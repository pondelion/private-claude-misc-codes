# Kimi K3 - 簡略化疑似コード集

Kimi K3 (Kimi Team / Moonshot AI, 2026) の理解を目的とした簡略化疑似コード集です。

論文: *Kimi K3: Open Frontier Intelligence* (Kimi Team, Technical Report, 2026)
公式重み: [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)

このリポジトリのコードは全て **実際に動作する PyTorch 実装** です (末尾に検証コマンドあり)。
ただし実モデルは 2.8T パラメータのため、動作確認は縮小スケールのトイモデルで行います。

## 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [主要イノベーション](#主要イノベーション)
- [Pre-Training / Post-Training](#pre-training--post-training)
- [形状ガイド](#形状ガイド)
- [簡略化した点・省略した点](#簡略化した点省略した点)
- [動作確認](#動作確認)
- [FAQ](#faq)

---

## 概要

**Kimi K3 の特徴:**
- **2.8T パラメータ MoE** (104B 活性化パラメータ)。896個中16個のルーテッドエキスパートを活性化 (sparsity=56)
- **Hybrid Attention**: 3層の Kimi Delta Attention (KDA, 線形アテンション) + 1層の Gated MLA (グローバルアテンション) の繰り返し
- **Attention Residuals (AttnRes)**: 通常の残差接続を「深さ方向のアテンション」に置き換え、各層が全ての先行層の出力を選択的に読み出せる
- **Stable LatentMoE**: フル幅の共有エキスパート + 低次元潜在空間で動作するルーテッドエキスパート。SiTU-GLU (有界GLU) と Quantile Balancing (分位点ベースの補助損失フリー負荷分散) で極端なスパース性を安定化
- **Native Vision (MoonViT-V2)**: contrastive事前学習 (SigLIP等) を使わず next-token prediction のみでゼロから学習する 0.4B パラメータの ViT
- **1M トークンコンテキスト**: NoPE (No Positional Encoding) + KDA の再帰的減衰ゲートにより、位置符号化の再調整なしに 1M トークンへ外挿
- **Multi-Teacher On-Policy Distillation (MOPD)**: 3ドメイン x 3 reasoning-effort = 9個の専門家モデルを単一モデルへ蒸留
- **RL (SFT → RL)**: グループ相対的優位度 (K個の応答をサンプルしグループ内で正規化) と予算制御報酬による強化学習。`rl_training_example.py` で実際に方策が改善することを検証

**タスク:**
- 長時間コーディングエージェント (SWE, カーネル最適化, Web開発)
- 汎用エージェント (深層リサーチ, 長文執筆, パーソナルアシスタント)
- マルチモーダル推論 (画像・動画・vision-in-the-loop ツール利用)

**Kimi K2 → K3 の主要な変更 (§3.2, Table "k2-k3-comparison"):**

| | Kimi K2 | Kimi K3 | 変化 |
|---|---|---|---|
| 総パラメータ | 1.04T | 2.78T | ↑167% |
| 活性化パラメータ | 32.6B | 104.2B | ↑220% |
| レイヤー数 | 61 | 93 | ↑52% |
| ルーテッドエキスパート | 384 | 896 | ↑133% |
| トークンあたり活性エキスパート | 8 | 16 | ↑100% |
| 共有エキスパート | 1 | 2 | ↑100% |
| アテンション | MLA のみ | Hybrid KDA (69層) + MLA (24層) | -- |
| 活性化関数 | SwiGLU | SiTU-GLU | -- |
| 訓練コンテキスト長 | 128K | 1M | 8x |
| Vision | なし | MoonViT-V2 (401M, 27層) | -- |
| 全体スケーリング効率 | 基準 | **約2.5倍改善** | -- |

---

## アーキテクチャ全体像

```
入力: テキストトークン + 画像/動画 (可変解像度)
    │
┌───▼─────────────────────────────────────────────────────────┐
│ 1. Native Vision (MoonViT-V2)               [native_vision.py] │
│    画像を14x14パッチに分割 (パディングなし)                        │
│    ViT 27層 (RMSNorm, bias無し, 2D RoPE, next-token-predictionで│
│    ゼロから学習 = SigLIP等の contrastive事前学習を使わない)         │
│    2x2 pixel-shuffle (空間4倍圧縮) + 動画は時間方向プーリング       │
│    MLP Projector で d_llm 次元へ写像                             │
│    → visual_tokens: (N_v, d_llm)                              │
└───┬─────────────────────────────────────────────────────────┘
    │ テキスト埋め込みの画像プレースホルダ位置に visual_tokens を埋め込む
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Hybrid Attention Backbone × L 層 (K3実値 L=93)               │
│    ┌───────────────────────────────────────────────────┐     │
│    │ 3層の Kimi Delta Attention (線形アテンション)         │     │
│    │   [kda_attention.py]                                │     │
│    │   channel-wise forget gate 付き delta-rule 再帰       │     │
│    │   状態 S: (d_k, d_v) を token 数に依らず固定サイズで保持│     │
│    ├───────────────────────────────────────────────────┤     │
│    │ 1層の Gated MLA (グローバルアテンション, NoPE)         │     │
│    │   [gated_mla.py]                                    │     │
│    │   低ランクKV圧縮 + フルランク出力ゲート                │     │
│    └───────────────────────────────────────────────────┘     │
│         × 23セット (3:1比) + 末尾に追加のGated MLA 1層           │
│                                                                │
│    各 attention 層 → Stable LatentMoE [stable_latent_moe.py]  │
│      896エキスパート中16個活性化、SiTU-GLU、Quantile Balancing    │
│                                                                │
│    層をまたぐ残差 → Block Attention Residuals                   │
│      [attention_residuals.py]  (8ブロック, ブロックサイズ12)      │
└───┬─────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 最終 RMSNorm → lm_head                    [main_flow.py]   │
│    → logits: (B, T, vocab_size)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## ファイル構成

| ファイル | 内容 | 主要クラス/関数 |
|---------|------|---------------|
| [kda_attention.py](kda_attention.py) | Kimi Delta Attention (§2.1.1) | `KimiDeltaAttention`, `kda_recurrent_reference`, `kda_chunkwise_forward` |
| [gated_mla.py](gated_mla.py) | Gated Multi-head Latent Attention, NoPE (§2.1.2) | `GatedMLA` |
| [attention_residuals.py](attention_residuals.py) | Full / Block Attention Residuals (§2.2) | `FullAttentionResidual`, `BlockAttentionResidual` |
| [stable_latent_moe.py](stable_latent_moe.py) | Stable LatentMoE: SiTU-GLU + Quantile Balancing (§2.3) | `StableLatentMoE`, `SiTUGLU`, `quantile_balancing_update` |
| [native_vision.py](native_vision.py) | MoonViT-V2: 可変解像度 ViT + Projector (§2.4) | `MoonViTV2`, `navit_patchify` |
| [main_flow.py](main_flow.py) | 全体フォワードパス (上記全てを統合) | `KimiK3ForConditionalGeneration`, `KimiK3DecoderLayer` |
| [loss_computation.py](loss_computation.py) | Post-Training の報酬・損失 (§4.1) | `reasoning_effort_budget_reward`, `mopd_loss`, `eagle3_lk_loss` |
| [finetuning_example.py](finetuning_example.py) | 実際に動く SFT ファインチューニングスクリプト | `KimiK3SFTDataset`, `KimiK3Collator`, `train()` |
| [rl_training_example.py](rl_training_example.py) | 実際に動く RL 学習スクリプト (§4.1 "Reinforcement Learning") | `train_rl()`, `group_relative_advantage`, `clipped_policy_gradient_loss` |

各ファイルは単体で `python <file>.py` として実行可能で、末尾に数値検証付きの動作確認コードが入っています (下記「動作確認」参照)。

---

## 主要イノベーション

### 1. Kimi Delta Attention (KDA) の Lower-Bounded Decay

Kimi Linear の delta-rule 再帰にチャネルワイズ忘却ゲートを組み合わせたもの:

```
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t
```

Kimi Linear は decay logit を `g_t = -e^{A} Softplus(z_t)` (下限なし) で写像するが、これは
チャンク並列計算時の鍵の逆数リスケーリング `1/Γ` を発散させうる。Kimi K3 は代わりに
**scaled sigmoid** で decay を下限付きにする:

```
g_t = g_min * Sigmoid(e^{A} z_t) ∈ (g_min, 0),   g_min = -5 (固定)
```

これにより 16トークンタイル内の累積 log-decay が `(-80, 0)` に収まり、BF16 の
ダイナミックレンジ内でチャンク並列計算の対角/非対角タイル両方を Tensor Core の
密行列積で計算できるようになる (診断的な対角パス計算が不要になる)。

### 2. Attention Residuals: 深さ方向のアテンション

通常の残差接続 `h_l = h_{l-1} + f(h_{l-1})` は全層の情報を単一ベクトルに圧縮してしまう。
AttnRes は各層に学習可能な擬似クエリ `w_l` を持たせ、softmaxカーネル
`φ(q,k) = exp(q^T RMSNorm(k))` を使って全ての先行層 (embedding含む) の出力から
選択的に読み出す:

```
α_{i→l} = φ(w_l, h_i) / Σ_j φ(w_l, h_j)
h_l = Σ_i α_{i→l} · h_i
```

L 層全てを保持すると O(L²d) の計算・O(Ld) のメモリを要するため、Kimi K3 は
S=12層ずつ N=8ブロックに分割し、ブロック内は総和で1つの代表ベクトルに集約してから
ブロック間だけで full attention を行う **Block AttnRes** を採用 (メモリ O(Ld)→O(Nd))。

### 3. Stable LatentMoE: 極端なスパース性 (896個中16個) の安定化

- **Normalized LatentMoE**: ルーテッドエキスパートの集約結果 `u` を `W_up` 適用前に RMSNorm
- **SiTU-GLU**: `SwiGLU = Sigmoid(Wg x)·(Wg x)·(Wu x)` の両乗数を `softcap(z,β)=β·tanh(z/β)` で
  有界化 (`β1=4, β2=25` → `|f(x)| ≤ 100`)。原点近傍では SwiGLU と一致しつつ、大振幅入力での
  活性化爆発を防ぐ
- **Quantile Balancing (QB)**: 補助損失フリーのバイアス `b_j` を固定ステップの符号更新ではなく、
  「ターゲット負荷 `q=mk/n` に対応する分位点」から直接計算する (最適バランス割当のラグランジュ
  双対から導出)。1回の forward で収束方向のバイアスを得られ、ハイパーパラメータ (学習率相当) 不要

### 4. Native Vision: Contrastive事前学習なしの MoonViT-V2

Kimi K2.5 まではSigLIP等で contrastive事前学習した ViT を LLM に接続していたが、
これは共同最適化時に不安定 (勾配ノルムのスパイクが頻発) だった。Kimi K3 は
MoonViT-V2 を **next-token predictionのみでゼロから学習** することで、
SigLIP初期化ベースラインと同等の視覚性能を安定して達成する。

---

## Pre-Training / Post-Training

### Pre-Training (§3)
- Web Text, Code, Mathematics, Knowledge の4ドメイン + 大規模ビジョンコーパス
- Per-Head Muon (Newton-Schulz直交化をヘッド単位で適用) + weight-clipping + QB
- コンテキスト長: 8K → 64K (pre-training) → 256K → 1M (cooldown) の4段階カリキュラム
- NoPE (KDAの再帰的減衰が位置情報を暗黙的に運ぶ) のため、1Mへの外挿にRoPE再調整が不要

### Post-Training (§4) -- `loss_computation.py`, `rl_training_example.py`
3段階パイプライン: **SFT → RL → Multi-Teacher On-Policy Distillation (MOPD)**

1. **SFT**: XTMLベースのチャットテンプレートでエージェント軌跡を直列化。SFT段階からMXFP4量子化認識学習 (QAT) を開始
   (`finetuning_example.py`)
2. **RL**: 3ドメイン (general / general agents / coding agents) × 3 reasoning-effort (low/high/max) = 9専門家モデル
   (`rl_training_example.py`)
   - **グループサンプリング**: N個のプロンプトそれぞれにK個の応答をサンプルする ("K completions for
     each of N prompts", §4.1 "Algorithm")
   - **Reasoning Effort RL**: 予算 `b0(x)` を超えたトラジェクトリの報酬を `-1` に上書き (`reasoning_effort_budget_reward`)
   - **partial rollout**: `λNK` 個の完了を待つだけで次イテレーションへ進み、long-tail latencyを緩和
   - **per-token 正則化**: 「更新をローカルな近傍に制約する」ことでオフポリシー・データの陳腐化に頑健になる
     (§4.1 "Algorithm")。正確な更新式は別論文 Kimi K2.5 に委譲されているため、同じ性質 (グループ相対的
     優位度 + per-tokenの重要度比クリップ) を持つ GRPO/PPO系の clipped surrogate で代替 (`rl_training_example.py` docstring参照)
3. **MOPD**: per-token 報酬 `r_opd = clip(sg(log(π_teacher/π_student)), -R_max, R_max)` で9専門家を単一モデルへ蒸留 (`mopd_loss`)
4. **Draft Model (EAGLE-3) Fine-Tuning**: 事前学習済みMTP層を投機的デコーディングのdraftモデルへ転用。
   受理率を直接最大化する LK損失 `L_LK = -log Σ_x min(p(x),q(x))` で学習 (`eagle3_lk_loss`)

---

## 形状ガイド

### 軸の意味

| 変数 | 意味 | K3実値 |
|------|------|--------|
| B | バッチサイズ (本リポジトリの縮小実装は AttnRes の都合上 B=1 前提) | -- |
| T | シーケンス長 (テキスト+視覚トークン) | 〜1M |
| d | モデル隠れ次元 (hidden_size) | 7168 |
| L | バックボーンの層数 | 93 (69 KDA + 24 Gated MLA、うち先頭1層は first_k_dense_replace で密FFN) |
| H_kda, d_k(kda) | KDAのヘッド数、ヘッドあたりq/k/v次元 | 96, 128 |
| H_mla | Gated MLAのヘッド数 | 96 |
| d_qc, d_c | MLAのクエリ/KV低ランク圧縮次元 (q_lora_rank / kv_lora_rank) | 1536, 512 |
| d_h, d_r, d_v(mla) | MLAのnope次元/rope次元/value次元 | 128, 64, 128 |
| ℓ | LatentMoEのルーテッドエキスパート潜在次元 (routed_expert_hidden_size) | 3584 |
| n | ルーテッドエキスパート総数 | 896 |
| k | トークンあたり活性化エキスパート数 | 16 |
| n_s | 共有エキスパート数 | 2 |
| d_ffn(moe) | ルーテッドエキスパートのFFN中間次元 (moe_intermediate_size) | 3072 |
| N, S | AttnRes のブロック数, ブロックサイズ (attn_res_block_size) | N=8, S=12 |
| d_v(vision) | Vision Encoder 隠れ次元 (vt_hidden_size) | 1024 |
| L_v | Vision Encoder レイヤー数 (vt_num_hidden_layers) | 27 |
| P | ViT パッチサイズ | 14 |
| vocab_size | 語彙サイズ | 163840 |

上記の値は全て公式重み [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
の `config.json` から直接確認した実値です (論文本文からの推定ではありません)。`main_flow.py` の
`KimiK3Config` の各フィールドにも同じ値をコメントで併記しています。

### 各段階のテンソル形状

| 段階 | テンソル名 | 形状 | 説明 |
|------|-----------|------|------|
| **Vision前処理** | pixel_values | `(N_patches, 3, P, P)` | 全パッチをパッチごとの画像ブロックとして平坦化 |
| | grid_thw | `(3,)` | `[T, H/P, W/P]` パッチグリッドサイズ |
| **Vision出力** | visual_tokens | `(N_v, d)` | `N_v = (H/P/2)*(W/P/2)` (2x2圧縮後、d=LLM隠れ次元) |
| **KDA内部** | recurrent_state | `(B, H, d_k, d_v)` | トークン数に依らない固定サイズの状態 |
| **Gated MLA** | latent_kv (キャッシュ対象) | `(B, T, d_c + d_r)` | `d_c`=kv_lora_rank, `d_r`=qk_rope_head_dim (フルKVより小さい) |
| **AttnRes** | block_residual | `(N, M, d)` | `N=B*T`, `M`=確定済みブロック数 (最大N=8) |
| **MoE** | u (集約後) | `(N, ℓ)` | ルーテッド経路の潜在表現 |
| **バックボーン出力** | hidden_states | `(B, T, d)` | 最終隠れ状態 |
| **ロジット** | logits | `(B, T, vocab_size)` | 語彙分布 |

---

## 実装の忠実性について (config.json との照合)

上記の実値はいずれも公式重み [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
の `config.json` および `modeling_kimi_linear.py` と直接突き合わせて確認済みです。特に以下2点は
論文本文だけからは読み取れないため、公式実装の該当箇所を根拠にコードへ反映しています:

- **Hybrid Attention の周期**: 「3層KDA + 1層Gated MLA」は周期4のパターン (KDA,KDA,KDA,MLA の
  繰り返し) で、かつ最終層は周期に関わらず必ず Gated MLA になる (`linear_attn_config.full_attn_layers`
  が `[4, 8, 12, ..., 92, 93]` と末尾だけ2層連続している通り)。`main_flow.py` の
  `KimiK3DecoderLayer.__init__` はこの2条件で KDA / Gated MLA を切り替えている。
- **KDAのdecay-logit低ランク射影の次元**: `f_a_proj`/`f_b_proj` の低ランクボトルネック次元は
  `head_dim` そのもの (全ヘッドで共有)。`kda_attention.py` の `alpha_rank = head_dim` に対応する。
- **first_k_dense_replace**: 先頭の1層 (layer_idx=0) はMoEではなく通常の密なFFNである。
  `main_flow.py` の `KimiK3DecoderLayer` は `is_dense_ffn` フラグでこれを切り替えている。

## 簡略化した点・省略した点

- **KDA チャンク並列カーネル (UT transform)**: 論文はチャンク内の並列行列演算の導出を
  別論文 (Kimi Linear) に委譲しているため (§2.1.1 "We refer readers to Kimi Linear for the
  UT transform and the full derivation")、本実装は数学的に同一の出力を返す
  チャンク境界での状態伝播 + チャンク内逐次計算で代替している (`kda_attention.py` の
  docstring と `if __name__` 内の数値一致検証を参照)。
- **MXFP4量子化認識学習 (QAT)**: OCP Microscaling仕様という外部標準文書に定義される
  数値フォーマットの実装であり、損失関数ではなくインフラ/カーネル技術のため対象外とした
  (`loss_computation.py` 冒頭コメント参照)。
- **実トークナイザ・実重み**: Kimi K3 の実トークナイザ (tiktoken, `tokenization_kimi.py`) は
  `tiktoken.model` という巨大な外部語彙ファイルを必要とし、実重みは2.8Tパラメータ
  (数百GB) のため、`finetuning_example.py` はバイトレベルの代替トークナイザと
  縮小スケールモデルで「学習ループが最初から最後まで正しく動く」ことを実証する
  (詳細は同ファイルの docstring 参照)。
- **公式コード非公開部分**: Infrastructure章 (§5, MoonEP, KDA Context Parallelism, AgentENV
  サンドボックス等) はシステム実装 (分散学習・サービング) であり、モデルの数式や損失関数を
  対象とする本リポジトリの範囲外としてコード化していない (README内で概要のみ説明)。
- **RLのポリシー最適化アルゴリズム**: §4.1 "Algorithm" は "policy optimization... follows the
  algorithm in Kimi K2.5" と明記しており、具体的な更新式は別論文 (Kimi K2.5、本タスクで参照可能な
  tex ソースには含まれない) に委譲されている。`rl_training_example.py` は、論文が述べる性質
  (K個の応答をグループサンプリング、per-tokenの近傍制約によるオフポリシー耐性) を満たす
  GRPO/PPO系の clipped surrogate という標準的な定式化で代替している (同ファイル冒頭のdocstring参照)。

---

## 動作確認

全ファイルが単体で実行可能で、実際に forward/backward が通ることを検証しています。

```bash
python kda_attention.py        # KDA: 逐次形とチャンク形の出力が完全一致することを検証
python gated_mla.py            # Gated MLA: 出力shape + 因果性 (未来のトークンが過去に影響しないこと) を検証
python attention_residuals.py  # Full/Block AttnRes の出力shapeを検証
python stable_latent_moe.py    # SiTU-GLUの有界性 + Quantile Balancingによる負荷分散を検証
python native_vision.py        # MoonViT-V2: 静止画・動画混在バッチのforwardを検証
python main_flow.py            # 統合モデル: テキストのみ/テキスト+画像のforwardを検証
python loss_computation.py     # 各損失関数: 教師に近いほど損失が小さくなることを検証
python finetuning_example.py   # 極小データセットで実際にSFT学習し、lossが低下することを検証
python rl_training_example.py  # SFTコールドスタート後、RL (group-relative clipped policy gradient)
                                # でさらに平均報酬が向上することを検証
```

(本セッションでは torch 2.4.1+cu124 の Docker イメージ上で全ファイルの実行を確認済みです。)

---

## FAQ

### Q1: KDAとGated MLAはどちらもKVキャッシュを持つのか？

**A**: 両方とも持つが性質が異なる。KDAは系列長に依らない**固定サイズ**の再帰状態
`S ∈ R^{d_k×d_v}` のみを保持する。Gated MLAは低ランク圧縮された潜在ベクトル
`c_t ∈ R^{d_c+d_r}` を系列長分キャッシュするが、通常のMHAのフルKV (`num_heads×head_dim×2`)
よりずっと小さい。公式実装ではこの2種類のキャッシュを1つのpagedプールに統一している
(§5.3.1 "Unified cache layout for hybrid KDA--MLA attention")。

### Q2: なぜ全てのMLA層でNoPE (位置符号化なし) にしたのか？

**A**: KDAの再帰的減衰ゲートが既に位置依存・近接性を考慮したトークン混合を提供しているため、
MLA層は「制約のないグローバルなコンテンツ相互作用」に専念できる (§2.1.2)。
副次効果として、コンテキスト長を延長する際にRoPEの基底周波数の再調整やYaRN適用が不要になる。

### Q3: Quantile Balancingは何が嬉しいのか？

**A**: 従来のauxiliary-loss-freeバランシング (DeepSeek-V3) は `b_j += γ·sign(load誤差)` という
固定ステップ更新で、ステップ幅 `γ` のチューニングが必要かつ収束が遅い。QBは
「バランス割当問題の双対」から導出された**分位点**を直接バイアスに設定するため、
ハイパーパラメータなしで数ステップのうちに ~900個のエキスパートの負荷を均衡させられる
(§2.3.3, Appendix "Relation to sign-based loss-free updates")。

### Q4: Attention Residualsは通常の残差接続と比べて何が改善するのか？

**A**: 通常の残差 `h_l = h_{l-1} + f(h_{l-1})` は全ての先行層の情報を単一のベクトルに
逐次圧縮してしまう (RNNの時間方向ボトルネックと同じ構造)。AttnResは各層が
「どの先行層 (embeddingを含む) から何を読み出すか」を softmax アテンションで
選択的に決められるため、深いネットワークでも情報が減衰しにくい。

### Q5: MOPDと通常のKD (Knowledge Distillation) は何が違うのか？

**A**: 通常のKDはオフポリシー (教師の出力分布に対するKL) だが、MOPDは生徒が
オンポリシーでサンプルした軌跡 `y〜π_student` に対し教師とのlog確率比を
**per-tokenの密な報酬**として使い、RLの枠組み (partial rolloutなど) にそのまま
統合できる (§4.1.3)。9個の (ドメイン×reasoning-effort) 専門家を1つのモデルに
集約するのに使われている。

### Q6: MoonViT-V2はなぜcontrastive事前学習 (SigLIP等) を使わないのか？

**A**: SigLIP初期化のViT (先代のMoonViT-3D) をLLMと共同最適化すると、
勾配ノルムが持続的に高くスパイクが頻発し訓練が不安定になることが分かった
(Fig.vitgradnorm)。next-token predictionのみでゼロから学習することで、
視覚表現が直接言語モデリング目的関数によって形成され、かつ最終的な視覚性能は
SigLIP初期化ベースラインと同等になる (§2.4)。

### Q7: なぜ `rl_training_example.py` はいきなりRLから始めずSFTコールドスタートを挟むのか？

**A**: §4.1 "Reinforcement Learning" が "While SFT provides a solid cold-start foundation,
RL is critical to unlocking higher-order reasoning..." と述べている通り、Kimi K3のRLは
SFT済みモデルを起点とする。ランダム初期化のままRLだけを回すと、極小タスクであっても
グループ内のK個のサンプルが全て報酬0になりやすく、`group_relative_advantage` の分散が
0になって学習信号が消える (実際に本リポジトリでも最初にこの現象を確認した)。これは
「コールドスタートが無いとRLは機能しにくい」という論文の主張を裏付ける挙動であり、
`rl_training_example.py` は `finetuning_example.train()` を再利用した軽いSFTで方策を
初期化してからRLに入る。

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装 (公式重み・
公式カーネル) とは数値的に一致しない場合がありますが、論文で示された数式・アルゴリズムの
入出力形状と計算内容についてはできる限り忠実に再現しています。
