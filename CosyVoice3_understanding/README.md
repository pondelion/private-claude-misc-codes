# CosyVoice3 アーキテクチャ理解用ドキュメント

CosyVoice 3 (Towards In-the-wild Speech Generation via Scaling-up and Post-training) の複雑なコードベースを理解するための簡略化された疑似コードとドキュメントを提供します。

## 📁 ファイル構成

```
CosyVoice3_understanding/
├── README.md                    # このファイル
├── main_flow.py                # 全体パイプラインと推論フロー
├── speech_tokenizer.py         # FSQ-MinMo 音声トークナイザ
├── llm.py                      # Qwen2ベース言語モデル (0.5B/1.5B)
├── flow_matching.py            # Conditional Flow Matching + DiT
├── vocoder.py                  # Causal HiFT ボコーダ
└── diffro.py                   # DiffRO 後処理 (強化学習)
```

## 🎯 各ファイルの役割

### 1. [main_flow.py](main_flow.py)
**CosyVoice3全体のパイプライン**

- テキスト入力から音声波形出力までの完全な処理フロー
- Zero-shot合成、指示付き合成、クロスリンガル合成の3つの推論モード
- 各ステージでの入出力shape、軸の意味を詳細に記載

**5つの処理ステージ:**
1. プロンプト音声 → Speech Tokenizer → 音声トークン (25Hz)
2. テキスト → BPE Tokenizer → テキストトークン
3. LLM (Qwen2) で自己回帰的に音声トークン列を生成
4. CFM (DiT) で音声トークン → メルスペクトログラム
5. HiFT Vocoder でメルスペクトログラム → 波形 (24kHz)

**重要な入出力:**
- 入力: テキスト (文字列) + プロンプト音声 (24kHz波形)
- 出力: 合成音声 `(1, T_audio)` - 24kHz波形

---

### 2. [speech_tokenizer.py](speech_tokenizer.py)
**FSQ-MinMoベースの音声トークナイザ**

主要コンポーネント:

#### `SpeechTokenizer`
- MinMo (140万時間事前学習の大規模マルチモーダルモデル) をベースにFSQ量子化を挿入
- マルチタスク学習: ASR (36.5万h), LID (8.5万h), SER (4.8万h), AED (2.1万h), SA (1.1万h)
- 入力: 音声波形 `(B, T_samples)` - 24kHz
- 出力: 離散トークン `(B, T_frames)` - 25Hz, 各値 ∈ [0, 6560]

#### `FiniteScalarQuantizer (FSQ)`
- 連続表現を有限の離散値に量子化 (VQ-VAEのcodebookなし)
- Proj_down: D_enc → D_fsq (低ランク射影)
- ROUND + clamp [-K, K]: 各次元を独立に丸め量子化
- トークンインデックス: D_fsq次元の(2K+1)進数 → 単一整数
- 語彙サイズ: (2K+1)^D = 6561

**CosyVoice2との違い:**

| 側面 | CosyVoice2 | CosyVoice3 |
|------|-----------|-----------|
| **ベースモデル** | SenseVoice-Large (ASR) | MinMo (マルチモーダルLLM) |
| **事前学習データ** | 限定的 | 140万時間 |
| **学習タスク** | ASRのみ | ASR + LID + SER + AED + SA |
| **学習データ** | 限定的 | 53万時間 |

---

### 3. [llm.py](llm.py)
**Qwen2ベース音声言語モデル**

主要コンポーネント:

#### `CosyVoice3LM`
- テキストトークンとプロンプト音声トークンから音声トークン列を自己回帰生成
- Qwen2事前学習モデルをベースに音声生成用にファインチューン
- 入力:
  - テキストトークン: `(B, L_text)` → Conformer → Affine → `(B, L_text, 896)`
  - プロンプト音声トークン: `(B, L_prompt_speech)` → Embedding → `(B, L_prompt_speech, 896)`
- 出力: 音声トークン列 `(B, T_speech)` - 各値 ∈ [0, 6560]

#### `Qwen2ForCausalLM`
- 0.5B: 22層, 14ヘッド, 2 KVヘッド (GQA), 隠れ次元896
- 1.5B: 28層, 12ヘッド, 2 KVヘッド (GQA), 隠れ次元1536
- KVキャッシュによる高速自己回帰推論

#### `ras_sampling`
- RAS (Repetition-Aware Sampling): 繰り返しを検出してペナルティ
- Top-K=25 + Top-P=0.8 のNucleus sampling

**LLM入力シーケンス:**
```
[SOS] [text_embeds (L_text)] [prompt_speech_embeds (L_prompt)] [自己回帰生成]
```

---

### 4. [flow_matching.py](flow_matching.py)
**Conditional Flow Matching + Diffusion Transformer**

主要コンポーネント:

#### `CausalMaskedDiffWithDiT`
- 音声トークン → メルスペクトログラム変換の全体管理
- トークン埋め込み → 補間 (×2) → CFMデコーダ
- 入力:
  - 音声トークン: `(B, T_speech)` → Embedding → `(B, T_speech, 896)` → Proj → `(B, T_speech, 80)` → 補間 → `(B, T_mel, 80)`
  - 話者埋め込み: `(B, 192)` → Linear → `(B, 80)`
- 出力: メルスペクトログラム `(B, 80, T_mel)`, T_mel = T_speech × 2

#### `CausalConditionalCFM`
- Optimal Transport CFM (Matcha-TTSベース)
- Classifier-Free Guidance: 学習時 cfg_rate=0.2, 推論時 cfg_rate=0.7
- Cosineスケジュールの時刻サンプリング
- Euler ODE Solver (10ステップ)

#### `DiT` (Diffusion Transformer)
- 22層, 16ヘッド × 64次元 = 1024次元, 300Mパラメータ
- AdaLayerNormZero: 時刻埋め込みから6つの変調パラメータ生成
- Rotary Position Embedding (RoPE) で位置情報
- Long Skip Connections: 前半→後半レイヤーのスキップ接続
- 入力: ノイズ付きメル + トークン条件 + 時刻 + 話者 → 速度場推定
  - `x`: `(B, T_mel, 80)`, `mu`: `(B, T_mel, 80)`, `t`: `(B,)`, `spks`: `(B, 80)`
- 出力: 速度場 `(B, T_mel, 80)`

**CosyVoice2との違い:**

| 側面 | CosyVoice2 | CosyVoice3 |
|------|-----------|-----------|
| **アーキテクチャ** | U-Netベース | DiT (Diffusion Transformer) |
| **パラメータ数** | ~100M | 300M |
| **テキストエンコーダ** | 必要 | 不要 (DiT内で条件付け) |
| **長さ正規化** | 専用モジュール | 単純な補間 (nearest) |

---

### 5. [vocoder.py](vocoder.py)
**Causal HiFT ボコーダ**

主要コンポーネント:

#### `CausalHiFTGenerator`
- HiFi-GAN + Neural Source Filter (NSF)
- 入力: メルスペクトログラム `(B, 80, T_mel)`
- 出力: 音声波形 `(B, T_audio)`, T_audio ≈ T_mel × 240

#### `CausalConvRNNF0Predictor`
- メルスペクトログラムからF0 (基本周波数) を予測
- 因果畳み込み + GRU による予測
- 出力: `(B, 1, T_mel)` - Hz単位のF0

#### `SourceGenerator (NSF)`
- F0からサイン波ベースの源信号を生成
- 基本波 + 8高調波 → 9チャンネル
- 有声区間: サイン波, 無声区間: ガウスノイズ

#### `ResBlockWithSnake`
- Snake活性化関数: `Snake(x) = x + (1/α) × sin²(αx)`
- 周期的活性化で音声波形の周期構造を効果的に捕捉

#### `ISTFTSynthesis`
- 最終段でmagnitude/phaseを予測しiSTFTで波形合成

**アップサンプリング:**
```
メル (B, 80, T) → Conv (B, 512, T) → ×8 (B, 256, T×8)
→ ×5 (B, 128, T×40) → ×3 (B, 64, T×120) → ISTFT → 波形 (B, T_audio)
```

---

### 6. [diffro.py](diffro.py)
**DiffRO (Differentiable Reward Optimization) 後処理**

主要コンポーネント:

#### `DiffROTrainer`
- 従来のRL-for-TTSの問題: CFM/Vocoderが非微分可能 or 高コスト
- DiffROの解決策: 音声トークン上で直接最適化 (CFM/Vocoderをスキップ)
- Gumbel-Softmax で離散トークンを微分可能にサンプリング
- 目的関数: `max E[R(Y)] - β × D_KL(pi_theta || pi_ref)`

#### `Token2TextRewardModel`
- ASRライクな構造: 音声トークン → テキスト対数尤度
- ソフトトークン (Gumbel-Softmax出力) を直接入力可能
- 報酬: `R_ASR = log P(text | speech_tokens)`

#### `MultiTaskReward`
- ASR報酬に加えて SER, MOS, AED 等のタスク報酬を統合
- DiffRO-EMO: 感情認識報酬で感情制御能力を向上

**DiffROの効果:**
- SEED test-zh CER: 1.27% → 0.75% (41% 改善)
- SEED test-en WER: 2.46% → 1.76% (28% 改善)
- 低リソース言語 (韓国語): 68.7% 相対改善

---

## 🔍 CosyVoice3の全体アーキテクチャ

### データフロー図

```
┌─────────────────────────────────────────────────────────────┐
│                      入力データ                              │
│  ・テキスト (文字列)                                         │
│  ・プロンプト音声 (24kHz波形)                                 │
│  ・(オプション) 指示テキスト (感情、方言等)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐              ┌───────▼────────┐
│ Qwen BPE       │              │ Speech         │
│ Tokenizer      │              │ Tokenizer      │
│                │              │ (FSQ-MinMo)    │
│ テキスト→トークン│              │ 音声→トークン   │
│ (1, L_text)    │              │ (1, L_prompt)  │
└───────┬────────┘              └───────┬────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
               ┌────────▼────────┐
               │ CosyVoice3 LLM  │
               │ (Qwen2ベース)    │
               │ 0.5B / 1.5B     │
               │                 │
               │ 自己回帰生成     │
               │ RAS sampling    │
               │                 │
               │ 出力: 音声トークン│
               │ (1, T_speech)   │
               └────────┬────────┘
                        │
                        │  各値 ∈ [0, 6560], 25Hz
                        │
               ┌────────▼────────┐
               │ CFM Decoder     │
               │ (DiT, 300M)     │
               │                 │
               │ トークン埋め込み  │
               │ → 補間 (×2)     │
               │ → Euler ODE     │
               │   (10ステップ)   │
               │                 │
               │ 出力: メル       │
               │ (1, 80, T_mel)  │
               └────────┬────────┘
                        │
               ┌────────▼────────┐
               │ HiFT Vocoder   │
               │                 │
               │ F0予測          │
               │ NSF源信号       │
               │ ×8→×5→×3       │
               │ ISTFT合成       │
               │                 │
               │ 出力: 波形      │
               │ (1, T_audio)   │
               │ 24kHz          │
               └─────────────────┘
```

---

## 📊 主要な次元とその意味

### テンソル形状一覧

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | テキスト | 文字列 | 合成したいテキスト |
| | プロンプト音声 | `(1, T_samples)` | 24kHz波形, T_samples = 秒数×24000 |
| **トークン化** | テキストトークン | `(1, L_text)` | Qwen BPEトークン |
| | プロンプト音声トークン | `(1, L_prompt)` | FSQ-MinMoトークン, 25Hz |
| **LLM** | テキスト埋め込み | `(1, L_text, 896)` | Conformer→Affine後 |
| | 音声埋め込み | `(1, L_prompt, 896)` | Embedding層出力 |
| | LLM入力 | `(1, 1+L_text+L_prompt+t, 896)` | 自己回帰中のシーケンス |
| | logits | `(1, 1, 6761)` | 各ステップの予測分布 |
| | 生成音声トークン | `(1, T_speech)` | 25Hz, 各値 ∈ [0, 6560] |
| **CFM** | トークン特徴 | `(1, T_speech, 896)` | Embedding出力 |
| | 条件 (mu) | `(1, 80, T_mel)` | 射影+補間後, T_mel=T_speech×2 |
| | 話者埋め込み | `(1, 192)` → `(1, 80)` | 射影後 |
| | 速度場 | `(1, T_mel, 80)` | DiT出力 |
| | メルスペクトログラム | `(1, 80, T_mel)` | Euler ODE出力 |
| **Vocoder** | F0 | `(1, 1, T_mel)` | 基本周波数 (Hz) |
| | 源信号 | `(1, 9, T_audio)` | 基本波+8高調波 |
| | 中間特徴 | `(1, 512→256→128→64, T↑)` | アップサンプリング中 |
| | 出力波形 | `(1, T_audio)` | 24kHz, T_audio≈T_mel×240 |

### 時間軸の関係

```
テキスト  ──→  可変長トークン列

音声時間  ──→  T_speech = 秒数 × 25        (音声トークン, 25Hz)
          ──→  T_mel    = T_speech × 2     (メルフレーム, 50Hz)
          ──→  T_audio  ≈ T_mel × 240      (波形サンプル, 24kHz)

例: 5秒の音声
  → T_speech = 125 トークン
  → T_mel    = 250 フレーム
  → T_audio  ≈ 60,000 サンプル (= 120,000 / 2)
```

---

## 🧩 CosyVoice3の主要イノベーション

### 1. **FSQ-MinMoベースの音声トークナイザ**
**問題**: 従来のASR専用エンコーダでは感情・プロソディ等のパラリンギスティック情報を十分に捕捉できない
**解決**: 140万時間で事前学習されたMinMo (マルチモーダルLLM) にFSQを挿入し、ASR+LID+SER+AED+SAのマルチタスク学習
**効果**: CosyVoice2比でCER 12%改善、話者類似度 1.2%改善
**実装**: [speech_tokenizer.py](speech_tokenizer.py)

### 2. **DiffRO (Differentiable Reward Optimization)**
**問題**: 従来のRL-for-TTSはCFM/Vocoderを通す必要があり計算コストが高い
**解決**: Gumbel-Softmaxで離散トークンを微分可能にし、Token2Text報酬モデルでトークン上で直接最適化
**効果**: CER 41%改善 (test-zh), WER 28%改善 (test-en), 韓国語68.7%改善
**実装**: [diffro.py](diffro.py)

### 3. **DiTベースのCFM**
**問題**: U-NetベースのCFMではテキストエンコーダと長さ正規化モジュールが必要で複雑
**解決**: Diffusion Transformer (DiT) に変更し、トークン特徴を直接条件付け。長さ不一致は単純な補間で解決
**効果**: アーキテクチャの大幅な簡素化、パラメータ100M→300Mでスケーリング
**実装**: [flow_matching.py](flow_matching.py)

### 4. **大規模データスケーリング**
**問題**: CosyVoice2は1万時間のデータで主に中英2言語
**解決**: 100万時間に拡大、9言語+18中国方言、多様なドメイン・テキスト形式をカバー
**効果**: 多言語ベンチマークでSOTA、クロスリンガル合成が劇的改善 (日→中: 48.1%→3.05% WER)

### 5. **Pronunciation Inpainting (発音修正)**
**問題**: 多音字 (中国語) / 多義語 (英語) の誤読
**解決**: 語彙を拡張してピンイン/ARPABETを混在可能に。単音字のみ発音に置換 (RepMono + MixPhn)
**効果**: 中国語・英語ともに100%の発音修正率

---

## 🏋️ 学習パイプライン

### 5ステージの学習フロー

```
Stage 1: 大規模事前学習
├── データ: ~100万時間 (全データ)
├── 対象: LLM + CFM
├── 損失: Next-Token Prediction (CE) + CFM Loss
└── 出力: ベースZero-shot LM & CFM

    ↓

Stage 2: DiffRO後処理
├── データ: 選択されたサブセット
├── 対象: LLMのみ (CFMは固定)
├── 損失: max E[R_ASR] - β × D_KL
├── Token2Text報酬モデルを別途学習
└── 出力: DiffRO適用済みLM

    ↓

Stage 3: 継続事前学習
├── データ: 感情、指示追従、多言語データ
├── 対象: LLM
├── 目的: 能力転移 (指示追従、感情、多言語)
└── 出力: SFT用ベースLM

    ↓

Stage 4: 話者ファインチューン
├── データ: 多話者データ
├── 対象: LLM + CFM
├── 話者埋め込み: 教師なしクラスタリングでセンター取得
└── 出力: 最終SFTモデル
```

---

## 📉 損失関数

CosyVoice3は各コンポーネントを個別に学習するため、4種類の損失関数が使用されます。

### 1. LLM損失: Cross-Entropy (音声トークン予測)

テキストトークンと音声トークンを連結した入力に対し、**音声部分のみ**でCross-Entropyを計算します。

```
入力シーケンス:  [SOS] [text_embeds (L_text)] [speech_embeds (T_speech-1)]
ターゲット:      [------- IGNORE -------]     [speech_tokens[1:] + EOS  ]

logits = LLM_Decoder(Qwen2(lm_input))
# logits: (B, 1 + L_text + T_speech - 1, 6761)

speech_logits = logits[:, 1 + L_text:, :]
# speech_logits: (B, T_speech - 1, 6761) ← 音声部分のみ抽出

L_LLM = CrossEntropy(speech_logits, speech_tokens[:, 1:], ignore_index=-1)
```

**ポイント:**
- テキスト位置のlogitsは損失計算から除外 (テキストは条件であり予測対象ではない)
- Label Smoothingあり (公式config: `lsm_weight: 0.1`)
- 語彙サイズ 6761 = 6561 (音声トークン) + 200 (特殊トークン)
- 実装: [llm.py:229-241](llm.py#L229-L241)

### 2. CFM損失: MSE (フローマッチング速度場)

Conditional Flow Matching の Optimal Transport パスに基づく速度場推定の二乗誤差です。

```
t ~ U(0, 1)                               # ランダム時刻
t = 1 - cos(t × π / 2)                    # Cosineスケジュール
z ~ N(0, I)                               # (B, 80, T_mel)

# 補間パス
x_t = (1 - (1-σ)t) × z + t × x1          # (B, 80, T_mel)

# DiTで速度場推定
v_theta = DiT(x_t, t, mu, spks)           # (B, T_mel, 80)

# ターゲット速度場
target_flow = x1 - (1-σ) × z             # (B, 80, T_mel)

# 損失
L_CFM = ||v_theta - target_flow||^2       # MSE
```

**ポイント:**
- Cosineスケジュール: `t = 1 - cos(t × π/2)` でt=0近傍を重視 (ノイズが多い時刻を重点的に学習)
- σ_min = 1e-6 (ノイズフロアパラメータ)
- CFG学習: 確率0.2で条件 (mu) をゼロマスクし、無条件速度場も同時に学習
- 入力 x1 はターゲットメルスペクトログラム `(B, 80, T_mel)`
- 実装: [flow_matching.py:298-364](flow_matching.py#L298-L364)

### 3. HiFiGAN損失: GAN敵対学習 + 再構成

HiFTボコーダの学習にはGAN損失とメル再構成損失を組み合わせます。

```
# Discriminator損失
L_D = Σ_k E[(D_k(x_real) - 1)^2 + D_k(x_fake)^2]

# Generator損失
L_G_adv = Σ_k E[(D_k(x_fake) - 1)^2]           # 敵対損失
L_G_mel = ||mel(x_fake) - mel(x_real)||_1        # メル再構成
L_G_fm  = Σ_k Σ_l ||D_k^l(x_fake) - D_k^l(x_real)||_1  # Feature matching

L_G = L_G_adv + λ_mel × L_G_mel + λ_fm × L_G_fm
```

**ポイント:**
- Multi-Period Discriminator (MPD) + Multi-Resolution Discriminator (MRD)
- GeneratorとDiscriminatorを交互に更新 (executor.pyの `train_one_epoc_gan`)
- Feature matching損失: Discriminatorの中間特徴を一致させ学習を安定化
- 学習フローは `Executor.train_one_epoc_gan()` で管理

### 4. DiffRO損失: 報酬最大化 + KL正則化

DiffROはLLMを後処理で最適化する強化学習ベースの損失です。

```
# Equation 5: 目的関数
L_DiffRO = -E[R_ASR(Y)] + β × D_KL(pi_theta || pi_ref)

  ここで:
    R_ASR(Y) = log P_ASR(Y_n | Y_{1:n-1}; μ_bar)     # ASR報酬 (Eq.4)
    μ_bar_t  = GumbelSoftmax(P_{pi_theta}(μ_t))       # 微分可能サンプリング (Eq.3)
    D_KL     = Σ_t Σ_k P_theta(k) × log(P_theta(k) / P_ref(k))  # KL正則化 (Eq.6)
```

**計算フロー:**
```
policy_logits = PolicyLLM(text, speech_tokens)     # (B, T, 6561)
ref_logits    = RefLLM(text, speech_tokens)         # (B, T, 6561) [勾配不要]
     ↓
soft_tokens = GumbelSoftmax(policy_logits, τ)       # (B, T, 6561) 微分可能
     ↓
reward = Token2TextReward(soft_tokens, text)        # (B,) ASR対数尤度
kl_div = KL(softmax(policy_logits), softmax(ref_logits))  # (B,)
     ↓
loss = -reward.mean() + β × kl_div.mean()
```

**ポイント:**
- Gumbel-Softmax温度 τ: 離散トークンを微分可能に近似 (hard=False)
- β: KL正則化の強度。参照モデルからの逸脱を抑制し学習の安定性を確保
- Token2Text報酬モデル: ASRライクな構造で、ソフトトークンから直接テキスト尤度を計算
- CFM/Vocoderを経由せずトークン空間で直接最適化 → 計算コスト大幅削減
- 実装: [diffro.py:71-183](diffro.py#L71-L183)

---

## 🌊 Flow Matching 詳解

Flow Matchingは確率的生成モデルの一手法で、CosyVoice3では音声トークン→メルスペクトログラム変換に使われています。ここではDiffusionとの比較を交えながら、数学的な直感から実装まで解説します。

### そもそも何がしたいのか

目標は「ノイズ (ランダムな数値の羅列) → データ (メルスペクトログラム)」の変換を学習することです。

```
z ~ N(0, I)     ←── 標準正規分布からのランダムノイズ
     ↓ 学習した変換 T
x1 = T(z)       ←── メルスペクトログラム (データ分布に従う)
```

この「変換 T」をどう構成・学習するかが手法の違いです。

### Diffusion vs Flow Matching

#### Diffusion Model (DDPM系)

Diffusion はデータに**少しずつノイズを加えていく過程** (Forward SDE) を定義し、その**逆過程** (Reverse SDE) を学習します。

```
Forward (固定):   x0 → x_t = √ᾱ_t × x0 + √(1-ᾱ_t) × ε      (tステップのノイズ化)
学習対象:         ε_θ(x_t, t) ≈ ε                              (ノイズ予測)
推論:             x_T ~ N(0,I) → x_{T-1} → ... → x_0          (逆拡散, 数百~数千ステップ)
```

**問題点:**
- 確率的 (SDE) なので推論時に多数のステップが必要 (遅い)
- ノイズスケジュール (β_t) の設計が複雑
- 理論がSDE/スコアマッチングに立脚し、導出が重い

#### Flow Matching (本手法)

Flow Matching は「ノイズ → データ」を**決定的なODE (常微分方程式)** で結ぶ**直線パス**を考えます。

```
直線パス:         x_t = (1 - t) × z + t × x1                  (t=0でノイズ, t=1でデータ)
速度場:           dx/dt = x1 - z                               (一定の速度で直線的に移動)
学習対象:         v_θ(x_t, t) ≈ x1 - z                        (速度場予測)
推論:             z ~ N(0,I) → ODE求解 → x1                   (10~32ステップで十分)
```

**Flow Matching の利点:**
- **決定的 ODE**: 確率的な項がないので少ないステップで高品質
- **直線パス**: ノイズ→データ間の最短経路を学習 (効率的)
- **シンプルな損失**: MSE `||v_θ - (x1 - z)||²` だけ
- **ノイズスケジュール不要**: パスの形が自明 (直線)

### 数学的な詳細

#### 1. Conditional Flow Matching (CFM)

直接「全データ分布の速度場」を学ぶのは困難なので、**個々のデータ点 x1 に条件付けた速度場**を学びます (Conditional Flow Matching)。

```
データ点 x1 ごとに:
  条件付きパス:  ψ_t(z | x1) = (1 - (1-σ)t) × z + t × x1
  条件付き速度:  u_t(x | x1) = (x1 - (1-σ)z) / 1          ← パスの微分

  σ = σ_min ≈ 1e-6  (ノイズフロア: t=1でもわずかなノイズを残す安定化項)
```

このとき、CFMの損失は:

```
L_CFM = E_{t, x1, z} [ ||v_θ(ψ_t(z|x1), t) - u_t(ψ_t(z|x1)|x1)||² ]
      = E_{t, x1, z} [ ||v_θ(x_t, t) - (x1 - (1-σ)z)||² ]
```

**直感**: 時刻tでの「ノイズ混合データ x_t」を見せて、「データ方向への速度 (x1 - (1-σ)z)」を予測させる。

#### 2. Optimal Transport パス

CosyVoice3/F5-TTSが使う Optimal Transport CFM では、各ノイズ z を最も近いデータ点 x1 とペアリングします。これにより:

- 輸送コスト (移動距離の合計) が最小化される
- パスが交差しにくくなり、速度場がスムーズになる
- 少ないODEステップでも高品質な生成が可能

```
通常のCFM:    ランダムな z と x1 をペアリング → パスが交差しやすい
OT-CFM:       最近傍の z と x1 をペアリング → パスが平行に近い → スムーズ
```

#### 3. Cosineスケジュール (CosyVoice3固有)

一様分布 `t ~ U(0,1)` から時刻をサンプリングする代わりに、cosine変換で `t=0` 近傍を密にサンプリングします:

```
u ~ U(0, 1)
t = 1 - cos(u × π/2)

u=0.0 → t=0.000   ← ノイズが多い時刻を重点サンプリング
u=0.1 → t=0.012
u=0.2 → t=0.049
u=0.3 → t=0.109
u=0.5 → t=0.293   ← 中間
u=0.7 → t=0.541
u=0.9 → t=0.844
u=1.0 → t=1.000
```

**なぜ?**: t≈0 付近 (ほぼノイズ) での速度推定が最も難しいため、そこを重点的に学習する。

### 学習の全体像

```
for each batch:
    # 1. ターゲットデータ取得
    x1 = target_mel                            # (B, 80, T_mel) - 正解メルスペクトログラム
    mu = token_features                        # (B, 80, T_mel) - トークン条件 (補間済み)
    spks = speaker_embedding                   # (B, 80)        - 話者埋め込み

    # 2. ランダム時刻サンプリング (Cosine)
    t = 1 - cos(rand(B) × π/2)                # (B,) ∈ [0, 1]

    # 3. ノイズサンプリング
    z = randn_like(x1)                         # (B, 80, T_mel)

    # 4. 補間パスでノイズ混合データを生成
    x_t = (1 - (1-σ)t) × z + t × x1           # (B, 80, T_mel)

    # 5. CFG: 確率0.2で条件を無効化 (無条件学習)
    if rand() < 0.2:
        mu = zeros_like(mu)

    # 6. DiTで速度場を予測
    v_θ = DiT(x_t, t, mu, spks)               # (B, T_mel, 80)

    # 7. ターゲット速度場
    target = x1 - (1-σ) × z                   # (B, 80, T_mel)

    # 8. MSE損失
    loss = MSE(v_θ, target)
    loss.backward()
```

### 推論の全体像

学習で得た速度場 v_θ を使って、ノイズから初めてODEを解きます。

```
# 初期値: 純粋ノイズ
x_0 = randn(1, 80, T_mel)                     # z ~ N(0, I)

# Euler法で ODE dx/dt = v_θ(x, t) を解く (10ステップ)
dt = 1.0 / 10                                 # ステップ幅 = 0.1
x = x_0

for i in range(10):
    t = i * dt                                 # t = 0.0, 0.1, ..., 0.9

    # CFG (Classifier-Free Guidance): 条件方向を増幅
    v_cond   = DiT(x, t, mu, spks)             # 条件付き速度場
    v_uncond = DiT(x, t, 0,  spks)             # 無条件速度場
    v = (1 + α) × v_cond - α × v_uncond        # α=0.7 で条件方向を1.7倍に増幅

    # Euler更新
    x = x + dt × v

# x がメルスペクトログラムになっている
mel = x                                        # (1, 80, T_mel)
```

### 視覚的な理解

```
t=0 (ノイズ)                              t=1 (データ)
┌──────────────┐                         ┌──────────────┐
│░░░░░░░░░░░░░░│                         │♪♪♪♪♪♪♪♪♪♪♪♪♪♪│
│░░ランダム░░░░│  ────── v_θ ──────→    │♪メル♪♪♪♪♪♪♪♪│
│░░ノイズ░░░░░░│   ODE (10ステップ)       │♪スペクトロ♪♪│
│░░░░░░░░░░░░░░│                         │♪グラム♪♪♪♪♪♪│
└──────────────┘                         └──────────────┘

     z ~ N(0,I)         x_t = x + dt × v_θ(x,t)        mel

学習時: x_t を見せて「データ方向への速度」を予測させる
推論時: z=0 からスタートして速度場に沿って移動 → データに到着
```

### Diffusionとの比較まとめ

| 観点 | DDPM (Diffusion) | Flow Matching |
|------|-----------------|---------------|
| **パス** | 確率的 (SDE) | 決定的 (ODE) |
| **学習対象** | ノイズ ε を予測 | 速度場 v を予測 |
| **パスの形** | ガウシアン拡散 (複雑) | 直線 (シンプル) |
| **損失関数** | `\|\|ε_θ - ε\|\|²` | `\|\|v_θ - (x1 - z)\|\|²` |
| **推論ステップ** | 数百~数千 (DDPM) / 20~50 (DDIM) | **10~32** |
| **スケジュール** | β_t の設計が必要 | 不要 (直線パス) |
| **理論** | Score Matching / SDE | ODE / 連続正規化フロー |
| **品質 (同ステップ)** | 劣る | **優れる** (OTパスのため) |

### CosyVoice3での具体的な使われ方

CosyVoice3ではFlow Matchingが「**音声トークン → メルスペクトログラム**」変換に使用されます:

```
入力条件:
  - speech_tokens: (B, T_speech) → Embedding → 射影 → 補間 → mu: (B, 80, T_mel)
  - speaker_embedding: (B, 192) → Linear → spks: (B, 80)

Flow Matching:
  - 学習: MSE(v_θ(x_t, t, mu, spks), x1 - (1-σ)z)
  - 推論: z ~ N(0,I) → Euler ODE 10ステップ → mel: (B, 80, T_mel)

後段:
  - mel → HiFT Vocoder → waveform: (B, T_audio)
```

条件情報 (mu, spks) は DiT の入力融合モジュール `InputEmbedding` で x_t と結合され、AdaLayerNormZero で時刻 t が注入されます。これにより DiT は「この時刻 t でのこの条件での速度はいくつか」を予測します。

### 実装対応

| 概念 | 実装箇所 |
|------|---------|
| ODE求解 (推論) | [flow_matching.py:367-450](flow_matching.py#L367-L450) の `inference()` |
| 損失計算 (学習) | [flow_matching.py:298-364](flow_matching.py#L298-L364) の `compute_loss()` |
| DiT (速度場推定器) | [flow_matching.py:520-603](flow_matching.py#L520-L603) の `DiT.forward()` |
| 時刻埋め込み | [flow_matching.py:682-713](flow_matching.py#L682-L713) の `TimestepEmbedding` |
| AdaLN変調 | [flow_matching.py:746-773](flow_matching.py#L746-L773) の `AdaLayerNormZero` |

---

## 📈 実験結果

### SEED-TTS-Eval (コンテンツ一貫性)

| モデル | test-zh CER↓ | test-en WER↓ | test-hard CER↓ |
|--------|-------------|-------------|----------------|
| Human | 1.26 | 2.14 | - |
| MaskGCT | 2.27 | 2.62 | 10.27 |
| Seed-TTS | 1.12 | 2.25 | 7.59 |
| CosyVoice 2 | 1.45 | 2.57 | 6.83 |
| **CosyVoice 3-0.5B+DiffRO** | **0.75** | **1.76** | **5.09** |
| **CosyVoice 3-1.5B+DiffRO** | **0.71** | **1.45** | **5.66** |

### MOS評価 (主観品質)

| モデル | 中国語 MOS | 英語 MOS | 平均 MOS |
|--------|-----------|---------|---------|
| Human | 4.66 | 4.50 | 4.58 |
| CosyVoice 2 | 4.47 | 4.25 | 4.36 |
| CosyVoice 3-0.5B | 4.48 | 4.36 | 4.42 |
| **CosyVoice 3-1.5B** | **4.46** | **4.43** | **4.45** |

---

## 🤔 よくある質問

### Q1: 音声トークンの語彙サイズ6561はどう決まる?
**A**: FSQのパラメータ D (低ランク次元) と K (量子化レベル) から `(2K+1)^D` で決定されます。実装では `(2K+1)^D = 6561` となるパラメータ (例: K=4, D≈5 で 9^5=59049、実際は別のD,K値) が使われます。

### Q2: なぜ25Hzのフレームレート?
**A**: 音声の言語的構造 (音素) は通常20-50ms程度で変化するため、40ms (25Hz) のフレームレートで十分な情報を保持できます。高すぎるとLLMの生成シーケンスが長くなり推論コストが増大します。

### Q3: token_mel_ratio=2の意味は?
**A**: 音声トークン1つがメルスペクトログラム2フレームに対応します。25Hz→50Hzの変換で、メルの時間解像度はトークンの2倍です。この不一致は単純なnearest補間で解決されます。

### Q4: Classifier-Free Guidance (CFG) の役割は?
**A**: 推論時に条件付き/無条件の速度場を混合して品質を向上させます。`v = (1+0.7)*v_cond - 0.7*v_uncond` のように、条件付き予測を強調することで、トークン特徴に忠実なメル生成を実現します。

### Q5: DiffROはなぜGumbel-Softmaxを使う?
**A**: LLMの出力は離散トークンですが、離散サンプリングは勾配を伝播できません。Gumbel-Softmaxは連続的な近似分布を生成しつつ、温度パラメータで離散性を制御できるため、報酬モデルへの微分可能な経路を確保できます。

### Q6: Long Skip Connectionsの効果は?
**A**: DiTの前半レイヤーの出力を後半レイヤーに直接接続し、U-Netライクな情報の流れを実現します。低レベル特徴 (細かい音響情報) を高レベル特徴と融合させ、メルスペクトログラムの再構成品質を向上させます。

### Q7: NSF (Neural Source Filter) はなぜ必要?
**A**: 音声の物理モデル (声帯振動→声道フィルタリング) を模倣する設計です。F0に基づくサイン波の源信号を生成し、ニューラルネットワークでフィルタリングすることで、特にピッチが明確な有声音の品質が大幅に向上します。

### Q8: RAS (Repetition-Aware Sampling) の目的は?
**A**: LLMの自己回帰生成では同じトークンパターンの繰り返し (ループ) が発生しやすい問題があります。RASは直近10トークンの繰り返しを検出してペナルティを与え、自然なバリエーションのある音声生成を促進します。

### Q9: Pronunciation Inpaintingとは?
**A**: 多音字 (例: 中国語の「给」) の誤読を防ぐため、テキスト中にピンインやARPABETを直接埋め込む手法です。例: `报道[j][ǐ]予好评` で「给予」の「给」を正しく「jǐ」と読ませます。RepMono+MixPhnで100%の修正率を達成。

### Q10: ポリグロット話者とは?
**A**: 元々1言語しか話さない話者を、ファインチューニングにより9言語で合成可能にする機能です。補助データセット (ランダム話者の各言語データ + 話者ID/言語IDの自然言語指示) を混合して継続学習することで実現します。

---

## 📚 参考資料

### CosyVoice3関連
- CosyVoice 3 論文: [arXiv:2505.17589](https://arxiv.org/abs/2505.17589)
- CosyVoice 公式実装: [github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- CosyVoice 3 公式ページ: [funaudiollm.github.io/cosyvoice3](https://funaudiollm.github.io/cosyvoice3/)

### 関連技術
- **Qwen2**: "Qwen2 Technical Report" (Alibaba, 2024)
- **MinMo**: 大規模マルチモーダル音声理解モデル (140万時間事前学習)
- **Matcha-TTS**: "Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching" (ICASSP 2024)
- **DiT**: "Scalable Diffusion Models with Transformers" (ICCV 2023)
- **HiFi-GAN**: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (NeurIPS 2020)
- **Gumbel-Softmax**: "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017)
- **FSQ**: "Finite Scalar Quantization: VQ-VAE Made Simple" (ICLR 2024)

---

## ✨ まとめ

このリポジトリは、CosyVoice3の複雑な実装を理解するための教育的な疑似コードです。

**カバー範囲:**
1. **全体パイプライン**: main_flow.py - テキスト→音声の5ステージ
2. **音声トークナイザ**: speech_tokenizer.py - FSQ-MinMoによる離散化
3. **言語モデル**: llm.py - Qwen2ベースの自己回帰生成
4. **フローマッチング**: flow_matching.py - DiTベースのメル生成
5. **ボコーダ**: vocoder.py - HiFT + NSFによる波形合成
6. **後処理**: diffro.py - DiffROによるRL最適化

**CosyVoice3の主要な強み:**
- 100万時間データで学習した多言語 (9言語+18方言) 音声合成
- DiffROにより発話内容の正確性が大幅に向上 (CER 0.71%)
- ゼロショット話者クローニングで人間に匹敵する品質 (MOS 4.45)
- ストリーミング推論対応 (因果的アーキテクチャ)
