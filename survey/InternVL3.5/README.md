# InternVL3.5 理解ガイド

## 概要

InternVL3.5 は上海AI研究所 (Shanghai AI Lab) が開発したオープンソースのマルチモーダル大規模言語モデル (MLLM) シリーズです。前バージョン InternVL3 から以下の3つの主要革新を導入しています：

1. **Cascade RL** - オフラインRL (MPO) → オンラインRL (GSPO) の2段階強化学習
2. **Visual Resolution Router (ViR)** - 視覚トークン数を動的に削減する解像度ルーター
3. **Decoupled Vision-Language Deployment (DvD)** - ViTとLLMを別GPUに分離したデプロイ戦略

これらにより前バージョン比でReasoningスコア **+16.0%** 向上・推論速度 **4.05倍** を達成します。

---

## ファイル構成

```
InternVL3.5_understanding/
├── README.md                      # 本ファイル（詳細解説）
├── model_architecture.py          # InternViT / MLP projector / InternVLChatModel
├── dynamic_resolution.py          # Dynamic High Resolution + Pixel Shuffle
├── cascade_rl.py                  # Cascade RL (MPO offline + GSPO online)
├── visual_resolution_router.py    # ViR + ViCO (Visual Consistency Learning)
├── main_flow.py                   # 全体フォワードパス
└── finetuning_example.py          # DataFrameベースのファインチューニングサンプル
```

---

## アーキテクチャ全体図

```
入力画像 (H_orig, W_orig, 3)
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  Dynamic High Resolution (動的高解像度タイリング)        │
│  448×448 パッチに分割 → 最大 max_dynamic_patch 枚       │
│  サムネイル1枚 + タイルN枚 = 合計 (N+1) パッチ          │
└──────────────────────────────────────────────────────┘
         │  pixel_values: (N_patches, 3, 448, 448)
         ▼
┌──────────────────────────────────────────────────────┐
│  InternViT-6B / InternViT-300M (Vision Encoder)      │
│  patch_size=14 → 1024 トークン/パッチ                  │
│  48層 or 24層 Transformer + QK Normalization          │
└──────────────────────────────────────────────────────┘
         │  vit_embeds: (N_patches, 1025, D_vit)  ← [CLS] + 1024 patch tokens
         │  CLS除去後:  (N_patches, 1024, D_vit)
         ▼
┌──────────────────────────────────────────────────────┐
│  Pixel Shuffle (空間方向トークン圧縮)                    │
│  1024 → 256 トークン/パッチ (downsample_ratio=0.5)     │
│  InternVL3.5-Flash: さらに 256 → 64 (ViR選択時)        │
└──────────────────────────────────────────────────────┘
         │  (N_patches, 256, D_vit*4)
         ▼
┌──────────────────────────────────────────────────────┐
│  MLP Projector (mlp1)                                │
│  LayerNorm → Linear → GELU → Linear                  │
│  D_vit*4 → D_llm                                    │
└──────────────────────────────────────────────────────┘
         │  vit_embeds: (N_patches, 256, D_llm)
         ▼
┌──────────────────────────────────────────────────────┐
│  LLM Embedding空間への埋め込み                          │
│  テキストトークン列の <IMG_CONTEXT> を視覚特徴で置換      │
└──────────────────────────────────────────────────────┘
         │  input_embeds: (B, L_total, D_llm)
         ▼
┌──────────────────────────────────────────────────────┐
│  Language Model (Qwen3 / GPT-OSS)                    │
│  自己回帰デコーダー                                     │
│  Dense: 1B / 8B / 14B / 38B                          │
│  MoE:  20B-A4B / 30B-A3B / 241B-A28B                │
└──────────────────────────────────────────────────────┘
         │  logits: (B, L_total, V)  V=語彙サイズ
         ▼
      テキスト出力
```

---

## 1. モデルアーキテクチャ詳細

### 1.1 ViT-MLP-LLM パラダイム

InternVL3.5 は "ViT-MLP-LLM" パラダイムを採用しています。

| モデル          | Vision Encoder      | LLM Backbone     | 合計パラメータ |
|----------------|---------------------|------------------|-------------|
| InternVL3.5-1B  | InternViT-300M      | Qwen3-0.6B       | ~1B         |
| InternVL3.5-2B  | InternViT-300M      | Qwen3-1.7B       | ~2B         |
| InternVL3.5-4B  | InternViT-300M      | Qwen3-4B         | ~4B         |
| InternVL3.5-8B  | InternViT-300M      | Qwen3-8B         | ~8B         |
| InternVL3.5-14B | InternViT-6B        | Qwen3-14B        | ~14B        |
| InternVL3.5-38B | InternViT-6B        | Qwen3-32B        | ~38B        |
| InternVL3.5-20B-A4B | InternViT-300M  | GPT-OSS-20B(MoE) | ~20B        |
| InternVL3.5-30B-A3B | InternViT-6B    | GPT-OSS-30B(MoE) | ~30B        |
| InternVL3.5-241B-A28B | InternViT-6B  | Qwen3-235B(MoE)  | ~241B       |

### 1.2 InternViT (Vision Encoder)

**InternViT-6B 設定:**
- `image_size`: 448 (Dynamic High Resolution 後の各パッチサイズ)
- `patch_size`: 14
- `hidden_size`: 3200
- `num_attention_heads`: 25
- `intermediate_size`: 12800
- `num_hidden_layers`: 48
- `qk_normalization`: True ← Q/K に RMSNorm を適用 (学習安定化)
- `norm_type`: 'rms_norm'

**トークン数計算:**
```
n_patch_tokens = (image_size / patch_size)^2
               = (448 / 14)^2 = 32^2 = 1024 トークン/パッチ
```

**QK Normalization:**
標準的な Vision Transformer に追加した重要な安定化技術です。QとKにRMSNormを適用することで、大規模モデル学習時の attention score の発散を防ぎます。

```
Q_norm = RMSNorm(Q)   # (B, H, N, D_head) → (B, H, N, D_head)
K_norm = RMSNorm(K)   # (B, H, N, D_head) → (B, H, N, D_head)
attn = softmax(Q_norm @ K_norm^T / sqrt(D_head))
```

### 1.3 Pixel Shuffle (トークン圧縮)

ViTの出力 1024 トークンをLLMに渡す前に圧縮します。

```
入力: (N, H_t, W_t, C)    # H_t=W_t=32, C=D_vit=3200
  ↓ view(N, W_t, H_t*s, C/s)  s=downsample_ratio=0.5
  → (N, 32, 16, 6400)
  ↓ permute(0,2,1,3)
  → (N, 16, 32, 6400)
  ↓ view(N, 16, 16, 12800)
  → 出力: (N, 256, 12800)  # 256=16*16 トークン/パッチ
```

### 1.4 MLP Projector

```
入力: (N_patches*256, D_vit*4) = (N_patches*256, 12800)
  LayerNorm(12800)
  Linear(12800 → D_llm)
  GELU
  Linear(D_llm → D_llm)
出力: (N_patches*256, D_llm)
```

---

## 2. Dynamic High Resolution (動的高解像度)

### 2.1 タイリング戦略

高解像度画像をモデルに入力するため、画像を448×448のタイルに分割します。

```
元画像: (H_orig, W_orig, 3)
  ↓ アスペクト比を保ちながら最適なグリッドサイズを選択
    例: 1344×672 → 3×2 グリッド
  ↓ 各タイルを 448×448 にリサイズ
  ↓ サムネイル画像 (1枚) を全体コンテキストとして追加
  ↓ 全パッチを結合
出力: (N_patches, 3, 448, 448)
  N_patches = タイル数 + 1(サムネイル)
  最大 max_dynamic_patch + 1 枚 (デフォルト max=6 → 最大7パッチ)
```

### 2.2 <img> トークン列の構築

各パッチはLLMに対して以下のトークン列として入力されます：

```
<img>  [IMG_CONTEXT × 256]  </img>   ← 1パッチあたり
```

N パッチの画像の場合:
```
<img>  [IMG_CONTEXT × (256 × N_patches)]  </img>
```

---

## 3. Cascade Reinforcement Learning

### 3.1 なぜ強化学習か？

SFT（教師あり学習）の限界：
- 正解データのみを学習 → 誤答パターンを明示的に抑制できない
- 多様なサンプリング軌跡を探索できない

RL の利点：
- **負例サンプル**を活用して誤答領域を明示的に排除
- 報酬信号により探索空間を自律的に制御

### 3.2 Cascade RL の2段階

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Offline RL (MPO - Mixed Preference Optimization) │
│                                                          │
│  ・既存ロールアウト (MMPR-v1.2 ~200K ペア) を使用         │
│  ・ロールアウト収集とパラメータ更新を分離                  │
│  ・報酬ハッキングを防止 (安定性↑)                         │
│  ・複数モデルでロールアウトを共有 → コスト削減             │
│                                                          │
│  損失: L_MPO = wp*L_DPO + wq*L_BCO + wg*L_LM            │
└─────────────────────────────────────────────────────────┘
         │  高品質な初期モデル
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Online RL (GSPO - Group Sampling Policy Opt.)  │
│                                                          │
│  ・MPO後の強化された初期モデルからリアルタイムにサンプリング │
│  ・MMPR-Tiny (~70K クエリ) を使用                        │
│  ・参照モデル制約なし → 密・MoEモデルで効果的              │
│  ・GRPOと同様にグループ正規化アドバンテージを使用           │
│                                                          │
│  Ā_i = (r(x,y_i) - mean(r)) / std(r)                    │
└─────────────────────────────────────────────────────────┘
```

### 3.3 MPO 損失の詳細

```
L_MPO = wp * L_preference + wq * L_quality + wg * L_generation

L_preference: DPO損失 (選好ペア学習)
L_quality:    BCO損失 (品質フィルタ)
L_generation: LM損失  (生成能力維持)
```

### 3.4 GSPO 損失の詳細

GRPOと類似しますが、重要サンプリング比に **幾何平均** を使用（per-token比のexp平均）：

```
s_i(θ) = exp(1/|y_i| * Σ_t log(π_θ(y_i,t|x) / π_ref(y_i,t|x)))

L_GSPO = E[1/G * Σ_i min(s_i(θ) * Ā_i, clip(s_i(θ), 1-ε, 1+ε) * Ā_i)]
```

GRPOとの違い: `π_ref` を参照モデルではなく同バッチ内の旧ポリシーとして扱い、参照モデル制約 (KL penalty) を除去しています。

---

## 4. Visual Resolution Router (ViR)

### 4.1 概要

InternVL3.5-Flash ではViRにより各パッチを動的に圧縮率選択します：

```
パッチルーター (binary classifier)
  ├─ 0 (低圧縮): 256 tokens/patch  ← 視覚情報が豊富なパッチ
  └─ 1 (高圧縮):  64 tokens/patch  ← 単純なパッチ
```

平均50%のトークン削減で性能はほぼ100%維持。

### 4.2 ViCO (Visual Consistency Learning) 2段階

**Stage 1 - 一貫性学習:**

256トークン表現と64トークン表現の出力分布の乖離をKL divergenceで最小化：

```
L_ViCO = E_ξ [1/N * Σ_i KL(π_θ,ξ(y|I_ξ) || π_θ_prior(y|I_256))]

ξ ∈ {1/4 (256tokens), 1/16 (64tokens)}  # ランダムに選択
π_θ_prior: 凍結された InternVL3.5 (参照モデル)
```

**Stage 2 - ルーター学習:**

各パッチに対する圧縮の影響度を測定し、二値分類器を学習：

```
r_i = L_ViCO(y_i | I_64) / L_ViCO(y_i | I_256)   # 損失比率

y_router = {
  0 (低圧縮 → 256tokens): r_i < τ    # 圧縮の影響が小さい
  1 (高圧縮 →  64tokens): r_i ≥ τ    # 圧縮の影響が大きい
}

τ: スライディングウィンドウのk-パーセンタイル (動的閾値)
```

---

## 5. Decoupled Vision-Language Deployment (DvD)

```
┌─────────────────────────────────────────────┐
│  Vision Server                              │
│  ┌──────────┐   ┌─────┐   ┌───────────┐   │
│  │InternViT │ → │ MLP │ → │  ViR      │   │
│  └──────────┘   └─────┘   └───────────┘   │
│           (並列処理・バッチ可能)              │
└─────────────────┬───────────────────────────┘
                  │  BF16 視覚特徴 (TCP/RDMA)
                  ▼
┌─────────────────────────────────────────────┐
│  Language Server                            │
│  ┌──────────────────────────────────────┐  │
│  │  LLM (Prefill + Decode)             │  │
│  └──────────────────────────────────────┘  │
│     (視覚計算に邪魔されず推論に専念)          │
└─────────────────────────────────────────────┘
```

非同期3段パイプライン: Vision処理 → 特徴転送 → LLM処理 がオーバーラップ実行。

---

## 6. 学習フロー (全体)

```
Stage 0: Pre-Training
  ├─ 全パラメータを更新 (ViT + MLP + LLM)
  ├─ 116M サンプル / ~250B トークン
  ├─ テキスト:マルチモーダル比 ≈ 1:2.5
  ├─ NTP損失 + Square Averaging (weight = 1/N^0.6)
  └─ 最大コンテキスト長 32K

Stage 1: Supervised Fine-Tuning (SFT)
  ├─ 高品質な会話データで全パラメータ更新
  ├─ ~56M サンプル / ~130B トークン
  └─ 新スキル: GUI操作, Embodied, SVG理解・生成

Stage 2: Cascade RL
  ├─ Offline RL (MPO): MMPR-v1.2 ~200K ペア
  └─ Online RL (GSPO): MMPR-Tiny ~70K クエリ

Stage 3: ViCO (Flash variant only)
  ├─ 一貫性学習: SFTデータでKL divergence最小化
  └─ ルーター学習: OCR/QVAデータで分類器訓練
```

---

## 7. テスト時スケーリング (Test-Time Scaling)

### Deep Thinking (推論深度)
"Thinking"モードを有効化することで、段階的な推論プロセス (CoT) を強制します。

### Parallel Thinking (推論幅)
Best-of-N (BoN) 戦略: N個の回答を生成し **VisualPRM-v1.1** (報酬モデル) で最良選択。

---

## 8. 主要ベンチマーク結果

| ベンチマーク      | InternVL3.5-38B | InternVL3.5-241B-A28B | GPT-5  |
|-----------------|-----------------|----------------------|--------|
| MMMU            | 73.4            | 77.7                 | 81.6   |
| MathVista       | —               | 82.7                 | —      |
| MMVet           | —               | —                    | —      |
| OCRBench        | —               | —                    | —      |
| ScreenSpot      | —               | —                    | —      |
| AIME24          | —               | —                    | —      |

InternVL3.5-241B-A28B は GPT-5 との差を **3.9%** まで縮小（一般マルチモーダルタスク）。

---

## 9. 各ファイルの概要

| ファイル | 内容 |
|---------|------|
| [model_architecture.py](model_architecture.py) | InternViT, MLP Projector, InternVLChatModel の実装 |
| [dynamic_resolution.py](dynamic_resolution.py) | 動的高解像度タイリング + Pixel Shuffle 実装 |
| [cascade_rl.py](cascade_rl.py) | MPO (offline RL) と GSPO (online RL) の学習ロジック |
| [visual_resolution_router.py](visual_resolution_router.py) | ViR + ViCO (一貫性学習 + ルーター学習) |
| [main_flow.py](main_flow.py) | 画像→テキスト生成の全フォワードパス |
| [finetuning_example.py](finetuning_example.py) | DataFrameベースのPyTorchファインチューニング例 |

---

## 参考

- 論文: InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency
- 公式コード: https://github.com/OpenGVLab/InternVL
- モデル: https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B
