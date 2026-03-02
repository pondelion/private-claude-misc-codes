# Kronos Understanding - 簡略化疑似コード集

Kronos (金融K線データのための基盤モデル) の理解を目的とした簡略化疑似コード集です。

論文: [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)

## 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [Kronosの主要イノベーション](#kronosの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**Kronosの特徴:**
- **金融特化基盤モデル**: K線(ローソク足)データ専用に設計された初の大規模事前学習モデル
- **離散化トークナイザ**: Binary Spherical Quantization (BSQ)で連続的なOHLCVAデータを階層的離散トークンに変換
- **階層的Coarse-to-Fine予測**: 粗粒度サブトークン(s1)→細粒度サブトークン(s2)の2段階自己回帰生成
- **大規模事前学習**: 45取引所、12億本超のK線レコード、7種類の時間粒度で学習
- **Test-Time Scaling**: 複数サンプルパスの平均化で推論精度を向上

**タスク:**
- 価格系列予測 (Price Series Forecasting)
- リターン予測 (Return Forecasting)
- 実現ボラティリティ予測 (Realized Volatility Forecasting)
- 合成K線生成 (Synthetic K-line Generation)
- 投資シミュレーション (Investment Simulation)

**性能:**
- 価格予測 RankIC: 既存TSFMの93%向上、非事前学習モデルの87%向上
- ボラティリティ予測 MAE: 9%低減
- 合成K線 忠実度: 22%向上

**モデルファミリー:**

| モデル | レイヤー数 | d_model | d_ff | ヘッド数 | 語彙(2^k) | パラメータ |
|--------|-----------|---------|------|---------|-----------|-----------|
| Kronos_small | 8 | 512 | 1024 | 8 | 20 | 24.7M |
| Kronos_base | 12 | 832 | 2048 | 16 | 20 | 102.3M |
| Kronos_large | 18 | 1664 | 3072 | 32 | 20 | 499.2M |

---

## アーキテクチャ全体像

```
入力K線系列 x = (x_1, ..., x_T)
各 x_t ∈ R^6: [Open, High, Low, Close, Volume, Amount]

===== Phase 1: K-line Tokenization (KronosTokenizer) =====

x_t (B, T, 6)  ──正規化──→  x_norm (B, T, 6)
                              ↓
┌──────────────────────────────────────────┐
│ Tokenizer Encoder (3層 Transformer)       │
│   Linear(6, 256) → 3x TransformerBlock   │
│   → Linear(256, 20)                       │
│   入力: (B, T, 6)                         │
│   出力: (B, T, 20)  ← codebook_dim       │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Binary Spherical Quantization (BSQ)       │
│   L2正規化 → sign関数で二値化 {-1, +1}   │
│   → スケーリング (1/√20)                 │
│                                           │
│   20ビット → s1 (10ビット) + s2 (10ビット)│
│   語彙サイズ: 2^10 = 1024 (各サブトークン)│
│   入力: (B, T, 20)                        │
│   出力: s1_indices (B, T), s2_indices (B, T)│
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Tokenizer Decoder (3層 Transformer)       │
│   階層的再構成:                           │
│   ・Coarse (s1のみ): Linear(10, 256)      │
│     → 3x TransformerBlock → Linear(256,6) │
│   ・Fine (s1+s2全体): Linear(20, 256)     │
│     → 3x TransformerBlock → Linear(256,6) │
│   出力: x̂_coarse (B, T, 6), x̂ (B, T, 6) │
└──────────────────────────────────────────┘

損失: L_tokenizer = L_coarse + L_fine + λ * L_quant

===== Phase 2: Autoregressive Pre-training (Kronos) =====

トークン列 b = (b_1, ..., b_T), b_t = [s1_t, s2_t]

┌──────────────────────────────────────────┐
│ HierarchicalEmbedding                     │
│   emb_s1(s1_ids): (B, T) → (B, T, d)     │
│   emb_s2(s2_ids): (B, T) → (B, T, d)     │
│   fusion: cat → Linear(2d, d)             │
│   出力: (B, T, d_model)                   │
│                                           │
│ + TemporalEmbedding                       │
│   minute + hour + weekday + day + month   │
│   各: Embedding → (B, T, d_model)         │
│   合計して加算                             │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Causal Transformer (N層)                  │
│   各層: RMSNorm → CausalSelfAttn(RoPE)   │
│        → RMSNorm → SwiGLU FFN            │
│   入力: (B, T, d_model)                   │
│   出力: h = (B, T, d_model)              │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Coarse Subtoken Prediction (s1)           │
│   h → Linear(d_model, 2^10)              │
│   s1_logits: (B, T, 1024)               │
│   → サンプリング → ŝ1_t                  │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ DependencyAwareLayer (Cross-Attention)    │
│   query = emb_s1(ŝ1_t): (B, T, d_model) │
│   key, value = h: (B, T, d_model)        │
│   → CrossAttn(RoPE) + Residual + Norm    │
│   出力: h_update (B, T, d_model)         │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Fine Subtoken Prediction (s2)             │
│   h_update → Linear(d_model, 2^10)       │
│   s2_logits: (B, T, 1024)               │
└──────────────────────────────────────────┘

損失: L_ar = -Σ [log p(s1_t | b_{<t}) + log p(s2_t | b_{<t}, s1_t)]
```

---

## ファイル構成

### 1. [tokenizer.py](tokenizer.py) (K-line Tokenizer)
**K線データの離散化: Transformer Autoencoder + BSQ**

KronosTokenizerの全体構造を簡略化して記述。連続的なOHLCVA値を階層的な離散トークンに変換する。

```python
class KronosTokenizer(nn.Module):
    def forward(self, x):
        # x: (B, T, 6) → (z_pre, z): 再構成, bsq_loss, quantized, z_indices
```

**重要ポイント:**
- Encoder/Decoderは各3層のTransformerBlock (d_model=256, 4ヘッド, ff_dim=512)
- BSQで20ビットバイナリコードに量子化 → s1(10bit) + s2(10bit)に分割
- 階層的再構成損失: Coarse(s1のみ)とFine(s1+s2全体)の2段階

---

### 2. [autoregressive_model.py](autoregressive_model.py) (Kronos本体)
**階層的自己回帰Transformer**

トークン化されたK線系列を入力として、次のタイムステップのトークンを階層的に予測する。

```python
class Kronos(nn.Module):
    def forward(self, s1_ids, s2_ids, stamp=None):
        # s1_ids, s2_ids: (B, T) → s1_logits: (B, T, 1024), s2_logits: (B, T, 1024)
```

**重要ポイント:**
- HierarchicalEmbedding: s1とs2の埋め込みをfusion_proj(Linear(2d, d))で結合
- TemporalEmbedding: minute/hour/weekday/day/monthの5つの時間特徴
- Causal Self-Attention + RoPE (Rotary Position Embedding)
- DependencyAwareLayer: Cross-Attentionでs2予測をs1に条件付け
- DualHead: s1用とs2用の独立した線形ヘッド

---

### 3. [inference.py](inference.py) (推論パイプライン)
**自己回帰推論 + Monte Carloロールアウト**

事前学習済みモデルによる未来K線の生成。複数サンプルの平均化でロバストな予測を実現。

```python
def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, ...):
    # x: (B, T, 6) → preds: (B, T+H, 6) (H: 予測長)
```

**重要ポイント:**
- スライディングウィンドウ方式でコンテキスト長を管理 (最大512)
- s1→s2の逐次生成 (Coarse-to-Fine)
- Temperature + Top-p (nucleus) サンプリングで確率的生成
- sample_count個のパスを生成し平均化 (Test-Time Scaling)

---

### 4. [training.py](training.py) (学習パイプライン)
**2段階学習: Tokenizer → Predictor**

Tokenizer学習とPredictor学習の両方のパイプラインを記述。

```python
def train_tokenizer(tokenizer, dataset, ...):
    # 再構成損失 + BSQ損失で学習

def train_predictor(model, tokenizer, dataset, ...):
    # Cross-Entropy損失で自己回帰学習
```

**重要ポイント:**
- Tokenizerは再構成品質最適化 (L_coarse + L_fine + λ*L_quant)
- Predictorはトークン予測精度最適化 (CE_s1 + CE_s2)
- 学習時: s1はサンプリング (teacher forcingではなく)
- AdamW + Cosine LR Schedule + Linear Warmup

---

## Kronosの主要イノベーション

### 1. **Binary Spherical Quantization (BSQ) によるK線トークン化**
**問題**: 連続的な金融時系列データをLLMのような自己回帰モデルで扱えない
**解決**:
- 各K線アイテム (OHLCVA) をTransformerエンコーダで特徴抽出
- L2正規化後、sign関数でk=20ビットのバイナリコードに量子化
- 実効語彙サイズ 2^20 ≈ 100万だが、2分割して2^10 × 2^10 = 1024 × 1024に

**従来手法との比較**:
- Chronos: スケーリング + 均一量子化 → 金融データの微細構造を損失
- TOTEM: VQ-VAE → コードブック崩壊のリスク
- **Kronos (BSQ)**: 球面射影 → 量子化誤差の上界が保証、ヘビーテールに強い

**数学的保証**:
```
E_a ||u - û|| < √(2 - 2/√L) < √2
```
(Lはコードブック次元。次元が増えるほど誤差上界が縮小)

**実装**: [tokenizer.py](tokenizer.py)

---

### 2. **階層的Coarse-to-Fine トークン予測**
**問題**: 2^20の巨大語彙に対する直接予測は計算コスト・パラメータが膨大
**解決**:
- 20ビットトークンをs1 (10ビット, coarse) + s2 (10ビット, fine) に分割
- s1を先に予測 → s1に条件付けてs2を予測
- 語彙パラメータ: 2^20 → 2×2^10 = 2048 (99.8%削減)

**分割の効果 (Kronos_base, 約97.5Mコアパラメータ)**:

| 分割数 n | Sub-Vocab | Vocabパラメータ | 総パラメータ | 推論ステップ |
|----------|-----------|---------------|------------|-------------|
| 1 (分割なし) | 1,048,576 | 1744.8M | 1842.3M | 1× |
| **2 (Kronos)** | **1,024** | **3.4M** | **102.3M** | **2×** |
| 4 | 32 | 0.2M | 100.5M | 4× |

**実装**: [autoregressive_model.py](autoregressive_model.py)

---

### 3. **DependencyAwareLayer (条件付きs2生成)**
**問題**: s1とs2を並列予測すると精度が低下 (Kronos-Parallel ablation: 性能劣化)
**解決**:
- s1予測後、サンプリングしたŝ1の埋め込みをqueryとしてCross-Attention
- Transformerの隠れ状態をkey/valueとして使用
- これによりs2生成がs1に明示的に条件付けられる

```python
# Cross-Attention
query = emb_s1(ŝ1)           # (B, T, d_model)
key = value = h              # (B, T, d_model) Transformerの出力
h_update = CrossAttn(q, k, v) + h  # Residual接続
s2_logits = proj_s2(h_update)
```

**訓練時の工夫**: Ground-truth s1ではなくサンプリングしたs1を使用 → exposure biasの軽減

**実装**: [autoregressive_model.py](autoregressive_model.py)

---

### 4. **TemporalEmbedding (時間的文脈)**
**問題**: 金融市場は日中パターン、週次、月次の周期性を持つ
**解決**:
- 5種類の時間特徴: minute, hour, weekday, day, month
- 各特徴を独立したEmbedding層でd_modelに射影
- 全て加算してトークン埋め込みに足し込み

**効果**: 同じ価格パターンでも時間帯によって異なる意味を持つことをモデルが学習可能

**実装**: [autoregressive_model.py](autoregressive_model.py)

---

### 5. **Test-Time Scaling (推論時精度向上)**
**問題**: 確率的サンプリングは1回のパスでは不安定
**解決**:
- 同じコンテキストから複数回 (N回) サンプリング
- N個の予測パスをトークナイザでデコード後、平均化
- Nを増やすほどIC/RankICが改善

**効果** (Figure 7):
- N=1 → N=20: 価格予測IC約30%改善
- 計算コストはN倍だが予測精度との良好なトレードオフ

**実装**: [inference.py](inference.py)

---

## 処理フロー詳細

### 事前学習フロー

```python
# ===== Stage 1: Tokenizer学習 =====
# 入力: 正規化済みK線データ
x = normalize(raw_kline)              # (B, T, 6) OHLCVA

# Forward
(x_coarse, x_recon), bsq_loss, quantized, z_indices = tokenizer(x)
# x_coarse: (B, T, 6) - s1のみからの再構成
# x_recon:  (B, T, 6) - s1+s2全体からの再構成

# Loss
L_coarse = MSE(x, x_coarse)          # 粗粒度再構成
L_fine   = MSE(x, x_recon)           # 高精度再構成
L_quant  = bsq_loss                   # commit + entropy
L_total  = L_coarse + L_fine + λ * L_quant

# ===== Stage 2: Predictor学習 =====
# Tokenizer凍結 → トークン化
s1_ids, s2_ids = tokenizer.encode(x, half=True)
# s1_ids: (B, T)  各値 ∈ [0, 1023]
# s2_ids: (B, T)  各値 ∈ [0, 1023]

# Stamp作成
stamp = calc_time_stamps(timestamps)   # (B, T, 5): [min, hour, wday, day, month]

# Forward (自己回帰: 入力[0:T-1] → 予測[1:T])
s1_logits, s2_logits = model(s1_ids[:, :-1], s2_ids[:, :-1], stamp[:, :-1])
# s1_logits: (B, T-1, 1024)
# s2_logits: (B, T-1, 1024)

# Loss
CE_s1 = CrossEntropy(s1_logits, s1_ids[:, 1:])
CE_s2 = CrossEntropy(s2_logits, s2_ids[:, 1:])
L_ar = (CE_s1 + CE_s2) / 2
```

---

### 推論フロー

```python
# 入力: 履歴K線 + 時間情報
x = normalize(raw_kline)               # (B, T, 6)
x_stamp = calc_time_stamps(x_times)    # (B, T, 5)
y_stamp = calc_time_stamps(y_times)    # (B, H, 5)  H: 予測長

# Step 1: 複数サンプルに拡張
x = x.repeat(1, sample_count, 1, 1).reshape(-1, T, 6)
# (B*N, T, 6)  N: サンプル数

# Step 2: 履歴をトークン化
s1_ids, s2_ids = tokenizer.encode(x, half=True)
# s1_ids: (B*N, T), s2_ids: (B*N, T)

# Step 3: 自己回帰生成ループ
for i in range(H):
    # コンテキストウィンドウ管理 (最大512)
    window = buffer[:, :min(T+i, max_context)]
    stamp = full_stamp[:, start:T+i, :]

    # s1予測
    s1_logits, context = model.decode_s1(window_s1, window_s2, stamp)
    s1_logits = s1_logits[:, -1, :]              # (B*N, 1024)
    s1_logits = s1_logits / temperature
    s1_logits = top_p_filtering(s1_logits, top_p)
    s1_token = multinomial(softmax(s1_logits))   # (B*N, 1)

    # s2予測 (s1に条件付け)
    s2_logits = model.decode_s2(context, s1_token)
    s2_logits = s2_logits[:, -1, :]              # (B*N, 1024)
    s2_token = multinomial(softmax(s2_logits/T)) # (B*N, 1)

    # バッファ更新 (スライディングウィンドウ)
    buffer_s1 = roll_and_append(buffer_s1, s1_token)
    buffer_s2 = roll_and_append(buffer_s2, s2_token)

# Step 4: デコード + 平均化
full_tokens = cat(history_tokens, generated_tokens)
z = tokenizer.decode([full_s1, full_s2], half=True)
# z: (B*N, T+H, 6)

z = z.reshape(B, N, T+H, 6)
preds = z.mean(axis=1)  # Monte Carlo平均
# preds: (B, T+H, 6)

# Step 5: 逆正規化
preds = preds * (x_std + 1e-5) + x_mean
```

---

### 前処理フロー

```python
# 入力: 生K線データ
raw = [Open, High, Low, Close, Volume, Amount]  # 各 (T,)

# Step 1: Z-score正規化 (各特徴独立)
for d in range(D):   # D=6
    x[:, d] = (x[:, d] - mean_d) / (std_d + 1e-5)

# Step 2: クリッピング
x = clip(x, -5, 5)  # 外れ値対策

# Step 3: 時間特徴抽出
stamp = [minute, hour, weekday, day, month]  # 各整数値
# minute ∈ [0, 59], hour ∈ [0, 23], weekday ∈ [0, 6],
# day ∈ [1, 31], month ∈ [1, 12]
```

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | x (生データ) | `(B, T, 6)` | OHLCVA K線データ |
| | stamp | `(B, T, 5)` | 時間特徴 [min, hour, wday, day, month] |
| **Tokenizer Encoder** | embed出力 | `(B, T, 256)` | Linear(6→256) |
| | encoder出力 | `(B, T, 256)` | 3層Transformer |
| | quant入力 | `(B, T, 20)` | Linear(256→20) |
| **BSQ** | 量子化済み | `(B, T, 20)` | バイナリ値 ∈ {-1/√20, +1/√20} |
| | s1_indices | `(B, T)` | Coarseインデックス ∈ [0, 1023] |
| | s2_indices | `(B, T)` | Fineインデックス ∈ [0, 1023] |
| **Tokenizer Decoder** | x̂_coarse | `(B, T, 6)` | s1のみから再構成 |
| | x̂_fine | `(B, T, 6)` | s1+s2から再構成 |
| **Embedding** | s1_emb | `(B, T, d_model)` | Embedding(1024, d) |
| | s2_emb | `(B, T, d_model)` | Embedding(1024, d) |
| | fused | `(B, T, d_model)` | Linear(2d→d) |
| | time_emb | `(B, T, d_model)` | 5種類の合計 |
| **Transformer** | hidden | `(B, T, d_model)` | N層Causal Transformer出力 |
| **DualHead** | s1_logits | `(B, T, 1024)` | Coarseトークン分布 |
| **DependencyAwareLayer** | h_update | `(B, T, d_model)` | Cross-Attn更新済み |
| **DualHead (cond)** | s2_logits | `(B, T, 1024)` | Fineトークン分布 |
| **推論出力** | preds | `(B, H, 6)` | 予測K線 (逆正規化後) |

### 軸の意味

- **B**: バッチサイズ
- **T**: 系列長 (過去の時間ステップ数)
- **H**: 予測長 (未来の時間ステップ数)
- **6**: K線特徴数 (Open, High, Low, Close, Volume, Amount)
- **5**: 時間特徴数 (minute, hour, weekday, day, month)
- **d_model**: モデル次元 (512/832/1664, モデルサイズ依存)
- **20**: コードブック次元 (k = s1_bits + s2_bits = 10 + 10)
- **256**: トークナイザのモデル次元
- **1024**: サブトークン語彙サイズ (2^10)
- **N**: Monte Carloサンプル数 (Test-Time Scaling)

---

## FAQ

### Q1: なぜ連続値予測ではなく離散トークン予測?

**A**: 3つの理由があります。

**1. ノイズ抑制と安定性**:
```python
# 連続値予測 (Direct-AR)
pred = linear(h)  # MSE損失 → 外れ値に敏感、勾配爆発リスク

# 離散トークン予測 (Kronos)
logits = linear(h)  # CE損失 → 分類問題、外れ値の影響が限定的
```
- BSQの球面射影により量子化誤差の上界が保証される
- Flash Crash等の極端な値にもロバスト

**2. サンプル効率と汎化**:
- 類似パターンが同じトークンにマッピング → 各状態のサンプル数増加
- 希少なマーケットイベント (流動性ショック等) もトークン空間でモデル化可能

**3. 過学習抑制**:
- 量子化がノイズを除去する正則化として機能
- Ablation結果: 離散モデルが連続モデルを大幅に上回る (Table 2)

**実験結果 (Kronos_small規模)**:

| モデル | 予測空間 | 価格IC↑ | 価格RankIC↑ | Volatility MAE↓ |
|--------|---------|---------|------------|-----------------|
| Direct-AR | 連続 | 0.0212 | 0.0149 | 0.0565 |
| Prob-AR | 連続 | 0.0179 | 0.0102 | 0.0464 |
| **Kronos** | **離散** | **0.0431** | **0.0254** | **0.0384** |

---

### Q2: BSQの20ビットで金融データを十分に表現できる?

**A**: はい。コードブック使用率が高く、十分な表現力を確認しています。

**コードブック使用率** (Table 11):

| コードブック | サイズ | 使用率 |
|-------------|--------|--------|
| Coarse (s1) | 2^10 = 1024 | 97.66% |
| Fine (s2) | 2^10 = 1024 | 85.25% |

**語彙サイズの影響** (Figure 6):
- k=14 → k=20: 再構成MAEが約10%改善
- k=14 → k=20: 予測IC/RankICも一貫して改善
- 大きい語彙 = より精密な表現 → より良い下流タスク性能

**高頻度/低頻度/未使用トークンの意味**:
- 高頻度トークン → 安定した市場状態 (一般的なK線パターン)
- 低頻度トークン → 極端な市場状態 (長いヒゲ、高ボラティリティ)
- 未使用トークン → 極めて稀 or 非現実的なパターン

---

### Q3: s1/s2分割 (n=2) が最適な理由は?

**A**: パラメータ効率と推論レイテンシのバランスが最良だからです。

**パラメータ削減**:
```
n=1 (分割なし): 語彙パラメータ 1744.8M → 総パラメータ 1842.3M
n=2 (Kronos):   語彙パラメータ 3.4M    → 総パラメータ 102.3M  (99.8%削減!)
n=4:            語彙パラメータ 0.2M    → 総パラメータ 100.5M  (わずか2%追加削減)
```

**推論レイテンシ**:
- n=2: 各トークン生成に2ステップ
- n=4: 各トークン生成に4ステップ (2倍遅い)
- n=2→n=4: パラメータ削減2%に対しレイテンシ2倍 → 割に合わない

**結論**: n=2が「99.8%のパラメータ削減効果を得つつ、レイテンシ増加を最小限」

---

### Q4: Kronos-Parallelが劣る理由は?

**A**: s1とs2間の依存関係を無視するためです。

```python
# Kronos-Parallel (Ablation)
s1_logits = head_s1(h)  # (B, T, 1024)
s2_logits = head_s2(h)  # (B, T, 1024) ← hのみに依存、s1の情報なし

# Kronos (Sequential)
s1_logits = head_s1(h)           # (B, T, 1024)
ŝ1 = sample(s1_logits)
h_update = CrossAttn(emb(ŝ1), h)  # s1情報を注入
s2_logits = head_s2(h_update)    # (B, T, 1024) ← s1に条件付き
```

**結果 (Table 2)**:

| モデル | 価格IC↑ | 価格RankIC↑ |
|--------|---------|------------|
| Kronos-Parallel | 0.0345 | 0.0226 |
| **Kronos (Sequential)** | **0.0431** | **0.0254** |

s2はs1の「残差情報」を担うため、s1を知らないと正確に予測できない。

---

### Q5: TemporalEmbeddingはなぜ重要?

**A**: 金融市場の周期的パターンを捉えるためです。

**例**:
```
同じ価格パターンでも...
- 寄り付き直後 (9:30) → 高ボラティリティが正常
- 昼休み前 (11:30) → 出来高減少が正常
- 月末 → ポートフォリオリバランスの影響
```

**実装**:
```python
# 5つの独立したEmbedding
minute_emb = Embedding(60, d_model)   # 分: 0-59
hour_emb   = Embedding(24, d_model)   # 時: 0-23
wday_emb   = Embedding(7, d_model)    # 曜日: 0-6
day_emb    = Embedding(32, d_model)   # 日: 1-31
month_emb  = Embedding(13, d_model)   # 月: 1-12

# 全て加算
time_emb = minute + hour + weekday + day + month  # (B, T, d_model)
x = token_emb + time_emb  # トークン埋め込みに加算
```

---

### Q6: Temperature / Top-p の使い分けは?

**A**: タスクによって最適値が異なります (Table 6)。

| タスク | Temperature | Top-p | サンプル数 N |
|--------|------------|-------|-------------|
| 価格予測 | 0.6 | 0.90 | 10 |
| リターン予測 | 0.6 | 0.90 | 10 |
| ボラティリティ予測 | 0.9 | 0.90 | 1 |
| 合成K線生成 | 1.0 | 0.95 | 1 |
| 投資シミュレーション | 0.6 | 0.90 | 10 |

**原則**:
- **予測タスク (精度重視)**: 低温 (T≈0.6) → シャープな分布、高確信予測
- **生成タスク (多様性重視)**: 高温 (T≈1.0) → フラットな分布、多様なサンプル
- **サンプル数**: 予測タスクでN>1のMonte Carlo平均が有効

---

### Q7: 低品質データのフィルタリングはどうしている?

**A**: 3段階のフィルタリングパイプラインを適用しています (Algorithm 1)。

**Step 1: 構造的ブレイクによる分割**
```python
# 前日の終値と当日の始値の乖離をチェック
if |open_t / close_{t-1} - 1| > θ_jump:  # 例: 1min → 0.10
    split_here()  # 株式分割、契約ロールオーバー等を検出
```

**Step 2: 非流動性期間の除去**
```python
# 出来高がゼロ/ほぼゼロの連続バーをチェック
if consecutive_zero_volume_bars > θ_illiquid:  # 例: 1min → 15本
    flag_as_invalid()
```

**Step 3: 価格停滞期間の除去**
```python
# 終値が変わらない連続バーをチェック
if consecutive_same_close_bars > θ_stagnant:  # 例: 1min → 45本
    flag_as_invalid()
```

**最終段階**: 有効セグメントのみ保持、最小長要件を適用 (例: 1min → 2048本)

---

### Q8: 事前学習データの規模は?

**A**: 金融特化TSFMとして最大規模です。

| 項目 | 値 |
|------|-----|
| 総レコード数 | 12億本超 |
| 取引所数 | 45以上 (30カ国超) |
| 時間粒度 | 7種類 (1分〜週次) |
| 資産クラス | 株式、暗号通貨、FX、先物 |
| 期間 | 〜2024年6月 |

**データリバランス**: 暗号通貨・先物・FXのサンプリング重みを増加し、資産クラス間の不均衡を補正。

**比較**: 一般時系列TSFMの金融データ比率は通常 < 1.6% → Kronosは100%金融特化。

---

### Q9: コンテキスト長512の制限はどう影響する?

**A**: 時間粒度の切り替えで実質的に任意のホライゾンに対応可能です。

```
1分足 × 512 = 約8.5時間   → 超短期予測
5分足 × 512 = 約42.7時間  → 短期予測
日足  × 512 = 約2年       → 中長期予測
週足  × 512 = 約10年      → 長期予測
```

**実験でのLookback/Forecast設定** (Table 8):

| 頻度 | Lookback | Forecast |
|------|----------|----------|
| 5min | 480 | 96 |
| 15min | 160 | 32 |
| 1hour | 80 | 12 |
| Daily | 40 | 12 |

---

### Q10: 投資シミュレーションでの実用性は?

**A**: CSI 300/CSI 800でのバックテストで最高のAERとIRを達成しています (Table 10, Figure 9)。

**戦略**:
```python
# Top-k/Drop-n ポートフォリオ
for each_trading_day:
    # H日先の予測リターンを算出
    R = (mean(pred_close[1:H]) - current_close) / current_close

    # 上位k銘柄を等ウェイトでロング
    portfolio = top_k(R, k=50)  # CSI300: k=50

    # 1日最大n銘柄入替、最小5日保有
    apply_turnover_constraint(n=5, min_hold=5)
```

**結果 (Kronos_large)**:
- CSI 300: AER=0.2193, IR=1.4177
- CSI 800: AER=0.1974, IR=1.8805
- 全ベースライン (TSFMs含む) を上回る

---

## まとめ

Kronosは以下の5つのイノベーションで金融K線データの基盤モデルを実現:

1. **BSQトークン化**: 連続K線データを階層的離散トークンに変換、ノイズ耐性と表現力を両立
2. **Coarse-to-Fine予測**: s1→s2の逐次予測で語彙パラメータ99.8%削減
3. **DependencyAwareLayer**: Cross-Attentionでs2をs1に条件付け、サブトークン間依存を明示的にモデル化
4. **TemporalEmbedding**: 5種類の時間特徴で金融市場の周期性を捕捉
5. **Test-Time Scaling**: 複数サンプル平均で推論時に精度向上 (計算コストとのトレードオフ)

**性能**: 価格予測RankICで既存TSFM比93%、非事前学習モデル比87%の改善

**用途**:
- 価格系列予測 / リターン予測
- ボラティリティ予測
- 合成K線生成
- 投資シミュレーション / バックテスト

**推奨設定**:
- 予測タスク: T=0.6, top_p=0.90, N=10 (Monte Carlo平均)
- 生成タスク: T=1.0, top_p=0.95, N=1 (多様性重視)

---

## 参考文献

- 論文: [Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)
- 公式実装: [https://github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- 関連研究:
  - BSQ (2024): [Image and video tokenization with binary spherical quantization](https://arxiv.org/abs/2406.07548)
  - Chronos (2024): Learning the language of time series
  - TimesFM (2024): A decoder-only foundation model for time-series forecasting
  - TimeMOE (2025): Billion-Scale Time Series Foundation Models with Mixture of Experts

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装とは異なる場合があります。
