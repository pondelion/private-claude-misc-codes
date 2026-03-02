# SiT Understanding - 簡略化疑似コード集

SiT (Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers) の理解を目的とした簡略化疑似コード集です。

論文: [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740) (Ma et al., 2024, ICML)
公式実装: [https://github.com/willisma/SiT](https://github.com/willisma/SiT)

## 📋 目次

- [概要](#概要)
- [DiTからの変更点](#ditからの変更点)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [Interpolant Framework詳解](#interpolant-framework詳解)
- [パス設計](#パス設計)
- [予測対象と損失関数](#予測対象と損失関数)
- [推論: ODE vs SDE](#推論-ode-vs-sde)
- [形状ガイド](#形状ガイド)
- [実験結果](#実験結果)
- [DiT・SiT・F5-TTS・CosyVoice3の関係](#ditsitf5-ttscosyvoice3の関係)

---

## 概要

**SiTの特徴:**
- **DiTのアーキテクチャをそのまま使用**: Transformerバックボーンは変更なし
- **Diffusion → Interpolant Framework**: DDPM離散拡散からStochastic Interpolantの連続フレームワークに移行
- **統一フレームワーク**: 1つのモデルで velocity / score / noise 予測を切り替え可能
- **ODE + SDE 推論**: 確率的 (SDE) と決定的 (ODE) の両方で生成可能
- **3つのパス設計**: Linear / GVP (cosine) / VP を統一的に扱う

**タスク:**
- クラス条件付き画像生成 (ImageNet 256x256, 512x512)

**性能:**
- ImageNet 256x256: **FID 2.06** (SiT-XL/2, SDE sampler, cfg=1.80)
- DiT-XL/2 の FID 2.27 を**同じアーキテクチャ・同じ学習量**で改善

---

## DiTからの変更点

**変わらないもの (アーキテクチャ):**
- SiTBlock = DiTBlock (adaLN-Zero, 完全同一)
- FinalLayer (adaLN)
- TimestepEmbedder, LabelEmbedder
- PatchEmbed, 位置符号化
- モデルバリエーション (S/B/L/XL, /2/4/8)
- VAE (Stable Diffusion)

**変わるもの (学習・推論フレームワーク):**

| 項目 | DiT | SiT |
|------|-----|-----|
| **フレームワーク** | DDPM (離散, 1000ステップ) | Stochastic Interpolant (連続, t∈[0,1]) |
| **Forward process** | `x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε` | `x_t = α_t × x_1 + σ_t × x_0` (パス依存) |
| **予測対象** | ε (ノイズ) のみ | velocity / score / noise を選択可能 |
| **損失関数** | MSE + VLB | MSE (+ optional weighting) |
| **パス設計** | Linear β schedule 固定 | Linear / GVP / VP を選択可能 |
| **推論** | DDPM逆拡散 (SDE的) | ODE (dopri5) or SDE (Euler/Heun) |
| **時刻入力** | 離散 t ∈ {0,...,999} | 連続 t ∈ [0, 1] |
| **出力チャンネル** | 8 (ε + learned σ) | 4 (velocity のみ, learn_sigma 不要) |

→ **「同じTransformerを、より良いフレームワークで学習・推論する」**

---

## アーキテクチャ全体像

### 学習時

```
入力画像 x (B, 3, 256, 256)
    ↓
┌──────────────────────────────────────┐
│ 1. VAE Encoder (事前学習済み, 固定)    │  → x_1: (B, 4, 32, 32)
│    × 0.18215 正規化                   │     データ (t=1側)
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 2. Interpolantで補間 (連続)           │  → x_t: (B, 4, 32, 32)
│    x_0 ~ N(0,I), t ~ U[0,1]         │     x_t = α_t×x_1 + σ_t×x_0
│    target: u_t = dα/dt×x_1 + dσ/dt×x_0│   ← 速度場ターゲット
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 3. SiT (= DiT Backbone)              │  → v_θ: (B, 4, 32, 32)
│    adaLN-Zero, 条件: t_emb + y_emb   │     速度場予測
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 4. 損失計算                          │  → loss
│    MSE(v_θ, u_t)                     │
└──────────────────────────────────────┘
```

### 推論時 (ODE)

```
z ~ N(0, I): (B, 4, 32, 32)      ← t=0 (ノイズ)
    ↓
┌──────────────────────────────────────┐
│ ODE求解 (dopri5 / Euler)             │
│ dx/dt = v_θ(x_t, t, y)              │  t: 0 → 1
│ CFG: v = v_uncond + s×(v_cond-v_uncond)│ s=4.0
└──────────────────────────────────────┘
    ↓
x_1: (B, 4, 32, 32)              ← t=1 (データ)
    ↓
┌──────────────────────────────────────┐
│ VAE Decoder                          │  → 画像: (B, 3, 256, 256)
└──────────────────────────────────────┘
```

### 推論時 (SDE)

```
z ~ N(0, I): (B, 4, 32, 32)
    ↓
┌──────────────────────────────────────┐
│ SDE求解 (Euler-Maruyama / Heun)       │
│ dx = [drift + diffusion²×score]dt    │  t: 0 → 1
│      + √(2×diffusion) dW            │  ← 確率的ノイズ項
│ score = velocity→score変換           │
│ last_step: Mean / Tweedie / Euler    │
└──────────────────────────────────────┘
    ↓
x_1: (B, 4, 32, 32)
    ↓
VAE Decode → 画像
```

---

## ファイル構成

```
SiT_understanding/
├── README.md                    # このファイル
├── models.py                   # SiTモデル (DiTと構造同一, forward差分あり)
├── transport.py                # Interpolant Framework (パス, 損失, ODE/SDE)
└── train_sample.py             # 学習ループ + 推論パイプライン
```

### 1. [models.py](models.py)
**SiTモデルアーキテクチャ**

DiTと同一構造。唯一の違い:
- `forward()`: `learn_sigma=True`の場合、出力の前半4chのみ返す (後半は破棄)
- DiTでは8ch全てを返す (ε予測 + 分散予測)
- SiTでは4chのみ (velocity/score/noise予測)

### 2. [transport.py](transport.py)
**SiTの核心: Stochastic Interpolant Framework**

- `ICPlan` / `GVPCPlan` / `VPCPlan`: 3つのパス設計
- `Transport`: 学習損失計算 (velocity/score/noise予測)
- `Sampler`: ODE/SDEサンプリング

### 3. [train_sample.py](train_sample.py)
**学習・推論パイプライン**

- DiTとほぼ同じ学習ループ
- 推論はODE/SDEの選択が可能

---

## Interpolant Framework詳解

### Stochastic Interpolant とは

データ x_1 とノイズ x_0 を連続的に結ぶ「パス」を定義し、そのパスに沿った速度場を学習する。

```
x_t = α(t) × x_1 + σ(t) × x_0

ここで:
  x_0 ~ N(0, I)                    ← ノイズ (t=0側)
  x_1 ~ p_data                     ← データ (t=1側)
  α(t): データ係数    α(0)=0, α(1)=1
  σ(t): ノイズ係数    σ(0)=1, σ(1)=0
```

**速度場 (velocity field)**:
```
u_t(x_t) = dα/dt × x_1 + dσ/dt × x_0

学習: v_θ(x_t, t) ≈ u_t(x_t)
損失: L = E_{t,x_0,x_1} ||v_θ(x_t, t) - u_t||²
```

**Flow Matching との関係:**
SiTのLinearパス + velocity予測 = Conditional Flow Matching (CFM) そのもの。
SiTはこれをGVP/VPパスやscore/noise予測にまで拡張した統一フレームワーク。

---

## パス設計

### 1. Linear (ICPlan) - デフォルト

```
α(t) = t                    dα/dt = 1
σ(t) = 1 - t                dσ/dt = -1

x_t = t × x_1 + (1-t) × x_0        ← 直線パス
u_t = x_1 - x_0                     ← 一定速度
```

CosyVoice3/F5-TTSが使うOptimal Transport CFMと同じ形式。

### 2. GVP (Geodesic Variational Path)

```
α(t) = sin(πt/2)            dα/dt = (π/2) × cos(πt/2)
σ(t) = cos(πt/2)            dσ/dt = -(π/2) × sin(πt/2)

x_t = sin(πt/2) × x_1 + cos(πt/2) × x_0   ← 測地線パス (球面上)
```

Cosineスケジュールに対応。t=0付近でゆっくり、中間で速く変化。
CosyVoice3の `t = 1 - cos(u × π/2)` はこのGVPのスケジューリングに相当。

### 3. VP (Variance Preserving)

```
α(t) = exp(-0.25(1-t)²(σ_max-σ_min) - 0.5(1-t)σ_min)
σ(t) = √(1 - α(t)²)

DDPM/Score Matchingと等価なパス。
σ_min=0.1, σ_max=20.0 (VP-SDE のデフォルト)
```

### パスの比較

```
        Linear              GVP (Cosine)         VP
α(t)  ──┐                  ╱                   ╱
     1  │    ╱           1 │  ╱              1 │     ╱
        │  ╱               │╱                  │   ╱
     0  │╱_________     0  │_________       0  │╱________
        0    0.5   1       0    0.5   1        0    0.5   1

x_1重み: 線形増加        sin曲線 (緩→急→緩)    指数的増加

σ(t)
     1  │╲              1  │╲                1  │╲
        │  ╲               │ ╲                  │  ╲
     0  │____╲___       0  │___╲____        0  │____╲___
        0    0.5   1       0    0.5   1        0    0.5   1

x_0重み: 線形減少        cos曲線              √指数的減少
```

---

## 予測対象と損失関数

### 3つの予測モード

同じモデル出力 `f_θ(x_t, t)` を、3つの異なるターゲットで学習できる:

#### 1. Velocity prediction (デフォルト, 推奨)

```
target: u_t = dα/dt × x_1 + dσ/dt × x_0
loss:   L = ||f_θ(x_t, t) - u_t||²

ODE推論: dx/dt = f_θ(x, t)  ← そのまま速度場として使える
```

#### 2. Score prediction

```
target: ∇_x log p_t(x_t)
        = (α(t)×x_1 - x_t) / σ(t)²   (Gaussianの場合)
loss:   L = ||f_θ(x_t, t) - score||²

ODE推論: dx/dt = -drift_mean + drift_var × f_θ(x, t)
         ← 変数変換が必要
```

#### 3. Noise prediction

```
target: x_0 (入力ノイズ)
loss:   L = ||f_θ(x_t, t) - x_0||²

ODE推論: score = f_θ / (-σ_t) に変換後使用
         ← DDPMのε予測に相当
```

### 損失重み付け

```
L = E_t [w(t) × ||f_θ(x_t, t) - target||²]

重み w(t) の選択肢:
1. None (uniform):   w(t) = 1
2. Velocity:         w(t) = (drift_var / σ_t)²
3. Likelihood:       w(t) = drift_var / σ_t²
```

Velocity予測 + uniform重み が最もシンプルかつ安定 (ε=0で数値安定)。

---

## 推論: ODE vs SDE

### ODE推論

```
dx/dt = v_θ(x, t)                    ← 速度場に沿った決定的な流れ

ソルバー:
  - dopri5 (適応的ステップ, デフォルト)  ← Runge-Kutta 4/5次
  - Euler (固定ステップ)
  - Heun (固定ステップ, 2次精度)

atol=1e-6, rtol=1e-3

利点: 決定的、尤度計算可能、少ないステップで高品質
欠点: 多様性がやや低い
```

### SDE推論

```
dx = [drift + diffusion² × score] dt + √(2 × diffusion) dW

drift: v_θ(x, t)  (velocity → score変換して使用)
diffusion: σ(t) or 定数 等、複数の形式を選択可能

ソルバー:
  - Euler-Maruyama (デフォルト)
  - Heun

last_step: 最終ステップの処理
  - Mean:    x_final = x + drift × dt
  - Tweedie: x_final = x/α + σ²/α × score
  - Euler:   x_final = x + v × dt

利点: 確率的ノイズにより多様性が高い
欠点: ステップ数が多く必要
```

### Diffusion形式の選択肢

SDE推論時の diffusion 係数:
```
form      | diffusion(t)
----------|---------------------------
constant  | norm (=1.0)
SBDM      | norm × drift_var(t)
sigma     | norm × σ(t)               ← デフォルト
linear    | norm × (1 - t)
decreasing| 0.25 × (norm×cos(πt)+1)²
```

---

## 形状ガイド

DiTと同一。SiT-XL/2 の場合:

```
画像空間:
  入力画像:         (B, 3, 256, 256)
  VAE潜在:          (B, 4, 32, 32)        ← ×0.18215

Transformer内部:
  パッチ埋め込み後:  (B, 256, 1152)       ← 16×16, dim=1152
  SiTBlock出力:     (B, 256, 1152)
  FinalLayer出力:   (B, 256, 32)

出力:
  unpatchify後:     (B, 8, 32, 32)        ← learn_sigma=True
  → forward戻り値:  (B, 4, 32, 32)        ← 前半のみ (velocity)

条件:
  時刻埋め込み:     (B, 1152)             ← 連続t ∈ [0,1]
  ラベル埋め込み:   (B, 1152)
  条件ベクトル:     (B, 1152)
```

---

## 実験結果

### ImageNet 256×256 (class-conditional)

```
モデル                      | FID↓  | IS↑    | Precision | Recall |
---------------------------|-------|--------|-----------|--------|
DiT-XL/2 (DDPM)           | 2.27  | 278.24 | 0.83      | 0.57   |
SiT-XL/2 (Linear+vel+ODE) | 2.06  | 270.27 | 0.82      | 0.59   |
SiT-XL/2 (Linear+vel+SDE) | 2.06  | 277.57 | 0.82      | 0.59   |
```

### パス × 予測 × 推論の組み合わせ比較 (SiT-XL/2, 400K steps)

```
パス      | 予測      | 推論 | FID↓  |
----------|----------|------|-------|
Linear    | velocity | ODE  | 21.00 |  ← Flow Matchingと等価
Linear    | velocity | SDE  | 15.80 |
GVP       | velocity | ODE  | 21.34 |
GVP       | velocity | SDE  | 13.40 |  ← SDE×GVPが400Kでは最良
VP        | velocity | ODE  | 24.70 |
VP        | velocity | SDE  | 14.81 |
Linear    | score    | ODE  | 27.70 |
Linear    | noise    | ODE  | 36.61 |
```

→ velocity予測が全パスで最良。SDEは短い学習で優位だが、収束後はODEと同等。

### CFG scale の効果

```
CFG   | FID↓ (ODE) | FID↓ (SDE) |
------|------------|------------|
1.0   | 8.61       | 7.31       |
1.25  | 3.15       | 2.93       |
1.50  | 2.14       | 2.10       |
1.80  | 2.06       | 2.06       |
4.00  | 3.57       | 2.68       |
```

---

## DiT・SiT・F5-TTS・CosyVoice3の関係

```
DiT (2023)
│  Diffusion + Transformer
│  adaLN-Zero条件付け
│  ImageNet画像生成
│
├── SiT (2024)
│   │  DiTと同じアーキテクチャ
│   │  Diffusion → Interpolant/Flow Matching
│   │  ODE/SDE両対応
│   │
│   └─── 設計空間の探索結果:
│        - Linear path + velocity = Flow Matching
│        - GVP path = Cosine schedule
│        - SDE推論で多様性向上
│
├── F5-TTS (2024)
│   │  DiT + Flow Matching (OT-CFM)
│   │  画像→音声 (メルスペクトログラム)
│   │  テキスト前処理: ConvNeXt V2
│   │  Sway Sampling (不均一ステップ)
│   │  adaLN (not Zero) + RoPE
│   │
│   └─── SiTのLinear+velocity設定に相当
│
└── CosyVoice3 (2024)
    │  DiT + Flow Matching (OT-CFM)
    │  画像→音声 (メルスペクトログラム)
    │  adaLN-Zero + RoPE + Long Skip Connection
    │  Cosineスケジュール ≈ SiTのGVPパス
    │  10ステップEuler ODE
    │
    └─── SiTのGVP+velocity+ODE設定に近い
```

**共通する核心思想:**
1. Transformerはスケーリングに優れる (DiTが証明)
2. adaLN-Zero で条件を注入するのが最も効率的 (DiTが証明)
3. Flow Matching / Interpolant はDDPMより効率的 (SiTが証明)
4. 上記の組み合わせは画像以外 (音声) にもそのまま適用可能

---

## 実装対応

| 概念 | 実装箇所 |
|------|---------|
| SiTBlock (= DiTBlock) | [models.py:120-157](models.py#L120-L157) |
| SiT forward (velocity出力) | [models.py:186-336](models.py#L186-L336) |
| Linear Path (ICPlan) | [transport.py:95-340](transport.py#L95-L340) |
| GVP Path (Cosine) | [transport.py:344-380](transport.py#L344-L380) |
| VP Path | [transport.py:382-465](transport.py#L382-L465) |
| Transport (学習損失) | [transport.py:468-760](transport.py#L468-L760) |
| ODE推論 (dopri5) | [transport.py:762-845](transport.py#L762-L845) |
| SDE推論 (Euler-Maruyama) | [transport.py:848-978](transport.py#L848-L978) |
| Sampler (ODE/SDE選択) | [transport.py:981-1295](transport.py#L981-L1295) |
| create_transport ファクトリ | [transport.py:1297-1370](transport.py#L1297-L1370) |
| 学習ループ | [train_sample.py:81-248](train_sample.py#L81-L248) |
| ODE推論パイプライン | [train_sample.py:252-355](train_sample.py#L252-L355) |
| SDE推論パイプライン | [train_sample.py:359-460](train_sample.py#L359-L460) |
