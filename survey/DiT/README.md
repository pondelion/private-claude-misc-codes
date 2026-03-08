# DiT Understanding - 簡略化疑似コード集

DiT (Scalable Diffusion Models with Transformers) の理解を目的とした簡略化疑似コード集です。

論文: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (Peebles & Xie, 2023, ICCV)
公式実装: [https://github.com/facebookresearch/DiT](https://github.com/facebookresearch/DiT)

## 📋 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [DiTの主要イノベーション](#ditの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [形状ガイド](#形状ガイド)
- [損失関数](#損失関数)
- [条件付け手法の比較実験](#条件付け手法の比較実験)
- [実験結果](#実験結果)
- [後続研究への影響](#後続研究への影響)

---

## 概要

**DiTの特徴:**
- **U-NetをTransformerで置換**: Diffusion Modelのバックボーンを標準的なViTアーキテクチャに置き換え
- **adaLN-Zero条件付け**: Adaptive Layer Normalizationのゼロ初期化による安定した条件注入
- **スケーラビリティ**: モデルサイズ・計算量の増加に対してFIDが一貫して改善 (Gflops vs FID が強い相関)
- **Latent空間で動作**: Stable DiffusionのVAE潜在空間で学習・生成 (計算効率)

**タスク:**
- クラス条件付き画像生成 (ImageNet 256x256, 512x512)

**性能:**
- ImageNet 256x256: **FID 2.27** (state-of-the-art at publication)
- ImageNet 512x512: FID 3.04
- DiT-XL/2 (最大モデル): 675M パラメータ

---

## アーキテクチャ全体像

### 学習時

```
入力画像 x (B, 3, 256, 256)
    ↓
┌──────────────────────────────────────┐
│ 1. VAE Encoder (事前学習済み, 固定)    │  → z: (B, 4, 32, 32)
│    Stable Diffusion VAE              │     潜在空間に圧縮 (×0.18215 正規化)
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 2. ノイズ付加 (Forward Diffusion)     │  → x_t: (B, 4, 32, 32)
│    x_t = √ᾱ_t × z + √(1-ᾱ_t) × ε  │     t ~ U{0, ..., 999}
│    Linear β schedule                 │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 3. DiT (Transformer Backbone)        │  → ε_θ, σ_θ: (B, 8, 32, 32)
│    Patch Embed + N × DiTBlock        │     ε予測 + 学習分散
│    + FinalLayer                      │     (learn_sigma=True → 8ch)
│    条件: t_emb + y_emb               │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ 4. 損失計算                          │  → loss
│    MSE(ε_θ, ε) + VLB(σ_θ)           │
└──────────────────────────────────────┘
```

### 推論時

```
z_T ~ N(0, I): (B, 4, 32, 32)  ← 純粋ノイズ (VAE潜在空間)
    ↓
┌──────────────────────────────────────┐
│ DDPM逆拡散 (250ステップ)              │
│ for t = T, T-1, ..., 0:             │
│   ε_cond = DiT(z_t, t, y)           │
│   ε_uncond = DiT(z_t, t, ∅)         │
│   ε = ε_uncond + s×(ε_cond-ε_uncond)│  CFG scale s=4.0
│   z_{t-1} = denoise_step(z_t, ε)    │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ VAE Decoder                          │  → 画像: (B, 3, 256, 256)
│ x = Decoder(z_0 / 0.18215)          │
└──────────────────────────────────────┘
```

---

## ファイル構成

```
DiT_understanding/
├── README.md                    # このファイル
├── models.py                   # DiTモデル全体 (パッチ埋め込み, DiTBlock, FinalLayer)
└── train_sample.py             # 学習ループ + 推論パイプライン
```

### 1. [models.py](models.py)
**DiTモデルのコアアーキテクチャ**

主要コンポーネント:
- `TimestepEmbedder`: 正弦波位置符号化 + MLP → 時刻埋め込み
- `LabelEmbedder`: クラスラベル → 埋め込み (CFG用ドロップアウト付き)
- `DiTBlock`: adaLN-Zero条件付きTransformerブロック
- `FinalLayer`: adaLN + 線形射影 → パッチ空間への出力
- `DiT`: 全体モデル (パッチ化 → Nブロック → アンパッチ化)

**重要な入出力:**
- 入力: `x (B, 4, 32, 32)` ノイズ付き潜在表現, `t (B,)` 時刻, `y (B,)` クラスラベル
- 出力: `(B, 8, 32, 32)` ノイズ予測 + 分散予測

### 2. [train_sample.py](train_sample.py)
**学習・推論パイプライン**

- `train()`: DDP学習ループ (ImageNet + VAE + DDPM)
- `sample()`: DDPM逆拡散 + CFG + VAEデコード

---

## DiTの主要イノベーション

### 1. U-Net → Transformer への置換

従来のDiffusion Model (DDPM, ADM, LDM) はU-Netバックボーンを使用。DiTはこれを**標準的なViT (Vision Transformer)** で完全に置き換えた最初の成功例。

```
従来 (U-Net):
  ダウンサンプリング → ボトルネック → アップサンプリング
  ResBlock + Self-Attention + Cross-Attention
  スキップ接続 (エンコーダ→デコーダ)

DiT (Transformer):
  画像パッチ化 → N × TransformerBlock → アンパッチ化
  全トークンが全トークンに注意 (フルアテンション)
  スキップ接続なし (等方的アーキテクチャ)
```

**利点:**
- アーキテクチャがシンプル (ViTそのもの)
- スケーリング則がViTの知見をそのまま活用可能
- マルチスケール特徴マップの設計が不要

### 2. adaLN-Zero (Adaptive Layer Normalization Zero)

DiTの最大の貢献の一つ。条件情報 (時刻 t + クラス y) をTransformerブロックに注入する方法。

```
c = t_emb + y_emb                          # (B, D) 条件ベクトル

# adaLN-Zero: c から6つの変調パラメータを生成
[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
    SiLU(c) → Linear(D, 6D)               # (B, 6D) → 6分割

# Attention側
x_norm = LayerNorm(x) × (1 + scale_msa) + shift_msa   # FiLM変調
attn_out = MHSA(x_norm)
x = x + gate_msa × attn_out                             # ゲート付き残差

# FFN側
x_norm = LayerNorm(x) × (1 + scale_mlp) + shift_mlp
ff_out = FFN(x_norm)
x = x + gate_mlp × ff_out
```

**"Zero" の意味:**
- `Linear(D, 6D)` の重みとバイアスを**全て0で初期化**
- 学習初期: `scale=0, shift=0, gate=0`
- → ブロック出力 = `x + 0 × attn_out = x` (恒等写像)
- → 深いネットワークでも学習初期は信号がそのまま通過し、勾配が安定

### 3. Latent Diffusion

DiTはピクセル空間ではなく、Stable DiffusionのVAE潜在空間で動作:

```
画像 (3, 256, 256)  →  VAE Encode  →  潜在 (4, 32, 32)  →  DiT  →  潜在  →  VAE Decode  →  画像

圧縮率: 256×256×3 = 196,608 → 32×32×4 = 4,096 (約48倍)
```

### 4. スケーリング則

DiTの重要な発見: **モデルのGflopsとFIDの間に強い負の相関がある**

```
モデル         | 深さ | 幅   | ヘッド | Gflops | パラメータ | FID (256) |
--------------|------|------|-------|--------|-----------|-----------|
DiT-S/2       | 12   | 384  | 6     | 6      | 33M       | 68.4      |
DiT-B/2       | 12   | 768  | 12    | 23     | 130M      | 43.5      |
DiT-L/2       | 24   | 1024 | 16    | 80     | 458M      | 9.62      |
DiT-XL/2      | 28   | 1152 | 16    | 119    | 675M      | 2.27      |
```

パッチサイズの影響 (DiT-XL):
```
パッチサイズ | トークン数 | Gflops | FID   |
-----------|----------|--------|-------|
/8         | 16       | 29     | 不収束 |
/4         | 64       | 41     | 9.69  |
/2         | 256      | 119    | 2.27  |
```

→ **パッチが小さいほど (=トークンが多いほど) 性能向上** (ただし計算量も増加)

---

## 処理フロー詳細

### DiTモデルの forward

```python
def forward(x, t, y):
    """
    入力:
      x: (B, 4, 32, 32)  ← ノイズ付きVAE潜在表現
      t: (B,)             ← 離散時刻ステップ {0, ..., 999}
      y: (B,)             ← クラスラベル {0, ..., 999}

    出力:
      out: (B, 8, 32, 32) ← ε予測(4ch) + 分散予測(4ch)
    """
    # 1. パッチ埋め込み
    x = PatchEmbed(x)          # (B, 4, 32, 32) → (B, 256, 1152)
    # patch_size=2: 32/2=16, 16×16=256トークン
    x = x + pos_embed          # (B, 256, 1152) 固定sin-cos位置符号化

    # 2. 条件埋め込み
    t_emb = TimestepEmbedder(t)  # (B,) → sinusoidal → MLP → (B, 1152)
    y_emb = LabelEmbedder(y)     # (B,) → Embedding → (B, 1152)
    c = t_emb + y_emb             # (B, 1152) 条件ベクトル

    # 3. N個のDiTブロック
    for block in blocks:           # 28ブロック (DiT-XL)
        x = block(x, c)           # (B, 256, 1152) → (B, 256, 1152)

    # 4. 最終層
    x = FinalLayer(x, c)          # (B, 256, 1152) → (B, 256, 32)
    # 32 = patch_size² × out_channels = 2² × 8

    # 5. アンパッチ化
    x = unpatchify(x)             # (B, 256, 32) → (B, 8, 32, 32)

    return x
```

### DiTBlock の詳細

```python
def DiTBlock_forward(x, c):
    """
    入力:
      x: (B, 256, 1152)  ← トークン列
      c: (B, 1152)        ← 条件ベクトル (t + y)

    出力:
      x: (B, 256, 1152)
    """
    # adaLN変調パラメータ生成 (6つ)
    modulation = SiLU(c) → Linear(1152, 6912)  # (B, 6912)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(6)
    # 各: (B, 1152)

    # --- Attention ---
    x_norm = LayerNorm(x)                              # (B, 256, 1152)
    x_norm = x_norm × (1 + scale_msa[:,None,:]) + shift_msa[:,None,:]
    # scale_msa[:,None,:]: (B, 1152) → (B, 1, 1152) ブロードキャスト
    attn_out = MultiHeadAttention(x_norm)              # (B, 256, 1152)
    # 16ヘッド, head_dim=72, qkv_bias=True
    x = x + gate_msa[:,None,:] × attn_out             # ゲート付き残差

    # --- FFN ---
    x_norm = LayerNorm(x)
    x_norm = x_norm × (1 + scale_mlp[:,None,:]) + shift_mlp[:,None,:]
    ff_out = GELU_tanh(Linear(1152, 4608)) → Linear(4608, 1152)
    # mlp_ratio=4.0: 1152 × 4 = 4608
    x = x + gate_mlp[:,None,:] × ff_out

    return x
```

### FinalLayer の詳細

```python
def FinalLayer_forward(x, c):
    """
    入力:
      x: (B, 256, 1152)
      c: (B, 1152)

    出力:
      x: (B, 256, 32)  ← patch_size² × out_channels = 4 × 8 = 32
    """
    # adaLN (2パラメータのみ: shift, scale。ゲートなし)
    shift, scale = (SiLU(c) → Linear(1152, 2304)).chunk(2)   # 各 (B, 1152)
    x = LayerNorm(x) × (1 + scale[:,None,:]) + shift[:,None,:]  # (B, 256, 1152)
    x = Linear(1152, 32)(x)                                      # (B, 256, 32)
    return x
```

---

## 形状ガイド

### DiT-XL/2 (デフォルト設定)

```
画像空間:
  入力画像:         (B, 3, 256, 256)
  VAE潜在:          (B, 4, 32, 32)        ← ×0.18215 正規化

パッチ空間 (Transformer内部):
  パッチ埋め込み後:  (B, 256, 1152)       ← 16×16=256トークン, dim=1152
  DiTBlock出力:     (B, 256, 1152)
  FinalLayer出力:   (B, 256, 32)          ← 2²×8=32

出力空間:
  unpatchify後:     (B, 8, 32, 32)        ← ε(4ch) + σ(4ch)

条件空間:
  時刻埋め込み:     (B, 1152)             ← sinusoidal(256) → MLP
  ラベル埋め込み:   (B, 1152)             ← Embedding(1001, 1152)
  条件ベクトル:     (B, 1152)             ← t_emb + y_emb
  adaLN出力:        (B, 6912)             ← Linear(1152, 6×1152)
```

### モデルバリエーション

```
モデル   | depth | hidden_size | num_heads | head_dim | params |
---------|-------|-------------|-----------|----------|--------|
DiT-S    | 12    | 384         | 6         | 64       | 33M    |
DiT-B    | 12    | 768         | 12        | 64       | 130M   |
DiT-L    | 24    | 1024        | 16        | 64       | 458M   |
DiT-XL   | 28    | 1152        | 16        | 72       | 675M   |
```

パッチサイズバリエーション (DiT-XL, 入力32×32潜在):
```
パッチ | グリッド | トークン数 | 出力dim               |
-------|---------|----------|----------------------|
/2     | 16×16   | 256      | 2²×8 = 32 per token  |
/4     | 8×8     | 64       | 4²×8 = 128 per token |
/8     | 4×4     | 16       | 8²×8 = 512 per token |
```

---

## 損失関数

DiTはImproved DDPM (Nichol & Dhariwal, 2021) の損失を使用:

### メイン損失: MSE (ノイズ予測)

```
ε ~ N(0, I)                                    # (B, 4, 32, 32)
t ~ U{0, ..., 999}                             # (B,)

x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε            # Forward process

model_output = DiT(x_t, t, y)                 # (B, 8, 32, 32)
ε_θ = model_output[:, :4]                     # ノイズ予測
σ_θ = model_output[:, 4:]                     # 分散予測

L_simple = ||ε_θ - ε||²                       # MSE
```

### 分散学習: LEARNED_RANGE

```
# 分散は FIXED_SMALL と FIXED_LARGE の間の値を予測
# β_t = linear schedule: 0.0001 → 0.02 (1000ステップ)

β_tilde_t = β_t × (1 - ᾱ_{t-1}) / (1 - ᾱ_t)   # FIXED_SMALL (posterior variance)
β_t                                                # FIXED_LARGE

# モデル出力を [0,1] に変換して内挿
v = (σ_θ + 1) / 2                                 # [-1,1] → [0,1]
log_var = v × log(β_t) + (1-v) × log(β_tilde_t)   # 内挿

L_vlb = KL(q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t))  # VLB項
```

### 全体損失

```
L = L_simple + L_vlb        (VLB項はstop_gradientなしで直接加算)
```

---

## 条件付け手法の比較実験

DiT論文の重要な貢献の一つは、4つの条件付け手法を体系的に比較したこと:

### 1. In-context conditioning

```
トークン列にt,yを追加トークンとして結合:
tokens = [t_token, y_token, x_patch_1, ..., x_patch_N]
→ N+2 トークンを通常のTransformerで処理
```

### 2. Cross-attention

```
条件トークン c = [t_emb, y_emb]    # (B, 2, D)
各ブロックでCross-Attention追加:
x = x + CrossAttention(Q=x, K=c, V=c)
```

### 3. adaLN (Adaptive Layer Norm)

```
shift, scale = Linear(c)
x = LayerNorm(x) × (1 + scale) + shift
→ ゲートなし
```

### 4. adaLN-Zero (DiTの採用手法)

```
shift, scale, gate = Linear(c)   ← ゼロ初期化
x = LayerNorm(x) × (1 + scale) + shift
x = x + gate × sublayer_output   ← ゲート付き残差
```

### 比較結果 (DiT-XL/2, 400K steps)

```
手法               | FID↓   | 追加パラメータ | 追加Gflops |
------------------|--------|-------------|-----------|
In-context        | 175.0  | 最小         | ~0        |
Cross-attention   | 56.7   | 15%増        | 有意       |
adaLN             | 31.7   | 最小         | ~0        |
adaLN-Zero        | 23.3   | 最小         | ~0        |
```

→ **adaLN-Zeroが圧倒的に優秀** (追加コストほぼゼロで最高性能)

---

## 実験結果

### ImageNet 256×256 (class-conditional)

```
モデル              | FID↓  | sFID↓ | IS↑    | Precision | Recall |
-------------------|-------|-------|--------|-----------|--------|
ADM                | 10.94 | 6.02  | 100.98 | 0.69      | 0.63   |
ADM-U              | 7.49  | 5.13  | 127.49 | 0.72      | 0.63   |
ADM-G              | 4.59  | 5.25  | 186.70 | 0.82      | 0.52   |
LDM-4              | 10.56 | -     | 103.49 | 0.71      | 0.62   |
LDM-4-G (cfg=1.5)  | 3.60  | -     | 247.67 | -         | -      |
DiT-XL/2           | 9.62  | 6.85  | 121.50 | 0.67      | 0.67   |
DiT-XL/2-G (cfg=1.5)| 2.27 | 4.60 | 278.24 | 0.83      | 0.57   |
```

### ImageNet 512×512

```
モデル              | FID↓  | sFID↓ | IS↑    |
-------------------|-------|-------|--------|
ADM-G              | 3.85  | 5.86  | 221.72 |
DiT-XL/2-G        | 3.04  | 5.02  | 240.82 |
```

### Classifier-Free Guidance スケール

```
CFG scale | FID↓   | IS↑    |
----------|--------|--------|
1.0       | 9.62   | 121.50 |
1.25      | 3.22   | 205.30 |
1.50      | 2.27   | 278.24 |
4.0       | 2.91   | 316.98 | ← サンプリングデフォルト
```

---

## 後続研究への影響

DiTは以下の後続研究のベースアーキテクチャとなった:

| 研究 | 年 | DiTからの変更点 |
|------|-----|---------------|
| **SiT** | 2024 | Diffusion → Flow Matching (Interpolant Framework) |
| **F5-TTS** | 2024 | 画像 → 音声, adaLN + ConvNeXt V2テキスト前処理 |
| **CosyVoice3** | 2024 | 画像 → 音声, adaLN-Zero + RoPE + Long Skip |
| **Stable Diffusion 3** | 2024 | adaLN-Zero + MM-DiT (text/image joint attention) |
| **Sora** (推定) | 2024 | 画像 → 動画, spatiotemporal patches |
| **PixArt-α** | 2023 | T5テキスト条件 + 効率的学習 |

**DiTが確立したパターン:**
1. adaLN-Zero条件付け → Diffusion/Flow Matching Transformerの事実上の標準
2. パッチ埋め込み → 画像以外 (音声メル、動画フレーム) にも適用
3. ViT的等方アーキテクチャ → スケーリング則の活用
4. 潜在空間での学習 → VAE/Tokenizer + Transformer の2段構成が主流に

---

## 実装対応

| 概念 | 実装箇所 |
|------|---------|
| DiTBlock (adaLN-Zero) | [models.py:54-125](models.py#L54-L125) |
| FinalLayer (adaLN) | [models.py:129-174](models.py#L129-L174) |
| DiT全体 (forward) | [models.py:178-293](models.py#L178-L293) |
| TimestepEmbedder | [models.py:30-51](models.py#L30-L51) |
| LabelEmbedder (CFG dropout) | [models.py:60-101](models.py#L60-L101) |
| 学習ループ | [train_sample.py:32-137](train_sample.py#L32-L137) |
| 推論 (DDPM + CFG) | [train_sample.py:140-208](train_sample.py#L140-L208) |
| 初期化 (ゼロ初期化) | [models.py:214-260](models.py#L214-L260) |
