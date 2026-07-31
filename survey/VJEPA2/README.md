# V-JEPA 2.1 理解のための簡略化疑似コード集

**論文**: [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)
**著者**: Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, et al. (FAIR at Meta)
**公開**: 2026年3月

---

## 目次

1. [概要](#概要)
2. [V-JEPA 2 との違い・進化点](#v-jepa-2-との違い進化点)
3. [アーキテクチャ全体像](#アーキテクチャ全体像)
4. [ファイル構成](#ファイル構成)
5. [4つの主要イノベーション](#4つの主要イノベーション)
   - [Dense Prediction Loss](#1-dense-prediction-loss-密予測損失)
   - [Deep Self-Supervision](#2-deep-self-supervision-深層自己教師あり学習)
   - [Multi-Modal Tokenizer](#3-multi-modal-tokenizer-マルチモーダルトークナイザ)
   - [Data & Model Scaling](#4-data--model-scaling)
6. [3D スパシオテンポラルマスキング](#3d-スパシオテンポラルマスキング)
7. [処理フロー詳細](#処理フロー詳細)
8. [学習設定・ハイパーパラメータ](#学習設定ハイパーパラメータ)
9. [下流タスク性能](#下流タスク性能)
10. [形状ガイド](#形状ガイド)
11. [FAQ](#faq)
12. [参考文献](#参考文献)

---

## 概要

V-JEPA 2.1 は Meta FAIR が開発した**映像・画像統一の自己教師あり表現学習**フレームワーク。
前作 V-JEPA 2 が「動作認識などグローバル理解」には強力だったものの、
「セグメンテーションや深度推定などの密予測タスク」に弱かった問題を解決。

### キーコントリビューション

| 要素 | 内容 |
|------|------|
| **Dense Prediction Loss** | マスクトークンだけでなく可視コンテキストトークンにも損失を適用 |
| **Deep Self-Supervision** | エンコーダの複数中間層出力を階層的に監督 |
| **Multi-Modal Tokenizer** | 画像(2D Conv)と動画(3D Conv)を専用トークナイザで処理 |
| **Data/Model Scaling** | VisionMix163M(142M画像+動画)、ViT-G(2B params) |

### 性能ハイライト（ViT-G, frozen encoder）

| タスク | V-JEPA 2 | V-JEPA 2.1 | 改善率 |
|--------|-----------|------------|--------|
| SSv2 Action Recognition | 72.8% | 77.7% | +7% |
| ADE20K Segmentation (mIoU) | 22.2 | 47.9 | **+116%** |
| NYUv2 Depth Est. (RMSE↓) | 0.682 | 0.307 | **-55%** |
| Ego4D STA (mAP) | – | 7.71 | SOTA |
| EK Action Anticipation (R@5) | – | 40.8 | SOTA |

---

## V-JEPA 2 との違い・進化点

```
                   V-JEPA 2                    V-JEPA 2.1
                  ──────────                  ────────────
学習目標       マスクトークンのみ予測      可視+マスクトークン両方予測
                                           (Dense Prediction)

エンコーダ出力  最終層のみ                 中間層4層 + 最終層を連結
                                           (Deep Self-Supervision)

画像処理       16フレームの静止動画扱い    専用2D Convで処理
                                           (Multi-Modal Tokenizer)

モダリティ     動画のみ                    画像+動画の統一学習
                                           (Modality Embedding)

モデルサイズ   ViT-g (1B)                 ViT-G (2B) + 蒸留モデル

データ         VideoMix22M                 VisionMix163M
                                           (142M画像 + 動画)
```

---

## アーキテクチャ全体像

```
入力: 動画 (B, 3, T, H, W)  または  画像 (B, 3, H, W)
         │                              │
    ┌────┴────┐                   ┌─────┴─────┐
    │3D Conv  │                   │  2D Conv  │
    │(tubelet)│                   │  (patch)  │
    └────┬────┘                   └─────┬─────┘
         └──────────┬───────────────────┘
                    │ パッチトークン (B, N, D)
               ┌────┴────┐
               │3D RoPE  │  ← 時空間位置符号化
               │+ Modal  │  ← モダリティ埋め込み(img/vid)
               │Embedding│
               └────┬────┘
                    │
        ┌───────────┴──────────────┐
        │   3D Block Mask Sampling  │
        │  ┌──────────────┐         │
        │  │  mask blocks  │  npred=8 (small) + 2 (large)
        │  └──────────────┘         │
        └───────────┬──────────────┘
                    │
        ┌──────────────────────────┐
        │  x-エンコーダ(Student)   │  y-エンコーダ(EMA Teacher)
        │  可視トークンのみ処理     │  全トークン処理(勾配なし)
        │                          │
        │  ViT Transformer×depth   │  ViT Transformer×depth
        │                          │
        │  中間層[l1,l2,l3] + 最終 │  中間層[l1,l2,l3] + 最終
        │  → 連結→MLP fusion       │  → 連結 (ターゲット)
        │  z_ctx: (B, N_ctx, D×4)  │  h: (B, N, D×4)
        └──────────┬───────────────┘
                   │
         z_ctx ────┤
                   │ context tokens
               ┌───┴──────┐
               │Predictor │  mask tokens (位置情報付き)
               │ViT×L_pred│  ────────────────────────→
               └───┬───┬──┘
                   │   │
             z_pred│   │z_context
           (マスク) │   │(可視)
                   │   │
        ┌──────────┴───┴──────────────────────────────┐
        │              損失計算                        │
        │                                              │
        │  L_predict = (1/|M|) Σ_{i∈M} |z_pred_i - h_i|   │
        │                                              │
        │  L_context = (1/|C|) Σ_{i∈C} λ_i |z_ctx_i - h_i|│
        │      λ_i = λ / sqrt(d_min(i, M))             │
        │      d_min: コンテキスト↔最近マスクの距離    │
        │                                              │
        │  L_dense = L_predict + λ * L_context         │
        └──────────────────────────────────────────────┘
                   │
            EMA更新: θ_teacher ← m·θ_teacher + (1-m)·θ_student
                      m ≈ 0.99925 (固定)
```

---

## ファイル構成

```
VJEPA2/
├── README.md              本ファイル（アーキテクチャ詳細説明）
├── encoder.py             [10.5 KB] VisionTransformer (x-encoder/y-encoder)
│                            PatchEmbed2D, PatchEmbed3D, ViTブロック
├── predictor.py           [11.2 KB] VisionTransformerPredictor
│                            マスクトークン、コンテキスト+マスク統合処理
├── mask_generator.py      [9.8 KB]  3Dスパシオテンポラルマスク生成
│                            _MaskGenerator, MaskCollator
├── loss_computation.py    [12.1 KB] Dense Prediction Loss
│                            L_predict, L_context, 距離重み付け
├── main_flow.py           [15.2 KB] VJEPA2全体フロー
│                            Student/Teacher構造、EMAアップデート、学習ステップ
└── finetune_example.py    [18.5 KB] ファインチューニングサンプル
                             動画分類・密予測タスクへの適用
```

### 各ファイルの役割

| ファイル | 対応する実装 | 主要クラス |
|----------|-------------|-----------|
| `encoder.py` | `src/models/vision_transformer.py` | `VisionTransformer`, `PatchEmbed3D` |
| `predictor.py` | `src/models/predictor.py` | `VisionTransformerPredictor` |
| `mask_generator.py` | `src/masks/multiseq_multiblock3d.py` | `MaskCollator`, `_MaskGenerator` |
| `loss_computation.py` | `app/vjepa_2_1/train.py` の損失部分 | `DensePredictionLoss` |
| `main_flow.py` | `app/vjepa_2_1/train.py` | `VJEPA2`, `train_step` |
| `finetune_example.py` | `evals/video_classification_frozen/` | `VideoClassifier` |

---

## 4つの主要イノベーション

### 1. Dense Prediction Loss（密予測損失）

#### 問題提起
V-JEPA 2 では可視コンテキストトークンに損失がかからないため、
コンテキストトークンがグローバル情報を集約するだけになり、局所的な空間構造が失われる。

```
V-JEPA 2 の問題:
  Predictor入力: [context_tokens | mask_tokens]
                  ↑                ↑
              損失なし            損失あり ← マスクトークンの予測のみ

→ コンテキストトークンが「グローバルアグリゲータ」として振る舞い
  空間局所情報が無視される (DINOのregisterトークンと同様の現象)
```

#### 解決策: コンテキスト損失 L_context

```
L_predict = (1/|M|) Σ_{i∈M}  ||z_pred_i    - sg(h_i)||_1   ← 元のV-JEPA損失
L_context = (1/|C|) Σ_{i∈C}  λ_i ||z_ctx_i - sg(h_i)||_1   ← 新規追加

L_dense = L_predict + L_context

記号:
  M: マスクされたパッチのインデックス集合
  C: コンテキスト(可視)パッチのインデックス集合
  z_pred_i: Predictorが予測したマスクトークンの表現
  z_ctx_i: Predictorが出力したコンテキストトークンの表現
  h_i: Teacherエンコーダの出力 (stop-gradient)
  sg(): stop-gradient演算子
  λ_i: 距離重み (後述)
```

#### 距離重み付け λ_i

コンテキストパッチとマスク領域の距離に基づく動的重み:

```python
λ_i = λ / sqrt(d_min(i, M))

# d_min(i, M): コンテキストパッチiから最近傍マスクトークンまでの
#              ブロック単位の時空間距離

# 効果: マスク境界付近のコンテキストパッチに高い重みをかけ
#       「局所的連続性」を強制する
```

| λ係数 | ADE20k(mIoU↑) | SSv2(Acc↑) | 備考 |
|-------|--------------|-----------|------|
| 0 (V-JEPA 2) | 22.2 | 72.8 | ベースライン |
| 0.2 (固定) | 29.6 | 62.5 | 分類低下 |
| 0.5 (warmup) | 33.8 | 62.5 | 少し改善 |
| 距離重み (最終) | 33.9 | 62.5 | バランス最良 |

### 2. Deep Self-Supervision（深層自己教師あり学習）

エンコーダの中間層出力を活用して、ネットワーク全体に学習信号を伝播させる。

```
エンコーダブロック構造 (例: ViT-L, depth=24):

Block 0  → Block 1  → ... → Block 5  → ... → Block 11 → ... → Block 23
                              ↓                   ↓                 ↓
                           level_1            level_2           level_3
                              ↓                   ↓                 ↓
                         └──────────────────────────────────────┘
                                        concat (dim方向)
                                    (B, N, D×4) ← 4層分
                                        ↓
                                  MLP fusion
                                    (B, N, D)
                                        ↓
                                 Predictor へ入力
                                        ↓
                              4つのレベルそれぞれに対して
                              L_predict + L_context を計算
```

**効果**: Deep Self-Supervisionによりコンテキスト損失で失われた分類性能を回復
(SSv2: 62.5 → 72.1, ADE20k: 33.8 → 38.6)

### 3. Multi-Modal Tokenizer（マルチモーダルトークナイザ）

```
V-JEPA 2 (問題あり):
  画像 (B, 3, H, W)
      ↓ 時間方向に16回複製
  疑似動画 (B, 3, 16, H, W) ← 非効率・偏ったバイアス
      ↓ 3D Conv (patch=16, tubelet=2)
  トークン (B, N_video, D)


V-JEPA 2.1 (改善):
  動画 (B, 3, T, H, W)          画像 (B, 3, H, W)
      ↓ 3D Conv                      ↓ 2D Conv
      │  (kernel: 2×16×16)            │  (kernel: 16×16)
  (B, N_vid, D)              (B, N_img, D)
       └────────────┬──────────────────┘
                    │
             + Modality Embedding  ← 学習可能な「動画/画像」識別トークン
                    │
              Shared ViT Encoder
```

### 4. Data & Model Scaling

#### VisionMix163M データセット

| ソース | サンプル数 | タイプ | V-JEPA2重み | V-JEPA2.1重み |
|--------|-----------|--------|------------|--------------|
| SSv2 | 168K | Ego動画 | 0.056 | 0.170 |
| Kinetics | 733K | Exo動画 | 0.188 | 0.010 |
| HowTo100M | 1.1M | Exo動画 | 0.318 | 0.100 |
| ImageNet | 1M | 画像 | 0.250 | 0 |
| YT-1B | 19M | Exo動画 | 0.188 | 0.720 |
| LVD-142M | 142M | キュレーション画像 | 0 | (含む) |

#### モデルサイズ

| モデル | Params | embed_dim | depth | heads |
|--------|--------|-----------|-------|-------|
| ViT-B  | 80M    | 768       | 12    | 12    |
| ViT-L  | 300M   | 1024      | 24    | 16    |
| ViT-g  | 1B     | 1408      | 40    | 22    |
| ViT-G  | 2B     | 1664      | 48    | 26    |

---

## 3D スパシオテンポラルマスキング

### マスク生成戦略

```
グリッドサイズ (ViT-L, 256px, 16フレーム, patch=16, tubelet=2):
  時間方向: T/tubelet = 16/2 = 8 フレーム
  空間方向: H/patch × W/patch = 16 × 16 パッチ
  総パッチ数: 8 × 16 × 16 = 2048

マスク設定 (V-JEPA 2.1 config):
  Block 1-8: spatial_scale=[0.15, 0.15], temporal_scale=[1.0, 1.0]
             → 空間の15%を時間全体でマスク × 8ブロック
  Block 9-10: spatial_scale=[0.7, 0.7], temporal_scale=[1.0, 1.0]
              → 空間の70%を時間全体でマスク × 2ブロック

マスク生成手順:
  1. ブロックサイズをサンプリング:
     t ← uniform(T_min, T_max) × duration
     s ← uniform(s_min, s_max) × H × W  (空間面積)
     ar ← uniform(ar_min, ar_max)
     h = sqrt(s × ar), w = sqrt(s / ar)

  2. ブロック位置をサンプリング:
     start ← randint(0, duration-t)
     top   ← randint(0, height-h)
     left  ← randint(0, width-w)

  3. 3Dマスク作成:
     mask[start:start+t, top:top+h, left:left+w] = 0  (マスク)
     mask[その他] = 1  (コンテキスト)

  4. npredブロック分繰り返してAND結合

  5. 出力:
     masks_enc: コンテキストパッチのインデックス (B, N_ctx) ← エンコーダへ
     masks_pred: ターゲットパッチのインデックス (B, N_pred) ← Predictorが予測
```

---

## 処理フロー詳細

### Step 1: パッチ埋め込み

```python
# 動画入力
x: (B, 3, T=16, H=256, W=256)
→ PatchEmbed3D(kernel=(2,16,16)):
→ (B, T/2 * H/16 * W/16, D) = (B, 8*16*16, 1024) = (B, 2048, 1024)

# 画像入力 (V-JEPA 2.1)
x: (B, 3, H=256, W=256)
→ PatchEmbed2D(kernel=(16,16)):
→ (B, H/16 * W/16, D) = (B, 16*16, 1024) = (B, 256, 1024)
```

### Step 2: マスキングとエンコーダ処理

```python
# masks_enc: (B, N_ctx)  ← 可視パッチのインデックス
# N_ctx ≈ 2048 × (1 - masking_ratio) ≈ 500~800 程度

# x-エンコーダ (可視トークンのみ処理)
z = encoder(x, masks=masks_enc)  # (B, N_ctx, D=1024)

# 中間層出力を収集 (V-JEPA 2.1 Deep Self-Supervision)
# ViT-L (24層)の場合、例えば層[5, 11, 17, 23]から取得
# 各中間層出力: (B, N_ctx, D=1024)
# 連結後: (B, N_ctx, D*4=4096)
# MLP fusion後: (B, N_ctx, D=1024)

# y-エンコーダ (全トークン処理, 勾配なし, EMAパラメータ)
with torch.no_grad():
    h = target_encoder(x)  # (B, N_total=2048, D=1024)
    # 各中間層から (B, N_total, D) → 連結 → (B, N_total, D*4)
```

### Step 3: Predictor処理

```python
# Predictorへ入力
# コンテキストトークン: z (B, N_ctx, D)
# マスクトークン: learnable (B, N_pred, D_pred)  ← 位置情報付き

# 内部処理:
# 1. 線形投影: D → D_pred (例: 1024 → 384)
# 2. コンテキスト+マスクトークンを連結: (B, N_ctx+N_pred, D_pred)
# 3. パッチIDでソートして元の順序に整理
# 4. Transformer Blocks × L_pred 層 (例: 12層)
# 5. 線形投影: D_pred → D_out (Teacherと同じ次元)

# return_all_tokens=True (V-JEPA 2.1) の場合:
z_pred    # (B, N_pred, D_out) ← マスクトークンに対応する出力
z_context # (B, N_ctx, D_out)  ← コンテキストトークンに対応する出力
```

### Step 4: 損失計算

```python
# Teacherターゲットからマスクトークン位置を選択
h_pred = apply_masks(h, masks_pred)  # (B, N_pred, D)
h_ctx  = apply_masks(h, masks_enc)   # (B, N_ctx, D)

# L_predict: マスクトークン予測損失
L_predict = mean(|z_pred - h_pred|) / 1.0   # loss_exp=1 → L1

# L_context: コンテキストトークン予測損失 (距離重み付き)
d_weights = compute_distance_weights(masks_pred, masks_enc)  # (B, N_ctx)
L_context = mean(|z_context - h_ctx| * (1/d_weights)) / 1.0

# 合計損失
lambda_value = progressive_warmup(epoch)  # 0→0.5 (epoch 50-100でwarmup)
L_total = L_predict + lambda_value * L_context
```

### Step 5: EMAアップデート

```python
# Teacherエンコーダのパラメータを学生エンコーダのEMAで更新
m = 0.99925  # momentum係数 (固定)
for θ_teacher, θ_student in zip(target_encoder.params(), encoder.params()):
    θ_teacher = m * θ_teacher + (1 - m) * θ_student
```

---

## 学習設定・ハイパーパラメータ

### 事前学習 (V-JEPA 2.1 ViT-L)

| パラメータ | 値 |
|-----------|-----|
| エポック数 | (135K itr + 12K cooldown) |
| バッチサイズ | 128動画 + 2304画像 (グローバル) |
| 学習率 | warmup→定常(5.25e-4) |
| weight decay | 0.04 |
| EMA momentum | 0.99925 (固定) |
| dtype | bfloat16 |
| 動画解像度(main) | 16frames × 256×256 |
| 動画解像度(cooldown) | 64frames × 384×384 |
| 画像解像度(main) | 256×256 |
| 画像解像度(cooldown) | 512×512 |

### マスク設定

```yaml
# V-JEPA 2.1 masking config
mask:
  - num_blocks: 8
    spatial_scale: [0.15, 0.15]   # 空間の15%をマスク
    temporal_scale: [1.0, 1.0]    # 全フレームをマスク
    aspect_ratio: [0.75, 1.5]
  - num_blocks: 2
    spatial_scale: [0.7, 0.7]     # 空間の70%をマスク (大ブロック)
    temporal_scale: [1.0, 1.0]
    aspect_ratio: [0.75, 1.5]
```

### コンテキスト損失設定

```yaml
loss:
  loss_exp: 1.0                   # L1損失
  predict_all: true               # コンテキストも予測
  weight_distance_loss: true      # 距離重み付き
  lambda_value_vid: 0.5           # 動画用λ
  lambda_value_img: 0.5           # 画像用λ
  lambda_progressive: true        # epoch 50-100でwarmup
```

---

## 下流タスク性能

### 密予測タスク (frozen encoder + linear probe)

| タスク | V-JEPA 2 ViT-g | V-JEPA 2.1 ViT-G | 改善 |
|--------|---------------|-----------------|------|
| ADE20K (mIoU) | 22.2 | 47.9 | +116% |
| NYUv2 Depth (RMSE↓) | 0.682 | 0.307 | -55% |
| DAVIS Tracking (J&F) | – | 72.7 | – |
| Pascal VOC Seg (mIoU) | – | 85.0 | – |

### グローバル理解タスク (frozen encoder + attentive probe)

| タスク | V-JEPA 2 ViT-g | V-JEPA 2.1 ViT-G | 改善 |
|--------|---------------|-----------------|------|
| SSv2 (Acc) | 72.8 | 77.7 | +7% |
| K400 (Acc) | 85.1 | 88.1 | +4% |
| IN1K (Acc) | 82.2 | 85.5 | +4% |

### 予測・anticipationタスク

| タスク | 指標 | 結果 |
|--------|------|------|
| Ego4D STA | mAP@5 | 7.71 (SOTA) |
| EPIC-KITCHENS AA | Recall@5 | 40.8 (SOTA) |

---

## 形状ガイド

### テンソル記号定義

| 記号 | 意味 |
|------|------|
| B | バッチサイズ |
| C | チャネル数 (通常3 = RGB) |
| T | 時間フレーム数 |
| t | チューブレット数 = T/tubelet_size |
| H, W | 空間解像度 (高さ, 幅) |
| h, w | パッチグリッド数 = H/patch_size, W/patch_size |
| N | 総パッチ数 = t × h × w (動画) or h × w (画像) |
| N_ctx | コンテキスト(可視)パッチ数 |
| N_pred | ターゲット(マスク)パッチ数 |
| D | エンコーダ埋め込み次元 |
| D_pred | Predictor内部次元 |
| D_out | Predictor出力次元 (= D) |
| L | Predictorのレイヤー数 |
| K | 蒸留での階層レベル数 (通常4) |

### 主要テンソルの形状

```
入力動画:              (B, 3, T, H, W)
  ↓ PatchEmbed3D
パッチトークン:        (B, N, D)  N = (T/2) * (H/16) * (W/16)
  ↓ マスキング (masks_enc: (B, N_ctx))
コンテキストトークン:  (B, N_ctx, D)
  ↓ エンコーダ
エンコーダ出力:        (B, N_ctx, D)  [または (B, N_ctx, D*K) でK層連結]
  ↓ Predictor
  ├→ z_pred:           (B, N_pred, D_out)  マスクトークン予測
  └→ z_context:        (B, N_ctx, D_out)   コンテキストトークン予測

Teacherエンコーダ出力: (B, N, D)  [全トークン, 勾配なし]
  ↓ apply_masks
ターゲット(pred用):    (B, N_pred, D)
ターゲット(ctx用):     (B, N_ctx, D)

距離重み:              (B, N_ctx)  各コンテキストパッチの重み λ_i
```

### 標準的な数値例 (ViT-L, 256px, 16フレーム)

```
T=16, H=W=256, patch=16, tubelet=2
N = (16/2) * (256/16) * (256/16) = 8 * 16 * 16 = 2048

典型的なマスキング後:
  N_ctx ≈ 700~900 パッチ (コンテキスト)
  N_pred ≈ 1100~1400 パッチ (予測ターゲット)

エンコーダ次元:
  D = 1024 (ViT-L)
  D_pred = 384 (Predictor内部)
  D_out = 1024 (Predictor出力)
```

---

## FAQ

**Q: V-JEPA 2 と V-JEPA 2.1 はどう違う？**
A: 4つの主要変更点がある。①Dense Prediction Loss（コンテキストにも損失）、②Deep Self-Supervision（中間層監督）、③Multi-Modal Tokenizer（画像/動画専用Conv）、④VisionMix163Mデータセット。これらにより密予測性能が大幅向上。

**Q: なぜ可視トークンに損失をかけると密特徴が改善する？**
A: V-JEPA 2ではコンテキストトークンに損失がないため、モデルはコンテキストトークンをグローバル情報のアグリゲータとして使う（DINOのregisterトークンと類似）。距離重み付きL_contextを追加することで、各パッチが自分の局所空間情報を保持するよう強制される。

**Q: コンテキスト損失λを大きくすると分類性能が下がるのはなぜ？**
A: コンテキスト損失が大きすぎると、モデルが「コンテキストトークンをそのままコピー」する解に収束しやすく、グローバル表現が崩れる。Deep Self-Supervisionとλのウォームアップがこのトレードオフを緩和する。

**Q: EMAモメンタムはなぜ高い値(0.99925)を使う？**
A: Teacherエンコーダの出力を安定したターゲットとするため、急激な変化を防ぐ。モメンタムが低いと学生と教師が同一化してCollapseが起きやすい。

**Q: Predictorでトークンをソートする理由は？**
A: RoPE（Rotary Position Embedding）を使う場合、トークンの時空間位置に基づいて正しく位置符号化するため、元の位置順に並べ直す必要がある。

**Q: 蒸留はどのように行う？**
A: ①EMA Teacherを学習済みViT-G（固定）に置き換え、②学生エンコーダのEMAコピーを最終モデルとして保持、③損失は最終層のみ（Deep Self-Supervisionなし）、④Predictorを12層+最終Linear（Teacher次元へ投影）に変更。

---

## 参考文献

- **V-JEPA 2.1**: Mur-Labadia et al., 2026 ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482))
- **V-JEPA 2**: Assran et al., 2025
- **V-JEPA**: Bardes et al., 2024
- **I-JEPA**: Assran et al., 2023
- **JEPA原論文**: LeCun, 2022
- **DINOv2**: Oquab et al., 2023
- **DINOv3**: Siméoni et al., 2025
- **公式コード**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
