# DINOv3 Understanding - 簡略化疑似コード集

DINOv3 の理解を目的とした簡略化疑似コード集です。

論文: [DINOv3](https://arxiv.org/abs/2508.10104) (Meta AI Research, 2025)

## 📋 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [DINOv3の主要イノベーション](#dinov3の主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**DINOv3の特徴:**
- **自己教師あり学習 (SSL)**: ラベル不要でユニバーサルな視覚特徴を学習
- **大規模モデル**: ViT-7B (67億パラメータ) による圧倒的な表現力
- **Gram Anchoring**: 長時間学習での密特徴劣化を解決する新手法
- **マルチスケール蒸留**: 7Bモデルを ViT-S/B/L/H+ や ConvNeXt に蒸留

**タスク:**
- 画像分類 (線形プロービング)
- セマンティックセグメンテーション
- 単眼深度推定
- 物体検出
- 3D対応点推定
- 動画セグメンテーション追跡
- 教師なし物体発見

**性能:**

| タスク | データセット | DINOv3 (7B) | DINOv2 (ViT-g) | 改善幅 |
|--------|-------------|-------------|-----------------|--------|
| セグメンテーション (線形) | ADE20k mIoU | **55.9** | 49.5 | +6.4 |
| 深度推定 (線形) | NYUv2 RMSE | **0.309** | 0.372 | -0.063 |
| 画像分類 (線形) | ImageNet-1k | **88.4%** | 87.3% | +1.1 |
| 3D対応点 | NAVI recall | **64.4** | 60.1 | +4.3 |
| 動画追跡 | DAVIS J&F | **83.3** | 76.6 | +6.7 |
| 物体発見 | VOC07 CorLoc | **66.1** | 55.6 | +10.5 |
| 物体検出 (frozen) | COCO mAP | **66.1** | - | SOTA |

---

## アーキテクチャ全体像

```
入力画像群 (B, 3, H, W)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Augmentation: Multi-Crop                                │
│    - Global Crops: 2枚 (256×256)                                │
│    - Local Crops: 8枚 (112×112)                                 │
│    - マスク生成: ランダムブロック (10-50%パッチ)                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Student Backbone: ViT-7B + Axial RoPE                       │
│    - Patch Embedding: (B, 3, 256, 256) → (B, 256, 4096)        │
│    - CLS + 4 Storage Tokens → (B, 261, 4096)                   │
│    - 40 Transformer Blocks (SwiGLU FFN)                         │
│    - マスクされたパッチは mask_token で置換                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Teacher Backbone: EMA (momentum=0.999)                       │
│    - Global Crops のみ処理                                      │
│    - Student と同一アーキテクチャ                                │
│    - パラメータは Student の指数移動平均                          │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. DINO Head (画像レベル): CLS → MLP → K prototypes             │
│    - Student CLS: (B, 4096) → (B, 256K)                        │
│    - Teacher CLS: (B, 4096) → (B, 256K)                        │
│    - Sinkhorn-Knopp centering on Teacher                        │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. iBOT Head (パッチレベル): masked patches → MLP → K protos    │
│    - Student masked: (n_masked, 4096) → (n_masked, 96K)        │
│    - Teacher visible: (n_masked, 4096) → (n_masked, 96K)       │
│    - Sinkhorn-Knopp centering on Teacher                        │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Gram Teacher (frozen snapshot, optional)                     │
│    - 初期学習段階 (~200k iter) のモデル凍結コピー               │
│    - 高解像度 (2x) 入力で Gram 行列を計算                       │
│    - Student の密特徴構造を正則化                                │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Loss: L_DINO + L_iBOT + L_KoLeo + L_Gram                   │
│    - DINO: CLS token 自己蒸留 (cross-entropy)                   │
│    - iBOT: masked patch 再構成 (cross-entropy)                  │
│    - KoLeo: 特徴空間の均一性正則化                               │
│    - Gram: 二次統計量マッチング (Frobenius norm)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ファイル構成

### 1. [main_flow.py](main_flow.py)
**DINOv3の全体フロー**

メインクラス `DINOv3SSLFramework` で Teacher-Student 学習を実装:
- Student + Teacher バックボーン
- DINO / iBOT / KoLeo / Gram の4つの損失関数
- EMA による Teacher 更新

```python
class DINOv3SSLFramework(nn.Module):
    def forward_backward(self, data, teacher_temp):
        # data["collated_global_crops"]: (2*B, 3, 256, 256)
        # data["collated_local_crops"]: (8*B, 3, 112, 112)
        # data["collated_masks"]: (2*B, P) - P=256パッチ

        teacher_out = self.get_teacher_output(global_crops)
        student_out = self.get_student_output(global_crops, local_crops, masks)

        loss = self.compute_losses(teacher_out, student_out)
        # loss: scalar
        return loss
```

**重要ポイント:**
- Teacher は Global Crops のみ処理
- Student は Global + Local 全てを処理
- iBOT は Global Crops のマスクされたパッチのみ対象

---

### 2. [backbone.py](backbone.py)
**ViT-7B + Axial RoPE**

#### 🔑 **キー・イノベーション: Axial RoPE + Box Jittering**

```python
class DinoVisionTransformer(nn.Module):
    """
    従来手法 (DINOv2):
      - 学習可能な位置埋め込み
      - 異なる解像度への汎化が困難

    DINOv3:
      - Axial RoPE で相対位置をエンコード
      - Box Jittering: [-1,1] → [-s,s] (s∈[0.5,2]) でロバスト性向上
      - 任意の解像度・アスペクト比に対応
    """
```

#### Transformer Block 詳細

```python
class SelfAttentionBlock(nn.Module):
    """
    LayerNorm → Self-Attention (with RoPE) → LayerScale → Residual
    → LayerNorm → SwiGLU FFN → LayerScale → Residual

    入力: (B, 1+R+P, D) → 出力: (B, 1+R+P, D)
    """
```

---

### 3. [gram_anchoring.py](gram_anchoring.py)
**Gram Anchoring (核心的新手法)**

#### 🔑 **キー・イノベーション: 密特徴劣化の解決**

```python
class GramLoss(nn.Module):
    """
    問題:
      - 長時間 SSL 学習で密特徴 (パッチ特徴) が劣化
      - CLS とパッチの類似度が増加し空間局所性が失われる
      - DINOv2, Web-DINO で深刻 (ADE20k: 42.7 mIoU)

    解決:
      - 初期学習段階の Gram 行列 (二次統計量) を保存
      - Student の Gram 行列を Teacher の Gram 行列に近づける
      - 特徴自体ではなく「類似度構造」を保存 → 特徴は自由に移動可能

    効果:
      - ADE20k: 49.5 → 55.9 mIoU (+6.4)
      - DAVIS: 76.6 → 83.3 J&F (+6.7)
    """
```

---

### 4. [loss_computation.py](loss_computation.py)
**全損失関数の詳細実装**

#### DINO Loss (画像レベル自己蒸留)
- CLS token 間の交差エントロピー
- Sinkhorn-Knopp 正規化

#### iBOT Loss (パッチレベル再構成)
- マスクされたパッチの予測損失

#### KoLeo Loss (均一性正則化)
- 最近傍距離の対数の平均

#### Gram Loss (二次統計量マッチング)
- Frobenius ノルムで Gram 行列の差を最小化

---

### 5. [training_example.py](training_example.py)
**学習サンプルコード**

DINOv3 の学習ループを簡略化した実行可能なサンプル:
- ダミーデータでの学習
- 各損失の計算と表示
- EMA Teacher 更新

---

## DINOv3の主要イノベーション

### 1. **Gram Anchoring**
**問題**: SSL を長時間学習すると、グローバル精度は向上するが密特徴 (セグメンテーション、深度推定等) が劣化する

**解決**:
- 学習初期 (~200k iter) のモデルスナップショットを Gram Teacher として凍結
- Student の L2 正規化パッチ特徴の Gram 行列を計算: `G_S = X_S @ X_S^T` (P×P)
- Gram Teacher の Gram 行列との MSE を損失に追加: `L_Gram = ||G_S - G_G||_F^2`
- 特徴空間の移動は許容しつつ、パッチ間の類似度「構造」を維持

**高解像度 Gram**: Teacher に 2x 解像度で入力し bicubic でダウンサンプルすることで、より滑らかな Gram 行列を得る (+2 mIoU)

**実装**: [gram_anchoring.py](gram_anchoring.py)

---

### 2. **定数ハイパーパラメータスケジュール**
**問題**: コサインスケジュールは学習ホライズンを事前に決定する必要がある

**解決**:
- 学習率: ウォームアップ後は定数 0.0004
- 重み減衰: 定数 0.04
- EMA momentum: 定数 0.999
- 無期限の学習が可能

**実装**: [training_example.py](training_example.py)

---

### 3. **Axial RoPE + Box Jittering**
**問題**: 学習可能な位置埋め込みは異なる解像度への汎化が困難

**解決**:
- Axial RoPE: 各パッチに正規化座標 [-1, 1] を割り当て、相対位置をエンコード
- Box Jittering: 座標を [-s, s] (s ∈ [0.5, 2]) にランダムスケール → 解像度/アスペクト比のロバスト性向上

**実装**: [backbone.py](backbone.py)

---

### 4. **専用 LayerNorm の分離**
**問題**: Global Crops と Local Crops で特徴分布が異なる

**解決**:
- CLS/Storage tokens と Patch tokens で別々の LayerNorm
- Global Crops と Local Crops の CLS で別々の LayerNorm
- ImageNet kNN +0.2, ADE20k +1 mIoU

**実装**: [backbone.py](backbone.py)

---

### 5. **大規模データキュレーション (LVD-1689M)**
**問題**: データの品質と多様性が SSL 性能に直結

**解決**:
- ~170億 Instagram 画像から 16.89 億画像を選別
- DINOv2 埋め込みで階層的 k-means クラスタリング (5レベル)
- ImageNet-1k 同質バッチを 10% 混合

---

### 6. **マルチスケール蒸留**
**問題**: 7B モデルはデプロイに大きすぎる

**解決**:
- 7B Teacher から ViT-S/S+/B/L/H+ と ConvNeXt-T/S/B/L に蒸留
- 同じ DINO + iBOT + KoLeo 目的関数を使用 (固定 Teacher)
- 1M iter + 250k iter コサイン LR クールダウン
- Gram Anchoring は不要 (小モデルでは密特徴劣化が起きない)

---

## 処理フロー詳細

### 推論フロー

```python
# === 推論時 (frozen backbone) ===

# 入力
image = load_image()  # (1, 3, 512, 512)

# 1. パッチ埋め込み
patches = patch_embed(image)  # (1, 32, 32, 4096) → flatten → (1, 1024, 4096)

# 2. CLS + Storage tokens 追加
tokens = cat([cls_token, storage_tokens, patches])  # (1, 1029, 4096)
#         cls=1, storage=4, patches=1024

# 3. RoPE 計算
rope = compute_rope(H=32, W=32)  # (sin, cos) each (1024, 4096)

# 4. 40 Transformer Blocks
for block in blocks:
    tokens = block(tokens, rope=rope)  # (1, 1029, 4096)

# 5. 出力正規化
cls_token = norm(tokens[:, 0])           # (1, 4096)
patch_tokens = norm(tokens[:, 5:])       # (1, 1024, 4096)

# 下流タスクへ:
# - 分類: cls_token → linear → (1, num_classes)
# - セグメンテーション: patch_tokens → reshape → (1, 4096, 32, 32) → decoder
# - 検出: intermediate layers [10,20,30,40] → (1, 1024, 16384) → DETR
```

### 学習フロー

```python
# === 学習時 ===
for batch in dataloader:
    # 1. データ拡張
    global_crops = augment_global(batch)   # 2枚: (2*B, 3, 256, 256)
    local_crops = augment_local(batch)     # 8枚: (8*B, 3, 112, 112)
    masks = generate_masks()               # (2*B, 256) - ランダムブロック

    # 2. Teacher Forward (勾配なし)
    with torch.no_grad():
        t_cls = teacher.backbone(global_crops)["x_norm_clstoken"]  # (2*B, 4096)
        t_patch = teacher.backbone(global_crops)["x_norm_patchtokens"]  # (2*B, 256, 4096)
        t_cls_head = teacher.dino_head(t_cls)  # (2*B, 256K)
        t_cls_centered = sinkhorn_knopp(t_cls_head / teacher_temp)  # (2, B, 256K)
        t_patch_head = teacher.ibot_head(t_patch[masked])  # (n_masked, 96K)
        t_patch_centered = sinkhorn_knopp(t_patch_head / teacher_temp)

    # 3. Student Forward (マスクあり)
    s_out = student.backbone([global_crops, local_crops], masks=masks)
    s_cls_global = s_out[0]["x_norm_clstoken"]  # (2*B, 4096)
    s_cls_local = s_out[1]["x_norm_clstoken"]   # (8*B, 4096)
    s_patch = s_out[0]["x_norm_patchtokens"]     # (2*B, 256, 4096)

    s_cls_head = student.dino_head(cat([s_cls_global, s_cls_local]))
    # (10*B, 256K) → split to global (2, B, 256K) + local (8, B, 256K)

    s_patch_head = student.ibot_head(s_patch[masked])  # (n_masked, 96K)

    # 4. 損失計算
    loss_dino = dino_loss(s_cls_head, t_cls_centered)
    loss_ibot = ibot_loss(s_patch_head, t_patch_centered, masks)
    loss_koleo = koleo_loss(s_cls_global)

    # Gram Anchoring (Phase 2 以降)
    if use_gram:
        gram_target = gram_teacher.backbone(global_crops_2x)["x_norm_patchtokens"]
        gram_target = bicubic_downsample(gram_target)  # 2x → 1x
        loss_gram = gram_loss(s_patch, gram_target)  # Gram行列MSE

    loss = loss_dino + loss_ibot + 0.1 * loss_koleo + 2.0 * loss_gram

    # 5. 更新
    loss.backward()
    optimizer.step()

    # 6. EMA Teacher 更新
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data = 0.999 * t_param.data + 0.001 * s_param.data
```

---

## 学習データフォーマット

### データセット構成

```
LVD-1689M/
├── instagram/            # ~16.89億画像 (Instagram から選別)
│   ├── shard_000000/
│   │   ├── image_000.jpg
│   │   ├── image_001.jpg
│   │   └── ...
│   └── ...
├── imagenet1k/           # 128万画像 (10% 同質バッチ)
├── imagenet22k/          # 1400万画像
└── mapillary/            # ストリートレベル画像
```

### 学習バッチの構成

| 項目 | 値 | 形状 |
|------|-----|------|
| バッチサイズ | 4096 画像 | - |
| Global Crops | 2枚/画像 | (2×B, 3, 256, 256) |
| Local Crops | 8枚/画像 | (8×B, 3, 112, 112) |
| マスク | Global のみ | (2×B, 256) |
| マスク率 | 10-50% | ランダム |
| マスク適用確率 | 50% | - |
| 総トークン数/バッチ | ~3.7M | - |

### Gram Teacher 用入力

| 項目 | 値 | 形状 |
|------|-----|------|
| 高解像度入力 | 2x | (2×B, 3, 512, 512) |
| 出力パッチ | 32×32 | (2×B, 1024, 4096) |
| ダウンサンプル後 | 16×16 | (2×B, 256, 4096) |

---

## 形状ガイド

### 入力・中間・出力形状 (ViT-7B/16, 256×256入力)

| 段階 | 変数名 | 形状 | 説明 |
|------|--------|------|------|
| **入力** | image | `(B, 3, 256, 256)` | RGB画像 |
| **Patch Embed** | patches | `(B, 16, 16, 4096)` | パッチ埋め込み |
| **Flatten** | patches | `(B, 256, 4096)` | P=16×16=256 |
| **Token追加** | tokens | `(B, 261, 4096)` | 1 CLS + 4 Storage + 256 Patch |
| **Block出力** | tokens | `(B, 261, 4096)` | 40ブロック通過後 |
| **CLS出力** | cls | `(B, 4096)` | 正規化済み |
| **Patch出力** | patches | `(B, 256, 4096)` | 正規化済み |
| **DINO Head** | dino_out | `(B, 256K)` | K=256,000 prototypes |
| **iBOT Head** | ibot_out | `(n_masked, 96K)` | K=96,000 prototypes |

### モデルバリアント別

| モデル | params | embed_dim (D) | depth | heads | head_dim | FFN hidden | patch_size |
|--------|--------|---------------|-------|-------|----------|-----------|------------|
| ViT-S | 21M | 384 | 12 | 6 | 64 | 1536 | 16 |
| ViT-B | 86M | 768 | 12 | 12 | 64 | 3072 | 16 |
| ViT-L | 300M | 1024 | 24 | 16 | 64 | 4096 | 16 |
| ViT-H+ | 840M | 1280 | 32 | 20 | 64 | 5120 | 16 |
| ViT-g | 1.1B | 1536 | 40 | 24 | 64 | 6144 | 14 |
| **ViT-7B** | **6.7B** | **4096** | **40** | **32** | **128** | **8192** | **16** |

### 解像度別パッチ数 (patch_size=16)

| 解像度 | パッチグリッド | パッチ数 (P) | 総トークン数 (1+R+P) |
|--------|---------------|-------------|---------------------|
| 112×112 | 7×7 | 49 | 54 |
| 224×224 | 14×14 | 196 | 201 |
| 256×256 | 16×16 | 256 | 261 |
| 512×512 | 32×32 | 1024 | 1029 |
| 768×768 | 48×48 | 2304 | 2309 |
| 1024×1024 | 64×64 | 4096 | 4101 |

### 軸の意味

- **B**: バッチサイズ
- **D**: 埋め込み次元 (embed_dim)
- **P**: パッチ数 = (H/patch_size) × (W/patch_size)
- **R**: Storage tokens 数 (デフォルト: 4)
- **K**: プロトタイプ数 (DINO: 256K, iBOT: 96K)
- **H, W**: 画像の高さ・幅 (ピクセル)
- **num_heads**: Attention ヘッド数

---

## FAQ

### Q1: DINOv3 と DINOv2 の最大の違いは？

**A**: 最大の違いは **Gram Anchoring** です。DINOv2 では長時間学習すると密特徴 (セグメンテーション、深度推定に使うパッチ特徴) が劣化する問題がありました。DINOv3 はこの問題を Gram 行列の構造保存によって解決し、グローバル精度と密特徴品質の両立を実現しています。

| 比較項目 | DINOv2 | DINOv3 |
|----------|--------|--------|
| モデルサイズ | ViT-g (1.1B) | ViT-7B (6.7B) |
| 位置エンコーディング | 学習可能 | Axial RoPE |
| LRスケジュール | コサイン | 定数 |
| 密特徴品質 | 劣化 (>200k iter) | 維持 (Gram Anchoring) |
| ADE20k mIoU | 49.5 | **55.9** |
| 学習データ | 142M | **1,689M** |

---

### Q2: Gram Anchoring はなぜ二次統計量 (Gram 行列) を使うのか？

**A**: 特徴そのものをマッチングすると、特徴空間の自由な発展を阻害してしまいます。Gram 行列 `G = X @ X^T` (P×P) はパッチ間の類似度構造のみをキャプチャするため、特徴ベクトルは任意の回転・並進が許容されます。つまり、「どのパッチが似ているか」の関係性だけを保存し、特徴自体は自由に改善できます。

```python
# Gram 行列の概念:
X_S = F.normalize(student_patches, dim=-1)  # (P, D) L2正規化
X_G = F.normalize(gram_teacher_patches, dim=-1)  # (P, D)

G_S = X_S @ X_S.T  # (P, P) 全ペア類似度
G_G = X_G @ X_G.T  # (P, P)

loss_gram = F.mse_loss(G_S, G_G)  # 構造の差を最小化
```

---

### Q3: なぜ定数スケジュール (コサイン不使用) なのか？

**A**: コサインスケジュールは事前に学習ステップ数を決定する必要があります。DINOv3 は100万イテレーション以上の長時間学習を行うため、「いつ学習を止めるべきか」を事前に決定できません。定数スケジュールにより、学習をいつでも停止・再開でき、Gram Anchoring の効果と組み合わせることで無期限に学習を続けられます。

---

### Q4: Storage Tokens (Register Tokens) の役割は？

**A**: DINOv1/v2 では一部のパッチに異常な高ノルム (outlier) が発生し、密特徴を汚染する問題がありました。Storage tokens (4個) はこれらの情報を吸収する専用のバッファです。DINOv2 の Register Tokens と同じ概念で、パッチ特徴のクリーンさを維持します。

```python
# Token構成
tokens = cat([
    cls_token,       # (1, 1, D) - グローバル表現
    storage_tokens,  # (1, 4, D) - 情報バッファ (推論時は破棄)
    patch_tokens,    # (1, P, D) - 密特徴
])
# 推論時: CLS → 分類, Patch → 密タスク, Storage → 無視
```

---

### Q5: SwiGLU FFN と標準 MLP の違いは？

**A**: SwiGLU は GLU (Gated Linear Unit) の SiLU 版で、2つの並列線形変換のゲーティングを行います。

```python
# 標準 MLP:
hidden = GELU(W1 @ x)       # (B, N, 4D)
output = W2 @ hidden        # (B, N, D)

# SwiGLU FFN:
x1 = W1 @ x                 # (B, N, hidden)
x2 = W2 @ x                 # (B, N, hidden)
hidden = SiLU(x1) * x2      # ゲーティング
output = W3 @ hidden        # (B, N, D)
# hidden = int(D * 2/3) をアライメントしたサイズ
```

SwiGLU は標準 MLP よりも表現力が高く、LLM でも広く採用されています (LLaMA 等)。

---

### Q6: 蒸留モデルの性能はどの程度か？

**A**: ViT-H+ (840M) は ViT-7B (6.7B) の 1/8 のサイズながら、ほぼ同等の性能を達成しています。

| モデル | params | ADE20k mIoU | NYUv2 RMSE | IN-1k |
|--------|--------|-------------|------------|-------|
| ViT-B (蒸留) | 86M | 50.7 | 0.336 | 85.6 |
| ViT-L (蒸留) | 300M | 53.9 | 0.317 | 87.4 |
| ViT-H+ (蒸留) | 840M | 55.5 | 0.310 | 88.1 |
| ViT-7B (Teacher) | 6.7B | 55.9 | 0.309 | 88.4 |

蒸留では Gram Anchoring は不要です (小モデルでは密特徴劣化が発生しないため)。

---

### Q7: Masked K Bias とは何か？

**A**: QKV 線形層のバイアスで、K (Key) 部分のみバイアスをゼロマスクする手法です。Q と V にはバイアスを保持します。これにより Key のバイアスが除去され、Self-Attention の安定性が向上します。

```python
class LinearKMaskedBias(nn.Linear):
    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        # bias shape: (3 * D,) → Q部(D), K部(D), V部(D)
        # K部のバイアスをゼロに → Attention score のバイアスを除去
        return output
```

---

### Q8: Sinkhorn-Knopp 正規化の役割は？

**A**: Teacher 出力のソフトマックスを均一化する手法です。DINOv1 のセンタリング (平均減算) を置き換えます。反復的に行/列の正規化を行い、各プロトタイプが均等に使われるようにします。これにより mode collapse (全ての特徴が同じプロトタイプに集中) を防ぎます。

```python
def sinkhorn_knopp(teacher_output, temp, n_iter=3):
    Q = torch.exp(teacher_output / temp).T  # (K, B)
    Q /= Q.sum()  # 全体正規化
    for _ in range(n_iter):
        Q /= Q.sum(dim=1, keepdim=True) * K  # 行正規化
        Q /= Q.sum(dim=0, keepdim=True) * B  # 列正規化
    return Q.T  # (B, K)
```

---

### Q9: DINOv3 の計算コストは？

**A**: ViT-7B の事前学習は 256 GPU (H100-SXM5) で 1M イテレーション、約 61,440 GPU 時間 (47 MWh) です。DINOv2 の約 3 倍ですが、MetaCLIP の 1/3 以下です。

| モデル | GPU | ステップ | GPU時間 | 電力 (MWh) |
|--------|-----|---------|---------|-----------|
| DINOv3 | 256×H100 | 1M | 61,440 | 47 |
| DINOv2 | 64×A100 | 625k | 22,016 | 9.7 |
| MetaCLIP | 1024×A100 | 390k | 368,640 | 160 |

---

### Q10: DINOv3 の限界・制約は？

**A**: 主な制約は以下です:

1. **計算コスト**: 7B モデルの推論は ViT-L の約 20 倍の計算量
2. **テキスト理解**: LiT でテキスト整合しているが、ネイティブな言語理解ではない (後付け)
3. **学習データ**: Instagram ベースのため、医療画像やリモートセンシング等の特殊ドメインでの性能は不明
4. **パッチサイズ**: 16 に固定で、非常に細かいディテールの復元には限界
5. **蒸留の限界**: ViT-S/B 等の小モデルでは 7B Teacher の全能力を移転しきれない

実用上は蒸留モデル (ViT-L/H+) が最も有用で、7B はさらなる蒸留や研究用として位置づけられます。
