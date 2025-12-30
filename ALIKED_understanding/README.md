# ALIKED Understanding - 簡略化疑似コード集

ALIKED (A LIghter Keypoint and descriptor Extraction network with Deformable transformation) の理解を目的とした簡略化疑似コード集です。

論文: [ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://arxiv.org/abs/2304.03608)

## 📋 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [ALIKEDの主要イノベーション](#alikedの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**ALIKEDの特徴:**
- **超軽量**: 0.19M (Tiny) ~ 0.98M (Normal-32) パラメータ
- **高速**: 125 FPS (Tiny) ~ 75 FPS (Normal-32) @ RTX 2060
- **SDDH**: Sparse Deformable Descriptor Head (変形可能記述子)
- **DKD**: Differentiable Keypoint Detection (sub-pixel精度)
- **Sparse NRE Loss**: 密→スパースに緩和した学習

**タスク:**
- Keypoint Detection (キーポイント検出)
- Descriptor Extraction (記述子抽出)
- Image Matching (画像マッチング)

**性能** (HPatches @ 5K keypoints):
- ALIKED-T(16): 78.70% MHA, 125 FPS
- ALIKED-N(16): 77.22% MHA, 77 FPS
- ALIKED-N(32): 74.44% MHA, 76 FPS

---

## アーキテクチャ全体像

```
入力画像 (B, 3, H, W)
    ↓
┌─────────────────────────────────────┐
│ 1. Feature Encoding (4ブロック)     │
│    Block1 (stride=1)  → F1          │
│    Block2 (stride=2)  → F2          │
│    Block3 (stride=8)  → F3 + DCN    │ ← Deformable Conv
│    Block4 (stride=32) → F4 + DCN    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Feature Aggregation              │
│    F1,F2,F3,F4 → Upsample & Concat  │ → F (B, dim, H, W)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Score Map Head (SMH)             │
│    Conv3x3 × 3 + Sigmoid            │ → S (B, 1, H, W)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. DKD (Differentiable Detection)   │
│    - NMS (2回適用)                   │
│    - Thresholding & Top-K           │
│    - Soft-argmax refinement         │ → Keypoints (B, N, 2)
└─────────────────────────────────────┘                Scores (B, N)
    ↓
┌─────────────────────────────────────┐
│ 5. SDDH (Sparse Deformable Desc)    │
│    - K×Kパッチ抽出                   │
│    - Mデformable位置推定             │
│    - 変形可能サンプリング             │
│    - 記述子集約                      │ → Descriptors (B, N, dim)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6. Loss (訓練時)                     │
│    - Reprojection Loss              │
│    - Peaky Loss                     │
│    - Sparse NRE Loss                │
│    - Reliable Loss                  │
└─────────────────────────────────────┘
```

---

## ファイル構成

### 1. [main_flow.py](main_flow.py) (14KB)
**ALIKEDの全体フロー**

メインクラス `ALIKED` で5段階処理を実装:
- Stage 1: Feature Encoding (Multi-scale)
- Stage 2: Feature Aggregation
- Stage 3: Score Map Estimation
- Stage 4: Differentiable Keypoint Detection
- Stage 5: Sparse Deformable Descriptor Extraction

```python
class ALIKED(nn.Module):
    def forward(self, images, top_k=5000, scores_th=0.2):
        # images: (B, 3, H, W)

        # Feature Encoding & Aggregation
        features = self.encode_and_aggregate(images)  # (B, dim, H, W)

        # Score Map
        score_map = self.score_head(features)  # (B, 1, H, W)

        # Keypoint Detection
        keypoints, scores = self.dkd(score_map)  # (B, N, 2), (B, N)

        # Descriptor Extraction
        descriptors = self.sddh(features, keypoints)  # (B, N, dim)

        return {'keypoints': keypoints, 'descriptors': descriptors, ...}
```

**モデルバリアント:**
| Model | Channels | Dim | M | Size | FPS | Use Case |
|-------|----------|-----|---|------|-----|----------|
| aliked-t16 | 8,16,32,64 | 64 | 16 | 0.19M | 126 FPS | Mobile/Real-time |
| aliked-n16 | 16,32,64,128 | 128 | 16 | 0.68M | 77 FPS | Standard |
| aliked-n32 | 16,32,64,128 | 128 | 32 | 0.98M | 76 FPS | High accuracy |

---

### 2. [blocks.py](blocks.py) (18KB)
**Building Blocks**

#### 🔑 **キー・イノベーション: SDDH (Sparse Deformable Descriptor Head)**

```python
class SDDH(nn.Module):
    """
    従来手法 (DMH: Dense Descriptor Map Head):
      - 全ピクセルで記述子計算: O(H × W × C^2)
      - メモリ: H × W × C
      - 冗長: キーポイント周辺以外も計算

    SDDH (Sparse Deformable Descriptor Head):
      - スパースキーポイントのみ: O(N × M × C)
      - メモリ: N × C
      - 効率: 300倍以上高速化!

    処理フロー:
      1. キーポイント周辺K×Kパッチ抽出
      2. M個の変形可能サンプル位置推定
      3. 変形可能サンプリング
      4. 記述子集約

    変形可能性:
      - 各キーポイントで最適なサンプリング位置を学習
      - 幾何学的変換に対する不変性を獲得
    """

    def forward(self, features, keypoints):
        # features: (B, dim, H, W)
        # keypoints: (B, N, 2)

        # Step 1: K×Kパッチ抽出
        patches = self._extract_patches(features, keypoints, K=3)

        # Step 2: Mデformable位置推定
        offsets = self.offset_net(patches)  # (B*N, M, 2)

        # Step 3: 変形可能サンプリング
        sampled = self._sample_features(features, keypoints + offsets)

        # Step 4: 集約
        descriptors = sampled.mean(dim=2)  # Average pooling
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        return descriptors  # (B, N, dim)
```

#### その他のブロック

- **ConvBlock**: 基本畳み込み (Conv3x3 × 2 + BN + SELU)
- **ResBlock**: 残差ブロック (Deformable Conv対応)
- **DeformableConv2d**: 変形可能畳み込み (DCNv2風)

**実装**: [blocks.py:81-340](blocks.py)

---

### 3. [soft_detect.py](soft_detect.py) (11KB)
**Differentiable Keypoint Detection (DKD)**

#### 🔑 **キー・イノベーション: Sub-pixel Soft-argmax**

```python
class DKD(nn.Module):
    """
    従来手法:
      - NMS → ピクセルレベルキーポイント
      - 微分不可能 → end-to-end学習困難

    DKD:
      - NMS → Soft-argmax refinement
      - 完全に微分可能
      - Sub-pixel精度

    処理:
      1. NMS (2回) → 局所最大値
      2. Thresholding & Top-K
      3. Soft-argmax → sub-pixel refinement
    """

    def _soft_argmax_refine(self, score_map, keypoints_pix):
        """
        Soft-argmax による Sub-pixel refinement

        数式:
        p_refined = Σ (p_i × exp(s_i / T)) / Σ exp(s_i / T)

        効果:
        - ピクセルレベル: [50, 60]
        - Sub-pixel: [50.3, 60.7] ← より正確!
        """
        # ウィンドウ内でsoftmax weighted average
        # Temperature T で鋭さ調整 (T=0.1)
```

#### Score Dispersity

```python
def compute_score_dispersity(score_map, keypoints):
    """
    スコアの分散度測定:
    - 低分散 → キーポイント位置が確実
    - 高分散 → キーポイント位置が不確実

    Peaky Lossの訓練目標
    """
```

**実装**: [soft_detect.py:43-180](soft_detect.py)

---

### 4. [loss_computation.py](loss_computation.py) (20KB)
**5つの損失関数**

#### 🔑 **キー・イノベーション: Sparse NRE Loss**

```python
class ALIKEDLossWrapper(nn.Module):
    """
    従来のNRE Loss (Neural Reprojection Error):
      - 密な記述子マップが必要
      - 2D確率マップ構築: (H, W)
      - Cross-entropy loss
      - GPU memory: 大量

    Sparse NRE Loss:
      - スパース記述子のみ
      - 1D確率ベクトル構築: (N_keypoints,)
      - GPU memory: 50倍削減!

    処理:
      1. 幾何的対応 → Reprojection Probability (binary)
      2. 記述子類似度 → Matching Probability (softmax)
      3. Cross-Entropy最小化
    """

    def _sparse_nre_loss(self, outputs_a, outputs_b, H_ab):
        # キーポイントAをImage Bに投影
        kpts_a_warped = warp_keypoints(kpts_a, H_ab)

        # 最近傍マッチング
        matches = find_nearest_neighbors(kpts_a_warped, kpts_b)

        for idx_a, idx_b in matches:
            # 記述子類似度
            sim = desc_b @ desc_a[idx_a]  # (N_b,)

            # Matching probability vector
            q_m = softmax((sim - 1.0) / t_des)  # (N_b,)

            # Loss: -log(matching probability)
            loss += -log(q_m[idx_b])
```

#### その他の損失

**1. Reprojection Loss**:
```python
def _reprojection_loss(outputs_a, outputs_b, H_ab):
    """
    キーポイントの幾何的整合性

    L_rp = 1/2 * (||pA - pBA|| + ||pB - pAB||)

    - pA → Image B に投影 → pAB
    - pB → Image A に投影 → pBA
    - 双方向距離を最小化
    """
```

**2. Peaky Loss**:
```python
def _peaky_loss(outputs_a, outputs_b):
    """
    スコアマップの鋭さ強化

    L_pk = mean(softmax(s_patch) · ||p - c||)

    - スコアが鋭くピークを持つように訓練
    - Score dispersity最小化
    """
```

**3. Reliable Loss**:
```python
def _reliable_loss(outputs_a, outputs_b):
    """
    記述子の信頼性考慮

    L_re = Σ (1 - r(pA, I_B)) * sA

    - 明確にマッチする → 高信頼性 → 高スコア
    - 曖昧なマッチ → 低信頼性 → 低スコア
    """
```

**損失重み**:
- w_rp: 1.0
- w_pk: 0.5
- w_ds: 5.0 (最も重要)
- w_re: 1.0

**実装**: [loss_computation.py:28-540](loss_computation.py)

---

## ALIKEDの主要イノベーション

### 1. **SDDH (Sparse Deformable Descriptor Head)**
**問題**: 密な記述子マップ計算は冗長かつ高コスト

**解決**:
- スパースキーポイントのみで記述子抽出
- 各キーポイントでM個の変形可能サンプル位置を学習
- 幾何学的変換に対する不変性を獲得

**効果**:
- 計算量: 300倍削減 (H×W×C² → N×M×C)
- メモリ: 50倍削減 (H×W×C → N×C)
- 精度: 同等以上 (+1.5% MHA)

**実装**: [blocks.py:81-340](blocks.py)

---

### 2. **DKD (Differentiable Keypoint Detection)**
**問題**: 従来のNMSは微分不可能 → end-to-end学習困難

**解決**:
- Soft-argmaxによるsub-pixel refinement
- 完全に微分可能
- Score dispersityで信頼度評価

**効果**:
- Sub-pixel精度: ±0.5ピクセル以下
- Reprojection error直接最適化可能
- MHA: +2.1% 向上

**実装**: [soft_detect.py:43-180](soft_detect.py)

---

### 3. **Sparse NRE Loss**
**問題**: 密なNRE Lossは大量のGPUメモリ必要

**解決**:
- 2D確率マップ → 1D確率ベクトルに緩和
- スパース記述子のみで学習
- 記述子マッチング品質を直接最適化

**効果**:
- GPUメモリ: 50倍削減 (800×800訓練で11GB → 3GB)
- 訓練速度: 3倍高速化
- 精度: ほぼ同等 (-0.5% MHA)

**実装**: [loss_computation.py:150-260](loss_computation.py)

---

### 4. **Deformable Convolution (Block3&4)**
**問題**: 通常のConvは固定受容野 → 幾何学的不変性不足

**解決**:
- 学習可能なオフセットで柔軟なサンプリング
- 各ピクセルで最適な受容野を獲得
- Block3&4のみ使用 (効率のため)

**効果**:
- 幾何学的不変性: 大幅向上
- Rotation/Scale/Viewpoint変化に頑健
- 計算量増加: わずか (+0.1 GFLOPs)

**実装**: [blocks.py:50-80](blocks.py)

---

## 処理フロー詳細

### 推論フロー

```python
# 1. 画像入力
images = torch.randn(2, 3, 640, 480)  # (B, 3, H, W)

# 2. Feature Encoding (4 blocks)
x1 = block1(images)                   # (B, 16, 640, 480) stride=1
x2 = block2(pool2(x1))                # (B, 32, 320, 240) stride=2
x3 = block3_dcn(pool3(x2))            # (B, 64, 80, 60)   stride=8
x4 = block4_dcn(pool4(x3))            # (B, 128, 20, 15)  stride=32

# 3. Feature Aggregation
f1 = ublock1(x1)                      # (B, 32, 640, 480)
f2 = ublock2(x2)                      # (B, 32, 640, 480)
f3 = ublock3(x3)                      # (B, 32, 640, 480)
f4 = ublock4(x4)                      # (B, 32, 640, 480)
features = concat([f1, f2, f3, f4])   # (B, 128, 640, 480)

# 4. Score Map
score_map = score_head(features)      # (B, 1, 640, 480)

# 5. Keypoint Detection
nms_map = simple_nms(score_map) × 2   # 2回NMS
keypoints_pix = threshold_topk(nms_map, top_k=1000, th=0.2)
keypoints = soft_argmax_refine(score_map, keypoints_pix)  # Sub-pixel
# keypoints: (B, N, 2), N ≤ 1000

# 6. Descriptor Extraction
patches = extract_patches(features, keypoints, K=3)       # (B*N, 128, 3, 3)
offsets = offset_net(patches)                             # (B*N, M, 2)
sampled = sample_features(features, keypoints + offsets)  # (B, N, M, 128)
descriptors = aggregate(sampled)                          # (B, N, 128)
descriptors = F.normalize(descriptors, p=2, dim=-1)

# 7. Output
outputs = {
    'keypoints': keypoints,       # (B, N, 2)
    'descriptors': descriptors,   # (B, N, 128)
    'scores': scores,             # (B, N)
    'score_map': score_map        # (B, 1, 640, 480)
}
```

---

### 訓練フロー

```python
# 1. 画像ペア
img_a = torch.randn(2, 3, 800, 800)
img_b = torch.randn(2, 3, 800, 800)
H_ab = get_homography()  # or depth, R, t for perspective

# 2. フォワードパス
out_a = model(img_a)
out_b = model(img_b)

# 3. 損失計算
losses = loss_wrapper(out_a, out_b, H_ab)
# losses = {
#     'loss_rp': Reprojection Loss
#     'loss_pk': Peaky Loss
#     'loss_ds': Sparse NRE Loss
#     'loss_re': Reliable Loss
#     'total_loss': Weighted sum
# }

# 4. バックプロパゲーション
total_loss = losses['total_loss']
total_loss.backward()
optimizer.step()
```

**訓練データセット**:
- MegaDepth (135 scenes, 10K pairs/scene): Perspective
- R2D2 Homographic: Homography augmentation

**訓練設定**:
- Resolution: 800×800
- Batch size: 2 (gradient accumulation: 6)
- Top-K keypoints: 400 (detection) + 400 (random)
- Steps: 100K
- Optimizer: Adam (betas: 0.9, 0.999)

---

## 学習データフォーマット

ALIKEDは2種類のデータセット形式をサポートしています:

### 1. Homographic Dataset (合成データ)

**特徴**:
- カメラパラメータ不要
- 1枚の画像から合成ペアを生成
- 幾何変換のみで対応関係を生成

**処理フロー**:
```python
# 1. 画像を1枚ロード
image = load_image("photo.jpg")  # (H, W, 3)

# 2. ランダムなHomography行列を生成
H_ab = generate_random_homography()  # (3, 3)
# H = Translation × Rotation × Scale × Shear × Perspective

# 3. 画像をワープして画像ペアを作成
image_a = image
image_b = warp_image(image, H_ab)

# 4. 幾何的対応関係は既知
# 点p_aは H_ab @ p_a で p_b に変換される
```

**データ構成**:
- 入力: 単一画像のみ
- ラベル: Homography行列 `H_ab` (3×3) - 実行時に生成
- 利点: 大量のデータを効率的に生成可能

### 2. Perspective Dataset (実画像ペア - MegaDepth)

**特徴**:
- 実際の画像ペアを使用
- COLMAPで事前計算されたカメラパラメータを利用
- 奥行き情報を含む

**データ構成**:
各シーンに対して `scene_info.npz` が存在:
```python
scene_info = {
    'intrinsics': (N, 3, 3),    # カメラ内部パラメータ K
    'poses': (N, 4, 4),          # カメラポーズ [R|t]
    'depth_paths': List[str],    # 各画像の深度マップパス
    'pairs': List[(i, j)],       # 画像ペアのインデックス
    'image_paths': List[str]     # 画像パス
}
```

**データ取得例**:
```python
# ペア (i, j) をロード
pair_idx = (5, 12)
image_a = load_image(scene_info['image_paths'][5])
image_b = load_image(scene_info['image_paths'][12])
depth_a = load_depth(scene_info['depth_paths'][5])

# カメラパラメータを取得
K_a = scene_info['intrinsics'][5]      # (3, 3)
K_b = scene_info['intrinsics'][12]     # (3, 3)

# カメラポーズから相対変換を計算
pose_a = scene_info['poses'][5]        # (4, 4) = [R_a | t_a]
pose_b = scene_info['poses'][12]       # (4, 4) = [R_b | t_b]

# 相対ポーズ計算: b = R_ab @ a + t_ab
T_ab = pose_b @ np.linalg.inv(pose_a)
R_ab = T_ab[:3, :3]  # (3, 3)
t_ab = T_ab[:3, 3]   # (3,)
```

**幾何的対応関係の計算**:
```python
# 画像A上の点 p_a = (x_a, y_a) に対して、画像B上の対応点を計算

# 1. ピクセル座標 → カメラ座標
p_cam_a = K_a_inv @ [x_a, y_a, 1] * depth_a[y_a, x_a]

# 2. カメラAからカメラBへ変換
p_cam_b = R_ab @ p_cam_a + t_ab

# 3. カメラ座標 → ピクセル座標
p_b = K_b @ p_cam_b
x_b, y_b = p_b[0] / p_b[2], p_b[1] / p_b[2]
```

### 訓練時の使用方法

**Homographic Dataset**:
```python
dataset = HomographicDataset(
    image_paths=glob("images/*.jpg"),
    image_size=(800, 800)
)
# 返り値: image_a, image_b, H_ab
```

**Perspective Dataset**:
```python
dataset = MegaDepthDataset(
    scene_info_dir="/path/to/megadepth/scene_info/",
    image_size=(800, 800)
)
# 返り値: image_a, image_b, depth_a, K_a, K_b, R_ab, t_ab
```

**詳細実装**: [training_data.py](training_data.py) を参照してください。データローダーの完全な実装が含まれています。

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | images | `(B, 3, H, W)` | RGB画像 |
| **Encoding** | x1 | `(B, c1, H, W)` | Block1出力 (stride=1) |
| | x2 | `(B, c2, H/2, W/2)` | Block2出力 (stride=2) |
| | x3 | `(B, c3, H/8, W/8)` | Block3出力 (stride=8, DCN) |
| | x4 | `(B, c4, H/32, W/32)` | Block4出力 (stride=32, DCN) |
| **Aggregation** | features | `(B, dim, H, W)` | 統合特徴 |
| **Score Map** | score_map | `(B, 1, H, W)` | スコアマップ [0,1] |
| **Keypoints** | keypoints | `(B, N, 2)` | Sub-pixel座標 `[x, y]` |
| | scores | `(B, N)` | キーポイントスコア [0,1] |
| **Descriptors** | descriptors | `(B, N, dim)` | L2正規化記述子 |
| **SDDH内部** | patches | `(B*N, dim, K, K)` | キーポイントパッチ (K=3) |
| | offsets | `(B*N, M, 2)` | 変形可能オフセット |
| | sampled | `(B, N, M, dim)` | サンプリング特徴 |

### 軸の意味

- **B**: バッチサイズ
- **3**: RGB チャネル
- **H, W**: 画像の高さ・幅
- **H/N, W/N**: N倍ダウンサンプリング後の高さ・幅
- **c1~c4**: 各ブロックのチャネル数 (モデルバリアントで異なる)
- **dim**: 記述子次元 (64 for Tiny, 128 for Normal)
- **N**: キーポイント数 (top_kで制御, 通常 ≤ 5000)
- **M**: 変形可能サンプル位置数 (16 or 32)
- **K**: パッチサイズ (3 or 5)

---

## FAQ

### Q1: ALIKEDと従来手法 (SuperPoint, R2D2) の違いは?

**A**: 主な違いは3点:

1. **記述子抽出方法**
   - SuperPoint/R2D2: 密な記述子マップを全体で計算 → サンプリング
   - ALIKED: スパースキーポイントのみで記述子抽出 (SDDH)
   - 効果: 300倍高速化, 50倍メモリ削減

2. **幾何学的不変性**
   - SuperPoint/R2D2: 通常のConv → 固定受容野
   - ALIKED: Deformable Conv + SDDH → 適応的受容野
   - 効果: Rotation/Scale/Viewpoint変化に頑健

3. **キーポイント検出**
   - SuperPoint/R2D2: 微分不可能なNMS
   - ALIKED: Differentiable Soft-argmax (DKD)
   - 効果: Sub-pixel精度, end-to-end学習

**結果**: HPatches 78.70% MHA @ 126 FPS (ALIKED-T vs SuperPoint 70.19% @ 53 FPS)

---

### Q2: SDDHの変形可能性はどう機能する?

**A**: 各キーポイントで最適なサンプリング位置を学習します。

**通常の記述子**:
```python
# 固定グリッド (例: 3x3)
positions = [
    [-1,-1], [-1,0], [-1,1],
    [0,-1],  [0,0],  [0,1],
    [1,-1],  [1,0],  [1,1]
]
descriptor = aggregate(features[keypoint + positions])
```

**SDDH (変形可能)**:
```python
# 学習可能なオフセット
offsets = offset_net(patch)  # Network learns optimal positions
positions = deformable_positions + offsets  # Adaptive!

# 例: Rotated feature
offsets = [
    [-0.7,0.7], [0,1], [0.7,0.7],
    [-1,0],     [0,0], [1,0],
    [-0.7,-0.7],[0,-1],[0.7,-0.7]
]  # Automatically adapts to rotation!

descriptor = aggregate(features[keypoint + offsets])
```

**利点**:
- Rotation: オフセットが回転方向に適応
- Scale: オフセットの大きさが適応
- Perspective: 複雑な変形にも対応

**可視化**: 論文 Fig.7 参照

---

### Q3: Sparse NRE Lossのメモリ削減効果は?

**A**: 約50倍のメモリ削減です。

**Dense NRE Loss**:
```python
# 密な記述子マップ必要
desc_map_a = dense_head(features_a)  # (B, dim, H, W)
desc_map_b = dense_head(features_b)  # (B, dim, H, W)

# 2D確率マップ構築
prob_map_a = compute_probability_map(desc_map_a, desc_map_b)
# (B, H, W, H, W) ← 非常に大きい!

# Memory: B × H × W × dim × 4 bytes
# 例: 2 × 800 × 800 × 128 × 4 = 655 MB per batch
```

**Sparse NRE Loss (ALIKED)**:
```python
# スパース記述子のみ
desc_a = sddh(features_a, keypoints_a)  # (B, N, dim)
desc_b = sddh(features_b, keypoints_b)  # (B, N, dim)

# 1D確率ベクトル構築
for each match (kpt_a, kpt_b):
    prob_vec = softmax(desc_b @ desc_a)  # (N_b,) ← 小さい!

# Memory: B × N × dim × 4 bytes
# 例: 2 × 800 × 128 × 4 = 0.8 MB per batch
```

**削減率**: 655 MB / 0.8 MB ≈ 820倍 (実際は~50倍、他の要因含む)

**訓練可能性**:
- Dense: 800×800訓練で11GB GPU必要
- Sparse: 800×800訓練で3GB GPU (Batch size=2, accumulation=6)

---

### Q4: Soft-argmaxのSub-pixel精度はどの程度?

**A**: 約±0.3ピクセルの精度です。

**ピクセルレベル検出 (通常のNMS)**:
```python
# Pixel-level keypoint
kpt_pix = [50, 60]  # Integer coordinates

# 実際の最大値位置が [50.7, 60.3] だとしても
# [50, 60] に丸められる → 誤差 ±0.5ピクセル
```

**Soft-argmax refinement (ALIKED)**:
```python
# Score patch (5x5 window around pixel [50, 60])
scores = [
    [0.1, 0.2, 0.3, 0.2, 0.1],
    [0.2, 0.4, 0.6, 0.4, 0.2],
    [0.3, 0.6, 0.9, 0.6, 0.3],  # Center at [50, 60]
    [0.2, 0.4, 0.6, 0.4, 0.2],
    [0.1, 0.2, 0.3, 0.2, 0.1]
]

# Weighted average
weights = softmax(scores / temperature)
refined = Σ (position × weights)
# Result: [50.7, 60.3] ← Sub-pixel!
```

**実験結果** (HPatches Repeatability):
- Pixel-level: 40.2%
- Sub-pixel (DKD): 43.4% (+3.2%)

**可視化**: 論文 Table IX, Row 7 vs Row 8

---

### Q5: M (サンプル位置数) はどう選ぶ?

**A**: 速度と精度のトレードオフです。

**実験結果** (IMW-validation):

| M | GFLOPs | Running Time | mAA(10°) | MS@3 |
|---|--------|--------------|----------|------|
| 8 | 3.48 | 0.28 ms | 64.72% | 88.28% |
| **16** | **4.05** | **0.57 ms** | **65.39%** | **88.93%** |
| 24 | 4.62 | 0.86 ms | 67.59% | 90.29% |
| 32 | 4.62 | 1.14 ms | 67.78% | 90.12% |

**推奨**:
- **M=16**: 最良のバランス (ALIKED-N16)
  - 速度: 77 FPS
  - 精度: 77.22% MHA
  - パラメータ: 0.68M

- M=32: 高精度用途 (ALIKED-N32)
  - 速度: 76 FPS (わずかに遅い)
  - 精度: 74.44% MHA
  - パラメータ: 0.98M

**理由**: M=16で十分な受容野を確保、M>16は性能飽和

---

### Q6: なぜBlock3&4のみDeformable Conv?

**A**: 効率と性能のバランスです。

**全ブロックでDCN使用した場合**:
```python
# Block1 (H×W) + DCN
block1_dcn_flops = H × W × c1^2 × K^2 × 2
# = 640 × 480 × 16^2 × 9 × 2 = 354M

# Block2 (H/2×W/2) + DCN
block2_dcn_flops = (H/2) × (W/2) × c2^2 × K^2 × 2
# = 320 × 240 × 32^2 × 9 × 2 = 354M

# Total: 708M additional FLOPs!
```

**Block3&4のみDCN使用 (ALIKED)**:
```python
# Block3 (H/8×W/8) + DCN
block3_dcn_flops = (H/8) × (W/8) × c3^2 × K^2 × 2
# = 80 × 60 × 64^2 × 9 × 2 = 35M

# Block4 (H/32×W/32) + DCN
block4_dcn_flops = (H/32) × (W/32) × c4^2 × K^2 × 2
# = 20 × 15 × 128^2 × 9 × 2 = 8.8M

# Total: 44M additional FLOPs (acceptable!)
```

**性能比較** (IMW-validation):
- No DCN: 57.00% mAA(10°)
- Block3&4 DCN: 63.58% mAA(10°) (+6.58%)
- All blocks DCN: 64.1% mAA(10°) (+0.52%, not worth it)

**結論**: Block3&4のDCNで十分な幾何学的不変性を獲得

---

### Q7: 訓練時と推論時で何が違う?

**A**: 主に3点異なります。

**1. キーポイント数**:
```python
# 訓練時
top_k = 400  # DKD detected
random_k = 400  # Randomly sampled
total = 800  # More diverse for training

# 推論時
top_k = 1000~5000  # User specified
random_k = 0
total = top_k
```

**2. 損失計算**:
```python
# 訓練時
outputs_a = model(img_a)
outputs_b = model(img_b)
losses = loss_wrapper(outputs_a, outputs_b, H_ab)
total_loss.backward()  # Backpropagation

# 推論時
outputs = model(img)
# No loss, no backprop
```

**3. NMS適用回数**:
```python
# 訓練時
nms_map = simple_nms(score_map)  # 1回のみ (高速化)

# 推論時
nms_map = score_map
for _ in range(2):
    nms_map = simple_nms(nms_map)  # 2回 (精度向上)
```

**その他は同一**: Feature extraction, DKD, SDDH は同じ処理

---

### Q8: カスタムCUDA実装の役割は?

**A**: パッチ抽出の高速化です。

**PyTorch実装 (標準)**:
```python
def extract_patches_pytorch(features, keypoints, K):
    # Grid sample使用
    patches = F.grid_sample(features, grid)
    # 速度: ~1.0 ms (1000 keypoints)
```

**カスタムCUDA実装**:
```cpp
// custom_ops/get_patches_cuda.cu
__global__ void extract_patches_kernel(...) {
    // Optimized parallel extraction
}
```

```python
def extract_patches_cuda(features, keypoints, K):
    # Custom CUDA kernel
    patches = get_patches_cuda.forward(features, keypoints, K)
    # 速度: ~0.3 ms (1000 keypoints) ← 3倍高速!
```

**利点**:
- 並列化最適化
- メモリアクセスパターン最適化
- Backward pass最適化

**Fallback**: CUDA利用不可時は自動的にPyTorch実装使用

---

### Q9: ALIKEDの限界は?

**A**: 主に2つの限界があります。

**1. 大規模スケール&視点変化**:
```
問題: スケール差4倍以上 + 視点変化大

例: 遠景ビル (scale 1x) ↔ 近景ビル (scale 5x, viewpoint 45°)

結果:
- ASLFeat(MS): 数マッチ回復 (multi-scale strategy)
- DISK: 数マッチ回復 (強力な記述子)
- ALIKED: ほぼ失敗

理由:
- Single-scale feature extraction
- Deformable convは1レイヤーのみ → 限定的なモデリング
```

**解決策**:
- Multi-scale matching strategy追加
- Deformable convを複数レイヤーに (コスト増)
- Learned matcher (SuperGlue等) 併用

**2. ハードウェアフレンドリーでない**:
```
問題:
- Grid sampling: 標準演算でない
- Deformable conv: 特殊な実装必要
- 32-bit float descriptors: メモリ帯域幅

モバイルデプロイ時:
- TensorRT最適化必要
- 量子化対応必要
- カスタムカーネル実装必要
```

**今後の方向**:
- Binary descriptors (1-bit) 検討
- Hardware-friendly architecture設計

**実験結果**: 論文 Fig. 8, Section VI-E参照

---

### Q10: ALIKEDのベストユースケースは?

**A**: 以下の用途に最適です。

**1. リアルタイムSLAM**:
```
要件:
- 高速: >60 FPS
- 低レイテンシ
- 適度な精度

ALIKED-T(16):
- 126 FPS @ RTX 2060
- 0.19M parameters
- 78.70% MHA (SuperPointより高い)

用途: ドローンSLAM, ロボットナビゲーション
```

**2. 標準画像マッチング**:
```
要件:
- 高精度
- 適度な速度
- 幾何学的不変性

ALIKED-N(16):
- 77 FPS @ RTX 2060
- 77.22% MHA (SOTA級)
- Rotation/Scale頑健

用途: SfM, 3D再構成, Visual Localization
```

**3. エッジデバイス**:
```
要件:
- 超軽量
- 低メモリ
- バッテリー効率

ALIKED-T(16):
- 0.19M parameters (SuperPointの1/7)
- 1.37 GFLOPs (SuperPointの1/19)
- 推論: 8ms @ GPU

用途: モバイルAR, エッジカメラ
```

**避けるべきケース**:
- 超大規模スケール変化 (>4x)
- 極端な視点変化 + スケール変化
- 超高精度要求 (DISK/ASLFeat推奨)

---

## まとめ

ALIKEDは以下の3つのイノベーションで超軽量・高速・高精度を実現:

1. **SDDH**: Sparse Deformable Descriptor Head
   - 計算量300倍削減
   - メモリ50倍削減
   - 幾何学的不変性獲得

2. **DKD**: Differentiable Keypoint Detection
   - Sub-pixel精度
   - End-to-end学習可能
   - Reprojection error直接最適化

3. **Sparse NRE Loss**: 密→スパース緩和
   - GPUメモリ50倍削減
   - 訓練速度3倍向上
   - 高解像度訓練可能

**性能**: HPatches 78.70% MHA @ 126 FPS (ALIKED-T16)

**用途**:
- リアルタイムSLAM (ドローン, ロボット)
- 標準画像マッチング (SfM, 3D再構成)
- エッジデバイス (モバイルAR, 組み込み)

**推奨設定**:
- Real-time: ALIKED-T(16) - 126 FPS
- Balanced: ALIKED-N(16) - 77 FPS (推奨)
- Accuracy: ALIKED-N(32) - 76 FPS

**制限**:
- 大規模スケール&視点変化: Multi-scale strategy推奨
- モバイル最適化: TensorRT/量子化必要

---

## 参考文献

- 論文: [ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://arxiv.org/abs/2304.03608)
- コードベース: Original ALIKED implementation
- 関連研究:
  - SuperPoint (2018): Homographic Adaptation
  - R2D2 (2019): Repeatability and Reliability
  - DISK (2020): Reinforcement Learning
  - ALIKE (2022): Differentiable Keypoint Detection (ALIKEDの前身)
  - DCNv2 (2019): Deformable Convolution

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装とは異なる場合があります。
