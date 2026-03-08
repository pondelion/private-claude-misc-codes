# LightGlue Understanding - 簡略化疑似コード集

LightGlue (Local Feature Matching at Light Speed) の理解を目的とした簡略化疑似コード集です。

論文: [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643) (ICCV 2023)

## 📋 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [LightGlueの主要イノベーション](#lightglueの主要イノベーション)
- [SuperGlueとの比較](#superglueとの比較)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**LightGlueの特徴:**
- **高速**: SuperGlueの2.5倍以上高速（Adaptive機構により）
- **高精度**: SuperGlueと同等以上の精度
- **学習容易**: 2 GPU-daysで訓練可能（SuperGlueは7+ days）
- **適応的**: 画像ペアの難易度に応じて計算量を調整

**タスク:**
- Local Feature Matching（局所特徴量マッチング）
- Sparse Correspondence Estimation（スパース対応推定）
- Outlier Rejection（外れ値除去）

**性能** (MegaDepth-1500, SuperPoint features):
| Method | AUC@5° | AUC@10° | AUC@20° | Time (ms) |
|--------|--------|---------|---------|-----------|
| SuperGlue | 49.7 | 67.1 | 80.6 | 70.0 |
| SGMNet | 43.2 | 61.6 | 75.6 | 73.8 |
| **LightGlue** | **49.9** | **67.0** | **80.1** | **44.2** |
| LightGlue (adaptive) | 49.4 | 67.2 | 80.1 | **31.4** |

---

## アーキテクチャ全体像

```
入力: 2つの画像からの局所特徴量
    Image A: keypoints (M, 2), descriptors (M, D)
    Image B: keypoints (N, 2), descriptors (N, D)
        ↓
┌─────────────────────────────────────────────┐
│ 1. Input Projection                          │
│    descriptors → state vectors (256-dim)     │
│    Linear(D, 256) if D != 256                │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ 2. Positional Encoding (Rotary)              │
│    Learnable Fourier Features                │
│    positions → rotary embeddings             │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Transformer Layers × L (L=9)                         │
│    ┌─────────────────────────────────────┐              │
│    │ Self-Attention (Image A)            │← Rotary PE   │
│    │ Self-Attention (Image B)            │← Rotary PE   │
│    └─────────────────────────────────────┘              │
│                     ↓                                   │
│    ┌─────────────────────────────────────┐              │
│    │ Cross-Attention (A↔B)               │ Bidirectional│
│    │ 類似度行列を一度だけ計算              │              │
│    └─────────────────────────────────────┘              │
│                     ↓                                   │
│    ┌─────────────────────────────────────┐              │
│    │ Confidence Classifier               │              │
│    │ → 早期終了判定 (Adaptive Depth)     │              │
│    └─────────────────────────────────────┘              │
│                     ↓                                   │
│    ┌─────────────────────────────────────┐              │
│    │ Point Pruning                       │              │
│    │ → マッチ不可能点を除外 (Adaptive Width)│              │
│    └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ 4. Matching Head                             │
│    - Similarity Matrix: S = proj(x_A)ᵀ proj(x_B)  │
│    - Matchability: σ = sigmoid(Linear(x))    │
│    - Assignment: P = σ_A σ_B softmax(S) softmax(Sᵀ) │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ 5. Match Filtering                           │
│    - Mutual nearest neighbor check          │
│    - Threshold filtering (τ=0.1)            │
└─────────────────────────────────────────────┘
        ↓
出力:
    matches: List of (i, j) correspondences
    scores: Matching confidence scores
    stop: Layer at which inference stopped
```

---

## ファイル構成

### 1. [main_flow.py](main_flow.py)
**LightGlueの全体フロー**

メインクラス `LightGlue` でマッチング処理を実装:
- Input Projection（次元変換）
- Positional Encoding（Rotary PE）
- Transformer Layers（Self + Cross Attention）
- Adaptive Inference（Early Stop + Pruning）
- Matching Head（Assignment予測）

```python
class LightGlue(nn.Module):
    def forward(self, data):
        # data = {'image0': {...}, 'image1': {...}}

        # 1. 特徴量抽出
        kpts0, kpts1 = data['image0']['keypoints'], data['image1']['keypoints']
        desc0, desc1 = data['image0']['descriptors'], data['image1']['descriptors']

        # 2. 位置正規化 & エンコーディング
        kpts0 = normalize_keypoints(kpts0, size0)  # [-1, 1]
        encoding0 = self.posenc(kpts0)  # Rotary embeddings

        # 3. Transformer Layers
        for i in range(self.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

            # Early stopping check
            if self.check_if_stop(token0, token1, i):
                break

            # Point pruning
            if should_prune:
                desc0 = desc0[keep_mask0]
                desc1 = desc1[keep_mask1]

        # 4. Matching
        scores = self.log_assignment[i](desc0, desc1)
        matches = filter_matches(scores, threshold=0.1)

        return {'matches': matches, 'scores': scores, 'stop': i+1}
```

---

### 2. [transformer_blocks.py](transformer_blocks.py)
**Transformer構成要素**

#### 🔑 **キー・イノベーション: Rotary Positional Encoding**

```python
class LearnableFourierPositionalEncoding(nn.Module):
    """
    従来手法 (SuperGlue):
      - 絶対位置エンコーディング: MLP(p) → x に加算
      - 深いレイヤーで位置情報が薄れる

    LightGlue:
      - 相対位置エンコーディング (Rotary)
      - 各self-attentionレイヤーでquery/keyに適用
      - 位置情報が常に保持される

    数学的表現:
      R(p) = diag(R̂(b₁ᵀp), R̂(b₂ᵀp), ...)
      R̂(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]

      attention_score = qᵢᵀ R(pⱼ - pᵢ) kⱼ
    """

    def forward(self, positions):
        # positions: (B, N, 2) normalized to [-1, 1]
        projected = self.Wr(positions)  # (B, N, F_dim//2)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        return torch.stack([cosines, sines], 0)  # (2, B, N, F_dim)
```

#### 🔑 **キー・イノベーション: Bidirectional Cross-Attention**

```python
class CrossBlock(nn.Module):
    """
    従来手法:
      - A→Bの類似度: sim_AB = q_A @ k_Bᵀ
      - B→Aの類似度: sim_BA = q_B @ k_Aᵀ
      - 計算量: O(2 × N × M × d)

    LightGlue:
      - 共有key: k_A = k_B の投影を共有
      - sim_AB = k_Aᵀ @ k_B = sim_BAᵀ
      - 計算量: O(N × M × d) ← 半分!

    効果:
      - 20%の高速化
      - 精度に影響なし
    """

    def forward(self, x0, x1):
        # 共有key projection
        qk0 = self.to_qk(x0)  # Query/Key shared
        qk1 = self.to_qk(x1)
        v0 = self.to_v(x0)
        v1 = self.to_v(x1)

        # 類似度行列を一度だけ計算
        sim = einsum('bhid, bhjd -> bhij', qk0, qk1)

        # 双方向のattention
        attn01 = softmax(sim, dim=-1)      # A→B
        attn10 = softmax(sim.T, dim=-1).T  # B→A (転置で計算)

        m0 = einsum('bhij, bhjd -> bhid', attn01, v1)
        m1 = einsum('bhji, bhjd -> bhid', attn10.T, v0)

        return x0 + FFN([x0, m0]), x1 + FFN([x1, m1])
```

---

### 3. [matching_head.py](matching_head.py)
**マッチング予測ヘッド**

#### 🔑 **キー・イノベーション: Double Softmax + Matchability**

```python
class MatchAssignment(nn.Module):
    """
    従来手法 (SuperGlue):
      - Sinkhorn Algorithm (最適輸送問題)
      - 100イテレーション必要
      - Dustbin で unmatchable を表現
      - 計算量大、メモリ大

    LightGlue:
      - Double Softmax (行・列両方向)
      - Matchability score (unary)
      - 1回の計算で完了
      - 計算量小、メモリ小、勾配クリーン

    数学的表現:
      Similarity: Sᵢⱼ = Linear(xᴬᵢ)ᵀ Linear(xᴮⱼ)
      Matchability: σᵢ = sigmoid(Linear(xᵢ))
      Assignment: Pᵢⱼ = σᴬᵢ × σᴮⱼ × softmax(S)ᵢ × softmax(Sᵀ)ⱼ
    """

    def forward(self, desc0, desc1):
        # Similarity matrix
        mdesc0 = self.final_proj(desc0) / d**0.25
        mdesc1 = self.final_proj(desc1) / d**0.25
        sim = einsum('bmd, bnd -> bmn', mdesc0, mdesc1)

        # Matchability scores (unary)
        z0 = self.matchability(desc0)  # (B, M, 1)
        z1 = self.matchability(desc1)  # (B, N, 1)

        # Assignment matrix
        scores = sigmoid_log_double_softmax(sim, z0, z1)

        return scores
```

```python
def sigmoid_log_double_softmax(sim, z0, z1):
    """
    Log-domain assignment matrix computation

    数式:
      certainties = log_sigmoid(z0) + log_sigmoid(z1)ᵀ
      scores[i,j] = log_softmax(S, dim=1)[i,j]
                  + log_softmax(S, dim=0)[i,j]
                  + certainties[i,j]

      scores[:, -1] = unmatchable scores for image 0
      scores[-1, :] = unmatchable scores for image 1
    """
    b, m, n = sim.shape

    # Matchability (certainty that point has a match)
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).T

    # Double softmax in log domain
    scores0 = F.log_softmax(sim, dim=2)      # Row-wise
    scores1 = F.log_softmax(sim.T, dim=2).T  # Column-wise

    # Combined assignment (M+1 x N+1)
    scores = torch.zeros(b, m+1, n+1)
    scores[:, :m, :n] = scores0 + scores1 + certainties

    # Unmatchable scores (dustbin equivalent)
    scores[:, :-1, -1] = F.logsigmoid(-z0)  # A is unmatchable
    scores[:, -1, :-1] = F.logsigmoid(-z1)  # B is unmatchable

    return scores
```

---

### 4. [adaptive_inference.py](adaptive_inference.py)
**適応的推論（Adaptive Depth & Width）**

#### 🔑 **キー・イノベーション: Adaptive Depth (Early Stopping)**

```python
class TokenConfidence(nn.Module):
    """
    各レイヤーで予測の確信度を推定

    処理:
      1. 各点の状態から確信度を予測
      2. 確信度が高い点の割合を計算
      3. 閾値αを超えたら推論終了

    数式:
      cᵢ = sigmoid(MLP(xᵢ)) ∈ [0, 1]

      exit = (1/(M+N) × Σ[cᵢ > λₗ]) > α

    where:
      λₗ = 0.8 + 0.1 × exp(-4ℓ/L)  (層による閾値減衰)
      α = 0.95 (depth_confidence)
    """

    def __init__(self, dim):
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, desc0, desc1):
        # 勾配を切断（確信度予測はマッチング精度に影響させない）
        return self.token(desc0.detach()), self.token(desc1.detach())


def check_if_stop(confidences0, confidences1, layer_index, num_points):
    """
    早期終了の判定

    効果 (MegaDepth):
      - Easy pairs: 3-4層で終了 → 1.86倍高速化
      - Medium pairs: 5-6層で終了 → 1.33倍高速化
      - Hard pairs: 7-9層で終了 → 1.16倍高速化
    """
    confidences = torch.cat([confidences0, confidences1], -1)
    threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / n_layers)
    ratio_confident = (confidences >= threshold).float().mean()
    return ratio_confident > depth_confidence  # 0.95
```

#### 🔑 **キー・イノベーション: Point Pruning (Adaptive Width)**

```python
def get_pruning_mask(confidences, matchability_scores, layer_index):
    """
    マッチ不可能な点を早期に除外

    条件:
      - 確信度が高い (confident)
      - マッチ可能性が低い (unmatchable)
      → 以降のレイヤーから除外

    数式:
      unmatchable(i) = (cᵢ > λₗ) AND (σᵢ < β)

    where:
      λₗ: 層依存の確信度閾値
      β = 0.01 (width_confidence相当)

    効果:
      - 計算量: O(N²) → O((N-pruned)²)
      - Easy pairs: ~20%のpointsを除外
      - Hard pairs: ~28%のpointsを除外
    """
    keep = matchability_scores > (1 - width_confidence)  # 0.99

    # Low-confidence points are never pruned
    if confidences is not None:
        keep |= confidences <= confidence_threshold[layer_index]

    return keep  # True = keep, False = prune
```

---

### 5. [loss_computation.py](loss_computation.py)
**損失関数**

#### 🔑 **キー・イノベーション: Deep Supervision**

```python
class LightGlueLoss(nn.Module):
    """
    従来手法 (SuperGlue):
      - 最終レイヤーのみで損失計算
      - Sinkhorn が重いため中間出力困難

    LightGlue:
      - 全レイヤーで損失計算 (Deep Supervision)
      - 軽量なヘッドで各層の予測が可能
      - 収束が速く、中間レイヤーでも意味のある予測

    損失関数:
      L = (1/L) × Σₗ L_assignment(ℓ)

      L_assignment = L_positive + L_negative
    """

    def forward(self, predictions, ground_truth):
        total_loss = 0

        for layer_idx in range(n_layers):
            # 各レイヤーの予測を取得
            P = predictions[layer_idx]  # Assignment matrix

            # === Positive Loss (正しいマッチ) ===
            # Ground truth matches: M = {(i, j)}
            loss_positive = -log(P[i, j]) for (i, j) in matches
            loss_positive = loss_positive.mean() / |M|

            # === Negative Loss (unmatchable points) ===
            # Unmatchable in A: Ā = points with no correspondence
            loss_neg_A = -log(1 - σᴬᵢ) for i in Ā
            loss_neg_B = -log(1 - σᴮⱼ) for j in B̄
            loss_negative = (loss_neg_A.mean() + loss_neg_B.mean()) / 2

            total_loss += loss_positive + loss_negative

        return total_loss / n_layers
```

#### Confidence Classifier の学習

```python
def train_confidence_classifier(predictions):
    """
    2段階目の学習: 確信度分類器

    Ground truth:
      - 各レイヤーの予測が最終レイヤーと同じか？
      - label_i = (match_at_layer_ℓ == match_at_layer_L)

    損失:
      L_conf = BCE(cᵢ, label_i)

    注意:
      - 勾配は状態に伝播させない (detach)
      - マッチング精度に影響させない
    """
    for layer_idx in range(n_layers - 1):
        # 各点のマッチ結果
        match_at_layer = get_match(layer_idx)
        match_at_final = get_match(n_layers - 1)

        # Ground truth: 予測が一致しているか
        labels = (match_at_layer == match_at_final).float()

        # BCE loss
        confidence = confidence_classifiers[layer_idx](desc.detach())
        loss = F.binary_cross_entropy(confidence, labels)
```

---

## LightGlueの主要イノベーション

### 1. **Rotary Positional Encoding**
**問題**: SuperGlueの絶対位置エンコーディングは深いレイヤーで薄れる

**解決**:
- 相対位置を使用（Rotary encoding）
- 各self-attentionレイヤーでquery/keyに適用
- 位置情報が常に保持される

**効果**:
- 精度: +2% precision
- 幾何パターンの学習が容易

**実装**: [transformer_blocks.py](transformer_blocks.py)

---

### 2. **Bidirectional Cross-Attention**
**問題**: 標準のcross-attentionは類似度を2回計算

**解決**:
- Query/Keyの投影を共有
- 類似度行列を1回だけ計算
- 転置で双方向のattention取得

**効果**:
- 計算量: 50%削減（cross-attention部分）
- 全体: 20%高速化
- 精度: 変化なし

**実装**: [transformer_blocks.py](transformer_blocks.py)

---

### 3. **Double Softmax + Matchability**
**問題**: SuperGlueのSinkhorn Algorithmは重い

**解決**:
- 行方向・列方向の両Softmax
- 別途Matchability scoreで unmatchableを表現
- Dustbinを分離して表現

**効果**:
- 計算量: 大幅削減（100イテレーション → 1回）
- 勾配: よりクリーン
- 学習: 安定化

**実装**: [matching_head.py](matching_head.py)

---

### 4. **Deep Supervision**
**問題**: SuperGlueは最終レイヤーのみで教師信号

**解決**:
- 全レイヤーで損失計算
- 中間レイヤーでも意味のある予測を強制
- Early stoppingの前提条件

**効果**:
- 収束: 3倍高速化
- 学習コスト: 2 GPU-days（SuperGlue: 7+ days）

**実装**: [loss_computation.py](loss_computation.py)

---

### 5. **Adaptive Depth (Early Stopping)**
**問題**: 簡単なペアでも全レイヤー計算は無駄

**解決**:
- 確信度分類器で各レイヤーの予測信頼度を推定
- 十分な確信度に達したら推論終了
- 層依存の閾値で早期レイヤーの不確実性を考慮

**効果**:
- Easy pairs: 平均4.7層で終了、1.86倍高速化
- Medium pairs: 平均5.5層で終了、1.33倍高速化
- Hard pairs: 平均6.9層で終了、1.16倍高速化

**実装**: [adaptive_inference.py](adaptive_inference.py)

---

### 6. **Point Pruning (Adaptive Width)**
**問題**: マッチ不可能な点も後続レイヤーで処理

**解決**:
- 早期にマッチ不可能と判断された点を除外
- Attentionの計算量O(N²)を削減
- 探索空間を縮小

**効果**:
- 平均23.7%の点を除外
- 特にHard pairs（低overlap）で効果的
- 計算量削減 + 精度維持

**実装**: [adaptive_inference.py](adaptive_inference.py)

---

## SuperGlueとの比較

| 側面 | SuperGlue | LightGlue |
|------|-----------|-----------|
| **位置エンコーディング** | 絶対位置 (MLP) | 相対位置 (Rotary) |
| **Cross-Attention** | 標準 (2回計算) | 双方向 (1回計算) |
| **Assignment** | Sinkhorn (100 iter) | Double Softmax |
| **Unmatchable** | Dustbin (entangled) | Matchability (separated) |
| **教師信号** | 最終層のみ | 全層 (Deep Supervision) |
| **適応性** | なし | Depth + Width |
| **学習時間** | 7+ GPU-days | 2 GPU-days |
| **推論速度** | 70ms | 31-44ms |
| **精度 (AUC@5°)** | 49.7% | 49.4-49.9% |

---

## 処理フロー詳細

### 推論フロー

```python
# 1. 入力データ準備
data = {
    'image0': {
        'keypoints': torch.randn(B, M, 2),      # (B, M, 2) ピクセル座標
        'descriptors': torch.randn(B, M, 256),  # (B, M, D) 記述子
        'image_size': torch.tensor([H, W])      # 画像サイズ
    },
    'image1': {
        'keypoints': torch.randn(B, N, 2),
        'descriptors': torch.randn(B, N, 256),
        'image_size': torch.tensor([H, W])
    }
}

# 2. キーポイント正規化
kpts0 = normalize_keypoints(kpts0, size0)  # [-1, 1]に正規化
kpts1 = normalize_keypoints(kpts1, size1)

# 3. 入力投影
desc0 = input_proj(desc0)  # (B, M, 256)
desc1 = input_proj(desc1)  # (B, N, 256)

# 4. 位置エンコーディング（キャッシュ）
encoding0 = posenc(kpts0)  # (2, B, M, head_dim)
encoding1 = posenc(kpts1)  # (2, B, N, head_dim)

# 5. Transformerレイヤー
for i in range(9):  # L=9 layers
    # Self-attention (with Rotary PE)
    desc0 = self_attn(desc0, encoding0)
    desc1 = self_attn(desc1, encoding1)

    # Cross-attention (bidirectional)
    desc0, desc1 = cross_attn(desc0, desc1)

    if i < 8:  # Last layer: no early stop
        # Confidence check
        token0, token1 = token_confidence[i](desc0, desc1)
        if check_if_stop(token0, token1, i, M + N):
            break

        # Point pruning
        if desc0.shape[1] > 1024:
            keep0 = get_pruning_mask(token0, matchability0, i)
            desc0 = desc0[:, keep0]
            encoding0 = encoding0[..., keep0, :]

# 6. Matching head
scores, sim = log_assignment[i](desc0, desc1)
# scores: (B, M+1, N+1) log assignment matrix

# 7. Match filtering
m0, m1, mscores0, mscores1 = filter_matches(scores, threshold=0.1)

# 8. 出力
output = {
    'matches0': m0,           # (B, M) 各点のマッチ先 (-1 = unmatched)
    'matches1': m1,           # (B, N)
    'matching_scores0': mscores0,  # (B, M) マッチ信頼度
    'matching_scores1': mscores1,  # (B, N)
    'matches': matches,       # List[(Si, 2)] バッチごとのマッチペア
    'scores': mscores,        # List[(Si,)] マッチスコア
    'stop': i + 1,            # 終了レイヤー
    'prune0': prune0,         # (B, M) pruning layer
    'prune1': prune1          # (B, N)
}
```

---

### 訓練フロー

```python
# === Stage 1: マッチング予測の学習 ===

for epoch in range(50):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch)

        # Ground truth from reprojection
        matches_gt = compute_ground_truth_matches(
            kpts0, kpts1,
            homography=batch.get('H'),
            depth=batch.get('depth'),
            pose=batch.get('pose')
        )

        # Loss at each layer
        total_loss = 0
        for layer_idx in range(n_layers):
            P = predictions['scores_at_layer'][layer_idx]

            # Positive: correct matches
            loss_pos = -log(P[matches_gt]).mean()

            # Negative: unmatchable points
            loss_neg = -log(1 - matchability[unmatchable]).mean()

            total_loss += (loss_pos + loss_neg) / n_layers

        total_loss.backward()
        optimizer.step()

# === Stage 2: 確信度分類器の学習 ===

# マッチング部分の重みを固定
for param in model.transformers.parameters():
    param.requires_grad = False
for param in model.log_assignment.parameters():
    param.requires_grad = False

for epoch in range(10):
    for batch in dataloader:
        predictions = model(batch)

        # Ground truth: 各レイヤーの予測が最終レイヤーと同じか
        for layer_idx in range(n_layers - 1):
            match_at_layer = get_match_at_layer(predictions, layer_idx)
            match_at_final = get_match_at_layer(predictions, n_layers - 1)
            labels = (match_at_layer == match_at_final).float()

            confidence = token_confidence[layer_idx](desc.detach())
            loss = F.binary_cross_entropy(confidence, labels)

        loss.backward()
        optimizer.step()
```

**訓練データセット**:
- **Pre-training**: Oxford-Paris 1M (synthetic homographies)
- **Fine-tuning**: MegaDepth (196 landmarks, 1M images)

**訓練設定**:
- Batch size: 32
- Keypoints: 2048 per image
- Learning rate: 1e-4 (pre-train), 1e-5 (fine-tune)
- GPU: 2× RTX 3090
- Time: ~2 GPU-days

---

## 学習データフォーマット

### 1. Homographic Dataset (Pre-training)

```python
# 合成Homography による自己教師あり学習
class HomographicDataset:
    """
    特徴:
    - 1枚の画像から合成ペア生成
    - 完全なground truth（ノイズなし）
    - 極端な変換も可能

    使用データ:
    - Oxford-Paris 1M distractors (170K images)
    """

    def __getitem__(self, idx):
        image = load_image(self.paths[idx])

        # Random homography
        H = sample_random_homography()

        # Warp image
        image_b = warp_perspective(image, H)

        # Extract features
        kpts_a, desc_a = extractor(image)
        kpts_b, desc_b = extractor(image_b)

        return {
            'image0': {'keypoints': kpts_a, 'descriptors': desc_a},
            'image1': {'keypoints': kpts_b, 'descriptors': desc_b},
            'H': H  # Ground truth transformation
        }
```

### 2. MegaDepth Dataset (Fine-tuning)

```python
# 実画像ペアによる教師あり学習
class MegaDepthDataset:
    """
    特徴:
    - 実際の画像ペア
    - SfM + MVS による depth/pose
    - 現実的な変化を含む

    データ構成:
    - 196 landmarks
    - 1M crowd-sourced images
    - Scene splits for train/val/test
    """

    def __getitem__(self, idx):
        scene_info = load_scene_info(self.scenes[idx])

        # Sample pair by covisibility
        img_a, img_b = sample_pair_by_overlap(scene_info)

        # Load depth and poses
        depth_a = load_depth(scene_info, img_a)
        pose_a = scene_info['poses'][img_a]
        pose_b = scene_info['poses'][img_b]
        K_a = scene_info['intrinsics'][img_a]
        K_b = scene_info['intrinsics'][img_b]

        # Extract features
        kpts_a, desc_a = extractor(img_a)
        kpts_b, desc_b = extractor(img_b)

        return {
            'image0': {'keypoints': kpts_a, 'descriptors': desc_a},
            'image1': {'keypoints': kpts_b, 'descriptors': desc_b},
            'depth0': depth_a,
            'K0': K_a, 'K1': K_b,
            'T_0to1': pose_b @ inv(pose_a)
        }
```

### Ground Truth マッチの計算

```python
def compute_ground_truth_matches(kpts_a, kpts_b, H=None, depth=None, pose=None):
    """
    Homography の場合:
        kpts_a_warped = H @ kpts_a
        match if ||kpts_a_warped - kpts_b|| < 3px

    Depth + Pose の場合:
        p_cam_a = K_a^-1 @ kpts_a * depth_a
        p_cam_b = R @ p_cam_a + t
        kpts_a_warped = K_b @ p_cam_b
        match if ||kpts_a_warped - kpts_b|| < 3px

    Unmatchable:
        - Reprojection error > 5px for all points
        - No depth available
        - Large epipolar error
    """
    # ... implementation
    return matches, unmatchable_A, unmatchable_B
```

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | keypoints | `(B, M/N, 2)` | ピクセル座標 |
| | descriptors | `(B, M/N, D)` | D=128 or 256 |
| | image_size | `(B, 2)` or `(2,)` | [H, W] |
| **正規化** | kpts_norm | `(B, M/N, 2)` | [-1, 1]正規化 |
| **投影後** | desc | `(B, M/N, 256)` | 統一次元 |
| **Rotary PE** | encoding | `(2, B, M/N, head_dim)` | cos/sin |
| **Self-Attn内** | qkv | `(B, H, M/N, 3, head_dim)` | H=4 heads |
| **Cross-Attn内** | sim | `(B, H, M, N)` | 類似度行列 |
| **Confidence** | token | `(B, M/N)` | [0, 1] |
| **Matchability** | σ | `(B, M/N)` | [0, 1] |
| **Assignment** | scores | `(B, M+1, N+1)` | log確率 |
| **出力** | matches0 | `(B, M)` | マッチ先index |
| | matches | `List[(Si, 2)]` | ペアリスト |

### 軸の意味

- **B**: バッチサイズ
- **M**: Image Aのキーポイント数
- **N**: Image Bのキーポイント数
- **D**: 入力記述子次元 (128 for DISK/ALIKED, 256 for SuperPoint)
- **256**: 内部状態次元 (descriptor_dim)
- **H**: Attention heads (4)
- **head_dim**: 64 (= 256 / 4)
- **L**: レイヤー数 (9)

---

## FAQ

### Q1: LightGlueとSuperGlueの最大の違いは？

**A**: 3つの主要な違いがあります。

1. **Assignment方法**
   - SuperGlue: Sinkhorn Algorithm（最適輸送、100イテレーション）
   - LightGlue: Double Softmax + Matchability（1回の計算）

   ```python
   # SuperGlue
   for _ in range(100):
       scores = scores - scores.logsumexp(1)
       scores = scores - scores.logsumexp(0)

   # LightGlue
   scores = log_softmax(sim, dim=1) + log_softmax(sim, dim=0) + certainties
   ```

2. **適応性**
   - SuperGlue: 固定深度（9層全て実行）
   - LightGlue: Adaptive depth + width

   ```python
   # LightGlue
   for i in range(9):
       desc0, desc1 = transformer[i](desc0, desc1)
       if check_if_stop(confidence, i):
           break  # Early exit!
       desc0 = desc0[keep_mask]  # Point pruning!
   ```

3. **学習効率**
   - SuperGlue: 7+ GPU-days, 最終層のみ教師信号
   - LightGlue: 2 GPU-days, Deep Supervision

---

### Q2: Rotary Positional Encodingの利点は？

**A**: 3つの利点があります。

1. **相対位置の表現**
   ```python
   # 絶対位置 (SuperGlue)
   attention = softmax(q @ k.T)  # 位置情報なし

   # 相対位置 (LightGlue)
   attention = softmax(q @ R(p_j - p_i) @ k.T)  # 相対位置を考慮
   ```

2. **位置情報の保持**
   - 各レイヤーで再適用
   - 深いレイヤーでも位置を参照可能

3. **幾何パターンの学習**
   - 「右上にある点」「近くの点」などのパターン
   - 画像間で比較可能な表現

---

### Q3: Adaptive Depthはどう機能する？

**A**: 確信度ベースの早期終了です。

```
Layer 1: 50%の点が確信度高い → 続行
Layer 2: 60%の点が確信度高い → 続行
Layer 3: 80%の点が確信度高い → 続行
Layer 4: 96%の点が確信度高い → 停止! (>95%閾値)
```

**効果の例**:
| ペアタイプ | 平均停止層 | 速度向上 |
|-----------|-----------|---------|
| Easy (高overlap) | 4.7層 | 1.86倍 |
| Medium | 5.5層 | 1.33倍 |
| Hard (低overlap) | 6.9層 | 1.16倍 |

---

### Q4: Point Pruningの判定基準は？

**A**: 2つの条件の組み合わせです。

```python
prune_point = (confidence > threshold) AND (matchability < 0.01)
```

1. **確信度が高い**: 予測が安定している
2. **マッチ可能性が低い**: マッチ相手がいない

**どんな点が除外される？**
- オクルージョン領域
- 視野外
- テクスチャレス領域
- 動的物体

---

### Q5: 学習の2段階とは？

**A**: マッチング予測と確信度予測を分離して学習します。

**Stage 1: マッチング予測**
```python
# 目的: 正しい対応関係を予測
loss = -log(P[gt_matches]) - log(1 - σ[unmatchable])
```

**Stage 2: 確信度分類器**
```python
# 目的: 各レイヤーの予測が最終層と同じか予測
# 重要: マッチング部分は固定（勾配を伝播させない）
label = (match_at_layer_i == match_at_final_layer)
loss = BCE(confidence, label)
```

**なぜ分離？**
- 確信度予測はマッチング精度に影響させたくない
- 確信度は「早期終了の判定」にのみ使用
- Stage 1の収束後にStage 2を学習

---

### Q6: 対応可能な特徴量は？

**A**: 複数の局所特徴量に対応しています。

| 特徴量 | 入力次元 | 追加情報 | 用途 |
|--------|---------|---------|------|
| SuperPoint | 256 | なし | 一般用途 |
| DISK | 128 | なし | 高精度 |
| ALIKED | 128 | なし | 高速 |
| SIFT | 128 | scale, orientation | 古典的 |
| DoG-HardNet | 128 | scale, orientation | ハイブリッド |

```python
# 使用例
model = LightGlue(features='superpoint')  # or 'disk', 'aliked', 'sift'

# カスタム特徴量
model = LightGlue(features=None, input_dim=128)
```

---

### Q7: メモリ効率化のテクニックは？

**A**: 複数の手法を組み合わせています。

1. **FlashAttention**
   ```python
   # 標準attention: O(N²)メモリ
   # FlashAttention: O(N)メモリ
   if FLASH_AVAILABLE:
       v = F.scaled_dot_product_attention(q, k, v)
   ```

2. **Gradient Checkpointing**
   ```python
   # 訓練時のメモリ削減
   # Forward時に中間結果を保存しない
   # Backward時に再計算
   ```

3. **Mixed Precision**
   ```python
   with torch.autocast(device_type='cuda'):
       output = model(data)
   ```

4. **Point Pruning**
   - 不要な点を除外
   - O(N²) → O((N-pruned)²)

---

### Q8: Dense Matcherとの比較は？

**A**: 速度で大きく優位、精度は同等レベルです。

| Method | Type | AUC@5° | Time (ms) | Speed |
|--------|------|--------|-----------|-------|
| LoFTR | Dense | 52.8 | 181 | 5.5 fps |
| MatchFormer | Dense | 53.3 | 388 | 2.6 fps |
| ASpanFormer | Dense | 55.3 | 369 | 2.7 fps |
| **LightGlue** | Sparse | 49.9 | 44 | **22.7 fps** |
| LightGlue (adaptive) | Sparse | 49.4 | 31 | **32.3 fps** |

**トレードオフ**:
- Dense: 高精度、低速、メモリ大
- Sparse (LightGlue): やや低精度、高速、メモリ小

**LightGlueの強み**:
- リアルタイムアプリケーション
- 大規模SfM/SLAM
- エッジデバイス

---

### Q9: LightGlueの限界は？

**A**: 主に以下の制限があります。

1. **Sparse特徴量への依存**
   - キーポイント検出の品質に依存
   - テクスチャレス領域では困難

2. **極端な変化への対応**
   - 大きなスケール変化（4倍以上）
   - 極端な視点変化

3. **繰り返しパターン**
   - 建物のファサード
   - タイル状のテクスチャ
   → 誤マッチが発生しやすい

**対策**:
- Multi-scale特徴量の使用
- より強力な特徴量（DISK, ALIKED）
- 後処理（RANSAC, MAGSAC）

---

### Q10: 推奨設定は？

**A**: 用途に応じた設定を推奨します。

**リアルタイムSLAM**:
```python
model = LightGlue(
    features='superpoint',
    n_layers=9,
    depth_confidence=0.95,  # Adaptive depth ON
    width_confidence=0.99,  # Point pruning ON
    flash=True
)
# 30+ fps @ 2048 keypoints
```

**高精度SfM**:
```python
model = LightGlue(
    features='disk',  # or 'aliked'
    n_layers=9,
    depth_confidence=-1,   # Adaptive depth OFF
    width_confidence=-1,   # Point pruning OFF
    flash=True
)
# 20+ fps @ 2048 keypoints, highest accuracy
```

**エッジデバイス**:
```python
model = LightGlue(
    features='aliked',
    n_layers=5,            # Reduced layers
    depth_confidence=0.90,  # Aggressive early stopping
    width_confidence=0.95,  # Aggressive pruning
    flash=False            # CPU inference
)
# Real-time on mobile
```

---

## まとめ

LightGlueは以下の6つのイノベーションでSuperGlueを超越:

1. **Rotary PE**: 相対位置エンコーディングで位置情報を保持
2. **Bidirectional Cross-Attention**: 類似度計算を半減
3. **Double Softmax + Matchability**: Sinkhornを置き換え
4. **Deep Supervision**: 全層で教師信号、収束高速化
5. **Adaptive Depth**: 簡単なペアは早期終了
6. **Point Pruning**: 不要な点を除外

**性能**:
- 速度: SuperGlueの2.5倍（adaptive時）
- 精度: 同等以上
- 学習: 3倍高速化（2 GPU-days）

**用途**:
- リアルタイムSLAM（30+ fps）
- 大規模3D再構成
- Visual Localization
- 画像検索・照合

---

## 参考文献

- 論文: [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643)
- 公式コード: [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue)
- 関連研究:
  - SuperGlue (2020): 元祖Deep Matcher
  - LoFTR (2021): Dense Matcher
  - RoFormer (2021): Rotary Position Embedding
  - FlashAttention (2022): 効率的Attention

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装とは異なる場合があります。
