# MOTIP Understanding

**MOTIP (Multiple Object Tracking as ID Prediction)** の簡潔な擬似コード実装とドキュメント

論文: [Multiple Object Tracking as ID Prediction](https://arxiv.org/abs/2403.16848)
公式実装: https://github.com/MCG-NJU/MOTIP

---

## 目次

1. [概要](#概要)
2. [核心的アイデア](#核心的アイデア)
3. [アーキテクチャ](#アーキテクチャ)
4. [主要イノベーション](#主要イノベーション)
5. [処理フロー](#処理フロー)
6. [訓練](#訓練)
7. [推論](#推論)
8. [データセット](#データセット)
9. [実験結果](#実験結果)
10. [形状ガイド](#形状ガイド)
11. [FAQ](#faq)

---

## 概要

MOTIPは**Multiple Object Tracking (MOT) をIn-Context ID予測タスク**として定式化した、シンプルかつ効果的な手法です。

### 従来のMOT vs MOTIP

```
【従来のMOT】
1. 物体検出 (YOLO, DETR, etc.)
2. 特徴抽出 (ReID, etc.)
3. コスト行列計算 (IoU, 外観類似度, etc.)
4. ハンガリアンアルゴリズム (二部グラフマッチング)
→ ヒューリスティックな手法、複雑な調整が必要

【MOTIP】
1. 物体検出 (Deformable DETR)
2. ID Decoder (Transformer)
3. ID分類 (直接予測)
→ End-to-Endの学習可能な手法、シンプルで高性能
```

### 主要な特徴

- **In-Context ID予測**: 履歴軌跡のID情報を文脈として、現在フレームの検出にIDを予測
- **End-to-End学習**: DETR検出 + ID予測を統合した損失関数で訓練
- **シンプルなアーキテクチャ**: 標準的なTransformer Decoderのみで実装
- **State-of-the-Art性能**: DanceTrack (69.6 HOTA), SportsMOT (72.6 HOTA), BFT (70.5 HOTA)

---

## 核心的アイデア

### In-Context ID予測とは

MOTのIDラベルは「一貫性」を示すだけで、固定ラベルである必要はありません。

**例**: 4つの物体A, B, C, Dに対して

```
Ground Truth: ID = [1, 2, 3, 4]
別解:         ID = [8, 5, 7, 3]
```

両方とも正解です! (時間を通じて一貫していれば)

### この性質の活用

この性質により、MOTを**分類タスク**として定式化できます:

1. **ID辞書**: K個の学習可能なID埋め込み `{i^1, i^2, ..., i^K, i^spec}`
2. **In-Context Prompt**: 履歴軌跡に対応するID埋め込みを付与
3. **ID予測**: 現在フレームの検出に対して、どのIDに属するかを分類

```python
# 履歴軌跡トークン
τ^{m,km} = concat(f^m, i^km)
# f^m: 物体特徴 (256-dim)
# i^km: 対応するID埋め込み (256-dim)

# 現在フレームトークン
τ^n = concat(f^n, i^spec)
# i^spec: 新規物体用の特別トークン

# ID Decoder (Transformer Decoder)
output = IDDecoder(τ^n, memory=τ^{m,km})

# ID分類
id_logits = linear(output)  # (K+1,)
id = argmax(id_logits)
```

### 従来手法との違い

| 手法 | 訓練 | 推論 | 問題点 |
|------|------|------|--------|
| **ReID** | 分類損失 | コサイン類似度 | 訓練と推論の不一致 |
| **MOTIP** | 分類損失 | 分類予測 | なし (End-to-End) |

---

## アーキテクチャ

MOTIPは3つの主要コンポーネントで構成されます:

```
┌─────────────────────────────────────────────────────────┐
│                        MOTIP                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ DETR Detector │→ │ Trajectory   │→ │ ID Decoder  │ │
│  │ (Deformable)  │  │ Modeling     │  │ (6 Layers)  │ │
│  └───────────────┘  └──────────────┘  └─────────────┘ │
│         ↓                  ↓                  ↓        │
│  pred_boxes         trajectory         id_logits       │
│  pred_logits        _features          (K+1 classes)   │
│  output_embeddings  (256-dim)                          │
│  (256-dim)                                             │
└─────────────────────────────────────────────────────────┘
```

### 1. DETR Detector (Deformable DETR)

**役割**: 物体検出 + 物体レベル特徴抽出

```python
# 入力: 画像
images: (B, T, 3, H, W)

# 出力
pred_logits: (B, T, N, num_classes)  # クラス確率
pred_boxes: (B, T, N, 4)             # バウンディングボックス (cx, cy, w, h)
output_embeddings: (B, T, N, 256)    # 物体特徴 (軌跡表現として使用)
```

**構成**:
- ResNet-50 Backbone
- Deformable Transformer Encoder (4レベルのマルチスケール特徴)
- Deformable Transformer Decoder (300クエリ)
- 検出ヘッド (クラス分類 + ボックス回帰)

### 2. Trajectory Modeling

**役割**: DETR特徴を軌跡表現に変換 (軽量FFNアダプター)

```python
# 入力
output_embeddings: (B, T, N, 256)

# 処理: FFN → LayerNorm
trajectory_features = FFN(output_embeddings)
trajectory_features = LayerNorm(trajectory_features)

# 出力
trajectory_features: (B, T, N, 256)
```

### 3. ID Decoder

**役割**: In-Context ID予測 (6層Transformer Decoder)

```python
# 入力トークン構築
current_tokens = concat(
    trajectory_features,  # (B, T, N, 256)
    i^spec,               # (B, T, N, 256) - 新規物体用トークン
)  # → (B, T, N, 512)

trajectory_tokens = concat(
    historical_features,  # (G, M, 256)
    i^km,                 # (G, M, 256) - 対応するID埋め込み
)  # → (G, M, 512)

# ID Decoder (6層)
output_features = IDDecoder(
    current_tokens,      # Query
    trajectory_tokens,   # Key/Value (Memory)
)  # → (B, T, N, 256)

# ID分類
id_logits = linear(output_features)  # (B, T, N, K+1)
```

**レイヤー構成**:
- **Layer 1**: Cross-Attentionのみ
- **Layer 2-6**: Self-Attention + Cross-Attention + FFN

**Self-Attention** (Layer 2以降):
- 同一フレーム内の複数検出間で情報交換
- 類似物体の区別、グローバル最適解の発見

**Cross-Attention**:
- Query: 現在フレームの検出
- Key/Value: 履歴軌跡
- **Relative Position Encoding**で時間的文脈を考慮

---

## 主要イノベーション

### 1. In-Context ID予測

**問題**: 従来のReID手法は訓練 (分類) と推論 (コサイン類似度) が不一致

**解決**: MOTのIDラベルの性質を活用
- IDラベルは「一貫性」を示すだけ
- In-Context PromptとしてID埋め込みを使用
- 訓練と推論で同じタスク (分類) を実行

### 2. Relative Position Encoding

**目的**: 時間的な相対位置を考慮したアテンション

**実装**:
```python
# 各ヘッドごとに学習可能な相対位置バイアス
rel_pos_embeddings: (num_heads, rel_pe_length)
# rel_pe_length = 30 (最大30フレームの履歴)

# アテンションスコアにバイアスを追加
offset = t_current - t_trajectory
attn_score[t, m] += rel_pos_embeddings[head_idx, offset]

# 未来の軌跡はマスク (offset < 0)
if offset < 0:
    attn_score[t, m] = -inf
```

**効果**:
- 近い時刻の軌跡に高い重み
- 未来の情報漏洩を防ぐ (因果マスク)

### 3. Trajectory Augmentation

**Trajectory Occlusion** (prob=0.5):
```python
# 軌跡の一部をランダムにマスク
mask = torch.rand(G, T, N, 1) > 0.5
trajectory_tokens = trajectory_tokens * mask
```

**Trajectory Switching** (prob=0.5):
```python
# 同一フレーム内で2つの軌跡IDを入れ替え
# 推論時のID割り当てエラーをシミュレート
```

**効果**:
- オクルージョンへの頑健性向上
- ID割り当てエラーへの頑健性向上

### 4. ID辞書管理

**ID辞書**: K個の学習可能なID埋め込み

```python
id_embeddings = nn.Embedding(K + 1, id_dim)
# i^1, i^2, ..., i^K: 通常のID
# i^spec: 新規物体用の特別トークン
```

**推論時のID管理**:
- 利用可能なIDプール: `[1, 2, ..., K]`
- 軌跡が終了したらIDを再利用
- ID辞書サイズ K=50 (デフォルト)

---

## 処理フロー

### 訓練時のフロー

```
入力: T+1フレームのシーケンス
├─ 最初のTフレーム: 履歴軌跡
└─ 最後のフレーム: 現在フレーム (ID予測対象)

┌─────────────────────────────────────────┐
│ ステップ1: DETR検出 (全フレーム)       │
├─────────────────────────────────────────┤
│ images: (B, T+1, 3, H, W)               │
│ ↓ Deformable DETR                       │
│ output_embeddings: (B, T+1, N, 256)     │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ2: Trajectory Modeling          │
├─────────────────────────────────────────┤
│ trajectory_features: (B, T+1, N, 256)   │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ3: 履歴軌跡トークン構築        │
├─────────────────────────────────────────┤
│ historical_features: (B, T, N, 256)     │
│ + id_embeddings: (B, T, N, 256)         │
│ ↓ concat                                │
│ trajectory_tokens: (B, T, N, 512)       │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ4: 現在フレームトークン構築    │
├─────────────────────────────────────────┤
│ current_features: (B, 1, N, 256)        │
│ + i^spec: (B, 1, N, 256)                │
│ ↓ concat                                │
│ current_tokens: (B, 1, N, 512)          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ5: ID Decoder                   │
├─────────────────────────────────────────┤
│ IDDecoder(current_tokens,               │
│           trajectory_tokens)            │
│ ↓                                       │
│ id_logits: (B, 1, N, K+1)               │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ6: 損失計算                     │
├─────────────────────────────────────────┤
│ L = λ_cls*L_cls + λ_L1*L_L1 +           │
│     λ_giou*L_giou + λ_id*L_id           │
└─────────────────────────────────────────┘
```

### 推論時のフロー (RuntimeTracker)

```
各フレームごとに:

┌─────────────────────────────────────────┐
│ ステップ1: DETR検出                     │
├─────────────────────────────────────────┤
│ frame: (3, H, W)                        │
│ ↓ Deformable DETR                       │
│ detections: (N, 4), scores: (N,)        │
│ features: (N, 256)                      │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ2: 信頼度フィルタリング        │
├─────────────────────────────────────────┤
│ active_mask = scores > det_thresh       │
│ active_detections: (M, 4)               │
│ active_features: (M, 256)               │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ3: ID Decoder (履歴軌跡使用)   │
├─────────────────────────────────────────┤
│ 履歴軌跡から trajectory_tokens 構築    │
│ ↓ IDDecoder                             │
│ id_probs: (M, K+1)                      │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ4: ID割り当て                   │
├─────────────────────────────────────────┤
│ Assignment Protocol:                    │
│ - hungarian / id-max / object-max /     │
│   object-priority / id-priority         │
│ ↓                                       │
│ id_labels: (M,)                         │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ5: 軌跡更新                     │
├─────────────────────────────────────────┤
│ - 既存軌跡に追加                        │
│ - 新規軌跡作成 (score > newborn_thresh) │
│ - miss_count更新                        │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ ステップ6: 非アクティブ軌跡削除        │
├─────────────────────────────────────────┤
│ miss_count > miss_tolerance → 削除      │
└─────────────────────────────────────────┘
           ↓
出力: tracks = [{'id': 1, 'box': [...], 'score': 0.9}, ...]
```

---

## 訓練

### 訓練設定

**データセット**:
- DanceTrack: 10 epochs (学習率: epoch 5, 9で1/10)
- SportsMOT: 13 epochs (学習率: epoch 8, 12で1/10)
- BFT: 22 epochs (学習率: epoch 16, 20で1/10)

**シーケンスサンプリング**:
- サンプリング長: T+1 = 30フレーム (DanceTrack), 60フレーム (SportsMOT), 20フレーム (BFT)
- サンプリング間隔: 1~4フレーム (ランダム)
- 解像度: 短辺800, 長辺1440

**DETR訓練フレーム**:
- 全フレームでDETR forward
- 最初の4フレームのみ勾配計算
- 残りのフレームは `torch.no_grad()`
- メモリ効率化

**損失関数**:
```python
L = λ_cls * L_cls + λ_L1 * L_L1 + λ_giou * L_giou + λ_id * L_id
```

**ハイパーパラメータ**:
- λ_cls = 2.0, λ_L1 = 5.0, λ_giou = 2.0, λ_id = 1.0
- Optimizer: Adam (betas: 0.9, 0.999)
- Base LR: 1e-4 (Backbone: 0.1×, Linear projection: 0.05×)
- Batch size: 1 per GPU × 8 GPUs

**Data Augmentation**:
- Random resize, crop, color jitter
- Trajectory Occlusion (prob=0.5)
- Trajectory Switching (prob=0.5)

### 並列化訓練

MOTIPの大きな利点: **GPU並列化が容易**

```python
# 従来のTracking-by-Propagation (MOTR, MeMOTR, etc.)
# RNN的な逐次処理 → 並列化困難
for t in range(T):
    features_t = DETR(frame_t, track_queries)
    track_queries = update(track_queries, features_t)

# MOTIP: 検出と関連付けを分離 → 完全並列化
# すべてのフレームを並列に処理
features = DETR(frames)  # (B, T, N, C) - 並列処理
id_logits = IDDecoder(features, trajectory_tokens)  # Attention mask使用
```

---

## 推論

### RuntimeTracker

**初期化**:
```python
tracker = RuntimeTracker(
    model=motip_model,
    num_id_vocabulary=50,          # ID辞書サイズ
    max_trajectory_length=30,      # 軌跡の最大長
    assignment_protocol='object-max',  # ID割り当てプロトコル
    det_thresh=0.3,                # 検出信頼度閾値
    id_thresh=0.2,                 # ID確率閾値
    newborn_thresh=0.6,            # 新規物体の検出信頼度閾値
    miss_tolerance=30,             # 検出されないフレーム数の許容値
)
```

**フレームごとの更新**:
```python
for frame_idx, frame in enumerate(video):
    tracks = tracker.update(frame, frame_idx)
    # tracks: [{'id': 1, 'box': [x, y, w, h], 'score': 0.9}, ...]
```

### ID割り当てプロトコル

MOTIPは5種類のAssignment Protocolをサポート:

| Protocol | 説明 | 特徴 |
|----------|------|------|
| **object-max** | 各検出に対して最高確率のIDを割り当て | デフォルト、シンプルで高速 |
| **hungarian** | ハンガリアンアルゴリズムで最適割り当て | グローバル最適、やや複雑 |
| **id-max** | 各IDに対して最高確率の検出を割り当て | ID側から見た貪欲法 |
| **object-priority** | 検出信頼度が高い順に処理 | 高信頼度の検出を優先 |
| **id-priority** | ID確率が高い順に処理 | 確実なマッチを優先 |

**object-max (デフォルト)**:
```python
# 各検出の最大ID確率とラベル
max_probs, max_labels = torch.max(id_probs[:, :-1], dim=1)

# 閾値以下は新規物体
id_labels = torch.where(
    max_probs > id_thresh,
    max_labels,
    newborn_label,
)

# 重複排除: 同じIDが複数の検出に割り当てられた場合、
# 最高確率の検出のみ保持、他は新規物体
```

### ID辞書管理

**IDプール**:
```python
# 利用可能なIDラベル
available_ids = deque([1, 2, ..., K])

# 軌跡作成時にIDを取得
id_label = available_ids.popleft()

# 軌跡削除時にIDを返却
available_ids.append(id_label)

# IDプールが枯渇した場合: 最も古い軌跡のIDを再利用
```

**ID辞書容量**:
- K=50 (デフォルト)
- ほとんどの場合、利用率は40%以下
- 混雑シーンではK=200に増やす (MOT17など)

---

## データセット

### 対応データセット

| Dataset | 特徴 | シーケンス数 | HOTA (MOTIP) |
|---------|------|--------------|--------------|
| **DanceTrack** | 頻繁なオクルージョン、不規則な動き | 100 | **69.6** |
| **SportsMOT** | カメラ移動、高速移動 | 240 | **72.6** |
| **BFT** | 3次元空間での高機動性 | 106 | **70.5** |
| **MOT17** | 歩行者トラッキング | 7 (train) | 59.3 |

### データフォーマット

**フレームアノテーション**:
```python
{
    'id': tensor([id1, id2, ...]),              # (M,) - 物体ID
    'category': tensor([cat1, cat2, ...]),      # (M,) - クラスラベル
    'bbox': tensor([[x, y, w, h], ...]),        # (M, 4) - バウンディングボックス
    'visibility': tensor([0.5, 1.0, ...]),      # (M,) - 可視性スコア
    'is_legal': bool,                            # フレームの有効性フラグ
}
```

**軌跡アノテーション (訓練時)**:
```python
{
    'trajectory_id_labels': (G, T, N),          # 軌跡IDラベル
    'trajectory_id_masks': (G, T, N),           # パディングマスク
    'trajectory_times': (G, T, N),              # タイムスタンプ
    'unknown_id_labels': (G, 1, N),             # 現在フレームのID (教師信号)
    'unknown_id_masks': (G, 1, N),              # 現在フレームのマスク
}
```

詳細は[training_data.py](training_data.py)を参照してください。

---

## 実験結果

### DanceTrack Test Set

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|------|
| ByteTrack | 47.7 | 71.0 | 32.1 | 89.6 | 53.9 |
| OC-SORT | 55.1 | 80.3 | 38.3 | 92.0 | 54.6 |
| Hybrid-SORT | 62.2 | / | / | 91.6 | 63.0 |
| MeMOTR | 63.4 | 77.0 | 52.3 | 85.4 | 65.5 |
| CO-MOT | 65.3 | 80.1 | 53.5 | 89.3 | 66.5 |
| **MOTIP (ours)** | **69.6** | 80.4 | **60.4** | 90.6 | **74.7** |

### SportsMOT Test Set

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|------|
| ByteTrack | 62.1 | 76.5 | 50.5 | 93.4 | 69.1 |
| OC-SORT | 68.1 | 84.8 | 54.8 | 93.4 | 68.0 |
| MeMOTR | 68.8 | 82.0 | 57.8 | 90.2 | 69.9 |
| **MOTIP (ours)** | **72.6** | 83.5 | **63.2** | 92.4 | **77.1** |

### BFT Test Set

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|------|
| SORT | 61.2 | 60.6 | 62.3 | 75.5 | 77.2 |
| ByteTrack | 62.5 | 61.2 | 64.1 | 77.2 | 82.3 |
| OC-SORT | 66.8 | 65.4 | 68.7 | 77.1 | 79.3 |
| **MOTIP (ours)** | **70.5** | 69.6 | **71.8** | 77.1 | **82.1** |

### 主要な観察

1. **複雑なシーンで大幅な改善**:
   - DanceTrack: +4.3 HOTA (vs CO-MOT)
   - SportsMOT: +3.8 HOTA (vs MeMOTR)

2. **関連付け精度 (AssA) で大幅なリード**:
   - DanceTrack: 60.4 AssA (+6.9 vs CO-MOT)
   - SportsMOT: 63.2 AssA (+5.4 vs MeMOTR)

3. **シンプルな設計で高性能**:
   - 標準的なTransformer Decoderのみ
   - 特別なアーキテクチャ不要

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | images | `(B, T, 3, H, W)` | RGB画像シーケンス |
| **DETR検出** | pred_logits | `(B, T, N, num_classes)` | クラス確率 |
| | pred_boxes | `(B, T, N, 4)` | バウンディングボックス (cx, cy, w, h) |
| | output_embeddings | `(B, T, N, 256)` | DETR出力埋め込み |
| **Trajectory Modeling** | trajectory_features | `(B, T, N, 256)` | 軌跡特徴 |
| **ID Decoder入力** | current_tokens | `(B, T, N, 512)` | 現在フレームトークン (256+256) |
| | trajectory_tokens | `(G, M, 512)` | 履歴軌跡トークン |
| | trajectory_times | `(G, M)` | タイムスタンプ |
| **ID Decoder出力** | output_features | `(B, T, N, 256)` | デコード済み特徴 |
| | id_logits | `(B, T, N, K+1)` | ID予測ロジット |

### 軸の意味

- **B**: バッチサイズ
- **T**: フレーム数 (訓練: 30, 推論: 1)
- **N**: DETRクエリ数 (300)
- **K**: ID辞書サイズ (50)
- **G**: 軌跡グループ数 (可変)
- **M**: 各グループの軌跡長 (可変、最大30)

### データ型

- 画像: `torch.float32`, 正規化済み `[0, 1]`
- バウンディングボックス: `torch.float32`, 正規化座標 `[0, 1]`
- クラスラベル: `torch.long`, `0 ~ num_classes-1`
- IDラベル: `torch.long`, `1 ~ K` (0はパディング、K+1は新規物体)

---

## FAQ

### Q1: MOTIPとReID手法の違いは?

**A**: 訓練と推論の一貫性

| 手法 | 訓練 | 推論 | 問題点 |
|------|------|------|--------|
| **ReID** | 分類損失 (ID分類) | コサイン類似度 | 訓練と推論が不一致 |
| **MOTIP** | 分類損失 (In-Context ID予測) | 分類予測 (argmax) | なし (End-to-End) |

ReID手法は訓練時に分類タスクとして学習するが、推論時はコサイン類似度で関連付けを行います。これは訓練と推論の不一致を生み、性能低下の原因となります。

MOTIPはIn-Context ID予測により、訓練と推論で同じタスク (分類) を実行します。

### Q2: MOTIPとTracking-by-Propagation (MOTR, MeMOTR) の違いは?

**A**: 検出と関連付けの分離

| 手法 | アプローチ | 問題点 |
|------|-----------|--------|
| **Tracking-by-Propagation** | Track QueryをRNN的に伝播 | 検出と関連付けの競合、並列化困難 |
| **MOTIP** | 検出 → ID予測 (分離) | 競合なし、並列化容易 |

Tracking-by-Propagation手法 (MOTR, MeMOTR, etc.) は、Track Queryを時間方向に伝播させることで関連付けを行います。しかし、これは以下の問題を引き起こします:

1. **検出と関連付けの競合**: 同じクエリで検出と追跡を行うため、どちらかの性能が低下
2. **並列化困難**: RNN的な逐次処理が必要で、GPU並列化が困難

MOTIPは検出 (DETR) と関連付け (ID Decoder) を分離し、これらの問題を解決しています。

### Q3: ID辞書のサイズ K はどう決める?

**A**: フレーム内の最大物体数以上に設定

```python
# 推奨設定
DanceTrack: K = 50  (フレーム内最大物体数 ~20)
SportsMOT: K = 50   (フレーム内最大物体数 ~15)
BFT: K = 50         (フレーム内最大物体数 ~10)
MOT17: K = 200      (混雑シーン、最大 ~50)
```

一般的に、利用率は40%以下です。K=50で十分なケースがほとんどです。

### Q4: Relative Position Encodingの役割は?

**A**: 時間的文脈を考慮したアテンション

```python
# 各ヘッドごとに学習可能な相対位置バイアス
rel_pos_embeddings: (num_heads, rel_pe_length)

# アテンションスコアにバイアスを追加
offset = t_current - t_trajectory
attn_score[t, m] += rel_pos_embeddings[head_idx, offset]

# 未来の軌跡はマスク
if offset < 0:
    attn_score[t, m] = -inf
```

**効果**:
1. 近い時刻の軌跡に高い重み
2. 未来の情報漏洩を防ぐ (因果マスク)
3. 時間的な相対関係を学習

### Q5: Trajectory Augmentationは必須?

**A**: はい、性能向上に大きく寄与

| Augmentation | HOTA | AssA | IDF1 |
|--------------|------|------|------|
| なし | 59.5 | 47.2 | 61.1 |
| Occlusion のみ | 60.7 | 49.4 | 62.7 |
| **Occlusion + Switching** | **62.2** | **51.5** | **64.8** |

Trajectory Augmentationにより:
- オクルージョンへの頑健性向上
- ID割り当てエラーへの頑健性向上
- +2.7 HOTA, +4.3 AssA の改善

### Q6: 推論速度は?

**A**: ほぼリアルタイム (FP16で22.8 FPS)

| Precision | FPS | GPU |
|-----------|-----|-----|
| FP32 | 12.7 | NVIDIA RTX A5000 |
| FP16 | 22.8 | NVIDIA RTX A5000 |

MOTRと同等の速度で、より高い精度を実現しています。

### Q7: Hungarian Algorithmは使うべき?

**A**: デフォルトの object-max で十分

| Protocol | HOTA | AssA | IDF1 | 特徴 |
|----------|------|------|------|------|
| object-max (default) | **62.2** | **51.5** | **64.8** | シンプル、高速 |
| hungarian | 62.2 | 51.8 | 65.6 | やや複雑 |

MOTIPのID Decoderは、self-attentionにより自動的にグローバル最適解を見つける能力を持っています。そのため、Hungarian Algorithmを使用しても大きな改善は見られません。

### Q8: CrowdHumanとの混在訓練は推奨される?

**A**: 慎重に (論文Appendix C参照)

**問題点**:
1. **アノテーション基準の不一致**:
   - CrowdHuman: すべての人物
   - SportsMOT: アスリートのみ

2. **過度に単純化された動画生成**:
   - MultiSimulate: 平行移動 + スケールのみ
   - 実際の動画とは大きく異なる

3. **長期モデリングへの悪影響**:
   - 単純な動きのパターンを学習
   - 複雑な動きに対応できない

**推奨**:
- 訓練データが十分な場合 (DanceTrack, SportsMOT, BFT): 混在訓練は不要
- 訓練データが不足する場合 (MOT17): 慎重に混在訓練

### Q9: Self-Attentionは必要?

**A**: はい、グローバル最適解の発見に重要

| Self-Attention | HOTA | AssA | IDF1 |
|----------------|------|------|------|
| なし (Layer 1のみ) | 59.5 | 47.2 | 61.1 |
| Layer 2以降 | **62.2** | **51.5** | **64.8** |

Self-Attentionにより:
- 同一フレーム内の複数検出間で情報交換
- 類似物体の区別
- グローバル最適解の発見

### Q10: IDラベルのシャッフルは訓練時に必要?

**A**: はい、汎化性能向上に重要

MOTIPはIDラベルが「一貫性」を示すだけという性質を活用しています。訓練時にIDラベルをシャッフルすることで:

1. 固定ラベルへの過学習を防ぐ
2. In-Context ID予測の汎化性能向上
3. 推論時の未知IDへの対応力向上

実装では、各エポックでランダムにIDラベルを割り当てることで実現しています。

---

## ファイル構成

```
MOTIP_understanding/
├── README.md                   # 本ファイル
├── main_flow.py                # メインフロー (MOTIP全体)
├── id_decoder.py               # ID Decoder (核心的イノベーション)
├── loss_computation.py         # 損失関数 (DETR + ID損失)
├── runtime_tracker.py          # 推論時のトラッキング管理
└── training_data.py            # データセットフォーマット
```

各ファイルの詳細は対応するファイルを参照してください。

---

## 参考文献

- 論文: [Multiple Object Tracking as ID Prediction](https://arxiv.org/abs/2403.16848)
- 公式実装: https://github.com/MCG-NJU/MOTIP
- Deformable DETR: https://arxiv.org/abs/2010.04159
- DanceTrack: https://arxiv.org/abs/2111.14690
- SportsMOT: https://arxiv.org/abs/2304.05170

---

## まとめ

MOTIPは**In-Context ID予測**という新しい視点でMOTを定式化し、以下を実現しました:

1. **シンプルな設計**: 標準的なTransformer Decoderのみ
2. **End-to-End学習**: 訓練と推論の一貫性
3. **State-of-the-Art性能**: 複数のベンチマークで最高性能
4. **並列化容易**: GPU効率的な訓練

今後の発展の余地も大きく、有望なベースライン手法として期待されます。
