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

**定数定義**:
```python
N_QUERIES = 300     # DETRクエリ数 (N)
FEATURE_DIM = 256   # 特徴次元
K = 148             # ID語彙サイズ (DanceTrackの場合)
T = 5               # 履歴フレーム数 (例)
```

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

### Ground Truth ID埋め込みの扱い（重要）

訓練時、ID Decoderが正しい時系列関連付けを学習するため、**履歴フレームにはGround TruthのID埋め込みを使用**します。

#### 前提知識: ID語彙サイズKの定義

**K = データセット全体でのユニークID数**

- DanceTrack: K=148 (データセット全体で148個のユニークID)
- SportsMOT: K=90 (データセット全体で90個のユニークID)
- MOT17: K=500 (データセット全体で500個のユニークID)

**重要**: Kはフレームごとの最大物体数ではなく、データセット全体でのID総数です。

#### ステップ1: ID埋め込み層の実装（重要）

**実装: nn.Linear + One-hot エンコーディング（nn.Embeddingではない！）**

```python
# 例: DanceTrackの場合
K = 148  # データセット全体のユニークID数

# ID埋め込み層: One-hot (149次元) → 埋め込みベクトル (256次元)
self.word_to_embed = nn.Linear(K + 1, 256, bias=False)
# 重み形状: (256, K+1) = (256, 149)
# 入力: One-hotベクトル (149次元)
# 出力: 埋め込みベクトル (256次元)

# スロット構成:
# - スロット 0~147: 148個のID用スロット
# - スロット 148: 新規物体トークン

# ID埋め込みの取得
def id_label_to_embed(id_labels):
    """
    id_labels: (N,) - スロット番号 [0~147: ID, 148: 新規物体]
    """
    one_hot = torch.eye(K + 1)[id_labels]  # (N, 149)
    id_embeds = self.word_to_embed(one_hot)  # (N, 256)
    return id_embeds

# 各スロットの埋め込みベクトル:
# スロット0の埋め込み = self.word_to_embed.weight[:, 0]  # (256,)
# スロット1の埋め込み = self.word_to_embed.weight[:, 1]  # (256,)
# ...
# スロット148の埋め込み = self.word_to_embed.weight[:, 148]  # (256,)
```

**なぜnn.Linearを使うのか？**

```python
# nn.Embedding の場合:
embedding = nn.Embedding(K+1, 256)
embed = embedding(id_label)  # id_label: スカラーインデックス

# nn.Linear + One-hot の場合 (MOTIP):
linear = nn.Linear(K+1, 256, bias=False)
one_hot = torch.eye(K+1)[id_label]  # One-hot化
embed = linear(one_hot)  # Linear変換

# 数学的に等価だが、実装の柔軟性が異なる
# MOTIPはOne-hot表現を明示的に使用する設計
```

#### ID埋め込みの本質（重要）

**ID埋め込み = インスタンスの外観特徴とは無関係**

1. **固定の学習済みベクトル**
   - Linear層の重み `word_to_embed.weight[:, i]` として保存
   - 訓練中に学習されるが、推論時は固定
   - 各スロット (0~K) は独立した256次元ベクトル

2. **スロットの意味**
   - 訓練時: GT ID → スロット番号の固定マッピング
   - 推論時: 動的に再利用されるスロット
   - 埋め込みベクトル自体は変わらない

3. **学習されるもの**
   - Linear層の重み (256, K+1)
   - 各スロットが「どのような文脈で使われるべきか」
   - ID Decoderが「スロットの使い方」を学習

4. **インスタンス外観とは独立**
   - ID埋め込みはインスタンスの見た目とは無関係
   - DETR特徴 (256次元) がインスタンス外観を表現
   - ID埋め込み (256次元) はラベル/スロット情報を表現

#### GT ID → スロット番号のマッピング

訓練時、各GT IDに固定のスロット番号を割り当てます:

```python
# データセット全体のユニークID: [1, 5, 7, 12, 18, ..., 148] (148個)
# スロット番号: [0, 1, 2, 3, 4, ..., 147]

id_mapping = {
    1: 0,      # GT ID 1 → スロット 0
    5: 1,      # GT ID 5 → スロット 1
    7: 2,      # GT ID 7 → スロット 2
    12: 3,     # GT ID 12 → スロット 3
    18: 4,     # GT ID 18 → スロット 4
    # ...
    148: 147,  # GT ID 148 → スロット 147
}

# このマッピングは訓練中ずっと固定
# 同じGT IDは常に同じスロット番号を使用
```

#### ステップ2: Hungarian Matchingで予測とGTを対応付け

各履歴フレームで、DETR予測（300個のクエリ）とGround Truthをマッチング:

```python
# 例: フレーム0の場合
gt_boxes = [[100, 200, 50, 80],   # GT 0: ID 1
            [300, 150, 60, 90],   # GT 1: ID 5
            [450, 300, 70, 100]]  # GT 2: ID 7
gt_ids = [1, 5, 7]

pred_boxes = model(frame_0)  # (N_QUERIES, 4) = (300, 4) - 300個のクエリの予測

# Hungarian Matching: コスト = L1距離 + GIoU距離
cost_matrix = compute_cost(pred_boxes, gt_boxes)  # (N_QUERIES, N_GT) = (300, 3)
matched_pred_indices, matched_gt_indices = hungarian_algorithm(cost_matrix)

# マッチング結果例:
# クエリ15 ← GT 0 (ID 1)
# クエリ42 ← GT 1 (ID 5)
# クエリ7  ← GT 2 (ID 7)
# クエリ0-14, 16-41, 43-299 ← マッチなし（背景）
```

#### ステップ3: GT ID埋め込みの取得

マッチングした予測にGT IDを割り当て:

```python
# 履歴フレーム（T-1個）の各クエリにGT IDを割り当て
historical_ids = torch.full((T-1, N_QUERIES), K)  # (T-1, 300), デフォルト: 新規物体トークン(K=148)

for t in range(T-1):  # 履歴フレームのみ
    for pred_idx, gt_idx in zip(matched_pred_indices[t], matched_gt_indices[t]):
        gt_id_original = gt_ids[t][gt_idx]  # 例: 1, 5, 7
        gt_id_vocab = id_mapping[gt_id_original]  # 語彙インデックス: 0, 1, 2
        historical_ids[t, pred_idx] = gt_id_vocab

# GT ID埋め込みを取得
gt_id_embeddings = self.id_embeddings(historical_ids)  # (T-1, N_QUERIES, FEATURE_DIM) = (T-1, 300, 256)

# 履歴軌跡トークン: DETR特徴 + GT ID埋め込み
historical_tokens = torch.cat([
    historical_features,  # (T-1, N_QUERIES, FEATURE_DIM) = (T-1, 300, 256)
    gt_id_embeddings,     # (T-1, N_QUERIES, FEATURE_DIM) = (T-1, 300, 256)
], dim=-1)  # (T-1, N_QUERIES, FEATURE_DIM*2) = (T-1, 300, 512)
```

#### ステップ4: 現在フレームのID埋め込み（重要）

現在フレームのID埋め込みは、**訓練時**と**推論時**で異なります。

**訓練時: 履歴に存在するかで条件分岐**

```python
# 現在フレームのID埋め込みを準備
current_ids = torch.full((N_QUERIES,), K)  # (300,), デフォルト: 新規物体トークン(K=148)

# Hungarian Matchingで現在フレームの予測とGTを対応付け
matched_pred_indices, matched_gt_indices = hungarian_algorithm(...)

# 履歴フレーム(0~T-1)に出現したIDを収集
historical_gt_ids = set()
for t in range(T-1):
    historical_gt_ids.update(gt_ids[t])
# 例: {1, 5, 7} - 履歴フレームに登場したGT ID

# 現在フレームの各マッチング
for pred_idx, gt_idx in zip(matched_pred_indices, matched_gt_indices):
    gt_id = gt_ids_current[gt_idx]  # 例: 1, 5, 18

    if gt_id in historical_gt_ids:
        # 履歴に存在するID → 対応するスロット番号を使用
        current_ids[pred_idx] = id_mapping[gt_id]  # 例: ID 1→スロット0, ID 5→スロット1
    else:
        # シーケンス内で初出現のID → 新規物体トークンを使用
        current_ids[pred_idx] = K  # ID 18は履歴にないので新規物体トークン(K=148)

# One-hot化してLinear変換
current_one_hot = torch.eye(K + 1)[current_ids]  # (N_QUERIES, K+1) = (300, 149)
current_embeddings = self.word_to_embed(current_one_hot)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# 現在フレームトークン: DETR特徴 + ID埋め込み
current_tokens = torch.cat([
    current_features,      # (N_QUERIES, FEATURE_DIM) = (300, 256) - インスタンス外観特徴
    current_embeddings,    # (N_QUERIES, FEATURE_DIM) = (300, 256) - ID埋め込み (スロット情報)
], dim=-1)  # (N_QUERIES, FEATURE_DIM*2) = (300, 512)
```

**推論時: すべて新規物体トークン**

```python
# 推論時は正解IDが分からないため、全て新規物体トークン
current_ids = torch.full((N_QUERIES,), K)  # (300,), K = num_id_vocabulary (例: 148)
current_one_hot = torch.eye(K + 1)[current_ids]  # (N_QUERIES, K+1) = (300, 149)
current_embeddings = self.word_to_embed(current_one_hot)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# 履歴軌跡のID埋め込みは、過去の予測結果から取得
# (後述の「推論時のスロット再利用」参照)
```

#### ステップ5: ID Decoder & 損失計算

```python
# ID Decoder: 履歴のGT ID埋め込みを文脈として、現在フレームのIDを予測
id_output = id_decoder(
    query=current_tokens,      # (N_QUERIES, FEATURE_DIM*2) = (300, 512) - 現在フレーム（新規トークン付き）
    memory=historical_tokens,  # (T-1*N_QUERIES, FEATURE_DIM*2) = (T-1*300, 512) - 履歴（GT ID付き）
)

# ID予測
id_logits = id_head(id_output)  # (N_QUERIES, K+1) = (300, 149) - 148個のID + 1個の新規トークン

# 現在フレームでもHungarian Matchingして正解ラベルを作成
gt_boxes_current = [[110, 210, 50, 80],  # GT 0: ID 1（継続）
                    [310, 160, 60, 90],  # GT 1: ID 5（継続）
                    [500, 350, 65, 95]]  # GT 2: ID 18（新規出現）
gt_ids_current = [1, 5, 18]

matched_pred_indices, matched_gt_indices = hungarian_algorithm(...)

# ID損失の正解ラベルを作成
id_targets = torch.full((N_QUERIES,), K)  # (300,), デフォルト: 新規トークン(K=148)
for pred_idx, gt_idx in zip(matched_pred_indices, matched_gt_indices):
    gt_id_original = gt_ids_current[gt_idx]
    gt_id_vocab = id_mapping[gt_id_original]  # 1→0, 5→1, 18→17
    id_targets[pred_idx] = gt_id_vocab

# ID損失
loss_id = cross_entropy(id_logits, id_targets)
```

#### 訓練時の具体例: 新規IDの扱い

シーケンス内で初出現するIDの処理を、具体例で説明します。

**シナリオ**:
- フレーム0~4 (履歴): GT ID 1, 5, 7 が存在
- フレーム5 (現在): GT ID 1, 5, 18 が存在 (ID 18は初出現)

```python
# ========================================
# 1. GT ID → スロット番号マッピング (固定)
# ========================================
id_mapping = {
    1: 0,    # GT ID 1 → スロット0
    5: 1,    # GT ID 5 → スロット1
    7: 2,    # GT ID 7 → スロット2
    18: 17,  # GT ID 18 → スロット17
    # ...
}

# ========================================
# 2. 履歴フレーム (0~4) のID埋め込み
# ========================================
# フレームごとにGT IDを収集: {1, 5, 7}
# Hungarian Matchingで予測とGTを対応付け
# マッチした予測にスロット番号を割り当て

historical_id_labels = [0, 1, 2, K, K, ...]  # (N_QUERIES,) = (300,) - 各DETRクエリのスロット
# 例: クエリ15→スロット0(ID 1), クエリ42→スロット1(ID 5), クエリ7→スロット2(ID 7)
#     残りのクエリ→スロットK=148(背景)

historical_one_hot = torch.eye(K+1)[historical_id_labels]  # (N_QUERIES, K+1) = (300, 149)
historical_id_embeds = self.word_to_embed(historical_one_hot)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# ========================================
# 3. 現在フレーム (5) のID埋め込み
# ========================================
# 履歴に存在したID: {1, 5, 7}
# 現在フレームのGT ID: {1, 5, 18}
# → ID 18は履歴に無い (初出現)

current_ids = torch.full((N_QUERIES,), K)  # (300,), デフォルト: 新規物体トークン(K=148)

# Hungarian Matchingで現在フレームの予測とGTを対応付け
for pred_idx, gt_idx in zip(matched_pred_indices, matched_gt_indices):
    gt_id = gt_ids_current[gt_idx]

    if gt_id == 1:
        current_ids[pred_idx] = 0   # ID 1→スロット0 (履歴に存在)
    elif gt_id == 5:
        current_ids[pred_idx] = 1   # ID 5→スロット1 (履歴に存在)
    elif gt_id == 18:
        current_ids[pred_idx] = K   # ID 18→新規物体トークン (履歴に無い!), K=148

current_one_hot = torch.eye(K+1)[current_ids]  # (N_QUERIES, K+1) = (300, 149)
current_id_embeds = self.word_to_embed(current_one_hot)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# ========================================
# 4. ID Decoderへの入力
# ========================================
# 履歴トークン: [DETR特徴, スロット0/1/2の埋め込み]
# 現在トークン: [DETR特徴, スロット0/1/新規トークンの埋め込み]

historical_tokens = torch.cat([
    historical_detr_features,  # (T*N_QUERIES, FEATURE_DIM) = (T*300, 256)
    historical_id_embeds,      # (T*N_QUERIES, FEATURE_DIM) = (T*300, 256) - スロット0,1,2,K,...
], dim=-1)  # (T*N_QUERIES, FEATURE_DIM*2) = (T*300, 512)

current_tokens = torch.cat([
    current_detr_features,  # (N_QUERIES, FEATURE_DIM) = (300, 256)
    current_id_embeds,      # (N_QUERIES, FEATURE_DIM) = (300, 256) - スロット0,1,K,...
], dim=-1)  # (N_QUERIES, FEATURE_DIM*2) = (300, 512)

# ========================================
# 5. ID Decoder & 損失計算
# ========================================
id_output = id_decoder(current_tokens, historical_tokens)  # (N_QUERIES, FEATURE_DIM) = (300, 256)
id_logits = id_head(id_output)  # (N_QUERIES, K+1) = (300, 149) - 各スロット+新規トークンのスコア

# 正解ラベル作成
id_targets = torch.full((N_QUERIES,), K)  # (300,), デフォルト: 新規トークン(K=148)
for pred_idx, gt_idx in zip(matched_pred_indices, matched_gt_indices):
    gt_id = gt_ids_current[gt_idx]
    id_targets[pred_idx] = id_mapping[gt_id]
    # ID 1 → スロット0
    # ID 5 → スロット1
    # ID 18 → スロット17 ← 重要! 履歴に無いが、正解ラベルはスロット17

# 損失計算
loss_id = cross_entropy(id_logits, id_targets)

# ========================================
# 6. 学習されるもの
# ========================================
# ID Decoderは次を学習:
#
# 入力文脈:
#   - 履歴: スロット0, 1, 2の埋め込み
#   - 現在: 新規物体トークンの埋め込み
#
# 予測タスク:
#   - ID 1, 5 (継続): スロット0, 1を予測
#   - ID 18 (新規): スロット17を予測
#
# 学習内容:
#   - 「履歴に同じスロットがあれば継続と判断」→ そのスロットを予測
#   - 「履歴に無いスロットを予測すべきと判断」→ 別のスロット(17)を予測
#   - 新規物体トークンは「どのスロットを予測すべきか」のヒント
```

#### 推論時の具体例: スロットの再利用

推論時は、ID Vocabularyを**動的に再利用可能なスロット**として管理します。

**重要ポイント**:
- Linear層の重み (`word_to_embed.weight`) は訓練時のまま固定
- スロット番号の意味が動的に変わる (訓練時: GT ID固定, 推論時: 動的割り当て)
- 同じスロット埋め込みベクトルを異なる物体に再利用

```python
# ========================================
# 1. 初期化
# ========================================
K = 148  # 訓練時と同じID語彙サイズ
id_queue = deque([0, 1, 2, ..., 147])  # 利用可能なスロット (148個)
id_label_to_id = {}  # スロット番号 → 実際のトラックID
active_trajectories = {}  # トラックID → 軌跡情報
next_track_id = 0  # 次に割り当てるトラックID

# ========================================
# 2. フレームt=0の処理
# ========================================
# DETR検出
detections = detr(frame_0)  # (N_QUERIES, 4) = (300, 4)
features = trajectory_modeling(detections)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# 現在フレームトークン: 全て新規物体トークン
current_ids = torch.full((N_QUERIES,), K)  # (300,), K=148 (新規物体トークン)
current_one_hot = torch.eye(K + 1)[current_ids]  # (N_QUERIES, K+1) = (300, 149)
current_embeds = word_to_embed(current_one_hot)  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# ID Decoder (履歴が無いので新規物体として検出)
id_logits = id_decoder(current_embeds, memory=None)  # (N_QUERIES, K+1) = (300, 149)
id_pred = argmax(id_logits, dim=-1)  # (N_QUERIES,) = (300,)

# 新規物体の割り当て
for det_idx in range(N_QUERIES):
    if score[det_idx] > newborn_thresh and id_pred[det_idx] == K:
        # 未使用スロットを取得
        slot = id_queue.popleft()  # 例: slot=0
        track_id = next_track_id  # track_id=0
        next_track_id += 1

        id_label_to_id[slot] = track_id  # {0: 0}
        active_trajectories[track_id] = {
            'slot': slot,  # スロット0
            'boxes': [boxes[det_idx]],
            'features': [features[det_idx]],
            'times': [0],
        }

# 結果: track_id=0がスロット0に割り当て

# ========================================
# 3. フレームt=1の処理
# ========================================
detections = detr(frame_1)
features = trajectory_modeling(detections)

# 履歴軌跡トークンを構築
historical_tokens = []
for track_id, traj in active_trajectories.items():
    slot = traj['slot']  # 例: 0
    slot_one_hot = torch.eye(K + 1)[slot]  # スロット0のOne-hot
    slot_embed = word_to_embed(slot_one_hot)  # スロット0の埋め込み

    traj_token = torch.cat([traj['features'][-1], slot_embed], dim=-1)
    historical_tokens.append(traj_token)

# 現在フレームトークン: 全て新規物体トークン
current_ids = torch.full((N_QUERIES,), K)  # (300,)
current_embeds = word_to_embed(torch.eye(K + 1)[current_ids])  # (N_QUERIES, FEATURE_DIM) = (300, 256)

# ID Decoder
id_logits = id_decoder(current_embeds, memory=historical_tokens)  # (N_QUERIES, K+1) = (300, 149)
id_pred = argmax(id_logits, dim=-1)  # (N_QUERIES,) = (300,)

# ID割り当て
for det_idx in range(N_QUERIES):
    if id_pred[det_idx] == 0:  # スロット0を予測 (継続)
        track_id = id_label_to_id[0]  # track_id=0
        active_trajectories[track_id]['boxes'].append(boxes[det_idx])
        active_trajectories[track_id]['features'].append(features[det_idx])
        active_trajectories[track_id]['times'].append(1)
    elif id_pred[det_idx] == 148:  # 新規物体トークンを予測 (新規)
        slot = id_queue.popleft()  # 例: slot=1
        track_id = next_track_id  # track_id=1
        next_track_id += 1

        id_label_to_id[slot] = track_id  # {0: 0, 1: 1}
        active_trajectories[track_id] = {...}

# ========================================
# 4. トラック消失時のスロット解放
# ========================================
# miss_countが閾値を超えたトラックを削除
for track_id in list(active_trajectories.keys()):
    if active_trajectories[track_id]['miss_count'] > miss_tolerance:
        slot = active_trajectories[track_id]['slot']
        del active_trajectories[track_id]
        del id_label_to_id[slot]
        id_queue.append(slot)  # スロットを再利用可能に
        # 例: スロット0が解放 → 将来的に別の物体に割り当て可能

# ========================================
# 5. 重要ポイント
# ========================================
# - Linear層の重み (word_to_embed.weight) は訓練時のまま固定
# - スロット0の埋め込みベクトルは常に word_to_embed.weight[:, 0]
# - 訓練時: スロット0 = GT ID 1 (固定)
# - 推論時: スロット0 = track_id 0, その後解放されて別の物体に再利用
# - ID Decoderは「スロットの使い方」を学習しているため、
#   異なる物体でも同じスロットを適切に使い分けられる
```

#### 推論時の詳細: どうやってトラッキングするのか？

推論時の目的は、**現在フレームの各検出が、前フレームのどのトラックに紐づくか、または新規出現か**を判定することです。

**推論時の処理フロー (フレームt) - 実際のコードに基づく**:

```python
# ========================================
# 定数定義
# ========================================
N_QUERIES = 300  # DETRクエリ数
FEATURE_DIM = 256  # 特徴次元
K = 148  # ID語彙サイズ (DanceTrackの場合)
MAX_HISTORY_LEN = 30  # 履歴の最大長 (miss_tolerance)

# ========================================
# 前提: RuntimeTrackerが管理しているデータ
# ========================================
# trajectory_features: (T_history, N_active_tracks, 256)
#   - T_history: 過去のフレーム数 (最大30)
#   - N_active_tracks: アクティブなトラック数 (可変)
#   - 各トラックの過去T個のフレームのDETR特徴
#
# trajectory_id_labels: (T_history, N_active_tracks)
#   - 各トラックが使用しているスロット番号
#
# trajectory_boxes: (T_history, N_active_tracks, 4)
# trajectory_times: (T_history, N_active_tracks)
# trajectory_masks: (T_history, N_active_tracks)

# 例: 3つのアクティブトラックがある場合
# trajectory_features.shape = (5, 3, 256)  # 過去5フレーム、3トラック
# trajectory_id_labels[0] = [0, 1, 5]  # 最新フレームでのスロット番号

# ========================================
# ステップ1: 現在フレーム (t) の検出
# ========================================
# DETR forward
frame_t_tensor = preprocess(frame_t)  # (1, 3, H, W)
detr_out = model.detr(frame_t_tensor)

# DETR出力
pred_logits = detr_out['pred_logits'][0]  # (N_QUERIES, num_classes)
pred_boxes = detr_out['pred_boxes'][0]    # (N_QUERIES, 4)
detr_features = detr_out['outputs'][0]     # (N_QUERIES, FEATURE_DIM)
#                                          # = (300, 256)

# 信頼度フィルタリング
scores = pred_logits.sigmoid().max(dim=-1)[0]  # (N_QUERIES,)
active_mask = scores > det_thresh  # 例: det_thresh=0.3

# アクティブな検出のみ抽出
active_boxes = pred_boxes[active_mask]      # (N_active, 4)
active_features = detr_features[active_mask]  # (N_active, 256)
active_scores = scores[active_mask]         # (N_active,)

# ========================================
# ステップ2: Trajectory Modeling
# ========================================
# DETR特徴をID Decoder用に変換
active_features = trajectory_modeling(active_features)  # (N_active, 256)

# ========================================
# ステップ3: seq_info構築（実際の実装）
# ========================================
# 履歴が無い場合（最初のフレーム）
if trajectory_features.shape[0] == 0:
    # 全て新規物体として扱う
    id_pred_labels = torch.full((N_active,), K, dtype=torch.int64)
else:
    # 履歴軌跡情報を準備
    seq_info = {
        # 履歴軌跡 (過去T個のフレーム、N_active_tracksトラック)
        "trajectory_features": trajectory_features[None, None, ...],
        # shape: (1, 1, T_history, N_active_tracks, 256)

        "trajectory_id_labels": trajectory_id_labels[None, None, ...],
        # shape: (1, 1, T_history, N_active_tracks)
        # 各トラックが使用しているスロット番号

        "trajectory_boxes": trajectory_boxes[None, None, ...],
        "trajectory_times": trajectory_times[None, None, ...],
        "trajectory_masks": trajectory_masks[None, None, ...],

        # 現在フレームの検出 (まだIDが不明)
        "unknown_features": active_features[None, None, None, ...],
        # shape: (1, 1, 1, N_active, 256)

        "unknown_boxes": active_boxes[None, None, None, ...],
        "unknown_masks": torch.zeros((1, 1, 1, N_active), dtype=torch.bool),
        "unknown_times": torch.full((1, 1, 1, N_active), T_history, dtype=torch.int64),
    }

    # Trajectory Modeling (ID埋め込み付与)
    seq_info = model.trajectory_modeling(seq_info)

    # ID Decoder forward
    # 内部で以下の処理が行われる:
    # 1. 履歴軌跡トークン構築:
    #    trajectory_tokens = concat([trajectory_features,
    #                                id_label_to_embed(trajectory_id_labels)], dim=-1)
    #    shape: (T_history * N_active_tracks, 512)
    #
    # 2. 現在フレームトークン構築:
    #    current_tokens = concat([unknown_features,
    #                            新規物体トークン埋め込み], dim=-1)
    #    shape: (N_active, 512)
    #
    # 3. Transformer Decoder:
    #    Query: current_tokens
    #    Memory: trajectory_tokens

    id_logits, _, _ = model.id_decoder(seq_info)
    # id_logits.shape = (1, 1, 1, N_active, K+1) = (1, 1, 1, N_active, 149)

    id_logits = id_logits[0, 0, 0]  # (N_active, 149)
    id_scores = id_logits.softmax(dim=-1)  # (N_active, 149)

# id_logits[i] の意味:
# - id_logits[i, 0]: 検出iがスロット0 (=track_id_0) である確率
# - id_logits[i, 1]: 検出iがスロット1 (=track_id_1) である確率
# - id_logits[i, 5]: 検出iがスロット5 (=track_id_2) である確率
# - id_logits[i, 2~4, 6~147]: 使用されていないスロット (低い確率)
# - id_logits[i, 148]: 検出iが新規物体である確率

# ========================================
# ステップ5: ID割り当て (object-max protocol)
# ========================================
# 各検出について、最も確率が高いスロットを選択
id_pred = torch.argmax(id_logits, dim=-1)  # (N_active,)

# 例:
# id_pred[10] = 0   → 検出10はスロット0 (track_id_0) に紐づく
# id_pred[25] = 1   → 検出25はスロット1 (track_id_1) に紐づく
# id_pred[50] = K   → 検出50は新規物体 (K=148)
# id_pred[80] = 5   → 検出80はスロット5 (track_id_2) に紐づく

# ========================================
# ステップ6: トラック更新
# ========================================
for det_idx in range(N_active):
    if score[det_idx] < det_thresh:
        continue  # 信頼度が低い検出は無視

    pred_slot = id_pred[det_idx]

    if pred_slot == K:  # 新規物体トークン (148)
        # 新規トラック作成
        if score[det_idx] > newborn_thresh:  # 新規物体の閾値
            new_slot = id_queue.popleft()  # 未使用スロットを取得 (例: 2)
            new_track_id = next_track_id
            next_track_id += 1

            active_tracks[new_track_id] = {
                'slot': new_slot,  # スロット2を新規トラックに割り当て
                'bbox': [bbox[det_idx]],
                'features': [features_t[det_idx]],
                'times': [t],
            }
    else:  # 既存スロットを予測 (0~147)
        # そのスロットを使用しているトラックを探す
        track_id = None
        for tid, track_info in active_tracks.items():
            if track_info['slot'] == pred_slot:
                track_id = tid
                break

        if track_id is not None:
            # 既存トラックに検出を追加
            active_tracks[track_id]['bbox'].append(bbox[det_idx])
            active_tracks[track_id]['features'].append(features_t[det_idx])
            active_tracks[track_id]['times'].append(t)
            active_tracks[track_id]['miss_count'] = 0
        # else: スロットは予測されたが対応するトラックがない (稀)

# ========================================
# ステップ7: 非アクティブトラックの削除
# ========================================
for track_id in list(active_tracks.keys()):
    if active_tracks[track_id]['miss_count'] > miss_tolerance:
        # トラックが長期間検出されなかった
        slot = active_tracks[track_id]['slot']
        del active_tracks[track_id]
        id_queue.append(slot)  # スロットを解放して再利用可能に
```

**重要なポイント**:

1. **履歴ID埋め込み = 各トラックが使用しているスロットのID埋め込み**
   - フレームt-1で track_id_0 がスロット0を使用していた
   - フレームtの履歴トークンにスロット0の埋め込みを付与
   - これで「スロット0 = track_id_0」という情報を ID Decoder に伝える

2. **現在ID埋め込み = 全て新規物体トークン**
   - フレームtの検出は「どのトラックに属するか分からない」状態
   - だから全て新規物体トークンを付与
   - ID Decoder が判断する

3. **ID Decoder の判断**
   - 入力文脈:
     - 履歴: 「スロット0/1/5がアクティブ」という情報
     - 現在: 「新規物体かもしれない検出」
   - 出力: 各検出がどのスロットに属するか
     - スロット0/1/5のどれか → 既存トラックに紐づく
     - スロット148 (新規物体トークン) → 新規トラック

4. **トラックIDとスロットの関係**
   - スロットは固定サイズ (K個)
   - トラックIDは無限に増える
   - スロット → トラックID のマッピングを動的に管理
   - トラック終了時にスロットを解放して再利用

#### まとめ: 訓練時と推論時の違い

| 項目 | 訓練時 | 推論時 |
|------|--------|--------|
| **K (ID語彙サイズ)** | データセット全体のユニークID数 (例: 148) | 訓練時と同じ (例: 148) |
| **Linear層の重み** | 学習される | 固定 (訓練時の重みを使用) |
| **スロット番号の意味** | GT ID → スロット番号の固定マッピング | 各トラックに動的に割り当て |
| **履歴ID埋め込み** | GT IDに対応するスロット埋め込み<br>(Hungarian Matchingで取得) | 各トラックが使用しているスロット埋め込み<br>(前フレームの追跡結果から) |
| **現在ID埋め込み** | 履歴に存在するID: GT ID埋め込み<br>初出現ID: 新規物体トークン | **全て新規物体トークン**<br>(どのトラックか不明なため) |
| **ID Decoderの出力** | GT IDスロット番号の分類 (教師あり) | どのスロットor新規物体かの分類 (教師なし) |
| **正解ラベル** | GT IDのスロット番号 | なし (予測のみ) |
| **スロット管理** | 固定マッピング (ID 5 → スロット1) | 動的マッピング (track_id_0 → スロット1)<br>キュー管理、割り当て・解放 |

#### SimpleMOTIPとの違い

**SimpleMOTIP（demo_training.py）**:
- 履歴フレーム: ランダムなID埋め込み
- 現在フレーム: 新規物体トークン（常に）
- 時系列関連付けが学習されない（処理フロー理解用のみ）

**実際のMOTIP（訓練時）**:
- 履歴フレーム: Ground TruthのID埋め込み
- 現在フレーム: 履歴に存在するID → GT ID埋め込み、初出現ID → 新規物体トークン
- 正解ラベル: Hungarian Matchingで取得したGT IDの語彙インデックス
- 正しい時系列関連付けを学習

**実際のMOTIP（推論時）**:
- 履歴フレーム: 過去の予測結果のID埋め込み（スロット番号）
- 現在フレーム: 新規物体トークン（常に）
- ID Vocabularyをスロットとして再利用し、可変数のトラックを管理

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
