# SAM3 アーキテクチャ理解用ドキュメント

このリポジトリは、SAM3 (Segment Anything Model 3) の複雑なコードベースを理解するための簡略化された疑似コードとドキュメントを提供します。

## 📁 ファイル構成

```
sam3_understanding/
├── README.md                    # このファイル
├── main_flow.py                # SAM3のメインフロー全体像 (画像セグメンテーション)
├── encoder.py                  # Transformer Encoder詳細
├── decoder.py                  # Transformer Decoder詳細
├── segmentation_head.py        # セグメンテーションヘッド詳細
├── video_tracking.py           # 動画追跡とメモリメカニズム
└── loss_computation.py         # 学習時のロス計算とマッチング
```

## 🎯 各ファイルの役割

### 1. [main_flow.py](main_flow.py)
**SAM3全体のデータフロー**

- 画像入力からマスク出力までの完全な処理パイプライン
- 各ステージでの入出力shape、軸の意味を詳細に記載
- 6つの主要ステージ:
  1. バックボーン特徴抽出 (Vision + Text)
  2. プロンプトエンコーディング
  3. Transformer Encoder (画像-プロンプト融合)
  4. Transformer Decoder (オブジェクトクエリ精製)
  5. セグメンテーションヘッド (マスク生成)
  6. スコアリング (信頼度計算)

**重要な入出力:**
- 入力: `(B, 3, 1008, 1008)` 画像
- 出力: `(B, 200, 1008, 1008)` マスク、`(B, 200, 4)` ボックス、`(B, 200)` スコア

---

### 2. [encoder.py](encoder.py)
**画像とプロンプトの融合**

主要コンポーネント:

#### `TransformerEncoder`
- Multi-scale画像特徴とプロンプトをクロスアテンションで融合
- 入力:
  - Vision: `List[(B, 256, H/4, W/4), (B, 256, H/8, W/8), ...]` - 4スケールの特徴
  - Prompt: `(L_prompt, B, 256)` - テキスト+幾何プロンプト
- 出力: `(HW, B, 256)` - 融合メモリ (全空間位置を平坦化)

#### `GeometryEncoder`
- 点、ボックス、マスクプロンプトを埋め込みに変換
- PositionEmbeddingRandom: フーリエ特徴ベースの位置エンコード
- 出力: `(L_geom, B, 256)` - 幾何プロンプト特徴

#### `VLCombiner`
- Vision Backbone (ViT) とText Encoderを統合
- Vision: Multi-scale特徴ピラミッド生成
- Text: BPEトークナイゼーション + Transformer

**処理フロー:**
```
Multi-scale特徴 (4レベル)
  → 平坦化 + レベル埋め込み追加
  → プロンプトとのクロスアテンション (6レイヤー)
  → 融合メモリ出力
```

---

### 3. [decoder.py](decoder.py)
**オブジェクトクエリの精製とボックス予測**

主要コンポーネント:

#### `TransformerDecoder`
- 200個の学習可能なクエリを使ってオブジェクトを検出
- **反復的ボックス改善**: 各レイヤーでボックスを精製
- 入力:
  - Queries: `(200, B, 256)` - オブジェクトクエリ
  - Memory: `(HW, B, 256)` - Encoderメモリ
  - Text: `(L_text, B, 256)` - テキスト特徴
- 出力 (6レイヤー分):
  - `hs`: `(6, B, 200, 256)` - 各レイヤーの隠れ状態
  - `pred_boxes`: `(6, B, 200, 4)` - 各レイヤーの予測ボックス
  - `pred_logits`: `(6, B, 200, 1)` - 信頼度

#### `TransformerDecoderLayer`
アテンション構成:
1. **Self-Attention**: クエリ間の相互作用
2. **Text Cross-Attention**: テキストプロンプトへの注意
3. **Memory Cross-Attention**: 画像メモリへの注意 (位置バイアス付き)
4. **FFN**: 非線形変換

#### `DotProductScoring`
- クエリとテキスト特徴のドット積で信頼度計算
- 温度パラメータで調整
- 出力: `(B, 200)` スコア

**重要な特徴:**
- **Reference Point**: 各クエリの参照座標を管理
- **Box Refinement**: デルタ値 `[delta_cx, delta_cy, log(w), log(h)]` で更新
- **Iterative**: 各レイヤーで段階的にボックス精度向上

---

### 4. [segmentation_head.py](segmentation_head.py)
**ピクセル単位のマスク生成**

主要コンポーネント:

#### `SegmentationHead`
- MaskFormerスタイルのマスク予測
- **Einstein Sum**: クエリ×ピクセル埋め込みでマスク生成
  ```python
  pred_masks = einsum('bqc,bchw->bqhw', mask_queries, pixel_embeddings)
  ```
- 入力:
  - Queries: `(B, 200, 256)` - Decoderクエリ
  - Vision: `List[(B, 256, H_i, W_i)]` - Multi-scale特徴
- 出力: `(B, 200, H, W)` - マスクロジット

#### `PixelDecoder`
- **FPN風のアップサンプリング**: Multi-scale特徴を統合
- Top-Down経路で低解像度→高解像度へ逐次統合
- 3段階のアップサンプリングで元画像サイズに復元
- 出力: `(B, 256, H, W)` - 高解像度ピクセル埋め込み

#### `MaskPostProcessor`
ポストプロセス機能:
- **threshold_masks**: ロジット→二値マスク変換
- **filter_by_score**: スコアでフィルタ&ソート
- **resize_masks**: マスクのリサイズ

**処理フロー:**
```
Multi-scale特徴
  → Lateral convs (各スケールを統一次元に)
  → Top-Down統合 (FPN)
  → 3段階アップサンプリング
  → 高解像度ピクセル埋め込み (B, 256, H, W)

クエリ (B, 200, 256)
  → MLP投影 → マスククエリ (B, 200, 256)

Einstein Sum: (B, 200, 256) × (B, 256, H, W)
  → マスク (B, 200, H, W)
```

---

### 5. [video_tracking.py](video_tracking.py)
**動画におけるオブジェクト追跡**

主要コンポーネント:

#### `Sam3VideoInference`
- フレーム間でメモリを維持して時間的一貫性を保つ
- 画像モード (is_image_only=True) と動画モード (False) をサポート
- 入力:
  - Video: `(T, 3, H, W)` - T フレームの動画
  - Prompts: 初期フレームでの点/ボックス/マスク
- 出力: `Dict[frame_idx -> (num_obj, H, W)]` - 全フレームのマスク

#### `SimpleMaskEncoder`
- 予測マスクをメモリ特徴に変換
- 処理: マスクダウンサンプル → Vision特徴と融合 → ConvNeXt精製
- 出力:
  - `maskmem_features`: `(B, 64, H_mem, W_mem)` - 空間メモリ
  - `obj_ptr`: `(B, 256)` - オブジェクトポインタ (dense表現)

#### メモリメカニズム
- **空間メモリ**: 過去フレームのマスク特徴 (空間的な位置情報)
- **オブジェクトポインタ**: オブジェクトの密な表現 (クラス・外観情報)
- **時間的位置エンコーディング**: フレーム間距離に基づく1D Sine波
- メモリ選択戦略:
  - 位置0: 条件付きフレーム (ユーザー指定、距離=0)
  - 位置1: 直前/直後フレーム (距離=1)
  - 位置2〜: temporal_stride ごとのフレーム

**追跡フロー:**
```
初期フレーム (frame 0)
  → ユーザープロンプトでセグメンテーション
  → マスクをメモリにエンコード

次フレーム (frame t)
  → バックボーン特徴抽出
  → メモリフレーム選択 (最大 num_maskmem=7 フレーム)
  → Transformer Encoder でメモリと融合
     - 空間メモリ: (HW_mem, B, 64)
     - オブジェクトポインタ: (num_mem, B, 256) + 時間位置エンコード
  → SAM Decoder でマスク生成
  → 新しいメモリにエンコード
  → 次フレームへ繰り返し
```

**重要なパラメータ:**
- `num_maskmem=7`: 使用するメモリフレーム数
- `memory_temporal_stride=1`: メモリサンプリングストライド
- `mem_dim=64`: メモリ特徴次元 (圧縮)
- `hidden_dim=256`: オブジェクトポインタ次元

**入出力shape詳細:**

| 処理ステージ | データ | Shape | 意味 |
|-------------|-------|-------|------|
| 入力動画 | video | (T, 3, H, W) | T=フレーム数, 3=RGB, H/W=1008 |
| バックボーン出力 | vision_feat | (HW, B, 256) | HW=空間位置, B=バッチ |
| 空間メモリ | maskmem_features | (B, 64, 126, 126) | 64=mem_dim, 126x126=H/8 |
| オブジェクトポインタ | obj_ptr | (num_mem, B, 256) | num_mem=メモリフレーム数 |
| 時間位置エンコード | t_pos_enc | (num_mem, B, 256) | フレーム距離に基づく |
| 予測マスク | pred_masks | (B, H, W) | フレームごとのマスク |

---

### 6. [loss_computation.py](loss_computation.py)
**学習時のロス計算**

主要コンポーネント:

#### `Sam3LossWrapper`
- 全ロスを統合管理
- Deep Supervision: 各Decoderレイヤーでロス計算
- One-to-One (O2O) + One-to-Many (O2M) マッチング
- 出力: `Dict['core_loss': Tensor, 'loss_ce': Tensor, ...]`

#### `BinaryHungarianMatcher`
- **ハンガリアンアルゴリズム**: 予測とGTの最適1対1マッチング
- コスト行列の構成:
  - `cost_class * 2.0`: 分類コスト (Focal Loss)
  - `cost_bbox * 5.0`: ボックスL1距離
  - `cost_giou * 2.0`: 負のGIoU
- 出力: `List[Tuple(src_idx, tgt_idx)]` - バッチごとのマッチング結果

#### `BinaryOneToManyMatcher`
- **貪欲マッチング**: 1つのGTに複数予測をマッチ可能
- スコア × IoU でソート、上位 topk=4 個を選択
- 補助出力 (aux_outputs) 用

#### ロス関数

**1. ClassificationLoss (分類)**
- Sigmoid Focal Loss
- 正例重み: `pos_weight=10.0`
- alpha=0.25, gamma=2.0
- 重み: `loss_ce: 20.0`

**2. BoxLoss (ボックス)**
- L1 Loss: 正規化座標 (cx, cy, w, h) の距離
- GIoU Loss: 一般化IoU
- 重み: `loss_bbox: 5.0`, `loss_giou: 2.0`

**3. MaskLoss (マスク)**
- Sigmoid Focal Loss: 不確実性の高いピクセルを重視
- Dice Loss: 領域の重なりを最大化
- ポイントサンプリング: 効率化のため 12544 点をサンプル
- 重み: `loss_mask: 200.0`, `loss_dice: 10.0`

**ロス計算フロー:**
```
モデル出力 + Ground Truth
  ↓
ハンガリアンマッチング (O2O)
  → (src_idx, tgt_idx) マッチング結果
  ↓
各ロス関数適用
  → loss_ce, loss_bbox, loss_giou, loss_mask, loss_dice
  ↓
O2M マッチング (aux_outputs用)
  → 同様のロス計算 × o2m_weight=2.0
  ↓
Deep Supervision (各Decoderレイヤー)
  → 各レイヤーでマッチング + ロス計算
  → サフィックス _aux_0, _aux_1, ... を追加
  ↓
総ロス = Σ(重み付きロス) / num_boxes
```

**正規化:**
- `normalization="global"`: 全GPU平均 (分散学習)
- `num_boxes`: バッチ内のGT総数 (最小1)

**入出力shape詳細:**

| データ | Shape | 意味 |
|-------|-------|------|
| **モデル出力** | | |
| pred_logits | (B, 200, 1) | 分類スコア |
| pred_boxes | (B, 200, 4) | ボックス (cx,cy,w,h) |
| pred_boxes_xyxy | (B, 200, 4) | ボックス (x1,y1,x2,y2) |
| pred_masks | (B, 200, H, W) | マスクロジット |
| **Ground Truth** | | |
| boxes | (N_total, 4) | 全GTボックス (flatten) |
| boxes_xyxy | (N_total, 4) | GT (x1,y1,x2,y2) |
| masks | (N_total, H, W) | GTマスク |
| num_boxes | (B,) | 各画像のGT数 |
| **マッチング結果** | | |
| indices | List[Tuple] 長さB | 各画像の (src_idx, tgt_idx) |
| src_idx | (M,) | マッチした予測インデックス |
| tgt_idx | (M,) | マッチしたGTインデックス |

**ロス重みのデフォルト値:**
```yaml
loss_ce: 20.0         # 分類
loss_bbox: 5.0        # ボックスL1
loss_giou: 2.0        # GIoU
loss_mask: 200.0      # マスクFocal
loss_dice: 10.0       # Dice
o2m_weight: 2.0       # O2Mロスの全体重み
```

---

## 🔍 SAM3の全体アーキテクチャ

### データフロー図

```
┌─────────────────────────────────────────────────────────────┐
│                      入力データ                              │
│  ・画像 (B, 3, 1008, 1008)                                   │
│  ・テキストプロンプト ["cat", "dog", ...]                     │
│  ・幾何プロンプト (点/ボックス/マスク)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐              ┌───────▼────────┐
│  Vision Backbone│              │ Text Encoder   │
│  (ViT + Neck)  │              │ (Transformer)  │
│                │              │                │
│ Multi-scale:   │              │ BPE Tokenizer  │
│ 4x, 2x, 1x, 0.5x│              │ + 24 layers    │
└───────┬────────┘              └───────┬────────┘
        │                               │
        │         ┌─────────────────────┤
        │         │                     │
        │   ┌─────▼──────┐              │
        │   │ Geometry   │              │
        │   │ Encoder    │              │
        │   │ (点/ボックス) │              │
        │   └─────┬──────┘              │
        │         │                     │
        └─────────┴─────────────────────┘
                  │
         ┌────────▼────────┐
         │ Transformer     │
         │ Encoder         │
         │ (6 layers)      │
         │                 │
         │ Image-Prompt    │
         │ Fusion          │
         └────────┬────────┘
                  │
           Memory (HW, B, 256)
                  │
         ┌────────▼────────┐
         │ Transformer     │
         │ Decoder         │
         │ (6 layers)      │
         │                 │
         │ 200 Queries     │
         │ Box Refinement  │
         └────────┬────────┘
                  │
      ┌───────────┴────────────┐
      │                        │
┌─────▼──────┐        ┌────────▼────────┐
│Segmentation│        │ Box Predictor   │
│   Head     │        │ + Scoring       │
│            │        │                 │
│ Pixel Dec  │        │ Reference Points│
│ + Einsum   │        │ Iterative Refine│
└─────┬──────┘        └────────┬────────┘
      │                        │
      └───────────┬────────────┘
                  │
         ┌────────▼────────┐
         │   出力           │
         │                 │
         │ ・Masks         │
         │   (B,200,H,W)   │
         │ ・Boxes         │
         │   (B,200,4)     │
         │ ・Scores        │
         │   (B,200)       │
         └─────────────────┘
```

---

## 📊 主要な次元とその意味

### バッチ次元
- `B`: バッチサイズ (通常1-4程度)

### 空間次元
- `H, W`: 画像の高さ・幅 (通常1008x1008)
- `H/4, W/4`: 1/4解像度 (252x252)
- `H/8, W/8`: 1/8解像度 (126x126)
- `H/16, W/16`: 1/16解像度 (63x63)
- `H/32, W/32`: 1/32解像度 (31x31)
- `HW`: 全空間位置のflatten (H/4*W/4 + H/8*W/8 + ...)

### シーケンス次元
- `L_text`: テキストトークン長 (可変、20-50程度)
- `L_geom`: 幾何プロンプト数 (点数+ボックス数+マスク数)
- `L_prompt`: 全プロンプト長 (L_text + L_geom)
- `N_q`: オブジェクトクエリ数 (固定で200)

### 時間次元 (動画)
- `T`: 動画のフレーム数 (可変)
- `num_maskmem`: 各フレームで使用するメモリフレーム数 (通常7)
- `memory_temporal_stride`: メモリサンプリングストライド (通常1)
- `t_pos`: 現在フレームからの時間的距離 (フレーム数)

### 特徴次元
- `256`: 統一特徴次元 (d_model)
- `1024`: ViT内部の埋め込み次元
- `2048`: FFN中間次元
- `64`: メモリ特徴次元 (mem_dim) - 圧縮された表現

### レイヤー次元
- `6`: Encoder/Decoderのレイヤー数
- `8`: アテンションヘッド数

---

## 🧩 重要な処理とテクニック

### 1. Multi-scale特徴抽出
- **ViTDet**: Vision Transformerベースの検出器
- **Neck (FPN)**: 4つの解像度レベル (4x, 2x, 1x, 0.5x)
- **レベル埋め込み**: 各スケールを識別する学習可能な埋め込み

### 2. Vision-Language融合
- **Text Encoder**: Vision-Encoder風のTransformer (24層)
- **Cross-Attention**: 画像特徴がテキスト特徴に注目
- **Prompt統合**: テキスト+幾何プロンプトを結合

### 3. オブジェクトクエリ
- **学習可能**: 200個の固定クエリ (データから学習)
- **反復精製**: 各Decoderレイヤーでボックスを改善
- **Reference Point**: 各クエリの注目すべき空間座標

### 4. マスク生成
- **Einstein Sum**: クエリとピクセルのドット積
  - メモリ効率的: 200×256 と 256×H×W のみ
  - 並列化可能: 全クエリを同時処理
- **FPN風デコーダ**: Top-Down統合で高解像度復元

### 5. 位置エンコーディング
- **RoPE**: Rotary Position Embedding (ViT用)
- **Sine-based**: 正弦波ベースの絶対位置
- **Fourier Features**: ランダムフーリエ特徴 (幾何プロンプト用)

### 6. 効率化テクニック
- **Activation Checkpointing**: メモリ削減
- **Divide-and-Conquer (DAC)**: One-to-Many マッチング効率化
- **torch.compile**: 実行時最適化

### 7. 動画追跡のメモリメカニズム
- **Spatial Memory**: ConvNeXtベースのマスクエンコーダーで生成
  - 入力マスクを16倍ダウンサンプル + Vision特徴と融合
  - 出力を64次元に圧縮 (メモリ効率化)
- **Object Pointer**: オブジェクトの密な表現
  - SAM Decoderの出力トークンから抽出
  - 時間的位置エンコーディングと共に使用
- **Memory Selection**: 効率的なメモリ管理
  - 条件付きフレーム (ユーザー指定) を優先
  - temporal_stride でサンプリング
  - 最大 num_maskmem 個まで保持

### 8. ハンガリアンマッチング
- **二部グラフマッチング**: 予測とGTの最適割り当て
- **コスト関数**: 分類 + ボックスL1 + GIoU の重み付き和
- **scipyの実装**: `linear_sum_assignment` を使用
- **バッチ処理**: 各画像で独立にマッチング

### 9. Deep Supervision
- **各レイヤーでロス**: 6つのDecoderレイヤー全てで予測とロス計算
- **段階的学習**: 浅いレイヤーから深いレイヤーまで監督
- **補助出力**: aux_outputs として保存、サフィックス付きでロス記録

---

## 💡 使用方法

### 1. コード全体の理解
まず `main_flow.py` を読んで全体像を把握:
```python
# main_flow.pyを実行して処理フローを確認
python main_flow.py
```

### 2. 各モジュールの詳細理解
興味のあるコンポーネントの詳細を確認:
```python
# Encoderの詳細
python encoder.py

# Decoderの詳細
python decoder.py

# Segmentation Headの詳細
python segmentation_head.py
```

### 3. 動画追跡の実行
```python
# video_tracking.pyを実行して動画追跡を確認
python video_tracking.py
```

### 4. ロス計算の確認
```python
# loss_computation.pyを実行してロス計算を確認
python loss_computation.py
```

---

## 🔗 SAM3コードベースとの対応

### 簡略化の方針

✅ **含まれる内容:**
- データフローの全体構造
- 入出力のshapeと軸の意味
- 主要なアテンションメカニズム
- マスク生成の計算方法
- 動画追跡のメモリメカニズム
- ロス計算とマッチング戦略

❌ **省略した詳細:**
- ViTの内部実装 (timmライブラリ使用を想定)
- RoPEの詳細計算
- Deformable Attentionの実装
- 分散学習の詳細 (SPMD等)
- データローダーとプリプロセス
- 評価メトリクス (mAP, mIoU)
- ビデオ推論の最適化詳細 (メモリオフロード等)

---

## 📚 参考資料

### SAM3関連
- SAM3論文: Segment Anything Model 3 (2024)

### 関連技術
- **Vision Transformer (ViT)**: "An Image is Worth 16x16 Words" (ICLR 2021)
- **DETR**: "End-to-End Object Detection with Transformers" (ECCV 2020)
- **MaskFormer**: "Per-Pixel Classification is Not All You Need" (NeurIPS 2021)
- **RoPE**: "Rotary Position Embedding" (arXiv 2021)

### 実装ライブラリ
- **timm**: PyTorch Image Models (ViT実装)
- **torch.nn**: 標準的なTransformer層
- **einops**: テンソル操作ライブラリ

---

## 🤔 よくある質問

### Q1: なぜクエリ数は200個固定なのか?
A: DETR系のアーキテクチャでは、最大検出数をあらかじめ設定します。200個は一般的な画像内のオブジェクト数を十分カバーする数です。実際には、スコアでフィルタリングして有効なオブジェクトのみ使用します。

### Q2: Multi-scaleはなぜ必要?
A: 小さいオブジェクト(高解像度特徴が必要)と大きいオブジェクト(広い受容野が必要)の両方を検出するため、複数の解像度を使います。

### Q3: Einstein Sumとは?
A: `torch.einsum` を使った効率的なテンソル積です。マスク生成では、各クエリ(クエリ埋め込み)と各ピクセル(ピクセル埋め込み)の類似度をドット積で計算します。

### Q4: Reference Pointの役割は?
A: 各クエリが注目すべき空間座標を指定します。これにより、Decoderのアテンションを効率化し、ボックス予測の精度を向上させます。

### Q5: なぜ6レイヤー全ての出力を保存?
A: 学習時に各レイヤーの予測に対してロスを計算する(Deep Supervision)ため、全レイヤーの出力が必要です。推論時は最終レイヤーのみ使用します。

### Q6: 動画追跡でなぜメモリを圧縮 (64次元) するのか?
A: 長い動画では多数のフレームのメモリを保持する必要があります。256次元のまま保存するとGPUメモリを大量に消費するため、64次元に圧縮して効率化しています。空間的な情報は低次元でも十分表現できます。

### Q7: Object Pointerとは何か?
A: オブジェクトの「何であるか」(クラス、外観)を表す密なベクトル表現です。空間メモリが「どこにあるか」を表すのに対し、Object Pointerは「何か」を表します。これにより、オブジェクトが動いても追跡できます。

### Q8: なぜハンガリアンマッチングが必要なのか?
A: DETR系モデルでは200個のクエリが並列に予測を出力しますが、どのクエリがどのGTオブジェクトに対応するかは決まっていません。ハンガリアンアルゴリズムで最適な対応付けを見つけることで、各予測に正しいGTを割り当ててロスを計算できます。

### Q9: One-to-Many (O2M) マッチングの目的は?
A: 1つのGTに対して複数の予測をマッチさせることで、補助出力の学習を強化します。これにより、浅いDecoderレイヤーでも多様な予測を学習でき、最終的な精度向上に繋がります。

### Q10: Deep Supervisionの効果は?
A: 各Decoderレイヤーの出力に対してロスを計算することで、浅いレイヤーから深いレイヤーまで段階的に学習が進みます。これにより、勾配消失を防ぎ、学習を安定化させます。

---

## ✨ まとめ

このリポジトリは、SAM3の複雑な実装を理解するための教育的な疑似コードです。

**カバー範囲:**
1. ✅ **画像セグメンテーション**: main_flow.py, encoder.py, decoder.py, segmentation_head.py
2. ✅ **動画追跡**: video_tracking.py - メモリメカニズムと時間的一貫性
3. ✅ **学習**: loss_computation.py - ロス計算とマッチング戦略

**実際の実装に必要なもの:**
1. **ViTバックボーン**: timmやtorchvisionから取得
2. **Text Encoder**: CLIP等の事前学習モデル
3. **学習ループ**: オプティマイザ、スケジューラ、データローダー
4. **評価コード**: mAP、mIoU等のメトリクス
5. **推論最適化**: メモリオフロード、torch.compile等

この疑似コードをベースに、SAM3の画像・動画・学習の全体像を理解できます!
