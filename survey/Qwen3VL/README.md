# Qwen3VL Understanding - 簡略化疑似コード集

Qwen3-VL (Qwen3 Visual Language Model) の理解を目的とした簡略化疑似コード集です。

論文: [Qwen3-VL Technical Report](https://arxiv.org/abs/2506.xxxxx) (2025)
公式コード: [github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

## 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [主要イノベーション](#主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**Qwen3-VLの特徴:**
- **動的解像度**: 画像を14×14pxのパッチで処理、解像度に応じてトークン数が変動
- **Interleaved MRoPE**: Qwen2.5-VLのChunked方式に対し、時間的位置をタイムスタンプで表現
- **DeepStack Vision**: 中間ViT特徴量をLLMの各層に注入してVision-Language融合を強化
- **Text Timestamps**: 動画内のフレームに秒数タイムスタンプを付与し時間的理解を向上
- **長コンテキスト**: 128K tokens のコンテキスト長

**タスク:**
- 画像理解・VQA・OCR
- 動画理解（長時間動画対応）
- Agentic タスク (GUI操作、コーディング)
- 時空間的理解 (Grounding, Temporal Localization)

**利用可能なモデルサイズ:**

| モデル | パラメータ数 | Vision Encoder | LLM Backbone |
|--------|------------|---------------|--------------|
| Qwen3-VL-2B | 2B | SigLIP-2 (400M) | Qwen3-2B |
| Qwen3-VL-7B | 7B | SigLIP-2 (400M) | Qwen3-7B |
| Qwen3-VL-72B | 72B | SigLIP-2 (400M) | Qwen3-72B |
| Qwen3-VL-30B-A3B | 30B(MoE) | SigLIP-2 (400M) | Qwen3-30B-A3B |

**性能 (Qwen3-VL-72B、主要ベンチマーク):**

| ベンチマーク | タスク | スコア |
|-------------|--------|--------|
| MMMU | 大学レベルVQA | 74.3 |
| MathVista | 数学的推論 | 81.6 |
| DocVQA | 文書理解 | 96.5 |
| VideoMME | 動画理解 | 73.5 |
| RealWorldQA | 実世界QA | 74.0 |
| MathVision | 視覚数学 | 55.4 |

---

## アーキテクチャ全体像

```
入力: 画像/動画 + テキスト
    image: PIL.Image (任意解像度)
    text: "この画像を説明してください"
        ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 前処理 (Processor)                                     │
│    smart_resize(): アスペクト比維持でリサイズ               │
│    patch_size=14, merge_size=2                           │
│    → pixel_values: (N_patches, C×P×P)                   │
│    → image_grid_thw: (1, 3) [T=1, H_patches, W_patches] │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Vision Encoder (SigLIP-2 ViT)                        │
│    patch_embed: (N_patches, C×P×P) → (N_patches, D_v)   │
│    ViT layers × L_v (32層)                               │
│    DeepStack: 中間層の特徴量を保存 (6層ごとに)              │
│    → visual_features: (N_patches, D_v)                  │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. MLP Merger (Vision-Language Connector)               │
│    2×2空間圧縮: N_patches → N_patches / 4               │
│    Linear: D_v × 4 → D_llm                              │
│    → visual_tokens: (N_v, D_llm)                        │
│    N_v = H_patches/2 × W_patches/2 (per image)          │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Interleaved MRoPE 位置ID計算                          │
│    テキストトークン: t=h=w=連続インデックス                  │
│    画像トークン: t=0, h=[0,H_grid), w=[0,W_grid)          │
│    動画トークン: t=タイムスタンプ(秒)×2, h/w=空間インデックス │
│    position_ids: (3, B, T_seq)                          │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. LLM (Qwen3)                                          │
│    入力: visual_tokens + text_tokens (混合シーケンス)      │
│    Qwen3 Transformer layers × L_llm                     │
│    各層でDeepStack特徴を受け取る (cross-attn or add)       │
│    3D MRoPE: (3, B, T_seq) → cos/sin embeddings         │
│    → logits: (B, T_seq, vocab_size)                     │
└─────────────────────────────────────────────────────────┘
        ↓
出力: 生成テキスト / logits
```

---

## ファイル構成

| ファイル | 内容 | 主要クラス/関数 |
|---------|------|---------------|
| [main_flow.py](main_flow.py) | 全体フォワードパス | `Qwen3VLForConditionalGeneration.forward()` |
| [vision_encoder.py](vision_encoder.py) | SigLIP-2 ViT + MLP Merger | `Qwen3VisionTransformerPretrainedModel`, `PatchMerger` |
| [rope_and_position.py](rope_and_position.py) | Interleaved MRoPE | `get_rope_index_3()`, `apply_multimodal_rotary_pos_emb()` |
| [deepstack.py](deepstack.py) | DeepStack中間特徴注入 | `DeepStackLayer`, `DeepStackConfig` |
| [loss_computation.py](loss_computation.py) | 学習損失計算 | `compute_loss()`, ラベルマスキング |
| [finetuning_example.py](finetuning_example.py) | 実際に動くファインチューニングスクリプト | `Qwen3VLDataset`, `Qwen3VLCollator`, `train()` |

---

## 主要イノベーション

### 1. Interleaved MRoPE (vs Qwen2.5-VL の Chunked MRoPE)

**Qwen2.5-VL (Chunked)**:
```
動画フレーム位置: t = 0, 0, ..., 1, 1, ..., 2, 2, ...  (フレームインデックス)
時間間隔: second_per_grid_t × 2 でスケーリング
```

**Qwen3-VL (Interleaved with Timestamps)**:
```
フレーム間にタイムスタンプトークン <t1>, <t2>, ... を挿入
各フレームは単独でt=0として処理 (llm_grid_t=1 に分割)
時間情報はトークン列の構造で表現 (位置IDではなくテキストで)

例:
  <t0.0> <vision_start> [frame1 patches] <vision_end>
  <t0.5> <vision_start> [frame2 patches] <vision_end>
  "動画の説明をしてください"
```

**利点**: LLMがタイムスタンプを直接テキストとして読むため、時間的位置関係を自然言語として理解できる。

### 2. DeepStack Vision

**問題**: 標準的なViT-LLM接続では最終層の特徴量のみを使用

**解決策**:
```
ViT層 1  →
ViT層 2  →
...
ViT層 6  → 中間特徴量 → LLMの第1レイヤーへ注入
ViT層 12 → 中間特徴量 → LLMの第2レイヤーへ注入
...
ViT層 32 → 最終特徴量 → MLP Merger → LLMの入力埋め込みへ
```

**効果**: ViTの異なる抽象度の特徴量がLLMの各深さに対応して注入される。低レベル特徴(テクスチャ)はLLMの浅い層、高レベル特徴(意味)はLLMの深い層へ。

### 3. Text Timestamps

動画フレームに秒数タイムスタンプを付与:
```
入力トークン列:
  "<|video_start|> <t0.00> <|vision_start|> [frame_patches] <|vision_end|>
   <t0.50> <|vision_start|> [frame_patches] <|vision_end|>
   <t1.00> <|vision_start|> [frame_patches] <|vision_end|> <|video_end|>"
```

モデルは「XXX秒の動作は何ですか？」という質問に秒数で回答可能。

### 4. 動的解像度 (Native Resolution)

```python
# 画像を14×14pxのパッチに分割（パディングなし）
patch_size = 14  # ViTのパッチサイズ
merge_size = 2   # MLP Mergerの空間圧縮倍率

# 例: 448×448px画像
# → (448/14) × (448/14) = 32×32 = 1024 patches (ViT入力)
# → MLP Mergerで2×2圧縮: 512 visual tokens (LLM入力)

# 例: 896×448px画像
# → 64×32 = 2048 patches (ViT入力)
# → 1024 visual tokens (LLM入力)
```

---

## 処理フロー詳細

### 推論フロー

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# モデル・Processor ロード
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-7B-Instruct")

# 入力準備
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "この画像を詳しく説明してください。"},
        ],
    }
]

# テキスト + 画像前処理
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
)
# inputs:
#   input_ids:        (1, T_seq) int64
#   attention_mask:   (1, T_seq) int64
#   pixel_values:     (N_patches, C×P×P) float  # 全パッチをバッチ次元に展開
#   image_grid_thw:   (1, 3) int  # [T=1, H_patches, W_patches]

# 生成
output_ids = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(output_ids[0][len(inputs.input_ids[0]):])
```

### 訓練フロー

```python
# ファインチューニングの詳細は finetuning_example.py 参照

# 主な手順:
# 1. Dataset: conversations形式のJSONLからロード
# 2. Collator: visual token + text token の混合シーケンスを生成
#    - pixel_values: (total_patches, C×P×P)
#    - input_ids: (B, T_seq) with image placeholder tokens
#    - labels: -100でvision/prefix部分をマスク、assistant応答のみが損失計算対象
#    - position_ids: (3, B, T_seq) MRoPE用
# 3. Forward:
#    vision_encoder → merger → LLM
#    loss = cross_entropy(logits[:, :-1], labels[:, 1:])
#           (labels == -100 の部分は除外)
```

---

## 形状ガイド

### 軸の意味

| 変数 | 意味 | 典型値 |
|------|------|--------|
| B | バッチサイズ | 1〜32 |
| T_seq | LLMのシーケンス長 (テキスト+視覚トークン) | 〜128K |
| N_patches | ViT入力パッチ数 (全画像の合計) | 画像に依存 |
| N_v | LLM入力の視覚トークン数 = N_patches / 4 | N_patches/4 |
| P | パッチサイズ = 14 | 14 |
| C | 入力チャンネル数 = 3 (RGB) | 3 |
| D_v | Vision Encoder隠れ次元 | 1152 |
| D_llm | LLM隠れ次元 | 3584 (7B) |
| H_v | Vision Encoderのアテンションヘッド数 | 16 |
| L_v | Vision Encoderのレイヤー数 | 32 |
| L_llm | LLMのレイヤー数 | 28 (7B) |
| vocab_size | 語彙サイズ | 151936 |
| H_patches | パッチグリッドの高さ = 画像高さ / P | 画像に依存 |
| W_patches | パッチグリッドの幅 = 画像幅 / P | 画像に依存 |
| merge_size | MLP Mergerの空間圧縮倍率 = 2 | 2 |

### 各段階のテンソル形状

| 段階 | テンソル名 | 形状 | 説明 |
|------|-----------|------|------|
| **前処理後** | pixel_values | `(N_patches, C×P²)` = `(N_patches, 3×196)` | 全パッチを平坦化 |
| | image_grid_thw | `(num_images, 3)` | 各画像の [T,H,W] パッチ数 |
| **ViT内部** | patch_embeds | `(N_patches, D_v)` | パッチ埋め込み後 |
| | attention_output | `(N_patches, D_v)` | ViT各層の出力 |
| **DeepStack** | intermediate_feats | `List[(N_patches, D_v)]` | 中間層特徴量 |
| **Merger後** | visual_tokens | `(N_v, D_llm)` | LLMに入力される視覚トークン |
| **LLM入力** | inputs_embeds | `(B, T_seq, D_llm)` | 混合テキスト+視覚埋め込み |
| **MRoPE** | position_ids | `(3, B, T_seq)` | [temporal, height, width] |
| **LLM出力** | hidden_states | `(B, T_seq, D_llm)` | 最終隠れ状態 |
| **ロジット** | logits | `(B, T_seq, vocab_size)` | 語彙分布 |

---

## FAQ

### Q1: pixel_values の形状はなぜ (N_patches, C×P²) なのか？

**A**: Qwen3-VLはNative Dynamic Resolutionに対応するため、バッチ内の画像サイズがすべて異なります。そのため画像次元でバッチ化せず、すべてのパッチを1次元に並べています。

```
画像1: 448×448 → 32×32=1024 patches
画像2: 672×336 → 48×24=1152 patches
合計 N_patches = 1024 + 1152 = 2176
pixel_values: (2176, 3×196)
image_grid_thw: [[1, 32, 32], [1, 48, 24]]
```

### Q2: Interleaved MRoPE vs Chunked MRoPE の違いは？

**A**: 動画の時間的位置をどう表現するかの違いです。

```
Qwen2.5-VL (Chunked):
  位置ID: t=[0,0,0,0, 1,1,1,1, 2,2,2,2]  ← フレームインデックス
  video_grid_thw: [3, H, W]  ← 複数フレームが1まとまり

Qwen3-VL (Interleaved):
  各フレームが独立: video_grid_thw = [[1,H,W], [1,H,W], [1,H,W]]
  時間情報はテキストタイムスタンプ: "<t0.00>", "<t0.50>", "<t1.00>"
  位置ID: すべてt=0 (時間的相対位置は不要)
```

### Q3: DeepStackの中間特徴量はどう利用されるか？

**A**: ViTの特定レイヤー（例: 6層ごと）の出力をLLMの対応するレイヤーに注入します。注入方式は加算またはクロスアテンションです。

```python
# ViTのL_v=32層を分割してLLMのL_llm=28層に対応
# injection_schedule = [6, 12, 18, 24, 32]  → LLM層[5, 11, 17, 23, 27]

# LLM forward内:
for llm_layer_idx, llm_layer in enumerate(llm_layers):
    if llm_layer_idx in injection_points:
        vit_feat = intermediate_vit_features[injection_points[llm_layer_idx]]
        hidden_state = hidden_state + cross_attn(hidden_state, vit_feat)
    hidden_state = llm_layer(hidden_state)
```

### Q4: 動的解像度での最大・最小トークン数は？

**A**: コードの定数から:

```python
IMAGE_MIN_TOKEN_NUM = 4      # 最小4トークン
IMAGE_MAX_TOKEN_NUM = 16384  # 最大16384トークン (merger後)
VIDEO_MIN_TOKEN_NUM = 128    # 動画フレームあたり最小128トークン
VIDEO_MAX_TOKEN_NUM = 768    # 動画フレームあたり最大768トークン
```

### Q5: LoRAでのファインチューニングのターゲットモジュールは？

**A**: 公式コードより:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_r = 64
lora_alpha = 128
lora_dropout = 0.05
```

Vision Encoder (`model.visual`) はデフォルトで凍結し、LLMの注意機構のみを学習させるのが推奨です。

### Q6: Vision Encoderのパッチ次元は？

**A**: SigLIP-2 ViTの仕様:
```
patch_size: 14 (14×14ピクセル/パッチ)
embed_dim D_v: 1152 (7B/72Bモデル共通)
num_heads H_v: 16
num_layers L_v: 32
MLP ratio: 4 (FFN内部次元 = 4 × 1152)
```

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装とは異なる場合があります。
