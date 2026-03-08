# Qwen2.5-Omni アーキテクチャ理解用ドキュメント

このリポジトリは、Qwen2.5-Omni の複雑なコードベースを理解するための簡略化された疑似コードとドキュメントを提供します。

## ファイル構成

```
Qwen2.5Omni_understanding/
├── README.md                          # このファイル
├── main_flow.py                       # 全体のデータフロー (入力 → Thinker → Talker → 音声出力)
├── audio_encoder.py                   # Whisper系 Audio Encoder 詳細
├── vision_encoder.py                  # ViT系 Vision Encoder + PatchMerger 詳細
├── thinker.py                         # Thinker LLM (マルチモーダル統合 + テキスト生成)
├── talker.py                          # Talker (テキスト表現 → 音声コードトークン生成)
├── token2wav.py                       # Token2Wav (DiT + BigVGAN による音声波形生成)
├── loss_computation.py                # 学習時のロス計算 (Pre-training + Post-training)
├── lora_finetune_text_image.py        # LoRA ファインチューニング サンプル (Text-Image)
└── lora_finetune_text_audio.py        # LoRA ファインチューニング サンプル (Text-Audio)
```

## 各ファイルの役割

### 1. [main_flow.py](main_flow.py)
**Qwen2.5-Omni全体のデータフロー**

- テキスト/画像/動画/音声入力からテキスト+音声出力までの完全な処理パイプライン
- 各ステージでの入出力shape、軸の意味を詳細に記載
- 5つの主要ステージ:
  1. 入力前処理 (テキストトークン化、メルスペクトログラム、画像パッチ化)
  2. エンコーダ (Audio Encoder、Vision Encoder)
  3. Thinker LLM (マルチモーダル特徴統合 + テキスト生成)
  4. Talker (テキスト表現 → 音声コードトークン)
  5. Token2Wav (コードトークン → 音声波形)

**重要な入出力:**
- 入力: テキスト `(B, L_text)`, 音声 `(B, 128, T_mel)`, 画像 `(N_patches, C_hidden)`, 動画 `(T*H*W, C_hidden)`
- 出力: テキスト `(B, L_gen, 151643)`, 音声波形 `(1, N_samples)` (24kHz)

---

### 2. [audio_encoder.py](audio_encoder.py)
**Whisperベースの音声エンコーダ**

主要コンポーネント:

#### `AudioEncoder`
- Whisper-large-v3 から初期化
- 16kHz音声 → 128チャネルメルスペクトログラム → 特徴ベクトル系列
- 入力: `(batch, 128, T_mel)` - メルスペクトログラム
- 出力: `(batch, T_mel//4, output_dim)` - 4倍ダウンサンプリング後の音声特徴
- ブロック単位アテンション (2秒ブロック) でストリーミング対応

**処理フロー:**
```
メルスペクトログラム (B, 128, T_mel)
  → Conv1D + GELU (B, d_model, T_mel)
  → Conv1D stride=2 + GELU (B, d_model, T_mel//2) ← 2倍ダウン
  → 正弦波位置エンコーディング追加
  → Transformer Encoder (flash attention, cu_seqlens)
  → AvgPool1D stride=2 (B, d_model, T_mel//4) ← さらに2倍ダウン
  → LayerNorm + Linear射影 (B, T_mel//4, output_dim)
```

---

### 3. [vision_encoder.py](vision_encoder.py)
**ViTベースの画像/動画エンコーダ**

主要コンポーネント:

#### `VisionEncoder`
- Qwen2.5-VL と同一の Vision Transformer (~675Mパラメータ)
- パッチサイズ14、temporal_patch_size=2
- ウィンドウアテンション + フルアテンション (選択レイヤー)
- PatchMerger: 隣接 2x2 パッチを1トークンに統合

#### `PatchMerger`
- spatial_merge_size=2: 4パッチ → 1トークン (解像度1/4)
- MLP投影で次元変換

**処理フロー:**
```
画像/動画ピクセル
  → パッチ埋め込み (patch_size=14, temporal_patch_size=2)
  → 2D RoPE位置エンコーディング (height, width 独立)
  → ウィンドウアテンション + フルアテンション (depth レイヤー)
  → PatchMerger (2x2統合)
  → 出力特徴 (N_merged, hidden_size)
```

---

### 4. [thinker.py](thinker.py)
**マルチモーダル統合 LLM (Thinker)**

主要コンポーネント:

#### `ThinkerForConditionalGeneration`
- Qwen2.5ベースの7B LLM
- TMRoPE (Time-aligned Multimodal RoPE) で時空間位置を統一
- テキスト/音声/画像/動画の特徴を統合してテキスト生成
- Low-VRAM モード: エンコーダをCPU↔GPU間で動的移動

#### `TMRoPE`
- 3軸位置エンコーディング: (temporal, height, width)
- テキスト: 3軸同一 (1D-RoPEと等価)
- 音声: 3軸同一、1 temporal ID = 40ms
- 画像: temporal固定、height/widthは空間位置
- 動画: temporalがフレーム間で動的増分 (1 ID = 40ms)
- 動画+音声: 2秒ごとにインターリーブ (視覚→聴覚)

**処理フロー:**
```
テキスト埋め込み (B, L_text, 4096)
  + 音声特徴 (audio_token位置にscatter)
  + 画像特徴 (image_token位置にscatter)
  + 動画特徴 (video_token位置にscatter)
  → 統合入力 (B, L_total, 4096)
  → TMRoPE位置ID: (3, B, L_total) [temporal, height, width]
  → Transformer Decoder (32層)
  → LM Head → logits (B, L_total, 151643)
```

---

### 5. [talker.py](talker.py)
**音声コードトークン生成 (Talker)**

主要コンポーネント:

#### `TalkerForConditionalGeneration`
- Thinkerの隠れ状態を受け取り、音声コードトークンを自己回帰生成
- テキストトークン埋め込み + コードトークン埋め込みの加算融合
- thinker_to_talker_proj で次元変換

**処理フロー:**
```
Thinkerの隠れ状態 (B, L_text, hidden)
  + codec_bos_token (開始トークン)
  → 自己回帰ループ:
    codec_embed (B, 1, hidden) + thinker_hidden[:, :1, :] ← 1トークンずつ消費
    → thinker_to_talker_proj → (B, 1, talker_hidden)
    → Talker Transformer
    → codec_head → logits (B, 1, codebook_size)
    → サンプリング → 次のcodecトークン
  → 音声コード系列 (B, L_codec)
```

---

### 6. [token2wav.py](token2wav.py)
**音声波形生成パイプライン (Token2Wav)**

主要コンポーネント:

#### `Token2WavModel`
- 2段階パイプライン: DiT + BigVGAN
- ストリーミング対応 (チャンク単位生成)

#### `DiTModel` (Diffusion Transformer)
- Flow-Matching方式のメルスペクトログラム生成
- Euler ODE Solver (10ステップ)
- スライディングウィンドウブロックアテンション (受容野4ブロック)
- guidance_scale=0.5, sway_coefficient=-1.0

#### `BigVGANModel` (Neural Vocoder)
- メルスペクトログラム → 音声波形 (24kHz)
- チャンク単位処理可能

**処理フロー:**
```
音声コードトークン (B, L_codec)
  → DiT Flow-Matching (10ステップ)
    noise (1, 30000, mel_dim) → 段階的にデノイズ
  → メルスペクトログラム (B, T_mel, 160)
  → BigVGAN Vocoder
  → 音声波形 (1, N_samples) at 24kHz
```

---

### 7. [loss_computation.py](loss_computation.py)
**学習時のロス計算**

主要コンポーネント:

#### Pre-training ロス
- **ステージ1 (エンコーダ学習)**: LLM凍結、音声/画像エンコーダのみ学習
- **ステージ2 (全パラメータ学習)**: 800Bトークン(画像/動画) + 300B(音声) + 100B(動画+音声)
- **ステージ3 (長系列学習)**: max_length 8192 → 32768 に拡張

#### Thinker Post-training ロス
- ChatML形式の指示追従データ
- 標準的なCross-Entropy Loss

#### Talker Post-training ロス (3段階)
1. **ICL (In-Context Learning)**: 音声継続タスク、next-token prediction
2. **DPO**: 音声生成品質向上
3. **Speaker Fine-tuning**: 特定話者への適応

#### DPO Loss
```
L_DPO = -E[log σ(β * (log P_θ(y_w|x)/P_ref(y_w|x) - log P_θ(y_l|x)/P_ref(y_l|x)))]
```

---

### 8. [lora_finetune_text_image.py](lora_finetune_text_image.py)
**LoRA ファインチューニング サンプル (Text-Image)**

- LLaMA-Factory / ms-swift 両方の設定例
- ShareGPT形式のデータセット準備
- Vision Tower凍結 + Thinker LLMのみLoRA適用
- 学習率1e-4、rank=8、target=all-linear

---

### 9. [lora_finetune_text_audio.py](lora_finetune_text_audio.py)
**LoRA ファインチューニング サンプル (Text-Audio)**

- 音声理解タスク向けのLoRAファインチューニング
- 音声前処理 (16kHz リサンプリング、メルスペクトログラム生成)
- Audio Tower凍結 + Thinker LLMのみLoRA適用

---

## Qwen2.5-Omni の全体アーキテクチャ

### データフロー図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          入力データ                                      │
│  ・テキスト "Describe this image"                                        │
│  ・画像 (B, 3, H, W) 可変解像度                                          │
│  ・動画 (T, 3, H, W) 動的フレームレート                                   │
│  ・音声 (N_samples,) at 16kHz                                            │
└───────────┬──────────────┬───────────────┬──────────────┬───────────────┘
            │              │               │              │
    ┌───────▼───────┐ ┌───▼───────┐ ┌─────▼──────┐ ┌────▼────────┐
    │ Text Tokenizer│ │Audio Enc. │ │Vision Enc. │ │Vision Enc.  │
    │ (BPE 151643)  │ │(Whisper)  │ │(ViT, 画像) │ │(ViT, 動画)  │
    │               │ │           │ │            │ │             │
    │ vocab=151643  │ │128ch mel  │ │patch=14    │ │patch=14     │
    │               │ │→ d_model  │ │+PatchMerge │ │temporal=2   │
    └───────┬───────┘ │→ 4x down  │ └─────┬──────┘ └─────┬───────┘
            │         │→ proj     │       │               │
            │         └─────┬─────┘       │               │
            │               │             │               │
    ┌───────▼───────────────▼─────────────▼───────────────▼───────────────┐
    │                                                                      │
    │                    Thinker (Qwen2.5-7B LLM)                          │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐    │
    │  │ テキスト埋め込み + masked_scatter (audio/image/video tokens)   │    │
    │  │ → 統合入力 (B, L_total, 4096)                                 │    │
    │  │                                                               │    │
    │  │ TMRoPE: position_ids (3, B, L_total)                          │    │
    │  │   [temporal, height, width]                                    │    │
    │  │                                                               │    │
    │  │ Transformer Decoder × 32 layers                               │    │
    │  │   hidden_size=4096, num_heads=32, intermediate=14336          │    │
    │  │                                                               │    │
    │  │ → LM Head → logits (B, L_total, 151643)                      │    │
    │  └──────────────────────────────┬───────────────────────────────┘    │
    │                                 │                                    │
    └─────────────────────────────────┼────────────────────────────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
          ┌───────▼───────┐   ┌───────▼───────┐          │
          │ テキスト出力   │   │ 隠れ状態      │          │
          │ (B, L, vocab) │   │ (B, L, 4096)  │          │
          │               │   │               │          │
          │ argmax/sample │   │ Talkerへ渡す  │          │
          └───────┬───────┘   └───────┬───────┘          │
                  │                   │                   │
                  │           ┌───────▼───────────────┐   │
                  │           │                       │   │
                  │           │   Talker              │   │
                  │           │                       │   │
                  │           │ thinker_hidden        │   │
                  │           │ + codec_embed         │   │
                  │           │ → proj → Transformer  │   │
                  │           │ → codec_head          │   │
                  │           │ → 音声コードトークン   │   │
                  │           │   (B, L_codec)         │   │
                  │           └───────┬───────────────┘   │
                  │                   │                   │
                  │           ┌───────▼───────────────┐   │
                  │           │                       │   │
                  │           │   Token2Wav           │   │
                  │           │                       │   │
                  │           │ DiT (Flow-Matching)   │   │
                  │           │ → mel spectrogram     │   │
                  │           │ → BigVGAN             │   │
                  │           │ → waveform (24kHz)    │   │
                  │           │   (1, N_samples)      │   │
                  │           └───────┬───────────────┘   │
                  │                   │                   │
          ┌───────▼───────┐   ┌───────▼───────┐          │
          │  テキスト応答  │   │  音声応答     │          │
          │  (文字列)      │   │  (WAVファイル) │          │
          └───────────────┘   └───────────────┘          │
```

---

## 主要な次元とその意味

### バッチ・シーケンス次元
- `B`: バッチサイズ
- `L_text`: テキストトークン長 (可変)
- `L_total`: 全モダリティ統合後のシーケンス長
- `L_gen`: 生成テキストトークン数
- `L_codec`: 生成音声コードトークン数

### 音声関連次元
- `T_mel`: メルスペクトログラムのフレーム数 (= 音声秒数 × 100)
- `128`: メルスペクトログラムのチャンネル数 (メルビン)
- `d_model=768`: Audio Encoder内部の隠れ次元 (Whisper-large-v3)
- `output_dim=1024`: Audio Encoder出力次元 → Thinkerに入力
- `n_window`: チャンク処理のウィンドウサイズ (2秒=50フレーム)
- `mel_dim=160`: Token2WavのDiTが生成するメルスペクトログラム次元

### 画像/動画関連次元
- `patch_size=14`: ViTのパッチサイズ (14x14ピクセル)
- `temporal_patch_size=2`: 時間方向のパッチサイズ (2フレーム)
- `spatial_merge_size=2`: PatchMergerの統合サイズ (2x2パッチ → 1トークン)
- `hidden_size=1024`: Vision Encoderの隠れ次元
- `depth=24-32`: Vision Transformerの層数
- `num_heads=16`: Vision Transformerのアテンションヘッド数
- `IMAGE_FACTOR=28`: 画像サイズが28の倍数であること (patch_size×spatial_merge_size)

### Thinker LLM 次元
- `hidden_size=4096`: LLMの隠れ次元
- `num_layers=32`: Transformerレイヤー数
- `num_heads=32`: アテンションヘッド数
- `intermediate_size=14336`: FFN中間次元
- `vocab_size=151643`: テキストボキャブラリサイズ (BPE)

### Talker 次元
- `codebook_size`: 音声コードブックサイズ (~8000-9000)
- `codec_bos=8292`, `codec_eos=8294`: 音声コード特殊トークン

### Token2Wav 次元
- `mel_dim=160`: DiT出力のメルスペクトログラム次元
- `dit_chunk_size=48`: DiTのストリーミングチャンクサイズ
- `vocoder_upsample_rate=240`: BigVGANの1メルフレームあたりのサンプル数
- `24000`: 出力音声のサンプリングレート (24kHz)

---

## 主要イノベーション

### 1. Thinker-Talker アーキテクチャ
**問題**: テキストと音声を同時にストリーミング生成することの困難さ

**解決**:
- Thinker (脳): マルチモーダル理解 + テキスト生成
- Talker (口): Thinkerの高レベル表現から音声を並行生成
- Talkerは Thinkerの隠れ状態を1トークンずつ消費しながら音声コードを生成

**効果**:
- テキストと音声の同時ストリーミング生成が可能
- 音声生成にワードレベルのアライメント不要
- Thinkerのセマンティック表現がTalkerの音声品質を向上

### 2. TMRoPE (Time-aligned Multimodal RoPE)
**問題**: 異なるモダリティ (テキスト、音声、画像、動画) の位置情報を統一的にエンコードする方法がない

**解決**:
- RoPEを3軸に分解: (temporal, height, width)
- テキスト: 3軸同一 → 標準1D-RoPEと等価
- 音声: 3軸同一、1 temporal ID = 40ms → 絶対時間エンコーディング
- 画像: temporal固定、height/widthは空間位置
- 動画+音声: 2秒チャンクでインターリーブ (視覚先、聴覚後)

**効果**: 動画と音声の時間的同期が自然に実現

### 3. ストリーミング対応設計
**問題**: 初回パケットレイテンシの最小化

**解決**:
- Audio Encoder: フルアテンション → ブロックワイズアテンション (2秒ブロック)
- Vision Encoder: Flash Attention + PatchMerger (2x2統合)
- Token2Wav DiT: スライディングウィンドウブロックアテンション (受容野4ブロック)
- BigVGAN: チャンク単位波形生成

**効果**: リアルタイム音声対話が可能

### 4. qwen-tts-tokenizer (カスタム音声コーデック)
**問題**: 音声情報を効率的に離散トークンで表現する必要性

**解決**:
- カスタム設計の効率的な音声コーデック
- 因果的デコーダでストリーミングデコード可能
- テキストとのワード/タイムスタンプアライメント不要

**効果**: 高品質な音声生成 (WER 1.42% on SEED test-zh)

---

## 性能比較

### テキスト理解 (Text-to-Text)

| ベンチマーク | Qwen2.5-Omni-7B | Qwen2.5-7B | GPT-4o-mini |
|------------|-----------------|------------|-------------|
| MMLU-Pro | 47.0 | 56.3 | 54.2 |
| MATH | 71.5 | 75.5 | 70.2 |
| HumanEval | 78.7 | 84.8 | 87.2 |
| GSM8K | 88.7 | 91.6 | 93.2 |

### 画像理解 (Image-to-Text)

| ベンチマーク | Qwen2.5-Omni-7B | Qwen2.5-VL-7B | GPT-4o-mini |
|------------|-----------------|---------------|-------------|
| MMMU_val | 59.2 | 58.6 | 60.0 |
| DocVQA_test | 95.2 | 96.0 | 90.4 |
| ChartQA_test | 85.3 | 83.6 | 78.1 |
| MME_sum | 2340 | 2335 | 2003 |

### 音声理解 (Audio-to-Text)

| ベンチマーク | Qwen2.5-Omni-7B | Qwen2-Audio | Gemini-1.5-Pro |
|------------|-----------------|-------------|----------------|
| MMAU (Avg) | 65.60 | 56.46 | 56.25 |
| VoiceBench (Avg) | 74.12 | - | 65.63 |

### 動画理解 (Video-to-Text)

| ベンチマーク | Qwen2.5-Omni-7B | Qwen2.5-VL-7B | GPT-4o-mini |
|------------|-----------------|---------------|-------------|
| Video-MME (w/o sub) | 64.3 | 65.1 | 64.8 |
| MVBench | 70.3 | 69.6 | - |

### 音声生成 (Speech Generation, SEED Benchmark)

| モデル | WER (test-zh) | WER (test-en) | Speaker Sim |
|--------|-------------|-------------|-------------|
| Qwen2.5-Omni (RL) | **1.42** | **2.33** | 0.754 |
| MaskGCT | 2.63 | 2.63 | 0.753 |
| CosyVoice 2 | 3.63 | 3.13 | 0.746 |

---

## 形状ガイド (全処理ステージ)

### 音声処理パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| 入力 | raw_audio | `(N_samples,)` | 16kHz モノラル音声 |
| メル変換 | mel_spec | `(B, 128, T_mel)` | 128ビンのメルスペクトログラム |
| Conv1 | after_conv1 | `(B, 768, T_mel)` | 次元変換 (128→768) |
| Conv2 | after_conv2 | `(B, 768, T_mel//2)` | stride=2 で2倍ダウン |
| Transformer | encoder_out | `(B, T_mel//2, 768)` | ブロックワイズアテンション後 |
| AvgPool | pooled | `(B, T_mel//4, 768)` | stride=2 でさらに2倍ダウン |
| 射影 | audio_features | `(B, T_mel//4, 1024)` | LLM入力次元へ射影 |

### 画像処理パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| 入力 | image | `(B, 3, H, W)` | RGB画像 (Hは28の倍数) |
| パッチ化 | patches | `(N_patches, hidden=1024)` | N_patches = H/14 × W/14 |
| RoPE | pos_emb | `(N_patches, head_dim)` | 2D回転位置埋め込み |
| ViTブロック | features | `(N_patches, 1024)` | ウィンドウ/フルアテンション |
| PatchMerger | merged | `(N_patches/4, out_hidden)` | 2x2統合 → 1/4トークン数 |

### 動画処理パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| 入力 | video | `(T, 3, H, W)` | Tフレーム動画 |
| パッチ化 | patches | `(T/2 × H/14 × W/14, 1024)` | temporal_patch=2 |
| ViTブロック | features | `(N_total, 1024)` | 時空間アテンション |
| PatchMerger | merged | `(N_total/4, out_hidden)` | 空間2x2統合 |

### Thinker LLM パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| テキスト入力 | input_ids | `(B, L_text)` | トークンID |
| テキスト埋め込み | text_embeds | `(B, L_text, 4096)` | ワード埋め込み |
| 統合入力 | merged_embeds | `(B, L_total, 4096)` | audio+image+video統合後 |
| TMRoPE | position_ids | `(3, B, L_total)` | [temporal, height, width] |
| Transformer出力 | hidden_states | `(B, L_total, 4096)` | 32層後 |
| LM Head | logits | `(B, L_total, 151643)` | ボキャブラリ全体の確率 |

### Talker パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| Thinker隠れ状態 | thinker_hidden | `(B, L_text, 4096)` | Thinkerから受取 |
| コード埋め込み | codec_embed | `(B, 1, hidden)` | 前ステップのcodecトークン |
| 融合 | fused | `(B, 1, hidden)` | codec_embed + thinker_hidden[:, :1] |
| 射影 | projected | `(B, 1, talker_hidden)` | thinker_to_talker_proj |
| Talker出力 | logits | `(B, 1, codebook_size)` | 次のcodecトークン予測 |

### Token2Wav パイプライン

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| コードトークン | codes | `(B, L_codec)` | 音声コード系列 |
| DiTノイズ | noise | `(1, 30000, 160)` | 初期ランダムノイズ |
| DiT出力 | mel_spec | `(B, T_mel, 160)` | メルスペクトログラム |
| BigVGAN出力 | waveform | `(1, N_samples)` | 24kHz音声波形 |

### 軸の意味凡例

```python
# 基本次元
B       # バッチサイズ
L_text  # テキストシーケンス長
L_total # 統合後の全シーケンス長
L_gen   # 生成テキスト長
L_codec # 生成音声コード長

# 音声次元
T_mel   # メルスペクトログラムのフレーム数 (= 音声秒数 × 100, 10msホップ)
128     # メルスペクトログラムビン数
768     # Audio Encoder内部隠れ次元 (d_model, Whisper-large-v3)
1024    # Audio Encoder出力次元 (output_dim)
160     # DiT生成メルスペクトログラムの次元

# 画像/動画次元
H, W        # 画像の高さ・幅 (28の倍数)
T           # 動画のフレーム数
patch_size  # 14 (ViTパッチサイズ)
1024        # Vision Encoder隠れ次元

# LLM次元
4096    # Thinker隠れ次元
32      # アテンションヘッド数
14336   # FFN中間次元
151643  # ボキャブラリサイズ (BPE)
```

---

## よくある質問

### Q1: Thinker-Talkerアーキテクチャの利点は何ですか？
A: Thinkerが「脳」として高レベルなセマンティック理解を行い、Talkerが「口」として音声を生成する分離設計により、テキストと音声の同時ストリーミング生成が可能になります。TalkerはThinkerの隠れ状態を1トークンずつ消費するため、テキスト生成が完了する前に音声生成を開始できます。

### Q2: TMRoPEはなぜ3軸に分解するのですか？
A: テキスト(1次元)、画像(2次元空間)、動画(3次元時空間)、音声(1次元時間)という異なるモダリティの位置情報を統一的に扱うためです。テキストと音声は3軸を同一にすることで通常の1D-RoPEと等価に、画像はheight/widthで2D空間を、動画はさらにtemporal軸で時間を表現します。

### Q3: Audio Encoderの4倍ダウンサンプリングとは？
A: Conv2(stride=2)で2倍、AvgPool(stride=2)で2倍の合計4倍です。元のメルスペクトログラムは10msホップなので、出力の1フレームは約40msの音声に対応します。これがTMRoPEの「1 temporal ID = 40ms」と整合します。

### Q4: PatchMergerの役割は何ですか？
A: 隣接する2x2パッチ(4トークン)を1トークンに統合することで、Vision Encoderの出力トークン数を1/4に削減します。これによりThinker LLMの計算コストが大幅に減少し、長い動画や高解像度画像の処理が実用的になります。

### Q5: 動画+音声のインターリーブはどう機能しますか？
A: 動画+音声は2秒ごとのチャンクに分割され、各チャンク内で視覚表現が前、聴覚表現が後に配置されます。これにより、モデルは視覚と聴覚の情報を同時に受け取りながら、時間的に整合した理解が可能になります。

### Q6: Low-VRAMモードとは何ですか？
A: Audio EncoderとVision EncoderをCPU上に配置し、必要な時だけGPUに転送する方式です。Prefill時にのみエンコーダをGPUに移動→処理→CPU返送→CUDAキャッシュクリアを行うことで、ピークVRAM使用量を大幅に削減できます (BF16 31GB → GPTQ-Int4 12GB)。

### Q7: Token2Wavのストリーミングはどう実現されていますか？
A: 2段階のストリーミングを実装しています。(1) DiTがcodecトークンをチャンク(48トークン)単位でメルスペクトログラムに変換。左2ブロック+現在1ブロック+右1ブロックの受容野で高品質を維持。(2) BigVGANがメルチャンクを波形チャンクに変換。両段階とも左右コンテキストを保持してシームレスな結合を実現します。

### Q8: LoRAファインチューニングでは何が学習されますか？
A: ThinkerのLLM部分のみがLoRA適用対象です。Audio Tower、Visual Tower、Talker、Token2Wavは全て凍結されます。これは、エンコーダの特徴抽出能力は十分に訓練済みであり、LLMの指示追従能力や特定タスクへの適応のみが必要なためです。

### Q9: DPOによる音声品質向上はどう機能しますか？
A: Talkerの生成音声を、WER(Word Error Rate)と句読点停止エラー率に基づいてランキングし、好ましい音声(y_w)と好ましくない音声(y_l)のペアを作成します。DPO損失によりTalkerは低WERで自然な韻律の音声を生成するよう学習します。

### Q10: Qwen2.5-OmniとQwen2.5-VLの違いは何ですか？
A: Qwen2.5-VLはテキスト+画像/動画のみを扱うのに対し、Qwen2.5-Omniはさらに音声入力と音声出力(Talker + Token2Wav)を追加した統合モデルです。Vision Encoderは同一ですが、Audio Encoder、Talker、Token2Wavが新規追加されています。TMRoPEはM-RoPEを音声の時間軸に対応するよう拡張したものです。

---

## まとめ

このリポジトリは、Qwen2.5-Omniの複雑な実装を理解するための教育的な疑似コードです。

**カバー範囲:**
1. **マルチモーダル入力処理**: audio_encoder.py, vision_encoder.py
2. **テキスト理解・生成**: thinker.py (TMRoPE + Transformer)
3. **音声生成**: talker.py, token2wav.py
4. **学習**: loss_computation.py (Pre-training 3段階 + Post-training 3段階)
5. **LoRAファインチューニング**: lora_finetune_text_image.py, lora_finetune_text_audio.py

**実際の実装に必要なもの:**
1. **transformers >= 4.52.4**: HuggingFace公式サポート
2. **flash-attn**: 高速アテンション (推奨)
3. **qwen-omni-utils**: マルチモーダル入力の前処理
4. **LLaMA-Factory / ms-swift**: LoRAファインチューニングフレームワーク

---

## 参考資料

### Qwen2.5-Omni関連
- [Qwen2.5-Omni論文](https://arxiv.org/abs/2503.20215) (arXiv:2503.20215, 2025-03-27)
- [公式GitHubリポジトリ](https://github.com/QwenLM/Qwen2.5-Omni)
- [HuggingFace モデルカード](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

### 関連技術
- **Whisper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2023)
- **Vision Transformer (ViT)**: "An Image is Worth 16x16 Words" (ICLR 2021)
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- **Flow Matching**: "Flow Matching for Generative Modeling" (Lipman et al., ICLR 2023)
- **BigVGAN**: "BigVGAN: A Universal Neural Vocoder" (Lee et al., ICLR 2023)
- **DPO**: "Direct Preference Optimization" (Rafailov et al., 2023)

### ファインチューニングフレームワーク
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): template=qwen2_omni
- [ms-swift](https://github.com/modelscope/ms-swift): Qwen2.5-Omni対応
