# Qwen3-Omni アーキテクチャ理解用ドキュメント

このリポジトリは、Qwen3-Omni の複雑なコードベースを理解するための簡略化された疑似コードとドキュメントを提供します。

## ファイル構成

```
Qwen3Omni_understanding/
├── README.md                          # このファイル
├── main_flow.py                       # 全体のデータフロー (入力 → Thinker → Talker → 音声出力)
├── audio_encoder.py                   # AuT (Audio Transformer) 詳細
├── vision_encoder.py                  # SigLIP2-So400m Vision Encoder 詳細
├── thinker.py                         # MoE Thinker LLM (マルチモーダル統合 + テキスト生成)
├── talker.py                          # MoE Talker + MTP (テキスト表現 → 音声コードトークン生成)
├── code2wav.py                        # Code2Wav (ConvNet による音声波形生成)
├── loss_computation.py                # 学習時のロス計算 (Pre-training + Post-training)
├── lora_finetune_text_image.py        # LoRA ファインチューニング サンプル (Text-Image)
└── lora_finetune_text_audio.py        # LoRA ファインチューニング サンプル (Text-Audio)
```

## Qwen2.5-Omni との主要な差分

| 項目 | Qwen2.5-Omni | Qwen3-Omni |
|------|-------------|------------|
| **Audio Encoder** | Whisper-large-v3 (~640M) | AuT (Audio Transformer) (~650M), 20Mh学習データでスクラッチ学習 |
| **Vision Encoder** | ViT (~675M) | SigLIP2-So400m (~540M) |
| **Thinker** | Dense Transformer 7B | **MoE** Transformer 30B-A3B (30B総パラメータ, 3Bアクティブ) |
| **Talker** | Dense Transformer | **MoE** Transformer 3B-A0.3B |
| **コーデック方式** | シングルコードブック | **マルチコードブック** + MTP Module (80M) |
| **音声合成** | DiT (Flow-Matching) + BigVGAN | **Code2Wav** (軽量 Causal ConvNet, 200M) |
| **位置エンコーディング** | M-RoPE (2秒チャンク固定) | **TM-RoPE** (絶対時間アンカー, 修正角度割当) |
| **音声入力上限** | 制限あり | **40分以上**対応 |
| **言語** | 限定的 | テキスト119言語, 音声入力19言語, 音声出力10言語 |
| **First-Packet Latency** | 高め | **234ms** (1並行, Audio/Video) |
| **HF クラス名** | `Qwen2_5OmniForConditionalGeneration` | `Qwen3OmniMoeForConditionalGeneration` |
| **transformers バージョン** | >=4.52.4 | **>=4.57.3** |

## 各ファイルの役割

### 1. [main_flow.py](main_flow.py)
**Qwen3-Omni全体のデータフロー**

- テキスト/画像/動画/音声入力からテキスト+音声出力までの完全な処理パイプライン
- 各ステージでの入出力shape、軸の意味を詳細に記載
- 6つの主要ステージ:
  1. 入力前処理 (テキストトークン化、メルスペクトログラム、画像パッチ化)
  2. エンコーダ (AuT Audio Encoder、SigLIP2 Vision Encoder)
  3. MoE Thinker LLM (マルチモーダル特徴統合 + テキスト生成)
  4. MoE Talker (テキスト表現 → マルチコードブック音声コード)
  5. MTP Module (残余コードブック予測)
  6. Code2Wav (コードトークン → 音声波形)

**重要な入出力:**
- 入力: テキスト `(B, L_text)`, 音声 `(B, 128, T_mel)`, 画像 `(N_patches, C_hidden)`, 動画 `(T*H*W, C_hidden)`
- 出力: テキスト `(B, L_gen, 151643)`, 音声波形 `(1, N_samples)` (24kHz)

---

### 2. [audio_encoder.py](audio_encoder.py)
**AuT (Audio Transformer) 音声エンコーダ**

主要コンポーネント:

#### `AuTEncoder`
- 20M時間の教師あり音声データでスクラッチ学習
- 16kHz音声 → 128チャネルメルスペクトログラム → 特徴ベクトル系列
- 入力: `(batch, 128, T_mel)` - メルスペクトログラム (10ms hop)
- 出力: `(batch, T_tokens, output_dim)` - 12.5Hz (80ms/token)
- 3× Conv2D ダウンサンプリング (8倍) + 32層 Self-Attention

#### `AuTDecoder`
- 8層 デコーダ (Cross-Attention + Self-Attention)
- ASR・音声理解の両タスクで学習

**処理フロー:**
```
メルスペクトログラム (B, 128, T_mel)   ← 10ms frame shift
  → 3× Downsampling Conv2D (8倍ダウン)
  → 32層 Self-Attention Encoder (flash attention, 動的ウィンドウ 1-8秒)
  → AuT Hidden: 12.5Hz トークン列
  → [オプション] 8層 Decoder (Cross-Attn + Self-Attn)
  → 出力: (B, T_tokens, output_dim)  ← 12.5Hz = 80ms/token
```

**Qwen2.5-Omniとの差分:**
- Whisper → AuT (スクラッチ学習, encoder-decoder)
- ダウンサンプリング: Conv1D stride=2×2 (4倍) → Conv2D×3 (8倍)
- ブロックアテンション: 固定2秒 → 動的1-8秒ウィンドウ
- トークンレート: 25Hz (40ms) → 12.5Hz (80ms)

---

### 3. [vision_encoder.py](vision_encoder.py)
**SigLIP2-So400m 画像/動画エンコーダ**

主要コンポーネント:

#### `VisionEncoder`
- SigLIP2-So400m から初期化 (~540M パラメータ)
- Qwen3-VL と共通のビジョンエンコーダ
- 画像・動画の両方に対応
- PatchMerger: 隣接パッチを統合してトークン数を削減

**処理フロー:**
```
画像/動画フレーム
  → パッチ分割 (patch_size=14, temporal_patch_size=2)
  → パッチ埋め込み → SigLIP2 Transformer
  → PatchMerger (spatial_merge_size=2: 4パッチ→1トークン)
  → 出力: (N_merged, hidden_dim)
```

**Qwen2.5-Omniとの差分:**
- ViT ~675M → SigLIP2-So400m ~540M
- Qwen2.5-VL → Qwen3-VL ベース

---

### 4. [thinker.py](thinker.py)
**MoE Thinker LLM**

主要コンポーネント:

#### `ThinkerMoEForConditionalGeneration`
- Qwen3 MoE ベース (30B-A3B: 30B総パラメータ, 3Bアクティブ)
- Mixture-of-Experts: 高並行性・高速推論
- TM-RoPE (Time-aligned Multimodal RoPE) による位置エンコーディング
- 入力: テキスト + 音声特徴 + 画像/動画特徴
- 出力: テキストトークン + 中間層隠れ状態 (Talkerへ)

#### `TM-RoPE (Time-aligned Multimodal Rotary Position Embedding)`
- M-RoPE を拡張し、絶対時間エンコーディングを導入
- 3次元: temporal, height, width
- 角度割当: temporal=24, height=20, width=20 (head_dim=64の場合)
- テキスト: 3軸同一 → 標準1D-RoPEと等価
- 音声: 共有位置ID + 80ms単位の絶対時間ID
- 画像: temporal固定 + height/width空間座標
- 動画: 80ms単位で単調増加するtemporal ID + height/width

**Qwen2.5-Omniとの差分:**
- Dense 7B → MoE 30B-A3B
- M-RoPE → TM-RoPE (修正角度割当: 16,24,24 → 24,20,20)
- 2秒チャンク固定 → 絶対時間アンカー
- コンテキスト長: 8192 → 32768 (S3)

---

### 5. [talker.py](talker.py)
**MoE Talker + MTP Module**

主要コンポーネント:

#### `TalkerMoEForConditionalGeneration`
- MoE Transformer (3B-A0.3B)
- Thinkerの中間層隠れ状態 + マルチモーダル特徴を直接受け取る
- マルチコードブック自己回帰生成: 1ステップで1コードブックフレーム
- RVQ (Residual Vector Quantization) トークンを直接操作

#### `MTPModule` (Multi-Token Prediction)
- 超軽量 固定ステップ自己回帰 Dense Transformer (80M)
- 残余コードブック層の予測
- 低メモリ帯域幅・高スループット

**入力:**
- Thinkerからの高次元マルチモーダル特徴 (中間層から抽出)
- 過去のテキストトークン
- 現在のストリーミングテキスト

**出力:**
- マルチコードブック離散音声トークン (RVQフレーム)

**処理フロー:**
```
Thinker中間層隠れ状態 + テキストトークン + マルチモーダル特徴
  → MoE Talker Backbone
    → 第0コードブック予測 (Linear Head)
  → MTP Module
    → 残余コードブック予測 (固定ステップ自己回帰)
  → マルチコードブックフレーム出力
  → Code2Wavへ
```

**Qwen2.5-Omniとの差分:**
- Dense Talker → MoE Talker (3B-A0.3B)
- シングルコードブック → マルチコードブック + MTP
- Thinkerのテキスト隠れ状態のみ → マルチモーダル特徴も直接受信
- ブロックコンテキスト待ち必要 → 最初のトークンで即座に波形出力可能

---

### 6. [code2wav.py](code2wav.py)
**Code2Wav (ConvNet Vocoder)**

主要コンポーネント:

#### `Code2WavModel`
- 軽量 Causal ConvNet (200M パラメータ)
- マルチコードブック → 波形変換
- ストリーミング対応 (80msフレーム単位)
- 出力: 24kHz 音声波形

**処理フロー:**
```
マルチコードブック RVQトークン (12.5Hz)
  → Code2Wav ConvNet (causal, ストリーミング)
  → 音声波形 (24kHz)
```

**Qwen2.5-Omniとの差分:**
- DiT (Flow-Matching, 拡散モデル) + BigVGAN → 軽量 Causal ConvNet
- 大幅なFLOP削減、ハードウェアアクセラレーション対応
- ブロック単位処理 → フレーム単位ストリーミング (最初のコードフレームから即座に波形生成)

---

### 7. [loss_computation.py](loss_computation.py)
**学習時のロス計算**

#### Pre-training ロス
- **S1 (Encoder Alignment)**: LLM凍結、エンコーダ+アダプタのみ学習
- **S2 (General)**: 全パラメータ学習、~2Tトークン
  - text 0.57T, audio 0.77T, image 0.82T, video 0.05T, video-audio 0.05T
- **S3 (Long Context)**: コンテキスト長 8192 → 32768

#### Post-training ロス (Thinker)
1. **SFT Loss**: ChatML形式の教師あり学習
2. **Strong-to-Weak Distillation**:
   - Off-policy: 教師モデル (Qwen3-32B / Qwen3-235B-A22B) からの応答蒸留
   - On-policy: 生徒モデル生成 → KLダイバージェンス最小化
3. **GSPO (Graduate-level Self-Preference Optimization)**:
   - Rule-based Reward (数学, コーディング, 指示追従)
   - Model-based Reward (LLM-as-judge)

#### Post-training ロス (Talker, 4段階)
1. **Stage 1**: マルチモーダルコンテキスト付き音声データで学習
2. **Stage 2 (CPT)**: 高品質データで継続事前学習 + 長コンテキスト学習
3. **Stage 3 (DPO)**: 多言語話者選好ペアで Direct Preference Optimization
4. **Stage 4 (Speaker FT)**: 話者ファインチューニング

---

### 8. [lora_finetune_text_image.py](lora_finetune_text_image.py)
**LoRA ファインチューニング (Text-Image)**

- Thinker MoE LLM に LoRA 適用 (MoE の各エキスパートにも適用可能)
- Audio Encoder, Vision Encoder, Talker, MTP, Code2Wav は凍結
- pandas DataFrame 形式でデータセットを受け取る
- 実データがある想定の動作するスクリプト

---

### 9. [lora_finetune_text_audio.py](lora_finetune_text_audio.py)
**LoRA ファインチューニング (Text-Audio)**

- Thinker MoE LLM に LoRA 適用
- 音声理解タスク向け (ASR, 音声分類, 音声QA, 音楽理解等)
- pandas DataFrame 形式でデータセットを受け取る
- 実データがある想定の動作するスクリプト

---

## アーキテクチャ概要図

```
テキスト入力 ────────────────────┐
                                │
音声入力 (16kHz) ──→ [128-ch mel] ──→ [AuT Encoder (650M)]      │
  3× Conv2D (8倍ダウン)                                          │
  32× Self-Attention                                             │
  12.5Hz トークン ──────────────────────────────────────────────→ │
                                                                  │
画像/動画 ──→ [SigLIP2-So400m (540M)]                            │
  パッチ化 + ViT + PatchMerger ──────────────────────────────→ │
                                                                  ↓
                    ┌──────────────────────────────────────────────┐
                    │        MoE Thinker (30B-A3B)                 │
                    │  TM-RoPE位置エンコーディング                    │
                    │  MoE Transformer Layers                      │
                    │  → テキストトークン生成                        │
                    │  → 中間層隠れ状態 (Talkerへ)                  │
                    └──────────────┬───────────────────────────────┘
                                   │
           テキスト出力 ←──────────┤
                                   │ 中間層隠れ状態 + マルチモーダル特徴
                                   ↓
                    ┌──────────────────────────────────────────────┐
                    │        MoE Talker (3B-A0.3B)                 │
                    │  マルチコードブック自己回帰生成                  │
                    │  → 第0コードブック (Linear Head)               │
                    │  → MTP Module (80M): 残余コードブック          │
                    └──────────────┬───────────────────────────────┘
                                   │ マルチコードブック RVQトークン
                                   ↓
                    ┌──────────────────────────────────────────────┐
                    │        Code2Wav (200M)                        │
                    │  軽量 Causal ConvNet                          │
                    │  ストリーミング 80ms フレーム単位               │
                    └──────────────┬───────────────────────────────┘
                                   │
                                   ↓
                    音声出力 (24kHz)
```

## モデルサイズ一覧 (Qwen3-Omni-30B-A3B)

| コンポーネント | アーキテクチャ | パラメータ数 | ストリーミング |
|-------------|------------|-----------|------------|
| Audio Encoder | AuT | 650M | ✓ |
| Vision Encoder | SigLIP2-So400m | 540M | - |
| Thinker | MoE Transformer | 30B-A3B | ✓ |
| Talker | MoE Transformer | 3B-A0.3B | ✓ |
| MTP Module | Dense Transformer | 80M | ✓ |
| Code2Wav | ConvNet | 200M | ✓ |

## レイテンシ (Theoretical First-Packet, Audio/Video)

| 項目 | 1並行 | 4並行 | 6並行 |
|------|-------|-------|-------|
| Thinker-Talker前処理 | 72/160ms | 94/180ms | 100/200ms |
| Thinker TTFT | 88/160ms | 468/866ms | 673/1330ms |
| Talker TTFT | 57/210ms | 145/450ms | 376/734ms |
| MTP コスト/トークン | 14ms | 16ms | 18ms |
| Code2Wav コスト/コード | 3ms | 5ms | 5ms |
| **合計** | **234/547ms** | **728/1517ms** | **1172/2284ms** |

## 話者タイプ

| 話者名 | 性別 | 説明 |
|--------|------|------|
| Ethan | 男性 | 明るく元気で、感染力のあるエネルギーと温かさを持つ声 |
| Chelsie | 女性 | 蜜のように滑らかで、柔らかな温かさと透明感のある声 |
| Aiden | 男性 | 温かくリラックスしたアメリカンな声、優しい少年のような魅力 |

## 言語サポート

| モダリティ | 言語数 | 言語 |
|-----------|-------|------|
| テキスト | 119 | Qwen3 と同一 |
| 音声入力 | 19 | ar, de, en, es, fr, id, it, ja, ko, ms, nl, pl, ru, th, tr, ur, vi, yue, zh |
| 音声出力 | 10 | de, en, es, fr, it, ja, ko, pt, ru, zh |

## 公式リソース

- 論文: [Qwen3-Omni Technical Report](https://arxiv.org/pdf/2509.17765)
- GitHub: https://github.com/QwenLM/Qwen3-Omni
- HuggingFace: https://huggingface.co/collections/Qwen/qwen3-omni-68d100a86cd0906843ceccbe
- 必要パッケージ: `transformers>=4.57.3`, `qwen-omni-utils`, `flash-attn`
- HF クラス: `Qwen3OmniMoeForConditionalGeneration`, `Qwen3OmniMoeProcessor`
