# Qwen3-ASR アーキテクチャ理解用ドキュメント

このリポジトリは、Qwen3-ASR (Automatic Speech Recognition) の複雑なコードベースを理解するための簡略化された疑似コードとドキュメントを提供します。

論文: [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337) (2026-01-30)
公式実装: [GitHub - QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

## 📁 ファイル構成

```
Qwen3ASR_understanding/
├── README.md                    # このファイル
├── main_flow.py                # Qwen3-ASRのメインフロー全体像 (推論パイプライン)
├── audio_encoder.py            # AuT Audio Encoder詳細
├── text_decoder.py             # Qwen3 LM Text Decoder詳細
├── forced_aligner.py           # Qwen3-ForcedAligner タイムスタンプ予測
├── streaming_inference.py      # ストリーミング推論パイプライン
└── finetuning_example.py       # ファインチューニング (Trainerなし)
```

## 🎯 各ファイルの役割

### 1. [main_flow.py](main_flow.py)
**Qwen3-ASR全体のデータフロー**

- 音声入力からテキスト出力までの完全な処理パイプライン
- 各ステージでの入出力shape、軸の意味を詳細に記載
- 5つの主要ステージ:
  1. 音声特徴抽出 (WhisperFeatureExtractor)
  2. AuT Audio Encoder (CNN + Transformer)
  3. Audio-Text Embedding統合
  4. Qwen3 LM Text Decoder (自己回帰生成)
  5. 出力パース (言語識別 + テキスト抽出)

**重要な入出力:**
- 入力: `(num_samples,)` 16kHz モノラル音声 (float32)
- 中間: `(B, T_audio//8, 3584)` Audio Encoder出力
- 出力: テキスト文字列 `"language English<asr_text>Hello world"`

---

### 2. [audio_encoder.py](audio_encoder.py)
**AuT (Audio Transformer) Encoder**

主要コンポーネント:

#### `Qwen3ASRAudioEncoder`
- 3段階のConv2dダウンサンプリング + 32層のTransformerエンコーダ
- 入力: `(B, 128, T_mel)` - メルスペクトログラム (128 mel bins, 100Hz)
- 出力: `(B, T_mel//8, 3584)` - 12.5Hzの音声表現

#### `Qwen3ASRAudioAttention`
- Multi-Head Attention (20ヘッド, d_model=1280)
- 動的Flash Attention Window (1秒〜8秒)
- ストリーミング/オフライン統一推論を実現

#### `SinusoidsPositionEmbedding`
- 正弦波ベースの位置エンコーディング
- 出力: `(1, T, d_model)` - チャンクごとの位置情報

**処理フロー:**
```
Mel Spectrogram (B, 128, T_mel) @ 100Hz
  → Conv2d ×3 (stride=2 each, 1→480 channels)
  → (B, 480, 16, T_mel//8) → reshape → (B, T_mel//8, 7680)
  → Linear → (B, T_mel//8, 1280)
  → + Positional Embedding
  → 32× Transformer Encoder Layer (Self-Attention + FFN)
  → proj1: Linear(1280→1280) + GELU
  → proj2: Linear(1280→3584)
  → 出力: (B, T_mel//8, 3584) @ 12.5Hz
```

---

### 3. [text_decoder.py](text_decoder.py)
**Qwen3 LM Text Decoder**

主要コンポーネント:

#### `Qwen3ASRThinkerForConditionalGeneration`
- Audio EncoderとQwen3 LMを統合する最上位モデル
- Audio特徴をテキスト埋め込み空間にマッピング
- 入力: `input_ids (B, T_text)` + `input_features (B, T_feat, 3584)`
- 出力: `logits (B, T_combined, 151936)`

#### `Qwen3ASRThinkerTextModel`
- 32層のTransformer Decoder (causal attention)
- Multi-axis RoPE (MRoPE) による3D位置エンコーディング
- SwiGLU活性化関数のMLP

#### `Qwen3ASRTextAttention`
- 32ヘッド, head_dim=128
- RoPE (Rotary Position Embedding) 適用
- Query/Key正規化 (RMSNorm)

**処理フロー:**
```
Token IDs (B, T_text)
  → Token Embedding (B, T_text, 4096)
  → Audio特徴を<audio>トークン位置に置換
  → MRoPE Position IDs生成 (3, B, T_combined, 128)
  → 32× Decoder Layer:
     ├─ RMSNorm → Causal Self-Attention (RoPE) → Residual
     └─ RMSNorm → SwiGLU MLP (4096→22016→4096) → Residual
  → Final RMSNorm
  → LM Head: Linear(4096→151936)
  → logits (B, T_combined, 151936)
```

---

### 4. [forced_aligner.py](forced_aligner.py)
**Qwen3-ForcedAligner-0.6B タイムスタンプ予測**

主要コンポーネント:

#### `Qwen3ForcedAligner`
- 非自己回帰 (NAR) のタイムスタンプ予測モデル
- 音声とテキストのペアから単語/文字レベルのタイムスタンプを推定
- slot-filling形式: テキスト中の `[time]` トークンにタイムスタンプインデックスを予測
- 11言語対応、最大300秒の音声

#### タイムスタンプ変換
- AuT Encoder出力のフレームレート: 80ms/フレーム
- タイムスタンプインデックス × 80ms = 実際のタイムスタンプ
- 最大クラス数: 3,750 (= 300s / 80ms)

**処理フロー:**
```
音声 (B, num_samples) + テキスト "Hello world"
  ↓
テキスト整形: "Hello [time][time] world [time][time]"
  → Tokenize → (B, T_text)
  ↓
Audio Encoder → (B, T_audio//8, 3584)
  ↓
Qwen3-0.6B LM → hidden_states (B, T_combined, hidden_size)
  ↓
Timestamp Prediction Layer → (B, T_combined, 3750)
  ↓
[time]トークン位置のみ抽出 → argmax → タイムスタンプインデックス
  ↓
× 80ms → [(text, start_sec, end_sec), ...]
```

---

### 5. [streaming_inference.py](streaming_inference.py)
**ストリーミング推論パイプライン**

主要コンポーネント:

#### `ASRStreamingState`
- チャンクベースのストリーミング状態管理
- 累積音声バッファ + ロールバック戦略
- unfixed_chunk_num: 固定されない末尾チャンク数 (デフォルト2)

#### ストリーミングフロー
```
Audio Stream → 2秒チャンクにバッファリング
  ↓
チャンク蓄積 (audio_accum: 累積的に増加)
  ↓
各チャンクで:
  ├─ chunk_id < unfixed_chunk_num: prefix = ""
  └─ else: 末尾Kトークンをロールバック
  ↓
model.generate(audio_accum + prefix)
  ↓
state.text 更新 (部分認識結果)
  ↓
finish_streaming_transcribe() → 最終結果
```

---

### 6. [finetuning_example.py](finetuning_example.py)
**ファインチューニング (Trainerなし)**

主要コンポーネント:

#### カスタム訓練ループ
- transformers Trainerを使わない手動訓練
- 音声-テキストペアデータセットからASRモデルをファインチューニング
- Prefix (system prompt + audio) は損失計算から除外 (labels=-100)

#### データ準備
```
JSONL形式:
{"audio": "/path/to/audio.wav", "text": "transcript text", "prompt": "optional context"}

→ Processor: audio → mel features, text → token IDs
→ Labels: prefix部分=-100, target部分=token IDs
```

---

## 🔍 Qwen3-ASRの全体アーキテクチャ

### データフロー図

```
┌─────────────────────────────────────────────────────────────┐
│                      入力データ                              │
│  ・音声 (num_samples,) @ 16kHz モノラル                      │
│  ・(オプション) コンテキスト: エンティティリスト等             │
│  ・(オプション) 言語指定: "Chinese", "English" 等             │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐              ┌───────▼────────┐
│ Feature        │              │ Chat Template  │
│ Extractor      │              │ + Tokenizer    │
│                │              │                │
│ WhisperFE      │              │ Qwen2Tokenizer │
│ FFT=400        │              │ vocab=151936   │
│ hop=160        │              │                │
│ 128 mel bins   │              │ System prompt  │
│                │              │ + <audio> token│
└───────┬────────┘              └───────┬────────┘
        │                               │
        │ Mel Spectrogram               │ Token IDs
        │ (B, 128, T_mel)              │ (B, T_text)
        │                               │
┌───────▼────────┐                      │
│ AuT Audio      │                      │
│ Encoder        │                      │
│                │                      │
│ Conv2d ×3      │                      │
│ (stride=2 each)│                      │
│ 32× Transformer│                      │
│ Encoder Layers │                      │
│                │                      │
│ 100Hz→12.5Hz   │                      │
│ 8倍ダウン       │                      │
└───────┬────────┘                      │
        │                               │
        │ Audio Features                │
        │ (B, T_audio//8, 3584)        │
        │                               │
        └───────────────┬───────────────┘
                        │
               ┌────────▼────────┐
               │ Embedding       │
               │ 統合             │
               │                 │
               │ <audio>トークン  │
               │ 位置にAudio特徴  │
               │ をmasked scatter│
               │                 │
               │ MRoPE Position  │
               │ IDs生成          │
               └────────┬────────┘
                        │
                        │ Combined Embeddings
                        │ (B, T_combined, 4096)
                        │
               ┌────────▼────────┐
               │ Qwen3 LM       │
               │ Text Decoder   │
               │                 │
               │ 32 Layers:     │
               │ ・Causal Attn  │
               │   (RoPE, 32h)  │
               │ ・SwiGLU MLP   │
               │   (22016 dim)  │
               └────────┬────────┘
                        │
                        │ Hidden States
                        │ (B, T_combined, 4096)
                        │
               ┌────────▼────────┐
               │ LM Head        │
               │ Linear(→151936) │
               └────────┬────────┘
                        │
                        │ Logits
                        │ (B, T_combined, 151936)
                        │
               ┌────────▼────────┐
               │ Auto-regressive │
               │ Decoding        │
               │                 │
               │ Greedy / Beam   │
               └────────┬────────┘
                        │
               ┌────────▼────────────────┐
               │ 出力パース               │
               │                         │
               │ "language English       │
               │  <asr_text>Hello world" │
               │                         │
               │ → language: "English"   │
               │ → text: "Hello world"   │
               └─────────────────────────┘
```

---

## 📊 主要な次元とその意味

### バッチ次元
- `B`: バッチサイズ (通常1-8程度)

### 時間次元
- `num_samples`: 生の音声サンプル数 (16kHz × 秒数)
- `T_mel`: メルスペクトログラムのフレーム数 (≈ num_samples / 160)
- `T_audio`: Audio Encoder出力のフレーム数 (= T_mel // 8, 12.5Hz)
- `T_text`: テキストトークン数 (プロンプト + <audio>プレースホルダー)
- `T_combined`: Audio特徴置換後の全トークン数 (= T_text - 1 + T_audio)

### 特徴次元
- `128`: メル周波数ビン数
- `480`: Conv2dダウンサンプリング中間チャネル数
- `1280`: AuT Encoder内部次元 (d_model)
- `3584`: AuT Encoder出力次元 (→Qwen3 LMへの入力)
- `4096`: Qwen3 LM隠れ次元 (hidden_size)
- `22016`: SwiGLU MLP中間次元
- `151936`: 語彙サイズ (vocab_size)

### Attention次元
- AuT Encoder: 20ヘッド × 64次元 = 1280
- Qwen3 LM: 32ヘッド × 128次元 = 4096

### レイヤー次元
- AuT Encoder: 32層 (Self-Attention + FFN)
- Qwen3 LM: 32層 (Causal Self-Attention + SwiGLU MLP)

---

## 🧩 重要な処理とテクニック

### 1. AuT (Audio Transformer) プリトレーニング
- **AED (Attention-Encoder-Decoder)** フレームワークで事前学習
- 約4,000万時間の擬似ラベル付きASRデータを使用
- 8倍ダウンサンプリングで100Hz → 12.5Hzトークンレート
- 動的Flashアテンションウィンドウ (1秒〜8秒)

### 2. 4段階学習パイプライン
1. **AuTプリトレーニング**: 大規模ラベル付きデータでEncoder学習
2. **Omniプリトレーニング**: Qwen3-Omni基盤モデル (3兆トークン)
3. **ASR SFT**: スタイル転移 + 多言語 + コンテキストバイアス
4. **ASR RL (GSPO)**: ノイズ耐性・安定性向上 (5万発話)

### 3. 出力フォーマット
```
認識結果あり:
<|im_start|>assistant
language English<asr_text>Today we release Qwen3-ASR.<|im_end|>

認識結果なし:
<|im_start|>assistant
language None<asr_text><|im_end|>
```

### 4. コンテキストバイアス (Context Biasing)
- System promptにエンティティリストを指定可能
- 固有名詞・専門用語の認識精度向上
- 例: `"Entities: Qwen, Qwen-Omni, Tongyi Lab"`

### 5. Multi-axis RoPE (MRoPE)
- 3次元の位置エンコーディング: Temporal (24次元), Height (20次元), Width (20次元)
- 音声トークンとテキストトークンで異なる位置ID割り当て
- インターリーブ配置: [T, H, W, T, H, W, ...]

### 6. 動的アテンションウィンドウ
- ストリーミング推論: 短いチャンクで1秒ウィンドウ
- オフライン推論: 長いクエリで8秒ウィンドウ
- 同一モデルでストリーミング/オフライン統一推論

### 7. 音声チャンキング
- 最大入力長: 1,200秒 (20分)
- 長い音声は低エネルギー境界で分割
- スライディングウィンドウでエネルギー検出
- 最小チャンク: 0.5秒 (パディング)

### 8. ForcedAligner: Slot-Filling方式
- テキスト中に `[time]` トークンを挿入
- 非自己回帰 (NAR) で全スロットを同時予測
- AuT Encoder出力フレームレート: 80ms
- タイムスタンプインデックス × 80ms = 実際のタイムスタンプ
- LIS (最長増加部分列) ベースの単調性修正

---

## 📊 モデルバリアント比較

| パラメータ | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B | Qwen3-ForcedAligner-0.6B |
|-----------|----------------|----------------|-------------------------|
| LLM | Qwen3-1.7B | Qwen3-0.6B | Qwen3-0.6B |
| AuT Encoder | 300M (d=1024) | 180M (d=896) | 180M (d=896) |
| 言語数 | 52 (30言語+22方言) | 52 (30言語+22方言) | 11言語 |
| 最大入力長 | 1,200秒 | 1,200秒 | 300秒 |
| 推論方式 | オフライン/ストリーミング | オフライン/ストリーミング | NAR (非自己回帰) |
| TTFT (Conc=1) | 102ms | 92ms | - |
| RTF (Conc=128) | 0.105 | 0.064 | 0.001 |
| スループット (Conc=128) | 1,220 sec/s | 2,000 sec/s | 649 sec/s |

---

## 📊 ベンチマーク結果 (代表値)

### 英語 (WER %)
| ベンチマーク | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-1.7B |
|-------------|-------------------|------------------|----------------|
| LibriSpeech clean\|other | 1.39\|3.75 | 1.51\|3.97 | 1.63\|3.38 |
| GigaSpeech | 25.50 | 9.76 | 8.45 |
| CV-en | 9.08 | 9.90 | 7.39 |

### 中国語 (CER %)
| ベンチマーク | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-1.7B |
|-------------|-------------------|------------------|----------------|
| WenetSpeech net\|meeting | 15.30\|32.27 | 9.86\|19.11 | 4.97\|5.88 |
| AISHELL-2-test | 4.24 | 5.06 | 2.71 |
| Fleurs-zh | 2.44 | 4.09 | 2.41 |

---

## 🤔 よくある質問

### Q1: AuT Encoderのトークンレートが12.5Hzとはどういう意味か?
**A**: 1秒あたり12.5フレームの音声表現を出力するということです。100Hz (10ms/frame) のメルスペクトログラムに対して3段階のstride=2 Conv2dで8倍ダウンサンプリングすることで実現しています。つまり80ms/frameの解像度になります。

### Q2: なぜAuT EncoderとQwen3 LMの次元が異なるのか?
**A**: AuT Encoderは内部次元1280 (1.7Bモデルの場合) で動作しますが、出力時にproj2で3584次元に射影します。これはQwen3 LMの入力次元4096とは異なりますが、Qwen3 LMのtoken embeddingレイヤーが`<audio>`トークン位置にAudio特徴を直接scatter (3584→4096のLinear projectionはなく、embed_tokensの重みで変換)することで統合されます。

### Q3: コンテキストバイアスはどのように機能するか?
**A**: System promptにエンティティリスト (`"Entities: Qwen, Tongyi Lab, ..."`) を含めることで、モデルはこれらを背景知識として利用します。LLMベースのため、自然言語プロンプトとしてコンテキストを理解し、固有名詞の認識精度を向上させます。

### Q4: ストリーミング推論のロールバック戦略とは?
**A**: ストリーミングでは最新の数チャンクの認識結果は不安定です。unfixed_chunk_num (デフォルト2) で指定された数の末尾チャンクの認識結果をロールバック (巻き戻し) し、新しいチャンクを追加して再認識します。これにより認識結果の安定性を確保します。

### Q5: ASR RL (GSPO) とは何か?
**A**: Group Sequence Policy Optimization の略で、強化学習を用いてASRの品質を向上させます。RL段階では約5万発話を使用し、ノイズ耐性、転写安定性、難しいケースの分析能力を改善します。従来のSFTだけでは達成できない品質向上を実現します。

### Q6: ForcedAlignerが非自己回帰 (NAR) である利点は?
**A**: 全タイムスタンプスロットを同時に予測するため、自己回帰モデルと比べて推論速度が大幅に高速です。RTF 0.001 (=1秒で1,000秒の音声を処理可能) を実現しています。また、因果学習 (causal training) により先行コンテキストを考慮でき、グローバルな一貫性を保てます。

### Q7: MRoPE (Multi-axis RoPE) はなぜ必要か?
**A**: 音声トークンとテキストトークンでは位置の概念が異なります。MRoPEは3つの軸 (Temporal, Height, Width) で独立した位置エンコーディングを適用することで、マルチモーダルな入力に対して適切な位置情報を付与します。これはQwen3-Omniの基盤モデルから引き継がれた設計です。

### Q8: なぜ出力に言語識別 (LID) が含まれるのか?
**A**: Qwen3-ASRは認識テキストの前に言語を出力する形式 (`"language English<asr_text>..."`) です。これにより、言語識別と音声認識を単一モデルで同時に行え、52言語/方言をサポートします。言語を強制指定することも可能です。

### Q9: 歌声認識はなぜ難しいのか?
**A**: 歌声には (1) ピッチドリフト, (2) 音素の引き延ばし, (3) リズミカルな歌詞変化, (4) BGM (背景音楽) との混合があり、通常の音声認識モデルでは困難です。Qwen3-ASRはLALMパラダイムにより音声の高レベルな理解を行い、さらに歌声認識データでのSFTにより対応しています。

### Q10: Qwen3-ASR-0.6BとQwen3-ASR-1.7Bの使い分けは?
**A**: 0.6Bはオンデバイスデプロイメントや低レイテンシが求められるシナリオに最適で、TTFT 92ms、スループット 2,000 sec/sを達成します。1.7Bはより高い認識精度が必要な場合に推奨され、特に多言語・方言・ノイズ環境で安定した性能向上を示します。

---

## 📚 参考資料

### Qwen3-ASR関連
- Qwen3-ASR Technical Report (arXiv:2601.21337, 2026)
- Qwen3-Omni Technical Report (arXiv:2509.17765, 2025)
- LLM-ForcedAligner (arXiv:2601.18220, 2026)

### 関連技術
- **Whisper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (ICML 2023)
- **AED**: "Listen, Attend and Spell" (ICASSP 2016)
- **Transducer**: "Sequence Transduction with Recurrent Neural Networks" (2012)
- **RoPE**: "Rotary Position Embedding" (2021)
- **GSPO**: "Group Sequence Policy Optimization" (2025)

### 実装ライブラリ
- **transformers**: HuggingFace Transformers (モデル定義・推論)
- **vLLM**: 高速LLM推論エンジン (ストリーミング対応)
- **librosa**: 音声処理ライブラリ
- **WhisperFeatureExtractor**: メルスペクトログラム抽出

---

## ✨ まとめ

このリポジトリは、Qwen3-ASRの複雑な実装を理解するための教育的な疑似コードです。

**カバー範囲:**
1. **バッチ推論**: main_flow.py - 完全な推論パイプライン
2. **音声エンコーダ**: audio_encoder.py - AuT CNN + Transformer
3. **テキストデコーダ**: text_decoder.py - Qwen3 LM統合
4. **タイムスタンプ予測**: forced_aligner.py - NAR slot-filling
5. **ストリーミング**: streaming_inference.py - チャンクベース推論
6. **ファインチューニング**: finetuning_example.py - Trainerなし手動訓練

**実際の利用に必要なもの:**
1. **モデル重み**: HuggingFace `Qwen/Qwen3-ASR-1.7B` or `Qwen/Qwen3-ASR-0.6B`
2. **GPU**: bfloat16/float16推論にCUDA対応GPU推奨
3. **vLLM**: ストリーミング推論・高スループット推論に必要
4. **音声データ**: 16kHz モノラル PCM (自動変換対応)
