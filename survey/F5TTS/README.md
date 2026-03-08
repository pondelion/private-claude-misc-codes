# F5-TTS Understanding - 簡略化疑似コード集

F5-TTS (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching) の理解を目的とした簡略化疑似コード集です。

論文: [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)
公式実装: [https://github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)

## 目次

- [概要](#概要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [F5-TTSの主要イノベーション](#f5-ttsの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [形状ガイド](#形状ガイド)
- [FAQ](#faq)

---

## 概要

**F5-TTSの特徴:**
- **完全非自己回帰**: Flow Matching + Diffusion Transformer (DiT) ベース
- **シンプルなパイプライン**: 音素アラインメント、Duration Predictor、テキストエンコーダ不要
- **高速収束**: ConvNeXt V2によるテキスト前処理で学習収束を大幅加速
- **Sway Sampling**: 推論時のフローステップサンプリング戦略で性能・効率向上

**タスク:**
- Zero-shot Text-to-Speech (メイン)
- Speech Editing (マスク領域の再生成)
- Code-switching (多言語混在音声合成)

**性能:**
- LibriSpeech-PC test-clean: WER 2.42%, SIM-o 0.66, RTF 0.15 (32 NFE)
- Seed-TTS test-en: WER 1.83%, SIM-o 0.67
- Seed-TTS test-zh: WER 1.56%, SIM-o 0.76

---

## アーキテクチャ全体像

### 学習時 (Text-guided Speech Infilling)

```
音声サンプル x (waveform)
    |
    v
┌──────────────────────────────────────┐
│ MelSpec: Mel-Spectrogram抽出         │  → x1: (B, N, F)
│  24kHz, hop=256, n_mel=100           │     F=100 (mel次元), N=時間フレーム数
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ ランダムマスク生成                    │  → m: (B, N)
│  70%-100%の連続スパンをマスク         │     マスク=生成対象, 非マスク=条件
└──────────────────────────────────────┘
    |
    |  マスク音声: cond = (1-m) ⊙ x1    (B, N, F)  非マスク部分を条件として保持
    |  ノイズ混合: φ_t = (1-t)*x0 + t*x1 (B, N, F)  x0~N(0,I), t~U[0,1]
    |  フロー目標: flow = x1 - x0        (B, N, F)
    |
    v
┌──────────────────────────────────────┐
│ テキスト処理                          │
│  文字列 y → 文字トークン列 z          │  → z: (B, N)
│  フィラートークン ⟨F⟩ でmel長にパディング│     文字数M + (N-M)個のフィラー
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ TextEmbedding                        │  → text_embed: (B, N, text_dim)
│  Embedding(2546+1, 512) +            │     text_dim=512
│  SinusoidalPosEmb +                  │
│  ConvNeXt V2 Blocks (×4)            │     ← テキスト前処理の核心
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ InputEmbedding                       │  → x_in: (B, N, dim)
│  Linear([φ_t, cond, text], dim) +    │     dim=1024
│  ConvPositionalEmbedding             │     入力 = [φ_t; cond; text] 結合
└──────────────────────────────────────┘
    |
    |  TimestepEmbedding: t → t_emb (B, dim)
    |  Sinusoidal → MLP
    |
    v
┌──────────────────────────────────────┐
│ DiT Blocks (×22)                     │
│  各ブロック:                          │
│    AdaLN-zero (時刻条件付き正規化)    │
│    Multi-Head Self-Attention + RoPE  │  → x: (B, N, dim)
│    FeedForward (dim → dim*ff_mult)   │     dim=1024, heads=16, head_dim=64
│    ゲート付き残差接続                  │
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ AdaLayerNorm_Final + Linear          │  → pred_flow: (B, N, F)
│  最終変調 + 射影 (dim → mel_dim)     │     F=100 (mel次元)
└──────────────────────────────────────┘
    |
    v
  Loss = MSE(pred_flow, flow) [マスク領域のみ]
```

### 推論時

```
参照音声 x_ref + 参照テキスト y_ref + 生成テキスト y_gen
    |
    v
┌──────────────────────────────────────┐
│ 1. Duration推定                      │  → N (生成mel長)
│    N = len(x_ref) + len(x_ref) *     │     テキスト長比で線形推定
│        len(y_gen) / len(y_ref)       │
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ 2. 条件・テキスト準備                 │
│    cond: [x_ref, 0...0] (B, N, F)   │  参照mel + ゼロパディング
│    text: [z_ref, z_gen]  (B, N)     │  参照+生成テキスト結合
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ 3. ノイズ初期化                       │
│    y0 ~ N(0, I)          (B, N, F)  │  ガウスノイズから開始
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ 4. Sway Sampling タイムステップ生成   │
│    t_k = t_k + s*(cos(π/2 * t_k)    │  s=-1 (左寄り: 初期ステップ重視)
│            - 1 + t_k)               │  NFE=32ステップ
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ 5. ODE求解 (Euler法)                 │
│    for k = 0 to NFE-1:              │
│      v_cond = DiT(y_k, cond,        │  条件付き予測
│                   text, t_k)         │
│      v_uncond = DiT(y_k, 0,         │  無条件予測 (CFG)
│                     0, t_k)          │
│      v = v_cond + α*(v_cond          │  α=2.0 (CFG強度)
│                      - v_uncond)     │
│      y_{k+1} = y_k + (t_{k+1}-t_k)  │  → 最終出力: (B, N, F)
│                      * v             │
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│ 6. 条件部分置換 + Vocoder             │
│    out = where(cond_mask, cond, y_N) │  参照部分を元に戻す
│    mel → Vocos → waveform            │  → audio: (B, nw)
└──────────────────────────────────────┘
```

---

## ファイル構成

### 1. [main_flow.py](main_flow.py)
**F5-TTSの全体フロー (CFM + DiT)**

Conditional Flow Matching ラッパーと DiT バックボーンの統合:
- 学習時: ランダムマスク → ノイズ混合 → フロー予測 → MSE損失
- 推論時: ODE求解 (Sway Sampling + CFG) → mel生成 → Vocoder

```python
class F5TTS(nn.Module):
    def forward(self, audio, text):
        # audio: (B, nw) → mel: (B, N, 100)
        # ランダムマスク → infilling学習
        # 返値: loss, cond, pred_flow

    def sample(self, cond, text, duration, steps=32):
        # ODE求解でmel生成
        # 返値: generated_mel (B, N, 100), trajectory
```

**重要ポイント:**
- Flow Matching: φ_t = (1-t)*x0 + t*x1, flow = x1 - x0
- CFG: 2段階ドロップ (audio_drop=0.3, cond_drop=0.2)
- Sway Sampling: 初期ステップに集中して品質向上

---

### 2. [dit_model.py](dit_model.py)
**DiTバックボーン + ConvNeXt V2テキスト処理**

DiT (Diffusion Transformer) の詳細実装:
- TextEmbedding: 文字埋め込み + ConvNeXt V2 ブロック
- InputEmbedding: [ノイズ音声, 条件音声, テキスト] → 統合埋め込み
- DiTBlock: AdaLN-zero + Self-Attention (RoPE) + FFN
- AdaLayerNorm: 時刻条件付き正規化 (6パラメータ: shift/scale/gate × 2)

```python
class DiT(nn.Module):
    def forward(self, x, cond, text, time, mask=None):
        # x: (B, N, 100) noised mel
        # cond: (B, N, 100) masked mel
        # text: (B, nt) character tokens
        # time: (B,) flow step
        # → pred_flow: (B, N, 100)
```

**重要ポイント:**
- ConvNeXt V2: テキストを音声と同じモダリティ空間に近づける
- AdaLN-zero: ゼロ初期化で学習安定化 (DiT論文の手法)
- RoPE: 回転位置埋め込みで可変長シーケンス対応

---

### 3. [training_inference.py](training_inference.py)
**学習パイプライン + 推論パイプライン**

学習:
- データ読み込み (Emilia 95K時間)
- 動的バッチサイズ (フレーム数ベース)
- EMA (Exponential Moving Average)
- Accelerateによる分散学習

推論:
- 参照音声前処理
- Duration推定
- ODE求解 + Sway Sampling
- チャンク分割 + クロスフェード結合

```python
class Trainer:
    def train(self, dataset):
        # バッチサイズ: 307,200フレーム/バッチ (8 GPU)
        # AdamW, lr=7.5e-5, warmup=20K
        # 1.2Mアップデート

class InferencePipeline:
    def generate(self, ref_audio, ref_text, gen_text):
        # → waveform (nw,), sample_rate=24000
```

---

## F5-TTSの主要イノベーション

### 1. **ConvNeXt V2によるテキスト前処理 (TextEmbedding)**
**問題**: E2 TTSではテキスト(文字列)とメルスペクトログラムが直接結合されるが、
意味情報と音響特徴量の間に大きな情報ギャップがあり、学習収束が遅く頑健性が低い。

**解決**:
- テキスト埋め込み後にConvNeXt V2ブロック(4層)を適用
- テキストに個別のモデリング空間を与え、音声との結合前に前処理
- Depthwise Conv (kernel=7) + GRN (Global Response Normalization) + GELU

**効果**:
- WER: 9.63% (E2 TTS) → 4.17% (F5-TTS) at 800K updates
- 学習速度: E2 TTSより大幅に高速な収束

**実装**: [dit_model.py - TextEmbedding](dit_model.py)

---

### 2. **adaLN-zero DiTブロック (U-Net置換)**
**問題**: E2 TTSのFlat U-Net Transformerはスキップ接続付きだが、
テキストアラインメント学習にはより柔軟な時刻条件付けが必要。

**解決**:
- U-Netを廃止し、DiT (Diffusion Transformer) を採用
- adaLN-zero: フローステップ t をAdaptive Layer Normの条件として使用
- 6個の変調パラメータ (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
- ゼロ初期化: 学習初期は恒等写像に近い動作

**効果**:
- GFLOPs: 301 (E2 TTS) → 173 (F5-TTS) — スキップ接続なしで高速
- テキストアラインメントの頑健性が大幅向上

**実装**: [dit_model.py - DiTBlock](dit_model.py)

---

### 3. **Sway Sampling (推論時サンプリング戦略)**
**問題**: フローステップを均一にサンプリングすると、
初期ステップ(大まかな構造決定)に十分な計算リソースが割り当てられない。

**解決**:
- 推論時のみ非一様サンプリングを適用 (学習時は均一U[0,1])
- `f_sway(u; s) = u + s * (cos(π/2 * u) - 1 + u)`
- s < 0: 左寄り (初期ステップ密、後期疎) → テキスト忠実度向上
- s = -1 をデフォルト使用

**効果**:
- 16 NFEでRTF 0.15を達成 (32 NFEと同等品質)
- 既存のCFMモデルに再学習なしで適用可能

**実装**: [main_flow.py - sample()](main_flow.py), [training_inference.py](training_inference.py)

---

### 4. **Classifier-Free Guidance (CFG)**
**問題**: 条件付き生成の忠実度をさらに向上させたい。

**解決**:
- 学習時: 2段階のランダムドロップ
  - 音声条件ドロップ: p=0.3 (マスク音声をゼロに)
  - 全条件ドロップ: p=0.2 (音声+テキスト両方をゼロに)
- 推論時: 条件付き/無条件予測の線形外挿
  - `v = v_cond + α * (v_cond - v_uncond)`, α=2.0

**効果**:
- 話者類似度 (SIM-o) と テキスト忠実度 (WER) の両方を向上

**実装**: [main_flow.py - forward(), sample()](main_flow.py)

---

## 処理フロー詳細

### 学習フロー

```python
# 1. 入力準備
audio = load_audio()                    # (B, nw) 24kHz waveform
text = ["Hello world", "こんにちは"]    # テキスト文字列リスト

# 2. Mel-Spectrogram抽出
mel = mel_spec(audio)                   # (B, 100, N) mel spectrogram
x1 = mel.permute(0, 2, 1)              # (B, N, 100) → sequence-first

# 3. テキストトークン化
z = tokenize(text)                      # (B, nt) 文字 → インデックス
z = pad_to_length(z, N, filler=0)       # (B, N)  フィラートークンでパディング

# 4. ランダムマスク生成 (70%-100%をマスク)
frac = uniform(0.7, 1.0)               # マスク比率
mask = random_span_mask(lens, frac)     # (B, N) True=マスク対象

# 5. 条件音声: マスク外の部分を保持
cond = where(mask, zeros, x1)           # (B, N, 100) マスク部分はゼロ

# 6. ノイズサンプリング & フロー混合
x0 = randn_like(x1)                    # (B, N, 100) ガウスノイズ
t = rand(B)                             # (B,) 時刻 U[0,1]
φ_t = (1-t)*x0 + t*x1                  # (B, N, 100) ノイズ混合 (OT path)
flow = x1 - x0                          # (B, N, 100) フロー目標

# 7. DiT予測
pred_flow = dit(x=φ_t, cond=cond,      # (B, N, 100) 予測フロー
                text=z, time=t)

# 8. 損失計算 (マスク領域のみ)
loss = mse(pred_flow[mask], flow[mask])  # スカラー
```

### 推論フロー

```python
# 1. 入力準備
ref_audio = load_audio("prompt.wav")    # (1, nw) 参照音声
ref_text = "Are you OK?"               # 参照テキスト
gen_text = "I'm fine!"                  # 生成テキスト

# 2. mel抽出 & Duration推定
ref_mel = mel_spec(ref_audio)           # (1, 100, N_ref)
ref_mel = ref_mel.permute(0, 2, 1)     # (1, N_ref, 100)
N_gen = N_ref * len(gen_text) / len(ref_text)
N = N_ref + N_gen                       # 全体の長さ

# 3. 条件構築
cond = pad(ref_mel, [0, N-N_ref])       # (1, N, 100) 参照mel + ゼロ
text = tokenize(ref_text + gen_text)    # (1, N) テキスト結合

# 4. ノイズ初期化
y0 = randn(1, N, 100)                  # (1, N, 100) ガウスノイズ

# 5. Sway Samplingタイムステップ
t = linspace(0, 1, 33)                  # 0.0, 0.03, ..., 1.0  (32 NFE)
t = t + (-1) * (cos(π/2 * t) - 1 + t)  # 左寄りサンプリング (s=-1)

# 6. ODE求解 (Euler + CFG)
for k in range(32):
    # CFG: 条件付き + 無条件を同時計算
    pred = dit(y_k, cond, text, t[k], cfg_infer=True)  # (2, N, 100)
    v_cond, v_uncond = pred.chunk(2)
    v = v_cond + 2.0 * (v_cond - v_uncond)              # CFG strength=2.0
    y_{k+1} = y_k + (t[k+1] - t[k]) * v                 # Eulerステップ

# 7. 後処理
out = where(cond_mask, cond, y_N)       # (1, N, 100) 参照部分を復元
out = out.permute(0, 2, 1)              # (1, 100, N)
waveform = vocos(out)                   # (1, nw) mel → 波形
# 参照部分を捨てて生成部分のみ出力
```

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | audio (raw) | `(B, nw)` | 24kHz waveform |
| | text (raw) | `list[str]` | 文字列リスト |
| **Mel抽出** | mel | `(B, 100, N)` | Mel-spectrogram (Vocos backend) |
| | mel (transposed) | `(B, N, 100)` | sequence-first形式 |
| **テキスト** | tokens | `(B, nt)` | 文字トークンインデックス |
| | tokens (padded) | `(B, N)` | フィラーでmel長にパディング |
| **TextEmbedding** | text_embed | `(B, N, 512)` | ConvNeXt V2処理後 |
| **InputEmbedding** | x_in | `(B, N, 1024)` | [ノイズ,条件,テキスト]統合 |
| **TimestepEmb** | t_emb | `(B, 1024)` | 時刻条件ベクトル |
| **DiT Block** | hidden | `(B, N, 1024)` | Transformer隠れ状態 |
| | Q, K, V | `(B, 16, N, 64)` | Attention (16 heads, dim=64) |
| **出力** | pred_flow | `(B, N, 100)` | 予測フローベクトル |
| **推論** | generated_mel | `(B, N, 100)` | 生成メルスペクトログラム |
| | waveform | `(B, nw)` | 最終音声波形 |

### 軸の意味

- **B**: バッチサイズ
- **nw**: 波形サンプル数 (24kHz × 秒数)
- **N**: メルスペクトログラムのフレーム数 (= nw / hop_length, hop=256)
- **F** (= 100): メル周波数ビン数 (n_mel_channels)
- **nt**: テキストトークン長 (文字数)
- **dim** (= 1024): DiT隠れ次元
- **text_dim** (= 512): テキスト埋め込み次元
- **heads** (= 16): Attention ヘッド数
- **head_dim** (= 64): ヘッドあたりの次元
- **ff_mult** (= 2): FFN中間次元の倍率 (1024 × 2 = 2048)
- **depth** (= 22): DiTブロック数
- **vocab_size** (= 2546): 文字語彙サイズ (英字+記号+漢字ピンイン+フィラー+その他言語)

### 時間と長さの関係

```
1秒の音声:
  waveform: 24,000 samples (24kHz)
  mel frames: 24,000 / 256 = 93.75 ≈ 94 frames

10秒の音声:
  waveform: 240,000 samples
  mel frames: ~938 frames

最大長: 65,536 frames ≈ 699秒 ≈ 11.6分
```

---

## FAQ

### Q1: F5-TTSとE2 TTSの違いは?

**A**: 主な違いは3点:

1. **テキスト前処理**
   - E2 TTS: テキスト文字列をそのままmel長にパディングして結合
   - F5-TTS: ConvNeXt V2で前処理してから結合 → 情報ギャップ低減

2. **バックボーン**
   - E2 TTS: Flat U-Net Transformer (24層, スキップ接続あり)
   - F5-TTS: DiT with adaLN-zero (22層, スキップ接続なし) → 軽量・高速

3. **GFLOPs**
   - E2 TTS: 301 GFLOPs (161Mパラメータ)
   - F5-TTS: 173 GFLOPs (158Mパラメータ) ← 42%少ない計算量

**結果**: F5-TTSはE2 TTSより学習が大幅に高速で頑健 (WER: 9.63% → 4.17%)

---

### Q2: なぜ音素(phoneme)ではなく文字(character)を使うのか?

**A**: パイプラインの簡素化と多言語対応のためです。

**音素ベース (従来手法)**:
- テキスト → G2P変換 → 音素列 → Duration Predictor → アラインメント
- 言語ごとにG2Pモジュールが必要
- Duration Predictorの精度がボトルネック

**文字ベース (F5-TTS)**:
- テキスト → 文字トークン → フィラーでパディング → そのままモデル入力
- G2P不要、Duration Predictor不要
- モデル自身がアラインメントを学習

**多言語対応**:
- 英語: アルファベット + 記号 (直接使用)
- 中国語: ピンイン変換 (pypinyin使用)
- 語彙サイズ: 2546トークン (全言語統一)

---

### Q3: Flow Matchingとは何か? Diffusionとの違いは?

**A**: Flow Matchingは確率的フロー (ODE) ベースの生成モデルです。

**拡散モデル (DDPM)**:
- 前方過程: x_t = √(α_t) * x + √(1-α_t) * ε (ノイズスケジュール依存)
- 逆過程: SDE/ODEで逐次デノイズ
- 多数のステップが必要 (50-1000)

**Flow Matching (Optimal Transport)**:
- フロー: ψ_t(x) = (1-t)*x0 + t*x1 (線形補間、単純)
- 目標: x1 - x0 のベクトル場を学習
- ODE: dψ/dt = v_t(ψ_t) を積分
- 少ないステップで高品質 (16-32)

**F5-TTSでの利点**:
- 学習: t ~ U[0,1] を均一サンプリングするだけ
- 推論: Euler法で16-32ステップ
- RTF 0.15 (10秒音声を1.5秒で生成)

---

### Q4: Sway Samplingはなぜ効果的か?

**A**: フローステップの重要度に応じた非一様サンプリングです。

**直感的理解**:
- 初期ステップ (t ≈ 0): 大まかな音声構造・テキストアラインメントを決定
- 後期ステップ (t ≈ 1): 細部の音響品質を仕上げ

**s < 0 (左寄り)**:
- 初期ステップを密にサンプリング
- テキストに忠実な音声構造の生成を重視
- WER改善 + 話者類似度維持

**数式**:
```
t' = t + s * (cos(π/2 * t) - 1 + t)

s = -1, 均一 u=0.1 の場合:
  t' = 0.1 + (-1) * (cos(π/20) - 0.9) ≈ 0.0124
  → 0.1が0.0124に縮小 (初期密度増加)

s = -1, 均一 u=0.9 の場合:
  t' = 0.9 + (-1) * (cos(9π/20) - 0.1) ≈ 0.9564
  → 0.9が0.9564に拡大 (後期密度低下)
```

**再学習不要**: 推論時のみ適用するため、既存モデルにそのまま使用可能

---

### Q5: CFG (Classifier-Free Guidance) の学習と推論は?

**A**: 学習時にランダムドロップ、推論時に線形外挿です。

**学習時 (2段階ドロップ)**:
```python
# Stage 1: 音声条件のみドロップ (p=0.3)
if random() < 0.3:
    cond = zeros_like(cond)  # 参照音声をゼロに

# Stage 2: 全条件ドロップ (p=0.2) — Stage 1に上書き
if random() < 0.2:
    cond = zeros_like(cond)  # 音声もテキストも
    text = zeros_like(text)  # 両方ゼロに
```

**推論時 (CFG線形外挿)**:
```python
# バッチサイズを2倍にして効率化
pred_cfg = dit(x, cond, text, t, cfg_infer=True)  # (2B, N, 100)
v_cond, v_uncond = pred_cfg.chunk(2)

# 外挿: 条件方向に強調
v = v_cond + α * (v_cond - v_uncond)  # α=2.0
```

**CFG強度α**:
- α=0: 無条件生成 (多様だが不忠実)
- α=1: 通常の条件付き生成
- α=2: 条件を強調 (デフォルト、忠実度↑)
- α>3: 過剰 (アーティファクト発生)

---

### Q6: テキストのフィラートークンとは?

**A**: テキスト長とmel長を揃えるためのパディングトークンです。

**背景**:
- テキスト文字数 M << mel フレーム数 N (例: 20文字 vs 938フレーム)
- DiTの入力はテキストとmelを同じ時間軸で結合する必要がある

**フィラー戦略**:
```python
# テキスト: "Hello" (5文字) → mel長: 938 フレーム
z = [H, e, l, l, o, ⟨F⟩, ⟨F⟩, ..., ⟨F⟩]  # 5文字 + 933フィラー
#                                             合計 = 938 トークン

# ⟨F⟩ = フィラートークン (index 0)
# モデルは⟨F⟩の位置に暗黙的にDuration情報を学習
```

**Voicebox/E2 TTSとの違い**:
- Voicebox: 音素レベルのDuration Predictorで明示的アラインメント
- E2 TTS/F5-TTS: フィラーパディングで暗黙的アラインメント (シンプル)

---

### Q7: adaLN-zeroとは何か? なぜゼロ初期化?

**A**: Adaptive Layer Normalization with Zero-initialization です。

**通常のLayerNorm**:
```python
# 固定のスケール/シフト
x = LayerNorm(x)  # 学習パラメータ: γ, β
```

**adaLN-zero**:
```python
# 時刻tに応じた動的スケール/シフト + ゲート
t_emb = time_embed(t)                                     # (B, dim)
shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp = Linear(SiLU(t_emb))     # 各 (B, dim)

# Attention前
x_normed = LayerNorm(x) * (1 + scale_msa) + shift_msa
attn_out = SelfAttention(x_normed)
x = x + gate_msa * attn_out                               # ゲート付き残差

# FFN前
x_normed = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
ff_out = FFN(x_normed)
x = x + gate_mlp * ff_out                                 # ゲート付き残差
```

**ゼロ初期化の理由**:
- Linear層の重み・バイアスをゼロで初期化
- 学習初期: gate=0 → 各ブロックが恒等写像 (x → x)
- 段階的に非ゼロ値を学習 → 安定した学習開始
- DiT論文 (Peebles & Xie, 2023) で有効性が実証済み

---

### Q8: 位置エンコーディングの使い分けは?

**A**: F5-TTSでは3種類の位置エンコーディングを使い分けます。

| 位置エンコーディング | 適用先 | 目的 |
|---|---|---|
| **Sinusoidal PE** | テキスト文字列 (ConvNeXt V2前) | テキストの絶対位置を示す |
| **Conv Position Embedding** | InputEmbedding後 (結合シーケンス) | 局所的な位置情報を注入 |
| **RoPE** | DiTブロックのSelf-Attention | 相対位置を距離に基づいてエンコード |

**Conv Position Embedding**:
```python
# 2層のGrouped 1D Conv (kernel=31)
# Voiceboxと同じ設定
x = Conv1d(x, kernel=31, groups=16)  # 局所的位置
x = Mish(x)
x = Conv1d(x, kernel=31, groups=16)
x = Mish(x)
```

**RoPE (Rotary Position Embedding)**:
- Self-Attentionのみに適用 (FFNには不適用)
- 相対位置に基づく回転で距離減衰を実現
- ALiBi biasの代わりにRoPEを使用 (対称的な双方向性)

---

### Q9: Vocoderの役割と種類は?

**A**: メルスペクトログラムから音声波形に変換するモジュールです。

**対応Vocoder**:

| Vocoder | 品質 | 速度 | 備考 |
|---|---|---|---|
| **Vocos** | 高品質 | 高速 | デフォルト。周波数-時間領域変換 |
| **BigVGAN** | 最高品質 | やや遅い | 大規模学習済み。音質重視 |

**Mel-Spectrogram設定**:
```python
MelSpec(
    n_fft=1024,
    hop_length=256,          # 256サンプル = 10.67ms
    win_length=1024,         # 1024サンプル = 42.67ms
    n_mel_channels=100,      # mel周波数ビン数
    target_sample_rate=24000 # 24kHz
)
```

**Vocos方式**: `MelSpectrogram(power=1) → clamp(min=1e-5) → log()`
**BigVGAN方式**: `STFT → |magnitude| → mel_basis @ mag → clamp → log()`

---

### Q10: F5-TTSの限界と将来展望は?

**A**: 主に2つの限界があります。

1. **メルスペクトログラムの長さ**:
   - mel長 >> テキスト長 (例: 938フレーム vs 20文字)
   - フィラートークンによる冗長性
   - 将来: より効率的な連続表現の研究

2. **パラ言語情報の制御**:
   - 感情・イントネーションの明示的制御が困難
   - Zero-shotで話者音色は模倣できるが、細かい表現は制御不可
   - 将来: 感情ラベルやスタイルトークンの統合

**改善の方向性**:
- 学習時ノイズスケジューラとSway Samplingの組み合わせ
- 蒸留技術による1-4ステップ生成
- より効率的な音声表現 (neural codec等) の統合

---

## 参考文献

- 論文: [F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching](https://arxiv.org/abs/2410.06885)
- 公式実装: [https://github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- 関連研究:
  - E2 TTS (Eskimez et al., 2024): Embarrassingly easy fully non-autoregressive zero-shot TTS
  - Voicebox (Le et al., 2024): Text-guided multilingual universal speech generation at scale
  - DiT (Peebles & Xie, 2023): Scalable diffusion models with transformers
  - Flow Matching (Lipman et al., 2022): Flow matching for generative modeling
  - ConvNeXt V2 (Woo et al., 2023): Co-designing and scaling convnets with masked autoencoders
  - Sway Sampling: 本論文で提案された推論時サンプリング戦略

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。実際の実装とは異なる場合があります。
