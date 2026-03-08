# 🎙️ VibeVoice Technical Report 完全解説

VibeVoice は Microsoft Research が開発した、**長時間・複数話者対応の会話音声合成モデル**である。Next-Token Diffusion フレームワークにより、最大90分・4話者の自然な会話音声を64Kコンテキストウィンドウ内で合成できる。

- **論文**: [VibeVoice Technical Report (arXiv:2508.19205)](https://arxiv.org/abs/2508.19205)
- **公式コード**: [github.com/microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- **Hugging Face**: [microsoft/VibeVoice](https://huggingface.co/microsoft/VibeVoice)

---

## 📋 目次

1. [概要](#1-概要)
2. [アーキテクチャ全体像](#2-アーキテクチャ全体像)
3. [ファイル構成](#3-ファイル構成)
4. [Speech Tokenizer（音声トークナイザ）](#4-speech-tokenizer音声トークナイザ)
5. [VibeVoice 本体（LLM + Diffusion）](#5-vibevoice-本体llm--diffusion)
6. [Diffusion Head（拡散ヘッド）](#6-diffusion-head拡散ヘッド)
7. [Streaming TTS（リアルタイム音声合成）](#7-streaming-ttsリアルタイム音声合成)
8. [ASR モデル（音声認識）](#8-asr-モデル音声認識)
9. [学習の詳細](#9-学習の詳細)
10. [主要イノベーション](#10-主要イノベーション)
11. [形状ガイド](#11-形状ガイド)
12. [FAQ](#12-faq)
13. [まとめ](#13-まとめ)
14. [参考文献](#14-参考文献)

---

## 1. 概要

### 背景と動機

従来の TTS システムは短い単一話者発話では高品質だが、**長時間の複数話者会話**（ポッドキャスト、オーディオブックなど）では以下の課題があった：

- 個別発話の連結では自然なターンテイキングが困難
- シーケンス長の爆発（50Hz以上のフレームレートでは長時間音声を扱えない）
- 複数話者の音色・感情の制御が困難

### VibeVoice の解決策

| 技術 | 概要 | 効果 |
|------|------|------|
| **超低フレームレート Tokenizer** | 7.5 Hz（3200倍圧縮） | 90分の音声を64Kトークン内に収容可能 |
| **Dual Tokenization** | Acoustic（音質）+ Semantic（意味）の分離 | 高音質と意味理解の両立 |
| **Next-Token Diffusion** | LLM の隠れ状態から拡散ヘッドで連続潜在変数を予測 | ベクトル量子化不要で音質劣化なし |
| **LLM バックボーン** | Qwen2.5（1.5B / 7B） | 複雑なスクリプトの理解と多話者制御 |

### 性能比較

| モデル | Realism ↑ | Richness ↑ | Preference ↑ | Average ↑ | WER (Whisper) ↓ | WER (Nemo) ↓ | SIM ↑ |
|--------|-----------|------------|--------------|-----------|-----------------|--------------|-------|
| SesameAILabs-CSM | 2.89 | 3.03 | 2.75 | 2.89 | 2.66 | 3.05 | 0.685 |
| Gemini 2.5 Pro Preview | 3.55 | 3.78 | 3.65 | 3.66 | 1.73 | 2.43 | - |
| Elevenlabs v3 alpha | 3.34 | 3.48 | 3.38 | 3.40 | 2.39 | 2.47 | 0.623 |
| **VibeVoice-1.5B** | 3.59 | 3.59 | 3.44 | 3.54 | 1.11 | 1.82 | 0.548 |
| **VibeVoice-7B** | **3.71** | **3.81** | **3.75** | **3.76** | 1.29 | 1.95 | **0.692** |

> VibeVoice-7B は主観・客観評価の両方で全モデルを上回る。

---

## 2. アーキテクチャ全体像

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VibeVoice 全体パイプライン                             │
│                                                                             │
│  【入力】                                                                    │
│  ┌──────────────┐  ┌───────────────────────────────────────────────────┐    │
│  │ Voice Prompts │  │ Text Scripts                                     │    │
│  │ (各話者の音声) │  │ Speaker1: "Welcome to ..."                       │    │
│  │              │  │ Speaker2: "Thanks for having ..."                 │    │
│  │ [B, samples] │  │ Speaker3: "Hello, uh, I'm ..."                   │    │
│  └──────┬───────┘  └───────────────────┬───────────────────────────────┘    │
│         │                               │                                   │
│         ▼                               │                                   │
│  ┌──────────────────────┐               │                                   │
│  │ Acoustic Tokenizer   │               │                                   │
│  │ Encoder (σ-VAE)      │               │                                   │
│  │ 24kHz→7.5Hz (3200x)  │               │                                   │
│  │ [B, T, 64]           │               │                                   │
│  └──────┬───────────────┘               │                                   │
│         │                               │                                   │
│         ▼                               ▼                                   │
│  ┌──────────────────┐  ┌──────────────────────┐                             │
│  │ Acoustic Connector│  │ Text Tokenizer       │                             │
│  │ 64→hidden_size   │  │ (Qwen2 BPE)          │                             │
│  │ fc1→RMSNorm→fc2  │  │ [B, T_text]          │                             │
│  └──────┬───────────┘  └──────────┬───────────┘                             │
│         │                          │                                        │
│         ▼                          ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │              Qwen2.5 LLM (1.5B or 7B)                           │       │
│  │                                                                  │       │
│  │  入力: [Voice_Prompt_Tokens] + [Text_Tokens] + [<Start>]        │       │
│  │                                                                  │       │
│  │  Auto-regressive 生成:                                           │       │
│  │    各トークン位置で hidden_state h_i を出力                       │       │
│  │    [B, T_total, hidden_size]                                     │       │
│  └──────────────────────────────────┬───────────────────────────────┘       │
│                                      │                                      │
│         ┌────────────────────────────┤                                      │
│         │                            │                                      │
│         ▼                            ▼                                      │
│  ┌──────────────────┐  ┌──────────────────────────────────┐                 │
│  │ LM Head          │  │ Diffusion Head (4層)              │                 │
│  │ logits→vocab_size│  │                                    │                 │
│  │ (テキストトークン) │  │  条件: h_i (LLM隠れ状態)          │                 │
│  └──────────────────┘  │  入力: ノイズ z_t [B, T, 64]      │                 │
│                        │  出力: 予測速度 v [B, T, 64]       │                 │
│                        │  20ステップ DPM-Solver++ で推論    │                 │
│                        │  CFG (guidance_scale=1.3)          │                 │
│                        └──────────────┬─────────────────────┘                │
│                                       │                                     │
│                                       ▼                                     │
│                        ┌──────────────────────────────────┐                 │
│                        │ Acoustic Tokenizer Decoder        │                 │
│                        │ 7.5Hz → 24kHz (3200x アップ)      │                 │
│                        │ [B, T, 64] → [B, 1, T_audio]     │                 │
│                        └──────────────┬─────────────────────┘                │
│                                       │                                     │
│                                       ▼                                     │
│                              ┌──────────────┐                               │
│                              │  音声波形出力  │                               │
│                              │ [B,1,T_audio] │                               │
│                              │ 24kHz, mono   │                               │
│                              └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ASR（音声認識）パイプライン

```
音声入力 [B, samples]
    │
    ├─── Acoustic Tokenizer Encode ──→ [B, T, 64] ──→ Acoustic Connector ──┐
    │                                                                       │
    └─── Semantic Tokenizer Encode ──→ [B, T, 128] ──→ Semantic Connector ─┤
                                                                            │
                                                                            ▼
                                                               [B, T, hidden_size]
                                                                   (合算)
                                                                            │
                                          ┌─────────────────────────────────┘
                                          ▼
                                    Qwen2.5 LLM
                                          │
                                          ▼
                                    LM Head → テキスト出力
                                    (JSON形式: 発話内容 + タイムスタンプ + 話者ID)
```

---

## 3. ファイル構成

### 本リポジトリ（VibeVoice_understanding）

| ファイル | 役割 | 主要クラス |
|---------|------|-----------|
| `README.md` | 全体のドキュメントハブ | - |
| `main_flow.py` | TTS全体のデータフロー | `VibeVoiceForConditionalGeneration` |
| `speech_tokenizer.py` | σ-VAE ベースの音声トークナイザ | `AcousticTokenizer`, `SemanticTokenizer` |
| `diffusion_head.py` | トークンレベル拡散ヘッド | `VibeVoiceDiffusionHead` |
| `streaming_inference.py` | ストリーミングTTS推論 | `VibeVoiceStreamingInference` |
| `asr_model.py` | ASR（音声認識）モデル | `VibeVoiceASR` |
| `loss_and_training.py` | 損失関数と学習の詳細 | `DiffusionLoss`, `TrainingPipeline` |

### 公式リポジトリ（参照先）

```
VibeVoice/
├── vibevoice/
│   ├── modular/
│   │   ├── configuration_vibevoice.py           # 全構成クラス
│   │   ├── configuration_vibevoice_streaming.py  # Streaming構成
│   │   ├── modeling_vibevoice.py                 # TTS本体モデル
│   │   ├── modeling_vibevoice_asr.py             # ASRモデル
│   │   ├── modeling_vibevoice_streaming.py       # Streaming TTS基底
│   │   ├── modeling_vibevoice_streaming_inference.py # Streaming推論
│   │   ├── modular_vibevoice_tokenizer.py        # 音声トークナイザ
│   │   ├── modular_vibevoice_diffusion_head.py   # 拡散ヘッド
│   │   └── streamer.py                           # 音声ストリーミング
│   ├── processor/
│   │   ├── vibevoice_processor.py                # TTS プロセッサ
│   │   ├── vibevoice_asr_processor.py            # ASR プロセッサ
│   │   ├── vibevoice_streaming_processor.py      # Streaming プロセッサ
│   │   └── audio_utils.py                        # 音声ユーティリティ
│   ├── schedule/
│   │   ├── dpm_solver.py                         # DPM-Solver++
│   │   └── timestep_sampler.py                   # タイムステップサンプラ
│   └── configs/
│       ├── qwen2.5_1.5b_64k.json                # 1.5Bモデル設定
│       └── qwen2.5_7b_32k.json                  # 7Bモデル設定
├── demo/                                         # デモスクリプト
├── finetuning-asr/                               # ASR LoRA ファインチューニング
└── vllm_plugin/                                  # vLLM統合
```

---

## 4. Speech Tokenizer（音声トークナイザ）

### 🔑 キー・イノベーション：超低フレームレート（7.5 Hz）σ-VAE

**問題**: 従来のコーデック（Encodec: 600トークン/秒, DAC: 400トークン/秒）では長時間音声がLLMの文脈長を超える

**解決**: σ-VAE による 7.5 Hz（1秒あたり7.5トークン）の超高圧縮トークナイザ

**効果**:
- Encodec 比 80倍の圧縮率改善
- 90分音声 → 約40,500トークン（64Kコンテキスト内に収まる）
- speech-to-text トークン比 ≈ 2:1（2音声トークン ≈ 1 BPEテキストトークン）

### Acoustic Tokenizer

σ-VAE (Variational Autoencoder) に基づく音声圧縮。分散 σ を固定（学習しない）とすることで、自己回帰モデリングでの分散崩壊を防ぐ。

```
【エンコーダ構造】
24kHz 音声入力 [B, 1, T_samples]
    │
    ├── Stem Conv: [1 → 32] channels
    │
    ├── Downsample Stage 1: stride=2, [32 → 64], Block1D × 8
    ├── Downsample Stage 2: stride=2, [64 → 128], Block1D × 3
    ├── Downsample Stage 3: stride=4, [128 → 256], Block1D × 3
    ├── Downsample Stage 4: stride=5, [256 → 512], Block1D × 3
    ├── Downsample Stage 5: stride=5, [512 → 1024], Block1D × 3
    ├── Downsample Stage 6: stride=8, [1024 → 2048], Block1D × 3
    │   (累積ダウンサンプリング: 2×2×4×5×5×8 = 3200倍)
    │
    ├── RMSNorm
    │
    └── Head Conv: [2048 → 64 (vae_dim)]
        出力: [B, 64, T_latent] → permute → [B, T_latent, 64]
        T_latent = ceil(T_samples / 3200) ≈ 7.5 tokens/sec

【サンプリング（σ-VAE）】
z = μ + σ ⊙ ε
  μ: エンコーダ出力（mean）
  σ ~ N(0, C_σ): 固定分布（fix_std=0.5）
  ε ~ N(0, 1): 標準正規ノイズ

【デコーダ構造】（エンコーダの鏡像）
入力 [B, 64, T_latent]
    │
    ├── Stem Conv: [64 → 2048]
    │
    ├── Upsample Stage 1: stride=8, [2048 → 1024], ConvTranspose1d
    ├── Upsample Stage 2: stride=5, [1024 → 512]
    ├── Upsample Stage 3: stride=5, [512 → 256]
    ├── Upsample Stage 4: stride=4, [256 → 128]
    ├── Upsample Stage 5: stride=2, [128 → 64]
    ├── Upsample Stage 6: stride=2, [64 → 32]
    │
    ├── RMSNorm
    │
    └── Head Conv: [32 → 1]
        出力: [B, 1, T_samples] (24kHz 波形)
```

**パラメータ**: エンコーダ/デコーダそれぞれ約340Mパラメータ。学習目的関数は DAC [KSL+23] に従い、判別器＋損失関数を使用。

### Semantic Tokenizer

Acoustic Tokenizer のエンコーダと同構造だが、**デコーダなし**（エンコーダのみ）。
決定論的な特徴抽出（サンプリングなし：`std_dist_type='none'`）。

```
24kHz 音声入力 [B, 1, T_samples]
    │
    ├── Acoustic Tokenizer と同構造のエンコーダ
    │   (vae_dim=128, fix_std=0)
    │
    └── 出力: [B, T_latent, 128] (決定論的な意味特徴)
```

**学習**: ASR（自動音声認識）をプロキシタスクとして使用し、エンコーダの出力をTransformerデコーダ層でテキスト転写を予測するように学習。学習後デコーダは破棄。

### Tokenizer の再構成品質（LibriTTS）

| モデル | N_q | Token Rate | test-clean PESQ ↑ | test-clean UTMOS ↑ | test-other PESQ ↑ | test-other UTMOS ↑ |
|--------|-----|------------|--------------------|--------------------|--------------------|---------------------|
| Encodec | 8 | 600 | 2.72 | 3.04 | 2.682 | 3.483 |
| DAC | 4 | 400 | 2.738 | 3.433 | 2.595 | 2.945 |
| WavTokenizer | 1 | 75 | 2.373 | 4.049 | 2.261 | 3.431 |
| **Ours (Acoustic)** | **1** | **7.5** | **3.068** | **4.181** | **2.848** | **3.724** |

> 7.5 Hz という超低フレームレートにもかかわらず、PESQとUTMOSの両方で最高性能を達成。

実装詳細: [speech_tokenizer.py](speech_tokenizer.py)

---

## 5. VibeVoice 本体（LLM + Diffusion）

### 入力表現

モデルへの入力 X は、音声プロンプトとテキストスクリプトの連結：

```
X = [Speaker₁ : z₁, Speaker₂ : z₂, ..., Speaker_N : z_N]   ← 音声プロンプト（各話者の声質参照）
  + [Speaker₁ : T₁, Speaker₂ : T₂, ..., Speaker_N : T_N]   ← テキストスクリプト（各話者の台詞）

z_k: 話者 k の音声を Acoustic Tokenizer でエンコードした潜在表現 [T_voice, 64]
T_k: 話者 k のテキスト台詞（BPEトークン列）
```

### Token-Level Diffusion

LLM は各トークン位置で隠れ状態 **h_i** を出力。この h_i を**条件**として、軽量な Diffusion Head が Acoustic VAE の潜在変数 z_{a,i} を予測する。

```python
# 学習時
for each speech token position i:
    h_i = LLM.hidden_state[i]           # [hidden_size]
    z_a_i = acoustic_vae_features[i]    # [64] (正解)

    t = random_timestep(0, 1000)
    noise = torch.randn_like(z_a_i)
    z_noisy = scheduler.add_noise(z_a_i, noise, t)

    v_pred = diffusion_head(z_noisy, t, h_i)    # [64]
    loss = mse(v_pred, target_velocity)

# 推論時
for each speech token position i:
    h_i = LLM.hidden_state[i]
    z = torch.randn(64)  # 純粋ノイズから開始

    for t in dpm_solver_timesteps:  # 20ステップ (or 10ステップ)
        # Classifier-Free Guidance (CFG)
        v_cond = diffusion_head(z, t, h_i)         # 条件付き予測
        v_uncond = diffusion_head(z, t, zero_cond)  # 無条件予測
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        z = scheduler.step(v, t, z)

    acoustic_latent_i = z  # [64]
```

### 学習設定

| パラメータ | 値 |
|-----------|-----|
| LLM | Qwen2.5 (1.5B or 7B) |
| Diffusion Head 層数 | 4 |
| 学習時 Diffusion ステップ | 1000 |
| 推論時 Diffusion ステップ | 10（VibeVoice）/ 20（Streaming） |
| CFG スケール | 1.3 |
| 予測タイプ | v_prediction |
| ベータスケジュール | cosine |
| Curriculum Learning | 4,096 → 65,536 トークン漸増 |
| 凍結パラメータ | Acoustic Tokenizer, Semantic Tokenizer |
| 学習パラメータ | LLM, Diffusion Head |

実装詳細: [main_flow.py](main_flow.py)

---

## 6. Diffusion Head（拡散ヘッド）

### 🔑 キー・イノベーション：軽量トークンレベル拡散

**問題**: 従来の音声生成拡散モデルは重く、自己回帰生成と組み合わせにくい

**解決**: わずか4層の Transformer ブロックで構成される軽量拡散ヘッド

**効果**: LLM の隠れ状態を条件として高品質な音声潜在変数を効率的に予測

### アーキテクチャ

```
VibeVoiceDiffusionHead
│
├── noisy_images_proj: Linear(64 → hidden_size, no bias)
│   ノイズ付き潜在変数をヘッド次元に射影
│
├── cond_proj: Linear(hidden_size → cond_dim, no bias)
│   LLM 隠れ状態を条件次元に射影
│
├── t_embedder: TimestepEmbedder
│   │
│   ├── Sinusoidal PE (dim=256, max_period=10000)
│   │   t: [B] → [B, 256]
│   │
│   └── MLP: Linear(256 → cond_dim) → SiLU → Linear(cond_dim → cond_dim)
│       出力: [B, cond_dim]
│
├── layers: HeadLayer × 4
│   │
│   │  各 HeadLayer の構造:
│   │  ┌─────────────────────────────────────────────┐
│   │  │ Input x: [B, T, embed_dim]                  │
│   │  │ Condition c: [B, T, cond_dim]                │
│   │  │                                              │
│   │  │ adaLN_modulation(c):                         │
│   │  │   SiLU → Linear(cond_dim → 3*embed_dim)     │
│   │  │   → shift, scale, gate を chunk(3)           │
│   │  │                                              │
│   │  │ 変調: y = norm(x) * (1 + scale) + shift     │
│   │  │ FFN(SwiGLU): gate_proj → SiLU → ×up_proj    │
│   │  │              → down_proj                      │
│   │  │                                              │
│   │  │ 出力: x + gate * FFN(modulate(norm(x)))      │
│   │  └─────────────────────────────────────────────┘
│   │
│   └── embed_dim = latent_size (64)
│       ffn_dim = hidden_size * head_ffn_ratio (768 * 3.0 = 2304)
│       cond_dim = hidden_size (768 for 1.5B)
│
└── final_layer: FinalLayer
    │
    ├── norm_final: RMSNorm (affine なし)
    ├── adaLN_modulation: SiLU → Linear(cond_dim → 2*hidden_size)
    │   → shift, scale を chunk(2)
    └── linear: Linear(hidden_size → latent_size=64, no bias)
        出力: [B, T, 64] (予測ノイズ or 速度)
```

### Forward のデータフロー

```python
def forward(noisy_images, timesteps, condition):
    """
    Args:
        noisy_images: [B, T_latent, 64]    ノイズ付き音声潜在変数
        timesteps:    [B]                   拡散タイムステップ (0~999)
        condition:    [B, T_latent, hidden_size]  LLM隠れ状態

    Returns:
        output:       [B, T_latent, 64]     予測速度 (v_prediction)
    """
    x = noisy_images_proj(noisy_images)      # [B, T, hidden_size]
    t = t_embedder(timesteps)                 # [B, cond_dim]
    c = cond_proj(condition) + t.unsqueeze(1) # [B, T, cond_dim]

    for layer in layers:
        x = layer(x, c)                       # [B, T, embed_dim] (AdaLN + FFN)

    x = final_layer(x, c)                    # [B, T, 64]
    return x
```

### 重み初期化戦略

- **AdaLN の最終層**: ゼロ初期化（変調を恒等写像からスタート）
- **FinalLayer の出力**: ゼロ初期化（予測を0からスタート → 安定した学習初期段階）
- **TimestepEmbedder の MLP**: `normal(std=0.02)`

実装詳細: [diffusion_head.py](diffusion_head.py)

---

## 7. Streaming TTS（リアルタイム音声合成）

### 🔑 キー・イノベーション：ウィンドウベースのストリーミング生成

**問題**: 長いテキスト全体を処理してから音声を出力するのは遅延が大きい

**解決**: テキストをウィンドウ（5トークン）に分割し、ウィンドウごとに音声（6潜在変数）を生成

### モデル分割アーキテクチャ

Streaming モデルでは Qwen2 の Transformer 層を2つに分割：

```
Qwen2.5 全28層
    │
    ├── Lower Layers（下位 8層）: language_model
    │   テキストのみを処理（音声入力なし）
    │   テキストの文脈理解に特化
    │
    └── Upper Layers（上位 20層）: tts_language_model
        テキスト隠れ状態 + 音声潜在変数を処理
        音声生成に特化
        tts_input_types: Embedding(2, hidden_size) で
          テキスト(1) と 音声(0) を区別
```

### ストリーミング生成フロー

```
TTS_TEXT_WINDOW_SIZE = 5   (テキスト5トークンずつ処理)
TTS_SPEECH_WINDOW_SIZE = 6 (テキスト5に対し音声6トークン生成)

テキスト入力: "Hello, this is a test of streaming speech synthesis..."
              ↓ BPEトークナイズ
Token列: [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, t₉, t₁₀, ...]

Window 1: [t₁, t₂, t₃, t₄, t₅]
  │
  ├── Lower LM (8層) → hidden_text₁₋₅
  ├── Upper TTS LM (20層) → EOS予測 + 条件ベクトル
  │
  ├── Speech Token 1: CFG拡散 → z₁ [64]
  │   → Acoustic Connector → embedding → TTS LM に入力
  ├── Speech Token 2: CFG拡散 → z₂ [64]
  │   → Acoustic Connector → embedding → TTS LM に入力
  ├── ... (計6トークン)
  ├── Speech Token 6: CFG拡散 → z₆ [64]
  │
  ├── z₁₋₆ を Acoustic Decoder（ストリーミングキャッシュ付き）で波形に変換
  └── 音声チャンクを AudioStreamer に送信（リアルタイム出力）

Window 2: [t₆, t₇, t₈, t₉, t₁₀]
  │
  ├── Lower LM（Window 1 のKVキャッシュ再利用）
  ...

EOS 検出（BinaryClassifier の logit > 0.5）で生成停止
```

### AudioStreamer

バッチ対応の非同期音声ストリーミングシステム：

```python
AudioStreamer
├── audio_queues: List[Queue]     # バッチサンプルごとにキュー
├── finished_flags: List[bool]    # 完了フラグ
├── put(audio, sample_indices)    # 音声チャンクをキューに追加
├── end(sample_indices)           # 完了信号送信
└── get_stream(sample_idx)        # イテレータ取得
    ├── AudioSampleIterator       # 単一サンプル（ブロッキング）
    └── AudioBatchIterator        # バッチ（ノンブロッキング）

AsyncAudioStreamer               # asyncio 版
├── asyncio.Queue ベース
└── AsyncAudioBatchIterator      # asyncio.wait(FIRST_COMPLETED)
```

実装詳細: [streaming_inference.py](streaming_inference.py)

---

## 8. ASR モデル（音声認識）

### 構造

ASR モデルは TTS モデルと類似するが、**Diffusion Head なし**。音声→テキスト方向の生成に特化。

```
音声入力 [B, T_samples]
    │
    ├── Acoustic Tokenizer Encode → [B, T_latent, 64]
    │   └── Acoustic Connector → [B, T_latent, hidden_size]
    │
    ├── Semantic Tokenizer Encode → [B, T_latent, 128]
    │   └── Semantic Connector → [B, T_latent, hidden_size]
    │
    └── 合算 → [B, T_latent, hidden_size]
         │
         ▼
    テンプレートトークン列:
    [System] + [User: <speech_start>...<speech_end> + コンテキスト] + [Assistant:]
    ↓
    speech token 位置に acoustic+semantic features を挿入
    ↓
    Qwen2.5 LLM (自己回帰テキスト生成)
    ↓
    JSON 出力: {"Start time": "0.00", "End time": "5.32", "Speaker ID": "0", "Content": "..."}
```

### ストリーミングエンコード（長時間音声対応）

60秒を超える音声は自動的にストリーミングエンコードに切り替え：

```python
def encode_speech(speech_tensors, streaming_segment_duration=60.0):
    """
    60秒未満: 一括エンコード
    60秒以上: 60秒セグメントに分割し、ストリーミングキャッシュでエンコード
    """
    if total_samples <= segment_samples:
        # 一括処理
        encoder_output = acoustic_tokenizer.encode(audio)
        audio_tokens = encoder_output.sample()    # [B, T, 64]
    else:
        # ストリーミング処理
        for segment in split_audio(audio, segment_duration=60):
            acoustic_out = acoustic_tokenizer.encode(
                segment, cache=acoustic_cache, use_cache=True
            )
            semantic_out = semantic_tokenizer.encode(
                segment, cache=semantic_cache, use_cache=True
            )
            collect(acoustic_out.mean, semantic_out.mean)

        audio_tokens = concat_and_sample(all_means)  # [B, T_total, 64]
```

### LoRA ファインチューニング

ASR モデルは PEFT (Parameter-Efficient Fine-Tuning) の LoRA でファインチューニング可能：

```python
# finetuning-asr/lora_finetune.py
peft_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    ...
)
# HuggingFace Trainer で学習
trainer = Trainer(model=model, args=training_args, data_collator=collator)
```

実装詳細: [asr_model.py](asr_model.py)

---

## 9. 学習の詳細

### 損失関数

VibeVoice の学習損失は2つの成分から構成：

```
L_total = L_text + L_diffusion

L_text: テキストトークンの Cross-Entropy Loss
  → LLM Head の出力 logits と正解テキストトークンの交差エントロピー

L_diffusion: 拡散損失
  → 音声トークン位置での MSE Loss
```

### 拡散損失の計算

```python
def compute_diffusion_loss(hidden_states, target_features, speech_masks):
    """
    Args:
        hidden_states: [B, T_total, hidden_size]  LLMの隠れ状態
        target_features: [N_speech, 64]            正解音声潜在変数
        speech_masks: [B, T_total] bool            音声トークン位置マスク
    """
    # 1. 条件ベクトル抽出
    condition = hidden_states[speech_masks]           # [N_speech, hidden_size]

    # 2. ddpm_batch_mul 倍のバッチで損失計算（学習効率向上）
    target_expanded = target_features.repeat(ddpm_batch_mul, 1)  # [N*4, 64]
    condition_expanded = condition.repeat(ddpm_batch_mul, 1)     # [N*4, hidden_size]

    # 3. ランダムなタイムステップとノイズ
    timesteps = torch.randint(0, 1000, (N_speech * ddpm_batch_mul,))
    noise = torch.randn_like(target_expanded)                    # [N*4, 64]

    # 4. ノイズ追加
    noisy_latents = scheduler.add_noise(target_expanded, noise, timesteps)

    # 5. 予測
    pred = diffusion_head(noisy_latents, timesteps, condition_expanded)

    # 6. v_prediction のターゲット計算
    target = scheduler.get_velocity(target_expanded, noise, timesteps)

    # 7. MSE損失（正規化）
    loss = F.mse_loss(pred, target) / (latent_size * ddpm_batch_mul)
    #                                   = / (64 * 4)
    return loss
```

### 音声特徴の正規化

学習開始時に動的に計算される正規化パラメータ：

```python
# 最初のバッチで計算（DDP環境対応）
if speech_scaling_factor is NaN:
    valid_tokens = audio_tokens[speech_masks]     # マスクされた有効トークンのみ
    mean_val = valid_tokens.mean()                # グローバル平均
    std_val = valid_tokens.std()                  # グローバル標準偏差

    if torch.distributed.is_initialized():
        # 全GPU間で同期
        all_reduce(mean_val, op=SUM) / world_size
        all_reduce(std_val, op=SUM) / world_size

    speech_bias_factor = -mean_val
    speech_scaling_factor = 1.0 / std_val

# 正規化適用
audio_features = (audio_tokens + speech_bias_factor) * speech_scaling_factor
```

### Curriculum Learning

入力シーケンス長を段階的に増加：
```
Stage 1: max_seq_len = 4,096 トークン
Stage 2: max_seq_len = 8,192
Stage 3: max_seq_len = 16,384
Stage 4: max_seq_len = 32,768
Stage 5: max_seq_len = 65,536 (最終)
```

実装詳細: [loss_and_training.py](loss_and_training.py)

---

## 10. 主要イノベーション

### イノベーション 1: 超低フレームレート σ-VAE Tokenizer

**問題**: 既存コーデック（Encodec 8量子化器 × 75Hz = 600トークン/秒）では長時間音声がLLM文脈長を超える

**解決**:
- σ-VAE で連続潜在変数を直接出力（量子化不要）
- 7段階の深さ方向畳み込みで 3200倍ダウンサンプリング
- 因果畳み込みでストリーミング対応

**効果**:
- 1秒あたりわずか 7.5 トークン（Encodec の 1/80）
- 64Kコンテキストで最大 90分の音声合成が可能
- それでいて PESQ/UTMOS で既存手法を上回る再構成品質

### イノベーション 2: Dual Tokenization（二重トークン化）

**問題**: 単一トークナイザでは音質と意味理解のトレードオフがある

**解決**:
- **Acoustic Tokenizer**: σ-VAE で音響的特徴（音色、韻律、音質）を保存
- **Semantic Tokenizer**: ASR プロキシタスクで言語的意味を抽出
- 両者の特徴を SpeechConnector で LLM 次元に射影し合算

**効果**:
- 低 WER（Whisper: 1.11%, Nemo: 1.82%）→ 正確な発話内容
- 高 SIM（0.692）→ 話者音色の忠実な再現
- 高い主観評価（Realism: 3.71, Richness: 3.81）

### イノベーション 3: Next-Token Diffusion

**問題**: ベクトル量子化（VQ）はコードブック崩壊や情報損失を引き起こす

**解決**:
- LatentLM [SBW+24] の手法を音声に適用
- LLM の各トークン位置の隠れ状態を条件として、軽量 Diffusion Head が連続潜在変数を予測
- DPM-Solver++ で 10~20 ステップの高速推論

**効果**:
- VQ 不要で音質劣化なし
- 4層の軽量ヘッドで計算コスト最小
- CFG による生成品質の制御が可能

### イノベーション 4: ウィンドウベースストリーミング

**問題**: 長いテキスト全体の処理完了を待つと初期遅延が大きい

**解決**:
- Transformer 層を下位（テキストエンコード）と上位（音声生成）に分割
- テキスト5トークンのウィンドウごとに音声6トークンを生成
- KVキャッシュの再利用で効率的なインクリメンタル生成
- BinaryClassifier で EOS を自動検出

**効果**:
- テキスト入力と同時に音声出力を開始（低遅延）
- 非同期 AudioStreamer でリアルタイム再生

---

## 11. 形状ガイド

### 主要テンソル

| ステージ | 名前 | 形状 | 説明 |
|---------|------|------|------|
| 入力 | audio_raw | `[B, T_samples]` | 24kHz モノラル波形 |
| Acoustic Encoder | acoustic_mean | `[B, T_latent, 64]` | σ-VAE の μ出力 |
| Acoustic Sample | acoustic_tokens | `[B, T_latent, 64]` | サンプリング後の潜在変数 |
| Semantic Encoder | semantic_tokens | `[B, T_latent, 128]` | 決定論的意味特徴 |
| Acoustic Connector | acoustic_embed | `[B, T_latent, hidden_size]` | LLM次元に射影 |
| Semantic Connector | semantic_embed | `[B, T_latent, hidden_size]` | LLM次元に射影 |
| テキスト | input_ids | `[B, T_text]` | BPEトークンID |
| テキスト埋め込み | text_embed | `[B, T_text, hidden_size]` | 埋め込みベクトル |
| LLM入力 | combined | `[B, T_total, hidden_size]` | 音声+テキスト統合 |
| LLM出力 | hidden_states | `[B, T_total, hidden_size]` | 隠れ状態 |
| LM Head | logits | `[B, T_total, vocab_size]` | テキスト予測分布 |
| Diffusion入力 | noisy_latents | `[B, T_latent, 64]` | ノイズ付き潜在変数 |
| Diffusion出力 | predicted_v | `[B, T_latent, 64]` | 予測速度 |
| デコーダ出力 | audio_output | `[B, 1, T_samples]` | 復元波形 |

### 次元の意味

| シンボル | 意味 | 典型値 |
|---------|------|--------|
| B | バッチサイズ | 1~16 |
| T_samples | 音声サンプル数 | 24000 × 秒数 |
| T_latent | 潜在トークン数 | ceil(T_samples / 3200) |
| T_text | テキストトークン数 | 可変 |
| T_total | 全トークン数 | T_voice + T_text + T_speech |
| hidden_size | LLM隠れ次元 | 1536 (1.5B) / 3584 (7B) |
| vocab_size | 語彙サイズ | 151,936 (1.5B) / 152,064 (7B) |
| vae_dim_a | Acoustic VAE次元 | 64 |
| vae_dim_s | Semantic VAE次元 | 128 |
| cond_dim | Diffusion条件次元 | = hidden_size |
| latent_size | Diffusion潜在次元 | 64 (= vae_dim_a) |

### モデルサイズ

| コンポーネント | 1.5B モデル | 7B モデル |
|---------------|------------|----------|
| Qwen2.5 LLM | ~1.5B | ~7B |
| Acoustic Tokenizer (Enc) | ~340M | ~340M |
| Acoustic Tokenizer (Dec) | ~340M | ~340M |
| Semantic Tokenizer (Enc) | ~340M | ~340M |
| Diffusion Head | 小 (4層) | 小 (4層) |
| Acoustic Connector | 64→1536 | 64→3584 |
| Semantic Connector | 128→1536 | 128→3584 |

---

## 12. FAQ

### Q1: なぜ σ-VAE を使うのか？通常の VAE と何が違う？

通常の VAE は分散 σ も学習するが、自己回帰モデルで使うと**分散崩壊**（σ→0 に収束し、ほぼ決定論的になる）が起きやすい。σ-VAE は分散を固定分布 N(0, C_σ) からサンプリングすることで、潜在空間の多様性を維持しつつ安定した学習を実現する。

### Q2: 7.5 Hz でなぜ高品質な音声再構成が可能？

6段階のダウンサンプリング（2×2×4×5×5×8=3200倍）を深い残差ブロック（各ステージ3~8ブロック）で行い、情報の損失を最小限に抑えている。また、連続潜在変数（64次元ベクトル）を使うことで、離散コードブックの情報ボトルネックを回避している。

### Q3: Acoustic Tokenizer と Semantic Tokenizer の違いは？

- **Acoustic**: 音声の音響的特徴（音色、韻律、周波数特性）を VAE で圧縮。再構成可能な双方向構造。
- **Semantic**: 音声の言語的内容（何を言っているか）を抽出。ASR タスクで学習したエンコーダのみ。デコーダは学習後破棄。

### Q4: Diffusion Head が4層で十分なのはなぜ？

LLM が既に音声の高レベルな構造（テキスト内容、韻律、話者性）を理解しているため、Diffusion Head は LLM の隠れ状態を条件として **残りの音響的ディテール**を補完するだけでよい。これにより少ない層数でも十分な品質が得られる。

### Q5: CFG (Classifier-Free Guidance) はどのように使われる？

推論時、Diffusion Head に「LLM隠れ状態あり（条件付き）」と「ゼロベクトル（無条件）」の2つの予測を行い、条件付き方向を強調する：
```
v = v_uncond + cfg_scale × (v_cond - v_uncond)
```
cfg_scale=1.3 で、テキスト内容に忠実かつ高品質な音声を生成。

### Q6: ストリーミング推論のウィンドウサイズはなぜ 5:6？

テキスト5トークンに対して音声6トークン（約0.8秒）を生成。speech-to-text トークン比が約2:1であることと、音声トークンのオーバーヘッドを考慮した設計。ウィンドウが小さすぎると品質が低下し、大きすぎると遅延が増す。

### Q7: ASR モデルはなぜ Diffusion Head が不要？

ASR は音声→テキスト方向の変換であり、音声の潜在変数を**生成**する必要がない。Acoustic/Semantic Tokenizer でエンコードした特徴を LLM に入力し、テキストトークンを自己回帰的に生成するだけで十分。

### Q8: 学習時の正規化（scaling_factor, bias_factor）はなぜ動的に計算？

音声データの統計量はデータセットに依存するため、学習開始時の最初のバッチで動的に計算される。DDP（Distributed Data Parallel）環境では全 GPU 間で同期して一貫した正規化を保証する。これにより、異なるデータセットでの再学習が容易になる。

### Q9: なぜ Qwen2.5 を LLM バックボーンに選んだのか？

Qwen2.5 は多言語対応（50+言語）が強く、テキスト理解能力が高い。1.5B と 7B のスケールを提供し、用途に応じた選択が可能。7B モデルはより豊かな音色、自然なイントネーション、クロスリンガル能力を示す。

### Q10: 90分の音声を本当に1回の推論で生成できるのか？

はい。7.5 Hz のトークナイザにより、90分 = 5400秒 → 5400 × 7.5 = 40,500 音声トークン。音声プロンプトとテキストを含めても 64K コンテキストウィンドウ内に収まる。ただし、Streaming モデルを使用する場合はウィンドウ単位で逐次生成するため、メモリ効率も良い。

---

## 13. まとめ

VibeVoice は以下の3つの柱で長時間・複数話者会話音声合成を実現：

1. **超低フレームレート σ-VAE Tokenizer**（7.5 Hz, 3200倍圧縮）により、90分の音声を64Kトークン内で処理可能にしつつ、PESQ/UTMOS で最高品質を達成
2. **Dual Tokenization + Next-Token Diffusion** により、VQ を使わず連続潜在変数で高忠実度の音声生成を実現
3. **ウィンドウベースストリーミング** により、リアルタイムの音声合成を低遅延で提供

1.5B→7B のスケーリングで主観評価が大幅に向上し、Gemini 2.5 Pro Preview TTS や Elevenlabs v3 alpha を含む全ての既存システムを上回る性能を達成。

---

## 14. 参考文献

- [VibeVoice Technical Report](https://arxiv.org/abs/2508.19205) - Peng et al., Microsoft Research, 2025
- [LatentLM (SBW+24)](https://arxiv.org/abs/2412.08635) - Sun et al., 2024 (Next-Token Diffusion の元手法)
- [DAC (KSL+23)](https://proceedings.neurips.cc/paper_files/paper/2023/) - Kumar et al., NeurIPS 2023 (Acoustic Tokenizer の学習手法)
- [Qwen2.5 (YYZ+24)](https://arxiv.org/abs/2412.15115) - Yang et al., 2024 (LLM バックボーン)
- [DPM-Solver++ (LZB+22)](https://arxiv.org/abs/2206.00927) - Lu et al., NeurIPS 2022 (拡散サンプラー)
- [DPM-Solver++ Guided (LZB+25)](https://arxiv.org/abs/2305.08891) - Lu et al., 2025
- [公式 GitHub リポジトリ](https://github.com/microsoft/VibeVoice)
