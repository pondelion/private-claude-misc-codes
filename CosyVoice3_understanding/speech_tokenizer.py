"""
CosyVoice3 Speech Tokenizer (FSQ-MinMo) - 簡略化疑似コード
=============================================================

音声波形を離散トークンに変換するモジュール。
MinMo (大規模マルチモーダル音声理解モデル, 140万時間で事前学習) をベースに、
Finite Scalar Quantization (FSQ) を挿入して学習。

論文: CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
公式実装: cosyvoice/llm/llm.py (SpeechTokenExtractor クラス)

CosyVoice2との違い:
- CosyVoice2: SenseVoice-Large (ASRエンコーダ) にFSQ挿入
- CosyVoice3: MinMo (マルチモーダルLLM, 1.4M時間事前学習) にFSQ挿入

学習タスク (マルチタスク学習, 計53万時間):
1. ASR (多言語音声認識): 36.5万時間
2. LID (言語識別): 8.5万時間
3. SER (感情認識): 4.8万時間
4. AED (音響イベント検出): 2.1万時間
5. SA (話者分析): 1.1万時間

Shape Convention
============================================================
B: バッチサイズ
T_samples: 入力波形のサンプル数
T_frames: トークンフレーム数 (= T_samples / sample_rate * 25)
D_enc: エンコーダ隠れ次元 (MinMoの内部次元)
D_fsq: FSQ低ランク空間次元 (D)
K: FSQ量子化レベル (各次元 [-K, K] に量子化)
Q: 音声トークン語彙サイズ (= (2K+1)^D = 6561)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpeechTokenizer(nn.Module):
    """
    FSQ-MinMoベースのSpeech Tokenizer

    アーキテクチャ全体像:
    ┌──────────────────────────────────────────────────────┐
    │ 音声波形 (1, T_samples)                               │
    │     ↓                                                │
    │ Voice Encoder_1 (12層 Transformer + RoPE)             │
    │     ↓                                                │
    │ H: 中間表現 (B, T_frames, D_enc)                      │
    │     ↓                                                │
    │ ┌────────────────────────────┐                        │
    │ │ FSQ Module                 │                        │
    │ │  Proj_down: D_enc → D_fsq  │                        │
    │ │  ROUND: 各次元を [-K, K] に │                        │
    │ │  Proj_up: D_fsq → D_enc    │                        │
    │ └────────────────────────────┘                        │
    │     ↓                                                │
    │ H_hat: 量子化表現 (B, T_frames, D_enc)                │
    │     ↓                                                │
    │ [学習時のみ]                                           │
    │ Voice Encoder_2 + MinMo LLM → マルチタスク予測         │
    │     ↓                                                │
    │ 音声トークン mu: (B, T_frames)                         │
    │   各フレームは [0, Q-1] の整数 (Q = 6561)              │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_enc: int = 1024,          # エンコーダ隠れ次元 (MinMo内部)
        d_fsq: int = 5,             # FSQ低ランク次元 D
        k_levels: int = 4,          # 各次元の量子化レベル [-K, K], K=4 → 9値
        num_encoder1_layers: int = 12,  # Voice Encoder_1 のレイヤー数
        frame_rate: int = 25,        # トークンフレームレート (25Hz)
    ):
        super().__init__()

        self.d_fsq = d_fsq
        self.k_levels = k_levels
        self.frame_rate = frame_rate

        # 語彙サイズ: (2K+1)^D
        # K=4, D=5 の場合: 9^5 = 59049 (論文では別パラメータで6561)
        # 実際の設定: (2K+1)^D = 6561
        self.vocab_size = (2 * k_levels + 1) ** d_fsq

        # ========================================
        # Voice Encoder_1: 12層 Transformer + RoPE
        # ========================================
        # MinMoの前半部分 (frozen or fine-tuned)
        self.voice_encoder_1 = VoiceEncoder(
            num_layers=num_encoder1_layers,
            d_model=d_enc,
            use_rope=True,  # Rotary Position Embedding
        )
        # 入力: 音声特徴 (B, T_frames, D_enc)
        # 出力: H (B, T_frames, D_enc)

        # ========================================
        # FSQ Module: Finite Scalar Quantization
        # ========================================
        self.fsq = FiniteScalarQuantizer(
            d_enc=d_enc,
            d_fsq=d_fsq,
            k_levels=k_levels,
        )
        # 入力: H (B, T_frames, D_enc)
        # 出力: H_hat (B, T_frames, D_enc), H_bar (B, T_frames, D_fsq)

        # ========================================
        # [学習時のみ] Voice Encoder_2 + MinMo LLM
        # ========================================
        # マルチタスク予測のための後段モジュール
        # 推論時は使用しない (破線ブロック)
        self.voice_encoder_2 = None  # 学習時のみ
        self.minmo_llm = None        # 学習時のみ

    def tokenize(
        self,
        audio: torch.Tensor,     # (B, T_samples)
    ) -> torch.Tensor:
        """
        音声波形を離散トークン列に変換 (推論時)

        入力:
            audio: (B, T_samples) - 24kHz音声波形
                B: バッチサイズ
                T_samples: サンプル数

        出力:
            tokens: (B, T_frames) - 離散音声トークン
                T_frames = T_samples / 24000 * 25
                各値は [0, vocab_size-1] の整数

        実装メモ:
            実際の推論ではONNXモデル (SpeechTokenExtractor) として実行
        """
        # Step 1: Voice Encoder_1 で中間表現を生成
        H = self.voice_encoder_1(audio)
        # H: (B, T_frames, D_enc)
        #   T_frames = T_samples / sample_rate * frame_rate
        #   例: 3秒 (72000サンプル) → 75フレーム

        # Step 2: FSQ で量子化
        H_hat, H_bar = self.fsq(H)
        # H_hat: (B, T_frames, D_enc) - 量子化後の表現 (元次元に復元)
        # H_bar: (B, T_frames, D_fsq) - 低ランク量子化ベクトル (整数値)

        # Step 3: 量子化ベクトルをトークンインデックスに変換
        tokens = self.fsq.encode_to_index(H_bar)
        # tokens: (B, T_frames)
        #   各値は [0, vocab_size-1] の整数

        return tokens

    def forward(
        self,
        audio: torch.Tensor,         # (B, T_samples)
        text_labels: torch.Tensor,   # (B, L_text) - ASR用正解テキスト
        emotion_labels: torch.Tensor, # (B,) - 感情ラベル
        language_labels: torch.Tensor, # (B,) - 言語ラベル
    ) -> Dict:
        """
        学習時のフォワードパス (マルチタスク学習)

        ※ Straight-Through Estimation (STE) で勾配を近似
        """
        # Voice Encoder_1
        H = self.voice_encoder_1(audio)
        # H: (B, T_frames, D_enc)

        # FSQ量子化 (STE: 勾配は量子化をスキップして伝播)
        H_hat, H_bar = self.fsq(H)
        # H_hat: (B, T_frames, D_enc) - 量子化表現
        # 学習時: H_hat = H + sg(ROUND(Proj_down(H)) - Proj_down(H))
        #         sg = stop gradient (straight-through)

        # Voice Encoder_2 + MinMo LLM (マルチタスク予測)
        predictions = self.voice_encoder_2_and_llm(H_hat)
        # predictions: Dict containing:
        #   'asr_logits': (B, L_text, vocab_text)  - ASR予測
        #   'emotion_logits': (B, num_emotions)     - 感情分類
        #   'language_logits': (B, num_languages)   - 言語識別
        #   'event_logits': (B, num_events)         - 音響イベント検出
        #   'speaker_logits': (B, num_speakers)     - 話者分析

        # 各タスクのロス計算
        losses = compute_multitask_losses(predictions, text_labels,
                                          emotion_labels, language_labels)
        return losses


class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ)

    連続表現を有限の離散値に量子化するモジュール。
    VQ-VAEのcodebook lookupと異なり、各次元を独立に丸め量子化。

    数式:
        H_bar = ROUND(Proj_down(H))           ... (1)
        H_hat = Proj_up(H_bar)

    トークンインデックス:
        mu_i = Σ_{j=0}^{D-1} h_bar_{i,j} × (2K+1)^j    ... (2)
        (D次元の(2K+1)進数 → 単一整数)
    """

    def __init__(
        self,
        d_enc: int = 1024,     # エンコーダ隠れ次元
        d_fsq: int = 5,        # 低ランク空間次元 D
        k_levels: int = 4,     # 量子化レベル [-K, K]
    ):
        super().__init__()

        self.d_fsq = d_fsq
        self.k_levels = k_levels
        self.vocab_size = (2 * k_levels + 1) ** d_fsq

        # 射影行列
        self.proj_down = nn.Linear(d_enc, d_fsq)    # D_enc → D_fsq
        self.proj_up = nn.Linear(d_fsq, d_enc)      # D_fsq → D_enc

    def forward(
        self,
        H: torch.Tensor,  # (B, T, D_enc) - エンコーダ出力
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FSQ量子化

        入力:
            H: (B, T, D_enc) - Voice Encoder_1 の出力
                B: バッチサイズ
                T: フレーム数
                D_enc: エンコーダ隠れ次元

        出力:
            H_hat: (B, T, D_enc) - 量子化後の表現 (元次元に復元)
            H_bar: (B, T, D_fsq) - 低ランク量子化ベクトル (整数値)

        処理:
            1. Proj_down: (B, T, D_enc) → (B, T, D_fsq)
            2. ROUND + clamp [-K, K]: (B, T, D_fsq)
            3. Proj_up: (B, T, D_fsq) → (B, T, D_enc)
        """
        # Step 1: 低ランク空間に射影
        H_low = self.proj_down(H)
        # H_low: (B, T, D_fsq)

        # Step 2: 丸め量子化 + 範囲制限
        H_bar = torch.round(H_low)
        H_bar = torch.clamp(H_bar, -self.k_levels, self.k_levels)
        # H_bar: (B, T, D_fsq) - 各値は {-K, ..., -1, 0, 1, ..., K} の整数

        # Straight-Through Estimation (学習時)
        # 順伝播: H_bar (量子化値) を使用
        # 逆伝播: 量子化をスキップして勾配を直接伝播
        H_bar_ste = H_low + (H_bar - H_low).detach()

        # Step 3: 元の次元に復元
        H_hat = self.proj_up(H_bar_ste)
        # H_hat: (B, T, D_enc)

        return H_hat, H_bar

    def encode_to_index(
        self,
        H_bar: torch.Tensor,  # (B, T, D_fsq) - 量子化ベクトル
    ) -> torch.Tensor:
        """
        量子化ベクトル → トークンインデックス変換

        数式 (Equation 2):
            mu_i = Σ_{j=0}^{D-1} (h_bar_{i,j} + K) × (2K+1)^j

        入力:
            H_bar: (B, T, D_fsq) - 量子化ベクトル, 各値 ∈ [-K, K]

        出力:
            indices: (B, T) - トークンインデックス, 各値 ∈ [0, Q-1]
                     Q = (2K+1)^D = 6561

        例 (D=5, K=4):
            H_bar[i] = [-2, 0, 3, 1, -4]
            → shifted = [2, 4, 7, 5, 0]  (K=4を加算)
            → index = 2×9^0 + 4×9^1 + 7×9^2 + 5×9^3 + 0×9^4
                    = 2 + 36 + 567 + 3645 + 0 = 4250
        """
        base = 2 * self.k_levels + 1  # 9

        # [-K, K] → [0, 2K] にシフト
        H_shifted = H_bar + self.k_levels
        # H_shifted: (B, T, D_fsq) - 各値 ∈ [0, 2K]

        # (2K+1)進数 → 10進数に変換
        powers = torch.pow(
            base,
            torch.arange(self.d_fsq, device=H_bar.device).float()
        )
        # powers: (D_fsq,) = [1, 9, 81, 729, 6561, ...]

        indices = (H_shifted * powers).sum(dim=-1).long()
        # indices: (B, T) - 各値 ∈ [0, Q-1]

        return indices


class VoiceEncoder(nn.Module):
    """
    Voice Encoder_1: MinMoの前半エンコーダ部分

    12層のTransformer + Rotary Position Embedding (RoPE)

    入力: audio_features (B, T_frames, D_enc) - 音声フロントエンド特徴
    出力: H (B, T_frames, D_enc) - 中間表現
    """

    def __init__(
        self,
        num_layers: int = 12,
        d_model: int = 1024,
        num_heads: int = 16,
        use_rope: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio: torch.Tensor,  # (B, T_samples) or (B, T_frames, D_enc)
    ) -> torch.Tensor:
        """
        入力:
            audio: (B, T_samples) - 生波形 or
                   (B, T_frames, D_enc) - フロントエンド特徴

        出力:
            H: (B, T_frames, D_enc)
                T_frames = T_samples / sample_rate * frame_rate (25Hz)
                D_enc = 1024
        """
        # フロントエンド (FBank等) で特徴抽出 → (B, T_frames, D_enc)
        x = self.frontend(audio)

        # 12層のTransformer
        for layer in self.layers:
            x = layer(x)
            # x: (B, T_frames, D_enc)

        return x


class TransformerBlock(nn.Module):
    """Transformer Block with RoPE"""
    def __init__(self, d_model: int, num_heads: int, use_rope: bool):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
        # 出力: (B, T, D)
