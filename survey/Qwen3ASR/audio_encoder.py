"""
Qwen3-ASR - AuT Audio Encoder 詳細
====================================

このファイルはQwen3-ASRのAuT (Audio Transformer) Encoderの
詳細な処理フローを理解するための疑似コードです。

論文: https://arxiv.org/abs/2601.21337
公式実装: qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

処理フロー:
1. メルスペクトログラム入力 (128 mel bins, 100Hz)
2. 3段階Conv2dダウンサンプリング (stride=2 each → 8倍)
3. Linear射影 + 正弦波位置エンコーディング
4. 32層Transformer Encoder (Self-Attention + FFN)
5. proj1 (Linear + GELU) → proj2 (Linear)
6. 出力: 12.5Hz音声表現 (D_aut_out次元)

============================================================
Shape Convention
============================================================
B:            バッチサイズ (Audio Encoderでは常に1)
D_mel:        メル周波数ビン数 (128)
T_mel:        メルスペクトログラムフレーム数 (100Hz)
T_conv:       Conv2d出力の時間フレーム数 (≈ T_mel // 8)
D_conv:       Conv2d出力チャネル数 (480)
F_conv:       Conv2d後の周波数次元 (16)
D_model:      Encoder内部次元 (1280 for 1.7B / 896 for 0.6B)
D_out:        Encoder出力次元 (3584 for 1.7B / ? for 0.6B)
N_heads:      アテンションヘッド数 (20 for 1.7B)
D_head:       ヘッド次元 (= D_model / N_heads = 64)
D_ffn:        FFN中間次元 (5120 for 1.7B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# 1. Audio Encoder 全体
# ============================================================

class Qwen3ASRAudioEncoder(nn.Module):
    """
    AuT Audio Encoder

    ========================================
    アーキテクチャ概要
    ========================================
    AuT (Audio Transformer) は AED (Attention-Encoder-Decoder) ベースの
    音声エンコーダで、メルスペクトログラムを12.5Hzの音声表現に変換する。

    ・約4,000万時間の擬似ラベル付きデータで事前学習
    ・動的Flash Attentionウィンドウ (1秒〜8秒)
    ・ストリーミング/オフライン統一推論を実現

    ========================================
    モデルサイズ
    ========================================
    1.7Bモデル: 300M params, d_model=1280, 32 layers, 20 heads
    0.6Bモデル: 180M params, d_model=896, 32 layers, 14 heads
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ========================================
        # Conv2dダウンサンプリング (3段階, stride=2 each → 8倍)
        # ========================================
        # Conv2d_0: (1, D_mel, T_mel) → (D_conv, D_mel//2, T_mel//2)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.downsample_hidden_size,  # 480
            kernel_size=3, stride=2, padding=1,
        )
        # Conv2d_1: (D_conv, D_mel//2, T_mel//2) → (D_conv, D_mel//4, T_mel//4)
        self.conv2 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,   # 480
            out_channels=config.downsample_hidden_size,  # 480
            kernel_size=3, stride=2, padding=1,
        )
        # Conv2d_2: (D_conv, D_mel//4, T_mel//4) → (D_conv, D_mel//8, T_mel//8)
        self.conv3 = nn.Conv2d(
            in_channels=config.downsample_hidden_size,   # 480
            out_channels=config.downsample_hidden_size,  # 480
            kernel_size=3, stride=2, padding=1,
        )
        self.gelu = nn.GELU()

        # ========================================
        # Linear射影: flatten(D_conv × F_conv) → D_model
        # ========================================
        # F_conv = ((D_mel + 1) // 2 + 1) // 2 ... の繰り返し → 16
        # D_conv × F_conv = 480 × 16 = 7680
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * (config.num_mel_bins // 8),  # 480 × 16 = 7680
            config.d_model,  # 1280
        )

        # ========================================
        # 正弦波位置エンコーディング
        # ========================================
        self.embed_positions = SinusoidsPositionEmbedding(
            config.d_model,  # 1280
        )

        # ========================================
        # Transformer Encoder (32層)
        # ========================================
        self.layers = nn.ModuleList([
            Qwen3ASRAudioEncoderLayer(config)
            for _ in range(config.encoder_layers)  # 32
        ])

        # 出力LayerNorm
        self.layer_norm = nn.LayerNorm(config.d_model)  # 1280

        # ========================================
        # 出力射影 (2段階)
        # ========================================
        self.proj1 = nn.Linear(config.d_model, config.d_model)    # 1280 → 1280
        self.proj2 = nn.Linear(config.d_model, config.output_dim)  # 1280 → 3584

        # チャンク処理パラメータ
        self.n_window = config.n_window            # 100 (学習時チャンクサイズ)
        self.n_window_infer = config.n_window_infer  # 400 (推論時チャンクサイズ)

    def forward(
        self,
        input_features: torch.Tensor,      # (B, D_mel=128, T_mel)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T_mel)
    ) -> torch.Tensor:
        """
        Audio Encoder Forward Pass

        ========================================
        Shape
        ========================================
        入力:
            input_features: (B, D_mel, T_mel)
                - B: バッチサイズ (通常1、個別処理)
                - D_mel: 128 mel周波数ビン
                - T_mel: メルスペクトログラムフレーム数 (100Hz)

        出力:
            audio_features: (B, T_conv, D_out)
                - T_conv: ≈ T_mel // 8 (12.5Hz)
                - D_out: 3584 (1.7Bモデル)

        ========================================
        処理詳細
        ========================================
        """
        B = input_features.shape[0]

        # ========================================
        # Stage 1: Conv2d ダウンサンプリング (×3)
        # ========================================
        # 入力を4D (B, 1, D_mel, T_mel) にreshape
        x = input_features.unsqueeze(1)  # (B, 1, 128, T_mel)

        # Conv2d_0: stride=2 → 周波数・時間を半分に
        x = self.gelu(self.conv1(x))
        # x: (B, 480, 64, T_mel//2)
        # 128 → (128-3+2*1)//2 + 1 = 64
        # T_mel → (T_mel-3+2*1)//2 + 1 ≈ T_mel//2

        # Conv2d_1: stride=2 → さらに半分
        x = self.gelu(self.conv2(x))
        # x: (B, 480, 32, T_mel//4)

        # Conv2d_2: stride=2 → さらに半分
        x = self.gelu(self.conv3(x))
        # x: (B, 480, 16, T_mel//8)
        # 最終的に周波数: 128 → 64 → 32 → 16
        # 時間: T_mel → T_mel//2 → T_mel//4 → T_mel//8

        # ========================================
        # Stage 2: Flatten + Linear射影
        # ========================================
        # (B, 480, 16, T_mel//8) → (B, T_mel//8, 480×16=7680)
        T_conv = x.shape[3]  # T_mel // 8
        x = x.permute(0, 3, 1, 2)  # (B, T_mel//8, 480, 16)
        x = x.reshape(B, T_conv, -1)  # (B, T_mel//8, 7680)

        # Linear射影: 7680 → 1280
        x = self.conv_out(x)
        # x: (B, T_mel//8, 1280)

        # ========================================
        # Stage 3: チャンク分割 + 位置エンコーディング
        # ========================================
        # 推論時はn_window_infer=400フレーム (≈32秒分) ごとにチャンク分割
        # 各チャンクに独立した位置エンコーディングを付与
        n_window = self.n_window_infer  # 400

        # チャンク数の計算
        num_chunks = (T_conv + n_window - 1) // n_window

        # 各チャンクに位置エンコーディングを追加
        chunks = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * n_window
            end = min(start + n_window, T_conv)
            chunk = x[:, start:end, :]  # (B, chunk_len, 1280)

            # 正弦波位置エンコーディング (チャンクごとに0からリセット)
            pos_emb = self.embed_positions(chunk)  # (1, chunk_len, 1280)
            chunk = chunk + pos_emb  # (B, chunk_len, 1280)

            chunks.append(chunk)

        # チャンクを結合
        x = torch.cat(chunks, dim=1)
        # x: (B, T_conv, 1280) - 位置エンコーディング付き

        # ========================================
        # Stage 4: 32層 Transformer Encoder
        # ========================================
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        # x: (B, T_conv, 1280)

        # 出力LayerNorm
        x = self.layer_norm(x)
        # x: (B, T_conv, 1280)

        # ========================================
        # Stage 5: 出力射影 (2段階)
        # ========================================
        # proj1: Linear(1280→1280) + GELU
        x = self.gelu(self.proj1(x))
        # x: (B, T_conv, 1280)

        # proj2: Linear(1280→3584)
        x = self.proj2(x)
        # x: (B, T_conv, 3584)

        return x


# ============================================================
# 2. Transformer Encoder Layer
# ============================================================

class Qwen3ASRAudioEncoderLayer(nn.Module):
    """
    Audio Encoder の1レイヤー

    ========================================
    構成
    ========================================
    1. Self-Attention (Pre-LayerNorm)
    2. FFN (Pre-LayerNorm)
    3. 残差接続

    ========================================
    Shape
    ========================================
    入力/出力: (B, T, D_model=1280)
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model  # 1280

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = Qwen3ASRAudioAttention(
            embed_dim=config.d_model,            # 1280
            num_heads=config.encoder_attention_heads,  # 20
            dropout=config.attention_dropout,     # 0.0
        )

        # FFN
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)  # 1280 → 5120
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)  # 5120 → 1280
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, 1280)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            x: (B, T, D_model=1280)

        出力:
            x: (B, T, D_model=1280)

        ========================================
        処理詳細
        ========================================
        """
        # ========================================
        # 1. Self-Attention Block (Pre-LayerNorm + Residual)
        # ========================================
        residual = x
        x = self.self_attn_layer_norm(x)   # (B, T, 1280)
        x = self.self_attn(x, attention_mask=attention_mask)  # (B, T, 1280)
        x = residual + x                   # (B, T, 1280)

        # ========================================
        # 2. FFN Block (Pre-LayerNorm + Residual)
        # ========================================
        residual = x
        x = self.final_layer_norm(x)       # (B, T, 1280)
        x = self.activation_fn(self.fc1(x))  # (B, T, 5120)
        x = self.dropout(x)
        x = self.fc2(x)                    # (B, T, 1280)
        x = self.dropout(x)
        x = residual + x                   # (B, T, 1280)

        return x


# ============================================================
# 3. Audio Attention
# ============================================================

class Qwen3ASRAudioAttention(nn.Module):
    """
    Multi-Head Attention for Audio Encoder

    ========================================
    特徴
    ========================================
    - 双方向 (non-causal) attention
    - 動的Flash Attentionウィンドウ対応
      - ストリーミング: 1秒ウィンドウ (≈13フレーム)
      - オフライン: 8秒ウィンドウ (≈100フレーム)
    - Flash Attention v2 対応可能

    ========================================
    Shape
    ========================================
    入力:  (B, T, D_model=1280)
    出力:  (B, T, D_model=1280)
    Q,K,V: (B, N_heads=20, T, D_head=64)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim    # 1280
        self.num_heads = num_heads    # 20
        self.head_dim = embed_dim // num_heads  # 64
        self.scaling = self.head_dim ** -0.5     # 1/8 = 0.125

        # Q, K, V, Out 射影
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # 1280 → 1280
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # 1280 → 1280
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # 1280 → 1280
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # 1280 → 1280

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,      # (B, T, 1280)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            hidden_states: (B, T, D_model=1280)
            attention_mask: (B, T) or None

        中間:
            q: (B, N_heads, T, D_head) = (B, 20, T, 64)
            k: (B, N_heads, T, D_head) = (B, 20, T, 64)
            v: (B, N_heads, T, D_head) = (B, 20, T, 64)
            attn_weights: (B, N_heads, T, T) = (B, 20, T, T)

        出力:
            output: (B, T, D_model=1280)
        """
        B, T, _ = hidden_states.shape

        # ========================================
        # 1. Q, K, V 射影
        # ========================================
        q = self.q_proj(hidden_states)  # (B, T, 1280)
        k = self.k_proj(hidden_states)  # (B, T, 1280)
        v = self.v_proj(hidden_states)  # (B, T, 1280)

        # Reshape to multi-head: (B, T, 1280) → (B, N_heads, T, D_head)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 20, T, 64)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 20, T, 64)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 20, T, 64)

        # ========================================
        # 2. Attention Score計算
        # ========================================
        # Q × K^T / sqrt(d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attn_weights: (B, 20, T, T)

        # マスク適用 (パディング位置を-inf)
        if attention_mask is not None:
            # attention_mask: (B, T) → (B, 1, 1, T) にブロードキャスト
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))

        # ========================================
        # 3. Softmax + Dropout
        # ========================================
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 20, T, T)
        attn_weights = self.dropout(attn_weights)

        # ========================================
        # 4. Attention × V
        # ========================================
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (B, 20, T, 64)

        # ========================================
        # 5. Multi-head結合 + 出力射影
        # ========================================
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embed_dim)
        # attn_output: (B, T, 1280)

        output = self.out_proj(attn_output)
        # output: (B, T, 1280)

        return output


# ============================================================
# 4. 正弦波位置エンコーディング
# ============================================================

class SinusoidsPositionEmbedding(nn.Module):
    """
    正弦波ベースの位置エンコーディング

    ========================================
    数式
    ========================================
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    ========================================
    特徴
    ========================================
    - チャンクごとにposition 0からリセット
    - n_window_infer=400フレームごとに独立した位置情報
    - これにより長い音声でも安定した位置表現

    ========================================
    Shape
    ========================================
    入力:  (B, T, D_model)
    出力:  (1, T, D_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model  # 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            x: (B, T, D_model=1280)
                - T: チャンク内のフレーム数

        出力:
            pos_emb: (1, T, D_model=1280)
                - ブロードキャストで加算可能
        """
        T = x.shape[1]

        # 位置インデックス
        positions = torch.arange(T, dtype=torch.float32, device=x.device)
        # positions: (T,)

        # 周波数計算
        half_dim = self.d_model // 2  # 640
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=x.device) / half_dim
        )
        # freq: (640,)

        # 外積: (T, 1) × (1, 640) → (T, 640)
        angles = positions.unsqueeze(1) * freq.unsqueeze(0)
        # angles: (T, 640)

        # sin/cos交互配置
        pos_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # pos_emb: (T, 1280)

        return pos_emb.unsqueeze(0)  # (1, T, 1280)


# ============================================================
# 入出力shape一覧表
# ============================================================
"""
========================================
Audio Encoder Shape遷移 (1.7Bモデル、10秒音声の例)
========================================

T_mel = 1000 (100Hz × 10秒)

| 段階                  | テンソル名     | Shape                    | 計算                            |
|----------------------|---------------|-------------------------|---------------------------------|
| メルスペクトログラム   | input         | (1, 128, 1000)          | 128 mel bins × 1000 frames      |
| unsqueeze            | x             | (1, 1, 128, 1000)       | チャネル次元追加                   |
| Conv2d_0 + GELU      | x             | (1, 480, 64, 500)       | stride=2: 128→64, 1000→500      |
| Conv2d_1 + GELU      | x             | (1, 480, 32, 250)       | stride=2: 64→32, 500→250        |
| Conv2d_2 + GELU      | x             | (1, 480, 16, 125)       | stride=2: 32→16, 250→125        |
| permute + reshape    | x             | (1, 125, 7680)          | 480×16=7680, T=125              |
| conv_out (Linear)    | x             | (1, 125, 1280)          | 7680→1280                       |
| +Position Embedding  | x             | (1, 125, 1280)          | sin/cos位置情報追加               |
| 32× Encoder Layer    | x             | (1, 125, 1280)          | Self-Attn(20h) + FFN(5120)      |
| LayerNorm            | x             | (1, 125, 1280)          | 正規化                           |
| proj1 + GELU         | x             | (1, 125, 1280)          | Linear + 活性化                  |
| proj2                | output        | (1, 125, 3584)          | 1280→3584                       |

========================================
Conv2dダウンサンプリングの詳細計算
========================================
入力: (B, C_in, H, W) → 出力: (B, C_out, H_out, W_out)
H_out = (H + 2*padding - kernel_size) // stride + 1
      = (H + 2*1 - 3) // 2 + 1
      = (H - 1) // 2 + 1

例 (D_mel=128):
  Conv1: (128-1)//2 + 1 = 64
  Conv2: (64-1)//2 + 1 = 32
  Conv3: (32-1)//2 + 1 = 16

例 (T_mel=1000):
  Conv1: (1000-1)//2 + 1 = 500
  Conv2: (500-1)//2 + 1 = 250
  Conv3: (250-1)//2 + 1 = 125

最終: 128→16 (周波数8倍ダウン), T→T//8 (時間8倍ダウン)

========================================
動的Flash Attentionウィンドウ
========================================
ストリーミング推論時:
  - チャンクサイズ: 2秒 (=25フレーム @ 12.5Hz)
  - ウィンドウ: 1秒 (=13フレーム)
  - 各フレームは前後1秒の範囲のみattend

オフライン推論時:
  - チャンクサイズ: n_window_infer=400フレーム (=32秒)
  - ウィンドウ: 8秒 (=100フレーム)
  - 各フレームは前後8秒の範囲をattend

→ 同一モデルでストリーミング/オフライン統一推論を実現
"""
