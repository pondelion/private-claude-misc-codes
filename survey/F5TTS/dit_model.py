"""
F5-TTS DiTモデル - 簡略化疑似コード
==========================================

DiT (Diffusion Transformer) バックボーンの詳細実装。
対応ファイル: src/f5_tts/model/backbones/dit.py
              src/f5_tts/model/modules.py

主要コンポーネント:
1. TextEmbedding: 文字トークン → ConvNeXt V2 処理 → テキスト埋め込み
2. InputEmbedding: [ノイズmel, 条件mel, テキスト] → 統合埋め込み
3. TimestepEmbedding: フローステップ t → 時刻条件ベクトル
4. DiTBlock (×22): adaLN-zero + Self-Attention (RoPE) + FFN
5. 最終層: AdaLayerNorm_Final + Linear射影

============================================================
Shape Convention
============================================================
B: バッチサイズ
N: シーケンス長 (mel フレーム数)
F: mel周波数ビン数 (= 100)
nt: テキストトークン長
dim: 隠れ次元 (= 1024)
text_dim: テキスト埋め込み次元 (= 512)
heads: Attentionヘッド数 (= 16)
head_dim: ヘッドあたりの次元 (= 64)
ff_dim: FFN中間次元 (= dim * ff_mult = 2048)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# ConvNeXt V2 Block
# ============================================================

class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 ブロック

    テキスト前処理の核心。Depthwise Conv + GRN + GELU で
    テキスト埋め込みを音声モダリティに近い表現に変換。

    ========================================
    Shape
    ========================================
    入力: x (B, N, text_dim)   例: (B, N, 512)
    出力: x (B, N, text_dim)   例: (B, N, 512) 残差接続

    内部:
        Depthwise Conv: (B, text_dim, N) → (B, text_dim, N)  kernel=7
        LayerNorm:      (B, N, text_dim)
        Pointwise1:     (B, N, text_dim) → (B, N, intermediate_dim)
        GELU:           (B, N, intermediate_dim)
        GRN:            (B, N, intermediate_dim)
        Pointwise2:     (B, N, intermediate_dim) → (B, N, text_dim)

    ========================================
    処理詳細
    ========================================
    ConvNeXt V2の主要改善点:
    - GRN (Global Response Normalization): チャネル間の応答正規化
    - Depthwise Conv: チャネルごとの空間畳み込み (kernel=7)
    """

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2

        # Depthwise Conv: 各チャネル独立に空間方向の畳み込み
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding,
            groups=dim, dilation=dilation
        )  # groups=dim → depthwise

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # Pointwise (1×1) Conv
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x (B, N, dim) 例: (B, N, 512)
        出力: x (B, N, dim) 例: (B, N, 512)
        """
        residual = x                          # (B, N, dim)

        x = x.transpose(1, 2)                 # (B, N, dim) → (B, dim, N)
        x = self.dwconv(x)                    # (B, dim, N) → (B, dim, N) depthwise conv
        x = x.transpose(1, 2)                 # (B, dim, N) → (B, N, dim)

        x = self.norm(x)                      # (B, N, dim) LayerNorm
        x = self.pwconv1(x)                   # (B, N, dim) → (B, N, intermediate_dim)
        x = self.act(x)                       # (B, N, intermediate_dim) GELU
        x = self.grn(x)                       # (B, N, intermediate_dim) Global Response Norm
        x = self.pwconv2(x)                   # (B, N, intermediate_dim) → (B, N, dim)

        return residual + x                   # (B, N, dim) 残差接続


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2の核心)

    チャネル間の応答を正規化し、特徴の多様性を促進。

    ========================================
    Shape
    ========================================
    入力: x (B, N, dim)
    出力: x (B, N, dim)

    ========================================
    処理詳細
    ========================================
    Gx = ||x||_2 (dim=1方向のL2ノルム)  → (B, 1, dim)
    Nx = Gx / mean(Gx)                   → (B, 1, dim) 正規化
    out = γ * (x * Nx) + β + x           → 学習可能なスケール+残差
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)           # (B, 1, dim)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)       # (B, 1, dim)
        return self.gamma * (x * Nx) + self.beta + x            # (B, N, dim)


# ============================================================
# Sinusoidal Position Embedding
# ============================================================

class SinusPositionEmbedding(nn.Module):
    """
    正弦波位置埋め込み

    ========================================
    Shape
    ========================================
    入力: x (B,) or (N,) - 位置インデックス or 時刻値
    出力: emb (B, dim) or (N, dim) - 位置埋め込み

    ========================================
    処理詳細
    ========================================
    emb_k = sin/cos(position * exp(-k * log(10000) / (D/2)))
    偶数次元: sin, 奇数次元: cos
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)  # (B, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)   # (B, D)
        return emb


# ============================================================
# Timestep Embedding
# ============================================================

class TimestepEmbedding(nn.Module):
    """
    フローステップ t → 時刻条件ベクトル

    ========================================
    Shape
    ========================================
    入力: timestep (B,) - フローステップ t ∈ [0, 1]
    出力: t_emb (B, dim) - 時刻条件ベクトル

    ========================================
    処理詳細
    ========================================
    t → SinusPositionEmbedding(256) → Linear(256, dim) → SiLU → Linear(dim, dim)
    例: t=0.5 → sin/cos(0.5 * freq) → MLP → (B, 1024)
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),   # 256 → 1024
            nn.SiLU(),
            nn.Linear(dim, dim),              # 1024 → 1024
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        入力: timestep (B,) 例: [0.1, 0.5, 0.9, 0.3]
        出力: t_emb (B, dim) 例: (4, 1024)
        """
        time_hidden = self.time_embed(timestep)  # (B,) → (B, 256)
        time = self.time_mlp(time_hidden)         # (B, 256) → (B, 1024)
        return time


# ============================================================
# Convolutional Position Embedding
# ============================================================

class ConvPositionEmbedding(nn.Module):
    """
    畳み込み位置埋め込み (Voiceboxと同じ設定)

    ========================================
    Shape
    ========================================
    入力: x (B, N, dim) - 入力シーケンス
    出力: x (B, N, dim) - 位置情報付加済み (残差接続)

    ========================================
    処理詳細
    ========================================
    2層のGrouped 1D Convolution:
    Conv1d(dim, dim, kernel=31, groups=16) → Mish → Conv1d → Mish
    kernel=31: 約31フレーム (≈330ms @ 24kHz) の局所的位置情報
    """

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        入力: x (B, N, dim) 例: (B, N, 1024)
        出力: x (B, N, dim) 例: (B, N, 1024)
        """
        x_in = x                                # (B, N, dim)
        x = x.permute(0, 2, 1)                  # (B, N, dim) → (B, dim, N)
        x = self.conv1d(x)                       # (B, dim, N) → (B, dim, N)
        x = x.permute(0, 2, 1)                  # (B, dim, N) → (B, N, dim)
        return x  # 呼び出し側で + x_in する


# ============================================================
# AdaLayerNorm (adaLN-zero)
# ============================================================

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization

    時刻条件 t に応じてLayerNormのスケール/シフトを動的に決定。
    6個の変調パラメータを生成 (Attention用3個 + FFN用3個)。

    ========================================
    Shape
    ========================================
    入力:
        x:   (B, N, dim) - 入力特徴
        emb: (B, dim)    - 時刻条件ベクトル (TimestepEmbeddingの出力)

    出力:
        x_norm:    (B, N, dim) - 正規化+変調済み (Attention入力用)
        gate_msa:  (B, dim)    - Attentionゲート
        shift_mlp: (B, dim)    - FFN用シフト
        scale_mlp: (B, dim)    - FFN用スケール
        gate_mlp:  (B, dim)    - FFNゲート

    ========================================
    処理詳細
    ========================================
    emb → SiLU → Linear(dim, dim*6) → 6分割
    x_norm = LayerNorm(x) * (1 + scale_msa) + shift_msa

    ゼロ初期化: Linear.weight=0, Linear.bias=0
    → 学習初期: scale=0, shift=0, gate=0
    → 恒等写像に近い動作 (学習安定化)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)   # dim → dim*6 (6パラメータ)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        入力: x (B, N, dim), emb (B, dim)
        出力: (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb = self.linear(self.silu(emb))        # (B, dim) → (B, dim*6)

        # 6分割: 各 (B, dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            torch.chunk(emb, 6, dim=1)

        # 正規化 + 変調
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # scale_msa[:, None]: (B, dim) → (B, 1, dim) ブロードキャスト
        # → x: (B, N, dim)

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNorm_Final(nn.Module):
    """
    最終層用AdaLayerNorm (スケール/シフトのみ、ゲートなし)

    ========================================
    Shape
    ========================================
    入力: x (B, N, dim), emb (B, dim)
    出力: x (B, N, dim) 正規化+変調済み
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)    # dim → dim*2 (scale, shift)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))         # (B, dim) → (B, dim*2)
        scale, shift = torch.chunk(emb, 2, dim=1) # 各 (B, dim)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]  # (B, N, dim)
        return x


# ============================================================
# FeedForward Network
# ============================================================

class FeedForward(nn.Module):
    """
    フィードフォワードネットワーク

    ========================================
    Shape
    ========================================
    入力: x (B, N, dim)     例: (B, N, 1024)
    出力: x (B, N, dim)     例: (B, N, 1024)

    内部: dim → dim*mult → dim
    例: 1024 → 2048 → 1024 (ff_mult=2)
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),            # dim → inner_dim
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),            # inner_dim → dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


# ============================================================
# Self-Attention with RoPE
# ============================================================

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE

    ========================================
    Shape
    ========================================
    入力:
        x:    (B, N, dim)    例: (B, N, 1024)
        mask: (B, N) or None  True=有効位置
        rope: tuple(freqs, xpos_scale) RoPE周波数

    出力:
        x:    (B, N, dim)    例: (B, N, 1024)

    内部:
        Q, K, V: (B, heads, N, head_dim) 例: (B, 16, N, 64)
        Attention: softmax(Q @ K^T / sqrt(64)) @ V

    ========================================
    処理詳細
    ========================================
    1. Q, K, V = Linear(x) → reshape → (B, H, N, D_h)
    2. RoPE適用: Q, K の各ヘッドに回転位置埋め込み
    3. Scaled Dot-Product Attention (PyTorch 2.0 SDPA)
    4. 出力射影: (B, N, H*D_h) → Linear → (B, N, dim)
    """

    def __init__(
        self,
        dim: int,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.1,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head     # 16 * 64 = 1024

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # QK正規化 (オプション)
        if qk_norm == 'rms_norm':
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        入力: x (B, N, dim=1024), mask (B, N), rope (freqs, scale)
        出力: x (B, N, dim=1024)
        """
        B, N, _ = x.shape

        # --- QKV射影 ---
        q = self.to_q(x)                                  # (B, N, 1024)
        k = self.to_k(x)                                  # (B, N, 1024)
        v = self.to_v(x)                                  # (B, N, 1024)

        # --- Reshape: (B, N, H*D) → (B, H, N, D) ---
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # (B, 16, N, 64)
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # (B, 16, N, 64)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # (B, 16, N, 64)

        # --- QK正規化 (オプション) ---
        if self.q_norm is not None:
            q = self.q_norm(q)                             # (B, 16, N, 64)
        if self.k_norm is not None:
            k = self.k_norm(k)                             # (B, 16, N, 64)

        # --- RoPE適用 ---
        if rope is not None:
            freqs, xpos_scale = rope
            # q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
            # k_rot = k * cos(θ) + rotate_half(k) * sin(θ)
            q = apply_rotary_pos_emb(q, freqs)             # (B, 16, N, 64)
            k = apply_rotary_pos_emb(k, freqs)             # (B, 16, N, 64)

        # --- Attention計算 ---
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)     # (B, N) → (B, 1, 1, N)
            attn_mask = attn_mask.expand(B, self.heads, N, N)  # (B, 16, N, N)
        else:
            attn_mask = None

        # PyTorch 2.0 Scaled Dot-Product Attention
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )                                                  # (B, 16, N, 64)

        # --- Reshape + 出力射影 ---
        x = x.transpose(1, 2).reshape(B, N, -1)           # (B, N, 1024)
        x = self.to_out(x)                                 # (B, N, 1024)

        # マスク外をゼロに
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def apply_rotary_pos_emb(x, freqs):
    """
    Rotary Position Embedding 適用

    ========================================
    Shape
    ========================================
    入力:
        x: (B, H, N, D) - Q or K
        freqs: (N, D) - 事前計算された回転周波数

    出力:
        x_rot: (B, H, N, D) - 回転位置埋め込み適用済み

    ========================================
    処理詳細
    ========================================
    x_rot = x * cos(θ) + rotate_half(x) * sin(θ)
    θ = position * base_freq^(-2k/D)

    相対位置: dot(q_rot_i, k_rot_j) = f(q, k, i-j)
    → 位置iとjの距離に応じて自動的に減衰
    """
    # 実際にはx_transformers.RotaryEmbeddingを使用
    # freqs: (N, D//2) の cos/sin ペア
    cos_freqs = freqs.cos()  # (N, D//2)
    sin_freqs = freqs.sin()  # (N, D//2)

    # freqsをx.shapeに合わせてブロードキャスト
    cos_freqs = cos_freqs[None, None, :, :]  # (1, 1, N, D//2)
    sin_freqs = sin_freqs[None, None, :, :]  # (1, 1, N, D//2)

    # rotate_half: xの前半と後半を入れ替えて符号反転
    # x = [x1, x2] → rotate_half(x) = [-x2, x1]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    # rotated: (B, H, N, D)

    x_rot = x * cos_freqs.repeat(1, 1, 1, 2) + rotated * sin_freqs.repeat(1, 1, 1, 2)
    # x_rot: (B, H, N, D)
    return x_rot


# ============================================================
# DiT Block
# ============================================================

class DiTBlock(nn.Module):
    """
    DiT Transformer Block (adaLN-zero)

    ========================================
    Shape
    ========================================
    入力:
        x:    (B, N, dim)  例: (B, N, 1024)
        t:    (B, dim)     時刻条件ベクトル
        mask: (B, N) or None
        rope: tuple        RoPE周波数

    出力:
        x:    (B, N, dim)  例: (B, N, 1024)

    ========================================
    処理詳細
    ========================================
    1. AdaLN (pre-norm + 変調)
       x_norm = LayerNorm(x) * (1 + scale_msa) + shift_msa
    2. Self-Attention + RoPE
       attn_out = Attention(x_norm, mask, rope)
    3. ゲート付き残差接続 (Attention)
       x = x + gate_msa * attn_out
    4. FFN pre-norm + 変調
       x_norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
    5. FeedForward
       ff_out = FFN(x_norm)
    6. ゲート付き残差接続 (FFN)
       x = x + gate_mlp * ff_out
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        # AdaLN for attention
        self.attn_norm = AdaLayerNorm(dim)

        # Self-Attention with RoPE
        self.attn = Attention(
            dim=dim, heads=heads, dim_head=dim_head,
            dropout=dropout, qk_norm=qk_norm,
        )

        # LayerNorm for FFN (affine=False, adaLNで変調)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # FeedForward
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        入力: x (B, N, 1024), t (B, 1024), mask (B, N), rope
        出力: x (B, N, 1024)
        """
        # --- 1. AdaLN + Attention ---
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        # norm:      (B, N, dim) 正規化+変調済み
        # gate_msa:  (B, dim)    Attentionゲート
        # shift/scale/gate_mlp: 各 (B, dim) FFN用パラメータ

        attn_output = self.attn(x=norm, mask=mask, rope=rope)  # (B, N, dim)

        # ゲート付き残差 (Attention)
        x = x + gate_msa.unsqueeze(1) * attn_output            # (B, N, dim)
        # gate_msa.unsqueeze(1): (B, dim) → (B, 1, dim)

        # --- 2. AdaLN + FFN ---
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        # (B, N, dim) * (B, 1, dim) + (B, 1, dim) → (B, N, dim)

        ff_output = self.ff(norm)                              # (B, N, dim)

        # ゲート付き残差 (FFN)
        x = x + gate_mlp.unsqueeze(1) * ff_output             # (B, N, dim)

        return x


# ============================================================
# TextEmbedding
# ============================================================

class TextEmbedding(nn.Module):
    """
    テキスト文字トークン → 埋め込み + ConvNeXt V2 前処理

    F5-TTSの核心的イノベーション。
    テキストをConvNeXt V2で前処理してから音声と結合することで、
    意味-音響ギャップを軽減し、学習収束を大幅に加速。

    ========================================
    Shape
    ========================================
    入力:
        text: (B, nt) - 文字トークンインデックス (-1=パディング)
        seq_len: int or (B,) - mel長 (パディング先の長さ)
        drop_text: bool - CFGドロップフラグ

    出力:
        text_embed: (B, N, text_dim) 例: (B, N, 512)
            N = max(seq_len) にパディング済み
            フィラートークン部分はゼロ (mask_padding=True)

    ========================================
    処理詳細
    ========================================
    1. text + 1 (フィラー=0用にオフセット)
    2. Embedding(2547, 512) → (B, N, 512)
    3. Sinusoidal位置埋め込み加算
    4. ConvNeXt V2 Block × 4
       各ブロック: Depthwise Conv(k=7) → LN → Linear → GELU → GRN → Linear
    5. パディング部分をゼロマスク
    """

    def __init__(
        self,
        text_num_embeds: int = 2546,
        text_dim: int = 512,
        mask_padding: bool = True,
        conv_layers: int = 4,
        conv_mult: int = 2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim  # +1: index 0 をフィラートークンに
        )
        self.mask_padding = mask_padding

        # ConvNeXt V2 ブロック群
        if conv_layers > 0:
            self.extra_modeling = True
            # Sinusoidal位置埋め込み (事前計算)
            precompute_max_pos = 8192  # ≈ 87秒 @ 24kHz
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, precompute_max_pos),
                persistent=False,
            )
            # ConvNeXt V2ブロック ×4
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                  for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: torch.Tensor,          # (B, nt)
        seq_len: int,                 # mel長N
        drop_text: bool = False,
    ) -> torch.Tensor:
        """
        入力: text (B, nt) テキストトークン, seq_len: mel長N
        出力: text_embed (B, N, text_dim=512)
        """
        text = text + 1   # フィラーオフセット: -1→0(パディング), 0→1, ...

        max_seq_len = int(seq_len) if not torch.is_tensor(seq_len) else int(seq_len.max())

        # mel長に合わせてカット or パディング
        text = text[:, :max_seq_len]                          # (B, min(nt, N))
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)  # (B, N)

        # パディングマスク
        text_mask = (text == 0)                               # (B, N) True=パディング/フィラー

        # CFGドロップ: テキスト全体をゼロに
        if drop_text:
            text = torch.zeros_like(text)

        # Embedding
        text = self.text_embed(text)                          # (B, N) → (B, N, text_dim=512)

        # ConvNeXt V2前処理
        if self.extra_modeling:
            # Sinusoidal位置埋め込み加算
            freqs = self.freqs_cis[:max_seq_len, :]           # (N, text_dim)
            text = text + freqs                               # (B, N, text_dim)

            # ConvNeXt V2 ×4
            if self.mask_padding:
                text = text.masked_fill(
                    text_mask.unsqueeze(-1), 0.0
                )                                             # パディング部分ゼロ
                for block in self.text_blocks:
                    text = block(text)                         # (B, N, text_dim)
                    text = text.masked_fill(
                        text_mask.unsqueeze(-1), 0.0
                    )                                         # 各ブロック後もゼロマスク
            else:
                text = self.text_blocks(text)                  # (B, N, text_dim)

        return text  # (B, N, text_dim=512)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Sinusoidal位置埋め込みの事前計算

    ========================================
    Shape
    ========================================
    入力: dim (int), end (int) 最大位置数
    出力: freqs (end, dim) - cos/sin結合

    処理:
        freqs[pos, k] = cos/sin(pos / 10000^(2k/dim))
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # (end, dim)


# ============================================================
# InputEmbedding
# ============================================================

class InputEmbedding(nn.Module):
    """
    入力統合埋め込み: [ノイズmel, 条件mel, テキスト埋め込み] → 統合

    ========================================
    Shape
    ========================================
    入力:
        x:          (B, N, F)          ノイズ混合mel (F=100)
        cond:       (B, N, F)          条件mel (非マスク部分)
        text_embed: (B, N, text_dim)   テキスト埋め込み (text_dim=512)

    出力:
        x:          (B, N, dim)        統合埋め込み (dim=1024)

    内部:
        concat: (B, N, F + F + text_dim) = (B, N, 100+100+512) = (B, N, 712)
        Linear: (B, N, 712) → (B, N, 1024)
        ConvPosEmb: (B, N, 1024) → (B, N, 1024) (残差加算)
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        # 入力次元: mel_dim*2 + text_dim = 100*2 + 512 = 712
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)  # 712 → 1024
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, 100) ノイズmel
        cond: torch.Tensor,            # (B, N, 100) 条件mel
        text_embed: torch.Tensor,      # (B, N, 512) テキスト埋め込み
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:
        """
        入力: x (B,N,100), cond (B,N,100), text_embed (B,N,512)
        出力: x (B, N, 1024)
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)                  # CFGドロップ: 条件ゼロ

        # 結合 + 射影
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        # cat: (B, N, 100+100+512) = (B, N, 712)
        # proj: (B, N, 712) → (B, N, 1024)

        # 畳み込み位置埋め込み + 残差
        x = self.conv_pos_embed(x) + x                    # (B, N, 1024)

        return x


# ============================================================
# DiT (Diffusion Transformer) 全体
# ============================================================

class DiT(nn.Module):
    """
    F5-TTS DiT バックボーン

    ========================================
    アーキテクチャ
    ========================================
    TextEmbedding:  text (B, nt) → (B, N, 512)   [ConvNeXt V2 ×4]
    InputEmbedding: [x, cond, text] → (B, N, 1024) [Linear + ConvPosEmb]
    TimestepEmb:    t (B,) → (B, 1024)            [Sinusoidal + MLP]
    DiTBlock ×22:   (B, N, 1024) → (B, N, 1024)   [adaLN + Attn + FFN]
    Final:          (B, N, 1024) → (B, N, 100)     [adaLN_Final + Linear]

    ========================================
    Base Model設定
    ========================================
    dim=1024, depth=22, heads=16, dim_head=64
    ff_mult=2 (FFN: 1024→2048→1024)
    text_dim=512, text_num_embeds=2546, conv_layers=4
    mel_dim=100
    total params: 335.8M (うちConvNeXt V2: 4層, text_dim=512)
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 2,
        mel_dim: int = 100,
        text_num_embeds: int = 2546,
        text_dim: int = 512,
        text_mask_padding: bool = True,
        qk_norm: Optional[str] = None,
        conv_layers: int = 4,
        long_skip_connection: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # --- 1. TimestepEmbedding ---
        self.time_embed = TimestepEmbedding(dim)
        # t (B,) → (B, 1024)

        # --- 2. TextEmbedding (ConvNeXt V2) ---
        self.text_embed = TextEmbedding(
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,    # 4層のConvNeXt V2
        )
        # text (B, nt) → (B, N, 512)

        # --- 3. InputEmbedding ---
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        # [x, cond, text_embed] → (B, N, 1024)

        # --- 4. RoPE ---
        # RotaryEmbedding(dim_head=64) from x_transformers
        # 事前計算された回転周波数

        # --- 5. DiTBlocks ×22 ---
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head,
                ff_mult=ff_mult, dropout=dropout, qk_norm=qk_norm,
            )
            for _ in range(depth)
        ])

        # --- 6. Long Skip Connection (オプション) ---
        # F5-TTS+LongSkip: 第1層→最終層のスキップ接続
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False)
            if long_skip_connection else None
        )

        # --- 7. 最終層 ---
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)   # 1024 → 100

        # --- 8. ゼロ初期化 ---
        self.initialize_weights()

    def initialize_weights(self):
        """
        adaLN-zero初期化

        全DiTBlockのadaLN Linear層をゼロ初期化:
        → 学習初期は gate=0 で恒等写像
        → 段階的に非ゼロを学習して安定開始
        """
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, F=100)  ノイズ混合mel
        cond: torch.Tensor,            # (B, N, F=100)  条件mel
        text: torch.Tensor,            # (B, nt)        テキストトークン
        time: torch.Tensor,            # (B,) or ()     フローステップ
        mask: Optional[torch.Tensor] = None,  # (B, N)  有効位置
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,       # CFG推論モード
        cache: bool = False,           # テキストキャッシュ
    ) -> torch.Tensor:
        """
        DiT フォワードパス

        ========================================
        Shape
        ========================================
        入力:
            x:    (B, N, 100)  ノイズ混合mel (φ_t = (1-t)*x0 + t*x1)
            cond: (B, N, 100)  条件mel (非マスク部分保持)
            text: (B, nt)      テキストトークン
            time: (B,)         フローステップ t ∈ [0, 1]
            mask: (B, N)       有効位置マスク

        出力:
            pred_flow: (B, N, 100) 予測フローベクトル

        cfg_infer=True の場合:
            入力を2倍にバッチ結合 (条件付き + 無条件)
            出力: (2B, N, 100)

        ========================================
        内部処理フロー
        ========================================
        1. TimestepEmbedding: time → t_emb (B, 1024)
        2. TextEmbedding: text → text_embed (B, N, 512) [ConvNeXt V2]
        3. InputEmbedding: [x, cond, text_embed] → x (B, N, 1024)
        4. DiTBlock ×22: x → x (adaLN + RoPE Attn + FFN)
        5. AdaLayerNorm_Final: x → x (最終変調)
        6. proj_out: x → pred_flow (B, N, 100)
        """
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # --- Step 1: 時刻埋め込み ---
        t = self.time_embed(time)                              # (B,) → (B, 1024)

        # --- Step 2-3: 入力埋め込み ---
        if cfg_infer:
            # CFG推論: 条件付き + 無条件をバッチ結合
            # 条件付き: drop_audio=False, drop_text=False
            x_cond = self._get_input_embed(x, cond, text, False, False)   # (B, N, 1024)
            # 無条件: drop_audio=True, drop_text=True
            x_uncond = self._get_input_embed(x, cond, text, True, True)   # (B, N, 1024)

            x = torch.cat((x_cond, x_uncond), dim=0)          # (2B, N, 1024)
            t = torch.cat((t, t), dim=0)                       # (2B, 1024)
            if mask is not None:
                mask = torch.cat((mask, mask), dim=0)          # (2B, N)
        else:
            x = self._get_input_embed(
                x, cond, text, drop_audio_cond, drop_text
            )                                                  # (B, N, 1024)

        # --- Step 4: RoPE事前計算 ---
        # rope = RotaryEmbedding.forward_from_seq_len(seq_len)
        rope = None  # 実際はRotaryEmbeddingから取得

        # --- Step 5: Long Skip (オプション) ---
        if self.long_skip_connection is not None:
            residual = x                                       # (B, N, 1024)

        # --- Step 6: DiTBlocks ×22 ---
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)              # (B, N, 1024)

        # Long Skip接続
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(
                torch.cat((x, residual), dim=-1)
            )                                                  # (B, N, 2048) → (B, N, 1024)

        # --- Step 7: 最終層 ---
        x = self.norm_out(x, t)                                # (B, N, 1024) 最終変調
        output = self.proj_out(x)                              # (B, N, 1024) → (B, N, 100)

        return output  # pred_flow: (B, N, 100)

    def _get_input_embed(self, x, cond, text, drop_audio_cond, drop_text):
        """テキスト埋め込み + 入力統合"""
        seq_len = x.shape[1]
        text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
        # text: (B, nt) → text_embed: (B, N, 512)

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        # [x(100), cond(100), text(512)] → (B, N, 1024)

        return x

    def clear_cache(self):
        """推論後のテキストキャッシュクリア
        CFG推論時にテキスト埋め込みをキャッシュして条件付き/無条件の
        2回のforward間で再利用する。推論完了後にキャッシュを解放。
        """
        self._cached_text_embed = None


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=== F5-TTS DiT Model ===")
    print()
    print("TextEmbedding:")
    print("  text (B, nt) → Embedding(2547, 512)")
    print("  → SinusoidalPosEmb加算")
    print("  → ConvNeXt V2 ×4 [DWConv(k=7) + GRN + GELU]")
    print("  → (B, N, 512)")
    print()
    print("InputEmbedding:")
    print("  concat([x(100), cond(100), text(512)]) = (B, N, 712)")
    print("  → Linear(712, 1024)")
    print("  → ConvPosEmb(k=31, groups=16) + residual")
    print("  → (B, N, 1024)")
    print()
    print("DiTBlock ×22:")
    print("  AdaLN-zero(x, t_emb) → Self-Attention(RoPE, H=16, D=64)")
    print("  → gate_msa * attn + residual")
    print("  → AdaLN-zero → FFN(1024→2048→1024)")
    print("  → gate_mlp * ff + residual")
    print()
    print("Final:")
    print("  AdaLayerNorm_Final(x, t_emb) → Linear(1024, 100)")
    print("  → pred_flow (B, N, 100)")
    print()
    print("パラメータ数:")
    print("  Transformer: 22層, dim=768→1024, 18heads→16heads")
    print("  ConvNeXt V2: 4層, dim=512")
    print("  合計: ~335.8M (F5-TTS Base)")
