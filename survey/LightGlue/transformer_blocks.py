"""
LightGlue - Transformer Blocks
==============================

LightGlueのTransformerブロックを疑似コードで示します。

主要コンポーネント:
1. Learnable Fourier Positional Encoding (Rotary)
2. Self-Attention Block
3. Cross-Attention Block (Bidirectional)
4. Transformer Layer

論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)

============================================================
Shape Convention
============================================================
B: バッチサイズ
M: Image 0 のキーポイント数
N: Image 1 のキーポイント数
C: embed_dim = 256 (特徴記述子の埋め込み次元)
H: num_heads = 4 (Attention head数)
head_dim: C // H = 64 (ヘッドあたりの次元)
pos_dim: 2 (x, y) or 4 (x, y, scale, orientation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


# ============================================================
# Rotary Positional Encoding
# ============================================================

class LearnableFourierPositionalEncoding(nn.Module):
    """
    学習可能なFourier位置エンコーディング (Rotary)

    ========================================
    SuperGlueとの比較
    ========================================

    SuperGlue (絶対位置エンコーディング):
        x = desc + MLP(position)
        問題: 深いレイヤーで位置情報が薄れる

    LightGlue (相対位置エンコーディング, Rotary):
        q' = q * cos(θ) + rotate_half(q) * sin(θ)
        k' = k * cos(θ) + rotate_half(k) * sin(θ)
        利点: 各self-attentionで位置を再注入

    ========================================
    数学的表現
    ========================================

    Rotary Encoding:
        R(p) = diag(R̂(b₁ᵀp), R̂(b₂ᵀp), ..., R̂(b_{d/2}ᵀp))

        R̂(θ) = [[cos θ, -sin θ],
                 [sin θ,  cos θ]]

    Attention score:
        a_{ij} = q_i^T R(p_j - p_i) k_j          ← 論文の定義（相対位置）

        実装では各位置に独立にRを適用:
            q_i' = R(p_i) q_i
            k_j' = R(p_j) k_j

        内積を展開すると:
            q_i'^T k_j' = (R(p_i) q_i)^T (R(p_j) k_j)
                        = q_i^T R(p_i)^T R(p_j) k_j
                        = q_i^T R(-p_i) R(p_j) k_j    ← R は直交行列: R^T = R^{-1} = R(-θ)
                        = q_i^T R(p_j - p_i) k_j       ← 回転の合成則: R(a)R(b) = R(a+b)

        → 差分 p_j - p_i を明示計算せずに相対位置が自動的に表現される

    ========================================
    Shape Summary
    ========================================
    入力: positions (B, N, pos_dim)  ← 正規化座標 [-1, 1]
    出力: encoding (2, B, 1, N, head_dim)  ← [cos, sin] のペア
    """

    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0):
        """
        Args:
            M: pos_dim - 入力位置の次元 (2 for x,y, 4 for x,y,scale,ori)
            dim: head_dim - 出力エンコーディング次元 (64)
            F_dim: Fourier特徴次元 (default: dim = 64)
            gamma: 初期化のスケール
        """
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma

        # 学習可能な基底 (Fourier features)
        # Wr: (pos_dim, F_dim // 2) = (2, 32)
        # 入力座標を異なる周波数で射影
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)

        # 初期化: Fourier特徴のスケールを制御
        nn.init.normal_(self.Wr.weight.data, mean=0, std=gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置エンコーディングの計算

        ========================================
        Shape
        ========================================
        入力:
            x: (B, N, pos_dim)
                - B: バッチサイズ
                - N: キーポイント数
                - pos_dim: 2 (x, y) or 4 (x, y, scale, ori)

        出力:
            encoding: (2, B, 1, N, head_dim)
                - 2: [cos, sin]
                - B: バッチサイズ
                - 1: broadcastable head dimension
                - N: キーポイント数
                - head_dim: 64

        ========================================
        処理フロー
        ========================================
        1. 学習可能な基底に射影: (B, N, pos_dim) -> (B, N, F_dim//2)
        2. cos/sin 計算: (B, N, F_dim//2) -> (B, N, F_dim//2) × 2
        3. stack: (2, B, N, F_dim//2)
        4. unsqueeze: (2, B, 1, N, F_dim//2)
        5. repeat_interleave: (2, B, 1, N, head_dim)
        """
        B, N, pos_dim = x.shape

        # Step 1: 基底への射影
        # x: (B, N, pos_dim) @ Wr: (pos_dim, F_dim//2)
        # -> projected: (B, N, F_dim//2) = (B, N, 32)
        projected = self.Wr(x)

        # Step 2: Fourier特徴
        # cosines: (B, N, F_dim//2) = (B, N, 32)
        cosines = torch.cos(projected)
        # sines: (B, N, F_dim//2) = (B, N, 32)
        sines = torch.sin(projected)

        # Step 3: cos/sinをスタック
        # emb: (2, B, N, F_dim//2) = (2, B, N, 32)
        emb = torch.stack([cosines, sines], dim=0)

        # Step 4: head次元を追加（broadcastのため）
        # emb: (2, B, 1, N, F_dim//2) = (2, B, 1, N, 32)
        emb = emb.unsqueeze(-3)

        # Step 5: 次元を拡張 (各要素を2回繰り返し)
        # (2, B, 1, N, F_dim//2) → (2, B, 1, N, F_dim)
        # これにより rotate_half との組み合わせが可能に
        # emb: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)
        emb = emb.repeat_interleave(2, dim=-1)

        return emb


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    ベクトルの半分を回転

    ========================================
    Shape
    ========================================
    入力:
        x: (..., head_dim) = (..., 64)

    出力:
        x_rotated: (..., head_dim) = (..., 64)

    ========================================
    処理
    ========================================
    入力: x = [x_0, x_1, x_2, x_3, ...]
    出力: [-x_1, x_0, -x_3, x_2, ...]

    これは R̂(θ) の一部を実装:
        R̂(θ) @ [a, b]^T = [a*cos - b*sin, a*sin + b*cos]

    行列形式:
        x_rotated = x * cos(θ) + rotate_half(x) * sin(θ)
    """
    # x: (..., head_dim) -> (..., head_dim//2, 2) = (..., 32, 2)
    x = x.unflatten(-1, (-1, 2))

    # x1, x2: (..., head_dim//2) = (..., 32)
    x1, x2 = x.unbind(dim=-1)

    # stack: (..., head_dim//2, 2) = (..., 32, 2)
    # flatten: (..., head_dim) = (..., 64)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    キャッシュされたRotary embeddingを適用

    ========================================
    Shape
    ========================================
    入力:
        freqs: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)
            - freqs[0]: cos = (B, 1, N, head_dim)
            - freqs[1]: sin = (B, 1, N, head_dim)
        t: (B, H, N, head_dim) = (B, 4, N, 64)
            - query or key

    出力:
        t_rotated: (B, H, N, head_dim) = (B, 4, N, 64)

    ========================================
    数式
    ========================================
    t' = t * cos + rotate_half(t) * sin

    これにより:
        q_i' @ k_j' = q_i^T @ R(p_j - p_i) @ k_j

    つまり相対位置を考慮したattention scoreが計算される
    """
    # cos_freqs: (B, 1, N, head_dim) = (B, 1, N, 64)
    # broadcast to (B, H, N, head_dim) when multiplied with t
    cos_freqs = freqs[0]
    sin_freqs = freqs[1]

    # t * cos_freqs: (B, H, N, head_dim) * (B, 1, N, head_dim) -> (B, H, N, head_dim)
    # rotate_half(t): (B, H, N, head_dim)
    # rotate_half(t) * sin_freqs: (B, H, N, head_dim)
    # result: (B, H, N, head_dim)
    return t * cos_freqs + rotate_half(t) * sin_freqs


# ============================================================
# Attention Module
# ============================================================

class Attention(nn.Module):
    """
    Attention計算モジュール

    FlashAttention対応:
        - torch >= 2.0: scaled_dot_product_attention
        - flash-attn package: FlashCrossAttention

    ========================================
    Shape Summary
    ========================================
    入力:
        q: (B, H, N, head_dim)
        k: (B, H, M, head_dim)
        v: (B, H, M, head_dim)
        mask: (B, H, N, M) (optional)

    出力:
        out: (B, H, N, head_dim)
    """

    def __init__(self, allow_flash: bool = True):
        super().__init__()
        self.enable_flash = allow_flash and self._check_flash_available()

    def _check_flash_available(self) -> bool:
        """FlashAttentionが利用可能か確認"""
        return hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention計算

        ========================================
        Shape
        ========================================
        入力:
            q: (B, H, N, head_dim) = (B, 4, N, 64)
                - query vectors
            k: (B, H, M, head_dim) = (B, 4, M, 64)
                - key vectors
            v: (B, H, M, head_dim) = (B, 4, M, 64)
                - value vectors
            mask: (B, H, N, M) (optional)
                - attention mask

        中間:
            sim: (B, H, N, M)
                - similarity matrix: q @ k^T
            attn: (B, H, N, M)
                - attention weights: softmax(sim)

        出力:
            out: (B, H, N, head_dim) = (B, 4, N, 64)
                - output: attn @ v
        """
        B, H, N_q, head_dim = q.shape
        _, _, N_k, _ = k.shape

        if N_q == 0 or N_k == 0:
            return q.new_zeros(B, H, N_q, v.shape[-1])

        if self.enable_flash and q.device.type == 'cuda':
            # FlashAttention (PyTorch 2.0+)
            # Internally computes: softmax(q @ k^T / sqrt(d)) @ v
            out = F.scaled_dot_product_attention(
                q.contiguous(),  # (B, H, N, head_dim)
                k.contiguous(),  # (B, H, M, head_dim)
                v.contiguous(),  # (B, H, M, head_dim)
                attn_mask=mask   # (B, H, N, M) or None
            )
            # out: (B, H, N, head_dim)
            return out.nan_to_num() if mask is not None else out

        else:
            # 標準実装
            # scale: 1 / sqrt(head_dim)
            scale = head_dim ** -0.5

            # sim = q @ k^T * scale
            # q: (B, H, N, head_dim), k: (B, H, M, head_dim)
            # einsum 'bhid, bhjd -> bhij': (B, H, N, head_dim) @ (B, H, M, head_dim)^T
            # -> sim: (B, H, N, M)
            sim = torch.einsum('bhid, bhjd -> bhij', q, k) * scale

            if mask is not None:
                # sim: (B, H, N, M)
                # mask: (B, H, N, M)
                sim = sim.masked_fill(~mask, float('-inf'))

            # attn = softmax(sim, dim=-1)
            # attn: (B, H, N, M)
            attn = F.softmax(sim, dim=-1)

            # out = attn @ v
            # attn: (B, H, N, M), v: (B, H, M, head_dim)
            # einsum 'bhij, bhjd -> bhid'
            # -> out: (B, H, N, head_dim)
            out = torch.einsum('bhij, bhjd -> bhid', attn, v)
            return out


# ============================================================
# Self-Attention Block
# ============================================================

class SelfBlock(nn.Module):
    """
    Self-Attention Block (with Rotary PE)

    ========================================
    処理フロー
    ========================================

    1. Q, K, V を計算
    2. Rotary PE を Q, K に適用 (位置情報注入)
    3. Attention 計算
    4. 出力投影
    5. FFN + 残差接続

    ========================================
    Shape Summary
    ========================================
    入力:
        x: (B, N, C) = (B, N, 256)
        encoding: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)

    出力:
        x_updated: (B, N, C) = (B, N, 256)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        flash: bool = False,
        bias: bool = True
    ):
        """
        Args:
            embed_dim: C = 256 (特徴記述子の埋め込み次元)
            num_heads: H = 4 (Attention head数)
            flash: FlashAttention使用フラグ
            bias: バイアス使用フラグ
        """
        super().__init__()
        self.embed_dim = embed_dim      # C = 256
        self.num_heads = num_heads      # H = 4
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads  # head_dim = 64

        # Q, K, V を一括で計算 (効率化)
        # Wqkv: (C, 3*C) = (256, 768)
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # Attention module
        self.inner_attn = Attention(flash)

        # 出力投影
        # out_proj: (C, C) = (256, 256)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # FFN: [x, message] → x
        # Linear(2*C, 2*C) + LayerNorm + GELU + Linear(2*C, C)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),  # (512, 512)
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),  # (512, 256)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Self-Attention の forward

        ========================================
        Shape
        ========================================
        入力:
            x: (B, N, C) = (B, N, 256)
                - 特徴記述子の埋め込みベクトル
            encoding: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)
                - Rotary encoding [cos, sin]
            mask: (B, H, N, N) (optional)
                - attention mask

        中間:
            qkv: (B, N, 3*C) = (B, N, 768)
            q, k, v: (B, H, N, head_dim) = (B, 4, N, 64)
            context: (B, H, N, head_dim) = (B, 4, N, 64)
            message: (B, N, C) = (B, N, 256)

        出力:
            x_updated: (B, N, C) = (B, N, 256)
        """
        B, N, C = x.shape
        H = self.num_heads         # H = 4
        head_dim = self.head_dim   # head_dim = 64

        # ========================================
        # Step 1: Q, K, V 計算
        # ========================================
        # x: (B, N, C) @ Wqkv: (C, 3*C)
        # -> qkv: (B, N, 3*C) = (B, N, 768)
        qkv = self.Wqkv(x)

        # Multi-head形式に変形
        # (B, N, 3*C) -> (B, N, H, head_dim, 3) = (B, N, 4, 64, 3)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3))

        # (B, N, H, head_dim, 3) -> (B, H, N, head_dim, 3)
        qkv = qkv.transpose(1, 2)

        # q, k, v: (B, H, N, head_dim) = (B, 4, N, 64)
        q = qkv[..., 0]
        k = qkv[..., 1]
        v = qkv[..., 2]

        # ========================================
        # Step 2: Rotary PE を各位置に独立適用
        # ========================================
        # q_i' = R(p_i) q_i,  k_j' = R(p_j) k_j
        # 内積 q_i'^T k_j' = q_i^T R(p_j - p_i) k_j となり
        # 差分を明示計算せずに相対位置が自動的に反映される (RoPEの性質)
        # encoding: (2, B, 1, N, head_dim)
        # q: (B, H, N, head_dim) -> (B, H, N, head_dim)
        q = apply_cached_rotary_emb(encoding, q)
        # k: (B, H, N, head_dim) -> (B, H, N, head_dim)
        k = apply_cached_rotary_emb(encoding, k)

        # ========================================
        # Step 3: Attention
        # ========================================
        # q: (B, H, N, head_dim), k: (B, H, N, head_dim), v: (B, H, N, head_dim)
        # -> context: (B, H, N, head_dim) = (B, 4, N, 64)
        context = self.inner_attn(q, k, v, mask=mask)

        # ========================================
        # Step 4: 出力投影
        # ========================================
        # context: (B, H, N, head_dim) -> (B, N, H, head_dim)
        context = context.transpose(1, 2)
        # (B, N, H, head_dim) -> (B, N, C) = (B, N, 256)
        context = context.flatten(start_dim=-2)
        # message: (B, N, C) = (B, N, 256)
        message = self.out_proj(context)

        # ========================================
        # Step 5: FFN + 残差接続
        # ========================================
        # [x, message] を concat: (B, N, 2*C) = (B, N, 512)
        concat = torch.cat([x, message], dim=-1)
        # FFN: (B, N, 512) -> (B, N, 256)
        # x_updated = x + ffn([x, message])
        # x_updated: (B, N, C) = (B, N, 256)
        x_updated = x + self.ffn(concat)

        return x_updated


# ============================================================
# Cross-Attention Block (Bidirectional)
# ============================================================

class CrossBlock(nn.Module):
    """
    Bidirectional Cross-Attention Block

    ========================================
    キー・イノベーション
    ========================================

    従来の Cross-Attention:
        A→B: sim_AB = q_A @ k_B^T, attn = softmax(sim_AB) @ v_B
        B→A: sim_BA = q_B @ k_A^T, attn = softmax(sim_BA) @ v_A
        → 2回の類似度計算が必要

    Bidirectional (LightGlue):
        Q = K を共有: qk_A = to_qk(x_A), qk_B = to_qk(x_B)
        sim = qk_A @ qk_B^T  ← 1回だけ!
        A→B: attn = softmax(sim, dim=-1) @ v_B
        B→A: attn = softmax(sim^T, dim=-1)^T @ v_A
        → 計算量 50% 削減

    ========================================
    Shape Summary
    ========================================
    入力:
        x0: (B, M, C) = (B, M, 256)
        x1: (B, N, C) = (B, N, 256)

    出力:
        x0_updated: (B, M, C) = (B, M, 256)
        x1_updated: (B, N, C) = (B, N, 256)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        flash: bool = False,
        bias: bool = True
    ):
        """
        Args:
            embed_dim: C = 256
            num_heads: H = 4
            flash: FlashAttention使用フラグ
            bias: バイアス使用フラグ
        """
        super().__init__()
        self.num_heads = num_heads          # H = 4
        self.head_dim = embed_dim // num_heads  # head_dim = 64
        self.scale = self.head_dim ** -0.5  # 1/sqrt(64) = 0.125

        inner_dim = self.head_dim * num_heads  # C = 256

        # 共有 Query/Key 投影 (Bidirectional の鍵)
        # to_qk: (C, C) = (256, 256)
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)

        # Value は別々
        # to_v: (C, C) = (256, 256)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)

        # 出力投影
        # to_out: (C, C) = (256, 256)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)

        # FFN: [x, message] → x
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),  # (512, 512)
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),      # (512, 256)
        )

        # FlashAttention (optional)
        self.flash = Attention(flash) if flash else None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        """両方のテンソルに同じ関数を適用"""
        return func(x0), func(x1)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional Cross-Attention

        ========================================
        Shape
        ========================================
        入力:
            x0: (B, M, C) = (B, M, 256)
                - Image A の特徴記述子埋め込み
            x1: (B, N, C) = (B, N, 256)
                - Image B の特徴記述子埋め込み
            mask: (B, H, M, N) (optional)
                - attention mask

        中間:
            qk0, qk1: (B, H, M/N, head_dim) = (B, 4, M/N, 64)
            v0, v1: (B, H, M/N, head_dim) = (B, 4, M/N, 64)
            sim: (B, H, M, N)
            attn01: (B, H, M, N)
            attn10: (B, H, N, M)
            m0: (B, H, M, head_dim) -> (B, M, C)
            m1: (B, H, N, head_dim) -> (B, N, C)

        出力:
            x0_updated: (B, M, C) = (B, M, 256)
            x1_updated: (B, N, C) = (B, N, 256)
        """
        B, M, C = x0.shape
        _, N, _ = x1.shape
        H = self.num_heads         # H = 4
        head_dim = self.head_dim   # head_dim = 64

        # ========================================
        # Step 1: Q/K, V 計算
        # ========================================
        # 共有 Query/Key
        # x0: (B, M, C) -> qk0: (B, M, C)
        # x1: (B, N, C) -> qk1: (B, N, C)
        qk0, qk1 = self.map_(self.to_qk, x0, x1)

        # Value は別々
        # v0: (B, M, C), v1: (B, N, C)
        v0, v1 = self.map_(self.to_v, x0, x1)

        # Multi-head 形式に変形
        # (B, M/N, C) -> (B, M/N, H, head_dim) -> (B, H, M/N, head_dim)
        # qk0: (B, H, M, head_dim) = (B, 4, M, 64)
        # qk1: (B, H, N, head_dim) = (B, 4, N, 64)
        # v0: (B, H, M, head_dim), v1: (B, H, N, head_dim)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1)
        )

        # ========================================
        # Step 2: 類似度行列計算 (1回のみ!)
        # ========================================
        if self.flash is not None and qk0.device.type == 'cuda':
            # FlashAttention使用
            # m0: (B, H, M, head_dim)
            m0 = self.flash(qk0, qk1, v1, mask)
            # m1: (B, H, N, head_dim)
            m1 = self.flash(
                qk1, qk0, v0,
                mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            # 標準実装
            # スケーリング (sqrt を分割)
            # qk0_scaled: (B, H, M, head_dim)
            qk0_scaled = qk0 * (self.scale ** 0.5)
            # qk1_scaled: (B, H, N, head_dim)
            qk1_scaled = qk1 * (self.scale ** 0.5)

            # 類似度行列 (1回だけ計算)
            # sim = qk0 @ qk1^T
            # qk0_scaled: (B, H, M, head_dim), qk1_scaled: (B, H, N, head_dim)
            # einsum 'bhid, bhjd -> bhij'
            # -> sim: (B, H, M, N)
            sim = torch.einsum('bhid, bhjd -> bhij', qk0_scaled, qk1_scaled)

            if mask is not None:
                # sim: (B, H, M, N), mask: (B, H, M, N)
                sim = sim.masked_fill(~mask, float('-inf'))

            # ========================================
            # Step 3: 双方向の Attention
            # ========================================
            # A → B
            # attn01 = softmax(sim, dim=-1)
            # attn01: (B, H, M, N)
            attn01 = F.softmax(sim, dim=-1)
            # m0 = attn01 @ v1
            # attn01: (B, H, M, N), v1: (B, H, N, head_dim)
            # einsum 'bhij, bhjd -> bhid'
            # -> m0: (B, H, M, head_dim) = (B, 4, M, 64)
            m0 = torch.einsum('bhij, bhjd -> bhid', attn01, v1)

            # B → A (転置を利用)
            # attn10 = softmax(sim^T, dim=-1)
            # sim.transpose(-2, -1): (B, H, N, M)
            # attn10: (B, H, N, M)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            # m1 = attn10^T @ v0  (逆方向の注意)
            # attn10.transpose(-2, -1): (B, H, M, N) (元に戻す)
            # v0: (B, H, M, head_dim)
            # einsum 'bhji, bhjd -> bhid'
            # -> m1: (B, H, N, head_dim) = (B, 4, N, 64)
            m1 = torch.einsum('bhji, bhjd -> bhid', attn10.transpose(-2, -1), v0)

            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()

        # ========================================
        # Step 4: 出力投影
        # ========================================
        # m0: (B, H, M, head_dim) -> (B, M, H, head_dim) -> (B, M, C)
        # m1: (B, H, N, head_dim) -> (B, N, H, head_dim) -> (B, N, C)
        m0, m1 = self.map_(
            lambda t: t.transpose(1, 2).flatten(start_dim=-2),
            m0, m1
        )
        # m0: (B, M, C), m1: (B, N, C)
        m0, m1 = self.map_(self.to_out, m0, m1)

        # ========================================
        # Step 5: FFN + 残差接続
        # ========================================
        # concat: (B, M, 2*C) = (B, M, 512)
        # x0_updated = x0 + ffn([x0, m0])
        # x0_updated: (B, M, C) = (B, M, 256)
        x0_updated = x0 + self.ffn(torch.cat([x0, m0], dim=-1))
        # x1_updated: (B, N, C) = (B, N, 256)
        x1_updated = x1 + self.ffn(torch.cat([x1, m1], dim=-1))

        return x0_updated, x1_updated


# ============================================================
# Transformer Layer
# ============================================================

class TransformerLayer(nn.Module):
    """
    LightGlue Transformer Layer

    構成:
        Self-Attention (Image A) + Rotary PE
        Self-Attention (Image B) + Rotary PE
        Cross-Attention (A ↔ B) Bidirectional

    ========================================
    Shape Summary
    ========================================
    入力:
        desc0: (B, M, C) = (B, M, 256)
        desc1: (B, N, C) = (B, N, 256)
        encoding0: (2, B, 1, M, head_dim) = (2, B, 1, M, 64)
        encoding1: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)

    出力:
        desc0_updated: (B, M, C) = (B, M, 256)
        desc1_updated: (B, N, C) = (B, N, 256)
    """

    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False):
        """
        Args:
            embed_dim: C = 256
            num_heads: H = 4
            flash: FlashAttention使用フラグ
        """
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads, flash)
        self.cross_attn = CrossBlock(embed_dim, num_heads, flash)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transformer Layer の forward

        ========================================
        Shape
        ========================================
        入力:
            desc0: (B, M, C) = (B, M, 256)
                - Image A の特徴記述子埋め込み
            desc1: (B, N, C) = (B, N, 256)
                - Image B の特徴記述子埋め込み
            encoding0: (2, B, 1, M, head_dim) = (2, B, 1, M, 64)
                - A の Rotary encoding [cos, sin]
            encoding1: (2, B, 1, N, head_dim) = (2, B, 1, N, 64)
                - B の Rotary encoding [cos, sin]
            mask0, mask1: (B, H, M, M), (B, H, N, N) (optional)
                - self-attention masks (for compiled version)

        出力:
            desc0_updated: (B, M, C) = (B, M, 256)
            desc1_updated: (B, N, C) = (B, N, 256)
        """
        if mask0 is not None and mask1 is not None:
            # Compiled version with padding
            return self._masked_forward(
                desc0, desc1, encoding0, encoding1, mask0, mask1
            )
        else:
            # Standard forward
            # Self-attention (with Rotary PE)
            # desc0: (B, M, C) -> (B, M, C)
            desc0 = self.self_attn(desc0, encoding0)
            # desc1: (B, N, C) -> (B, N, C)
            desc1 = self.self_attn(desc1, encoding1)

            # Cross-attention (bidirectional)
            # desc0, desc1: (B, M, C), (B, N, C) -> (B, M, C), (B, N, C)
            desc0, desc1 = self.cross_attn(desc0, desc1)

            return desc0, desc1

    def _masked_forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masked forward (for torch.compile with padding)

        ========================================
        Shape
        ========================================
        入力:
            desc0: (B, M, C)
            desc1: (B, N, C)
            encoding0: (2, B, 1, M, head_dim)
            encoding1: (2, B, 1, N, head_dim)
            mask0: (B, 1, 1, M) - image0の有効点マスク
            mask1: (B, 1, 1, N) - image1の有効点マスク

        中間:
            mask: (B, 1, M, N) - cross-attention mask
            mask0_self: (B, 1, M, M) - self-attention mask for image0
            mask1_self: (B, 1, N, N) - self-attention mask for image1

        出力:
            desc0_updated: (B, M, C)
            desc1_updated: (B, N, C)
        """
        # Cross-attention mask
        # mask0: (B, 1, 1, M), mask1: (B, 1, 1, N)
        # mask: (B, 1, M, N)
        mask = mask0 & mask1.transpose(-1, -2)

        # Self-attention masks (自身の点のみ attend)
        # mask0_self: (B, 1, M, M)
        mask0_self = mask0 & mask0.transpose(-1, -2)
        # mask1_self: (B, 1, N, N)
        mask1_self = mask1 & mask1.transpose(-1, -2)

        # Self-attention
        # desc0: (B, M, C) -> (B, M, C)
        desc0 = self.self_attn(desc0, encoding0, mask0_self)
        # desc1: (B, N, C) -> (B, N, C)
        desc1 = self.self_attn(desc1, encoding1, mask1_self)

        # Cross-attention
        # desc0, desc1: (B, M, C), (B, N, C) -> (B, M, C), (B, N, C)
        return self.cross_attn(desc0, desc1, mask)


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Transformer blocksの使用例

    ========================================
    Shape Summary
    ========================================
    B = 2 (バッチサイズ)
    M = 128 (Image A のキーポイント数)
    N = 100 (Image B のキーポイント数)
    C = 256 (embed_dim)
    H = 4 (num_heads)
    head_dim = 64
    """
    print("=== Transformer Blocks Example ===\n")

    B, M, N = 2, 128, 100
    C = 256         # embed_dim
    H = 4           # num_heads
    head_dim = C // H  # 64

    # 入力データ
    desc0 = torch.randn(B, M, C)  # (2, 128, 256)
    desc1 = torch.randn(B, N, C)  # (2, 100, 256)

    # 位置座標 (正規化済み)
    pos0 = torch.rand(B, M, 2) * 2 - 1  # (2, 128, 2) in [-1, 1]
    pos1 = torch.rand(B, N, 2) * 2 - 1  # (2, 100, 2) in [-1, 1]

    print(f"Input shapes:")
    print(f"  desc0: {desc0.shape}")  # (2, 128, 256)
    print(f"  desc1: {desc1.shape}")  # (2, 100, 256)
    print(f"  pos0: {pos0.shape}")    # (2, 128, 2)
    print(f"  pos1: {pos1.shape}")    # (2, 100, 2)
    print()

    # ========================================
    # Positional Encoding
    # ========================================
    posenc = LearnableFourierPositionalEncoding(M=2, dim=head_dim)
    # pos0: (B, M, 2) -> encoding0: (2, B, 1, M, head_dim)
    encoding0 = posenc(pos0)  # (2, 2, 1, 128, 64)
    encoding1 = posenc(pos1)  # (2, 2, 1, 100, 64)

    print(f"Positional encoding shapes:")
    print(f"  encoding0: {encoding0.shape}")  # (2, 2, 1, 128, 64)
    print(f"  encoding1: {encoding1.shape}")  # (2, 2, 1, 100, 64)
    print()

    # ========================================
    # Transformer Layer
    # ========================================
    layer = TransformerLayer(C, H, flash=False)
    # desc0: (B, M, C), desc1: (B, N, C) -> desc0_out, desc1_out
    desc0_out, desc1_out = layer(desc0, desc1, encoding0, encoding1)

    print(f"TransformerLayer output:")
    print(f"  desc0_out: {desc0_out.shape}")  # (2, 128, 256)
    print(f"  desc1_out: {desc1_out.shape}")  # (2, 100, 256)
    print()

    # ========================================
    # Self-Attention のみ
    # ========================================
    self_attn = SelfBlock(C, H, flash=False)
    # desc0: (B, M, C) -> desc0_self: (B, M, C)
    desc0_self = self_attn(desc0, encoding0)
    print(f"SelfBlock output:")
    print(f"  desc0_self: {desc0_self.shape}")  # (2, 128, 256)
    print()

    # ========================================
    # Cross-Attention のみ
    # ========================================
    cross_attn = CrossBlock(C, H, flash=False)
    # desc0, desc1 -> desc0_cross, desc1_cross
    desc0_cross, desc1_cross = cross_attn(desc0, desc1)
    print(f"CrossBlock output:")
    print(f"  desc0_cross: {desc0_cross.shape}")  # (2, 128, 256)
    print(f"  desc1_cross: {desc1_cross.shape}")  # (2, 100, 256)


if __name__ == "__main__":
    example_usage()
