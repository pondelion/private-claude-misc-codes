"""
Qwen3VL - Vision Encoder
=========================

SigLIP-2 ベースの Vision Encoder と MLP Merger の疑似コードです。

主要コンポーネント:
1. PatchEmbed: 画像パッチを埋め込みベクトルに変換
2. Qwen3VisionBlock: ViT のトランスフォーマーブロック (RMSNorm + Attention + FFN)
3. Qwen3VisionAttention: ViT用アテンション (2D RoPE付き)
4. PatchMerger (MLP Merger): 2×2空間圧縮でトークン数を1/4に削減
5. DeepStack: 中間特徴量の抽出 (各Nブロックごとに保存)

論文: Qwen3-VL Technical Report (2025)

============================================================
Shape Convention
============================================================
N_patches: ViT入力パッチ数 (全バッチ・全画像の合計)
P: パッチサイズ = 14 (px)
C: チャンネル数 = 3 (RGB)
D_v: Vision Encoder隠れ次元 = 1152
D_llm: LLM隠れ次元 = 3584 (7Bモデル)
H_v: Vision Encoderのアテンションヘッド数 = 16
head_dim_v: D_v // H_v = 72
L_v: Vision Encoderのレイヤー数 = 32
merge_size: MLP Mergerの空間圧縮倍率 = 2
N_v: LLM入力の視覚トークン数 = N_patches / (merge_size²) = N_patches / 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
# 設定
# ============================================================

VISION_CONFIG = {
    "embed_dim": 1152,        # D_v: Vision Encoder隠れ次元
    "num_heads": 16,          # H_v: アテンションヘッド数
    "num_layers": 32,         # L_v: ViTレイヤー数
    "patch_size": 14,         # P: パッチサイズ (px)
    "spatial_merge_size": 2,  # merge_size: MLP Merger空間圧縮倍率
    "fullatt_block_indexes": [7, 15, 23, 31],  # Full Attentionを使うレイヤーインデックス
    # それ以外は Window Attention (窓サイズ: 112×112px = 8×8 patches)
    "deepstack_interval": 4,  # DeepStack: 何レイヤーごとに中間特徴量を抽出するか
}

MLP_MERGER_CONFIG = {
    "in_features": 1152 * 4,   # D_v × merge_size² = 1152 × 4
    "out_features": 3584,       # D_llm (LLMの隠れ次元)
    "hidden_features": 1152 * 4,  # 中間次元
}


# ============================================================
# 1. パッチ埋め込み
# ============================================================

class PatchEmbed(nn.Module):
    """
    画像パッチをベクトルに埋め込む (ViTの最初の層)

    ========================================
    Shape
    ========================================
    入力:
        pixel_values: (N_patches, C×P²) = (N_patches, 588)
            - 事前にパッチ分割済みのピクセル値
            - C=3, P=14 → C×P²=3×196=588

    出力:
        patch_embeds: (N_patches, D_v) = (N_patches, 1152)

    ========================================
    処理詳細
    ========================================
    Linear(C×P², D_v):
        weight: (D_v, C×P²) = (1152, 588)
        bias: (D_v,) = (1152,)

    数式: patch_embeds = pixel_values @ weight.T + bias
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1152):
        super().__init__()
        self.patch_size = patch_size
        # C×P² → D_v の線形変換
        self.proj = nn.Linear(
            in_features=in_channels * patch_size * patch_size,  # 3 × 14² = 588
            out_features=embed_dim,                              # 1152
            bias=True,
        )
        # weight: (1152, 588)
        # bias: (1152,)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        入力:
            pixel_values: (N_patches, C×P²) = (N_patches, 588)

        出力:
            patch_embeds: (N_patches, D_v) = (N_patches, 1152)
        """
        patch_embeds = self.proj(pixel_values)
        # (N_patches, 588) @ (588, 1152) + (1152,)
        # → (N_patches, 1152)
        return patch_embeds


# ============================================================
# 2. ViT Attention (2D Rotary PE + Window/Full Attention)
# ============================================================

class Qwen3VisionAttention(nn.Module):
    """
    Vision Encoder 用のアテンション

    特徴:
    - 2D Rotary Positional Encoding (h次元とw次元で独立してRoPEを適用)
    - Full Attention: 特定レイヤー (layer_idx = 7, 15, 23, 31)
    - Window Attention: それ以外のレイヤー (窓サイズ: 8×8 patches = 112×112px)

    ========================================
    Shape
    ========================================
    入力:
        hidden_states: (N_patches, D_v) = (N_patches, 1152)
        grid_thw: (num_images, 3) - 各画像の [T, H_patches, W_patches]
        rotary_pos_emb: (N_patches, head_dim_v//2, 2) - cos/sin

    出力:
        output: (N_patches, D_v) = (N_patches, 1152)

    ========================================
    内部形状
    ========================================
    head_dim_v = D_v // H_v = 1152 // 16 = 72
    QKV projection:
        weight: (3×D_v, D_v) = (3456, 1152)
        q, k, v: 各 (N_patches, H_v, head_dim_v) = (N_patches, 16, 72)
    """

    def __init__(self, embed_dim: int = 1152, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim          # D_v = 1152
        self.num_heads = num_heads          # H_v = 16
        self.head_dim = embed_dim // num_heads  # head_dim_v = 72

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        # weight: (3456, 1152)
        # 出力: (N_patches, 3456) → reshape → q/k/v: 各 (N_patches, H_v, head_dim_v)

        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        # 出力投影: (N_patches, 1152)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            hidden_states: (N_patches, D_v) = (N_patches, 1152)
            cu_seqlens: (num_seqs+1,) - パック済みシーケンスの累積長
                例: [0, 256, 768, ...] (パッチ数の累積和)
            rotary_pos_emb: (N_patches, head_dim_v//2, 2)

        内部:
            q, k, v: 各 (N_patches, H_v, head_dim_v) = (N_patches, 16, 72)
            attn_weights: (N_patches, N_patches) - Full Attention
                          または (N_patches, window_size²) - Window Attention

        出力:
            output: (N_patches, D_v) = (N_patches, 1152)
        """
        N_patches, D_v = hidden_states.shape

        # QKV 投影
        qkv = self.qkv(hidden_states)
        # qkv: (N_patches, 3×D_v) = (N_patches, 3456)

        qkv = qkv.reshape(N_patches, 3, self.num_heads, self.head_dim)
        # qkv: (N_patches, 3, H_v, head_dim_v) = (N_patches, 3, 16, 72)

        qkv = qkv.permute(1, 0, 2, 3)
        # qkv: (3, N_patches, H_v, head_dim_v)

        q, k, v = qkv.unbind(0)
        # q, k, v: 各 (N_patches, H_v, head_dim_v) = (N_patches, 16, 72)

        # 2D Rotary PE を適用
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
            # q, k: (N_patches, H_v, head_dim_v) (変化なし、内部で回転)

        # Attention 計算
        # 注: FlashAttention with varlen (パック済みシーケンス対応)
        # Full/Window Attentionの切り替えはcu_seqlensで制御
        output = flash_attention_varlen(q, k, v, cu_seqlens)
        # output: (N_patches, H_v, head_dim_v)

        output = output.reshape(N_patches, D_v)
        # output: (N_patches, D_v) = (N_patches, 1152)

        output = self.proj(output)
        # output: (N_patches, D_v) = (N_patches, 1152)

        return output


def apply_rotary_pos_emb_vision(
    q_or_k: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Vision Encoder 用の2D RoPE適用

    ========================================
    Shape
    ========================================
    入力:
        q_or_k: (N_patches, H_v, head_dim_v) = (N_patches, 16, 72)
        rotary_pos_emb: (N_patches, head_dim_v//2, 2)
            - 2: [cos, sin]
            - head_dim_v//2 = 36 (h次元18 + w次元18)

    出力:
        rotated: (N_patches, H_v, head_dim_v)

    ========================================
    処理詳細 (RoPE回転)
    ========================================
    q_or_k を head_dim_v/2 個の複素数として解釈:
        q_complex: (N_patches, H_v, head_dim_v//2) as complex

    cos, sin: (N_patches, head_dim_v//2)

    回転:
        q_rotated = q_complex * (cos + i*sin)
        = (q_r*cos - q_i*sin) + i*(q_r*sin + q_i*cos)
    """
    cos = rotary_pos_emb[..., 0]   # (N_patches, head_dim_v//2)
    sin = rotary_pos_emb[..., 1]   # (N_patches, head_dim_v//2)

    # (N_patches, H_v, head_dim_v) → 複素数表現
    # 注: 実際の実装はrotate_half()関数を使用
    # rotate_half: [..., ::2] と [..., 1::2] の要素で回転
    q_r = q_or_k[..., ::2]   # (N_patches, H_v, head_dim_v//2) - 実部
    q_i = q_or_k[..., 1::2]  # (N_patches, H_v, head_dim_v//2) - 虚部

    # cos/sinをブロードキャスト: (N_patches, 1, head_dim_v//2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rotated_r = q_r * cos - q_i * sin  # (N_patches, H_v, head_dim_v//2)
    q_rotated_i = q_r * sin + q_i * cos  # (N_patches, H_v, head_dim_v//2)

    # インターリーブして元の形状に戻す
    rotated = torch.stack([q_rotated_r, q_rotated_i], dim=-1).flatten(-2)
    # rotated: (N_patches, H_v, head_dim_v)

    return rotated


# ============================================================
# 3. ViT ブロック
# ============================================================

class Qwen3VisionBlock(nn.Module):
    """
    Vision Transformer ブロック (Pre-RMSNorm)

    構造:
        hidden = hidden + Attention(RMSNorm(hidden))
        hidden = hidden + FFN(RMSNorm(hidden))

    ========================================
    Shape
    ========================================
    入力:
        hidden_states: (N_patches, D_v) = (N_patches, 1152)

    出力:
        hidden_states: (N_patches, D_v) = (N_patches, 1152)

    ========================================
    内部次元
    ========================================
    D_v = 1152
    FFN中間次元 = D_v × 4 = 4608 (SigLIP-2のFFN比率)
    """

    def __init__(self, embed_dim: int = 1152, num_heads: int = 16):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

        self.attn = Qwen3VisionAttention(embed_dim=embed_dim, num_heads=num_heads)
        # weight: (3456+1152, 1152) for QKV+proj

        self.mlp = Qwen3VisionMLP(embed_dim=embed_dim, mlp_ratio=4)
        # FFN: D_v → 4×D_v → D_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        入力:
            hidden_states: (N_patches, D_v) = (N_patches, 1152)
            cu_seqlens: (num_images+1,) - パック済みシーケンスの累積長

        出力:
            hidden_states: (N_patches, D_v) = (N_patches, 1152)
        """
        # Pre-Norm Attention (残差接続)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)  # (N_patches, 1152)
        hidden_states = self.attn(hidden_states, cu_seqlens, rotary_pos_emb)  # (N_patches, 1152)
        hidden_states = residual + hidden_states   # (N_patches, 1152)

        # Pre-Norm FFN (残差接続)
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)  # (N_patches, 1152)
        hidden_states = self.mlp(hidden_states)    # (N_patches, 1152)
        hidden_states = residual + hidden_states   # (N_patches, 1152)

        return hidden_states


class Qwen3VisionMLP(nn.Module):
    """
    Vision Encoder の FFN (Feed-Forward Network)

    ========================================
    Shape
    ========================================
    入力: (N_patches, D_v) = (N_patches, 1152)
    出力: (N_patches, D_v) = (N_patches, 1152)

    内部:
        fc1: (N_patches, 1152) → (N_patches, 4608)
        act: GELU
        fc2: (N_patches, 4608) → (N_patches, 1152)
    """

    def __init__(self, embed_dim: int = 1152, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio  # 1152 × 4 = 4608
        self.fc1 = nn.Linear(embed_dim, hidden_dim)   # (1152, 4608)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)   # (4608, 1152)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)    # (N_patches, 4608)
        x = self.act(x)    # (N_patches, 4608)
        x = self.fc2(x)    # (N_patches, 1152)
        return x


# ============================================================
# 4. MLP Merger (Visual Token Compressor)
# ============================================================

class PatchMerger(nn.Module):
    """
    MLP Merger: 視覚トークンを2×2空間圧縮してLLM次元に変換

    Qwen2.5-VLと同じ構造: 2×2の隣接パッチを連結してMLPで圧縮

    ========================================
    処理詳細
    ========================================
    merge_size = 2 の場合:
        入力: (H_patches × W_patches, D_v)
        → reshape: (H_patches/2, 2, W_patches/2, 2, D_v)
        → permute + flatten: (H_patches/2 × W_patches/2, D_v × 4)
        → MLP: (H_patches/2 × W_patches/2, D_llm)

    合計パッチ数の変化:
        N_patches → N_patches / 4 (2×2圧縮)

    ========================================
    Shape
    ========================================
    入力:
        hidden_states: (N_patches, D_v) = (N_patches, 1152)
        grid_thw: (num_images, 3) = [[T, H_patches, W_patches], ...]

    出力:
        merged: (N_v, D_llm) = (N_patches/4, 3584)

    ========================================
    MLP 構造
    ========================================
    in_features = D_v × merge_size² = 1152 × 4 = 4608
    hidden_features = D_v × merge_size² = 4608 (中間次元)
    out_features = D_llm = 3584

    MLP:
        Linear(4608, 4608)
        GELU
        Linear(4608, 3584)
    """

    def __init__(
        self,
        in_features: int = 1152 * 4,   # D_v × merge_size² = 4608
        out_features: int = 3584,       # D_llm
        hidden_features: int = 1152 * 4,  # 中間次元
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size  # = 2
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),   # (4608, 4608)
            nn.GELU(),
            nn.Linear(hidden_features, out_features),  # (4608, 3584)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        """
        入力:
            hidden_states: (N_patches, D_v) = (N_patches, 1152)
            grid_thw: (num_images, 3)
                例: [[1, 32, 32], [1, 48, 24]]

        出力:
            merged: (N_v, D_llm) = (N_patches/4, 3584)
        """
        m = self.spatial_merge_size  # = 2

        # 各画像を個別に処理
        merged_list = []
        patch_idx = 0

        for thw in grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            # T=1 (画像) or T=フレーム数 (動画、Qwen3-VLでは常に1)
            N_i = T * H * W  # この画像のパッチ数

            # この画像のパッチを取り出す
            patches_i = hidden_states[patch_idx : patch_idx + N_i]
            # patches_i: (T×H×W, D_v) = (N_i, 1152)

            # 2×2空間圧縮: (H×W, D_v) → (H/m × W/m, D_v × m²)
            patches_i = patches_i.view(T, H // m, m, W // m, m, -1)
            # patches_i: (T, H/m, m, W/m, m, D_v) = (1, 16, 2, 16, 2, 1152) [32×32の例]

            patches_i = patches_i.permute(0, 1, 3, 2, 4, 5)
            # patches_i: (T, H/m, W/m, m, m, D_v) = (1, 16, 16, 2, 2, 1152)

            patches_i = patches_i.reshape(T * (H // m) * (W // m), m * m * patches_i.shape[-1])
            # patches_i: (T×H/m×W/m, m²×D_v) = (256, 4608) [32×32→16×16の例]

            # MLP で D_llm に変換
            merged_i = self.mlp(patches_i)
            # merged_i: (T×H/m×W/m, D_llm) = (256, 3584)

            merged_list.append(merged_i)
            patch_idx += N_i

        merged = torch.cat(merged_list, dim=0)
        # merged: (N_v, D_llm) = (sum(T×H/m×W/m), 3584)
        # N_v = N_patches / (m²) = N_patches / 4

        return merged


# ============================================================
# 5. Qwen3VisionTransformerPretrainedModel (全体)
# ============================================================

class Qwen3VisionTransformerPretrainedModel(nn.Module):
    """
    SigLIP-2 ベースの Vision Encoder + MLP Merger

    ========================================
    全体構造
    ========================================
    入力: pixel_values (N_patches, 588), grid_thw (num_images, 3)
        ↓
    PatchEmbed: (N_patches, 588) → (N_patches, D_v=1152)
        ↓
    [ViT Block × L_v=32 層]
      各ブロック:
        - RMSNorm + Window/Full Attention (2D RoPE) + 残差
        - RMSNorm + FFN (GELU) + 残差
      DeepStack: deepstack_interval=4ブロックごとに中間特徴を保存
        ↓
    PatchMerger (MLP Merger):
        (N_patches, D_v) → (N_v=N_patches/4, D_llm)
        ↓
    出力: visual_tokens (N_v, D_llm)

    ========================================
    Shape
    ========================================
    入力:
        pixel_values: (N_patches, C×P²) = (N_patches, 588)
        grid_thw: (num_images, 3)

    出力:
        visual_tokens: (N_v, D_llm) = (N_patches/4, 3584)
        intermediate_features: List[(N_patches, D_v)] - DeepStack用

    ========================================
    Full Attention vs Window Attention
    ========================================
    fullatt_block_indexes = [7, 15, 23, 31]:
        - 全パッチ間でグローバルアテンション
        - 計算量: O(N_patches²)
        - 目的: グローバルコンテキストの統合

    Window Attention (それ以外):
        - 局所的な窓 (112×112px = 8×8 patches) 内でのみアテンション
        - 計算量: O(N_patches × window_size²)
        - 目的: 局所的な特徴抽出の効率化
    """

    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config.get("embed_dim", 1152)
        num_heads = config.get("num_heads", 16)
        num_layers = config.get("num_layers", 32)
        self.spatial_merge_size = config.get("spatial_merge_size", 2)
        fullatt_block_indexes = config.get("fullatt_block_indexes", [7, 15, 23, 31])
        deepstack_interval = config.get("deepstack_interval", 4)

        # パッチ埋め込み
        self.patch_embed = PatchEmbed(
            patch_size=config.get("patch_size", 14),
            in_channels=3,
            embed_dim=embed_dim,
        )
        # (N_patches, 588) → (N_patches, 1152)

        # ViT ブロック
        self.blocks = nn.ModuleList([
            Qwen3VisionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)  # L_v = 32
        ])

        # MLP Merger
        self.merger = PatchMerger(
            in_features=embed_dim * self.spatial_merge_size ** 2,  # 1152 × 4 = 4608
            out_features=3584,  # D_llm (7Bモデル)
            spatial_merge_size=self.spatial_merge_size,
        )
        # (N_patches, 1152) → (N_v, 3584)

        self.fullatt_block_indexes = fullatt_block_indexes
        self.deepstack_interval = deepstack_interval

        # 位置埋め込み用 RMSNorm
        self.norm = nn.RMSNorm(embed_dim)

    def get_rotary_pos_emb(self, grid_thw: torch.LongTensor) -> torch.Tensor:
        """
        2D Rotary Position Embedding の cos/sin を計算

        ========================================
        Shape
        ========================================
        入力:
            grid_thw: (num_images, 3) - 各画像の [T, H_patches, W_patches]

        出力:
            rotary_pos_emb: (N_patches, head_dim_v//2, 2)
                - head_dim_v//2 = 36 (h次元18 + w次元18)
                - 2: [cos, sin]

        ========================================
        計算詳細
        ========================================
        各パッチ (t, h, w) に対して:
            θ_h = h × inv_freq_h  (次元18の周波数)
            θ_w = w × inv_freq_w  (次元18の周波数)
            rotary_pos_emb = cat([cos(θ_h), cos(θ_w)], [sin(θ_h), sin(θ_w)])
        """
        head_dim = 1152 // 16  # = 72
        # h次元: head_dim//4 = 18, w次元: head_dim//4 = 18
        # 合計: 36 = head_dim//2

        pos_emb_list = []
        for thw in grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            # T: 時間パッチ数 (画像は常に1)
            # H, W: 高さ・幅パッチ数

            # h, w のインデックスをメッシュグリッドで生成
            h_idx = torch.arange(H)   # (H,)
            w_idx = torch.arange(W)   # (W,)
            # T次元は無視 (Qwen3-VLでは動画も T=1 に分割)

            # 2D グリッドに展開
            h_grid, w_grid = torch.meshgrid(h_idx, w_idx, indexing="ij")
            # h_grid, w_grid: (H, W)

            # 逆周波数の計算 (RoPE標準)
            dim = head_dim // 4  # = 18
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            # inv_freq: (dim//2,) = (9,)

            # cos/sin 計算
            h_emb = torch.outer(h_grid.flatten(), inv_freq)  # (H×W, 9)
            w_emb = torch.outer(w_grid.flatten(), inv_freq)  # (H×W, 9)

            cos_h = torch.cos(h_emb)  # (H×W, 9)
            sin_h = torch.sin(h_emb)  # (H×W, 9)
            cos_w = torch.cos(w_emb)  # (H×W, 9)
            sin_w = torch.sin(w_emb)  # (H×W, 9)

            # (H×W, 36) の cos/sin に結合
            # 注: 実際の実装では h, w を head_dim//4 ずつ交互に配置
            cos = torch.cat([cos_h, cos_h, cos_w, cos_w], dim=-1)  # (H×W, 36)
            sin = torch.cat([sin_h, sin_h, sin_w, sin_w], dim=-1)  # (H×W, 36)

            emb = torch.stack([cos, sin], dim=-1)  # (H×W, 36, 2)
            pos_emb_list.append(emb.repeat(T, 1, 1))  # (T×H×W, 36, 2)

        rotary_pos_emb = torch.cat(pos_emb_list, dim=0)
        # rotary_pos_emb: (N_patches, head_dim//2, 2) = (N_patches, 36, 2)

        return rotary_pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Vision Encoder フォワードパス

        ========================================
        Shape
        ========================================
        入力:
            pixel_values: (N_patches, C×P²) = (N_patches, 588)
            grid_thw: (num_images, 3)

        出力:
            visual_tokens: (N_v, D_llm) = (N_patches/4, 3584)

        ========================================
        処理フロー
        ========================================
        Step 1: PatchEmbed
            (N_patches, 588) → (N_patches, 1152)
        Step 2: 2D RoPE計算
        Step 3: ViT Blocks × 32 (Full/Window Attention切替)
            DeepStackのため中間特徴量を保存 (4層ごと)
        Step 4: MLP Merger
            (N_patches, 1152) → (N_v, 3584)
        """
        # ========================================
        # Step 1: パッチ埋め込み
        # ========================================
        hidden_states = self.patch_embed(pixel_values)
        # hidden_states: (N_patches, D_v) = (N_patches, 1152)

        # ========================================
        # Step 2: 2D Rotary PE の計算
        # ========================================
        rotary_pos_emb = self.get_rotary_pos_emb(grid_thw)
        # rotary_pos_emb: (N_patches, head_dim//2, 2) = (N_patches, 36, 2)

        # パック済みシーケンスの累積長を計算 (FlashAttentionのvarlen用)
        seqlens = []
        for thw in grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            seqlens.append(T * H * W)
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), dim=0).tolist()),
                                   dtype=torch.int32)
        # cu_seqlens: (num_images+1,)
        # 例: [0, 1024, 2176] (画像1が1024パッチ、画像2が1152パッチの場合)

        # ========================================
        # Step 3: ViT Blocks (DeepStack付き)
        # ========================================
        intermediate_features = []  # DeepStack用中間特徴量

        # ウィンドウアテンション用の cu_seqlens を計算
        # 窓サイズ: 8×8 patches (112×112px) ← Qwen3VL の window_size
        window_size = 8
        cu_seqlens_win_list = [0]
        for thw in grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            # 各画像を window_size×window_size の窓に分割
            n_win_h = (H + window_size - 1) // window_size  # 高さ方向の窓数
            n_win_w = (W + window_size - 1) // window_size  # 幅方向の窓数
            win_size = window_size * window_size              # 窓あたりのパッチ数 (= 64)
            n_wins = T * n_win_h * n_win_w                   # 全窓数
            for _ in range(n_wins):
                cu_seqlens_win_list.append(cu_seqlens_win_list[-1] + win_size)
        cu_seqlens_win = torch.tensor(cu_seqlens_win_list, dtype=torch.int32)
        # cu_seqlens_win: (n_wins_total + 1,)

        for block_idx, block in enumerate(self.blocks):
            # Full Attention (block_idx in fullatt_block_indexes): cu_seqlens = global sequence lengths
            # Window Attention (others): cu_seqlens = window-local sequence lengths
            # 実装では block() の内部でcu_seqlensを使ってFull/Windowを切り替える
            if block_idx in self.fullatt_block_indexes:
                cu_seqlens_block = cu_seqlens         # グローバルアテンション: 全パッチ間
            else:
                cu_seqlens_block = cu_seqlens_win     # ウィンドウアテンション: 窓内のみ

            hidden_states = block(hidden_states, cu_seqlens_block, rotary_pos_emb)
            # hidden_states: (N_patches, D_v) = (N_patches, 1152)

            # DeepStack: deepstack_interval(=4)ブロックごとに中間特徴量を保存
            if (block_idx + 1) % self.deepstack_interval == 0:
                intermediate_features.append(hidden_states.clone())
                # 例 (L_v=32, deepstack_interval=4): 8個の中間特徴量を保存
                # indices: 3, 7, 11, 15, 19, 23, 27, 31

        # 最終正規化
        hidden_states = self.norm(hidden_states)
        # hidden_states: (N_patches, D_v) = (N_patches, 1152)

        # ========================================
        # Step 4: MLP Merger (2×2空間圧縮)
        # ========================================
        visual_tokens = self.merger(hidden_states, grid_thw)
        # visual_tokens: (N_v, D_llm) = (N_patches/4, 3584)

        # 注: intermediate_features はDeepStack機構でLLMに注入
        # 詳細は deepstack.py 参照
        self._last_intermediate_features = intermediate_features

        return visual_tokens
        # visual_tokens: (N_v, D_llm)


# ============================================================
# ダミー関数 (FlashAttention)
# ============================================================

def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    変長シーケンス対応 FlashAttention

    ========================================
    Shape
    ========================================
    入力:
        q, k, v: (N_patches, H_v, head_dim_v)
        cu_seqlens: (num_seqs+1,) - 累積シーケンス長

    出力:
        output: (N_patches, H_v, head_dim_v)

    ========================================
    注
    ========================================
    実際の実装では flash_attn.flash_attn_varlen_func() を使用
    Full AttentionとWindow Attentionの切り替えも内部で行われる

    Full Attention (fullatt_block_indexes):
        - causal=False, window_size=(-1, -1)
    Window Attention:
        - causal=False, window_size=(窓サイズ, 窓サイズ)
        - 窓: 8×8 patches = 112×112px
    """
    # 注: 簡略化のため、標準的なスケールドドット積アテンションで代替
    scale = q.shape[-1] ** -0.5
    N_patches, H_v, head_dim = q.shape
    q = q.permute(1, 0, 2)  # (H_v, N_patches, head_dim)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)
    attn_weights = torch.bmm(q, k.transpose(-2, -1)) * scale  # (H_v, N, N)
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.bmm(attn_weights, v)   # (H_v, N_patches, head_dim)
    output = output.permute(1, 0, 2)     # (N_patches, H_v, head_dim)
    return output


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Vision Encoder の使用例

    ========================================
    Shape Summary (448×448 画像の例)
    ========================================
    Input:
        pixel_values:   (1024, 588)   # N_patches=32×32=1024, C×P²=588
        grid_thw:       (1, 3) = [[1, 32, 32]]

    Internal:
        patch_embed出力: (1024, 1152)
        rotary_pos_emb:  (1024, 36, 2)
        ViT block出力:   (1024, 1152) ×32層
        intermediate:    8個 × (1024, 1152)  [DeepStack用]

    Output:
        visual_tokens:   (256, 3584)
        N_v = 1024 / 4 = 256 (2×2圧縮後)
    """
    print("=== Vision Encoder Example ===\n")

    # --- PatchEmbed 単体テスト ---
    N_patches = 1024  # 32×32
    patch_embed = PatchEmbed(patch_size=14, in_channels=3, embed_dim=1152)
    pixel_values = torch.randn(N_patches, 588)   # (N_patches, C×P²)
    patch_out = patch_embed(pixel_values)
    print(f"[PatchEmbed]")
    print(f"  pixel_values:   {pixel_values.shape}")   # (1024, 588)
    print(f"  patch_embed:    {patch_out.shape}")       # (1024, 1152)
    print()

    # --- Qwen3VisionAttention 単体テスト ---
    attn = Qwen3VisionAttention(embed_dim=1152, num_heads=16)
    cu_seqlens = torch.tensor([0, N_patches], dtype=torch.int32)  # 1画像のみ
    rotary_pos_emb = torch.randn(N_patches, 36, 2)
    attn_out = attn(patch_out, cu_seqlens, rotary_pos_emb)
    print(f"[Qwen3VisionAttention]")
    print(f"  hidden_states:  {patch_out.shape}")  # (1024, 1152)
    print(f"  attn output:    {attn_out.shape}")   # (1024, 1152)
    print()

    # --- PatchMerger 単体テスト ---
    merger = PatchMerger(
        in_features=1152 * 4,   # D_v × merge_size²
        out_features=3584,
        spatial_merge_size=2,
    )
    grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long)  # T=1, H=32, W=32
    merged = merger(patch_out, grid_thw)
    print(f"[PatchMerger]")
    print(f"  input:          {patch_out.shape}")   # (1024, 1152)
    print(f"  merged:         {merged.shape}")       # (256, 3584)
    print()

    # --- Qwen3VisionTransformerPretrainedModel 全体テスト (小さいサイズで実行) ---
    # 注: 32層フルモデルは時間がかかるため、2層の軽量版で形状を確認
    mini_config = {
        "embed_dim": 1152,
        "num_heads": 16,
        "num_layers": 2,          # 実際は32層、ここでは形状確認用に2層
        "patch_size": 14,
        "spatial_merge_size": 2,
        "fullatt_block_indexes": [1],  # 2層中の最終層をFull Attention
        "deepstack_interval": 1,
    }
    mini_model = Qwen3VisionTransformerPretrainedModel(mini_config)
    mini_model.eval()
    with torch.no_grad():
        visual_tokens = mini_model(pixel_values, grid_thw)
    print(f"[Qwen3VisionTransformerPretrainedModel (2層版)]")
    print(f"  pixel_values:   {pixel_values.shape}")    # (1024, 588)
    print(f"  grid_thw:       {grid_thw.shape}")        # (1, 3)
    print(f"  visual_tokens:  {visual_tokens.shape}")   # (256, 3584)
    print(f"  N_v = N_patches / 4 = {N_patches} / 4 = {N_patches // 4}")


if __name__ == "__main__":
    example_usage()
