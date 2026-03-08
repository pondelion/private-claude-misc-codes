"""
DINOv3 Backbone - 簡略化疑似コード
====================================

ViT-7B + Axial RoPE + SwiGLU FFN の詳細実装
論文: https://arxiv.org/abs/2508.10104

主要コンポーネント:
  1. PatchEmbed: 画像をパッチトークンに変換
  2. RoPE: Rotary Position Embedding (Axial)
  3. SelfAttentionBlock: Multi-Head Self-Attention + SwiGLU FFN
  4. DinoVisionTransformer: 全体構造

公式実装参照:
  - dinov3/models/vision_transformer.py
  - dinov3/layers/attention.py
  - dinov3/layers/block.py
  - dinov3/layers/swiglu_ffn.py
"""

import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Patch Embedding
# ============================================================
class PatchEmbed(nn.Module):
    """
    画像をパッチトークンに変換

    Conv2d でパッチサイズ分のストライドで畳み込み
    → 各パッチが1つのトークンに変換される

    入力: (B, 3, H, W)
    出力: (B, H/P, W/P, D) - P=patch_size, D=embed_dim
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 4096,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 256/16 = 16 → 16*16 = 256 patches

        # パッチ畳み込み: kernel=stride=patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) - 入力画像

        Returns:
            patches: (B, H//P, W//P, D) - パッチ埋め込み
        """
        # (B, 3, 256, 256) → (B, 4096, 16, 16)
        x = self.proj(x)
        # (B, 4096, 16, 16) → (B, 16, 16, 4096)
        x = x.permute(0, 2, 3, 1)
        return x


# ============================================================
# 2. Rotary Position Embedding (RoPE)
# ============================================================
class RopePositionEmbedding(nn.Module):
    """
    Axial Rotary Position Embedding

    各パッチに2D正規化座標を割り当て、
    Attention の Q, K に回転行列として適用

    特徴:
      - 学習可能パラメータなし
      - 任意の解像度に対応
      - Box Jittering で解像度ロバスト性向上

    参考: RoFormer (Su et al., 2021) の2D拡張
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        num_heads: int = 32,
        base: float = 100.0,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        rescale_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads  # 4096/32 = 128
        self.normalize_coords = normalize_coords
        self.rescale_coords = rescale_coords
        self.jitter_coords = jitter_coords

        # 周波数の計算: theta_i = base^(-2i/d) for i in [0, d/2)
        half_dim = self.head_dim // 2  # 64 (x方向 + y方向で分割)
        quarter_dim = half_dim // 2     # 32

        # 各次元の角周波数
        freqs = 1.0 / (base ** (torch.arange(0, quarter_dim).float() / quarter_dim))
        self.register_buffer("freqs", freqs)  # (32,)

    def forward(
        self,
        H: int,
        W: int,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定グリッドサイズの RoPE (sin, cos) を計算

        Args:
            H: パッチグリッドの高さ (例: 16)
            W: パッチグリッドの幅 (例: 16)
            training: 学習時は Box Jittering 適用

        Returns:
            sin: (H*W, head_dim) - sin成分
            cos: (H*W, head_dim) - cos成分
        """
        # 1. 正規化座標の生成 [-1, 1]
        coords_h = torch.linspace(-1, 1, H)  # (H,)
        coords_w = torch.linspace(-1, 1, W)  # (W,)

        # 2. Box Jittering (学習時のみ)
        if training and self.rescale_coords is not None:
            # [-1,1] → [-s,s] where s ∈ [1/r, r], r=rescale_coords
            r = self.rescale_coords
            s = torch.empty(1).uniform_(1.0 / r, r).item()
            coords_h = coords_h * s
            coords_w = coords_w * s

        if training and self.jitter_coords is not None:
            j = self.jitter_coords
            log_scale = torch.empty(1).uniform_(-math.log(j), math.log(j)).item()
            coords_h = coords_h * math.exp(log_scale)
            coords_w = coords_w * math.exp(log_scale)

        # 3. 座標 × 周波数 → 角度
        # coords_h: (H,), freqs: (quarter_dim,) → angles_h: (H, quarter_dim)
        angles_h = torch.outer(coords_h, self.freqs)  # (H, 32)
        angles_w = torch.outer(coords_w, self.freqs)  # (W, 32)

        # 4. 2D グリッドに展開
        # angles_h: (H, 1, 32) + angles_w: (1, W, 32) → (H, W, 32) それぞれ
        angles_h = angles_h.unsqueeze(1).expand(-1, W, -1)  # (H, W, 32)
        angles_w = angles_w.unsqueeze(0).expand(H, -1, -1)  # (H, W, 32)

        # 5. H方向とW方向を連結 → (H, W, 64)
        angles = torch.cat([angles_h, angles_w], dim=-1)  # (H, W, 64)

        # 6. sin, cos を複製して head_dim に合わせる
        # (H, W, 64) → repeat → (H, W, 128=head_dim)
        angles = angles.repeat(1, 1, 2)  # 各方向の sin/cos 用

        # 7. Flatten: (H*W, head_dim)
        sin = torch.sin(angles.flatten(0, 1))  # (H*W, 128)
        cos = torch.cos(angles.flatten(0, 1))  # (H*W, 128)

        return sin, cos


# ============================================================
# 3. Self-Attention with RoPE
# ============================================================
class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE

    入力: (B, N, D) → 出力: (B, N, D)
    N = 1 (CLS) + R (Storage) + P (Patches)

    RoPE は Patch tokens の Q, K にのみ適用
    (CLS, Storage tokens には適用しない)
    """

    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads     # 4096/32 = 128
        self.scale = self.head_dim ** -0.5   # 1/sqrt(128)

        # QKV 統合線形層
        if mask_k_bias:
            self.qkv = LinearKMaskedBias(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # 出力投影
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - 入力トークン
               N = 1 + R + P (CLS + Storage + Patches)
            rope: (sin, cos) each (P, head_dim) - RoPE 埋め込み
               P = patch tokens のみ

        Returns:
            out: (B, N, D)
        """
        B, N, D = x.shape

        # === QKV 投影 ===
        qkv = self.qkv(x)                                    # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        # q, k, v: each (B, num_heads, N, head_dim)

        # === RoPE 適用 (Patch tokens のみ) ===
        if rope is not None:
            sin, cos = rope  # each (P, head_dim)
            n_prefix = N - sin.shape[0]  # CLS + Storage = 1 + R

            # Patch tokens 部分にのみ RoPE を適用
            q_patch = q[:, :, n_prefix:]     # (B, H, P, head_dim)
            k_patch = k[:, :, n_prefix:]     # (B, H, P, head_dim)

            q_patch = rope_apply(q_patch, sin, cos)
            k_patch = rope_apply(k_patch, sin, cos)

            q = torch.cat([q[:, :, :n_prefix], q_patch], dim=2)
            k = torch.cat([k[:, :, :n_prefix], k_patch], dim=2)

        # === Scaled Dot-Product Attention ===
        # PyTorch の効率的な SDPA を使用
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # (B, num_heads, N, head_dim)

        # === 出力投影 ===
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(attn_out)                               # (B, N, D)

        return out


class LinearKMaskedBias(nn.Linear):
    """
    K (Key) のバイアスをゼロマスクする線形層

    QKV統合層のバイアス: (3*D,)
    → Q部(D): バイアスあり
    → K部(D): バイアスなし (ゼロ)
    → V部(D): バイアスあり
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        if bias:
            # K部分のマスク: [0...0, 1...1, 0...0] → 反転して [1,0,1]
            d = out_features // 3
            mask = torch.ones(out_features)
            mask[d:2*d] = 0  # K部分をゼロ
            self.register_buffer("bias_mask", mask)

    def forward(self, x):
        if self.bias is not None:
            masked_bias = self.bias * self.bias_mask
            return F.linear(x, self.weight, masked_bias)
        return F.linear(x, self.weight)


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE の回転操作: [x1, x2] → [-x2, x1]

    Args:
        x: (..., D)

    Returns:
        rotated: (..., D)
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE を適用: x * cos + rotate_half(x) * sin

    Args:
        x: (B, num_heads, P, head_dim) - Q or K
        sin: (P, head_dim) - sin成分
        cos: (P, head_dim) - cos成分

    Returns:
        out: (B, num_heads, P, head_dim)
    """
    return x * cos + rope_rotate_half(x) * sin


# ============================================================
# 4. SwiGLU FFN
# ============================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network

    標準 MLP との比較:
      MLP:    x → Linear → GELU → Linear
      SwiGLU: x → [Linear_1, Linear_2] → SiLU(L1) * L2 → Linear_3

    入力: (B, N, D) → 出力: (B, N, D)

    hidden_features = int(D * ffn_ratio * 2/3) をアライメント
    例: D=4096, ffn_ratio=3 → hidden=4096*3=12288 → 12288*2/3=8192
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        align_to: int = 8,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # SwiGLU の隠れ次元: hidden * 2/3 をアライメント
        d = int(hidden_features * 2 / 3)
        swiglu_hidden = d + (-d % align_to)  # align_to の倍数に切り上げ

        self.w1 = nn.Linear(in_features, swiglu_hidden, bias=bias)
        self.w2 = nn.Linear(in_features, swiglu_hidden, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - 入力

        Returns:
            out: (B, N, D)
        """
        x1 = self.w1(x)            # (B, N, swiglu_hidden)
        x2 = self.w2(x)            # (B, N, swiglu_hidden)
        hidden = F.silu(x1) * x2   # SiLU ゲーティング: (B, N, swiglu_hidden)
        out = self.w3(hidden)       # (B, N, D)
        return out


# ============================================================
# 5. LayerScale
# ============================================================
class LayerScale(nn.Module):
    """
    Layer Scale: 残差接続前にスケーリング

    各チャネルに学習可能なスケールパラメータ (初期値: 小さい値)
    → 学習初期の安定性向上

    参考: CaiT (Touvron et al., 2021)
    """

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - Attention or FFN 出力

        Returns:
            scaled: (B, N, D)
        """
        return x * self.gamma  # (B, N, D) * (D,) → (B, N, D)


# ============================================================
# 6. Transformer Block (Self-Attention + FFN)
# ============================================================
class SelfAttentionBlock(nn.Module):
    """
    1つの Transformer Block

    構成:
      LayerNorm → Self-Attention (with RoPE) → LayerScale → DropPath → Residual
      → LayerNorm → SwiGLU FFN → LayerScale → DropPath → Residual

    入力: (B, N, D) → 出力: (B, N, D)
    N = 1 (CLS) + R (Storage) + P (Patches)
    """

    def __init__(
        self,
        dim: int = 4096,
        num_heads: int = 32,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path: float = 0.0,
        layerscale_init: float = 1e-5,
        mask_k_bias: bool = False,
    ):
        super().__init__()

        # Attention path
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()

        # FFN path
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * ffn_ratio)
        self.mlp = SwiGLUFFN(
            in_features=dim,
            hidden_features=hidden_features,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()

        # Stochastic Depth
        self.drop_path_rate = drop_path

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - 入力トークン
            rope: (sin, cos) - RoPE 埋め込み

        Returns:
            out: (B, N, D)
        """
        # --- Attention path ---
        residual = x                                # (B, N, D)
        x = self.norm1(x)                           # (B, N, D)
        x = self.attn(x, rope=rope)                 # (B, N, D)
        x = self.ls1(x)                             # (B, N, D)
        if self.training and self.drop_path_rate > 0:
            x = self._drop_path(x)
        x = residual + x                            # (B, N, D)

        # --- FFN path ---
        residual = x                                # (B, N, D)
        x = self.norm2(x)                           # (B, N, D)
        x = self.mlp(x)                             # (B, N, D)
        x = self.ls2(x)                             # (B, N, D)
        if self.training and self.drop_path_rate > 0:
            x = self._drop_path(x)
        x = residual + x                            # (B, N, D)

        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic Depth: ランダムにサンプルをスキップ"""
        if not self.training or self.drop_path_rate == 0:
            return x
        keep_prob = 1 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob


# ============================================================
# 7. DINOv3 Vision Transformer (全体)
# ============================================================
class DinoVisionTransformer(nn.Module):
    """
    DINOv3 Vision Transformer

    構成:
      1. PatchEmbed: (B, 3, H, W) → (B, H/P, W/P, D)
      2. CLS + Storage tokens 追加: (B, 1+R+P, D)
      3. RoPE計算: (P, head_dim)
      4. depth 個の SelfAttentionBlock: (B, 1+R+P, D)
      5. LayerNorm: 正規化出力

    ViT-7B 構成:
      embed_dim=4096, depth=40, num_heads=32, patch_size=16
      ffn_ratio=3 (SwiGLU), n_storage_tokens=4
      → パラメータ数: 6.7B
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 4096,
        depth: int = 40,
        num_heads: int = 32,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.4,
        layerscale_init: float = 1e-5,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = True,
        # RoPE 設定
        rope_base: float = 100.0,
        rope_normalize_coords: str = "separate",
        rope_rescale_coords: Optional[float] = 2.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens
        self.depth = depth
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm

        # === パッチ埋め込み ===
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # === 特殊トークン ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.randn(1, n_storage_tokens, embed_dim) * 0.02
            )
        else:
            self.storage_tokens = None

        # === マスクトークン (iBOT用) ===
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # === RoPE ===
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            normalize_coords=rope_normalize_coords,
            rescale_coords=rope_rescale_coords,
        )

        # === Transformer Blocks ===
        # Stochastic Depth: 線形増加 (0 → drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                layerscale_init=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for i in range(depth)
        ])

        # === 出力正規化 ===
        self.norm = nn.LayerNorm(embed_dim)
        if untie_cls_and_patch_norms:
            self.cls_norm = nn.LayerNorm(embed_dim)
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = nn.LayerNorm(embed_dim)

    def prepare_tokens_with_masks(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        画像をトークンに変換 + マスク適用

        Args:
            x: (B, 3, H, W) - 入力画像
            masks: (B, P) - マスク (True=マスク), P = (H/P_size) * (W/P_size)

        Returns:
            tokens: (B, 1+R+P, D) - CLS + Storage + Patch tokens
            hw: (H_patch, W_patch) - パッチグリッドサイズ
        """
        B = x.shape[0]

        # 1. パッチ埋め込み
        patches = self.patch_embed(x)  # (B, H/16, W/16, D)
        H_patch, W_patch = patches.shape[1], patches.shape[2]
        patches = patches.flatten(1, 2)  # (B, P, D), P = H_patch * W_patch

        # 2. マスク適用 (マスクされたパッチを mask_token に置換)
        if masks is not None:
            mask_expanded = masks.unsqueeze(-1)  # (B, P, 1)
            patches = torch.where(
                mask_expanded.bool(),
                self.mask_token.expand_as(patches),
                patches,
            )  # (B, P, D)

        # 3. CLS token + Storage tokens を先頭に追加
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, patches], dim=1)  # (B, 1+P, D)

        if self.storage_tokens is not None:
            storage = self.storage_tokens.expand(B, -1, -1)  # (B, R, D)
            tokens = torch.cat([cls, storage, patches], dim=1)  # (B, 1+R+P, D)

        return tokens, (H_patch, W_patch)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass

        Args:
            x: (B, 3, H, W) or list of tensors
               学習時: [global_crops, local_crops]
            masks: マスクテンソル (学習時のみ)

        Returns:
            dict or list of dicts with:
                x_norm_clstoken: (B, D) - 正規化CLS
                x_storage_tokens: (B, R, D) - Storage tokens
                x_norm_patchtokens: (B, P, D) - 正規化パッチ
                x_prenorm: (B, 1+R+P, D) - 正規化前の特徴
                masks: (B, P) - マスク情報
        """
        # リストモードの場合
        if isinstance(x, list):
            return self._forward_list(x, masks)

        # 単一テンソルモード
        return self._forward_single(x, masks)

    def _forward_single(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        単一テンソルの Forward

        Args:
            x: (B, 3, H, W)
            masks: (B, P) optional

        Returns:
            dict with features
        """
        B = x.shape[0]

        # 1. トークン準備
        tokens, (H_p, W_p) = self.prepare_tokens_with_masks(x, masks)
        # tokens: (B, 1+R+P, D)

        # 2. RoPE 計算
        rope = self.rope_embed(H_p, W_p, training=self.training)
        # (sin, cos) each (P, head_dim)

        # 3. Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens, rope=rope)  # (B, 1+R+P, D)

        # 4. 出力正規化
        x_prenorm = tokens  # (B, 1+R+P, D)

        R = self.n_storage_tokens
        n_prefix = 1 + R  # CLS + Storage

        if self.untie_cls_and_patch_norms:
            # CLS/Storage と Patch で別々の LayerNorm
            cls_storage = self.cls_norm(tokens[:, :n_prefix])  # (B, 1+R, D)
            patches = self.norm(tokens[:, n_prefix:])           # (B, P, D)
        else:
            normed = self.norm(tokens)                          # (B, 1+R+P, D)
            cls_storage = normed[:, :n_prefix]                  # (B, 1+R, D)
            patches = normed[:, n_prefix:]                      # (B, P, D)

        return {
            "x_norm_clstoken": cls_storage[:, 0],       # (B, D)
            "x_storage_tokens": cls_storage[:, 1:],      # (B, R, D)
            "x_norm_patchtokens": patches,               # (B, P, D)
            "x_prenorm": x_prenorm,                      # (B, 1+R+P, D)
            "masks": masks,                               # (B, P) or None
        }

    def _forward_list(
        self,
        x_list: List[torch.Tensor],
        masks_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        リストモードの Forward (学習時の効率化)

        Global Crops と Local Crops を連結して一括処理

        Args:
            x_list: [global_crops (2*B, 3, 256, 256), local_crops (8*B, 3, 112, 112)]
            masks_list: [global_masks (2*B, 256), None]

        Returns:
            list of dicts with features for each input
        """
        if masks_list is None:
            masks_list = [None] * len(x_list)

        # 各入力をトークンに変換
        tokens_list = []
        hw_list = []
        for x, m in zip(x_list, masks_list):
            tokens, hw = self.prepare_tokens_with_masks(x, m)
            tokens_list.append(tokens)
            hw_list.append(hw)

        # 連結して一括処理 (異なるシーケンス長の場合はパディング等が必要)
        # ここでは概念的な処理を示す
        # 実際の実装では cat_keep_shapes / uncat_with_shapes を使用

        # RoPE はそれぞれのグリッドサイズで計算
        rope_list = [
            self.rope_embed(h, w, training=self.training)
            for h, w in hw_list
        ]

        # Transformer Blocks
        for block in self.blocks:
            tokens_list = block.forward_list(tokens_list, rope_list)

        # 正規化して返却
        results = []
        for i, (tokens, masks) in enumerate(zip(tokens_list, masks_list)):
            R = self.n_storage_tokens
            n_prefix = 1 + R

            # Local Crops の場合は別の CLS norm を使用
            if i > 0 and self.untie_global_and_local_cls_norm:
                normed = self.norm(tokens)
                cls_token = self.local_cls_norm(tokens[:, 0:1])[:, 0]
            elif self.untie_cls_and_patch_norms:
                cls_storage = self.cls_norm(tokens[:, :n_prefix])
                patches = self.norm(tokens[:, n_prefix:])
                cls_token = cls_storage[:, 0]
            else:
                normed = self.norm(tokens)
                cls_token = normed[:, 0]
                patches = normed[:, n_prefix:]

            if not self.untie_cls_and_patch_norms:
                patches = self.norm(tokens[:, n_prefix:]) if i > 0 else normed[:, n_prefix:]

            results.append({
                "x_norm_clstoken": cls_token,                                # (B_i, D)
                "x_storage_tokens": tokens[:, 1:n_prefix],                   # (B_i, R, D)
                "x_norm_patchtokens": patches,                               # (B_i, P_i, D)
                "x_prenorm": tokens,                                          # (B_i, 1+R+P_i, D)
                "masks": masks,                                               # (B_i, P_i) or None
            })

        return results

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int]] = 4,
    ) -> List[torch.Tensor]:
        """
        中間層の出力を取得 (下流タスク用)

        Args:
            x: (B, 3, H, W)
            n: 取得する層数 or 層インデックスのリスト
               例: [10, 20, 30, 40] → 4つの中間層

        Returns:
            features: list of (B, P, D) - 各層のパッチトークン
        """
        tokens, (H_p, W_p) = self.prepare_tokens_with_masks(x)
        rope = self.rope_embed(H_p, W_p, training=False)

        if isinstance(n, int):
            # 最後の n 層を取得
            layer_indices = list(range(self.depth - n, self.depth))
        else:
            layer_indices = n

        outputs = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, rope=rope)
            if i + 1 in layer_indices:  # 1-indexed
                normed = self.norm(tokens)
                R = self.n_storage_tokens
                patches = normed[:, 1 + R:]  # (B, P, D) - CLS + Storage を除く
                outputs.append(patches)

        return outputs


# ============================================================
# プリセットモデル構成
# ============================================================
def vit_small(**kwargs) -> DinoVisionTransformer:
    """ViT-Small: 21M params"""
    return DinoVisionTransformer(
        embed_dim=384, depth=12, num_heads=6, ffn_ratio=4.0, **kwargs
    )

def vit_base(**kwargs) -> DinoVisionTransformer:
    """ViT-Base: 86M params"""
    return DinoVisionTransformer(
        embed_dim=768, depth=12, num_heads=12, ffn_ratio=4.0, **kwargs
    )

def vit_large(**kwargs) -> DinoVisionTransformer:
    """ViT-Large: 300M params"""
    return DinoVisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0, **kwargs
    )

def vit_giant2(**kwargs) -> DinoVisionTransformer:
    """ViT-Giant2: 1.1B params (DINOv2 相当)"""
    return DinoVisionTransformer(
        embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4.0, **kwargs
    )

def vit_7b(**kwargs) -> DinoVisionTransformer:
    """ViT-7B: 6.7B params (DINOv3 メインモデル)"""
    return DinoVisionTransformer(
        embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3.0,
        n_storage_tokens=4, mask_k_bias=True,
        rope_rescale_coords=2.0,
        untie_global_and_local_cls_norm=True,
        **kwargs
    )
