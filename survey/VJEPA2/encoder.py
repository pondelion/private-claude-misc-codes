"""
V-JEPA 2.1 エンコーダ - 簡略化疑似コード
==========================================

VisionTransformerエンコーダ。動画・画像両方に対応。
V-JEPA 2.1では以下の拡張が加わる:
  - Multi-Modal Tokenizer: 2D Conv(画像) or 3D Conv(動画)
  - Modality Embedding: img/vid識別用の学習可能トークン
  - Hierarchical Outputs: 複数中間層の出力を収集 (Deep Self-Supervision用)

対応する公式実装:
  - src/models/vision_transformer.py          (V-JEPA 2 ベース)
  - app/vjepa_2_1/models/vision_transformer.py (V-JEPA 2.1 拡張)
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# パッチ埋め込み: 画像用 (2D Conv)
# ============================================================

class PatchEmbed2D(nn.Module):
    """
    画像のパッチ埋め込み (2D畳み込み)

    V-JEPA 2.1 の Multi-Modal Tokenizer における画像用トークナイザ。
    stride=patch_sizeにより非重複パッチに分割して埋め込む。

    入力:
        x: (B, C, H, W)
            B: バッチサイズ
            C: チャネル数 (通常3=RGB)
            H: 高さ (px)
            W: 幅  (px)

    出力:
        x: (B, N_patches, embed_dim)
            N_patches = (H / patch_size) * (W / patch_size)
    """

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        # kernel_size=patch_size, stride=patch_size → 非重複パッチ
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)      # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)      # (B, embed_dim, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, embed_dim)
        return x


# ============================================================
# パッチ埋め込み: 動画用 (3D Conv, チューブレット)
# ============================================================

class PatchEmbed3D(nn.Module):
    """
    動画のパッチ埋め込み (3D畳み込み, チューブレット方式)

    時間方向に tubelet_size フレームをまとめて1トークンとする。
    空間方向は通常のパッチ埋め込みと同じ。

    入力:
        x: (B, C, T, H, W)
            T: フレーム数

    出力:
        x: (B, N_patches, embed_dim)
            N_patches = (T / tubelet_size) * (H / patch_size) * (W / patch_size)

    例: T=16, H=W=256, patch_size=16, tubelet_size=2
        N_patches = 8 * 16 * 16 = 2048
    """

    def __init__(self, patch_size: int = 16, tubelet_size: int = 2,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        # 3D Conv: kernel=(tubelet_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)      # (B, embed_dim, T/tubelet, H/patch, W/patch)
        x = x.flatten(2)      # (B, embed_dim, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, embed_dim)
        return x


# ============================================================
# 3D Sinusoidal 位置埋め込み
# ============================================================

def get_3d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int,
                             grid_d: int, uniform_power: bool = False) -> torch.Tensor:
    """
    3D Sin-Cos位置埋め込みを生成する。

    入力:
        embed_dim: 埋め込み次元
        grid_h:    空間グリッド高さ (H/patch_size)
        grid_w:    空間グリッド幅   (W/patch_size)
        grid_d:    時間グリッド深さ (T/tubelet_size)

    出力:
        pos_embed: (N_patches, embed_dim)
                    N_patches = grid_h * grid_w * grid_d
    """
    # 空間・時間方向の次元配分
    if uniform_power:
        # 空間(h,w)と時間(d)に均等に次元を割り当て
        embed_dim_spatial = embed_dim * 2 // 3
        embed_dim_temporal = embed_dim - embed_dim_spatial
    else:
        # 空間に2/3、時間に1/3
        embed_dim_spatial = embed_dim * 2 // 3
        embed_dim_temporal = embed_dim // 3

    # ========================================
    # 空間位置埋め込み (2D Sin-Cos)
    # ========================================
    # グリッドの各位置を (y, x) として表現
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid_2d = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid_2d = torch.stack(grid_2d, dim=0)  # (2, grid_h, grid_w)

    # h方向のSin-Cos
    omega_h = torch.arange(embed_dim_spatial // 4, dtype=torch.float32)
    omega_h = 1.0 / (10000 ** (omega_h / (embed_dim_spatial // 4)))
    pos_h = grid_2d[0].reshape(-1, 1) * omega_h.reshape(1, -1)  # (N_spatial, D/4)
    emb_h = torch.cat([pos_h.sin(), pos_h.cos()], dim=-1)        # (N_spatial, D/2)

    # w方向のSin-Cos
    omega_w = torch.arange(embed_dim_spatial // 4, dtype=torch.float32)
    omega_w = 1.0 / (10000 ** (omega_w / (embed_dim_spatial // 4)))
    pos_w = grid_2d[1].reshape(-1, 1) * omega_w.reshape(1, -1)  # (N_spatial, D/4)
    emb_w = torch.cat([pos_w.sin(), pos_w.cos()], dim=-1)        # (N_spatial, D/2)

    # 空間位置埋め込みを結合
    emb_spatial = torch.cat([emb_h, emb_w], dim=-1)  # (N_spatial, D_spatial)
    # N_spatial = grid_h * grid_w

    # ========================================
    # 時間位置埋め込み (1D Sin-Cos)
    # ========================================
    grid_t = torch.arange(grid_d, dtype=torch.float32)
    omega_t = torch.arange(embed_dim_temporal // 2, dtype=torch.float32)
    omega_t = 1.0 / (10000 ** (omega_t / (embed_dim_temporal // 2)))
    pos_t = grid_t.reshape(-1, 1) * omega_t.reshape(1, -1)  # (grid_d, D_temp/2)
    emb_temporal = torch.cat([pos_t.sin(), pos_t.cos()], dim=-1)  # (grid_d, D_temporal)

    # ========================================
    # 3D位置埋め込みを構築
    # 時間×空間の全組み合わせ
    # ========================================
    N_spatial = grid_h * grid_w
    # 空間埋め込みをgrid_d回繰り返し: (grid_d * N_spatial, D_spatial)
    emb_spatial_3d = emb_spatial.unsqueeze(0).expand(grid_d, -1, -1).reshape(-1, embed_dim_spatial)
    # 時間埋め込みをN_spatial回繰り返し: (grid_d * N_spatial, D_temporal)
    emb_temporal_3d = emb_temporal.unsqueeze(1).expand(-1, N_spatial, -1).reshape(-1, embed_dim_temporal)

    # 最終位置埋め込み: (N_patches, embed_dim)
    pos_embed = torch.cat([emb_spatial_3d, emb_temporal_3d], dim=-1)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    2D Sin-Cos位置埋め込みを生成する (画像用)。

    出力:
        pos_embed: (N_patches, embed_dim)
                    N_patches = grid_size * grid_size
    """
    grid_y = torch.arange(grid_size, dtype=torch.float32)
    grid_x = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)

    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))

    pos_y = grid[0].reshape(-1, 1) * omega.reshape(1, -1)
    pos_x = grid[1].reshape(-1, 1) * omega.reshape(1, -1)

    emb_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
    emb_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)

    pos_embed = torch.cat([emb_y, emb_x], dim=-1)  # (N_patches, embed_dim)
    return pos_embed


def apply_masks(x: torch.Tensor, masks: list) -> torch.Tensor:
    """
    パッチトークンからマスクインデックスに対応するものを選択・取り出す。

    入力:
        x:     (B, N, D)  全パッチトークン
        masks: list of (B, K) インデックステンソル
               K: 保持するパッチ数

    出力:
        (B*len(masks), K, D) または concat後の (B*len(masks), K, D)

    仕組み: torch.gatherで指定インデックスのトークンを抽出
    """
    all_x = []
    for m in masks:
        # m: (B, K)
        # index用に次元拡張: (B, K, D)
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=idx))  # (B, K, D)
    return torch.cat(all_x, dim=0)  # (B*len(masks), K, D)


# ============================================================
# Transformer ブロック (単純化)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention

    入力:
        x:    (B, N, D)
        mask: (B, N) または None  トークンのインデックス (RoPE用)

    出力:
        x:    (B, N, D)
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None, attn_mask=None,
                T=None, H_patches=None, W_patches=None) -> torch.Tensor:
        B, N, D = x.shape

        # QKV分解
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)           # 各 (B, num_heads, N, head_dim)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)  # (B, N, num_heads, head_dim)
        x = x.reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer ブロック: LayerNorm → Attention →残差 + LayerNorm → MLP → 残差

    入力:
        x:    (B, N, D)
    出力:
        x:    (B, N, D)  ← 形状不変
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_silu: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        act = nn.SiLU if use_silu else nn.GELU
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
        # Stochastic Depth (DropPath)
        self.drop_path = nn.Identity() if drop_path == 0 else _DropPath(drop_path)

    def forward(self, x: torch.Tensor, mask=None, attn_mask=None,
                T=None, H_patches=None, W_patches=None) -> torch.Tensor:
        # (B, N, D) → (B, N, D)
        x = x + self.drop_path(
            self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask,
                      T=T, H_patches=H_patches, W_patches=W_patches)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _DropPath(nn.Module):
    """Stochastic Depth per sample"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor = torch.floor(rand_tensor + keep_prob)
        return x.div(keep_prob) * rand_tensor


# ============================================================
# Vision Transformer エンコーダ (V-JEPA 2.1 対応)
# ============================================================

class VisionTransformer(nn.Module):
    """
    V-JEPA 2.1 のエンコーダ: Vision Transformer

    動画・画像の両モダリティに対応。
    V-JEPA 2.1では:
      - Multi-Modal Tokenizer: 動画=3D Conv, 画像=2D Conv
      - Modality Embedding: 学習可能な img/vid 識別トークン
      - Hierarchical Outputs: out_layers指定時に中間層出力も返す

    入力:
        x: (B, 3, T, H, W) 動画  または (B, 3, H, W) 画像
        masks: list of (B, N_ctx) コンテキストパッチのインデックス (省略可)

    出力 (out_layers=None):
        x: (B, N_ctx, D)  ← マスク適用後のパッチ数

    出力 (out_layers指定):
        list of (B, N_ctx, D)  ← 各指定層のnorm済み出力のリスト

    記号:
        B: バッチサイズ
        N: 総パッチ数 = (T/tubelet) * (H/patch) * (W/patch)
        N_ctx: コンテキスト(可視)パッチ数
        D: embed_dim
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        out_layers: list = None,         # Deep Self-Supervision用: 出力する中間層インデックス
        use_silu: bool = False,
        use_rope: bool = False,          # Rotary Position Embedding使用フラグ
        modality_embedding: bool = False, # V-JEPA 2.1: モダリティ識別埋め込み
        img_temporal_dim_size: int = None, # 画像の時間次元サイズ (1の場合に画像として処理)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.img_temporal_dim_size = img_temporal_dim_size
        self.is_video = num_frames > 1

        # ========================================
        # Multi-Modal Tokenizer
        # 動画: 3D Conv (tubelet), 画像: 2D Conv
        # ========================================
        if self.is_video:
            self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
            self.num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        else:
            self.patch_embed = PatchEmbed2D(patch_size, in_chans, embed_dim)
            self.num_patches = (img_size // patch_size) ** 2

        # ========================================
        # 位置埋め込み
        # use_rope=True の場合はRoPEを各ブロックで適用
        # use_rope=False の場合はSin-Cos位置埋め込みを加算
        # ========================================
        self.use_rope = use_rope
        if not use_rope:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )
            # Sin-Cos位置埋め込みで初期化
            self._init_pos_embed()

        # ========================================
        # Modality Embedding (V-JEPA 2.1 新機能)
        # 画像/動画を区別する学習可能なトークン
        # ========================================
        self.modality_embedding = modality_embedding
        if modality_embedding:
            self.img_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.vid_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ========================================
        # Transformer Blocks
        # ========================================
        # Stochastic Depth: 深い層ほどdrop_pathを大きく
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], use_silu=use_silu,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_pos_embed(self):
        """Sin-Cos位置埋め込みで初期化"""
        if self.is_video:
            grid_size = int(self.num_patches ** (1/3) + 0.5)  # 近似
            grid_h = grid_w = int((self.num_patches // (self.num_frames // self.tubelet_size)) ** 0.5)
            grid_d = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(self.embed_dim, grid_h, grid_w, grid_d)
        else:
            grid_size = int(self.num_patches ** 0.5)
            sincos = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
        self.pos_embed.data.copy_(sincos.unsqueeze(0))

    def _init_weights(self):
        """重み初期化: Linear=truncated normal, LayerNorm=0/1"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # ブロックの出力投影を深さに応じてスケーリング
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp[-3].weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _determine_modality(self, x: torch.Tensor) -> str:
        """入力テンソルからモダリティを判定"""
        if x.ndim == 4:
            return "image"
        if self.img_temporal_dim_size is not None and x.shape[2] == self.img_temporal_dim_size:
            return "image"
        return "video"

    def interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力サイズが学習時と異なる場合に位置埋め込みを補間
        (cooldown phaseでの高解像度処理等に使用)

        入力:  x (B, C, T, H, W) or (B, C, H, W)
        出力:  pos_embed (1, N, embed_dim) 補間済み
        """
        pos_embed = self.pos_embed
        if x.ndim == 5:
            _, _, T, H, W = x.shape
            T_grid = T // self.tubelet_size
            H_grid = H // self.patch_size
            W_grid = W // self.patch_size
            N_new = T_grid * H_grid * W_grid
            if N_new == pos_embed.shape[1]:
                return pos_embed
            # trilinear補間で時空間補間
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = int((pos_embed.shape[1] // N_t) ** 0.5)
            D = pos_embed.shape[-1]
            pe = pos_embed.reshape(1, N_t, N_h, N_w, D).permute(0, 4, 1, 2, 3)
            scale = (T_grid / N_t, H_grid / N_h, W_grid / N_w)
            pe = F.interpolate(pe, scale_factor=scale, mode="trilinear")
            pe = pe.permute(0, 2, 3, 4, 1).reshape(1, -1, D)
            return pe
        else:
            _, _, H, W = x.shape
            N_new = (H // self.patch_size) * (W // self.patch_size)
            if N_new == pos_embed.shape[1]:
                return pos_embed
            # bicubic補間
            N = pos_embed.shape[1]
            n = int(N ** 0.5)
            D = pos_embed.shape[-1]
            pe = pos_embed.reshape(1, n, n, D).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(H // self.patch_size, W // self.patch_size), mode="bicubic")
            pe = pe.permute(0, 2, 3, 1).reshape(1, -1, D)
            return pe

    def forward(self, x: torch.Tensor, masks: list = None) -> torch.Tensor:
        """
        入力:
            x:     (B, 3, T, H, W) 動画  または  (B, 3, H, W) 画像
            masks: list of (B, N_ctx) 可視パッチのインデックス (省略可)
                   省略時は全パッチを処理

        出力:
            out_layers=None の場合:  (B*len(masks), N_ctx, D)
            out_layers指定の場合:   list of (B*len(masks), N_ctx, D)
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # ========================================
        # Step 1: モダリティ判定とパッチ埋め込み
        # ========================================
        modality = self._determine_modality(x)

        # x: (B, 3, T, H, W) or (B, 3, H, W)
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)

        # ========================================
        # Step 2: 位置埋め込みの加算
        # ========================================
        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x if masks is None else x)
            x = x + pos_embed  # (B, N_patches, embed_dim)

        # ========================================
        # Step 3: Modality Embedding の加算 (V-JEPA 2.1)
        # ========================================
        if self.modality_embedding:
            if modality == "image":
                x = x + self.img_embed  # (B, N, D) + broadcast(1, 1, D)
            else:
                x = x + self.vid_embed

        # ========================================
        # Step 4: マスキング (コンテキストトークンのみ残す)
        # ========================================
        if masks is not None:
            x = apply_masks(x, masks)
            # x: (B*len(masks), N_ctx, embed_dim)
            masks_cat = torch.cat(masks, dim=0)  # (B*len(masks), N_ctx)
        else:
            masks_cat = None

        # ========================================
        # Step 5: Transformer Blocksを通す
        # ========================================
        # T, H_patches, W_patches はRoPE用の時空間次元情報
        # (sincos pos_embedの場合は使われない)
        T_grid = None
        if hasattr(self, 'num_frames') and self.is_video:
            pass  # RoPE使用時のみ必要

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks_cat)  # (B*len(masks), N_ctx, D)
            # Deep Self-Supervision: 指定された中間層の出力を収集
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))  # LayerNorm後の中間出力

        # ========================================
        # Step 6: 最終LayerNormと出力
        # ========================================
        if self.out_layers is not None:
            # 最終層も収集してリストで返す
            outs.append(self.norm(x))
            # 各要素: (B*len(masks), N_ctx, D)
            return outs

        x = self.norm(x)
        return x  # (B*len(masks), N_ctx, D)


# ============================================================
# ViTバリアント定義
# ============================================================

def vit_base(**kwargs):
    """ViT-B: 80M params (蒸留モデル)"""
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)


def vit_large(**kwargs):
    """ViT-L: 300M params"""
    return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def vit_giant(**kwargs):
    """ViT-g: 1B params (V-JEPA 2)"""
    return VisionTransformer(embed_dim=1408, depth=40, num_heads=22,
                              mlp_ratio=48/11, **kwargs)


def vit_gigantic(**kwargs):
    """ViT-G: 2B params (V-JEPA 2.1)"""
    return VisionTransformer(embed_dim=1664, depth=48, num_heads=26,
                              mlp_ratio=64/13, **kwargs)


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("V-JEPA 2.1 エンコーダ 動作確認")
    print("=" * 60)

    # ----------------------------------------
    # ViT-L 動画エンコーダ (V-JEPA 2 スタイル)
    # ----------------------------------------
    print("\n[1] ViT-L 動画エンコーダ (マスクなし)")
    encoder = vit_large(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1024,
        depth=4,        # テスト用に浅く
        num_heads=16,
    )
    encoder.eval()

    B, C, T, H, W = 2, 3, 16, 256, 256
    x_vid = torch.randn(B, C, T, H, W)
    out = encoder(x_vid)
    # N = (16/2) * (256/16) * (256/16) = 8 * 16 * 16 = 2048
    print(f"  入力:  {x_vid.shape}")
    print(f"  出力:  {out.shape}")
    assert out.shape == (B, 2048, 1024), f"Expected (2, 2048, 1024), got {out.shape}"

    # ----------------------------------------
    # マスク付きエンコーダ
    # ----------------------------------------
    print("\n[2] ViT-L 動画エンコーダ (マスクあり)")
    N_ctx = 700  # 可視トークン数
    masks_enc = [torch.randperm(2048)[:N_ctx].unsqueeze(0).expand(B, -1)
                 for _ in range(1)]  # 1種類のマスク
    out_masked = encoder(x_vid, masks=masks_enc)
    print(f"  入力:  {x_vid.shape}")
    print(f"  masks: 各 {masks_enc[0].shape}  (可視パッチインデックス)")
    print(f"  出力:  {out_masked.shape}")
    assert out_masked.shape == (B, N_ctx, 1024)

    # ----------------------------------------
    # V-JEPA 2.1: 中間層出力あり (Deep Self-Supervision)
    # ----------------------------------------
    print("\n[3] ViT-L V-JEPA 2.1 (中間層出力 + Modality Embedding)")
    encoder_21 = vit_large(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        depth=12,       # テスト用12層
        num_heads=16,
        out_layers=[2, 5, 8, 11],  # 4つの中間層からhierarchical出力
        modality_embedding=True,
    )
    encoder_21.eval()

    out_hierarchical = encoder_21(x_vid, masks=masks_enc)
    print(f"  中間層出力の数: {len(out_hierarchical)}")
    for i, o in enumerate(out_hierarchical):
        print(f"    layer {i}: {o.shape}")
    # 各要素: (B, N_ctx, D=1024)
    assert all(o.shape == (B, N_ctx, 1024) for o in out_hierarchical)

    # ----------------------------------------
    # 画像エンコーダ (V-JEPA 2.1 Multi-Modal)
    # ----------------------------------------
    print("\n[4] ViT-L 画像エンコーダ (2D Conv)")
    encoder_img = vit_large(
        img_size=256,
        patch_size=16,
        num_frames=1,  # 画像はnum_frames=1
        tubelet_size=1,
        depth=4,
        num_heads=16,
        modality_embedding=True,
    )
    encoder_img.eval()

    x_img = torch.randn(B, 3, 256, 256)
    out_img = encoder_img(x_img)
    # N = (256/16) * (256/16) = 256
    print(f"  入力:  {x_img.shape}")
    print(f"  出力:  {out_img.shape}")
    assert out_img.shape == (B, 256, 1024)

    print("\n全テスト通過!")
