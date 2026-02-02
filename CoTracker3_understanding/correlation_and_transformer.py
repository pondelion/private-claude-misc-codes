"""
CoTracker3 - 4D相関計算 + EfficientUpdateFormer (Transformer)
論文: https://arxiv.org/abs/2410.11831

【このファイルの概要】
CoTracker3の核心部分:
1. 4D相関ボリューム計算 (49×49 → MLP → 256)
2. Transformer入力の組み立て (1110次元)
3. EfficientUpdateFormer (時間アテンション + Virtual Tracksベース空間アテンション)
4. 反復リファインメント (M=4回)

【公式コードの対応箇所】
- cotracker/models/core/cotracker/cotracker3_online.py → forward_window (相関計算)
- cotracker/models/core/cotracker/cotracker.py → EfficientUpdateFormer, CrossAttnBlock
- cotracker/models/core/cotracker/blocks.py → Attention, AttnBlock, Mlp

========================================
4D相関の設計思想 (論文 Section 3.1)
========================================

従来手法 (CoTracker2):
  - track_feat: (B, S, N, C) のグローバル特徴
  - fmapsとの全体相関 → CorrBlock.corr() + CorrBlock.sample()
  - 問題: 特徴が1ベクトルのみ → 表現力に限界

CoTracker3:
  - track_feat_support: (B, N, 7, 7, C)  ← 49点のサポートグリッド
  - 推定位置の近傍: (B, T, N, 7, 7, C)  ← 49点の近傍グリッド
  - einsum → (B, T, N, 7, 7, 7, 7) = 49×49 の密な4D相関ボリューム
  - MLP: 2401 → 384 → 256

利点:
  - 局所的な空間構造を保持した相関
  - LocoTrackのadaptive/local correlation volumeと同等の精度を、
    単純なMLP処理で達成

========================================
Transformer入力の構成 (1110次元)
========================================

vis            : 1    可視性 (logit)
conf           : 1    信頼度 (logit)
corr_embs      : 1024  4D相関特徴 (256 × 4レベル)
rel_pos_emb    : 84    相対変位Fourier Encoding

計: 1 + 1 + 1024 + 84 = 1110

========================================
EfficientUpdateFormer の構造
========================================

入力射影: Linear(1110 → 384)

Virtual Tracks を結合: (B, N+64, T, 384)

3回繰り返し:
  [Time Attention]  (B*(N+64), T, 384)  — 各点の全フレーム間
  [Space Attention via Virtual Tracks]  (B*T, N+64, 384):
    1. virtual_tokens ← CrossAttn(virtual, real)   実→仮想
    2. virtual_tokens ← SelfAttn(virtual)           仮想同士
    3. point_tokens ← CrossAttn(real, virtual)      仮想→実

Virtual Tracks を除去: (B, N, T, 384)

出力ヘッド:
  flow_head:     Linear(384 → 2)  [delta_x, delta_y]
  vis_conf_head: Linear(384 → 2)  [delta_vis, delta_conf]
  → concat → (B, N, T, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# ========================================
# 4D相関ボリューム計算
# ========================================
class CorrelationComputer:
    """
    4D相関ボリュームの計算

    ========================================
    概要
    ========================================
    クエリ点の「サポート特徴」と現在の推定位置の「近傍特徴」間の
    密な4D相関ボリュームを計算し、MLPで圧縮する。

    ========================================
    入力
    ========================================
    track_feat_support: (B, N, 7, 7, D)   ← クエリフレームの7×7サポート特徴
    corr_feat:          (B, T, N, 7, 7, D) ← 各フレームの推定座標周辺7×7特徴
    corr_mlp:           MLP(2401 → 384 → 256)

    ========================================
    計算フロー
    ========================================
    1. einsum('btnhwc,bnijc->btnhwij', corr_feat, track_feat_support)
       → (B, T, N, 7, 7, 7, 7) = 49×49の密な相関

    2. reshape → (B*T*N, 2401)

    3. MLP → (B*T*N, 256)

    4. reshape → (B, T, N, 256)

    これを4レベル分繰り返して結合 → (B, T, N, 1024)
    """

    def __init__(self, corr_radius: int = 3, corr_levels: int = 4, latent_dim: int = 128):
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.latent_dim = latent_dim
        r = 2 * corr_radius + 1  # = 7

        # 相関MLP: 49×49=2401 → 384 → 256
        # 全レベルで同じMLPを共有
        self.corr_mlp = nn.Sequential(
            nn.Linear(r * r * r * r, 384),  # 2401 → 384
            nn.GELU(approximate="tanh"),
            nn.Linear(384, 256),             # 384 → 256
        )

    def compute_4d_correlation(
        self,
        fmaps_pyramid: List[torch.Tensor],
        coords: torch.Tensor,
        track_feat_support_pyramid: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        4レベルの4D相関ボリュームを計算・結合

        ========================================
        Shape
        ========================================
        入力:
          fmaps_pyramid: [
            Level 0: (B, T, 128, H/4,  W/4),
            Level 1: (B, T, 128, H/8,  W/8),
            Level 2: (B, T, 128, H/16, W/16),
            Level 3: (B, T, 128, H/32, W/32),
          ]
          coords:      (B, T, N, 2) — 特徴マップスケール (Level 0基準)
          track_feat_support_pyramid: [
            Level i: (B, 1, 7, 7, N, 128) → (B, N, 7, 7, 128) にreshape
          ]

        出力: (B, T, N, 1024)  — 256 × 4レベル

        ========================================
        処理フロー (各レベル)
        ========================================
        for level in range(4):
            # 1. 座標をレベルに合わせてスケーリング
            coords_level = coords / (2 ** level)

            # 2. 推定座標周辺の近傍特徴取得
            corr_feat = get_correlation_feat(fmaps_pyramid[level], coords_level)
            # corr_feat: (B, T, N, 7, 7, 128)

            # 3. サポート特徴を取得
            track_support = track_feat_support_pyramid[level]
            # track_support: (B, N, 7, 7, 128)

            # 4. 4D相関ボリューム計算
            corr_volume = einsum('btnhwc,bnijc->btnhwij', corr_feat, track_support)
            # corr_volume: (B, T, N, 7, 7, 7, 7)

            # 5. MLP で圧縮
            corr_emb = corr_mlp(corr_volume.reshape(B*T*N, 2401))
            # corr_emb: (B*T*N, 256)

        # 全レベル結合: (B, T, N, 256×4) = (B, T, N, 1024)
        """
        B, T, N, _ = coords.shape
        r = 2 * self.corr_radius + 1  # = 7
        all_corr_embs = []

        for level in range(self.corr_levels):
            # === 1. 座標スケーリング ===
            # Level i では空間解像度が 1/2^i なので座標も 1/2^i
            coords_level = coords.reshape(B * T, N, 2) / (2 ** level)

            # === 2. 近傍特徴取得 ===
            # get_correlation_feat(fmaps_pyramid[level], coords_level)
            # → (B, T, N, 7, 7, 128)
            corr_feat = self._sample_neighbor_features(
                fmaps_pyramid[level], coords_level
            )  # (B, T, N, r, r, D)

            # === 3. サポート特徴 ===
            # (B, 1, r, r, N, D) → (B, N, r, r, D)
            track_support = (
                track_feat_support_pyramid[level]
                .view(B, 1, r, r, N, self.latent_dim)
                .squeeze(1)
                .permute(0, 3, 1, 2, 4)
            )
            # track_support: (B, N, r, r, D)

            # === 4. 4D相関ボリューム ===
            # einsum: (B,T,N,h,w,c) × (B,N,i,j,c) → (B,T,N,h,w,i,j)
            corr_volume = torch.einsum(
                "btnhwc,bnijc->btnhwij", corr_feat, track_support
            )
            # corr_volume: (B, T, N, 7, 7, 7, 7) = 49×49

            # === 5. MLP圧縮 ===
            corr_emb = self.corr_mlp(
                corr_volume.reshape(B * T * N, r * r * r * r)
            )
            # corr_emb: (B*T*N, 256)
            all_corr_embs.append(corr_emb)

        # === 全レベル結合 ===
        corr_embs = torch.cat(all_corr_embs, dim=-1)  # (B*T*N, 256×4=1024)
        corr_embs = corr_embs.reshape(B, T, N, -1)     # (B, T, N, 1024)
        return corr_embs

    def _sample_neighbor_features(
        self, fmaps: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """近傍特徴のサンプリング (擬似実装)"""
        B, T, D, H, W = fmaps.shape
        N = coords.shape[1]
        r = 2 * self.corr_radius + 1
        return torch.randn(B, T, N, r, r, D, device=fmaps.device)


# ========================================
# Transformer入力の組み立て
# ========================================
def build_transformer_input(
    vis: torch.Tensor,
    conf: torch.Tensor,
    corr_embs: torch.Tensor,
    coords: torch.Tensor,
    model_resolution: Tuple[int, int] = (384, 512),
    stride: int = 4,
) -> torch.Tensor:
    """
    Transformer入力トークンを構築

    ========================================
    Shape
    ========================================
    入力:
      vis:       (B, T, N, 1)   可視性 logit
      conf:      (B, T, N, 1)   信頼度 logit
      corr_embs: (B, T, N, 1024) 4D相関特徴
      coords:    (B, T, N, 2)    座標推定 (特徴マップスケール)

    出力: (B, N, T, 1110)

    ========================================
    相対変位の計算
    ========================================
    前方変位: coords[t] - coords[t+1]  (次フレームとの差分)
    後方変位: coords[t] - coords[t-1]  (前フレームとの差分)

    → 結合: [forward_x, forward_y, backward_x, backward_y] = 4次元
    → Fourier Encoding (deg 0~9): 4 + 4×10×2 = 84次元

    ========================================
    スケーリング
    ========================================
    変位はモデル解像度で正規化:
      scale = [model_resolution[1], model_resolution[0]] / stride
            = [512, 384] / 4 = [128, 96]
      rel_forward = rel_forward / scale
      rel_backward = rel_backward / scale

    → 相対変位が [-1, 1] 程度の範囲に正規化される
    """
    B, T, N, _ = coords.shape
    device = coords.device

    # === 相対変位計算 ===
    # 前方: coords[t] - coords[t+1] (t=0..T-2, 最後は0パディング)
    rel_forward = coords[:, :-1] - coords[:, 1:]
    rel_forward = F.pad(rel_forward, (0, 0, 0, 0, 0, 1))  # 時間方向の末尾に0

    # 後方: coords[t] - coords[t-1] (t=1..T-1, 最初は0パディング)
    rel_backward = coords[:, 1:] - coords[:, :-1]
    rel_backward = F.pad(rel_backward, (0, 0, 0, 0, 1, 0))  # 時間方向の先頭に0

    # モデル解像度でスケーリング
    scale = (
        torch.tensor(
            [model_resolution[1], model_resolution[0]],
            device=device, dtype=coords.dtype,
        )
        / stride
    )
    rel_forward = rel_forward / scale    # (B, T, N, 2) / (2,) = (B, T, N, 2)
    rel_backward = rel_backward / scale

    # === Fourier Positional Encoding ===
    rel_coords = torch.cat([rel_forward, rel_backward], dim=-1)  # (B, T, N, 4)
    rel_pos_emb = posenc(rel_coords, min_deg=0, max_deg=10)      # (B, T, N, 84)

    # === 結合 ===
    transformer_input = torch.cat([
        vis,          # (B, T, N, 1)
        conf,         # (B, T, N, 1)
        corr_embs,    # (B, T, N, 1024)
        rel_pos_emb,  # (B, T, N, 84)
    ], dim=-1)
    # transformer_input: (B, T, N, 1110)

    # (B, T, N, 1110) → (B, N, T, 1110)  ※Transformerは各点のフレーム列として処理
    transformer_input = transformer_input.permute(0, 2, 1, 3)
    return transformer_input


def posenc(x: torch.Tensor, min_deg: int = 0, max_deg: int = 10) -> torch.Tensor:
    """Fourier Positional Encoding (main_flow.py と同じ)"""
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x, four_feat], dim=-1)


# ========================================
# Attention / AttnBlock / CrossAttnBlock
# ========================================
class Attention(nn.Module):
    """Multi-Head Attention (Self/Cross対応)"""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        dim_head: int = 48,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads  # = 384
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.num_heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
          x:         (B, N1, C)
          context:   (B, N2, C) or None (Self-Attention)
          attn_bias: (B, heads, N1, N2) or None

        出力: (B, N1, C)

        ========================================
        計算
        ========================================
        Q = W_q(x):       (B, N1, heads, head_dim)
        K = W_k(ctx):     (B, N2, heads, head_dim)
        V = W_v(ctx):     (B, N2, heads, head_dim)

        attn = softmax(Q @ K^T / sqrt(d) + attn_bias)
        out = attn @ V
        """
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)

        ctx = context if context is not None else x
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        N2 = ctx.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N1, N2)

        if attn_bias is not None:
            sim = sim + attn_bias

        attn = sim.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(out)


class Mlp(nn.Module):
    """2層MLP (GELU)"""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttnBlock(nn.Module):
    """
    Pre-LayerNorm Self-Attention Block

    x = x + Attn(LN(x), attn_bias)
    x = x + MLP(LN(x))
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        mask: (B*T, N) or None
        → attn_bias: (B*T, heads, N, N)  マスク外は -inf
        """
        attn_bias = None
        if mask is not None:
            # (B*T, N) → (B*T, N, N) ペアワイズ可視性マスク
            mask_2d = mask[:, None] * mask[:, :, None]  # (B*T, N, N)
            mask_2d = mask_2d.unsqueeze(1).expand(-1, self.attn.num_heads, -1, -1)
            max_neg = -torch.finfo(x.dtype).max
            attn_bias = (~mask_2d) * max_neg

        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """
    Cross-Attention Block (Virtual Tracks用)

    x = x + CrossAttn(LN(x), LN(context))
    x = x + MLP(LN(x))

    ========================================
    使用箇所
    ========================================
    1. space_virtual2point_blocks: virtual ← CrossAttn(virtual, real_points)
       → 実トラックの情報を仮想トラックに集約

    2. space_point2virtual_blocks: real ← CrossAttn(real_points, virtual)
       → 仮想トラックから実トラックに情報を分配

    ========================================
    Mask処理
    ========================================
    Onlineモードでは、まだ出現していないクエリ点を
    マスクで無視する。mask: (B*T, N) or (B*T, 64)
    """

    def __init__(
        self, hidden_size: int, context_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        x:       (B*T, N_query, C)  e.g. virtual tokens (B*T, 64, 384)
        context: (B*T, N_ctx, C)    e.g. point tokens  (B*T, N, 384)
        mask:    (B*T, N) or None

        出力: (B*T, N_query, C)
        """
        attn_bias = None
        if mask is not None:
            # mask の形状に応じてattn_biasを構築
            if mask.shape[1] == x.shape[1]:
                # xがマスク対象 (Q側マスク)
                attn_bias = mask[:, None, :, None].expand(
                    -1, self.cross_attn.num_heads, -1, context.shape[1]
                )
            else:
                # contextがマスク対象 (KV側マスク)
                attn_bias = mask[:, None, None].expand(
                    -1, self.cross_attn.num_heads, x.shape[1], -1
                )
            max_neg = -torch.finfo(x.dtype).max
            attn_bias = (~attn_bias) * max_neg

        x = x + self.cross_attn(
            self.norm1(x),
            context=self.norm_context(context),
            attn_bias=attn_bias,
        )
        x = x + self.mlp(self.norm2(x))
        return x


# ========================================
# EfficientUpdateFormer (詳細版)
# ========================================
class EfficientUpdateFormer(nn.Module):
    """
    CoTracker3のTransformerモジュール (詳細実装)

    ========================================
    Virtual Tracks の仕組み (論文 Section 3.2)
    ========================================

    背景:
    - N個の追跡点間の相互情報交換 (Space Attention) が重要
    - 直接的なSelf-Attention: O(N²) → N=1000以上で計算コスト大

    解決策: Virtual Tracks
    - 64個の学習可能なプロキシトークン (nn.Parameter)
    - 実トラック → 仮想トラック → 実トラック の2段階交換
    - 計算量: O(N×64 + 64² + N×64) = O(N×128 + 64²) ≈ O(N)

    処理順序 (公式コード cotracker.py):
    1. virtual ← CrossAttn(virtual, point_tokens)  # 実→仮想
    2. virtual ← SelfAttn(virtual)                  # 仮想同士
    3. point  ← CrossAttn(point_tokens, virtual)    # 仮想→実

    ※ この順序は公式コードの通り。
      論文の図では 仮想SelfAttn → 実→仮想CrossAttn → 仮想→実CrossAttn
      のように見えるが、実装は上記の順序。

    ========================================
    Time/Space の交互実行
    ========================================
    time_depth=3, space_depth=3 の場合:
    - time_block[0] → space[0]
    - time_block[1] → space[1]
    - time_block[2] → space[2]

    一般的に: time_depth >= space_depth で、
    i % (time_depth // space_depth) == 0 のときに空間アテンションを実行。
    (time_depth=6, space_depth=3 なら、偶数回のみ空間)

    ========================================
    出力ヘッド: flow_head と vis_conf_head の分離
    ========================================
    linear_layer_for_vis_conf=True (CoTracker3のデフォルト):
    - flow_head:     Linear(384 → 2)  [delta_x, delta_y]
    - vis_conf_head: Linear(384 → 2)  [delta_vis, delta_conf]

    分離の理由:
    - Pseudo-Label学習時に vis_conf_head のみフリーズ可能
    - 教師なし実データでの学習時、可視性・信頼度の劣化を防ぐ
    """

    def __init__(
        self,
        input_dim: int = 1110,
        hidden_size: int = 384,
        output_dim: int = 4,
        time_depth: int = 3,
        space_depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_virtual_tracks: int = 64,
        add_space_attn: bool = True,
        linear_layer_for_vis_conf: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tracks = num_virtual_tracks
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf

        # === 入力射影: 1110 → 384 ===
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)

        # === 出力ヘッド ===
        if linear_layer_for_vis_conf:
            self.flow_head = nn.Linear(hidden_size, output_dim - 2, bias=True)  # 384 → 2
            self.vis_conf_head = nn.Linear(hidden_size, 2, bias=True)           # 384 → 2
        else:
            self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)      # 384 → 4

        # === Virtual Tracks ===
        self.virtual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)  # (1, 64, 1, 384)
        )

        # === Time Attention Blocks ===
        self.time_blocks = nn.ModuleList([
            AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(time_depth)
        ])

        # === Space Attention Blocks (Virtual Tracks経由) ===
        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList([
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])
            self.space_virtual2point_blocks = nn.ModuleList([
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])
            self.space_point2virtual_blocks = nn.ModuleList([
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(space_depth)
            ])

        # === 重み初期化 ===
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier + 出力ヘッドは小さいstdで初期化

        output_dim=4 → 初期のdeltaがほぼ0に近い
        → 最初の反復では座標変化がほぼ0 → 安定した学習
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # 出力ヘッドは小さい値で初期化 (初期delta ≈ 0)
        nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
        if self.linear_layer_for_vis_conf:
            nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

    def forward(
        self,
        input_tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_space_attn: bool = True,
    ) -> torch.Tensor:
        """
        ========================================
        処理フロー (入力 → 出力)
        ========================================

        入力: input_tensor (B, N, T, 1110)

        Step 1: 入力射影
          tokens = Linear(input_tensor)  → (B, N, T, 384)

        Step 2: Virtual Tracks 結合
          virtual = self.virtual_tracks.expand(B, 64, T, 384)
          tokens = cat([tokens, virtual])  → (B, N+64, T, 384)

        Step 3: 交互アテンション (3回)
          for i in range(3):
            # Time Attention: 各点の全フレーム
            tokens → (B*(N+64), T, 384) → time_block → (B*(N+64), T, 384)

            # Space Attention via Virtual Tracks
            tokens → (B*T, N+64, 384) → split: point(N) + virtual(64)
              virtual ← CrossAttn(virtual, point)   # 情報集約
              virtual ← SelfAttn(virtual)            # 仮想間交換
              point ← CrossAttn(point, virtual)      # 情報分配
            tokens ← cat([point, virtual])

        Step 4: Virtual Tracks 除去
          tokens = tokens[:, :N]  → (B, N, T, 384)

        Step 5: 出力ヘッド
          flow = flow_head(tokens)            → (B, N, T, 2)
          vis_conf = vis_conf_head(tokens)    → (B, N, T, 2)
          delta = cat([flow, vis_conf])       → (B, N, T, 4)

        ========================================
        Shape変換のまとめ
        ========================================
        Time Attn:  (B, N+64, T, C) → reshape (B*(N+64), T, C) → 各点内Self-Attn
        Space Attn: (B, N+64, T, C) → permute (B, T, N+64, C) → reshape (B*T, N+64, C)
                    → split → Cross/SelfAttn → cat → reshape back
        """
        B, N_input, T, _ = input_tensor.shape

        # === Step 1: 入力射影 ===
        tokens = self.input_transform(input_tensor)  # (B, N, T, 384)

        # === Step 2: Virtual Tracks 結合 ===
        virtual_tokens = self.virtual_tracks.repeat(B, 1, T, 1)  # (B, 64, T, 384)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)       # (B, N+64, T, 384)

        _, N_total, _, _ = tokens.shape  # N_total = N + 64
        space_idx = 0

        # === Step 3: 交互アテンション ===
        for i in range(len(self.time_blocks)):
            # --- Time Attention ---
            # 各点 (仮想含む) の全フレーム間で Self-Attention
            time_tokens = tokens.contiguous().view(B * N_total, T, -1)
            time_tokens = self.time_blocks[i](time_tokens)
            tokens = time_tokens.view(B, N_total, T, -1)

            # --- Space Attention (条件付き実行) ---
            should_do_space = (
                add_space_attn
                and self.add_space_attn
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            )

            if should_do_space:
                # (B, N+64, T, C) → (B*T, N+64, C)
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N_total, -1)

                # Split: 実トラック vs 仮想トラック
                point_tokens = space_tokens[:, :N_total - self.num_virtual_tracks]  # (B*T, N, C)
                virt_tokens = space_tokens[:, N_total - self.num_virtual_tracks:]   # (B*T, 64, C)

                # Step A: 実→仮想 (Cross-Attention)
                virt_tokens = self.space_virtual2point_blocks[space_idx](
                    virt_tokens, point_tokens, mask=mask
                )

                # Step B: 仮想同士 (Self-Attention)
                virt_tokens = self.space_virtual_blocks[space_idx](virt_tokens)

                # Step C: 仮想→実 (Cross-Attention)
                point_tokens = self.space_point2virtual_blocks[space_idx](
                    point_tokens, virt_tokens, mask=mask
                )

                # 再結合
                space_tokens = torch.cat([point_tokens, virt_tokens], dim=1)  # (B*T, N+64, C)
                tokens = space_tokens.view(B, T, N_total, -1).permute(0, 2, 1, 3)
                space_idx += 1

        # === Step 4: Virtual Tracks 除去 ===
        tokens = tokens[:, :N_total - self.num_virtual_tracks]  # (B, N, T, 384)

        # === Step 5: 出力ヘッド ===
        flow = self.flow_head(tokens)  # (B, N, T, 2)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)  # (B, N, T, 2)
            delta = torch.cat([flow, vis_conf], dim=-1)  # (B, N, T, 4)
        else:
            delta = flow  # (B, N, T, 4) if output_dim=4

        return delta


# ========================================
# 反復リファインメント
# ========================================
def iterative_refinement(
    updateformer: EfficientUpdateFormer,
    fmaps_pyramid: List[torch.Tensor],
    track_feat_support_pyramid: List[torch.Tensor],
    queried_coords: torch.Tensor,
    model_resolution: Tuple[int, int],
    stride: int,
    time_emb: torch.Tensor,
    iters: int = 4,
    corr_radius: int = 3,
    corr_levels: int = 4,
    latent_dim: int = 128,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    M回の反復リファインメントの全体フロー

    ========================================
    アルゴリズム
    ========================================
    初期化:
      coords = queried_coords  (全フレームにコピー)
      vis = 0
      conf = 0

    for m = 1 to M:
      coords = coords.detach()  # 勾配切断

      # 1. 4D相関計算 (4レベル)
      corr_embs = compute_4d_correlation(fmaps_pyramid, coords, support)

      # 2. 相対変位 Fourier Encoding
      rel_pos_emb = posenc([forward_disp, backward_disp])

      # 3. Transformer入力組み立て
      x = cat([vis, conf, corr_embs, rel_pos_emb])  # (B, T, N, 1110)

      # 4. 時間埋め込み加算
      x = x + time_emb  # (B*N, T, 1110) + (1, T, 1110)

      # 5. Transformer更新
      delta = updateformer(x)  # (B, N, T, 4)

      # 6. 座標・可視性・信頼度更新
      coords += delta[:2]
      vis += delta[2]
      conf += delta[3]

      # 7. 予測保存
      preds.append(coords * stride, sigmoid(vis), sigmoid(conf))

    ========================================
    勾配切断 (coords.detach()) の意味
    ========================================
    - 各反復でcoordsの勾配を切断
    - 効果: 反復1の勾配が反復2以降に影響しない
    - 理由: 安定した学習 (RAFT/RAFTに由来)
    - 損失は各反復の予測に指数減衰重みで適用

    ========================================
    累積更新 vs 直接予測
    ========================================
    CoTracker3は「累積更新」方式:
    - coords = coords + delta_coords (差分を加算)
    - vis = vis + delta_vis
    - conf = conf + delta_conf

    出力時:
    - coords → coords * stride (画像座標に復元)
    - vis → sigmoid(vis) (0~1に変換)
    - conf → sigmoid(conf) (0~1に変換)
    """
    B, T, _, _, _ = fmaps_pyramid[0].shape
    N = queried_coords.shape[1]
    device = queried_coords.device
    r = 2 * corr_radius + 1

    # === 初期化 ===
    coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()
    vis = torch.zeros((B, T, N), device=device).float()
    conf = torch.zeros((B, T, N), device=device).float()

    coord_preds, vis_preds, conf_preds = [], [], []

    corr_computer = CorrelationComputer(corr_radius, corr_levels, latent_dim)
    corr_computer.corr_mlp = corr_computer.corr_mlp  # 本来はモデルのMLPを共有

    for it in range(iters):
        # === 勾配切断 ===
        coords = coords.detach()  # 重要: 反復間の勾配を切断

        # === 4D相関計算 ===
        corr_embs = corr_computer.compute_4d_correlation(
            fmaps_pyramid, coords, track_feat_support_pyramid
        )
        # corr_embs: (B, T, N, 1024)

        # === Transformer入力組み立て ===
        transformer_input = build_transformer_input(
            vis[..., None], conf[..., None],
            corr_embs, coords,
            model_resolution, stride,
        )
        # transformer_input: (B, N, T, 1110)

        # === 時間埋め込み加算 ===
        x = transformer_input.reshape(B * N, T, -1)
        x = x + time_emb  # (1, T, 1110) をブロードキャスト
        x = x.view(B, N, T, -1)

        # === Transformer更新 ===
        delta = updateformer(x)  # (B, N, T, 4)

        # === 座標・可視性・信頼度更新 ===
        delta_coords = delta[..., :2].permute(0, 2, 1, 3)   # (B, T, N, 2)
        delta_vis = delta[..., 2].permute(0, 2, 1)           # (B, T, N)
        delta_conf = delta[..., 3].permute(0, 2, 1)          # (B, T, N)

        coords = coords + delta_coords
        vis = vis + delta_vis
        conf = conf + delta_conf

        # === 予測保存 ===
        coord_preds.append(coords[..., :2] * float(stride))  # 画像座標に復元
        vis_preds.append(torch.sigmoid(vis))
        conf_preds.append(torch.sigmoid(conf))

    return coord_preds, vis_preds, conf_preds


# ========================================
# デモ
# ========================================
def demo_correlation_and_transformer():
    """
    4D相関 + Transformerの処理フローをデモ
    """
    print("=" * 60)
    print("CoTracker3 4D相関 + Transformer デモ")
    print("=" * 60)

    # === パラメータ ===
    B, T, N = 1, 24, 100
    H, W = 384, 512
    D = 128
    stride = 4
    corr_radius = 3
    corr_levels = 4
    r = 2 * corr_radius + 1  # = 7

    # === 4D相関ボリューム ===
    print("\n[4D相関ボリューム]")
    print(f"  サポートグリッド: {r}×{r} = {r**2}点 (クエリフレームの固定特徴)")
    print(f"  近傍グリッド:     {r}×{r} = {r**2}点 (各フレームの推定位置周辺)")
    print(f"  相関ボリューム:   ({r},{r},{r},{r}) = {r**4} 要素")
    print(f"  MLP圧縮:         {r**4} → 384 → 256")
    print(f"  × {corr_levels} levels = {256 * corr_levels} 次元")

    # === Transformer入力 ===
    print("\n[Transformer入力]")
    print(f"  vis:       1")
    print(f"  conf:      1")
    print(f"  corr_embs: {256 * corr_levels}")
    print(f"  rel_pos:   84  (4次元 × Fourier deg=10 × sin/cos)")
    print(f"  合計:      {1 + 1 + 256 * corr_levels + 84}")

    # === EfficientUpdateFormer ===
    print("\n[EfficientUpdateFormer]")
    model = EfficientUpdateFormer(
        input_dim=1110,
        hidden_size=384,
        output_dim=4,
        time_depth=3,
        space_depth=3,
        num_virtual_tracks=64,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {param_count:,}")

    # テスト入力
    x = torch.randn(B, N, T, 1110)
    delta = model(x)
    print(f"  入力:  {x.shape}")
    print(f"  出力:  {delta.shape}")
    print(f"    delta[:2] = delta_coords (特徴マップスケール)")
    print(f"    delta[2]  = delta_vis (logit)")
    print(f"    delta[3]  = delta_conf (logit)")

    # === Virtual Tracks ===
    print("\n[Virtual Tracks の計算量比較]")
    print(f"  直接 Space Self-Attention: O(N²) = O({N}²) = {N**2:,}")
    print(f"  Virtual Tracks:")
    V = 64
    print(f"    real→virtual:    O(N×V) = O({N}×{V}) = {N*V:,}")
    print(f"    virtual SelfAttn: O(V²) = O({V}²) = {V**2:,}")
    print(f"    virtual→real:    O(N×V) = O({N}×{V}) = {N*V:,}")
    total_vt = 2 * N * V + V ** 2
    print(f"    合計: {total_vt:,}")
    print(f"    削減率: {N**2 / total_vt:.1f}x")

    # === 反復リファインメント ===
    print(f"\n[反復リファインメント]")
    print(f"  反復回数: M=4")
    print(f"  各反復:")
    print(f"    1. coords.detach()  → 勾配切断")
    print(f"    2. 4D相関計算      → (B, T, N, 1024)")
    print(f"    3. 変位Encoding     → (B, T, N, 84)")
    print(f"    4. Transformer入力  → (B, N, T, 1110)")
    print(f"    5. Transformer更新  → (B, N, T, 4)")
    print(f"    6. coords += delta  → 累積更新")
    print(f"  損失: 指数減衰重み γ^(M-m) で全反復の予測に適用")


if __name__ == "__main__":
    demo_correlation_and_transformer()
