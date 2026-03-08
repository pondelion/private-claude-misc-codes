"""
MiniCPM-V 4.5 - 統一3D-Resampler
================================================

2D Perceiver-Resamplerの実装と、動画用3D拡張（時間位置埋め込み追加）。
MiniCPM-V 4.5の効率性の核心モジュール。

論文: MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes
公式実装: omnilmm/model/resampler.py: Resampler

処理の流れ:
1. [画像] 各スライスの視覚特徴をcross-attentionでQ個のトークンに圧縮 (2D)
2. [動画] 隣接フレームをパッケージ化し、時間位置埋め込みを追加して圧縮 (3D)
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
N       : バッチ/スライス数
L_vis   : ViT出力のパッチトークン数 = (H/P) * (W/P)
Q       : 学習可能クエリ数 = grid_size^2 (64)
D_vis   : ViTの隠れ次元 (1792)
D_llm   : LLMの隠れ次元 = embed_dim (4096)
T_pkg   : 動画パッケージ数
F       : パッケージ内フレーム数
============================================================
"""

import math
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ========================================
# 位置埋め込みユーティリティ
# ========================================
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    2D正弦余弦位置埋め込みを生成する

    公式実装: omnilmm/model/resampler.py: get_2d_sincos_pos_embed()
    参考: MAE (facebook/mae)

    ========================================
    入力:
        embed_dim: 埋め込み次元 (4096)
        grid_size: グリッドサイズ (8)
        cls_token: CLSトークン用の位置を追加するか

    出力:
        pos_embed: (grid_size^2, embed_dim) or (1+grid_size^2, embed_dim)
            - 2D正弦余弦位置埋め込み
            - 前半D/2がH方向、後半D/2がW方向

    処理:
        1. H軸とW軸の1Dグリッドを作成
        2. 各軸で1D正弦余弦埋め込みを計算 (embed_dim/2 次元)
        3. H方向とW方向を結合 → embed_dim次元
    ========================================
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # W方向が先
    grid = np.stack(grid, axis=0)
    # grid: (2, grid_size, grid_size)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # pos_embed: (grid_size^2, embed_dim)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    2Dグリッドから位置埋め込みを計算

    ========================================
    入力:
        embed_dim: 埋め込み次元
        grid: (2, 1, grid_size, grid_size) - H方向とW方向

    出力:
        emb: (grid_size^2, embed_dim)
    ========================================
    """
    assert embed_dim % 2 == 0

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    # emb_h: (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    # emb_w: (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)
    # emb: (H*W, D)

    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    1D正弦余弦位置埋め込みを計算

    ========================================
    入力:
        embed_dim: 出力次元 (D/2)
        pos: 位置列 (M,)

    出力:
        emb: (M, D/2) - [sin, cos] を結合

    数式:
        omega_k = 1 / 10000^(2k/D)
        emb[m, 2k]   = sin(pos[m] * omega_k)
        emb[m, 2k+1] = cos(pos[m] * omega_k)
    ========================================
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    # omega: (D/4,)

    pos = pos.reshape(-1)
    # pos: (M,)

    out = np.einsum("m,d->md", pos, omega)
    # out: (M, D/4) - 外積

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    # emb: (M, D/2)

    return emb


def get_abs_pos(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
    """
    位置埋め込みをターゲットサイズに補間する

    公式実装: omnilmm/model/resampler.py: get_abs_pos()

    ========================================
    入力:
        abs_pos: (L_src, D) - 元の位置埋め込み
            L_src = src_grid_size^2
        tgt_size: ターゲットのトークン数
            L_tgt = tgt_grid_size^2

    出力:
        pos: (L_tgt, D) - 補間された位置埋め込み

    処理:
        src_size != tgt_size の場合、bicubic補間で
        (src_grid_size, src_grid_size) → (tgt_grid_size, tgt_grid_size)
    ========================================
    """
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
        # 出力: (tgt_size^2, D)
    else:
        return abs_pos


# ========================================
# 1D時間位置埋め込み (3D-Resampler用)
# ========================================
def get_1d_sincos_temporal_embed(embed_dim: int, max_len: int) -> np.ndarray:
    """
    1D正弦余弦時間位置埋め込みを生成する (動画の3D拡張用)

    論文Section 2.1.1:
        「We augment the learnable queries with both 2D spatial positional
         embedding, as used in image encoding, and temporal positional embedding.」

    ========================================
    入力:
        embed_dim: 埋め込み次元 (4096)
        max_len: 最大フレーム数 (例: 1080)

    出力:
        temporal_embed: (max_len, embed_dim)
    ========================================
    """
    positions = np.arange(max_len, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, positions)


# ========================================
# 2D-Resampler (画像用)
# ========================================
class Resampler2D(nn.Module):
    """
    2D Perceiver-Resampler

    公式実装: omnilmm/model/resampler.py: Resampler

    1層のcross-attentionで、grid_size^2個の学習可能クエリを使い
    可変長の視覚特徴を固定長に圧縮する。
    2D正弦余弦位置埋め込みを使用。

    ========================================
    入力:
        x: (N, L_vis, D_vis)
            - N: バッチ/スライス数
            - L_vis: ViT出力のパッチトークン数 (可変)
            - D_vis: ViT出力次元 (1792)

    出力:
        out: (N, Q, D_llm)
            - Q: grid_size^2 = 64 (固定)
            - D_llm: LLM埋め込み次元 (4096)

    圧縮率:
        448x448画像: L_vis=1024 → Q=64 (16倍圧縮)
    ========================================
    """

    def __init__(
        self,
        grid_size: int = 8,          # クエリグリッドサイズ (8 → Q=64)
        embed_dim: int = 4096,       # 出力/クエリ次元 = D_llm
        num_heads: int = 32,         # アテンションヘッド数 (embed_dim // 128)
        kv_dim: int = 1792,          # 入力次元 = D_vis (Noneならembed_dimと同じ)
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_queries = grid_size ** 2  # 64
        self.embed_dim = embed_dim         # 4096
        self.num_heads = num_heads         # 32

        # ========================================
        # 1. 固定の2D正弦余弦位置埋め込み
        # ========================================
        self.pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(embed_dim, grid_size)
            ).float()
        ).requires_grad_(False)
        # pos_embed: (Q, D_llm) = (64, 4096)
        # 固定パラメータ（学習しない）

        # ========================================
        # 2. 学習可能クエリ
        # ========================================
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)
        # query: (Q, D_llm) = (64, 4096)

        # ========================================
        # 3. KV射影 (D_vis → D_llm)
        # ========================================
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        # kv_proj: (*, D_vis) → (*, D_llm) = (*, 1792) → (*, 4096)

        # ========================================
        # 4. Cross-Attention
        # ========================================
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        # Q: (Q, N, D_llm), K: (L_vis, N, D_llm), V: (L_vis, N, D_llm)
        # → out: (Q, N, D_llm)

        # ========================================
        # 5. Layer Normalization
        # ========================================
        self.ln_q = norm_layer(embed_dim)     # クエリ正規化
        self.ln_kv = norm_layer(embed_dim)    # KV正規化
        self.ln_post = norm_layer(embed_dim)  # 出力正規化

        # ========================================
        # 6. 出力射影
        # ========================================
        self.proj = nn.Parameter(
            (embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim)
        )
        # proj: (D_llm, D_llm) = (4096, 4096)
        # ランダム行列射影（スケーリング付き）

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """重み初期化: Linear → truncated normal, LayerNorm → 標準初期化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        2D-Resamplerのフォワードパス

        ========================================
        入力:
            x: (N, L_vis, D_vis)
                - N: バッチ/スライス数
                - L_vis: 可変長のパッチトークン数
                - D_vis: ViT出力次元 (1792)

            attn_mask: Optional (Q, L_vis) or None
                - クロスアテンション用マスク

        出力:
            out: (N, Q, D_llm)
                - Q: grid_size^2 = 64
                - D_llm: 4096

        処理の流れ:
            1. 位置埋め込みの補間 (L_visに合わせる)
            2. KV射影 + LayerNorm
            3. クエリ + 位置埋め込み → LayerNorm
            4. Cross-Attention (Q, K, V)
            5. 出力LayerNorm + 射影
        ========================================
        """
        # --- 1. 位置埋め込みの補間 ---
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))
        # pos_embed: (L_vis, D_llm)
        #   L_visに合わせてbicubic補間

        # --- 2. KV射影 + LayerNorm ---
        x = self.kv_proj(x)
        # x: (N, L_vis, D_llm) ← (N, L_vis, D_vis) を射影

        x = self.ln_kv(x).permute(1, 0, 2)
        # x: (L_vis, N, D_llm) ← MultiheadAttentionの入力形式

        N = x.shape[1]  # バッチサイズ

        # --- 3. クエリの準備 ---
        q = self.ln_q(self.query)
        # q: (Q, D_llm)

        # --- 4. Cross-Attention ---
        # Q = query + pos_embed (空間位置情報)
        # K = x + pos_embed (視覚特徴 + 空間位置)
        # V = x (視覚特徴のみ)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),   # Q: (Q, N, D_llm)
            x + pos_embed.unsqueeze(1),                          # K: (L_vis, N, D_llm)
            x,                                                    # V: (L_vis, N, D_llm)
            attn_mask=attn_mask,
        )[0]
        # out: (Q, N, D_llm)

        x = out.permute(1, 0, 2)
        # x: (N, Q, D_llm)

        # --- 5. 出力正規化 + 射影 ---
        x = self.ln_post(x)
        x = x @ self.proj
        # x: (N, Q, D_llm)
        #   Q = 64, D_llm = 4096

        return x

    def _repeat(self, query: torch.Tensor, N: int) -> torch.Tensor:
        """クエリをバッチ次元でリピート: (Q, D) → (Q, N, D)"""
        return query.unsqueeze(1).repeat(1, N, 1)


# ========================================
# 3D-Resampler (動画用拡張)
# ========================================
class Unified3DResampler(nn.Module):
    """
    統一3D-Resampler（画像・動画両対応）

    論文 Section 2.1.1:
        「We extend the 2D-Resampler to a 3D-Resampler that jointly
         compresses spatial-temporal information for videos.」

    画像の場合は2D-Resamplerと同じ動作。
    動画の場合は隣接フレームをパッケージ化し、時間位置埋め込みを追加して
    空間・時間の冗長性を同時に圧縮する。

    ========================================
    画像モード:
        入力: (N_slices, L_vis, D_vis)
        出力: (N_slices, Q, D_llm)
        圧縮率: 最大16倍 (L_vis=1024 → Q=64)

    動画モード:
        入力: (T_pkg, F*L_vis, D_vis)  ※Fフレーム分を結合
        出力: (T_pkg, Q, D_llm)
        圧縮率: 最大96倍 (6フレーム × 1024 = 6144 → Q=64)
    ========================================
    """

    def __init__(
        self,
        grid_size: int = 8,
        embed_dim: int = 4096,
        num_heads: int = 32,
        kv_dim: int = 1792,
        max_temporal_len: int = 1080,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_queries = grid_size ** 2  # 64
        self.embed_dim = embed_dim

        # ========================================
        # 1. 2D-Resamplerのコンポーネント（画像・動画共通）
        # ========================================
        self.resampler_2d = Resampler2D(
            grid_size=grid_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
        )

        # ========================================
        # 2. 1D時間位置埋め込み（動画用追加コンポーネント）
        # ========================================
        # 論文: 「temporal positional embedding」
        # 各フレームの時間位置をエンコード
        self.temporal_embed = nn.Parameter(
            torch.from_numpy(
                get_1d_sincos_temporal_embed(embed_dim, max_temporal_len)
            ).float()
        ).requires_grad_(False)
        # temporal_embed: (max_temporal_len, D_llm) = (1080, 4096)
        # 固定パラメータ

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        画像モード: 2D-Resamplerと同じ動作

        ========================================
        入力:
            x: (N_slices, L_vis, D_vis)

        出力:
            out: (N_slices, Q, D_llm)
        ========================================
        """
        return self.resampler_2d(x)

    def forward_video(
        self,
        frame_features: List[torch.Tensor],
        pkg_size: int = 6,
    ) -> torch.Tensor:
        """
        動画モード: 3D-Resamplerによる空間・時間の同時圧縮

        論文 Section 2.1.1:
            「For each video, we first split it into packages along the temporal
             dimension, where each package contains adjacent frames. We resample
             the frame features from the visual encoder in each package into a
             fixed-length feature sequence through cross-attention.」

        ========================================
        入力:
            frame_features: List[Tensor]
                各フレームの視覚特徴: (L_vis, D_vis)
                len = 総フレーム数 T (例: 12)

            pkg_size: パッケージあたりのフレーム数 (例: 6)

        出力:
            video_tokens: (T_pkg, Q, D_llm)
                - T_pkg = ceil(T / pkg_size): パッケージ数
                - Q = 64
                - D_llm = 4096

        処理の流れ:
            1. フレームをパッケージに分割
            2. 各パッケージ内のフレーム特徴を結合
            3. 時間位置埋め込みを追加
            4. Cross-Attentionで圧縮

        圧縮例 (6秒2fps, 448x448):
            T=12フレーム, L_vis=1024/フレーム
            pkg_size=6 → 2パッケージ
            入力: 2 × (6*1024) = 2 × 6144 トークン
            出力: 2 × 64 = 128 トークン
            圧縮率: 12*1024 / 128 = 96倍
        ========================================
        """
        T = len(frame_features)
        L_vis = frame_features[0].shape[0]
        D_vis = frame_features[0].shape[1]

        # --- 1. フレームをパッケージに分割 ---
        num_packages = math.ceil(T / pkg_size)
        packages = []
        for pkg_idx in range(num_packages):
            start = pkg_idx * pkg_size
            end = min(start + pkg_size, T)
            pkg_frames = frame_features[start:end]
            # pkg_frames: F個の (L_vis, D_vis)

            # --- 2. パッケージ内のフレーム特徴を結合 ---
            F_actual = len(pkg_frames)
            pkg_features = torch.cat(pkg_frames, dim=0)
            # pkg_features: (F*L_vis, D_vis)

            # --- 3. 時間位置埋め込みを追加 ---
            # 各パッチトークンに、そのフレームの時間位置を加算
            temporal_pos = []
            for f_idx in range(F_actual):
                global_frame_idx = start + f_idx
                t_emb = self.temporal_embed[global_frame_idx]
                # t_emb: (D_llm,) = (4096,)
                # L_vis個のパッチトークンに同じ時間位置を付与
                temporal_pos.append(t_emb.unsqueeze(0).expand(L_vis, -1))
            temporal_pos = torch.cat(temporal_pos, dim=0)
            # temporal_pos: (F*L_vis, D_llm)

            packages.append((pkg_features, temporal_pos))

        # --- 4. 各パッケージをcross-attentionで圧縮 ---
        # 実際にはkv_projの後に temporal_pos を加算する形
        pkg_features_batch = torch.stack([p[0] for p in packages], dim=0)
        # pkg_features_batch: (T_pkg, F*L_vis, D_vis)

        # kv_proj: D_vis → D_llm
        pkg_projected = self.resampler_2d.kv_proj(pkg_features_batch)
        # pkg_projected: (T_pkg, F*L_vis, D_llm)

        # 時間位置埋め込みを追加
        temporal_pos_batch = torch.stack([p[1] for p in packages], dim=0)
        # temporal_pos_batch: (T_pkg, F*L_vis, D_llm)
        pkg_projected = pkg_projected + temporal_pos_batch
        # pkg_projected: (T_pkg, F*L_vis, D_llm)

        # LayerNorm + Cross-Attention
        pkg_projected = self.resampler_2d.ln_kv(pkg_projected).permute(1, 0, 2)
        # pkg_projected: (F*L_vis, T_pkg, D_llm)

        T_pkg = pkg_projected.shape[1]
        q = self.resampler_2d.ln_q(self.resampler_2d.query)
        # q: (Q, D_llm)

        # 空間位置埋め込み (L_visに対して補間)
        # 注: 3Dモードでは F*L_vis 全体に対する空間位置は
        #     各フレームのL_visに対して繰り返し適用
        spatial_pos = get_abs_pos(self.resampler_2d.pos_embed, L_vis)
        # spatial_pos: (L_vis, D_llm)

        # F個のフレーム分を繰り返し
        F_max = pkg_size
        spatial_pos_repeated = spatial_pos.unsqueeze(0).repeat(F_max, 1, 1).reshape(-1, self.embed_dim)
        # spatial_pos_repeated: (F*L_vis, D_llm)
        # ※パディングが必要な場合は末尾を切り詰め

        out = self.resampler_2d.attn(
            self.resampler_2d._repeat(q, T_pkg) + self.resampler_2d.pos_embed.unsqueeze(1),
            pkg_projected + spatial_pos_repeated[:pkg_projected.shape[0]].unsqueeze(1),
            pkg_projected,
        )[0]
        # out: (Q, T_pkg, D_llm)

        out = out.permute(1, 0, 2)
        # out: (T_pkg, Q, D_llm)

        out = self.resampler_2d.ln_post(out)
        out = out @ self.resampler_2d.proj
        # out: (T_pkg, Q, D_llm)

        return out

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "image",
        frame_features: Optional[List[torch.Tensor]] = None,
        pkg_size: int = 6,
    ) -> torch.Tensor:
        """
        統一フォワードパス

        ========================================
        入力:
            x: (N, L_vis, D_vis) - 画像モード用
            mode: "image" or "video"
            frame_features: List[Tensor] - 動画モード用
            pkg_size: パッケージサイズ (動画モード用)

        出力:
            画像: (N, Q, D_llm)
            動画: (T_pkg, Q, D_llm)
        ========================================
        """
        if mode == "image":
            return self.forward_image(x)
        elif mode == "video":
            assert frame_features is not None
            return self.forward_video(frame_features, pkg_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ========================================
# 使用例
# ========================================
def example_usage():
    """
    Resamplerの動作デモ

    画像モードと動画モードの両方を示す。
    """
    D_VIS = 1792
    D_LLM = 4096
    GRID_SIZE = 8
    Q = GRID_SIZE ** 2  # 64

    resampler = Unified3DResampler(
        grid_size=GRID_SIZE,
        embed_dim=D_LLM,
        num_heads=D_LLM // 128,
        kv_dim=D_VIS,
    )

    # ========================================
    # 画像モード
    # ========================================
    # 7スライス (ソース + 2x3グリッド), 各スライス 448x448
    N_slices = 7
    L_vis = (448 // 14) * (448 // 14)  # = 1024
    x_image = torch.randn(N_slices, L_vis, D_VIS)
    # x_image: (7, 1024, 1792)

    out_image = resampler(x_image, mode="image")
    # out_image: (7, 64, 4096)
    print(f"画像モード: {x_image.shape} → {out_image.shape}")
    print(f"  圧縮率: {N_slices * L_vis} → {N_slices * Q} = {N_slices * L_vis / (N_slices * Q):.1f}x")
    # → 7168 → 448 = 16.0x

    # ========================================
    # 動画モード
    # ========================================
    # 12フレーム (6秒2fps), 各フレーム 448x448, パッケージサイズ6
    T = 12
    pkg_size = 6
    frame_features = [torch.randn(L_vis, D_VIS) for _ in range(T)]
    # frame_features: 12個の (1024, 1792)

    out_video = resampler(
        x=None,
        mode="video",
        frame_features=frame_features,
        pkg_size=pkg_size,
    )
    # out_video: (2, 64, 4096)
    #   T_pkg = ceil(12/6) = 2 パッケージ
    T_pkg = math.ceil(T / pkg_size)
    total_in = T * L_vis
    total_out = T_pkg * Q
    print(f"動画モード: {T}フレーム × {L_vis} → ({T_pkg}, {Q})")
    print(f"  圧縮率: {total_in} → {total_out} = {total_in / total_out:.1f}x")
    # → 12288 → 128 = 96.0x


if __name__ == "__main__":
    example_usage()
