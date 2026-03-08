"""
CoTracker3 - メインフロー (全体パイプライン)
論文: https://arxiv.org/abs/2410.11831

【核心的アイデア】
従来のPoint Tracker: CNN特徴 → グローバルマッチング → ローカルリファインメント
CoTracker3: CNN特徴 → 4D相関 → Transformer反復リファインメント (共同追跡)

【主要コンポーネント】
1. BasicEncoder: CNN多スケール特徴抽出 (stride=4, d=128)
2. 4D Correlation: クエリ点近傍 × 推定位置近傍 の密な相関ボリューム
3. EfficientUpdateFormer: 時間+空間アテンション + Virtual Tracks

【重要な設計選択】
- グローバルマッチングなし → 4D局所相関 + 反復更新で十分
- MLP相関処理 → LocoTrackのアドホックモジュールより単純かつ高速
- Virtual Tracks → O(N×64)でO(N²)のクロストラックアテンションを近似
- vis/conf統合出力 → 可視性と信頼度をTransformer出力ヘッドで直接予測

========================================
Shape Convention (形状規約)
========================================
B: バッチサイズ
T: フレーム数 (Online: window_len=16, Offline: 動画全体)
N: 追跡点数
H, W: 画像の高さ・幅
C: チャネル数 (RGB=3)
D: 特徴次元 (latent_dim=128)
S: ウィンドウ長 (Online=16, Offline=T)
M: 反復回数 (iters=4~6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# ========================================
# Positional Encoding (Fourier Encoding)
# ========================================
def posenc(x: torch.Tensor, min_deg: int = 0, max_deg: int = 10) -> torch.Tensor:
    """
    Fourier Positional Encoding

    入力xに対して sin(x * 2^i), cos(x * 2^i) (i = min_deg, ..., max_deg-1) を計算し、
    元のxと結合する。

    ========================================
    Shape
    ========================================
    入力: x (*, D)  - 任意の形状、最終次元がエンコード対象
    出力: (*, D + D * (max_deg - min_deg) * 2)

    例: x: (B, T, N, 4)  [rel_fw_x, rel_fw_y, rel_bw_x, rel_bw_y]
        → 出力: (B, T, N, 4 + 4*10*2) = (B, T, N, 84)

    ========================================
    Processing Details
    ========================================
    scales = [2^0, 2^1, ..., 2^9] = [1, 2, 4, ..., 512]
    xb = x * scales  → (*, D, 10)
    → sin(xb) と cos(xb) = sin(xb + π/2) を結合
    → 元のx と結合
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )
    # x: (*, D) → x[..., None, :] * scales[:, None] → (*, num_scales, D)
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    # sin と cos をまとめて計算 (cos(x) = sin(x + π/2))
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x, four_feat], dim=-1)


# ========================================
# Sinusoidal Time Embedding
# ========================================
def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """
    1次元のSinusoidal位置埋め込みを生成

    ========================================
    Shape
    ========================================
    出力: (1, length, embed_dim)

    用途: 各フレームの時間的位置をエンコード
    """
    position = torch.arange(length).unsqueeze(1).float()  # (length, 1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, length, embed_dim)


# ========================================
# MLP (Multi-Layer Perceptron)
# ========================================
class Mlp(nn.Module):
    """
    2層MLP (GELU活性化)

    用途:
    - 4D相関ボリュームの次元削減 (49*49 → 256)
    - Transformer内のFeed-Forward Network
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ========================================
# BasicEncoder (CNN特徴抽出)
# ========================================
class BasicEncoder(nn.Module):
    """
    多スケールCNN特徴抽出器

    ========================================
    Architecture
    ========================================
    conv1 (7×7, stride=2) → 64ch
        ↓
    layer1: ResBlock×2 → a: 64ch  (H/2)
        ↓
    layer2: ResBlock×2 → b: 96ch  (H/4)
        ↓
    layer3: ResBlock×2 → c: 128ch (H/8)
        ↓
    layer4: ResBlock×2 → d: 128ch (H/16)

    全スケールをbilinear補間でH/4×W/4に揃えて結合
    → conv2 (3×3) → conv3 (1×1) → 128ch

    ========================================
    Shape
    ========================================
    入力: (B*T, 3, H, W)
    出力: (B*T, 128, H/4, W/4)

    InstanceNormを使用 (バッチ間の統計量に依存しない)
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 128, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.in_planes = output_dim // 2  # 64

        # 初期畳み込み
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, 7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 4段のResidual Block (各段2ブロック)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)    # 64ch
        self.layer2 = self._make_layer(output_dim * 3 // 4, stride=2)  # 96ch
        self.layer3 = self._make_layer(output_dim, stride=2)         # 128ch
        self.layer4 = self._make_layer(output_dim, stride=2)         # 128ch

        # 多スケール統合
        # 64+96+128+128 = 416 → 256 → 128
        self.conv2 = nn.Conv2d(output_dim * 3 + output_dim // 4, output_dim * 2, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, 1)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """ResidualBlock × 2"""
        # 簡略版: 実際はResidualBlock (conv3×3 + conv3×3 + shortcut)
        layers = nn.Sequential(
            nn.Conv2d(self.in_planes, dim, 3, stride=stride, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.in_planes = dim
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B*T, 3, H, W)
        出力: (B*T, 128, H/4, W/4)
        """
        _, _, H, W = x.shape

        x = self.relu1(self.norm1(self.conv1(x)))  # (B*T, 64, H/2, W/2)

        a = self.layer1(x)   # (B*T, 64,  H/2,  W/2)
        b = self.layer2(a)   # (B*T, 96,  H/4,  W/4)
        c = self.layer3(b)   # (B*T, 128, H/8,  W/8)
        d = self.layer4(c)   # (B*T, 128, H/16, W/16)

        # 全スケールを H/stride × W/stride にリサイズ
        target_size = (H // self.stride, W // self.stride)
        a = F.interpolate(a, target_size, mode="bilinear", align_corners=True)
        b = F.interpolate(b, target_size, mode="bilinear", align_corners=True)
        c = F.interpolate(c, target_size, mode="bilinear", align_corners=True)
        d = F.interpolate(d, target_size, mode="bilinear", align_corners=True)

        # 結合 + 射影
        x = torch.cat([a, b, c, d], dim=1)  # (B*T, 416, H/4, W/4)
        x = self.relu2(self.norm2(self.conv2(x)))  # (B*T, 256, H/4, W/4)
        x = self.conv3(x)                          # (B*T, 128, H/4, W/4)
        return x


# ========================================
# Attention Module
# ========================================
class Attention(nn.Module):
    """
    Multi-Head Attention

    ========================================
    Shape
    ========================================
    入力 x: (B, N1, C)
    context (optional): (B, N2, C)  # Noneの場合はSelf-Attention
    出力: (B, N1, C)
    """

    def __init__(self, query_dim: int, num_heads: int = 8, dim_head: int = 48):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(query_dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)

        ctx = context if context is not None else x
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        N2 = ctx.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(out)


# ========================================
# AttnBlock (Pre-Norm Attention Block)
# ========================================
class AttnBlock(nn.Module):
    """
    Pre-LayerNorm Attention Block + MLP

    x = x + Attn(LN(x))
    x = x + MLP(LN(x))
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ========================================
# CrossAttnBlock (Cross Attention Block)
# ========================================
class CrossAttnBlock(nn.Module):
    """
    Cross-Attention: Q=x, KV=context

    Virtual Tracks と 実トラック間の情報交換に使用
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_ctx = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), context=self.norm_ctx(context))
        x = x + self.mlp(self.norm2(x))
        return x


# ========================================
# EfficientUpdateFormer
# ========================================
class EfficientUpdateFormer(nn.Module):
    """
    CoTracker3のTransformerモジュール

    ========================================
    構造
    ========================================
    input_transform: Linear(1110 → 384)

    time_depth=3 回繰り返し:
      - TimeAttnBlock: 各点の全フレーム間Self-Attention  (B*N, T, 384)
      - SpaceAttnBlock (Virtual Tracks経由):
        - SelfAttn(virtual)                           (B*T, 64, 384)
        - CrossAttn(virtual ← real)                   (B*T, 64, 384)
        - CrossAttn(real ← virtual)                   (B*T, N, 384)

    flow_head: Linear(384 → 4)  [delta_x, delta_y, delta_vis, delta_conf]

    ========================================
    Shape
    ========================================
    入力:  x: (B, N, T, 1110)
    出力:  delta: (B, N, T, 4)
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

        # 入力射影: 1110 → 384
        self.input_transform = nn.Linear(input_dim, hidden_size)

        # 時間方向Self-Attentionブロック
        self.time_blocks = nn.ModuleList(
            [AttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(time_depth)]
        )

        if add_space_attn:
            # Virtual Tracks: 学習可能パラメータ (1, 64, 1, 384)
            self.virtual_tracks = nn.Parameter(
                torch.randn(1, num_virtual_tracks, 1, hidden_size)
            )

            # 空間アテンションブロック (Virtual Tracks経由)
            self.space_virtual_blocks = nn.ModuleList(
                [AttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(space_depth)]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(space_depth)]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(space_depth)]
            )

        # 出力ヘッド: 384 → 4 [delta_x, delta_y, delta_vis, delta_conf]
        if linear_layer_for_vis_conf:
            # vis/confは別の線形層 (Pseudo-Label学習時にフリーズ可能)
            self.flow_head = nn.Linear(hidden_size, 2)       # delta_coords
            self.vis_conf_head = nn.Linear(hidden_size, 2)   # delta_vis, delta_conf
        else:
            self.flow_head = nn.Linear(hidden_size, output_dim)

        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf

    def forward(
        self, x: torch.Tensor, add_space_attn: bool = True
    ) -> torch.Tensor:
        """
        ========================================
        処理フロー
        ========================================
        入力: x: (B, N, T, 1110)
        1. 入力射影: (B, N, T, 1110) → (B, N, T, 384)
        2. 時間アテンション: (B*N, T, 384) — 各点の全フレーム間
        3. 空間アテンション (Virtual Tracks経由):
           - (B*T, N+64, 384) → 仮想トラックと実トラックの情報交換
        4. 出力ヘッド: (B, N, T, 384) → (B, N, T, 4)
        """
        B, N, T, _ = x.shape
        # === 1. 入力射影 ===
        x = self.input_transform(x)  # (B, N, T, 384)

        # === 2-3. Time + Space アテンション ===
        for i, time_block in enumerate(self.time_blocks):
            # --- 時間方向 Self-Attention ---
            # (B, N, T, 384) → (B*N, T, 384)
            x_time = x.reshape(B * N, T, self.hidden_size)
            x_time = time_block(x_time)
            x = x_time.reshape(B, N, T, self.hidden_size)

            if add_space_attn and self.add_space_attn:
                # --- 空間方向: Virtual Tracks経由 ---
                # (B, N, T, 384) → (B*T, N, 384)
                x_space = x.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_size)

                # Virtual Tracks: (1, 64, 1, 384) → (B*T, 64, 384)
                virtual = self.virtual_tracks.expand(B, -1, T, -1)
                virtual = virtual.permute(0, 2, 1, 3).reshape(B * T, self.num_virtual_tracks, self.hidden_size)

                # Step 1: 仮想トークン自己注意
                virtual = self.space_virtual_blocks[i](virtual)

                # Step 2: 実トラック → 仮想トークン (Cross-Attention)
                virtual = self.space_point2virtual_blocks[i](virtual, context=x_space)

                # Step 3: 仮想トークン → 実トラック (Cross-Attention)
                x_space = self.space_virtual2point_blocks[i](x_space, context=virtual)

                # (B*T, N, 384) → (B, N, T, 384)
                x = x_space.reshape(B, T, N, self.hidden_size).permute(0, 2, 1, 3)

        # === 4. 出力ヘッド ===
        if self.linear_layer_for_vis_conf:
            delta_coords = self.flow_head(x)     # (B, N, T, 2)
            delta_vis_conf = self.vis_conf_head(x)  # (B, N, T, 2)
            delta = torch.cat([delta_coords, delta_vis_conf], dim=-1)  # (B, N, T, 4)
        else:
            delta = self.flow_head(x)  # (B, N, T, 4)

        return delta


# ========================================
# CoTracker3 Offline Model
# ========================================
class CoTracker3Offline(nn.Module):
    """
    CoTracker3 Offline版

    全フレームを一括処理。双方向の追跡が可能。
    オクルージョン追跡に強い。

    ========================================
    Shape
    ========================================
    入力:
      video: (B, T, 3, H, W)  [0-255]
      queries: (B, N, 3)  [frame_idx, x, y]

    出力:
      tracks: (B, T, N, 2)  画像座標
      visibility: (B, T, N)  [0, 1]
      confidence: (B, T, N)  [0, 1]
    """

    def __init__(
        self,
        stride: int = 4,
        corr_radius: int = 3,
        corr_levels: int = 4,
        num_virtual_tracks: int = 64,
        model_resolution: Tuple[int, int] = (384, 512),
    ):
        super().__init__()
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.latent_dim = 128
        self.model_resolution = model_resolution

        # 1. CNN特徴抽出器
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)

        # 2. 相関MLP: 49×49=2401 → 256 (各レベル共有)
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        # 3. Transformer
        # 入力次元: vis(1) + conf(1) + corr(256×4) + rel_pos(84) = 1110
        self.updateformer = EfficientUpdateFormer(
            input_dim=1110,
            hidden_size=384,
            output_dim=4,
            time_depth=3,
            space_depth=3,
            num_virtual_tracks=num_virtual_tracks,
        )

        # 4. 時間埋め込み (Offlineではフレーム数に合わせて補間)
        # 60フレーム分を事前計算し、実行時に線形補間
        self.time_emb = get_1d_sincos_pos_embed(1110, 60)  # (1, 60, 1110)

    def forward(
        self,
        video: torch.Tensor,
        queries: torch.Tensor,
        iters: int = 4,
        is_train: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[tuple]]:
        """
        ========================================
        Offline推論フロー
        ========================================

        ステップ1: 前処理
        ステップ2: CNN特徴抽出 + L2正規化
        ステップ3: 相関ピラミッド構築 (4レベル)
        ステップ4: クエリ点のサポート特徴取得
        ステップ5: M回の反復リファインメント
        ステップ6: 後処理 (stride復元, sigmoid)
        """
        B, T, C, H, W = video.shape
        device = video.device
        N = queries.shape[1]

        # ========================================
        # ステップ1: 前処理
        # ========================================
        video = 2 * (video / 255.0) - 1.0           # [-1, 1]に正規化
        queried_frames = queries[:, :, 0].long()     # (B, N) クエリフレーム
        queried_coords = queries[:, :, 1:3]          # (B, N, 2) クエリ座標 (ピクセル)
        queried_coords = queried_coords / self.stride  # 特徴マップスケールに変換

        # ========================================
        # ステップ2: CNN特徴抽出 + L2正規化
        # ========================================
        fmaps = self.fnet(video.reshape(-1, C, H, W))  # (B*T, 128, H/4, W/4)

        # L2正規化: 特徴ベクトルを単位長に正規化
        fmaps = fmaps.permute(0, 2, 3, 1)  # → (B*T, H/4, W/4, 128)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(fmaps ** 2, dim=-1, keepdim=True),
                torch.tensor(1e-12, device=device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2)  # → (B*T, 128, H/4, W/4)
        fmaps = fmaps.reshape(B, T, self.latent_dim, H // self.stride, W // self.stride)

        # ========================================
        # ステップ3: 相関ピラミッド構築 (4レベル)
        # ========================================
        fmaps_pyramid = [fmaps]
        for i in range(self.corr_levels - 1):
            fmaps_down = fmaps_pyramid[-1].reshape(
                B * T, self.latent_dim, fmaps_pyramid[-1].shape[-2], fmaps_pyramid[-1].shape[-1]
            )
            fmaps_down = F.avg_pool2d(fmaps_down, 2, stride=2)
            fmaps_down = fmaps_down.reshape(
                B, T, self.latent_dim, fmaps_down.shape[-2], fmaps_down.shape[-1]
            )
            fmaps_pyramid.append(fmaps_down)
        # fmaps_pyramid[0]: (B, T, 128, H/4,  W/4)
        # fmaps_pyramid[1]: (B, T, 128, H/8,  W/8)
        # fmaps_pyramid[2]: (B, T, 128, H/16, W/16)
        # fmaps_pyramid[3]: (B, T, 128, H/32, W/32)

        # ========================================
        # ステップ4: クエリ点のサポート特徴取得
        # ========================================
        r = 2 * self.corr_radius + 1  # = 7
        track_feat_support_pyramid = []
        for level in range(self.corr_levels):
            # クエリフレームのクエリ座標周辺 7×7=49 のサポート点特徴を抽出
            # track_feat_support: (B, 49, N, 128)  ← bilinear sampling
            track_feat_support = self._get_track_support(
                fmaps_pyramid[level], queried_frames, queried_coords / (2 ** level)
            )
            # 全フレームで同じサポート特徴を使用
            track_feat_support_pyramid.append(track_feat_support)

        # ========================================
        # ステップ5: M回の反復リファインメント
        # ========================================
        # 初期化: クエリ座標を全フレームにコピー
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()
        vis = torch.zeros((B, T, N), device=device).float()
        conf = torch.zeros((B, T, N), device=device).float()

        coord_preds, vis_preds, conf_preds = [], [], []

        for it in range(iters):
            coords = coords.detach()  # 勾配を切断 (反復間の安定性)
            coords_flat = coords.reshape(B * T, N, 2)

            # --- 4D相関計算 (4レベル) ---
            corr_embs = []
            for level in range(self.corr_levels):
                # 現在の推定座標周辺の近傍特徴を取得
                corr_feat = self._get_correlation_feat(
                    fmaps_pyramid[level], coords_flat / (2 ** level)
                )
                # corr_feat: (B, T, N, 7, 7, 128)

                # 4D相関ボリューム: サポート特徴 × 近傍特徴
                track_support = track_feat_support_pyramid[level]
                # track_support: (B, N, 7, 7, 128) → reshape for einsum
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_support
                )
                # corr_volume: (B, T, N, 7, 7, 7, 7) = (B, T, N, 49, 49)

                # MLP で次元削減: 49×49=2401 → 256
                corr_emb = self.corr_mlp(
                    corr_volume.reshape(B * T * N, r * r * r * r)
                )
                # corr_emb: (B*T*N, 256)
                corr_embs.append(corr_emb)

            # 全レベル結合: 256×4 = 1024
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.reshape(B, T, N, 1024)

            # --- 相対変位 Fourier Encoding ---
            # 前方変位: coords[t] - coords[t+1]
            rel_forward = coords[:, :-1] - coords[:, 1:]
            rel_forward = F.pad(rel_forward, (0, 0, 0, 0, 0, 1))  # 最後のフレームを0パディング

            # 後方変位: coords[t] - coords[t-1]
            rel_backward = coords[:, 1:] - coords[:, :-1]
            rel_backward = F.pad(rel_backward, (0, 0, 0, 0, 1, 0))  # 最初のフレームを0パディング

            # モデル解像度でスケーリング
            scale = torch.tensor(
                [self.model_resolution[1], self.model_resolution[0]],
                device=device
            ) / self.stride  # [128, 96]
            rel_forward = rel_forward / scale
            rel_backward = rel_backward / scale

            # Fourier Encoding: (B, T, N, 4) → (B, T, N, 84)
            rel_pos_emb = posenc(
                torch.cat([rel_forward, rel_backward], dim=-1),
                min_deg=0, max_deg=10,
            )

            # --- Transformer入力組み立て ---
            transformer_input = torch.cat([
                vis[..., None],      # (B, T, N, 1)
                conf[..., None],     # (B, T, N, 1)
                corr_embs,           # (B, T, N, 1024)
                rel_pos_emb,         # (B, T, N, 84)
            ], dim=-1)
            # transformer_input: (B, T, N, 1110)

            # (B, T, N, 1110) → (B*N, T, 1110)
            x = transformer_input.permute(0, 2, 1, 3).reshape(B * N, T, -1)

            # 時間埋め込みを加算 (Tに合わせて線形補間)
            time_emb = self._interpolate_time_embed(T)  # (1, T, 1110)
            x = x + time_emb

            # (B*N, T, 1110) → (B, N, T, 1110)
            x = x.reshape(B, N, T, -1)

            # --- Transformer で更新量予測 ---
            delta = self.updateformer(x)  # (B, N, T, 4)

            # --- 座標・可視性・信頼度を更新 ---
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)   # (B, T, N, 2)
            delta_vis = delta[..., 2].permute(0, 2, 1)           # (B, T, N)
            delta_conf = delta[..., 3].permute(0, 2, 1)          # (B, T, N)

            coords = coords + delta_coords
            vis = vis + delta_vis
            conf = conf + delta_conf

            # 予測を保存 (stride倍して画像座標に復元)
            coord_preds.append(coords[..., :2] * float(self.stride))
            vis_preds.append(torch.sigmoid(vis))
            conf_preds.append(torch.sigmoid(conf))

        # ========================================
        # ステップ6: 後処理
        # ========================================
        tracks = coord_preds[-1]              # (B, T, N, 2) 画像座標
        visibility = vis_preds[-1]            # (B, T, N) [0, 1]
        confidence = conf_preds[-1]           # (B, T, N) [0, 1]

        if is_train:
            # 訓練時: 全反復の予測を返す (損失計算用)
            valid_mask = torch.ones_like(visibility, device=device)
            train_data = (
                [coord_preds],     # List[List[(B, T, N, 2)]]
                [vis_preds],       # List[List[(B, T, N)]]
                [conf_preds],      # List[List[(B, T, N)]]
                valid_mask,
            )
        else:
            train_data = None

        return tracks, visibility, confidence, train_data

    def _get_track_support(
        self, fmaps: torch.Tensor, queried_frames: torch.Tensor, queried_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        クエリフレームのクエリ座標周辺からサポート特徴を取得

        ========================================
        Shape
        ========================================
        fmaps: (B, T, 128, H', W')
        queried_frames: (B, N)
        queried_coords: (B, N, 2) 特徴マップスケール
        出力: (B, N, 7, 7, 128)  ← 7×7サポート点の128次元特徴
        """
        B, T, D, H, W = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius  # 3

        # bilinear samplingでクエリ座標周辺7×7の特徴を取得
        # (実際の実装では sample_features5d を使用)
        # ここでは擬似的に形状のみ示す
        support = torch.randn(B, N, 2 * r + 1, 2 * r + 1, D, device=fmaps.device)
        return support

    def _get_correlation_feat(
        self, fmaps: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        各フレームの推定座標周辺から近傍特徴を取得

        ========================================
        Shape
        ========================================
        fmaps: (B, T, 128, H', W')  → (B*T, 128, H', W')
        coords: (B*T, N, 2) 特徴マップスケール
        出力: (B, T, N, 7, 7, 128)
        """
        B_T, N, _ = coords.shape
        # bilinear samplingで7×7近傍の特徴を取得
        # (実際の実装では bilinear_sampler を使用)
        D = self.latent_dim
        r = self.corr_radius
        B = fmaps.shape[0]
        T = fmaps.shape[1]
        feat = torch.randn(B, T, N, 2 * r + 1, 2 * r + 1, D, device=fmaps.device)
        return feat

    def _interpolate_time_embed(self, T: int) -> torch.Tensor:
        """
        時間埋め込みをTフレームに補間

        事前計算: (1, 60, 1110)
        → 線形補間: (1, T, 1110)
        """
        if T == self.time_emb.shape[1]:
            return self.time_emb
        time_emb = F.interpolate(
            self.time_emb.permute(0, 2, 1), size=T, mode="linear"
        ).permute(0, 2, 1)
        return time_emb


# ========================================
# CoTracker3 Online Model
# ========================================
class CoTracker3Online(CoTracker3Offline):
    """
    CoTracker3 Online版

    スライディングウィンドウ方式で動画をチャンクごとに処理。
    前方のみの追跡。リアルタイム/ストリーミングに適する。

    ========================================
    主な違い (Offline との比較)
    ========================================
    - window_len=16, step=8 (50%オーバーラップ)
    - 前方のみの追跡 (後方フレームは参照不可)
    - 特徴キャッシュ: ウィンドウ間でクエリ点特徴を保持
    - 予測の引き継ぎ: 前ウィンドウの後半予測を初期値に使用
    """

    def __init__(self, window_len: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.window_len = window_len

    def init_video_online_processing(self):
        """オンライン処理の初期化 (新しい動画の処理前に呼ぶ)"""
        self.online_ind = 0                                      # 現在のウィンドウ開始位置
        self.online_track_feat = [None] * self.corr_levels       # クエリ特徴キャッシュ
        self.online_track_support = [None] * self.corr_levels    # サポート特徴キャッシュ
        self.online_coords_predicted = None                      # 累積座標予測
        self.online_vis_predicted = None                         # 累積可視性予測
        self.online_conf_predicted = None                        # 累積信頼度予測

    def forward_online(
        self,
        video_chunk: torch.Tensor,
        queries: torch.Tensor,
        iters: int = 4,
    ):
        """
        ========================================
        Online推論フロー (1チャンク分)
        ========================================

        1. 動画チャンクをウィンドウ長にパディング
        2. CNN特徴抽出
        3. 新規クエリ点の特徴をキャッシュに追加
        4. 前ウィンドウの予測を初期値として引き継ぎ
        5. 反復リファインメント (Offlineと同じ forward_window)
        6. 予測をキャッシュに保存
        """
        B, T_chunk, C, H, W = video_chunk.shape
        S = self.window_len  # = 16
        step = S // 2        # = 8
        device = video_chunk.device
        N = queries.shape[1]

        # --- パディング ---
        pad = S - T_chunk
        if pad > 0:
            video_chunk = F.pad(video_chunk, (0, 0, 0, 0, 0, 0, 0, pad))

        # --- CNN特徴抽出 ---
        video_normed = 2 * (video_chunk / 255.0) - 1.0
        fmaps = self.fnet(video_normed.reshape(-1, C, H, W))
        # → L2正規化 → pyramid構築 (Offlineと同じ)

        # --- クエリ特徴のキャッシュ更新 ---
        queried_frames = queries[:, :, 0].long()
        left = 0 if self.online_ind == 0 else self.online_ind + step
        right = self.online_ind + S
        # このウィンドウに含まれるクエリフレームの特徴のみ追加
        sample_mask = (queried_frames >= left) & (queried_frames < right)
        # online_track_support[level] += new_support * sample_mask

        # --- 前ウィンドウの予測を引き継ぎ ---
        if self.online_ind > 0:
            overlap = S - step  # = 8
            # 前ウィンドウの後半 (frames [ind, ind+overlap)) の予測を初期値に
            copy_mask = queried_frames < (self.online_ind + overlap)
            # coords_init = where(copy_mask, prev_prediction, query_coords)

        # --- 反復リファインメント ---
        # forward_window(...) — Offline版と同じTransformerパイプライン
        # ただしウィンドウ長Sのフレームのみ処理

        # --- キャッシュ更新 ---
        self.online_ind += step
        # self.online_coords_predicted[:, :self.online_ind+S] = new_prediction

        # 返り値: (tracks, visibility, confidence) の累積結果
        print(f"[Online] Processed window {self.online_ind-step}~{self.online_ind+step}")


# ========================================
# メインフロー: 推論デモ
# ========================================
def demo_inference():
    """
    CoTracker3の推論デモ (Offline版)

    ========================================
    処理の流れ
    ========================================
    1. モデル構築
    2. ダミー入力生成
    3. フォワードパス
    4. 結果表示
    """
    print("=" * 60)
    print("CoTracker3 Offline 推論デモ")
    print("=" * 60)

    # --- モデル構築 ---
    model = CoTracker3Offline(
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
    )
    print(f"\nパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # --- ダミー入力 ---
    B, T, H, W = 1, 24, 384, 512
    N = 100  # 追跡点数
    video = torch.randint(0, 256, (B, T, 3, H, W), dtype=torch.float32)
    queries = torch.zeros(B, N, 3)
    queries[:, :, 0] = 0                                    # 全点フレーム0で開始
    queries[:, :, 1] = torch.randint(0, W, (B, N)).float()  # x座標
    queries[:, :, 2] = torch.randint(0, H, (B, N)).float()  # y座標

    print(f"\n入力:")
    print(f"  video:   {video.shape}  (B, T, C, H, W) [0-255]")
    print(f"  queries: {queries.shape}  (B, N, 3) [frame, x, y]")

    # --- フォワードパス ---
    with torch.no_grad():
        tracks, visibility, confidence, _ = model(video, queries, iters=4)

    print(f"\n出力:")
    print(f"  tracks:     {tracks.shape}  (B, T, N, 2) 画像座標")
    print(f"  visibility: {visibility.shape}  (B, T, N) [0, 1]")
    print(f"  confidence: {confidence.shape}  (B, T, N) [0, 1]")

    # --- 中間形状の確認 ---
    print(f"\n中間形状 (stride=4):")
    print(f"  特徴マップ: (B, T, 128, {H // 4}, {W // 4})")
    print(f"  ピラミッド Level 0: (B, T, 128, {H // 4}, {W // 4})")
    print(f"  ピラミッド Level 1: (B, T, 128, {H // 8}, {W // 8})")
    print(f"  ピラミッド Level 2: (B, T, 128, {H // 16}, {W // 16})")
    print(f"  ピラミッド Level 3: (B, T, 128, {H // 32}, {W // 32})")
    print(f"  相関 (per level): (B, T, N, 7, 7, 7, 7) = 49×49ボリューム")
    print(f"  相関MLP後 (per level): (B*T*N, 256)")
    print(f"  相関結合 (4levels): (B, T, N, 1024)")
    print(f"  変位Fourier: (B, T, N, 84)")
    print(f"  Transformer入力: (B, N, T, 1110)")
    print(f"  Transformer内部: (B, N, T, 384)")
    print(f"  Virtual Tracks: (1, 64, 1, 384)")
    print(f"  Transformer出力: (B, N, T, 4) [Δx, Δy, Δvis, Δconf]")


if __name__ == "__main__":
    demo_inference()
