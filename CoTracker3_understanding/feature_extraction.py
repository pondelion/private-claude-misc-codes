"""
CoTracker3 - 特徴抽出 (BasicEncoder + L2正規化 + ピラミッド構築)
論文: https://arxiv.org/abs/2410.11831

【このファイルの概要】
CoTracker3の特徴抽出パイプラインを詳細に解説。
CNN (BasicEncoder) → L2正規化 → 多スケールピラミッド → サポート特徴取得

【公式コードの対応箇所】
- cotracker/models/core/cotracker/blocks.py → ResidualBlock, BasicEncoder
- cotracker/models/core/cotracker/cotracker3_online.py → L2正規化, ピラミッド構築, get_track_feat, get_correlation_feat
- cotracker/models/core/model_utils.py → sample_features5d, bilinear_sampler

========================================
BasicEncoder のアーキテクチャ詳細
========================================

入力: (B*T, 3, H, W)  ※ H=384, W=512 が想定

conv1 (7×7, stride=2, pad=3)        → (B*T, 64, H/2, W/2)    = (*, 64, 192, 256)
InstanceNorm2d + ReLU
    ↓
layer1: ResBlock×2 (stride=1)        → a: (B*T, 64, H/2, W/2)  = (*, 64, 192, 256)
    ↓
layer2: ResBlock×2 (stride=2)        → b: (B*T, 96, H/4, W/4)  = (*, 96, 96, 128)
    ↓
layer3: ResBlock×2 (stride=2)        → c: (B*T, 128, H/8, W/8) = (*, 128, 48, 64)
    ↓
layer4: ResBlock×2 (stride=2)        → d: (B*T, 128, H/16,W/16)= (*, 128, 24, 32)

全スケールを bilinear補間で H/4 × W/4 に揃え:
concat([a, b, c, d]) → (B*T, 64+96+128+128, H/4, W/4) = (*, 416, 96, 128)
    ↓
conv2 (3×3, pad=1) + InstanceNorm + ReLU → (B*T, 256, H/4, W/4)
    ↓
conv3 (1×1) → (B*T, 128, H/4, W/4)

========================================
ResidualBlock の構造
========================================
y = ReLU(InstanceNorm(Conv3×3(x)))
y = ReLU(InstanceNorm(Conv3×3(y)))
if stride != 1:
    x = InstanceNorm(Conv1×1_stride(x))  # ショートカット
return ReLU(x + y)

【設計上のポイント】
1. InstanceNorm使用 → バッチサイズ非依存 (推論時の安定性)
2. 多スケール融合 → 低解像度(グローバルコンテキスト) + 高解像度(局所精度)
3. stride=4 → H/4 × W/4 が特徴マップの空間解像度
4. 出力128次元 → L2正規化後にドット積で相関計算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ========================================
# ResidualBlock
# ========================================
class ResidualBlock(nn.Module):
    """
    残差ブロック (InstanceNorm版)

    ========================================
    構造
    ========================================
    Conv3×3 (stride=stride) → InstanceNorm → ReLU
    Conv3×3 (stride=1)      → InstanceNorm → ReLU
    + ショートカット (strideが1でない場合はConv1×1でダウンサンプル)

    ========================================
    Shape
    ========================================
    入力: (B, in_planes, H, W)
    出力: (B, planes, H/stride, W/stride)
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # ショートカット: 空間サイズまたはチャネル数が変わる場合
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride),
                nn.InstanceNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B, in_planes, H, W)
        出力: (B, planes, H/stride, W/stride)
        """
        identity = x

        y = self.relu(self.norm1(self.conv1(x)))    # (B, planes, H/stride, W/stride)
        y = self.relu(self.norm2(self.conv2(y)))     # (B, planes, H/stride, W/stride)

        if self.downsample is not None:
            identity = self.downsample(x)            # (B, planes, H/stride, W/stride)

        return self.relu(identity + y)


# ========================================
# BasicEncoder (多スケールCNN特徴抽出器)
# ========================================
class BasicEncoder(nn.Module):
    """
    CoTracker3のCNN特徴抽出器

    ========================================
    パラメータ (デフォルト)
    ========================================
    input_dim  = 3   (RGB)
    output_dim = 128 (latent_dim)
    stride     = 4   (空間ダウンサンプリング比率)

    ========================================
    レイヤー構成とチャネル数
    ========================================
    in_planes の遷移:
    64 → 64 (layer1) → 96 (layer2) → 128 (layer3) → 128 (layer4)

    各layer = ResidualBlock × 2

    多スケール融合:
    64 + 96 + 128 + 128 = 416
    → conv2 (3×3) → 256ch
    → conv3 (1×1) → 128ch

    ========================================
    Kaiming初期化
    ========================================
    - Conv2d: kaiming_normal_ (fan_out, relu)
    - InstanceNorm2d: weight=1, bias=0
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 128, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.in_planes = output_dim // 2  # = 64

        # === 初期畳み込み ===
        # 7×7 conv, stride=2 → 空間を H/2 × W/2 に
        self.conv1 = nn.Conv2d(
            input_dim, self.in_planes,  # 3 → 64
            kernel_size=7, stride=2, padding=3
        )
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        # === 4段のResidual Block ===
        # 各段: ResidualBlock × 2 (self._make_layer)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)        # 64 → 64,  H/2
        self.layer2 = self._make_layer(output_dim * 3 // 4, stride=2)    # 64 → 96,  H/4
        self.layer3 = self._make_layer(output_dim, stride=2)             # 96 → 128, H/8
        self.layer4 = self._make_layer(output_dim, stride=2)             # 128→ 128, H/16

        # === 多スケール統合 ===
        # 全スケール結合: output_dim*3 + output_dim//4 = 128*3+32 = 416
        # 注: 実際は 64+96+128+128 = 416
        total_ch = output_dim * 3 + output_dim // 4  # = 416
        self.conv2 = nn.Conv2d(total_ch, output_dim * 2, 3, padding=1)  # 416 → 256
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, 1)           # 256 → 128

        # === 重み初期化 ===
        self._init_weights()

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """
        ResidualBlock × 2 を作成

        layer1 = ResBlock(in_planes → dim, stride=stride)
        layer2 = ResBlock(dim → dim, stride=1)

        ※ self.in_planes は各呼び出しで更新される
        """
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _init_weights(self):
        """Kaiming初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ========================================
        処理フロー (H=384, W=512 の場合)
        ========================================
        入力: x (B*T, 3, 384, 512)

        Step 1: 初期畳み込み
          conv1(7×7, s=2) → (B*T, 64, 192, 256)

        Step 2: 4段のResBlock
          a = layer1 → (B*T, 64,  192, 256)  stride=1 (サイズ変わらず)
          b = layer2 → (B*T, 96,  96,  128)  stride=2
          c = layer3 → (B*T, 128, 48,  64)   stride=2
          d = layer4 → (B*T, 128, 24,  32)   stride=2

        Step 3: 全スケール → H/stride × W/stride にリサイズ
          target = (96, 128) = (H/4, W/4)
          a → bilinear → (B*T, 64,  96, 128)  ※既にこのサイズ
          b → bilinear → (B*T, 96,  96, 128)  ※既にこのサイズ
          c → bilinear → (B*T, 128, 96, 128)  2倍アップサンプル
          d → bilinear → (B*T, 128, 96, 128)  4倍アップサンプル

        Step 4: 結合 + 射影
          cat → (B*T, 416, 96, 128)
          conv2 → (B*T, 256, 96, 128)
          conv3 → (B*T, 128, 96, 128)

        出力: (B*T, 128, H/4, W/4)
        """
        _, _, H, W = x.shape

        # Step 1: 初期畳み込み
        x = self.conv1(x)    # (B*T, 64, H/2, W/2)
        x = self.norm1(x)
        x = self.relu1(x)

        # Step 2: 4段のResBlock
        a = self.layer1(x)   # (B*T, 64,  H/2,  W/2)
        b = self.layer2(a)   # (B*T, 96,  H/4,  W/4)
        c = self.layer3(b)   # (B*T, 128, H/8,  W/8)
        d = self.layer4(c)   # (B*T, 128, H/16, W/16)

        # Step 3: 全スケールを H/stride × W/stride にリサイズ
        target_h, target_w = H // self.stride, W // self.stride

        def _bilinear_interpolate(feat):
            return F.interpolate(
                feat, (target_h, target_w),
                mode="bilinear", align_corners=True,
            )

        a = _bilinear_interpolate(a)  # (B*T, 64,  H/4, W/4)
        b = _bilinear_interpolate(b)  # (B*T, 96,  H/4, W/4)
        c = _bilinear_interpolate(c)  # (B*T, 128, H/4, W/4)
        d = _bilinear_interpolate(d)  # (B*T, 128, H/4, W/4)

        # Step 4: 結合 + 射影
        x = torch.cat([a, b, c, d], dim=1)  # (B*T, 416, H/4, W/4)
        x = self.conv2(x)                    # (B*T, 256, H/4, W/4)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)                    # (B*T, 128, H/4, W/4)
        return x


# ========================================
# L2正規化
# ========================================
def l2_normalize_features(fmaps: torch.Tensor) -> torch.Tensor:
    """
    特徴マップをL2正規化

    ========================================
    目的
    ========================================
    相関計算 (ドット積) でコサイン類似度となるように、
    特徴ベクトルを単位長に正規化する。

    ========================================
    Shape
    ========================================
    入力: fmaps (B*T, D, H', W')  ※ D=128
    出力: fmaps (B*T, D, H', W')  各ピクセル位置で ||f||=1

    ========================================
    実装
    ========================================
    1. (B*T, D, H, W) → (B*T, H, W, D)  permute
    2. norm = sqrt(max(sum(f^2, dim=-1), 1e-12))
    3. f = f / norm
    4. (B*T, H, W, D) → (B*T, D, H, W)  permute back

    ※ max(., 1e-12) でゼロ除算を防止
    """
    fmaps = fmaps.permute(0, 2, 3, 1)  # (B*T, H', W', D)
    fmaps = fmaps / torch.sqrt(
        torch.maximum(
            torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
            torch.tensor(1e-12, device=fmaps.device),
        )
    )
    fmaps = fmaps.permute(0, 3, 1, 2)  # (B*T, D, H', W')
    return fmaps


# ========================================
# 特徴マップピラミッド構築
# ========================================
def build_feature_pyramid(
    fmaps: torch.Tensor,
    num_levels: int = 4,
) -> List[torch.Tensor]:
    """
    Average Poolingで多スケール特徴ピラミッドを構築

    ========================================
    Shape
    ========================================
    入力: fmaps (B, T, D, H', W')  ※ D=128
    出力: List[Tensor] of length num_levels
      Level 0: (B, T, 128, H/4,  W/4)   = (B, T, 128, 96, 128)
      Level 1: (B, T, 128, H/8,  W/8)   = (B, T, 128, 48, 64)
      Level 2: (B, T, 128, H/16, W/16)  = (B, T, 128, 24, 32)
      Level 3: (B, T, 128, H/32, W/32)  = (B, T, 128, 12, 16)

    ========================================
    処理
    ========================================
    各レベルでavg_pool2d(kernel=2, stride=2)を適用。
    → 空間解像度が半分になる

    注意: fmapsは5次元 (B, T, D, H, W) なので、
    pooling前に (B*T, D, H, W) にreshapeが必要
    """
    B, T, D, H, W = fmaps.shape
    pyramid = [fmaps]  # Level 0: 元の解像度

    current_fmaps = fmaps
    for i in range(num_levels - 1):
        # (B, T, D, H_i, W_i) → (B*T, D, H_i, W_i) → pool → (B*T, D, H_i/2, W_i/2)
        fmaps_flat = current_fmaps.reshape(
            B * T, D, current_fmaps.shape[-2], current_fmaps.shape[-1]
        )
        fmaps_flat = F.avg_pool2d(fmaps_flat, 2, stride=2)
        current_fmaps = fmaps_flat.reshape(
            B, T, D, fmaps_flat.shape[-2], fmaps_flat.shape[-1]
        )
        pyramid.append(current_fmaps)

    return pyramid


# ========================================
# クエリ点のサポート特徴取得
# ========================================
def get_track_feat(
    fmaps: torch.Tensor,
    queried_frames: torch.Tensor,
    queried_coords: torch.Tensor,
    support_radius: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    クエリ点の中心特徴 + 周辺サポート特徴を取得

    ========================================
    概要
    ========================================
    各クエリ点について、指定フレームの指定座標周辺 (2r+1)×(2r+1)=7×7 の
    特徴をbilinear samplingで取得する。

    - 中心特徴: track_feat (相関計算のターゲット)
    - サポート特徴: track_feat_support (4D相関のグリッド)

    ========================================
    Shape
    ========================================
    入力:
      fmaps:          (B, T, D, H', W')   ※特定レベルの特徴マップ
      queried_frames: (B, N)               ※各クエリのフレーム番号
      queried_coords: (B, N, 2)            ※特徴マップスケールの座標
      support_radius: 3                    ※サポートグリッド半径

    出力:
      track_feat:         (B, 1, N, D)     ※中心点の特徴
      track_feat_support: (B, 49, N, D)    ※7×7サポートグリッドの特徴

    ========================================
    処理の詳細
    ========================================
    1. sample_coords を構築:
       - queried_frames → (B, 1, N, 1)  ※フレーム次元
       - queried_coords → (B, 1, N, 2)  ※x, y座標
       - 結合 → (B, 1, N, 3)  ※[frame, x, y]

    2. サポートポイント生成:
       - get_support_points() で中心座標周辺の 7×7 グリッドを生成
       - delta: [-3,-2,-1,0,1,2,3] × [-3,-2,-1,0,1,2,3] = 49点
       - 出力: (B, 49, N, 3)

    3. sample_features5d() で特徴をサンプリング:
       - fmaps の各 (frame, x, y) 位置でbilinear補間
       - 出力: (B, 49, N, D)

    4. 中心特徴 = support[49//2] = support[24] (中央のインデックス)

    ========================================
    実際の実装 (cotracker3_online.py)
    ========================================
    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):
        sample_frames = queried_frames[:, None, :, None]   # (B, 1, N, 1)
        sample_coords = cat([sample_frames, queried_coords[:, None]], dim=-1)  # (B, 1, N, 3)
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],  # 中心
            support_track_feats,  # 全49点
        )
    """
    B, T, D, H, W = fmaps.shape
    N = queried_coords.shape[1]
    r = support_radius  # = 3
    grid_size = 2 * r + 1  # = 7

    # === サポートグリッド生成 ===
    # 中心からの相対オフセット: [-3, -2, -1, 0, 1, 2, 3]
    dx = torch.linspace(-r, r, grid_size, device=fmaps.device)
    dy = torch.linspace(-r, r, grid_size, device=fmaps.device)
    xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
    # xgrid, ygrid: (7, 7)
    zgrid = torch.zeros_like(xgrid)  # フレーム方向のオフセットは0
    delta = torch.stack([zgrid, xgrid, ygrid], dim=-1)  # (7, 7, 3)

    # クエリ座標を中心にサポートグリッドを配置
    # sample_coords: (B, 1, N, 3) = [frame, y, x]
    sample_frames = queried_frames[:, None, :, None]  # (B, 1, N, 1)
    sample_coords = torch.cat([
        sample_frames.float(),
        queried_coords[:, None],  # (B, 1, N, 2)
    ], dim=-1)  # (B, 1, N, 3)

    centroid = sample_coords.reshape(B, N, 1, 1, 3)   # (B, N, 1, 1, 3)
    delta = delta.view(1, 1, grid_size, grid_size, 3)  # (1, 1, 7, 7, 3)
    support_points = centroid + delta                    # (B, N, 7, 7, 3)

    # → (B, 49, N, 3) にreshape
    support_points = support_points.reshape(B, N, grid_size * grid_size, 3)
    support_points = support_points.permute(0, 2, 1, 3)  # (B, 49, N, 3)

    # === Bilinear Sampling ===
    # sample_features5d(fmaps, support_points) で各サポート点の特徴を取得
    # 出力: (B, 49, N, D)
    # ここでは擬似実装
    track_feat_support = torch.randn(B, grid_size * grid_size, N, D, device=fmaps.device)

    # 中心特徴: 49//2 = 24番目 (0-indexed)
    center_idx = grid_size * grid_size // 2  # = 24
    track_feat = track_feat_support[:, None, center_idx]  # (B, 1, N, D)

    return track_feat, track_feat_support


# ========================================
# 推定座標周辺の近傍特徴取得
# ========================================
def get_correlation_feat(
    fmaps: torch.Tensor,
    queried_coords: torch.Tensor,
    corr_radius: int = 3,
    latent_dim: int = 128,
) -> torch.Tensor:
    """
    各フレームの現在の推定座標周辺から近傍特徴を取得

    ========================================
    概要
    ========================================
    反復リファインメントの各ステップで、
    現在の座標推定周辺の 7×7 近傍特徴をサンプリングする。
    これは4D相関ボリュームの「現在位置側」の特徴。

    ========================================
    Shape
    ========================================
    入力:
      fmaps:          (B, T, D, H', W')  ※特定レベルの特徴マップ
      queried_coords: (B*T, N, 2)        ※特徴マップスケールの推定座標
      corr_radius:    3                   ※近傍半径

    出力: (B, T, N, 7, 7, D)  ※各フレーム×各点の7×7近傍特徴

    ========================================
    処理
    ========================================
    1. 推定座標を中心に 7×7 サポートグリッドを生成
    2. bilinear_sampler で特徴をサンプリング
    3. 出力を (B, T, N, 7, 7, D) にreshape

    ========================================
    実際の実装 (cotracker3_online.py)
    ========================================
    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius

        # フレームオフセット0の3D座標に変換
        sample_coords = cat([zeros_like(queried_coords[..., :1]), queried_coords], dim=-1)
        support_points = self.get_support_points(sample_coords[:, None], r, reshape_back=False)

        # bilinear sampling (5Dテンソルとして処理)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B*T, D, 1, H_, W_),
            support_points
        )
        return correlation_feat.view(B, T, D, N, (2*r+1), (2*r+1)).permute(0,1,3,4,5,2)
        # 出力: (B, T, N, 7, 7, D)
    """
    B_T, N, _ = queried_coords.shape
    r = corr_radius
    grid_size = 2 * r + 1  # = 7
    B, T, D, H, W = fmaps.shape

    # サポートグリッドの生成 (get_correlation_feat と同じロジック)
    dx = torch.linspace(-r, r, grid_size, device=fmaps.device)
    dy = torch.linspace(-r, r, grid_size, device=fmaps.device)
    xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
    zgrid = torch.zeros_like(xgrid)
    delta = torch.stack([zgrid, xgrid, ygrid], dim=-1)  # (7, 7, 3)

    # 推定座標にフレームオフセット0を追加
    sample_coords = torch.cat([
        torch.zeros_like(queried_coords[..., :1]),
        queried_coords,
    ], dim=-1)  # (B*T, N, 3)

    centroid = sample_coords[:, None, :, None, None, :]  # (B*T, 1, N, 1, 1, 3)
    delta = delta.view(1, 1, 1, grid_size, grid_size, 3)
    support_points = centroid + delta  # (B*T, 1, N, 7, 7, 3)

    # bilinear sampling (実際の実装)
    # → (B, T, N, 7, 7, D)
    # ここでは擬似実装
    feat = torch.randn(B, T, N, grid_size, grid_size, D, device=fmaps.device)
    return feat


# ========================================
# 特徴抽出パイプライン全体のデモ
# ========================================
def demo_feature_extraction():
    """
    BasicEncoder → L2正規化 → ピラミッド → サポート特徴
    の全体フローをデモ

    ========================================
    処理の流れ
    ========================================
    1. BasicEncoder: (B*T, 3, H, W) → (B*T, 128, H/4, W/4)
    2. L2正規化: 各ピクセル位置で特徴ベクトルを単位長に
    3. ピラミッド: 4レベルの特徴マップ (avg_pool2d で段階的にダウンサンプル)
    4. サポート特徴: クエリ点周辺の 7×7 特徴グリッド
    5. 近傍特徴: 推定座標周辺の 7×7 特徴グリッド
    """
    print("=" * 60)
    print("CoTracker3 特徴抽出パイプライン デモ")
    print("=" * 60)

    # === パラメータ ===
    B, T, H, W = 1, 24, 384, 512
    N = 100
    D = 128
    stride = 4
    corr_radius = 3
    corr_levels = 4

    # === 1. BasicEncoder ===
    print("\n[Step 1] BasicEncoder")
    encoder = BasicEncoder(input_dim=3, output_dim=D, stride=stride)
    print(f"  パラメータ数: {sum(p.numel() for p in encoder.parameters()):,}")

    video = torch.randn(B * T, 3, H, W)
    fmaps = encoder(video)
    print(f"  入力:  {video.shape}")
    print(f"  出力:  {fmaps.shape}")
    assert fmaps.shape == (B * T, D, H // stride, W // stride)

    # === 2. L2正規化 ===
    print("\n[Step 2] L2正規化")
    fmaps_normed = l2_normalize_features(fmaps)
    print(f"  入力:  {fmaps.shape}")
    print(f"  出力:  {fmaps_normed.shape}")

    # 正規化確認: 各ピクセル位置のL2ノルム ≈ 1.0
    norms = torch.sqrt(torch.sum(fmaps_normed ** 2, dim=1))
    print(f"  L2ノルム (平均): {norms.mean().item():.4f}")
    print(f"  L2ノルム (最小): {norms.min().item():.4f}")
    print(f"  L2ノルム (最大): {norms.max().item():.4f}")

    # === 3. ピラミッド構築 ===
    print("\n[Step 3] 特徴ピラミッド構築")
    fmaps_5d = fmaps_normed.reshape(B, T, D, H // stride, W // stride)
    pyramid = build_feature_pyramid(fmaps_5d, num_levels=corr_levels)
    for i, level_fmaps in enumerate(pyramid):
        print(f"  Level {i}: {level_fmaps.shape}")

    # === 4. サポート特徴取得 ===
    print("\n[Step 4] クエリ点のサポート特徴取得")
    queried_frames = torch.zeros(B, N, dtype=torch.long)
    queried_coords = torch.randn(B, N, 2) * 10 + 50  # 適当な座標

    track_feat, track_feat_support = get_track_feat(
        pyramid[0], queried_frames, queried_coords, support_radius=corr_radius
    )
    print(f"  中心特徴:       {track_feat.shape}")       # (B, 1, N, D)
    print(f"  サポート特徴:   {track_feat_support.shape}") # (B, 49, N, D)

    # === 5. 近傍特徴取得 ===
    print("\n[Step 5] 推定座標の近傍特徴取得")
    coords = queried_coords[:, None].expand(B, T, N, 2).reshape(B * T, N, 2)
    corr_feat = get_correlation_feat(
        pyramid[0], coords, corr_radius=corr_radius, latent_dim=D
    )
    print(f"  入力座標:     (B*T, N, 2) = ({B*T}, {N}, 2)")
    print(f"  近傍特徴:     {corr_feat.shape}")  # (B, T, N, 7, 7, D)

    # === ピラミッドレベルごとの座標スケーリング ===
    print("\n[補足] ピラミッドレベルごとの座標スケーリング")
    for level in range(corr_levels):
        scale = 2 ** level
        print(f"  Level {level}: coords / {scale} = coords * {1/scale:.2f}")
        print(f"    特徴マップ: (B, T, {D}, {H//(stride*scale):.0f}, {W//(stride*scale):.0f})")

    # === 4D相関ボリュームの構造 ===
    print("\n[補足] 4D相関ボリュームの構造")
    r = corr_radius
    grid = 2 * r + 1
    print(f"  サポートグリッド: {grid}×{grid} = {grid**2}点")
    print(f"  近傍グリッド:     {grid}×{grid} = {grid**2}点")
    print(f"  相関ボリューム:   {grid**2}×{grid**2} = {grid**4}")
    print(f"    = einsum('btnhwc,bnijc->btnhwij')")
    print(f"    = (B, T, N, {grid}, {grid}, {grid}, {grid})")
    print(f"    → reshape → (B*T*N, {grid**4})")
    print(f"    → corr_mlp ({grid**4} → 384 → 256)")
    print(f"    × {corr_levels} levels → 256 × {corr_levels} = {256 * corr_levels}")


if __name__ == "__main__":
    demo_feature_extraction()
