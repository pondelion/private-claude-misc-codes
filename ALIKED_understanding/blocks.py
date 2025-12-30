"""
ALIKED Building Blocks - 簡略化疑似コード
=========================================

主要コンポーネント:
- ConvBlock: 基本畳み込みブロック
- ResBlock: 残差ブロック (Deformable Convオプション)
- SDDH: Sparse Deformable Descriptor Head (KEY INNOVATION)
- DeformableConv2d: 変形可能畳み込み
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """
    基本畳み込みブロック

    構成:
      - Conv3x3 → BN → SELU
      - Conv3x3 → BN → SELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SELU()
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """
    残差ブロック

    Block3/4で使用:
      - Deformable Conv使用可能
      - Skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dcn: bool = False
    ):
        super().__init__()

        if use_dcn:
            # Deformable Convolution使用
            self.conv1 = DeformableConv2d(in_channels, out_channels)
            self.conv2 = DeformableConv2d(out_channels, out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.selu(out)

        return out


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution (DCNv2風)

    通常の畳み込みと異なり:
      - 各ピクセルで学習可能なオフセットを推定
      - オフセットに基づいて柔軟にサンプリング
      - 幾何学的変換に対する不変性を獲得

    処理:
      1. オフセット推定: Conv3x3 → offsets (2 * K^2 channels)
      2. Deformable sampling: grid_sample
      3. 畳み込み適用
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # オフセット推定ネットワーク
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,  # (dx, dy) for each position
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # 通常の畳み込み
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

    def forward(self, x):
        """
        入力:
            x: (B, C_in, H, W)

        出力:
            out: (B, C_out, H', W')
        """

        # オフセット推定
        offsets = self.offset_conv(x)
        # offsets: (B, 2*K*K, H', W')

        # Deformable sampling
        x_offset = self._deform_sampling(x, offsets)
        # x_offset: (B, C_in, H', W')

        # 畳み込み適用
        out = self.regular_conv(x_offset)

        return out

    def _deform_sampling(self, x, offsets):
        """
        変形可能サンプリング

        グリッドサンプリングを使用して、
        オフセット位置から特徴をサンプリング
        """
        B, C, H, W = x.shape
        K = self.kernel_size

        # グリッド生成 (簡略化版)
        # 実装では torch.nn.functional.grid_sample を使用
        # ここでは概念的な処理のみ記載

        return x  # 簡略化のため元の特徴を返す


class SDDH(nn.Module):
    """
    Sparse Deformable Descriptor Head (SDDH)

    🔑 ALIKEDの最大のイノベーション:
    ========================================

    従来手法 (DMH: Descriptor Map Head):
      - 密な記述子マップを全体で計算
      - キーポイント位置からサンプリング
      - 計算量: O(H × W × C^2)
      - メモリ: 多大

    SDDH:
      - スパースなキーポイントのみで記述子抽出
      - 各キーポイントで変形可能なサンプリング位置を学習
      - 計算量: O(N × M × C) where N=キーポイント数, M=サンプル位置数
      - メモリ: 大幅削減 (50倍以上)

    処理フロー:
      1. キーポイント周辺のK×Kパッチを抽出
      2. パッチからMデformableサンプル位置を推定
      3. 特徴マップから変形可能サンプリング
      4. サンプリング特徴を集約して記述子生成
    """

    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 128,
        M: int = 16,           # デformableサンプル位置数
        K: int = 3             # パッチサイズ
    ):
        super().__init__()

        self.M = M
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        # ========================================
        # Offset推定ネットワーク
        # ========================================
        # K×Kパッチ → M個の2Dオフセット

        self.offset_net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=K, padding=0),  # No padding
            nn.SELU(),
            nn.Conv2d(in_dim, 2 * M, kernel_size=1)
        )


        # ========================================
        # 特徴エンコーダー
        # ========================================
        # サンプリングされた特徴をエンコード

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.SELU()
        )


        # ========================================
        # 記述子集約
        # ========================================
        # M個の特徴 → 1つの記述子

        # Learnable weights for aggregation
        self.agg_weights = nn.Parameter(torch.randn(M, out_dim, out_dim))

        # または Conv1x1ベースの集約
        self.agg_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)


    def forward(
        self,
        features: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparse Deformable Descriptor抽出

        入力:
            features: (B, in_dim, H, W) - 集約された特徴マップ
            keypoints: (B, N, 2) - キーポイント座標 [x, y]

        出力:
            descriptors: (B, N, out_dim) - 記述子
        """

        B, C, H, W = features.shape
        B, N, _ = keypoints.shape

        # ========================================
        # Step 1: K×Kパッチ抽出
        # ========================================

        # 各キーポイント周辺のK×Kパッチを抽出
        patches = self._extract_patches(features, keypoints, self.K)
        # patches: (B*N, in_dim, K, K)


        # ========================================
        # Step 2: Deformableサンプル位置推定
        # ========================================

        offsets = self.offset_net(patches)
        # offsets: (B*N, 2*M, 1, 1)

        offsets = offsets.view(B * N, self.M, 2)
        # offsets: (B*N, M, 2)

        # オフセットをクランプ (極端な変位を防止)
        max_offset = max(H, W) / 4
        offsets = torch.clamp(offsets, -max_offset, max_offset)


        # ========================================
        # Step 3: Deformable Sampling
        # ========================================

        # キーポイント座標 + オフセット
        keypoints_flat = keypoints.view(B * N, 2)  # (B*N, 2)
        sample_positions = keypoints_flat.unsqueeze(1) + offsets
        # sample_positions: (B*N, M, 2)

        # 特徴マップからサンプリング
        sampled_features = self._sample_features(
            features,
            sample_positions.view(B, N, self.M, 2)
        )
        # sampled_features: (B, N, M, in_dim)


        # ========================================
        # Step 4: 特徴エンコーディング
        # ========================================

        # (B, N, M, in_dim) → (B*N*M, in_dim, 1, 1) for conv
        sampled_features = sampled_features.reshape(B * N * self.M, self.in_dim, 1, 1)

        encoded = self.feature_encoder(sampled_features)
        # encoded: (B*N*M, out_dim, 1, 1)

        encoded = encoded.squeeze(-1).squeeze(-1)
        # encoded: (B*N*M, out_dim)

        encoded = encoded.view(B, N, self.M, self.out_dim)
        # encoded: (B, N, M, out_dim)


        # ========================================
        # Step 5: 記述子集約
        # ========================================

        # Method 1: Learnable weighted sum
        # descriptors = torch.einsum('bnmc,mcd->bnd', encoded, self.agg_weights)

        # Method 2: Simple average
        descriptors = encoded.mean(dim=2)
        # descriptors: (B, N, out_dim)

        # L2正規化
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        return descriptors


    def _extract_patches(
        self,
        features: torch.Tensor,
        keypoints: torch.Tensor,
        patch_size: int
    ) -> torch.Tensor:
        """
        キーポイント周辺のパッチ抽出

        入力:
            features: (B, C, H, W)
            keypoints: (B, N, 2)
            patch_size: int

        出力:
            patches: (B*N, C, patch_size, patch_size)
        """

        B, C, H, W = features.shape
        B, N, _ = keypoints.shape

        # Grid sample用に座標を正規化 [-1, 1]
        kpts_norm = keypoints.clone()
        kpts_norm[:, :, 0] = 2.0 * kpts_norm[:, :, 0] / (W - 1) - 1.0
        kpts_norm[:, :, 1] = 2.0 * kpts_norm[:, :, 1] / (H - 1) - 1.0

        # パッチグリッド生成
        half = patch_size // 2
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=features.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=features.device),
            indexing='ij'
        )

        # 正規化
        grid_x = 2.0 * grid_x / (W - 1)
        grid_y = 2.0 * grid_y / (H - 1)

        grid = torch.stack([grid_x, grid_y], dim=-1)
        # grid: (patch_size, patch_size, 2)

        # キーポイント位置に移動
        kpts_norm = kpts_norm.view(B, N, 1, 1, 2)
        grid = grid.view(1, 1, patch_size, patch_size, 2)

        sampling_grid = kpts_norm + grid
        # sampling_grid: (B, N, patch_size, patch_size, 2)

        # Grid sample
        features_expanded = features.unsqueeze(1).expand(B, N, C, H, W)
        features_flat = features_expanded.reshape(B * N, C, H, W)

        sampling_grid_flat = sampling_grid.reshape(B * N, patch_size, patch_size, 2)

        patches = F.grid_sample(
            features_flat,
            sampling_grid_flat,
            mode='bilinear',
            align_corners=False
        )
        # patches: (B*N, C, patch_size, patch_size)

        return patches


    def _sample_features(
        self,
        features: torch.Tensor,
        sample_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        任意位置から特徴サンプリング

        入力:
            features: (B, C, H, W)
            sample_positions: (B, N, M, 2) - [x, y]座標

        出力:
            sampled: (B, N, M, C)
        """

        B, C, H, W = features.shape
        B, N, M, _ = sample_positions.shape

        # 正規化 [-1, 1]
        pos_norm = sample_positions.clone()
        pos_norm[:, :, :, 0] = 2.0 * pos_norm[:, :, :, 0] / (W - 1) - 1.0
        pos_norm[:, :, :, 1] = 2.0 * pos_norm[:, :, :, 1] / (H - 1) - 1.0

        # Grid sample
        features_expanded = features.unsqueeze(1).expand(B, N, C, H, W)
        features_flat = features_expanded.reshape(B * N, C, H, W)

        pos_norm_flat = pos_norm.reshape(B * N, M, 1, 2)

        sampled = F.grid_sample(
            features_flat,
            pos_norm_flat,
            mode='bilinear',
            align_corners=False
        )
        # sampled: (B*N, C, M, 1)

        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        # sampled: (B*N, M, C)

        sampled = sampled.view(B, N, M, C)

        return sampled


# ============================================
# 使用例
# ============================================

def example_sddh():
    """SDDH使用例"""

    # SDDH作成
    sddh = SDDH(
        in_dim=128,
        out_dim=128,
        M=16,    # 16個のdeformableサンプル位置
        K=3      # 3×3パッチ
    )

    # ダミー入力
    features = torch.randn(2, 128, 160, 120)  # (B, dim, H, W)
    keypoints = torch.randint(0, 100, (2, 500, 2)).float()  # (B, N, 2)

    # 記述子抽出
    descriptors = sddh(features, keypoints)

    print(f"Features: {features.shape}")
    print(f"Keypoints: {keypoints.shape}")
    print(f"Descriptors: {descriptors.shape}")  # (2, 500, 128)

    # 効率性の確認
    print("\n=== Efficiency Comparison ===")
    print("Dense Descriptor Map (DMH):")
    print(f"  Operations: H × W × C^2 = 160 × 120 × 128^2 = 314M")
    print(f"  Memory: H × W × C = 160 × 120 × 128 = 2.5MB")

    print("\nSparse Deformable Descriptor (SDDH):")
    print(f"  Operations: N × M × C = 500 × 16 × 128 = 1.0M")
    print(f"  Memory: N × C = 500 × 128 = 64KB")
    print(f"  Speedup: ~300x faster!")


if __name__ == "__main__":
    example_sddh()
