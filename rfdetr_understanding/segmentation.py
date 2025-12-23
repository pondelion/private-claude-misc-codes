"""
RF-DETR Segmentation Head - 簡略化疑似コード
==========================================

RF-DETR-Seg: 高速インスタンスセグメンテーション
"""

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """
    RF-DETR Segmentation Head

    RF-DETRの特徴:
    - 軽量なDepthwise Convolutionベース
    - クエリで条件付けされたマスク生成
    - 学習時はPoint Sampling (PointRend風) で効率化
    - 1/4解像度で高速推論
    """

    def __init__(
        self,
        in_dim: int = 256,           # 入力次元
        num_blocks: int = 4,         # 処理ブロック数
        bottleneck_ratio: int = 1,   # ボトルネック比率
        downsample_ratio: int = 4    # 出力ダウンサンプリング比率
    ):
        super().__init__()

        self.in_dim = in_dim
        self.downsample_ratio = downsample_ratio

        # ========================================
        # Spatial Feature Processing
        # ========================================
        # Depthwise Convolutionで空間特徴を処理
        self.spatial_blocks = nn.ModuleList([
            DepthwiseConvBlock(
                in_channels=in_dim if i == 0 else in_dim,
                out_channels=in_dim,
                bottleneck_ratio=bottleneck_ratio
            )
            for i in range(num_blocks)
        ])

        # ========================================
        # Query Feature Processing
        # ========================================
        # クエリ特徴をマスク埋め込みに変換
        self.query_proj = MLPBlock(
            in_dim=in_dim,
            hidden_dim=in_dim,
            out_dim=in_dim,
            num_layers=2
        )


    def forward(
        self,
        query_features: torch.Tensor,    # (B, num_queries, 256)
        spatial_features: torch.Tensor   # (B, 768, H/14, W/14)
    ) -> torch.Tensor:
        """
        セグメンテーションヘッド フォワードパス

        処理フロー:
        1. 空間特徴を段階的に処理 (Depthwise Conv)
        2. クエリ特徴をマスク埋め込みに変換
        3. Einstein Sumでマスク生成

        入力:
            query_features: (B, num_queries, 256) - Decoderクエリ
            spatial_features: (B, 768, H/14, W/14) - 早期レイヤーの高解像度特徴

        出力:
            pred_masks: (B, num_queries, H/4, W/4) - マスクロジット
        """

        B, num_queries, C_query = query_features.shape
        B, C_spatial, H_spatial, W_spatial = spatial_features.shape

        # ========================================
        # Step 1: 空間特徴の処理
        # ========================================

        # 次元を統一 (768 -> 256)
        if C_spatial != self.in_dim:
            spatial_proj = nn.Conv2d(C_spatial, self.in_dim, kernel_size=1).to(spatial_features.device)
            spatial_feat = spatial_proj(spatial_features)
        else:
            spatial_feat = spatial_features
        # spatial_feat: (B, 256, H/14, W/14)

        # Depthwise Conv Blocksで段階的に処理
        for block in self.spatial_blocks:
            spatial_feat = block(spatial_feat)
        # spatial_feat: (B, 256, H/14, W/14)

        # ターゲット解像度にアップサンプリング (H/4, W/4)
        target_h = H_spatial * 14 // self.downsample_ratio
        target_w = W_spatial * 14 // self.downsample_ratio

        spatial_feat = nn.functional.interpolate(
            spatial_feat,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        # spatial_feat: (B, 256, H/4, W/4) - ピクセル埋め込み


        # ========================================
        # Step 2: クエリ特徴の処理
        # ========================================

        # クエリをマスク埋め込みに変換
        mask_embed = self.query_proj(query_features)
        # mask_embed: (B, num_queries, 256)


        # ========================================
        # Step 3: Einstein Sumでマスク生成
        # ========================================

        # クエリ埋め込み × ピクセル埋め込み
        # (B, num_queries, 256) × (B, 256, H/4, W/4) -> (B, num_queries, H/4, W/4)
        pred_masks = torch.einsum(
            'bqc,bchw->bqhw',
            mask_embed,
            spatial_feat
        )
        # pred_masks: (B, num_queries, H/4, W/4)

        return pred_masks


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block

    効率的な空間特徴処理
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: int = 1,
        kernel_size: int = 3
    ):
        super().__init__()

        bottleneck_channels = in_channels // bottleneck_ratio

        self.conv = nn.Sequential(
            # Depthwise Conv
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            # Pointwise Conv (Bottleneck)
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),

            # Pointwise Conv (Expansion)
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = (in_channels == out_channels)


    def forward(self, x):
        out = self.conv(x)
        if self.shortcut:
            out = out + x
        return nn.functional.relu(out)


class MLPBlock(nn.Module):
    """シンプルなMLP"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_dim
            out_d = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ============================================
# Point Sampling (学習時の効率化)
# ============================================

def get_uncertain_point_coords_with_randomness(
    pred_masks: torch.Tensor,
    num_points: int = 12544
) -> torch.Tensor:
    """
    不確実性ベースのポイントサンプリング

    RF-DETRの学習効率化:
    - マスク全体ではなく不確実なポイントのみでロス計算
    - Sigmoid値が0.5に近い(不確実な)ポイントを優先的にサンプリング
    - 計算量とメモリを大幅削減

    入力:
        pred_masks: (B, num_queries, H, W) - マスクロジット

    出力:
        point_coords: (B, num_queries, num_points, 2) - サンプリング座標
    """

    B, N, H, W = pred_masks.shape

    # Sigmoid適用
    pred_probs = pred_masks.sigmoid()  # (B, N, H, W)

    # 不確実性計算: |0.5 - prob|が小さいほど不確実
    uncertainty = -torch.abs(pred_probs - 0.5)
    # uncertainty: (B, N, H, W)

    # 各マスクごとにTop-Kポイントを選択
    uncertainty_flat = uncertainty.reshape(B, N, -1)
    # (B, N, H*W)

    # Top-K選択
    _, indices = torch.topk(uncertainty_flat, k=num_points, dim=2)
    # indices: (B, N, num_points)

    # インデックスを(y, x)座標に変換
    y_coords = indices // W
    x_coords = indices % W

    # 正規化座標 [0, 1]
    y_coords_normalized = y_coords.float() / (H - 1)
    x_coords_normalized = x_coords.float() / (W - 1)

    # (B, N, num_points, 2)
    point_coords = torch.stack([x_coords_normalized, y_coords_normalized], dim=-1)

    return point_coords


def point_sample(
    input: torch.Tensor,
    point_coords: torch.Tensor
) -> torch.Tensor:
    """
    指定座標でマスクをサンプリング

    入力:
        input: (B, N, H, W) - マスク
        point_coords: (B, N, K, 2) - サンプリング座標 [0,1]

    出力:
        sampled: (B, N, K) - サンプリングされた値
    """

    B, N, H, W = input.shape
    K = point_coords.shape[2]

    # grid_sample用に座標を変換 [-1, 1]
    point_coords = point_coords * 2 - 1
    # (B, N, K, 2)

    # grid_sample
    # input: (B, N, H, W)
    # point_coords: (B, N, K, 2) -> (B*N, K, 1, 2)
    input_flat = input.reshape(B * N, 1, H, W)
    point_coords_flat = point_coords.reshape(B * N, K, 1, 2)

    sampled = nn.functional.grid_sample(
        input_flat,
        point_coords_flat,
        mode='bilinear',
        align_corners=False
    )
    # sampled: (B*N, 1, K, 1)

    sampled = sampled.reshape(B, N, K)
    # (B, N, K)

    return sampled


# ============================================
# 使用例
# ============================================

def example_segmentation():
    """セグメンテーションヘッドの使用例"""

    # セグメンテーションヘッド
    seg_head = SegmentationHead(
        in_dim=256,
        num_blocks=4,
        downsample_ratio=4
    )

    # ダミー入力
    query_features = torch.randn(2, 200, 256)          # (B, num_queries, 256)
    spatial_features = torch.randn(2, 768, 46, 46)    # (B, 768, H/14, W/14)

    # フォワードパス
    pred_masks = seg_head(query_features, spatial_features)

    print(f"Predicted masks shape: {pred_masks.shape}")
    # Expected: (2, 200, 108, 108) - H/4, W/4 for 432x432 input

    # Point Sampling (学習時)
    point_coords = get_uncertain_point_coords_with_randomness(
        pred_masks,
        num_points=12544
    )
    print(f"Sampled point coords shape: {point_coords.shape}")
    # Expected: (2, 200, 12544, 2)

    # サンプリング
    sampled_values = point_sample(pred_masks, point_coords)
    print(f"Sampled values shape: {sampled_values.shape}")
    # Expected: (2, 200, 12544)


if __name__ == "__main__":
    example_segmentation()
