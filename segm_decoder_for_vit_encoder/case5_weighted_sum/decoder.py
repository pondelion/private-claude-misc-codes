"""
Case 5: Weighted Sum with Learnable Coefficients (超速度重視)

特徴:
- 各層に1x1 convで次元を統一
- 学習可能な重みで加重平均
- オプションで軽量なrefinement conv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTDecoderWeightedSum(nn.Module):
    """
    ViT Decoder using Weighted Sum with Learnable Coefficients.

    各エンコーダー層を1x1 convで統一次元に射影し、学習可能な重みで加重平均。
    最も高速かつメモリ効率が良い実装。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        use_refinement=True,
        num_refinement_blocks=2,
        use_softmax_weights=True,  # True: softmaxで正規化, False: sigmoidで独立
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            use_refinement: Whether to use refinement convolutions
            num_refinement_blocks: Number of refinement conv blocks
            use_softmax_weights: If True, use softmax for weights (sum to 1).
                                If False, use sigmoid (independent weights).
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.use_softmax_weights = use_softmax_weights
        self.num_layers = len(encoder_channels)

        # 1x1 convolutions to project all features to the same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels
        ])

        # Learnable weights for each layer
        self.layer_weights = nn.Parameter(torch.ones(len(encoder_channels)))

        # Optional refinement blocks
        self.use_refinement = use_refinement
        if use_refinement:
            refinement_blocks = []
            for _ in range(num_refinement_blocks):
                refinement_blocks.extend([
                    nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True)
                ])
            self.refinement = nn.Sequential(*refinement_blocks)

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        # Project all features to the same channel dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Normalize weights
        if self.use_softmax_weights:
            weights = F.softmax(self.layer_weights, dim=0)
        else:
            weights = torch.sigmoid(self.layer_weights)

        # Weighted sum
        x = sum(w * feat for w, feat in zip(weights, projected))

        # Optional refinement
        if self.use_refinement:
            x = self.refinement(x)

        # Final upsampling
        if self.final_upsampling > 1:
            x = F.interpolate(
                x,
                scale_factor=self.final_upsampling,
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == 'bilinear' else None
            )

        return x

    def get_layer_weights(self):
        """Get the current layer weights (for visualization/debugging)."""
        if self.use_softmax_weights:
            return F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        else:
            return torch.sigmoid(self.layer_weights).detach().cpu().numpy()


class ViTDecoderWeightedSumV2(nn.Module):
    """
    ViT Decoder using Weighted Sum with Spatial-Aware Weights.

    各位置で異なる重みを学習する、より柔軟なバージョン。
    """

    def __init__(
        self,
        encoder_channels,
        decoder_channels=256,
        use_refinement=True,
        num_refinement_blocks=2,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            use_refinement: Whether to use refinement convolutions
            num_refinement_blocks: Number of refinement conv blocks
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.num_layers = len(encoder_channels)

        # 1x1 convolutions to project all features to the same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels
        ])

        # Weight prediction network (predicts spatial weights for each layer)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(decoder_channels * len(encoder_channels), 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, len(encoder_channels), kernel_size=1),
            nn.Softmax(dim=1)  # Softmax over layer dimension
        )

        # Optional refinement blocks
        self.use_refinement = use_refinement
        if use_refinement:
            refinement_blocks = []
            for _ in range(num_refinement_blocks):
                refinement_blocks.extend([
                    nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True)
                ])
            self.refinement = nn.Sequential(*refinement_blocks)

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        # Project all features to the same channel dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Stack projected features: (B, num_layers, C, H, W)
        stacked = torch.stack(projected, dim=1)

        # Predict spatial weights for each layer
        # Concatenate all projected features
        concat_feat = torch.cat(projected, dim=1)  # (B, num_layers*C, H, W)
        weights = self.weight_predictor(concat_feat)  # (B, num_layers, H, W)

        # Apply weighted sum with spatial weights
        # Reshape weights: (B, num_layers, 1, H, W)
        weights = weights.unsqueeze(2)
        # Weighted sum: (B, C, H, W)
        x = (stacked * weights).sum(dim=1)

        # Optional refinement
        if self.use_refinement:
            x = self.refinement(x)

        # Final upsampling
        if self.final_upsampling > 1:
            x = F.interpolate(
                x,
                scale_factor=self.final_upsampling,
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == 'bilinear' else None
            )

        return x


# Usage example
if __name__ == "__main__":
    # ViT encoder channels (all same resolution)
    encoder_channels = [384, 384, 384]

    print("=" * 50)
    print("Testing ViTDecoderWeightedSum (Simple Version)")
    print("=" * 50)

    decoder_v1 = ViTDecoderWeightedSum(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_refinement=True,
        num_refinement_blocks=2,
        use_softmax_weights=True,
        final_upsampling=16
    )

    # Simulate ViT encoder outputs
    B, H, W = 2, 32, 80
    dummy_features = [
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
    ]

    output = decoder_v1(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (B={B}, C=256, H={H*16}, W={W*16})")
    print(f"Layer weights: {decoder_v1.get_layer_weights()}")

    # Count parameters
    total_params = sum(p.numel() for p in decoder_v1.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 50)
    print("Testing ViTDecoderWeightedSumV2 (Spatial-Aware Version)")
    print("=" * 50)

    decoder_v2 = ViTDecoderWeightedSumV2(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_refinement=True,
        num_refinement_blocks=2,
        final_upsampling=16
    )

    output = decoder_v2(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in decoder_v2.parameters())
    print(f"Total parameters: {total_params:,}")
