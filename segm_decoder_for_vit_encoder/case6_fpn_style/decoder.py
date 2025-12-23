"""
Case 6: FPN-style Lateral Connection (ViT adapted)

特徴:
- 解像度は同じだが、FPNのlateral connection的な発想
- 浅い層→深い層への情報伝達路を作る
- 各層を段階的にrefine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralConnection(nn.Module):
    """Lateral connection for feature fusion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FeatureRefinement(nn.Module):
    """Feature refinement block with convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ViTDecoderFPNStyle(nn.Module):
    """
    ViT Decoder using FPN-style Lateral Connections.

    解像度は同じだが、FPNの思想を適用。
    深い層から浅い層へ段階的に情報を伝達し、各段階でrefinement。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        use_refinement=True,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            use_refinement: Whether to use refinement blocks
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.num_stages = len(encoder_channels)

        # Lateral connections for each encoder feature
        self.lateral_convs = nn.ModuleList([
            LateralConnection(ch, decoder_channels)
            for ch in encoder_channels
        ])

        # Top-down pathway (refinement for merged features)
        self.fpn_convs = nn.ModuleList([
            FeatureRefinement(decoder_channels)
            for _ in range(len(encoder_channels) - 1)
        ])

        # Final fusion of all pyramid features
        self.final_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Final refinement
        if use_refinement:
            self.final_refinement = FeatureRefinement(decoder_channels)
        else:
            self.final_refinement = nn.Identity()

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)
                     Ordered from shallow to deep

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        # Apply lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # Build top-down pathway (from deep to shallow)
        # Start from the deepest feature
        fpn_features = [laterals[-1]]

        for i in range(len(features) - 2, -1, -1):
            # Merge top-down feature with lateral feature
            top_down = fpn_features[0]
            lateral = laterals[i]
            merged = top_down + lateral

            # Refine merged feature
            refined = self.fpn_convs[i](merged)
            fpn_features.insert(0, refined)

        # Concatenate all pyramid features
        x = torch.cat(fpn_features, dim=1)

        # Final fusion
        x = self.final_fusion(x)

        # Final refinement
        x = self.final_refinement(x)

        # Final upsampling
        if self.final_upsampling > 1:
            x = F.interpolate(
                x,
                scale_factor=self.final_upsampling,
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == 'bilinear' else None
            )

        return x


class ViTDecoderFPNStyleV2(nn.Module):
    """
    ViT Decoder using FPN-style with Progressive Refinement.

    各FPN levelの出力を個別に保持し、最後にマルチスケールfusion。
    より柔軟なバージョン。
    """

    def __init__(
        self,
        encoder_channels,
        decoder_channels=256,
        use_attention=False,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            use_attention: Whether to use attention for fusion
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.num_stages = len(encoder_channels)
        self.use_attention = use_attention

        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            LateralConnection(ch, decoder_channels)
            for ch in encoder_channels
        ])

        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            FeatureRefinement(decoder_channels)
            for _ in range(len(encoder_channels) - 1)
        ])

        # Attention for fusion (optional)
        if use_attention:
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(decoder_channels * len(encoder_channels), len(encoder_channels), kernel_size=1),
                nn.Softmax(dim=1)
            )

        # Final projection
        if use_attention:
            self.final_proj = nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.final_proj = nn.Sequential(
                nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        # Apply lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # Build top-down pathway
        fpn_features = [laterals[-1]]

        for i in range(len(features) - 2, -1, -1):
            top_down = fpn_features[0]
            lateral = laterals[i]
            merged = top_down + lateral
            refined = self.fpn_convs[i](merged)
            fpn_features.insert(0, refined)

        # Fusion
        if self.use_attention:
            # Attention-based fusion
            stacked = torch.stack(fpn_features, dim=1)  # (B, num_stages, C, H, W)
            concat_feat = torch.cat(fpn_features, dim=1)  # (B, num_stages*C, H, W)

            # Compute attention weights
            attn_weights = self.fusion_attention(concat_feat)  # (B, num_stages, H, W)
            attn_weights = attn_weights.unsqueeze(2)  # (B, num_stages, 1, H, W)

            # Apply attention
            x = (stacked * attn_weights).sum(dim=1)  # (B, C, H, W)
            x = self.final_proj(x)
        else:
            # Concatenation-based fusion
            x = torch.cat(fpn_features, dim=1)
            x = self.final_proj(x)

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
    print("Testing ViTDecoderFPNStyle (Basic Version)")
    print("=" * 50)

    decoder_v1 = ViTDecoderFPNStyle(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_refinement=True,
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

    # Count parameters
    total_params = sum(p.numel() for p in decoder_v1.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 50)
    print("Testing ViTDecoderFPNStyleV2 (with Attention)")
    print("=" * 50)

    decoder_v2 = ViTDecoderFPNStyleV2(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_attention=True,
        final_upsampling=16
    )

    output = decoder_v2(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in decoder_v2.parameters())
    print(f"Total parameters: {total_params:,}")
