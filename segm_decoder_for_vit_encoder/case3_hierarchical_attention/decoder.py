"""
Case 3: Hierarchical Channel Attention (バランス型)

特徴:
- エンコーダー層を段階的にfuse
- 各段階でchannel attentionを適用
- 重要な特徴を選択的に保持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionGate(nn.Module):
    """Channel attention gate for feature fusion."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class FeatureFusionBlock(nn.Module):
    """Fusion block for combining two feature maps with attention."""

    def __init__(self, in_channels1, in_channels2, out_channels, use_attention=True):
        super().__init__()

        self.use_attention = use_attention
        total_channels = in_channels1 + in_channels2

        # Channel alignment (if needed)
        self.align_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention gate
        if use_attention:
            self.attention = ChannelAttentionGate(out_channels)

        # Refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: First feature map (B, C1, H, W)
            x2: Second feature map (B, C2, H, W)

        Returns:
            Fused feature map (B, out_channels, H, W)
        """
        # Concatenate
        x = torch.cat([x1, x2], dim=1)

        # Channel alignment
        x = self.align_conv(x)

        # Apply attention
        if self.use_attention:
            x = self.attention(x)

        # Refinement
        x = self.refine(x)

        return x


class ViTDecoderHierarchicalAttention(nn.Module):
    """
    ViT Decoder using Hierarchical Channel Attention.

    エンコーダー層を段階的に融合し、各段階でattentionを適用。
    速度と精度のバランスが良い実装。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        use_attention=True,
        attention_reduction=16,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            use_attention: Whether to use attention in fusion blocks
            attention_reduction: Reduction ratio for attention
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.num_stages = len(encoder_channels)

        # First projection (for the first encoder feature)
        self.first_proj = nn.Sequential(
            nn.Conv2d(encoder_channels[0], decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Hierarchical fusion blocks
        self.fusion_blocks = nn.ModuleList()
        for i in range(1, len(encoder_channels)):
            fusion_block = FeatureFusionBlock(
                in_channels1=decoder_channels,  # Previous fused features
                in_channels2=encoder_channels[i],  # Current encoder features
                out_channels=decoder_channels,
                use_attention=use_attention
            )
            self.fusion_blocks.append(fusion_block)

        # Final refinement
        self.final_refinement = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
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
        # Start with the first feature
        x = self.first_proj(features[0])

        # Hierarchically fuse remaining features
        for i, fusion_block in enumerate(self.fusion_blocks):
            feat = features[i + 1]
            x = fusion_block(x, feat)

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


# Usage example
if __name__ == "__main__":
    # ViT encoder channels (all same resolution)
    encoder_channels = [384, 384, 384]

    decoder = ViTDecoderHierarchicalAttention(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_attention=True,
        final_upsampling=16
    )

    # Simulate ViT encoder outputs
    B, H, W = 2, 32, 80
    dummy_features = [
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
    ]

    output = decoder(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (B={B}, C=256, H={H*16}, W={W*16})")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")
