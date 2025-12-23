"""
Case 2: Multi-Scale Feature Pyramid Fusion (精度重視)

特徴:
- 各層の特徴量を異なる受容野で処理
- Atrous Spatial Pyramid Pooling (ASPP)的なアプローチ
- 異なるdilation rateで並列処理後に融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    """Atrous convolution with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Sequential):
    """Global average pooling with 1x1 conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, out_channels, dilation_rates=(6, 12, 18)):
        super().__init__()

        modules = []
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous convolutions with different dilation rates
        for rate in dilation_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        total_out_channels = out_channels * (len(dilation_rates) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class MultiScaleFeatureProcessor(nn.Module):
    """Process a single feature map with multi-scale context."""

    def __init__(self, in_channels, out_channels, dilation_rates=(6, 12, 18)):
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels, dilation_rates)

    def forward(self, x):
        return self.aspp(x)


class ViTDecoderMultiScaleFPN(nn.Module):
    """
    ViT Decoder using Multi-Scale Feature Pyramid Fusion.

    各エンコーダー層をASPPで処理してマルチスケール情報を抽出。
    Channel/Spatial Attentionで重要な特徴を選択的に融合。
    精度重視の実装。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        dilation_rates=(6, 12, 18),
        use_channel_attention=True,
        use_spatial_attention=True,
        upsampling_mode='bilinear',
        final_upsampling=16,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels for decoder
            dilation_rates: Dilation rates for ASPP
            use_channel_attention: Whether to use channel attention
            use_spatial_attention: Whether to use spatial attention
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling

        # Multi-scale processors for each encoder layer
        self.processors = nn.ModuleList([
            MultiScaleFeatureProcessor(ch, decoder_channels, dilation_rates)
            for ch in encoder_channels
        ])

        # Total channels after processing all layers
        total_channels = decoder_channels * len(encoder_channels)

        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Attention modules
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        if use_channel_attention:
            self.channel_attention = ChannelAttention(decoder_channels)

        if use_spatial_attention:
            self.spatial_attention = SpatialAttention()

        # Refinement layers
        self.refinement = nn.Sequential(
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
        # Process each feature with multi-scale context
        processed_features = []
        for feat, processor in zip(features, self.processors):
            processed = processor(feat)
            processed_features.append(processed)

        # Concatenate all processed features
        x = torch.cat(processed_features, dim=1)

        # Fusion
        x = self.fusion_conv(x)

        # Apply attention
        if self.use_channel_attention:
            x = self.channel_attention(x)

        if self.use_spatial_attention:
            x = self.spatial_attention(x)

        # Refinement
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

    decoder = ViTDecoderMultiScaleFPN(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        dilation_rates=(6, 12, 18),
        use_channel_attention=True,
        use_spatial_attention=True,
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
