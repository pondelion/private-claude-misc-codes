"""
Case 1: Simple Concat + MLP Mixer (速度重視)

特徴:
- 全エンコーダー層をチャネル方向にconcat
- MLP Mixerライクな構造でチャネル・空間方向を交互に混合
- 最後に1x1 convでチャネル削減
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """MLP block with GELU activation and optional dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    """Mixer block: channel mixing followed by spatial (token) mixing."""

    def __init__(self, num_channels, num_tokens, channel_expansion=4, token_expansion=0.5, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(num_channels)
        self.token_mixing = MLPBlock(
            num_tokens,
            int(num_tokens * token_expansion),
            num_tokens,
            dropout
        )

        self.norm2 = nn.LayerNorm(num_channels)
        self.channel_mixing = MLPBlock(
            num_channels,
            int(num_channels * channel_expansion),
            num_channels,
            dropout
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Token mixing (spatial mixing)
        # Reshape to (B, C, H*W) for token mixing
        x_flat = x.reshape(B, C, H * W)
        x_norm = self.norm1(x_flat.transpose(1, 2))  # (B, H*W, C)
        x_mixed = self.token_mixing(x_norm.transpose(1, 2))  # (B, C, H*W)
        x_flat = x_flat + x_mixed

        # Channel mixing
        x_norm = self.norm2(x_flat.transpose(1, 2))  # (B, H*W, C)
        x_mixed = self.channel_mixing(x_norm)  # (B, H*W, C)
        x_flat = x_flat + x_mixed.transpose(1, 2)  # (B, C, H*W)

        # Reshape back
        x = x_flat.reshape(B, C, H, W)
        return x


class ViTDecoderMLPMixer(nn.Module):
    """
    ViT Decoder using MLP Mixer approach.

    全エンコーダー層の特徴をconcatして、MLP Mixerで処理。
    速度重視の軽量な実装。
    """

    def __init__(
        self,
        encoder_channels,  # e.g., [384, 384, 384] for ViT
        decoder_channels=256,
        num_mixer_blocks=4,
        channel_expansion=4,
        token_expansion=0.5,
        dropout=0.0,
        upsampling_mode='bilinear',
        final_upsampling=16,  # ViTの場合、通常16倍アップサンプリング
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
            decoder_channels: Output channels after channel reduction
            num_mixer_blocks: Number of MLP Mixer blocks
            channel_expansion: Expansion ratio for channel mixing MLP
            token_expansion: Expansion ratio for token mixing MLP
            dropout: Dropout rate
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor to match input resolution
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling

        # Total channels after concatenation
        total_channels = sum(encoder_channels)

        # Channel reduction (1x1 conv)
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(total_channels, decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # MLP Mixer blocks
        # Note: num_tokens will be determined at runtime based on spatial size
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(
                num_channels=decoder_channels,
                num_tokens=0,  # Will be set dynamically
                channel_expansion=channel_expansion,
                token_expansion=token_expansion,
                dropout=dropout
            )
            for _ in range(num_mixer_blocks)
        ])

        # Update mixer blocks to accept dynamic token size
        self.num_mixer_blocks = num_mixer_blocks
        self.channel_expansion = channel_expansion
        self.token_expansion = token_expansion
        self.dropout = dropout

    def forward(self, *features):
        """
        Args:
            features: Tuple of encoder outputs, each of shape (B, C, H, W)
                     For ViT, all features have the same spatial resolution

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)
        """
        # Concatenate all encoder features along channel dimension
        x = torch.cat(features, dim=1)  # (B, ΣC, H, W)

        # Channel reduction
        x = self.channel_reduction(x)  # (B, decoder_channels, H, W)

        B, C, H, W = x.shape
        num_tokens = H * W

        # Initialize mixer blocks with correct num_tokens on first forward
        if not hasattr(self, '_mixer_initialized') or self._num_tokens != num_tokens:
            self.mixer_blocks = nn.ModuleList([
                MixerBlock(
                    num_channels=C,
                    num_tokens=num_tokens,
                    channel_expansion=self.channel_expansion,
                    token_expansion=self.token_expansion,
                    dropout=self.dropout
                ).to(x.device)
                for _ in range(self.num_mixer_blocks)
            ])
            self._mixer_initialized = True
            self._num_tokens = num_tokens

        # Apply MLP Mixer blocks
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        # Final upsampling to match input resolution
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

    decoder = ViTDecoderMLPMixer(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        num_mixer_blocks=4,
        final_upsampling=16
    )

    # Simulate ViT encoder outputs (all same spatial resolution)
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
