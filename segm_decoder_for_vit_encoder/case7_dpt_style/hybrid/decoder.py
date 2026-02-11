"""
DPT-Hybrid Decoder

異なる解像度の特徴（ResNet中間層 + Transformer層）を受け取り、
段階的に融合・アップサンプリングするデコーダ。

=== Shape Flow (例: 入力512x1280) ===

入力特徴（異なる解像度）:
    - R0:      (B, 256, 128, 320)   # 1/4 res from ResNet Block1
    - R1:      (B, 512, 64, 160)    # 1/8 res from ResNet Block2
    - layer9:  (B, 768, 32, 80)     # 1/16 res from Transformer
    - layer12: (B, 768, 32, 80)     # 1/16 res from Transformer
    ↓ Reassemble (Project to 256ch, NO spatial resampling)
    - layer_rn[0]: (B, 256, 128, 320)  # 1/4 res
    - layer_rn[1]: (B, 256, 64, 160)   # 1/8 res
    - layer_rn[2]: (B, 256, 32, 80)    # 1/16 res
    - layer_rn[3]: (B, 256, 32, 80)    # 1/16 res → downsample to 1/32
                   → (B, 256, 16, 40)  # 1/32 res
    ↓ FeatureFusionBlocks (Deep → Shallow, 2x upsample each)
    - path = fusion[3](layer_rn[3])           → (B, 256, 32, 80)   # 1/16 res
    - path = fusion[2](path, layer_rn[2])     → (B, 256, 64, 160)  # 1/8 res
    - path = fusion[1](path, layer_rn[1])     → (B, 256, 128, 320) # 1/4 res
    - path = fusion[0](path, layer_rn[0])     → (B, 256, 256, 640) # 1/2 res
    ↓ (デコーダ出力)
出力: (B, 256, 256, 640)  ※1/2解像度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Import from parent module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from decoder import ResidualConvUnit, FeatureFusionBlock


class ReassembleHybrid(nn.Module):
    """
    Reassemble module for hybrid features.

    Unlike pure ViT version, this handles features at DIFFERENT resolutions.
    Only projects channels, spatial resampling is handled separately.

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)  ※空間サイズは変化しない
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Args:
            in_channels: Input channel size
            out_channels: Output channel size (typically 256)
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C_in, H, W)

        Returns:
            Projected tensor (B, C_out, H, W)
        """
        return self.proj(x)


class ViTDecoderDPTHybrid(nn.Module):
    """
    DPT-Hybrid Decoder for mixed ResNet + Transformer features.

    Handles features at different resolutions:
    - R0: 1/4 resolution (from ResNet Block1)
    - R1: 1/8 resolution (from ResNet Block2)
    - Transformer layers: 1/16 resolution

    Architecture:
        Multi-resolution features
            ↓ Reassemble (channel projection only)
        [1/4, 1/8, 1/16, 1/32] feature pyramid
            ↓ Progressive fusion (deep → shallow)
        Final output at 1/2 resolution

    Shape (例: 入力512x1280):
        Input:  [
            (B, 256, 128, 320),   # R0, 1/4 res
            (B, 512, 64, 160),    # R1, 1/8 res
            (B, 768, 32, 80),     # Transformer layer, 1/16 res
            (B, 768, 32, 80),     # Transformer layer, 1/16 res
        ]
        Output: (B, 256, 256, 640)  ※1/2解像度
    """

    def __init__(
        self,
        encoder_channels: List[int],
        encoder_strides: List[int],
        decoder_channels: int = 256,
        use_bn: bool = True,
        align_corners: bool = True,
        upsampling_mode: str = 'bilinear',
    ):
        """
        Args:
            encoder_channels: List of encoder output channels
                              e.g., [256, 512, 768, 768] for DPT-Hybrid
            encoder_strides: List of encoder output strides (downsampling factors)
                             e.g., [4, 8, 16, 16] for DPT-Hybrid
            decoder_channels: Decoder feature dimension (default: 256)
            use_bn: Whether to use batch normalization
            align_corners: align_corners for bilinear interpolation
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.align_corners = align_corners
        self.num_stages = len(encoder_channels)
        self.encoder_strides = encoder_strides

        # Validate input
        assert len(encoder_channels) == len(encoder_strides), \
            "encoder_channels and encoder_strides must have same length"
        assert len(encoder_channels) == 4, \
            "DPT-Hybrid expects exactly 4 feature levels"

        # Reassemble layers (channel projection only)
        self.reassemble_layers = nn.ModuleList()
        for enc_ch in encoder_channels:
            self.reassemble_layers.append(
                ReassembleHybrid(enc_ch, decoder_channels)
            )

        # Downsample for deepest feature (1/16 → 1/32)
        self.downsample_deep = nn.Conv2d(
            decoder_channels, decoder_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        # Feature Fusion Blocks
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    decoder_channels,
                    use_bn=use_bn,
                    upsample=True,  # Always 2x upsample
                    align_corners=align_corners
                )
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder outputs with different resolutions
                      [R0 (1/4), R1 (1/8), transformer1 (1/16), transformer2 (1/16)]

        Returns:
            Decoded feature map (B, decoder_channels, H/2, W/2)

        Shape Flow (例: 入力512x1280):
            Input features:
                feat[0]: (B, 256, 128, 320)  # R0, 1/4 res
                feat[1]: (B, 512, 64, 160)   # R1, 1/8 res
                feat[2]: (B, 768, 32, 80)    # Transformer, 1/16 res
                feat[3]: (B, 768, 32, 80)    # Transformer, 1/16 res

            After Reassemble (channel proj only):
                layer_rn[0]: (B, 256, 128, 320)  # 1/4 res
                layer_rn[1]: (B, 256, 64, 160)   # 1/8 res
                layer_rn[2]: (B, 256, 32, 80)    # 1/16 res
                layer_rn[3]: (B, 256, 16, 40)    # 1/32 res (downsampled)

            After Fusion (deep → shallow, 2x upsample each):
                fusion[3](layer_rn[3])           → (B, 256, 32, 80)
                fusion[2](path, layer_rn[2])     → (B, 256, 64, 160)
                fusion[1](path, layer_rn[1])     → (B, 256, 128, 320)
                fusion[0](path, layer_rn[0])     → (B, 256, 256, 640)

            Output: (B, 256, 256, 640)  # 1/2 res
        """
        assert len(features) == self.num_stages, \
            f"Expected {self.num_stages} features, got {len(features)}"

        # === Reassemble: project channels ===
        layer_rn = []
        for feat, reassemble in zip(features, self.reassemble_layers):
            layer_rn.append(reassemble(feat))

        # === Downsample deepest feature to 1/32 ===
        # This creates the 1/32 resolution level for the pyramid
        # Shape: (B, 256, 32, 80) → (B, 256, 16, 40)
        layer_rn[3] = self.downsample_deep(layer_rn[3])

        # === Progressive fusion from deep to shallow ===
        # Start with deepest (1/32 res)
        # fusion[3]: (B, 256, 16, 40) → (B, 256, 32, 80) after 2x up
        path = self.fusion_blocks[3](layer_rn[3])

        # fusion[2]: fuse with 1/16, then 2x up → 1/8 res
        # path (32, 80) + layer_rn[2] (32, 80) → (B, 256, 64, 160)
        path = self.fusion_blocks[2](path, layer_rn[2])

        # fusion[1]: fuse with 1/8, then 2x up → 1/4 res
        # path (64, 160) + layer_rn[1] (64, 160) → (B, 256, 128, 320)
        path = self.fusion_blocks[1](path, layer_rn[1])

        # fusion[0]: fuse with 1/4, then 2x up → 1/2 res
        # path (128, 320) + layer_rn[0] (128, 320) → (B, 256, 256, 640)
        path = self.fusion_blocks[0](path, layer_rn[0])

        return path


class DPTHybridModel(nn.Module):
    """
    Complete DPT-Hybrid model combining encoder and decoder.

    便利なラッパークラス。
    """

    def __init__(
        self,
        encoder_channels: List[int] = [256, 512, 768, 768],
        encoder_strides: List[int] = [4, 8, 16, 16],
        decoder_channels: int = 256,
        use_bn: bool = True,
    ):
        """
        Args:
            encoder_channels: Channel sizes from encoder
            encoder_strides: Stride (downsampling factor) from encoder
            decoder_channels: Decoder feature dimension
            use_bn: Whether to use batch normalization
        """
        super().__init__()

        self.decoder = ViTDecoderDPTHybrid(
            encoder_channels=encoder_channels,
            encoder_strides=encoder_strides,
            decoder_channels=decoder_channels,
            use_bn=use_bn,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List from DPTHybridEncoder

        Returns:
            Decoded features at 1/2 resolution
        """
        return self.decoder(features)


# Usage example
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ViTDecoderDPTHybrid")
    print("=" * 60)

    # DPT-Hybrid typical configuration
    encoder_channels = [256, 512, 768, 768]
    encoder_strides = [4, 8, 16, 16]

    decoder = ViTDecoderDPTHybrid(
        encoder_channels=encoder_channels,
        encoder_strides=encoder_strides,
        decoder_channels=256,
        use_bn=True,
    )

    # Simulate encoder outputs for 512x1280 input
    # 512/4=128, 1280/4=320 (R0)
    # 512/8=64, 1280/8=160 (R1)
    # 512/16=32, 1280/16=80 (Transformer)
    B = 2
    dummy_features = [
        torch.randn(B, 256, 128, 320),   # R0: 1/4 res
        torch.randn(B, 512, 64, 160),    # R1: 1/8 res
        torch.randn(B, 768, 32, 80),     # Transformer layer 9: 1/16 res
        torch.randn(B, 768, 32, 80),     # Transformer layer 12: 1/16 res
    ]

    output = decoder(dummy_features)

    print("Input feature shapes:")
    for i, feat in enumerate(dummy_features):
        stride = encoder_strides[i]
        print(f"  [{i}]: {feat.shape}  (1/{stride} resolution)")

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (B={B}, C=256, H=256, W=640)  # 1/2 resolution")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Memory comparison")
    print("=" * 60)

    # Calculate tensor sizes
    print("Intermediate tensor sizes (per sample, float32):")
    print(f"  layer_rn[0] (1/4):  {256*128*320*4/1024/1024:.1f} MB")
    print(f"  layer_rn[1] (1/8):  {256*64*160*4/1024/1024:.1f} MB")
    print(f"  layer_rn[2] (1/16): {256*32*80*4/1024/1024:.1f} MB")
    print(f"  layer_rn[3] (1/32): {256*16*40*4/1024/1024:.1f} MB")
    print(f"  output (1/2):       {256*256*640*4/1024/1024:.1f} MB")
