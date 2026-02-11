"""
Case 7: DPT-Style Decoder (Dense Prediction Transformer)

特徴:
- マルチスケール特徴ピラミッドの構築（ViT特徴をリサンプリング）
- ResidualConvUnitによる特徴精緻化
- RefineNet風のFeatureFusionBlockで段階的に融合・アップサンプリング
- 深い層から浅い層へのcoarse-to-fine refinement

参考: "Vision Transformers for Dense Prediction" (Ranftl et al., 2021)
https://github.com/intel-isl/DPT

=== Shape Flow (例: 入力512x1280, ViT patch_size=16, 4段階) ===

入力画像: (B, 3, 512, 1280)
    ↓ ViT Encoder (timm)
ViT特徴 (全て同一解像度 1/16):
    - feat[0]: (B, 384, 32, 80)  # Early layer
    - feat[1]: (B, 384, 32, 80)
    - feat[2]: (B, 384, 32, 80)
    - feat[3]: (B, 384, 32, 80)  # Deep layer
    ↓ Reassemble (Project + Resample)
マルチスケールピラミッド:
    - layer_rn[0]: (B, 256, 128, 320)  # 4x upsample → 1/4 res
    - layer_rn[1]: (B, 256, 64, 160)   # 2x upsample → 1/8 res
    - layer_rn[2]: (B, 256, 32, 80)    # no change  → 1/16 res
    - layer_rn[3]: (B, 256, 16, 40)    # 2x downsample → 1/32 res
    ↓ FeatureFusionBlocks (Deep → Shallow)
    - path = fusion[3](layer_rn[3])           → (B, 256, 32, 80)   # 2x up
    - path = fusion[2](path, layer_rn[2])     → (B, 256, 64, 160)  # 2x up
    - path = fusion[1](path, layer_rn[1])     → (B, 256, 128, 320) # 2x up
    - path = fusion[0](path, layer_rn[0])     → (B, 256, 256, 640) # 2x up (1/2 res)
    ↓ (デコーダ出力は1/2解像度)
出力: (B, 256, 256, 640)  ※論文通り1/2解像度

※フル解像度へのアップサンプリングはOutput Head（セグメンテーションヘッド等）で行う
※Output Headでチャネル数を削減しながらアップサンプリングするのでメモリ効率が良い
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualConvUnit(nn.Module):
    """
    Residual Convolutional Unit from RefineNet/DPT.

    Structure: ReLU → Conv3x3 → (BN) → ReLU → Conv3x3 → (BN) → + input

    Shape:
        Input:  (B, C, H, W)
        Output: (B, C, H, W)  ※空間サイズ・チャネル数は変化しない
    """

    def __init__(self, features: int, use_bn: bool = True):
        """
        Args:
            features: Number of input/output channels
            use_bn: Whether to use batch normalization
        """
        super().__init__()

        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not use_bn
        )

        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        out = self.relu(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """
    Feature Fusion Block from DPT/RefineNet.

    Combines features from deeper level with skip connection from current level,
    applies residual conv units, and upsamples by 2x.

    Shape (upsample=True の場合):
        Input xs[0] (path from deeper): (B, C, H, W)
        Input xs[1] (skip connection):  (B, C, H, W)  ※optional
        Output: (B, C, 2H, 2W)  ※2倍にアップサンプリング

    Shape (upsample=False の場合):
        Input xs[0]: (B, C, H, W)
        Input xs[1]: (B, C, H, W)  ※optional
        Output: (B, C, H, W)  ※サイズ変化なし
    """

    def __init__(
        self,
        features: int,
        use_bn: bool = True,
        upsample: bool = True,
        align_corners: bool = True,
    ):
        """
        Args:
            features: Number of channels
            use_bn: Whether to use batch normalization
            upsample: Whether to apply 2x upsampling
            align_corners: align_corners for bilinear interpolation
        """
        super().__init__()

        self.upsample = upsample
        self.align_corners = align_corners

        # ResidualConvUnit for skip connection
        self.resConfUnit1 = ResidualConvUnit(features, use_bn=use_bn)
        # ResidualConvUnit for fused features
        self.resConfUnit2 = ResidualConvUnit(features, use_bn=use_bn)

        # Output projection (optional, used in DPT for channel adjustment)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xs: Variable number of input tensors
                - If one tensor: Only process and upsample (for deepest layer)
                - If two tensors: Fuse path from deeper level with skip connection

        Returns:
            Fused and upsampled tensor (B, C, 2H, 2W) if upsample=True
        """
        output = xs[0]

        if len(xs) == 2:
            # Fuse with skip connection
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if self.upsample:
            output = F.interpolate(
                output,
                scale_factor=2,
                mode='bilinear',
                align_corners=self.align_corners
            )

        output = self.out_conv(output)

        return output


class Reassemble(nn.Module):
    """
    Reassemble module that projects and resamples ViT features.

    Since timm ViT returns features in (B, C, H, W) format,
    we only need to project channels and resample spatially.

    Shape (例: in_channels=384, out_channels=256, H=32, W=80):
        scale_factor=4.0:  (B, 384, 32, 80) → (B, 256, 128, 320)  # 4x upsample
        scale_factor=2.0:  (B, 384, 32, 80) → (B, 256, 64, 160)   # 2x upsample
        scale_factor=1.0:  (B, 384, 32, 80) → (B, 256, 32, 80)    # no change
        scale_factor=0.5:  (B, 384, 32, 80) → (B, 256, 16, 40)    # 2x downsample
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float = 1.0,
    ):
        """
        Args:
            in_channels: Input channel size
            out_channels: Output channel size (typically 256)
            scale_factor: Spatial resampling factor
                - > 1.0: upsample
                - < 1.0: downsample
                - == 1.0: no spatial change
        """
        super().__init__()

        self.scale_factor = scale_factor

        # Channel projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Spatial resampling
        if scale_factor > 1.0:
            # Upsample using transposed convolution
            scale_int = int(scale_factor)
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels,
                kernel_size=scale_int, stride=scale_int, padding=0, bias=False
            )
        elif scale_factor < 1.0:
            # Downsample using strided convolution
            scale_int = int(1.0 / scale_factor)
            self.resample = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=scale_int, padding=1, bias=False
            )
        else:
            self.resample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C_in, H, W)

        Returns:
            Projected and resampled tensor (B, C_out, H', W')
        """
        x = self.proj(x)

        if self.resample is not None:
            x = self.resample(x)

        return x


class ViTDecoderDPT(nn.Module):
    """
    DPT-Style Decoder for ViT encoder.

    Creates multi-scale feature pyramid from constant-resolution ViT features,
    then progressively fuses and upsamples using RefineNet-style fusion blocks.

    Architecture:
        ViT Features (all at 1/16 resolution)
            ↓ Reassemble to multi-scale
        [1/4, 1/8, 1/16, 1/32] feature pyramid
            ↓ Progressive fusion (deep → shallow)
        Final output at 1/2 resolution
            ↓ final_upsampling
        Full resolution output

    Shape (例: 入力512x1280, patch_size=16, 4段階):
        Input:  List of (B, 384, 32, 80) × 4
        Output: (B, 256, 256, 640)  ※1/2解像度（論文通り）

    Note:
        論文では最終アップサンプリングはOutput Head（セグメンテーションヘッド等）で行う。
        Output Headでチャネル数を削減(256→128→32→num_classes)しながら
        アップサンプリングするのでメモリ効率が良い。
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: int = 256,
        use_bn: bool = True,
        align_corners: bool = True,
        upsampling_mode: str = 'bilinear',
        final_upsampling: int = 2,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels for each stage
                              e.g., [384, 384, 384, 384] for ViT-S with 4 stages
            decoder_channels: Decoder feature dimension (default: 256)
            use_bn: Whether to use batch normalization in fusion blocks
            align_corners: align_corners for bilinear interpolation
            upsampling_mode: Final upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor
                              - 2 (default, 論文通り): 出力は1/2解像度 → Output Headでフル解像度へ
                              - 16: 出力はフル解像度（メモリ注意）
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.align_corners = align_corners
        self.num_stages = len(encoder_channels)

        # Reassemble modules to create multi-scale feature pyramid
        # For ViT with patch_size=16, features are at 1/16 resolution
        # We create a pyramid at: 1/4, 1/8, 1/16, 1/32 of input
        # Scale factors relative to ViT features (which are at 1/16):
        #   - 1/4:  4x upsample (scale=4)
        #   - 1/8:  2x upsample (scale=2)
        #   - 1/16: no change (scale=1)
        #   - 1/32: 2x downsample (scale=0.5)

        # Default scale factors for 4-stage setup
        if self.num_stages >= 4:
            self.scale_factors = [4.0, 2.0, 1.0, 0.5][:self.num_stages]
        elif self.num_stages == 3:
            self.scale_factors = [4.0, 2.0, 1.0]
        elif self.num_stages == 2:
            self.scale_factors = [2.0, 1.0]
        else:
            self.scale_factors = [1.0]

        # Reassemble layers (project and resample)
        self.reassemble_layers = nn.ModuleList()
        for i, (enc_ch, scale) in enumerate(zip(encoder_channels, self.scale_factors)):
            self.reassemble_layers.append(
                Reassemble(enc_ch, decoder_channels, scale_factor=scale)
            )

        # Feature Fusion Blocks (RefineNet-style)
        # Process from deepest (lowest resolution) to shallowest (highest resolution)
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            # Last fusion block doesn't need to upsample if it's at target resolution
            upsample = True
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    decoder_channels,
                    use_bn=use_bn,
                    upsample=upsample,
                    align_corners=align_corners
                )
            )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tuple of ViT encoder outputs, each of shape (B, C, H, W)
                      All features should have the same spatial resolution (ViT property)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H*final_upsampling, W*final_upsampling)

        Shape Flow (例: 入力512x1280, patch_size=16, 4段階, decoder_channels=256):
            Input features (all at 1/16 = 32x80):
                feat[0]: (B, 384, 32, 80)
                feat[1]: (B, 384, 32, 80)
                feat[2]: (B, 384, 32, 80)
                feat[3]: (B, 384, 32, 80)

            After Reassemble (multi-scale pyramid):
                layer_rn[0]: (B, 256, 128, 320)  # 1/4 res
                layer_rn[1]: (B, 256, 64, 160)   # 1/8 res
                layer_rn[2]: (B, 256, 32, 80)    # 1/16 res
                layer_rn[3]: (B, 256, 16, 40)    # 1/32 res

            After Progressive Fusion (deep → shallow, each 2x upsample):
                fusion[3](layer_rn[3])           → (B, 256, 32, 80)
                fusion[2](path, layer_rn[2])     → (B, 256, 64, 160)
                fusion[1](path, layer_rn[1])     → (B, 256, 128, 320)
                fusion[0](path, layer_rn[0])     → (B, 256, 256, 640)  # 1/2 res

            After Final Upsampling (final_upsampling=2, 論文通り):
                output: (B, 256, 256, 640)  # 1/2 res ※フル解像度はOutput Headで
        """
        # === Reassemble features into multi-scale pyramid ===
        # layer_rn[0] is highest resolution (from early layer)
        # layer_rn[-1] is lowest resolution (from deep layer)
        # Shape: feat (B, enc_ch, H, W) → layer_rn (B, dec_ch, H', W')
        layer_rn = []
        for i, (feat, reassemble) in enumerate(zip(features, self.reassemble_layers)):
            layer_rn.append(reassemble(feat))

        # === Progressive fusion from deep to shallow ===
        # Start with deepest layer (lowest resolution)
        # Shape: (B, 256, H_deep, W_deep) → (B, 256, 2*H_deep, 2*W_deep)
        path = self.fusion_blocks[-1](layer_rn[-1])

        # Fuse with progressively shallower layers
        # Each fusion: add skip → refine → 2x upsample
        for i in range(self.num_stages - 2, -1, -1):
            # Shape: path (B, 256, H, W), layer_rn[i] (B, 256, H, W) → (B, 256, 2H, 2W)
            path = self.fusion_blocks[i](path, layer_rn[i])

        # === Final upsampling to target resolution ===
        if self.final_upsampling > 1:
            # After 4 fusion stages with 2x each: 1/32 → 1/2 resolution
            # Need additional upsampling to reach full resolution
            # final_upsampling=16, after fusion we're at 1/2, so need 8x more
            # Shape: (B, 256, H/2, W/2) → (B, 256, H, W)
            path = F.interpolate(
                path,
                scale_factor=self.final_upsampling // 2,  # Adjust for 2x from fusion
                mode=self.upsampling_mode,
                align_corners=self.align_corners if self.upsampling_mode == 'bilinear' else None
            )

        return path


class ViTDecoderDPTSimplified(nn.Module):
    """
    Simplified DPT-Style Decoder for ViT encoder.

    This version works better when all ViT features have the same resolution.
    Instead of creating a multi-scale pyramid, it directly fuses features
    and uses the DPT-style residual conv units for refinement.

    More efficient for standard ViT where all features are at 1/16 resolution.

    === Shape Flow (例: 入力512x1280, ViT patch_size=16, 3段階) ===

    入力画像: (B, 3, 512, 1280)
        ↓ ViT Encoder
    ViT特徴 (全て同一解像度 1/16 = 32x80):
        - feat[0]: (B, 384, 32, 80)
        - feat[1]: (B, 384, 32, 80)
        - feat[2]: (B, 384, 32, 80)
        ↓ Project (1x1 Conv)
    投影後:
        - proj[0]: (B, 256, 32, 80)
        - proj[1]: (B, 256, 32, 80)
        - proj[2]: (B, 256, 32, 80)
        ↓ FeatureFusionBlocks (Deep → Shallow, no upsample)
        - path = fusion[2](proj[2])         → (B, 256, 32, 80)
        - path = fusion[1](path, proj[1])   → (B, 256, 32, 80)
        - path = fusion[0](path, proj[0])   → (B, 256, 32, 80)
        ↓ Final RCUs
        - path = rcu1(path)  → (B, 256, 32, 80)
        - path = rcu2(path)  → (B, 256, 32, 80)
        ↓ Final Upsampling (8x, デフォルト)
    出力: (B, 256, 256, 640)  ※1/2解像度、フル解像度はOutput Headで
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: int = 256,
        use_bn: bool = True,
        upsampling_mode: str = 'bilinear',
        final_upsampling: int = 8,
    ):
        """
        Args:
            encoder_channels: List of encoder output channels
            decoder_channels: Output channels for decoder
            use_bn: Whether to use batch normalization
            upsampling_mode: Upsampling mode ('bilinear' or 'nearest')
            final_upsampling: Final upsampling factor
                              - 8 (default): 1/16 → 1/2解像度 (論文に近い設計)
                              - 16: 1/16 → フル解像度（メモリ注意）
        """
        super().__init__()

        self.upsampling_mode = upsampling_mode
        self.final_upsampling = final_upsampling
        self.num_stages = len(encoder_channels)

        # Project all encoder features to decoder channels
        self.proj_layers = nn.ModuleList()
        for enc_ch in encoder_channels:
            self.proj_layers.append(
                nn.Conv2d(enc_ch, decoder_channels, kernel_size=1, bias=False)
            )

        # Fusion blocks (one for each stage, processing deep → shallow)
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_stages):
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    decoder_channels,
                    use_bn=use_bn,
                    upsample=False,  # Don't upsample between layers (same resolution)
                    align_corners=True
                )
            )

        # Final refinement with 2 RCUs
        self.final_rcu1 = ResidualConvUnit(decoder_channels, use_bn=use_bn)
        self.final_rcu2 = ResidualConvUnit(decoder_channels, use_bn=use_bn)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tuple of ViT encoder outputs, each of shape (B, C, H, W)

        Returns:
            Decoded feature map of shape (B, decoder_channels, H_up, W_up)

        Shape Flow (例: 入力512x1280, patch_size=16, 3段階, decoder_channels=256):
            Input features (all at 1/16 = 32x80):
                feat[0]: (B, 384, 32, 80)
                feat[1]: (B, 384, 32, 80)
                feat[2]: (B, 384, 32, 80)

            After Projection:
                proj[0]: (B, 256, 32, 80)
                proj[1]: (B, 256, 32, 80)
                proj[2]: (B, 256, 32, 80)

            After Fusion (no spatial change):
                path: (B, 256, 32, 80)

            After Final RCUs:
                path: (B, 256, 32, 80)

            After Final Upsampling (8x, デフォルト):
                output: (B, 256, 256, 640)  # 1/2 res ※フル解像度はOutput Headで
        """
        # === Project all features to decoder channels ===
        # Shape: (B, enc_ch, H, W) → (B, dec_ch, H, W)
        proj_features = []
        for feat, proj in zip(features, self.proj_layers):
            proj_features.append(proj(feat))

        # === Progressive fusion from deep to shallow ===
        # Start with deepest (last) feature
        # Shape: (B, 256, H, W) → (B, 256, H, W) (no upsample in this version)
        path = self.fusion_blocks[-1](proj_features[-1])

        # Fuse with progressively shallower features
        # Shape: path + proj[i] → (B, 256, H, W)
        for i in range(self.num_stages - 2, -1, -1):
            path = self.fusion_blocks[i](path, proj_features[i])

        # === Final refinement with 2 RCUs ===
        # Shape: (B, 256, H, W) → (B, 256, H, W)
        path = self.final_rcu1(path)
        path = self.final_rcu2(path)

        # === Final upsampling ===
        # Shape: (B, 256, H, W) → (B, 256, H*16, W*16)
        if self.final_upsampling > 1:
            path = F.interpolate(
                path,
                scale_factor=self.final_upsampling,
                mode=self.upsampling_mode,
                align_corners=False if self.upsampling_mode == 'bilinear' else None
            )

        return path


# Usage example
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ViTDecoderDPT (Full Multi-Scale Version)")
    print("=" * 60)

    # ViT encoder channels (all same resolution for pure ViT)
    encoder_channels = [384, 384, 384, 384]

    # final_upsampling=2 (default): 出力は1/2解像度（論文通り）
    decoder = ViTDecoderDPT(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        use_bn=True,
        final_upsampling=2  # 1/2解像度出力（デフォルト）
    )

    # Simulate ViT encoder outputs (all at 1/16 resolution)
    # 入力画像: 512x1280 → ViT特徴: 32x80 (1/16)
    B, H, W = 2, 32, 80
    dummy_features = [
        torch.randn(B, 384, H, W),  # Early layer
        torch.randn(B, 384, H, W),  # Mid layer
        torch.randn(B, 384, H, W),  # Deeper layer
        torch.randn(B, 384, H, W),  # Deepest layer
    ]

    output = decoder(*dummy_features)
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (B={B}, C=256, H={H*8}, W={W*8})  # 1/2解像度")
    print("※フル解像度(512x1280)へはOutput Head(セグメンテーションヘッド等)でアップサンプリング")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Testing ViTDecoderDPTSimplified")
    print("=" * 60)

    # Test with 3-stage setup (more common)
    encoder_channels_3 = [384, 384, 384]

    # final_upsampling=8 (default): 1/16 → 1/2解像度
    decoder_simple = ViTDecoderDPTSimplified(
        encoder_channels=encoder_channels_3,
        decoder_channels=256,
        use_bn=True,
        final_upsampling=8  # 1/2解像度出力（デフォルト）
    )

    dummy_features_3 = [
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
        torch.randn(B, 384, H, W),
    ]

    output_simple = decoder_simple(*dummy_features_3)
    print(f"Input feature shapes: {[f.shape for f in dummy_features_3]}")
    print(f"Output shape: {output_simple.shape}")
    print(f"Expected output shape: (B={B}, C=256, H={H*8}, W={W*8})  # 1/2解像度")

    total_params_simple = sum(p.numel() for p in decoder_simple.parameters())
    print(f"Total parameters: {total_params_simple:,}")
