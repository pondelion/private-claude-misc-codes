"""
DPT-Hybrid Encoder Wrapper for timm

timmのHybrid ViTモデルからResNet中間特徴とTransformer特徴を
同時に抽出するためのラッパー。

=== DPT-Hybrid の特徴抽出 ===

Input Image: (B, 3, H, W)
    ↓
ResNet50 Stem
    ↓
ResNet Block1 → R0: (B, 256, H/4, W/4)   ← skip connection用
    ↓
ResNet Block2 → R1: (B, 512, H/8, W/8)   ← skip connection用
    ↓
ResNet Block3 → tokens: (B, 1024, H/16, W/16) → flatten → ViT入力
    ↓
ViT Transformer Layers
    ↓
Layer 9  → (B, 768, H/16, W/16)
Layer 12 → (B, 768, H/16, W/16)

出力: [R0, R1, layer9, layer12]
      異なる解像度・チャネル数の特徴リスト
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import timm


class DPTHybridEncoder(nn.Module):
    """
    DPT-Hybrid style encoder using timm's hybrid ViT.

    ResNet中間層（R0, R1）とTransformer層の特徴を同時に抽出。

    Shape:
        Input:  (B, 3, H, W)
        Output: List of tensors with different resolutions:
            - R0:      (B, C_r0, H/4, W/4)    e.g., (B, 256, 128, 320)
            - R1:      (B, C_r1, H/8, W/8)    e.g., (B, 512, 64, 160)
            - layer_a: (B, C_vit, H/16, W/16) e.g., (B, 768, 32, 80)
            - layer_b: (B, C_vit, H/16, W/16) e.g., (B, 768, 32, 80)
    """

    def __init__(
        self,
        model_name: str = 'vit_base_r50_s16_384',
        pretrained: bool = True,
        transformer_indices: Tuple[int, int] = (8, 11),  # 0-indexed, so 9th and 12th layers
    ):
        """
        Args:
            model_name: timm model name for hybrid ViT
                        e.g., 'vit_base_r50_s16_384', 'vit_base_r50_s16_224'
            pretrained: Whether to use pretrained weights
            transformer_indices: Which transformer layers to extract (0-indexed)
                                 Default (8, 11) = 9th and 12th layers (1-indexed)
        """
        super().__init__()

        self.transformer_indices = transformer_indices

        # Load hybrid ViT model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Storage for intermediate features
        self._features = {}

        # Register hooks for ResNet intermediate layers
        self._register_hooks()

        # Get channel info
        self._init_channel_info()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""

        # Hook for ResNet Block1 output (R0)
        def hook_r0(module, input, output):
            self._features['r0'] = output

        # Hook for ResNet Block2 output (R1)
        def hook_r1(module, input, output):
            self._features['r1'] = output

        # Hook for transformer layers
        def make_transformer_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        # Access ResNet backbone in patch_embed
        # timm's hybrid ViT structure: model.patch_embed.backbone
        backbone = self.model.patch_embed.backbone

        # Register hooks on ResNet stages
        # stage1 = layer1, stage2 = layer2 in ResNet terminology
        if hasattr(backbone, 'layer1'):
            backbone.layer1.register_forward_hook(hook_r0)
        if hasattr(backbone, 'layer2'):
            backbone.layer2.register_forward_hook(hook_r1)

        # Register hooks on transformer blocks
        for idx in self.transformer_indices:
            if hasattr(self.model, 'blocks') and idx < len(self.model.blocks):
                self.model.blocks[idx].register_forward_hook(
                    make_transformer_hook(f'transformer_{idx}')
                )

    def _init_channel_info(self):
        """Initialize channel information by running a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            _ = self.forward(dummy)

            self.channels = {
                'r0': self._features.get('r0', torch.zeros(1, 256, 1, 1)).shape[1],
                'r1': self._features.get('r1', torch.zeros(1, 512, 1, 1)).shape[1],
            }

            # Transformer features
            for idx in self.transformer_indices:
                key = f'transformer_{idx}'
                if key in self._features:
                    feat = self._features[key]
                    if feat.ndim == 3:  # (B, N, C)
                        self.channels[key] = feat.shape[-1]
                    else:
                        self.channels[key] = feat.shape[1]

            self._features.clear()

    def get_output_channels(self) -> List[int]:
        """
        Get output channel sizes for decoder initialization.

        Returns:
            List of channel sizes: [C_r0, C_r1, C_transformer, C_transformer]
        """
        channels = [self.channels['r0'], self.channels['r1']]
        for idx in self.transformer_indices:
            channels.append(self.channels.get(f'transformer_{idx}', 768))
        return channels

    def get_output_strides(self) -> List[int]:
        """
        Get output stride (downsampling factor) for each feature.

        Returns:
            List of strides: [4, 8, 16, 16]
        """
        return [4, 8, 16, 16]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            List of features:
                - R0: (B, C_r0, H/4, W/4)
                - R1: (B, C_r1, H/8, W/8)
                - transformer_a: (B, C_vit, H/16, W/16)
                - transformer_b: (B, C_vit, H/16, W/16)
        """
        self._features.clear()
        B, _, H, W = x.shape

        # Forward through the model (hooks will capture features)
        _ = self.model.forward_features(x)

        # Collect features
        features = []

        # R0 and R1 from ResNet
        if 'r0' in self._features:
            features.append(self._features['r0'])
        if 'r1' in self._features:
            features.append(self._features['r1'])

        # Transformer features
        # Need to reshape from (B, N, C) to (B, C, H, W)
        h, w = H // 16, W // 16

        for idx in self.transformer_indices:
            key = f'transformer_{idx}'
            if key in self._features:
                feat = self._features[key]
                if feat.ndim == 3:  # (B, N, C) -> (B, C, H, W)
                    # Remove CLS token if present
                    if feat.shape[1] == h * w + 1:
                        feat = feat[:, 1:, :]  # Remove CLS token
                    feat = feat.transpose(1, 2).reshape(B, -1, h, w)
                features.append(feat)

        return features


class DPTHybridEncoderSimple(nn.Module):
    """
    Simplified DPT-Hybrid encoder using separate ResNet and ViT.

    timmのhybridモデルが使いにくい場合の代替実装。
    ResNetとViTを別々に使い、ResNet特徴をViTに渡す。

    Note: これは近似的な実装で、本来のDPT-Hybridとは若干異なる。
    """

    def __init__(
        self,
        resnet_name: str = 'resnet50',
        vit_name: str = 'vit_base_patch16_384',
        pretrained: bool = True,
        vit_indices: Tuple[int, ...] = (5, 11),
    ):
        """
        Args:
            resnet_name: timm ResNet model name
            vit_name: timm ViT model name
            pretrained: Whether to use pretrained weights
            vit_indices: Which ViT layers to extract
        """
        super().__init__()

        self.vit_indices = vit_indices

        # ResNet for feature extraction
        self.resnet = timm.create_model(
            resnet_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # layer1, layer2, layer3
        )

        # ViT for global context
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Projection from ResNet to ViT dimension
        vit_dim = self.vit.embed_dim
        resnet_out_ch = 1024  # ResNet layer3 output
        self.proj = nn.Conv2d(resnet_out_ch, vit_dim, kernel_size=1)

        self._vit_features = {}
        self._register_vit_hooks()

    def _register_vit_hooks(self):
        """Register hooks for ViT layers."""
        def make_hook(name):
            def hook(module, input, output):
                self._vit_features[name] = output
            return hook

        for idx in self.vit_indices:
            if idx < len(self.vit.blocks):
                self.vit.blocks[idx].register_forward_hook(make_hook(f'vit_{idx}'))

    def get_output_channels(self) -> List[int]:
        """Get output channel sizes."""
        # [R0, R1, vit_layer1, vit_layer2]
        return [256, 512, self.vit.embed_dim, self.vit.embed_dim]

    def get_output_strides(self) -> List[int]:
        """Get output strides."""
        return [4, 8, 16, 16]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features.

        Args:
            x: Input (B, 3, H, W)

        Returns:
            List of [R0, R1, vit_feat1, vit_feat2]
        """
        B, _, H, W = x.shape
        self._vit_features.clear()

        # ResNet features
        resnet_feats = self.resnet(x)
        r0 = resnet_feats[0]  # 1/4 res
        r1 = resnet_feats[1]  # 1/8 res
        r2 = resnet_feats[2]  # 1/16 res

        # Project ResNet features and reshape for ViT
        proj_feat = self.proj(r2)  # (B, vit_dim, H/16, W/16)

        # Reshape to sequence for ViT
        h, w = proj_feat.shape[2], proj_feat.shape[3]
        tokens = proj_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        # Add position embedding (simplified - just use ViT's)
        # Note: This is approximate, real DPT-Hybrid handles this differently
        pos_embed = self.vit.pos_embed
        if pos_embed.shape[1] > tokens.shape[1] + 1:
            # Interpolate position embedding
            pos_embed = self._interpolate_pos_embed(pos_embed, h, w)

        # Add CLS token
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + pos_embed[:, :tokens.shape[1], :]

        # Forward through ViT blocks
        for blk in self.vit.blocks:
            tokens = blk(tokens)

        # Collect ViT features
        vit_feats = []
        for idx in self.vit_indices:
            key = f'vit_{idx}'
            if key in self._vit_features:
                feat = self._vit_features[key]
                # Remove CLS, reshape to spatial
                feat = feat[:, 1:, :].transpose(1, 2).reshape(B, -1, h, w)
                vit_feats.append(feat)

        return [r0, r1] + vit_feats

    def _interpolate_pos_embed(self, pos_embed, h, w):
        """Interpolate position embedding to target size."""
        # Simplified interpolation
        N = pos_embed.shape[1] - 1
        dim = pos_embed.shape[2]

        cls_pos = pos_embed[:, 0:1, :]
        patch_pos = pos_embed[:, 1:, :]

        # Assume square original
        orig_size = int(N ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=(h, w), mode='bilinear')
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)


# Usage example
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DPTHybridEncoder")
    print("=" * 60)

    # Note: This requires the actual timm hybrid model to work
    # Some hybrid models may have different internal structures

    try:
        encoder = DPTHybridEncoder(
            model_name='vit_base_r50_s16_384.orig_in21k',
            pretrained=False,  # Set True for actual use
        )

        x = torch.randn(2, 3, 384, 384)
        features = encoder(x)

        print(f"Input shape: {x.shape}")
        print(f"Output channels: {encoder.get_output_channels()}")
        print(f"Output strides: {encoder.get_output_strides()}")
        print("Feature shapes:")
        for i, feat in enumerate(features):
            print(f"  [{i}]: {feat.shape}")

    except Exception as e:
        print(f"DPTHybridEncoder test failed: {e}")
        print("This may be due to model architecture differences in timm versions.")

    print("\n" + "=" * 60)
    print("Testing DPTHybridEncoderSimple")
    print("=" * 60)

    try:
        encoder_simple = DPTHybridEncoderSimple(
            pretrained=False,
        )

        x = torch.randn(2, 3, 384, 384)
        features = encoder_simple(x)

        print(f"Input shape: {x.shape}")
        print(f"Output channels: {encoder_simple.get_output_channels()}")
        print("Feature shapes:")
        for i, feat in enumerate(features):
            print(f"  [{i}]: {feat.shape}")

    except Exception as e:
        print(f"DPTHybridEncoderSimple test failed: {e}")
