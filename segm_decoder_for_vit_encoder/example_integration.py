"""
Example integration of ViT decoders with SignalSegModelV7-style architecture.

Shows how to integrate any of the proposed decoders into the existing
SignalSegModelV7 framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path

# Add decoder paths
for case_dir in Path(__file__).parent.glob("case*"):
    sys.path.insert(0, str(case_dir))

from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention


# Constants (from original code)
SIG_SEG_IMAGE_SIZE = (512, 1280)


class SegmentationHead(nn.Sequential):
    """Segmentation head with optional upsampling and activation."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling_layer = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )

        # Simplified activation (no custom Activation class needed for this example)
        if activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        elif activation == "softmax":
            activation_layer = nn.Softmax(dim=1)
        else:
            activation_layer = nn.Identity()

        super().__init__(conv2d, upsampling_layer, activation_layer)


class SignalSegModelViT(nn.Module):
    """
    Signal segmentation model using ViT encoder with custom decoder.

    Similar architecture to SignalSegModelV7, but adapted for ViT encoders.
    """

    def __init__(
        self,
        num_classes=1,
        encoder_model_name="vit_small_patch16_224",  # ViT model
        decoder_type="hierarchical_attention",  # Which decoder to use
        decoder_channels=256,
        fixed_y_coord_scale=2 * 0.01,
        detach_y_coord_path=True,
        apply_lead_pe=True,
        # Decoder-specific kwargs
        decoder_kwargs=None,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            encoder_model_name: Name of ViT model from timm
            decoder_type: Which decoder to use
                - "mlp_mixer"
                - "multiscale_fpn"
                - "hierarchical_attention"
                - "cross_attention"
                - "weighted_sum"
                - "fpn_style"
            decoder_channels: Number of decoder output channels
            fixed_y_coord_scale: Scale for y-coordinate prediction
            detach_y_coord_path: Whether to detach segmentation for y-coord path
            apply_lead_pe: Whether to apply lead-wise positional encoding
            decoder_kwargs: Additional kwargs for decoder
        """
        super().__init__()

        self.num_classes = num_classes
        self.detach_y_coord_path = detach_y_coord_path
        self.fixed_y_coord_scale = fixed_y_coord_scale
        self.apply_lead_pe = apply_lead_pe
        self.n_leads = 16

        # ---------------------------------------------------------------------
        # Encoder (ViT)
        # ---------------------------------------------------------------------
        # Get more features from ViT (e.g., blocks 0, 4, 8, 11 for 12-layer ViT)
        # Adjust based on model depth
        self._encoder = timm.create_model(
            encoder_model_name,
            features_only=True,
            pretrained=False,
            # out_indices can be customized based on model depth
            # For 12-layer ViT: (0, 4, 11) gives shallow, mid, deep features
        )

        encoder_channels = self._encoder.feature_info.channels()
        encoder_reductions = [info["reduction"] for info in self._encoder.feature_info.info]

        print(f"Encoder channels: {encoder_channels}")
        print(f"Encoder reductions: {encoder_reductions}")

        # For ViT, all reductions should be the same (e.g., 16)
        assert len(set(encoder_reductions)) == 1, "ViT should have constant spatial resolution"
        self.encoder_reduction = encoder_reductions[0]

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        decoder_kwargs = decoder_kwargs or {}

        if decoder_type == "mlp_mixer":
            from case1_mlp_mixer.decoder import ViTDecoderMLPMixer
            self._decoder = ViTDecoderMLPMixer(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        elif decoder_type == "multiscale_fpn":
            from case2_multiscale_fpn.decoder import ViTDecoderMultiScaleFPN
            self._decoder = ViTDecoderMultiScaleFPN(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        elif decoder_type == "hierarchical_attention":
            from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention
            self._decoder = ViTDecoderHierarchicalAttention(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        elif decoder_type == "cross_attention":
            from case4_cross_attention.decoder import ViTDecoderCrossAttention
            self._decoder = ViTDecoderCrossAttention(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        elif decoder_type == "weighted_sum":
            from case5_weighted_sum.decoder import ViTDecoderWeightedSum
            self._decoder = ViTDecoderWeightedSum(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        elif decoder_type == "fpn_style":
            from case6_fpn_style.decoder import ViTDecoderFPNStyle
            self._decoder = ViTDecoderFPNStyle(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                final_upsampling=self.encoder_reduction,
                **decoder_kwargs
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # ---------------------------------------------------------------------
        # Segmentation head
        # ---------------------------------------------------------------------
        self._segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        # ---------------------------------------------------------------------
        # Lead-wise positional encoding
        # ---------------------------------------------------------------------
        if apply_lead_pe:
            self.lead_pe = nn.Parameter(
                torch.zeros(self.n_leads, 3, *SIG_SEG_IMAGE_SIZE)
            )

        # ---------------------------------------------------------------------
        # Y-coordinate extraction (simplified for this example)
        # ---------------------------------------------------------------------
        # In practice, you would use your DenoisingYCoordExtractionWeightedAvgHeadV3
        self.y_scale_param = nn.Parameter(torch.tensor(1.0))
        self.y_offset_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N_LEADS, C, H, W)

        Returns:
            pred_mask: Segmentation mask (B, N_LEADS, num_classes, H, W)
            pred_y_coord: Placeholder for y-coordinates (B, N_LEADS, W)
        """
        bs, n_leads, ch, h, w = x.shape
        assert n_leads == self.n_leads, f"Expected {self.n_leads} leads, got {n_leads}"

        # Apply lead-wise PE
        if self.apply_lead_pe:
            x = x + self.lead_pe.unsqueeze(0)

        # Merge batch and lead dimensions
        x = x.reshape(bs * n_leads, ch, h, w)

        # Encoder
        features = self._encoder(x)

        # Decoder
        dec_out = self._decoder(*features)

        # Segmentation head
        pred_mask = self._segmentation_head(dec_out)

        # Restore batch and lead dimensions
        pred_mask = pred_mask.reshape(bs, n_leads, self.num_classes, h, w)

        # Placeholder for y-coordinate prediction
        # In practice, implement your y-coord extraction head here
        pred_y_coord = torch.zeros(bs, n_leads, w, device=x.device)

        return pred_mask, pred_y_coord


# Usage example
def main():
    print("=" * 80)
    print("Example: Integrating ViT Decoder with SignalSegModelV7-style Architecture")
    print("=" * 80)

    # Create model with hierarchical attention decoder
    model = SignalSegModelViT(
        num_classes=1,
        encoder_model_name="vit_small_patch16_224",
        decoder_type="hierarchical_attention",
        decoder_channels=256,
        apply_lead_pe=True,
        decoder_kwargs={
            "use_attention": True,
        }
    )

    # Input: (B, N_LEADS, C, H, W)
    B, N_LEADS, C, H, W = 2, 16, 3, 512, 1280
    x = torch.randn(B, N_LEADS, C, H, W)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    pred_mask, pred_y_coord = model(x)

    print(f"Output mask shape: {pred_mask.shape}")
    print(f"Output y_coord shape: {pred_y_coord.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Try different decoder types
    print("\n" + "=" * 80)
    print("Comparing different decoder types:")
    print("=" * 80)

    decoder_types = [
        "mlp_mixer",
        "multiscale_fpn",
        "hierarchical_attention",
        "cross_attention",
        "weighted_sum",
        "fpn_style",
    ]

    for dec_type in decoder_types:
        try:
            model = SignalSegModelViT(
                num_classes=1,
                encoder_model_name="vit_small_patch16_224",
                decoder_type=dec_type,
                decoder_channels=256,
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{dec_type:<25} {total_params:>15,} params")
        except Exception as e:
            print(f"{dec_type:<25} ERROR: {e}")


if __name__ == "__main__":
    main()
