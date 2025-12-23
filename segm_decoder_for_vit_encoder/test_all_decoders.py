"""
Quick test script to verify all decoders work correctly.

Tests:
1. Import check
2. Forward pass check
3. Output shape check
"""

import torch
import sys
from pathlib import Path

# Add all case directories to path
for case_dir in Path(__file__).parent.glob("case*"):
    sys.path.insert(0, str(case_dir))


def test_decoder(name, decoder_class, **kwargs):
    """Test a single decoder implementation."""
    print(f"\nTesting {name}...")
    print("-" * 40)

    try:
        # Create decoder
        decoder = decoder_class(**kwargs)
        print(f"✓ Decoder created successfully")

        # Create dummy features (ViT-style: all same resolution)
        B, H, W = 2, 32, 80
        features = [
            torch.randn(B, 384, H, W),
            torch.randn(B, 384, H, W),
            torch.randn(B, 384, H, W),
        ]
        print(f"✓ Input features created: {[f.shape for f in features]}")

        # Forward pass
        with torch.inference_mode():
            output = decoder(*features)
        print(f"✓ Forward pass successful")

        # Check output shape
        expected_shape = (B, kwargs['decoder_channels'], H * 16, W * 16)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Output shape correct: {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in decoder.parameters())
        print(f"✓ Total parameters: {total_params:,}")

        print(f"✅ {name} - ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ {name} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("Testing All ViT Decoder Implementations")
    print("=" * 80)

    # Common kwargs for all decoders
    common_kwargs = {
        'encoder_channels': [384, 384, 384],
        'decoder_channels': 256,
        'final_upsampling': 16,
    }

    # Test all decoders
    results = {}

    # Case 1: MLP Mixer
    from case1_mlp_mixer.decoder import ViTDecoderMLPMixer
    results['Case 1: MLP Mixer'] = test_decoder(
        'Case 1: MLP Mixer',
        ViTDecoderMLPMixer,
        **common_kwargs,
        num_mixer_blocks=4
    )

    # Case 2: Multi-Scale FPN
    from case2_multiscale_fpn.decoder import ViTDecoderMultiScaleFPN
    results['Case 2: Multi-Scale FPN'] = test_decoder(
        'Case 2: Multi-Scale FPN',
        ViTDecoderMultiScaleFPN,
        **common_kwargs
    )

    # Case 3: Hierarchical Attention
    from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention
    results['Case 3: Hierarchical Attention'] = test_decoder(
        'Case 3: Hierarchical Attention',
        ViTDecoderHierarchicalAttention,
        **common_kwargs
    )

    # Case 4: Cross-Attention
    from case4_cross_attention.decoder import ViTDecoderCrossAttention
    results['Case 4: Cross-Attention'] = test_decoder(
        'Case 4: Cross-Attention',
        ViTDecoderCrossAttention,
        **common_kwargs,
        num_decoder_layers=4,
        num_heads=8
    )

    # Case 5: Weighted Sum (V1)
    from case5_weighted_sum.decoder import ViTDecoderWeightedSum
    results['Case 5: Weighted Sum V1'] = test_decoder(
        'Case 5: Weighted Sum V1',
        ViTDecoderWeightedSum,
        **common_kwargs
    )

    # Case 5: Weighted Sum (V2)
    from case5_weighted_sum.decoder import ViTDecoderWeightedSumV2
    results['Case 5: Weighted Sum V2'] = test_decoder(
        'Case 5: Weighted Sum V2',
        ViTDecoderWeightedSumV2,
        **common_kwargs
    )

    # Case 6: FPN-Style (V1)
    from case6_fpn_style.decoder import ViTDecoderFPNStyle
    results['Case 6: FPN-Style V1'] = test_decoder(
        'Case 6: FPN-Style V1',
        ViTDecoderFPNStyle,
        **common_kwargs
    )

    # Case 6: FPN-Style (V2)
    from case6_fpn_style.decoder import ViTDecoderFPNStyleV2
    results['Case 6: FPN-Style V2'] = test_decoder(
        'Case 6: FPN-Style V2',
        ViTDecoderFPNStyleV2,
        **common_kwargs
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<40} {status}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
