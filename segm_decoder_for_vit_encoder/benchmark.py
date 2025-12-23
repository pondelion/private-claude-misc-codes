"""
Benchmark script to compare all decoder implementations.

Compares:
- Parameter count
- Memory usage
- Inference speed
- Output shape
"""

import torch
import time
import sys
from pathlib import Path

# Add all case directories to path
for case_dir in Path(__file__).parent.glob("case*"):
    sys.path.insert(0, str(case_dir))

from case1_mlp_mixer.decoder import ViTDecoderMLPMixer
from case2_multiscale_fpn.decoder import ViTDecoderMultiScaleFPN
from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention
from case4_cross_attention.decoder import ViTDecoderCrossAttention
from case5_weighted_sum.decoder import ViTDecoderWeightedSum, ViTDecoderWeightedSumV2
from case6_fpn_style.decoder import ViTDecoderFPNStyle, ViTDecoderFPNStyleV2


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_speed(model, dummy_features, num_runs=100, warmup=10):
    """Benchmark inference speed."""
    model.eval()

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(*dummy_features)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.inference_mode():
        for _ in range(num_runs):
            _ = model(*dummy_features)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    return avg_time


def get_memory_usage(model, dummy_features):
    """Get approximate memory usage (forward pass only)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _ = model(*dummy_features)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return memory_mb
    else:
        # Approximate CPU memory (not accurate)
        return 0.0


def main():
    print("=" * 80)
    print("ViT Decoder Benchmark")
    print("=" * 80)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ViT encoder channels (same as vit_small_patch16_dinov3.lvd1689m output)
    encoder_channels = [384, 384, 384]
    decoder_channels = 256
    final_upsampling = 16

    # Input configuration
    B, H, W = 2, 32, 80  # Batch=2, H/16=32, W/16=80 (original: 512x1280)
    print(f"Input shape per feature: (B={B}, C=384, H={H}, W={W})")
    print(f"Expected output shape: (B={B}, C={decoder_channels}, H={H*final_upsampling}, W={W*final_upsampling})")

    # Create dummy features
    dummy_features = [
        torch.randn(B, 384, H, W).to(device),
        torch.randn(B, 384, H, W).to(device),
        torch.randn(B, 384, H, W).to(device),
    ]

    # Define all models
    models = {
        "Case 1: MLP Mixer": ViTDecoderMLPMixer(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_mixer_blocks=4,
            final_upsampling=final_upsampling
        ),
        "Case 2: Multi-Scale FPN": ViTDecoderMultiScaleFPN(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dilation_rates=(6, 12, 18),
            final_upsampling=final_upsampling
        ),
        "Case 3: Hierarchical Attention": ViTDecoderHierarchicalAttention(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_attention=True,
            final_upsampling=final_upsampling
        ),
        "Case 4: Cross-Attention": ViTDecoderCrossAttention(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_decoder_layers=4,
            num_heads=8,
            final_upsampling=final_upsampling
        ),
        "Case 5: Weighted Sum (Simple)": ViTDecoderWeightedSum(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_refinement=True,
            num_refinement_blocks=2,
            final_upsampling=final_upsampling
        ),
        "Case 5: Weighted Sum (Spatial)": ViTDecoderWeightedSumV2(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_refinement=True,
            num_refinement_blocks=2,
            final_upsampling=final_upsampling
        ),
        "Case 6: FPN-Style (Basic)": ViTDecoderFPNStyle(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_refinement=True,
            final_upsampling=final_upsampling
        ),
        "Case 6: FPN-Style (Attention)": ViTDecoderFPNStyleV2(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_attention=True,
            final_upsampling=final_upsampling
        ),
    }

    # Benchmark results
    results = []

    print("\n" + "=" * 80)
    print("Benchmarking...")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n{name}")
        print("-" * 40)

        model = model.to(device)
        model.eval()

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Test forward pass
        try:
            with torch.inference_mode():
                output = model(*dummy_features)
            print(f"  Output shape: {output.shape}")

            # Benchmark speed
            avg_time = benchmark_speed(model, dummy_features, num_runs=50, warmup=5)
            print(f"  Avg inference time: {avg_time:.2f} ms")

            # Memory usage (GPU only)
            if torch.cuda.is_available():
                memory_mb = get_memory_usage(model, dummy_features)
                print(f"  Peak memory: {memory_mb:.2f} MB")
            else:
                memory_mb = 0.0

            results.append({
                'name': name,
                'params': total_params,
                'time_ms': avg_time,
                'memory_mb': memory_mb,
                'output_shape': output.shape
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': name,
                'params': total_params,
                'time_ms': float('inf'),
                'memory_mb': float('inf'),
                'output_shape': 'N/A'
            })

    # Summary table
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<35} {'Params':<15} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 80)

    for r in results:
        params_str = f"{r['params']:,}" if r['params'] < float('inf') else "ERROR"
        time_str = f"{r['time_ms']:.2f}" if r['time_ms'] < float('inf') else "ERROR"
        mem_str = f"{r['memory_mb']:.2f}" if r['memory_mb'] < float('inf') else "ERROR"
        print(f"{r['name']:<35} {params_str:<15} {time_str:<12} {mem_str:<12}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find fastest
    fastest = min(results, key=lambda x: x['time_ms'])
    print(f"\n⚡ Fastest: {fastest['name']} ({fastest['time_ms']:.2f} ms)")

    # Find smallest
    smallest = min(results, key=lambda x: x['params'])
    print(f"🪶 Smallest: {smallest['name']} ({smallest['params']:,} params)")

    # Best balance (weighted score)
    for r in results:
        # Normalize metrics (lower is better)
        norm_params = r['params'] / max(res['params'] for res in results if res['params'] < float('inf'))
        norm_time = r['time_ms'] / max(res['time_ms'] for res in results if res['time_ms'] < float('inf'))
        r['balance_score'] = norm_params * 0.3 + norm_time * 0.7  # Weight speed more

    balanced = min(results, key=lambda x: x.get('balance_score', float('inf')))
    print(f"⚖️  Best Balance: {balanced['name']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
