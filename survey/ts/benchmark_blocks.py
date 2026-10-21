"""Measure real forward/backward latency for the five reusable blocks.

Examples:
  python benchmark_blocks.py ModernTCN --device cpu --batch 32 --length 256
  python benchmark_blocks.py Mamba --device cuda --batch 64 --length 512

No paper benchmark or accuracy claim is implied by synthetic timing.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch


def load_model(name: str,
               in_dim: int,
               out_dim: int,
               backend: str,
               version: int = 1,
               mimo: bool = False,
               length: int = 128):
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("selected_block",
                                                  root / name / "ts_block.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / name))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    if name == "PsLSTM":
        return module.PsLSTMBlock(in_dim, out_dim, patch_size=16, stride=16)
    if name == "Mamba":
        return module.MambaBlock(in_dim,
                                 out_dim,
                                 backend=backend,
                                 version=version,
                                 is_mimo=mimo)
    if name == "DSTMamba":
        return module.DSTMambaBlock(in_dim,
                                    out_dim,
                                    seq_len=length,
                                    ds_layers=0 if length < 4 else 2,
                                    backend=backend)
    if name == "WaveTS":
        return module.WaveTSBlock(in_dim, out_dim, seq_len=length)
    return module.ModernTCNBlock(in_dim, out_dim)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, x, warmup: int, iters: int, training: bool):
    device = x.device
    elapsed = []
    model.train(training)
    for index in range(warmup + iters):
        if training:
            model.zero_grad(set_to_none=True)
        synchronize(device)
        start = time.perf_counter()
        if training:
            y = model(x)
            y.square().mean().backward()
        else:
            with torch.inference_mode():
                y = model(x)
        synchronize(device)
        if index >= warmup:
            elapsed.append((time.perf_counter() - start) * 1000)
    return statistics.median(elapsed), min(elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["PsLSTM", "Mamba", "DSTMamba", "WaveTS", "ModernTCN"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--backend",
                        choices=["auto", "official", "torch"],
                        default="auto")
    parser.add_argument("--version",
                        type=int,
                        choices=(1, 2, 3),
                        default=1,
                        help="Mamba only: mixer architecture version")
    parser.add_argument("--mimo",
                        action="store_true",
                        help="Mamba-3 MIMO variant")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--in-dim", type=int, default=8)
    parser.add_argument("--out-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.length, args.in_dim, args.out_dim,
           args.iters) < 1 or args.warmup < 0:
        parser.error(
            "batch/length/dim/iters must be positive and warmup nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this environment")
    device = torch.device(args.device)
    model = load_model(args.model, args.in_dim, args.out_dim, args.backend,
                       args.version, args.mimo, args.length).to(device)
    x = torch.randn(args.batch,
                    args.length,
                    args.in_dim,
                    device=device,
                    requires_grad=True)
    params = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    infer_median, infer_min = measure(model,
                                      x,
                                      args.warmup,
                                      args.iters,
                                      training=False)
    train_median, train_min = measure(model,
                                      x,
                                      args.warmup,
                                      args.iters,
                                      training=True)
    version = f" version={args.version} mimo={args.mimo}" if args.model == "Mamba" else ""
    print(
        f"model={args.model}{version} device={device} shape={tuple(x.shape)} params={params:,}"
    )
    print(f"inference_ms median={infer_median:.3f} min={infer_min:.3f}")
    print(f"forward_backward_ms median={train_median:.3f} min={train_min:.3f}")
    if device.type == "cuda":
        print(
            f"cuda_peak_allocated_mb={torch.cuda.max_memory_allocated(device)/1024**2:.1f}"
        )


if __name__ == "__main__":
    main()
