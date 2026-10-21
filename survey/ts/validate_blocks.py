"""Dependency-light numerical smoke tests for all five reusable blocks.

Run: python validate_blocks.py. This does not benchmark paper accuracy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_block",
                                                  ROOT / name / "ts_block.py")
    module = importlib.util.module_from_spec(spec)
    # Mamba keeps its independently reusable v2/v3 mixers in a sibling file.
    sys.path.insert(0, str(ROOT / name))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def check_shape_grad(name, model):
    for length in (1, 7, 19, 32):
        x = torch.randn(2, length, 3, requires_grad=True)
        y = model(x)
        assert y.shape == (2, length, 5), (name, length, y.shape)
        assert torch.isfinite(y).all(), (name, length)
        model.zero_grad(set_to_none=True)
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), (name,
                                                                     length)
    print(f"{name}: shape, finite, backward OK")


def main():
    torch.set_num_threads(1)
    ps = load("PsLSTM")
    check_shape_grad(
        "PsLSTM",
        ps.PsLSTMBlock(3,
                       5,
                       patch_size=8,
                       stride=4,
                       d_model=16,
                       heads=4,
                       depth=1))

    ma = load("Mamba")
    for version, mimo in ((1, False), (2, False), (3, False), (3, True)):
        label = f"Mamba-{version}" + (" MIMO" if mimo else "")
        mamba = ma.MambaBlock(3,
                              5,
                              d_model=16,
                              d_state=4,
                              backend="torch",
                              version=version,
                              is_mimo=mimo)
        check_shape_grad(label, mamba)
        mamba.eval()
        x1 = torch.randn(2, 15, 3)
        x2 = x1.clone()
        x2[:, 8:] += 10.0
        torch.testing.assert_close(mamba(x1)[:, :8], mamba(x2)[:, :8])
        print(f"{label}: causal-prefix invariance OK")

    dst = load("DSTMamba")
    # Official DST-Mamba uses fixed learned L->D and L_i->H matrices.
    for length in (1, 7, 19, 32):
        model = dst.DSTMambaBlock(3,
                                  5,
                                  seq_len=length,
                                  d_model=16,
                                  node_dim=4,
                                  ds_layers=0 if length == 1 else 2,
                                  d_state=4,
                                  backend="torch")
        x = torch.randn(2, length, 3, requires_grad=True)
        y = model(x)
        assert y.shape == (2, length, 5) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    print("DSTMamba: fixed-window shape, finite, backward OK")
    dst_equal = dst.DSTMambaBlock(3,
                                  3,
                                  seq_len=7,
                                  d_model=16,
                                  node_dim=4,
                                  ds_layers=2,
                                  d_state=4,
                                  backend="torch")
    assert isinstance(dst_equal.output, torch.nn.Linear)
    assert dst_equal(torch.randn(2, 7, 3)).shape == (2, 7, 3)

    wave = load("WaveTS")
    # WaveTS has learned per-coefficient Linear maps and therefore fixed T.
    for length in (7, 19, 32):
        model = wave.WaveTSBlock(3, 5, seq_len=length, levels=3)
        x = torch.randn(2, length, 3, requires_grad=True)
        y = model(x)
        assert y.shape == (2, length, 5) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    print("WaveTS: fixed-window shape, finite, backward OK")
    wave_equal = wave.WaveTSBlock(3, 3, seq_len=7, levels=3)
    assert isinstance(wave_equal.output, torch.nn.Linear)
    assert wave_equal(torch.randn(2, 7, 3)).shape == (2, 7, 3)
    print("DST-Mamba/WaveTS: equal-width learnable output fusion OK")
    for length in (7, 19, 32):
        x = torch.randn(2, 3, length)
        low, highs, lengths = wave._haar_forward(x, 3)
        torch.testing.assert_close(wave._haar_inverse(low, highs, lengths),
                                   x,
                                   rtol=1e-5,
                                   atol=1e-5)
    print("WaveTS: odd/even exact Haar reconstruction OK")

    mt = load("ModernTCN")
    modern = mt.ModernTCNBlock(3, 5, d_model=8, depth=1)
    check_shape_grad("ModernTCN", modern)
    modern.eval()
    with torch.inference_mode():
        x = torch.randn(2, 23, 3)
        before = modern(x)
        modern.merge_kernels()
        after = modern(x)
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    print("ModernTCN: fused inference equivalence OK")
    multistage = mt.ModernTCNBlock(3,
                                   5,
                                   d_model=8,
                                   depth=1,
                                   stage_dims=(8, 16, 32),
                                   stage_depths=(1, 1, 1))
    check_shape_grad("ModernTCN multi-stage", multistage)


if __name__ == "__main__":
    main()
