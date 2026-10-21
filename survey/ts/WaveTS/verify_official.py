"""Check the fixed-length FRU and Haar path against official WaveTS.

Run: python verify_official.py --official-repo /path/to/WaveTS
Requires pytorch_wavelets for direct DWT/iDWT and full forward comparisons.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from ts_block import FRU, WaveTSBlock, _haar_forward, _haar_inverse


def main(repo: Path):
    path = repo / 'models/WaveTS.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'FRU')
    env = dict(math=math, torch=torch, nn=nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
         env)
    official = env['FRU'](order=2,
                          scale=3,
                          lookback_window=16,
                          predict_length=0)
    ours = FRU(16, 3)
    ours.low.load_state_dict(official.mlp_LL.state_dict())
    for a, b in zip(ours.highs, official.mlp_LH):
        a.load_state_dict(b.state_dict())
    low = torch.randn(2, 4, 2, requires_grad=True)
    highs = [torch.randn(2, 4, n, requires_grad=True) for n in (8, 4, 2)]
    a_low, a_highs = ours(low, highs)
    b_low, b_highs = official(low, highs)
    torch.testing.assert_close(a_low, b_low)
    for a, b in zip(a_highs, b_highs):
        torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(sum(z.square().mean() for z in [a_low, *a_highs]),
                             [low, *highs],
                             retain_graph=True)
    gb = torch.autograd.grad(sum(z.square().mean() for z in [b_low, *b_highs]),
                             [low, *highs])
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    print('WaveTS FRU: official forward/gradient parity PASS')

    model = WaveTSBlock(4, 4, seq_len=16, levels=3)
    x = torch.randn(2, 4, 16)
    lo, hi, lens = _haar_forward(x, 3)
    torch.testing.assert_close(_haar_inverse(lo, hi, lens),
                               x,
                               rtol=1e-5,
                               atol=1e-5)
    print('WaveTS Haar forward/inverse: reconstruction PASS')

    # Directly compare the boundary-sensitive transform. In particular, odd
    # input lengths must use reflect extension rather than edge replication.
    for length in (16, 17, 31, 32):
        sample = torch.randn(2, 4, length)
        low, highs, lengths = _haar_forward(sample, 3)
        ref_low, ref_highs = DWT1DForward(J=3, wave='db1',
                                         mode='reflect')(sample)
        torch.testing.assert_close(low, ref_low, atol=1e-6, rtol=1e-6)
        for actual_high, expected_high in zip(highs, ref_highs):
            torch.testing.assert_close(actual_high, expected_high,
                                       atol=1e-6, rtol=1e-6)
        official_inverse = DWT1DInverse(wave='db1', mode='reflect')(
            (ref_low, ref_highs))
        torch.testing.assert_close(_haar_inverse(ref_low, ref_highs,
                                                  lengths),
                                   official_inverse[..., :length],
                                   atol=1e-6, rtol=1e-6)
    print('WaveTS actual pytorch_wavelets Haar coefficients/inverse PASS')

    # First compare the shared Haar backend as a diagnostic control. Below,
    # compare again against the actual pytorch_wavelets backend.
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
               n.name in {'WaveletDerivativeTransform',
                          'InverseWaveletDerivativeTransform', 'FRU', 'Model'}]

    class SharedDWT(nn.Module):
        def __init__(self, J, wave, mode):
            super().__init__()
            self.levels = J

        def forward(self, value):
            low, details, _ = _haar_forward(value, self.levels)
            # The official FRU calls `.view`, as its wavelet backend returns
            # contiguous coefficients. Preserve that backend contract here.
            return low.contiguous(), [detail.contiguous() for detail in details]

    class SharedIDWT(nn.Module):
        def __init__(self, wave, mode):
            super().__init__()

        def forward(self, bands):
            low, details = bands
            lengths = [16 // (2 ** i) for i in range(len(details))]
            return _haar_inverse(low, details, lengths)

    namespace = dict(math=math, torch=torch, nn=nn,
                     DWT1DForward=SharedDWT, DWT1DInverse=SharedIDWT)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path),
                 'exec'), namespace)
    config = SimpleNamespace(order=2, scale=3, seq_len=16, pred_len=0,
                             enc_in=4, wavelet='db1')
    reference = namespace['Model'](config)
    with torch.no_grad():
        for source, destination in zip(model.frus, reference.OAWCNs):
            destination.mlp_LL.load_state_dict(source.low.state_dict())
            for source_high, destination_high in zip(
                    source.highs, destination.mlp_LH):
                destination_high.load_state_dict(source_high.state_dict())
        reference.projection_time.load_state_dict(
            model.projection_time.state_dict())
        model.output.weight.copy_(torch.eye(4))
        model.output.bias.zero_()
    sample = torch.randn(2, 16, 4)
    torch.testing.assert_close(model(sample), reference(sample),
                               rtol=2e-5, atol=2e-5)
    print('WaveTS full official forward with shared Haar backend PASS')

    # Repeat the end-to-end check without replacing the library backend.
    spec = importlib.util.spec_from_file_location('official_wavets', path)
    assert spec is not None and spec.loader is not None
    official_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_module)
    real_reference = official_module.Model(config)
    with torch.no_grad():
        for source, destination in zip(model.frus, real_reference.OAWCNs):
            destination.mlp_LL.load_state_dict(source.low.state_dict())
            for source_high, destination_high in zip(
                    source.highs, destination.mlp_LH):
                destination_high.load_state_dict(source_high.state_dict())
        real_reference.projection_time.load_state_dict(
            model.projection_time.state_dict())
    torch.testing.assert_close(model(sample), real_reference(sample),
                               rtol=2e-5, atol=2e-5)
    print('WaveTS full official forward with real Haar backend PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.official_repo)
