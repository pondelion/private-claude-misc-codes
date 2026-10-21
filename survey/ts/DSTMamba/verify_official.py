"""FP32 checks against importable official DST-Mamba submodules.

Run: python verify_official.py --official-repo /path/to/DST-Mamba
The official complete model requires mamba_ssm; this check compares its
pure-PyTorch decomposition, trend mixer and encoder-layer ordering.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import torch

from ts_block import DSTMambaBlock, _TorchSelectiveMixer, _moving_average


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + '_official',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(repo: Path):
    root = repo / 'baselines/DSTMamba'
    series = _load(root / 'SeriesDec.py')
    mixing = _load(root / 'SeriesMix.py')
    encoder = _load(root / 'MambaEnc.py')
    torch.manual_seed(29)
    model = DSTMambaBlock(3,
                          3,
                          seq_len=16,
                          d_model=16,
                          node_dim=4,
                          ds_layers=2,
                          depth=1,
                          d_state=4,
                          backend='torch',
                          dropout=0)
    model.eval()
    x = torch.randn(2, 16, 3, requires_grad=True)
    official_dec = series.Temporal_Decomposition(model.trend_kernel)
    expected_trend = official_dec(x)[1]
    actual_trend = _moving_average(x, model.trend_kernel)
    torch.testing.assert_close(actual_trend, expected_trend)
    print('DST decomposition: official parity PASS')

    # Feed both encoder layers the *same* SSM modules so the comparison
    # isolates the residual, norm, activation and FFN ordering.
    ours = model.layers[0]
    theirs = encoder.EncoderLayer(ours.forward_scan, ours.reverse_scan, 16,
                                  ours.conv1.out_channels, 0.0, 'gelu')
    theirs.norm1.load_state_dict(ours.norm1.state_dict())
    theirs.norm2.load_state_dict(ours.norm2.state_dict())
    theirs.conv1.load_state_dict(ours.conv1.state_dict())
    theirs.conv2.load_state_dict(ours.conv2.state_dict())
    nodes = torch.randn(2, 3, 16, requires_grad=True)
    a, b = ours(nodes), theirs(nodes)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    ga = torch.autograd.grad(a.square().mean(), nodes, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().mean(), nodes)[0]
    torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-5)
    print('DST bidirectional encoder: official forward/gradient parity PASS')

    theirs_mix = mixing.MultiScaleTrendMixing(16, 16, 3, 2, 2)
    theirs_mix.up_sampling.load_state_dict(model.up_sampling.state_dict())
    trends = [torch.randn(2, n, 3) for n in (16, 8, 4)]
    expected = theirs_mix(trends)
    reverse = [t.transpose(1, 2) for t in reversed(trends)]
    low = reverse[0]
    got = [low]
    for finer, up in zip(reverse[1:], model.up_sampling):
        low = finer + up(low)
        got.append(low)
    got = [t.transpose(1, 2) for t in reversed(got)]
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    print('DST coarse-to-fine trend mixer: official parity PASS')

    # The official complete model imports the fused Mamba library. Replace
    # that *one dependency* with the same Torch mixer on both sides, then
    # compare the entire input-to-output graph, including RevIN and trends.
    stub = types.ModuleType('mamba_ssm')
    stub.Mamba = lambda d_model, d_state, d_conv, expand: _TorchSelectiveMixer(
        d_model, d_state, d_conv, expand)
    previous = sys.modules.get('mamba_ssm')
    sys.modules['mamba_ssm'] = stub
    sys.path.insert(0, str(repo / 'baselines'))
    try:
        from DSTMamba.DSTMamba_Arch import DSTMamba as OfficialDSTMamba
    finally:
        if previous is None:
            del sys.modules['mamba_ssm']
        else:
            sys.modules['mamba_ssm'] = previous
    args = dict(history_seq_len=16, future_seq_len=16, num_channels=3,
                d_model=16, use_norm=True, emb_dropout=0.0,
                decom_type='STD', std_kernel=7, rank=4, node_dim=4,
                e_layers=1, d_state=4, d_conv=4, expand=2, d_ff=32,
                ffn_dropout=0.0, ffn_activation='gelu', ds_type='avg',
                ds_layers=2, ds_window=2, initial_tre_w=1.0)
    official = OfficialDSTMamba(**args)
    candidate = DSTMambaBlock(3, 3, seq_len=16, d_model=16, node_dim=4,
                              rank=4, trend_kernel=7, ds_layers=2,
                              ds_window=2, depth=1, d_state=4, d_conv=4,
                              expand=2, d_ff=32, dropout=0.0,
                              backend='torch')
    official.revin_layer.affine_weight.data.copy_(candidate.revin_weight)
    official.revin_layer.affine_bias.data.copy_(candidate.revin_bias)
    official.embedding.ValueEmb.load_state_dict(candidate.embedding.state_dict())
    official.adapter.data.copy_(candidate.adapter)
    official.lora.load_state_dict(candidate.lora.state_dict())
    official.encoder.norm_layer.load_state_dict(
        candidate.encoder_norm.state_dict())
    for source, destination in zip(candidate.layers,
                                   official.encoder.ssm_layers):
        destination.ssm = source.forward_scan
        destination.ssm_r = source.reverse_scan
        for name in ('norm1', 'norm2', 'conv1', 'conv2'):
            getattr(destination, name).load_state_dict(
                getattr(source, name).state_dict())
    official.projector.load_state_dict(candidate.projector.state_dict())
    official.ms_mixing.up_sampling.load_state_dict(
        candidate.up_sampling.state_dict())
    official.linear_mappings.load_state_dict(
        candidate.linear_mappings.state_dict())
    official.tre_w.data.copy_(candidate.trend_weight)
    with torch.no_grad():
        candidate.output.weight.copy_(torch.eye(3))
        candidate.output.bias.zero_()
    sample = torch.randn(2, 16, 3)
    actual = candidate(sample)
    expected = official(sample[..., None])[..., 0]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    print('DST complete official forward with shared Torch Mamba PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.official_repo)
