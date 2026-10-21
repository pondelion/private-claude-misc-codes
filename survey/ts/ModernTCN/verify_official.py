"""Compare one reusable unit with the released ModernTCN Block.

Run: python verify_official.py --official-repo /path/to/ModernTCN
Only the paper's one-resolution Block is compared; our stem/readout adapt the
multi-stage fixed-horizon model to a same-length reusable module.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ts_block import ModernTCNBlock, ModernTCNUnit


def _load_block(path: Path):
    names = {
        'get_conv1d', 'get_bn', 'conv_bn', 'fuse_bn', 'ReparamLargeKernelConv',
        'Block'
    }
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in nodes} != names:
        raise RuntimeError('official Block source changed')
    env = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),
         env)
    return env['Block']


def main(repo: Path):
    path = repo / 'ModernTCN-Long-term-forecasting/models/ModernTCN.py'
    Official = _load_block(path)
    torch.manual_seed(19)
    ours = ModernTCNUnit(3, 8, 2, 15, 3, 0.0)
    theirs = Official(large_size=15,
                      small_size=3,
                      dmodel=8,
                      dff=16,
                      nvars=3,
                      drop=0.0)
    pairs = [(ours.dw.large[0], theirs.dw.lkb_origin.conv),
             (ours.dw.large[1], theirs.dw.lkb_origin.bn),
             (ours.dw.small[0], theirs.dw.small_conv.conv),
             (ours.dw.small[1], theirs.dw.small_conv.bn),
             (ours.bn, theirs.norm), (ours.ffn_var[0], theirs.ffn1pw1),
             (ours.ffn_var[3], theirs.ffn1pw2),
             (ours.ffn_cross[0], theirs.ffn2pw1),
             (ours.ffn_cross[3], theirs.ffn2pw2)]
    for a, b in pairs:
        a.load_state_dict(b.state_dict())
    x = torch.randn(2, 3, 8, 23, requires_grad=True)
    a, b = ours(x), theirs(x)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    ga = torch.autograd.grad(a.square().mean(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().mean(), x)[0]
    torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-5)
    print('ModernTCN unit: official forward/gradient parity PASS')

    # Compare the full reusable backbone, not only an isolated Block. The
    # official forecast head is excluded because it maps patches to a fixed
    # future horizon; our adapter maps those features back to input time.
    tree = ast.parse(path.read_text())
    model_node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == 'ModernTCN')
    method = next(n for n in model_node.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'forward_feature')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path),
                 'exec'), namespace)
    backbone = ModernTCNBlock(3, 3, d_model=8, patch_size=8, patch_stride=4,
                              stage_dims=(8, 12), stage_depths=(1, 1),
                              large_kernel=15, small_kernel=3,
                              dropout=0.0)
    official_stages = []
    for stage_index, stage in enumerate(backbone.stages):
        official_stage = []
        for unit in stage:
            peer = Official(large_size=15, small_size=3,
                            dmodel=unit.dim, dff=unit.dim * 2,
                            nvars=3, drop=0.0)
            module_pairs = [
                (unit.dw.large[0], peer.dw.lkb_origin.conv),
                (unit.dw.large[1], peer.dw.lkb_origin.bn),
                (unit.dw.small[0], peer.dw.small_conv.conv),
                (unit.dw.small[1], peer.dw.small_conv.bn),
                (unit.bn, peer.norm),
                (unit.ffn_var[0], peer.ffn1pw1),
                (unit.ffn_var[3], peer.ffn1pw2),
                (unit.ffn_cross[0], peer.ffn2pw1),
                (unit.ffn_cross[3], peer.ffn2pw2),
            ]
            for source, destination in module_pairs:
                destination.load_state_dict(source.state_dict())
            official_stage.append(peer)
        official_stages.append(nn.Sequential(*official_stage))
    reference_model = SimpleNamespace(
        num_stage=2, patch_size=8, patch_stride=4,
        downsample_ratio=2,
        downsample_layers=[nn.Sequential(backbone.stem, backbone.stem_bn),
                           backbone.downsamples[0]],
        stages=official_stages,
    )
    # The final unit output is the exact core boundary before our readout.
    for length in (16, 17, 23, 31):
        captured = []
        hook = backbone.stages[-1][-1].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output))
        try:
            sample = torch.randn(2, length, 3)
            backbone(sample)
        finally:
            hook.remove()
        reference = namespace['forward_feature'](
            reference_model, sample.transpose(1, 2))
        torch.testing.assert_close(captured[0], reference,
                                   rtol=2e-5, atol=2e-5)
    print('ModernTCN stem-to-final-stage official parity PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.official_repo)
