"""Compare stabilized sLSTM recurrence with released xLSTM vanilla function.

Run: python verify_official.py --official-repo /path/to/P-sLSTM
Patch reconstruction and the same-length head are task adapters, not the
paper's fixed-horizon flatten head.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import torch
from einops import rearrange
from torch.nn.functional import logsigmoid

from ts_block import PsLSTMBlock, SLSTMCell


def main(repo: Path):
    path = repo / 'models/xlstm/blocks/slstm/src/vanilla/slstm.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'slstm_forward_pointwise')
    env = dict(torch=torch, logsigmoid=logsigmoid)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
         env)
    reference = env['slstm_forward_pointwise']
    torch.manual_seed(7)
    cell = SLSTMCell(8, heads=2)
    B, D, H = 3, 8, 2
    x = torch.randn(B, D, requires_grad=True)
    x_conv = torch.randn(B, D, requires_grad=True)
    h = torch.randn(B, D, requires_grad=True)
    c = torch.randn(B, D, requires_grad=True)
    n = torch.rand(B, D, requires_grad=True)
    m = torch.randn(B, D, requires_grad=True)
    ours, _ = cell(x, x_conv, (h, c, n, m))

    def gates(proj, value):
        raw = proj(value.unsqueeze(-1)).squeeze(-1)
        # Grouped Conv emits [head, gate, head_channel]; reference uses gate-major.
        return rearrange(raw, "b (h g d) -> b (g h d)", h=H, g=2, d=D // H)

    Wx = torch.cat((gates(cell.input_conv, x_conv), gates(cell.input_raw, x)),
                   -1)
    Ry = cell.recurrent(h.unsqueeze(-1)).squeeze(-1)
    Ry = rearrange(Ry, "b (h g d) -> b (g h d)", h=H, g=4, d=D // H)
    expected, _ = reference(Wx, Ry, cell.bias, torch.stack((h, c, n, m)), {})
    torch.testing.assert_close(ours, expected[0], rtol=2e-5, atol=2e-5)
    ga = torch.autograd.grad(ours.square().mean(), (x, x_conv, h, c, n, m),
                             retain_graph=True)
    gb = torch.autograd.grad(expected[0].square().mean(),
                             (x, x_conv, h, c, n, m))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    print('P-sLSTM cell: official forward/gradient parity PASS')

    # Load the released stack with its pure-PyTorch vanilla backend.  The
    # horizon forecast head is intentionally replaced by our same-length
    # patch readout; compare every operation up to the stack output instead.
    sys.path.insert(0, str(repo))
    from models.xlstm.blocks.slstm.block import sLSTMBlockConfig
    from models.xlstm.blocks.slstm.layer import sLSTMLayerConfig
    from models.xlstm.components.feedforward import FeedForwardConfig
    from models.xlstm.xlstm_block_stack import (xLSTMBlockStack,
                                                xLSTMBlockStackConfig)

    ours_block = PsLSTMBlock(3, 3, patch_size=4, stride=4,
                             d_model=8, heads=2, depth=2)
    config = xLSTMBlockStackConfig(
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(num_heads=2, conv1d_kernel_size=4,
                                   backend='vanilla', dtype='float32',
                                   dtype_b='float32'),
            feedforward=FeedForwardConfig()),
        embedding_dim=8, num_blocks=2, context_length=4, slstm_at='all')
    official_stack = xLSTMBlockStack(config)
    with torch.no_grad():
        for source, destination in zip(ours_block.layers,
                                       official_stack.blocks):
            destination.xlstm_norm.weight.copy_(source.norm1.weight - 1)
            destination.xlstm.conv1d.conv.load_state_dict(
                source.preconv.state_dict())
            dim_per_head = source.cell.dim // source.heads
            conv_gates = rearrange(source.cell.input_conv.weight[..., 0],
                                   '(h g d) i -> h g d i',
                                   h=source.heads, g=2, d=dim_per_head)
            raw_gates = rearrange(source.cell.input_raw.weight[..., 0],
                                  '(h g d) i -> h g d i',
                                  h=source.heads, g=2, d=dim_per_head)
            destination.xlstm.fgate.weight.copy_(conv_gates[:, 0])
            destination.xlstm.igate.weight.copy_(conv_gates[:, 1])
            destination.xlstm.zgate.weight.copy_(raw_gates[:, 0])
            destination.xlstm.ogate.weight.copy_(raw_gates[:, 1])
            cell = destination.xlstm.slstm_cell
            cell._recurrent_kernel_.copy_(rearrange(
                source.cell.recurrent.weight[..., 0],
                '(h g d) i -> h (g d) i', h=source.heads, g=4,
                d=dim_per_head))
            cell._bias_.copy_(source.cell.bias)
            destination.xlstm.group_norm.weight.copy_(
                source.group_weight - 1)
            destination.ffn_norm.weight.copy_(source.norm2.weight - 1)
            destination.ffn.proj_up.load_state_dict(
                source.ffn_up.state_dict())
            destination.ffn.proj_down.load_state_dict(
                source.ffn_down.state_dict())
        official_stack.post_blocks_norm.weight.copy_(
            ours_block.post_norm.weight - 1)
    sample = torch.randn(2, 4, 8, requires_grad=True)
    actual = sample
    for layer in ours_block.layers:
        actual = layer(actual)
    actual = ours_block.post_norm(actual)
    expected = official_stack(sample)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    captured = []
    hook = ours_block.post_norm.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    try:
        raw = torch.randn(2, 16, 3)
        ours_block(raw)
    finally:
        hook.remove()
    patches = rearrange(raw, 'b t c -> (b c) t').unfold(-1, 4, 4)
    torch.testing.assert_close(
        captured[0], official_stack(ours_block.embed(patches)),
        rtol=2e-5, atol=2e-5)
    print('P-sLSTM complete xLSTM stack: official forward parity PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.official_repo)
