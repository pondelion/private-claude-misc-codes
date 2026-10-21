"""Compare portable mixers with independent reference functions in an official checkout.

Run: python verify_official.py --official-repo /path/to/mamba
No local checkout path is embedded in this publishable file.  Fused CUDA
kernel parity additionally needs a CUDA system with mamba_ssm installed.
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_versions import TorchMamba, TorchMamba2, TorchMamba3
from ts_block import MambaBlock


def _load_functions(path: Path, names: set[str]) -> dict:
    """Load only pure reference functions, avoiding optional Triton imports."""
    tree = ast.parse(path.read_text())
    selected = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {n.name for n in selected} != names:
        raise RuntimeError(f'missing expected reference function in {path}')
    env = dict(torch=torch,
               F=F,
               math=math,
               rearrange=rearrange,
               repeat=repeat,
               Optional=Optional,
               Tuple=Tuple)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'),
        env)
    return env


def _capture_preprojection(model, x):
    got = []
    hook = model.out_proj.register_forward_pre_hook(
        lambda _, args: got.append(args[0]))
    try:
        model(x)
    finally:
        hook.remove()
    return got[0]


def _assert_forward_backward(actual, expected, input_tensor):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    ga = torch.autograd.grad(actual.square().mean(),
                             input_tensor,
                             retain_graph=True)[0]
    ge = torch.autograd.grad(expected.square().mean(), input_tensor)[0]
    torch.testing.assert_close(ga, ge, rtol=3e-4, atol=3e-5)


def check_v1(repo: Path):
    ref = _load_functions(repo / 'mamba_ssm/ops/selective_scan_interface.py',
                          {'selective_scan_ref'})['selective_scan_ref']
    model = TorchMamba(8, d_state=4, d_conv=3)
    u = torch.randn(2, 7, 8, requires_grad=True)
    actual = _capture_preprojection(model, u)
    B, T, _ = u.shape
    x, z = model.in_proj(u).chunk(2, -1)
    x = F.silu(model.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
    raw, b, c = model.x_proj(x).split(
        [model.dt_proj.in_features, model.d_state, model.d_state], -1)
    delta = model.dt_proj(raw)
    expected = ref(x.transpose(1, 2),
                   delta.transpose(1, 2),
                   -model.A_log.exp(),
                   b.transpose(1, 2),
                   c.transpose(1, 2),
                   D=model.D,
                   z=z.transpose(1, 2),
                   delta_softplus=True).transpose(1, 2)
    _assert_forward_backward(actual, expected, u)
    torch.testing.assert_close(model(u), model.out_proj(expected),
                               rtol=2e-5, atol=2e-5)
    print('Mamba-1: official selective_scan_ref forward/gradient parity PASS')


def check_v2(repo: Path):
    funcs = _load_functions(repo / 'mamba_ssm/modules/ssd_minimal.py',
                            {'segsum', 'ssd_minimal_discrete'})
    # ssd_minimal_discrete resolves segsum via its globals, already shared.
    model = TorchMamba2(8, d_state=4, d_conv=3, headdim=4, ngroups=2)
    x = torch.randn(2, 8, 8, requires_grad=True)
    actual = _capture_preprojection(model, x)
    B, T, _ = x.shape
    z, xbc, raw_dt = model.in_proj(x).split([
        model.inner, model.inner + 2 * model.ngroups * model.d_state,
        model.nheads
    ], -1)
    xbc = F.silu(model.conv1d(xbc.transpose(1, 2))[..., :T].transpose(1, 2))
    v, k, q = xbc.split([
        model.inner, model.ngroups * model.d_state,
        model.ngroups * model.d_state
    ], -1)
    # Match the official SSD axes: H heads of width P, G B/C state groups.
    v = rearrange(v, "b t (h p) -> b t h p", p=model.headdim)
    k = rearrange(k, "b t (g n) -> b t g n",
                  g=model.ngroups).repeat_interleave(model.nheads //
                                                     model.ngroups,
                                                     dim=2)
    q = rearrange(q, "b t (g n) -> b t g n",
                  g=model.ngroups).repeat_interleave(model.nheads //
                                                     model.ngroups,
                                                     dim=2)
    dt = F.softplus(raw_dt + model.dt_bias)
    # Official SSD listing uses X*dt and A*dt to represent the recurrence.
    y, _ = funcs['ssd_minimal_discrete'](v.float() * dt[..., None],
                                         dt * (-model.A_log.exp()),
                                         k.float(),
                                         q.float(),
                                         block_len=4)
    y = y + model.D[None, None, :, None] * v
    y = rearrange(y, "b t h p -> b t (h p)") * F.silu(z)
    expected = y * torch.rsqrt(y.square().mean(-1, keepdim=True) +
                               1e-5) * model.norm_weight
    _assert_forward_backward(actual, expected, x)
    torch.testing.assert_close(model(x), model.out_proj(expected),
                               rtol=2e-5, atol=2e-5)
    print('Mamba-2: official SSD reference forward/gradient parity PASS')


def check_v3(repo: Path, mimo: bool, outproj_norm: bool = False):
    path = (repo / 'tests/ops/tilelang/test_mamba3_mimo.py' if mimo else repo /
            'tests/ops/triton/test_mamba3_siso.py')
    name = 'mamba3_MIMO_step_ref' if mimo else 'mamba3_siso_step_ref'
    ref = _load_functions(path, {name})[name]
    model = TorchMamba3(8,
                        d_state=8,
                        headdim=4,
                        ngroups=2,
                        is_mimo=mimo,
                        mimo_rank=2,
                        is_outproj_norm=outproj_norm)
    x = torch.randn(2, 7, 8, requires_grad=True)
    actual = _capture_preprojection(model, x)
    B, T, _ = x.shape
    sizes = [
        model.inner, model.inner, model.rank * model.ngroups * model.d_state,
        model.rank * model.ngroups * model.d_state, model.nheads, model.nheads,
        model.nheads, model.nangles
    ]
    z, v, k, q, ddt, dda, trap, ang = model.in_proj(x).split(sizes, -1)
    z = rearrange(z, "b t (h p) -> b t h p", p=model.headdim)
    v = rearrange(v, "b t (h p) -> b t h p", p=model.headdim)

    def normalize(t, w):
        # [B,T,R*G*N] -> [B,T,R,G,N] before state-wise RMSNorm.
        t = rearrange(t,
                      "b t (r g n) -> b t r g n",
                      r=model.rank,
                      g=model.ngroups)
        return t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-5) * w

    k = normalize(k, model.B_norm_weight)
    q = normalize(q, model.C_norm_weight)
    dt = F.softplus(ddt + model.dt_bias)
    aa = -(dda.clamp_min(0) +
           torch.reciprocal(1 - dda.clamp_max(0))).clamp_min(model.A_floor)
    angles = ang[:, :, None, :].expand(-1, -1, model.nheads, -1)
    args = dict(Q=q,
                K=k,
                V=v,
                ADT=(aa * dt).transpose(1, 2),
                DT=dt.transpose(1, 2),
                Trap=trap.transpose(1, 2),
                Q_bias=model.C_bias,
                K_bias=model.B_bias,
                Angles=angles,
                D=model.D,
                Z=z)
    if mimo:
        args.update(MIMO_V=model.mimo_x,
                    MIMO_O=model.mimo_o,
                    MIMO_Z=model.mimo_z)
        if outproj_norm:
            args.update(fused_norm=True,
                        outproj_norm_weight=rearrange(model.norm_weight,
                                                      "(h p) -> h p",
                                                      p=model.headdim))
    else:
        args['Q'] = q.squeeze(2)
        args['K'] = k.squeeze(2)
        args['Q_bias'] = model.C_bias.squeeze(1)
        args['K_bias'] = model.B_bias.squeeze(1)
        if outproj_norm:
            args['Z'] = None
    expected, _ = ref(**args)
    if outproj_norm and not mimo:
        # SISO official reference exposes raw SSM output; Mamba3.forward then
        # applies its headwise norm_before_gate=True RMSNormGated.
        w = rearrange(model.norm_weight, "(h p) -> h p", p=model.headdim)
        expected = expected * torch.rsqrt(
            expected.square().mean(-1, keepdim=True) + 1e-5) * w[None, None]
        expected = expected * F.silu(z)
    expected = rearrange(expected, "b t h p -> b t (h p)")
    _assert_forward_backward(actual, expected, x)
    torch.testing.assert_close(model(x), model.out_proj(expected),
                               rtol=2e-5, atol=2e-5)
    print(
        f'Mamba-3 {"MIMO" if mimo else "SISO"} norm={outproj_norm}: official step reference forward/gradient parity PASS'
    )


def check_generic_wrapper(repo: Path):
    """Compare the full adapter to released Block ordering plus width maps."""
    path = repo / 'mamba_ssm/modules/block.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'Block')
    env = dict(torch=torch, nn=torch.nn, Tensor=torch.Tensor,
               Optional=Optional, RMSNorm=None, layer_norm_fn=None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path),
                 'exec'), env)
    OfficialBlock = env['Block']

    class MixerCallAdapter(torch.nn.Module):
        def __init__(self, mixer):
            super().__init__()
            self.mixer = mixer

        def forward(self, hidden_states, inference_params=None, **kwargs):
            return self.mixer(hidden_states)

    for version in (1, 2, 3):
        candidate = MambaBlock(5, 7, d_model=8,
                               d_state=8 if version == 3 else 4,
                               d_conv=3, depth=2, dropout=0.0,
                               backend='torch', version=version,
                               headdim=4, ngroups=2)
        peers = []
        for norm, mixer in zip(candidate.norms, candidate.mixers):
            peer = OfficialBlock(8, mixer_cls=lambda _dim, m=mixer:
                                 MixerCallAdapter(m),
                                 mlp_cls=torch.nn.Identity,
                                 fused_add_norm=False,
                                 residual_in_fp32=False)
            peer.norm.load_state_dict(norm.state_dict())
            peers.append(peer)
        sample = torch.randn(2, 7, 5)
        hidden, residual = candidate.input(sample), None
        for peer in peers:
            hidden, residual = peer(hidden, residual)
        expected = candidate.output(hidden + residual)
        torch.testing.assert_close(candidate(sample), expected,
                                   rtol=2e-5, atol=2e-5)
    print('Mamba-1/2/3 full adapter: official Block ordering PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-repo', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(11)
    check_v1(args.official_repo)
    check_v2(args.official_repo)
    check_v3(args.official_repo, False)
    check_v3(args.official_repo, True)
    check_v3(args.official_repo, False, True)
    check_v3(args.official_repo, True, True)
    check_generic_wrapper(args.official_repo)
