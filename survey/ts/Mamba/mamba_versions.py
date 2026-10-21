"""Readable PyTorch reference mixers for official Mamba-1, -2 and -3.

These are *not* replacements for the official fused GPU kernels in a throughput
benchmark.  They follow the official projection order and recurrent equations,
and keep every time-step in the autograd graph.  Axes: B=batch, T=time,
H=heads, P=head width, N=state width, G=B/C groups, R=MIMO rank.
"""
from __future__ import annotations

import math
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _rms(x: torch.Tensor,
         weight: torch.Tensor,
         eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm over last axis, with float32 statistics and learned scale."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) *
            weight.float()).to(x.dtype)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class TorchMamba(nn.Module):
    """Official Mamba-1 recurrence: [B,T,D] -> [B,T,D], causal in T."""

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: int | None = None):
        super().__init__()
        if min(d_model, d_state, d_conv, expand) < 1:
            raise ValueError("all dimensions must be positive")
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.inner = d_model * expand
        rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank
        if rank < 1:
            raise ValueError("dt_rank must be positive")
        # Official projection order: signal x and gate z, each [B,T,inner].
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner,
                              self.inner,
                              d_conv,
                              groups=self.inner,
                              padding=d_conv - 1,
                              bias=True)
        self.x_proj = nn.Linear(self.inner, rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(rank, self.inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -rank**-0.5, rank**-0.5)
        dt = torch.exp(
            torch.empty(self.inner).uniform_(math.log(1e-3), math.log(1e-1)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(_inverse_softplus(dt))
        self.A_log = nn.Parameter(
            torch.arange(1, d_state + 1).float().log().expand(self.inner,
                                                              -1).clone())
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model or x.size(1) < 1:
            raise ValueError(
                f"expected [B,T,{self.d_model}], got {tuple(x.shape)}")
        batch, time, _ = x.shape
        signal, gate = self.in_proj(x).chunk(2, dim=-1)  # each [B,T,inner]
        # Causal depthwise convolution preserves [B,T,inner].
        signal = F.silu(
            self.conv(signal.transpose(1, 2))[..., :time].transpose(1, 2))
        dt_raw, b_t, c_t = self.x_proj(signal).split(
            [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(
            self.dt_proj(dt_raw).float())  # [B,T,H], positive step
        a = -self.A_log.float().exp()  # [H,N], stable continuous A
        state = torch.zeros(batch,
                            self.inner,
                            self.d_state,
                            device=x.device,
                            dtype=torch.float32)
        outputs = []
        # Exact zero-order-hold A and Euler B discretization. State is
        # [B,inner,N]; B_t/C_t [B,N] broadcast across inner channels.
        for t in range(time):
            step = delta[:, t, :, None]  # [B,H,1]
            state = (step * a).exp() * state + step * b_t[:, t, None, :].float(
            ) * signal[:, t, :, None].float()
            y_t = (state * c_t[:, t, None, :].float()).sum(-1)
            y_t = y_t + self.D.float() * signal[:, t].float()
            outputs.append((y_t.to(x.dtype) * F.silu(gate[:, t])).unsqueeze(1))
        return self.out_proj(torch.cat(outputs, dim=1))


class TorchMamba2(nn.Module):
    """Official Mamba2Simple architecture, recurrent reference [B,T,D]→[B,T,D]."""

    def __init__(self,
                 d_model: int,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 128,
                 ngroups: int = 1,
                 dt_limit: tuple[float, float] = (0.0, float('inf')),
                 learnable_init_states: bool = False):
        super().__init__()
        if min(d_model, d_state, d_conv, expand, headdim, ngroups) < 1:
            raise ValueError('dimensions must be positive')
        self.d_model, self.d_state, self.headdim, self.ngroups = d_model, d_state, headdim, ngroups
        self.inner = d_model * expand
        if self.inner % headdim or (self.inner // headdim) % ngroups:
            raise ValueError(
                'expand*d_model must be divisible by headdim*ngroups')
        self.nheads = self.inner // headdim
        if dt_limit[0] < 0 or dt_limit[0] > dt_limit[1]:
            raise ValueError('invalid dt_limit')
        self.dt_limit = dt_limit
        self.in_proj = nn.Linear(d_model,
                                 2 * self.inner + 2 * ngroups * d_state +
                                 self.nheads,
                                 bias=False)
        conv_dim = self.inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim,
                                conv_dim,
                                d_conv,
                                padding=d_conv - 1,
                                groups=conv_dim,
                                bias=True)
        dt = torch.exp(
            torch.empty(self.nheads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.A_log = nn.Parameter(
            torch.empty(self.nheads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm_weight = nn.Parameter(torch.ones(self.inner))
        self.init_states = nn.Parameter(
            torch.zeros(self.nheads, headdim,
                        d_state)) if learnable_init_states else None
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 3 or u.shape[-1] != self.d_model or u.shape[1] < 1:
            raise ValueError(f'expected [B,T,{self.d_model}] with T>0')
        batch, time, _ = u.shape
        z, xbc, dt_raw = self.in_proj(u).split([
            self.inner, self.inner + 2 * self.ngroups * self.d_state,
            self.nheads
        ], -1)
        # z: [B,T,inner], xbc: [B,T,inner+2*G*N], dt_raw: [B,T,H].
        # The convolution sees x, B and C jointly; z and dt bypass it.
        xbc = F.silu(
            self.conv1d(xbc.transpose(1, 2))[..., :time].transpose(1, 2))
        x, b, c = xbc.split([
            self.inner, self.ngroups * self.d_state,
            self.ngroups * self.d_state
        ], -1)
        # [B,T,H*P] -> [B,T,H,P]; B/C have G independent state groups.
        x = rearrange(x, "b t (h p) -> b t h p", p=self.headdim)
        b = rearrange(b, "b t (g n) -> b t g n", g=self.ngroups)
        c = rearrange(c, "b t (g n) -> b t g n", g=self.ngroups)
        dt = F.softplus(dt_raw.float() +
                        self.dt_bias.float()).clamp(*self.dt_limit)
        a = -self.A_log.float().exp()
        state = (torch.zeros(
            batch, self.nheads, self.headdim, self.d_state, device=u.device)
                 if self.init_states is None else
                 self.init_states.float().unsqueeze(0).expand(
                     batch, -1, -1, -1))
        outputs = []
        for t in range(time):
            # Repeat each B/C group over its assigned contiguous heads (GQA layout).
            bt = b[:, t].repeat_interleave(self.nheads // self.ngroups,
                                           dim=1).float()  # [B,H,N]
            ct = c[:, t].repeat_interleave(self.nheads // self.ngroups,
                                           dim=1).float()  # [B,H,N]
            xt = x[:, t].float()  # [B,H,P]
            step = dt[:, t]  # [B,H]
            # State [B,H,P,N] decays per head and receives outer product x*B.
            decay = torch.exp(step * a).unsqueeze(-1).unsqueeze(-1)
            state = state * decay + (step[..., None, None] * xt[..., None] *
                                     bt[:, :, None, :])
            yt = (state * ct[:, :, None, :]
                  ).sum(-1) + self.D.float()[None, :, None] * xt
            outputs.append(yt)
        # Concatenate the H heads back to the model's inner feature axis.
        y = rearrange(torch.stack(outputs, 1), "b t h p -> b t (h p)")
        # Mamba2Simple has norm_before_gate=False: RMSNorm(y * SiLU(z)).
        y = _rms(y * F.silu(z.float()), self.norm_weight).to(u.dtype)
        return self.out_proj(y)


def _rotate_pairs(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply cumulative-angle RoPE to adjacent state-coordinate pairs."""
    even, odd = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((even * theta.cos() - odd * theta.sin(),
                       even * theta.sin() + odd * theta.cos()), -1)
    return rearrange(rot, "... pair component -> ... (pair component)")


def _heavy_tail(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


class TorchMamba3(nn.Module):
    """Mamba-3 SISO/MIMO reference: data-dependent A, rotary Q/K, trapezoidal scan."""

    def __init__(self,
                 d_model: int,
                 d_state: int = 128,
                 expand: int = 2,
                 headdim: int = 64,
                 ngroups: int = 1,
                 rope_fraction: float = 0.5,
                 is_mimo: bool = False,
                 mimo_rank: int = 4,
                 is_outproj_norm: bool = False,
                 A_floor: float = 1e-4):
        super().__init__()
        if min(d_model, d_state, expand, headdim, ngroups, mimo_rank) < 1:
            raise ValueError('dimensions must be positive')
        if rope_fraction not in (
                0.5, 1.0) or d_state % 2 or d_state * rope_fraction < 2:
            raise ValueError(
                'd_state must be even with at least one rotary pair')
        self.d_model, self.d_state, self.headdim, self.ngroups = d_model, d_state, headdim, ngroups
        self.inner = d_model * expand
        if self.inner % headdim or (self.inner // headdim) % ngroups:
            raise ValueError(
                'expand*d_model must be divisible by headdim*ngroups')
        self.nheads = self.inner // headdim
        self.rank = mimo_rank if is_mimo else 1
        self.is_mimo, self.is_outproj_norm, self.A_floor = is_mimo, is_outproj_norm, A_floor
        self.nangles = int(d_state * rope_fraction) // 2
        size = 2 * self.inner + 2 * d_state * ngroups * self.rank + 3 * self.nheads + self.nangles
        self.in_proj = nn.Linear(d_model, size, bias=False)
        dt = torch.exp(
            torch.empty(self.nheads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.rank, d_state))
        self.B_norm_weight = nn.Parameter(torch.ones(d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(self.nheads))
        if is_mimo:
            self.mimo_x = nn.Parameter(
                torch.full((self.nheads, self.rank, headdim), 1 / self.rank))
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.rank, headdim))
            self.mimo_o = nn.Parameter(
                torch.full((self.nheads, self.rank, headdim), 1 / self.rank))
        if is_outproj_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 3 or u.shape[-1] != self.d_model or u.shape[1] < 1:
            raise ValueError(f'expected [B,T,{self.d_model}] with T>0')
        batch, time, _ = u.shape
        parts = [
            self.inner, self.inner, self.rank * self.ngroups * self.d_state,
            self.rank * self.ngroups * self.d_state, self.nheads, self.nheads,
            self.nheads, self.nangles
        ]
        z, x, b, c, dd_dt, dd_a, trap, angles = self.in_proj(u).split(
            parts, -1)
        # One projection supplies gate/value, rank-grouped B/C, and per-head
        # step, decay, trapezoid weight, and rotary increments.
        # [B,T,H*P] -> [B,T,H,P]; B/C add rank R and group G axes.
        x = rearrange(x, "b t (h p) -> b t h p", p=self.headdim)
        z = rearrange(z, "b t (h p) -> b t h p", p=self.headdim)
        b = _rms(
            rearrange(b,
                      "b t (r g n) -> b t r g n",
                      r=self.rank,
                      g=self.ngroups), self.B_norm_weight)
        c = _rms(
            rearrange(c,
                      "b t (r g n) -> b t r g n",
                      r=self.rank,
                      g=self.ngroups), self.C_norm_weight)
        # B/C are normalized *before* group broadcast and bias, as in Mamba3.forward.
        b = b.repeat_interleave(self.nheads // self.ngroups, dim=3)
        c = c.repeat_interleave(self.nheads // self.ngroups, dim=3)
        dt = F.softplus(dd_dt.float() + self.dt_bias.float())  # [B,T,H]
        a = -_heavy_tail(dd_a.float()).clamp_min(self.A_floor)
        angle_state = torch.zeros(batch,
                                  self.nheads,
                                  self.nangles,
                                  device=u.device)
        state = torch.zeros(batch,
                            self.nheads,
                            self.headdim,
                            self.d_state,
                            device=u.device)
        old_k = torch.zeros(batch,
                            self.nheads,
                            self.rank,
                            self.d_state,
                            device=u.device)
        old_v = torch.zeros(batch,
                            self.nheads,
                            self.rank,
                            self.headdim,
                            device=u.device)
        outputs = []
        for t in range(time):
            step = dt[:, t]  # [B,H]
            # Integrate per-head angle increments, then rotate Q/K [B,H,R,N].
            angle_state = torch.remainder(
                angle_state + torch.tanh(angles[:, t].float())[:, None, :] *
                math.pi * step[..., None], 2 * math.pi)
            theta = F.pad(angle_state, (0, self.d_state // 2 - self.nangles))
            qt = c[:, t].float().permute(0, 2, 1,
                                         3) + self.C_bias.float()[None]
            kt = b[:, t].float().permute(0, 2, 1,
                                         3) + self.B_bias.float()[None]
            q = _rotate_pairs(qt, theta[:, :, None, :])  # [B,H,R,N]
            k = _rotate_pairs(kt, theta[:, :, None, :])
            v = x[:, t].float()[:, :, None, :] * (self.mimo_x.float()[None]
                                                  if self.is_mimo else 1)
            gate = z[:, t].float()[:, :, None, :] * (self.mimo_z.float()[None]
                                                     if self.is_mimo else 1)
            alpha = torch.exp(a[:, t] * step)
            mix = torch.sigmoid(trap[:, t].float())
            # Trapezoidal update combines previous and current K⊗V; the
            # recurrent state and both outer products have shape [B,H,P,N].
            weight_old = (1 - mix) * step * alpha
            weight_new = mix * step
            prev_kv = torch.einsum('bhrn,bhrp->bhpn', old_k, old_v)
            curr_kv = torch.einsum('bhrn,bhrp->bhpn', k, v)
            state = alpha[..., None, None] * state + weight_old[
                ..., None, None] * prev_kv + weight_new[..., None,
                                                        None] * curr_kv
            y = torch.einsum('bhpn,bhrn->bhrp', state,
                             q) + self.D.float()[None, :, None, None] * v
            if self.is_outproj_norm:
                # Official norm_before_gate=True, per head group for SISO and MIMO.
                weight = rearrange(self.norm_weight,
                                   "(h p) -> h p",
                                   p=self.headdim)
                y = _rms(y, weight[None, :, None, :]) * F.silu(gate)
            else:
                y = y * F.silu(gate)
            y = (y * self.mimo_o.float()[None]
                 ).sum(2) if self.is_mimo else y.squeeze(2)
            outputs.append(y)
            old_k, old_v = k, v
        # [B,T,H,P] -> [B,T,H*P] before the output projection.
        y = rearrange(torch.stack(outputs, 1),
                      "b t h p -> b t (h p)").to(u.dtype)
        return self.out_proj(y)
