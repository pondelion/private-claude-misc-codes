"""DST-Mamba fixed-window, same-length adapter built from its released flow.

The original forecasts H future values from L history values.  This block
sets H=L and optionally maps the N sensor outputs to Cout features.  L must
be chosen at construction because embedding, multi-scale MLP and projector
are learned matrices with length-dependent shapes in the official model.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class _TorchSelectiveMixer(nn.Module):
    """Mamba-1 node-axis scan, [B,N,D] -> [B,N,D]."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.inner, self.d_state = d_model * expand, d_state
        rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner,
                              self.inner,
                              d_conv,
                              padding=d_conv - 1,
                              groups=self.inner)
        self.x_proj = nn.Linear(self.inner, rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(rank, self.inner)
        dt = torch.exp(
            torch.empty(self.inner).uniform_(math.log(1e-3), math.log(1e-1)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(
            torch.arange(1, d_state + 1).float().log().expand(self.inner,
                                                              -1).clone())
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        # Node-axis selective scan: [B,N,D] -> signal/gate [B,N,D*expand].
        signal, gate = self.in_proj(x).chunk(2, -1)
        signal = F.silu(
            self.conv(signal.transpose(1, 2))[..., :nodes].transpose(1, 2))
        raw, b, c = self.x_proj(signal).split(
            [self.dt_proj.in_features, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(raw).float())
        a = -self.A_log.float().exp()
        state = torch.zeros(batch,
                            self.inner,
                            self.d_state,
                            device=x.device,
                            dtype=torch.float32)
        outputs = []
        for t in range(nodes):
            # State [B,inner,state_dim]; one node is consumed per scan step.
            delta = dt[:, t, :, None]
            state = (delta * a).exp() * state + delta * b[:, t, None, :].float(
            ) * signal[:, t, :, None].float()
            read = (state * c[:, t, None, :].float()
                    ).sum(-1) + self.D.float() * signal[:, t].float()
            outputs.append(
                (read.to(x.dtype) * F.silu(gate[:, t])).unsqueeze(1))
        return self.out_proj(torch.cat(outputs, 1))


def _make_mamba(dim: int, state: int, conv: int, expand: int, backend: str):
    if backend == 'official' or (backend == 'auto'
                                 and torch.cuda.is_available()):
        try:
            from mamba_ssm import Mamba
            return Mamba(d_model=dim,
                         d_state=state,
                         d_conv=conv,
                         expand=expand)
        except ImportError:
            if backend == 'official':
                raise ImportError(
                    "backend='official' requires mamba_ssm") from None
    return _TorchSelectiveMixer(dim, state, conv, expand)


class _BiNodeLayer(nn.Module):
    """Exact EncoderLayer order: two scans -> post-LN -> FFN -> residual/LN."""

    def __init__(self, dim: int, state: int, conv: int, expand: int, d_ff: int,
                 dropout: float, backend: str):
        super().__init__()
        self.forward_scan = _make_mamba(dim, state, conv, expand, backend)
        self.reverse_scan = _make_mamba(dim, state, conv, expand, backend)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both scans consume [B,N,D]; reversing N supplies opposite context.
        spatial = self.forward_scan(x) + self.reverse_scan(x.flip(1)).flip(1)
        encoded = self.norm1(spatial)
        # Pointwise FFN uses Conv1d layout [B,D,N], then returns [B,N,D].
        ff = self.dropout(F.gelu(self.conv1(encoded.transpose(1, 2))))
        ff = self.dropout(self.conv2(ff).transpose(1, 2))
        return self.norm2(encoded + ff)


def _moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Official SeriesDec edge replication; [B,T,N] -> [B,T,N]."""
    half = (kernel - 1) // 2
    padded = F.pad(x.transpose(1, 2), (half, half), mode='replicate')
    return F.avg_pool1d(padded, kernel, stride=1).transpose(1, 2)


class DSTMambaBlock(nn.Module):
    """[B,seq_len,N] -> [B,seq_len,Cout], N=in_dim fixed sensor nodes.

    Uses fixed-length learned time maps, official coarse-to-fine trend MLP,
    bidirectional node Mamba, per-node trend weighting, and RevIN. The only
    architectural adapter is setting forecast H=L and mapping N->Cout.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int = 96,
                 d_model: int = 64,
                 node_dim: int = 16,
                 rank: int = 4,
                 trend_kernel: int = 7,
                 ds_layers: int = 2,
                 ds_window: int = 2,
                 ds_type: str = 'avg',
                 depth: int = 1,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 d_ff: int | None = None,
                 dropout: float = 0.0,
                 backend: str = 'auto',
                 initial_trend_weight: float = 1.0):
        super().__init__()
        if min(in_dim, out_dim, seq_len, d_model, node_dim, rank, trend_kernel,
               ds_window, depth, d_state, d_conv, expand) < 1:
            raise ValueError('dimensions must be positive')
        if node_dim >= d_model or trend_kernel % 2 != 1 or ds_layers < 0 or backend not in {
                'auto', 'official', 'torch'
        } or ds_type not in {'avg', 'max'}:
            raise ValueError(
                'invalid node_dim, trend_kernel, ds_layers, ds_type or backend'
            )
        lengths = [seq_len // (ds_window**i) for i in range(ds_layers + 1)]
        if min(lengths) < 1:
            raise ValueError('seq_len too short for ds_layers and ds_window')
        self.in_dim, self.out_dim, self.seq_len = in_dim, out_dim, seq_len
        self.trend_kernel = trend_kernel
        self.down_pool = (nn.AvgPool1d(ds_window, stride=ds_window) if ds_type
                          == 'avg' else nn.MaxPool1d(ds_window,
                                                     stride=ds_window))
        embed_dim = d_model - node_dim
        self.embedding = nn.Linear(seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.adapter = nn.Parameter(torch.empty(in_dim, embed_dim, rank))
        nn.init.xavier_uniform_(self.adapter)
        self.lora = nn.Linear(rank, node_dim, bias=False)
        self.layers = nn.ModuleList(
            _BiNodeLayer(d_model, d_state, d_conv, expand, d_ff or 2 *
                         d_model, dropout, backend) for _ in range(depth))
        self.encoder_norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, seq_len)
        # SeriesMix.py traverses scales from coarsest to finest.
        self.up_sampling = nn.ModuleList(
            nn.Sequential(nn.Linear(lengths[i + 1], lengths[i]), nn.GELU(),
                          nn.Linear(lengths[i], lengths[i]))
            for i in reversed(range(ds_layers)))
        self.linear_mappings = nn.ModuleList(
            nn.Linear(n, seq_len) for n in lengths)
        self.trend_weight = nn.Parameter(
            torch.full((in_dim, ), initial_trend_weight))
        self.revin_weight = nn.Parameter(torch.ones(in_dim))
        self.revin_bias = nn.Parameter(torch.zeros(in_dim))
        # Keep a learnable node-to-output map for equal widths as well. The
        # spatial Mamba already mixes nodes; this readout never bypasses them.
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[
                2] != self.in_dim:
            raise ValueError(f'expected [B,{self.seq_len},{self.in_dim}]')
        # RevIN.py: mean and population stdev are detached per sample/node.
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) +
                         1e-5).detach()
        # RevIN statistics [B,1,N] normalize each sample and sensor separately.
        normalized = (
            x - mean) / std * self.revin_weight + self.revin_bias  # [B,L,N]
        seasonal = normalized - _moving_average(normalized, self.trend_kernel)
        base = self.embed_dropout(self.embedding(seasonal.transpose(
            1, 2)))  # [B,N,E]
        adaptation = F.relu(self.lora(self.adapter))  # [N,E,Da]
        spatial = torch.einsum('bne,ned->bnd', base, adaptation)  # [B,N,Da]
        token = torch.cat((base, spatial), -1)  # [B,N,D]
        for layer in self.layers:
            token = layer(token)
        # Spatial tokens [B,N,D] project over the fixed L-step output axis.
        sea = self.projector(self.encoder_norm(token)).transpose(1,
                                                                 2)  # [B,L,N]

        # DSTMamba_Arch: downsample the normalized *full signal* first,
        # decompose each scale separately, then mix trends coarse -> fine.
        scales = [normalized]
        current = normalized.transpose(1, 2)
        for _ in self.up_sampling:
            # Downsample time only: [B,N,L_i] -> [B,N,L_(i+1)].
            current = self.down_pool(current)
            scales.append(current.transpose(1, 2))
        trends = [
            _moving_average(s, self.trend_kernel).transpose(1, 2)
            for s in scales
        ]
        reverse = list(reversed(trends))
        coarse = reverse[0]
        mixed = [coarse]
        for finer, up in zip(reverse[1:], self.up_sampling):
            # Learned coarse-to-fine trend fusion, each [B,N,L_i].
            coarse = finer + up(coarse)
            mixed.append(coarse)
        mixed.reverse()
        projected = [
            mapping(t).transpose(1, 2)
            for mapping, t in zip(self.linear_mappings, mixed)
        ]
        trend = torch.stack(projected, -1).sum(-1)  # [B,L,N]
        pred = sea + self.trend_weight[None, None, :] * trend
        # RevIN denormalization happens before optional N->Cout projection.
        pred = (pred - self.revin_bias) / (self.revin_weight +
                                           1e-10) * std + mean
        return self.output(pred)


if __name__ == '__main__':
    model = DSTMambaBlock(4,
                          6,
                          seq_len=16,
                          d_model=16,
                          node_dim=4,
                          ds_layers=2,
                          d_state=4,
                          backend='torch')
    x = torch.randn(2, 16, 4, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == (2, 16, 6) and x.grad is not None
