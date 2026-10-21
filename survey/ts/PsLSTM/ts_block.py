"""P-sLSTM-inspired reusable block; requires PyTorch and einops.

Patching and channel independence follow P-sLSTM.  The overlap-add head is a
length-preserving adaptation; the paper uses a fixed-horizon flattening head.
"""
from __future__ import annotations

import math
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class SLSTMCell(nn.Module):
    """Stabilized multi-head scalar LSTM; x,h: [B,D], states: [B,D]."""

    def __init__(self,
                 dim: int,
                 heads: int = 4,
                 block_idx: int = 0,
                 num_blocks: int = 1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads = dim, heads
        # Official sLSTM uses convolved patch features for input/forget gates,
        # and raw patch features for cell candidate/output gate.
        # Four input gate projections are headwise in the released xLSTM.
        # A grouped 1x1 convolution is the same block-diagonal linear map.
        self.input_conv = nn.Conv1d(dim, 2 * dim, 1, groups=heads, bias=False)
        self.input_raw = nn.Conv1d(dim, 2 * dim, 1, groups=heads, bias=False)
        # Grouped 1x1 is a block-diagonal recurrent matrix: one memory mixer/head.
        self.recurrent = nn.Conv1d(dim, 4 * dim, 1, groups=heads, bias=False)
        # Official sLSTMCell has a separate trainable gate bias (the released
        # initialization uses a block-dependent power law for forget gates).
        self.bias = nn.Parameter(torch.zeros(4 * dim))
        nn.init.zeros_(self.recurrent.weight)
        # Published configuration: powerlaw_blockdependent forget bias.
        head_dim = dim // heads
        ratio = block_idx / (num_blocks - 1) if num_blocks > 1 else 0.0
        positions = torch.arange(head_dim).float() / max(head_dim - 1, 1)
        forget = 5.0 - 12.0 * positions.pow(0.3 + 1.3 * ratio)
        with torch.no_grad():
            # Bias layout is [gate, head, channel]; gate 1 is the forget gate.
            gate_bias = rearrange(self.bias, "(g h d) -> g h d", g=4, h=heads)
            gate_bias[1].copy_(forget.expand(heads, -1))

    def forward(self, x: torch.Tensor, x_conv: torch.Tensor,
                state: tuple[torch.Tensor, ...]):
        """One patch step: x/x_conv [B,D], state four [B,D] -> same."""
        # h=hidden, c=cell, n=normalizer, m=log-domain stabilizer.
        h, c, n, m = state
        # Grouped conv emits [head, gate, head_channel]. The dense input
        # projection emits [gate, head, head_channel]; reorder before adding.
        recurrent = self.recurrent(h.unsqueeze(-1)).squeeze(-1)
        # [B,D] -> [B,D,1] -> grouped Conv [B,4D,1] -> [B,4D].
        recurrent = rearrange(recurrent,
                              "b (h g d) -> b (g h d)",
                              h=self.heads,
                              g=4,
                              d=self.dim // self.heads)

        # Grouped Conv produces [head, gate, channel], while cell gates are
        # concatenated as [gate, head, channel]. Never add in mismatched order.
        def gate_major(proj, value):
            y = proj(value.unsqueeze(-1)).squeeze(-1)
            # [B,D] -> [B,2D] in head-major gate layout.
            # Grouped convolution orders [H,G,Dh]; gates require [G,H,Dh].
            return rearrange(y,
                             "b (h g d) -> b (g h d)",
                             h=self.heads,
                             g=2,
                             d=self.dim // self.heads)

        gates = torch.cat(
            (gate_major(self.input_conv, x_conv), gate_major(
                self.input_raw, x)), -1) + recurrent + self.bias
        # Two [B,2D] inputs concatenate to [B,4D], then split into 4 gates.
        i_raw, f_raw, z_raw, o_raw = gates.chunk(4, -1)
        # m is a log-domain normalizer. With m0=-inf, the first input weight is 1.
        # Released xLSTM vanilla sLSTM uses sigmoid forget gate; its log is
        # added to the stabilizer before the normalized exponential update.
        log_f_plus_m = F.logsigmoid(f_raw) + m
        m_new = torch.maximum(log_f_plus_m, i_raw)
        i = torch.exp(i_raw - m_new)
        f = torch.exp(log_f_plus_m - m_new)
        c_new = f * c + i * torch.tanh(z_raw)
        n_new = f * n + i
        h_new = torch.sigmoid(o_raw) * (c_new / n_new.clamp_min(1e-6))
        return h_new, (h_new, c_new, n_new, m_new)


class SLSTMLayer(nn.Module):
    """One recurrent layer: [B,P,D] -> [B,P,D], P=number of patches."""

    def __init__(self,
                 dim: int,
                 heads: int,
                 dropout: float,
                 block_idx: int = 0,
                 num_blocks: int = 1):
        super().__init__()
        # xLSTMBlock uses affine LayerNorm without a learned bias.
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.preconv = nn.Conv1d(dim, dim, 4, padding=3, groups=dim)
        self.cell = SLSTMCell(dim, heads, block_idx, num_blocks)
        self.heads = heads
        self.group_weight = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, bias=False)
        # Official xLSTM block uses a gated FFN; projected width is rounded
        # upward to 64, as in UpProjConfigMixin with proj_factor=1.3.
        hidden = math.ceil(1.3 * dim / 64) * 64
        self.ffn_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.ffn_down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,N,D] -> [B,N,D], N=patch tokens, D=hidden features."""
        batch_size, patch_count, hidden_dim = x.shape
        h = x.new_zeros(batch_size, hidden_dim)
        c = x.new_zeros(batch_size, hidden_dim)
        n = x.new_zeros(batch_size, hidden_dim)
        m = x.new_full((batch_size, hidden_dim), -torch.inf)
        outputs = []
        normalized_patches = self.norm1(x)  # [B,N,D]
        # Depthwise temporal preconv: [B,N,D] -> [B,D,N] -> [B,N,D].
        convolved_patches = F.silu(
            self.preconv(rearrange(normalized_patches,
                                   "b n d -> b d n"))[..., :patch_count]
            .transpose(1, 2))
        for patch_index in range(patch_count):
            y, (h, c, n, m) = self.cell(
                normalized_patches[:, patch_index],
                convolved_patches[:, patch_index],
                                        (h, c, n, m))
            outputs.append(y.unsqueeze(1))
        y = torch.cat(outputs, 1)  # [B,N,D]
        # MultiHeadLayerNorm applies GroupNorm independently to each patch.
        # GroupNorm treats each (batch, patch) pair as one sample: [B*P,D].
        y = rearrange(y, "b p d -> (b p) d")  # [B*N,D]
        y = F.group_norm(y, self.heads, weight=self.group_weight)
        y = rearrange(y, "(b p) d -> b p d",
                      b=batch_size, p=patch_count)  # [B,N,D]
        residual_tokens = x + self.dropout(y)  # [B,N,D]
        gate, value = self.ffn_up(self.norm2(residual_tokens)).chunk(2, -1)
        # Each gated FFN branch [B,N,Dff] -> down-projected [B,N,D].
        return residual_tokens + self.dropout(
            self.ffn_down(F.gelu(gate) * value))


class PsLSTMBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout]. Channels share the sLSTM backbone.

    Nonoverlapping patches are fastest. Overlap is allowed; reconstructed
    samples are normalized by coverage. The module is noncausal *within* each
    patch, so use it only on fully observed windows for forecasting.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 patch_size: int = 16,
                 stride: int | None = None,
                 d_model: int = 64,
                 heads: int = 4,
                 depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        if min(in_dim, out_dim, patch_size, d_model, heads, depth) < 1:
            raise ValueError("dimensions must be positive")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        if not 1 <= self.stride <= patch_size:
            raise ValueError(
                "stride must be in [1, patch_size] for full coverage")
        self.embed = nn.Linear(patch_size, d_model)
        self.layers = nn.ModuleList(
            SLSTMLayer(d_model, heads, dropout, i, depth)
            for i in range(depth))
        # The released xLSTMBlockStack applies one final LayerNorm after all
        # recurrent blocks, before the task-specific projection.
        self.post_norm = nn.LayerNorm(d_model, bias=False)
        self.to_patch = nn.Linear(d_model, patch_size)
        self.channel_mix = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> [B,T,Cout], sharing sLSTM across input series."""
        if x.ndim != 3 or x.size(-1) != self.in_dim or x.size(1) < 1:
            raise ValueError(f"expected [B,T,{self.in_dim}] with T>=1")
        batch_size, seq_len, num_series = x.shape
        patch_count = max(
            1, math.ceil(max(0, seq_len - self.patch_size) / self.stride) + 1)
        padded_len = (patch_count - 1) * self.stride + self.patch_size
        # [B,T,Cin] -> [B*Cin,T]; shared backbone, no early series fusion.
        series_streams = rearrange(x, "b t c -> (b c) t")
        series_streams = F.pad(
            series_streams, (0, padded_len - seq_len), mode="replicate")
        # [B*Cin,T] -> [B*Cin,Tpad] -> [B*Cin,N,K].
        patch_values = series_streams.unfold(
            -1, self.patch_size, self.stride)
        patch_tokens = self.embed(patch_values)  # [B*Cin,N,D]
        for layer in self.layers:
            patch_tokens = layer(patch_tokens)  # [B*Cin,N,D]
        patch_tokens = self.post_norm(patch_tokens)
        # [B*Cin,N,D], official encoder boundary.
        values = self.to_patch(patch_tokens)  # [B*Cin,N,K]
        if self.stride == self.patch_size:
            # Nonoverlapping patches need no scatter/add and are the fast path.
            out = rearrange(values, "bc p k -> bc (p k)")
            # [B*Cin,N,K] -> [B*Cin,Tpad] for nonoverlapping patches.
        else:
            # F.fold performs overlap-add on a 1xT image; count rescales overlaps.
            cols = rearrange(values, "bc n k -> bc k n")  # [B*Cin,K,N]
            out = F.fold(cols, (1, padded_len), (1, self.patch_size),
                         stride=(1, self.stride)).squeeze(2).squeeze(1)
            count = F.fold(torch.ones_like(cols), (1, padded_len),
                           (1, self.patch_size),
                           stride=(1, self.stride)).squeeze(2).squeeze(1)
            # Fold produces [B*Cin,Tpad]; divide by overlap count pointwise.
            out = out / count.clamp_min(1)
        out = rearrange(out, "(b c) t -> b c t",
                        b=batch_size, c=num_series)[:, :, :seq_len]
        # [B*Cin,Tpad] -> [B,Cin,Tpad] -> [B,Cin,T].
        # The task-specific forecast head has been replaced by a same-length
        # patch readout; the backbone features are not bypassed by raw input.
        # [B,Cin,T] -> [B,T,Cin] -> [B,T,Cout].
        return self.channel_mix(rearrange(out, "b c t -> b t c"))


if __name__ == "__main__":
    model = PsLSTMBlock(3, 5, patch_size=8, stride=4, d_model=16)
    x = torch.randn(2, 19, 3, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == (2, 19, 5) and x.grad is not None
