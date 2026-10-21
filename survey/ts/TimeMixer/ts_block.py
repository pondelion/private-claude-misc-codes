"""TimeMixer Past-Decomposable-Mixing as a same-length task encoder.

The original Future-Multipredictor-Mixing maps each scale to a forecast
horizon.  It is deliberately outside this reusable feature block.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Replicate-edge moving average, [B,T,D] -> [B,T,D]."""
    if kernel_size % 2 != 1 or kernel_size < 1:
        raise ValueError("moving_avg must be a positive odd integer")
    pad = (kernel_size - 1) // 2
    extended = torch.cat(
        (x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)),
        dim=1)
    return rearrange(
        F.avg_pool1d(rearrange(extended, "b t d -> b d t"),
                     kernel_size,
                     stride=1), "b d t -> b t d")


class PastDecomposableMixing(nn.Module):
    """Published bottom-up seasonal / top-down trend mixer on J+1 scales."""

    def __init__(self, lengths: list[int], d_model: int, d_ff: int,
                 moving_avg: int, dropout: float,
                 channel_independence: bool) -> None:
        super().__init__()
        self.lengths = lengths
        self.moving_avg = moving_avg
        self.channel_independence = channel_independence
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cross_layer = (None if channel_independence else nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)))
        self.season_down = nn.ModuleList([
            nn.Sequential(nn.Linear(high, low), nn.GELU(), nn.Linear(low, low))
            for high, low in zip(lengths[:-1], lengths[1:])
        ])
        self.trend_up = nn.ModuleList([
            nn.Sequential(nn.Linear(low, high), nn.GELU(),
                          nn.Linear(high, high))
            for high, low in reversed(list(zip(lengths[:-1], lengths[1:])))
        ])
        self.out_cross_layer = nn.Sequential(nn.Linear(d_model,
                                                       d_ff), nn.GELU(),
                                             nn.Linear(d_ff, d_model))

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(scales) != len(self.lengths):
            raise ValueError("unexpected number of scales")
        seasons, trends = [], []
        for x, expected in zip(scales, self.lengths):
            if x.size(1) != expected:
                raise ValueError("wrong scale length")
            trend = moving_average(x, self.moving_avg)  # [B,T_i,D]
            season = x - trend  # [B,T_i,D]
            if self.cross_layer is not None:
                season, trend = self.cross_layer(season), self.cross_layer(
                    trend)
            seasons.append(rearrange(season, "b t d -> b d t"))
            trends.append(rearrange(trend, "b t d -> b d t"))

        # Fine -> coarse. The time-MLP maps T_i to T_(i+1).
        mixed_season = [seasons[0]]
        for i, layer in enumerate(self.season_down):
            mixed_season.append(seasons[i + 1] + layer(mixed_season[-1]))

        # Coarse -> fine. Reversed layer order matches the official code.
        mixed_trend = [None] * len(trends)
        mixed_trend[-1] = trends[-1]
        for layer, i in zip(self.trend_up, range(len(trends) - 2, -1, -1)):
            mixed_trend[i] = trends[i] + layer(mixed_trend[i + 1])

        outputs = []
        for original, season, trend in zip(scales, mixed_season, mixed_trend):
            out = rearrange(season + trend, "b d t -> b t d")
            if self.channel_independence:
                out = original + self.out_cross_layer(out)
            outputs.append(out)  # each [B,T_i,D]
        return outputs


class TimeMixerBlock(nn.Module):
    """PDM encoder, [B,seq_len,Cin] -> [B,seq_len,Cout].

    ``channel_independence=False`` retains multivariate interactions in the
    token embedding.  When True, every input channel is encoded separately
    with shared weights; the final projection fuses their learned features.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int,
                 d_model: int = 64,
                 d_ff: int = 128,
                 e_layers: int = 2,
                 down_sampling_layers: int = 2,
                 down_sampling_window: int = 2,
                 moving_avg: int = 5,
                 dropout: float = 0.0,
                 channel_independence: bool = False,
                 down_sampling_method: str = "avg",
                 use_norm: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, d_model, d_ff, e_layers,
               down_sampling_window) < 1 or down_sampling_layers < 1:
            raise ValueError("dimensions and layer counts must be positive")
        if moving_avg % 2 != 1:
            raise ValueError("moving_avg must be odd")
        if down_sampling_method not in {"avg", "max"}:
            raise ValueError("down_sampling_method must be avg or max")
        lengths = [seq_len]
        for _ in range(down_sampling_layers):
            lengths.append(lengths[-1] // down_sampling_window)
        if min(lengths) < 1:
            raise ValueError("too many downsampling layers for seq_len")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.window = down_sampling_window
        self.method = down_sampling_method
        self.channel_independence = channel_independence
        self.use_norm = use_norm
        self.moving_avg = moving_avg
        # Released forecast path normalizes *each* input resolution before
        # embedding. Each scale owns independent affine parameters [Cin].
        self.norm_weight = nn.ParameterList([
            nn.Parameter(torch.ones(in_dim)) for _ in lengths
        ])
        self.norm_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(in_dim)) for _ in lengths
        ])
        self.embedding = nn.Conv1d(1 if channel_independence else in_dim,
                                   d_model,
                                   kernel_size=3,
                                   padding=1,
                                   padding_mode="circular",
                                   bias=False)
        self.embed_dropout = nn.Dropout(dropout)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(lengths, d_model, d_ff, moving_avg, dropout,
                                   channel_independence)
            for _ in range(e_layers)
        ])
        self.output = nn.Linear(
            d_model * in_dim if channel_independence else d_model, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return official forecast-encoder finest scale before task adapter.

        [B,T,Cin] -> [B,T,D] (CI=0) or [B,T,Cin*D] (CI=1).
        Future prediction FMM is intentionally outside the encoder contract.
        """
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError("x must have shape [B,seq_len,in_dim]")
        batch = x.size(0)
        scales = [x]
        for _ in range(len(self.pdm_blocks[0].lengths) - 1):
            source = rearrange(scales[-1], "b t c -> b c t")
            if self.method == "avg":
                reduced = F.avg_pool1d(source, self.window)
            else:
                reduced = F.max_pool1d(source, self.window)
            scales.append(rearrange(reduced, "b c t -> b t c"))

        embedded = []
        for index, scale in enumerate(scales):
            if self.use_norm:
                mean = scale.mean(dim=1, keepdim=True).detach()
                stdev = (scale.var(dim=1, keepdim=True, unbiased=False) +
                         1e-5).sqrt().detach()
                scale = (scale - mean) / stdev
                scale = (scale * self.norm_weight[index] +
                         self.norm_bias[index])  # [B,T_i,Cin]
            if not self.channel_independence:
                # Official forecast pre_enc passes seasonal (not raw x) to
                # DataEmbedding_wo_pos; the trend is reserved for its FMM.
                scale = scale - moving_average(scale, self.moving_avg)
            if self.channel_independence:
                tokens = rearrange(scale, "b t c -> (b c) 1 t")
            else:
                tokens = rearrange(scale, "b t c -> b c t")
            encoded = self.embedding(tokens)  # [B or B*Cin,D,T_i]
            embedded.append(
                self.embed_dropout(rearrange(encoded, "b d t -> b t d")))
        for block in self.pdm_blocks:
            embedded = block(embedded)
        finest = embedded[0]  # [B or B*Cin,T,D]
        if self.channel_independence:
            finest = rearrange(finest,
                               "(b c) t d -> b t (c d)",
                               b=batch,
                               c=self.in_dim)
        return finest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        finest = self.forward_features(x)
        return self.output(finest)  # [B,T,Cout]
