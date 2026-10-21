"""AdaWaveNet adaptive lifting path in a same-length encoder interface.

The released decoder has *independent* P/U weights and does not algebraically
invert the encoder's modified lifting. This implementation intentionally
preserves that published computation rather than substituting an exact inverse.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Replicate-edge trend smoothing, [B,C,T] -> [B,C,T]."""
    left = kernel_size - 1 - (kernel_size - 1) // 2
    right = (kernel_size - 1) // 2
    extended = F.pad(x, (left, right), mode="replicate")
    return F.avg_pool1d(extended, kernel_size, stride=1)


class _LearnedFilter(nn.Module):
    """Depthwise Conv1d -> GELU -> fixed-length LayerNorm, [B,C,L]."""

    def __init__(self, channels: int, length: int, kernel_size: int) -> None:
        super().__init__()
        self.left = kernel_size // 2
        self.right = kernel_size - 1 - self.left
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm((channels, length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official LiftingScheme uses ReflectionPad1d. Reflection is undefined
        # when a coarse band is too short; only then replicate its boundary.
        mode = "reflect" if x.size(-1) > max(self.left,
                                             self.right) else "replicate"
        padded = F.pad(x, (self.left, self.right), mode=mode)
        return self.norm(self.activation(self.conv(padded)))


def _instance_normalize(x: torch.Tensor) -> torch.Tensor:
    """Official affine-free InstanceNorm1d over a coefficient's time axis.

    [B,C,L] -> [B,C,L]. The explicit formula also supports L=1, for which
    torch's InstanceNorm1d raises in training mode.
    """
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + 1e-5)


class AdaptiveLifting(nn.Module):
    """One invertible learned lifting level over [B,C,T]."""

    def __init__(self,
                 channels: int,
                 input_length: int,
                 kernel_size: int = 3) -> None:
        super().__init__()
        if min(channels, input_length, kernel_size) < 1:
            raise ValueError("dimensions must be positive")
        self.length = input_length
        half_length = (input_length + 1) // 2
        self.update = _LearnedFilter(channels, half_length, kernel_size)
        self.predict = _LearnedFilter(channels, half_length, kernel_size)

    def forward(self, x: torch.Tensor):
        if x.size(-1) != self.length:
            raise ValueError("wrong lifting input length")
        if self.length % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        even, odd = x[..., ::2], x[..., 1::2]  # each [B,C,ceil(T/2)]
        coarse = even + self.update(odd)
        detail = odd - self.predict(coarse)
        return coarse, detail  # both [B,C,ceil(T/2)]

    def inverse(self, coarse: torch.Tensor,
                detail: torch.Tensor) -> torch.Tensor:
        if coarse.shape != detail.shape:
            raise ValueError("coarse/detail must have matching shapes")
        odd = detail + self.predict(coarse)
        even = coarse - self.update(odd)
        interleaved = rearrange(torch.stack((even, odd), dim=-1),
                                "b c t pair -> b c (t pair)")
        return interleaved[..., :self.length]  # [B,C,T]


class _PublishedInverseLifting(nn.Module):
    """Released InverseLiftingScheme, [B,C,L] pair -> [B,C,2L]."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        left = kernel_size // 2
        right = kernel_size - 1 - left

        def network() -> nn.Sequential:
            return nn.Sequential(
                nn.ReflectionPad1d((left, right)),
                nn.Conv1d(channels, 2 * channels, kernel_size,
                          groups=channels),
                nn.Tanh(),
                nn.Conv1d(2 * channels, channels, 1, groups=channels),
                nn.Tanh(),
            )

        self.update = network()
        self.predict = network()

    def forward(self, coarse: torch.Tensor,
                detail: torch.Tensor) -> torch.Tensor:
        # Published order (not the inverse of modified encoder lifting).
        even = coarse - self.update(detail)  # [B,C,L]
        odd = detail + self.predict(even)  # [B,C,L]
        return rearrange(torch.stack((even, odd), dim=-1),
                         "b c t pair -> b c (t pair)")


class _PublishedAttention(nn.Module):
    """Unmasked FullAttention + AttentionLayer over channel tokens."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.query(x), "b c (h d) -> b h c d", h=self.heads)
        k = rearrange(self.key(x), "b c (h d) -> b h c d", h=self.heads)
        v = rearrange(self.value(x), "b c (h d) -> b h c d", h=self.heads)
        scores = torch.matmul(q, k.transpose(-1, -2)) / q.size(-1)**0.5
        weights = self.dropout(torch.softmax(scores, dim=-1))
        result = torch.matmul(weights, v)  # [B,heads,C,d_head]
        return self.output(rearrange(result, "b h c d -> b c (h d)"))


class _PublishedEncoderLayer(nn.Module):
    """Official post-norm attention/pointwise-Conv encoder layer."""

    def __init__(self, d_model: int, heads: int, d_ff: int,
                 dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = _PublishedAttention(d_model, heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        y = self.activation(self.conv1(rearrange(x, "b c d -> b d c")))
        y = self.dropout(y)
        y = rearrange(self.conv2(y), "b d c -> b c d")
        return self.norm2(x + self.dropout(y))


class AdaWaveNetBlock(nn.Module):
    """Adaptive wavelet encoder, [B,seq_len,Cin]->[B,seq_len,Cout].

    Decomposes into trend/seasonal, applies the released channel-token
    encoder, and decodes with independently learned inverse lifting filters.
    The official clustered trend head is kept; cluster IDs are a per-channel
    input, not inferred with K-means inside the block.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int,
                 levels: int = 2,
                 d_model: int = 64,
                 heads: int = 4,
                 attention_layers: int = 1,
                 trend_kernel: int = 25,
                 lifting_kernel: int = 3,
                 dropout: float = 0.0,
                 d_ff: int | None = None,
                 activation: str = "gelu",
                 n_clusters: int = 1) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, levels, d_model, heads,
               attention_layers, n_clusters) < 1:
            raise ValueError("dimensions must be positive")
        if d_model % heads or (d_ff is not None and d_ff < 1):
            raise ValueError(
                "d_model must be divisible by heads; d_ff must be positive")
        if seq_len < 2**levels:
            raise ValueError("seq_len must be at least 2**levels")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.trend_kernel = trend_kernel
        self.n_clusters = n_clusters
        lengths = [seq_len]
        self.lifting = nn.ModuleList()
        self.inverse_lifting = nn.ModuleList()
        self.detail_mixer = nn.ModuleList()
        self.detail_decoder = nn.ModuleList()
        self.coarse_skip = nn.ModuleList()
        for _ in range(levels):
            length = (lengths[-1] + 1) // 2
            self.lifting.append(
                AdaptiveLifting(in_dim, lengths[-1], lifting_kernel))
            self.detail_mixer.append(nn.Linear(length, length))
            self.detail_decoder.append(nn.Linear(length, length))
            self.coarse_skip.append(nn.Linear(length, length))
            self.inverse_lifting.append(
                _PublishedInverseLifting(in_dim, lifting_kernel))
            lengths.append(length)
        self.coarse_embedding = nn.Linear(lengths[-1], d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.channel_encoder = nn.ModuleList([
            _PublishedEncoderLayer(d_model, heads, d_ff or 4 * d_model,
                                   dropout, activation)
            for _ in range(attention_layers)
        ])
        self.channel_norm = nn.LayerNorm(d_model)
        self.coarse_readout = nn.Linear(d_model, lengths[-1])
        self.trend_mixer = nn.ModuleList([
            nn.Linear(seq_len, seq_len) for _ in range(n_clusters)
        ])
        self.trend_affine_weight = nn.Parameter(torch.ones(in_dim))
        self.trend_affine_bias = nn.Parameter(torch.zeros(in_dim))
        self.season_projection = nn.Linear(seq_len, seq_len)
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor,
                clusters: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError("x must have shape [B,seq_len,in_dim]")
        channels_first = rearrange(x, "b t c -> b c t")
        trend = _moving_average(channels_first, self.trend_kernel)
        seasonal = channels_first - trend  # [B,Cin,T]
        trend_mean = trend.mean(dim=-1, keepdim=True).detach()
        trend_scale = (trend.var(dim=-1, keepdim=True, unbiased=False) +
                       1e-5).sqrt().detach()
        trend = (trend - trend_mean) / trend_scale
        trend = (trend * self.trend_affine_weight[None, :, None] +
                 self.trend_affine_bias[None, :, None])
        mean = seasonal.mean(dim=-1, keepdim=True).detach()
        scale = (seasonal.var(dim=-1, keepdim=True, unbiased=False) +
                 1e-5).sqrt().detach()
        coarse = (seasonal - mean) / scale
        details, skips = [], []
        for lifting, detail_mixer, coarse_skip in zip(
                self.lifting, self.detail_mixer, self.coarse_skip):
            coarse, detail = lifting(coarse)
            # AdaWaveNet's AdpWaveletBlock applies separate affine-free
            # InstanceNorm1d to c and d before any branch projection.
            coarse = _instance_normalize(coarse)
            detail = _instance_normalize(detail)
            skips.append(coarse_skip(coarse))  # [B,Cin,L_i]
            details.append((detail, detail_mixer(detail)))

        # Treat channels as attention tokens, with coarse time samples as
        # features: [B,Cin,L_J] -> [B,Cin,d_model] -> [B,Cin,L_J].
        tokens = self.embedding_dropout(self.coarse_embedding(coarse))
        for encoder_layer in self.channel_encoder:
            tokens = encoder_layer(tokens)
        coarse = self.coarse_readout(self.channel_norm(tokens))
        for inverse, skip, (detail, encoded), decoder in zip(
                reversed(self.inverse_lifting), reversed(skips),
                reversed(details), reversed(self.detail_decoder)):
            detail = encoded + decoder(detail)  # [B,Cin,L_i]
            # Odd-length windows are a generic-block extension: the released
            # model assumes lengths divisible by 2**levels.
            coarse = inverse(coarse[..., :skip.size(-1)] + skip, detail)
        coarse = coarse[..., :self.seq_len]
        seasonal_out = self.season_projection(coarse)
        seasonal_out = seasonal_out * scale + mean  # [B,Cin,T]
        if clusters is None:
            clusters = torch.zeros(self.in_dim, dtype=torch.long,
                                   device=x.device)
        if clusters.shape != (self.in_dim,):
            raise ValueError("clusters must have shape [Cin]")
        if torch.any(clusters < 0) or torch.any(clusters >= self.n_clusters):
            raise ValueError("cluster IDs out of range")
        trend_out = torch.cat([
            self.trend_mixer[int(clusters[channel])](
                trend[:, channel:channel + 1])
            for channel in range(self.in_dim)
        ], dim=1)
        trend_out = ((trend_out - self.trend_affine_bias[None, :, None]) /
                     (self.trend_affine_weight[None, :, None] + 1e-10))
        trend_out = trend_out * trend_scale + trend_mean
        combined = rearrange(seasonal_out + trend_out, "b c t -> b t c")
        return self.output(combined)  # [B,T,Cout]
