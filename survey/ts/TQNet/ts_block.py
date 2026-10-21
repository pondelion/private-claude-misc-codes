"""TQNet temporal-query encoder with an independent same-length readout.

Official encoder boundary: after ``hidden + input_proj`` [B,C,D].
The phase argument is the first history sample's index modulo cycle_len.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class TQNetBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout], with optional phase [B].

    Unlike a channel-independent temporal model, MultiheadAttention uses
    the learned query for each channel to retrieve all channel histories.
    Attention's ``embed_dim`` is T, exactly as in the official implementation.
    """

    def __init__(self, in_dim: int, out_dim: int, seq_len: int,
                 cycle_len: int, d_model: int = 64, heads: int = 4,
                 attention_dropout: float = 0.5, dropout: float = 0.0,
                 use_norm: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, cycle_len, d_model, heads) < 1:
            raise ValueError("dimensions must be positive")
        if seq_len % heads:
            raise ValueError("official channel attention requires T % heads = 0")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.cycle_len = cycle_len
        self.use_norm = use_norm
        self.temporal_query = nn.Parameter(torch.zeros(cycle_len, in_dim))
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=seq_len, num_heads=heads, batch_first=True,
            dropout=attention_dropout)
        self.input_proj = nn.Linear(seq_len, d_model)
        self.model = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU())
        # Task adapter: the paper instead uses D -> fixed forecast horizon.
        self.time_readout = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, seq_len))
        self.output = nn.Linear(in_dim, out_dim)

    def forward_features(self, x: torch.Tensor,
                         phase: torch.Tensor | None = None) -> torch.Tensor:
        """Return official pre-head representation [B,Cin,D]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        batch = x.shape[0]  # B; input axes are [B,T,Cin].
        if phase is None:
            # A zero origin is meaningful only when each window begins at
            # phase zero. Pass measured phases for genuine periodic data.
            phase = torch.zeros(batch, dtype=torch.long, device=x.device)
        if phase.shape != (batch,):
            raise ValueError("phase must be [B]")
        phase = phase.to(device=x.device, dtype=torch.long)
        if self.use_norm:
            # Match official torch.var default (sample variance, unbiased).
            mean = x.mean(dim=1, keepdim=True)  # [B,1,Cin]
            std = torch.sqrt(x.var(dim=1, keepdim=True) + 1e-5)
            x = (x - mean) / std  # [B,T,Cin]
        histories = rearrange(x, "b t c -> b c t")  # [B,C,T]
        # [B,1]+[1,T] -> [B,T] periodic row indices.
        positions = (phase[:, None] + torch.arange(
            self.seq_len, device=x.device)[None, :]) % self.cycle_len
        queries = rearrange(self.temporal_query[positions],
                            "b t c -> b c t")  # [B,C,T]
        information, _ = self.channel_attention(
            queries, histories, histories, need_weights=False)
        # Q,K,V:[B,Cin,T] -> attention over Cin -> [B,Cin,T].
        # [B,C,T] -> [B,C,D]; model MLP and residual match official order.
        embedded = self.input_proj(histories + information)
        return embedded + self.model(embedded)  # [B,Cin,D]

    def forward(self, x: torch.Tensor,
                phase: torch.Tensor | None = None) -> torch.Tensor:
        # [B,C,D] -> [B,C,T] -> [B,T,C] -> [B,T,Cout]. The official
        # de-normalization is part of its forecasting head, not this encoder.
        tokens = self.forward_features(x, phase)
        # Learned same-length adapter: [B,Cin,D] -> [B,Cin,T] -> [B,T,Cin].
        series = rearrange(self.time_readout(tokens), "b c t -> b t c")
        return self.output(series)  # [B,T,Cout], including Cin=Cout.


if __name__ == "__main__":
    encoder = TQNetBlock(4, 6, seq_len=32, cycle_len=24)
    sample = torch.randn(3, 32, 4, requires_grad=True)
    phases = torch.tensor([0, 4, 7])
    encoded = encoder(sample, phases)
    assert encoded.shape == (3, 32, 6)
    encoded.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
