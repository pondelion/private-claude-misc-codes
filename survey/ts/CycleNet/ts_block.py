"""CycleNet recurrent-cycle feature extractor with a separate task head."""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class RecurrentCycle(nn.Module):
    """Gather a learned periodic template as [B,T,C] from phase [B]."""

    def __init__(self, cycle_len: int, channels: int) -> None:
        super().__init__()
        if cycle_len < 1 or channels < 1:
            raise ValueError("cycle_len and channels must be positive")
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channels))

    def forward(self, phase: torch.Tensor, length: int) -> torch.Tensor:
        """[B] start phases -> [B,T,C] periodic values; T=length."""
        if phase.ndim != 1 or length < 1:
            raise ValueError("phase must be [B], length must be positive")
        offsets = torch.arange(length, device=phase.device)  # [T]
        # Broadcast [B,1]+[1,T] -> [B,T] cycle-table row indices.
        indices = (phase.long()[:, None] + offsets[None, :]) % self.cycle_len
        return self.data[indices]  # [B,T,C]


class CycleNetBlock(nn.Module):
    """Use remove-cycle -> temporal mixer -> restore-cycle, [B,T,Cin]->[B,T,Cout].

    This is noncausal and requires a fixed ``seq_len`` because the original
    channel-independent mixer is an nn.Linear over the complete time axis.
    ``phase`` has exactly the official ``cycle_index`` semantics: the index
    used to subtract the input cycle. The output cycle starts at phase + T.
    If omitted, all windows are assumed to have index zero.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int,
                 cycle_len: int,
                 model_type: str = "mlp",
                 d_model: int = 64,
                 use_revin: bool = True,
                 eps: float = 1e-5,
                 variance_unbiased: bool = False) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, cycle_len, d_model) < 1:
            raise ValueError("dimensions must be positive")
        if model_type not in {"linear", "mlp"}:
            raise ValueError("model_type must be 'linear' or 'mlp'")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.use_revin = use_revin
        self.eps = eps
        self.variance_unbiased = variance_unbiased
        self.cycle = RecurrentCycle(cycle_len, in_dim)
        if model_type == "linear":
            self.temporal = nn.Linear(seq_len, seq_len)
        else:
            self.temporal = nn.Sequential(nn.Linear(seq_len,
                                                    d_model), nn.ReLU(),
                                          nn.Linear(d_model, seq_len))
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self,
                x: torch.Tensor,
                phase: torch.Tensor | None = None) -> torch.Tensor:
        """[B,T,Cin] and start phase [B] -> [B,T,Cout]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError("x must have shape [B,seq_len,in_dim]")
        batch = x.size(0)
        if phase is None:
            phase = torch.zeros(batch, dtype=torch.long, device=x.device)
        if phase.shape != (batch, ):
            raise ValueError("phase must have shape [B]")
        if phase.dtype not in (torch.int8, torch.int16, torch.int32,
                               torch.int64, torch.uint8):
            raise TypeError("phase must contain integer cycle indices")
        phase = phase.to(device=x.device)
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)  # [B,1,Cin]
            scale = (
                x.var(dim=1, keepdim=True, unbiased=self.variance_unbiased) +
                self.eps).sqrt()  # [B,1,Cin]
            x = (x - mean) / scale  # [B,T,Cin]

        input_cycle = self.cycle(phase, self.seq_len)  # [B,T,Cin]
        # Subtract periodic template, then expose time to shared Linear:
        # [B,T,Cin] -> [B,Cin,T]; C is independent until output Linear.
        residual = rearrange(x - input_cycle, "b t c -> b c t")
        mixed = self.temporal(residual)  # [B,Cin,T]
        # Match official Model.forward even when T is not a multiple of W.
        # The predicted horizon is T in this encoder adaptation.
        future_phase = (phase + self.seq_len) % self.cycle.cycle_len
        output_cycle = self.cycle(future_phase, self.seq_len)  # [B,T,Cin]
        # [B,Cin,T] -> [B,T,Cin], then restore next-window cycle.
        restored = rearrange(mixed, "b c t -> b t c") + output_cycle
        if self.use_revin:
            restored = restored * scale + mean  # [B,T,Cin]
        return self.output(restored)  # [B,T,Cout]
