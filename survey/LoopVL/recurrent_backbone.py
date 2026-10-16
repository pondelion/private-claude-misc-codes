"""LoopVLのL/H nested recurrenceとvisual anchor。実行例は``main_flow.py``。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from attention import apply_rope


class RotaryAttention(nn.Module):
    """``x:(B,S,D), allowed:(B,S,S) -> (B,S,D)``のRoPE self-attention。"""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, rope_positions: torch.Tensor) -> torch.Tensor:
        b, length, width = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.heads, self.head_dim)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(2))
        q, k = apply_rope(q, k, rope_positions)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        mixed = scores.softmax(-1) @ v
        return self.out(mixed.transpose(1, 2).contiguous().view(b, length, width))


class LoopBlock(nn.Module):
    """pre-norm attentionとSwiGLU FFN。入出力は``(B,S,D)``。"""

    def __init__(self, width: int, heads: int, expansion: int = 3) -> None:
        super().__init__()
        hidden = expansion * width
        self.norm1, self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width)
        self.attn = RotaryAttention(width, heads)
        self.gate, self.value, self.down = nn.Linear(width, hidden), nn.Linear(width, hidden), nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), allowed, positions)
        normalized = self.norm2(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.value(normalized))


class ModuleStack(nn.Module):
    """1回のL/H invocationで全layerを順に適用するparameter stack。"""

    def __init__(self, width: int, heads: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LoopBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, allowed, positions)
        return x


class VisualGate(nn.Module):
    """各visual tokenとinstruction要約から``gate in (0,2)``を計算する。"""

    def __init__(self, width: int, gate_width: int) -> None:
        super().__init__()
        self.visual, self.query = nn.Linear(width, gate_width, bias=False), nn.Linear(width, gate_width, bias=False)
        self.out = nn.Linear(gate_width, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 初期gate=1

    def forward(self, visual: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.out(F.silu(self.visual(visual) + self.query(instruction)[:, None])).squeeze(-1))


@dataclass
class RecurrentOutput:
    hidden: torch.Tensor                # (B,S,D): 最終H state
    invocation_names: list[str]
    visual_update_rms: list[torch.Tensor]
    anchor_scales: list[torch.Tensor]


class LoopVLBackbone(nn.Module):
    """独立parameterのL/H stackを共有再利用するLoopVL backbone。

    公式は各stack 16 layers、``H_cycles=2, L_cycles=3``なので
    ``2*(3+1)*16=128`` layer applications。教材版はlayer数だけ縮小できる。
    """

    def __init__(self, width: int, heads: int, layers_per_stack: int,
                 h_cycles: int = 2, l_cycles: int = 3, gate_width: int = 32,
                 anchor_max_scale: float = 0.5) -> None:
        super().__init__()
        self.h_cycles, self.l_cycles = h_cycles, l_cycles
        self.L, self.H = ModuleStack(width, heads, layers_per_stack), ModuleStack(width, heads, layers_per_stack)
        self.gate = VisualGate(width, gate_width)
        self.raw_alpha = nn.Parameter(torch.full((h_cycles,), -2.0))
        self.anchor_max_scale = anchor_max_scale

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(x.dtype)[..., None]
        return (x * weight).sum(1) / weight.sum(1).clamp_min(1)

    def forward(self, embeddings: torch.Tensor, allowed: torch.Tensor, rope_positions: torch.Tensor,
                visual_mask: torch.Tensor, instruction_mask: torch.Tensor) -> RecurrentOutput:
        high = embeddings                                      # (B,S,D)
        low = torch.zeros_like(high)                           # persistent L state
        anchor = embeddings.clone()                            # 初期projected visualを固定保存
        names, updates, scales = [], [], []
        for h_index in range(self.h_cycles):
            before_visual = high[visual_mask]
            instruction = self._masked_mean(high, instruction_mask)  # (B,D)
            gates = self.gate(high, instruction).masked_fill(~visual_mask, 0.0)
            alpha = self.anchor_max_scale * torch.sigmoid(self.raw_alpha[h_index])
            high = high + alpha * gates[..., None] * anchor * visual_mask[..., None]
            scales.append(alpha)
            for l_index in range(self.l_cycles):
                low = self.L(low + high, allowed, rope_positions)
                names.append(f"H{h_index + 1}-L{l_index + 1}")
            high = self.H(high + low, allowed, rope_positions)
            names.append(f"H{h_index + 1}-H")
            after_visual = high[visual_mask]
            updates.append((after_visual - before_visual).square().mean().sqrt())
        return RecurrentOutput(high, names, updates, scales)
