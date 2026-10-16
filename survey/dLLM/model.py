"""理解用の小型bidirectional Transformer。実行例: ``python model.py``。"""

from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """許可maskを直接受け取れるmulti-head self-attention。

    入力 ``x: (B,L,D)``、``allowed: (L,L)`` または ``(B,L,L)``、出力 ``(B,L,D)``。
    ``allowed[q,k]=True`` のときだけquery qからkey kを参照できる。
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        b, length, width = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B,H,L,Dh)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B,H,L,L)
        if allowed is not None:
            if allowed.ndim == 2:
                allowed = allowed[None, None, :, :]
            elif allowed.ndim == 3:
                allowed = allowed[:, None, :, :]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = scores.softmax(dim=-1)
        mixed = (probs @ v).transpose(1, 2).contiguous().view(b, length, width)
        return self.out(mixed)


class TransformerBlock(nn.Module):
    """pre-norm attention + MLP。shapeは常に ``(B,L,D)``。"""

    def __init__(self, width: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = SelfAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(), nn.Linear(expansion * width, width))

    def forward(self, x: torch.Tensor, allowed: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), allowed)
        return x + self.mlp(self.norm2(x))


class TinyDiffusionLM(nn.Module):
    """DLMの情報流を再現する教材model（大規模checkpointの層構成再現ではない）。"""

    def __init__(self, vocab_size: int, max_length: int = 256, width: int = 128, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, allowed: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids: (B,L) -> logits: (B,L,V)``。DLM既定は全位置を見られる。"""
        b, length = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=input_ids.device)[None].expand(b, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for block in self.blocks:
            x = block(x, allowed)
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    model = TinyDiffusionLM(32, width=32, layers=2)
    assert model(torch.randint(0, 32, (2, 7))).shape == (2, 7, 32)
    print("model checks passed")
