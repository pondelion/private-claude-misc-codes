"""LoopVLのPrefixLM maskとhybrid 2D RoPE。実行: ``python attention.py``。"""

from __future__ import annotations

import torch


def prefix_lm_attention(token_types: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """PrefixLMの参照許可maskを返す。

    入力は``token_types, valid: (B,S)``。type 1はheader/image/instructionからなる
    bidirectional prefix、type 0はresponse。返値``allowed: (B,S,S)``は
    ``allowed[b,q,k]=True``のときquery qがkey kを参照できる。
    """
    if token_types.shape != valid.shape:
        raise ValueError("token_types and valid must have the same shape")
    _, length = token_types.shape
    q = torch.arange(length, device=token_types.device)[None, :, None]
    k = torch.arange(length, device=token_types.device)[None, None, :]
    q_prefix = token_types[:, :, None].eq(1)
    k_prefix = token_types[:, None, :].eq(1)
    # prefix queryはprefix全体、response queryはprefix全体+過去/現在responseを読む。
    allowed = (q_prefix & k_prefix) | (~q_prefix & (k_prefix | (k <= q)))
    return allowed & valid[:, :, None] & valid[:, None, :]


def build_rope_positions(token_types: torch.Tensor, visual_mask: torch.Tensor,
                         grid_height: int, grid_width: int) -> torch.Tensor:
    """visual tokenだけ2D、text tokenは(row=col)となる座標を作る。

    ``token_types/visual_mask: (B,S) -> positions: (2,B,S)``。各sampleのvisual
    token数はこの教材版では共通の``grid_height*grid_width``とする。
    """
    batch, length = token_types.shape
    positions = torch.arange(length, device=token_types.device)[None].expand(batch, -1)
    rope = positions[None].expand(2, -1, -1).clone()
    count = grid_height * grid_width
    for row in range(batch):
        indices = visual_mask[row].nonzero(as_tuple=False).flatten()
        if len(indices) != count:
            raise ValueError(f"expected {count} visual slots, got {len(indices)}")
        offset = int(indices[0])
        rope[0, row, indices] = torch.arange(grid_height, device=rope.device).repeat_interleave(grid_width) + offset
        rope[1, row, indices] = torch.arange(grid_width, device=rope.device).repeat(grid_height) + offset
    return rope


def rotary_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """公式hybrid RoPEと同じaxis結合。``(2,B,S) -> cos/sin (B,S,Dh)``。"""
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    freq = positions.float().unsqueeze(-1) * inv[None, None, None, :]
    angles = torch.cat((freq[0], freq[1]), dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q,k: (B,H,S,Dh)``へ2D/text-compatible RoPEを適用する。"""
    cos, sin = rotary_cos_sin(positions, q.shape[-1])
    cos, sin = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


if __name__ == "__main__":
    types = torch.tensor([[1, 1, 1, 0, 0]])
    valid = torch.ones_like(types, dtype=torch.bool)
    mask = prefix_lm_attention(types, valid)
    assert mask[0, 0, 2] and not mask[0, 0, 3] and mask[0, 4, 3]
    print("attention checks passed")
