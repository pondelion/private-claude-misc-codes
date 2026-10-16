"""BD3-LMの二重系列attentionとloss。実行例: ``python bd3lm.py``。"""

from __future__ import annotations

import torch

from mdlm import Corruption, mdlm_loss
from noise_schedule import AlphaSchedule


def bd3_training_attention(length: int, block_size: int, device: torch.device | None = None) -> torch.Tensor:
    """公式BD3-LM学習mask ``allowed: (2L,2L)`` を構築する。

    連結順は ``[x_t(noisy), x_0(clean)]``。許可される情報流は次の3つだけ。

    - noisy query -> 同じnoisy block全体（block内双方向）
    - noisy query -> 過去のclean block（現在targetのleakを防ぐ）
    - clean query -> 現在以前のclean block（block-causal）
    """
    pos = torch.arange(2 * length, device=device)
    q, k = pos[:, None], pos[None, :]
    q_clean, k_clean = q >= length, k >= length
    q_local = torch.where(q_clean, q - length, q)
    k_local = torch.where(k_clean, k - length, k)
    q_block, k_block = q_local // block_size, k_local // block_size
    noisy_same = (~q_clean) & (~k_clean) & (q_block == k_block)
    noisy_to_history = (~q_clean) & k_clean & (q_block > k_block)
    clean_causal = q_clean & k_clean & (q_block >= k_block)
    return noisy_same | noisy_to_history | clean_causal


def bd3lm_forward_and_loss(model, clean_ids: torch.Tensor, eligible: torch.Tensor,
                           corruption: Corruption, schedule: AlphaSchedule,
                           block_size: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``x_t``と``x_0``を1 forwardし、noisy側だけでclean tokenを予測する。

    入力 ``clean_ids: (B,L)``、model入力 ``(B,2L)``、全logits ``(B,2L,V)``、
    loss対象logits ``(B,L,V)``。両streamは同じ論理position 0..L-1を共有する。
    """
    b, length = clean_ids.shape
    concat = torch.cat([corruption.noisy_ids, clean_ids], dim=1)
    positions = torch.arange(length, device=clean_ids.device)[None].expand(b, -1)
    positions = torch.cat([positions, positions], dim=1)
    allowed = bd3_training_attention(length, block_size, clean_ids.device)
    all_logits = model(concat, allowed=allowed, position_ids=positions)
    loss, stats = mdlm_loss(all_logits[:, :length], clean_ids, eligible, corruption, schedule)
    return loss, {**stats, "concat_length": torch.tensor(2 * length, device=clean_ids.device)}


if __name__ == "__main__":
    mask = bd3_training_attention(8, 4)
    # noisy block 1はclean block 0を読めるが、clean block 1（自身の答え）は読めない。
    assert mask[4, 8] and not mask[4, 12]
    assert mask.shape == (16, 16)
    print("BD3-LM attention checks passed")
