"""confidence-guided MDLM/BD3-LM反復復号。実行例は``main_flow.py``。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class SamplingOutput:
    sequences: torch.Tensor       # (B,P+G): prompt + generation
    histories: list[torch.Tensor] # 各commit後の同shape canvas


def _predict_and_commit(model, x: torch.Tensor, mutable: torch.Tensor, count: int,
                        mask_id: int, allowed: torch.Tensor | None = None,
                        temperature: float = 0.0) -> torch.Tensor:
    """未確定位置を全て予測し、confidence上位``count``個/sampleだけ確定する。"""
    logits = model(x, allowed=allowed)  # (B,L,V)
    # MASKはforward process専用の吸収状態で、clean data vocabularyの候補ではない。
    # これを候補に残すと「commitしたのにMASKのまま」という状態になってしまう。
    logits[..., mask_id] = -torch.inf
    if temperature > 0:
        # Gumbel-max。logits自体はconfidence計算用にnoiseなしのまま保持する。
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        candidates = (logits + temperature * (-torch.log(-torch.log(u)))).argmax(-1)
    else:
        candidates = logits.argmax(-1)
    probs = F.softmax(logits, dim=-1)
    confidence = probs.gather(-1, candidates[..., None]).squeeze(-1)
    confidence = confidence.masked_fill(~mutable, -torch.inf)
    updated = x.clone()
    for row in range(x.shape[0]):
        k = min(count, int(mutable[row].sum()))
        if k:
            chosen = confidence[row].topk(k).indices
            updated[row, chosen] = candidates[row, chosen]
    return updated


@torch.no_grad()
def sample_mdlm(model, prompt: torch.Tensor, mask_id: int, generation_length: int,
                steps: int, temperature: float = 0.0) -> SamplingOutput:
    """全generation領域をMASKで開始するMDLM sampling。

    ``prompt: (B,P)`` -> canvas ``(B,P+G)``。各stepで残りMASKの一部を並列commitし、
    promptと一度commitしたtokenは変更しない。
    """
    b, _ = prompt.shape
    tail = torch.full((b, generation_length), mask_id, dtype=torch.long, device=prompt.device)
    x = torch.cat([prompt, tail], dim=1)
    histories = [x.clone()]
    for step in range(steps):
        mutable = x.eq(mask_id)
        if not mutable.any():
            break
        remaining_steps = steps - step
        count = max(1, math.ceil(int(mutable.sum(dim=1).max()) / remaining_steps))
        x = _predict_and_commit(model, x, mutable, count, mask_id, temperature=temperature)
        histories.append(x.clone())
    return SamplingOutput(x, histories)


@torch.no_grad()
def sample_bd3lm(model, prompt: torch.Tensor, mask_id: int, generation_length: int,
                 block_size: int, steps_per_block: int, temperature: float = 0.0) -> SamplingOutput:
    """blockは左から順、各block内は並列に復元するBD3-LM sampling。

    ``prompt: (B,P)`` -> ``(B,P+G)``。簡潔さのためbatch内prompt長は共通とする。
    block-causal maskにより現在blockは内部双方向、未来blockは不可視となる。
    """
    b, prompt_len = prompt.shape
    x = torch.cat([prompt, torch.full((b, generation_length), mask_id, dtype=torch.long, device=prompt.device)], 1)
    histories = [x.clone()]
    total = x.shape[1]
    block_ids = torch.arange(total, device=x.device) // block_size
    allowed = block_ids[None, :] <= block_ids[:, None]  # allowed[q,k]
    for start in range(prompt_len, prompt_len + generation_length, block_size):
        end = min(start + block_size, prompt_len + generation_length)
        for step in range(steps_per_block):
            mutable = torch.zeros_like(x, dtype=torch.bool)
            mutable[:, start:end] = x[:, start:end].eq(mask_id)
            if not mutable.any():
                break
            count = max(1, math.ceil((end - start) / (steps_per_block - step)))
            x = _predict_and_commit(model, x, mutable, count, mask_id, allowed, temperature)
            histories.append(x.clone())
    return SamplingOutput(x, histories)
