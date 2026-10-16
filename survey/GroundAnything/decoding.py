"""Entropy-guided blockwise diffusion decodingのframework非依存実装。"""
from __future__ import annotations
import torch


def token_entropy(logits: torch.Tensor, excluded_ids=()) -> torch.Tensor:
    """logits ``(...,V)``からentropy ``(...)``。MASK等をgeneratable vocabularyから除外する。"""
    work = logits.float().clone()
    if excluded_ids:
        work[..., list(excluded_ids)] = -torch.inf
    p = work.softmax(-1)
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


@torch.no_grad()
def entropy_guided_decode(model, prefix: torch.Tensor, mask_id: int, eos_id: int,
                          block_size=32, entropy_threshold=1.0, max_blocks=8):
    """``prefix (1,P)``にblockを追加し、低entropy位置から並列commitする。

    物理blockは既知anchor + B-1 masksという公式契約を簡略化し、ここではB個のmaskを生成する。
    各passでthreshold以下を全commitし、該当なしなら最小entropyを1個commitして停止性を保証する。
    committed tokenは再maskしない。block完了後、次blockは完成prefixをcausal historyとして読む。
    """
    output = prefix.clone()
    for _ in range(max_blocks):
        block_start = output.shape[1]
        work = torch.cat([output, output.new_full((1, block_size), mask_id)], 1)
        while (work[:, block_start:] == mask_id).any():
            T = work.shape[1]
            allowed = torch.ones(T, T, dtype=torch.bool, device=work.device).tril()
            # 新block内だけ双方向、prefixからblockへの逆参照は禁止。
            allowed[block_start:, block_start:] = True
            logits = model(token_ids=work, allowed_attention=allowed)[0]
            positions = (work[0] == mask_id).nonzero(as_tuple=False).flatten()
            positions = positions[positions >= block_start]
            pred_logits = logits[positions - 1]  # next-token shift
            entropy = token_entropy(pred_logits, excluded_ids=(mask_id,))
            chosen = positions[entropy <= entropy_threshold]
            if chosen.numel() == 0:
                chosen = positions[entropy.argmin().view(1)]
            chosen_logits = logits[chosen - 1].clone()
            chosen_logits[:, mask_id] = -torch.inf
            work[0, chosen] = chosen_logits.argmax(-1)
        block = work[:, block_start:]
        eos = (block[0] == eos_id).nonzero(as_tuple=False)
        if eos.numel():
            return torch.cat([output, block[:, :int(eos[0])+1]], 1)
        output = work
    return output
