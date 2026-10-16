"""AR-to-Diffusion conversionのcorruption、attention mask、joint objective。

T: clean sequence長、R: response位置数、B: diffusion block size、V: vocabulary。
公式は2つの相補viewとclean streamを1 joint sequenceへpackingする。本実装はmaskの意味を
明示するためstream indexを保ち、同じvisibility ruleをbool attention maskとして返す。
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class CorruptedViews:
    view1: torch.Tensor          # (T,)
    view2: torch.Tensor          # (T,)
    masked1: torch.Tensor        # (T,) bool: MDM loss対象
    masked2: torch.Tensor        # (T,) bool
    block_ids: torch.Tensor      # (T,), response外は-1


def complementary_corruption(tokens, response_mask, mask_id, eos_id, block_size=32, generator=None):
    """responseを連続B-token blockへ分け、相補的な2 noisy viewを作る。

    blockごとにt~Uniform(0,1)、各非EOS位置にm1~Bernoulli(t)、m2=1-m1を使う。
    よって非EOS response tokenは必ず片方だけでmaskされ、EOSは両viewでmaskされる。
    prompt/image/paddingはresponse_mask=Falseなので破壊しない。
    """
    if tokens.ndim != 1 or response_mask.shape != tokens.shape:
        raise ValueError("tokens and response_mask must be aligned 1-D tensors")
    response_positions = response_mask.nonzero(as_tuple=False).flatten()
    block_ids = torch.full_like(tokens, -1)
    for order, position in enumerate(response_positions.tolist()):
        block_ids[position] = order // block_size
    m1 = torch.zeros_like(response_mask)
    m2 = torch.zeros_like(response_mask)
    for block in range(int(block_ids.max().item()) + 1 if response_positions.numel() else 0):
        positions = (block_ids == block).nonzero(as_tuple=False).flatten()
        non_eos = positions[tokens[positions] != eos_id]
        t = torch.rand((), generator=generator, device=tokens.device)
        draws = torch.rand(non_eos.numel(), generator=generator, device=tokens.device) < t
        m1[non_eos], m2[non_eos] = draws, ~draws
        eos = positions[tokens[positions] == eos_id]
        m1[eos], m2[eos] = True, True
    v1, v2 = tokens.clone(), tokens.clone()
    v1[m1], v2[m2] = mask_id, mask_id
    return CorruptedViews(v1, v2, m1, m2, block_ids)


def conversion_attention_mask(block_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """1 noisy view用の可視性``(T,T)``を返す。Trueはqueryがkeyを読める。

    prompt positionsはcausal。response queryはprompt、過去block、同一block全体を読めるが、
    未来blockは読めない。同一block内の双方向性が通常causal LMとの差分である。
    """
    T = block_ids.numel()
    q = torch.arange(T)[:, None]
    k = torch.arange(T)[None, :]
    allowed = k <= q  # prompt/clean causal baseline
    for i in response_mask.nonzero(as_tuple=False).flatten().tolist():
        current = int(block_ids[i])
        # 全promptと、responseのcurrent以前のblockを可視にする。
        allowed[i] = (~response_mask) | ((block_ids >= 0) & (block_ids <= current))
    return allowed


def shifted_cross_entropy(logits, targets, supervised):
    """position i-1でtoken iを予測するshifted CE。logits ``(T,V)`` -> scalar。"""
    valid = supervised[1:]
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[:-1][valid], targets[1:][valid])


def conversion_loss(clean_logits, noisy1_logits, noisy2_logits, clean_tokens, response_mask,
                    masked1, masked2, mdm_weight=.5, ar_weight=.5):
    """論文式``0.5 L_MDM + 0.5 L_AR``を計算する。

    ARはclean response全位置、MDMは各noisy viewでmaskされたresponse位置だけをGTにする。
    prompt/image/paddingはresponse_mask=Falseなのでloss対象外。inverse-noise weightingは行わない。
    """
    ar = shifted_cross_entropy(clean_logits, clean_tokens, response_mask)
    # 2 viewのmasked tokenを合算してtoken平均にする。
    losses = []
    for logits, mask in ((noisy1_logits, masked1), (noisy2_logits, masked2)):
        valid = mask[1:]
        if valid.any():
            losses.append(F.cross_entropy(logits[:-1][valid], clean_tokens[1:][valid], reduction="sum"))
    denom = masked1[1:].sum() + masked2[1:].sum()
    mdm = torch.stack(losses).sum() / denom.clamp_min(1) if losses else clean_logits.sum() * 0
    return {"loss": mdm_weight * mdm + ar_weight * ar, "mdm_loss": mdm, "ar_loss": ar}
