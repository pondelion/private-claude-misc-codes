"""MDLMのforward corruptionと学習objective。実行例: ``python mdlm.py``。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from noise_schedule import AlphaSchedule, LinearSchedule


@dataclass
class Corruption:
    noisy_ids: torch.Tensor       # (B,L): x_t
    masked: torch.Tensor          # (B,L): 実際にMASKへ置換した位置
    timesteps: torch.Tensor       # (B,): 各sampleのt


def corrupt_tokens(clean_ids: torch.Tensor, eligible: torch.Tensor, mask_id: int,
                   schedule: AlphaSchedule, timesteps: torch.Tensor | None = None,
                   epsilon: float = 1e-3) -> Corruption:
    """clean tokenを独立にMASKへ置換するforward process。

    ``clean_ids, eligible: (B,L)``。prompt/paddingは``eligible=False``にすれば壊れず、
    lossにも入らない。tはtokenごとではなくsampleごとに1個という公式実装の設定。
    """
    b, length = clean_ids.shape
    if timesteps is None:
        timesteps = epsilon + (1.0 - epsilon) * torch.rand(b, device=clean_ids.device)
    p_mask = schedule.mask_probability(timesteps)[:, None].expand(b, length)
    masked = (torch.rand_like(p_mask) < p_mask) & eligible
    noisy = torch.where(masked, torch.as_tensor(mask_id, device=clean_ids.device), clean_ids)
    return Corruption(noisy, masked, timesteps)


def align_ar_logits(logits: torch.Tensor) -> torch.Tensor:
    """AR model変換時にnext-token logitsを同位置復元へ合わせる。``(B,L,V)->(B,L,V)``。

    公式の``right_shift_logits``に対応する。入力側にはBOSをprependしておく。
    純粋なbidirectional MDLMをscratch学習する場合は使用しない。
    """
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def mdlm_loss(logits: torch.Tensor, clean_ids: torch.Tensor, eligible: torch.Tensor,
              corruption: Corruption, schedule: AlphaSchedule) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MASK位置でclean token ``x_0`` を予測する重み付きCE。

    ``logits: (B,L,V)``, ``clean_ids/eligible: (B,L)``。linear scheduleでは、位置iが
    確率tで選ばれたlossへ1/tを掛けるため、期待値では全eligible tokenのclean-token CEになる。
    分母はMASK数ではなくeligible数であり、公式``loss_norm_type='token'``に対応する。
    """
    b, length, vocab = logits.shape
    if clean_ids.shape != (b, length):
        raise ValueError("logits and token shapes do not match")
    nll = F.cross_entropy(logits.reshape(-1, vocab), clean_ids.reshape(-1), reduction="none").view(b, length)
    weights = schedule.loss_weight(corruption.timesteps)[:, None]
    weighted = nll * weights * corruption.masked
    loss = weighted.sum() / eligible.sum().clamp_min(1)
    stats = {"masked_fraction": corruption.masked.float().sum() / eligible.sum().clamp_min(1),
             "mean_t": corruption.timesteps.mean(), "token_nll_sum": weighted.detach().sum()}
    return loss, stats


if __name__ == "__main__":
    torch.manual_seed(0)
    ids = torch.randint(0, 15, (3, 8))
    eligible = torch.ones_like(ids, dtype=torch.bool)
    corr = corrupt_tokens(ids, eligible, 15, LinearSchedule(), torch.tensor([0.25, 0.5, 0.75]))
    logits = torch.randn(3, 8, 16, requires_grad=True)
    loss, _ = mdlm_loss(logits, ids, eligible, corr, LinearSchedule())
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    print("MDLM checks passed")
