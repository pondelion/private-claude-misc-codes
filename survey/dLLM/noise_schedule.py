"""Masked diffusionの時刻schedule。実行例: ``python noise_schedule.py``。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class AlphaSchedule:
    """`alpha(t)=tokenがcleanなまま残る確率`を表す基底class。"""

    eps: float = 1e-6

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mask_probability(self, t: torch.Tensor) -> torch.Tensor:
        """入力 ``t: (B,)`` に対しmask確率 ``1-alpha(t): (B,)``。"""
        return 1.0 - self.alpha(t)

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """連続時間ELBOの重み ``-alpha'(t)/(1-alpha(t)): (B,)``。"""
        return -self.derivative(t) / (1.0 - self.alpha(t) + self.eps)

    def reverse_mask_probability(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """時刻tのMASKが、よりcleanな時刻sにもMASKで残る条件付き確率。``s,t: (B,)``。"""
        if not torch.all(s < t):
            raise ValueError("reverse process requires s < t")
        return (1.0 - self.alpha(s)) / (1.0 - self.alpha(t))


class LinearSchedule(AlphaSchedule):
    """論文の基本設定: alpha(t)=1-t、mask確率=t、loss重みはほぼ1/t。"""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class CosineSchedule(AlphaSchedule):
    """公式実装に含まれる、時刻端で緩やかに変化するschedule。"""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos((math.pi / 2.0) * (1.0 - t))

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2.0) * torch.sin((math.pi / 2.0) * (1.0 - t))


if __name__ == "__main__":
    times = torch.tensor([0.25, 0.5, 0.75])
    schedule = LinearSchedule()
    assert torch.allclose(schedule.mask_probability(times), times)
    assert torch.allclose(schedule.loss_weight(times), 1.0 / (times + schedule.eps))
    print("schedule checks passed")
