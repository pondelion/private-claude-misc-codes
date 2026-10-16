"""LoopVL論文のvisual-state/attention診断指標。実行: ``python diagnostics.py``。"""

from __future__ import annotations

import torch


def normalized_spatial_entropy(attention: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """visual attention ``(...,N)``の正規化entropy ``(...)``。0は集中、1は一様。"""
    probability = attention / attention.sum(-1, keepdim=True).clamp_min(eps)
    n = attention.shape[-1]
    return -(probability * probability.clamp_min(eps).log()).sum(-1) / torch.log(attention.new_tensor(float(n)))


def gini_concentration(attention: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """正規化visual attention ``(...,N)``のGini。大きいほど少数patchへ集中。"""
    probability = attention / attention.sum(-1, keepdim=True).clamp_min(eps)
    pairwise = (probability.unsqueeze(-1) - probability.unsqueeze(-2)).abs()
    return pairwise.sum(dim=(-1, -2)) / (2 * attention.shape[-1])


def attention_coverage(attention: torch.Tensor, target: float = 0.8) -> torch.Tensor:
    """attention質量``target``を占める最小token数の割合。``(...,N)->(...)``。"""
    probability = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
    cumulative = probability.sort(-1, descending=True).values.cumsum(-1)
    count = (cumulative < target).sum(-1) + 1
    return count / attention.shape[-1]


if __name__ == "__main__":
    uniform = torch.ones(1, 4)
    concentrated = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert normalized_spatial_entropy(uniform) > normalized_spatial_entropy(concentrated)
    assert gini_concentration(uniform) < gini_concentration(concentrated)
    print("diagnostic checks passed")
