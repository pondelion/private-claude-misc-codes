"""Grounding box rewardとGRPO objectiveの理解用実装。"""
from __future__ import annotations
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyxy boxes ``(N,4)``と``(M,4)``のpairwise IoU ``(N,M)``。座標scaleは任意。"""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb-lt).clamp_min(0).prod(-1)
    area_a = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(-1)[:, None]
    area_b = (b[:, 2:]-b[:, :2]).clamp_min(0).prod(-1)[None]
    return inter / (area_a + area_b - inter).clamp_min(1e-8)


def greedy_f1_iou(pred: torch.Tensor, target: torch.Tensor, threshold=.5):
    """同一label内で1対1greedy matchingし、F1とmatched mean IoUを返す。"""
    if len(pred) == 0 or len(target) == 0:
        return (1.0 if len(pred) == len(target) else 0.0), 0.0
    ious, used_p, used_t, matched = box_iou(pred, target), set(), set(), []
    while True:
        score, flat = ious.max(0).values.max(0)
        if float(score) < threshold: break
        p, t = divmod(int(ious.argmax()), ious.shape[1])
        if p in used_p or t in used_t:
            ious[p,t] = -1; continue
        used_p.add(p); used_t.add(t); matched.append(float(score)); ious[p,:] = -1; ious[:,t] = -1
    precision, recall = len(matched)/len(pred), len(matched)/len(target)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return f1, sum(matched)/max(1,len(matched))


def group_normalized_advantage(rewards: torch.Tensor, eps=1e-6):
    """同一promptのG response内でpopulation mean/stdにより標準化する。"""
    std = rewards.float().std(unbiased=False)
    return torch.zeros_like(rewards) if float(std) == 0 else (rewards-rewards.mean())/(std+eps)


def grpo_clipped_loss(current_logp, old_logp, advantages, valid_mask, epsilon=.2):
    """token-level clipped GRPO。入力shapeはすべて``(G,T)``、出力scalar。"""
    ratio = (current_logp-old_logp).exp()
    clipped = ratio.clamp(1-epsilon, 1+epsilon)
    surrogate = torch.minimum(ratio*advantages[:,None], clipped*advantages[:,None])
    per_response = -(surrogate*valid_mask).sum(1)/valid_mask.sum(1).clamp_min(1)
    return per_response.mean()
