"""
InnerJev - 自己蒸留複合損失 (重み付きCE + 順序回帰項 + KL) (ロス)
================================================================

InnerJev の学習目的関数は、2回の推論サンプル (thinkingあり) の平均を教師分布
として使う自己蒸留で、かつ回答の種類 (choice/noul/score) ごとに異なる重みと
追加の正則化項を持つ。

公式README (gh/InnerJev/README.md) の記述:
    "The loss is `w·CE(q,p) + 0.5w·I(Score)·((E[p]−E[q])/(K−1))² + κ·KL(p0||p)`.
     Choice/Noul/Score weights are 1 / 1.1666667 / 0.8.
     κ=0.2 for hard binary Noul and 0.05 otherwise."
公式コード: gh/InnerJev/innerjev/learning.py::loss_terms
      素のPyTorch実装であり、本ファイルはそのロジックをそのまま移植し、
      各項に詳細なコメントを付けたものである。

============================================================
記号 (README/公式コードとの対応)
============================================================
q  = train_target  : 教師分布 (2回のthinkingサンプルの平均)
p  = model distribution (softmax(logits))
p0 = baseline       : thinkingなしの元モデルの分布 (KL正則化の基準点)
K  = 選択肢数 (rowごとに異なる)
w  = type_weights[row['type']]  : choice=1, noul=1.1666667, score=0.8
κ  = 0.2 (hard Noul, gold_targetが1に近い) / 0.05 (それ以外)

============================================================
Shape Convention
============================================================
B : バッチサイズ
K : 選択肢ラベルの語彙サイズ (255, 全行共通の固定次元。各行の実際の選択肢数は
    rowごとに異なり、それを超える位置は-1e9でマスクされる)
"""

from __future__ import annotations

from typing import Dict, List

import torch


def loss_terms(
    logits: torch.Tensor,            # (B, K): Readoutの出力 (255個の選択肢ラベルに対するロジット)
    rows: List[Dict],                 # 長さB: 各サンプルの {options, type, train_target, p0, gold_target}
    weights: Dict[str, float],         # {"choice":1.0, "noul":1.1666667, "score":0.8}
):
    """自己蒸留の複合損失 w·CE(q,p) + 0.5w·I(Score)·((E[p]-E[q])/(K-1))^2 + κ·KL(p0||p) を計算する。

    ========================================
    出力
    ========================================
    (total, ce, ordinal, kl): 4つとも (B,) の per-sample 値 (呼び出し側で平均/合計する)
    """
    device = logits.device
    lengths = torch.tensor([len(row["options"]) for row in rows], device=device)  # (B,): 各行の実際の選択肢数

    # --- 有効な選択肢数を超える位置を -1e9 でマスクしてからsoftmax (論文の"legal options"制約) ---
    col_index = torch.arange(logits.shape[-1], device=device)[None, :]  # (1, K)
    masked = logits.masked_fill(col_index >= lengths[:, None], -1e9)     # (B, K)
    logs = masked.log_softmax(-1)                                         # (B, K): log p (K次元, 末尾はほぼ-inf)

    teacher = torch.zeros_like(logits)   # (B, K): q (train_target)
    baseline = torch.zeros_like(logits)  # (B, K): p0
    for i, row in enumerate(rows):
        teacher[i, : lengths[i]] = torch.tensor(row["train_target"], device=device)
        baseline[i, : lengths[i]] = torch.tensor(row["p0"], device=device)

    # --- (1) 重み付きCross-Entropy: w * CE(q, p) = w * -sum_k q_k * log p_k ---
    weight = torch.tensor([weights[row["type"]] for row in rows], device=device)  # (B,)
    ce = weight * -(teacher * logs).sum(-1)  # (B,)

    # --- (2) Score専用の順序回帰項: 0.5*w*((E[p]-E[q])/(K-1))^2 ---
    # E[p] = sum_k k*p_k, E[q] = sum_k k*q_k (選択肢インデックスを序数として期待値を取る)
    is_score = torch.tensor([row["type"] == "score" for row in rows], device=device, dtype=logits.dtype)
    ordinal = weight * 0.5 * is_score
    index = torch.arange(logits.shape[-1], device=device, dtype=logits.dtype)  # (K,): 0,1,2,...
    expectation_diff = ((logs.exp() - teacher) * index).sum(-1)  # (B,): E[p]-E[q]
    ordinal = ordinal * expectation_diff.square()
    ordinal = ordinal / (lengths - 1).square().clamp_min(1)  # (K-1)^2 で正規化 (K=1は発生しない前提)

    # --- (3) KL正則化: kappa * KL(p0 || p) = kappa * sum_k p0_k * (log p0_k - log p_k) ---
    # "hard Noul" = noulタイプかつgold_targetが(ほぼ)1 (=正解がほぼ確定している2値判定)
    is_hard_noul = torch.tensor(
        [row["type"] == "noul" and max(row["gold_target"]) >= 1 - 1e-6 for row in rows], device=device,
    )
    kappa = torch.where(is_hard_noul, torch.tensor(0.2, device=device), torch.tensor(0.05, device=device))
    kl = kappa * (baseline * (baseline.clamp_min(1e-30).log() - logs)).sum(-1)  # (B,)

    total = ce + ordinal + kl
    return total, ce, ordinal, kl


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== loss.py Example Usage ===\n")
    torch.manual_seed(0)

    type_weights = {"choice": 1.0, "noul": 1.1666667, "score": 0.8}

    rows = [
        {  # choiceタイプ: 教師がoption1を強く支持
            "type": "choice",
            "options": [{"key": "a"}, {"key": "b"}, {"key": "c"}],
            "train_target": [0.8, 0.15, 0.05],
            "p0": [0.5, 0.3, 0.2],
            "gold_target": [1.0, 0.0, 0.0],
        },
        {  # noul (2値): hard (gold=1.0に近い)
            "type": "noul",
            "options": [{"key": "yes"}, {"key": "no"}],
            "train_target": [0.95, 0.05],
            "p0": [0.6, 0.4],
            "gold_target": [1.0, 0.0],
        },
        {  # score (5段階の順序尺度)
            "type": "score",
            "options": [{"key": str(i)} for i in range(5)],
            "train_target": [0.0, 0.1, 0.6, 0.2, 0.1],
            "p0": [0.2, 0.2, 0.2, 0.2, 0.2],
            "gold_target": [0.0, 0.1, 0.6, 0.2, 0.1],
        },
    ]

    K = 255
    logits = torch.randn(len(rows), K, requires_grad=True)
    total, ce, ordinal, kl = loss_terms(logits, rows, type_weights)

    print("各サンプルの損失内訳:")
    for i, row in enumerate(rows):
        print(f"  [{row['type']:6s}] total={total[i].item():.4f}  ce={ce[i].item():.4f}  "
              f"ordinal={ordinal[i].item():.4f}  kl={kl[i].item():.4f}")

    loss = total.mean()
    loss.backward()
    print(f"\n平均loss={loss.item():.4f}  logitsへの勾度ノルム={logits.grad.norm().item():.4f}")

    print("\n--- ordinal項はscoreタイプ以外で常に0であることを確認 ---")
    print(f"choice: ordinal={ordinal[0].item():.6f} (0のはず)")
    print(f"noul:   ordinal={ordinal[1].item():.6f} (0のはず)")
    print(f"score:  ordinal={ordinal[2].item():.6f} (非ゼロのはず)")

    print("\n--- kappaがhard Noul(gold≈1)かどうかで切り替わることを確認 ---")
    print(f"noul(hard, gold=[1.0,0.0]): kl={kl[1].item():.3e}  (kappa=0.2)")
    rows[1]["gold_target"] = [0.6, 0.4]  # soft Noul (不確実) に変更
    _, _, _, kl2 = loss_terms(logits.detach(), rows, type_weights)
    print(f"noul(soft, gold=[0.6,0.4]): kl={kl2[1].item():.3e}  (kappa=0.05; "
          f"比率kl_hard/kl_soft={kl[1].item() / kl2[1].item():.2f} ≈ 0.2/0.05=4.0のはず)")


if __name__ == "__main__":
    example_usage()
