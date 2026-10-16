"""
OneStreamer - Proactive State Transition Learning (ロス計算)
================================================================

論文の核心的な学習テクニックである PSTL (Proactive State Transition Learning)
を実装する。密な状態トークン教師信号 (大半が </Silence> で占められる) による
"waiting" への過学習を避けつつ、状態変化・状態持続の両方の教師信号を残す。

論文: OneStreamer (arXiv:2610.01762)
      - Sec 3.3 Proactive State Transition Learning
      - Eq. (1) 供給クォータ q = max_X max_{a in A_tau} n_{X->a}
      - Eq. (2) 各遷移グループからのサンプリング数 |I_{X->Y}| = min(n_{X->Y}, q)
      - Eq. (3)/Appendix A.1 の状態損失 L_state
公式コード: 学習コードは非公開 (gh/OneStreamer には Inference/ と Eval/ のみ公開され、
      PSTL のサンプリング実装は含まれていない)。本ファイルは論文本文・Appendix の
      数式をそのまま実装した再構築である。

============================================================
アルゴリズム概要 (論文 Sec 3.3, Appendix A.1)
============================================================
1. 1つの学習シーケンス中の制御トークン列を、直前の制御状態 X から現在の状態 Y
   への遷移 (X -> Y) ごとにグループ化する。
2. タスクの出力アンカー集合 A_tau (例: PHCMなら {</Observe>, </Summary>}) へ向かう
   遷移グループのうち、最大のグループサイズをクォータ q とする (Eq.1)。
3. 各遷移グループから、クォータ q を超えないよう一様無復元抽出する (Eq.2)。
   抽出されなかった状態トークンは系列には残るが、状態損失からは除外される
   (注意機構やキャプション/応答への教師信号には影響しない)。
4. キャプション本文 (</Observe>, </Summary> に続くテキスト) や応答本文
   (</Response> に続くテキスト) は、PSTL の影響を受けず常に全トークン教師信号。

============================================================
Shape Convention
============================================================
T          : トークン列長 (1サンプル)
V          : 語彙サイズ
IGNORE     : -100 (cross_entropy の ignore_index)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
START_STATE = "<start>"  # シーケンス先頭の制御トークンに対する仮想的な「直前状態」


@dataclass(frozen=True)
class ControlTokenEvent:
    """1個の制御トークンの、系列内での位置と注釈された状態。"""

    position: int  # input_ids 内でその制御トークンが位置するインデックス (0-indexed)
    state: str     # 例: "</Silence>", "</Observe>", ...


# ============================================================
# 1. 供給クォータの計算とサンプリング (Eq. 1, 2)
# ============================================================

def _group_by_transition(
    events: Sequence[ControlTokenEvent],
) -> Dict[Tuple[str, str], List[int]]:
    """制御トークン列を (直前状態 X, 現在状態 Y) ごとにグループ化する。

    ========================================
    入力
    ========================================
    events: 系列内で出現順にソートされた ControlTokenEvent のリスト
            (長さ N_ctrl = そのシーケンス内の制御トークン総数)

    ========================================
    出力
    ========================================
    groups: Dict[(X, Y), List[position]]
        X: 直前の制御状態 (先頭要素は START_STATE)
        Y: 現在の制御状態
        value: その遷移に属する制御トークンの position 一覧
    """
    groups: Dict[Tuple[str, str], List[int]] = {}
    prev_state = START_STATE
    for ev in events:
        key = (prev_state, ev.state)
        groups.setdefault(key, []).append(ev.position)
        prev_state = ev.state
    return groups


def compute_supervision_quota(
    groups: Dict[Tuple[str, str], List[int]],
    anchors: FrozenSet[str],
) -> int:
    """供給クォータ q = max_X max_{a in A_tau} n_{X->a} を計算する (Eq. 1)。

    シーケンス中に出力アンカーへ向かう遷移が1つも存在しない場合 (理論上は
    起こらないはずだが、空の安全策として) は q=0 とする。
    """
    anchor_group_sizes = [
        len(indices) for (_, y), indices in groups.items() if y in anchors
    ]
    if not anchor_group_sizes:
        return 0
    return max(anchor_group_sizes)


def select_supervised_positions(
    events: Sequence[ControlTokenEvent],
    anchors: FrozenSet[str],
    rng: Optional[random.Random] = None,
) -> Tuple[set, int]:
    """PSTL のクォータに基づき、状態損失に使う position 集合を選ぶ (Eq. 1, 2)。

    ========================================
    戻り値
    ========================================
    selected: Set[int]  - 状態損失に使う制御トークンの position 集合 (I = union I_{X->Y})
    quota: int          - このシーケンスで計算されたクォータ q
    """
    if rng is None:
        rng = random.Random()

    groups = _group_by_transition(events)
    quota = compute_supervision_quota(groups, anchors)

    selected: set = set()
    for (_, _), indices in groups.items():
        k = min(len(indices), quota)
        if k <= 0:
            continue
        # 一様無復元抽出 (Eq. 2): |I_{X->Y}| = min(n_{X->Y}, q)
        selected.update(rng.sample(indices, k))
    return selected, quota


# ============================================================
# 2. ラベル構築: 状態損失のマスキングを既存ラベルに適用
# ============================================================

def build_labels_with_pstl(
    base_labels: torch.LongTensor,
    events: Sequence[ControlTokenEvent],
    anchors: FrozenSet[str],
    rng: Optional[random.Random] = None,
) -> Tuple[torch.LongTensor, Dict[str, int]]:
    """全トークン教師ラベルに PSTL の状態トークン選択マスクを適用する。

    `base_labels` は通常の「アシスタント発話のみ教師信号、それ以外は -100」の
    ラベル (論文のキャプション/応答テキストに対する全トークン教師信号と同じもの)。
    この関数は、制御トークン自身の位置のみを追加で見て、PSTL で選択されなかった
    制御トークンのラベルを -100 に上書きする。キャプション/応答の本文トークンや
    画像/システム/ユーザー発話のラベルには一切触れない
    (論文 Appendix A.1: "Masking affects only the state loss ... Caption text ...
    retains full next-token supervision")。

    ========================================
    Shape
    ========================================
    入力:
        base_labels: (T,) int64  - 通常のフルスーパービジョンラベル (-100 or token id)
        events: 系列内に現れる制御トークンの (position, state) リスト
    出力:
        labels: (T,) int64 - PSTL 適用後のラベル
        stats: Dict - {"num_control_tokens", "num_selected", "quota"} (分析用)
    """
    labels = base_labels.clone()

    selected, quota = select_supervised_positions(events, anchors, rng)

    num_masked = 0
    for ev in events:
        if ev.position not in selected:
            if labels[ev.position].item() != IGNORE_INDEX:
                num_masked += 1
            labels[ev.position] = IGNORE_INDEX

    stats = {
        "num_control_tokens": len(events),
        "num_selected": len(selected),
        "num_masked_from_state_loss": num_masked,
        "quota": quota,
    }
    return labels, stats


# ============================================================
# 3. 損失計算 (標準 Cross-Entropy, ラベルは上で構築したものを使う)
# ============================================================

def compute_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> torch.Tensor:
    """次トークン予測の Cross-Entropy Loss。

    状態損失 (選択された制御トークン) とキャプション/応答の文章損失は、
    同じラベル列の中で "教師信号を与えるか (-100 かどうか)" が異なるだけで、
    数式上は単一の Cross-Entropy で統一的に計算される (論文 Eq.3 の L_state は
    this の対象を制御トークン位置に限定したもの、と解釈できる)。

    ========================================
    Shape
    ========================================
    入力:
        logits: (B, T, V)
        labels: (B, T) - -100 (除外) or トークンID
    出力:
        loss: scalar
    """
    shift_logits = logits[..., :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[..., 1:].contiguous()       # (B, T-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    return loss


def compute_loss_breakdown(
    logits: torch.Tensor,
    labels_dense: torch.LongTensor,
    labels_pstl: torch.LongTensor,
    control_positions: Sequence[int],
) -> Dict[str, float]:
    """PSTL の効果を検証するための分析用ヘルパー。

    `labels_dense`: 全ての制御トークンを教師信号に使った場合のラベル
                     (= PSTL を適用しない、全状態トークンを supervise するベースライン)
    `labels_pstl` : build_labels_with_pstl() の出力

    ========================================
    出力
    ========================================
    {
        "loss_dense": float,            - 全状態トークンを supervise した場合の損失
        "loss_pstl": float,             - PSTL 適用後の損失
        "supervised_ratio_dense": float,
        "supervised_ratio_pstl": float,
    }
    """
    loss_dense = compute_loss(logits, labels_dense).item()
    loss_pstl = compute_loss(logits, labels_pstl).item()

    shift_dense = labels_dense[..., 1:]
    shift_pstl = labels_pstl[..., 1:]
    total = shift_dense.numel()

    return {
        "loss_dense": loss_dense,
        "loss_pstl": loss_pstl,
        "supervised_ratio_dense": float((shift_dense != IGNORE_INDEX).sum()) / total,
        "supervised_ratio_pstl": float((shift_pstl != IGNORE_INDEX).sum()) / total,
    }


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    """PSTL のクォータ計算・サンプリング・ラベルマスキングの動作例。

    論文 Fig. "State Transition" の状況 (</Silence> が大量に続き、ときどき
    </Observe>/</Summary> が挿入される PHCM シーケンス) を模した合成データで検証する。
    """
    print("=== pstl_loss.py Example Usage ===\n")

    from control_tokens import TASK_SPECS

    rng = random.Random(0)

    # ------------------------------------------------------------
    # 1. 合成シーケンス: </Silence> を25回、間に </Observe> x3, </Summary> x2 を混在
    #    状態順序 (抜粋): S S S O S S S S S S S O S S S S S S S S S S O S S S U S S
    #    (S=Silence, O=Observe, U=Summary)。遷移はほぼ S->S (持続) と S->O/S->U (変化)。
    # ------------------------------------------------------------
    state_sequence = (
        ["</Silence>"] * 3 + ["</Observe>"]
        + ["</Silence>"] * 7 + ["</Observe>"]
        + ["</Silence>"] * 9 + ["</Observe>"]
        + ["</Silence>"] * 4 + ["</Summary>"]
        + ["</Silence>"] * 2 + ["</Summary>"]
    )
    events = [
        ControlTokenEvent(position=10 * i, state=s)
        for i, s in enumerate(state_sequence)
    ]
    print(f"制御トークン総数: {len(events)}")

    spec = TASK_SPECS["phcm"]
    groups = _group_by_transition(events)
    print("\n遷移グループごとの出現数 n_{X->Y}:")
    for (x, y), idxs in sorted(groups.items()):
        marker = " <- アンカー" if y in spec.anchors else ""
        print(f"  {x:12s} -> {y:12s} : n={len(idxs):2d}{marker}")

    quota = compute_supervision_quota(groups, spec.anchors)
    print(f"\n供給クォータ q = max_X max_{{a in anchors}} n_{{X->a}} = {quota}")

    selected, _ = select_supervised_positions(events, spec.anchors, rng)
    print(f"状態損失に使われる制御トークン数 |I| = {len(selected)} / {len(events)}")
    print(f"=> 教師信号を与える割合 = {len(selected) / len(events):.1%} "
          f"(論文は OVOBench 系ablationで 27.5% という数値を報告)")

    # ------------------------------------------------------------
    # 2. ラベル構築 + 損失への影響をダミー logits で確認
    # ------------------------------------------------------------
    T = state_sequence_len = events[-1].position + 1  # 系列長の下限
    V = 50
    base_labels = torch.full((T,), IGNORE_INDEX, dtype=torch.long)
    for ev in events:
        # 制御トークン自身は常に通常ラベル(ダミーのトークンID)を持つ、という前提
        base_labels[ev.position] = hash(ev.state) % V

    labels_pstl, stats = build_labels_with_pstl(base_labels, events, spec.anchors, rng)
    print(f"\nbuild_labels_with_pstl の統計: {stats}")

    logits = torch.randn(1, T, V)
    loss_dense = compute_loss(logits, base_labels.unsqueeze(0))
    loss_pstl = compute_loss(logits, labels_pstl.unsqueeze(0))
    print(f"\n全状態トークンを supervise した場合の loss: {loss_dense.item():.4f}")
    print(f"PSTL 適用後の loss:                         {loss_pstl.item():.4f}")

    breakdown = compute_loss_breakdown(
        logits,
        base_labels.unsqueeze(0),
        labels_pstl.unsqueeze(0),
        [ev.position for ev in events],
    )
    print(f"\n詳細: {breakdown}")


if __name__ == "__main__":
    example_usage()
