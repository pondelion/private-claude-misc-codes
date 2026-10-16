"""
Context Language Models (CLMs) - In-Context Skill Evolution (ICL)
================================================================================

論文 Sec 3.2 "In-Context Learning ... for CLMs" で導入される、テキスト進化
(prompt-evolution loop, GEPA 系) によるスキル文書 (SKILL.md 相当) の最適化を
実装する。各ラウンドで: (1) 現在のスキルでロールアウトし、(2) 対比的なノートを
作り、(3) プロポーザが候補スキルを複数生成し、(4) 開発セットで評価し、
(5) 現行チャンピオンに対するゲート判定で合否を決める。

論文: Context Language Models (arXiv:2609.37725)
      - Sec 3.2 Eq.4: skl* = argmax_skl E_x[R(tau(x; skl))]
      - Appendix "In-context evolution" (開発セットでの選別手順, Pareto frontier)
公式コード: gh/context-language-models/clm/clm_icl/evolve.py
      `score()`, `diff_se()`, `gate()`, `pick()`, `pareto()` は外部依存のない
      純粋関数であり、公式実装からそのまま移植している (ロールアウト実行・
      プロポーザLLM呼び出しは Harbor/litellm に依存するため、本ファイルでは
      pluggable な callable に置き換えている)。

============================================================
ゲート規則 (公式 gate() のdocstringより)
============================================================
d = acc(candidate) - acc(incumbent), SE = diff_se(candidate, incumbent) の標準誤差
  - d >  SE: 合格 (有意に高精度)
  - d < -SE: 不合格 (有意に低精度)
  - それ以外 (統計的に同等): コストが低い場合のみ合格
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ============================================================
# 1. Episode / Score (公式 clm_icl/tasks.py, evolve.py の Episode/Score を移植)
# ============================================================

@dataclass
class Episode:
    """1つのタスクを1つのスキルで1回実行した結果。"""

    task_id: str
    rep: int
    reward: Optional[float]
    cost: Optional[float] = None
    cost_metric: Optional[str] = None
    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.reward is not None


@dataclass
class Score:
    n: int
    n_valid: int
    acc: Optional[float]
    cost: Optional[float]
    cost_metric: Optional[str]
    task_rewards: Dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.n > 0 and self.n_valid == self.n


def score(episodes: List[Episode]) -> Score:
    """採点済みエピソードの平均報酬。コストは単一メトリックで全エピソードに
    値がある場合のみ平均を取る (そうでなければ比較不能として None)。"""
    valid = [e for e in episodes if e.valid]
    acc = sum(e.reward for e in valid) / len(valid) if valid else None
    metrics = {e.cost_metric for e in valid}
    cost = metric = None
    if valid and len(metrics) == 1 and all(e.cost is not None for e in valid):
        cost, metric = sum(e.cost for e in valid) / len(valid), metrics.pop()
    per_task: Dict[str, List[float]] = {}
    for e in valid:
        per_task.setdefault(e.task_id, []).append(e.reward)
    task_rewards = {t: sum(v) / len(v) for t, v in per_task.items()}
    return Score(len(episodes), len(valid), acc, cost, metric, task_rewards)


def _sem(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def diff_se(a: Score, b: Score) -> Tuple[float, str]:
    """acc(a)-acc(b) の標準誤差。同じタスク集合なら paired (ペア差の標準誤差)、
    そうでなければ unpaired (両者の標準誤差を合成)。"""
    ta, tb = a.task_rewards or {}, b.task_rewards or {}
    if ta and set(ta) == set(tb):
        return _sem([ta[t] - tb[t] for t in ta]), "paired"
    return math.hypot(_sem(list(ta.values())), _sem(list(tb.values()))), "unpaired"


# ============================================================
# 2. ゲートと選択 (公式 gate() / pick() / pareto() をそのまま移植)
# ============================================================

def gate(cand: Score, inc: Score, eps: float = 1e-9) -> Tuple[bool, str]:
    if not cand.complete:
        return False, f"only {cand.n_valid}/{cand.n} episodes graded"
    if inc.acc is None:
        return True, "incumbent has no accuracy"
    d = cand.acc - inc.acc
    se, kind = diff_se(cand, inc)
    acc_txt = f"acc {cand.acc:.4f} vs {inc.acc:.4f}, diff {d:+.4f}, SE {se:.4f} ({kind})"
    if d > se + eps:
        return True, f"higher accuracy: {acc_txt}"
    if d < -se - eps:
        return False, f"lower accuracy: {acc_txt}"
    if cand.cost is None or inc.cost is None or cand.cost_metric != inc.cost_metric:
        return False, f"tie within 1 SE, cost not comparable: {acc_txt}"
    cost_txt = f"cost {cand.cost:.4g} vs {inc.cost:.4g} {cand.cost_metric}"
    if cand.cost < inc.cost:
        return True, f"tie within 1 SE, lower cost: {acc_txt}; {cost_txt}"
    return False, f"tie within 1 SE, not cheaper: {acc_txt}; {cost_txt}"


def pick(passing: List[Tuple[str, Score]]) -> str:
    """合格した候補の中で最高精度のものを基準に、1SE以内を同等とみなし、
    その中で最安のものを採用する。"""
    best = max(passing, key=lambda x: x[1].acc)[1]
    tied = [(sid, s) for sid, s in passing if best.acc - s.acc <= diff_se(best, s)[0] + 1e-9]
    inf = float("inf")
    return min(tied, key=lambda x: (x[1].cost if x[1].cost is not None else inf, -x[1].acc))[0]


def pareto(scores: Dict[str, Score]) -> List[str]:
    """他のどのスキルにも精度・コストの両方で支配されていないスキル一覧。"""
    pts = {k: s for k, s in scores.items() if s.acc is not None and s.cost is not None}
    out = []
    for k, s in pts.items():
        dominated = any(
            o.cost_metric == s.cost_metric and o.acc >= s.acc and o.cost <= s.cost
            and (o.acc > s.acc or o.cost < s.cost)
            for j, o in pts.items() if j != k
        )
        if not dominated:
            out.append(k)
    return sorted(out, key=lambda k: pts[k].cost)


# ============================================================
# 3. 進化ループ (公式 Evolution クラスの本質的な部分)
# ============================================================

#: 1タスクを1スキルで実行して Episode を返す callable (実運用では
#: Harbor + ClmAgent のロールアウト; ここでは pluggable)
RunEpisode = Callable[[str, str, int], Episode]  # (skill_text, task_id, rep) -> Episode
#: 現行スキル文書 + 対比ノート + タスク説明からN個の候補スキル文書を生成する callable
#: (実運用では外部LLMへのプロンプト; ここでは pluggable)
Proposer = Callable[[str, str, int], List[str]]  # (incumbent_skill, note, n) -> [skill_text, ...]


@dataclass
class EvolutionResult:
    incumbent_skill: str
    incumbent_score: Score
    history: List[dict]


def build_contrastive_note(episodes: List[Episode], k: int = 5) -> str:
    """良い実行 vs 悪い実行を対比するノート (公式 note.build_note の簡略版)。"""
    valid = [e for e in episodes if e.valid]
    if not valid:
        return "(no valid episodes to learn from)"
    ranked = sorted(valid, key=lambda e: e.reward, reverse=True)
    best, worst = ranked[: k // 2 + 1], ranked[-(k // 2) :]
    lines = ["Best episodes:"]
    lines += [f"  - {e.task_id} (rep {e.rep}): reward={e.reward:.3f}, cost={e.cost}" for e in best]
    lines.append("Worst episodes:")
    lines += [f"  - {e.task_id} (rep {e.rep}): reward={e.reward:.3f}, cost={e.cost}" for e in worst]
    return "\n".join(lines)


class SkillEvolution:
    """公式 `Evolution` クラスの中核ループ (Harbor依存を除去した再構成)。"""

    def __init__(
        self,
        task_ids: List[str],
        run_episode: RunEpisode,
        proposer: Proposer,
        *,
        n_candidates: int = 4,
        reps: int = 2,
    ) -> None:
        self.task_ids = task_ids
        self.run_episode = run_episode
        self.proposer = proposer
        self.n_candidates = n_candidates
        self.reps = reps
        self.skills: Dict[str, str] = {"base": ""}
        self.incumbent = "base"
        self.history: List[dict] = []

    def _evaluate(self, skill_text: str) -> Score:
        episodes = [
            self.run_episode(skill_text, tid, rep)
            for tid in self.task_ids
            for rep in range(self.reps)
        ]
        return score(episodes), episodes

    def run_round(self, k: int) -> dict:
        inc_text = self.skills[self.incumbent]
        inc_score, train_eps = self._evaluate(inc_text)
        note = build_contrastive_note(train_eps)

        candidate_texts = self.proposer(inc_text, note, self.n_candidates)
        results, passing = [], []
        for i, text in enumerate(candidate_texts):
            sid = f"r{k}-{i}"
            self.skills[sid] = text
            cand_score, _ = self._evaluate(text)
            ok, why = gate(cand_score, inc_score)
            results.append({"skill": sid, "acc": cand_score.acc, "cost": cand_score.cost,
                             "passed": ok, "reason": why})
            if ok:
                passing.append((sid, cand_score))

        winner = pick(passing) if passing else None
        if winner:
            self.incumbent = winner
        rec = {"round": k, "incumbent_before": inc_score.acc, "candidates": results,
               "winner": winner}
        self.history.append(rec)
        return rec

    def run(self, rounds: int) -> EvolutionResult:
        for k in range(1, rounds + 1):
            self.run_round(k)
        final_score, _ = self._evaluate(self.skills[self.incumbent])
        return EvolutionResult(self.incumbent, final_score, self.history)


# ============================================================
# 使用例: KV Store 風タスクで、ベースラインから進化させる
# ============================================================

def example_usage() -> None:
    """合成タスクに対し、決定論的な「仮想グレーダー」を使ってスキル進化を実際に走らせる。

    各スキル文書は「サンプル内で何か特定の語を含むか」でその効果を近似する
    (本物のLLMロールアウトの代わり)。進化ループ自体(ノート作成・ゲート判定・
    pick/pareto)は公式のものと全く同じ処理を実行する。
    """
    print("=== skill_evolution.py Example Usage ===\n")

    import random

    rng = random.Random(0)
    task_ids = [f"task{i}" for i in range(6)]

    # 「仮想グレーダー」: スキル文書に 'offload' という語があれば高報酬・低コスト、
    # 'verbose' という語があれば低報酬・高コストになる合成モデル。
    def run_episode(skill_text: str, task_id: str, rep: int) -> Episode:
        base_reward = 0.4 + rng.uniform(-0.05, 0.05)
        base_cost = 100.0 + rng.uniform(-5, 5)
        if "offload" in skill_text:
            base_reward += 0.35
            base_cost -= 40
        if "verbose" in skill_text:
            base_reward -= 0.1
            base_cost += 30
        return Episode(task_id=task_id, rep=rep, reward=max(0.0, min(1.0, base_reward)),
                        cost=max(1.0, base_cost), cost_metric="flops")

    # プロポーザ: ノートに "Worst" が多いことを見て 'offload' を含む候補を混ぜる
    # (実運用では外部LLMがノートを読んで自然言語のスキル文書を書く)。
    proposal_round = {"n": 0}

    def proposer(incumbent_skill: str, note: str, n: int) -> List[str]:
        proposal_round["n"] += 1
        r = proposal_round["n"]
        candidates = [
            incumbent_skill + f" Also, remember to offload large blocks (round {r}, cand 0).",
            incumbent_skill + f" Be extremely verbose and keep everything (round {r}, cand 1).",
            incumbent_skill + f" Offload promptly and keep notes short (round {r}, cand 2).",
        ]
        return candidates[:n]

    evo = SkillEvolution(task_ids, run_episode, proposer, n_candidates=3, reps=3)
    result = evo.run(rounds=3)

    print("--- ラウンドごとの結果 ---")
    for rec in result.history:
        print(f"round {rec['round']}: incumbent_acc_before={rec['incumbent_before']:.3f}")
        for c in rec["candidates"]:
            mark = "PASS" if c["passed"] else "reject"
            print(f"    {c['skill']:10s} acc={c['acc']:.3f} cost={c['cost']:.1f} [{mark}] {c['reason']}")
        print(f"  -> winner: {rec['winner']}")

    print(f"\n最終インカンベント: {result.incumbent_skill!r}")
    print(f"最終スコア: acc={result.incumbent_score.acc:.3f}, cost={result.incumbent_score.cost:.1f}")
    print(f"採用されたスキル文書: {evo.skills[result.incumbent_skill]!r}")


if __name__ == "__main__":
    example_usage()
