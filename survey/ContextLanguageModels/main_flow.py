"""
Context Language Models (CLMs) - メインフロー (ClmAgent の1タスク実行ループ)
================================================================================

``context_env.py`` の ContextEnv と、コンテキスト予算の状態機械 (BudgetController)
を統合し、公式 `clm_agent/harness.py::ClmAgent.run()` の本質的なループ構造を
再現する。LM 呼び出し部分だけを差し替え可能な `policy` callable とし、
それ以外 (ミラー同期、Bashコマンドの実サブプロセス実行、予算ナッジ、
ロールバック、純粋な圧縮ターンの無償化) は本物のロジックで実際に動かす。

論文: Context Language Models (arXiv:2609.37725) Sec 3.1
公式コード: gh/context-language-models/clm/clm_harness/
      - clm_agent/harness.py (ClmAgent.run, 本ファイルの主な参照元)
      - utils/budget.py (BudgetController; 本ファイルは主要ロジックのみ簡略移植)
本ファイルは Harbor サンドボックス・litellm 経由のLLM呼び出し・スキル読み込み等の
周辺機能を除き、「コンテキスト予算の管理 + ミラーファイル同期」という中核だけを
抜き出して実際に動かす。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from context_env import ContextEnv, approx_token_count

# ============================================================
# 1. BudgetController (公式 utils/budget.py の中核ロジックを簡略移植)
# ============================================================

@dataclass
class BudgetDecision:
    token_count: int
    action: str  # "continue" | "final_turn" | "stop"
    messages: List[Dict[str, Any]] = field(default_factory=list)


class BudgetController:
    """コンテキスト予算の決定論的な状態機械 (公式 BudgetController の簡略版)。

    * 予算の 25/50/75% で段階的にナッジを送る (再発火は圧縮後にのみ再武装)。
    * strict_target (= budget - reserve) を超えたら「最終ターン」を1回許可し、
      それでも超えたら run を停止する。
    """

    def __init__(
        self,
        budget_tokens: int,
        reserve_tokens: int = 20,
        nudge_ratios: Optional[List[float]] = None,
        count_tokens=approx_token_count,
    ) -> None:
        self.budget = int(budget_tokens or 0)
        self.reserve = int(reserve_tokens)
        self.strict_target = self.budget - self.reserve if self.budget else 0
        self.nudge_ratios = sorted(f for f in (nudge_ratios or [0.25, 0.5, 0.75]) if 0.0 < f < 1.0)
        self.count_tokens = count_tokens
        self._nudged: set = set()
        self._finalized = False
        self.n_nudges = 0

    @property
    def enabled(self) -> bool:
        return self.budget > 0

    def evaluate(self, messages: List[Dict[str, Any]]) -> BudgetDecision:
        if not self.enabled:
            return BudgetDecision(0, "continue")
        tc = self.count_tokens(messages)
        self._nudged = {f for f in self._nudged if tc >= int(self.budget * f)}

        if tc > self.strict_target:
            if self._finalized:
                return BudgetDecision(tc, "stop")
            self._finalized = True
            return BudgetDecision(
                tc, "final_turn",
                messages=[{"role": "user", "content": self._final_message(tc)}],
            )
        self._finalized = False

        crossed = [f for f in self.nudge_ratios if tc >= int(self.budget * f)]
        to_fire = [f for f in crossed if f not in self._nudged]
        if to_fire:
            tier = max(to_fire)
            self._nudged |= set(crossed)
            self.n_nudges += 1
            return BudgetDecision(
                tc, "continue",
                messages=[{"role": "user", "content": self._nudge_message(tier, tc)}],
            )
        return BudgetDecision(tc, "continue")

    def _final_message(self, tokens: int) -> str:
        return (
            f"[SYSTEM NOTICE] You have reached your context budget ({tokens}/{self.budget} "
            "tokens). This is your FINAL turn before the session ends."
        )

    def _nudge_message(self, frac: float, tokens: int) -> str:
        pct = int(round(frac * 100))
        return (
            f"CONTEXT BUDGET NUDGE: context is at ~{pct}% of your {self.budget}-token "
            f"budget ({tokens} tokens). Consider compacting your context file."
        )


# ============================================================
# 2. "何もしていないのに課金されない" ための無償ターン判定 (公式 ctx_turn_is_free)
# ============================================================

def ctx_turn_is_free(*, ctx_changed: bool, return_code: int, task_output_empty: bool) -> bool:
    """純粋な圧縮ターン (ミラーが変わり、タスク出力が空で、exit 0) はタスクの
    ステップを消費しない (公式 harness.py と同一の判定基準)。"""
    return bool(ctx_changed and task_output_empty and return_code == 0)


# ============================================================
# 3. ClmAgentLoop: 1タスクの実行ループ
# ============================================================

#: policy は現在のメッセージ列を受け取り、次に実行するBashコマンド (または
#: "DONE: ..." で終了) を返す callable。実運用では litellm 経由のLLM呼び出し。
Policy = Callable[[List[Dict[str, Any]]], str]


@dataclass
class RunResult:
    messages: List[Dict[str, Any]]
    steps_used: int
    free_ctx_turns: int
    n_nudges: int
    stopped_reason: str


class ClmAgentLoop:
    def __init__(
        self,
        policy: Policy,
        *,
        max_steps: int = 20,
        context_budget_tokens: int = 0,
        reserve_tokens: int = 20,
        protect: int = 2,
        command_timeout: int = 10,
    ) -> None:
        self.policy = policy
        self.max_steps = max_steps
        self.budget = BudgetController(context_budget_tokens, reserve_tokens=reserve_tokens)
        self.env = ContextEnv(
            protect=protect, context_budget_tokens=context_budget_tokens, command_timeout=command_timeout,
        )

    def run(self, system_prompt: str, task_prompt: str) -> RunResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        step = 0
        free_ctx_turns = 0
        stopped_reason = "max_steps"

        while step < self.max_steps:
            dec = self.budget.evaluate(messages)
            messages.extend(dec.messages)
            if dec.action == "stop":
                stopped_reason = "budget_stop"
                break

            command = self.policy(messages)
            if command.startswith("DONE"):
                stopped_reason = "done"
                break

            result = self.env.step(command, messages)
            messages.append({"role": "assistant", "content": f"$ {command}"})
            task_output_empty = not result.stdout.strip()
            messages.append({"role": "tool", "content": result.stdout + result.note})

            if ctx_turn_is_free(
                ctx_changed=result.ctx_changed,
                return_code=result.return_code,
                task_output_empty=task_output_empty,
            ):
                free_ctx_turns += 1
            else:
                step += 1

        self.env.close()
        return RunResult(
            messages=messages, steps_used=step, free_ctx_turns=free_ctx_turns,
            n_nudges=self.budget.n_nudges, stopped_reason=stopped_reason,
        )


# ============================================================
# 使用例: KV Store 風のタスクを、無限に蓄積するだけの素朴なポリシーと、
# 予算ナッジを受けて圧縮する賢いポリシーで比較する
# ============================================================

def example_usage() -> None:
    print("=== main_flow.py Example Usage ===\n")

    kv_pairs = [(f"k{i}", f"value_{i}_padding_text") for i in range(12)]

    class NaivePolicy:
        """コンテキストを気にせずデータを積み続けるだけのポリシー(ベースライン)。"""

        def __init__(self, pairs):
            self.pairs = list(pairs)
            self.i = 0

        def __call__(self, messages: List[Dict[str, Any]]) -> str:
            if self.i >= len(self.pairs):
                return "DONE"
            k, v = self.pairs[self.i]
            self.i += 1
            return f"echo 'stored {k}={v}'"

    class CompactingPolicy:
        """予算ナッジ(直前のtoolメッセージにNUDGEが含まれる)を見たら、実際に
        ミラーファイルを圧縮するBashコマンドを返すポリシー。"""

        def __init__(self, pairs):
            self.pairs = list(pairs)
            self.i = 0
            self.stored: Dict[str, str] = {}

        def __call__(self, messages: List[Dict[str, Any]]) -> str:
            last = messages[-1]["content"] if messages else ""
            if "CONTEXT BUDGET NUDGE" in last:
                return (
                    "python3 - <<'PY'\n"
                    "p = '{{CTX_FILE}}'\n"
                    f"open(p, 'w').write('[[CTX_TURN 1 role=assistant]]\\n'\n"
                    f"    'Summary: stored {len(self.stored)} keys so far.')\n"
                    "PY"
                )
            if self.i >= len(self.pairs):
                return "DONE"
            k, v = self.pairs[self.i]
            self.stored[k] = v
            self.i += 1
            return f"echo 'stored {k}={v}'"

    for name, policy_cls in [("NaivePolicy (圧縮しない)", NaivePolicy), ("CompactingPolicy (圧縮する)", CompactingPolicy)]:
        loop = ClmAgentLoop(
            policy=policy_cls(kv_pairs), max_steps=30,
            context_budget_tokens=80, reserve_tokens=10,
        )
        result = loop.run(
            system_prompt="You are a helpful assistant managing a KV store.",
            task_prompt="Store the following key-value pairs as instructed, one per turn.",
        )
        print(f"--- {name} ---")
        print(f"  stopped_reason={result.stopped_reason}  steps_used={result.steps_used}  "
              f"free_ctx_turns={result.free_ctx_turns}  n_nudges={result.n_nudges}")
        print(f"  最終メッセージ数: {len(result.messages)}")
        print(f"  最終トークン数(近似): {approx_token_count(result.messages)}")
        print()


if __name__ == "__main__":
    example_usage()
