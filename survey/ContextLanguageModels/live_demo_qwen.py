"""
Context Language Models (CLMs) - 実モデルで試す (Ollama + Qwen3:8b)
================================================================

CLM は「モデル自体を改造する手法」ではなく「既存の任意のLMに取り付ける
ハーネス」です。このスクリプトはローカルのOllama (`qwen3:8b`) に対して
実際に ``context_env.ContextEnv`` (本サーベイの Context-as-a-File 実装) を
取り付け、モデル自身が bash コマンドで自分のコンテキストファイルを
読み書きする様子をライブで確認します。

============================================================
前提
============================================================
- Ollama がローカルで起動済みで `qwen3:8b` が pull 済みであること
  (`ollama list` で確認; 無ければ `ollama pull qwen3:8b`)
- Ollama の native API (`/api/chat`) に `requests` で直接アクセスする
  (litellm 等の抽象化レイヤーは使わない; 追加インストールも不要)

============================================================
タスク設計 (論文 ContextBench の KV Store タスクを単純化したもの)
============================================================
環境 (このスクリプト) がSET操作を1つずつユーザー発話として送り込み、
モデルは好きなタイミングで bash コマンド (``` bash ... ``` ブロック) を
使ってコンテキストファイルを圧縮・退避してよい。最後にGET操作で
値を復元できるかを検証する。コンテキスト予算は小さく設定してあるので、
何もしないと ``[SYSTEM NOTICE]`` のナッジ/最終ターン通知で圧迫される
様子が実際に見える。

============================================================
使い方
============================================================
python live_demo_qwen.py                 # デフォルト設定で実行
python live_demo_qwen.py --budget 1200    # 予算を変えて試す
python live_demo_qwen.py --no-hint        # 「bashで圧縮できる」ヒントを与えない
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests

from context_env import ContextEnv, approx_token_count
from main_flow import BudgetController, ctx_turn_is_free

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"

_CODE_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def query_model(
    messages: List[Dict[str, Any]],
    max_tokens: int = 400,
    temperature: float = 0.3,
    timeout_sec: int = 180,
    max_retries: int = 2,
    think: bool = True,
) -> str:
    """Ollama の native API (``/api/chat``) を ``requests`` で直接叩く。

    litellm 等の抽象化は使わず、ペイロードとレスポンスをそのまま見える形にしている。
    ``think=True`` (既定) はQwen3のthinkingモードを有効にする。「いつ圧縮すべきか」
    はメタな判断なので、小さい8Bモデルでも推論させた方が実際に行動しやすい
    (遅くなる代わりに判断の質が上がる)。
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    last_error = ""
    for _ in range(max_retries):
        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            content = resp.json()["message"]["content"] or ""
            # thinkをOFFにしていても念のため<think>タグが混じった場合に備えて除去する。
            return _THINK_RE.sub("", content).strip()
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
    return f"[ERROR calling Ollama: {last_error}]"


def extract_command(text: str) -> Optional[str]:
    """応答テキストから ```bash ...``` ブロックのコマンドを取り出す。無ければ None。"""
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


def build_system_prompt(ctx_file: str, budget_tokens: int, give_hint: bool) -> str:
    hint = (
        f"\n\nYour live context is mirrored to the file {ctx_file}. You may rewrite that "
        "file with a bash command, and the new content becomes your own conversation "
        "history from the next turn on (it replaces everything except this system "
        "message and the first task message, which are protected). "
        "When you see a 'CONTEXT BUDGET NUDGE', immediately compact: rewrite the file to "
        "keep ONLY a short summary of the keys you have seen so far (e.g. 'stored k0..k5'), "
        "dropping the verbose SET history. Example:\n"
        "```bash\n"
        f"cat > {ctx_file} << 'EOF'\n"
        "[[CTX_TURN 1 role=assistant]]\n"
        "Summary: stored k0..k5 so far, values recorded.\n"
        "EOF\n"
        "```\n"
        "A rewrite that would make the file BIGGER than before is rejected, so only "
        "shrink it, never pad it."
        if give_hint else ""
    )
    return (
        "You are a helpful assistant tracking a key-value store as values stream in one "
        f"at a time. Your context budget is about {budget_tokens} tokens; you will be "
        "warned as you approach it, and the session ends if you go over.\n"
        "Respond in plain text. If you want to run a bash command this turn, put EXACTLY "
        "ONE command in a fenced block like:\n```bash\n<your command>\n```\n"
        "Only emit a bash block when you actually want to run a command. Otherwise just "
        "reply with a short acknowledgement, or (for a GET) the answer as "
        "`ANSWER <key>: <value>`." + hint
    )


# ============================================================
# KV Store 風タスクの操作列
# ============================================================

def make_kv_operations(n_sets: int, gets: List[int]) -> Tuple[List[str], Dict[str, str]]:
    """n_sets 個の SET 操作 + 指定インデックスへの GET 操作を作る。

    戻り値: (operations文字列のリスト, 正解の key->value 辞書)
    """
    words = ["apple", "banana", "cherry", "date", "elder", "fig", "grape", "honey",
             "ivy", "jade", "kiwi", "lime", "mango", "nectar", "olive", "peach"]
    kv = {f"k{i}": f"{words[i % len(words)]}-{i:02d}" for i in range(n_sets)}
    ops: List[str] = [f"SET k{i} = {kv[f'k{i}']}" for i in range(n_sets)]
    for idx in gets:
        ops.append(f"GET k{idx} (reply with `ANSWER k{idx}: <value>`)")
    return ops, kv


# ============================================================
# ライブ実行ループ
# ============================================================

def run_live_demo(
    budget_tokens: int = 500,
    n_sets: int = 14,
    get_indices: Optional[List[int]] = None,
    max_steps: int = 40,
    give_hint: bool = True,
    think: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    get_indices = get_indices if get_indices is not None else [2, 5, 8]
    operations, ground_truth = make_kv_operations(n_sets, get_indices)

    env = ContextEnv(protect=2, context_budget_tokens=budget_tokens, allow_edit_growth=True)
    budget = BudgetController(budget_tokens, reserve_tokens=max(20, budget_tokens // 20))

    system_prompt = build_system_prompt(str(env.mirror_path), budget_tokens, give_hint)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    op_queue: Deque[str] = deque(operations)
    pending_op: Optional[str] = None
    answers: Dict[str, str] = {}
    step = 0
    n_bash = 0
    n_ctx_syncs = 0
    consecutive_bash_for_op = 0
    #: 同じ保留中の操作に対して連続でbashターンを許す上限。これを超えたら強制的に
    #: 次の操作へ進める(モデルが同じ圧縮コマンドを繰り返すだけの無限ループ対策)。
    MAX_CONSECUTIVE_BASH = 2
    max_tokens = 900 if think else 300

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    while step < max_steps:
        if pending_op is None:
            if not op_queue:
                break
            pending_op = op_queue.popleft()
            messages.append({"role": "user", "content": pending_op})
            log(f"\n[env] -> {pending_op}")

        dec = budget.evaluate(messages)
        messages.extend(dec.messages)
        for m in dec.messages:
            log(f"[budget] {m['content']}")
        if dec.action == "stop":
            log("[budget] STOP: budget exceeded after final turn.")
            break

        raw = query_model(messages, think=think, max_tokens=max_tokens)
        log(f"[qwen3:8b] {raw[:300]}{' ...' if len(raw) > 300 else ''}")
        command = extract_command(raw)

        # ANSWER はbashコマンドと同じ応答に混在することもあるので、どちらの
        # 分岐でも独立にチェックする。
        m = re.search(r"ANSWER\s+(k\d+)\s*:\s*(.+)", raw, re.I)
        if m:
            answers[m.group(1).lower()] = m.group(2).strip().splitlines()[0].strip()

        if command and consecutive_bash_for_op < MAX_CONSECUTIVE_BASH:
            consecutive_bash_for_op += 1
            n_bash += 1
            result = env.step(command, messages)
            messages.append({"role": "assistant", "content": raw})
            tool_text = f"[bash exit={result.return_code}]\n{result.stdout.strip()}{result.note}"
            messages.append({"role": "user", "content": tool_text})
            log(f"[bash] $ {command}")
            log(f"[bash] {tool_text.strip()[:300]}")
            if result.ctx_changed:
                n_ctx_syncs += 1
            free = ctx_turn_is_free(
                ctx_changed=result.ctx_changed, return_code=result.return_code,
                task_output_empty=not result.stdout.strip(),
            )
            if not free:
                step += 1
                # Bashターンがタスク上の意味を持つ場合のみ、保留中の操作を消費する
                # (このデモではbashターンは常にコンテキスト管理用途と見なし、
                # 保留中の操作は次ラウンドでも再送する)。
        else:
            if command:
                log(f"[loop] 同じ操作に対するbashターンが{MAX_CONSECUTIVE_BASH}回連続したため、"
                    "次の操作へ強制的に進めます。")
            messages.append({"role": "assistant", "content": raw})
            pending_op = None
            consecutive_bash_for_op = 0
            step += 1

        log(f"[state] tokens≈{approx_token_count(messages)}/{budget_tokens}  "
            f"messages={len(messages)}  step={step}/{max_steps}")

    env.close()

    correct = sum(1 for k, v in ground_truth.items() if k in [f"k{i}" for i in get_indices]
                  and answers.get(k, "").strip() == v)
    total_gets = len(get_indices)
    return {
        "final_messages": messages,
        "final_tokens": approx_token_count(messages),
        "steps_used": step,
        "n_bash_commands": n_bash,
        "n_ctx_syncs": n_ctx_syncs,
        "n_nudges": budget.n_nudges,
        "answers": answers,
        "ground_truth": {f"k{i}": ground_truth[f"k{i}"] for i in get_indices},
        "correct": correct,
        "total_gets": total_gets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CLM を Ollama/Qwen3:8b に実際に取り付けて試す")
    parser.add_argument("--budget", type=int, default=500, help="コンテキスト予算(近似トークン数)")
    parser.add_argument("--n-sets", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--no-hint", action="store_true", help="bashで圧縮できるヒントを与えない")
    parser.add_argument("--no-think", action="store_true", help="Qwen3のthinkingモードを無効化(高速化)")
    args = parser.parse_args()

    print("=== live_demo_qwen.py: CLM x Ollama(qwen3:8b) ===")
    print(f"budget={args.budget} tokens, n_sets={args.n_sets}, hint={not args.no_hint}, "
          f"think={not args.no_think}\n")

    result = run_live_demo(
        budget_tokens=args.budget, n_sets=args.n_sets, max_steps=args.max_steps,
        give_hint=not args.no_hint, think=not args.no_think,
    )

    print("\n" + "=" * 60)
    print("=== 結果 ===")
    print(f"使用ステップ数: {result['steps_used']}")
    print(f"bashコマンド発行数: {result['n_bash_commands']} (うち実際にコンテキストが変化: {result['n_ctx_syncs']})")
    print(f"予算ナッジ回数: {result['n_nudges']}")
    print(f"最終トークン数(近似): {result['final_tokens']}")
    print(f"GET正答数: {result['correct']}/{result['total_gets']}")
    print(f"  正解: {result['ground_truth']}")
    print(f"  回答: {result['answers']}")


if __name__ == "__main__":
    main()
