"""
Context Language Models (CLMs) - Context-as-a-File (モデルブロック)
================================================================

CLM の核心機構である「context as a file」を実装する。LM の生きた会話コンテキスト
(messages) をディスク上の1枚のテキストファイルにミラーし、LM が通常の Bash
コマンド (sed, python3 heredoc, cat 等) でそのファイルを自由に編集できるようにする。
編集結果は次ターンの会話コンテキストに同期される。

論文: Context Language Models (arXiv:2609.37725)
      - Sec 3.1 "Formal Definition and Implementation with Context as a File"
        Eq.2: c_{t+1} = f^CLM_theta(c_t)  (標準LMの c_{t+1} = c_t ⊕ f^LM_theta(c_t) との対比)
公式コード: gh/context-language-models/clm/clm_harness/
      - context_utils/context_string.py (render_editable / parse_back, stdlib依存なし)
      - context_env/edit_gate.py (fit/shrinkゲート)
      - context_env/env.py (ContextEnv.step, Harbor sandbox前提)
本ファイルは render_editable/parse_back/edit_gate を公式実装からほぼそのまま移植し
(これらは元から「stdlib onlyでユニットテスト可能」と明記されたロジックであり、
省略の必要が無い)、ContextEnv.step() のみ Harbor サンドボックス依存を取り除いて
実際のローカルファイル + 実サブプロセスで動くように再構成している。

============================================================
会話のレンダリング形式
============================================================
[[CTX_TURN 1 role=assistant]]
<そのターンでモデルが見たテキスト>

[[CTX_TURN 2 role=tool]]
<観測結果...>

保護された先頭 (system + 最初のタスク指示, protect=2) は編集領域に含まれない。
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 1. Context-as-a-string round trip (公式 context_string.py を移植)
# ============================================================

_HEADER_RE = re.compile(r"^\[\[CTX_TURN\s+(\d+)\s+role=([A-Za-z]+)\]\]\s*$", re.M)
_VALID_ROLES = {"system", "user", "assistant", "tool"}


def content_text(content: Any) -> Optional[str]:
    """メッセージの content (str または [{"type":"text","text":...}] 形式) を文字列化する。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text") and isinstance(
                part.get("text"), str
            ):
                parts.append(part["text"])
            else:
                return None
        return "".join(parts)
    return None


def render_editable(messages: List[Dict[str, Any]], protect: int = 2) -> str:
    """編集対象領域 (turns >= protect) をヘッダ付き文字列としてレンダリングする。

    ========================================
    入力/出力
    ========================================
    入力: messages (List[Dict]) - 全会話履歴
          protect (int) - 保護するプレフィックスのターン数 (system+taskで通常2)
    出力: str - "[[CTX_TURN n role=...]]\\n<text>" を "\\n\\n" で連結した文字列
    """
    blocks: List[str] = []
    for n, i in enumerate(range(protect, len(messages)), start=1):
        m = messages[i]
        text = content_text(m.get("content")) or ""
        blocks.append(f"[[CTX_TURN {n} role={m.get('role', 'user')}]]\n{text}")
    return "\n\n".join(blocks)


def parse_back(edited: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """任意に編集された文字列を、合法なメッセージ列に戻す。

    ``protected`` (system+task) は常にオリジナルから再固定される(改ざん不可)。
    残存するヘッダがそのまま各ターンの role を定義し、ヘッダ間の孤立テキストは
    user ターンとして扱われる。assistant 以外の role は全て user に畳み込まれる
    (tool呼び出し構造は編集を経ると再構築できないため)。

    ========================================
    入力/出力
    ========================================
    入力: edited (str) - 編集後の文字列
          protected (List[Dict]) - 保護されたプレフィックス (オリジナルから)
    出力: List[Dict] - 合法なメッセージ列 (空ターン除去・同role連続ターンの結合済み)
    """
    text = edited if isinstance(edited, str) else str(edited)
    matches = list(_HEADER_RE.finditer(text))

    sections: List[Tuple[str, str]] = []
    if not matches:
        body = text.strip()
        if body:
            sections.append(("user", body))
    else:
        lead = text[: matches[0].start()].strip()
        if lead:
            sections.append(("user", lead))
        for k, mt in enumerate(matches):
            role = mt.group(2).lower()
            if role not in _VALID_ROLES:
                role = "user"
            start = mt.end()
            end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((role, body))

    msgs: List[Dict[str, Any]] = [dict(m) for m in protected]
    for role, body in sections:
        msgs.append({"role": "assistant" if role == "assistant" else "user", "content": body})
    return _normalize(msgs, protect=len(protected))


def _normalize(messages: List[Dict[str, Any]], protect: int) -> List[Dict[str, Any]]:
    """空ターンを除去し、連続する同roleターンを結合する。保護プレフィックスは不変。"""
    kept: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i < protect:
            kept.append(m)
            continue
        c = content_text(m.get("content"))
        if c and c.strip():
            kept.append(m)

    merged: List[Dict[str, Any]] = []
    for i, m in enumerate(kept):
        if (
            merged
            and i >= protect
            and len(merged) > protect
            and m.get("role") in ("user", "assistant")
            and merged[-1].get("role") == m.get("role")
        ):
            prev = content_text(merged[-1].get("content")) or ""
            cur = content_text(m.get("content")) or ""
            merged[-1] = {"role": m["role"], "content": (prev + "\n\n" + cur).strip()}
        else:
            merged.append(m)
    return merged


# ============================================================
# 2. Edit Gate: 編集を受理するかどうかのルール (公式 edit_gate.py を移植)
# ============================================================

SHRINK_RULE = "A compaction must SHRINK context"


def fit_rule(limit: int) -> str:
    return f"An edit must FIT the {limit}-token limit"


def may_grow(*, allow_growth: bool, limit: int, after: int) -> bool:
    """fit ゲート下では、結果がまだ予算に収まるなら成長 (grow) を許す。"""
    return bool(allow_growth and limit and after <= limit)


def reject_rule(*, allow_growth: bool, limit: int) -> str:
    return fit_rule(limit) if (allow_growth and limit) else SHRINK_RULE


def rejected_note(*, baseline: int, after: int, before: int, rule: str) -> str:
    return (
        f"\n[LIVE_CTX_MAIN.txt: edit REJECTED -- it GREW context "
        f"~{baseline}->{after} tokens, so it was NOT applied (still ~{before}). "
        f"{rule}: you likely duplicated/appended content -- replace stale text "
        "with a SHORTER summary instead.]"
    )


# ============================================================
# 3. トークン数の近似カウント (公式は tiktoken o200k を使うが、ここでは
#    外部依存を増やさずに空白分割近似で代替する。実運用では本物のトークナイザに
#    差し替える前提で、カウント関数は差し替え可能にしている)
# ============================================================

def approx_token_count(messages: List[Dict[str, Any]]) -> int:
    """空白分割による近似トークン数 (実運用では tiktoken 等に差し替え可能)。"""
    total = 0
    for m in messages:
        text = content_text(m.get("content")) or ""
        total += max(1, len(text.split())) if text else 0
    return total


# ============================================================
# 4. ContextEnv: 実ファイル + 実サブプロセスで動く「書き込み可能なコンテキストOS」
# ============================================================

@dataclass
class StepResult:
    stdout: str
    return_code: int
    ctx_changed: bool
    note: str
    tokens_before: int
    tokens_after: int


class ContextEnv:
    """公式 ``context_env/env.py::ContextEnv`` の簡略再構成。

    Harbor サンドボックスの代わりに、ローカルの一時ファイルと実際の ``subprocess``
    でBashコマンドを実行する。アルゴリズム (ミラー書き込み -> コマンド実行 ->
    ミラー読み戻し -> edit_gate 判定 -> メッセージ列への反映) は公式と同一。
    """

    def __init__(
        self,
        *,
        protect: int = 2,
        context_budget_tokens: int = 0,
        allow_edit_growth: bool = True,
        command_timeout: int = 10,
        count_tokens=approx_token_count,
    ) -> None:
        self.protect = protect
        self.context_budget_tokens = int(context_budget_tokens or 0)
        self.allow_edit_growth = allow_edit_growth
        self.command_timeout = command_timeout
        self.count_tokens = count_tokens
        fd, tmp = tempfile.mkstemp(prefix="live_ctx_mirror_", suffix=".txt")
        import os as _os

        _os.close(fd)
        self.mirror_path = Path(tmp)

    def close(self) -> None:
        self.mirror_path.unlink(missing_ok=True)

    def write_mirror(self, messages: List[Dict[str, Any]]) -> str:
        rendered = render_editable(messages, protect=self.protect)
        self.mirror_path.write_text(rendered, encoding="utf-8")
        return rendered

    def read_mirror(self) -> str:
        return self.mirror_path.read_text(encoding="utf-8", errors="replace")

    def execute(self, command: str) -> subprocess.CompletedProcess:
        """実際にBashコマンドを実行する (サブプロセス, モックなし)。

        ``{{CTX_FILE}}`` というプレースホルダをミラーファイルの実パスに展開するので、
        LM が生成するコマンドは公式と同じ固定パス記法 (``/tmp/.live_ctx/LIVE_CTX_MAIN.txt``)
        を書く代わりにこのプレースホルダを使う。
        """
        resolved = command.replace("{{CTX_FILE}}", str(self.mirror_path))
        return subprocess.run(
            ["bash", "-c", resolved],
            capture_output=True,
            text=True,
            timeout=self.command_timeout,
        )

    def step(self, command: str, messages: List[Dict[str, Any]]) -> StepResult:
        """1ターン分の env 側処理: ミラー書き込み -> 実行 -> 読み戻し -> ゲート判定。

        ``messages`` はその場で (in-place に近い形で) 更新される。
        """
        rendered = self.write_mirror(messages)
        before = self.count_tokens(messages)

        proc = self.execute(command)
        read_back = self.read_mirror()

        ctx_changed = False
        note = ""
        if read_back.strip() != rendered.strip():
            note, ctx_changed = self._apply_edit(messages, rendered, read_back, before)

        after = self.count_tokens(messages)
        return StepResult(
            stdout=(proc.stdout or "") + (proc.stderr or ""),
            return_code=proc.returncode,
            ctx_changed=ctx_changed,
            note=note,
            tokens_before=before,
            tokens_after=after,
        )

    def _apply_edit(
        self, messages: List[Dict[str, Any]], rendered: str, read_back: str, before: int,
    ) -> Tuple[str, bool]:
        protected = messages[: self.protect]
        candidate = parse_back(read_back, protected)
        after = self.count_tokens(candidate)
        baseline = self.count_tokens(parse_back(rendered, protected))
        limit = self.context_budget_tokens

        if after > baseline and not may_grow(
            allow_growth=self.allow_edit_growth, limit=limit, after=after
        ):
            rule = reject_rule(allow_growth=self.allow_edit_growth, limit=limit)
            return rejected_note(baseline=baseline, after=after, before=before, rule=rule), False

        messages[:] = candidate
        if after > baseline:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied but it GREW context "
                f"~{baseline}->{after} tokens (it fits, so it was kept).]"
            )
        elif limit and after > limit:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied -- context ~{before}->{after} tokens, "
                f"but STILL OVER the ~{limit}-token limit. Compact more NOW.]"
            )
        else:
            note = (
                f"\n[LIVE_CTX_MAIN.txt: edit applied -- context ~{before}->{after} tokens, "
                f"{len(messages)} turns]"
            )
        return note, True


# ============================================================
# 使用例: 実際のBashコマンドで自分のコンテキストファイルを圧縮させる
# ============================================================

def example_usage() -> None:
    print("=== context_env.py Example Usage ===\n")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Track a KV store. SET k1=apple, SET k2=banana, SET k3=cherry."},
        {"role": "assistant", "content": "Stored k1=apple."},
        {"role": "tool", "content": "SET k1=apple -> ok"},
        {"role": "assistant", "content": "Stored k2=banana."},
        {"role": "tool", "content": "SET k2=banana -> ok"},
        {"role": "assistant", "content": "Stored k3=cherry."},
        {"role": "tool", "content": "SET k3=cherry -> ok"},
    ]

    env = ContextEnv(protect=2, context_budget_tokens=0, allow_edit_growth=True)

    print("--- 編集前のミラー (render_editable) ---")
    print(render_editable(messages, protect=2))

    # LM が実際に発行する Bash コマンド: ミラーファイルを python で書き換えて
    # 全ターンを1行のサマリーに圧縮する (本物のサブプロセス実行)。
    compaction_command = (
        "python3 - <<'PY'\n"
        "p = '{{CTX_FILE}}'\n"
        "open(p, 'w').write('[[CTX_TURN 1 role=assistant]]\\n'\n"
        "    'Summary: stored k1=apple, k2=banana, k3=cherry (3 keys).')\n"
        "PY"
    )

    result = env.step(compaction_command, messages)

    print(f"\nコマンド実行結果: return_code={result.return_code}, "
          f"ctx_changed={result.ctx_changed}")
    print(f"トークン数: {result.tokens_before} -> {result.tokens_after}")
    print(f"note: {result.note.strip()}")

    print("\n--- 編集後のメッセージ列 ---")
    for m in messages:
        print(f"  [{m['role']:9s}] {m['content'][:60]}")

    # 成長する編集はゲートに拒否されることを確認する。
    print("\n--- 成長する編集 (拒否されるはず) ---")
    env2 = ContextEnv(protect=2, context_budget_tokens=50, allow_edit_growth=True)
    messages2 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "short note"},
    ]
    grow_command = (
        "python3 - <<'PY'\n"
        "p = '{{CTX_FILE}}'\n"
        "text = open(p).read()\n"
        "open(p, 'w').write(text + ' ' + 'padding word ' * 80)\n"
        "PY"
    )
    result2 = env2.step(grow_command, messages2)
    print(f"ctx_changed={result2.ctx_changed}  note={result2.note.strip()[:120]}")

    env.close()
    env2.close()


if __name__ == "__main__":
    example_usage()
