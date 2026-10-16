"""
OneStreamer - Control Tokens (モデルブロック: 制御トークン語彙)
================================================================

OneStreamer が Qwen3-VL-4B-Instruct に追加する「制御トークン」の定義と、
タスクごとの状態語彙・出力アンカー集合 (Proactive State Transition Learning
で使う $\\mathcal{A}_\\tau$) をまとめたモジュールです。

論文: OneStreamer: Unifying Perception, Memory, and Proactive Response in
      Streaming Video Interaction (arXiv:2610.01762)
      - Sec 3.1 Overall Architecture (制御トークンの意味)
      - Appendix A.1 / Table "Task-specific control-state tokens and output-anchor
        sets for PSTL" (タスクごとの状態語彙とアンカー集合)
公式コード: gh/OneStreamer/Eval/VLMEvalKit/vlmeval/vlm/qwen3_ovotiming/inference_fast.py
            の SYSTEM prompt 内で </Silence> / </Standby> / </Response> が使われている
            ことが確認できる (</Observe>, </Summary> は PHCM 用で、公開コードの
            推論デモ (StreamingSession) には含まれず、学習コードも非公開のため、
            本ファイルの語彙・アンカー定義は論文 Table (Appendix A.1) の記述を
            そのまま実装したもの)。
公開チェックポイント: huggingface.co/MCG-NJU/OneStreamer-4B の `added_tokens.json` を
            実際に確認すると、Qwen3-VL-4B-Instruct の既存特殊トークン (151643-151668)
            に続けて本ファイルの5つの制御トークンがそのまま登録されている:
                </Response>: 151669, </Standby>: 151670, </Silence>: 151671,
                </Summary>: 151672, </Observe>: 151673
            また config.json の `architectures` は `Qwen3VLForConditionalGeneration`
            (transformers 標準クラス) のみで、カスタム modeling_*.py は一切同梱
            されていない。つまり実運用では
                `from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer`
            をそのまま使えばよく、本サーベイが Qwen3-VL 本体を再実装しない判断
            (→ 姉妹フォルダ survey/Qwen3VL を参照) は公開物の実態とも一致する。

============================================================
制御トークンの意味 (論文 Sec 3.1)
============================================================
</Silence>  : 継続観察。キャプションも応答も生成しない (全タスク共通)。
</Observe>  : 直接観察された物体・動作・場面・状態変化の "ローカル詳細キャプション" を開始。
</Summary>  : 完了したイベント/セグメントの "セマンティックサマリー" を開始。
</Standby>  : 証拠が出現しつつあるが、まだ回答するには不十分。
</Response> : ユーザーに見える応答 (または他タスク出力) を開始。

============================================================
Shape Convention
============================================================
T       : トークン列長
V       : (拡張後の) 語彙サイズ = 元の語彙サイズ + len(CONTROL_TOKENS)
D_llm   : LLM の埋め込み次元
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ============================================================
# 1. 制御トークン語彙
# ============================================================

# 公式推論コード (qwen3_ovotiming/inference_fast.py の SYSTEM prompt, および
# Inference/inference.py の `'</Response>' in answer` 判定) で実際に使われている
# リテラル文字列をそのまま採用する。
SILENCE = "</Silence>"
OBSERVE = "</Observe>"
SUMMARY = "</Summary>"
STANDBY = "</Standby>"
RESPONSE = "</Response>"

# tokenizer.add_special_tokens(...) にそのまま渡す追加トークン一覧。
# 順序はトークンID割り当て順になるため固定しておく。
CONTROL_TOKENS: Tuple[str, ...] = (SILENCE, OBSERVE, SUMMARY, STANDBY, RESPONSE)


@dataclass(frozen=True)
class TaskSpec:
    """1つの学習タスクファミリーが使う状態語彙とPSTL出力アンカー集合。

    Appendix A.1 Table "Task-specific control-state tokens and output-anchor
    sets for PSTL" を構造化したもの。
    """

    name: str
    state_vocab: Tuple[str, ...]      # そのタスクで出現しうる制御トークン全体
    anchors: FrozenSet[str]           # $\mathcal{A}_\tau$: 出力を開始する状態


# Appendix A.1 Table の3行をそのまま定義。
TASK_SPECS: Dict[str, TaskSpec] = {
    # PHCM (Proactive Hierarchical Caption Memory): 記憶形成タスク。
    # </Observe> と </Summary> がともに出力アンカー (キャプション本文が続く)。
    "phcm": TaskSpec(
        name="phcm",
        state_vocab=(SILENCE, OBSERVE, SUMMARY),
        anchors=frozenset({OBSERVE, SUMMARY}),
    ),
    # Proactive QA: </Response> のみが出力アンカー (回答本文が続く)。
    "proactive_qa": TaskSpec(
        name="proactive_qa",
        state_vocab=(SILENCE, STANDBY, RESPONSE),
        anchors=frozenset({RESPONSE}),
    ),
    # Proactive Interaction (継続説明・アラート・能動カウント等): </Standby> を持たず
    # </Silence> <-> </Response> のみで構成される、より単純な状態機械。
    "proactive_interaction": TaskSpec(
        name="proactive_interaction",
        state_vocab=(SILENCE, RESPONSE),
        anchors=frozenset({RESPONSE}),
    ),
}


def task_spec(task: str) -> TaskSpec:
    """タスク名から TaskSpec を取得する。未知のタスク名は即座に例外を出す。"""
    if task not in TASK_SPECS:
        raise KeyError(f"未知のタスク: {task!r}. 選択肢: {sorted(TASK_SPECS)}")
    return TASK_SPECS[task]


# ============================================================
# 2. トークナイザ拡張 (実際の HF tokenizer/model に対して動く実装)
# ============================================================

def add_control_tokens(tokenizer) -> Dict[str, int]:
    """制御トークンをトークナイザの特殊トークンとして追加する。

    実運用では `transformers.AutoTokenizer.from_pretrained("MCG-NJU/OneStreamer-4B")`
    (あるいは Qwen3-VL-4B-Instruct の tokenizer) をそのまま渡す。
    `tokenizer.add_special_tokens` は HuggingFace tokenizer の標準APIであり、
    ここでモックは一切していない。

    ========================================
    Shape / 戻り値
    ========================================
    戻り値: Dict[str, int] - 各制御トークン文字列 -> 新規に割り当てられたトークンID
        (既存の語彙サイズが V0 だった場合、通常は V0, V0+1, ..., V0+4 の5連番になる)
    """
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(CONTROL_TOKENS)}
    )
    if added != len(CONTROL_TOKENS):
        # 一部トークンが既に語彙に存在していた場合 (再実行時など)。
        # 実害はないが、呼び出し側が気づけるように警告だけ出す。
        print(
            f"[control_tokens] 追加されたのは {added}/{len(CONTROL_TOKENS)} 個 "
            "(一部は既存語彙に存在していた可能性があります)"
        )
    return {tok: tokenizer.convert_tokens_to_ids(tok) for tok in CONTROL_TOKENS}


def resize_embeddings_for_control_tokens(
    model: nn.Module,
    tokenizer,
    embedding_attr: str = "get_input_embeddings",
) -> Dict[str, int]:
    """制御トークン追加後に埋め込み行列をリサイズし、新規行を既存埋め込みの平均で初期化する。

    `model.resize_token_embeddings(len(tokenizer))` は HuggingFace の
    `PreTrainedModel` が実装する標準APIで、ここではその呼び出し方と
    「新規行を平均ベクトルで初期化する」という一般的な実践 (ランダム初期化より
    学習序盤の出力分布を暴れさせない) を実装している。

    ========================================
    Shape
    ========================================
    埋め込み行列: (V_old, D_llm) -> (V_new, D_llm)
        V_new = V_old + len(CONTROL_TOKENS) (すでに一部が語彙にあれば差分のみ増加)
    """
    token_ids = add_control_tokens(tokenizer)

    old_num_tokens = model.get_input_embeddings().weight.shape[0]  # V_old
    model.resize_token_embeddings(len(tokenizer))  # (V_old, D) -> (V_new, D)

    embed = model.get_input_embeddings()  # nn.Embedding, weight: (V_new, D)
    with torch.no_grad():
        mean_vec = embed.weight[:old_num_tokens].mean(dim=0)  # (D,)
        new_ids = sorted(v for v in token_ids.values() if v >= old_num_tokens)
        for new_id in new_ids:
            embed.weight[new_id] = mean_vec  # (D,) をそのまま代入

    # LM Head が embedding と重み共有していない場合は別途リサイズが必要になるが、
    # Qwen系はtie_word_embeddings=Trueが既定であり、resize_token_embeddingsが
    # lm_head側も同期する (HuggingFace実装に準拠)。
    return token_ids


# ============================================================
# 3. 生成テキストから制御トークンを検出するユーティリティ
# ============================================================

def find_control_token(text: str) -> Optional[str]:
    """デコードされた1ステップの生成テキストから、含まれる制御トークンを1つ返す。

    公式 Inference/inference.py の `'</Response>' in answer` 判定を一般化したもの。
    1ステップの生成に制御トークンは1つしか含まれない前提 (学習時にそうなるよう
    教師強制されている) で、最初に見つかったものを返す。
    """
    for tok in CONTROL_TOKENS:
        if tok in text:
            return tok
    return None


def split_control_and_payload(text: str) -> Tuple[Optional[str], str]:
    """生成テキストを (制御トークン, その後のペイロード文字列) に分割する。

    例: "</Observe> 赤いマグカップが机の上に置かれている" ->
        ("</Observe>", "赤いマグカップが机の上に置かれている")
    """
    tok = find_control_token(text)
    if tok is None:
        return None, text.strip()
    payload = text.split(tok, 1)[-1].strip()
    return tok, payload


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    """control_tokens.py の動作例。

    実際の HuggingFace tokenizer/model をネットワーク接続なしで用意できないため、
    `add_special_tokens` / `resize_token_embeddings` と同じメソッド名・挙動を持つ
    最小限の自作クラスで代替する (OneStreamer固有のロジックである
    add_control_tokens / resize_embeddings_for_control_tokens 自体は
    本物のHF APIをそのまま呼んでおり、モックはトークナイザ/モデル側のみ)。
    """
    print("=== control_tokens.py Example Usage ===\n")

    class _TinyTokenizer:
        """add_special_tokens / convert_tokens_to_ids のみ実装した最小トークナイザ。"""

        def __init__(self, base_vocab: Sequence[str]):
            self.vocab: List[str] = list(base_vocab)

        def add_special_tokens(self, mapping: Dict[str, List[str]]) -> int:
            added = 0
            for tok in mapping.get("additional_special_tokens", []):
                if tok not in self.vocab:
                    self.vocab.append(tok)
                    added += 1
            return added

        def convert_tokens_to_ids(self, tok: str) -> int:
            return self.vocab.index(tok)

        def __len__(self) -> int:
            return len(self.vocab)

    class _TinyModel(nn.Module):
        """get_input_embeddings / resize_token_embeddings のみ実装した最小モデル。"""

        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embed

        def resize_token_embeddings(self, new_vocab_size: int) -> nn.Embedding:
            old = self.embed
            new = nn.Embedding(new_vocab_size, old.embedding_dim)
            with torch.no_grad():
                new.weight[: old.weight.shape[0]] = old.weight
            self.embed = new
            return new

    base_vocab = [f"<tok{i}>" for i in range(100)]  # V_old = 100
    tokenizer = _TinyTokenizer(base_vocab)
    model = _TinyModel(vocab_size=len(base_vocab), hidden_size=16)

    print(f"拡張前の語彙サイズ V_old = {len(tokenizer)}")
    print(f"拡張前の埋め込み形状 = {tuple(model.get_input_embeddings().weight.shape)}")

    token_ids = resize_embeddings_for_control_tokens(model, tokenizer)

    print(f"\n拡張後の語彙サイズ V_new = {len(tokenizer)}")
    print(f"拡張後の埋め込み形状 = {tuple(model.get_input_embeddings().weight.shape)}")
    print(f"制御トークン -> ID: {token_ids}")

    # 新規行が「既存埋め込みの平均」で初期化されていることを確認。
    embed = model.get_input_embeddings().weight
    mean_vec = embed[:100].mean(dim=0)
    new_row = embed[token_ids[OBSERVE]]
    print(
        f"\n新規行(</Observe>)が平均ベクトルと一致するか: "
        f"{torch.allclose(new_row, mean_vec)}"
    )

    print("\n--- タスクごとの状態語彙とPSTLアンカー ---")
    for name, spec in TASK_SPECS.items():
        print(f"{name:22s} state_vocab={spec.state_vocab}  anchors={sorted(spec.anchors)}")

    print("\n--- 生成テキストのパース例 ---")
    for text in [
        "</Silence>",
        "</Observe> 赤いマグカップが机の上に置かれている",
        "</Response> 合計3回出現しました",
    ]:
        tok, payload = split_control_and_payload(text)
        print(f"  {text!r:55s} -> state={tok!r:12s} payload={payload!r}")


if __name__ == "__main__":
    example_usage()
