"""
InnerJev - 単一トークン選択肢へのReadout射影 + プロンプト構築 (モデルブロック)
================================================================

InnerJev は「通常のLLM」を改造せず、既存の causal LM に**推論時の読み出し方だけ**を
変える: フルボキャブラリ (15万語)のソフトマックスではなく、あらかじめ用意された
255個の単一トークン選択肢ラベル (A, B, ..., Z, AA, ..., ZZ) だけに制限した
"legal-option" ロジットを、最後の有効トークン位置の隠れ状態から1回の forward で
読み出す。

論文/公式README: InnerJev (gh/InnerJev/README.md)
      "Inference uses the original chat template, thinking disabled, ending in
      `The correct answer is (`. A single forward pass supplies the legal
      option-token logits, normalized at temperature one."
公式コード: gh/InnerJev/innerjev/{learning.py, prompts.py, infer.py}
      これらは全て素のPyTorch/transformers実装であり、本ファイルはそのロジックを
      ほぼそのまま移植している (`Readout` クラス、`option_labels`/`body`/`prompt`
      関数は公式と同一のアルゴリズム)。本物のQwen3.5ベースモデルの代わりに、
      小さなスタンドインのcausal LMに対して同じロジックを適用する。

============================================================
Shape Convention
============================================================
B        : バッチサイズ
T        : 入力トークン列長 (プロンプト全体, パディング後)
D        : LLMの隠れ次元
V        : LLMの全語彙サイズ
K        : 選択肢ラベルの語彙サイズ (255, 固定)
"""

from __future__ import annotations

import json
import string
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ANSWER_CUE = "The correct answer is ("


# ============================================================
# 1. プロンプト構築 (公式 prompts.py とほぼ同一)
# ============================================================

def text(value: Any) -> str:
    """state/question を文字列化する (文字列ならそのまま、構造化データならJSON化)。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def option_labels(tokenizer) -> List[str]:
    """単一トークンにエンコードされる255個の選択肢ラベル (A,B,...,Z,AA,...,ZZ) を選ぶ。

    実運用のQwen3.5トークナイザでは必ず255個ちょうど見つかる前提 (公式の検証と同じ)。
    """
    candidates = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    labels = [label for label in candidates
              if len(tokenizer.encode(label, add_special_tokens=False)) == 1][:255]
    return labels


def build_body(row: Dict[str, Any], labels: List[str]) -> str:
    """state/question/options から、選択肢をA/B/...でラベル付けした本文を作る。"""
    choices = "\n".join(f"({labels[i]}) {o['key']}: {o['desc']}" for i, o in enumerate(row["options"]))
    return f"State:\n{text(row['state'])}\nQuestion: {text(row['question'])}\nOptions:\n{choices}".rstrip()


def build_prompt(tokenizer, row: Dict[str, Any], labels: List[str]) -> str:
    """チャットテンプレート + ANSWER_CUE で終わる、生成なしの読み出し用プロンプトを作る。"""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_body(row, labels)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return prefix + ANSWER_CUE


# ============================================================
# 2. Readout: legal-option ロジットへの射影 (公式 learning.py::Readout と同一)
# ============================================================

class Readout(nn.Module):
    """causal LM をラップし、最後の有効トークン位置の隠れ状態から
    「255個の選択肢ラベルのトークンだけ」のロジットを読み出す。

    ========================================
    Shape
    ========================================
    入力: input_ids (B, T), attention_mask (B, T)
    出力: logits (B, K=255)

    ``full_rank=False`` (LoRA) の場合、出力埋め込み (LM head) の該当255行は
    学習開始時点のベースモデルから凍結コピーして保持する (LoRAはLM head自体を
    更新しないため、公式と同じ設計)。``full_rank=True`` の場合は現在の (学習中の)
    出力埋め込みをそのまま使う。
    """

    def __init__(self, base: nn.Module, label_ids: List[int], full_rank: bool):
        super().__init__()
        self.base = base
        self.full_rank = full_rank
        self.register_buffer("label_ids", torch.tensor(label_ids), persistent=False)
        if not full_rank:
            # LoRA: base.get_base_model() で元のPreTrainedModel (凍結済み) にアクセス。
            weight = base.get_base_model().get_output_embeddings().weight[label_ids].detach().clone()
            self.register_buffer("legal_weight", weight, persistent=False)  # (K, D)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        base = self.base if self.full_rank else self.base.get_base_model()
        backbone = getattr(base.model, "language_model", base.model)
        hidden = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        ).last_hidden_state  # (B, T, D)

        # 各サンプルの「最後の有効トークン位置」= attention_mask.sum(-1)-1 (右パディング前提)。
        last_valid = attention_mask.sum(-1) - 1  # (B,)
        slot = hidden[torch.arange(len(hidden), device=hidden.device), last_valid]  # (B, D)

        weight = (
            base.get_output_embeddings().weight.index_select(0, self.label_ids)  # (K, D): 現在の重み (full-rank)
            if self.full_rank else self.legal_weight                              # (K, D): 凍結スナップショット (LoRA)
        )
        return F.linear(slot, weight).float()  # (B, K): 255個の選択肢トークンだけのロジット


# ============================================================
# 3. デモ用の極小causal LM (実運用ではQwen3.5-4B/27B等に差し替える)
# ============================================================

class TinyTransformerStack(nn.Module):
    """``get_input_embeddings``/``forward(..., return_dict=True).last_hidden_state``
    というHF PreTrainedModel互換の最小インターフェースを持つ極小Transformer。"""

    def __init__(self, vocab_size: int, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * 2, batch_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        x = self.embed_tokens(input_ids)  # (B, T, D)
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return SimpleNamespace(last_hidden_state=x)


class TinyCausalLM(nn.Module):
    """``Readout`` が要求する最小インターフェース (``model``, ``get_output_embeddings``,
    ``get_input_embeddings``) を持つ極小causal LM (デモ専用スタンドイン)。"""

    def __init__(self, vocab_size: int, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.model = TinyTransformerStack(vocab_size, dim, n_layers, n_heads)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== model.py Example Usage ===\n")
    torch.manual_seed(0)

    class _TinyTokenizer:
        """最小限のトークナイザ (A-Z,AA-ZZの単一トークン化保証 + 固定語彙)。"""

        def __init__(self):
            self.vocab = ["<pad>"] + list(string.ascii_uppercase) + \
                [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]

        def encode(self, text_: str, add_special_tokens: bool = True) -> List[int]:
            if text_ in self.vocab:
                return [self.vocab.index(text_)]
            # 単語をハッシュで固定サイズ領域に写す(デモ用の簡易トークナイザ)。
            offset = len(self.vocab)
            return [offset + (hash(w) % 500) for w in text_.split()] or [0]

        def __call__(self, texts: List[str], padding=True, return_tensors="pt", add_special_tokens=False):
            encoded = [self.encode(t, add_special_tokens) for t in texts]
            max_len = max(len(e) for e in encoded)
            input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
            attn = torch.zeros(len(texts), max_len, dtype=torch.long)
            for i, ids in enumerate(encoded):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                attn[i, : len(ids)] = 1
            return {"input_ids": input_ids, "attention_mask": attn}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return messages[0]["content"] + "\n"

    tokenizer = _TinyTokenizer()
    labels = option_labels(tokenizer)
    print(f"単一トークン選択肢ラベル数: {len(labels)} (先頭5個: {labels[:5]})")

    row = {
        "type": "choice", "state": "The user is reviewing a code change.",
        "question": "Which action should the assistant take next?",
        "options": [
            {"key": "approve", "desc": "Approve the pull request."},
            {"key": "request_changes", "desc": "Request changes with comments."},
            {"key": "comment", "desc": "Leave a neutral comment only."},
        ],
    }
    prompt_text = build_prompt(tokenizer, row, labels)
    print(f"\n構築されたプロンプト (末尾): ...{prompt_text[-60:]!r}")

    vocab_size = len(tokenizer.vocab) + 500
    base = TinyCausalLM(vocab_size=vocab_size, dim=32, n_layers=2, n_heads=4)
    label_ids = [tokenizer.encode(label)[0] for label in labels]

    readout = Readout(base, label_ids, full_rank=True)
    encoded = tokenizer([prompt_text], padding=True, return_tensors="pt")
    logits = readout(encoded["input_ids"], encoded["attention_mask"])
    print(f"\nReadout出力 logits shape: {tuple(logits.shape)}  (B=1, K=255)")

    n = len(row["options"])
    probs = logits[0, :n].softmax(-1)
    print(f"先頭{n}個(実際の選択肢数)の確率: {probs.tolist()}")

    loss = logits.square().mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in base.parameters() if p.grad is not None)
    print(f"\nloss={loss.item():.4f}  backboneの勾度ノルム合計={grad_norm:.4f} (0でなければ逆伝播成功)")


if __name__ == "__main__":
    example_usage()
