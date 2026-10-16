"""
InnerJev - pandasデータフレームからのLoRA自己蒸留ファインチューニング例
================================================================

公式の学習データは JSONL (1行=1問) で配布されるが、ここでは同じスキーマを
pandas DataFrame として持っている前提で、素のPyTorch + peft(LoRA) で
学習ループを組む最小例を示す。

公式スキーマ (gh/InnerJev/innerjev/data.py::read_rows が検証する項目):
    id            : str  (一意なID)
    type          : "choice" | "noul" | "score"
    state         : str または 構造化データ (JSON化される)
    question      : str
    options       : [{"key": str, "desc": str}, ...]  (2〜255個)
    train_target  : List[float]  (optionsと同じ長さ, 合計1, 教師分布q)
    p0            : List[float]  (optionsと同じ長さ, 合計1, 蒸留前ベースラインp0)
    gold_target   : List[float]  (optionsと同じ長さ, 合計1, 評価/hard-Noul判定用)

本物の学習 (train.py) は Qwen3.5-4B/27B + LoRA(r=64,alpha=128) を
DDPで学習するが、本ファイルはモデルサイズを ``model.py`` の
``TinyCausalLM`` に差し替えた以外は、同じ Readout / loss_terms /
LoRA構成のロジックをそのまま使う (スタンドインである理由はモデルの
事前学習済み重みが非公開/巨大なためであり、学習ロジック自体は公式と同一)。

============================================================
Shape Convention
============================================================
B : バッチサイズ
T : パディング後のトークン列長 (バッチ内最大長)
K : 選択肢ラベル語彙サイズ (255, 固定)
"""

from __future__ import annotations

import string
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from loss import loss_terms
from model import ANSWER_CUE, Readout, TinyCausalLM, option_labels


# ============================================================
# 1. 極小トークナイザ (model.py の example_usage 内のものと同じ設計。
#    実運用では AutoTokenizer.from_pretrained(Qwen3.5-...) に置き換える)
# ============================================================

class ToyTokenizer:
    """A-Z, AA-ZZ を単一トークンに保証し、それ以外の単語はハッシュで固定語彙へ写す。"""

    def __init__(self, extra_vocab: int = 2000):
        self.single_tokens = list(string.ascii_uppercase) + \
            [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        self.vocab = ["<pad>"] + self.single_tokens
        self.extra_vocab = extra_vocab
        self.padding_side = "right"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + self.extra_vocab

    def encode(self, text_: str, add_special_tokens: bool = True) -> List[int]:
        if text_ in self.vocab:
            return [self.vocab.index(text_)]
        offset = len(self.vocab)
        ids = [offset + (hash(word) % self.extra_vocab) for word in text_.split()]
        return ids or [0]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[0]["content"] + "\n"

    def __call__(self, texts: List[str], padding=True, return_tensors="pt", add_special_tokens=False):
        encoded = [self.encode(t, add_special_tokens) for t in texts]
        max_len = max(len(e) for e in encoded)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, ids in enumerate(encoded):
            # padding_side="right": 有効トークンを先頭に詰める (公式と同じ、Readoutの
            # 「最後の有効トークン位置 = attention_mask.sum(-1)-1」という読み出しに必要)
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_prompt(tokenizer: ToyTokenizer, row: Dict, labels: List[str]) -> str:
    choices = "\n".join(f"({labels[i]}) {o['key']}: {o['desc']}" for i, o in enumerate(row["options"]))
    body = f"State:\n{row['state']}\nQuestion: {row['question']}\nOptions:\n{choices}".rstrip()
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": body}])
    return prefix + ANSWER_CUE


# ============================================================
# 2. pandas DataFrame -> Dataset / collate_fn
# ============================================================

class DecisionFrameDataset(Dataset):
    """pandas DataFrame (公式JSONLと同じカラム) を1行=1サンプルとして返すDataset。"""

    REQUIRED_COLUMNS = ["id", "type", "state", "question", "options", "train_target", "p0", "gold_target"]

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrameに必須カラムが欠けています: {missing}")
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        record = self.df.iloc[index].to_dict()
        # options/train_target/p0/gold_target は DataFrame のセルに list として
        # 格納されている前提 (JSONLをpd.read_json(lines=True)で読んだ場合もこの形になる)。
        return record


def make_collate_fn(tokenizer: ToyTokenizer, labels: List[str]):
    def collate_fn(rows: List[Dict]):
        prompts = [build_prompt(tokenizer, row, labels) for row in rows]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        return encoded, rows  # rows はそのまま loss_terms に渡す (型/options数/教師分布を保持)
    return collate_fn


# ============================================================
# 3. 合成データセット (実際にJSONLファイルを書き出し、pandasで読み直す)
# ============================================================

def _make_synthetic_dataframe(path: str, n_rows: int = 24, seed: int = 0) -> pd.DataFrame:
    import json
    import random

    rng = random.Random(seed)
    topics = ["code review", "incident response", "travel planning", "data pipeline triage"]
    records = []
    for i in range(n_rows):
        kind = ["choice", "noul", "score"][i % 3]
        if kind == "choice":
            n_opt = rng.randint(3, 5)
            options = [{"key": f"opt_{j}", "desc": f"Take action {j} regarding {topics[i % len(topics)]}."}
                       for j in range(n_opt)]
            probs = [rng.random() for _ in range(n_opt)]
        elif kind == "noul":
            options = [{"key": "yes", "desc": "The condition holds."}, {"key": "no", "desc": "It does not."}]
            probs = [rng.random(), 0]
            probs[1] = 1 - probs[0]
        else:
            n_opt = 5
            options = [{"key": str(j), "desc": f"Severity level {j} out of {n_opt - 1}."} for j in range(n_opt)]
            probs = [rng.random() for _ in range(n_opt)]
        total = sum(probs)
        probs = [p / total for p in probs]
        # train_target(教師)は gold に近いノイズ付き分布、p0(蒸留前baseline)はより均一な分布として合成する。
        gold_index = max(range(len(probs)), key=probs.__getitem__)
        gold = [1.0 if j == gold_index else 0.0 for j in range(len(options))]
        uniform = [1.0 / len(options)] * len(options)
        record = {
            "id": f"row_{i:04d}",
            "type": kind,
            "state": f"Session {i}: monitoring {topics[i % len(topics)]}.",
            "question": "Which option best matches the current situation?",
            "options": options,
            "train_target": probs,
            "p0": uniform,
            "gold_target": gold,
        }
        records.append(record)

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return pd.read_json(path, lines=True)


# ============================================================
# 4. 学習ループ (LoRA, 公式 train.py の r=64/alpha=128 構成を小モデルに適用)
# ============================================================

def build_lora_model(tokenizer: ToyTokenizer, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
    from peft import LoraConfig, get_peft_model

    base = TinyCausalLM(vocab_size=tokenizer.vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads)
    # TinyTransformerStack の各層が持つ素のnn.Linearは linear1/linear2 (FFN)。
    # 本物のQwen3.5では target_modules は言語側のq/k/v/o/gate/up/down projであり、
    # ここではその役割に対応するFFN線形層をLoRA対象にする (接頭辞マッチなので全層に適用される)。
    lora_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0,
                              target_modules=["linear1", "linear2"], bias="none")
    peft_model = get_peft_model(base, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def train(csv_or_jsonl_path: str, epochs: int = 3, batch_size: int = 4, lr: float = 5e-5) -> None:
    import pandas as pd

    df = pd.read_json(csv_or_jsonl_path, lines=True) if csv_or_jsonl_path.endswith((".jsonl", ".json")) \
        else pd.read_csv(csv_or_jsonl_path)

    tokenizer = ToyTokenizer()
    labels = option_labels(tokenizer)

    peft_model = build_lora_model(tokenizer)
    # LoRA (full_rank=False): base.get_base_model() で凍結済みベースモデルのLM headから
    # legal_weight (255行だけの出力埋め込みスナップショット) を取得する (model.py::Readout と同一)。
    label_ids = [tokenizer.encode(label)[0] for label in labels]
    readout = Readout(peft_model, label_ids, full_rank=False)

    dataset = DecisionFrameDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=make_collate_fn(tokenizer, labels))

    type_weights = {"choice": 1.0, "noul": 1.1666667, "score": 0.8}
    trainable_params = [p for p in readout.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    readout.train()
    for epoch in range(epochs):
        epoch_loss, epoch_ce, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0
        for encoded, rows in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = readout(encoded["input_ids"], encoded["attention_mask"])  # (B, K=255)
            total, ce, ordinal, kl = loss_terms(logits, rows, type_weights)
            loss = total.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce += ce.mean().item()
            epoch_kl += kl.mean().item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs}: loss={epoch_loss / n_batches:.4f}  "
              f"ce={epoch_ce / n_batches:.4f}  kl={epoch_kl / n_batches:.4f}  "
              f"grad_norm={grad_norm:.4f}")

    return readout


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    import os
    import tempfile

    print("=== finetune_example.py Example Usage ===\n")
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "train.jsonl")
        df = _make_synthetic_dataframe(data_path, n_rows=24, seed=0)
        print(f"合成データセット: {len(df)}行, カラム: {list(df.columns)}")
        print(f"type分布:\n{df['type'].value_counts()}\n")

        readout = train(data_path, epochs=2, batch_size=4, lr=5e-4)

        print("\n--- 学習後、1サンプルで推論して確率を確認 ---")
        readout.eval()
        tokenizer = ToyTokenizer()
        labels = option_labels(tokenizer)
        row = df.iloc[0].to_dict()
        with torch.no_grad():
            encoded = tokenizer([build_prompt(tokenizer, row, labels)], padding=True, return_tensors="pt")
            logits = readout(encoded["input_ids"], encoded["attention_mask"])
        n = len(row["options"])
        probs = logits[0, :n].softmax(-1)
        print(f"type={row['type']}  options={[o['key'] for o in row['options']]}")
        print(f"予測確率: {[round(p, 4) for p in probs.tolist()]}")
        print(f"教師分布(train_target): {[round(p, 4) for p in row['train_target']]}")


if __name__ == "__main__":
    example_usage()
