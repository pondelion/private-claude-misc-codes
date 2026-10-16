"""DataFrameからscratch MDLMをfinetuneする完全例。実行: ``python finetuning_example.py``。

必要列は``prompt``と``response``。外部tokenizerを不要にするため、この例では文字tokenizerを
学習DataFrameから構築する。実モデルでは同じDataset/損失構造のままsubword tokenizerへ交換する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mdlm import corrupt_tokens, mdlm_loss
from model import TinyDiffusionLM
from noise_schedule import LinearSchedule
from sampling import sample_mdlm


class CharacterTokenizer:
    """教材用可逆tokenizer。PAD/BOS/SEP/EOS/MASKは通常文字と衝突しない。"""

    SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<mask>", "<unk>"]

    def __init__(self, texts: Sequence[str]) -> None:
        chars = sorted(set("".join(texts)))
        self.tokens = self.SPECIALS + chars
        self.to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.sep_id, self.eos_id, self.mask_id, self.unk_id = range(6)

    def encode(self, text: str) -> list[int]:
        return [self.to_id.get(char, self.unk_id) for char in text]

    def decode(self, ids: Sequence[int]) -> str:
        ignored = {self.pad_id, self.bos_id, self.sep_id, self.mask_id}
        return "".join(self.tokens[i] for i in ids if i not in ignored and i != self.eos_id)

    def __len__(self) -> int:
        return len(self.tokens)


class DataFrameSFTDataset(Dataset):
    """1行を``[BOS,prompt,SEP,response,EOS]``へ変換する。

    返値 ``ids: (L,)``、``eligible: (L,)``。promptは条件としてvisibleだが、corruption/loss
    対象はresponse+EOSだけ。これがinstruction tuningの「何をGTにするか」に相当する。
    """

    def __init__(self, frame: pd.DataFrame, tokenizer: CharacterTokenizer, max_length: int = 96) -> None:
        missing = {"prompt", "response"} - set(frame.columns)
        if missing:
            raise ValueError(f"missing DataFrame columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        prefix = [self.tokenizer.bos_id] + self.tokenizer.encode(str(row.prompt)) + [self.tokenizer.sep_id]
        target = self.tokenizer.encode(str(row.response)) + [self.tokenizer.eos_id]
        ids = (prefix + target)[:self.max_length]
        target_start = min(len(prefix), len(ids))
        eligible = [False] * target_start + [True] * (len(ids) - target_start)
        return torch.tensor(ids), torch.tensor(eligible)


@dataclass
class Batch:
    ids: torch.Tensor       # (B,Lmax)
    eligible: torch.Tensor  # (B,Lmax)
    valid: torch.Tensor     # (B,Lmax), non-padding


def collate(rows: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> Batch:
    max_len = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    eligible = torch.zeros((len(rows), max_len), dtype=torch.bool)
    valid = torch.zeros_like(eligible)
    for i, (row_ids, row_eligible) in enumerate(rows):
        ids[i, :len(row_ids)] = row_ids
        eligible[i, :len(row_ids)] = row_eligible
        valid[i, :len(row_ids)] = True
    return Batch(ids, eligible, valid)


def train(frame: pd.DataFrame, epochs: int = 20, device: str = "cpu") -> tuple[TinyDiffusionLM, CharacterTokenizer]:
    """DataFrame全行を実際に更新する基本PyTorch loop。"""
    tokenizer = CharacterTokenizer(frame.prompt.astype(str).tolist() + frame.response.astype(str).tolist())
    dataset = DataFrameSFTDataset(frame, tokenizer)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True,
                        collate_fn=lambda rows: collate(rows, tokenizer.pad_id))
    model = TinyDiffusionLM(len(tokenizer), max_length=96, width=96, layers=3, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    schedule = LinearSchedule()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            ids, eligible, valid = (batch.ids.to(device), batch.eligible.to(device), batch.valid.to(device))
            corruption = corrupt_tokens(ids, eligible, tokenizer.mask_id, schedule)
            # keyがpaddingの列だけを不可視化する。(B,Lquery,Lkey)
            allowed = valid[:, None, :].expand(-1, ids.shape[1], -1)
            logits = model(corruption.noisy_ids, allowed=allowed)
            loss, _ = mdlm_loss(logits, ids, eligible, corruption, schedule)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        if epoch in {0, epochs - 1}:
            print(f"epoch={epoch + 1:02d} loss={total / len(loader):.4f}")
    return model, tokenizer


def main() -> None:
    torch.manual_seed(1)
    frame = pd.DataFrame({
        "prompt": ["1+1=", "2+2=", "3+3=", "空の色は?", "雪の色は?", "猫の鳴き声は?"],
        "response": ["2", "4", "6", "青", "白", "にゃー"],
    })
    model, tokenizer = train(frame)
    prompt_text = "1+1="
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt_text) + [tokenizer.sep_id]
    output = sample_mdlm(model, torch.tensor([prompt_ids]), tokenizer.mask_id,
                         generation_length=4, steps=4)
    generated = output.sequences[0, len(prompt_ids):].tolist()
    print("generated:", tokenizer.decode(generated))
    assert output.sequences.shape == (1, len(prompt_ids) + 4)


if __name__ == "__main__":
    main()
