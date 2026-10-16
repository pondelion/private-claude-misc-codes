"""DataFrameから理解用LoopVLをfine-tuneする完全例。実行: ``python finetuning_example.py``。

DataFrame columnsは``image``（画像path）、``instruction``、``response``。
外部VLM frameworkは使わず、画像読込、token化、PrefixLM batch、response-only CE、
optimizer updateを基本PyTorchで実行する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from loopvl_model import TinyLoopVL, response_cross_entropy


class CharacterTokenizer:
    """sampleを単独実行可能にする文字tokenizer。実checkpointでは公式tokenizerへ置換する。"""

    SPECIALS = ["<pad>", "<bos>", "<visual>", "<sep>", "<eos>", "<unk>"]

    def __init__(self, texts: Sequence[str]) -> None:
        self.tokens = self.SPECIALS + sorted(set("".join(texts)))
        self.lookup = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.visual_id, self.sep_id, self.eos_id, self.unk_id = range(6)

    def encode(self, text: str) -> list[int]:
        return [self.lookup.get(char, self.unk_id) for char in text]

    def __len__(self) -> int:
        return len(self.tokens)


def load_image(path: str, image_size: int) -> torch.Tensor:
    """画像をRGB resizeし``(3,H,W)`` float32 [0,1]へ変換する。"""
    with Image.open(path) as source:
        image = source.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class LoopVLDataFrameDataset(Dataset):
    """1行を画像と``[BOS,visual slots,instruction,SEP,response,EOS]``へ変換する。"""

    def __init__(self, frame: pd.DataFrame, tokenizer: CharacterTokenizer,
                 visual_tokens: int, image_size: int = 32) -> None:
        missing = {"image", "instruction", "response"} - set(frame.columns)
        if missing:
            raise ValueError(f"missing DataFrame columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.tokenizer, self.visual_tokens, self.image_size = tokenizer, visual_tokens, image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        instruction_ids = self.tokenizer.encode(str(row.instruction))
        response_ids = self.tokenizer.encode(str(row.response)) + [self.tokenizer.eos_id]
        ids = ([self.tokenizer.bos_id] + [self.tokenizer.visual_id] * self.visual_tokens
               + instruction_ids + [self.tokenizer.sep_id] + response_ids)
        visual_start, visual_end = 1, 1 + self.visual_tokens
        instruction_start, instruction_end = visual_end, visual_end + len(instruction_ids)
        response_start = instruction_end + 1
        length = len(ids)
        visual = torch.zeros(length, dtype=torch.bool); visual[visual_start:visual_end] = True
        instruction = torch.zeros(length, dtype=torch.bool); instruction[instruction_start:instruction_end] = True
        response = torch.zeros(length, dtype=torch.bool); response[response_start:] = True
        return {"image": load_image(str(row.image), self.image_size), "ids": torch.tensor(ids),
                "visual": visual, "instruction": instruction, "response": response}


@dataclass
class Batch:
    images: torch.Tensor
    ids: torch.Tensor
    token_types: torch.Tensor
    valid: torch.Tensor
    visual: torch.Tensor
    instruction: torch.Tensor
    response: torch.Tensor


def collate(rows: list[dict[str, torch.Tensor]], pad_id: int) -> Batch:
    """可変長textをpadding。各maskは``(B,Smax)``、画像は``(B,3,H,W)``。"""
    maximum, batch = max(len(row["ids"]) for row in rows), len(rows)
    ids = torch.full((batch, maximum), pad_id, dtype=torch.long)
    valid = torch.zeros((batch, maximum), dtype=torch.bool)
    visual, instruction, response = (torch.zeros_like(valid) for _ in range(3))
    for index, row in enumerate(rows):
        length = len(row["ids"]); ids[index, :length] = row["ids"]; valid[index, :length] = True
        visual[index, :length] = row["visual"]; instruction[index, :length] = row["instruction"]
        response[index, :length] = row["response"]
    token_types = (~response & valid).long()  # prefix=1、response/padding=0
    return Batch(torch.stack([row["image"] for row in rows]), ids, token_types,
                 valid, visual, instruction, response)


def train(frame: pd.DataFrame, epochs: int = 15, batch_size: int = 2,
          device: str = "cpu") -> tuple[TinyLoopVL, CharacterTokenizer]:
    """visual projectorとL/H recurrent backboneをresponse-only CEで更新する。"""
    texts = frame.instruction.astype(str).tolist() + frame.response.astype(str).tolist()
    tokenizer = CharacterTokenizer(texts)
    model = TinyLoopVL(len(tokenizer), width=64, vision_width=32, heads=4, layers_per_stack=1).to(device)
    dataset = LoopVLDataFrameDataset(frame, tokenizer, model.visual_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda rows: collate(rows, tokenizer.pad_id))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            values = [batch.images, batch.ids, batch.token_types, batch.valid,
                      batch.visual, batch.instruction, batch.response]
            images, ids, types, valid, visual, instruction, response = [x.to(device) for x in values]
            output = model(images, ids, types, valid, visual, instruction)
            loss = response_cross_entropy(output.logits, ids, response)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            total += float(loss.detach())
        if epoch in {0, epochs - 1}:
            print(f"epoch={epoch + 1:02d} loss={total / len(loader):.4f}")
    return model, tokenizer


def _make_demo_images(directory: Path) -> list[str]:
    """self-contained smoke test用画像を作る。学習処理自体は通常のpathを受け取る。"""
    directory.mkdir(parents=True, exist_ok=True)
    colors = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255)]
    paths = []
    for index, color in enumerate(colors):
        path = directory / f"color_{index}.png"; Image.new("RGB", (40, 28), color).save(path); paths.append(str(path))
    return paths


def main() -> None:
    torch.manual_seed(2)
    with TemporaryDirectory(prefix="loopvl_demo_") as directory:
        paths = _make_demo_images(Path(directory))
        frame = pd.DataFrame({"image": paths, "instruction": ["色は?", "色は?", "何色?", "何色?"],
                              "response": ["赤", "青", "赤", "青"]})
        model, _ = train(frame)
        assert model.projector.fc1.weight.grad is not None
    print("DataFrame fine-tuning completed")


if __name__ == "__main__":
    main()
