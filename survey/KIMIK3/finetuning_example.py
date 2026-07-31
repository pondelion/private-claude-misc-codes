"""
Kimi K3 ファインチューニング (Trainer不使用, PyTorch基本機能のみ)
================================================================

pandas DataFrame で与えられた (prompt, response, image) のメタデータから
Kimi K3 を教師あり学習 (SFT) でファインチューニングするサンプルスクリプトです。
transformers の Trainer は使わず、手動の訓練ループで実装しています。

============================================================
このスクリプトが使うモデルについて
============================================================
Kimi K3 は 2.8T パラメータ (実重み ~数百GB) のモデルであり、実重みをダウンロードして
ロードすることは非現実的です (また本タスクでは HuggingFace から .py 以外のファイル
[safetensors 等] を取得しないよう指示されています)。そのため本スクリプトは、
main_flow.py で実装した KimiK3ForConditionalGeneration (KDA + Gated MLA +
Stable LatentMoE + AttnRes + MoonViT-V2 を全て含む、論文の数式に忠実な縮小スケール版)
をそのまま使い、実際に forward -> loss -> backward -> optimizer.step() が
最初から最後まで動くことを示します。

同様の理由で、実際の Kimi トークナイザ (tiktoken ベース, tokenization_kimi.py) は
外部の語彙ファイル (tiktoken.model, .py 以外の巨大バイナリ) を必要とするため
ロードできません。代わりに本スクリプトはバイト単位の決定的なトークナイザ
(`ByteLevelTokenizer`) を実装し、その旨を明記します。学習ロジック (labelマスキング、
勾配計算、最適化) 自体は本物のトークナイザに差し替えても変更不要です。

============================================================
DataFrameの期待するカラム
============================================================
必須:
    - prompt   (str): ユーザー入力 (指示文)
    - response (str): 学習させたいアシスタント応答

オプション:
    - image (str or None): 画像ファイルパス。指定した場合 prompt 中に
      画像プレースホルダトークンが1つ挿入され、対応する視覚トークンに置き換えられる。
    - reasoning_effort (str: "low"/"high"/"max"): §4.1 のreasoning effort レベルに
      対応するプロンプト接頭辞として使用 (任意)。

============================================================
使い方
============================================================
# スクリプト単体実行 (JSONL -> pandas DataFrame)
python finetuning_example.py \\
    --train_file /path/to/train.jsonl \\
    --output_dir ./kimik3-finetuned \\
    --epochs 3 --batch_size 1 --grad_acc 8 --lr 2e-5

# Python内からDataFrameで呼ぶ場合
import pandas as pd
df = pd.DataFrame({
    "prompt":   ["1+1は?", "画像の内容を説明して"],
    "response": ["2です。", "赤い花が写っています。"],
    "image":    [None, "path/to/flower.jpg"],
})
model, tokenizer, config = build_model_and_tokenizer()
train(model, tokenizer, config, train_df=df, output_dir="./out")
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from main_flow import KimiK3Config, KimiK3ForConditionalGeneration

IGNORE_INDEX = -100


# ============================================================
# 1. トークナイザ (バイトレベルの代替実装、理由は module docstring 参照)
# ============================================================
class ByteLevelTokenizer:
    """UTF-8 バイト単位の決定的トークナイザ。

    語彙: 0..255 (生バイト) + 特殊トークン {BOS, EOS, PAD, IMAGE}。
    実際の Kimi K3 は XTML ベースのチャットテンプレートと BPE (tiktoken) を使うが
    (§4.1.1 "we serialize all data with our XTML-based chat template")、
    ここでは外部語彙ファイルなしで完結させるための最小限の代替である。
    """

    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
        self.image_id = 259
        self.vocab_size = 260

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if i < 256)
        return raw.decode("utf-8", errors="replace")


# ============================================================
# 2. データセット定義
# ============================================================
class KimiK3SFTDataset(Dataset):
    """SFT用データセット (pandas DataFrame ベース)。

    DataFrame columns:
        prompt   (str, 必須)
        response (str, 必須)
        image    (str or None, オプション): 画像ファイルパス
        reasoning_effort (str, オプション): "low" | "high" | "max"
    """

    def __init__(self, df: pd.DataFrame, tokenizer: ByteLevelTokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        response = str(row["response"])
        image_path = row.get("image", None) if "image" in self.df.columns else None
        if isinstance(image_path, float) and math.isnan(image_path):  # pandas の NaN
            image_path = None
        effort = row.get("reasoning_effort", None) if "reasoning_effort" in self.df.columns else None
        if isinstance(effort, str):
            prompt = f"[effort={effort}] " + prompt
        return {
            "prompt": prompt,
            "response": response,
            "image_path": image_path,
        }


def _load_image_as_navit_input(image_path: str) -> torch.Tensor:
    """画像ファイルを native_vision.py が期待する (T=1, H, W, 3) float テンソルに変換する。

    H, W は patch_size(=14) の倍数かつ merge_kernel(2,2) 適用可能な偶数パッチ数になるよう
    28の倍数へ最近傍リサイズする (簡略化のための固定サイズポリシー)。
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((28, 28))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (28, 28, 3), [0, 1]
    arr = (arr - 0.5) / 0.5  # 簡易正規化
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 28, 28, 3)


# ============================================================
# 3. Collator (バッチ化 + ラベル生成)
# ============================================================
class KimiK3Collator:
    """1サンプルずつ (B=1) を処理する。

    main_flow.KimiK3Backbone は Block Attention Residuals の実装上 B=1 を前提としている
    (トークン軸のフラット化と系列軸の復元を単純化するため、main_flow.py の docstring 参照)。
    そのため本スクリプトのバッチサイズは実質1固定とし、`--grad_acc` で実効バッチサイズを稼ぐ。

    Labels の構成:
        full_text = "<BOS>" + prompt(+image placeholder) + response + "<EOS>"
        prompt 部分は -100 でマスクし、response 部分のみ損失計算の対象にする。
    """

    def __init__(self, tokenizer: ByteLevelTokenizer, media_placeholder_token_id: int):
        self.tokenizer = tokenizer
        self.media_placeholder_token_id = media_placeholder_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        assert len(features) == 1, "このモデルは B=1 のみサポート (docstring参照)"
        f = features[0]

        prompt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(f["prompt"])
        images = None
        if f["image_path"] is not None:
            prompt_ids = prompt_ids + [self.media_placeholder_token_id]
            images = [_load_image_as_navit_input(f["image_path"])]

        response_ids = self.tokenizer.encode(f["response"]) + [self.tokenizer.eos_id]

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, T)
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)        # (1, T)

        return {"input_ids": input_ids, "labels": labels, "images": images}


# ============================================================
# 4. モデル構築
# ============================================================
def build_model_and_tokenizer() -> tuple[KimiK3ForConditionalGeneration, ByteLevelTokenizer, KimiK3Config]:
    """縮小スケールの KimiK3 モデルとバイトレベルトークナイザを構築する。

    実運用の Kimi K3 (vocab_size=163840, hidden_size=7168, 93層, ...) をそのまま使うと
    このデモは実行不可能なため、main_flow.KimiK3Config のデフォルト (縮小スケール) を
    トークナイザの vocab_size に合わせて調整して使う。
    """
    tokenizer = ByteLevelTokenizer()
    config = KimiK3Config(
        vocab_size=tokenizer.vocab_size,
        media_placeholder_token_id=tokenizer.image_id,
    )
    model = KimiK3ForConditionalGeneration(config)
    return model, tokenizer, config


# ============================================================
# 5. 学習率スケジューラ
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine学習率スケジューラ with Warmup (§3.3 の cosine decay + 1% linear warmup に対応)。"""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 6. メイン訓練ループ
# ============================================================
def train(
    model: KimiK3ForConditionalGeneration,
    tokenizer: ByteLevelTokenizer,
    config: KimiK3Config,
    train_df: pd.DataFrame,
    output_dir: str = "./kimik3-finetuned",
    epochs: int = 3,
    grad_acc: int = 8,
    lr: float = 2e-5,
    warmup_ratio: float = 0.01,  # §3.3: "1% linear warmup"
    weight_decay: float = 0.1,   # §3.3: "Weight decay is set to 0.1 throughout"
    max_grad_norm: float = 1.0,
    log_steps: int = 5,
):
    """手動訓練ループ (Trainerなし)。

    ========================================
    Shape (訓練中、B=1 固定)
    ========================================
    入力:
        input_ids: (1, T) int64
        labels:    (1, T) int64   (-100 = 損失除外)
        images:    None or list[(1, H, W, 3) float]
    Forward出力:
        logits: (1, T, vocab_size)
    Loss:
        shift_logits = logits[:, :-1]   -> (1, T-1, vocab_size)
        shift_labels = labels[:, 1:]    -> (1, T-1)
        loss = cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=-100)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    dataset = KimiK3SFTDataset(train_df, tokenizer)
    collator = KimiK3Collator(tokenizer, config.media_placeholder_token_id)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(loader) / grad_acc)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("Training config:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Gradient accumulation: {grad_acc}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    accumulated_loss = 0.0
    loss_history: list[float] = []

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1}/{epochs} ===")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(loader, desc="train")):
            input_ids = batch["input_ids"].to(device)  # (1, T)
            labels = batch["labels"].to(device)        # (1, T)
            images = (
                [img.to(device) for img in batch["images"]] if batch["images"] is not None else None
            )

            logits = model(input_ids, images=images)  # (1, T, V)

            shift_logits = logits[:, :-1, :].contiguous()  # (1, T-1, V)
            shift_labels = labels[:, 1:].contiguous()      # (1, T-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            (loss / grad_acc).backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_steps == 0 or global_step == total_steps:
                    avg_loss = accumulated_loss / (grad_acc * log_steps) if global_step % log_steps == 0 else accumulated_loss / grad_acc
                    loss_history.append(avg_loss)
                    print(f"  step {global_step}/{total_steps} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                    accumulated_loss = 0.0

    ckpt_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nTraining complete! Model saved to {ckpt_path}")
    return loss_history


# ============================================================
# 7. エントリポイント
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Kimi K3 Fine-tuning (No Trainer)")
    p.add_argument("--train_file", type=str, default="",
                    help="JSONL file: {prompt, response, image(optional)} -> pd.read_json(lines=True)")
    p.add_argument("--output_dir", type=str, default="./kimik3-finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1, help="B=1 固定 (main_flow.py の制約)")
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer, config = build_model_and_tokenizer()

    if args.train_file:
        train_df = pd.read_json(args.train_file, lines=True)
    else:
        train_df = _build_demo_dataframe()

    train(
        model, tokenizer, config, train_df,
        output_dir=args.output_dir, epochs=args.epochs,
        grad_acc=args.grad_acc, lr=args.lr,
    )


def _build_demo_dataframe() -> pd.DataFrame:
    """外部データが無い場合に使う、動作確認用の極小デモデータセット。"""
    return pd.DataFrame({
        "prompt": [
            "1たす1は?",
            "日本の首都は?",
            "こんにちは",
            "2たす3は?",
        ],
        "response": [
            "2です。",
            "東京です。",
            "こんにちは、元気ですか？",
            "5です。",
        ],
        "image": [None, None, None, None],
        "reasoning_effort": ["low", "low", "low", "low"],
    })


if __name__ == "__main__":
    torch.manual_seed(0)

    model, tokenizer, config = build_model_and_tokenizer()
    df = _build_demo_dataframe()

    print(df)
    loss_history = train(
        model, tokenizer, config, df,
        output_dir="/tmp/kimik3-finetuned-demo",
        epochs=8, grad_acc=2, lr=5e-4, log_steps=2,
    )

    print("loss history:", [f"{v:.4f}" for v in loss_history])
    assert loss_history[-1] < loss_history[0], "極小デモデータセットで損失が低下していない"
    print("finetuning_example OK: loss decreased from "
          f"{loss_history[0]:.4f} to {loss_history[-1]:.4f}")
