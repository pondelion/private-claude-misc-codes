"""
Qwen3-Omni LoRA ファインチューニング (Text-Image)
====================================================

画像理解タスク (VQA, キャプション生成, OCR等) 向けの
LoRA ファインチューニング スクリプト

LoRA対象: Thinker LLM (MoE) のみ
凍結: AuT (Audio Transformer), SigLIP2 (Vision), Talker, Code2Wav

★ Qwen2.5-Omni との主な違い:
    - MoE アーキテクチャ (30B-A3B): 全エキスパートにLoRA適用
    - クラス名: Qwen3OmniMoeForConditionalGeneration / Qwen3OmniMoeProcessor
    - Vision Encoder: SigLIP2-So400m (Qwen2.5-Omni は独自ViT)
    - transformers>=4.57.3 が必要

必要パッケージ:
    pip install transformers>=4.57.3 peft accelerate datasets
    pip install flash-attn --no-build-isolation
    pip install qwen-omni-utils Pillow

使い方:
    # 単一GPU
    python lora_finetune_text_image.py \
        --model_name Qwen/Qwen3-Omni \
        --data_path /path/to/text_image_train.json \
        --output_dir ./output/text_image_lora \
        --num_epochs 3

    # マルチGPU (accelerate)
    accelerate launch --num_processes 4 lora_finetune_text_image.py \
        --model_name Qwen/Qwen3-Omni \
        --data_path /path/to/text_image_train.json \
        --output_dir ./output/text_image_lora

データセット JSON 形式:
    [
      {
        "messages": [
          {"role": "user", "content": "<image>この画像を説明してください。"},
          {"role": "assistant", "content": "画像には猫が写っています。"}
        ],
        "images": ["/absolute/path/to/image.jpg"]
      },
      ...
    ]

    ★ <image> タグの数と images リストの長さが一致すること
    ★ マルチターン、systemプロンプト、複数画像にも対応

別解 (ms-swift):
    MAX_PIXELS=1003520 swift sft \
        --model Qwen/Qwen3-Omni \
        --dataset /path/to/data.jsonl \
        --train_type lora --lora_rank 8 --lora_alpha 32 \
        --target_modules all-linear --freeze_vit true \
        --torch_dtype bfloat16 --num_train_epochs 3
"""

import argparse
import json
import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================
# データセット
# ============================================

class TextImageSFTDataset(Dataset):
    """
    Text-Image SFT データセット

    pandas DataFrame形式:
        必須カラム:
            messages: ChatML形式の会話リスト (list[dict])
            images: 画像ファイルパスのリスト (list[str])

        DataFrame例:
            df = pd.DataFrame([
                {
                    "messages": [
                        {"role": "user", "content": "<image>この画像を説明してください。"},
                        {"role": "assistant", "content": "猫が写っています。"},
                    ],
                    "images": ["/path/to/image.jpg"],
                },
                ...
            ])

    入出力shape (Processor適用後):
        input_ids:       (L,)          トークンID
        attention_mask:  (L,)          アテンションマスク
        pixel_values:    (N_patches, patch_dim)  パッチ化画像
        image_grid_thw:  (num_images, 3)  各画像の [T, H, W]
        labels:          (L,)          assistant部分のみ有効、他は -100

    ★ Qwen2.5-Omni との違い:
        - Processor名: Qwen3OmniMoeProcessor
        - SigLIP2ベースのVision Encoder (hidden=1152, depth=27)
        - pixel_values の内部表現が SigLIP2 に対応
    """

    def __init__(self, df: pd.DataFrame, processor: Qwen3OmniMoeProcessor, max_length: int = 3072):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        logger.info(f"Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = row["messages"]
        image_paths = row.get("images", []) if "images" in row.index else []

        # 画像読み込み
        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        # ChatML テンプレート適用 (学習用: add_generation_prompt=False)
        text_with_response = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # Processor で前処理
        inputs = self.processor(
            text=text_with_response,
            images=images if images else None,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # labels 作成: assistant 応答部分のみ
        input_ids = inputs["input_ids"].squeeze(0)  # (L,)
        labels = self._create_labels(input_ids, messages)

        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result

    def _create_labels(self, input_ids: torch.Tensor, messages: list) -> torch.Tensor:
        """
        assistant応答部分のみ学習対象にするlabelsを作成

        assistant応答以外のトークンは -100 (ignore_index) に設定
        """
        labels = input_ids.clone()

        # im_start/im_end パターンでassistant部分を特定
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        ids = input_ids.tolist()
        mask = torch.zeros_like(labels, dtype=torch.bool)

        i = 0
        while i < len(ids):
            if ids[i] == im_start_id:
                # <|im_start|> の次のトークンが role
                # "assistant" かどうかを確認
                role_text = self.processor.tokenizer.decode(
                    ids[i + 1 : min(i + 5, len(ids))], skip_special_tokens=False
                )
                if role_text.startswith("assistant"):
                    # assistant応答: <|im_start|>assistant\n の次から <|im_end|> まで
                    j = i + 1
                    while j < len(ids) and ids[j] != im_end_id:
                        j += 1
                    # "assistant\n" 部分をスキップし、応答本体 + <|im_end|> を学習対象に
                    assistant_token = self.processor.tokenizer.encode(
                        "assistant\n", add_special_tokens=False
                    )
                    content_start = i + 1 + len(assistant_token)
                    content_end = j + 1  # <|im_end|> を含む
                    mask[content_start:content_end] = True
                    i = j + 1
                    continue
            i += 1

        labels[~mask] = -100
        return labels


def collate_fn(batch):
    """可変長のpixel_values / image_grid_thw を処理するcollate関数"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch if key in item]
        if not values:
            continue
        if key in ("pixel_values", "image_grid_thw"):
            # 画像関連: バッチ内で結合 (可変サイズ)
            collated[key] = torch.cat(values, dim=0)
        else:
            collated[key] = torch.stack(values, dim=0)

    return collated


# ============================================
# 学習ループ
# ============================================

def train(args):
    logger.info(f"Loading model: {args.model_name}")

    # モデルロード
    # ★ Qwen3-Omni は MoE (30B-A3B) のため VRAM 消費が大きい
    #   device_map="auto" で自動分散配置
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    # 全パラメータ凍結 (AuT, SigLIP2, Talker, Code2Wav を含む)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Thinker LLM (MoE) に LoRA 適用
    # ★ MoE の全エキスパートの線形層にLoRAが適用される
    #   共有ゲート (gate_proj 等) にも適用
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.thinker = get_peft_model(model.thinker, lora_config)
    model.thinker.print_trainable_parameters()

    # データセット (JSONファイル → DataFrame)
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    dataset = TextImageSFTDataset(df, processor, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # オプティマイザ & スケジューラ
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 学習ループ
    model.train()
    global_step = 0
    accumulation_loss = 0.0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            # デバイスに転送
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model.thinker(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            accumulation_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = accumulation_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                    accumulation_loss = 0.0

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.thinker.save_pretrained(save_path)
                    processor.save_pretrained(save_path)
                    logger.info(f"Checkpoint saved: {save_path}")

    # 最終保存
    final_path = os.path.join(args.output_dir, "final")
    model.thinker.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Training complete. Final adapter saved: {final_path}")


# ============================================
# エントリポイント
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Omni Text-Image LoRA Fine-tuning")

    # モデル
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Omni")
    parser.add_argument("--max_length", type=int, default=3072)

    # データ
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON dataset file")
    parser.add_argument("--num_workers", type=int, default=4)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 学習
    parser.add_argument("--batch_size", type=int, default=1,
                        help="MoEモデルはVRAM消費が大きいため1推奨")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ログ・保存
    parser.add_argument("--output_dir", type=str, default="./output/text_image_lora")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
