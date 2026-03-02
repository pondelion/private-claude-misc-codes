"""
Qwen2.5-Omni LoRA ファインチューニング (Text-Audio)
====================================================

音声理解タスク (ASR, 音声分類, 音声QA, 音楽理解等) 向けの
LoRA ファインチューニング スクリプト

LoRA対象: Thinker LLM のみ
凍結: Audio Tower, Visual Tower, Talker, Token2Wav

必要パッケージ:
    pip install transformers>=4.52.4 peft accelerate datasets
    pip install flash-attn --no-build-isolation
    pip install qwen-omni-utils librosa soundfile

使い方:
    # 単一GPU
    python lora_finetune_text_audio.py \
        --model_name Qwen/Qwen2.5-Omni-7B \
        --data_path /path/to/text_audio_train.json \
        --output_dir ./output/text_audio_lora \
        --num_epochs 3

    # マルチGPU (accelerate)
    accelerate launch --num_processes 4 lora_finetune_text_audio.py \
        --model_name Qwen/Qwen2.5-Omni-7B \
        --data_path /path/to/text_audio_train.json \
        --output_dir ./output/text_audio_lora

データセット JSON 形式:
    [
      {
        "messages": [
          {"role": "user", "content": "<audio>この音声を書き起こしてください。"},
          {"role": "assistant", "content": "本日の会議では..."}
        ],
        "audios": ["/absolute/path/to/audio.wav"]
      },
      ...
    ]

    ★ <audio> タグの数と audios リストの長さが一致すること
    ★ マルチターン、systemプロンプト、複数音声にも対応
    ★ 音声フォーマット: WAV, MP3, FLAC, OGG 等 (librosa対応形式)
    ★ サンプリングレートは任意 (16kHzに自動リサンプリング)

別解 (ms-swift):
    swift sft \
        --model Qwen/Qwen2.5-Omni-7B \
        --dataset /path/to/data.jsonl \
        --train_type lora --lora_rank 8 --lora_alpha 32 \
        --target_modules all-linear --freeze_vit true \
        --torch_dtype bfloat16 --num_train_epochs 3
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Qwen2.5-Omni の音声サンプリングレート


# ============================================
# 音声読み込み
# ============================================

def load_audio(audio_path: str, max_duration: float = 30.0) -> np.ndarray:
    """
    音声ファイルを読み込み、16kHz モノラルに変換

    入力:
        audio_path: 音声ファイルパス
        max_duration: 最大音声長 (秒)

    出力:
        audio: (N_samples,) np.ndarray - 16kHz モノラル音声

    対応形式: WAV, MP3, FLAC, OGG, M4A 等
    """
    import librosa

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    # audio: (N_samples,) np.ndarray, float32, [-1, 1]

    # 最大長でクリップ
    max_samples = int(max_duration * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


# ============================================
# データセット
# ============================================

class TextAudioSFTDataset(Dataset):
    """
    Text-Audio SFT データセット

    pandas DataFrame形式:
        必須カラム:
            messages: ChatML形式の会話リスト (list[dict])
            audios: 音声ファイルパスのリスト (list[str])

        DataFrame例:
            df = pd.DataFrame([
                {
                    "messages": [
                        {"role": "user", "content": "<audio>この音声を書き起こしてください。"},
                        {"role": "assistant", "content": "本日の会議では..."},
                    ],
                    "audios": ["/path/to/audio.wav"],
                },
                ...
            ])

    入出力shape (Processor適用後):
        input_ids:              (L,)              トークンID
        attention_mask:         (L,)              アテンションマスク
        input_features:         (N_audios, 128, T_mel) メルスペクトログラム
        audio_feature_lengths:  (N_audios,)       各音声の有効フレーム数
        labels:                 (L,)              assistant部分のみ有効、他は -100
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: Qwen2_5OmniProcessor,
        max_length: int = 4096,
        max_audio_duration: float = 30.0,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.max_audio_duration = max_audio_duration
        logger.info(f"Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = row["messages"]
        audio_paths = row.get("audios", []) if "audios" in row.index else []

        # 音声読み込み
        audios = []
        for p in audio_paths:
            audio = load_audio(p, max_duration=self.max_audio_duration)
            audios.append(audio)

        # ChatML テンプレート適用 (学習用: add_generation_prompt=False)
        text_with_response = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # Processor で前処理
        inputs = self.processor(
            text=text_with_response,
            audios=audios if audios else None,
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

        # 音声特徴 (存在する場合)
        if "input_features" in inputs and inputs["input_features"] is not None:
            result["input_features"] = inputs["input_features"].squeeze(0)
        if "audio_feature_lengths" in inputs and inputs["audio_feature_lengths"] is not None:
            result["audio_feature_lengths"] = inputs["audio_feature_lengths"].squeeze(0)

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
    """可変長の音声特徴を処理するcollate関数"""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch if key in item]
        if not values:
            continue
        if key in ("input_features",):
            # 音声特徴: バッチ内で結合 (可変サイズ)
            collated[key] = torch.cat(values, dim=0)
        elif key in ("audio_feature_lengths",):
            # 音声長: 結合
            collated[key] = torch.cat(
                [v.unsqueeze(0) if v.dim() == 0 else v for v in values], dim=0
            )
        else:
            collated[key] = torch.stack(values, dim=0)

    return collated


# ============================================
# 学習ループ
# ============================================

def train(args):
    logger.info(f"Loading model: {args.model_name}")

    # モデルロード
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    # エンコーダ・Talker・Token2Wav を凍結
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Thinker LLM に LoRA 適用
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
    dataset = TextAudioSFTDataset(
        df,
        processor,
        max_length=args.max_length,
        max_audio_duration=args.max_audio_duration,
    )
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
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Text-Audio LoRA Fine-tuning")

    # モデル
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大シーケンス長 (音声は長いので大きめに設定)")

    # データ
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON dataset file")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_audio_duration", type=float, default=30.0,
                        help="最大音声長 (秒). 長音声はVRAM消費が大きい")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 学習
    parser.add_argument("--batch_size", type=int, default=1,
                        help="音声データはVRAM消費が大きいため1推奨")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="実効バッチサイズ = batch_size * gradient_accumulation_steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ログ・保存
    parser.add_argument("--output_dir", type=str, default="./output/text_audio_lora")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
