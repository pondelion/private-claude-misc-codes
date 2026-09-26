"""日本語音声データセットに対して OpenAI Whisper large-v3-turbo で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

使い方:
    uv run python scripts/run_whisper_large_v3_turbo.py --dataset jsut             # 全ファイル推論
    uv run python scripts/run_whisper_large_v3_turbo.py --dataset jsut --num-samples 5
"""

import os

# WSL2環境でHFの xet 転送プロトコルがネットワークエラーで失敗することがあるため、
# 通常のHTTPダウンロードにフォールバックさせる。huggingface_hub のimport前に設定必須。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

from datasets_ja import DATASETS, RESULTS_DIR

MODEL_NAME = "openai/whisper-large-v3-turbo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="推論するサンプル数(未指定の場合は全ファイル)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or RESULTS_DIR / f"whisper_large_v3_turbo_{args.dataset}_results.csv"

    samples = DATASETS[args.dataset]()
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    generate_kwargs = {"language": "ja", "task": "transcribe"}

    print(f"loading {MODEL_NAME} on {device} ...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred", "inference_time_sec"])

        for sample_id, audio_path, gt_text in tqdm(samples, desc="transcribing"):
            start = time.perf_counter()
            output = pipe(str(audio_path), generate_kwargs=generate_kwargs)
            elapsed = time.perf_counter() - start

            pred_text = output["text"]
            writer.writerow([sample_id, gt_text, pred_text, f"{elapsed:.4f}"])

    print(f"\nwrote {len(samples)} rows to {output_path}")


if __name__ == "__main__":
    main()


# openai/whisper-large-v3-turbo の学習データについて
#
# large-v3 の重みから、デコーダーを32層→4層に削減して追加学習(fine-tuning)した派生モデル。
# OpenAIの独自収集音声(68万時間、Web由来)で学習されており、ReazonSpeech/JSUTなど
# 日本語特化コーパスは学習データに含まれない(=リークの心配は基本的にない)。
# ただし翻訳(translate)タスクは学習から除外されているため未対応、transcribeのみ利用可能。
