"""日本語音声データセットに対して Qwen3-ASR で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

使い方:
    uv run python scripts/run_qwen3_asr.py --dataset jsut                       # 全ファイル推論(0.6B)
    uv run python scripts/run_qwen3_asr.py --dataset jsut --num-samples 5
    uv run python scripts/run_qwen3_asr.py --dataset jsut --model-size 1.7B
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
from transformers import AutoModelForMultimodalLM, AutoProcessor

from datasets_ja import DATASETS, RESULTS_DIR

MODEL_NAMES = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B-hf",
    "1.7B": "Qwen/Qwen3-ASR-1.7B-hf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--model-size", choices=MODEL_NAMES.keys(), default="0.6B")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="推論するサンプル数(未指定の場合は全ファイル)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model_size]
    output_path = args.output or RESULTS_DIR / f"qwen3_asr_{args.model_size}_{args.dataset}_results.csv"

    samples = DATASETS[args.dataset]()
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    print(f"loading {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, device_map="auto")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred", "inference_time_sec"])

        for sample_id, audio_path, gt_text in tqdm(samples, desc="transcribing"):
            start = time.perf_counter()

            inputs = processor.apply_transcription_request(
                audio=str(audio_path),
                language="Japanese",
            ).to(model.device, model.dtype)

            output_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            pred_text = processor.decode(generated_ids, return_format="transcription_only")[0]

            elapsed = time.perf_counter() - start
            writer.writerow([sample_id, gt_text, pred_text, f"{elapsed:.4f}"])

    print(f"\nwrote {len(samples)} rows to {output_path}")


if __name__ == "__main__":
    main()


# Qwen/Qwen3-ASR-{0.6B,1.7B}-hf の学習データについて
#
# Qwen3-Omni の事前学習(3兆トークン規模のマルチタスク音声/画像/テキストデータ)をベースに、
# 30言語・22中国語方言をカバーする独自の教師ありファインチューニングデータで学習されている。
# 具体的なデータセット名(ReazonSpeechやJSUTなど)は技術レポート上で明示されておらず、
# 学習データは非公開の自社データが中心と見られる。
# → ReazonSpeech/JSUTいずれについても学習データとの重複(リーク)の有無を確認できないため、
#   他モデルほど「フェアな評価」を断定はできない点に注意。
