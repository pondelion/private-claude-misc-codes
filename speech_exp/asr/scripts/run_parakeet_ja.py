"""日本語音声データセットに対して NVIDIA Parakeet TDT-CTC (日本語版) で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

使い方:
    uv run python scripts/run_parakeet_ja.py --dataset reazonspeech             # 全ファイル推論
    uv run python scripts/run_parakeet_ja.py --dataset jsut --num-samples 5     # ランダム5件のみ
    uv run python scripts/run_parakeet_ja.py --dataset jsut --output out.csv
"""

import os

# WSL2環境でHFの xet 転送プロトコルがネットワークエラーで失敗することがあるため、
# 通常のHTTPダウンロードにフォールバックさせる。huggingface_hub のimport前に設定必須。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import csv
import logging
import random
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from tqdm import tqdm

from datasets_ja import DATASETS, RESULTS_DIR

# NeMo の transcribe() は呼び出しの度に内部でログレベルを WARNING に戻してしまうため、
# set_verbosity では抑制できない。ロガーに直接フィルタを付けて WARNING 以下を常に捨てる。
_nemo_logger = logging.getLogger("nemo_logger")
_nemo_logger.addFilter(lambda record: record.levelno >= logging.ERROR)

MODEL_NAME = "nvidia/parakeet-tdt_ctc-0.6b-ja"


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

    output_path = args.output or RESULTS_DIR / f"parakeet_ja_{args.dataset}_results.csv"

    samples = DATASETS[args.dataset]()
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device} ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    asr_model = asr_model.to(device)
    asr_model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred", "inference_time_sec"])

        for sample_id, audio_path, gt_text in tqdm(samples, desc="transcribing"):
            start = time.perf_counter()
            output = asr_model.transcribe([str(audio_path)], batch_size=1, verbose=False)[0]
            elapsed = time.perf_counter() - start

            pred_text = output.text if hasattr(output, "text") else str(output)
            writer.writerow([sample_id, gt_text, pred_text, f"{elapsed:.4f}"])

    print(f"\nwrote {len(samples)} rows to {output_path}")


if __name__ == "__main__":
    main()


# nvidia/parakeet-tdt_ctc-0.6b-ja の学習データ (モデルカード README.md より)
#
# 学習に使用されたデータセット:
#   - ReazonSpeech v2.0 (https://huggingface.co/datasets/reazon-research/reazonspeech)
#     日本のTV番組から収集した35,000時間超の自然な日本語音声コーパス。
#     これが唯一の学習データセットであり、当スクリプトの --dataset reazonspeech は
#     このコーパスのサブセット (tiny) を使っているため、学習データとの重複(リーク)
#     を含む可能性が高く、精度評価としてはフェアではない。
#
# 評価(ベンチマーク)にのみ使用されたデータセット (学習には未使用):
#   - JSUT basic5000
#   - Common Voice (MCV) 8.0 test
#   - Common Voice (MCV) 16.1 dev / test
#   - TEDxJP-10k
#   これらは学習データに含まれないため、--dataset jsut はリークなしの
#   フェアな精度検証として使える。
