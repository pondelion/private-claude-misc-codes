"""日本語音声データセットに対して ReazonSpeech-NeMo v2 で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

使い方:
    uv run python scripts/run_reazonspeech_nemo.py --dataset jsut             # 全ファイル推論
    uv run python scripts/run_reazonspeech_nemo.py --dataset jsut --num-samples 5
    uv run python scripts/run_reazonspeech_nemo.py --dataset jsut --output out.csv
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
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding, tdt_beam_decoding
from tqdm import tqdm

from datasets_ja import DATASETS, RESULTS_DIR

# NeMo の transcribe() は呼び出しの度に内部でログレベルを WARNING に戻してしまうため、
# set_verbosity では抑制できない。ロガーに直接フィルタを付けて WARNING 以下を常に捨てる。
_nemo_logger = logging.getLogger("nemo_logger")
_nemo_logger.addFilter(lambda record: record.levelno >= logging.ERROR)

# ビームサーチデコーダーはサンプル毎に "Beam search progress:" という tqdm を
# 無条件で生成する(verbose=False でも止まらない)。常に disable=True にして黙らせる。
_disabled_tqdm = lambda *args, **kwargs: tqdm(*args, **{**kwargs, "disable": True})
rnnt_beam_decoding.tqdm = _disabled_tqdm
tdt_beam_decoding.tqdm = _disabled_tqdm

MODEL_NAME = "reazon-research/reazonspeech-nemo-v2"


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

    output_path = args.output or RESULTS_DIR / f"reazonspeech_nemo_{args.dataset}_results.csv"

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


# reazon-research/reazonspeech-nemo-v2 の学習データ
#
# 学習に使用されたデータセット:
#   - ReazonSpeech v2.0 (https://huggingface.co/datasets/reazon-research/reazonspeech)
#     Parakeet日本語版と同様、このモデル自体もReazonSpeechの開発元が
#     ReazonSpeechコーパスを使って学習している。
#     したがって --dataset reazonspeech はリークの可能性が高く、
#     フェアな精度評価には --dataset jsut を使うこと。
