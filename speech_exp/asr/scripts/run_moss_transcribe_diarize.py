"""日本語音声データセットに対して MOSS-Transcribe-Diarize で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

MOSS-Transcribe-Diarize: 書き起こし+話者分離(diarization)+タイムスタンプを
1モデルで一気にこなすend-to-endモデル(0.9B, 50言語以上対応、日本語含む)。
INTERSPEECH 2026 MLC-SLM Challenge(日本語含む14言語)優勝モデル。
https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize

出力は `[開始時刻][S01]発話内容[終了時刻]` 形式のタグ付きテキストになるため、
CERを見るために話者ラベル・タイムスタンプを除去したプレーンテキストも別列に出す。
JSUT/ReazonSpeechは単一話者なので話者分離の効果自体は見えないが、
書き起こし精度そのものは他モデルと同じ形式で比較できる。

使い方:
    uv run python scripts/run_moss_transcribe_diarize.py --dataset jsut
    uv run python scripts/run_moss_transcribe_diarize.py --dataset jsut --num-samples 5
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import csv
import random
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

from datasets_ja import DATASETS, RESULTS_DIR

MODEL_NAME = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# パッケージ付属の parse_transcript() は "[start][SXX]text[end]" のように
# 話者ラベルが必須の厳密フォーマットしかパースできず、モデルが稀に話者ラベルを
# 省略して "[start]text[end]" と出力した場合に空文字を返してしまう
# (実測でJSUT 5000件中259件で発生)。話者ラベルの有無に関わらず動く正規表現で代替する。
_SEGMENT_RE = re.compile(r"\[[\d.]+\](?:\[S\d+\])?([^\[\]]*)\[[\d.]+\]")


def plain_text_from_transcript(raw_text: str) -> str:
    """タイムスタンプ・話者ラベルを除いた発話内容だけを時系列順に連結する。"""
    return "".join(m.group(1) for m in _SEGMENT_RE.finditer(raw_text))


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

    output_path = args.output or RESULTS_DIR / f"moss_transcribe_diarize_{args.dataset}_results.csv"

    samples = DATASETS[args.dataset]()
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"loading {MODEL_NAME} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype="auto",
        attn_implementation="sdpa",
    ).to(dtype=dtype).to(device).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred", "pred_raw_with_tags", "inference_time_sec"])

        for sample_id, audio_path, gt_text in tqdm(samples, desc="transcribing"):
            start = time.perf_counter()

            messages = build_transcription_messages(str(audio_path))
            result = generate_transcription(
                model,
                processor,
                messages,
                max_new_tokens=512,
                do_sample=False,
                device=device,
                dtype=dtype,
            )
            raw_text = result["text"]
            pred_text = plain_text_from_transcript(raw_text)

            elapsed = time.perf_counter() - start
            writer.writerow([sample_id, gt_text, pred_text, raw_text, f"{elapsed:.4f}"])

    print(f"\nwrote {len(samples)} rows to {output_path}")


if __name__ == "__main__":
    main()


# OpenMOSS-Team/MOSS-Transcribe-Diarize の学習データについて
#
# 具体的な学習データセット名(ReazonSpeechやJSUTなど)はモデルカードに明示されていない。
# → ReazonSpeech/JSUTいずれについても学習データとの重複(リーク)の有無を確認できないため、
#   他モデルほど「フェアな評価」を断定はできない点に注意。
#
# なお本モデルの主眼は「複数話者の書き起こし+話者分離+タイムスタンプ」であり、
# JSUT/ReazonSpeechのような単一話者データでは強みの一部(話者分離)は評価できない。
# 複数話者の音声(会議・対談など)で試すとこのモデルの本領が見えるはず。
