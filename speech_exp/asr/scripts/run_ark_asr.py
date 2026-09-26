"""日本語音声データセットに対して ARK-ASR で推論し、
正解(GT)・予測書き起こし・推論時間をCSVに出力する。

ARK-ASR: Whisper系エンコーダー + Qwen2デコーダーの多言語ASR(19言語、日本語含む)。
teacher-data adaptation + online policy distillation で学習されたモデル。
https://huggingface.co/AutoArk-AI/ARK-ASR-0.6B

使い方:
    uv run python scripts/run_ark_asr.py --dataset jsut                       # 全ファイル推論(0.6B)
    uv run python scripts/run_ark_asr.py --dataset jsut --num-samples 5
    uv run python scripts/run_ark_asr.py --dataset jsut --model-size 3B
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from datasets_ja import DATASETS, RESULTS_DIR

MODEL_NAMES = {
    "0.6B": "AutoArk-AI/ARK-ASR-0.6B",
    "3B": "AutoArk-AI/ARK-ASR-3B",
}


def build_bad_words_ids(tokenizer):
    """特殊トークン(ASRテキスト以外のタグ等)が出力に混ざらないよう禁止リストを作る。"""
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )
    return [[token_id] for token_id in sorted(bad_ids)]


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

    model_path = MODEL_NAMES[args.model_size]
    output_path = args.output or RESULTS_DIR / f"ark_asr_{args.model_size}_{args.dataset}_results.csv"

    samples = DATASETS[args.dataset]()
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"loading {model_path} on {device} ...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    bad_words_ids = build_bad_words_ids(tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred", "inference_time_sec"])

        for sample_id, audio_path, gt_text in tqdm(samples, desc="transcribing"):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": str(audio_path)},
                        {"type": "text", "text": "Please transcribe this audio."},
                    ],
                }
            ]

            start = time.perf_counter()

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                sampling_rate=16000,
                audio_padding="longest",
                text_kwargs={"padding": "longest"},
                audio_max_length=30 * 16000,
            )
            inputs = inputs.to(device)
            if "audios" in inputs:
                inputs["audios"] = inputs["audios"].to(dtype=torch_dtype)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=bad_words_ids,
                )
            pred_text = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()

            elapsed = time.perf_counter() - start
            writer.writerow([sample_id, gt_text, pred_text, f"{elapsed:.4f}"])

    print(f"\nwrote {len(samples)} rows to {output_path}")


if __name__ == "__main__":
    main()


# AutoArk-AI/ARK-ASR-{0.6B,3B} の学習データについて
#
# teacher-data adaptation + online policy distillation (TD+OPD) という手法で学習。
# 静的な書き起こしデータだけでなく、生成した書き起こしをより強い教師モデルに
# トークンレベルでスコアリングさせながらオンラインで学習する方式。
# 具体的な学習データセット名(ReazonSpeechやJSUTなど)はモデルカードに明示されていない。
# → ReazonSpeech/JSUTいずれについても学習データとの重複(リーク)の有無を確認できないため、
#   他モデルほど「フェアな評価」を断定はできない点に注意。
