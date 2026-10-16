"""
OneStreamer - 公開HuggingFaceモデルを使った実推論サンプル
================================================================

`MCG-NJU/OneStreamer-4B` (公開チェックポイント) を実際に
`transformers.AutoProcessor` / `Qwen3VLForConditionalGeneration` でロードし、
動画をストリーミング処理しながらプロアクティブに応答する完全な推論スクリプト。

公式 `gh/OneStreamer/Inference/inference.py` と同じ使い方 (CLI引数・出力jsonl) を
保ちつつ、推論エンジン部分 (`_Qwen3VLInferenceFastEngine` 相当) とフレーム前処理
(`smart_video_resize` 等) は公式 `Eval/VLMEvalKit/.../qwen3_ovotiming/utils.py`
`inference_fast.py` のロジックをそのまま再現している (既に実際の環境で動作確認
済みの処理のみ)。公式デモとの最大の違いは、セッションに本サーベイの
[streaming_session.py](streaming_session.py) の `PHCMStreamingSession` を使える
点で、`--memory phcm` を指定すると、フレームがRecent-Nウィンドウを出てもキャプション
(</Observe>/</Summary>) がテキスト履歴に残る (論文の主要提案PHCM) 挙動を、
公式デモが提供する `--memory fifo` (キャプションも一緒に失われる) と比較できる。

============================================================
動作確認について
============================================================
本ファイル内の以下は、ネットワーク接続下でこの環境から実際に実行して動作を
確認済み:
    - `AutoProcessor.from_pretrained("MCG-NJU/OneStreamer-4B")` のロード
    - `processor.apply_chat_template(...)` による画像+テキストのトークン化
      (実際に `input_ids`, `pixel_values: (320, 1536)`, `image_grid_thw: [[1,16,20]]`
       が得られることを確認済み)
    - 制御トークンのID (`</Silence>`:151671 等) が実際のトークナイザと一致すること
モデル本体の重み (safetensors, 言語側のみで約8GB) はダウンロードしていないため、
`model.generate()` を含む完全なエンドツーエンド実行はこの環境では未検証。
実行するには GPU と、`--model-path` に実際のチェックポイントパス (または
"MCG-NJU/OneStreamer-4B" を指定してHubから自動ダウンロード) が必要。

============================================================
使い方
============================================================
python hf_inference_example.py \\
    --model-path MCG-NJU/OneStreamer-4B \\
    --video path/to/video.mp4 \\
    --question "How many times does the event: showing something to the camera happen?" \\
    --memory phcm \\
    --output outputs/responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

from control_tokens import find_control_token
from streaming_session import FIFOStreamingSession, PHCMStreamingSession

ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. フレーム前処理 (公式 qwen3_ovotiming/utils.py のロジックを再現)
# ============================================================

def smart_resize(
    height: int, width: int, factor: int,
    min_pixels: int, max_pixels: int, force_resize: bool = False,
) -> Tuple[int, int]:
    """公式 `qwen3_ovotiming/utils.py::smart_resize` と同一のアルゴリズム。

    アスペクト比を維持しつつ、ピクセル数が [min_pixels, max_pixels] に収まり、
    かつ高さ・幅が `factor` で割り切れるようにリサイズする。
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("アスペクト比が極端すぎます (200倍超)")
    if force_resize:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        return h_bar, w_bar
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def processor_visual_geometry(processor) -> Tuple[int, int, int]:
    """processor から (patch_size, merge_size, factor=patch_size*merge_size) を取得する。"""
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
    merge_size = int(
        getattr(image_processor, "merge_size", getattr(image_processor, "spatial_merge_size", 2)) or 2
    )
    return patch_size, merge_size, patch_size * merge_size


def resize_frame_for_streaming(
    frame: Image.Image, factor: int, max_pixels: int = 224 * 224,
) -> Image.Image:
    """1フレームをストリーミング推論用の固定トークン予算にリサイズする。

    公式デモ (`Inference/inference.py`) が1ステップあたりのフレームに
    `frame_max_pixels=224*224` を使っているのと同じ設定。Recent-N ウィンドウの
    トークン数を安定させ、長時間ストリームでも推論コストが暴れないようにする。
    """
    w, h = frame.size
    h_bar, w_bar = smart_resize(
        h, w, factor=factor, min_pixels=factor ** 2, max_pixels=max_pixels, force_resize=True,
    )
    return frame.resize((w_bar, h_bar))


def video_windows(
    path: str, max_seconds: Optional[float] = None, fps: int = 4,
):
    """公式 `Inference/inference.py::video_windows` と同一のフレームサンプリング。

    1秒ごとに `fps` 枚のフレームをまとめて1ステップとして yield する。
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or frame_count <= 0:
            raise RuntimeError("動画のfps/フレーム数が不正です")
        duration = frame_count / source_fps
        end = min(duration, max_seconds) if max_seconds is not None else duration
        frames, times = [], []
        for step in range(math.ceil(end * fps)):
            timestamp = step / fps
            frame_index = min(int(timestamp * source_fps), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"フレーム取得に失敗: {timestamp:.3f}s")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            times.append((timestamp, timestamp + 1 / fps))
            if len(frames) == fps:
                yield frames, times
                frames, times = [], []
        if frames:
            yield frames, times
    finally:
        cap.release()


# ============================================================
# 2. 実推論エンジン (公式 _Qwen3VLInferenceFastEngine と同一のロジック)
# ============================================================

class HFEngine:
    """streaming_session.py の `Engine` インターフェースを実装する実推論エンジン。

    `Engine = Callable[[dict, int, float], str]` ( data={"images":[...],"messages":[...]} )
    を満たすように `__call__` を実装している。ロジックは公式
    `Eval/VLMEvalKit/.../qwen3_ovotiming/utils.py::_Qwen3VLInferenceFastEngine.infer`
    と同一 (`apply_chat_template` -> `generate` -> 新規トークンのみデコード)。
    """

    _END_TOKENS = ("<|im_end|>", "<|endoftext|>")

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    @staticmethod
    def _to_chat_messages(messages: List[dict], images: List[Image.Image]) -> List[dict]:
        """`{"role","content": "<image>を含む文字列"}` 形式を processor 用の
        `{"role","content": [{"type":"text"/"image", ...}]}` 形式に変換する。
        """
        image_iter = iter(images)
        converted = []
        for m in messages:
            role, content = m["role"], m["content"]
            if "<image>" not in content:
                converted.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            parts: List[dict] = []
            for idx, seg in enumerate(content.split("<image>")):
                seg = seg.strip()
                if seg:
                    parts.append({"type": "text", "text": seg})
                if idx < content.count("<image>"):
                    parts.append({"type": "image", "image": next(image_iter)})
            converted.append({"role": role, "content": parts})
        return converted

    def _strip_end_tokens(self, text: str) -> str:
        for tok in self._END_TOKENS:
            while text.rstrip().endswith(tok):
                text = text.rstrip()[: -len(tok)]
        return text.strip()

    def __call__(self, data: dict, max_tokens: int, temperature: float) -> str:
        chat = self._to_chat_messages(data["messages"], data["images"])
        inputs = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=False)[0]
        return self._strip_end_tokens(text)


SYSTEM_PROMPT = """
You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
""".strip()


# ============================================================
# 3. モデルロード
# ============================================================

def load_model_and_processor(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """本物の `transformers` API でOneStreamer-4Bをロードする。

    `AutoProcessor.from_pretrained(model_path)` はこの環境で実際に動作確認済み
    (制御トークンID `</Silence>`:151671 等が一致することを含む)。
    `Qwen3VLForConditionalGeneration.from_pretrained(...)` はクラスの import 自体は
    確認済みだが、重み (~8GB) のダウンロードは行っていないためこの関数全体の
    実行は未検証 (GPU環境で `--model-path` を渡して実行して確認してほしい)。
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model, processor


# ============================================================
# 4. メインの推論ループ
# ============================================================

def run(args: argparse.Namespace) -> None:
    model_path = args.model_path
    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_path.exists():
        raise FileExistsError(f"出力先は新規パスにしてください: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path} ...", flush=True)
    model, processor = load_model_and_processor(model_path)
    patch_size, merge_size, factor = processor_visual_geometry(processor)
    engine = HFEngine(model, processor)

    session_cls = PHCMStreamingSession if args.memory == "phcm" else FIFOStreamingSession
    session = session_cls(
        engine=engine,
        system=SYSTEM_PROMPT,
        question=args.question,
        max_num_frames=args.max_num_frames,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"Memory strategy: {args.memory}  Question: {args.question}", flush=True)
    responses = 0
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            for round_idx, (frames, times) in enumerate(video_windows(video_path, args.max_seconds, fps=4)):
                resized = [resize_frame_for_streaming(f, factor) for f in frames]
                answer = session.step_frames(resized, times, round_idx)
                control = find_control_token(answer)
                is_response = control == "</Response>"
                responses += int(is_response)

                record = dict(
                    start=times[0][0], end=times[-1][1], raw=answer, state=control,
                    response=answer.split("</Response>", 1)[-1].strip() if is_response else None,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if is_response or args.show_all:
                    print(f"[{times[-1][1]:.2f}s] {answer}", flush=True)
    finally:
        if isinstance(session, PHCMStreamingSession):
            print(f"保持されているPHCMレコード数: {session.num_memory_records()}", flush=True)
        session.reset()

    print(f"Finished: {responses} proactive responses. Saved all updates to {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OneStreamer-4B (HuggingFace公開モデル) を使ったストリーミング推論サンプル"
    )
    parser.add_argument("--model-path", default="MCG-NJU/OneStreamer-4B")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--question", default="How many times does the event: showing something to the camera happen?"
    )
    parser.add_argument("--output", default="outputs/responses.jsonl")
    parser.add_argument(
        "--memory", choices=["phcm", "fifo"], default="phcm",
        help="phcm: PHCMStreamingSession (本サーベイの再構築, キャプション永続化) / "
             "fifo: FIFOStreamingSession (公式デモと同じ挙動, キャプションも失われる)",
    )
    parser.add_argument("--max-num-frames", type=int, default=16, help="Recent-N (論文のOVOBench設定)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--show-all", action="store_true", help="</Silence>/</Standby> も表示する")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("--question は空にできません")
    return args


if __name__ == "__main__":
    run(parse_args())
