"""ASR(Parakeet) -> LLM(Ollama, ストリーミング) -> TTS(VOICEVOX) のカスケードパイプライン。

LLMの出力は句読点(。！？)ごとに文単位で切り出し、文が完成するたびに
TTS合成をバックグラウンドスレッドプールに投げてLLMの続きの生成と並行させる
(文単位パイプライン化)。これにより「最初の一文の音声ができるまでの時間」が
LLM全文生成完了を待つ場合より短くなる、という設計を実測できるようにしてある。

このモジュール単体でASR/LLM/TTSの各処理とタイミング計測を担当し、
Gradio側(app.py)はこれを呼び出してUI表示するだけにする。
"""

import os
import re
import time
import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import requests
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr

_nemo_logger = logging.getLogger("nemo_logger")
_nemo_logger.addFilter(lambda record: record.levelno >= logging.ERROR)

# --- 設定 ---
ASR_MODEL_NAME = "nvidia/parakeet-tdt_ctc-0.6b-ja"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"
VOICEVOX_URL = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER = 3  # ずんだもん・ノーマル
VOICEVOX_SAMPLE_RATE = 24000

SYSTEM_PROMPT = (
    "あなたは音声で会話するアシスタントです。応答は音声合成でそのまま読み上げられます。\n"
    "・簡潔に1〜2文程度で日本語で応答する\n"
    "・Markdown記法(**太字**、見出し#、箇条書き-や1.、水平線---など)は絶対に使わず、"
    "話し言葉の文章だけを出力する"
)

_SENTENCE_END_RE = re.compile(r"[。！？\n]")


# --- ASRモデルはプロセス起動時に一度だけロードする ---
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading {ASR_MODEL_NAME} on {_device} ...")
_asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
_asr_model = _asr_model.to(_device)
_asr_model.eval()
print("ASR model ready.")


def transcribe(audio_path: str) -> tuple[str, float]:
    """録音済み音声ファイルを文字起こしする。(テキスト, 所要秒数) を返す。"""
    start = time.perf_counter()
    outputs = _asr_model.transcribe([audio_path], batch_size=1, verbose=False)
    text = outputs[0].text if hasattr(outputs[0], "text") else str(outputs[0])
    elapsed = time.perf_counter() - start
    return text.strip(), elapsed


_MARKDOWN_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)  # ---, ***, ___
_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)  # # 見出し
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)  # - item, 1. item
_MARKDOWN_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|__|_)(.+?)\1")  # **太字**, *斜体*
_MARKDOWN_INLINE_CODE_RE = re.compile(r"`([^`]*)`")


def strip_markdown(text: str) -> str:
    """LLMがMarkdown記法で出力してしまった場合に備え、TTSに渡す前に記号を除去する。
    (プロンプトで禁止していても完全には防げないため、防御的に処理する)
    """
    text = _MARKDOWN_HR_RE.sub("", text)
    text = _MARKDOWN_HEADING_RE.sub("", text)
    text = _MARKDOWN_LIST_RE.sub("", text)
    text = _MARKDOWN_EMPHASIS_RE.sub(r"\2", text)
    text = _MARKDOWN_INLINE_CODE_RE.sub(r"\1", text)
    return text.strip()


def synthesize(text: str, speaker: int = VOICEVOX_SPEAKER) -> tuple[np.ndarray, float]:
    """VOICEVOXで1文を音声合成する。(波形(float32, mono), 所要秒数) を返す。"""
    text = strip_markdown(text)
    if not text:
        return np.zeros(0, dtype="float32"), 0.0

    start = time.perf_counter()
    query = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    ).json()
    wav_bytes = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=30,
    ).content
    elapsed = time.perf_counter() - start

    import io
    data, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return data, elapsed


@dataclass
class TurnMetrics:
    asr_sec: float = 0.0
    llm_first_token_sec: float | None = None
    llm_first_sentence_sec: float | None = None
    tts_first_sentence_sec: float | None = None
    llm_total_sec: float | None = None
    total_sec: float | None = None
    sentence_count: int = 0

    def as_text(self) -> str:
        def fmt(v):
            return f"{v * 1000:.0f}ms" if v is not None else "-"

        return (
            f"ASR:                {fmt(self.asr_sec)}\n"
            f"LLM 初回トークンまで: {fmt(self.llm_first_token_sec)}\n"
            f"LLM 最初の1文まで:    {fmt(self.llm_first_sentence_sec)}\n"
            f"最初の1文のTTS合成:  {fmt(self.tts_first_sentence_sec)}\n"
            f"体感上の初回応答まで: {fmt(self._time_to_first_audio())}\n"
            f"LLM 全文生成:        {fmt(self.llm_total_sec)}\n"
            f"合計(全文音声完成まで): {fmt(self.total_sec)}\n"
            f"文の数:              {self.sentence_count}"
        )

    def _time_to_first_audio(self):
        if self.llm_first_sentence_sec is None or self.tts_first_sentence_sec is None:
            return None
        return self.llm_first_sentence_sec + self.tts_first_sentence_sec


def _llm_stream_producer(
    messages: list[dict], event_queue: "queue.Queue", metrics: TurnMetrics
):
    """Ollamaをストリーミングで呼び、トークン/文完成イベントをqueueに流す(別スレッドで実行)。"""
    start = time.perf_counter()
    buffer = ""
    full_text = ""
    sentence_count = 0

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
        for line in resp.iter_lines():
            if not line:
                continue
            import json

            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")

            if token:
                if metrics.llm_first_token_sec is None:
                    metrics.llm_first_token_sec = time.perf_counter() - start
                full_text += token
                buffer += token
                event_queue.put(("text_update", full_text))

            match = _SENTENCE_END_RE.search(buffer)
            while match:
                sentence = buffer[: match.end()].strip()
                buffer = buffer[match.end() :]
                if sentence:
                    sentence_count += 1
                    if metrics.llm_first_sentence_sec is None:
                        metrics.llm_first_sentence_sec = time.perf_counter() - start
                    event_queue.put(("sentence", sentence))
                match = _SENTENCE_END_RE.search(buffer)

            if chunk.get("done"):
                break

        # 末尾に句読点なしの断片が残っていたら最後の文として扱う
        if buffer.strip():
            sentence_count += 1
            event_queue.put(("sentence", buffer.strip()))

        metrics.llm_total_sec = time.perf_counter() - start
        metrics.sentence_count = sentence_count
        event_queue.put(("llm_done", full_text))

    except Exception as e:  # noqa: BLE001
        event_queue.put(("error", str(e)))


def run_turn(user_text: str, history_messages: list[dict], metrics: TurnMetrics):
    """
    1ターン分(ユーザー発話テキスト -> LLM応答生成 -> 文単位TTS)を処理するジェネレーター。

    yieldする内容: ("llm_text", 現時点までのLLM応答全文)
                  ("audio_chunk", (波形, サンプリングレート))  # 文が1つ合成できるたび
                  ("final", (LLM応答全文, 結合済み波形, サンプリングレート))
    """
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_text}]
    )

    event_queue: queue.Queue = queue.Queue()
    producer = __import__("threading").Thread(
        target=_llm_stream_producer, args=(messages, event_queue, metrics), daemon=True
    )
    producer.start()

    tts_pool = ThreadPoolExecutor(max_workers=2)
    pending_futures: list = []
    audio_chunks: list[np.ndarray] = []
    first_sentence_tts_measured = False
    full_text = ""
    llm_done = False

    while True:
        try:
            kind, payload = event_queue.get(timeout=0.05)
        except queue.Empty:
            kind, payload = None, None

        if kind == "text_update":
            full_text = payload
            yield ("llm_text", full_text)

        elif kind == "sentence":
            sentence = payload
            is_first = not first_sentence_tts_measured
            first_sentence_tts_measured = True

            def _synth_and_time(sentence=sentence, is_first=is_first):
                data, elapsed = synthesize(sentence)
                if is_first:
                    metrics.tts_first_sentence_sec = elapsed
                return data

            pending_futures.append(tts_pool.submit(_synth_and_time))

        elif kind == "llm_done":
            full_text = payload
            llm_done = True

        elif kind == "error":
            yield ("error", payload)
            tts_pool.shutdown(wait=False)
            return

        # 完了しているTTSタスクを順番に取り出して音声チャンクとしてyield
        while pending_futures and pending_futures[0].done():
            fut = pending_futures.pop(0)
            data = fut.result()
            if data.size == 0:
                continue  # Markdown記号のみ等、除去後に空文字だった断片はスキップ
            audio_chunks.append(data)
            yield ("audio_chunk", (data, VOICEVOX_SAMPLE_RATE))

        if llm_done and not pending_futures:
            break

    tts_pool.shutdown(wait=True)

    if audio_chunks:
        combined = np.concatenate(audio_chunks)
    else:
        combined = np.zeros(0, dtype="float32")

    yield ("final", (full_text, combined, VOICEVOX_SAMPLE_RATE))
