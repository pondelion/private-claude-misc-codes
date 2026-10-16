"""
OneStreamer - Streaming Inference Session (キー実装: FIFOウィンドウ & PHCM)
================================================================================

推論時にストリーミング動画を1ステップずつ処理し、Recent-N FIFO視覚ウィンドウと
テキスト履歴 (対話 + 記憶レコード) を保持しながら制御トークンを生成していく
セッションオブジェクトを実装する。

論文: OneStreamer (arXiv:2610.01762)
      - Sec 3.1 Overall Architecture (Recent-N FIFOウィンドウの定義)
      - Sec 3.2 Proactive Hierarchical Caption Memory (PHCM)
      - Appendix A.2 "Online Inference with PHCM"
          "At inference time, OneStreamer processes the video incrementally and
           maintains two complementary forms of history: a Recent-N FIFO visual
           window and an accumulated textual caption memory. ... The source visual
           frames may subsequently leave the FIFO window, while the generated
           record remains available to later predictions."

公式コード: gh/OneStreamer/Eval/VLMEvalKit/vlmeval/vlm/qwen3_ovotiming/inference_fast.py
      の `StreamingSession` クラスをベースにしている。ただし公式コードの
      `_trim_sliding_window()` は、ウィンドウから最古のフレームが押し出される際に
      そのフレームに対応する assistant の応答メッセージ (= </Observe>/</Summary> の
      キャプションを含む可能性がある) も **まとめて削除**している
      (`del messages[first_u : first_u + 2]`)。つまり公開されている推論デモは
      Recent-N FIFO ベースライン (論文 Table "Answer-stage efficiency" でいう
      "FIFO" 行) であり、論文の主要提案である PHCM (キャプションをフレームより
      長く保持する) のメモリ保持ロジック自体は公開コードに含まれていない。
      本ファイルはまず公式コードに忠実な `FIFOStreamingSession` を再現し、
      その後 Appendix A.2 の記述に基づいて PHCM を実装した
      `PHCMStreamingSession` を追加する (論文記述からの再構築であることを明記)。

============================================================
Shape Convention
============================================================
images (セッション内部): List[PIL.Image]  - Recent-N ウィンドウに残っているフレーム
messages: List[Dict]  - {"role": "system"/"user"/"assistant", "content": str}
max_num_frames: int   - Recent-N の N (論文 Table "Answer-stage efficiency" では
                         OVOBench/StreamingBench 向けに Recent-16 を使用)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from control_tokens import OBSERVE, SUMMARY, find_control_token


# ============================================================
# 0. 推論エンジンの抽象インターフェース
# ============================================================
#
# 公式 `_Qwen3VLInferenceFastEngine.infer(data, max_tokens, temperature) -> str` と
# 同じシグネチャ。実運用では Qwen3-VL-4B-Instruct (+ 追加された制御トークン) を
# ロードした transformers モデルをラップしたものを渡す
# (詳細な中身は姉妹フォルダ survey/Qwen3VL の main_flow.py / vision_encoder.py を参照。
#  OneStreamer はこの推論エンジンの「中身」を変えているわけではなく、
#  「どういう履歴 (data) を渡すか」を変えている点が本ファイルの主題)。
Engine = Callable[[dict, int, float], str]


@dataclass
class _TurnMeta:
    round_idx: int
    time_secs: List[Tuple[float, float]]


# ============================================================
# 1. 公式コード相当の Recent-N FIFO ベースライン
# ============================================================

class FIFOStreamingSession:
    """公式 `inference_fast.py::StreamingSession` を再現した Recent-N FIFO セッション。

    ウィンドウからフレームが押し出される際、そのフレームに対応する
    user(画像)メッセージと assistant(応答)メッセージの **両方** を削除する。
    したがって </Observe>/</Summary> で記録したキャプションも、
    そのフレームがウィンドウから出た時点で失われる
    (論文 Table "Answer-stage efficiency" の "FIFO" 行に相当)。
    """

    def __init__(
        self,
        engine: Engine,
        system: str,
        question: Optional[str] = None,
        question_time: int = 0,
        max_num_frames: int = 16,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_time = question_time
        self.max_num_frames = max_num_frames
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._messages: List[Dict] = []
        self._images: List = []
        self._turn_image_counts: List[int] = []
        self._turn_meta: List[_TurnMeta] = []
        self._last_answer: Optional[str] = None
        self._question_injected = False

    # -- メッセージ構築 -------------------------------------------------

    @staticmethod
    def _user_content(time_secs: Sequence[Tuple[float, float]]) -> str:
        return "\n".join(f"<{lo:.3f}s-{hi:.3f}s>\n<image>" for lo, hi in time_secs)

    def _maybe_inject_question(self, round_idx: int) -> None:
        if self.question is None or self._question_injected:
            return
        inject = self.global_question or (round_idx == self.question_time)
        if inject:
            self._messages.append({"role": "user", "content": self.question})
            self._question_injected = True

    def _append_turn(
        self, images: Sequence, time_secs: Sequence[Tuple[float, float]], round_idx: int
    ) -> None:
        if not self._messages:
            self._messages.append({"role": "system", "content": self.system})
        self._maybe_inject_question(round_idx)
        if self._last_answer is not None:
            self._messages.append({"role": "assistant", "content": self._last_answer})
        self._messages.append(
            {"role": "user", "content": self._user_content(time_secs)}
        )
        self._images.extend(images)
        self._turn_image_counts.append(len(images))
        self._turn_meta.append(_TurnMeta(round_idx, list(time_secs)))
        self._trim_sliding_window()

    def _first_image_message_index(self) -> int:
        for i, msg in enumerate(self._messages):
            if msg["role"] == "user" and "<image>" in msg["content"]:
                return i
        return len(self._messages)

    def _trim_sliding_window(self) -> None:
        """Recent-N を超えたら最古のフレーム一式 (画像+応答) を削除する (公式仕様)。"""
        while len(self._images) > self.max_num_frames:
            first_u = self._first_image_message_index()
            if first_u + 1 >= len(self._messages):
                break
            n0 = self._turn_image_counts[0]
            # 公式コード同様、user(画像) と直後の assistant(応答) をまとめて削除する。
            # => </Observe>/</Summary> のキャプションもここで一緒に失われる。
            del self._messages[first_u : first_u + 2]
            self._images = self._images[n0:]
            self._turn_image_counts.pop(0)
            self._turn_meta.pop(0)

    # -- 公開API ---------------------------------------------------------

    def step_frames(
        self, frames: Sequence, time_secs: Sequence[Tuple[float, float]], round_idx: int
    ) -> str:
        """1ステップ分のフレームを追加し、モデルに推論させて生成テキストを返す。"""
        self._append_turn(frames, time_secs, round_idx)
        data = {"images": list(self._images), "messages": list(self._messages)}
        answer = self.engine(data, self.max_tokens, self.temperature)
        self._last_answer = answer
        return answer

    @property
    def messages(self) -> List[Dict]:
        return list(self._messages)

    @property
    def window_size(self) -> int:
        return len(self._images)

    def reset(self) -> None:
        self.__init__(  # type: ignore[misc]
            self.engine, self.system, self.question, self.question_time,
            self.max_num_frames, self.global_question, self.max_tokens, self.temperature,
        )


# ============================================================
# 2. PHCM を実装したセッション (論文 Appendix A.2 の再構築)
# ============================================================

class PHCMStreamingSession(FIFOStreamingSession):
    """Proactive Hierarchical Caption Memory を実装したストリーミングセッション。

    `FIFOStreamingSession` との唯一の違いは `_trim_sliding_window` の挙動:
    押し出されるフレームに対応する応答が </Observe> または </Summary> だった場合、
    その **応答テキスト (キャプション) だけはテキスト履歴に永続的に残し**、
    画像メッセージのみを削除する。</Silence> だった場合は (記録すべき内容がないので)
    画像・応答ともに削除する (FIFO と同じ)。

    これにより「直近 Recent-N フレームの生の視覚特徴」と「それより前の時間帯の
    キャプション記憶」が共存するテキスト履歴が構築される
    (論文 Appendix A.2 の記述通り)。
    """

    def _trim_sliding_window(self) -> None:
        while len(self._images) > self.max_num_frames:
            first_u = self._first_image_message_index()
            if first_u + 1 >= len(self._messages):
                break
            n0 = self._turn_image_counts[0]

            # 押し出される画像ターンの直後にある assistant 応答 (あれば) を調べる。
            has_answer = (
                first_u + 1 < len(self._messages)
                and self._messages[first_u + 1]["role"] == "assistant"
            )
            answer_text = self._messages[first_u + 1]["content"] if has_answer else ""
            control = find_control_token(answer_text) if has_answer else None
            is_memory_record = control in (OBSERVE, SUMMARY)

            if is_memory_record:
                # 画像メッセージだけ削除し、キャプション(assistant応答)は
                # テキスト履歴に残す = PHCM のメモリレコードとして永続化。
                del self._messages[first_u]
            else:
                # </Silence> など記録すべき内容がない場合は FIFO と同様に両方削除。
                del self._messages[first_u : first_u + 2]

            self._images = self._images[n0:]
            self._turn_image_counts.pop(0)
            self._turn_meta.pop(0)

    def num_memory_records(self) -> int:
        """現在のテキスト履歴に残っている PHCM レコード (Observe/Summary) の数。"""
        count = 0
        for msg in self._messages:
            if msg["role"] == "assistant":
                tok = find_control_token(msg["content"])
                if tok in (OBSERVE, SUMMARY):
                    count += 1
        return count


# ============================================================
# 使用例: FIFO vs PHCM の挙動比較
# ============================================================

def example_usage() -> None:
    """合成フレーム列に対し、FIFO と PHCM でテキスト履歴がどう変わるかを比較する。

    `FakeEngine` は実モデルの代わりに、あらかじめ決めた制御トークンの列を
    順番に返すだけの決定的なスタブ (実モデルの重みがこの環境に無いため)。
    履歴の構築・トリミングロジック自体は本物であり、ここでは
    「フレームがウィンドウから出た後もキャプションが残るかどうか」という
    PHCM の核心的な挙動を実際に比較検証する (printでごまかしていない)。
    """
    print("=== streaming_session.py Example Usage ===\n")

    # 0..19 の20ステップ、各ステップ1フレーム。4ステップごとにキャプションを記録。
    scripted_answers = []
    for step in range(20):
        if step % 4 == 3:
            scripted_answers.append(f"</Observe> ステップ{step}時点の要約キャプション")
        else:
            scripted_answers.append("</Silence>")

    class FakeEngine:
        def __init__(self, answers: List[str]):
            self.answers = answers
            self.calls = 0

        def __call__(self, data: dict, max_tokens: int, temperature: float) -> str:
            ans = self.answers[self.calls]
            self.calls += 1
            return ans

    def run_session(session_cls) -> Tuple[List[Dict], int]:
        engine = FakeEngine(scripted_answers)
        session = session_cls(
            engine=engine,
            system="You are a helpful streaming video assistant.",
            question="showing somethingの回数を数えてください",
            max_num_frames=4,  # 小さくしてすぐトリミングが起きるようにする
        )
        for step in range(20):
            frame_placeholder = f"frame_{step}"  # 実画像の代わりに文字列で十分 (shapeの検証に画素値は不要)
            session.step_frames([frame_placeholder], [(float(step), float(step + 1))], step)
        return session.messages, engine.calls

    fifo_messages, fifo_calls = run_session(FIFOStreamingSession)
    phcm_messages, phcm_calls = run_session(PHCMStreamingSession)

    fifo_captions = sum(
        1 for m in fifo_messages if m["role"] == "assistant" and "</Observe>" in m["content"]
    )
    phcm_captions = sum(
        1 for m in phcm_messages if m["role"] == "assistant" and "</Observe>" in m["content"]
    )

    print(f"推論ステップ数: {fifo_calls} (FIFO) / {phcm_calls} (PHCM) (同一シナリオ)")
    print(f"最終的なメッセージ総数: FIFO={len(fifo_messages)}, PHCM={len(phcm_messages)}")
    print(f"履歴に残っている </Observe> キャプション数: FIFO={fifo_captions}, PHCM={phcm_captions}")
    print(
        "\n=> FIFO は Recent-4 ウィンドウを超えた古いキャプションを失うため "
        f"{fifo_captions} 件しか残らないが、PHCM はフレームがウィンドウを出た後も "
        f"キャプションをテキスト履歴として保持し続けるため {phcm_captions} 件が残る "
        "(直近ウィンドウ内にまだ残っている最新の </Observe> は self._last_answer 側に"
        "あり、次ステップが無いためまだ messages には追加されていない点は公式実装と同じ仕様)。"
    )

    print("\n--- PHCM の最終メッセージ履歴 (抜粋) ---")
    for msg in phcm_messages[:9]:
        content_preview = msg["content"][:40].replace("\n", " ")
        print(f"  [{msg['role']:9s}] {content_preview}")


if __name__ == "__main__":
    example_usage()
