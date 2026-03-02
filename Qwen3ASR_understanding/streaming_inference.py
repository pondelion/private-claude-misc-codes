"""
Qwen3-ASR - ストリーミング推論パイプライン 詳細
================================================

このファイルはQwen3-ASRのストリーミング (チャンクベース) 推論の
詳細な処理フローを理解するための疑似コードです。

論文: https://arxiv.org/abs/2601.21337 (Section 2.4, 4.5)
公式実装: qwen_asr/inference/qwen3_asr.py (LLM class)
          qwen_asr/core/vllm_backend/qwen3_asr.py

============================================================
概要
============================================================
Qwen3-ASRは動的アテンションウィンドウにより、同一モデルで
ストリーミング/オフライン統一推論を実現する。

ストリーミング推論のキーアイデア:
- 音声を2秒チャンクに分割
- 累積的にチャンクを蓄積して逐次認識
- ロールバック戦略で末尾の不安定な認識を修正
- vLLM Serve + prefix caching で高効率推論

============================================================
Shape Convention
============================================================
B:            バッチサイズ (ストリーミングでは通常1)
chunk_size:   チャンクサイズ (サンプル数, デフォルト 2秒 = 32000)
T_accum:      累積音声サンプル数 (チャンクごとに増加)
T_mel_accum:  累積メルフレーム数
T_audio_accum: 累積Audio Encoder出力フレーム数
"""

import torch
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


# ============================================================
# 1. ストリーミング状態管理
# ============================================================

@dataclass
class ASRStreamingState:
    """
    ストリーミング推論の状態

    ========================================
    フィールド
    ========================================
    - audio_buffer: 未処理の音声バッファ
    - audio_accum: 累積音声 (チャンク蓄積)
    - chunk_id: 現在のチャンク番号
    - language: 識別された言語
    - text: 認識テキスト (逐次更新)
    - prev_tokens: 前回の認識トークン列 (ロールバック用)
    """

    # 音声バッファ
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # チャンク管理
    chunk_id: int = 0
    chunk_size_samples: int = 32000  # 2秒 @ 16kHz

    # 認識結果
    language: str = ""
    text: str = ""
    prev_tokens: List[int] = field(default_factory=list)

    # ロールバック設定
    unfixed_chunk_num: int = 2   # 固定されない末尾チャンク数
    fallback_tokens: int = 5     # ロールバック時に巻き戻すトークン数

    # コンテキスト
    context: str = ""            # エンティティリスト等


# ============================================================
# 2. ストリーミング推論フロー
# ============================================================

class Qwen3ASRStreaming:
    """
    ストリーミングASR推論

    ========================================
    全体フロー
    ========================================
    1. init_streaming_state: 状態初期化
    2. streaming_transcribe: チャンクごとの逐次推論
    3. finish_streaming_transcribe: 最終フラッシュ

    ========================================
    vLLMベース推論
    ========================================
    vLLMのprefix cachingを活用:
    - Audio特徴のKV cacheを再利用
    - 新しいチャンクの追加分のみ計算
    - 大幅な推論速度向上
    """

    def __init__(self, model_path: str):
        """
        モデルロード (vLLMバックエンド)

        内部処理:
            1. vLLM LLMエンジン初期化
            2. Processor ロード
            3. SamplingParams設定
        """
        # vLLM推論エンジン
        self.llm = None   # vllm.LLM(model_path)
        self.processor = None  # Qwen3ASRProcessor

        # サンプリング設定
        self.sampling_params = {
            "max_tokens": 512,
            "temperature": 0.0,   # Greedy decoding
            "top_p": 1.0,
        }

    def init_streaming_state(
        self,
        language: Optional[str] = None,
        context: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        chunk_size_sec: float = 2.0,
    ) -> ASRStreamingState:
        """
        ストリーミング状態の初期化

        ========================================
        パラメータ
        ========================================
        language:          言語指定 (Noneで自動識別)
        context:           コンテキスト (エンティティリスト)
        unfixed_chunk_num: 固定されない末尾チャンク数 (デフォルト2)
                          - 直近2チャンク分の認識結果は「暫定」扱い
                          - 新チャンク到着時にロールバック+再認識
        chunk_size_sec:    チャンクサイズ (秒, デフォルト2.0)
        """
        state = ASRStreamingState(
            chunk_size_samples=int(chunk_size_sec * 16000),  # 2.0s × 16kHz = 32000
            unfixed_chunk_num=unfixed_chunk_num,
            context=context or "",
        )

        if language:
            state.language = language

        return state

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,       # (num_new_samples,) float32 @ 16kHz
        state: ASRStreamingState,
    ) -> ASRStreamingState:
        """
        チャンクごとのストリーミング推論

        ========================================
        Shape
        ========================================
        入力:
            pcm16k: (num_new_samples,) float32 @ 16kHz
                - リアルタイム音声ストリームからの新しいサンプル

        状態更新:
            audio_buffer: (buffer_len,) - 未処理バッファ
            audio_accum:  (accum_len,) - 累積音声 (チャンクごとに増加)
            text:         str - 認識テキスト (逐次更新)

        ========================================
        処理フロー詳細
        ========================================
        """

        # ========================================
        # 1. 音声バッファリング
        # ========================================
        # 新しいサンプルをバッファに追加
        state.audio_buffer = np.concatenate([state.audio_buffer, pcm16k])
        # audio_buffer: (buffer_len + num_new_samples,)

        # バッファがチャンクサイズに達するまで待機
        while len(state.audio_buffer) >= state.chunk_size_samples:
            # ========================================
            # 2. チャンク取り出し
            # ========================================
            chunk = state.audio_buffer[:state.chunk_size_samples]
            state.audio_buffer = state.audio_buffer[state.chunk_size_samples:]
            # chunk: (32000,) = 2秒分

            # ========================================
            # 3. 累積音声に追加
            # ========================================
            # ★ キーポイント: 累積的に音声を蓄積
            # パディングなし、チャンクごとに全体を再認識
            state.audio_accum = np.concatenate([state.audio_accum, chunk])
            # audio_accum: (accum_len + 32000,)
            # 例: chunk_id=0 → 32000, chunk_id=1 → 64000, ...

            # ========================================
            # 4. プレフィックス構築 (ロールバック戦略)
            # ========================================
            prefix = self._build_prefix(state)
            # prefix: str - 確定済み認識テキスト

            # ========================================
            # 5. プロンプト構築
            # ========================================
            # System prompt + 累積音声 + prefix
            messages = [
                {"role": "system", "content": state.context},
                {"role": "user", "content": [
                    {"type": "audio", "audio": state.audio_accum},
                ]},
            ]

            prompt = self.processor.apply_chat_template(
                [messages],
                add_generation_prompt=True,
                tokenize=False,
            )[0]

            # prefix がある場合、生成プロンプトの末尾に追加
            if prefix:
                prompt = prompt + prefix

            # ========================================
            # 6. vLLM推論
            # ========================================
            # vLLMのprefix cachingにより:
            # - 前回のAudio特徴のKV cacheを再利用
            # - 新しいチャンク分のみ追加計算
            output = self.llm.generate(
                prompt,
                sampling_params=self.sampling_params,
            )
            generated_text = output[0].outputs[0].text
            # generated_text: "language English<asr_text>Hello, this is..."

            # ========================================
            # 7. 認識結果更新
            # ========================================
            language, text = self._parse_output(generated_text)

            state.language = language
            state.text = text
            state.chunk_id += 1

            # prev_tokensを更新 (次回ロールバック用)
            state.prev_tokens = self.processor.tokenizer.encode(
                generated_text, add_special_tokens=False
            )

        return state

    def finish_streaming_transcribe(
        self,
        state: ASRStreamingState,
    ) -> ASRStreamingState:
        """
        ストリーミングの最終フラッシュ

        ========================================
        処理
        ========================================
        1. 残りのバッファをフラッシュ
        2. 最終認識を実行 (ロールバックなし)
        3. 最終結果を返却

        ※ 最後のチャンクではunfixed_chunk_num=0として扱い、
           全認識結果を確定させる
        """
        # 残りバッファがある場合
        if len(state.audio_buffer) > 0:
            # 短すぎる場合はゼロパディング (最小0.5秒)
            min_samples = int(0.5 * 16000)  # 8000 samples
            if len(state.audio_buffer) < min_samples:
                padding = np.zeros(min_samples - len(state.audio_buffer), dtype=np.float32)
                state.audio_buffer = np.concatenate([state.audio_buffer, padding])

            state.audio_accum = np.concatenate([state.audio_accum, state.audio_buffer])
            state.audio_buffer = np.array([], dtype=np.float32)

        # 最終推論 (ロールバックなし)
        messages = [
            {"role": "system", "content": state.context},
            {"role": "user", "content": [
                {"type": "audio", "audio": state.audio_accum},
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=False,
        )[0]

        output = self.llm.generate(prompt, sampling_params=self.sampling_params)
        generated_text = output[0].outputs[0].text

        language, text = self._parse_output(generated_text)
        state.language = language
        state.text = text

        return state

    def _build_prefix(self, state: ASRStreamingState) -> str:
        """
        ロールバック戦略によるプレフィックス構築

        ========================================
        ロールバック戦略
        ========================================
        ストリーミングASRでは、音声の末尾付近の認識結果は
        不安定 (後続の音声がないため不確実)。

        unfixed_chunk_num で指定した数の末尾チャンクの
        認識結果を「暫定」扱いにし、新チャンク到着時に
        ロールバック (巻き戻し) して再認識する。

        例 (unfixed_chunk_num=2):
        ─────────────────────────────────────
        chunk_id=0: ""        (prefix なし)
        chunk_id=1: ""        (prefix なし, < unfixed_chunk_num)
        chunk_id=2: "Hello"   (chunk_id=0 の結果が確定)
        chunk_id=3: "Hello, " (chunk_id=1 の結果が確定)
        chunk_id=4: "Hello, this " (chunk_id=2 の結果が確定)
        ─────────────────────────────────────

        具体的な処理:
        1. chunk_id < unfixed_chunk_num → prefix = ""
        2. chunk_id >= unfixed_chunk_num →
           前回の認識結果から末尾 fallback_tokens トークンを除去
           → 残りを prefix として使用
        """
        if state.chunk_id < state.unfixed_chunk_num:
            return ""

        if not state.prev_tokens:
            return ""

        # 末尾 fallback_tokens を除去
        fixed_tokens = state.prev_tokens[:-state.fallback_tokens]
        if not fixed_tokens:
            return ""

        # トークン列をテキストにデコード
        prefix = self.processor.tokenizer.decode(
            fixed_tokens, skip_special_tokens=False
        )

        return prefix

    def _parse_output(self, text: str):
        """
        出力テキストのパース

        "language English<asr_text>Hello world<|im_end|>"
        → ("English", "Hello world")
        """
        text = text.replace("<|im_end|>", "").strip()
        if text.startswith("language "):
            text = text[len("language "):]
            if "<asr_text>" in text:
                language, transcript = text.split("<asr_text>", 1)
                return language.strip(), transcript.strip()
        return "Unknown", text


# ============================================================
# 3. ストリーミング推論の使用例
# ============================================================

def streaming_example():
    """
    ストリーミング推論の使用例 (疑似コード)

    ========================================
    想定シナリオ
    ========================================
    マイクからの音声を2秒チャンクで受信し、
    リアルタイムで認識結果を表示する。
    """

    # 1. モデル初期化
    asr = Qwen3ASRStreaming("Qwen/Qwen3-ASR-1.7B")

    # 2. ストリーミング状態初期化
    state = asr.init_streaming_state(
        language="English",
        unfixed_chunk_num=2,   # 直近2チャンクはロールバック対象
        chunk_size_sec=2.0,     # 2秒チャンク
    )

    # 3. 音声チャンクごとの推論
    # (実際にはマイクからの音声ストリーム)
    audio_chunks = simulate_audio_stream()  # 2秒チャンクのジェネレータ

    for chunk in audio_chunks:
        # chunk: (32000,) float32 @ 16kHz = 2秒

        state = asr.streaming_transcribe(chunk, state)
        print(f"[Chunk {state.chunk_id}] {state.language}: {state.text}")

        # 出力例:
        # [Chunk 1] English: Hello
        # [Chunk 2] English: Hello, this is
        # [Chunk 3] English: Hello, this is Qwen speaking
        # [Chunk 4] English: Hello, this is Qwen speaking to you
        # ...

    # 4. 最終フラッシュ
    state = asr.finish_streaming_transcribe(state)
    print(f"[Final] {state.language}: {state.text}")


def simulate_audio_stream():
    """音声ストリームのシミュレーション (テスト用)"""
    import librosa
    audio, _ = librosa.load("test.wav", sr=16000, mono=True)
    chunk_size = 32000  # 2秒
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]


# ============================================================
# 入出力shape一覧表
# ============================================================
"""
========================================
ストリーミング推論のShape遷移 (10秒音声, 2秒チャンク)
========================================

5チャンク: chunk_0 (0-2s), chunk_1 (2-4s), chunk_2 (4-6s), chunk_3 (6-8s), chunk_4 (8-10s)

| チャンク | audio_accum       | mel frames | AuT frames | prefix       | 出力テキスト                    |
|---------|-------------------|-----------|------------|--------------|-------------------------------|
| 0       | (32000,)   = 2s   | 200       | 25         | ""           | "Hello"                       |
| 1       | (64000,)   = 4s   | 400       | 50         | ""           | "Hello, this is"              |
| 2       | (96000,)   = 6s   | 600       | 75         | "Hello"      | "Hello, this is Qwen"         |
| 3       | (128000,)  = 8s   | 800       | 100        | "Hello, this"| "Hello, this is Qwen speaking" |
| 4       | (160000,)  = 10s  | 1000      | 125        | "Hello, this is"| "Hello, this is Qwen speaking to you" |
| final   | (160000,)  = 10s  | 1000      | 125        | (なし)        | "Hello, this is Qwen speaking to you today" |

★ 注意: 各チャンクで累積音声全体を再認識するが、
   vLLMのprefix cachingにより前回のKV cacheを再利用して効率化

========================================
ストリーミング vs オフライン (Table 8 from paper)
========================================

| モデル          | モード      | LibriSpeech c|o  | Fleurs-en | Fleurs-zh | 平均  |
|----------------|-----------|------------------|-----------|-----------|-------|
| Qwen3-ASR-1.7B | Offline   | 1.63 | 3.38      | 3.35      | 2.41      | 2.69  |
| Qwen3-ASR-1.7B | Streaming | 1.95 | 4.51      | 4.02      | 2.84      | 3.33  |
| Qwen3-ASR-0.6B | Offline   | 2.11 | 4.55      | 4.39      | 2.88      | 3.48  |
| Qwen3-ASR-0.6B | Streaming | 2.54 | 6.27      | 5.38      | 3.40      | 4.40  |

→ ストリーミングはオフラインに比べて約20-30%の認識率低下
→ 同一モデルで両方対応可能なことが利点

========================================
動的アテンションウィンドウの効果
========================================
AuT Encoder内のSelf-Attentionが動的ウィンドウを使用:
- ストリーミング: 1秒ウィンドウ (≈13フレーム @ 12.5Hz)
  → 各フレームは前後1秒のコンテキストのみ参照
  → 低レイテンシ

- オフライン: 8秒ウィンドウ (≈100フレーム @ 12.5Hz)
  → 各フレームは前後8秒の広いコンテキストを参照
  → 高精度

この動的ウィンドウにより、AuT Encoderの重みを変更せずに
ストリーミング/オフラインを切り替え可能。
"""
