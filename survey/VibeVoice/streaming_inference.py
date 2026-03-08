"""
VibeVoice Streaming Inference - リアルタイムストリーミング音声合成

公式実装:
  - vibevoice/modular/modeling_vibevoice_streaming.py (学習用基底)
  - vibevoice/modular/modeling_vibevoice_streaming_inference.py (推論)
  - vibevoice/modular/streamer.py (音声ストリーミング)

ストリーミングTTSの核心アイデア:
  1. Qwen2 の Transformer 層を2つに分割
     - Lower Layers (8層): テキストエンコード専用
     - Upper Layers (20層): テキスト+音声の生成
  2. テキストを5トークンのウィンドウに分割し、
     ウィンドウごとに6つの音声潜在変数を生成
  3. 生成された潜在変数を即座にデコードして音声を出力

参照:
  - modeling_vibevoice_streaming.py
  - modeling_vibevoice_streaming_inference.py
  - streamer.py
"""

import torch
import torch.nn as nn
import asyncio
from queue import Queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterator


# ============================================================================
# 定数
# ============================================================================

TTS_TEXT_WINDOW_SIZE = 5    # テキスト5トークンずつ処理
TTS_SPEECH_WINDOW_SIZE = 6  # テキスト5に対し音声6トークン生成


# ============================================================================
# 出力データクラス
# ============================================================================

@dataclass
class VibeVoiceGenerationOutput:
    """
    ストリーミング生成の出力。

    Fields:
        sequences: [B, T_generated] 生成されたトークンID列
        speech_outputs: バッチ内の各サンプルの音声波形リスト
        reach_max_step_sample: 最大ステップに到達したサンプルのフラグ
    """
    sequences: torch.LongTensor = None
    speech_outputs: List[torch.Tensor] = field(default_factory=list)
    reach_max_step_sample: List[bool] = field(default_factory=list)


# ============================================================================
# AudioStreamer: リアルタイム音声出力
# ============================================================================

class AudioStreamer:
    """
    バッチ対応の同期型音声ストリーミングシステム。

    generate() が音声チャンクを生成するたびに put() でキューに追加し、
    クライアント側は get_stream() でイテレータを取得して順次再生する。

    バッチサイズ > 1 の場合、各サンプルに独立したキューを持つ。

    参照: streamer.py の AudioStreamer
    """

    def __init__(self, batch_size: int = 1, timeout: float = 30.0):
        """
        Args:
            batch_size: 同時生成するサンプル数
            timeout: キュー操作のタイムアウト（秒）
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.stop_signal = "STOP"

        # 各サンプル用のキュー
        self.audio_queues: List[Queue] = [Queue() for _ in range(batch_size)]
        self.finished_flags: List[bool] = [False] * batch_size

    def put(self, audio_chunk: torch.Tensor, sample_indices: List[int]):
        """
        音声チャンクをキューに追加。

        Args:
            audio_chunk: [T_audio] or [B_sub, T_audio] 音声波形チャンク
            sample_indices: チャンクに対応するサンプルインデックス
        """
        for i, idx in enumerate(sample_indices):
            if not self.finished_flags[idx]:
                chunk = audio_chunk[i] if audio_chunk.dim() > 1 else audio_chunk
                self.audio_queues[idx].put(chunk)

    def end(self, sample_indices: List[int]):
        """
        指定サンプルの生成完了を通知。

        Args:
            sample_indices: 完了したサンプルのインデックス
        """
        for idx in sample_indices:
            self.finished_flags[idx] = True
            self.audio_queues[idx].put(self.stop_signal)

    def get_stream(self, sample_idx: int = 0) -> 'AudioSampleIterator':
        """
        特定サンプルの音声ストリームイテレータを取得。

        Args:
            sample_idx: サンプルインデックス

        Returns:
            AudioSampleIterator: 音声チャンクを順次返すイテレータ
        """
        return AudioSampleIterator(self.audio_queues[sample_idx], self.stop_signal)


class AudioSampleIterator:
    """
    単一サンプルの音声ストリームイテレータ。
    キューから音声チャンクをブロッキング取得し、
    stop_signal を受信すると StopIteration を発生。

    使用例:
        for audio_chunk in streamer.get_stream(0):
            play_audio(audio_chunk)

    参照: streamer.py の AudioSampleIterator
    """

    def __init__(self, queue: Queue, stop_signal: str):
        self.queue = queue
        self.stop_signal = stop_signal

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        item = self.queue.get()  # ブロッキング
        if item == self.stop_signal:
            raise StopIteration
        return item  # [T_audio] 音声チャンク


class AudioBatchIterator:
    """
    バッチ全体の音声ストリームイテレータ（ノンブロッキング）。
    準備ができたサンプルのチャンクを dict で返す。

    使用例:
        for batch_chunks in AudioBatchIterator(streamer):
            for idx, chunk in batch_chunks.items():
                process_audio(idx, chunk)

    参照: streamer.py の AudioBatchIterator
    """

    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        self.active = set(range(streamer.batch_size))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[int, torch.Tensor]:
        if not self.active:
            raise StopIteration

        chunks = {}
        for idx in list(self.active):
            try:
                item = self.streamer.audio_queues[idx].get(block=False)
                if item == self.streamer.stop_signal:
                    self.active.discard(idx)
                else:
                    chunks[idx] = item
            except Exception:
                pass  # キューが空

        if not chunks and self.active:
            import time
            time.sleep(0.01)  # CPU 使用率を抑える
            return self.__next__()

        return chunks


# ============================================================================
# AsyncAudioStreamer: 非同期版
# ============================================================================

class AsyncAudioStreamer:
    """
    asyncio ベースの非同期音声ストリーマー。
    FastAPI/WebSocket サーバーでのリアルタイム TTS に使用。

    call_soon_threadsafe() でスレッドセーフにキューを操作。

    参照: streamer.py の AsyncAudioStreamer
    """

    def __init__(self, batch_size: int = 1, loop: asyncio.AbstractEventLoop = None):
        self.batch_size = batch_size
        self.loop = loop or asyncio.get_event_loop()
        self.stop_signal = "STOP"

        self.audio_queues: List[asyncio.Queue] = [
            asyncio.Queue() for _ in range(batch_size)
        ]
        self.finished_flags: List[bool] = [False] * batch_size

    def put(self, audio_chunk: torch.Tensor, sample_indices: List[int]):
        """スレッドセーフに音声チャンクを追加"""
        for i, idx in enumerate(sample_indices):
            if not self.finished_flags[idx]:
                chunk = audio_chunk[i] if audio_chunk.dim() > 1 else audio_chunk
                self.loop.call_soon_threadsafe(
                    self.audio_queues[idx].put_nowait, chunk
                )

    def end(self, sample_indices: List[int]):
        """生成完了通知"""
        for idx in sample_indices:
            self.finished_flags[idx] = True
            self.loop.call_soon_threadsafe(
                self.audio_queues[idx].put_nowait, self.stop_signal
            )


# ============================================================================
# BinaryClassifier: EOS 検出
# ============================================================================

class BinaryClassifier(nn.Module):
    """
    音声生成の終了（EOS）を検出する二値分類器。

    LLM の隠れ状態から「生成を終了すべきか」を判定。
    logit > 0.5 で EOS と判定。

    構造: Linear(hidden_size → hidden_size) → ReLU → Linear(hidden_size → 1)

    参照: modeling_vibevoice_streaming.py の BinaryClassifier
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_size] LLM の隠れ状態

        Returns:
            logit: [B, 1] EOS スコア（> 0.5 で終了）
        """
        return self.fc2(torch.relu(self.fc1(x)))


# ============================================================================
# VibeVoiceStreamingModel: ストリーミングTTS基底モデル
# ============================================================================

class VibeVoiceStreamingModel(nn.Module):
    """
    ストリーミングTTSのモデル構造。
    Qwen2 の Transformer 層を2つに分割して管理。

    構成:
      language_model (Lower Layers):
        Qwen2 の下位 8 層（28 - tts_backbone_num_hidden_layers）
        テキストのみを処理（音声入力なし）

      tts_language_model (Upper Layers):
        Qwen2 の上位 20 層（tts_backbone_num_hidden_layers）
        テキスト隠れ状態 + 音声潜在変数を処理

      tts_input_types: Embedding(2, hidden_size)
        テキスト(1) と 音声(0) を区別する型埋め込み

    参照: modeling_vibevoice_streaming.py の VibeVoiceStreamingModel
    """

    def __init__(self, config):
        super().__init__()
        # === Lower Transformer Layers (テキストエンコード) ===
        # 全28層のうち下位 8 層
        # (28 - tts_backbone_num_hidden_layers = 28 - 20 = 8)
        self.language_model = Qwen2Model(
            num_layers=config.decoder_config.num_hidden_layers
                       - config.tts_backbone_num_hidden_layers
        )
        # hidden_size: 1536 (1.5B) or 3584 (7B)

        # === Upper Transformer Layers (TTS生成) ===
        # 上位 20 層
        self.tts_language_model = Qwen2Model(
            num_layers=config.tts_backbone_num_hidden_layers  # 20
        )

        # === 型埋め込み ===
        # テキストトークン → 1, 音声トークン → 0
        self.tts_input_types = nn.Embedding(2, config.decoder_config.hidden_size)

        # === 音声コンポーネント ===
        self.acoustic_tokenizer = AcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, config.decoder_config.hidden_size
        )
        self.prediction_head = DiffusionHead(config.diffusion_head_config)
        self.noise_scheduler = DPMSolverMultistepScheduler()

        # 正規化バッファ
        self.register_buffer("speech_scaling_factor", torch.tensor(float('nan')))
        self.register_buffer("speech_bias_factor", torch.tensor(float('nan')))


# ============================================================================
# VibeVoiceStreamingForConditionalGenerationInference: 推論本体
# ============================================================================

class VibeVoiceStreamingForConditionalGenerationInference(nn.Module):
    """
    ストリーミングTTSの推論モデル。
    テキストをウィンドウ単位で処理し、音声を逐次生成。

    主要メソッド:
      forward_lm():     下位層でテキストをエンコード
      forward_tts_lm(): 上位層でテキスト+音声を処理
      generate():       ストリーミング生成ループ
      sample_speech_tokens(): CFG付き拡散サンプリング

    参照: modeling_vibevoice_streaming_inference.py
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VibeVoiceStreamingModel(config)
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )
        self.eos_classifier = BinaryClassifier(config.decoder_config.hidden_size)

    # ====================================================================
    # Lower LM: テキストエンコード
    # ====================================================================

    def forward_lm(
        self,
        input_ids: torch.LongTensor,              # [B, S] テキストトークンID
        attention_mask: Optional[torch.Tensor],     # [B, S]
        past_key_values: Optional[tuple] = None,   # KVキャッシュ
    ):
        """
        下位 Transformer 層（8層）でテキストをエンコード。

        テキストのみを処理し、音声は扱わない。
        結果は上位層 (forward_tts_lm) への入力となる。

        Args:
            input_ids: [B, S] テキストトークンID
            attention_mask: [B, S]
            past_key_values: 前回のKVキャッシュ

        Returns:
            BaseModelOutputWithPast
                last_hidden_state: [B, S, hidden_size]
                past_key_values: 更新されたKVキャッシュ

        データフロー:
            input_ids [B, S]
            → embed_tokens → [B, S, hidden_size]
            → Transformer Layer 0~7 (8層)
            → [B, S, hidden_size]
        """
        # Lower LM forward（テキストのみ）
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        return outputs
        # last_hidden_state: [B, S, hidden_size]

    # ====================================================================
    # Upper TTS LM: テキスト+音声生成
    # ====================================================================

    def forward_tts_lm(
        self,
        input_ids: torch.LongTensor,                  # [B, S]
        attention_mask: Optional[torch.Tensor],         # [B, S_total]
        lm_last_hidden_state: Optional[torch.Tensor],  # [B, S, hidden_size]
        tts_text_masks: Optional[torch.BoolTensor],     # [B, S] True=テキスト
        past_key_values: Optional[tuple] = None,
    ):
        """
        上位 Transformer 層（20層）でテキスト+音声を処理。

        下位層の隠れ状態を入力として受け取り、
        型埋め込み（テキスト/音声を区別）を加算した上で処理。

        Args:
            input_ids: [B, S] 入力ID（音声位置はダミーID）
            attention_mask: [B, S_total] 全体のマスク
            lm_last_hidden_state: [B, S, hidden_size] 下位層の出力
            tts_text_masks: [B, S] True=テキスト位置, False=音声位置
            past_key_values: TTS層のKVキャッシュ

        Returns:
            VibeVoiceCausalLMOutputWithPast
                logits: [B, S, 1] EOS予測スコア

        データフロー:
            lm_last_hidden_state [B, S, hidden_size]
            + tts_input_types(tts_text_masks.long()) [B, S, hidden_size]
              → テキスト位置: type_embed[1]
              → 音声位置:   type_embed[0]
            → Upper Transformer Layer 0~19 (20層)
            → hidden_states [B, S, hidden_size]
            → eos_classifier → logits [B, S, 1]
        """
        # テキスト/音声の型埋め込みを追加
        type_embed = self.model.tts_input_types(tts_text_masks.long())
        # [B, S, hidden_size]

        inputs_embeds = lm_last_hidden_state + type_embed
        # [B, S, hidden_size]

        # Upper TTS LM forward
        outputs = self.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = outputs.last_hidden_state
        # [B, S, hidden_size]

        # EOS 予測
        logits = self.eos_classifier(hidden_states[:, -1, :])
        # [B, 1]

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=outputs.past_key_values,
        )

    # ====================================================================
    # CFG 付き拡散サンプリング
    # ====================================================================

    def sample_speech_tokens(
        self,
        condition: torch.Tensor,   # [B, 1, hidden_size] 条件付き
        cfg_scale: float = 1.3,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance (CFG) 付きの拡散サンプリング。

        DPM-Solver++ で num_inference_steps ステップのイテレーティブデノイジングを行う。

        Args:
            condition: [B, 1, hidden_size]
                上位TTS層の隠れ状態（音声トークン位置）
            cfg_scale: CFG ガイダンススケール (1.3)
            num_inference_steps: 拡散推論ステップ数 (20)

        Returns:
            speech_latent: [B, 1, 64] デノイズされた音声潜在変数

        アルゴリズム:
            1. 条件付き(positive) + 無条件(negative) の2つを結合
            2. N(0,1) からノイズを初期化
            3. 各タイムステップ t で:
               a. ノイズサンプルを複製して条件/無条件の両方に
               b. Diffusion Head で予測
               c. CFG: v = v_uncond + scale * (v_cond - v_uncond)
               d. Scheduler で1ステップ更新
            4. 条件付き側（前半）を返す
        """
        prediction_head = self.model.prediction_head
        noise_scheduler = self.model.noise_scheduler

        # 無条件入力（ゼロベクトル）
        uncond = torch.zeros_like(condition)  # [B, 1, hidden_size]

        # 条件付き + 無条件 を結合
        cond_combined = torch.cat([condition, uncond], dim=0)
        # [2B, 1, hidden_size]

        # ノイズ初期化
        B = condition.shape[0]
        latent_size = self.config.diffusion_head_config.latent_size  # 64
        z = torch.randn(B, 1, latent_size, device=condition.device)
        # [B, 1, 64]

        # スケジューラ設定
        noise_scheduler.set_timesteps(num_inference_steps)

        for t in noise_scheduler.timesteps:
            # ノイズサンプルを複製
            z_doubled = torch.cat([z, z], dim=0)  # [2B, 1, 64]
            t_batch = t.expand(2 * B)              # [2B]

            # Diffusion Head で予測
            v_pred = prediction_head(z_doubled, t_batch, cond_combined)
            # [2B, 1, 64]

            # 条件付き / 無条件 に分割
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            # 各 [B, 1, 64]

            # CFG ガイダンス
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            # [B, 1, 64]

            # スケジューラで1ステップ進める
            z = noise_scheduler.step(v.squeeze(1), t, z.squeeze(1)).prev_sample
            z = z.unsqueeze(1)
            # [B, 1, 64]

        return z  # [B, 1, 64]

    # ====================================================================
    # メイン生成ループ
    # ====================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,       # [B, S_prompt]
        tts_text_ids: torch.LongTensor,     # [B, T_text] 生成対象テキスト
        attention_mask: torch.Tensor,       # [B, S_prompt]
        tts_lm_input_ids: torch.LongTensor, # [B, S_tts_prompt]
        tts_lm_attention_mask: torch.Tensor,
        max_new_tokens: int = 4096,
        cfg_scale: float = 1.3,
        num_inference_steps: int = 20,
        audio_streamer: Optional[AudioStreamer] = None,
        **kwargs,
    ) -> VibeVoiceGenerationOutput:
        """
        ストリーミングTTS生成のメインループ。

        テキストをTTS_TEXT_WINDOW_SIZE（5）トークンのウィンドウに分割し、
        各ウィンドウに対してTTS_SPEECH_WINDOW_SIZE（6）の音声トークンを生成。

        Args:
            input_ids: [B, S_prompt] Prefill用のプロンプトID（音声プロンプト含む）
            tts_text_ids: [B, T_text] 生成対象のテキストトークンID
            attention_mask: [B, S_prompt]
            tts_lm_input_ids: [B, S_tts_prompt] TTS層用のプロンプトID
            tts_lm_attention_mask: [B, S_tts_prompt]
            max_new_tokens: 最大生成トークン数
            cfg_scale: CFG スケール (1.3)
            num_inference_steps: 拡散ステップ数 (20)
            audio_streamer: リアルタイム音声出力用ストリーマー

        Returns:
            VibeVoiceGenerationOutput

        全体フロー:
        ```
        Phase 1: Prefill
          input_ids → Lower LM → lm_hidden
          tts_lm_input_ids → Upper TTS LM → tts_hidden
          両方のKVキャッシュを生成

        Phase 2: ウィンドウベース生成ループ
          for window in text_windows (5トークンずつ):

            Step A: テキストウィンドウ処理
              text_tokens → Lower LM(cache) → lm_hidden_new
              lm_hidden_new → Upper TTS LM(cache) → EOS判定

            Step B: 音声トークン生成 (×6)
              for i in range(TTS_SPEECH_WINDOW_SIZE):
                tts_hidden → sample_speech_tokens(CFG) → z_i [B, 1, 64]
                z_i → Acoustic Decoder(streaming cache) → audio_chunk
                audio_chunk → AudioStreamer.put()
                z_i → Acoustic Connector → embedding
                embedding → Upper TTS LM(cache) → 次の条件ベクトル

            Step C: EOS チェック
              if eos_logit > 0.5: break

        Phase 3: 後処理
          AudioStreamer.end() で完了通知
          残りの音声をデコード
        ```
        """
        B = input_ids.shape[0]
        acoustic_tokenizer = self.model.acoustic_tokenizer
        acoustic_connector = self.model.acoustic_connector
        scaling_factor = self.model.speech_scaling_factor
        bias_factor = self.model.speech_bias_factor

        # ========================================
        # Phase 1: Prefill
        # ========================================

        # Lower LM: プロンプトをエンコード
        lm_outputs = self.forward_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        lm_past_kv = lm_outputs.past_key_values
        # KVキャッシュ: 8層分

        # Upper TTS LM: TTS プロンプトをエンコード
        tts_outputs = self.forward_tts_lm(
            input_ids=tts_lm_input_ids,
            attention_mask=tts_lm_attention_mask,
            lm_last_hidden_state=lm_outputs.last_hidden_state,
            tts_text_masks=torch.ones_like(tts_lm_input_ids, dtype=torch.bool),
        )
        tts_past_kv = tts_outputs.past_key_values
        # KVキャッシュ: 20層分

        # ========================================
        # Phase 2: ウィンドウベース生成ループ
        # ========================================

        all_speech_latents = []
        decoder_cache = None  # Acoustic Decoder のストリーミングキャッシュ
        generated_tokens = []
        reach_max_step = [False] * B

        # テキストをウィンドウに分割
        T_text = tts_text_ids.shape[1]
        num_windows = (T_text + TTS_TEXT_WINDOW_SIZE - 1) // TTS_TEXT_WINDOW_SIZE
        total_steps = 0

        for w in range(num_windows):
            start = w * TTS_TEXT_WINDOW_SIZE
            end = min(start + TTS_TEXT_WINDOW_SIZE, T_text)
            window_ids = tts_text_ids[:, start:end]  # [B, ≤5]
            window_len = window_ids.shape[1]

            # ========================================
            # Step A: テキストウィンドウ処理
            # ========================================

            # Lower LM: テキストウィンドウをインクリメンタルにエンコード
            lm_outputs = self.forward_lm(
                input_ids=window_ids,
                attention_mask=None,  # KVキャッシュ使用時は不要
                past_key_values=lm_past_kv,
            )
            lm_past_kv = lm_outputs.past_key_values
            lm_hidden = lm_outputs.last_hidden_state
            # [B, window_len, hidden_size]

            # Upper TTS LM: テキスト隠れ状態を処理
            tts_text_masks = torch.ones(B, window_len, device=window_ids.device, dtype=torch.bool)
            tts_outputs = self.forward_tts_lm(
                input_ids=window_ids,
                attention_mask=None,
                lm_last_hidden_state=lm_hidden,
                tts_text_masks=tts_text_masks,
                past_key_values=tts_past_kv,
            )
            tts_past_kv = tts_outputs.past_key_values
            tts_hidden = tts_outputs.hidden_states
            # [B, window_len, hidden_size]

            # ========================================
            # Step B: 音声トークン生成 (×6)
            # ========================================

            for speech_idx in range(TTS_SPEECH_WINDOW_SIZE):
                total_steps += 1
                if total_steps > max_new_tokens:
                    reach_max_step = [True] * B
                    break

                # --- 条件ベクトル: TTS LM の最後の隠れ状態 ---
                condition = tts_hidden[:, -1:, :]  # [B, 1, hidden_size]

                # --- CFG付き拡散サンプリング ---
                speech_latent = self.sample_speech_tokens(
                    condition=condition,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                )
                # [B, 1, 64]

                # --- 逆正規化 ---
                # 学習時の正規化の逆: z_raw = z / scaling + (-bias)
                speech_raw = speech_latent / scaling_factor - bias_factor
                # [B, 1, 64]

                all_speech_latents.append(speech_raw.squeeze(1))

                # --- Acoustic Decoder で波形にデコード（ストリーミング） ---
                audio_chunk = acoustic_tokenizer.decode(
                    speech_raw,
                    cache=decoder_cache,
                    use_cache=True,
                )
                # [B, 1, chunk_samples]
                # chunk_samples ≈ 3200 サンプル ≈ 0.133秒 (24kHz)

                # --- AudioStreamer に送信 ---
                if audio_streamer is not None:
                    audio_streamer.put(
                        audio_chunk.squeeze(1),
                        sample_indices=list(range(B)),
                    )

                # --- 次の TTS LM 入力用に射影 ---
                speech_embed = acoustic_connector(speech_latent)
                # [B, 1, hidden_size]

                # --- Upper TTS LM に音声埋め込みを入力 ---
                tts_speech_masks = torch.zeros(B, 1, device=window_ids.device, dtype=torch.bool)
                # False = 音声トークン
                tts_outputs = self.forward_tts_lm(
                    input_ids=torch.zeros(B, 1, device=window_ids.device, dtype=torch.long),
                    attention_mask=None,
                    lm_last_hidden_state=speech_embed,
                    tts_text_masks=tts_speech_masks,
                    past_key_values=tts_past_kv,
                )
                tts_past_kv = tts_outputs.past_key_values
                tts_hidden = tts_outputs.hidden_states
                # [B, 1, hidden_size]

            # ========================================
            # Step C: EOS チェック
            # ========================================

            eos_logit = tts_outputs.logits  # [B, 1]
            if (eos_logit > 0.5).all():
                break

            if any(reach_max_step):
                break

        # ========================================
        # Phase 3: 後処理
        # ========================================

        if audio_streamer is not None:
            audio_streamer.end(list(range(B)))

        # 全潜在変数から完全な音声を構築
        if all_speech_latents:
            all_latents = torch.stack(all_speech_latents, dim=1)
            # [B, N_speech, 64]
            # （ストリーマー使用時は既にチャンクで出力済みだが、
            #   完全な音声も返す）

        return VibeVoiceGenerationOutput(
            sequences=torch.tensor(generated_tokens) if generated_tokens else None,
            speech_outputs=[
                torch.cat([lat for lat in all_speech_latents], dim=0)
            ] if all_speech_latents else [],
            reach_max_step_sample=reach_max_step,
        )


# ============================================================================
# ダミークラス
# ============================================================================

class Qwen2Model(nn.Module):
    """Qwen2.5 Transformer のプレースホルダ"""
    def __init__(self, num_layers=None): super().__init__()
    def forward(self, **kwargs): pass

class AcousticTokenizerModel(nn.Module):
    """Acoustic Tokenizer のプレースホルダ。詳細は speech_tokenizer.py を参照。"""
    def __init__(self, config): super().__init__()
    def decode(self, latents, cache=None, use_cache=False): pass

class SpeechConnector(nn.Module):
    """Speech Connector のプレースホルダ。詳細は main_flow.py を参照。"""
    def __init__(self, input_dim, output_dim): super().__init__()
    def forward(self, x): pass

class DiffusionHead(nn.Module):
    """Diffusion Head のプレースホルダ。詳細は diffusion_head.py を参照。"""
    def __init__(self, config): super().__init__()
    def forward(self, noisy_images, timesteps, condition): pass

class DPMSolverMultistepScheduler:
    """DPM-Solver++ のプレースホルダ。"""
    def __init__(self, **kwargs): pass
    def set_timesteps(self, n): self.timesteps = []
    def step(self, model_output, t, sample): pass


@dataclass
class VibeVoiceCausalLMOutputWithPast:
    """推論出力"""
    logits: torch.Tensor = None
    hidden_states: torch.Tensor = None
    past_key_values: tuple = None
