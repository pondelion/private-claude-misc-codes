"""
VibeVoice ASR Model - 音声認識モデル

公式実装:
  - vibevoice/modular/modeling_vibevoice_asr.py
  - vibevoice/processor/vibevoice_asr_processor.py

ASR モデルは TTS モデルと類似するが、Diffusion Head なし。
音声→テキスト方向の生成に特化。

構成:
  - Acoustic Tokenizer (Encoder only): 音声 → [B, T, 64]
  - Semantic Tokenizer (Encoder only): 音声 → [B, T, 128]
  - Acoustic Connector: 64 → hidden_size
  - Semantic Connector: 128 → hidden_size
  - Qwen2.5 LLM: 統合特徴からテキスト自己回帰生成
  - LM Head: hidden_size → vocab_size

特徴:
  - Diffusion Head 不要（テキスト生成のみ）
  - Dual Tokenizer による高精度認識（WER: 1.11%）
  - 60秒超の長音声は自動的にストリーミングエンコード
  - LoRA ファインチューニング対応

参照: modeling_vibevoice_asr.py, vibevoice_asr_processor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


# ============================================================================
# ASR プロセッサ
# ============================================================================

class VibeVoiceASRProcessor:
    """
    ASR タスク用の音声+テキスト前処理プロセッサ。

    音声ファイルをロード・前処理し、LLM への入力形式を構築する。

    設定値:
      - speech_tok_compress_ratio: 320（TTS の 3200 と異なる）
        ※ ASR では VAE の圧縮比ではなく、別のトークン長推定に使用
      - target_sample_rate: 24000 Hz
      - normalize_audio: True (dB正規化)

    特殊トークン:
      - <|speech_start|>: 音声開始マーカー
      - <|speech_end|>: 音声終了マーカー
      - <|speech_pad|>: 音声パディング

    参照: vibevoice_asr_processor.py の VibeVoiceASRProcessor
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant that transcribes audio input "
        "into text output in JSON format."
    )

    def __init__(self, tokenizer, audio_processor, compress_ratio=320):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = compress_ratio
        self.target_sample_rate = 24000

        # 特殊トークンID
        self.speech_start_id = tokenizer.convert_tokens_to_ids("<|speech_start|>")
        self.speech_end_id = tokenizer.convert_tokens_to_ids("<|speech_end|>")
        self.speech_pad_id = tokenizer.convert_tokens_to_ids("<|speech_pad|>")

    def __call__(
        self,
        audio,                    # str (ファイルパス) or np.ndarray or torch.Tensor
        sampling_rate: int = None,
        context_info: str = None,  # ホットワード等のコンテキスト
        use_streaming: bool = None,
    ) -> dict:
        """
        音声入力をモデル入力形式に変換。

        Args:
            audio: 音声データ（パス、配列、テンソル）
            sampling_rate: サンプリングレート（リサンプリング用）
            context_info: 追加コンテキスト情報
            use_streaming: 60秒超で自動True

        Returns:
            dict with:
                input_ids: [B, S] テキストテンプレート + 音声パッドトークン
                attention_mask: [B, S]
                acoustic_input_mask: [B, S] 音声特徴を挿入する位置
                speech_tensors: [B, T_samples] 音声波形

        入力トークン構造:
        ```
        [System] You are a helpful assistant...
        [User] <speech_start> <speech_pad>×N <speech_end>
               This is X.XX seconds audio, please transcribe...
        [Assistant]
        ```
        """
        return self._process_single_audio(audio, sampling_rate, context_info)

    def _process_single_audio(self, audio, sampling_rate, context_info):
        """
        単一音声の処理パイプライン。

        処理フロー:
        1. 音声ファイルのロード（FFmpeg or soundfile）
        2. target_sample_rate (24kHz) にリサンプリング
        3. float32 正規化 + dB正規化（-25 dB FS）
        4. VAEトークン長の計算: ceil(samples / compress_ratio)
        5. チャットテンプレートトークンの構築
        6. acoustic_input_mask の生成

        Args:
            audio: 音声データ
            sampling_rate: 元のサンプリングレート
            context_info: コンテキスト

        Returns:
            dict with input_ids, acoustic_input_mask, speech, vae_tok_len
        """
        # --- Step 1-3: 音声ロード・前処理 ---
        if isinstance(audio, str):
            waveform, sr = load_audio(audio)  # FFmpeg ベース
        else:
            waveform = audio
            sr = sampling_rate

        # リサンプリング
        if sr != self.target_sample_rate:
            waveform = resample(waveform, sr, self.target_sample_rate)

        # float32 正規化
        waveform = waveform.float()
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        # dB 正規化（ターゲット: -25 dB FS）
        waveform = db_normalize(waveform, target_db=-25)

        # --- Step 4: トークン長計算 ---
        num_samples = waveform.shape[-1]
        duration = num_samples / self.target_sample_rate
        vae_tok_len = -(-num_samples // self.speech_tok_compress_ratio)  # ceil division
        # 例: 5秒音声 → 120000 / 320 = 375 トークン

        # --- Step 5: チャットテンプレート構築 ---
        # System prompt
        system_tokens = self.tokenizer.encode(
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        )

        # User message: 音声パッドトークン + テキスト指示
        speech_tokens = (
            [self.speech_start_id]
            + [self.speech_pad_id] * vae_tok_len
            + [self.speech_end_id]
        )

        context_str = ""
        if context_info:
            context_str = f"\nContext: {context_info}"

        user_text = (
            f"This is {duration:.2f} seconds audio, "
            f"please transcribe it into text.{context_str}"
        )
        user_text_tokens = self.tokenizer.encode(user_text)

        user_tokens = self.tokenizer.encode("<|im_start|>user\n")
        user_tokens += speech_tokens + user_text_tokens
        user_tokens += self.tokenizer.encode("<|im_end|>\n")

        # Assistant start
        assistant_tokens = self.tokenizer.encode("<|im_start|>assistant\n")

        # 全トークンを結合
        input_ids = system_tokens + user_tokens + assistant_tokens

        # --- Step 6: acoustic_input_mask ---
        acoustic_input_mask = [False] * len(input_ids)
        # speech_pad_id の位置を True に
        for i, token_id in enumerate(input_ids):
            if token_id == self.speech_pad_id:
                acoustic_input_mask[i] = True

        return {
            'input_ids': torch.tensor(input_ids),          # [S]
            'acoustic_input_mask': torch.tensor(acoustic_input_mask),  # [S]
            'speech': waveform,                             # [T_samples]
            'vae_tok_len': vae_tok_len,                    # int
        }

    def post_process_transcription(self, generated_text: str) -> List[Dict]:
        """
        モデルの出力テキスト（JSON）を構造化データに変換。

        モデルは以下の JSON 形式で出力:
        ```json
        [
          {"Start time": "0.00", "End time": "5.32",
           "Speaker ID": "0", "Content": "Hello world"},
          ...
        ]
        ```

        Args:
            generated_text: モデルの生成テキスト（JSON文字列）

        Returns:
            List[Dict]: 各発話の情報
                - start_time: float
                - end_time: float
                - speaker_id: str
                - text: str
        """
        import json

        # Markdown コードブロックの除去
        text = generated_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [{"text": generated_text}]

        results = []
        for entry in data:
            results.append({
                'start_time': float(entry.get('Start time', 0)),
                'end_time': float(entry.get('End time', 0)),
                'speaker_id': entry.get('Speaker ID', '0'),
                'text': entry.get('Content', ''),
            })
        return results


# ============================================================================
# ASR モデル本体
# ============================================================================

class VibeVoiceASRModel(nn.Module):
    """
    ASR コアモデル（Diffusion Head なし）。

    構成:
      - language_model: Qwen2.5 LLM
      - acoustic_tokenizer: σ-VAE Encoder
      - semantic_tokenizer: Semantic Encoder
      - acoustic_connector: SpeechConnector (64 → hidden_size)
      - semantic_connector: SpeechConnector (128 → hidden_size)

    TTS モデルとの違い:
      - Diffusion Head なし
      - noise_scheduler なし
      - 音声からテキストへの一方向変換のみ

    参照: modeling_vibevoice_asr.py の VibeVoiceASRModel
    """

    def __init__(self, config):
        super().__init__()
        self.language_model = Qwen2Model(config.decoder_config)
        self.acoustic_tokenizer = AcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = SemanticTokenizerModel(config.semantic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim,   # 64
            config.decoder_config.hidden_size,
        )
        self.semantic_connector = SpeechConnector(
            config.semantic_vae_dim,   # 128
            config.decoder_config.hidden_size,
        )


class VibeVoiceASRForConditionalGeneration(nn.Module):
    """
    ASR（音声認識）の完全なモデル。
    GenerationMixin を継承し、beam search / sampling 等の生成が可能。

    処理フロー:
      1. 音声をエンコード（Acoustic + Semantic）
      2. コネクタで LLM 次元に射影して合算
      3. テキストテンプレートの acoustic_input_mask 位置に挿入
      4. LLM で自己回帰テキスト生成
      5. JSON 形式の転写結果を出力

    参照: modeling_vibevoice_asr.py の VibeVoiceASRForConditionalGeneration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VibeVoiceASRModel(config)
        self.lm_head = nn.Linear(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            bias=False,
        )

    # ====================================================================
    # 音声エンコード
    # ====================================================================

    @torch.no_grad()
    def encode_speech(
        self,
        speech_tensors: torch.Tensor,          # [B, T_samples] or [T_samples]
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.Tensor] = None,
        streaming_segment_duration: float = 60.0,
    ) -> torch.Tensor:
        """
        音声波形を Dual Tokenizer でエンコードし、LLM 特徴に変換。

        短い音声（< 60秒）は一括処理、長い音声はストリーミングエンコード。

        Args:
            speech_tensors: [B, T_samples] 24kHz モノラル波形
            speech_masks: [B, T_latent] 有効トークンマスク（オプション）
            speech_semantic_tensors: [B, T_latent, 128] 事前計算済みセマンティック
            streaming_segment_duration: ストリーミングセグメント長（秒）

        Returns:
            combined_features: [B, T_latent, hidden_size] or [N, hidden_size]
                Acoustic + Semantic の合算特徴

        データフロー:
        ```
        speech_tensors [B, T_samples]
            │
            ├── Acoustic Tokenizer Encode
            │   → encoder_output.mean [B, T_latent, 64]
            │   → sample(gaussian) → [B, T_latent, 64]
            │   → acoustic_connector → [B, T_latent, hidden_size]
            │
            ├── Semantic Tokenizer Encode
            │   → semantic_output.mean [B, T_latent, 128]
            │   → semantic_connector → [B, T_latent, hidden_size]
            │
            └── 合算 → [B, T_latent, hidden_size]
        ```
        """
        acoustic_tokenizer = self.model.acoustic_tokenizer
        semantic_tokenizer = self.model.semantic_tokenizer
        acoustic_connector = self.model.acoustic_connector
        semantic_connector = self.model.semantic_connector

        # 次元調整
        if speech_tensors.dim() == 1:
            speech_tensors = speech_tensors.unsqueeze(0)  # [T] → [1, T]

        sample_rate = 24000
        segment_samples = int(streaming_segment_duration * sample_rate)
        total_samples = speech_tensors.shape[-1]
        use_streaming = total_samples > segment_samples

        if not use_streaming:
            # ========================================
            # 短い音声: 一括エンコード
            # ========================================

            # Acoustic エンコード
            acoustic_output = acoustic_tokenizer.encode(speech_tensors)
            # mean: [B, T_latent, 64], std: 0.5
            acoustic_tokens = acoustic_output.sample(dist_type='gaussian')[0]
            # [B, T_latent, 64]

            acoustic_features = acoustic_connector(acoustic_tokens)
            # [B, T_latent, hidden_size]

            # Semantic エンコード
            if speech_semantic_tensors is None:
                semantic_output = semantic_tokenizer.encode(speech_tensors)
                semantic_tokens = semantic_output.sample(dist_type='none')[0]
                # [B, T_latent, 128]
            else:
                semantic_tokens = speech_semantic_tensors
                # [B, T_latent, 128]

            semantic_features = semantic_connector(semantic_tokens)
            # [B, T_latent, hidden_size]

            # 合算
            combined = acoustic_features + semantic_features
            # [B, T_latent, hidden_size]

        else:
            # ========================================
            # 長い音声: ストリーミングエンコード
            # ========================================

            from speech_tokenizer import VibeVoiceTokenizerStreamingCache

            acoustic_cache = VibeVoiceTokenizerStreamingCache()
            semantic_cache = VibeVoiceTokenizerStreamingCache()

            acoustic_means = []
            semantic_means = []

            num_segments = -(-total_samples // segment_samples)  # ceil

            for seg_idx in range(num_segments):
                start = seg_idx * segment_samples
                end = min(start + segment_samples, total_samples)
                segment = speech_tensors[:, start:end]
                is_final = (seg_idx == num_segments - 1)

                # Acoustic エンコード（キャッシュ付き）
                acoustic_out = acoustic_tokenizer.encode(
                    segment,
                    cache=acoustic_cache,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                acoustic_means.append(acoustic_out.mean)

                # Semantic エンコード（キャッシュ付き）
                semantic_out = semantic_tokenizer.encode(
                    segment,
                    cache=semantic_cache,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                semantic_means.append(semantic_out.mean)

            # 全セグメントの mean を結合
            full_acoustic_mean = torch.cat(acoustic_means, dim=1)
            # [B, T_latent_total, 64]
            full_semantic_mean = torch.cat(semantic_means, dim=1)
            # [B, T_latent_total, 128]

            # サンプリング（一括）
            acoustic_tokens = VibeVoiceTokenizerEncoderOutput(
                mean=full_acoustic_mean, std=0.5
            ).sample(dist_type='gaussian')[0]

            # Connector
            acoustic_features = acoustic_connector(acoustic_tokens)
            semantic_features = semantic_connector(full_semantic_mean)
            combined = acoustic_features + semantic_features
            # [B, T_latent_total, hidden_size]

        # マスク適用（オプション）
        if speech_masks is not None:
            combined = combined[speech_masks]
            # [N_valid, hidden_size]

        return combined

    # ====================================================================
    # Forward パス
    # ====================================================================

    def forward(
        self,
        input_ids: torch.LongTensor,                     # [B, S]
        attention_mask: Optional[torch.Tensor] = None,     # [B, S]
        labels: Optional[torch.LongTensor] = None,        # [B, S]
        speech_tensors: Optional[torch.Tensor] = None,     # [B, T_samples]
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.Tensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        ASR モデルのフォワードパス。

        処理フロー:
        1. テキスト埋め込み: input_ids → [B, S, hidden_size]
        2. 音声エンコード: encode_speech() → [N_valid, hidden_size]
        3. acoustic_input_mask 位置に音声特徴を挿入
        4. LLM Forward → hidden_states [B, S, hidden_size]
        5. LM Head → logits [B, S, vocab_size]
        6. (学習時) Cross-Entropy Loss

        Args:
            input_ids: [B, S] テンプレート + 特殊トークン
            attention_mask: [B, S]
            labels: [B, S] 正解テキスト (-100 で無視)
            speech_tensors: [B, T_samples] 音声波形
            speech_masks: [B, T_latent]
            speech_semantic_tensors: [B, T_latent, 128]
            acoustic_input_mask: [B, S] 音声特徴挿入位置

        Returns:
            CausalLMOutputWithPast(loss, logits, past_key_values)
        """
        # === Step 1: テキスト埋め込み ===
        embed_tokens = self.model.language_model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        # [B, S, hidden_size]

        # === Step 2: 音声エンコード + 挿入 ===
        if speech_tensors is not None and acoustic_input_mask is not None:
            speech_features = self.encode_speech(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_semantic_tensors=speech_semantic_tensors,
            )
            # [N_valid, hidden_size] or [B, T_latent, hidden_size]

            # acoustic_input_mask 位置に挿入
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[acoustic_input_mask] = speech_features
            # テンプレート中の <speech_pad> 位置が音声特徴で置換される

        # === Step 3: LLM Forward ===
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # [B, S, hidden_size]

        # === Step 4: LM Head ===
        logits = self.lm_head(hidden_states)
        # [B, S, vocab_size]

        # === Step 5: Loss ===
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, 'past_key_values', None),
        )

    # ====================================================================
    # 生成ヘルパー
    # ====================================================================

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        speech_tensors=None,
        speech_masks=None,
        speech_semantic_tensors=None,
        acoustic_input_mask=None,
        **kwargs,
    ):
        """
        自己回帰生成ループの各ステップで入力を準備。

        Qwen2-VL パターン:
          - 最初のステップ (cache_position[0] == 0): 全音声入力を含める
          - 2ステップ目以降: 音声入力を None に（KVキャッシュに既に格納済み）

        Args:
            input_ids: [B, S] 現在のトークンID
            past_key_values: 前ステップのKVキャッシュ
            speech_tensors: 音声波形（初回のみ使用）
            ...

        Returns:
            model_inputs: dict for forward()
        """
        model_inputs = {}

        if past_key_values is not None:
            # 2ステップ目以降: 最新トークンのみ
            past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, past_length:]

            # 音声入力はキャッシュ済みなので不要
            model_inputs.update({
                'input_ids': input_ids,
                'past_key_values': past_key_values,
                'speech_tensors': None,
                'speech_masks': None,
                'speech_semantic_tensors': None,
                'acoustic_input_mask': None,
            })
        else:
            # 初回: 全入力を含める
            model_inputs.update({
                'input_ids': input_ids,
                'speech_tensors': speech_tensors,
                'speech_masks': speech_masks,
                'speech_semantic_tensors': speech_semantic_tensors,
                'acoustic_input_mask': acoustic_input_mask,
            })

        return model_inputs


# ============================================================================
# LoRA ファインチューニング
# ============================================================================

class VibeVoiceASRLoRATrainer:
    """
    LoRA を使った ASR モデルのファインチューニング。

    PEFT (Parameter-Efficient Fine-Tuning) により、
    LLM の重みのごく一部のみを学習して特定ドメインに適応。

    対象モジュール:
      - q_proj, k_proj, v_proj, o_proj (Attention)
      - gate_proj, up_proj, down_proj (FFN)

    参照: finetuning-asr/lora_finetune.py

    使用例（擬似コード）:
    ```python
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,                    # LoRA ランク
        lora_alpha=32,           # スケーリング係数
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(...)
    model = get_peft_model(model, lora_config)

    # HuggingFace Trainer で学習
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./lora_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            num_train_epochs=3,
            bf16=True,
            gradient_checkpointing=True,
        ),
        data_collator=VibeVoiceASRDataCollator(processor),
        train_dataset=dataset,
    )
    trainer.train()
    ```
    """

    def __init__(self, model, processor, lora_config):
        self.model = model
        self.processor = processor
        self.lora_config = lora_config


class VibeVoiceASRDataCollator:
    """
    ASR 学習用のデータコレーター。

    可変長の音声+テキストデータをバッチにまとめる。
    左パディング（自己回帰生成に適合）。

    各サンプルのデータ形式:
    ```json
    {
      "audio_path": "/path/to/audio.wav",
      "text": "transcription text"
    }
    ```

    参照: finetuning-asr/lora_finetune.py の VibeVoiceASRDataCollator
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        バッチ構築。

        Args:
            features: List of dicts (各サンプル)

        Returns:
            batch: dict with:
                input_ids: [B, S_max] (左パディング)
                attention_mask: [B, S_max]
                acoustic_input_mask: [B, S_max]
                speech_tensors: [B, T_max]
                labels: [B, S_max]
        """
        processed = []
        for feat in features:
            processed.append(
                self.processor(
                    audio=feat['audio_path'],
                    context_info=feat.get('context', None),
                )
            )

        # バッチパディング（左パディング）
        max_len = max(p['input_ids'].shape[0] for p in processed)
        max_audio = max(p['speech'].shape[-1] for p in processed)

        batch_input_ids = []
        batch_masks = []
        batch_acoustic_masks = []
        batch_speech = []

        for p in processed:
            s_len = p['input_ids'].shape[0]
            pad_len = max_len - s_len

            # 左パディング
            batch_input_ids.append(
                F.pad(p['input_ids'], (pad_len, 0), value=0)
            )
            batch_masks.append(
                F.pad(torch.ones(s_len), (pad_len, 0), value=0)
            )
            batch_acoustic_masks.append(
                F.pad(p['acoustic_input_mask'], (pad_len, 0), value=False)
            )

            # 音声パディング（右）
            a_len = p['speech'].shape[-1]
            batch_speech.append(
                F.pad(p['speech'], (0, max_audio - a_len))
            )

        return {
            'input_ids': torch.stack(batch_input_ids),         # [B, S_max]
            'attention_mask': torch.stack(batch_masks),        # [B, S_max]
            'acoustic_input_mask': torch.stack(batch_acoustic_masks),  # [B, S_max]
            'speech_tensors': torch.stack(batch_speech),       # [B, T_max]
        }


# ============================================================================
# ダミークラス・ユーティリティ
# ============================================================================

@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[tuple] = None

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    mean: torch.Tensor = None
    std: float = 0.5
    def sample(self, dist_type='gaussian'):
        if dist_type == 'none':
            return self.mean, self.std
        noise = torch.randn_like(self.mean)
        return self.mean + self.std * noise, self.std

class Qwen2Model(nn.Module):
    def __init__(self, config=None): super().__init__()
    def get_input_embeddings(self): pass
    def forward(self, **kwargs): pass

class AcousticTokenizerModel(nn.Module):
    def __init__(self, config): super().__init__()
    def encode(self, audio, cache=None, use_cache=False, is_final_chunk=False): pass

class SemanticTokenizerModel(nn.Module):
    def __init__(self, config): super().__init__()
    def encode(self, audio, cache=None, use_cache=False, is_final_chunk=False): pass

class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim): super().__init__()
    def forward(self, x): pass

def load_audio(path): pass
def resample(waveform, orig_sr, target_sr): pass
def db_normalize(waveform, target_db=-25): pass
