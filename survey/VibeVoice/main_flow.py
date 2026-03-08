"""
VibeVoice メインフロー - TTS (Text-to-Speech) 全体のデータフローを擬似コードで解説

公式実装: vibevoice/modular/modeling_vibevoice.py

VibeVoice は Qwen2.5 LLM + Dual Tokenizer + Diffusion Head を組み合わせた
長時間・複数話者対応の音声合成モデル。

全体構成:
  1. SpeechConnector: 音声特徴を LLM 次元に射影
  2. VibeVoiceModel: 全コンポーネントを統合するコンテナ
  3. VibeVoiceForConditionalGeneration: 学習・推論の本体

参照: modeling_vibevoice.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ============================================================================
# 出力データクラス
# ============================================================================

@dataclass
class VibeVoiceCausalLMOutputWithPast:
    """
    VibeVoice のフォワードパス出力

    Fields:
        loss: Optional[torch.FloatTensor]
            テキスト CE Loss + 拡散 MSE Loss の合計（学習時のみ）
        diffusion_loss: Optional[torch.FloatTensor]
            拡散ヘッドの MSE 損失単体
        speech_token_num: Optional[int]
            バッチ内の音声トークン総数（ロギング用）
        logits: torch.FloatTensor
            [B, T_total, vocab_size] テキストトークン予測分布
        past_key_values: Optional[Tuple]
            KV キャッシュ（推論時の自己回帰生成用）
        hidden_states: Optional[Tuple]
            全層の隠れ状態（出力オプション）
        attentions: Optional[Tuple]
            全層のアテンション重み（出力オプション）
    """
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    speech_token_num: Optional[int] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None


@dataclass
class VibeVoiceGenerationOutput:
    """
    生成結果の出力

    Fields:
        sequences: torch.LongTensor
            [B, T_generated] 生成されたトークンID列
        speech_outputs: List[torch.Tensor]
            バッチ内の各サンプルの音声波形リスト [T_audio]
    """
    sequences: torch.LongTensor = None
    speech_outputs: List[torch.Tensor] = field(default_factory=list)


# ============================================================================
# SpeechConnector: 音声特徴 → LLM 次元への射影
# ============================================================================

class SpeechConnector(nn.Module):
    """
    音声トークナイザの潜在変数を LLM の隠れ次元に射影するコネクタ。
    2層の全結合 + RMSNorm で構成。

    Acoustic Connector: 64 → hidden_size (1536 or 3584)
    Semantic Connector: 128 → hidden_size (1536 or 3584)

    参照: modeling_vibevoice.py の SpeechConnector クラス
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: 入力次元 (vae_dim: 64 or 128)
            output_dim: 出力次元 (hidden_size: 1536 or 3584)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=1e-6)  # LlamaRMSNorm 相当
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T_latent, input_dim] 音声潜在変数

        Returns:
            output: [B, T_latent, output_dim] LLM次元に射影された特徴

        データフロー:
            [B, T, 64] → fc1 → [B, T, hidden_size] → RMSNorm → fc2 → [B, T, hidden_size]
        """
        x = self.fc1(features)       # [B, T, input_dim] → [B, T, output_dim]
        x = self.norm(x)             # [B, T, output_dim] RMSNorm 正規化
        x = self.fc2(x)             # [B, T, output_dim] → [B, T, output_dim]
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LlamaRMSNorm 相当)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, dim]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight  # [*, dim]


# ============================================================================
# VibeVoiceModel: 全コンポーネントの統合コンテナ
# ============================================================================

class VibeVoiceModel(nn.Module):
    """
    VibeVoice のモデル本体。以下のコンポーネントを統合:
    - language_model: Qwen2.5 LLM (テキスト+音声の統合処理)
    - acoustic_tokenizer: σ-VAE (音声→潜在変数)
    - semantic_tokenizer: Semantic Encoder (音声→意味特徴)
    - acoustic_connector: Acoustic 潜在変数 → LLM 次元
    - semantic_connector: Semantic 潜在変数 → LLM 次元
    - prediction_head: Diffusion Head (LLM隠れ状態→音声潜在変数)
    - noise_scheduler: DPM-Solver++ (拡散スケジューラ)

    参照: modeling_vibevoice.py の VibeVoiceModel クラス
    """

    def __init__(self, config):
        super().__init__()
        # === LLM ===
        # Qwen2.5 (1.5B: hidden=1536, heads=12, layers=28)
        # Qwen2.5 (7B:   hidden=3584, heads=28, layers=28)
        self.language_model = Qwen2Model(config.decoder_config)

        # === 音声トークナイザ（学習中は凍結） ===
        self.acoustic_tokenizer = AcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = SemanticTokenizerModel(config.semantic_tokenizer_config)

        # === コネクタ ===
        self.acoustic_connector = SpeechConnector(
            input_dim=config.acoustic_vae_dim,   # 64
            output_dim=config.decoder_config.hidden_size  # 1536 or 3584
        )
        self.semantic_connector = SpeechConnector(
            input_dim=config.semantic_vae_dim,   # 128
            output_dim=config.decoder_config.hidden_size
        )

        # === 拡散ヘッド ===
        self.prediction_head = DiffusionHead(config.diffusion_head_config)

        # === スケジューラ ===
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,      # 1000
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,         # "cosine"
            prediction_type=config.diffusion_head_config.prediction_type,          # "v_prediction"
        )

        # === 正規化バッファ（FSDP対応） ===
        # 学習開始時の最初のバッチで動的に計算される
        self.register_buffer("speech_scaling_factor", torch.tensor(float('nan')))
        self.register_buffer("speech_bias_factor", torch.tensor(float('nan')))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        LLM の forward のみ（推論用簡易パス）

        Args:
            input_ids: [B, T_text] テキストトークンID
            attention_mask: [B, T_text]

        Returns:
            BaseModelOutputWithPast (last_hidden_state: [B, T, hidden_size])
        """
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )


# ============================================================================
# VibeVoiceForConditionalGeneration: 学習・推論の本体
# ============================================================================

class VibeVoiceForConditionalGeneration(nn.Module):
    """
    VibeVoice の条件付き生成モデル。
    テキスト + 音声プロンプトから音声を自己回帰的に生成する。

    学習時: テキスト CE Loss + 拡散 MSE Loss を同時に最適化
    推論時: LLM でトークン列を生成し、各音声位置で Diffusion Head が潜在変数を予測

    参照: modeling_vibevoice.py の VibeVoiceForConditionalGeneration クラス
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VibeVoiceModel(config)

        # テキストトークン予測ヘッド
        hidden_size = config.decoder_config.hidden_size   # 1536 or 3584
        vocab_size = config.decoder_config.vocab_size     # 151936 or 152064
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    # ====================================================================
    # 音声特徴の前処理
    # ====================================================================

    def forward_speech_features(
        self,
        speech_tensors: torch.Tensor,       # [B, T_audio] 生の音声波形
        speech_masks: torch.BoolTensor,      # [B, T_latent] 有効トークンマスク
        speech_type: str = "audio",          # "audio" or "vae"
        return_unmask: bool = False,         # True: マスク前の全特徴も返す
    ):
        """
        音声波形をエンコードし、正規化・射影した特徴量を返す。

        処理フロー:
        1. Acoustic Tokenizer でエンコード
        2. σ-VAE サンプリング
        3. 正規化（mean/std の動的計算 + 適用）
        4. Acoustic Connector で LLM 次元に射影

        Args:
            speech_tensors: [B, T_audio] 24kHz モノラル波形
            speech_masks:   [B, T_latent] 有効音声トークン位置
            speech_type:    "audio"（生波形）or "vae"（事前計算済み潜在変数）
            return_unmask:  True なら全特徴と有効特徴の両方を返す

        Returns:
            speech_features: [N_valid, hidden_size] or [B, T_latent, hidden_size]
            speech_connect_features: [N_valid, hidden_size] (return_unmask時のみ)
        """
        acoustic_tokenizer = self.model.acoustic_tokenizer
        acoustic_connector = self.model.acoustic_connector

        # --- 音声が無い場合 ---
        if speech_tensors is None:
            return None  # 音声なし

        # --- Step 1: エンコード ---
        if speech_type == "audio":
            # 生波形からエンコード
            encoder_output = acoustic_tokenizer.encode(speech_tensors)
            # encoder_output.mean: [B, T_latent, 64]
            # encoder_output.std: fixed (0.5)

            # σ-VAE サンプリング: z = μ + σ ⊙ ε
            audio_tokens = encoder_output.sample(
                dist_type="gaussian"  # std_dist_type
            )[0]  # [B, T_latent, 64]

        elif speech_type == "vae":
            # 事前計算済み潜在変数（学習データの効率化用）
            vae_dim = self.config.acoustic_vae_dim  # 64
            audio_tokens = speech_tensors.reshape(
                speech_tensors.shape[0], -1, vae_dim
            )  # [B, T_latent, 64]

            # 学習時のノイズ追加（fix_std=0.5 × 0.8 = 0.4）
            fix_std = self.config.acoustic_tokenizer_config.fix_std  # 0.5
            noise = torch.randn_like(audio_tokens) * fix_std * 0.8
            audio_tokens = audio_tokens + noise  # [B, T_latent, 64]

        # --- Step 2: 正規化パラメータの動的計算（初回のみ） ---
        scaling_factor = self.model.speech_scaling_factor
        bias_factor = self.model.speech_bias_factor

        if torch.isnan(scaling_factor):
            # 有効トークンの統計量を計算
            valid_tokens = audio_tokens[speech_masks]  # [N_valid, 64]
            mean_val = valid_tokens.mean()
            std_val = valid_tokens.std()

            # DDP 環境での同期
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_val)
                mean_val /= torch.distributed.get_world_size()
                torch.distributed.all_reduce(std_val)
                std_val /= torch.distributed.get_world_size()

            bias_factor = -mean_val
            scaling_factor = 1.0 / std_val

            # バッファに保存（以降再計算不要）
            self.model.speech_bias_factor.fill_(bias_factor.item())
            self.model.speech_scaling_factor.fill_(scaling_factor.item())

        # --- Step 3: 正規化適用 ---
        audio_features = (audio_tokens + bias_factor) * scaling_factor
        # [B, T_latent, 64] → 零平均・単位分散に正規化

        # --- Step 4: LLM 次元に射影 ---
        connect_features = acoustic_connector(audio_features)
        # [B, T_latent, 64] → [B, T_latent, hidden_size]

        if return_unmask:
            # 学習時: マスク前の全特徴と、マスク後の有効特徴を返す
            speech_all_features = audio_features  # [B, T_latent, 64]
            speech_connect_features = connect_features[speech_masks]  # [N_valid, hidden_size]
            return speech_all_features, speech_connect_features
        else:
            # 推論時: マスクされた有効特徴のみ
            return connect_features[speech_masks]  # [N_valid, hidden_size]

    # ====================================================================
    # メインの Forward パス
    # ====================================================================

    def forward(
        self,
        input_ids: torch.LongTensor,                   # [B, T_total] 全トークンID
        attention_mask: Optional[torch.Tensor] = None,  # [B, T_total]
        labels: Optional[torch.LongTensor] = None,      # [B, T_total] テキスト正解
        speech_tensors: Optional[torch.Tensor] = None,   # [B, T_audio] 入力音声波形
        speech_masks: Optional[torch.BoolTensor] = None,  # [B, T_latent] 音声マスク
        speech_semantic_tensors: Optional[torch.Tensor] = None,  # [B, T_latent, 128]
        acoustic_input_mask: Optional[torch.BoolTensor] = None,   # [B, T_total] 音声位置マスク
        speeches_loss_input: Optional[torch.BoolTensor] = None,   # [B, T_latent] 拡散損失対象
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,    # [B, T_total] 損失計算位置
        **kwargs,
    ) -> VibeVoiceCausalLMOutputWithPast:
        """
        VibeVoice のメインフォワードパス。

        処理フロー:
        1. テキストトークンの埋め込み
        2. Semantic 特徴の射影
        3. Acoustic 特徴の処理（エンコード + 正規化 + 射影）
        4. 音声特徴をテキスト埋め込みに挿入
        5. LLM に通して隠れ状態を取得
        6. LM Head でテキスト logits を計算
        7. テキスト CE Loss を計算
        8. Diffusion Loss を計算（音声トークン位置のみ）

        Args:
            input_ids:             [B, T_total] テキスト + 特殊トークンのID列
            attention_mask:        [B, T_total] パディングマスク
            labels:                [B, T_total] テキスト正解（-100で無視）
            speech_tensors:        [B, T_audio] 入力音声の生波形（24kHz）
            speech_masks:          [B, T_latent] 有効な音声トークン位置
            speech_semantic_tensors: [B, T_latent, 128] Semantic Tokenizer の出力
            acoustic_input_mask:   [B, T_total] 音声特徴を挿入する位置
            speeches_loss_input:   [B, T_latent] 拡散損失を計算する音声位置
            acoustic_loss_mask:    [B, T_total] LLM隠れ状態から条件を抽出する位置

        Returns:
            VibeVoiceCausalLMOutputWithPast
        """
        # === Step 1: テキスト埋め込み ===
        embed_tokens = self.model.language_model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        # [B, T_total] → [B, T_total, hidden_size]

        # === Step 2: Semantic 特徴の射影 ===
        if speech_semantic_tensors is not None:
            semantic_features = self.model.semantic_connector(speech_semantic_tensors)
            # [B, T_latent, 128] → [B, T_latent, hidden_size]
        else:
            semantic_features = None

        # === Step 3 & 4: Acoustic 特徴の処理と挿入 ===
        diffusion_loss = None
        speech_token_num = 0

        if speeches_loss_input is not None:
            # --- 学習モード: 拡散損失あり ---

            # Acoustic 特徴をエンコード・正規化・射影（マスク前の全特徴も返す）
            speech_all_features, speech_connect_features = self.forward_speech_features(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_type="audio",
                return_unmask=True,
            )
            # speech_all_features: [B, T_latent, 64] (正規化済み、全トークン)
            # speech_connect_features: [N_valid, hidden_size] (射影済み、有効トークンのみ)

            # テキスト埋め込みの音声位置に acoustic + semantic 特徴を挿入
            inputs_embeds = inputs_embeds.clone()  # in-place 回避
            if semantic_features is not None:
                # acoustic_input_mask で指定された位置に合算特徴を挿入
                combined = speech_connect_features
                if semantic_features is not None:
                    semantic_connect = self.model.semantic_connector(
                        speech_semantic_tensors
                    )
                    # semantic_connect[speech_masks] と合算
                    combined = combined + semantic_connect[speech_masks]
                inputs_embeds[acoustic_input_mask] = combined
                # [B, T_total, hidden_size] の一部が音声特徴で置換

            # 拡散損失ターゲットの抽出
            target_latent_mask = speeches_loss_input & speech_masks  # [B, T_latent]
            target_features = speech_all_features[target_latent_mask]
            # [N_target, 64] 拡散損失の正解潜在変数

        else:
            # --- 推論モード: 拡散損失なし ---
            if speech_tensors is not None and acoustic_input_mask is not None:
                speech_features = self.forward_speech_features(
                    speech_tensors=speech_tensors,
                    speech_masks=speech_masks,
                    speech_type="audio",
                    return_unmask=False,
                )
                inputs_embeds = inputs_embeds.clone()
                if speech_features is not None:
                    inputs_embeds[acoustic_input_mask] = speech_features

        # === Step 5: LLM Forward ===
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # [B, T_total, hidden_size]

        # === Step 6: LM Head ===
        logits = self.lm_head(hidden_states)
        # [B, T_total, vocab_size]

        # === Step 7: テキスト CE Loss ===
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # === Step 8: Diffusion Loss（音声トークン位置のみ） ===
        if speeches_loss_input is not None and target_features.numel() > 0:
            diffusion_loss = self._compute_diffusion_loss(
                hidden_states=hidden_states,
                acoustic_loss_mask=acoustic_loss_mask,
                target_features=target_features,
            )

            # 合計損失
            if loss is not None:
                loss = loss + diffusion_loss
            else:
                loss = diffusion_loss

            speech_token_num = target_features.shape[0]

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            speech_token_num=speech_token_num,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        )

    # ====================================================================
    # 拡散損失の計算
    # ====================================================================

    def _compute_diffusion_loss(
        self,
        hidden_states: torch.Tensor,         # [B, T_total, hidden_size]
        acoustic_loss_mask: torch.BoolTensor, # [B, T_total]
        target_features: torch.Tensor,        # [N_speech, 64]
    ) -> torch.Tensor:
        """
        Diffusion Head の学習損失を計算。

        各音声トークン位置で:
        1. LLM隠れ状態を条件ベクトルとして抽出
        2. ランダムなタイムステップとノイズを生成
        3. ノイズを正解潜在変数に追加
        4. Diffusion Head が速度(v)を予測
        5. 正解速度との MSE を計算

        ddpm_batch_mul（デフォルト4）倍のバッチで計算し、学習効率を向上。

        Args:
            hidden_states:     [B, T_total, hidden_size] LLMの全隠れ状態
            acoustic_loss_mask: [B, T_total] 条件を抽出する位置
            target_features:    [N_speech, 64] 正解音声潜在変数

        Returns:
            loss: スカラー MSE 損失
        """
        prediction_head = self.model.prediction_head
        noise_scheduler = self.model.noise_scheduler
        config = self.config.diffusion_head_config

        ddpm_batch_mul = config.ddpm_batch_mul  # 4
        ddpm_num_steps = config.ddpm_num_steps  # 1000
        latent_size = config.latent_size         # 64
        prediction_type = config.prediction_type  # "v_prediction"

        # --- 条件ベクトル抽出 ---
        condition = hidden_states[acoustic_loss_mask]
        # [N_speech, hidden_size]

        # --- ddpm_batch_mul 倍に拡張 ---
        N_speech = target_features.shape[0]
        target_expanded = target_features.repeat(ddpm_batch_mul, 1)
        condition_expanded = condition.repeat(ddpm_batch_mul, 1)
        # [N_speech * 4, 64] と [N_speech * 4, hidden_size]

        # --- ランダムタイムステップ ---
        timesteps = torch.randint(
            0, ddpm_num_steps,
            (N_speech * ddpm_batch_mul,),
            device=target_features.device,
        )
        # [N_speech * 4]

        # --- ランダムノイズ ---
        noise = torch.randn_like(target_expanded)
        # [N_speech * 4, 64]

        # --- ノイズ追加: x_t = α_t × x_0 + σ_t × ε ---
        noisy_latents = noise_scheduler.add_noise(
            target_expanded, noise, timesteps
        )
        # [N_speech * 4, 64]

        # --- Diffusion Head で予測 ---
        # unsqueeze(1) で [N*4, 1, 64] と [N*4, 1, hidden_size] にして
        # トークンレベルの予測を行う
        pred = prediction_head(
            noisy_images=noisy_latents.unsqueeze(1),     # [N*4, 1, 64]
            timesteps=timesteps,                          # [N*4]
            condition=condition_expanded.unsqueeze(1),    # [N*4, 1, hidden_size]
        ).squeeze(1)
        # [N*4, 64]

        # --- ターゲット計算 ---
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                target_expanded, noise, timesteps
            )
            # v = α_t × ε - σ_t × x_0
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        # [N*4, 64]

        # --- MSE 損失 ---
        loss = F.mse_loss(pred, target)
        loss = loss / (latent_size * ddpm_batch_mul)
        # 正規化: / (64 * 4)

        return loss

    # ====================================================================
    # 推論時の生成（概要）
    # ====================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,          # [B, T_input]
        speech_tensors: torch.Tensor = None,   # [B, T_audio] 音声プロンプト
        speech_masks: torch.BoolTensor = None,
        acoustic_input_mask: torch.BoolTensor = None,
        max_new_tokens: int = 4096,
        cfg_scale: float = 1.3,
        num_inference_steps: int = 10,
        **kwargs,
    ):
        """
        TTS 推論の全体フロー（概要）。

        実際の推論は VibeVoiceForConditionalGeneration.generate() で
        HuggingFace の GenerationMixin を使用するが、
        ここでは音声生成部分のデータフローを擬似コードで示す。

        Args:
            input_ids: [B, T_input] 入力テキスト + 音声プロンプトのトークンID
            speech_tensors: [B, T_audio] 各話者の音声プロンプト波形
            max_new_tokens: 生成する最大トークン数
            cfg_scale: CFG ガイダンススケール (default: 1.3)
            num_inference_steps: 拡散推論ステップ数 (default: 10)

        Returns:
            VibeVoiceGenerationOutput
        """
        prediction_head = self.model.prediction_head
        noise_scheduler = self.model.noise_scheduler
        acoustic_tokenizer = self.model.acoustic_tokenizer
        acoustic_connector = self.model.acoustic_connector

        # --- Prefill: 音声プロンプトとテキストを処理 ---
        # 音声プロンプトを Acoustic Tokenizer でエンコード → Connector で射影
        # テキストトークンと統合して LLM に入力
        # → KV キャッシュ生成
        outputs = self.forward(
            input_ids=input_ids,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            acoustic_input_mask=acoustic_input_mask,
        )
        past_key_values = outputs.past_key_values

        # --- Auto-regressive 生成ループ ---
        generated_tokens = []
        speech_latents = []

        for step in range(max_new_tokens):
            # LLM が次のトークンを予測
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            next_token = logits.argmax(dim=-1)  # [B]

            # 音声トークン位置の場合
            if is_speech_token(next_token):
                # LLM の隠れ状態を条件として取得
                h_i = outputs.hidden_states[:, -1, :]  # [B, hidden_size]

                # --- CFG 付き拡散生成 ---
                noise_scheduler.set_timesteps(num_inference_steps)
                z = torch.randn(1, 1, 64, device=h_i.device)  # [B, 1, 64]

                for t in noise_scheduler.timesteps:
                    # 条件付き予測
                    z_input = torch.cat([z, z], dim=0)  # [2B, 1, 64]
                    cond = torch.cat([
                        h_i.unsqueeze(1),                # [B, 1, hidden_size] 条件付き
                        torch.zeros_like(h_i).unsqueeze(1),  # [B, 1, hidden_size] 無条件
                    ], dim=0)
                    # [2B, 1, hidden_size]

                    t_batch = t.expand(2 * h_i.shape[0])
                    v_pred = prediction_head(z_input, t_batch, cond)
                    # [2B, 1, 64]

                    v_cond, v_uncond = v_pred.chunk(2, dim=0)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    # [B, 1, 64] CFG ガイダンス適用

                    z = noise_scheduler.step(v.squeeze(1), t, z.squeeze(1))
                    z = z.prev_sample.unsqueeze(1)
                    # [B, 1, 64]

                speech_latents.append(z.squeeze(1))  # [B, 64]

                # 潜在変数を Connector で射影して次の LLM 入力に
                next_embed = acoustic_connector(z)  # [B, 1, hidden_size]

            generated_tokens.append(next_token)

        # --- 潜在変数を波形にデコード ---
        all_latents = torch.stack(speech_latents, dim=1)  # [B, T_speech, 64]
        audio = acoustic_tokenizer.decode(all_latents)     # [B, 1, T_audio]

        return VibeVoiceGenerationOutput(
            sequences=torch.stack(generated_tokens, dim=1),
            speech_outputs=[audio[i] for i in range(audio.shape[0])],
        )


# ============================================================================
# ダミークラス（実際のモデルの代替）
# ============================================================================

class Qwen2Model(nn.Module):
    """Qwen2.5 LLM のプレースホルダ。実際は transformers.Qwen2Model を使用。"""
    def __init__(self, config):
        super().__init__()
        # 実際: 28層の Transformer Decoder
        # hidden_size: 1536 (1.5B) or 3584 (7B)
        # num_attention_heads: 12 (1.5B) or 28 (7B)
        # vocab_size: 151936 (1.5B) or 152064 (7B)
        pass
    def get_input_embeddings(self):
        pass
    def forward(self, **kwargs):
        pass

class AcousticTokenizerModel(nn.Module):
    """Acoustic Tokenizer のプレースホルダ。詳細は speech_tokenizer.py を参照。"""
    def __init__(self, config): super().__init__()
    def encode(self, audio): pass
    def decode(self, latents): pass

class SemanticTokenizerModel(nn.Module):
    """Semantic Tokenizer のプレースホルダ。詳細は speech_tokenizer.py を参照。"""
    def __init__(self, config): super().__init__()
    def encode(self, audio): pass

class DiffusionHead(nn.Module):
    """Diffusion Head のプレースホルダ。詳細は diffusion_head.py を参照。"""
    def __init__(self, config): super().__init__()
    def forward(self, noisy_images, timesteps, condition): pass

class DPMSolverMultistepScheduler:
    """DPM-Solver++ のプレースホルダ。詳細は loss_and_training.py を参照。"""
    def __init__(self, **kwargs): pass
    def add_noise(self, original, noise, timesteps): pass
    def get_velocity(self, original, noise, timesteps): pass
    def set_timesteps(self, num_steps): pass
    def step(self, model_output, timestep, sample): pass


def is_speech_token(token_id):
    """音声トークンかどうかを判定するユーティリティ（実際はトークナイザの特殊トークンで判定）"""
    pass
