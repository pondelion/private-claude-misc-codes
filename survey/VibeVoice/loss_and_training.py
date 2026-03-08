"""
VibeVoice 損失関数と学習の詳細

公式実装:
  - vibevoice/modular/modeling_vibevoice.py (拡散損失計算部分)
  - vibevoice/schedule/dpm_solver.py (DPM-Solver++ スケジューラ)
  - vibevoice/schedule/timestep_sampler.py (タイムステップサンプラー)
  - vibevoice/processor/vibevoice_processor.py (データ前処理)

VibeVoice の学習は2つの損失を同時に最適化:
  1. L_text: テキストトークンの Cross-Entropy Loss（LLM Head）
  2. L_diffusion: 音声トークン位置での拡散 MSE Loss（Diffusion Head）

学習パラメータ:
  - 凍結: Acoustic Tokenizer, Semantic Tokenizer
  - 学習: LLM (Qwen2.5), Diffusion Head, Connectors
  - Curriculum Learning: 4096 → 65536 トークン漸増

参照:
  - modeling_vibevoice.py
  - dpm_solver.py
  - timestep_sampler.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union


# ============================================================================
# タイムステップサンプラー
# ============================================================================

class UniformSampler:
    """
    一様分布タイムステップサンプラー。
    [0, num_timesteps) の範囲から一様にサンプリング。

    標準的な DDPM 学習で使用。

    参照: timestep_sampler.py の UniformSampler
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch_size: サンプル数
            device: デバイス

        Returns:
            timesteps: [batch_size] ∈ [0, num_timesteps)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class LogitNormalSampler:
    """
    Logit-Normal 分布タイムステップサンプラー。

    一様分布よりも特定のタイムステップ範囲を重点的にサンプリング。
    Imagen/Photorealistic 系モデルで効果的とされる。

    数式:
      logit(t) ~ N(m, s²)
      p(t) ∝ exp(-0.5 * (logit(t) - m)² / s²) / (t * (1-t))

    デフォルト: m=0, s=1 → 中間タイムステップ付近を重視

    参照: timestep_sampler.py の LogitNormalSampler
    """

    def __init__(self, num_timesteps: int = 1000, m: float = 0.0, s: float = 1.0):
        self.num_timesteps = num_timesteps
        self.m = m
        self.s = s

        # 各タイムステップの確率を事前計算
        t = torch.linspace(0.5 / num_timesteps, 1 - 0.5 / num_timesteps, num_timesteps)
        logit_t = torch.log(t / (1 - t))
        prob = torch.exp(-0.5 * ((logit_t - m) / s) ** 2)
        self.prob = prob / prob.sum()  # 正規化

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch_size: サンプル数
            device: デバイス

        Returns:
            timesteps: [batch_size] ∈ [0, num_timesteps)
        """
        indices = torch.multinomial(self.prob, batch_size, replacement=True)
        return indices.to(device)


# ============================================================================
# DPM-Solver++ スケジューラ
# ============================================================================

class DPMSolverMultistepScheduler:
    """
    DPM-Solver++ マルチステップスケジューラ。
    高次の ODE ソルバーによる効率的な拡散サンプリング。

    10~20ステップで高品質なサンプリングが可能（DDPM の 1000ステップ相当）。

    設定値:
      - num_train_timesteps: 1000（学習時）
      - beta_schedule: "cosine"
      - prediction_type: "v_prediction"
      - algorithm_type: "dpmsolver++"
      - solver_order: 2（推奨）
      - solver_type: "midpoint"

    主要メソッド:
      - set_timesteps(): 推論ステップの設定
      - add_noise(): 順方向拡散（ノイズ追加）
      - step(): 逆方向拡散（1ステップデノイズ）
      - get_velocity(): v-prediction ターゲット計算

    参照: dpm_solver.py の DPMSolverMultistepScheduler
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        algorithm_type: str = "dpmsolver++",
        solver_order: int = 2,
        solver_type: str = "midpoint",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        self.solver_order = solver_order
        self.solver_type = solver_type

        # === ベータスケジュールの計算 ===
        if beta_schedule == "cosine":
            # Cosine スケジュール (Improved DDPM)
            # β(t) = 1 - ᾱ(t) / ᾱ(t-1)
            # ᾱ(t) = cos²((t/T + s) / (1 + s) × π/2)
            s = 0.008
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((t / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        elif beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(0.0001**0.5, 0.02**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # ᾱ_t: [T] 累積積

        # λ(t) = log(α_t / σ_t) = log(√ᾱ_t / √(1-ᾱ_t))
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # 推論用の変数
        self.timesteps = None
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order

    def set_timesteps(self, num_inference_steps: int):
        """
        推論用のタイムステップスケジュールを設定。

        num_train_timesteps (1000) を num_inference_steps (10 or 20) に
        均等に分割してタイムステップ列を生成。

        Args:
            num_inference_steps: 推論ステップ数 (10 or 20)

        設定後:
            self.timesteps: [num_inference_steps] 降順のタイムステップ
            例 (20ステップ): [999, 949, 899, ..., 49, 0]
        """
        self.num_inference_steps = num_inference_steps

        # 均等間隔でタイムステップを選択
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
        timesteps = torch.from_numpy(timesteps).long()
        timesteps = torch.flip(timesteps, [0])  # 降順

        self.timesteps = timesteps
        self.model_outputs = [None] * self.solver_order
        self.step_index = 0

    def add_noise(
        self,
        original_samples: torch.Tensor,   # [B, D] or [B, T, D]
        noise: torch.Tensor,               # same shape
        timesteps: torch.Tensor,           # [B]
    ) -> torch.Tensor:
        """
        順方向拡散: 元データにノイズを追加。

        数式: x_t = √ᾱ_t × x_0 + √(1 - ᾱ_t) × ε

        Args:
            original_samples: [B, D] 元の潜在変数 (x_0)
            noise: [B, D] ガウスノイズ (ε ~ N(0,1))
            timesteps: [B] タイムステップ

        Returns:
            noisy_samples: [B, D] ノイズ付きサンプル (x_t)
        """
        # ᾱ_t の取得
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # 次元調整
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy

    def get_velocity(
        self,
        sample: torch.Tensor,     # [B, D] x_0
        noise: torch.Tensor,      # [B, D] ε
        timesteps: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        v-prediction のターゲットを計算。

        数式: v = √ᾱ_t × ε - √(1 - ᾱ_t) × x_0

        v-prediction は epsilon-prediction と sample-prediction の
        線形結合であり、学習が安定する（特にコサインスケジュールで有効）。

        Args:
            sample: [B, D] 元の潜在変数 (x_0)
            noise: [B, D] ノイズ (ε)
            timesteps: [B]

        Returns:
            velocity: [B, D] 速度ターゲット (v)
        """
        alphas_cumprod = self.alphas_cumprod.to(sample.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while sqrt_alpha_prod.dim() < sample.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def step(
        self,
        model_output: torch.Tensor,  # [B, D] 予測出力
        timestep: torch.Tensor,       # スカラー or [B]
        sample: torch.Tensor,         # [B, D] 現在のノイズサンプル
    ) -> 'SchedulerOutput':
        """
        逆方向拡散: 1ステップのデノイズ。

        DPM-Solver++ のアルゴリズム:
        1. model_output を x_0 予測に変換（convert_model_output）
        2. ソルバー次数に応じて更新:
           - 1次: DDIM 相当
           - 2次: 中点法 (midpoint)
           - 3次: 3点法

        Args:
            model_output: [B, D] Diffusion Head の予測
            timestep: 現在のタイムステップ
            sample: [B, D] 現在のノイズ付きサンプル

        Returns:
            SchedulerOutput(prev_sample=[B, D])
        """
        t = timestep
        s = self.timesteps[min(self.step_index + 1, len(self.timesteps) - 1)]

        # --- model_output → x_0 予測 ---
        x0_pred = self._convert_model_output(model_output, t, sample)

        # --- ソルバー次数の選択 ---
        if self.step_index == 0 or self.solver_order == 1:
            # 1次ソルバー（DDIM 相当）
            prev_sample = self._first_order_update(x0_pred, t, s, sample)
        elif self.step_index == 1 and self.solver_order >= 2:
            # 2次ソルバー（中点法）
            prev_sample = self._second_order_update(
                [self.model_outputs[-1], x0_pred], t, s, sample
            )
        else:
            # 3次ソルバー
            prev_sample = self._third_order_update(
                [self.model_outputs[-2], self.model_outputs[-1], x0_pred],
                t, s, sample,
            )

        # model_outputs キャッシュ更新
        self.model_outputs = self.model_outputs[1:] + [x0_pred]
        self.step_index += 1

        return SchedulerOutput(prev_sample=prev_sample)

    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        モデル出力を x_0 予測に変換。

        prediction_type に応じた変換:
          "epsilon":     x_0 = (x_t - σ_t × ε_pred) / α_t
          "v_prediction": x_0 = α_t × x_t - σ_t × v_pred
          "sample":      x_0 = model_output (そのまま)
        """
        alphas_cumprod = self.alphas_cumprod.to(sample.device)

        if isinstance(timestep, torch.Tensor) and timestep.dim() == 0:
            t_idx = timestep.long()
        else:
            t_idx = timestep

        alpha_t = alphas_cumprod[t_idx] ** 0.5
        sigma_t = (1 - alphas_cumprod[t_idx]) ** 0.5

        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        if self.prediction_type == "epsilon":
            x0 = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "v_prediction":
            x0 = alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == "sample":
            x0 = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return x0

    def _first_order_update(self, x0_pred, t, s, sample):
        """1次ソルバー（DDIM相当）"""
        alphas_cumprod = self.alphas_cumprod.to(sample.device)
        alpha_s = alphas_cumprod[s] ** 0.5
        sigma_s = (1 - alphas_cumprod[s]) ** 0.5

        while alpha_s.dim() < sample.dim():
            alpha_s = alpha_s.unsqueeze(-1)
            sigma_s = sigma_s.unsqueeze(-1)

        # x_{t-1} = α_{t-1} × x_0 + σ_{t-1} × ε_pred
        # ε_pred は x_0 予測から逆算
        alpha_t = self.alphas_cumprod[t] ** 0.5
        sigma_t = (1 - self.alphas_cumprod[t]) ** 0.5
        while alpha_t.dim() < sample.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        eps_pred = (sample - alpha_t * x0_pred) / sigma_t
        prev_sample = alpha_s * x0_pred + sigma_s * eps_pred
        return prev_sample

    def _second_order_update(self, x0_preds, t, s, sample):
        """2次ソルバー（中点法）- 擬似実装"""
        # 実際の DPM-Solver++ 2次は λ 空間での線形補間を使用
        # ここでは簡略化して1次の結果を返す
        return self._first_order_update(x0_preds[-1], t, s, sample)

    def _third_order_update(self, x0_preds, t, s, sample):
        """3次ソルバー - 擬似実装"""
        return self._first_order_update(x0_preds[-1], t, s, sample)


@dataclass
class SchedulerOutput:
    """スケジューラの1ステップ出力"""
    prev_sample: torch.Tensor  # [B, D] デノイズされたサンプル


# ============================================================================
# 拡散損失の計算
# ============================================================================

class DiffusionLoss(nn.Module):
    """
    VibeVoice の拡散損失モジュール。

    学習時に LLM の隠れ状態を条件として、
    Diffusion Head の予測と正解速度の MSE を計算。

    ddpm_batch_mul（デフォルト4）で1つの音声トークンから
    複数のタイムステップをサンプリングし、学習効率を向上。

    数式:
      L_diffusion = (1 / (D × M)) × Σ ||v_pred - v_target||²
      D = latent_size (64)
      M = ddpm_batch_mul (4)

    参照: modeling_vibevoice.py の forward() 内の拡散損失計算部分
    """

    def __init__(self, config):
        super().__init__()
        self.ddpm_num_steps = config.ddpm_num_steps        # 1000
        self.ddpm_batch_mul = config.ddpm_batch_mul        # 4
        self.latent_size = config.latent_size               # 64
        self.prediction_type = config.prediction_type       # "v_prediction"

    def forward(
        self,
        prediction_head: nn.Module,          # Diffusion Head
        noise_scheduler: DPMSolverMultistepScheduler,
        hidden_states: torch.Tensor,         # [B, T_total, hidden_size]
        acoustic_loss_mask: torch.BoolTensor, # [B, T_total]
        target_features: torch.Tensor,        # [N_speech, 64]
    ) -> torch.Tensor:
        """
        拡散損失を計算。

        処理フロー:
        1. LLM隠れ状態から条件ベクトルを抽出
        2. ddpm_batch_mul 倍のバッチに拡張
        3. ランダムタイムステップとノイズを生成
        4. 正解潜在変数にノイズを追加
        5. Diffusion Head で速度/ノイズを予測
        6. v-prediction ターゲットと MSE

        Args:
            prediction_head: Diffusion Head モジュール
            noise_scheduler: DPM-Solver++ スケジューラ
            hidden_states: [B, T_total, hidden_size] LLM全隠れ状態
            acoustic_loss_mask: [B, T_total] 条件抽出位置
            target_features: [N_speech, 64] 正解音声潜在変数

        Returns:
            loss: スカラー MSE 損失

        データフロー:
        ```
        hidden_states [B, T_total, hidden_size]
          → mask抽出 → condition [N_speech, hidden_size]
          → repeat(4) → [N*4, hidden_size]

        target_features [N_speech, 64]
          → repeat(4) → [N*4, 64]

        timesteps = randint(0, 1000, [N*4])
        noise = randn([N*4, 64])

        noisy = add_noise(target, noise, timesteps)  [N*4, 64]
        pred = diffusion_head(noisy, timesteps, condition)  [N*4, 64]

        target_v = get_velocity(target, noise, timesteps)  [N*4, 64]

        loss = mse(pred, target_v) / (64 * 4)
        ```
        """
        # === 条件ベクトル抽出 ===
        condition = hidden_states[acoustic_loss_mask]
        # [N_speech, hidden_size]
        N_speech = condition.shape[0]

        if N_speech == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        # === ddpm_batch_mul 倍に拡張 ===
        target_expanded = target_features.repeat(self.ddpm_batch_mul, 1)
        condition_expanded = condition.repeat(self.ddpm_batch_mul, 1)
        # [N*4, 64] and [N*4, hidden_size]

        total_batch = N_speech * self.ddpm_batch_mul

        # === ランダムタイムステップ ===
        timesteps = torch.randint(
            0, self.ddpm_num_steps, (total_batch,),
            device=target_features.device,
        )
        # [N*4]

        # === ランダムノイズ ===
        noise = torch.randn_like(target_expanded)
        # [N*4, 64]

        # === 順方向拡散（ノイズ追加） ===
        noisy_latents = noise_scheduler.add_noise(
            target_expanded, noise, timesteps
        )
        # [N*4, 64]
        # x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

        # === Diffusion Head で予測 ===
        pred = prediction_head(
            noisy_images=noisy_latents.unsqueeze(1),     # [N*4, 1, 64]
            timesteps=timesteps,                          # [N*4]
            condition=condition_expanded.unsqueeze(1),    # [N*4, 1, hidden_size]
        ).squeeze(1)
        # [N*4, 64]

        # === ターゲット計算 ===
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(
                target_expanded, noise, timesteps
            )
            # v = √ᾱ_t × ε - √(1-ᾱ_t) × x_0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        # [N*4, 64]

        # === MSE 損失 ===
        loss = F.mse_loss(pred, target)
        loss = loss / (self.latent_size * self.ddpm_batch_mul)
        # 正規化: / (64 × 4) = / 256

        return loss


# ============================================================================
# 学習パイプライン
# ============================================================================

class TrainingPipeline:
    """
    VibeVoice の学習パイプライン概要。

    学習構成:
      - 凍結: Acoustic Tokenizer (Enc + Dec), Semantic Tokenizer (Enc)
      - 学習: LLM (Qwen2.5), Diffusion Head, Acoustic/Semantic Connector
      - LM Head は LLM の embedding weight と共有（tied weights）

    損失関数:
      L_total = L_text + L_diffusion

      L_text: テキストトークンの Cross-Entropy Loss
        → 音声位置は -100 でマスク（無視）
        → テキストスクリプト部分のみ計算

      L_diffusion: 拡散 MSE Loss
        → 音声トークン位置のみ計算
        → ddpm_batch_mul=4 で効率化

    Curriculum Learning:
      ステージ1: max_seq_len = 4,096
      ステージ2: max_seq_len = 8,192
      ステージ3: max_seq_len = 16,384
      ステージ4: max_seq_len = 32,768
      ステージ5: max_seq_len = 65,536

    ハイパーパラメータ:
      - LLM: Qwen2.5 (1.5B or 7B)
      - Diffusion Head: 4層
      - 学習 Diffusion steps: 1000
      - 推論 Diffusion steps: 10 (VibeVoice) / 20 (Streaming)
      - CFG scale: 1.3
      - Beta schedule: cosine
      - Prediction type: v_prediction
      - ddpm_batch_mul: 4
      - Tokenizer fix_std: 0.5 (Acoustic), 0 (Semantic)

    参照: modeling_vibevoice.py の forward() メソッド
    """

    def __init__(self, model, optimizer, scheduler_config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

    def train_step(
        self,
        input_ids: torch.LongTensor,           # [B, T_total]
        attention_mask: torch.Tensor,           # [B, T_total]
        labels: torch.LongTensor,              # [B, T_total]
        speech_tensors: torch.Tensor,          # [B, T_audio]
        speech_masks: torch.BoolTensor,        # [B, T_latent]
        speech_semantic_tensors: torch.Tensor, # [B, T_latent, 128]
        acoustic_input_mask: torch.BoolTensor, # [B, T_total]
        speeches_loss_input: torch.BoolTensor, # [B, T_latent]
        acoustic_loss_mask: torch.BoolTensor,  # [B, T_total]
    ):
        """
        1ステップの学習。

        処理フロー:
        ```
        === 入力準備 ===
        1. テキストトークンを埋め込み
        2. 音声波形を Acoustic Tokenizer でエンコード → σ-VAE サンプリング
        3. 正規化（speech_scaling_factor, speech_bias_factor）
        4. Acoustic Connector で LLM 次元に射影
        5. Semantic Tokenizer の出力を Semantic Connector で射影
        6. acoustic_input_mask 位置に acoustic + semantic 特徴を挿入

        === LLM Forward ===
        7. Qwen2.5 で全トークン列を処理 → hidden_states [B, T, hidden_size]

        === テキスト損失 ===
        8. LM Head → logits [B, T, vocab_size]
        9. Cross-Entropy(logits, labels) ※ 音声位置は -100 で無視

        === 拡散損失 ===
        10. acoustic_loss_mask で LLM 隠れ状態から条件抽出
        11. speeches_loss_input で正解音声潜在変数を抽出
        12. ランダムタイムステップ + ノイズ生成
        13. 正解潜在変数にノイズ追加
        14. Diffusion Head で速度予測
        15. MSE(予測速度, 正解速度) / (64 × 4)

        === 最適化 ===
        16. L_total = L_text + L_diffusion
        17. backward + optimizer.step
        ```
        """
        self.optimizer.zero_grad()

        # Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_semantic_tensors=speech_semantic_tensors,
            acoustic_input_mask=acoustic_input_mask,
            speeches_loss_input=speeches_loss_input,
            acoustic_loss_mask=acoustic_loss_mask,
        )

        # Backward
        loss = outputs.loss
        loss.backward()

        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'diffusion_loss': outputs.diffusion_loss.item() if outputs.diffusion_loss else 0,
            'speech_token_num': outputs.speech_token_num,
        }


# ============================================================================
# TTS データプロセッサ
# ============================================================================

class VibeVoiceProcessor:
    """
    TTS タスク用のデータ前処理プロセッサ。

    テキストスクリプト + 音声プロンプトから LLM 入力を構築。

    設定値:
      - speech_tok_compress_ratio: 3200（24kHz / 7.5Hz）
      - db_normalize: True
      - system_prompt: "Transform the text provided by various speakers
                        into speech output..."

    入力フォーマット:
    ```
    Speaker 0: Welcome to the show...
    Speaker 1: Thanks for having me...
    Speaker 2: Hello, I'm excited...
    ```

    出力トークン構造:
    ```
    [System Prompt]
    + [Voice Prompt Speaker 0]: <speech_start> <diffusion>×N <speech_end>
    + [Voice Prompt Speaker 1]: <speech_start> <diffusion>×N <speech_end>
    + [Text Input Section]:
        Speaker 0: Welcome to the show...
        Speaker 1: Thanks for having me...
    + [Speech Output]: <speech_start>
    ```

    参照: vibevoice_processor.py の VibeVoiceProcessor
    """

    SYSTEM_PROMPT = (
        "Transform the text provided by various speakers into speech output. "
        "Follow the script exactly, maintaining appropriate tone, emotion, "
        "and speaking style for each speaker."
    )

    def __init__(self, tokenizer, audio_processor, compress_ratio=3200):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = compress_ratio

    def __call__(self, text, voice_samples, **kwargs):
        """
        テキストスクリプトと音声プロンプトからモデル入力を構築。

        Args:
            text: str テキストスクリプト
            voice_samples: Dict[int, str/array] 話者ID → 音声サンプル

        Returns:
            BatchEncoding with:
                input_ids: [B, S]
                attention_mask: [B, S]
                speech_tensors: [B, N_speakers, T_audio]
                speech_masks: [B, N_speakers, T_latent]
                acoustic_input_mask: [B, S]
        """
        # スクリプト解析
        parsed = self._parse_script(text)
        # [(speaker_id, text), ...]

        # 音声プロンプトの処理
        voice_tokens, voice_speech, voice_masks = self._create_voice_prompt(
            voice_samples
        )

        # テキスト入力セクション
        text_section = ""
        for speaker_id, line in parsed:
            text_section += f"Speaker {speaker_id}: {line}\n"

        # トークン列の構築
        # [System] + [Voice Prompts] + [Text Input] + [Speech Output Start]
        tokens = self.tokenizer.encode(self.SYSTEM_PROMPT)
        tokens += voice_tokens
        tokens += self.tokenizer.encode(text_section)
        tokens += self.tokenizer.encode("<|speech_start|>")

        return {
            'input_ids': torch.tensor(tokens),
            'voice_speech': voice_speech,
            'voice_masks': voice_masks,
        }

    def _parse_script(self, text: str):
        """
        テキストスクリプトを解析。

        入力: "Speaker 0: Hello\nSpeaker 1: Hi there"
        出力: [(0, "Hello"), (1, "Hi there")]
        """
        import re
        lines = text.strip().split('\n')
        parsed = []
        for line in lines:
            match = re.match(r'Speaker\s+(\d+):\s*(.*)', line)
            if match:
                speaker_id = int(match.group(1))
                content = match.group(2).strip()
                parsed.append((speaker_id, content))
        return parsed

    def _create_voice_prompt(self, voice_samples):
        """
        各話者の音声サンプルをトークン化。

        各話者について:
        1. 音声をロード・正規化
        2. VAE トークン長を計算: ceil(samples / 3200)
        3. <speech_start> + <diffusion>×N + <speech_end> トークン列を作成

        Returns:
            (tokens, speech_inputs, speech_masks)
        """
        tokens = []
        speech_inputs = []
        speech_masks = []

        for speaker_id, audio in sorted(voice_samples.items()):
            if isinstance(audio, str):
                waveform = load_audio_file(audio)
            else:
                waveform = audio

            num_samples = waveform.shape[-1]
            vae_tok_len = -(-num_samples // self.speech_tok_compress_ratio)

            # トークン列: <speech_start> + <diffusion>×N + <speech_end>
            speaker_tokens = (
                self.tokenizer.encode(f"Speaker {speaker_id}: ")
                + [SPEECH_START_ID]
                + [DIFFUSION_TOKEN_ID] * vae_tok_len
                + [SPEECH_END_ID]
                + self.tokenizer.encode("\n")
            )

            tokens.extend(speaker_tokens)
            speech_inputs.append(waveform)
            speech_masks.append(torch.ones(vae_tok_len, dtype=torch.bool))

        return tokens, speech_inputs, speech_masks


# ============================================================================
# 音声特徴の正規化詳細
# ============================================================================

class SpeechNormalization:
    """
    音声特徴の正規化（dynamic scaling/bias）。

    学習開始時に最初のバッチの統計量から動的に計算。
    DDP 環境では全GPU間で同期。

    処理:
    1. 有効トークン（speech_masks=True）の mean と std を計算
    2. bias_factor = -mean
    3. scaling_factor = 1 / std
    4. 適用: features = (tokens + bias_factor) * scaling_factor
       → 零平均・単位分散に正規化

    DDP同期:
      all_reduce(mean, op=SUM) / world_size
      all_reduce(std, op=SUM) / world_size

    一度計算されたら buffer に保存し、以降再計算しない。

    参照: modeling_vibevoice.py の forward_speech_features() 内
    """

    @staticmethod
    def compute_and_apply(
        audio_tokens: torch.Tensor,          # [B, T, 64]
        speech_masks: torch.BoolTensor,       # [B, T]
        scaling_factor: torch.Tensor,         # buffer (初回 NaN)
        bias_factor: torch.Tensor,            # buffer (初回 NaN)
    ) -> tuple:
        """
        正規化パラメータの計算と適用。

        Args:
            audio_tokens: [B, T, 64] σ-VAE サンプリング後の潜在変数
            speech_masks: [B, T] 有効トークンマスク
            scaling_factor: 保存済みスケーリング（初回 NaN）
            bias_factor: 保存済みバイアス（初回 NaN）

        Returns:
            (normalized_features, scaling_factor, bias_factor)
            normalized_features: [B, T, 64] 正規化済み

        数値例:
            audio_tokens の統計:
              mean ≈ 0.12, std ≈ 0.45 (データ依存)
            → bias_factor = -0.12
            → scaling_factor = 1 / 0.45 ≈ 2.22
            → normalized = (tokens - 0.12) * 2.22
            → 零平均・単位分散
        """
        if torch.isnan(scaling_factor):
            valid = audio_tokens[speech_masks]  # [N_valid, 64]
            mean_val = valid.mean()
            std_val = valid.std()

            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(mean_val, op=torch.distributed.ReduceOp.SUM)
                mean_val /= world_size
                torch.distributed.all_reduce(std_val, op=torch.distributed.ReduceOp.SUM)
                std_val /= world_size

            bias_factor = -mean_val
            scaling_factor = 1.0 / std_val

        normalized = (audio_tokens + bias_factor) * scaling_factor
        return normalized, scaling_factor, bias_factor


# ============================================================================
# ユーティリティ
# ============================================================================

# 特殊トークンID（実際の値はトークナイザ依存）
SPEECH_START_ID = 151646
DIFFUSION_TOKEN_ID = 151648
SPEECH_END_ID = 151647

def load_audio_file(path: str) -> torch.Tensor:
    """音声ファイルのロード（プレースホルダ）"""
    pass
