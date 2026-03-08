"""
SiT - Stochastic Interpolant Transport Framework

対応:
  transport/__init__.py   (create_transport ファクトリ)
  transport/transport.py  (Transport, Sampler クラス)
  transport/path.py       (ICPlan, GVPCPlan, VPCPlan パス設計)
  transport/integrators.py (ODE/SDE ソルバー)

SiTの核心部分。DiTとの違いは全てここに集約されている。

========================================
構成
========================================
1. パス設計 (Coupling Plan)
   - ICPlan   (Linear): α(t)=t, σ(t)=1-t           → Flow Matching
   - GVPCPlan (Cosine): α(t)=sin(πt/2), σ(t)=cos(πt/2)  → 測地線パス
   - VPCPlan  (VP):     指数的α, √(1-α²)のσ        → VP-SDE互換

2. Transport
   - training_losses(): velocity/score/noise の3モード学習
   - get_drift():       ODE推論用の速度場関数
   - get_score():       SDE推論用のスコア関数

3. Sampler
   - sample_ode():      ODE推論 (dopri5 / Euler / Heun)
   - sample_sde():      SDE推論 (Euler-Maruyama / Heun)
   - sample_ode_likelihood(): 尤度計算付きODE推論

4. Integrators
   - ode: torchdiffeq.odeint ラッパー
   - sde: Euler-Maruyama / Heun ステッパー
"""

import torch
import torch.nn as nn
import numpy as np
import enum
from functools import partial
from torchdiffeq import odeint


# ============================================================
# ユーティリティ
# ============================================================

def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """時刻 t をデータ x の次元に合わせてreshape

    入力: t (B,), x (B, C, H, W) または (B, ...)
    出力: t (B, 1, 1, 1) etc.  ← xと同次元数

    対応: 公式 path.py L5-L13
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """空間次元にわたる平均 (バッチ次元は保持)

    入力: (B, C, H, W) → 出力: (B,)
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ============================================================
# 列挙型
# ============================================================

class ModelType(enum.Enum):
    """モデルの予測対象"""
    NOISE = enum.auto()     # ε (入力ノイズ x_0) を予測
    SCORE = enum.auto()     # ∇_x log p_t(x) を予測
    VELOCITY = enum.auto()  # u_t (速度場) を予測  ← デフォルト・推奨

class PathType(enum.Enum):
    """補間パスの種類"""
    LINEAR = enum.auto()    # ICPlan:  直線パス (= Flow Matching)
    GVP = enum.auto()       # GVPCPlan: 測地線パス (cosine)
    VP = enum.auto()        # VPCPlan:  VP-SDE互換パス

class WeightType(enum.Enum):
    """損失の重み付け"""
    NONE = enum.auto()      # w(t) = 1
    VELOCITY = enum.auto()  # w(t) = (drift_var / σ_t)²
    LIKELIHOOD = enum.auto()# w(t) = drift_var / σ_t²


# ============================================================
# パス設計 (Coupling Plan)
# ============================================================

class ICPlan:
    """
    Linear Coupling Plan (= Flow Matching / Optimal Transport CFM)

    α(t) = t        → dα/dt = 1
    σ(t) = 1 - t    → dσ/dt = -1

    x_t = t × x_1 + (1-t) × x_0     (直線補間)
    u_t = x_1 - x_0                   (一定速度場)

    ========================================
    記号の対応
    ========================================
    x_0: ノイズ ~ N(0, I)  (t=0: 純粋ノイズ)
    x_1: データ            (t=1: 純粋データ)
    x_t: 補間点            (0<t<1: 中間状態)
    u_t: 条件付き速度場    (学習ターゲット)

    ※ DiT/DDPMとは逆の convention:
      DiT:  t=0 がデータ, t=T がノイズ
      SiT:  t=0 がノイズ, t=1 がデータ

    対応: 公式 path.py L18-L136 (ICPlan)
    """

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma  # 未使用 (将来の拡張用)

    def compute_alpha_t(self, t: torch.Tensor):
        """データ係数 α(t) とその微分

        入力: t (B, 1, 1, 1) or (B,)
        出力: (α_t, dα/dt)  ← 各 t と同じ shape

        Linear: α(t) = t, dα/dt = 1
        """
        return t, 1

    def compute_sigma_t(self, t: torch.Tensor):
        """ノイズ係数 σ(t) とその微分

        入力: t (B, 1, 1, 1) or (B,)
        出力: (σ_t, dσ/dt)

        Linear: σ(t) = 1 - t, dσ/dt = -1
        """
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor):
        """dα/dt / α(t) = 1 / t

        SDE drift計算の数値安定性のための分離計算
        """
        return 1 / t

    def compute_drift(self, x: torch.Tensor, t: torch.Tensor):
        """SDE表現のドリフト項を計算

        x_t の確率流ODE: dx = -drift_mean × dt + drift_var × score × dt

        入力: x (B, 4, 32, 32), t (B,)
        出力: (drift_mean, drift_var)  各 (B, 4, 32, 32) or スカラー

        ========================================
        導出 (Linear path の場合)
        ========================================
        drift_mean  = (dα/dt / α_t) × x_t = x_t / t
        drift_var   = (dα/dt / α_t) × σ_t² - σ_t × dσ/dt
                    = (1/t) × (1-t)² - (1-t) × (-1)
                    = (1-t)²/t + (1-t)
                    = (1-t)(1-t+t)/t = (1-t)/t ... ← 実際の計算

        返り値: (-drift_mean, drift_var)
        符号反転に注意: ODE dx/dt = -(-drift_mean) + drift_var × score
                                   = drift_mean + drift_var × score

        対応: 公式 path.py L35-L43
        """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        # alpha_ratio: (B, 1, 1, 1) = 1/t

        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        # sigma_t: 1-t, d_sigma_t: -1

        drift = alpha_ratio * x
        # drift: (B, 4, 32, 32) = x_t / t

        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        # diffusion: (B, 1, 1, 1) = (1-t)²/t + (1-t) = (1-t)/t ← 簡約後

        return -drift, diffusion

    def compute_diffusion(self, x: torch.Tensor, t: torch.Tensor,
                          form: str = "constant", norm: float = 1.0):
        """SDE推論の拡散係数 g(t) を計算

        入力: x (B, 4, 32, 32), t (B,), form (str), norm (float)
        出力: diffusion ← xと同じ or broadcastable shape

        ========================================
        拡散形式の選択肢
        ========================================
        "constant":    norm                          (定数)
        "SBDM":        norm × drift_var(t)           (Score-Based Diffusion Matching)
        "sigma":       norm × σ(t)                   (ノイズ係数に比例)
        "linear":      norm × (1-t)                  (線形減少)
        "decreasing":  0.25 × (norm×cos(πt)+1)²      (cosine減少)
        "increasing-decreasing": norm × sin(πt)²     (山型)

        対応: 公式 path.py L45-L68
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        return choices[form]

    def get_score_from_velocity(self, velocity: torch.Tensor,
                                x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """velocity予測からscore (∇ log p_t) を復元

        入力:
          velocity: (B, 4, 32, 32)  ← モデルの velocity 予測
          x:        (B, 4, 32, 32)  ← x_t
          t:        (B,)

        出力:
          score: (B, 4, 32, 32) = ∇_x log p_t(x_t)

        ========================================
        導出
        ========================================
        x_t = α_t × x_1 + σ_t × x_0
        u_t = dα/dt × x_1 + dσ/dt × x_0

        x_1 = (x_t - σ_t × x_0) / α_t
        u_t = dα/dt × (x_t - σ_t × x_0) / α_t + dσ/dt × x_0

        score = ∇_x log p(x_t | x_0, x_1) = (α_t × x_1 - x_t) / σ_t²
        変形: score = (α_t/dα_t × velocity - x_t) / (σ_t² - α_t/dα_t × dσ_t × σ_t)

        Linear: α_t/dα_t = t/1 = t
                var = (1-t)² - t × (-1) × (1-t) = (1-t)² + t(1-t) = (1-t)
                score = (t × velocity - x_t) / (1-t)

        対応: 公式 path.py L70-L84
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)

        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        # Linear: reverse_alpha_ratio = t / 1 = t

        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        # Linear: var = (1-t)² - t × (-1) × (1-t) = (1-t)² + t(1-t) = (1-t)

        score = (reverse_alpha_ratio * velocity - mean) / var
        # score: (B, 4, 32, 32) = (t × v - x_t) / (1-t)
        return score

    def get_noise_from_velocity(self, velocity: torch.Tensor,
                                x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """velocity予測からnoise (x_0) を復元

        入力/出力: get_score_from_velocity と同じ shape

        対応: 公式 path.py L86-L100
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)

        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        # Linear: var = t × (-1) - (1-t) = -t - 1 + t = -1

        noise = (reverse_alpha_ratio * velocity - mean) / var
        # noise: (B, 4, 32, 32) = (t × v - x_t) / (-1) = x_t - t × v
        return noise

    def compute_mu_t(self, t: torch.Tensor,
                     x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """p_t の平均を計算: μ_t = α_t × x_1 + σ_t × x_0

        入力: t (B,), x0 (B, 4, 32, 32), x1 (B, 4, 32, 32)
        出力: mu_t (B, 4, 32, 32)

        対応: 公式 path.py L114-L119
        """
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t: torch.Tensor,
                   x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """x_t をサンプリング (確定的、σ=0の場合は μ_t そのまま)

        Linear: x_t = t × x_1 + (1-t) × x_0

        対応: 公式 path.py L121-L124
        """
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t: torch.Tensor,
                   x0: torch.Tensor, x1: torch.Tensor,
                   xt: torch.Tensor) -> torch.Tensor:
        """条件付き速度場 u_t を計算 (学習ターゲット)

        入力: t (B,), x0 (B, 4, 32, 32), x1 (B, 4, 32, 32), xt (未使用)
        出力: u_t (B, 4, 32, 32)

        u_t = dα/dt × x_1 + dσ/dt × x_0

        Linear: u_t = 1 × x_1 + (-1) × x_0 = x_1 - x_0  ← 定速度場

        対応: 公式 path.py L126-L131
        """
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t: torch.Tensor,
             x0: torch.Tensor, x1: torch.Tensor):
        """学習用: (t, x_t, u_t) を計算

        入力: t (B,), x0 (B, 4, 32, 32), x1 (B, 4, 32, 32)
        出力:
          t:  (B,)
          xt: (B, 4, 32, 32) = α_t × x_1 + σ_t × x_0
          ut: (B, 4, 32, 32) = dα/dt × x_1 + dσ/dt × x_0

        対応: 公式 path.py L133-L136
        """
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class GVPCPlan(ICPlan):
    """
    Geodesic Variational Path (= Cosine Schedule)

    α(t) = sin(πt/2)        → dα/dt = (π/2) × cos(πt/2)
    σ(t) = cos(πt/2)        → dσ/dt = -(π/2) × sin(πt/2)

    性質:
    - α²(t) + σ²(t) = sin² + cos² = 1  (分散保存)
    - t=0付近: ゆっくり変化 (ノイズが徐々に減少)
    - t=0.5付近: 最も速く変化
    - CosyVoice3の `t = 1 - cos(u × π/2)` スケジューリングと同系統

    ICPlan を継承し、compute_alpha_t / compute_sigma_t のみオーバーライド

    対応: 公式 path.py L174-L192 (GVPCPlan)
    """

    def __init__(self, sigma: float = 0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t: torch.Tensor):
        """α(t) = sin(πt/2), dα/dt = (π/2) cos(πt/2)"""
        alpha_t = torch.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: torch.Tensor):
        """σ(t) = cos(πt/2), dσ/dt = -(π/2) sin(πt/2)"""
        sigma_t = torch.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor):
        """dα/dt / α(t) = (π/2) cos(πt/2) / sin(πt/2) = π / (2 tan(πt/2))"""
        return np.pi / (2 * torch.tan(t * np.pi / 2))


class VPCPlan(ICPlan):
    """
    Variance Preserving Path (VP-SDE互換)

    log α(t) = -0.25 × (1-t)² × (σ_max - σ_min) - 0.5 × (1-t) × σ_min
    α(t)     = exp(log α(t))
    σ(t)     = √(1 - α(t)²)

    σ_min = 0.1, σ_max = 20.0 (VP-SDEのデフォルト)

    ========================================
    特徴
    ========================================
    - DDPMのlinear β scheduleに対応するパス
    - t=0付近: α≈0 (ほぼノイズ)
    - t=1付近: α≈1 (ほぼデータ)
    - Score Matching / DDPM との互換性のため用意

    ========================================
    VP-SDEとの対応
    ========================================
    VP-SDE: dx = -0.5 β(t) x dt + √β(t) dW
    β(t) = σ_min + (1-t)(σ_max - σ_min)

    ICPlan を継承し、α/σ/drift をオーバーライド

    対応: 公式 path.py L139-L171 (VPCPlan)
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # log α(t) の計算 (数値安定性のため対数で保持)
        self.log_mean_coeff = lambda t: (
            -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )
        # d/dt [log α(t)]
        self.d_log_mean_coeff = lambda t: (
            0.5 * (1 - t) * (self.sigma_max - self.sigma_min)
            + 0.5 * self.sigma_min
        )

    def compute_alpha_t(self, t: torch.Tensor):
        """α(t) = exp(log_mean_coeff(t))

        dα/dt = α(t) × d_log_mean_coeff(t)
        """
        alpha_t = self.log_mean_coeff(t)
        alpha_t = torch.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: torch.Tensor):
        """σ(t) = √(1 - α(t)²)

        dσ/dt = -α(t) × dα/dt / σ(t)
              = exp(2 log α) × (2 × d_log_mean_coeff) / (-2σ)
        """
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        d_sigma_t = torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor):
        """dα/dt / α(t) = d_log_mean_coeff(t)  ← 対数微分で安定計算"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x: torch.Tensor, t: torch.Tensor):
        """VP-SDEのドリフト

        dx = -0.5 β(t) x dt  →  drift_mean = -0.5 β(t) x
        β(t) = σ_min + (1-t)(σ_max - σ_min)

        出力: (drift_mean, drift_var) = (-0.5 β x, β/2)
        """
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


# ============================================================
# Transport クラス (学習フレームワーク)
# ============================================================

class Transport:
    """
    Stochastic Interpolant Transport

    学習時の補間・損失計算と、推論時のODE/SDE drift/score関数を提供する。

    ========================================
    フレームワーク概要
    ========================================
    1. 学習:
       t ~ U[0,1], x_0 ~ N(0,I), x_1 = data
       x_t = path.plan(t, x_0, x_1)
       loss = ||model(x_t, t) - target||²

    2. 推論:
       ODE: dx/dt = drift_fn(x, t, model)
       SDE: dx = [drift + g²×score] dt + √(2g²) dW

    対応: 公式 transport/transport.py L39-L210
    """

    def __init__(
        self,
        *,
        model_type: ModelType,
        path_type: PathType,
        loss_type: WeightType,
        train_eps: float,
        sample_eps: float,
    ):
        path_options = {
            PathType.LINEAR: ICPlan,
            PathType.GVP: GVPCPlan,
            PathType.VP: VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """標準正規分布の対数尤度

        入力: z (B, 4, 32, 32)
        出力: logp (B,)

        log p(z) = -N/2 × log(2π) - ||z||² / 2
        N = C × H × W = 4 × 32 × 32 = 4096

        尤度計算 (sample_ode_likelihood) で使用
        """
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - torch.sum(x ** 2) / 2.
        return torch.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        eval: bool = False,
        last_step_size: float = 0.0,
    ):
        """積分区間 [t0, t1] の決定

        ========================================
        数値安定性のための ε 調整
        ========================================
        - t=0, t=1 端点では 1/t や 1/(1-t) が発散
        - velocity + Linear/GVP は全域で安定 → ε=0 (調整不要)
        - score/noise 予測や VP パスでは ε > 0 で端点を避ける

        学習時: [train_eps, 1-train_eps] or [0, 1]
        推論時: [sample_eps, 1-sample_eps] or [0, 1]

        SDEの場合: last_step で最終ステップを特別処理するため t1 を調整

        対応: 公式 transport/transport.py L73-L100
        """
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps

        if type(self.path_sampler) in [VPCPlan]:
            # VP: 常に端点を避ける
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [ICPlan, GVPCPlan]
              and (self.model_type != ModelType.VELOCITY or sde)):
            # Linear/GVP で score/noise予測、またはSDE推論の場合
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        # velocity + Linear/GVP + ODE の場合: t0=0, t1=1 (ε不要)

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1: torch.Tensor):
        """学習用: ノイズ x_0 と時刻 t をサンプリング

        入力: x1 (B, 4, 32, 32) ← データ
        出力:
          t:  (B,) ∈ [t0, t1]   ← 一様分布
          x0: (B, 4, 32, 32)    ← ノイズ ~ N(0, I)
          x1: (B, 4, 32, 32)    ← データ (そのまま返す)

        対応: 公式 transport/transport.py L103-L113
        """
        x0 = torch.randn_like(x1)
        # x0: (B, 4, 32, 32)

        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        # t: (B,) ∈ [t0, t1]

        return t, x0, x1

    def training_losses(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        model_kwargs: dict = None,
    ) -> dict:
        """
        学習損失の計算

        入力:
          model: SiTモデル
          x1: (B, 4, 32, 32)  ← VAE潜在 (データ)
          model_kwargs: {"y": class_labels}

        出力:
          {"pred": (B, 4, 32, 32), "loss": (B,)}

        ========================================
        処理フロー
        ========================================
        1. t ~ U[0,1], x0 ~ N(0,I) をサンプリング
        2. x_t = α_t × x_1 + σ_t × x_0 を計算 (path.plan)
        3. u_t = dα/dt × x_1 + dσ/dt × x_0 をターゲットとして計算
        4. model_output = model(x_t, t, y)
        5. 予測モードに応じた損失計算

        ========================================
        予測モード別の損失
        ========================================
        velocity: L = ||model_output - u_t||²
        noise:    L = w(t) × ||model_output - x_0||²
        score:    L = w(t) × ||model_output × σ_t + x_0||²
                  (score × σ_t = -x_0 の関係から)

        対応: 公式 transport/transport.py L116-L158
        """
        if model_kwargs is None:
            model_kwargs = {}

        # --- Step 1: サンプリング ---
        t, x0, x1 = self.sample(x1)
        # t: (B,), x0: (B, 4, 32, 32), x1: (B, 4, 32, 32)

        # --- Step 2-3: 補間と速度場の計算 ---
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        # xt: (B, 4, 32, 32) = α_t × x_1 + σ_t × x_0
        # ut: (B, 4, 32, 32) = dα/dt × x_1 + dσ/dt × x_0
        # Linear の場合: xt = t×x_1 + (1-t)×x_0, ut = x_1 - x_0

        # --- Step 4: モデル推論 ---
        model_output = model(xt, t, **model_kwargs)
        # model_output: (B, 4, 32, 32)

        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        # --- Step 5: 損失計算 ---
        terms = {}
        terms['pred'] = model_output

        if self.model_type == ModelType.VELOCITY:
            # velocity予測: 最もシンプル、重み付け不要
            terms['loss'] = mean_flat((model_output - ut) ** 2)
            # loss: (B,) = E[||v_θ(x_t, t) - u_t||²]

        else:
            # score/noise予測: 重み付きMSE
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))

            # 重みの選択
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                # noise予測: target = x_0
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                # score予測: score × σ_t = -(x_t - α_t×x_1)/σ_t × σ_t = -(x_0 - ε項)
                # → model_output × σ_t + x_0 = 0 が理想
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    def get_drift(self):
        """ODE推論用の drift 関数を返す

        ========================================
        予測モード別の ODE drift
        ========================================
        velocity: dx/dt = model(x, t)
                  ← 直接速度場として使える (最もシンプル)

        score:    dx/dt = -drift_mean + drift_var × model(x, t)
                  ← 確率流ODE変換

        noise:    dx/dt = -drift_mean + drift_var × (model(x, t) / -σ_t)
                  ← noise → score → drift の2段変換

        対応: 公式 transport/transport.py L161-L193
        """
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output)

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output

        return body_fn

    def get_score(self):
        """SDE推論用の score 関数を返す

        ========================================
        予測モード別の score 変換
        ========================================
        noise:    score = model(x, t) / -σ_t
        score:    score = model(x, t)    ← そのまま
        velocity: score = path.get_score_from_velocity(model(x, t), x, t)

        対応: 公式 transport/transport.py L196-L210
        """
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: \
                model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: \
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()

        return score_fn


# ============================================================
# ODE / SDE ソルバー (Integrators)
# ============================================================

class ODESolver:
    """
    ODE ソルバー (torchdiffeq ラッパー)

    dx/dt = drift(x, t)  を t0 → t1 で積分

    ========================================
    サポートするソルバー
    ========================================
    - "dopri5": Dormand-Prince 4/5次適応ステップ (デフォルト)
    - "euler": 1次固定ステップ Euler法
    - "heun": 2次固定ステップ Heun法 (改良Euler)

    入力: x_init (B, 4, 32, 32) ← t=0 のノイズ
    出力: samples (num_steps, B, 4, 32, 32) ← 各時刻の状態

    対応: 公式 transport/integrators.py L77-L115 (ode class)
    """

    def __init__(
        self,
        drift,
        *,
        t0: float,
        t1: float,
        sampler_type: str,
        num_steps: int,
        atol: float,
        rtol: float,
    ):
        self.drift = drift
        self.t = torch.linspace(t0, t1, num_steps)
        # t: (num_steps,)  例: [0.0, 0.02, 0.04, ..., 1.0] (50ステップ)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """ODE求解

        入力: x (B, 4, 32, 32) or tuple((B, 4, 32, 32), (B,))  ← 尤度計算時
        出力: samples (num_steps, B, 4, 32, 32)

        ========================================
        torchdiffeq.odeint の呼び出し
        ========================================
        odeint(fn, x0, t, method="dopri5", atol=1e-6, rtol=1e-3)
        - fn(t, x): 速度場 dx/dt を返す関数
        - x0: 初期値
        - t: 評価時刻のリスト (昇順)
        - method: ソルバー種類
        - 戻り値: 各時刻での状態 (num_steps, *x0.shape)

        dopri5 (適応的):
          - 内部的に可変ステップで精密に積分
          - num_steps は出力の保存点数 (内部ステップ数ではない)
          - atol, rtol で精度制御

        euler/heun (固定ステップ):
          - num_steps がそのまま積分ステップ数
          - 大きいステップ数で精度向上
        """
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t_batch = (torch.ones(x[0].size(0)).to(device) * t
                       if isinstance(x, tuple)
                       else torch.ones(x.size(0)).to(device) * t)
            return self.drift(x, t_batch, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]

        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol,
        )
        # samples: (num_steps, B, 4, 32, 32)
        return samples


class SDESolver:
    """
    SDE ソルバー

    dx = drift(x, t) dt + √(2 × diffusion(x, t)) dW

    ========================================
    サポートするソルバー
    ========================================
    - "Euler": Euler-Maruyama法 (1次)
    - "Heun": Heun法 (2次)

    入力: x_init (B, 4, 32, 32) ← t=0 のノイズ
    出力: samples [(B, 4, 32, 32)] × (num_steps - 1)

    対応: 公式 transport/integrators.py L8-L75 (sde class)
    """

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0: float,
        t1: float,
        num_steps: int,
        sampler_type: str,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = torch.linspace(t0, t1, num_steps)
        # t: (num_steps,)  例: [0.0, 0.004, 0.008, ..., 0.96] (250ステップ)
        self.dt = self.t[1] - self.t[0]
        # dt: スカラー  例: 0.004 (250ステップの場合)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def _euler_maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        """Euler-Maruyama ステップ

        入力:
          x:      (B, 4, 32, 32) ← 現在の状態
          mean_x: (B, 4, 32, 32) ← 平均追跡 (last_step用)
          t:      スカラー

        出力:
          x_next:    (B, 4, 32, 32) ← 次の状態
          mean_next: (B, 4, 32, 32) ← 次の平均

        ========================================
        アルゴリズム
        ========================================
        dW = √dt × z    (z ~ N(0, I))
        mean = x + drift(x, t) × dt
        x_next = mean + √(2 × diffusion) × dW

        drift は SDE用: drift_ODE + diffusion² × score
        """
        w_cur = torch.randn(x.size()).to(x)
        t_batch = torch.ones(x.size(0)).to(x) * t
        dw = w_cur * torch.sqrt(self.dt)
        # dw: (B, 4, 32, 32) ← Wiener過程の増分

        drift = self.drift(x, t_batch, model, **model_kwargs)
        # drift: (B, 4, 32, 32) ← drift_ODE + g² × score

        diffusion = self.diffusion(x, t_batch)
        # diffusion: broadcastable ← g(t)

        mean_x = x + drift * self.dt
        # mean_x: (B, 4, 32, 32) ← 決定的な更新

        x = mean_x + torch.sqrt(2 * diffusion) * dw
        # x: (B, 4, 32, 32) ← 確率的ノイズ付き更新

        return x, mean_x

    def _heun_step(self, x, _, t, model, **model_kwargs):
        """Heun ステップ (2次精度)

        ========================================
        アルゴリズム
        ========================================
        1. ノイズ付加:  x̂ = x + √(2 × diffusion) × dW
        2. 1次予測:     x_pred = x̂ + dt × drift(x̂, t)
        3. 補正:        x_next = x̂ + 0.5 × dt × (drift(x̂, t) + drift(x_pred, t+dt))

        Euler-Maruyamaより精度が高いが、drift を2回評価するため計算コスト2倍
        """
        w_cur = torch.randn(x.size()).to(x)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def sample(self, init, model, **model_kwargs):
        """SDE求解ループ

        入力: init (B, 4, 32, 32) ← 初期ノイズ
        出力: samples [(B, 4, 32, 32)] × (num_steps - 1)

        各時刻 t0, t1, ..., t_{N-2} で1ステップずつ進める
        (最終ステップは Sampler.sample_sde 内で last_step_fn として別処理)
        """
        sampler_dict = {
            "Euler": self._euler_maruyama_step,
            "Heun": self._heun_step,
        }
        sampler = sampler_dict[self.sampler_type]

        x = init
        mean_x = init
        samples = []

        for ti in self.t[:-1]:
            with torch.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        # samples: [(B, 4, 32, 32)] × (num_steps - 1)
        return samples


# ============================================================
# Sampler クラス (推論フレームワーク)
# ============================================================

class Sampler:
    """
    Sampler: Transport の推論インターフェース

    Transport から drift (ODE用) と score (SDE用) 関数を取得し、
    各推論手法に必要な設定を構成して sample_fn を返す。

    対応: 公式 transport/transport.py L213-L440 (Sampler class)
    """

    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
    ):
        """SDE推論用の drift と diffusion を構成

        ========================================
        SDE: dx = sde_drift dt + √(2 × sde_diffusion) dW
        ========================================
        sde_drift     = ode_drift + diffusion × score
        sde_diffusion = g(x, t)  (compute_diffusion で選択)

        これは確率流ODE → SDE への変換 (Anderson, 1982):
        同じ p_t(x) を生成するODEとSDEの関係:
          ODE: dx = v(x,t) dt
          SDE: dx = [v(x,t) + g²(t)×∇log p_t(x)] dt + √(2g²(t)) dW

        対応: 公式 transport/transport.py L228-L245
        """
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        sde_drift = lambda x, t, model, **kwargs: \
            self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        return sde_drift, diffusion_fn

    def _get_last_step(self, sde_drift, *, last_step, last_step_size):
        """SDE推論の最終ステップ関数を構成

        ========================================
        最終ステップの選択肢
        ========================================
        None:     x_final = x                        (何もしない)
        "Mean":   x_final = x + sde_drift × dt       (ドリフトのみで1ステップ)
        "Tweedie":x_final = x/α + σ²/α × score      (Tweedieの公式で推定)
        "Euler":  x_final = x + ode_drift × dt       (ODE driftで1ステップ)

        SDE推論では最終ステップで確率的ノイズを除去するため、
        決定的な最終ステップ処理が品質向上に寄与する。

        対応: 公式 transport/transport.py L247-L277
        """
        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x

        elif last_step == "Mean":
            last_step_fn = lambda x, t, model, **model_kwargs: \
                x + sde_drift(x, t, model, **model_kwargs) * last_step_size

        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: \
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)

        elif last_step == "Euler":
            last_step_fn = lambda x, t, model, **model_kwargs: \
                x + self.drift(x, t, model, **model_kwargs) * last_step_size

        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_ode(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
    ):
        """ODE推論のサンプリング関数を返す

        ========================================
        ODE: dx/dt = drift(x, t)
        ========================================
        - velocity: drift = model(x, t)  ← そのまま
        - score:    drift = -drift_mean + drift_var × model(x, t)
        - noise:    drift = -drift_mean + drift_var × (model(x, t) / -σ_t)

        t: 0 (ノイズ) → 1 (データ)

        reverse=True の場合: t: 1 → 0 (データ → ノイズ、エンコード用)

        入力 (返り値の関数):
          init: (B, 4, 32, 32) ← 初期ノイズ z ~ N(0, I)
          model: SiTモデル
          **model_kwargs: {"y": labels, "cfg_scale": 4.0}

        出力:
          samples: (num_steps, B, 4, 32, 32)
                   samples[-1] が最終結果

        対応: 公式 transport/transport.py L341-L381
        """
        drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ODESolver(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: str = "Mean",
        last_step_size: float = 0.04,
        num_steps: int = 250,
    ):
        """SDE推論のサンプリング関数を返す

        ========================================
        SDE: dx = [drift + g² × score] dt + √(2g²) dW
        ========================================
        - drift: ODE drift (velocity/score/noise依存)
        - score: ∇ log p_t(x)
        - g: diffusion 係数 (diffusion_form で選択)

        t: 0 → 1 (last_step_size手前まで)
        最終ステップ: last_step で決定的に処理

        入力 (返り値の関数):
          init: (B, 4, 32, 32) ← 初期ノイズ z ~ N(0, I)
          model: SiTモデル
          **model_kwargs: {"y": labels, "cfg_scale": 4.0}

        出力:
          samples: [(B, 4, 32, 32)] × num_steps
                   samples[-1] が最終結果

        対応: 公式 transport/transport.py L279-L339
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = SDESolver(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self._get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            # xs: [(B, 4, 32, 32)] × (num_steps - 1)

            # 最終ステップ (決定的処理)
            ts = torch.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            # xs: [(B, 4, 32, 32)] × num_steps

            assert len(xs) == num_steps
            return xs

        return _sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        """尤度計算付きODE推論

        ========================================
        変分推論による対数尤度計算
        ========================================
        log p(x) = log p(z_T) - ∫₀ᵀ ∇·f(x_t, t) dt

        ∇·f の計算にはHutchinson推定量を使用:
          ∇·f ≈ ε^T (∂f/∂x) ε    (ε ~ Rademacher)

        逆方向 (データ→ノイズ) のODEを解きながら対数尤度変化を追跡

        入力 (返り値の関数):
          x: (B, 4, 32, 32) ← データ
          model: SiTモデル

        出力:
          logp: (B,)    ← 各サンプルの対数尤度
          z:    (B, 4, 32, 32) ← エンコードされた潜在変数

        対応: 公式 transport/transport.py L383-L440
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x  # tuple の場合 (x, logp) を分離
            # Rademacher ランダムベクトル (±1)
            eps = torch.randint(2, x.size(), dtype=torch.float, device=x.device) * 2 - 1
            t = torch.ones_like(t) * (1 - t)  # 時刻反転

            with torch.enable_grad():
                x.requires_grad = True
                # Hutchinson推定量: ε^T ∇f ε ≈ ∇·f
                grad = torch.autograd.grad(
                    torch.sum(self.drift(x, t, model, **model_kwargs) * eps), x
                )[0]
                logp_grad = torch.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                # logp_grad: (B,)

                drift = self.drift(x, t, model, **model_kwargs)
                # drift: (B, 4, 32, 32)

            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ODESolver(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = torch.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            # input: tuple((B, 4, 32, 32), (B,))

            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            # drift:      (num_steps, B, 4, 32, 32)
            # delta_logp: (num_steps, B)

            drift, delta_logp = drift[-1], delta_logp[-1]
            # drift: (B, 4, 32, 32) ← 最終時刻のz (ノイズ側)
            # delta_logp: (B,)

            prior_logp = self.transport.prior_logp(drift)
            # prior_logp: (B,) = log N(z; 0, I)

            logp = prior_logp - delta_logp
            # logp: (B,) = log p(x) = log p(z_T) - ∫∇·f dt

            return logp, drift

        return _sample_fn


# ============================================================
# ファクトリ関数
# ============================================================

def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str = None,
    train_eps: float = None,
    sample_eps: float = None,
) -> Transport:
    """Transport オブジェクトのファクトリ

    ========================================
    デフォルト設定
    ========================================
    path_type:  "Linear"    → ICPlan (= Flow Matching)
    prediction: "velocity"  → ModelType.VELOCITY
    loss_weight: None       → WeightType.NONE

    ========================================
    ε の自動設定
    ========================================
    velocity + Linear/GVP:    ε = 0     (全域安定)
    score/noise + Linear/GVP: ε = 1e-3  (端点回避)
    VP:                       train_ε = 1e-5, sample_ε = 1e-3

    対応: 公式 transport/__init__.py L3-L65
    """
    # 予測モード
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    # 損失重み
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    # パスタイプ
    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_type = path_choice[path_type]

    # ε の自動設定 (数値安定性)
    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR]
          and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        # velocity + Linear/GVP: 端点で安定 → ε=0
        train_eps = 0
        sample_eps = 0

    return Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )


# ============================================================
# メイン (確認用)
# ============================================================

if __name__ == "__main__":
    print("=== SiT Transport Framework ===")
    print()

    # パス設計の比較
    print("1. パス設計 (t=0: ノイズ, t=1: データ)")
    print("-" * 60)

    plans = {
        "Linear (ICPlan)": ICPlan(),
        "GVP (Cosine)":    GVPCPlan(),
        "VP":              VPCPlan(),
    }

    for name, plan in plans.items():
        print(f"\n  {name}:")
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.tensor([t_val])
            alpha, _ = plan.compute_alpha_t(t)
            sigma, _ = plan.compute_sigma_t(t)
            print(f"    t={t_val:.2f}: α={alpha.item():.4f}, σ={sigma.item():.4f}")

    print()
    print("2. 予測モードと推論の対応")
    print("-" * 60)
    print("  velocity + ODE: dx/dt = model(x, t)        ← 最もシンプル")
    print("  score    + ODE: dx/dt = -f + g² × model(x, t)")
    print("  noise    + ODE: dx/dt = -f + g² × (model(x, t)/-σ)")
    print("  any      + SDE: dx = [drift + g²×score]dt + √(2g²)dW")

    print()
    print("3. ε の自動設定")
    print("-" * 60)
    for path in ["Linear", "GVP", "VP"]:
        for pred in ["velocity", "score", "noise"]:
            t = create_transport(path, pred)
            print(f"  {path:7s} + {pred:8s}: train_ε={t.train_eps}, sample_ε={t.sample_eps}")

    print()
    print("4. Linear path velocity の場合の学習")
    print("-" * 60)
    print("  x_t = t × x_1 + (1-t) × x_0")
    print("  u_t = x_1 - x_0")
    print("  loss = ||model(x_t, t) - (x_1 - x_0)||²")
    print("  → Optimal Transport CFM / Flow Matching と完全一致")
