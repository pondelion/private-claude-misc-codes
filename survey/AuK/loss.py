"""
AuK - Flow-Matching損失 (Rectified Flow + 条件ドロップアウト) (ロス)
================================================================

AuKの学習目的は Rectified Flow (線形補間パスの速度場回帰) であり、VAE潜在
z1 (クリーンなターゲット) と z0 (ガウス雑音) の間の線形パス上の速度を予測する。

論文: AuK Technical Report (arXiv:2609.08936)
      - Eq.flow-matching-path: z_t=(1-t)z0+t*z1, v_t=z1-z0
      - Eq.flow-matching-loss: L_FM = ||m ⊙ (v̂_t - v_t)||^2 / sum(m)  (パディング除外マスク付きMSE)
      - "Condition Dropout and Reference Augmentation": 階層的な条件ドロップアウト
        (音響条件を0.3、さらにunconditionalを0.2でドロップ)
      - t = sigmoid(u), u~N(-0.8, 0.8^2) による logit-normal 時刻サンプリング
公式コード: gh/AuK/src/auk/model/cfm_edit.py (`CFMEdit.forward`, `sample_time`)
      これらは全て素のPyTorch実装であり、本ファイルはそのロジックをそのまま
      再構成したものである (本物のモデル(transformer, text_encoder)とは別に、
      「損失計算のオーケストレーション」だけを独立した関数/クラスとして
      切り出している)。

============================================================
Shape Convention
============================================================
B      : バッチサイズ
N      : ターゲット潜在のフレーム数
D_lat  : VAE潜在次元
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. フロー時刻のサンプリング (論文: 一様 or logit-normal)
# ============================================================

def sample_flow_time(
    batch: int, device, dtype,
    mode: str = "logistic_normal", p_mean: float = -0.8, p_std: float = 0.8,
) -> torch.Tensor:
    """flow time t in (0,1) をサンプリングする (公式 CFMEdit.sample_time と同一)。

    ``logistic_normal``: t = sigmoid(u), u ~ N(p_mean, p_std^2)。
    小さいtの近傍 (ノイズに近い状態) に学習を偏らせる (論文の設計意図)。
    """
    if mode == "uniform":
        return torch.rand((batch,), device=device, dtype=dtype)
    if mode == "logistic_normal":
        u = torch.randn((batch,), device=device, dtype=dtype) * p_std + p_mean
        return torch.sigmoid(u)
    raise ValueError(f"unknown t_sampling mode: {mode!r}")


# ============================================================
# 2. 階層的な条件ドロップアウト (論文 "Condition Dropout and Reference Augmentation")
# ============================================================

def sample_condition_dropout(audio_drop_prob: float = 0.3, cond_drop_prob: float = 0.2) -> Tuple[bool, bool]:
    """(drop_audio_cond, drop_text) を論文と同じ階層ルールでサンプリングする。

    1. 確率 audio_drop_prob で音響条件(参照潜在)をドロップ
    2. 確率 cond_drop_prob で "unconditional" を発生させ、その場合は
       音響条件・テキスト条件の両方を強制的にドロップする (1の結果を上書き)
    """
    import random

    drop_audio_cond = random.random() < audio_drop_prob
    if random.random() < cond_drop_prob:
        return True, True  # unconditional: 両方ドロップ
    return drop_audio_cond, False


# ============================================================
# 3. Rectified Flow の学習パスと目的関数
# ============================================================

@dataclass
class FlowMatchingBatch:
    z_t: torch.Tensor          # (B, N, D_lat): 補間された潜在 (モデルの入力)
    target_velocity: torch.Tensor  # (B, N, D_lat): 回帰対象の速度 z1-z0
    time: torch.Tensor         # (B,)


def build_flow_matching_batch(
    z1: torch.Tensor,  # (B, N, D_lat) クリーンなターゲット潜在 (VAEエンコーダ出力, x1)
    time_mode: str = "logistic_normal",
    x0: Optional[torch.Tensor] = None,
) -> FlowMatchingBatch:
    """z1 (クリーン潜在) から学習用の (z_t, v_target, t) を構築する (論文Eq.flow-matching-path)。"""
    B, N, D = z1.shape
    x0 = torch.randn_like(z1) if x0 is None else x0            # (B, N, D_lat): ガウス雑音
    t = sample_flow_time(B, z1.device, z1.dtype, mode=time_mode)  # (B,)
    t_ = t.view(B, 1, 1)
    z_t = (1 - t_) * x0 + t_ * z1                                 # (B, N, D_lat): 線形補間パス
    target_velocity = z1 - x0                                      # (B, N, D_lat): 速度目標 (tに依存しない!)
    return FlowMatchingBatch(z_t=z_t, target_velocity=target_velocity, time=t)


def masked_flow_matching_loss(
    v_pred: torch.Tensor,            # (B, N, D_lat)
    target_velocity: torch.Tensor,    # (B, N, D_lat)
    mask: Optional[torch.Tensor] = None,  # (B, N) bool, True=有効フレーム
) -> torch.Tensor:
    """パディング除外マスク付きMSE (論文Eq.flow-matching-loss)。"""
    diff2 = (v_pred - target_velocity).pow(2)  # (B, N, D_lat)
    if mask is None:
        # 全フレーム有効 = sum(diff2) / (B*N) (論文の定義で m_i が全て1の場合と一致)
        return diff2.sum(dim=-1).mean()
    mask_f = mask.to(diff2.dtype)                    # (B, N): フレーム単位の有効マスク
    masked_sum = (diff2.sum(dim=-1) * mask_f).sum()    # scalar: 有効フレームのみチャンネル方向にsum
    return masked_sum / mask_f.sum().clamp_min(1.0)     # 論文Eq: ||m⊙(v̂-v)||^2 / sum(m_i)


# ============================================================
# 4. 推論時のODE積分 (公式はtorchdiffeqのodeintを使うが、本ファイルは
#    依存を増やさないシンプルな固定ステップEuler法を自前実装する)
# ============================================================

@torch.no_grad()
def euler_ode_sample(
    velocity_fn,       # Callable[[t: float, z: Tensor], Tensor] -> 予測速度
    shape: Tuple[int, int, int],
    steps: int = 32,
    device="cpu", dtype=torch.float32,
) -> torch.Tensor:
    """z0~N(0,I) から z1 までEuler法でフロー常微分方程式を積分する (推論時サンプリング)。

    ========================================
    式
    ========================================
    dz/dt = v_theta(z, t),  z(0)=z0 ~ N(0,I)
    Euler: z_{k+1} = z_k + v_theta(z_k, t_k) * dt,  dt = 1/steps
    """
    z = torch.randn(shape, device=device, dtype=dtype)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((shape[0],), k * dt, device=device, dtype=dtype)
        v = velocity_fn(t, z)
        z = z + v * dt
    return z


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== loss.py Example Usage ===\n")
    torch.manual_seed(0)

    B, N, D_lat = 4, 20, 16
    z1 = torch.randn(B, N, D_lat)  # VAEエンコーダ出力(クリーン潜在)のダミー

    batch = build_flow_matching_batch(z1, time_mode="logistic_normal")
    print(f"z1 (クリーン潜在) shape: {tuple(z1.shape)}")
    print(f"z_t (補間潜在) shape: {tuple(batch.z_t.shape)}")
    print(f"target_velocity shape: {tuple(batch.target_velocity.shape)}")
    print(f"サンプルされたt (B,): {batch.time.tolist()}")

    # ダミーのモデル出力でマスク付きlossを確認 (末尾5フレームをパディングとしてマスク)
    v_pred = torch.randn(B, N, D_lat, requires_grad=True)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -5:] = False
    loss_masked = masked_flow_matching_loss(v_pred, batch.target_velocity, mask)
    loss_full = masked_flow_matching_loss(v_pred, batch.target_velocity, None)
    print(f"\nmask付きloss: {loss_masked.item():.4f}  (末尾5フレームを除外)")
    print(f"mask無しloss: {loss_full.item():.4f}")
    loss_masked.backward()
    print(f"v_predへの勾度ノルム: {v_pred.grad.norm().item():.4f}")

    print("\n--- 条件ドロップアウトの頻度を実際にサンプリングして確認 ---")
    n_trials = 20000
    counts = {"both": 0, "audio_only": 0, "none": 0}
    for _ in range(n_trials):
        drop_audio, drop_text = sample_condition_dropout()
        if drop_text:
            counts["both"] += 1
        elif drop_audio:
            counts["audio_only"] += 1
        else:
            counts["none"] += 1
    for k, v in counts.items():
        print(f"  {k:10s}: {v / n_trials:.1%}")
    print("=> 'both'(unconditional)は約20%、'audio_only'は約(0.3*0.8)=24%、'none'は約56%になるはず")

    print("\n--- Euler ODE積分によるサンプリング (ダミーの速度場) ---")
    def dummy_velocity_fn(t, z):
        return -z * 0.1  # 原点に収縮するだけの適当な速度場 (形状確認用)
    z_final = euler_ode_sample(dummy_velocity_fn, (2, N, D_lat), steps=10)
    print(f"Euler積分後のサンプル shape: {tuple(z_final.shape)}")


if __name__ == "__main__":
    example_usage()
