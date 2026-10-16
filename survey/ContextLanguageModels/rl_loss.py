"""
Context Language Models (CLMs) - Success-Gated Efficiency Advantage for Stepwise GRPO (ロス)
================================================================================================

CLM を強化学習で学習するための損失関数を実装する。通常の GRPO (group-relative
policy optimization) の「成果アドバンテージ」に加え、コンテキスト編集ターンの
トークンだけに効く「効率アドバンテージ」を合成する "dual-channel advantage" を
実装する。

論文: Context Language Models (arXiv:2609.37725)
      - Sec 3.2 "Reinforcement Learning for CLMs"
      - Eq.5: A_i^eff = clip((c̄_g - c_i)/c̄_g, -1, 1) for i in G_g^+ (成功した軌跡のみ), else 0
      - Eq.6: A_i = A_i^out + w_eff * A_i^eff
公式コード: 学習コード自体は非公開 ("RL training for CLM is not part of this
      repository", clm/clm_rl/README.md)。公開されているのは、外部の学習フレーム
      ワーク (ProRL-Agent-Server / Slime, いずれも別リポジトリ) に当てる約70行の
      パッチ (clm_rl/patches/0001-slime_bridge-dual-channel-advantage.patch) のみ。
      本ファイルはそのパッチに書かれているアルゴリズム
      (`custom_advantage_function`) をそのまま、スタンドアロンの PyTorch 実装
      として再構成したものである。

============================================================
Dual-channel advantage (公式パッチのコメントより)
============================================================
A[i, t] = r_i + w_eff * a_eff_i * m_i[t]
  - r_i    : group-normalized task reward (通常のGRPO, 全トークンに均一放送)
  - m_i[t] : トークン t が「コンテキスト編集のみを行うターン」に属するなら1、他は0
  - a_eff_i: グループ内で「タスクに成功した」軌跡だけを対象に計算する効率スコア
             (成功軌跡が2未満のグループは全員 a_eff=0)
  - w_eff  : 効率チャンネルの重み (公式設定: 0.25)

============================================================
Shape Convention
============================================================
G : 1グループ内のロールアウト数 (公式設定: 32)
T_i: 軌跡 i の応答トークン長 (軌跡間で異なる -> パディングして扱う)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# ============================================================
# 1. 成功ゲート付き効率アドバンテージ (Eq.5)
# ============================================================

def success_gated_efficiency_advantage(
    success: Sequence[bool],
    flops: Sequence[float],
) -> torch.Tensor:
    """1ロールアウトグループの a_eff を計算する (論文 Eq.5 / 公式パッチと同一)。

    ========================================
    Shape
    ========================================
    入力: success (G,) bool, flops (G,) float
    出力: a_eff (G,) float32
        - 成功軌跡が2個未満のグループ: 全員0
        - 失敗軌跡: 常に0
        - 成功軌跡: clip((mean_flops - flops_i) / mean_flops, -1, 1)
          (成功軌跡だけの平均FLOPsより安いほど +1 に近づく)
    """
    success_t = torch.tensor(list(success), dtype=torch.bool)
    flops_t = torch.tensor(list(flops), dtype=torch.float32)
    G = success_t.shape[0]
    a_eff = torch.zeros(G, dtype=torch.float32)

    succ_idx = torch.nonzero(success_t, as_tuple=True)[0]
    if succ_idx.numel() < 2:
        return a_eff  # 成功が2未満: 効率信号なし (公式仕様)

    mean_flops = flops_t[succ_idx].mean()
    if mean_flops <= 0:
        return a_eff
    rel = (mean_flops - flops_t[succ_idx]) / mean_flops
    a_eff[succ_idx] = torch.clamp(rel, -1.0, 1.0)
    return a_eff


# ============================================================
# 2. role_mask: どのトークンが「コンテキスト編集のみのターン」に属するか
# ============================================================

def build_role_mask(
    turn_token_lengths: Sequence[int],
    turn_is_pure_context_edit: Sequence[bool],
) -> torch.Tensor:
    """ターンごとの (トークン数, 純粋なコンテキスト編集ターンかどうか) から、
    応答トークン列全体に対する role_mask (1=編集ターン, 0=それ以外) を作る。

    「純粋なコンテキスト編集ターン」とは、そのターンの assistant アクションが
    ミラーファイルへの書き込みのみで、タスク上の出力 (stdout/stderr) が空の
    ターンのこと (公式 harness.py の `ctx_turn_is_free` と同じ判定基準)。

    ========================================
    Shape
    ========================================
    入力: turn_token_lengths (n_turns,), turn_is_pure_context_edit (n_turns,) bool
    出力: role_mask (T,) float32, T = sum(turn_token_lengths)
    """
    parts = []
    for length, is_edit in zip(turn_token_lengths, turn_is_pure_context_edit):
        value = 1.0 if is_edit else 0.0
        parts.append(torch.full((length,), value, dtype=torch.float32))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float32)


# ============================================================
# 3. Dual-channel advantage をトークン列に展開する (公式 custom_advantage_function)
# ============================================================

@dataclass
class TrajectorySample:
    """1ロールアウトの、GRPO学習に必要な最小限の情報。"""

    task_reward: float          # group-normalized task reward r_i (既に正規化済み)
    role_mask: torch.Tensor     # (T_i,) 1=コンテキスト編集ターンのトークン
    response_length: int        # T_i


def dual_channel_returns(
    samples: Sequence[TrajectorySample],
    a_eff: Sequence[float],
    w_eff: float = 0.25,
) -> List[torch.Tensor]:
    """公式 `custom_advantage_function` と同一のロジック: 各サンプルについて
    "task channel (全トークンに r_i を均一放送)" + "efficiency channel
    (role_mask==1のトークンにのみ w_eff*a_eff_i を加算)" を合成する。

    ========================================
    Shape
    ========================================
    出力: List[(T_i,)] (サンプルごとに異なる長さのテンソルのリスト)
    """
    returns = []
    for i, s in enumerate(samples):
        ret = torch.full((s.response_length,), float(s.task_reward), dtype=torch.float32)
        ae = float(a_eff[i])
        if ae != 0.0 and s.role_mask.shape[0] == s.response_length:
            ret = ret + w_eff * ae * s.role_mask
        returns.append(ret)
    return returns


# ============================================================
# 4. GRPO クリップ付きポリシー勾配損失 (トークン平均, 低分散KL)
# ============================================================

def grpo_clipped_loss(
    logprobs_new: torch.Tensor,     # (B, T) パディング済み、現在ポリシーのlogprob
    logprobs_old: torch.Tensor,     # (B, T) ロールアウト時のlogprob
    logprobs_ref: torch.Tensor,     # (B, T) 参照(KL)モデルのlogprob
    advantages: torch.Tensor,       # (B, T) dual_channel_returns をパディングしたもの
    loss_mask: torch.Tensor,        # (B, T) 1=実トークン(パディングでない)
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_coef: float = 0.01,
) -> dict:
    """GRPO の token-mean クリップ付きポリシー勾配損失 + 低分散KL損失。

    論文 clm_rl/README "Training configuration": clipping eps 0.2/0.28, KL 0.01
    (low-variance KL) をそのまま採用。

    ========================================
    Shape
    ========================================
    入力: logprobs_new/old/ref, advantages, loss_mask: (B, T) 全て同形状
    出力:
        loss: scalar
        pg_loss: scalar (クリップ付きポリシー勾配項)
        kl_loss: scalar
    """
    ratio = torch.exp(logprobs_new - logprobs_old)  # (B, T)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
    # advantage>=0 では上側(clip_high)、advantage<0 では下側(clip_low)が働くよう
    # min を取る (PPO/GRPOの標準的な非対称クリップと同じ)
    pg_per_token = -torch.min(unclipped, clipped)  # (B, T)

    # 低分散KL推定 (k3推定量): KL ≈ exp(ref - new) - (ref - new) - 1  (>=0)
    log_ratio_ref = logprobs_ref - logprobs_new
    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0  # (B, T)

    denom = loss_mask.sum().clamp(min=1.0)
    pg_loss = (pg_per_token * loss_mask).sum() / denom
    kl_loss = (kl_per_token * loss_mask).sum() / denom
    loss = pg_loss + kl_coef * kl_loss
    return {"loss": loss, "pg_loss": pg_loss.detach(), "kl_loss": kl_loss.detach()}


# ============================================================
# 使用例: 1グループ(G=4)の合成ロールアウトで dual-channel advantage を検証する
# ============================================================

def example_usage() -> None:
    print("=== rl_loss.py Example Usage ===\n")

    # 4本の軌跡: 2本成功(FLOPs 1.0e14, 1.4e14), 2本失敗
    success = [True, True, False, False]
    flops = [1.0e14, 1.4e14, 0.8e14, 2.0e14]
    a_eff = success_gated_efficiency_advantage(success, flops)
    print(f"success={success}")
    print(f"flops={flops}")
    print(f"a_eff={a_eff.tolist()}")
    print("=> 失敗軌跡は常に0。成功軌跡のみ平均FLOPsとの相対差でスコア付けされる"
          " (安い方=1.0e14 が正、高い方=1.4e14 が負)。\n")

    # role_mask: 各軌跡は [通常ターン, コンテキスト編集ターン, 通常ターン] の3ターン
    # 構成で、各ターン5トークンとする。
    samples = []
    for r, ae_dummy in zip([0.6, 0.6, -0.4, -0.4], a_eff):
        role_mask = build_role_mask(
            turn_token_lengths=[5, 5, 5],
            turn_is_pure_context_edit=[False, True, False],
        )
        samples.append(TrajectorySample(task_reward=r, role_mask=role_mask, response_length=15))

    returns = dual_channel_returns(samples, a_eff, w_eff=0.25)
    print("--- dual_channel_returns (各軌跡15トークン, 中央5トークンが編集ターン) ---")
    for i, ret in enumerate(returns):
        print(f"  trajectory {i}: {ret.tolist()}")
    print(
        "\n=> 成功した軌跡0(安い)は編集ターンの5トークンだけ task_reward(0.6)より"
        " 高いアドバンテージを得て、成功した軌跡1(高い)はそこだけ低くなる。"
        " 失敗した軌跡2,3は a_eff=0 なので全トークン均一のまま。"
    )

    # --- GRPO損失の計算 (ダミーのlogprobsでshapeと勾配の流れだけ確認) ---
    print("\n--- grpo_clipped_loss の動作確認 ---")
    B, T = 4, 15
    torch.manual_seed(0)
    padded_adv = torch.stack(returns)  # (B, T)
    logprobs_old = torch.randn(B, T) * 0.1
    logprobs_new = (logprobs_old + torch.randn(B, T) * 0.01).requires_grad_(True)
    logprobs_ref = logprobs_old + torch.randn(B, T) * 0.01
    loss_mask = torch.ones(B, T)

    out = grpo_clipped_loss(logprobs_new, logprobs_old, logprobs_ref, padded_adv, loss_mask)
    out["loss"].backward()
    print(f"loss={out['loss'].item():.4f}  pg_loss={out['pg_loss'].item():.4f}  "
          f"kl_loss={out['kl_loss'].item():.4f}")
    print(f"logprobs_new.grad is not None: {logprobs_new.grad is not None} "
          f"(勾配ノルム={logprobs_new.grad.norm().item():.4f})")


if __name__ == "__main__":
    example_usage()
