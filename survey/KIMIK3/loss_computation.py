"""Kimi K3 の Post-Training で使われる報酬・損失関数 -- 論文 §4 (Post-Training) の実装。

対象となる3つの仕組み:
    1. Reasoning Effort RL の予算制御報酬 (§4.1, sec:post-rl "Reasoning Effort RL" 段落)
       トラジェクトリのトークン消費量 T(y) が閾値 tau*b0(x) を超えると報酬を -1 に上書きする。
       同様の仕組みは Agentic GRM の冗長性抑制 (§4.1 "Agentic Generative Reward Model" 段落)
       にも使われる (出力長が sigma*l0 を超えると二値比較で自動的に負ける)。
    2. Multi-Teacher On-Policy Distillation (MOPD) の per-token 報酬 (§4.1.3, Eq.mopd の式)
       9個 (3ドメイン x 3 reasoning-effort) の教師モデルから単一モデルへ蒸留する。
    3. EAGLE-3 スタイルの draft モデル (MTP層) を投機的デコーディング用に fine-tune する
       LK損失 (§4.1.4, sec:post-eagle3, Eq: L_LK)。

NOTE: MXFP4 量子化認識学習 (QAT, §4.1.4 sec:mxfp4-qat) はここでは実装しない。
これは損失関数ではなく数値表現 (OCP Microscaling仕様) に関するインフラ技術であり、
その正確な仕様 (共有指数のブロックサイズ、E2M1フォーマットの丸め規則等) は
本論文の参照範囲外の外部標準文書 (Rouhani et al. 2023, microscaling) に定義されている。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Reasoning Effort RL: budget-clipped task reward (§4.1 "Reasoning Effort RL")
# ---------------------------------------------------------------------------
def reasoning_effort_budget_reward(
    task_reward: torch.Tensor,
    token_budget: torch.Tensor,
    initial_budget: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """予算超過トラジェクトリの報酬を -1 に上書きする (論文の式は \\iffalse ブロックで
    レンダリングされていないが、直前の本文で同一の規則が明記されている):

        r(x, y) = -1                     if T(y) > tau * b0(x)
                = R_task(y|x)            otherwise

    Args:
        task_reward:    (N,)  各トラジェクトリのタスク報酬 R_task(y|x)
        token_budget:   (N,)  実際に消費したトークン数 T(y)
            (一般タスクでは thinking token 数、エージェントタスクでは
             reasoning + tool-call 引数を含む累積出力トークン数)
        initial_budget: (N,)  コールドスタートモデルから推定した初期予算 b0(x)
        tau: 予算倍率 (max-effort では大きく、low-effort では小さくアニーリングされる)
    Returns:
        reward: (N,)
    """
    exceeded = token_budget > (tau * initial_budget)
    return torch.where(exceeded, torch.full_like(task_reward, -1.0), task_reward)


def agentic_grm_verbosity_gate(
    binary_win: torch.Tensor,
    length_a: torch.Tensor,
    length_b: torch.Tensor,
    initial_length: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Agentic GRM のトーナメント形式の二値比較に、冗長性抑制ルールを適用する。

    「初期の冗長性 l0 と倍率 sigma に対し、出力長が sigma*l0 を超えた候補は
    自動的に二値比較で敗北する」(§4.1 "Agentic Generative Reward Model" 段落末尾)。

    Args:
        binary_win: (N,) bool  GRMが判定した「候補Aが勝った」かどうか (元の判定)
        length_a, length_b: (N,)  候補A, Bの出力長
        initial_length: (N,)  コールドスタートモデルから推定した初期冗長性 l0
        sigma: 冗長性倍率
    Returns:
        adjusted_win: (N,) bool  冗長性ペナルティを適用した後の勝敗
    """
    a_too_long = length_a > (sigma * initial_length)
    b_too_long = length_b > (sigma * initial_length)
    win = binary_win.clone()
    win = torch.where(a_too_long & ~b_too_long, torch.zeros_like(win), win)  # Aが長すぎる -> A敗北
    win = torch.where(b_too_long & ~a_too_long, torch.ones_like(win), win)   # Bが長すぎる -> A勝利
    return win


# ---------------------------------------------------------------------------
# 2. Multi-Teacher On-Policy Distillation (MOPD, §4.1.3)
# ---------------------------------------------------------------------------
def mopd_per_token_reward(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    r_max: float,
) -> torch.Tensor:
    """Eq.(mopd の式, §4.1.3): per-token OPD 報酬。

        r_opd(y_t | e, x, y_<t) = clip( sg( log( pi_teacher(y_t) / pi_student(y_t) ) ), -R_max, R_max )

    teacher は9個 (3ドメイン x 3 reasoning-effort) の専門家モデルのうち、サンプル対象の
    (domain, effort) に対応する1つ。生徒がオンポリシーでサンプルした y に対し、
    教師とのlog確率比をクリップして密な per-token 報酬とする (stop-gradient なので
    勾配はこの報酬自体には流れない、生徒の対数尤度側にのみ流れる)。

    Args:
        teacher_logprobs: (B, T)  教師モデルによる y_t の対数尤度 (勾配不要)
        student_logprobs: (B, T)  生徒モデルによる y_t の対数尤度 (勾配が必要な側)
        r_max: クリップ閾値 R_max > 0
    Returns:
        reward: (B, T)  勾配を持たない (detach 済み) per-token 報酬
    """
    log_ratio = (teacher_logprobs - student_logprobs).detach()  # sg(.) の実体
    return torch.clamp(log_ratio, min=-r_max, max=r_max)


def mopd_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    r_max: float,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MOPD の per-token 報酬を用いたポリシー勾配損失。

    密な報酬 r_opd(y_t) を各トークンの対数尤度に対する優位度 (advantage) として用い、
    REINFORCE型の損失 L = -E[r_opd(y_t) * log pi_student(y_t)] を最小化する。
    論文は「この密な報酬シグナルは我々のRLフレームワークにシームレスに統合され、
    long-horizonタスク向けの partial rollout 訓練のようなインフラ最適化を自然に可能にする」
    (§4.1.3) と述べており、既存の RL 損失 (§4.1 "Algorithm") と同じ勾配推定形を使う。

    Args:
        student_logprobs: (B, T)  勾配を通す生徒モデルの対数尤度
        teacher_logprobs: (B, T)  対応する教師モデルの対数尤度 (勾配不要)
        r_max: クリップ閾値
        response_mask: (B, T) bool、生成部分のみ (prompt/paddingは除外) を対象にする場合に指定
    Returns:
        loss: スカラー
    """
    reward = mopd_per_token_reward(teacher_logprobs, student_logprobs, r_max)  # (B, T), detached
    per_token_loss = -reward * student_logprobs  # (B, T)

    if response_mask is None:
        return per_token_loss.mean()
    mask = response_mask.to(per_token_loss.dtype)
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# 3. EAGLE-3 style draft model fine-tuning: LK loss (§4.1.4, sec:post-eagle3)
# ---------------------------------------------------------------------------
def eagle3_lk_loss(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Eq.(LK損失, §4.1.4): 投機的デコーディングの受理率を直接最大化する損失。

        L_LK = -log sum_{x in V} min(p(x), q(x))

    p, q はそれぞれ target (target model, 凍結) と draft (MTP由来の1層モデル, 学習対象) の
    次トークン分布 (温度1)。通常のKL蒸留と異なり、lossless speculative sampling の
    受理率 sum_x min(p(x),q(x)) の対数を直接最大化する (論文: "minimizing the conventional
    KL-divergence surrogate does not guarantee maximizing this rate")。

    Args:
        draft_logits:  (N, V)  draft モデル (勾配あり) のロジット、温度1
        target_logits: (N, V)  target モデル (勾配なし, 通常 detach 済み) のロジット、温度1
    Returns:
        loss: スカラー (バッチ平均)
    """
    p = F.softmax(target_logits.detach(), dim=-1)  # (N, V) ターゲット分布 (勾配なし)
    q = F.softmax(draft_logits, dim=-1)             # (N, V) ドラフト分布 (勾配あり)
    acceptance_rate = torch.minimum(p, q).sum(dim=-1).clamp_min(1e-8)  # (N,)  sum_x min(p(x),q(x))
    loss = -torch.log(acceptance_rate)  # (N,)
    return loss.mean()


def eagle3_feature_fusion(
    low_level_feat: torch.Tensor,
    mid_level_feat: torch.Tensor,
    high_level_feat: torch.Tensor,
    fusion_weight: torch.Tensor,
) -> torch.Tensor:
    """§4.1.4: draft の入力は target の低・中・高レベル特徴 (AttnRes の 1番目・4番目・
    最終ブロックの出力) を結合し、bias無し行列 W_E3 で射影したもの。W_E3 は
    [0, 0, I] で初期化され、初期状態では高レベル特徴 h_h (MTP層の事前学習入力) と
    一致し、fine-tuning が進むにつれ低・中レベル特徴も取り込むようになる。

    Args:
        low_level_feat, mid_level_feat, high_level_feat: (N, d)
        fusion_weight: (d, 3*d)  W_E3 (呼び出し側で [0,0,I] 初期化を行う)
    Returns:
        (N, d)
    """
    fused_input = torch.cat([low_level_feat, mid_level_feat, high_level_feat], dim=-1)  # (N, 3d)
    return F.linear(fused_input, fusion_weight)  # (N, d)


def init_eagle3_fusion_weight(hidden_size: int) -> torch.Tensor:
    """W_E3 = [0, 0, I] の初期化 (§4.1.4: "initialized as [0 0 I] so that the fused
    representation coincides at initialization with the high-level feature")。"""
    w = torch.zeros(hidden_size, 3 * hidden_size)
    w[:, 2 * hidden_size:] = torch.eye(hidden_size)
    return w


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 8

    # --- 1. Reasoning Effort budget reward ---
    task_reward = torch.rand(N)
    token_budget = torch.tensor([80.0, 150.0, 50.0, 300.0, 90.0, 20.0, 500.0, 60.0])
    initial_budget = torch.full((N,), 100.0)
    reward = reasoning_effort_budget_reward(task_reward, token_budget, initial_budget, tau=1.5)
    print("budget reward:", reward.tolist())
    # token_budget > 150 (=tau*100) のもの (300, 500) は -1 に上書きされているはず
    assert reward[3].item() == -1.0 and reward[6].item() == -1.0
    assert reward[0].item() == task_reward[0].item()

    # --- Agentic GRM verbosity gate ---
    win = torch.tensor([True, False, True])
    len_a = torch.tensor([500.0, 50.0, 30.0])
    len_b = torch.tensor([60.0, 60.0, 40.0])
    l0 = torch.full((3,), 100.0)
    adjusted = agentic_grm_verbosity_gate(win, len_a, len_b, l0, sigma=2.0)
    print("verbosity-adjusted win:", adjusted.tolist())  # 1件目は A=500>200 なので敗北に反転
    assert adjusted[0].item() == 0.0

    # --- 2. MOPD loss: 生徒が教師に近いほど損失(絶対値)は小さくなることを確認 ---
    B, T = 2, 10
    teacher_logprobs = torch.log(torch.rand(B, T) * 0.5 + 0.5)
    student_close = teacher_logprobs + 0.01 * torch.randn(B, T)
    student_far = teacher_logprobs + 2.0 * torch.randn(B, T)
    student_close.requires_grad_(True)
    student_far.requires_grad_(True)

    loss_close = mopd_loss(student_close, teacher_logprobs, r_max=5.0)
    loss_far = mopd_loss(student_far, teacher_logprobs, r_max=5.0)
    print(f"MOPD |loss| (close teacher) = {loss_close.abs().item():.4f}")
    print(f"MOPD |loss| (far teacher)   = {loss_far.abs().item():.4f}")
    loss_close.backward()
    assert student_close.grad is not None

    # --- 3. EAGLE-3 LK loss: draft が target に近いほど損失が小さくなることを確認 ---
    V = 50
    target_logits = torch.randn(N, V) * 3
    draft_close = target_logits + 0.1 * torch.randn(N, V)
    draft_far = torch.randn(N, V) * 3
    lk_close = eagle3_lk_loss(draft_close, target_logits)
    lk_far = eagle3_lk_loss(draft_far, target_logits)
    print(f"LK loss (draft close to target) = {lk_close.item():.4f}")
    print(f"LK loss (draft far from target)  = {lk_far.item():.4f}")
    assert lk_close.item() < lk_far.item(), "draft が target に近いほど LK損失は小さいはず"

    # --- W_E3 初期化: 初期状態で high-level 特徴のみが通ることを確認 ---
    d = 16
    w = init_eagle3_fusion_weight(d)
    low, mid, high = torch.randn(4, d), torch.randn(4, d), torch.randn(4, d)
    fused = eagle3_feature_fusion(low, mid, high, w)
    assert torch.allclose(fused, high), "初期化直後は高レベル特徴とビット一致するはず"

    print("loss_computation OK")
