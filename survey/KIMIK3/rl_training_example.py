"""
Kimi K3 強化学習 (RL) 簡易学習サンプルスクリプト -- 論文 §4.1 "Reinforcement Learning" の実装。

============================================================
このスクリプトが再現する仕組み
============================================================
§4.1 "Algorithm" 段落 (post-training §4.1):
    "we sample K completions for each of N prompts, maintaining an active workload
     of N x K trajectories. ... Once all K responses for a prompt complete, they are
     immediately dispatched for policy optimization, which follows the algorithm in
     Kimi K2.5. ... Our policy optimization algorithm inherently tolerates such an
     extreme off-policy regime through a per-token regularization. By constraining
     policy updates within a localized neighborhood, this regularization enables the
     algorithm to robustly handle highly stale data and sustains training stability."

再現する具体的な仕組み:
    1. **グループサンプリング**: N個のプロンプトそれぞれに対しK個の応答をサンプルする
       ("sample K completions for each of N prompts")
    2. **予算制御報酬**: 予算超過トラジェクトリの報酬を -1 に上書き (§4.1 "Reasoning Effort RL",
       loss_computation.reasoning_effort_budget_reward を再利用)
    3. **per-token 正則化によるオフポリシー耐性**: 「ローカルな近傍に更新を制約する」正則化
       ("per-token regularization... constraining policy updates within a localized neighborhood")

============================================================
簡略化した点 (正当な理由あり)
============================================================
Kimi K3 の正確なポリシー最適化アルゴリズムは "follows the algorithm in Kimi K2.5" と
明記されている通り別論文 (Kimi K2.5) に委譲されており、本タスクで参照可能な Kimi K3 の
tex ソースにはその具体的な更新式は含まれていない。そのため本スクリプトは、上記の性質
(グループ相対的な優位度、per-tokenの近傍制約によるオフポリシー耐性) を満たす
**GRPO/PPO系のclipped surrogate**という広く使われる標準的な定式化で代替する:

    advantage_{i,k} = (r_{i,k} - mean_k(r_i)) / (std_k(r_i) + eps)     # グループ相対的優位度
    ratio_t         = exp(logp_new(y_t) - logp_old(y_t))               # per-token 重要度比
    loss_t          = -min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A)  # ローカル近傍制約

タスク環境も、実際のエージェント環境 (§4.1.2 "Verifiable Problems in Agentic Environments",
ソフトウェア開発やツール利用を伴う複雑なタスク) を再現する代わりに、決定的に検証可能な
極小タスク ("ECHO:xx" というプロンプトに対し xx をそのまま出力できたら正解) を使う。
これは「検証可能な報酬」という設計思想 (deterministic verifier) 自体は保っている。

============================================================
SFT コールドスタートについて
============================================================
§4.1 "Method": "Our post-training pipeline follows a three-stage paradigm: initializing
baseline agent capabilities via supervised fine-tuning (SFT), developing specialized domain
experts ... via Reinforcement Learning (RL), and consolidating ... " とある通り、RLは
SFTで初期化された方策から始まる (§4.1 "Reinforcement Learning": "While SFT provides a solid
cold-start foundation, RL is critical to unlocking higher-order reasoning...")。

ランダム初期化のまま RL だけを回すと、極小タスクであっても K サンプル全てが報酬0のまま
グループ内分散が0になり (`group_relative_advantage` の分母が常にeps) 学習信号が得られない。
これは実装のバグではなく「SFTなしのRLはコールドスタート問題を起こす」という論文の主張
そのものを裏付ける挙動である。そのため本スクリプトは `finetuning_example.train()` を
そのまま再利用して軽いSFTコールドスタートを行ってからRLに入る (`cold_start_sft` 関数)。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import pandas as pd

from finetuning_example import ByteLevelTokenizer, build_model_and_tokenizer
from finetuning_example import train as sft_train
from loss_computation import reasoning_effort_budget_reward
from main_flow import KimiK3ForConditionalGeneration


# ============================================================
# 1. 検証可能タスク環境 (§4.1.2 "Verifiable Problems in Agentic Environments" の簡略版)
# ============================================================
ECHO_ALPHABET = "ABCDEFGH"
ECHO_CODE_LEN = 1  # 1文字echo (8択)。トイスケールモデルがSFTで部分的にしか解けない難易度に調整


@dataclass
class EchoTask:
    prompt: str        # 例: "ECHO:AB"
    target: str        # 例: "AB"  (モデルはこれをそのまま出力すれば正解)


def sample_echo_task() -> EchoTask:
    code = "".join(random.choice(ECHO_ALPHABET) for _ in range(ECHO_CODE_LEN))
    return EchoTask(prompt=f"ECHO:{code}", target=code)


def verify_echo(generated_text: str, task: EchoTask) -> float:
    """決定的verifier: 生成文字列が target と完全一致すれば 1.0、そうでなければ 0.0。"""
    return 1.0 if generated_text == task.target else 0.0


# ============================================================
# 2. Rollout (グループサンプリング)
# ============================================================
def sample_response(
    model: KimiK3ForConditionalGeneration,
    tokenizer: ByteLevelTokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """自己回帰的に応答をサンプリングする (KVキャッシュ未実装のため毎ステップ全体を forward)。

    Args:
        prompt_ids: (1, T_prompt) int64
    Returns:
        full_ids:      (1, T_prompt + T_gen) int64  プロンプト+生成トークン
        old_logprobs:  (T_gen,)  各生成トークンをサンプルした時点の対数尤度 (勾配なし)
    """
    model.eval()
    ids = prompt_ids.clone()
    logprobs = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids)  # (1, T, vocab_size)
            next_logits = logits[0, -1] / temperature  # (vocab_size,)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)
            logp = torch.log(probs[next_token] + 1e-12)
            logprobs.append(logp)
            ids = torch.cat([ids, next_token.view(1, 1)], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    model.train()
    return ids, torch.cat(logprobs)


def compute_response_logprobs(
    model: KimiK3ForConditionalGeneration,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """teacher-forcing で応答部分の対数尤度を再計算する (勾配あり、GRPO/PPOの logp_new に相当)。

    Args:
        full_ids: (1, T_prompt + T_gen)
    Returns:
        (T_gen,)  応答部分 (プロンプトを除く) 各トークンの対数尤度
    """
    logits = model(full_ids)  # (1, T, vocab_size)
    log_probs_all = F.log_softmax(logits[0, prompt_len - 1:-1], dim=-1)  # (T_gen, vocab_size)
    target_ids = full_ids[0, prompt_len:]  # (T_gen,)
    return log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (T_gen,)


# ============================================================
# 3. グループ相対的優位度 (GRPO風) と clipped surrogate loss
# ============================================================
def group_relative_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        rewards: (K,)  同一プロンプトに対する K 個の応答の報酬
    Returns:
        (K,)  グループ内平均を引き、グループ内標準偏差で正規化した優位度
    """
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def clipped_policy_gradient_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """per-token の重要度比をクリップした方策勾配損失 (PPO/GRPO系のclipped surrogate)。

    「ローカルな近傍に更新を制約する」正則化 (§4.1 "Algorithm") に対応し、
    オンポリシーからのずれ (rollout 時点とパラメータ更新後のポリシーの差) が
    大きくなりすぎないようにする。

    Args:
        new_logprobs, old_logprobs: (T_gen,)  現在方策 / rollout時点の方策での対数尤度
        advantage: グループ相対的優位度 (このトラジェクトリ全体で共通のスカラー)
        clip_eps: クリップ幅
    Returns:
        スカラー損失 (このトラジェクトリの平均、符号は「最小化」方向)
    """
    ratio = torch.exp(new_logprobs - old_logprobs.detach())  # (T_gen,)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.minimum(surr1, surr2).mean()


# ============================================================
# 4. メイン RL 訓練ループ
# ============================================================
def train_rl(
    model: KimiK3ForConditionalGeneration,
    tokenizer: ByteLevelTokenizer,
    num_iterations: int = 40,
    num_prompts: int = 4,       # N: イテレーションあたりのプロンプト数
    group_size: int = 6,        # K: プロンプトあたりのサンプル数
    max_new_tokens: int = 4,
    temperature: float = 1.0,
    lr: float = 1e-3,
    tau: float = 1.5,           # §4.1 "Reasoning Effort RL" の予算倍率
    clip_eps: float = 0.2,
    log_every: int = 5,
):
    """
    ========================================
    1イテレーションの流れ (§4.1 "Algorithm" に対応)
    ========================================
    for N個のプロンプト:
        for K個のサンプル:                          # "K completions for each of N prompts"
            rollout: 応答をサンプリング (勾配なし)      -> old_logprobs, task_reward
            budget-clipped reward の計算              -> reasoning_effort_budget_reward
        グループ相対的優位度の計算                      -> group_relative_advantage
        for K個のサンプル:
            teacher-forcing で new_logprobs を再計算 (勾配あり)
            clipped surrogate loss を計算・蓄積
    -> 全 N*K トラジェクトリ分の損失を平均して1回だけ backward + optimizer.step()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    initial_budget_value = float(ECHO_CODE_LEN + 1)  # コード長 + EOS 1トークン分
    reward_history: list[float] = []

    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device)
        n_trajectories = 0
        iter_rewards = []

        for _ in range(num_prompts):
            task = sample_echo_task()
            prompt_ids = torch.tensor(
                [tokenizer.bos_id] + tokenizer.encode(task.prompt), dtype=torch.long
            ).unsqueeze(0).to(device)
            prompt_len = prompt_ids.shape[1]

            # --- (a) グループロールアウト: K個サンプリングして報酬を計算 ---
            group_full_ids, group_old_logprobs, group_rewards = [], [], []
            for _ in range(group_size):
                full_ids, old_logprobs = sample_response(
                    model, tokenizer, prompt_ids, max_new_tokens, temperature, device
                )
                gen_ids = full_ids[0, prompt_len:]
                gen_text = tokenizer.decode(
                    [i for i in gen_ids.tolist() if i != tokenizer.eos_id]
                )
                task_reward = verify_echo(gen_text, task)

                token_budget = torch.tensor([float(gen_ids.shape[0])])
                initial_budget = torch.tensor([initial_budget_value])
                reward = reasoning_effort_budget_reward(
                    torch.tensor([task_reward]), token_budget, initial_budget, tau=tau
                ).item()

                group_full_ids.append(full_ids)
                group_old_logprobs.append(old_logprobs)
                group_rewards.append(reward)
                iter_rewards.append(reward)

            # --- (b) グループ相対的優位度 ---
            advantages = group_relative_advantage(torch.tensor(group_rewards))

            # --- (c) teacher-forcing で new_logprobs を再計算し、clipped surrogate loss を蓄積 ---
            for full_ids, old_logprobs, adv in zip(group_full_ids, group_old_logprobs, advantages):
                if old_logprobs.numel() == 0:
                    continue  # 1トークン目でEOSが出た退化ケースはスキップ
                new_logprobs = compute_response_logprobs(model, full_ids, prompt_len)
                loss = clipped_policy_gradient_loss(new_logprobs, old_logprobs, adv.item(), clip_eps)
                total_loss = total_loss + loss
                n_trajectories += 1

        (total_loss / max(n_trajectories, 1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_reward = sum(iter_rewards) / len(iter_rewards)
        reward_history.append(avg_reward)
        if iteration % log_every == 0 or iteration == 1:
            print(f"  iter {iteration:3d}/{num_iterations} | avg reward (N*K={len(iter_rewards)}) = {avg_reward:.3f}")

    return reward_history


# ============================================================
# 5. SFT コールドスタート (§4.1 "Method": SFT -> RL -> MOPD の最初の段階)
# ============================================================
def cold_start_sft(model, tokenizer, config, num_examples: int = 24, epochs: int = 3, lr: float = 5e-4):
    """finetuning_example.train() をそのまま再利用し、ECHOタスクの軽いSFTで方策を初期化する。

    意図的に少ないエポック数に留め、SFT単独ではタスクを完全には解けない (reward < 1.0)
    ようにしている。これによりRL段階で改善する余地を残す。
    """
    prompts, responses = [], []
    for _ in range(num_examples):
        task = sample_echo_task()
        prompts.append(task.prompt)
        responses.append(task.target)
    df = pd.DataFrame({"prompt": prompts, "response": responses, "image": [None] * num_examples})

    print(f"--- SFT cold start ({num_examples} examples x {epochs} epochs) ---")
    sft_train(model, tokenizer, config, df, output_dir="/tmp/kimik3-rl-coldstart",
              epochs=epochs, grad_acc=1, lr=lr, log_steps=999999)  # 途中経過ログは抑制


@torch.no_grad()
def evaluate_avg_reward(model, tokenizer, num_samples: int, max_new_tokens: int, temperature: float) -> float:
    """現在の方策でECHOタスクを何回かサンプリングし、平均タスク成功率 (予算ペナルティ抜き) を測る。"""
    device = next(model.parameters()).device
    rewards = []
    for _ in range(num_samples):
        task = sample_echo_task()
        prompt_ids = torch.tensor(
            [tokenizer.bos_id] + tokenizer.encode(task.prompt), dtype=torch.long
        ).unsqueeze(0).to(device)
        full_ids, _ = sample_response(model, tokenizer, prompt_ids, max_new_tokens, temperature, device)
        gen_ids = full_ids[0, prompt_ids.shape[1]:]
        gen_text = tokenizer.decode([i for i in gen_ids.tolist() if i != tokenizer.eos_id])
        rewards.append(verify_echo(gen_text, task))
    return sum(rewards) / len(rewards)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    model, tokenizer, config = build_model_and_tokenizer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"policy model params (toy-scale): {n_params:,}")

    # --- Stage 1: SFT cold start (§4.1 "Method") ---
    # あえて完全収束させず (epochs=5) 、RLで改善する余地を残した「部分的に解けるコールドスタート」にする
    cold_start_sft(model, tokenizer, config, num_examples=40, epochs=5, lr=5e-4)
    reward_after_sft = evaluate_avg_reward(model, tokenizer, num_samples=80, max_new_tokens=2, temperature=1.0)
    print(f"avg task success rate right after SFT (before RL) = {reward_after_sft:.3f}")

    # --- Stage 2: RL (group-relative clipped policy gradient, budget-aware reward) ---
    print("\n--- RL training (ECHO verifiable-copy task) ---")
    reward_history = train_rl(
        model, tokenizer,
        num_iterations=30, num_prompts=4, group_size=6,
        max_new_tokens=2, temperature=1.0, lr=3e-4, tau=1.5,
    )

    print("\nreward history (first 3 / last 3 iterations):")
    print("  first:", [f"{r:.3f}" for r in reward_history[:3]])
    print("  last :", [f"{r:.3f}" for r in reward_history[-3:]])

    early_avg = sum(reward_history[:5]) / 5
    late_avg = sum(reward_history[-5:]) / 5
    print(f"\nSFT直後 = {reward_after_sft:.3f} | RL序盤(iter1-5) = {early_avg:.3f} | RL終盤(last5) = {late_avg:.3f}")
    assert late_avg > early_avg, "RLで平均報酬が向上していない"
    print("rl_training_example OK: SFT cold start -> RL improved the policy further "
          "via group-relative clipped policy gradient")
