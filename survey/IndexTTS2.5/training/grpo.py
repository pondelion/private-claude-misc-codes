"""ASR/speaker実測rewardを受け取るGRPO objective。rewardを捏造しない。

rollout音声の合成・ASR・speaker verificationはそれぞれの実モデルが必要。
論文は細かなclip/KL係数を公開しないため、以下の値は利用者が指定する。

使う段階:
    まずtraining_example.pyで教師あり学習したpolicyを用意する。同じ本文に
    G候補の音声を生成し、実ASRの認識誤りや話者類似度から、良いほど大きい
    rewards[B,G]を呼び出し側で作る。B=本文数、G=本文当たり候補数。
    本ファイルは音声生成やreward計算を行うtrainerではなく、loss部品である。

渡す値:
    現policy/rollout時policy/固定reference policyが実際に選択したtokenへ
    与えたlog確率を各[B,G,U]で渡す。U=token数。語彙方向の全logitsではない。
    現policyのlog_probsだけに勾配を残す。validはEOSまでの有効token mask。
    group内のrewardを中心化・標準化し、改善候補の確率を上げるclip付き
    surrogateとreferenceからのKL penaltyを計算する。

利用:
    loss = grpo_loss(log_probs, old_log_probs, reference_log_probs,
                     rewards, valid)
    loss.backward()を既存optimizerの更新へ組み込む。PyTorchのみを使用。
    戻り値はscalarで、音声品質を保証するrewardモデルは含まれない。
"""
import torch


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    clip: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """実測rewardからgroup標準化advantageを作りscalar GRPO lossを返す。

    各log-probabilityとvalid:[B,G,U]、rewards:[B,G]。B=本文数、G=本文当たり候補数（論文4）、
    U=生成token数。 validは有効tokenのbool mask、clipはpolicy比のclip幅、betaはKL係数。

    Args:
        log_probs: 現在policyのtoken対数確率[B,G,U]。Bは本文数、Gは候補数、Uはtoken数。
        old_log_probs: 更新前policyのtoken対数確率[B,G,U]。Bは本文数、Gは候補数、Uはtoken数。
        reference_log_probs: 参照policyのtoken対数確率[B,G,U]。Bは本文数、Gは候補数、
            Uはtoken数。
        rewards: 候補ごとの実測reward[B,G]。Bは本文数、Gは候補数。
        valid: 損失を計算するtokenのbool mask[B,G,U]。Bは本文数、Gは候補数、Uはtoken数。
        clip: policy比のclip幅。
        beta: 参照policyに対するKL項の係数。

    Returns:
        有効tokenで平均したGRPO損失[]。[]はscalar。
    """
    if rewards.size(1) < 2:
        raise ValueError("GRPO requires at least two candidates per prompt")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    advantages = centered / rewards.std(dim=1, keepdim=True,
                                        unbiased=False).clamp_min(1e-6)
    # [B,G]のadvantageをtoken軸へbroadcastし、候補内の全有効tokenへ配る。
    ratio = torch.exp(log_probs - old_log_probs.detach())
    clipped = ratio.clamp(1 - clip, 1 + clip)
    surrogate = torch.minimum(ratio * advantages.detach()[..., None],
                              clipped * advantages.detach()[..., None])
    difference = reference_log_probs.detach() - log_probs
    kl = difference.exp() - difference - 1  # 非負のKL推定項をtokenごとに計算する。
    token_loss = -surrogate + beta * kl
    # token平均[B,G]を作ってから本文/候補平均 -> scalar。
    return ((token_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
