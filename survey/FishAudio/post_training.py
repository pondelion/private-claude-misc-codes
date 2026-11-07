"""Fish S2のgroup-centered advantage + Slow/Fast policy objective。

論文に記載されたDr.GRPO型の中心化（標準偏差で割らない）を実装。
必要なrewardは実ASR・quality・speakerモデルの結果を外部から渡す。

前提となる作業:
    教師あり学習後のDual-ARで、同じ本文に対してG候補を生成する。
    実音声を評価し、良いほど大きいreward[B,G]を作る。B=本文数、G=候補数。
    本ファイルは候補生成・ASR採点・optimizer更新を行わず、その間のlossを担う。

使い方:
    Slowで実際に選んだIDのlog確率[B,G,T]、Fastで選んだ各book IDの
    log確率[B,G,T,N]を別々に計算する。T=生成位置数、N=book数。
    全語彙logitsではなく、生成IDでgatherした値を渡す。現policy側だけに
    勾配を残し、固定reference policyの同じ候補に対するlog確率も用意する。
    validは各log確率と同shapeで、有効位置またはbook重みを表す。
    grouped_policy_lossをSlow/Fastそれぞれへ呼び、係数gammaを使って
    loss_slow + gamma*loss_fastをbackwardする。gammaは呼び出し側で決める。

計算内容:
    rewardは本文内で平均を引く。IndexTTS側の実装と異なり標準偏差では割らない。
    良い候補のlog確率を上げ、KL項でreferenceからの変化を抑える。
    位置をmask付き平均し、本文/候補を平均したscalarを返す。PyTorchのみ使用。
"""
import torch


def grouped_policy_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """log_probs:[B,G,T]または[B,G,T,N]、reward:[B,G]、validは同shape。

    Slow/Fastを別々に呼び、loss_slow + gamma*loss_fastで学習する。
    Fastのcodebook重みを変える場合はvalidをfloat重みにして渡せる。B=本文数、G=本文当たり候補数、T=時刻token数、
    N=Fastのbook数。ref_log_probsはlog_probsと同shape、betaはKL係数、戻り値はscalar loss。

    Args:
        log_probs: 現在policyの対数確率[B,G,T]または[B,G,T,N]。Bは本文数、Gは候補数、Tは時刻数、
            Nはbook数。
        ref_log_probs: 参照policyのtoken対数確率。log_probsと同じ[B,G,T]または[B,G,T,N]。
            Bは本文数、Gは候補数、Tは時刻数、Nはbook数。
        rewards: 候補ごとの実測reward[B,G]。Bは本文数、Gは候補数。
        valid: 有効tokenの重み。log_probsと同shapeの[B,G,T]または[B,G,T,N]。Bは本文数、Gは候補数、
            Tは時刻数、Nはbook数。
        beta: 参照policyに対するKL項の係数。

    Returns:
        候補間の相対rewardに基づくpolicy損失[]。[]はscalar。
    """
    advantage = (rewards - rewards.mean(dim=1, keepdim=True)).detach()
    while advantage.ndim < log_probs.ndim:
        # Slow:[B,G]->[B,G,1]、Fast:[B,G]->[B,G,1,1]として位置軸へ配る。
        advantage = advantage.unsqueeze(-1)
    difference = ref_log_probs.detach() - log_probs
    kl = difference.exp() - difference - 1
    per_token = -advantage * log_probs + beta * kl
    reduce_dims = tuple(range(2, per_token.ndim))
    # 長いsampleのtoken数が大きいだけでpolicy寄与が増えないようsample平均。
    return ((per_token * valid).sum(reduce_dims) /
            valid.sum(reduce_dims).clamp_min(1)).mean()
