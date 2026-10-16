"""TDT forward algorithm、CTC、Hybrid loss。

教材用 TDT DP は全 alignment を厳密に log-sum-exp する。公式 NeMo は同じ目的関数を
Numba CUDA kernel で計算し、さらに sigma under-normalization と omega RNNT sampling を持つ。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def tdt_loss(token_logits, duration_logits, targets, input_lengths, target_lengths,
             durations=(0,1,2,3,4), blank_id=None, reduction="mean"):
    """Differentiable TDT negative log likelihood.

    Shapes: token logits ``(B,T,U+1,V+1)``, duration logits ``(B,T,U+1,K)``,
    targets ``(B,U)``, lengths ``(B,)``. A label may advance time by d>=0;
    blank must advance by d>0. This is the central TDT/RNN-T difference.
    """
    # tokenとdurationは独立なcategorical分布として正規化し、遷移確率は
    # log p(token|t,u) + log p(duration|t,u) として加算する。
    blank_id = token_logits.shape[-1] - 1 if blank_id is None else blank_id
    token_lp, duration_lp = token_logits.log_softmax(-1), duration_logits.log_softmax(-1)
    losses = []
    for b in range(token_logits.shape[0]):
        # padding領域をDPへ入れないようsample固有のT/Uだけを走査する。
        T, U = int(input_lengths[b]), int(target_lengths[b])
        # alpha[t][u] is log probability of reaching an ordinary lattice node.
        # As in NeMo's kernel, acceptance always uses a final blank transition.
        neg_inf = token_logits.new_tensor(float("-inf"))
        alpha = [[neg_inf for _ in range(U+1)] for _ in range(T+1)]
        alpha[0][0] = token_logits.new_zeros(())
        # tとuが増える方向だけのDAGなので、昇順走査でforward probabilityを確定できる。
        for t in range(T+1):
            for u in range(U+1):
                if t == T or torch.isneginf(alpha[t][u]).item(): continue
                # Emit next target and optionally jump d frames.
                if u < U:
                    # 正解系列以外のlabel遷移はlossのalignment集合に含めない。
                    label = int(targets[b, u])
                    for k, d in enumerate(durations):
                        if t+d <= T:
                            score = alpha[t][u] + token_lp[b,t,u,label] + duration_lp[b,t,u,k]
                            alpha[t+d][u+1] = torch.logaddexp(alpha[t+d][u+1], score)
                # Blank consumes at least one frame and no label.
                for k, d in enumerate(durations):
                    if d > 0 and t+d <= T:
                        score = alpha[t][u] + token_lp[b,t,u,blank_id] + duration_lp[b,t,u,k]
                        alpha[t+d][u] = torch.logaddexp(alpha[t+d][u], score)
        # NeMo CUDA kernelと同様、(T,U)へ直接到達した値ではなく、
        # duration付きのfinal blankで音声末尾Tへ到達する経路だけを受理する。
        final = neg_inf
        for k, d in enumerate(durations):
            if d > 0 and T-d >= 0:
                score = alpha[T-d][U] + token_lp[b,T-d,U,blank_id] + duration_lp[b,T-d,U,k]
                final = torch.logaddexp(final, score)
        losses.append(-final)
    result = torch.stack(losses)
    return result.mean() if reduction == "mean" else result


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_id=None):
    """padding済みtargetを連結し、PyTorch CTCへ渡す。

    入力は``log_probs (B,T,V+1)``、``targets (B,U_max)``、各length ``(B,)``。
    PyTorch APIが要求する時間優先``(T,B,C)``へtransposeし、scalar NLLを返す。
    ``zero_infinity``は入力frameが短すぎるsampleのinf lossを0にしてbatch全体の破綻を防ぐ。
    """
    blank_id = log_probs.shape[-1]-1 if blank_id is None else blank_id
    # padding tokenは正解系列ではないため、lengthで切って1次元へ連結する。
    flat = torch.cat([targets[b, :int(target_lengths[b])] for b in range(targets.shape[0])])
    return F.ctc_loss(log_probs.transpose(0,1), flat, input_lengths, target_lengths,
                      blank=blank_id, reduction="mean", zero_infinity=True)


def hybrid_loss(tdt, ctc, ctc_weight=.3):
    """公式Hybrid objective ``(1-w)*L_TDT + w*L_CTC``。

    2 lossを単純加算せず係数の和を1にするため、weight変更時にもloss scaleが急変しにくい。
    """
    return (1-ctc_weight)*tdt + ctc_weight*ctc
