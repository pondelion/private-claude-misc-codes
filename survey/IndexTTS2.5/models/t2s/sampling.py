"""自己回帰生成のscore処理とbeam samplingをPyTorchで実装する。

公式が固定するTransformers 4.52.1の順序に合わせ、log-softmax、repetition
penalty、temperature/top-k/top-p、beam累積scoreの順に処理する。
外部GenerationMixin/LogitsProcessorは呼ばない。
K=beam数、V=音声semantic tokenの語彙数、P=本文と条件を含むprefix長、A=生成済み位置数。
"""
from __future__ import annotations

from typing import Callable

from einops import rearrange
from torch.nn import functional as F
import torch


def process_scores(logits: torch.Tensor, history: torch.Tensor, penalty: float,
                   temperature: float, top_k: int, top_p: float,
                   minimum_keep: int) -> torch.Tensor:
    """logits[K,V]を履歴[K,P+A]に基づき処理し、対数score[K,V]を返す。

    Args:
        logits: 次tokenの未正規化score[K,V]。Kはbeam数、Vは語彙数。
        history: 生成履歴[K,P+A]。Kはbeam数、Pは開始列長、Aは生成済み長。
        penalty: 既出tokenのscoreへ掛ける繰り返し抑制係数。
        temperature: samplingまたは初期ノイズの温度係数。
        top_k: 候補語彙をscore上位何個へ制限するか。0では無効。
        top_p: 累積確率による候補語彙の打ち切り閾値。
        minimum_keep: 語彙候補を絞っても残す最小個数。

    Returns:
        処理済みの対数score[K,V]。Kはbeam数、Vは語彙数。禁止した候補は負の無限大。
    """
    if penalty <= 0 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    # 1. beam生成は最初にlog-softmaxを取り、その値へrepetition penaltyを適用。
    scores = F.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(scores, 1, history)
    selected = torch.where(selected < 0, selected * penalty,
                           selected / penalty)
    scores = scores.scatter(1, history, selected) / temperature
    # 2. 上位k候補とtop-pの確率質量を残す。除外候補は-inf。
    if top_k > 0:
        keep = min(max(top_k, minimum_keep), scores.size(-1))
        threshold = torch.topk(scores, keep, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        sorted_scores, order = torch.sort(scores, descending=False, dim=-1)
        remove = sorted_scores.softmax(-1).cumsum(-1) <= 1 - top_p
        remove[:, -minimum_keep:] = False
        scores = scores.masked_fill(remove.scatter(1, order, remove),
                                    -torch.inf)
    return scores


def candidate_score(candidate: tuple[float, torch.Tensor]) -> float:
    """完了候補の長さ正規化済みscoreを返す。

    Tensorの値は変更しない。

    Args:
        candidate: 累積scoreと生成token列[A]の組。Aは生成済みのtoken数。

    Returns:
        長さpenaltyで正規化した候補の累積score。
    """
    return candidate[0]


def beam_sample(
    step: Callable[[torch.Tensor], torch.Tensor],
    initial_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
    beams: int = 3,
    temperature: float = 0.8,
    top_k: int = 30,
    top_p: float = 0.8,
    repetition_penalty: float = 10.0,
    length_penalty: float = 0.0,
    do_sample: bool = True,
) -> torch.Tensor:
    """stepでlogitsを求め、単一発話の生成ID[1,A]を返す。

    initial_ids[1,P]は公式と同じfake prefix/BOS履歴。EOSを実際に生成した場合だけそのEOSを含める。
    上限到達時に架空のEOSを追加しない。

    Args:
        step: 履歴[K,P+A]から次token logits[K,V]を返す関数。Kはbeam数、Pは開始列長、Aは生成済み長、
            Vは語彙数。
        initial_ids: 開始token列[1,P]。1は単一発話、Pは開始列長。内部でbeam数Kへ展開する。
        eos_id: 生成を終了するtoken ID。
        max_new_tokens: 開始列の後へ追加するtoken数の上限。
        beams: beam数。
        temperature: samplingまたは初期ノイズの温度係数。
        top_k: 候補語彙をscore上位何個へ制限するか。0では無効。
        top_p: 累積確率による候補語彙の打ち切り閾値。
        repetition_penalty: 既出tokenのscoreへ掛ける繰り返し抑制係数。
        length_penalty: 候補scoreを長さで正規化する指数。
        do_sample: 確率samplingを使うか。Falseではscore上位を使う。

    Returns:
        最良候補のtoken ID[1,A]。1は単一候補、Aは開始列を除いた生成長。実際に生成したEOSは含み得る。
    """
    if initial_ids.size(0) != 1 or beams < 1 or max_new_tokens < 1:
        raise ValueError(
            "Expected one utterance and positive beam/token limits")
    prefix = initial_ids.size(1)
    history = initial_ids.expand(beams, -1).clone()
    scores = torch.zeros(beams, device=initial_ids.device)
    scores[1:] = -1e9
    finished: list[tuple[float, torch.Tensor]] = []
    for iteration in range(max_new_tokens):
        # 1. 各beamの次token score[K,V]へ過去累積score[K,1]を足す。
        logits = step(history)
        if do_sample:
            processed = process_scores(logits, history, repetition_penalty,
                                       temperature, top_k, top_p,
                                       2 if beams > 1 else 1)
        else:
            processed = process_scores(logits, history, repetition_penalty,
                                       1.0, 0, 1.0, 1)
        ranked = processed + scores[:, None]
        vocab = ranked.size(1)
        flat = rearrange(ranked, "k v -> 1 (k v)")
        count = min(2 * beams, flat.size(1))
        # 2. 4.52.1のbeam samplingと同じく、sample順をここでは並べ替えない。
        if do_sample:
            indices = torch.multinomial(flat.softmax(-1), count)[0]
            values = flat[0, indices]
        else:
            values, indices = torch.topk(flat[0], count)
        parents = torch.div(indices, vocab, rounding_mode="floor")
        tokens = indices % vocab
        candidates = torch.cat((history[parents], tokens[:, None]), dim=1)
        hit_eos = tokens == eos_id
        stopped = hit_eos | (iteration + 1 == max_new_tokens)
        # 3. 先頭K候補の完了列を既存の完了beamへ統合する。
        # sampleした2K候補全体を先にsortすると、公式の完了候補選択と異なる。
        generated_length = candidates.size(1) - prefix
        for index in range(min(beams, count)):
            if bool(stopped[index]):
                value = values[index] / generated_length**length_penalty
                finished.append((float(value), candidates[index,
                                                          prefix:].clone()))
        finished.sort(key=candidate_score, reverse=True)
        finished = finished[:beams]
        # 4. 完了候補を大きな負値で除外し、継続する上位K beamを選択する。
        running = values + stopped.to(values.dtype) * -1e9
        scores, order = torch.topk(running, beams)
        history = candidates[order]
        if iteration + 1 == max_new_tokens:
            break
        if len(finished) >= beams:
            upper_bound = float(scores.max() /
                                generated_length**length_penalty)
            if finished[-1][0] >= upper_bound:
                break
    if not finished:
        raise RuntimeError("Beam search produced no completed candidate")
    return finished[0][1][None]
