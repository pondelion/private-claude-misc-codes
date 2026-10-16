"""学習後の確率分布をtoken列へ変換するTDT/CTC greedy decoding。

両者を並べることで、CTCのframe同期処理と、label履歴・durationを持つTDTの違いを示す。
tokenizerによるID→文字列変換はmodel固有なので、このfileはtoken IDを返す。
"""
from __future__ import annotations
import torch


@torch.no_grad()
def tdt_greedy_decode(encoder, lengths, predictor, joint, max_symbols_per_step=10):
    """Return token ids and encoder-frame timestamps for ``encoder (B,T,D)``.

    TDT emits a token and a duration jointly. Duration advances t directly, unlike RNNT's
    one-frame blank loop. ``timestamp`` unit is encoder frame (FastConformerなら約80 ms).
    """
    results = []
    for b in range(encoder.shape[0]):
        # blankをstart symbolとしてprediction LSTMの初期状態g_0を作る。
        t, token, state, ids, stamps = 0, encoder.new_full((1,), predictor.blank_id, dtype=torch.long), None, [], []
        pred, state = predictor.step(token, state)
        same_t = 0
        while t < int(lengths[b]):
            # 現在の音声位置h_tとlabel履歴g_uだけで1つのjoint分布を評価する。
            tok_logits, dur_logits = joint.step(encoder[b:b+1,t], pred[:,-1])
            tok, k = int(tok_logits.argmax(-1)), int(dur_logits.argmax(-1))
            d = joint.durations[k]
            if tok == joint.blank_id:
                # blankは出力列を変えず時間だけ進める。duration=0 blankは禁止なので、
                # modelが誤って選んでも最低1 frame進めて停止性を保証する。
                t += max(1, d); same_t = 0
            else:
                # non-blankは出力へ追加し、次の予測がこのtokenへ条件付くようLSTMを更新する。
                ids.append(tok); stamps.append(t)
                token = encoder.new_tensor([tok], dtype=torch.long)
                pred, state = predictor.step(token, state)
                t += d; same_t += 1
                if d == 0 and same_t >= max_symbols_per_step:
                    # duration=0 labelは同一tで複数tokenを出せるが、未学習modelなどで
                    # 0が続くと無限loopになる。公式decoder同様に上限を設ける。
                    t += 1; same_t = 0
        results.append({"tokens": ids, "encoder_timestamps": stamps})
    return results


@torch.no_grad()
def ctc_greedy_decode(log_probs, lengths):
    """CTC best-path decoding: argmax→連続重複除去→blank除去。

    例えばframe path ``[a,a,blank,a,b,b]`` は``[a,a,b]``になる。blankを挟む同一tokenは
    別tokenとして残る点に注意。入力``(B,T,V+1)``、出力はsampleごとの可変長ID list。
    """
    blank, paths, out = log_probs.shape[-1]-1, log_probs.argmax(-1), []
    for b in range(paths.shape[0]):
        ids, previous = [], None
        for token in paths[b,:int(lengths[b])].tolist():
            if token != previous and token != blank: ids.append(token)
            previous = token
        out.append(ids)
    return out
