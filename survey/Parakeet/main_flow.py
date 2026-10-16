"""各教材moduleを接続し、Parakeet Hybrid TDT-CTCの全経路を俯瞰する。

目的は0.6B checkpointの代替ではなく、waveformからloss/decodeまでどのtensorが渡るかを
1 fileで追えるようにすること。``example_usage``はlossを表示するだけでなく、実際にbackwardし、
共有encoderへTDT/CTC双方の勾配が届くことをassertする。
"""
from __future__ import annotations
import torch
from torch import nn
from audio_frontend import LogMelFrontend, SpecAugment
from fastconformer import FastConformerEncoder
from transducer import PredictionNetwork, TDTJoint, CTCHead
from losses import tdt_loss, ctc_loss, hybrid_loss
from decoding import tdt_greedy_decode, ctc_greedy_decode


class ParakeetHybridModel(nn.Module):
    """共有FastConformerへTDT主headとCTC補助headを接続する。

    既定値は理解用の小型構成。論文の0.6B-v3は24層であり、ここでの4層既定値は
    architectureをCPUでも動かすための縮小である。処理そのものをmockにはしていない。
    """
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, pred_hidden=256,
                 durations=(0,1,2,3,4), ctc_weight=.3):
        super().__init__()
        # frontend/encoderは両lossで共有される。したがってCTC lossもencoder表現を更新する。
        self.frontend, self.augment = LogMelFrontend(), SpecAugment()
        self.encoder = FastConformerEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.predictor = PredictionNetwork(vocab_size, pred_hidden)
        self.joint = TDTJoint(d_model, pred_hidden, vocab_size, durations, pred_hidden)
        self.ctc_head, self.ctc_weight = CTCHead(d_model, vocab_size), ctc_weight

    def encode(self, waveform, waveform_lengths):
        """``(B,S)`` waveformを``(B,Te,D)`` encoder表現へ変換する。"""
        features, feature_lengths = self.frontend(waveform, waveform_lengths)  # (B,128,T10ms)
        # SpecAugmentはtrain時だけ。eval/transcribeで確率的maskを掛けてはいけない。
        if self.training:
            features = self.augment(features, feature_lengths)
        return self.encoder(features, feature_lengths)  # (B,T/8,D), (B,)

    def forward(self, waveform, waveform_lengths, targets, target_lengths):
        """teacher forcingでTDT/CTC lossを同時に計算する。

        targetsは``(B,U_max)``、target_lengthsは``(B,)``。返すCTC log probabilityは
        decode/可視化にも利用できるが、TDTの巨大joint logitsはmemory節約のため返さない。
        """
        encoded, encoded_lengths = self.encode(waveform, waveform_lengths)
        # targetの先頭にBOS相当を加え、各uのlabel-history状態を作る。
        prediction = self.predictor(targets)  # (B,U+1,H)
        token_logits, duration_logits = self.joint(encoded, prediction)  # (B,T,U+1,V+1/K)
        ctc_log_probs = self.ctc_head(encoded)  # (B,T,V+1)
        # TDTは(T,U)格子、CTCはT軸だけの別alignment集合を周辺化する。
        lt = tdt_loss(token_logits, duration_logits, targets, encoded_lengths, target_lengths,
                      self.joint.durations, self.joint.blank_id)
        lc = ctc_loss(ctc_log_probs, targets, encoded_lengths, target_lengths, self.joint.blank_id)
        return {"loss": hybrid_loss(lt, lc, self.ctc_weight), "tdt_loss": lt, "ctc_loss": lc,
                "encoded_lengths": encoded_lengths, "ctc_log_probs": ctc_log_probs}

    @torch.no_grad()
    def transcribe_ids(self, waveform, waveform_lengths, decoder="tdt"):
        """同じencoderをTDTまたはCTC greedy decoderで読む。

        一時的にeval modeへ切り替え、dropout/SpecAugmentを止める。呼出前のtrain/eval状態は
        処理後に復元するため、validationをtraining loopへ安全に挿入できる。
        """
        was_training = self.training; self.eval()
        encoded, lengths = self.encode(waveform, waveform_lengths)
        result = (tdt_greedy_decode(encoded, lengths, self.predictor, self.joint) if decoder == "tdt"
                  else ctc_greedy_decode(self.ctc_head(encoded), lengths))
        self.train(was_training)
        return result


def example_usage():
    """小型batchでforward、backward、実decodeを一続きに検証する。"""
    torch.manual_seed(0)
    model = ParakeetHybridModel(vocab_size=32, d_model=64, n_layers=2, n_heads=4, pred_hidden=64)
    # 0.5秒相当の2音声。第2sampleの末尾1600 sampleはpaddingとして扱われる。
    waveform = torch.randn(2, 8000)
    waveform_lengths = torch.tensor([8000, 6400])
    # 第2target末尾の32はpaddingで、target_lengths=2なのでlossには入らない。
    targets = torch.tensor([[1, 2, 3], [4, 5, 32]])
    target_lengths = torch.tensor([3, 2])
    output = model(waveform, waveform_lengths, targets, target_lengths)
    output["loss"].backward()  # frontendから両headまで実際にbackprop
    assert model.encoder.subsampling.out.weight.grad is not None
    print({k: float(v.detach()) for k,v in output.items() if k.endswith("loss")})
    print(model.transcribe_ids(waveform, waveform_lengths, "tdt"))


if __name__ == "__main__": example_usage()
