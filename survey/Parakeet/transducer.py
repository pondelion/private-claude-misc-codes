"""TDTのlabel-history経路、音声とのjoint、補助CTC headを分離して示す。

encoderだけを見るCTCに対し、TDTはencoder状態h_tと、出力済みtokenから作る
prediction状態g_uの組ごとに分布を作る。このため学習時の主要tensorは(T,U)格子になる。
"""
from __future__ import annotations
import torch
from torch import nn


class PredictionNetwork(nn.Module):
    """出力済みtoken列を表現するRNN-T/TDT prediction network。

    学習時は正解token ``(B,U)`` の先頭へblank/BOSを足し、teacher forcingで
    ``(B,U+1,H)``を一括計算する。推論時は新しく確定したtokenを1個ずつ``step``へ渡し、
    LSTM stateを再利用する。音声特徴を入力しない点がencoderとの重要な違い。
    """
    def __init__(self, vocab_size: int, hidden: int = 640):
        super().__init__()
        self.blank_id = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, hidden, padding_idx=self.blank_id)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
    def forward(self, tokens):
        # u=0はまだtokenを1つも出していない状態。NeMoのblank-as-pad構成に合わせ、
        # blank idをstart symbolとして兼用する。
        bos = tokens.new_full((tokens.shape[0], 1), self.blank_id)
        return self.rnn(self.embedding(torch.cat([bos, tokens], 1)))[0]  # (B,U+1,H)
    def step(self, token, state=None):
        """1 token ``(B,)`` と前stateから、次の状態``(B,1,H)``を作る。"""
        return self.rnn(self.embedding(token[:, None]), state)  # (B,1,H), recurrent state


class TDTJoint(nn.Module):
    """Encoder/label pair -> token and duration logits.

    Inputs ``encoder (B,T,D)``, ``prediction (B,U,H)``.
    Outputs ``token_logits (B,T,U,V+1)`` and ``duration_logits (B,T,U,K)``.
    Token blank id is V; duration values (e.g. 0,1,2,3,4) index K.

    ``encoder[:, :, None]``と``prediction[:, None]``のbroadcastにより、全(t,u)の
    組を明示的に作る。これは理解しやすい一方メモリ量がO(B*T*U*joint_dim)なので、
    実際のNeMo学習はfused sub-batch計算を利用できる。
    """
    def __init__(self, encoder_dim: int, pred_dim: int, vocab_size: int, durations=(0,1,2,3,4), joint_dim=640):
        super().__init__()
        self.vocab_size, self.blank_id = vocab_size, vocab_size
        self.durations = tuple(durations)
        self.enc, self.pred = nn.Linear(encoder_dim, joint_dim), nn.Linear(pred_dim, joint_dim, bias=False)
        self.out = nn.Linear(joint_dim, vocab_size + 1 + len(durations))
    def forward(self, encoder, prediction):
        # encoder projection: (B,T,J)->(B,T,1,J)
        # predictor projection: (B,U,J)->(B,1,U,J)
        # 加算後: (B,T,U,J)
        z = torch.tanh(self.enc(encoder)[:, :, None] + self.pred(prediction)[:, None])
        logits = self.out(z)
        return logits[..., :self.vocab_size+1], logits[..., self.vocab_size+1:]
    def step(self, encoder_t, prediction_u):
        """推論中の1格子点を計算する。入力は``(B,D)``, ``(B,H)``。"""
        logits = self.out(torch.tanh(self.enc(encoder_t) + self.pred(prediction_u)))
        return logits[:, :self.vocab_size+1], logits[:, self.vocab_size+1:]


class CTCHead(nn.Module):
    """履歴を使わない補助head: ``(B,T,D) -> (B,T,V+1)``。

    各encoder frameをtoken/blankへ独立射影し、CTC loss側で全単調alignmentを周辺化する。
    Hybrid学習ではencoderをTDTと共有するため、alignmentを促す補助教師として働く。
    """
    def __init__(self, d_model, vocab_size):
        super().__init__(); self.proj = nn.Linear(d_model, vocab_size + 1)
    def forward(self, x): return self.proj(x).log_softmax(-1)
