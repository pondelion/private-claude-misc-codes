"""感情frontendのConformer部品。射影・位置情報・畳み込み・attentionを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=入力特徴の時間長、D=特徴次元数、H=head数。hidden[B,T,D]、mask[B,1,T]。
"""
from __future__ import annotations

from typing import Any, Tuple
import math

from einops import rearrange
from torch import nn
import torch


class MultiHeadedAttention(nn.Module):
    """複数headによるscaled dot-product attentionを計算する。"""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float) -> None:
        """複数headによるscaled dot-product attentionを計算するための層と設定を構築する。

        Args:
            n_head: attention head数。
            n_feat: 全attention headを結合した特徴次元数。
            dropout_rate: dropout確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        assert n_feat % n_head == 0
        # keyとvalueのattention headごとの特徴次元数は共通とする。d=全特徴次元数/head数。
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """query、key、valueを射影し、head軸を分離する。

        Args:
            query: query特徴[B,N,D]。Bは発話数、Nはquery数、Dは特徴次元数。
            key: key特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。
            value: value特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。

        Returns:
            射影したquery[B,H,N,d]、key[B,H,M,d]、value[B,H,M,d]の組。Bは発話数、Hはhead数、
            Nはquery数、Mはkey数、dはattention headごとの特徴次元数。
        """
        n_batch = query.size(0)
        q = rearrange(self.linear_q(query), "b t (h d) -> b t h d", h=self.h)
        k = rearrange(self.linear_k(key), "b t (h d) -> b t h d", h=self.h)
        v = rearrange(self.linear_v(value), "b t (h d) -> b t h d", h=self.h)
        # query側の特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery長、dはattention headごとの特徴次元数。
        q = q.transpose(1, 2)
        # 参照側の特徴[B,H,M,d]。Bは発話数、Hはhead数、Mはkey長、dはattention headごとの特徴次元数。
        k = k.transpose(1, 2)
        # 参照側の特徴[B,H,M,d]。Bは発話数、Hはhead数、Mはkey長、dはattention headごとの特徴次元数。
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """mask付きscoreを正規化し、valueを重み付き集約する。

        Args:
            value: 射影済みvalue[B,H,M,d]。Bは発話数、Hはhead数、Mはkey数、
                dはattention headごとの特徴次元数。
            scores: attention score[B,H,N,M]。Bは発話数、Hはhead数、Nはquery数、Mはkey数。
            mask:有効位置mask[B,1,M]または[B,N,M]。Bは発話数、Nはquery数、Mはkey数。0は参照禁止。
                空Tensorではmaskなし。

        Returns:
            集約特徴[B,N,D]。Bは発話数、Nはquery数、Dは全headを結合した特徴次元数。
        """
        n_batch = value.size(0)
        # 有効maskがある経路ではpadding位置のscoreと重みを無効化する。
        if mask.size(2) > 0:  # 有効な参照位置maskが渡された場合だけmask処理を行う。
            # 禁止位置mask[B,1,Nmask,M]。Bは発話数、Nmaskは1またはquery長、Mはkey長。
            mask = mask.unsqueeze(1).eq(0)
            # 末尾chunkではmaskがscoreより長い場合があるため、key長へ切り詰める。
            # 禁止位置mask[B,1,Nmask,M]。Bは発話数、Nmaskは1またはquery長、Mはkey長。
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                # attention scoreまたは重み[B,H,N,M]。Bは発話数、Hはhead数、Nはquery長、Mはkey長。
                mask,
                0.0)
        # 空maskは参照制限なしを表す。
        else:
            # attention scoreまたは重み[B,H,N,M]。Bは発話数、Hはhead数、Nはquery長、Mはkey長。
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        # query側の特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery長、dはattention headごとの特徴次元数。
        x = torch.matmul(p_attn, value)
        # headを結合した特徴[B,N,D]。Bは発話数、Nはquery長、Dは全attention headを結合した特徴次元数。
        x = (rearrange(x, "b h t d -> b t (h d)"))

        # headを結合した特徴[B,N,D]。Bは発話数、Nはquery長、Dは全attention headを結合した特徴次元数。
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """複数headによるscaled dot-product attentionを計算する。

        Args:
            query: query特徴[B,N,D]。Bは発話数、Nはquery数、Dは特徴次元数。
            key: key特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。
            value: value特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。
            mask:有効位置mask[B,1,M]または[B,N,M]。Bは発話数、Nはquery数、Mはkey数。0は参照禁止。
                空Tensorではmaskなし。
            pos_emb: 位置特徴[1,Tpos,D]または[B,Tpos,D]。Bは発話数、Tposは位置表の長さ、Dは特徴次元数。
            cache:過去のkey/valueを特徴軸で連結したcache[B,H,K,2d]。Bは発話数、Hはhead数、Kは過去長、
                dはattention headごとの特徴次元数。空Tensorでは過去なし。

        Returns:
            出力特徴[B,N,D]と更新cache[B,H,K,2d]の組。Bは発話数、Nはquery数、
            Dは全attention headを結合した特徴次元数、Hはhead数、Kは過去を含むkey長、d=D/H。
        """
        q, k, v = self.forward_qkv(query, key, value)

        # cacheがあれば過去KVを時間軸へ連結する。長さ0のKVも連結可能。
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # 保持する過去長の切り詰めは、この層の呼び出し元で行う。
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """相対位置特徴をscoreへ加えたattentionを計算する。"""

    def __init__(self, n_head: Any, n_feat: Any, dropout_rate: float) -> None:
        """相対位置特徴をscoreへ加えたattentionを計算するための層と設定を構築する。

        Args:
            n_head: attention head数。
            n_feat: 全attention headを結合した特徴次元数。
            dropout_rate: dropout確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(n_head, n_feat, dropout_rate)
        # 相対位置特徴をattentionと同じ全attention headを結合した特徴次元数へ線形射影する。
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # 内容scoreと相対位置scoreに、それぞれ学習可能なhead別biasを加える。
        # 相対位置attentionの式はarXiv:1901.02860の3.3節に対応する。
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor, zero_triu: bool = False) -> Any:
        """相対位置scoreのメモリ順を組み替えて位置をずらす。

        Args:
            x: 相対位置score[B,H,N,M]。Bは発話数、Hはhead数、Nはquery数、Mは相対位置数。
            zero_triu: 相対位置scoreの未来側の三角領域をゼロにするか。

        Returns:
            位置をずらしたscore[B,H,N,M]。Bは発話数、Hはhead数、Nはquery数、Mは相対位置数。
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = rearrange(rearrange(x_padded, "b h t s -> b h (t s)"),
                             "b h (s t) -> b h s t",
                             s=x.size(3) + 1,
                             t=x.size(2))
        x = rearrange(rearrange(x_padded[:, :, 1:], "b h s t -> b h (s t)"),
                      "b h (t s) -> b h t s",
                      t=x.size(2),
                      s=x.size(3))

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """相対位置特徴をscoreへ加えたattentionを計算する。

        Args:
            query: query特徴[B,N,D]。Bは発話数、Nはquery数、Dは特徴次元数。
            key: key特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。
            value: value特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。
            mask:有効位置mask[B,1,M]または[B,N,M]。Bは発話数、Nはquery数、Mはkey数。0は参照禁止。
                空Tensorではmaskなし。
            pos_emb: 位置特徴[1,Tpos,D]または[B,Tpos,D]。Bは発話数、Tposは位置表の長さ、Dは特徴次元数。
            cache:過去のkey/valueを特徴軸で連結したcache[B,H,K,2d]。Bは発話数、Hはhead数、Kは過去長、
                dはattention headごとの特徴次元数。空Tensorでは過去なし。

        Returns:
            出力特徴[B,N,D]と更新cache[B,H,K,2d]の組。Bは発話数、Nはquery数、
            Dは全attention headを結合した特徴次元数、Hはhead数、Kは過去を含むkey長、d=D/H。
        """
        q, k, v = self.forward_qkv(query, key, value)
        # 位置射影[B,N,H,d]。Bは発話数、Nは位置数、Hはhead数、dはattention headごとの特徴次元数。
        q = q.transpose(1, 2)

        # cacheがあれば過去KVを時間軸へ連結する。長さ0のKVも連結可能。
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # 保持する過去長の切り詰めは、この層の呼び出し元で行う。
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = rearrange(self.linear_pos(pos_emb),
                      "b t (h d) -> b t h d",
                      h=self.h)
        # query側の特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery長、dはattention headごとの特徴次元数。
        p = p.transpose(1, 2)

        # query側の特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery長、dはattention headごとの特徴次元数。
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # query側の特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery長、dはattention headごとの特徴次元数。
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # 内容項と相対位置項を足し、attention headごとの特徴次元数で正規化したscoreを作る。
        # 内容queryと内容biasを足し、keyとの内積を計算する。
        # 相対位置attentionの式はarXiv:1901.02860の3.3節に対応する。
        # attention scoreまたは重み[B,H,N,M]。Bは発話数、Hはhead数、Nはquery長、Mはkey長。
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # 位置queryと位置biasを足し、相対位置特徴との内積を計算する。
        # attention scoreまたは重み[B,H,N,M]。Bは発話数、Hはhead数、Nはquery長、Mはkey長。
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # 対象の公開経路と同じく、このscoreへrel_shiftは適用しない。
        # 位置のずらし方を追加すると、公開経路と異なるattentionになる。

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            # attention scoreまたは重み[B,H,N,M]。Bは発話数、Hはhead数、Nはquery長、Mはkey長。
            self.d_k)

        return self.forward_attention(v, scores, mask), new_cache
