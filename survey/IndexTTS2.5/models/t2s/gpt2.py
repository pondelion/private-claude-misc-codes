"""PyTorch基本演算で実装したT2SのGPT-2 decoder。

外部GPT2Modelを使用しない。公式transformers_gpt2.pyのpre-norm、Conv1Dの
weight配置[in,out]、gelu_new、causal/padding mask、残差/dropout、最終normを
ローカルに実装する。位置埋め込みは上位で加算し、本体では追加しない。
state_dictのgpt.h.N.*名とshapeを公式に合わせ、重みの転置変換を不要にする。

B=発話数、T=今回の入力位置数、K=過去cacheを含む位置数、D=特徴次元数、H=head数。
本ファイルをimportしてもモデルを構築しない。Tensor演算は呼び出し時だけ行う。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch


class Conv1D(nn.Module):
    """GPT-2のLinear。重みをnn.Linearとは逆の[in_features,out_features]で保持。"""

    def __init__(self, out_features: int, in_features: int) -> None:
        """公式c_attn/c_proj等のweight/biasの名前・shapeで初期化する。

        Args:
            out_features: 出力特徴の次元数。
            in_features: 入力チャネル数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden[B,T,Din]→[B,T,Dout]。

        B/Tは維持し、特徴次元数だけを射影する。

        Args:
            hidden: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            射影特徴[B,T,Dout]。Bは発話数、Tは位置数、Doutは出力の特徴次元数。
        """
        batch, length = hidden.shape[:2]
        # 発話軸と時間軸を一時的にまとめる。[B,T,Din]→[B*T,Din]。B=発話数、T=位置数、Din=入力の特徴次元数。
        flat = rearrange(hidden, "b t d -> (b t) d")
        # 共有重み[Din,Dout]で射影する。[B*T,Din]→[B*T,Dout]。Doutは出力の特徴次元数。
        projected = torch.addmm(self.bias, flat, self.weight)
        # まとめた発話軸と時間軸を復元する。[B*T,Dout]→[B,T,Dout]。
        return rearrange(projected, "(b t) d -> b t d", b=batch, t=length)


class GPT2Attention(nn.Module):
    """QKV射影、過去KV cache、causal maskを備えるGPT-2 self-attention。"""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        """dimをheads個へ分割し、公式c_attn/c_proj名の層を構築する。

        Args:
            dim: 特徴の次元数。
            heads: attention head数。
            dropout: dropout確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        if dim % heads:
            raise ValueError("GPT width must be divisible by head count")
        self.heads = heads
        self.c_attn = Conv1D(3 * dim, dim)
        self.c_proj = Conv1D(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """[B,T,D]とmask[B,K]、任意cache[B,H,Kpast,D/H]→出力と新cache。

        causal位置は過去KVの長さを加味する。padding maskの0は参照を禁止する。

        Args:
            hidden: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            attention_mask: 有効位置のmask[B,K]。Bは発話数、Kは参照可能な位置数。1が有効。
            past:過去の(key,value)。各[B,H,Kpast,d]。Bは発話数、Hはhead数、Kpastは過去長、
                d=D/H。Noneでは過去なし。

        Returns:
            attention出力[B,T,D]と(key,value)の組。各cacheは[B,H,K,d]で、Hはhead数、
            Kは過去と現在の位置数、d=D/H。
        """
        # 1. [B,T,3D]を分割し、head軸へ展開する。各[B,H,T,D/H]。
        query, key, value = self.c_attn(hidden).chunk(3, dim=-1)
        query = rearrange(query, "b t (h d) -> b h t d", h=self.heads)
        key = rearrange(key, "b t (h d) -> b h t d", h=self.heads)
        value = rearrange(value, "b t (h d) -> b h t d", h=self.heads)
        if past is not None:
            # 過去のkeyと今回のkeyを時間軸で連結する。[B,H,Kpast,d]と[B,H,T,d]→[B,H,Kpast+T,d]。
            key = torch.cat((past[0], key), dim=2)
            # valueもkeyと同じ順序で連結する。[B,H,Kpast,d]と[B,H,T,d]→[B,H,Kpast+T,d]。
            value = torch.cat((past[1], value), dim=2)
        # 2. score[B,H,T,K]へcausal/padding maskを適用する。
        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(value.size(-1))
        queries, keys = query.size(-2), key.size(-2)
        causal = torch.arange(keys, device=hidden.device)[None] <= (
            torch.arange(queries, device=hidden.device)[:, None] + keys -
            queries)
        scores = torch.where(
            causal, scores,
            torch.full((),
                       torch.finfo(scores.dtype).min,
                       device=scores.device,
                       dtype=scores.dtype))
        if attention_mask is not None:
            additive = (1 - attention_mask[:, None, None, :].to(scores.dtype))
            scores = scores + additive * torch.finfo(scores.dtype).min
        # 3. 語彙ではなくkey位置方向で正規化し、Vを重み付き集約する。
        weights = self.attn_dropout(F.softmax(scores, dim=-1).to(value.dtype))
        # head別特徴[B,H,T,d]。Bは発話数、Hはhead数、Tは今回の位置数、d=D/H。
        attended = weights @ value
        # head軸を結合して射影する。[B,H,T,d]→[B,T,H*d]→[B,T,D]。
        output = self.c_proj(rearrange(attended, "b h t d -> b t (h d)"))
        return self.resid_dropout(output), (key, value)


class GPT2MLP(nn.Module):
    """特徴次元数を4倍に拡大し、近似GELUと出力射影を適用する。"""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        """特徴次元数を4倍に拡大し、近似GELUと出力射影を適用するための層と設定を構築する。

        Args:
            dim: 特徴の次元数。
            dropout: dropout確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.c_fc = Conv1D(4 * dim, dim)
        self.c_proj = Conv1D(dim, 4 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """特徴次元数を4倍に拡大し、近似GELUと出力射影を適用する。

        Args:
            hidden: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            FFN出力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        # 時刻ごとに特徴次元数を拡張する。[B,T,D]→[B,T,4D]。DはGPTの隠れ状態の次元数。
        expanded = self.c_fc(hidden)
        activated = 0.5 * expanded * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (expanded + 0.044715 * expanded.pow(3))))
        # GELU後の特徴を元の特徴次元数へ戻す。[B,T,4D]→[B,T,D]。
        return self.dropout(self.c_proj(activated))


class GPT2Block(nn.Module):
    """LayerNorm→attention→残差→LayerNorm→FFN→残差の1層。"""

    def __init__(self, dim: int, heads: int) -> None:
        """公式のLayerNorm eps=1e-5、dropout=0.1を維持する。

        Args:
            dim: 特徴の次元数。
            heads: attention head数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = GPT2Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = GPT2MLP(dim)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """hidden[B,T,D]と有効mask[B,K]から同shapeの隠れ状態とKVを返す。

        Args:
            hidden: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            mask: 有効key位置mask[B,K]。Bは発話数、Kは過去を含むkey長。0はpadding。
            past:過去の(key,value)。各[B,H,Kpast,d]。Bは発話数、Hはhead数、Kpastは過去長、
                d=D/H。Noneでは過去なし。

        Returns:
            更新した隠れ状態[B,T,D]と(key,value)の組。各cacheは[B,H,K,d]で、Hはhead数、
            Kは過去と現在の位置数、d=D/H。
        """
        attended, cache = self.attn(self.ln_1(hidden), mask, past)
        hidden = hidden + attended
        return hidden + self.mlp(self.ln_2(hidden)), cache


@dataclass
class DecoderOutput:
    """隠れ状態[B,T,D]と、層ごとのKV cacheをまとめる。

    Attributes:
        last_hidden_state: 隠れ状態[B,T,D]。Bは発話数、Tは現在の入力長、DはGPTの隠れ状態の次元数。
        past_key_values: 層ごとの(key,value)のリスト。各[B,H,K,d]。Bは発話数、Hはhead数、Kは過去を含む長さ
            、dはattention headごとの特徴次元数。
    """

    last_hidden_state: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


class GPT2Model(nn.Module):
    """外部Transformerクラスに依存しない、embedding入力のGPT-2本体。"""

    def __init__(self, layers: int, dim: int, heads: int) -> None:
        """公式のh/ln_f名で各層を構築し、GPT-2方式で重みを初期化する。

        Args:
            layers: Transformer層数。
            dim: 特徴の次元数。
            heads: attention head数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList(GPT2Block(dim, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dim, eps=1e-5)
        self.apply(self._initialize)
        # GPT-2は残差出力射影だけ1/sqrt(2*n_layer)で初期分散を調整する。
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        """Conv1D/Linear/Embedding/LayerNormの公式初期化を適用する。

        Args:
            module: 構築済みのPyTorchモジュール。

        Returns:
            None。対象モジュールの重み・バイアスを更新する。
        """
        if isinstance(module, (Conv1D, nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput:
        """embedding[B,T,D]を処理する。

        位置embeddingは上位で加算済み。

        Args:
            inputs_embeds: 埋め込み済み入力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            attention_mask: 有効位置のmask[B,K]。Bは発話数、Kは参照可能な位置数。1が有効。
            past_key_values:層ごとの過去(key,value)のリスト。各[B,H,Kpast,d]。Bは発話数、
                Hはhead数、Kpastは過去長、dはattention headごとの特徴次元数。Noneでは過去なし。
            return_dict: DecoderOutput形式で返すか。現在の実装は常に同形式を返す。

        Returns:
            隠れ状態[B,T,D]と層ごとのKV cacheを保持するDecoderOutput。各cacheは[B,H,K,d]。Bは発話数、
            Tは今回の位置数、DはGPTの隠れ状態の次元数、Hはhead数、Kは過去と今回の合計長、d=D/H。
        """
        hidden = self.drop(inputs_embeds)
        caches = []
        for index, layer in enumerate(self.h):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = layer(hidden, attention_mask, past)
            caches.append(cache)
        return DecoderOutput(self.ln_f(hidden), caches)
