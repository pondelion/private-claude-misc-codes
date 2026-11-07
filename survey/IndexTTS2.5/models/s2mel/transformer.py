"""DiTのRMSNorm、時刻変調、RoPE、GQA、SwiGLU、U-skipを実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=Mel位置数、D=隠れ状態の次元数、H=head数、d=attention headごとの特徴次元数。hidden[B,T,D]、
Q/K/V[B,H,T,d]。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F
import torch


def find_multiple(n: int, k: int) -> int:
    """指定整数以上でkの倍数となる最小整数を返す。

    cacheの時間長調整に使う。

    Args:
        n: 切り上げる対象の整数。
        k: 倍数の基準となる正の整数。

    Returns:
        n以上となる最小のkの倍数。
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


class AdaptiveLayerNorm(nn.Module):
    """時刻条件から計算した係数で正規化特徴を変調する。"""

    def __init__(self, d_model: int, norm: nn.Module) -> None:
        """時刻条件から計算した係数で正規化特徴を変調するための層と設定を構築する。

        Args:
            d_model: 位置特徴または隠れ状態の特徴次元数。
            norm: 使用する正規化方式または構築済み正規化層。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        """時刻条件から計算した係数で正規化特徴を変調する。

        特徴[B,T,D]と時刻[B,1,D]→[B,T,D]。B=発話数、T=Mel長、D=隠れ状態の次元数。

        Args:
            input: 正規化する隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            embedding: 時刻条件[B,1,D]。Bは発話数、Dは隠れ状態の特徴次元数。Noneでは通常の正規化。

        Returns:
            正規化・変調後の特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        if embedding is None:
            return self.norm(input)
        # 時刻条件[B,1,D]→射影[B,1,2D]→係数2個の各[B,1,D]。Bは発話数、Dは内部特徴の次元数。
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


@dataclass
class ModelArgs:
    """Transformerの特徴次元数/層数/head数と位置情報の設定を保持する。

    Attributes:
        block_size: 扱える最大位置数。
        vocab_size: 元のTransformer設定との互換用語彙数。DiTは埋め込み済み特徴を入力する。
        n_layer: Transformerの層数。
        n_head: queryのhead数。
        dim: 隠れ状態の特徴次元数D。
        intermediate_size: SwiGLUの中間特徴次元数。Noneなら構築時に決める。
        n_local_heads: key/valueのhead数。-1ならqueryのhead数と同じ。
        head_dim: 各headの特徴次元数d。
        rope_base: RoPEの周波数尺度の底。
        norm_eps: RMS正規化の安定化定数。
        has_cross_attention: 参照特徴へのcross-attentionを追加するか。
        context_dim: cross-attentionに入力する参照特徴次元数。
        uvit_skip_connection: 前半層の特徴を後半層へ接続するか。
        time_as_token: 時刻条件を追加tokenとして扱うか。
    """
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    has_cross_attention: bool = False
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

    def __post_init__(self) -> None:
        """未指定のKV head数とSwiGLU中間特徴次元数を設定値から導出する。

        Returns:
            None。未指定の設定値をインスタンスへ補完する。
        """
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

    @classmethod
    def from_name(cls, name: str) -> ModelArgs:
        """既知のモデル名に対応する特徴次元数・層数等の設定を選択する。

        重み取得は行わない。

        Args:
            name: 設定名、または保存ファイルを識別する名前。

        Returns:
            名前に対応する設定またはその設定で構築したモデル。呼び出し元クラスに従う。
        """
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # 名前の一部が一致する既知設定を候補にする。
        config = [
            config for config in transformer_configs
            if config.lower() in str(name).lower()
        ]

        # 複数の候補がある場合は、より具体的な長い名前を優先する。
        # 一致する名前の文字数が多い設定を先頭へ並べる。
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(
                config[1]), name  # 最長の一致候補が1つに定まることを確認する。

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf":
    dict(block_size=16384,
         vocab_size=32000,
         n_layer=32,
         dim=4096,
         rope_base=1000000),
    "7B":
    dict(n_layer=32, n_head=32, dim=4096),
    "13B":
    dict(n_layer=40, n_head=40, dim=5120),
    "30B":
    dict(n_layer=60, n_head=52, dim=6656),
    "34B":
    dict(n_layer=48,
         n_head=64,
         dim=8192,
         vocab_size=32000,
         n_local_heads=8,
         intermediate_size=22016,
         rope_base=1000000),
    "70B":
    dict(n_layer=80,
         n_head=64,
         dim=8192,
         n_local_heads=8,
         intermediate_size=28672),
    "Mistral-7B":
    dict(n_layer=32,
         n_head=32,
         n_local_heads=8,
         dim=4096,
         intermediate_size=14336,
         vocab_size=32000),
    "stories15M":
    dict(n_layer=6, n_head=6, dim=288),
    "stories110M":
    dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b":
    dict(block_size=8192,
         n_layer=32,
         n_head=32,
         n_local_heads=8,
         dim=4096,
         intermediate_size=14336,
         vocab_size=128256,
         rope_base=500000),
    "llama-3-70b":
    dict(block_size=8192,
         n_layer=80,
         n_head=64,
         n_local_heads=8,
         dim=8192,
         intermediate_size=28672,
         vocab_size=128256,
         rope_base=500000),
}


class KVCache(nn.Module):
    """過去key/valueを時間位置ごとに保持する。
    """

    def __init__(self,
                 max_batch_size: int,
                 max_seq_length: int,
                 n_heads: int,
                 head_dim: int,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        """過去key/valueを時間位置ごとに保持するための層と設定を構築する。

        max_batch_size=cacheが保持する最大发話数、max_seq_length=cacheが保持する最大時間長、
        n_heads=attention head数、head_dim=head内特徴次元数。

        Args:
            max_batch_size: キャッシュが受け付ける最大発話数。
            max_seq_length: キャッシュが保持できる最大位置数。
            n_heads: KVキャッシュのhead数。
            head_dim: 1 head当たりの特徴次元数。
            dtype: 生成するテンソルのdtype。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor,
               v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 位置[T]にkey/value[B,H,T,d]を書き込む。B=発話数、H=KV head数、
        # d=attention headごとの特徴次元数。
        """指定位置input_pos[S]へK/V[B,H,S,d]を書き込み、保存済みの全KVを返す。

        Args:
            input_pos: 今回処理する位置の整数列[T]。Tは現在の入力長。
            k_val: 追加するkey[B,H,T,d]。Bは発話数、HはKV head数、Tは今回の位置数、
                dはattention headごとの特徴次元数。
            v_val: 追加するvalue[B,H,T,d]。Bは発話数、HはKV head数、Tは今回の位置数、
                dはattention headごとの特徴次元数。

        Returns:
            更新したkeyとvalueの組。各[Bmax,H,Kmax,d]。Bmaxは最大発話数、HはKV head数、
            Kmaxは確保済み位置数、dはattention headごとの特徴次元数。
        """
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    """attention/FFN blockを順に適用し最終正規化する。
    """

    def __init__(self, config: ModelArgs) -> None:
        """attention/FFN blockを順に適用し最終正規化するための層と設定を構築する。

        config=特徴次元数・層数・head数等の構成。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNorm(config.dim,
                                      RMSNorm(config.dim, eps=config.norm_eps))

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self,
                     max_batch_size: int,
                     max_seq_length: int,
                     use_kv_cache: Any = True) -> None:
        """位置回転係数[T,d/2,2]とcausal mask[T,T]、必要なKV保存領域を準備する。

        Args:
            max_batch_size: キャッシュが受け付ける最大発話数。
            max_seq_length: キャッシュが保持できる最大位置数。
            use_kv_cache: key/valueのキャッシュを保持するか。

        Returns:
            None。位置表、mask、必要なKV保存領域をインスタンスへ登録する。
        """
        if (self.max_seq_length >= max_seq_length
                and self.max_batch_size >= max_batch_size):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        if not self.training and use_kv_cache:
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                                               self.config.n_local_heads,
                                               head_dim, dtype).to(device)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size,
                                              self.config.head_dim,
                                              self.config.rope_base,
                                              dtype).to(device)
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length,
                       self.max_seq_length,
                       dtype=torch.bool)).to(device)
        self.use_kv_cache = use_kv_cache
        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [
                i for i in range(self.config.n_layer)
                if i < self.config.n_layer // 2
            ]
            self.layers_receive_skip = [
                i for i in range(self.config.n_layer)
                if i > self.config.n_layer // 2
            ]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_input_pos: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None) -> Tensor:
        """attention/FFN blockを順に適用し最終正規化する。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            c: 時刻条件[B,D]または[B,1,D]。Bは発話数、Dは隠れ状態の次元数。受け付ける形は各射影の定義に従う。
            input_pos: 今回処理する位置の整数列[T]。Tは現在の入力長。
            mask: 参照を許可する位置のbool mask[B,1,T,K]。Bは発話数、Tはquery長、Kはkey長。
                Noneでは内部の因果maskを使う。
            context:参照特徴[B,M,Dcontext]。Bは発話数、Mは参照長、Dcontextは参照特徴次元数。
                Noneではself-attention。
            context_input_pos: 参照位置の整数列[M]。Mは参照長。
            cross_attention_mask: 参照を許可する位置のmask[B,1,T,M]。Bは発話数、Tはquery長、
                Mは参照長。

        Returns:
            最終正規化後の特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None:  # mask省略時は、保持している因果maskから対象位置を選ぶ。
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if context is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if self.uvit_skip_connection and i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs_cis, mask, context,
                      context_freqs_cis, cross_attention_mask, skip_in_x)
            if self.uvit_skip_connection and i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        x = self.norm(x, c)
        return x

    @classmethod
    def from_name(cls, name: str) -> Transformer:
        """既知のモデル名に対応する特徴次元数・層数等の設定を選択する。

        重み取得は行わない。

        Args:
            name: 設定名、または保存ファイルを識別する名前。

        Returns:
            名前に対応する設定またはその設定で構築したモデル。呼び出し元クラスに従う。
        """
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    """時刻変調付きattentionとSwiGLUに残差とU-skipを適用する。
    """

    def __init__(self, config: ModelArgs) -> None:
        """時刻変調付きattentionとSwiGLUに残差とU-skipを適用するための層と設定を構築する。

        config=特徴次元数・層数・head数等の構成。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(
            config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(
                config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        else:
            self.has_cross_attention = False

        if config.uvit_skip_connection:
            self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)
            self.uvit_skip_connection = True
        else:
            self.uvit_skip_connection = False

        self.time_as_token = config.time_as_token

    def forward(self,
                x: Tensor,
                c: Tensor,
                input_pos: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                skip_in_x: Optional[Tensor] = None) -> Tensor:
        """時刻変調付きattentionとSwiGLUに残差とU-skipを適用する。

        特徴[B,T,D]→[B,T,D]。attention内部は[B,H,T,d]、H=head数、
        d=attention headごとの特徴次元数。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            c: 時刻条件[B,D]または[B,1,D]。Bは発話数、Dは隠れ状態の次元数。受け付ける形は各射影の定義に従う。
            input_pos: 今回処理する位置の整数列[T]。Tは現在の入力長。
            freqs_cis: 現在位置のRoPE係数[T,d/2,2]。Tは位置数、dはattention headごとの特徴次元数、
                末尾2はcosとsin。
            mask: 参照を許可する位置のbool mask[B,1,T,K]。Bは発話数、Tはquery長、Kはkey長。
                Noneでは内部の因果maskを使う。
            context:参照特徴[B,M,Dcontext]。Bは発話数、Mは参照長、Dcontextは参照特徴次元数。
                Noneではself-attention。
            context_freqs_cis: 参照位置のRoPE係数[M,d/2,2]。Mは参照長、
                dはattention headごとの特徴次元数。
            cross_attention_mask: 参照を許可する位置のmask[B,1,T,M]。Bは発話数、Tはquery長、
                Mは参照長。
            skip_in_x: U字接続の前半層出力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。Noneでは連結なし。

        Returns:
            残差加算後の特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        c = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            # U字接続の特徴を結合して圧縮する。[B,T,D]×2→[B,T,2D]→[B,T,D]。
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask,
                               input_pos)
        if self.has_cross_attention:
            h = h + self.cross_attention(self.cross_attention_norm(
                h, c), freqs_cis, cross_attention_mask, input_pos, context,
                                         context_freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Attention(nn.Module):
    """Q/K/V射影とattentionによる位置間集約を計算する。
    """

    def __init__(self,
                 config: ModelArgs,
                 is_cross_attention: bool = False) -> None:
        """Q/K/V射影とattentionによる位置間集約を計算するための層と設定を構築する。

        config=特徴次元数・層数・head数等の構成。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。
            is_cross_attention: 参照特徴からkey/valueを計算するcross-attentionか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head +
                          2 * config.n_local_heads) * config.head_dim
        # Qと共有KVを1回の射影で作り、後で各headへ分離する。
        if is_cross_attention:
            self.wq = nn.Linear(config.dim,
                                config.n_head * config.head_dim,
                                bias=False)
            self.wkv = nn.Linear(config.context_dim,
                                 2 * config.n_local_heads * config.head_dim,
                                 bias=False)
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head,
                            config.dim,
                            bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                mask: Tensor,
                input_pos: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_freqs_cis: Optional[Tensor] = None) -> Tensor:
        """Q/K/V射影とattentionによる位置間集約を計算する。

        特徴[B,T,D]をheadへ分け、Q/K/V[B,H,T,d]で集約して[B,T,D]へ戻す。B=発話数、T=位置数、D=特徴次元数、
        H=head数、d=attention headごとの特徴次元数。

        Args:
            x: query入力[B,N,D]。Bは発話数、Nはquery数、Dは内部特徴次元数。
            freqs_cis: 現在位置のRoPE係数[T,d/2,2]。Tは位置数、dはattention headごとの特徴次元数、
                末尾2はcosとsin。
            mask: 参照を許可する位置のbool mask[B,1,T,K]。Bは発話数、Tはquery長、Kはkey長。
                Noneでは内部の因果maskを使う。
            input_pos: 今回処理する位置の整数列[T]。Tは現在の入力長。
            context:参照特徴[B,M,Dcontext]。Bは発話数、Mは参照長、Dcontextは参照特徴次元数。
                Noneではself-attention。
            context_freqs_cis: 参照位置のRoPE係数[M,d/2,2]。Mは参照長、
                dはattention headごとの特徴次元数。

        Returns:
            attention出力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        # queryのhead軸を分離する。[B,T,H*d]→[B,T,H,d]。Bは発話数、Tはquery長、Hはquery head数、dは
        # attention headごとの特徴次元数。
        q = rearrange(q, "b t (h d) -> b t h d", h=self.n_head)
        # 共有keyのhead軸を分離する。[B,M,Hkv*d]→[B,M,Hkv,d]。Mはkey長、HkvはKV head数。
        k = rearrange(k, "b t (h d) -> b t h d", h=self.n_local_heads)
        # 共有valueも[B,M,Hkv*d]→[B,M,Hkv,d]へ分離する。keyと位置順を一致させる。
        v = rearrange(v, "b t (h d) -> b t h d", h=self.n_local_heads)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(
            k,
            context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        # attention用にheadを時間の前へ移す。Qは[B,H,T,d]、K/Vは[B,Hkv,M,d]。
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # GQAの共有keyをquery headへ対応させる。[B,Hkv,M,d]→[B,H,M,d]。
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        # 共有valueもquery head数へ展開する。[B,Hkv,M,d]→[B,H,M,d]。
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q,
                                           k,
                                           v,
                                           attn_mask=mask,
                                           dropout_p=0.0)

        # headを結合する。[B,H,T,d]→[B,T,H*d]。後続の出力射影で内部特徴の次元数Dへ戻す。
        y = rearrange(y, "b h t d -> b t (h d)")

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    """位置ごとの特徴射影とgate付き非線形変換を計算する。
    """

    def __init__(self, config: ModelArgs) -> None:
        """位置ごとの特徴射影とgate付き非線形変換を計算するための層と設定を構築する。

        config=特徴次元数・層数・head数等の構成。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """位置ごとの特徴射影とgate付き非線形変換を計算する。

        入力/出力[B,T,D]。中間層で特徴を拡張してからDへ戻す。 B=発話数、T=位置数、D=特徴次元数。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            SwiGLU出力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        # SwiGLUで値とgateを掛ける。[B,T,D]→2本の[B,T,I]→[B,T,D]。IはFFNの中間特徴次元数。
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """最後の特徴軸をRMS正規化する。
    """

    def __init__(self, dim: int, eps: float = 1e-05) -> None:
        """最後の特徴軸をRMS正規化するための層と設定を構築する。

        dim=隠れ状態の次元数、eps=正規化分母の安定化係数。

        Args:
            dim: 特徴の次元数。
            eps: 正規化の分母へ加える安定化定数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """最後の特徴軸Dに沿った平均二乗からRMS正規化値を計算する。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            RMS正規化した特徴。入力と同shapeで末尾軸だけを正規化する。
        """
        return x * torch.rsqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """最後の特徴軸をRMS正規化する。

        入力/出力のshapeは同じ。最後の軸Dだけを正規化する。B=発話数、T=位置数。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。

        Returns:
            正規化後の特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int,
                         n_elem: int,
                         base: int = 10000,
                         dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """時刻Tと特徴対d/2に対応するcos/sin係数[T,d/2,2]を作る。

    Args:
        seq_len: 生成する位置表の長さ。
        n_elem: RoPEを適用する特徴次元数。
        base: RoPEの周波数尺度の底。
        dtype: 生成するテンソルのdtype。

    Returns:
        RoPE係数[T,d/2,2]。Tはseq_len、dはn_elem、末尾2はcosとsin。
    """
    freqs = 1.0 / (base**(torch.arange(0, n_elem, 2)[:(n_elem // 2)].float() /
                          n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 回転係数のcosとsinを末尾へ積む。[T,d/2]×2→[T,d/2,2]。Tは位置数、dはattention headごとの特徴次元数。
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Q/K[B,T,H,d]を隣接2要素ごとに回転し、同shapeでRoPEを適用する。

    Args:
        x: queryまたはkey[B,T,H,d]。Bは発話数、Tは位置数、Hはhead数、
            dはattention headごとの特徴次元数。
        freqs_cis: 現在位置のRoPE係数[T,d/2,2]。Tは位置数、dはattention headごとの特徴次元数、
            末尾2はcosとsin。

    Returns:
        RoPE適用後の特徴[B,T,H,d]。Bは発話数、Tは位置数、Hはhead数、dはattention headごとの特徴次元数。
    """
    # RoPE用に隣り合う2特徴を実部・虚部の組へ分ける。[B,T,H,d]→[B,T,H,d/2,2]。
    xshaped = rearrange(x.float(), "b t h (d r) -> b t h d r", r=2)
    # 位置ごとの回転係数[T,d/2,2]へ発話・headのbroadcast軸を足し、[1,T,1,d/2,2]にする。
    freqs_cis = rearrange(freqs_cis, "t d r -> 1 t 1 d r")
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] -
            xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] +
            xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
