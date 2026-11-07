"""感情特徴をlearned queryで集約するPerceiver。attention/FFNを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=参照特徴長、D=特徴次元数、N=learned query数（感情では1）。入力[B,T,D]→[B,N,1024]。
"""
from __future__ import annotations

from collections import namedtuple
from functools import wraps
from typing import Any

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum
from torch import nn
from torch.nn import functional as F
import torch


def exists(val: Any) -> bool:
    """値がNoneでないかをboolで返す。

    Tensorの値やshapeは変更しない。

    Args:
        val: Noneかどうかを判定する値。

    Returns:
        入力値がNoneでなければTrue。Tensorの要素値は参照しない。
    """
    return val is not None


def once(fn: Any) -> Any:
    """渡された関数を初回だけ呼ぶdecoratorを作る。

    モデル演算には関与しない。

    Args:
        fn: 初回の呼び出しだけを実行する関数。

    Returns:
        初回だけfnを呼ぶラッパー関数。
    """
    called = False

    @wraps(fn)
    def inner(x: Any) -> Any:
        """初回の呼び出しだけ元の関数を実行し、その後の重複通知を抑える。

        Args:
            x: 初回に元の関数へ渡す値。通知用の文字列等を受け取る。

        Returns:
            初回は元の関数の戻り値。2回目以降はNone。
        """
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


# 以下でattentionの計算とmask処理を定義する。
class Attend(nn.Module):
    """queryとkeyのscoreを正規化し、valueを重み付き集約する。"""

    def __init__(self,
                 dropout: float = 0.0,
                 causal: bool = False,
                 use_flash: bool = False) -> None:
        """queryとkeyのscoreを正規化し、valueを重み付き集約するための層と設定を構築する。

        Args:
            dropout: dropout確率。
            causal: 未来の時刻を参照しない処理にするか。
            use_flash: PyTorchのSDPA経路を使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__)
                    < version.parse("2.0.0")), (
                        "in order to use flash attention, "
                        "you must be using pytorch 2.0 or above")

        # CPUとGPUで利用可能なSDPA実行方式を選ぶ。
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"])
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(("A100 GPU detected, using flash "
                        "attention if input tensor is on cuda"))
            self.cuda_config = self.config(True, False, False)
        else:
            print_once(("Non-A100 GPU detected, using math or mem "
                        "efficient attention if input tensor is on cuda"))
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n: int, device: torch.device | str) -> torch.Tensor:
        """長さNの上三角bool mask[N,N]を構築し、未来位置への参照を禁止する。

        Args:
            n: 因果maskの位置数。
            device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

        Returns:
            未来位置がTrueの因果mask[N,N]。Nはqueryとkeyの位置数。
        """
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self,
                   q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        """Q[B,H,N,d]とK/V[B,H,M,d]または[B,M,d]をSDPAで集約し[B,H,N,d]を返す。

        Args:
            q: query[B,H,N,d]。Bは発話数、Hはhead数、Nはquery数、
                dはattention headごとの特徴次元数。
            k: key[B,H,M,d]または[B,M,d]。Bは発話数、Hはhead数、Mは参照長、
                dはattention headごとの特徴次元数。
            v: value[B,H,M,d]または[B,M,d]。Bは発話数、Hはhead数、Mは参照長、
                dはattention headごとの特徴次元数。
            mask: 参照の有効位置mask[B,M]。Bは発話数、Mはkeyの位置数。Trueが参照可能。Noneでは全位置。

        Returns:
            集約特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery数、dはattention headごとの特徴次元数。
        """
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # headを共有するkey/valueはhead軸を補って計算する。
        # 共有KV[B,M,d]→head付き[B,H,M,d]。Bは発話数、Hはhead数、Mは参照長、
        # dはattention headごとの特徴次元数。

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # 参照位置maskを各head・queryへbroadcastする。
        # mask[B,M]→[B,H,N,M]。Bは発話数、Hはhead数、Nはquery数、Mは参照長。

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # 入力デバイスに対応するSDPA実行設定を使用する。

        config = self.cuda_config if is_cuda else self.cpu_config

        # Q/K/VとmaskをPyTorchのSDPAへ渡し、同じhead軸の集約特徴を得る。

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal)

        return out

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """queryとkeyのscoreを正規化し、valueを重み付き集約する。

        attentionでは発話軸B、head軸H、query軸N、key軸M、head特徴軸dを区別する。Q[B,H,N,d]、
        K/V[B,H,M,d]または[B,M,d]→[B,H,N,d]。B=発話数、H=head数、N=query数、M=key数、
        d=attention headごとの特徴次元数。

        Args:
            q: query[B,H,N,d]。Bは発話数、Hはhead数、Nはquery数、
                dはattention headごとの特徴次元数。
            k: key[B,H,M,d]または[B,M,d]。Bは発話数、Hはhead数、Mは参照長、
                dはattention headごとの特徴次元数。
            v: value[B,H,M,d]または[B,M,d]。Bは発話数、Hはhead数、Mは参照長、
                dはattention headごとの特徴次元数。
            mask: 参照の有効位置mask[B,M]。Bは発話数、Mはkeyの位置数。Trueが参照可能。Noneでは全位置。

        Returns:
            集約特徴[B,H,N,d]。Bは発話数、Hはhead数、Nはquery数、dはattention headごとの特徴次元数。
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1]**-0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # QK内積でscore[B,H,N,M]を計算し、attention headごとの特徴次元数の平方根で割る。

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # 有効な参照位置以外のscoreを抑制する。

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 因果設定の場合だけ未来位置のscoreを抑制する。

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # key位置の軸でsoftmaxを取り、attention重みを得る。

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 重みとvalueを掛け、key位置を集約する。[B,H,N,M]×[B,H,M,d]→[B,H,N,d]。

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


def Sequential(*mods: Any) -> nn.Sequential:
    """Noneの部品を除いてnn.Sequentialを構築する。

    層の順序は維持する。

    Args:
        *mods: 元の順序で接続するモジュール。Noneは除く。

    Returns:
        None以外のモジュールを接続したnn.Sequential。
    """
    return nn.Sequential(*filter(exists, mods))


def exists(x: Any) -> bool:
    """値がNoneでないかをboolで返す。

    Tensorの値やshapeは変更しない。

    Args:
        x: Noneかどうかを判定する値。

    Returns:
        入力値がNoneでなければTrue。Tensorの要素値は参照しない。
    """
    return x is not None


def default(val: Any, d: Any) -> Any:
    """値がNoneなら既定値または既定値を生成する関数の結果を返す。

    Args:
        val: Noneかどうかを判定する値。
        d: valがNoneのときに用いる値、または値を返す関数。

    Returns:
        valがNoneでなければval。Noneならd、またはdを呼んだ結果。
    """
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    """最後の特徴軸をRMS正規化する。
    """

    def __init__(self,
                 dim: int,
                 scale: bool = True,
                 dim_cond: Any = None) -> None:
        """最後の特徴軸をRMS正規化するための層と設定を構築する。

        Args:
            dim: 特徴の次元数。
            scale: 学習可能な特徴ごとの係数を保持するか。
            dim_cond: 正規化条件の次元数。Noneでは条件射影を作らない。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim *
                                       2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """最後の特徴軸をRMS正規化する。

        入力/出力のshapeは同じ。最後の軸Dだけを正規化する。B=発話数、T=位置数。

        Args:
            x: 隠れ状態[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            cond: 正規化条件[B,Dcond]。Bは発話数、Dcondは条件の特徴次元数。Noneでは無条件。

        Returns:
            正規化後の特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        # 発話ごとの条件[B,Dcond]→2本の正規化係数[B,D]。Dcondは条件の特徴次元数、Dは正規化する特徴次元数。
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        # 係数へ時間broadcast軸を加える。[B,D]→[B,1,D]。
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"),
                          (gamma, beta))
        return out * gamma + beta


class CausalConv1d(nn.Conv1d):
    """左paddingを使って未来を参照しない時間Convを計算する。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """左paddingを使って未来を参照しない時間Convを計算するための層と設定を構築する。

        Args:
            *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(*args, **kwargs)
        (kernel_size, ) = self.kernel_size
        (dilation, ) = self.dilation
        (stride, ) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """左paddingを使って未来を参照しない時間Convを計算する。

        Args:
            x: 入力特徴[B,Cin,T]。Bは発話数、Cinは入力チャネル数、Tは時間長。

        Returns:
            畳み込み特徴[B,Cout,T]。Bは発話数、Coutは出力チャネル数、Tは入力と同じ時間長。
        """
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Module):
    """特徴を値とgateに分割しGELU gateを乗算する。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴を値とgateに分割しGELU gateを乗算する。

        特徴[B,T,2D]→[B,T,D]。B=発話数、T=位置数、D=gate後の特徴次元数。

        Args:
            x: 値とgateを連結した特徴[B,T,2D]。Bは発話数、Tは位置数、Dはgate適用後の特徴次元数。

        Returns:
            gateを乗算した特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim: int,
                mult: Any = 4,
                causal_conv: Any = False) -> nn.Sequential:
    """GEGLUと任意の因果畳み込みを含むFFNを構築する。

    Args:
        dim: 特徴の次元数。
        mult: FFNの中間次元を決める倍率。
        causal_conv: FFNへ左側padding付きの時間畳み込みを挿入するか。

    Returns:
        特徴[B,T,D]を同shapeへ変換するnn.Sequential。Bは発話数、Tは位置数、Dは入力の特徴次元数。
    """
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv,
                      nn.Linear(dim_inner, dim))


class PerceiverResampler(nn.Module):
    """参照特徴をlearned queryに集約する。
    """

    def __init__(self,
                 dim: int,
                 depth: int = 2,
                 dim_context: int | None = None,
                 num_latents: int = 32,
                 dim_head: int = 64,
                 heads: int = 8,
                 ff_mult: int = 4,
                 use_flash_attn: bool = False) -> None:
        """参照特徴をlearned queryに集約するための層と設定を構築する。

        Args:
            dim: 特徴の次元数。
            depth: attentionとFFNを繰り返す回数。
            dim_context: 参照特徴の次元数。Noneでは内部特徴次元数と同じ。
            num_latents: 学習可能な集約queryの個数。
            dim_head: 1 head当たりの特徴次元数。
            heads: attention head数。
            ff_mult: FFNの中間次元を決める倍率。
            use_flash_attn: Perceiver attentionでSDPA経路を使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(
            dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        use_flash=use_flash_attn,
                        cross_attn_include_queries=True,
                    ),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

        self.norm = RMSNorm(dim)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """参照特徴をlearned queryに集約する。

        Args:
            x: 参照特徴[B,T,Dcontext]。Bは発話数、Tは参照長、Dcontextは参照特徴次元数。
            mask: 参照の有効位置mask[B,M]。Bは発話数、Mはkeyの位置数。Trueが参照可能。Noneでは全位置。

        Returns:
            集約特徴[B,N,D]。Bは発話数、Nは学習可能なquery数、Dは出力の特徴次元数。
        """
        batch = x.shape[0]

        x = self.proj_context(x)

        # 学習query[N,D]を発話ごとに展開する。[N,D]→[B,N,D]。Bは発話数、Nはquery数、Dは内部特徴の次元数。
        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class Attention(nn.Module):
    """Q/K/V射影とattentionによる位置間集約を計算する。
    """

    def __init__(self,
                 dim: int,
                 *,
                 dim_context: int = None,
                 causal: bool = False,
                 dim_head: int = 64,
                 heads: int = 8,
                 dropout: float = 0.0,
                 use_flash: bool = False,
                 cross_attn_include_queries: Any = False) -> None:
        """Q/K/V射影とattentionによる位置間集約を計算するための層と設定を構築する。

        Args:
            dim: 特徴の次元数。
            dim_context: 参照特徴の次元数。Noneでは内部特徴次元数と同じ。
            causal: 未来の時刻を参照しない処理にするか。
            dim_head: 1 head当たりの特徴次元数。
            heads: attention head数。
            dropout: dropout確率。
            use_flash: PyTorchのSDPA経路を使うか。
            cross_attn_include_queries: 参照key/valueへ現在のqueryも連結するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal,
                             dropout=dropout,
                             use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Q/K/V射影とattentionによる位置間集約を計算する。

        query[B,N,D]とcontext[B,M,D]→出力[B,N,D]。B=発話数、N=query数、M=context位置数、
        D=特徴次元数。

        Args:
            x: query入力[B,N,D]。Bは発話数、Nはquery数、Dは内部特徴次元数。
            context:参照特徴[B,M,Dcontext]。Bは発話数、Mは参照長、Dcontextは参照特徴次元数。
                Noneではself-attention。
            mask: 参照の有効位置mask[B,M]。Bは発話数、Mはkeyの位置数。Trueが参照可能。Noneでは全位置。

        Returns:
            query位置の特徴[B,N,D]。Bは発話数、Nはquery数、Dは出力の特徴次元数。
        """
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            # queryも参照対象へ含める。[B,N,D]と[B,M,D]→[B,N+M,D]。Nはquery数、Mは参照長。
            context = torch.cat((x, context), dim=-2)

        # query[B,N,D]と参照[B,M,D]を射影する。Qは[B,N,H*d]、K/Vは各[B,M,H*d]。
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        # head軸を分離する。Qは[B,H,N,d]、K/Vは[B,H,M,d]。Hはhead数、
        # dはattention headごとの特徴次元数。
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                      (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        # headを連結して出力射影へ渡す。[B,H,N,d]→[B,N,H*d]。
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
