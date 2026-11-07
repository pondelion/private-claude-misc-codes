"""Fish Audio S2 Dual-ARの学習可能な実装。Drama3/MoEとは別モデル。

Slow ARは時刻方向、Fast ARは同時刻のcodebook深さ方向に因果的。
公式llama.pyのMCF、GQA、QK norm、RoPE、norm_fastlayer_inputを明示する。

2段ARの意味:
    1frameをN個のcodec IDで表す。Slowは発話の時間方向に進み、本文と過去の
    全book情報から次frameの先頭semantic IDを予測する。Fastはそのframeの
    Slow状態を受け、book方向に進んで残りIDを作る。時間とbook深さは別の軸。
    MCFは過去frameの複数bookをembedding加算でSlow入力へ融合する処理。

入力とGT:
    main_ids:[B,S]はtextまたは先頭semanticのglobal tokenizer ID。
    codes:[B,S,N]はbook内ID。B=発話数、S=text/audio混在位置数、N=book数。
    text位置のcodesはゼロでよく、main IDのsemantic band判定で融合を無効にする。
    data.pyがこれらのtrackとsupervise/valid maskを構築する。
    forwardはSlow次token CEとFast book CEのdictを返し、lossでbackwardする。

使い方:
    config_from_dictでDualARConfigを構築し、DualAR(config)を作る。
    学習例はtraining_example.py。生成はeval()後generateへ単一発話prefixを
    渡す。返るIDs[1,T,N]をcodec向け[1,N,T]へ転置して波形化する。
    その接続はmain_flow.pyに示す。生成時は全prefixを再計算し、KV cacheはない。
    PyTorchとeinopsを使用。語彙band/境界ID/各bookの有効語彙を実資産と揃える。
    本ファイルはS2-Pro向け構造で、後発Drama3やMoEモデルは実装対象ではない。
"""
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    """Slow/Fast共通のGQA Transformer構成。dimとheads*head_dimは別設定。

    Attributes:
        dim: 隠れ状態の特徴次元数D。
        intermediate: SwiGLUの中間特徴次元数I。
        layers: Transformer層数。
        heads: queryのhead数H。
        kv_heads: key/valueの共有head数G。
        head_dim: 1 headの特徴次元数d。
        rope_base: RoPEの周波数尺度の底。
        eps: RMS正規化の安定化定数。
        qk_norm: queryとkeyをhead内で正規化するか。
        qkv_bias: QKV射影へbiasを付けるか。
        output_bias: attention出力射影へbiasを付けるか。
        dropout: attentionのdropout確率。
    """

    dim: int = 512
    intermediate: int = 1536
    layers: int = 8
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    rope_base: float = 10000.0
    eps: float = 1e-5
    qk_norm: bool = True
    qkv_bias: bool = False
    output_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """正値、Q/KV head数の割り切れ、RoPEの偶数attention headごとの特徴次元数を確認する。

        Returns:
            None。未指定の設定値をインスタンスへ補完する。
        """
        if min(self.dim, self.intermediate, self.layers, self.heads,
               self.kv_heads, self.head_dim) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("invalid GQA or rotary head dimensions")


def _default_fast_config() -> TransformerConfig:
    """Fast AR用の既定Transformer構成を独立したinstanceで返す。

    Returns:
        独立したFast AR用のTransformerConfig。
    """
    return TransformerConfig(
        dim=256,
        intermediate=768,
        layers=4,
        heads=4,
        kv_heads=2,
        head_dim=64,
    )


@dataclass
class DualARConfig:
    """Slow/Fast構成、text/semantic語彙band、10book融合の設定を保持する。

    Attributes:
        vocab_size: 本文または共通tokenizerの語彙数。
        semantic_begin: 共通語彙におけるsemantic ID領域の開始位置。
        semantic_size: semantic codebookの有効語彙数。
        codebook_size: Fast分類と音声埋め込みで確保する語彙数。
        num_codebooks: 音声1フレームのcodebook数N。
        eos_id: 音声生成の終了token ID。
        tie_embeddings: 共通token埋め込みと出力headの重みを共有するか。
        scale_mcf: 音声位置の融合特徴をsqrt(N+1)で割るか。
            Nはbook数、追加の1は共通token埋め込みの寄与。
        norm_fast_input: Slow隠れ状態をFast入力へ渡す前に正規化するか。
        slow: 時刻方向のSlow ARのTransformer構成。
        fast: 同一時刻のcodebook方向のFast ARのTransformer構成。
    """

    # 既定は構造理解用の小型構成。論文のQwen3-4B構成値ではない。
    # 実際のscratch学習ではtokenizer/config.jsonに合わせて明示指定する。
    vocab_size: int = 36100
    semantic_begin: int = 32000
    semantic_size: int = 4096
    codebook_size: int = 4096  # Fast/MCF側の共通embedding語彙数。
    num_codebooks: int = 10
    eos_id: int = 2
    tie_embeddings: bool = True
    scale_mcf: bool = True
    norm_fast_input: bool = True
    slow: TransformerConfig = field(default_factory=TransformerConfig)
    fast: TransformerConfig = field(default_factory=_default_fast_config)


def config_from_dict(values: dict[str, Any]) -> DualARConfig:
    """JSON辞書を、slow/fastも型付き設定にしたDualARConfigへ変換する。

    Args:
        values: JSONから読み込んだ構成設定の辞書。

    Returns:
        入れ子のTransformer設定も型付きにしたモデル設定。
    """
    values = dict(values)
    for key in ("slow", "fast"):
        if key in values:
            values[key] = TransformerConfig(**values[key])
    return DualARConfig(**values)


class RMSNorm(nn.Module):
    """hiddenまたはQ/Kの最後の特徴軸を正規化するRMSNorm。"""

    def __init__(self, dim: int, eps: float) -> None:
        """正規化する特徴次元数dimと、分母安定化係数epsを指定する。

        Args:
            dim: 特徴の次元数。
            eps: 正規化の分母へ加える安定化定数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden[B,S,D]またはQ/K[B,H,S,d]を同shape/dtypeへ返す。

        B=発話/音声frame batch数、S=時間/深さ位置数、D=隠れ状態の次元数、H=attention head数、
        d=attention headごとの特徴次元数。最後の軸がconstructorのdimに対応する。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。

        Returns:
            入力と同shape・dtypeの正規化特徴。末尾の特徴軸だけを正規化する。
        """
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(hidden.dtype) * self.weight


class Attention(nn.Module):
    """Q/K normと隣接対RoPEを持つcausal Grouped-Query Attention。"""

    def __init__(self, config: TransformerConfig) -> None:
        """query/KV head数、attention headごとの特徴次元数、bias/dropout/正規化の設定から層を作る。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        if config.heads % config.kv_heads or config.head_dim % 2:
            raise ValueError("invalid GQA geometry")
        self.qkv = nn.Linear(config.dim, (config.heads + 2 * config.kv_heads) *
                             config.head_dim,
                             bias=config.qkv_bias)
        self.output = nn.Linear(config.heads * config.head_dim,
                                config.dim,
                                bias=config.output_bias)
        self.q_norm = RMSNorm(config.head_dim,
                              config.eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim,
                              config.eps) if config.qk_norm else nn.Identity()

    def rope(self, values: torch.Tensor) -> torch.Tensor:
        """Q/K[B,H,S,d]の隣接成分対を回し、同shapeへ返す。

        B=発話/音声frame batch数、H=QまたはKV head数、S=時間/深さ位置数、d=attention headごとの特徴次元数。
        内部で[B,H,S,d/2,2]へ分け、2成分を複素数対として扱う。

        Args:
            values: 回転前のqueryまたはkey[B,H,S,d]。Bは発話またはpatch数、Hはhead数、Sは位置数、
                dはattention headごとの特徴次元数。

        Returns:
            回転後のqueryまたはkey[B,H,S,d]。Bは発話またはpatch数、Hはhead数、Sは位置数、
            dはattention headごとの特徴次元数。
        """
        # 公式llama.pyのinterleaved RoPE。Qwenのhalf-split重みをそのまま
        # 読む場合は変換が必要なので、scratchモデルに公式checkpoint直loadはしない。
        length, width = values.size(-2), values.size(-1)
        frequency = self.config.rope_base**(
            -torch.arange(0, width, 2, device=values.device).float() / width)
        angle = torch.outer(
            torch.arange(length, device=values.device).float(), frequency)
        pairs = rearrange(values.float(), "b h t (d r) -> b h t d r", r=2)
        real, imaginary = pairs.unbind(-1)  # 各[B,H,S,d/2]。
        rotated = torch.stack((real * angle.cos() - imaginary * angle.sin(),
                               real * angle.sin() + imaginary * angle.cos()),
                              dim=-1)
        return rearrange(rotated, "b h t d r -> b h t (d r)").to(values.dtype)

    def forward(self,
                hidden: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """hidden[B,S,D]を過去位置のattentionで更新し、同shapeで返す。

        B=発話/音声frame batch数、S=時間/深さ位置数、D=隠れ状態の次元数。valid:[B,S]は有効key mask。
        H=query head数、G=KV head数、d=attention headごとの特徴次元数。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            attention出力[B,S,D]。Bは発話またはpatch数、Sは位置数、Dは内部特徴の次元数。
        """
        cfg = self.config
        # [B,S,D]->[B,S,(H+2G)*d]をQ:[B,S,H*d], K/V:[B,S,G*d]へ分ける。
        query, key, value = self.qkv(hidden).split(
            (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim,
             cfg.kv_heads * cfg.head_dim),
            dim=-1)
        # Q:[B,H,S,d]、K/V:[B,G,S,d]。normはattention headごとの特徴次元数dだけに適用する。
        query = self.q_norm(
            rearrange(query, "b t (h d) -> b h t d", h=cfg.heads))
        key = self.k_norm(
            rearrange(key, "b t (h d) -> b h t d", h=cfg.kv_heads))
        value = rearrange(value, "b t (h d) -> b h t d", h=cfg.kv_heads)
        query, key = self.rope(query), self.rope(key)
        # 共有するKV headをquery head数へ展開。[B,G,S,d]->[B,H,S,d]。
        key = key.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        value = value.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        mask = None
        if valid is not None:
            position = torch.arange(hidden.size(1), device=hidden.device)
            # [S,S]の因果条件と[B,1,1,S]のkey有効条件から[B,1,S,S]を作る。
            mask = (position[:, None]
                    >= position[None, :])[None, None] & valid[:, None, None]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=mask is None,
            dropout_p=cfg.dropout if self.training else 0.0)
        # [B,H,S,d]->[B,S,H*d]->Linear(H*d,D)->[B,S,D]。
        return self.output(rearrange(attended, "b h t d -> b t (h d)"))


class Layer(nn.Module):
    """pre-RMSNorm causal attentionとSwiGLU FFNの残差block。"""

    def __init__(self, config: TransformerConfig) -> None:
        """attentionとFFNを独立パラメータで構築する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.dim, config.eps), RMSNorm(
            config.dim, config.eps)
        self.attention = Attention(config)
        self.gate = nn.Linear(config.dim, config.intermediate, bias=False)
        self.up = nn.Linear(config.dim, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.dim, bias=False)

    def forward(self,
                hidden: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """[B,S,D]を同shapeで更新する。

        B=batch、S=時間/深さ位置数、D=隠れ状態の次元数。valid:[B,S]は任意のpadding key mask。
        FFN内部のI=config.intermediate。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            残差加算した隠れ状態[B,S,D]。Bは発話またはpatch数、Sは位置数、Dは内部特徴の次元数。
        """
        hidden = hidden + self.attention(self.norm1(hidden), valid)
        normalized = self.norm2(hidden)
        # 2経路[B,S,D]->[B,S,I]を積で融合し、Dへ戻して残差加算する。
        return hidden + self.down(
            F.silu(self.gate(normalized)) * self.up(normalized))


class Stack(nn.Module):
    """Slow/Fastで別々に構築するTransformer stackと最終RMSNorm。"""

    def __init__(self, config: TransformerConfig) -> None:
        """独立した層をconfig.layers個と、出力正規化層を構築する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.layers = nn.ModuleList(
            Layer(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.dim, config.eps)

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """norm前後のhiddenを各[B,S,D]で返す。

        SlowではB=発話数、S=混在列長、D=Slow幅。FastではB=音声frame総数、S=codebook深さ位置数、D=Fast幅。
        valid:[B,S]は任意の有効key mask。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            最終正規化前と後の隠れ状態の組。各[B,S,D]。Bは発話または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
        """
        for layer in self.layers:
            hidden = layer(hidden, valid)  # [B,S,D]または[Naudio,Nbook,Dfast]。
        return hidden, self.norm(hidden)


class DualAR(nn.Module):
    """時刻方向Slow ARと、同時刻のbook深さ方向Fast ARを接続するTTS本体。"""

    def __init__(self, config: DualARConfig) -> None:
        """MCF表、Slow/Fast stack、各headをrandom構築して初期化する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        if config.semantic_begin + config.semantic_size > config.vocab_size:
            raise ValueError("semantic token band exceeds vocabulary")
        if config.semantic_size > config.codebook_size:
            raise ValueError(
                "shared Fast embedding must cover semantic code IDs")
        if (config.num_codebooks < 2 or config.semantic_begin < 0
                or not 0 <= config.eos_id < config.vocab_size
                or config.semantic_begin <= config.eos_id <
                config.semantic_begin + config.semantic_size):
            raise ValueError(
                "invalid codebook counts or EOS/semantic vocabulary overlap")
        self.text_embedding = nn.Embedding(config.vocab_size, config.slow.dim)
        self.codebook_embeddings = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.slow.dim)
        self.slow = Stack(config.slow)
        self.slow_head = nn.Linear(config.slow.dim,
                                   config.vocab_size,
                                   bias=False)
        if config.tie_embeddings:
            self.slow_head.weight = self.text_embedding.weight
        self.fast_project = (nn.Linear(config.slow.dim, config.fast.dim)
                             if config.slow.dim != config.fast.dim else
                             nn.Identity())
        self.fast_embedding = nn.Embedding(config.codebook_size,
                                           config.fast.dim)
        self.fast = Stack(config.fast)
        self.fast_head = nn.Linear(config.fast.dim,
                                   config.codebook_size,
                                   bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        """Linear/Embeddingをstd=0.02、Linear biasを0で初期化する。

        Args:
            module: 構築済みのPyTorchモジュール。

        Returns:
            None。対象モジュールの重み・バイアスを更新する。
        """
        # 公開BaseTransformerの初期化規約。巨大なstd=1 embeddingで開始しない。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed(self, main_ids: torch.Tensor,
              codes: torch.Tensor) -> torch.Tensor:
        """main_ids[B,S], codes[B,S,N] -> [B,S,Dslow]。

        N=10 codebooks。Text位置はmain embeddingだけ。audio位置ではq0がmainとbook0の独立した2表現とし
        て加算される。コードIDの0はPAD判定に使わない。B=発話数、S=text/reference/targetを含む位置数、
        Dslow=Slow 隠れ状態の次元数。

        Args:
            main_ids: 本文・semantic共通語彙のID[B,S]。Bは発話数、Sは混在列の位置数。
            codes: 音声codec ID[B,S,N]。Bは発話数、Sは本文と音声の混在列長、Nはcodebook数。

        Returns:
            融合した特徴[B,S,Ds]。Bは発話数、Sは混在列の位置数、DsはSlow幅。
        """
        cfg = self.config
        audio = ((main_ids >= cfg.semantic_begin)
                 & (main_ids < cfg.semantic_begin + cfg.semantic_size))
        # 各bookの独立tableを1表内に置くoffset[N]。ID[B,S,N]へbroadcastする。
        offsets = torch.arange(cfg.num_codebooks,
                               device=codes.device) * cfg.codebook_size
        summed = self.codebook_embeddings(codes + offsets).sum(
            dim=2)  # [B,S,N,D]->[B,S,D]
        hidden = self.text_embedding(main_ids) + summed * audio[..., None]
        if cfg.scale_mcf:
            hidden = torch.where(audio[..., None],
                                 hidden / (cfg.num_codebooks + 1)**0.5, hidden)
        return hidden

    def slow_forward(
        self,
        main_ids: torch.Tensor,
        codes: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """混在列からSlow logits[B,S,V]とFast条件hidden[B,S,Ds]を得る。

        main_ids/valid:[B,S]、codes:[B,S,N]。B=発話数、S=混在列長、V=global語彙数、Ds=Slow幅、
        N=book数。valid省略時は全位置が有効。

        Args:
            main_ids: 本文・semantic共通語彙のID[B,S]。Bは発話数、Sは混在列の位置数。
            codes: 音声codec ID[B,S,N]。Bは発話数、Sは本文と音声の混在列長、Nはcodebook数。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            共通語彙logits[B,S,V]とFast条件[B,S,Ds]の組。Bは発話数、Sは列長、Vは語彙数、DsはSlow幅。
        """
        raw, normalized = self.slow(self.embed(main_ids, codes), valid)
        return self.slow_head(
            normalized), normalized if self.config.norm_fast_input else raw

    def fast_forward(self, slow_hidden: torch.Tensor,
                     target_codes: torch.Tensor) -> torch.Tensor:
        """[Na,Dslow], GT[Na,N] -> logits[Na,N,K]。

        入力位置0=hslow、位置1=Emb(q0)、...、位置N-1=Emb(qN-2)。出力位置kでqkを予測。
        qk自身を入力に含めずteacher forcingする。Na=教師対象の音声frame総数、Dslow=Slow幅、N=book数、
        K=Fast分類語彙数。Dfast=Fast幅。frameごとの文脈は他frameのFast文脈と混ぜない。

        Args:
            slow_hidden: Slow ARの隠れ状態[Na,Ds]。Naは教師対象の音声フレーム数、DsはSlow幅。
            target_codes: GTのcodec ID[Na,N]。Naは教師対象の音声フレーム数、Nはcodebook数。

        Returns:
            codebookごとのlogits[Na,N,K]。Naは音声フレーム数、Nはbook数、KはFast語彙数。
        """
        condition = self.fast_project(
            slow_hidden)[:, None]  # [Na,Ds]->[Na,1,Dfast]。
        prior_codes = self.fast_embedding(
            target_codes[:, :-1])  # [Na,N-1,Dfast]
        # 条件1位置＋直前book N-1位置を連結。[Na,N,Dfast]。
        _, hidden = self.fast(torch.cat((condition, prior_codes), dim=1))
        return self.fast_head(hidden)  # [Na,N,K]

    def forward(
        self,
        main_ids: torch.Tensor,
        codes: torch.Tensor,
        supervise: torch.Tensor,
        valid: torch.Tensor,
        codebook_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """compact conversation[B,S]を1shift。

        system/referenceはsupervise=False。codes[B,S,N]はaudio以外ゼロでよい。
        Slowは次のmain tokenを予測し、Fastは「その次の音声frame」の全bookを同じhslowから予測する。
        main_ids/supervise/valid:[B,S]、codes:[B,S,N]、codebook_weights:[N]。
        B=発話数、S=混在列長、N=book数。superviseは教師対象、validは非padding。
        戻り値loss/slow_ce/fast_ceはscalar tensor。

        Args:
            main_ids: 本文・semantic共通語彙のID[B,S]。Bは発話数、Sは混在列の位置数。
            codes: 音声codec ID[B,S,N]。Bは発話数、Sは本文と音声の混在列長、Nはcodebook数。
            supervise: 教師対象の位置が真のmask[B,S]。Bは発話数、Sは混在列長。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。
            codebook_weights: Fast CEのcodebook別重み[N]。Nはcodebook数。
                Noneでは既定の重み。

        Returns:
            loss・slow_ce・fast_ceを格納した辞書。各値は平均損失のscalar[]。
        """
        logits, hidden = self.slow_forward(main_ids[:, :-1], codes[:, :-1],
                                           valid[:, :-1])
        targets = main_ids[:, 1:]  # [B,S-1]。Slow入力の1位置先がGT。
        mask = supervise[:, 1:] & valid[:, 1:]
        errors = F.cross_entropy(rearrange(logits.float(), "b s v -> b v s"),
                                 targets,
                                 reduction="none")
        slow_loss = (errors * mask).sum() / mask.sum().clamp_min(1)
        cfg = self.config
        audio = (mask & (targets >= cfg.semantic_begin)
                 & (targets < cfg.semantic_begin + cfg.semantic_size))
        if audio.any():
            # bool選択でB/位置軸を教師audio frame総数Naへ畳む。
            audio_gt = codes[:, 1:][audio]  # [B,S-1,N]->[Na,N]。
            fast_logits = self.fast_forward(hidden[audio], audio_gt)
            fast_errors = F.cross_entropy(rearrange(fast_logits.float(),
                                                    "a n k -> a k n"),
                                          audio_gt,
                                          reduction="none")  # [Na,N]
            weights = (fast_errors.new_ones(cfg.num_codebooks)
                       if codebook_weights is None else torch.as_tensor(
                           codebook_weights,
                           device=fast_errors.device,
                           dtype=fast_errors.dtype))
            if (weights.shape != (cfg.num_codebooks, ) or (weights < 0).any()
                    or weights.sum() <= 0):
                raise ValueError("one nonnegative weight per codebook, "
                                 "positive total required")
            fast_loss = (fast_errors * weights[None]).sum(-1).div(
                weights.sum()).mean()
        else:
            fast_loss = hidden.sum() * 0  # text-only batchにはFast教師が存在しない。
        return {
            "loss": slow_loss + fast_loss,
            "slow_ce": slow_loss,
            "fast_ce": fast_loss
        }

    @torch.no_grad()
    def generate(
        self,
        main_ids: torch.Tensor,
        codes: torch.Tensor,
        max_frames: int = 2000,
        temperature: float = 0.8,
        acoustic_vocab_size: int | None = None,
    ) -> torch.Tensor:
        """1sample。

        semanticはSlow、残りbookはFastで逐次生成。[1,T,N]。full-prefix再計算の実装。
        Fast cacheは各frameでリセットされるという意味を、ここではframe内prefixを新規に作ることで表現する。
        入力main_ids:[1,S]、codes:[1,S,N]。S=conditioning混在列長、T=新規音声frame数、N=book数。
        max_framesはTの上限。acoustic_vocab_sizeは非semantic bookの有効語彙数、
        temperatureはsampling温度。

        Args:
            main_ids: 本文・semantic共通語彙のID[B,S]。Bは発話数、Sは混在列の位置数。
            codes: 音声codec ID[B,S,N]。Bは発話数、Sは本文と音声の混在列長、Nはcodebook数。
            max_frames: 生成する音声codecフレーム数の上限。
            temperature: samplingまたは初期ノイズの温度係数。
            acoustic_vocab_size: 非semantic codebookでsamplingする有効な語彙数。

        Returns:
            生成codec ID[1,T,N]。1は単一発話、Tは生成音声フレーム数、Nはbook数。
        """
        if self.training or main_ids.size(0) != 1 or temperature <= 0:
            raise ValueError("use eval(), batch=1 and positive temperature")
        cfg, frames = self.config, []
        for _ in range(max_frames):
            logits, hidden = self.slow_forward(main_ids, codes)
            last = logits[:, -1].float() / temperature
            constrained = torch.full_like(last, -torch.inf)
            begin = cfg.semantic_begin
            end = begin + cfg.semantic_size
            constrained[:, begin:end] = last[:, begin:end]
            constrained[:, cfg.eos_id] = last[:, cfg.eos_id]
            semantic = torch.multinomial(constrained.softmax(-1), 1)
            if int(semantic) == cfg.eos_id:
                break
            # tokenizer内のsemantic IDをcodec内のID[1,1]へ戻す。
            q0 = semantic - cfg.semantic_begin
            books = [q0]
            # Slow特徴次元数DsからFast特徴次元数Dfへ射影し、book位置軸を追加する。
            # [1,Ds] -> [1,Df] -> [1,1,Df]。
            projected = self.fast_project(hidden[:, -1])
            prefix = projected[:, None]
            for depth in range(1, cfg.num_codebooks):
                # 直前book ID[1,1]を埋め、Fast深さ位置を1個伸ばす -> [1,depth+1,Df]。
                prefix = torch.cat((prefix, self.fast_embedding(books[-1])),
                                   dim=1)
                _, fast_hidden = self.fast(prefix)
                depth_logits = self.fast_head(
                    fast_hidden[:, -1]).float() / temperature
                if acoustic_vocab_size is not None:
                    depth_logits[:, acoustic_vocab_size:] = -torch.inf
                books.append(torch.multinomial(depth_logits.softmax(-1),
                                               1))  # ID[1,1]。
            frame = torch.cat(books, dim=1)  # [1,N]。各bookのIDは別辞書のindex。
            frames.append(frame)
            main_ids = torch.cat((main_ids, semantic), dim=1)
            codes = torch.cat((codes, frame[:, None]),
                              dim=1)  # [1,S,N]->[1,S+1,N]。
        # T個のframe[1,N]を時間方向へstack -> [1,T,N]。
        return torch.stack(frames, dim=1) if frames else codes.new_empty(
            1, 0, cfg.num_codebooks)
