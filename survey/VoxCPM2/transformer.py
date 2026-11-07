"""MiniCPM4のGQA/RMSNorm/SwiGLU/LongRoPE/muP residualを明示した本体。

公式 modules/minicpm4/model.py に対応する学習用full-sequence経路。
KV cacheは速度最適化のためここでは持たず、生成時も実attentionを再計算する。

何に使われるか:
    model.pyのTSLM/RALMはcausal=Trueで過去の本文・音声だけを見る。
    local_modules.pyのLocEnc/LocDiTはcausal=Falseでpatch内を相互参照する。
    同じ実装形式でも各instanceのパラメータは独立している。

入力と設定:
    Transformer(config)(embeddings, causal=True, valid=mask)で呼ぶ。
    embeddings:[B,S,D]、mask:[B,S]、出力:[B,S,D]。
    B=発話またはpatchのbatch数、S=位置数、D=config.dim。
    token ID→embeddingは上位modelの担当で、このクラスへ整数IDは渡さない。
    heads=Q head数、kv_heads=共有KV head数、head_dim=attention headごとの特徴次元数。
    GQAではKVをQ head数まで共有展開し、attention後に特徴次元数Dへ射影する。

読む順序:
    RMSNorm→Attention（RoPE→GQA→mask→SDPA）→Layer（残差とSwiGLU）
    →Transformer（層stackと最終norm）。LongRoPEの係数は使用するconfigに
    合わせる。コンストラクタは重みを取得せずrandom初期化する。
    PyTorchとeinopsが必要。本ファイルを直接実行するCLIはない。
"""
import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    """MiniCPMの特徴次元数・層数・GQA構成と、RoPE/muPの計算設定。

    Attributes:
        dim: 隠れ状態の特徴次元数D。
        intermediate: SwiGLUの中間特徴次元数I。
        layers: Transformer層数。
        heads: queryのhead数H。
        kv_heads: key/valueの共有head数G。
        head_dim: 1 headの特徴次元数d。
        eps: RMS正規化の安定化定数。
        rope_theta: RoPEの周波数尺度の底。
        max_positions: 位置表に確保する最大位置数。
        original_positions: 位置拡張前の基準となる最大位置数。
        short_factors: 短い列に用いるLongRoPE係数[d/2]。dはattention headごとの特徴次元数。
        long_factors: 長い列に用いるLongRoPE係数[d/2]。dはattention headごとの特徴次元数。
        no_rope: 回転位置埋め込みを無効にするか。
        use_mup: muPに対応した埋め込み・残差のscaleを使うか。
        scale_depth: 層数に応じた残差係数を決める基準値。
        scale_emb: 埋め込み特徴へ掛ける係数。
    """

    dim: int = 2048
    intermediate: int = 8192
    layers: int = 28
    heads: int = 16
    kv_heads: int = 2
    head_dim: int = 128
    eps: float = 1e-6
    rope_theta: float = 10000.0
    max_positions: int = 8192
    original_positions: int = 8192
    short_factors: tuple[float, ...] | None = None
    long_factors: tuple[float, ...] | None = None
    no_rope: bool = False
    use_mup: bool = True
    scale_depth: float = 1.4
    scale_emb: float = 1.0

    def __post_init__(self) -> None:
        """正値、head共有数、RoPE位置範囲と各回転対の係数を確認する。

        Returns:
            None。未指定の設定値をインスタンスへ補完する。
        """
        if min(self.dim, self.intermediate, self.layers, self.heads,
               self.kv_heads, self.head_dim, self.max_positions) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("invalid GQA or rotary head dimensions")
        if (self.original_positions <= 1
                or self.max_positions < self.original_positions):
            raise ValueError("require max_positions >= original_positions > 1")
        for factors in (self.short_factors, self.long_factors):
            if factors is not None and (len(factors) != self.head_dim // 2
                                        or min(factors) <= 0):
                raise ValueError(
                    "LongRoPE requires positive factors, one per rotary pair")


class RMSNorm(nn.Module):
    """最後の特徴軸だけをFP32でRMS正規化する層。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """正規化する特徴次元数dimと分母安定化係数epsを設定する。

        Args:
            dim: 特徴の次元数。
            eps: 正規化の分母へ加える安定化定数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B,S,D]を同shape/dtypeへ返す。

        B=batch、S=位置数、D=特徴次元数。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。

        Returns:
            入力と同shape・dtypeの正規化特徴。末尾の特徴軸だけを正規化する。
        """
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(hidden.dtype) * self.weight


class Attention(nn.Module):
    """LongRoPEまたはNoPEを使うGrouped-Query Self-Attention。"""

    def __init__(self, config: TransformerConfig) -> None:
        """query/KV head数・attention headごとの特徴次元数を独立に指定して射影層を作る。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        if config.heads % config.kv_heads or config.head_dim % 2:
            raise ValueError("invalid GQA/head dimensions")
        self.q_proj = nn.Linear(config.dim,
                                config.heads * config.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(config.dim,
                                config.kv_heads * config.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(config.dim,
                                config.kv_heads * config.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(config.heads * config.head_dim,
                                config.dim,
                                bias=False)

    def rope(self, values: torch.Tensor) -> torch.Tensor:
        """Q/K[B,H,S,d]へhalf-split LongRoPEを適用し同shapeで返す。

        B=発話数、H=QまたはKVのhead数、S=位置数、d=head_dim。short/long factor各[d/2]を使い、
        位置角[S,d]をB/Hへbroadcastする。

        Args:
            values: 回転前のqueryまたはkey[B,H,S,d]。Bは発話またはpatch数、Hはhead数、Sは位置数、
                dはattention headごとの特徴次元数。

        Returns:
            回転後のqueryまたはkey[B,H,S,d]。Bは発話またはpatch数、Hはhead数、Sは位置数、
            dはattention headごとの特徴次元数。
        """
        cfg = self.config
        length = values.size(-2)
        if length > cfg.max_positions:
            raise ValueError("configured position range exceeded")
        # 公式はmax_positionsまでcacheを作る時点でshort/long factorを選ぶ。
        factors = (cfg.long_factors if cfg.max_positions
                   > cfg.original_positions else cfg.short_factors)
        if factors is None:
            factors = torch.ones(cfg.head_dim // 2, device=values.device)
        else:
            factors = torch.tensor(factors,
                                   device=values.device,
                                   dtype=torch.float32)
        if factors.numel() != cfg.head_dim // 2:
            raise ValueError("LongRoPE factor length must equal head_dim/2")
        inverse = 1.0 / (factors * cfg.rope_theta**(torch.arange(
            0, cfg.head_dim, 2, device=values.device).float() / cfg.head_dim))
        angles = torch.outer(
            torch.arange(length, device=values.device).float(), inverse)
        angles = torch.cat((angles, angles), dim=-1)  # [S,d/2]->[S,d]。
        scale = math.sqrt(
            1 + math.log(cfg.max_positions / cfg.original_positions) /
            math.log(cfg.original_positions))
        first, second = values.float().chunk(2, dim=-1)  # 各[B,H,S,d/2]。
        rotated = torch.cat((-second, first), dim=-1)
        return ((values.float() * angles.cos() + rotated * angles.sin()) *
                scale).to(values.dtype)

    def forward(self,
                hidden: torch.Tensor,
                causal: bool = True,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """hidden[B,S,D]をGQAで混合し、同shapeへ返す。

        B=発話数、S=位置数、D=隠れ状態の次元数、H=query head数、G=KV head数、
        d=attention headごとの特徴次元数。valid:[B,S]は有効key mask。causal=Falseはpatch内用。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
            causal: 未来の位置へのattentionを禁止するか。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            attention出力[B,S,D]。Bは発話またはpatch数、Sは位置数、Dは内部特徴の次元数。
        """
        cfg = self.config
        # Dは隠れ状態の次元数、Hはquery head数、GはKV head数、dはattention headごとの特徴次元数、Sは位置数。
        query = rearrange(self.q_proj(hidden),
                          "b s (h d) -> b h s d",
                          h=cfg.heads)
        key = rearrange(self.k_proj(hidden),
                        "b s (h d) -> b h s d",
                        h=cfg.kv_heads)
        value = rearrange(self.v_proj(hidden),
                          "b s (h d) -> b h s d",
                          h=cfg.kv_heads)
        if not cfg.no_rope:
            query, key = self.rope(query), self.rope(key)
        # GQAの共有KVをH個へ展開する。[B,G,S,d]->[B,H,S,d]。
        key = key.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        value = value.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        mask = None
        if valid is not None:
            mask = valid[:, None, None, :]  # [B,S]->[B,1,1,S]。
            if causal:
                positions = torch.arange(hidden.size(1), device=hidden.device)
                # 有効keyと因果条件をANDし[B,1,S,S]のquery/key maskにする。
                mask = mask & (positions[:, None] >= positions[None, :])[None,
                                                                         None]
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  attn_mask=mask,
                                                  is_causal=causal
                                                  and mask is None)
        # [B,H,S,d]->[B,S,H*d]->Linear(H*d,D)->[B,S,D]。
        return self.o_proj(rearrange(attended, "b h s d -> b s (h d)"))


class Layer(nn.Module):
    """muP residual scaleを持つpre-RMSNorm attention/SwiGLU block。"""

    def __init__(self, config: TransformerConfig) -> None:
        """attention、FFN、正規化と層数に依存する残差係数を構築する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.attention = Attention(config)
        self.norm1, self.norm2 = RMSNorm(config.dim, config.eps), RMSNorm(
            config.dim, config.eps)
        self.gate = nn.Linear(config.dim, config.intermediate, bias=False)
        self.up = nn.Linear(config.dim, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.dim, bias=False)
        self.residual_scale = (config.scale_depth / math.sqrt(config.layers)
                               if config.use_mup else 1.0)

    def forward(self, hidden: torch.Tensor, causal: bool,
                valid: torch.Tensor | None) -> torch.Tensor:
        """[B,S,D]を同shapeで更新する。

        B=発話数、S=位置数、D=特徴次元数。causalは未来key参照の禁止、valid:[B,S]はpadding keyの除外を指定する。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。
            causal: 未来の位置へのattentionを禁止するか。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            残差加算した隠れ状態[B,S,D]。Bは発話またはpatch数、Sは位置数、Dは内部特徴の次元数。
        """
        hidden = hidden + self.residual_scale * self.attention(
            self.norm1(hidden), causal, valid)
        normalized = self.norm2(hidden)
        # 特徴[B,S,D]を値・gateの2本の[B,S,I]へ射影し、積を元の特徴次元数Dへ戻す。
        # Bは発話数、Sは位置数、IはFFNの中間特徴の次元数。
        feed_forward = self.down(
            F.silu(self.gate(normalized)) * self.up(normalized))
        return hidden + self.residual_scale * feed_forward


class Transformer(nn.Module):
    """埋込済みの列へ複数のMiniCPM blockと最終RMSNormを適用する。"""

    def __init__(self, config: TransformerConfig) -> None:
        """独立パラメータのblockをconfig.layers個積み重ねる。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            Layer(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.dim, config.eps)

    def forward(self,
                embeddings: torch.Tensor,
                causal: bool = True,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """埋込[B,S,D]を同shapeの最終hiddenへ変換する。

        B=発話/patch batch数、S=token/patch内位置数、D=config.dim。
        valid:[B,S]は任意の有効key mask。causalで長期LM/局所encoderを切り替える。

        Args:
            embeddings: 埋め込み済みの列[B,S,D]。Bは発話数またはpatch数、Sは位置数、Dは内部特徴の次元数。
            causal: 未来の位置へのattentionを禁止するか。
            valid: 有効な位置のbool mask[B,S]。Bは発話数、Sは列長。Falseはpadding。

        Returns:
            最終正規化後の隠れ状態[B,S,D]。Bは発話またはpatch数、Sは位置数、Dは内部特徴の次元数。
        """
        # Text/LocEnc/LocDiTなど、embeddingの作成元は呼び出し側で明示する。
        for layer in self.layers:
            embeddings = layer(embeddings, causal, valid)  # [B,S,D]を保持。
        return self.norm(embeddings)
