"""Kimi K3 全体のメインフロー -- 論文 §2 (Model Architecture) 全体の統合実装。

テキスト+画像入力からロジット出力までの一連の流れを、これまでのファイルで実装した
各コンポーネントを組み合わせて再現する:

    入力: input_ids (テキスト+画像プレースホルダトークン), images (可変解像度)
        │
        ▼
    [native_vision.py] MoonViT-V2 で画像 -> 視覚トークン列 (d_llm次元)
        │
        ▼
    テキスト埋め込みの画像プレースホルダ位置に視覚トークンを埋め込む (トークン列の混合)
        │
        ▼
    Hybrid Attention Backbone を L 層 (K3実値 L=93) 通す:
        3層の KDA (kda_attention.py) + 1層の Gated MLA (gated_mla.py) の
        パターンを繰り返し、末尾にもう1層 Gated MLA を追加 (§2.1 本文)
        各層の attention 出力後に Stable LatentMoE (stable_latent_moe.py) を適用
        層をまたぐ残差は Block Attention Residuals (attention_residuals.py) で集約
        │
        ▼
    最終 RMSNorm -> lm_head -> logits: (B, T, vocab_size)

形状の記法 (このファイル全体で共通):
    B       : バッチサイズ (Block AttnRes の都合上、本実装は B=1 を前提とする。
              下記 KimiK3Backbone の docstring 参照)
    T       : シーケンス長 (テキスト+視覚トークンの合計)
    d       : モデル隠れ次元 (hidden_size, K3実値 7168)
    V       : 語彙サイズ (K3実値 163840)
    L       : バックボーンの層数 (K3実値 93 = 69 KDA + 24 Gated MLA)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from attention_residuals import BlockAttentionResidual, KimiRMSNorm
from gated_mla import GatedMLA
from kda_attention import KimiDeltaAttention
from native_vision import MoonViTV2
from stable_latent_moe import SharedExpertMLP, StableLatentMoE


@dataclass
class KimiK3Config:
    """K3実値をコメントに残しつつ、動作確認しやすい小規模な値を既定値とする。

    K3実値は公式重みの config.json (huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
    から直接確認した値であり、論文本文からの推定ではない。
    """

    vocab_size: int = 4096          # K3実値: 163840
    hidden_size: int = 64           # K3実値: 7168
    num_layers: int = 8             # K3実値: 93 (69 KDA + 24 Gated MLA)
    kda_group_size: int = 3         # 3層のKDAごとに1層のGated MLA (§2.1, 3:1 mixing ratio)
                                     # -> 周期は kda_group_size+1=4 (KDA,KDA,KDA,MLA の繰り返し)
    first_k_dense_replace: int = 1  # K3実値: 1 (最初の layer_idx=0 は MoE ではなく密な FFN)

    # KDA
    kda_num_heads: int = 4          # K3実値: 96
    kda_head_dim: int = 16          # K3実値: 128

    # Gated MLA
    mla_num_heads: int = 4          # K3実値: 96
    mla_q_lora_rank: int = 48       # K3実値: 1536
    mla_kv_lora_rank: int = 32      # K3実値: 512
    mla_qk_nope_head_dim: int = 12  # K3実値: 128
    mla_qk_rope_head_dim: int = 4   # K3実値: 64
    mla_v_head_dim: int = 16        # K3実値: 128

    # Stable LatentMoE
    moe_latent_dim: int = 32        # K3実値: 3584 (= routed_expert_hidden_size)
    moe_num_routed_experts: int = 16  # K3実値: 896
    moe_num_experts_per_token: int = 4  # K3実値: 16
    moe_num_shared_experts: int = 2   # K3実値: 2
    moe_routed_ffn_dim: int = 48    # K3実値: 3072 (moe_intermediate_size)
    moe_shared_ffn_dim: int = 96    # 密な first-k 層にも流用 (dense_ffn_dim, K3実値: 33792)

    # AttnRes
    attn_res_block_size: int = 3    # K3実値: 12 (S=12, N=8 blocks)

    # Native Vision (K3実値: 全て公式 vision_config と一致させてある)
    vision_patch_size: int = 14     # K3実値: 14
    vision_hidden_dim: int = 32     # K3実値: 1024 (vt_hidden_size)
    vision_num_layers: int = 2      # K3実値: 27
    vision_num_heads: int = 4       # K3実値: 12
    vision_mlp_dim: int = 64        # K3実値: 4096
    media_placeholder_token_id: int = 3  # K3実値: 163605


class KimiK3DecoderLayer(nn.Module):
    """1層分のデコーダブロック: (KDA または Gated MLA) -> Stable LatentMoE, Block AttnRes 付き。

    Block AttnRes (Eq.attnres-block) はトークンごとに独立な演算 (層をまたぐ「深さ方向」の
    アテンション) なので、このクラスの内部では系列トークンを (N=B*T, d) にフラット化して
    扱う。一方 KDA/Gated MLA は系列方向 (T軸) の因果構造に依存するため、self_attn の
    呼び出し直前だけ (B, T, d) に戻す。
    """

    def __init__(self, config: KimiK3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_size

        # 3:1 の Hybrid Attention パターン (周期 = kda_group_size+1 = KDA,KDA,KDA,MLA の繰り返し)。
        # さらに、最終層 (layer_idx == num_layers-1) は周期に関わらず必ず Gated MLA にする
        # (§2.1 本文: "An additional Gated MLA layer is placed at the end of the backbone,
        # ensuring that the final layer always performs global attention"。実際に config.json の
        # full_attn_layers は末尾が [..., 92, 93] と2層連続しており、通常の周期パターンでは
        # 生成されない末尾レイヤーが追加で挿入されていることを確認済み)。
        cycle_length = config.kda_group_size + 1
        self.is_mla = (layer_idx == config.num_layers - 1) or ((layer_idx + 1) % cycle_length == 0)

        # first_k_dense_replace: 最初の数層は MoE ではなく密な (routing なしの) FFN を使う
        self.is_dense_ffn = layer_idx < config.first_k_dense_replace
        if self.is_mla:
            self.self_attn = GatedMLA(
                hidden_size=d,
                num_heads=config.mla_num_heads,
                q_lora_rank=config.mla_q_lora_rank,
                kv_lora_rank=config.mla_kv_lora_rank,
                qk_nope_head_dim=config.mla_qk_nope_head_dim,
                qk_rope_head_dim=config.mla_qk_rope_head_dim,
                v_head_dim=config.mla_v_head_dim,
            )
        else:
            self.self_attn = KimiDeltaAttention(
                hidden_size=d, num_heads=config.kda_num_heads, head_dim=config.kda_head_dim
            )

        if self.is_dense_ffn:
            # first_k_dense_replace: ルーティングなしの通常 FFN (SharedExpertMLP を1個だけ流用)
            self.moe = SharedExpertMLP(hidden_size=d, ffn_dim=config.moe_shared_ffn_dim)
        else:
            self.moe = StableLatentMoE(
                hidden_size=d,
                latent_dim=config.moe_latent_dim,
                num_routed_experts=config.moe_num_routed_experts,
                num_experts_per_token=config.moe_num_experts_per_token,
                num_shared_experts=config.moe_num_shared_experts,
                routed_ffn_dim=config.moe_routed_ffn_dim,
                shared_ffn_dim=config.moe_shared_ffn_dim,
            )
        self.input_layernorm = KimiRMSNorm(d)
        self.post_attention_layernorm = KimiRMSNorm(d)

        self.attn_res_block_size = config.attn_res_block_size
        self.pre_attn_res = BlockAttentionResidual(d)   # attention直前に読み出す擬似クエリ w_l
        self.post_ffn_res = BlockAttentionResidual(d)   # FFN直前に読み出す擬似クエリ w_l

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        seq_len: int,
    ):
        """
        Args:
            prefix_sum:    (N, d)  Eq.(attnres-block) の部分和 b_n^{i-1} (N = B*T)
            block_residual:(N, M, d)  確定済みブロック代表 [b_0, ..., b_{n-1}]
            seq_len: T (self_attn 呼び出し時に (N,d) -> (B,T,d) へ戻すために必要)
        Returns:
            new_prefix_sum:    (N, d)
            new_block_residual:(N, M', d)
        """
        N, d = prefix_sum.shape
        B = N // seq_len

        # --- Eq.(attnres-block): attention直前の入力を過去ブロックからの読み出しで構成 ---
        if block_residual.shape[1] > 0:
            x = self.pre_attn_res(prefix_sum, block_residual)
        else:
            x = prefix_sum  # まだ確定ブロックが無い最初のブロックの最初の層 (softmaxが自明)

        # ブロック境界 (layer_idx が S の倍数): 現在の部分和を新しいブロック代表として確定
        is_boundary = (self.layer_idx % self.attn_res_block_size) == 0
        if is_boundary:
            block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
            residual_base = prefix_sum.new_zeros(N, d)
        else:
            residual_base = prefix_sum

        h = self.input_layernorm(x).view(B, seq_len, d)
        attn_out = self.self_attn(h).reshape(N, d)
        prefix_sum = residual_base + attn_out

        # --- FFN直前も同様に過去ブロックからの読み出しで構成 ---
        y = self.post_ffn_res(prefix_sum, block_residual)
        h2 = self.post_attention_layernorm(y).view(B, seq_len, d)
        moe_out = self.moe(h2).reshape(N, d)
        prefix_sum = prefix_sum + moe_out

        return prefix_sum, block_residual


class KimiK3Backbone(nn.Module):
    """埋め込み層 + L 層のデコーダ + 出力 AttnRes + 最終 RMSNorm。

    NOTE: Block AttnRes はトークンを (N=B*T, d) にフラット化して扱うが、
    KDA/Gated MLA の内部では改めて (B, T, d) に戻す (`KimiK3DecoderLayer` 参照)。
    本デモでは実装を単純にするため B=1 を前提とする (公式実装はバッチ次元を保ったまま
    トークンレベル演算をブロードキャストで処理する)。
    """

    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [KimiK3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.output_res = BlockAttentionResidual(config.hidden_size)
        self.norm = KimiRMSNorm(config.hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds: (B, T, d)  テキスト+視覚トークンの混合埋め込み (B=1 を想定)
        Returns:
            (B, T, d)  最終隠れ状態
        """
        B, T, d = inputs_embeds.shape
        assert B == 1, "このデモ実装は B=1 (単一シーケンス) のみをサポートする"
        x = inputs_embeds.reshape(-1, d)  # (N, d), N = B*T

        prefix_sum = x
        block_residual = x.new_zeros(x.shape[0], 0, d)
        for layer in self.layers:
            prefix_sum, block_residual = layer(prefix_sum, block_residual, seq_len=T)

        out = self.output_res(prefix_sum, block_residual)
        out = self.norm(out)
        return out.view(B, T, d)


class KimiK3ForConditionalGeneration(nn.Module):
    """MoonViT-V2 + Hybrid Attention Backbone + lm_head の統合モデル。"""

    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config
        self.vision_tower = MoonViTV2(
            patch_size=config.vision_patch_size,
            hidden_dim=config.vision_hidden_dim,
            num_layers=config.vision_num_layers,
            num_heads=config.vision_num_heads,
            mlp_dim=config.vision_mlp_dim,
            llm_hidden_dim=config.hidden_size,
        )
        self.model = KimiK3Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _merge_text_and_vision(
        self, input_ids: torch.Tensor, images: list[torch.Tensor] | None
    ) -> torch.Tensor:
        """テキスト埋め込みの `media_placeholder_token_id` 位置を視覚トークンで置き換える。

        簡略化のため B=1 (単一シーケンス) を前提とし、画像トークン数はプレースホルダ
        トークン数と一致する想定 (公式実装 `_merge_input_ids_with_image_features` の
        複数バッチ・左パディング対応を単純化したもの)。

        Args:
            input_ids: (1, T)
            images: 画像/動画のリスト (無ければ None)
        Returns:
            inputs_embeds: (1, T, d)
        """
        embeds = self.model.embed_tokens(input_ids)  # (1, T, d)
        if not images:
            return embeds

        visual_tokens = self.vision_tower(images)  # list of (N_v_i, d)
        visual_tokens = torch.cat(visual_tokens, dim=0)  # (sum N_v_i, d)

        placeholder_mask = (input_ids[0] == self.config.media_placeholder_token_id)
        n_placeholders = placeholder_mask.sum().item()
        assert n_placeholders == visual_tokens.shape[0], (
            f"画像プレースホルダ数 ({n_placeholders}) と視覚トークン数 "
            f"({visual_tokens.shape[0]}) が一致しません"
        )
        embeds = embeds.clone()
        embeds[0, placeholder_mask] = visual_tokens.to(embeds.dtype)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        images: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (1, T) int64  テキストトークンID列 (画像は media_placeholder_token_id で埋める)
            images: 可変解像度画像/動画のリスト、または None (テキストのみ)
        Returns:
            logits: (1, T, vocab_size)
        """
        inputs_embeds = self._merge_text_and_vision(input_ids, images)  # (1, T, d)
        hidden_states = self.model(inputs_embeds)  # (1, T, d)
        logits = self.lm_head(hidden_states)  # (1, T, vocab_size)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    config = KimiK3Config()
    model = KimiK3ForConditionalGeneration(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params (toy-scale model): {n_params:,}")

    # --- テキストのみの forward ---
    T = 20
    input_ids = torch.randint(4, config.vocab_size, (1, T))  # 4番以降はプレースホルダを避ける
    logits = model(input_ids)
    print("text-only logits shape:", logits.shape)  # (1, 20, 4096)
    assert logits.shape == (1, T, config.vocab_size)

    # --- テキスト+画像混在の forward ---
    image = torch.randn(1, 28, 28, 3)  # 1枚, 2x2 patch grid -> merge(2,2)で1視覚トークン
    input_ids2 = input_ids.clone()
    input_ids2[0, 5] = config.media_placeholder_token_id  # 1個のプレースホルダ = 1視覚トークンに対応
    logits2 = model(input_ids2, images=[image])
    print("text+image logits shape:", logits2.shape)  # (1, 20, 4096)
    assert logits2.shape == (1, T, config.vocab_size)

    print("KimiK3 main_flow OK")
