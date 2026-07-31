"""Gated MLA -- Kimi K3 論文 §2.1.2 (sec:gated-mla) の実装。

Multi-head Latent Attention (MLA, DeepSeek-V2) をベースに、Kimi K3 は
    (1) 全レイヤーで NoPE (No Position Encoding) を採用 (位置情報は KDA 層の
        再帰的減衰ゲートが暗黙的に運ぶため、MLA 層は明示的な位置符号化を持たない)
    (2) 入力依存・チャネルワイズのフルランク出力ゲート (Eq.gated-mla-output)
を追加している。3層の KDA + 1層の Gated MLA というブロックがバックボーン全体で
繰り返され (Hybrid Attention, §2.1)、さらにバックボーン末尾に追加の Gated MLA 層が
置かれることで最終層は必ずグローバルアテンションになる (§2.1 本文)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KimiRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


class GatedMLA(nn.Module):
    """Gated Multi-head Latent Attention (NoPE)。

    形状の記法:
        B            : バッチサイズ
        T            : シーケンス長
        H            : アテンションヘッド数 (num_heads)
        d            : モデル隠れ次元 (hidden_size)
        d_c          : KV 圧縮後の潜在次元 (kv_lora_rank)  -- KVキャッシュとして保持する次元
        d_qc         : クエリ圧縮後の潜在次元 (q_lora_rank)
        d_h          : ヘッドあたりの (nope) コンテンツ次元 (qk_nope_head_dim)
        d_r          : ヘッドあたりの (rope) 次元。K3 は NoPE なので rope 部分の
                       位置回転は適用しないが、DeepSeek-V2 由来の設計を踏襲し
                       次元自体は残す (qk_rope_head_dim)。
        d_v          : ヘッドあたりの value 次元 (v_head_dim)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.q_head_dim ** (-0.5)

        # --- クエリの低ランク圧縮: x -(down)-> c_q -(up)-> q_states ---
        self.q_down_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_down_norm = KimiRMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=False)

        # --- KVの低ランク圧縮: x -(down)-> c_t = W_c x  (キャッシュされる潜在ベクトル) ---
        self.kv_down_proj = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_down_norm = KimiRMSNorm(kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        # --- Eq.(gated-mla-output): フルランク出力ゲート ---
        self.g_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_latent_kv: bool = False,
    ):
        """
        Args:
            hidden_states: (B, T, d)
            attn_mask: (B, 1, T, T) の加算マスク (causal + padding)。None なら causal のみ。
            return_latent_kv: True の場合、推論時にキャッシュすべき圧縮潜在ベクトル c_t を返す
                (実運用では c_t : (B, T, d_c + d_r) のみを保持すれば良く、これが
                 MLA が KV キャッシュを削減できる理由である)。
        Returns:
            out: (B, T, d)
            latent_kv (optional): (B, T, kv_lora_rank + qk_rope_head_dim)
        """
        B, T, _ = hidden_states.shape

        # --- Query ---
        q = self.q_up_proj(self.q_down_norm(self.q_down_proj(hidden_states)))
        q = q.view(B, T, self.num_heads, self.q_head_dim).transpose(1, 2)  # (B, H, T, d_h+d_r)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # NoPE: 回転位置埋め込みは適用しない。q_rope はそのまま「位置非依存の追加チャネル」として使う。

        # --- Key / Value (低ランク圧縮からの復元) ---
        compressed_kv = self.kv_down_proj(hidden_states)  # (B, T, d_c + d_r)
        latent_c, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_up_proj(self.kv_down_norm(latent_c))
        kv = kv.view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)  # (B,H,T,d_h) / (B,H,T,d_v)

        # k_rope は全ヘッドで共有 (MQA的に1本だけ計算し、ヘッド次元へブロードキャスト)
        k_rope = k_rope.view(B, 1, T, self.qk_rope_head_dim).expand(-1, self.num_heads, -1, -1)

        query_states = torch.cat([q_nope, q_rope], dim=-1)  # (B, H, T, d_h+d_r)
        key_states = torch.cat([k_nope, k_rope], dim=-1)    # (B, H, T, d_h+d_r)

        # --- スケール済みドット積アテンション (グローバル, causal) ---
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling  # (B,H,T,T)
        causal = torch.triu(torch.ones(T, T, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = F.softmax(scores.float(), dim=-1).to(query_states.dtype)

        attn_out = torch.matmul(probs, v)  # (B, H, T, d_v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.num_heads * self.v_head_dim)

        # --- Eq.(gated-mla-output): y_t = W_o[Sigmoid(W_g x_t) ⊙ õ_t] ---
        gate = torch.sigmoid(self.g_proj(hidden_states))
        attn_out = gate * attn_out
        out = self.o_proj(attn_out)

        if return_latent_kv:
            return out, compressed_kv
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d = 2, 20, 128
    layer = GatedMLA(
        hidden_size=d,
        num_heads=8,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
    )
    x = torch.randn(B, T, d)
    out, latent_kv = layer(x, return_latent_kv=True)
    print("output shape:", out.shape)          # (2, 20, 128)
    print("latent kv cache shape:", latent_kv.shape)  # (2, 20, 32+4=36)  <- フルKV(8*16*2=256次元)よりずっと小さい

    # 因果性チェック: 未来のトークンを変えても過去の出力は変化しないこと
    x2 = x.clone()
    x2[:, -1] += 5.0
    out2 = layer(x2)
    print("causal check max diff on first T-1 tokens:", (out[:, :-1] - out2[:, :-1]).abs().max().item())
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5)
    print("Gated MLA causality OK")
