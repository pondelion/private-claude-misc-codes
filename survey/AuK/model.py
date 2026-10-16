"""
AuK - MMDiT/DiT ハイブリッドTransformerバックボーン + MLLM層融合 (モデルブロック/メインフロー)
================================================================================================

音声生成・編集のための統一指示インターフェースを実現する、FLUX-styleハイブリッド
Transformer (M個のdual-stream MMDiTブロック + N個のsingle-stream DiTブロック) と、
MLLM (多言語大規模言語モデル) の層ごとの隠れ状態を学習済み重みで融合する
"semantic layer fusion" を実装する。

論文: AuK Technical Report (arXiv:2609.08936)
      - Sec Model Design / Overall Architecture: 全体構成 (Fig.2)
      - Eq.2 (semantic-layer-fusion): c_sem = sum_l w_l * LayerNorm(h^(l))
      - Eq. condition-mapping: s^(0)=P_sem(c_sem), a^(0)=[P_ref(c_ac); P_tgt(z_t)]
      - Eq. mmdit-denoising / dit-denoising: M個のMMDiT + N個のDiTの2段構成
公式コード: gh/AuK/src/auk/model/{modules.py, flux2_edit.py, cfm_edit.py}
      これらは全て素のPyTorch実装であり、本ファイルはその構造をほぼそのまま移植
      している (AdaLayerNorm/SwiGLUFeedForward/MMDiTBlock/DiTBlockは公式
      modules.pyのロジックと同一)。唯一の違いは:
      (1) 公式は RoPE を外部パッケージ `x_transformers.RotaryEmbedding` に依存するが、
          本ファイルは標準的なRoPEを自前で実装し、依存を減らしている。
      (2) 公式 `cfm_edit.py::encode_text` は実際のQwen2.5-Omni (frozen) を使うが、
          本ファイルはモデルサイズの都合上、同じ「層融合」ロジックを小さな
          スタンドインのエンコーダに対して適用する (層融合の重み自体は本物の
          学習対象パラメータであり、エンコーダ本体だけがスタンドイン)。

============================================================
Shape Convention
============================================================
B      : バッチサイズ
N_tgt  : ターゲット(ノイズ付き)潜在のフレーム数
N_ref  : 参照音声潜在のフレーム数 (無い場合は0)
N_txt  : テキスト/意味条件のトークン数
D      : Transformerの隠れ次元
D_lat  : VAE潜在次元 (公式は64)
H      : アテンションヘッド数
D_h    : ヘッドあたりの次元
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. RoPE (公式は x_transformers.RotaryEmbedding に依存; 本ファイルは自前実装)
# ============================================================

def build_rope(seq_len: int, dim_head: int, device, dtype, theta: float = 10000.0) -> torch.Tensor:
    """標準的なRotary Position Embeddingの (cos, sin) をまとめて返す。

    ========================================
    Shape
    ========================================
    出力: (seq_len, dim_head) の cos と sin を結合した (2, seq_len, dim_head)
          (各ペアの軸を「半分ずつ複製」する rotate_half 方式)
    """
    half = dim_head // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))  # (half,)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # (seq_len,)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim_head): rotate_half方式は前後半で同じ角度を使う
    return torch.stack([freqs.cos(), freqs.sin()], dim=0).to(dtype)  # (2, seq_len, dim_head)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """RoPEを適用する。

    ========================================
    Shape
    ========================================
    入力: x (B, H, N, D_h), rope (2, N, D_h)  (cos, sinのペア)
    出力: (B, H, N, D_h)
    """
    cos, sin = rope[0], rope[1]  # (N, D_h)
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ============================================================
# 2. 基本層 (公式 modules.py から移植)
# ============================================================

class SinusPositionEmbedding(nn.Module):
    """フローステップ t in [0,1] を正弦波位置埋め込みへ写す。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # timestep: (B,) -> (B, dim)
        return self.time_mlp(self.time_embed(timestep).to(timestep.dtype))


class ConvPositionEmbedding(nn.Module):
    """畳み込みによる局所位置埋め込み (F5-TTS系で使われる設計)。"""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B, D, N) -> conv -> (B, N, D)
        return self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN (ゲート付き活性化、公式 modules.py と同一)。"""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.gate_fn = nn.SiLU()
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)         # (B, N, 2*inner_dim)
        x1, x2 = x.chunk(2, dim=-1)    # 各 (B, N, inner_dim)
        x = self.gate_fn(x1) * x2      # (B, N, inner_dim)
        return self.linear_out(x)      # (B, N, dim)


class AdaLayerNorm(nn.Module):
    """時間条件付きAdaLN。attn/mlp両方の shift/scale/gate を一括生成する (公式と同一)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # x: (B, N, D), emb: (B, D) -> 6つの (B, D) を返す
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """最終出力層用のAdaLN (mlp変調なし)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(emb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale)[:, None] + shift[:, None]


# ============================================================
# 3. アテンション (単一ストリーム + ジョイント, QK-RMSNorm + RoPE)
# ============================================================

class MultiHeadAttentionCore(nn.Module):
    """QKV射影・QK-RMSNorm・RoPE・SDPAをまとめた再利用可能なコア。単一/ジョイント両対応。"""

    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: Optional[int] = None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.context_dim = context_dim

        self.to_qkv = nn.Linear(dim, 3 * inner_dim)
        self.q_norm = nn.RMSNorm(dim_head)
        self.k_norm = nn.RMSNorm(dim_head)
        self.to_out = nn.Linear(inner_dim, dim)

        if context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * inner_dim)
            self.c_q_norm = nn.RMSNorm(dim_head)
            self.c_k_norm = nn.RMSNorm(dim_head)
            self.to_out_c = nn.Linear(inner_dim, context_dim)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, N, inner_dim) -> (B, H, N, D_h)
        B, N, _ = t.shape
        return t.view(B, N, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,                      # (B, N, D)
        c: Optional[torch.Tensor] = None,      # (B, N_txt, context_dim) : ジョイントの場合のみ
        mask: Optional[torch.Tensor] = None,   # (B, N) bool, True=有効
        c_mask: Optional[torch.Tensor] = None,  # (B, N_txt) bool
        rope: Optional[torch.Tensor] = None,     # (2, N, D_h)
        c_rope: Optional[torch.Tensor] = None,   # (2, N_txt, D_h)
    ):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)       # 各 (B, N, inner_dim)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)  # 各 (B, H, N, D_h)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)

        if c is None:
            # --- 単一ストリームのSDPA ---
            attn_mask = self._build_mask(mask, mask, q.shape[0]) if mask is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, N, D_h)
            out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)        # (B, N, inner_dim)
            return self.to_out(out)                                              # (B, N, D)

        # --- ジョイントアテンション (MMDiT) ---
        qc, kc, vc = self.to_qkv_c(c).chunk(3, dim=-1)
        qc, kc, vc = self._split_heads(qc), self._split_heads(kc), self._split_heads(vc)
        qc, kc = self.c_q_norm(qc), self.c_k_norm(kc)
        if c_rope is not None:
            qc, kc = apply_rope(qc, c_rope), apply_rope(kc, c_rope)

        q_all = torch.cat([q, qc], dim=2)  # (B, H, N+N_txt, D_h)
        k_all = torch.cat([k, kc], dim=2)
        v_all = torch.cat([v, vc], dim=2)

        joint_mask = None
        if mask is not None or c_mask is not None:
            B, N = x.shape[0], x.shape[1]
            m = mask if mask is not None else torch.ones(B, N, dtype=torch.bool, device=x.device)
            cm = c_mask if c_mask is not None else torch.ones(B, c.shape[1], dtype=torch.bool, device=x.device)
            full_mask = torch.cat([m, cm], dim=1)
            joint_mask = self._build_mask(full_mask, full_mask, B)

        out = F.scaled_dot_product_attention(q_all, k_all, v_all, attn_mask=joint_mask)  # (B, H, N+N_txt, D_h)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * self.dim_head)      # (B, N+N_txt, inner_dim)
        out_x, out_c = out[:, : x.shape[1]], out[:, x.shape[1]:]
        return self.to_out(out_x), self.to_out_c(out_c)  # (B,N,D), (B,N_txt,context_dim)

    @staticmethod
    def _build_mask(q_mask: torch.Tensor, k_mask: torch.Tensor, batch: int) -> torch.Tensor:
        # (B, N) -> SDPA用の (B, 1, 1, N) ブロードキャスト可能なbool mask
        return k_mask[:, None, None, :]


# ============================================================
# 4. MMDiT / DiT ブロック (公式 modules.py::MMDiTBlock / DiTBlock と同一構造)
# ============================================================

class DiTBlock(nn.Module):
    """単一ストリームのDiTブロック (自己注意 + SwiGLU FFN, 時間条件AdaLN)。"""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 4.0):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = MultiHeadAttentionCore(dim, heads, dim_head)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim, ff_mult)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask=None, rope=None) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        attn_out = self.attn(norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_out
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        return x


class MMDiTBlock(nn.Module):
    """dual-streamのMMDiTブロック: 意味条件 c と音響ストリーム x をジョイントアテンションで融合。"""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float = 4.0, context_dim: Optional[int] = None):
        super().__init__()
        context_dim = context_dim or dim
        self.attn_norm_c = AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = MultiHeadAttentionCore(dim, heads, dim_head, context_dim=context_dim)
        self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(context_dim, ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim, ff_mult)

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None, c_mask=None):
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, t)

        x_attn, c_attn = self.attn(norm_x, c=norm_c, mask=mask, c_mask=c_mask, rope=rope, c_rope=c_rope)

        c = c + c_gate_msa.unsqueeze(1) * c_attn
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp.unsqueeze(1) * self.ff_c(norm_c)

        x = x + x_gate_msa.unsqueeze(1) * x_attn
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp.unsqueeze(1) * self.ff_x(norm_x)
        return c, x


# ============================================================
# 5. MLLM 意味条件層融合 (論文 Eq.2, 公式 cfm_edit.py::encode_text)
# ============================================================

class SemanticLayerFusion(nn.Module):
    """多層MLLM隠れ状態を学習済み重みで融合する (ELMo的な層融合)。

    実運用ではこの前段に Qwen2.5-Omni (frozen) が入り、`all_hidden_states`
    (各層の出力) を渡す。本クラスはそのMLLM自体は含まず、融合ロジックのみを
    実装する (融合の重み `layer_weights`/`layer_scale` は本物の学習対象)。

    ========================================
    Shape
    ========================================
    入力: hidden_states List[(B, N_txt, D_llm)] (長さ num_layers)
    出力: (B, N_txt, D_llm)
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))  # softmaxのロジット
        self.layer_scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        d_llm = hidden_states[0].shape[-1]
        stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in hidden_states], dim=0)  # (L,B,N,D)
        weights = F.softmax(self.layer_weights, dim=0)                                     # (L,)
        return (stacked * weights[:, None, None, None]).sum(dim=0) * self.layer_scale       # (B,N,D)


# ============================================================
# 6. Flux2Edit バックボーン (公式 flux2_edit.py::Flux2Edit と同一構造)
# ============================================================

class AudioPromptEmbedding(nn.Module):
    """VAE潜在 (ターゲット/参照) をTransformer次元へ射影し、畳み込み位置埋め込みを加える。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.conv_pos_embed(x) + x


class Flux2EditBackbone(nn.Module):
    """M個のMMDiT + N個のDiTからなるハイブリッドTransformerバックボーン (論文Eq.mmdit/dit-denoising)。

    ========================================
    Shape
    ========================================
    入力: x (B, N_tgt, D_lat): ノイズ付きターゲット潜在
          text (B, N_txt, D_llm): MLLM融合済み意味条件
          time (B,): フロー時刻 t in [0,1]
          ref  (B, N_ref, D_lat) or None: 参照音声潜在
    出力: v_pred (B, N_tgt, D_lat): 予測されたフロー速度
    """

    def __init__(
        self, dim: int = 256, depth: int = 2, num_single_layers: int = 4,
        heads: int = 4, dim_head: int = 64, ff_mult: float = 4.0,
        latent_dim: int = 64, text_hidden_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.dim_head = dim_head

        self.mmdit_blocks = nn.ModuleList([
            MMDiTBlock(dim, heads, dim_head, ff_mult) for _ in range(depth)
        ])
        self.dit_blocks = nn.ModuleList([
            DiTBlock(dim, heads, dim_head, ff_mult) for _ in range(num_single_layers)
        ])
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        # zero-init: 公式と同様、AdaLNとproj_outを0初期化し、恒等写像から学習を始める。
        for block in self.mmdit_blocks:
            nn.init.zeros_(block.attn_norm_x.linear.weight)
            nn.init.zeros_(block.attn_norm_x.linear.bias)
            nn.init.zeros_(block.attn_norm_c.linear.weight)
            nn.init.zeros_(block.attn_norm_c.linear.bias)
        for block in self.dit_blocks:
            nn.init.zeros_(block.attn_norm.linear.weight)
            nn.init.zeros_(block.attn_norm.linear.bias)
        nn.init.zeros_(self.norm_out.linear.weight)
        nn.init.zeros_(self.norm_out.linear.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self, x: torch.Tensor, text: torch.Tensor, time: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, c_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        t = self.time_embed(time)                       # (B, D)
        c = self.txt_norm(self.txt_proj(text))            # (B, N_txt, D)

        if ref is not None and ref.shape[1] > 0:
            ref_emb = self.audio_embed(ref)                # (B, N_ref, D)
            x_emb = self.audio_embed(x)                     # (B, N_tgt, D)
            audio = torch.cat([ref_emb, x_emb], dim=1)       # (B, N_ref+N_tgt, D)
            n_ref = ref_emb.shape[1]
        else:
            audio = self.audio_embed(x)
            n_ref = 0

        n_txt = c.shape[1]
        rope_audio = build_rope(audio.shape[1], self.dim_head, x.device, x.dtype)
        rope_text = build_rope(n_txt, self.dim_head, x.device, x.dtype)

        for block in self.mmdit_blocks:
            c, audio = block(audio, c, t, mask=mask, rope=rope_audio, c_rope=rope_text, c_mask=c_mask)

        fused = torch.cat([c, audio], dim=1)                 # (B, N_txt+N_ref+N_tgt, D)
        rope_full = build_rope(fused.shape[1], self.dim_head, x.device, x.dtype)
        for block in self.dit_blocks:
            fused = block(fused, t, mask=None, rope=rope_full)

        target_part = fused[:, n_txt + n_ref:]               # (B, N_tgt, D): ターゲット位置のみ抽出
        target_part = self.norm_out(target_part, t)
        return self.proj_out(target_part)                     # (B, N_tgt, D_lat)


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== model.py Example Usage ===\n")
    torch.manual_seed(0)

    B, N_txt, N_ref, N_tgt = 2, 6, 4, 10
    D_llm, D_lat = 32, 16

    fusion = SemanticLayerFusion(num_layers=4)
    hidden_states = [torch.randn(B, N_txt, D_llm) for _ in range(4)]
    c_sem = fusion(hidden_states)
    print(f"MLLM層融合: {len(hidden_states)}層 (B,N_txt,D_llm)={tuple(hidden_states[0].shape)} "
          f"-> c_sem {tuple(c_sem.shape)}")

    backbone = Flux2EditBackbone(
        dim=32, depth=2, num_single_layers=2, heads=4, dim_head=8,
        latent_dim=D_lat, text_hidden_dim=D_llm,
    )
    x = torch.randn(B, N_tgt, D_lat, requires_grad=True)
    ref = torch.randn(B, N_ref, D_lat)
    time = torch.rand(B)

    v_pred = backbone(x, c_sem, time, ref=ref)
    print(f"\nバックボーン入力: x{tuple(x.shape)}, ref{tuple(ref.shape)}, text{tuple(c_sem.shape)}")
    print(f"バックボーン出力 (予測フロー速度 v_pred): {tuple(v_pred.shape)}  (= xと同形状)")
    print(f"初期化直後のv_predはゼロ初期化されたAdaLN/proj_outのため厳密に0: "
          f"max|v_pred|={v_pred.abs().max().item():.3e} (DiT系のzero-init設計として想定通り)")

    # 注意: v_predは上記の理由でゼロ初期化直後は厳密に0になるため、v_pred自身を
    # 目的関数にすると勾度も厳密に0になる (自明な自己参照損失のバグ)。
    # 実際のflow-matching損失は MSE(v_pred, target_velocity) であり、
    # target_velocity = z1 - z0 は一般に非ゼロなので、ここでも非ゼロなダミーの
    # ターゲットを使って本物の学習信号が流れることを確認する。
    target_velocity = torch.randn_like(v_pred)
    loss = F.mse_loss(v_pred, target_velocity)
    loss.backward()
    print(f"\nloss={loss.item():.6f} (target_velocityとの非自明なMSE)")

    n_nonzero = sum(1 for p in backbone.parameters() if p.grad is not None and p.grad.norm() > 0)
    n_total = sum(1 for _ in backbone.parameters())
    print(f"バックボーンの非ゼロ勾度パラメータ数: {n_nonzero}/{n_total}")
    print(f"proj_out.weight勾度ノルム: {backbone.proj_out.weight.grad.norm().item():.3e}  (これだけ非ゼロ)")
    print(f"xへの勾度ノルム: {x.grad.norm().item():.3e}  (ゼロのはず)")
    print(
        "\n=> AdaLN線形層とproj_outを全てゼロ初期化する公式設計 (initialize_weights) のため、"
        "全てのゲート(gate_msa/gate_mlp)が初期化直後は厳密に0になり、各ブロックはx_outputs=x_inputs"
        "という恒等写像になる。その結果、最後のproj_out (重みも0) だけが入力に依存した非ゼロ勾度を受け取り、"
        "それより手前 (全てのAdaLN・SRNL・層融合の重みを含む) には勾度が届かない。これはバグではなく、"
        "DiT系モデルで学習初期の出力を0にする意図的な設計 (zero-init trick) の数学的な帰結であり、"
        "最初の数ステップでproj_outが0から離れるにつれて、後続ステップで手前の層にも勾度が伝わるようになる。"
    )

    print("\n--- 実際に数ステップ最適化し、手前の層にも勾度が伝わり始めることを確認 ---")
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(fusion.parameters()), lr=1e-2
    )
    for step in range(5):
        optimizer.zero_grad()
        x2 = torch.randn(B, N_tgt, D_lat, requires_grad=True)
        c_sem2 = fusion(hidden_states)
        v_pred2 = backbone(x2, c_sem2, time, ref=ref)
        loss2 = F.mse_loss(v_pred2, torch.randn_like(v_pred2))
        loss2.backward()
        optimizer.step()
        if step == 4:
            early_grad = backbone.mmdit_blocks[0].attn_norm_x.linear.weight.grad.norm().item()
            print(f"step {step}: loss={loss2.item():.4f}  "
                  f"mmdit[0].attn_norm_x勾度ノルム={early_grad:.3e} "
                  f"(数ステップ後は0でなくなり、手前の層にも学習信号が伝わる)")


if __name__ == "__main__":
    example_usage()
