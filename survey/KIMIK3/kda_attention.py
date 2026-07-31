"""Kimi Delta Attention (KDA) -- Kimi K3 論文 §2.1.1 (sec:kda) の実装。

KDAは delta-rule recurrence (DeltaNet) にチャネルワイズの忘却ゲートを加えたもので、
Kimi K3 のバックボーンにおいて 3層に1層の Gated MLA を除く全ての層で使われる
(3:1 の Hybrid Attention, §2.1)。

参照する論文中の式:
    Eq.(recurrent_KDA)      : 状態更新の再帰式 (逐次形)
    Eq.(kda-param)          : q,k,v,beta,decay-logit z の生成
    Eq.(kda-cumulative-decay), Eq.(kda-chunkwise) : チャンク並列形
    Eq.(kda-forget-gate)    : lower-bounded decay (scaled sigmoid)
    Eq.(kda-output)         : full-rank output gate

【本ファイルで意図的に簡略化した部分】
論文は chunkwise 並列形の "UT transform" (擬似値 Ṽ_[t] = U_[t] - W_[t] S_[t] を
求める処理) について "We refer readers to Kimi Linear for the UT transform and
the full derivation" (§2.1.1, Eq.kda-chunkwise 直後) と明記しており、導出は
別論文 (Kimi Linear, team2025kimi) に委譲されている。そのため本実装では:
  1. `kda_recurrent_reference`: Eq.(recurrent_KDA) をそのまま1トークンずつ計算する
     "正解" 実装 (数値的に厳密、pass/mock なし)。
  2. `kda_chunkwise_forward`: チャンク境界ごとに状態 S を明示的に伝播させる
     (chunk-recurrent, Eq.kda-chunkwise の "inter-chunk" 項に相当) 一方で、
     チャンク内部の計算は UT-transform による行列並列化ではなく
     `kda_recurrent_reference` を chunk 内に限定して呼び出すことで代用する。
     出力はどちらの経路でも完全に一致する (下部の `if __name__` で数値検証)。
     実運用の FlashKDA カーネルはチャンク内部も Tensor Core 行列積で並列化するが、
     その具体的な導出手順は本論文の参照範囲外である。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Eq.(kda-param) の L2Norm。"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class ShortConv(nn.Module):
    """因果的な depthwise 1D 短畳み込み (Eq.kda-param の ShortConv)。

    入力  : (B, T, D)
    出力  : (B, T, D)  -- 未来のトークンを見ないよう左パディングした因果畳み込み
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T)
        y = self.conv(x.transpose(1, 2))[..., : x.shape[1]]
        return y.transpose(1, 2)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention 層 (1層分)。

    形状の記法:
        B          : バッチサイズ
        T          : シーケンス長 (トークン数)
        H          : KDA のヘッド数 (num_heads)
        d          : モデル隠れ次元 (hidden_size)
        d_k        : ヘッドあたりの query/key 次元 (head_dim)
        d_v        : ヘッドあたりの value 次元 (= d_k, KDA では query/key/value が同じ head_dim を共有)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        g_min: float = -5.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.g_min = g_min  # Eq.(kda-forget-gate) の下限 (論文では固定値 -5)
        proj_size = num_heads * head_dim

        # --- Eq.(kda-param): q, k, v, beta, decay-logit z の各射影 ---
        self.q_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.q_conv = ShortConv(proj_size, conv_kernel_size)
        self.k_conv = ShortConv(proj_size, conv_kernel_size)
        self.v_conv = ShortConv(proj_size, conv_kernel_size)

        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)  # Eq: beta_t^h = Sigmoid(W_beta x_t)

        # z_t^h = W_alpha^up W_alpha^down x_t + b_alpha^h  (低ランク射影 + head-wise bias)
        # 低ランク次元は head_dim そのもの (公式実装 modeling_kimi_linear.py:
        #   f_a_proj: Linear(hidden_size, head_dim), f_b_proj: Linear(head_dim, num_heads*head_dim)
        # に合わせている。全ヘッドで低ランクボトルネックを共有する設計)。
        alpha_rank = head_dim
        self.alpha_down = nn.Linear(hidden_size, alpha_rank, bias=False)
        self.alpha_up = nn.Linear(alpha_rank, num_heads * head_dim, bias=False)
        self.alpha_bias = nn.Parameter(torch.zeros(num_heads, head_dim))

        # A_h: head-wise log-scale (Eq.kda-forget-gate)。初期値 A_h=0 (論文の指定通り)。
        self.A_log_scale = nn.Parameter(torch.zeros(num_heads))

        # --- Eq.(kda-output): フルランク出力ゲート ---
        self.g_proj = nn.Linear(hidden_size, proj_size, bias=False)
        self.o_norm = nn.RMSNorm(head_dim, eps=1e-6)  # RMSNorm(õ_t) をヘッド単位で適用
        self.o_proj = nn.Linear(proj_size, hidden_size, bias=False)

    def _project_qkvbg(self, x: torch.Tensor):
        """(B, T, d) -> q, k, v: (B, T, H, d_k) / beta: (B, T, H) / g_log: (B, T, H, d_k)"""
        B, T, _ = x.shape
        q = l2norm(F.silu(self.q_conv(self.q_proj(x))))
        k = l2norm(F.silu(self.k_conv(self.k_proj(x))))
        v = F.silu(self.v_conv(self.v_proj(x)))
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        beta = torch.sigmoid(self.beta_proj(x))  # (B, T, H)

        z = self.alpha_up(self.alpha_down(x)).view(B, T, self.num_heads, self.head_dim)
        z = z + self.alpha_bias  # (B, T, H, d_k)  Eq.(kda-param) の z_t^h

        # Eq.(kda-forget-gate): g_t^h = g_min * sigmoid(e^{A_h} * z_t^h) in (g_min, 0)
        scale = torch.exp(self.A_log_scale).view(1, 1, self.num_heads, 1)
        g_log = self.g_min * torch.sigmoid(scale * z)  # (B, T, H, d_k), 各要素 in (g_min, 0)
        return q, k, v, beta, g_log

    def forward(self, hidden_states: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, d)
            chunk_size: None なら完全な逐次再帰、指定するとチャンク境界で状態を
                明示的に伝播させる chunk-recurrent 経路を通る (出力は数学的に同一)。
        Returns:
            (B, T, d)
        """
        q, k, v, beta, g_log = self._project_qkvbg(hidden_states)
        alpha = torch.exp(g_log)  # (B, T, H, d_k), Eq.(kda-forget-gate): alpha_t^h = exp(g_t^h)

        if chunk_size is None:
            o, _ = kda_recurrent_reference(q, k, v, alpha, beta)
        else:
            o, _ = kda_chunkwise_forward(q, k, v, alpha, beta, chunk_size=chunk_size)
        # o: (B, T, H, d_v)

        o = self.o_norm(o)  # ヘッド単位 RMSNorm (Eq.kda-output の RMSNorm(õ_t))
        g = self.g_proj(hidden_states).view_as(o)
        y = torch.sigmoid(g) * o  # Eq.(kda-output)
        y = y.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return self.o_proj(y)


def kda_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
):
    """Eq.(recurrent_KDA) の厳密な逐次実装 (1トークンずつ)。

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

    Args:
        q, k, alpha: (B, T, H, d_k)
        v:           (B, T, H, d_v)  (KDA では d_v == d_k)
        beta:        (B, T, H)
        state:       (B, H, d_k, d_v) or None (初期状態、無ければゼロ)
    Returns:
        o:     (B, T, H, d_v)
        state: (B, H, d_k, d_v)  -- 最終状態 (次チャンク/次デコードステップへ引き継ぐ)
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]
    if state is None:
        state = q.new_zeros(B, H, d_k, d_v)

    outputs = []
    for t in range(T):
        q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]  # (B, H, d_k) / (B, H, d_v)
        alpha_t, beta_t = alpha[:, t], beta[:, t]  # (B, H, d_k) / (B, H)

        # Diag(alpha_t) S_{t-1}: チャネルごとの減衰を状態の行 (d_k 軸) に適用
        decayed = state * alpha_t.unsqueeze(-1)  # (B, H, d_k, d_v)

        # (I - beta_t k_t k_t^T) を左からかける = decayed から beta_t*k_t*(k_t^T decayed) を引く
        kt_decayed = torch.einsum("bhk,bhkv->bhv", k_t, decayed)  # k_t^T @ decayed : (B, H, d_v)
        correction = beta_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * kt_decayed.unsqueeze(-2)
        state = decayed - correction  # (I - beta k k^T) Diag(alpha) S_{t-1}

        # + beta_t k_t v_t^T
        state = state + beta_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        o_t = torch.einsum("bhkv,bhk->bhv", state, q_t)  # S_t^T q_t : (B, H, d_v)
        outputs.append(o_t)

    o = torch.stack(outputs, dim=1)  # (B, T, H, d_v)
    return o, state


def kda_chunkwise_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    state: torch.Tensor | None = None,
):
    """チャンク境界で状態を明示的に伝播させる chunk-recurrent 実装。

    論文 Eq.(kda-chunkwise) の inter-chunk / intra-chunk 分解のうち、
    チャンク間の状態伝播 (recurrent across chunks) だけを明示化し、
    チャンク内部 (intra-chunk, 本来は Tril 行列積で並列化される) は
    `kda_recurrent_reference` に委譲する。KDA Context Parallelism
    (論文 §infra sec:kda-cp) が「チャンク=セグメント単位で状態を合成できる」
    という性質を利用するのと同じ発想に基づく。

    Returns:
        o:     (B, T, H, d_v)
        state: (B, H, d_k, d_v)  最終状態
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]
    if state is None:
        state = q.new_zeros(B, H, d_k, d_v)

    outs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        o_chunk, state = kda_recurrent_reference(
            q[:, start:end], k[:, start:end], v[:, start:end],
            alpha[:, start:end], beta[:, start:end], state=state,
        )
        outs.append(o_chunk)
    o = torch.cat(outs, dim=1)
    return o, state


def cumulative_decay(alpha: torch.Tensor) -> torch.Tensor:
    """Eq.(kda-cumulative-decay): gamma_[t]^r = prod_{i=1}^{r} alpha_[t]^i (チャンク先頭からの累積減衰)。

    Args:
        alpha: (B, C, H, d_k)  チャンク内の1トークンずつの減衰率
    Returns:
        gamma: (B, C, H, d_k)  各位置までの累積積 (log空間で cumsum してから exp するほうが数値安定)
    """
    log_alpha = torch.log(alpha.clamp_min(1e-8))
    return torch.exp(torch.cumsum(log_alpha, dim=1))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, d_k = 2, 37, 4, 16
    q = l2norm(torch.randn(B, T, H, d_k))
    k = l2norm(torch.randn(B, T, H, d_k))
    v = torch.randn(B, T, H, d_k)
    g_min = -5.0
    z = torch.randn(B, T, H, d_k)
    alpha = torch.exp(g_min * torch.sigmoid(z))  # Eq.(kda-forget-gate)
    beta = torch.sigmoid(torch.randn(B, T, H))

    o_seq, s_seq = kda_recurrent_reference(q, k, v, alpha, beta)
    o_chunk, s_chunk = kda_chunkwise_forward(q, k, v, alpha, beta, chunk_size=8)

    print("output shape:", o_seq.shape)  # (2, 37, 4, 16)
    print("state shape :", s_seq.shape)  # (2, 4, 16, 16)
    print("max |o_seq - o_chunk| =", (o_seq - o_chunk).abs().max().item())
    print("max |s_seq - s_chunk| =", (s_seq - s_chunk).abs().max().item())
    assert torch.allclose(o_seq, o_chunk, atol=1e-5)
    assert torch.allclose(s_seq, s_chunk, atol=1e-5)

    # --- レイヤー全体の動作確認 ---
    layer = KimiDeltaAttention(hidden_size=64, num_heads=H, head_dim=d_k)
    x = torch.randn(B, T, 64)
    y1 = layer(x, chunk_size=None)
    y2 = layer(x, chunk_size=8)
    print("layer output shape:", y1.shape)  # (2, 37, 64)
    print("max |y_seq - y_chunk| =", (y1 - y2).abs().max().item())
    assert torch.allclose(y1, y2, atol=1e-4)
    print("KDA recurrent form and chunkwise form match exactly. OK")
