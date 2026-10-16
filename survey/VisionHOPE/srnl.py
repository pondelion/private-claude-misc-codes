"""
VisionHOPE - Self-Referential Nested Learning (SRNL) (モデルブロック)
================================================================

論文の核心である SRNL (5メモリが連動して学習しあう再帰) を純粋な PyTorch で
再実装する。

論文: VisionHOPE: Visual Backbones as Self-Modifying Learning Systems (arXiv:2609.33325)
      - Sec 3.1 Nested Learning Foundations (Eq. 1-13): Delta Gradient Descent と
        Self-Referential DGD (SR-DGD) の導出
      - Sec 3.1.3 Chunk-Wise Linear Recurrence (Eq. 14-16): チャンク単位の並列化
      - Sec 3.2 Stability-Matched Step-Size Control (Eq. 17-21, Def.1): ソフト注入
        キャップ + スペクトラルクランプによる non-expansion 保証
公式コード: gh/VisionHOPE/models/srnl.py (`SRNL` クラス) は、実際の再帰演算を
      CUDA拡張 (models/srnl_core/csrc/*.cu, *.cuh) に委譲するラッパーのみを提供し、
      Python/PyTorchレベルでの再帰のソースコードは公開されていない (JITコンパイルされる
      生CUDAのみ)。本ファイルはCUDA拡張をビルドせず、CPU/GPUどちらでも動く
      PyTorch参照実装であり、論文の数式をそのまま実装したものである。
      活性化関数の具体形 (phi_eta=スケール付きsoftplus, phi_alpha=バイアス付きsigmoid)
      と数値安定化定数は、公式CUDAソースに実際に定義されている値を採用した:
        models/srnl_core/csrc/chunk.py の STEP_SCALE=0.025, MIN_RETENTION=0.0,
        RETENTION_BIAS=2.197224577 (=logit(0.9)), INJECTION_MARGIN=0.999,
        SPECTRAL_MARGIN=0.999999
        models/srnl_core/csrc/memory.cuh の soft_project_eta()/clamp_eta_to_alpha()
      (論文 Eq.17 の r_t=sqrt(||delta||^2+eps^2)+eps は、公式カーネルでは
       diff_norm+eps という単純な形に簡略化されている。本ファイルは論文の式を
       採用し、コメントでこの差異を明記する)。

============================================================
Shape Convention
============================================================
B      : バッチサイズ
N      : トークン数 (系列長)
D      : ヘッド次元 (head_dim; 4/8/16/32/64のいずれか)
H      : ヘッド数 (dim // head_dim)
C      : チャンク長
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 公式 models/srnl_core/csrc/chunk.py の環境変数デフォルト値そのまま。
STEP_SCALE = 0.025
ALPHA_MIN = 0.0
ALPHA_BIAS = 2.197224577  # = logit(0.9): 初期化時の retention を 0.9 付近にバイアス
INJECTION_MARGIN = 0.999  # 公式 STAB_MU
SPECTRAL_MARGIN = 0.999999
PRE_ACT_CLAMP = 10.0
STAB_EPS = 1e-6

SUPPORTED_HEAD_DIMS = (4, 8, 16, 32, 64)


def phi_eta(pre_activation: torch.Tensor) -> torch.Tensor:
    """学習率メモリの出力を正の DGD ステップへ写す phi_eta (論文 Eq.9, 公式 chunk.cu)。

    phi_eta(x) = STEP_SCALE * softplus(clamp(x, -10, 10))
    """
    clamped = pre_activation.clamp(-PRE_ACT_CLAMP, PRE_ACT_CLAMP)
    return STEP_SCALE * F.softplus(clamped)


def phi_alpha(pre_activation: torch.Tensor) -> torch.Tensor:
    """保持率メモリの出力を [0,1) の retention factor へ写す phi_alpha (論文 Eq.9)。

    phi_alpha(x) = ALPHA_MIN + (1-ALPHA_MIN) * sigmoid(clamp(x + ALPHA_BIAS, -10, 10))
    ALPHA_BIAS=logit(0.9) により、学習初期 (メモリ出力≈0) の retention は約0.9になる。
    """
    biased = (pre_activation + ALPHA_BIAS).clamp(-PRE_ACT_CLAMP, PRE_ACT_CLAMP)
    sigmoid = torch.sigmoid(biased)
    return ALPHA_MIN + (1.0 - ALPHA_MIN) * sigmoid


def stability_matched_step(
    eta_raw: torch.Tensor,     # (BH,) phi_etaを通した後の生ステップ
    alpha: torch.Tensor,       # (BH,)
    k: torch.Tensor,           # (BH, D) 正規化済みキー
    delta: torch.Tensor,       # (BH, D) = k - v
) -> torch.Tensor:
    """ソフト注入キャップ + スペクトラルクランプ (論文 Def.1, Eq.17-18)。

    ========================================
    式
    ========================================
    r_t = sqrt(||delta_t||^2 + eps^2) + eps                      (Eq.17)
    eta_inj = INJECTION_MARGIN * (1 - alpha_t) / r_t
    eta_bar = eta_inj * (1 - exp(-eta_t / eta_inj))               (soft injection cap)
    eta_spec = 2 * SPECTRAL_MARGIN * alpha_t / ||k_t||^2          (Eq.18, spectral clamp)
    eta_tilde = min(eta_bar, eta_spec)

    ========================================
    Shape
    ========================================
    入力: eta_raw, alpha: (BH,) / k, delta: (BH, D)
    出力: eta_tilde: (BH,) 実際に使われる executed step
    """
    diff_norm = torch.sqrt(delta.pow(2).sum(-1) + STAB_EPS ** 2) + STAB_EPS  # (BH,) 論文Eq.17
    eta_inj = INJECTION_MARGIN * (1.0 - alpha) / diff_norm
    z = eta_raw / eta_inj
    eta_bar = eta_inj * (-torch.expm1(-z))  # = eta_inj * (1 - exp(-z)), 数値的に安定な形

    # 公式CUDAカーネル (memory.cuh) は ETA2ALPHA_DENOM_EPS=1e-20 という非常に小さい
    # フロアを使うが、そちらは手で導出した解析的backwardを使うため 1/key_denom^2 が
    # float32で桁あふれしても安全に扱われている。本実装はPyTorchのautogradに
    # 除算をそのまま微分させるため、同じ1e-20だと ∂eta_spec/∂key_norm_sq ~ 1/key_norm_sq^2
    # がfloat32の範囲(~3.4e38)を超えてinfになり、torch.minimumの非選択側に
    # 0*inf=NaNが伝播する。そのため本参照実装では同じ安定化の意図を保ったまま
    # 1e-6 (他の箇所のSTAB_EPSと同スケール) を使う。
    key_norm_sq = k.pow(2).sum(-1).clamp_min(1e-6)
    eta_spec = 2.0 * SPECTRAL_MARGIN * alpha / key_norm_sq  # k≈0ならk_norm_sqが下限→eta_specは大きな値

    return torch.minimum(eta_bar, eta_spec)


def _stabilize_key(k_raw: torch.Tensor) -> torch.Tensor:
    """キーを ||k||_2 <= 1 に安定化する (論文 Def.1: "stabilized L2 normalization")。

    ノルムが1を超える場合のみ縮小する (単位ベクトルへの正規化ではなく上限クリップ)。
    """
    norm = k_raw.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return k_raw / norm


class SRNLReference(nn.Module):
    """Self-Referential Nested Learning の単一方向・単一インスタンス版 (参照実装)。

    ========================================
    Shape
    ========================================
    入力:
        x:       (B, N, dim)
        queries: (B, N, dim) or None (Noneならxを流用)
    出力:
        y: (B, N, dim)

    ``dim`` は ``head_dim`` で割り切れる必要があり、内部で ``H = dim // head_dim``
    個のヘッドに分割して独立に処理する (公式と同じ多ヘッド構成)。
    ``chunk_size`` は正の整数 (Noneなら64)。``N`` は ``chunk_size`` で割り切れる
    必要がある (割り切れない場合は呼び出し側でパディングする)。

    クエリ射影 W_q (論文 Eq.13: q_t=W_qx_t) は公式実装でも ``SRNL`` クラス自身には
    持たせず、呼び出し側 (``VisionHOPEOperator``) が外部パラメータとして持つ設計に
    なっている (SRNLは渡された queries をそのまま使うだけ)。本クラスもそれに
    合わせ、W_q は保持しない。
    """

    def __init__(self, dim: int, head_dim: int = 16, chunk_size: int | None = None):
        super().__init__()
        if head_dim not in SUPPORTED_HEAD_DIMS or dim <= 0 or dim % head_dim:
            raise ValueError(f"dim must be positive and divisible by head_dim in {SUPPORTED_HEAD_DIMS}")
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.chunk_size = chunk_size or 64

        H, D = self.num_heads, self.head_dim
        # 5つの学習済みメモリの初期値 (公式 SRNL.reset_parameters と同じ初期化方針)。
        self.M_m_0 = nn.Parameter(torch.empty(H, D, D))
        self.M_k_0 = nn.Parameter(torch.empty(H, D, D))
        self.M_v_0 = nn.Parameter(torch.empty(H, D, D))
        self.m_eta_0 = nn.Parameter(torch.zeros(H, D))
        self.m_alpha_0 = nn.Parameter(torch.zeros(H, D))
        self.reset_parameters()

    def reset_parameters(self):
        for p in (self.M_m_0, self.M_k_0, self.M_v_0):
            nn.init.trunc_normal_(p, std=0.01)
        nn.init.zeros_(self.m_eta_0)
        nn.init.zeros_(self.m_alpha_0)

    def forward(self, x: torch.Tensor, queries: torch.Tensor | None = None) -> torch.Tensor:
        B, N, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"SRNLReference expects last dim={self.dim}, got {dim}")
        if N % self.chunk_size:
            raise ValueError(f"sequence length {N} must be divisible by chunk_size={self.chunk_size}")
        if queries is None:
            queries = x

        H, D = self.num_heads, self.head_dim
        # (B, N, dim) -> (B, N, H, D) -> (B*H, N, D): ヘッドをバッチ次元に展開して
        # 以降の全ての演算をヘッド独立のバッチ処理として書く (公式と同じヘッド分割)。
        x_heads = x.view(B, N, H, D).permute(0, 2, 1, 3).reshape(B * H, N, D)
        q_heads = queries.view(B, N, H, D).permute(0, 2, 1, 3).reshape(B * H, N, D)

        # 学習済み初期メモリをバッチ方向に複製: (H,D,D) -> (B*H,D,D)。
        M_m = self.M_m_0.repeat(B, 1, 1)
        M_k = self.M_k_0.repeat(B, 1, 1)
        M_v = self.M_v_0.repeat(B, 1, 1)
        m_eta = self.m_eta_0.repeat(B, 1)
        m_alpha = self.m_alpha_0.repeat(B, 1)

        y_heads = srnl_chunkwise_recurrence(
            x_heads, q_heads, self.chunk_size, M_m, M_k, M_v, m_eta, m_alpha,
        )  # (B*H, N, D)

        # (B*H, N, D) -> (B, H, N, D) -> (B, N, H, D) -> (B, N, dim)
        y = y_heads.view(B, H, N, D).permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return y


def srnl_chunkwise_recurrence(
    x: torch.Tensor,        # (BH, N, D)
    q: torch.Tensor,        # (BH, N, D)
    chunk_size: int,
    M_m: torch.Tensor,      # (BH, D, D) 初期境界状態 (呼び出し後に書き換えない; 内部でクローンする)
    M_k: torch.Tensor,
    M_v: torch.Tensor,
    m_eta: torch.Tensor,    # (BH, D)
    m_alpha: torch.Tensor,
) -> torch.Tensor:
    """チャンク単位の SR-DGD 再帰 (論文 Sec 3.1.3 + Sec 3.2) を実行する。

    各チャンク内では:
      1. (並列化可能) チャンク境界の固定メモリから k_t, v_t, eta_t, alpha_t, y_t を
         チャンク内全トークンについて一括計算する (論文 Eq.14-16)。
      2. (系列順・逐次) 5つのメモリをトークン順に SR-DGD で更新する (論文 Eq.19-21)。
         読み出し y_t は境界の M_m だけに依存するため、チャンク内では再計算しない。

    ========================================
    Shape
    ========================================
    入力: x, q: (BH, N, D) / M_m,M_k,M_v: (BH, D, D) / m_eta,m_alpha: (BH, D)
    出力: y: (BH, N, D)
    """
    BH, N, D = x.shape
    num_chunks = N // chunk_size
    device, dtype = x.device, x.dtype
    eye = torch.eye(D, device=device, dtype=dtype).expand(BH, D, D)

    outputs = []
    for c in range(num_chunks):
        xc = x[:, c * chunk_size:(c + 1) * chunk_size, :]  # (BH, C, D)
        qc = q[:, c * chunk_size:(c + 1) * chunk_size, :]

        # --- 1. 並列生成フェーズ (境界状態 M_m, M_k, M_v, m_eta, m_alpha は固定) ---
        k_raw = torch.einsum('bij,bcj->bci', M_k, xc)   # (BH, C, D):  k_t = M_k @ x_t, 全tを並列計算
        k = _stabilize_key(k_raw)                        # ||k_t||<=1 (論文 Def.1)
        v = torch.einsum('bij,bcj->bci', M_v, xc)        # (BH, C, D)
        eta_pre = torch.einsum('bj,bcj->bc', m_eta, xc)      # (BH, C):  m_eta @ x_t (スカラー)
        alpha_pre = torch.einsum('bj,bcj->bc', m_alpha, xc)  # (BH, C)
        eta_t = phi_eta(eta_pre)                             # (BH, C)
        alpha_t = phi_alpha(alpha_pre)                       # (BH, C)
        y_chunk = torch.einsum('bij,bcj->bci', M_m, qc)  # (BH, C, D): y_t=M_m(境界)@q_t, Eq.20
        delta = k - v                                     # (BH, C, D)
        outputs.append(y_chunk)

        # --- 2. 逐次更新フェーズ (トークン順に5メモリを更新; 次チャンクの境界になる) ---
        for t in range(chunk_size):
            k_t, v_t, delta_t = k[:, t], v[:, t], delta[:, t]      # (BH, D)
            eta_raw_t, alpha_raw_t = eta_t[:, t], alpha_t[:, t]    # (BH,)
            eta_tilde = stability_matched_step(eta_raw_t, alpha_raw_t, k_t, delta_t)  # (BH,)

            kkT = torch.einsum('bi,bj->bij', k_t, k_t)  # (BH, D, D)
            # T_t = alpha*I - eta_tilde*k k^T: 全メモリ共通の遷移行列 (論文 Eq.19 A_t)
            T_t = alpha_raw_t.view(-1, 1, 1) * eye - eta_tilde.view(-1, 1, 1) * kkT

            def sr_dgd_update(M: torch.Tensor) -> torch.Tensor:
                """行列メモリの SR-DGD 更新: M_t = M_{t-1} @ T_t - eta*g*k^T (論文 Eq.19)。"""
                v_hat = torch.einsum('bij,bj->bi', M, v_t)            # M_{t-1} @ v_t  (自己生成ターゲット, Eq.11)
                g = torch.einsum('bij,bj->bi', M, k_t) - v_hat        # g^sq_t = M_{t-1}@k_t - v_hat (Eq.12)
                injected = eta_tilde.view(-1, 1, 1) * torch.einsum('bi,bj->bij', g, k_t)
                return torch.einsum('bij,bjk->bik', M, T_t) - injected

            def sr_dgd_update_vector(m: torch.Tensor) -> torch.Tensor:
                """行ベクトルメモリ(eta/alpha)の同じ更新を1×D線形写像として適用する。"""
                v_hat = (m * v_t).sum(-1)                              # (BH,) スカラー
                g = (m * k_t).sum(-1) - v_hat                          # (BH,)
                transitioned = torch.einsum('bj,bjk->bk', m, T_t)      # (BH, D)
                injected = (eta_tilde * g).unsqueeze(-1) * k_t         # (BH, D)
                return transitioned - injected

            M_m = sr_dgd_update(M_m)
            M_k = sr_dgd_update(M_k)
            M_v = sr_dgd_update(M_v)
            m_eta = sr_dgd_update_vector(m_eta)
            m_alpha = sr_dgd_update_vector(m_alpha)

    return torch.cat(outputs, dim=1)  # (BH, N, D)


# ============================================================
# 使用例: 勾配が実際に流れること、non-expansion (メモリノルムが発散しないこと) を確認する
# ============================================================

def example_usage() -> None:
    print("=== srnl.py Example Usage ===\n")
    torch.manual_seed(0)

    B, N, dim, head_dim, chunk = 2, 16, 32, 8, 4
    srnl = SRNLReference(dim=dim, head_dim=head_dim, chunk_size=chunk)
    x = torch.randn(B, N, dim, requires_grad=True)

    print(f"入力 shape: {tuple(x.shape)}  (B={B}, N={N}, dim={dim})")
    print(f"num_heads={srnl.num_heads}, head_dim={head_dim}, chunk_size={chunk}")

    y = srnl(x)
    print(f"出力 shape: {tuple(y.shape)}")

    loss = y.square().mean()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"\nloss={loss.item():.6f}  入力への勾配ノルム={grad_norm:.3e} "
          f"(0でなければ逆伝播成功; nonzero={bool((x.grad != 0).any())})")

    # --- Non-expansion の実測確認 (論文 Corollary 1: ||M_t||_F <= ||M_{t-1}||_F) ---
    print("\n--- Non-expansion の実測確認 (トークンを追加するたびにメモリノルムを記録) ---")
    with torch.no_grad():
        H, D = srnl.num_heads, srnl.head_dim
        M_m = srnl.M_m_0.repeat(1, 1, 1)  # (H, D, D), バッチ1つ分
        M_k = srnl.M_k_0.repeat(1, 1, 1)
        M_v = srnl.M_v_0.repeat(1, 1, 1)
        m_eta = srnl.m_eta_0.repeat(1, 1)
        m_alpha = srnl.m_alpha_0.repeat(1, 1)
        x_one = torch.randn(1, N, dim)
        x_heads = x_one.view(1, N, H, D).permute(0, 2, 1, 3).reshape(H, N, D)
        q_heads = x_heads

        norms = [M_m.norm().item()]
        eye = torch.eye(D)
        for c in range(N // chunk):
            xc = x_heads[:, c * chunk:(c + 1) * chunk, :]
            k_raw = torch.einsum('bij,bcj->bci', M_k, xc)
            k = _stabilize_key(k_raw)
            v = torch.einsum('bij,bcj->bci', M_v, xc)
            eta_pre = torch.einsum('bj,bcj->bc', m_eta, xc)
            alpha_pre = torch.einsum('bj,bcj->bc', m_alpha, xc)
            eta_t = phi_eta(eta_pre)
            alpha_t = phi_alpha(alpha_pre)
            delta = k - v
            for t in range(chunk):
                k_t, delta_t = k[:, t], delta[:, t]
                eta_tilde = stability_matched_step(eta_t[:, t], alpha_t[:, t], k_t, delta_t)
                kkT = torch.einsum('bi,bj->bij', k_t, k_t)
                T_t = alpha_t[:, t].view(-1, 1, 1) * eye - eta_tilde.view(-1, 1, 1) * kkT
                v_hat = torch.einsum('bij,bj->bi', M_m, v[:, t])
                g = torch.einsum('bij,bj->bi', M_m, k_t) - v_hat
                M_m = torch.einsum('bij,bjk->bik', M_m, T_t) - eta_tilde.view(-1, 1, 1) * torch.einsum(
                    'bi,bj->bij', g, k_t)
                norms.append(M_m.norm().item())
        print(f"M_m のフロベニウスノルムの推移 (先頭5件→末尾5件): "
              f"{[round(v, 4) for v in norms[:5]]} -> {[round(v, 4) for v in norms[-5:]]}")
        print(f"最大ノルム={max(norms):.4f}, 初期ノルム={norms[0]:.4f} "
              f"(発散せず初期値程度に留まっていればnon-expansionと整合)")


if __name__ == "__main__":
    example_usage()
