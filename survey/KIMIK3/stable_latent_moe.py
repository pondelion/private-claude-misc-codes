"""Stable LatentMoE -- Kimi K3 論文 §2.3 (sec:stable-latent-moe) の実装。

LatentMoE (Elango et al. 2026) は「フル幅の共有エキスパート」と「低次元潜在空間 ℓ で
動作するルーテッドエキスパート」を分離することで、896個中16個という極端なスパース性
(sparsity=56) を実現する (§2.3 冒頭)。このスパース性は2つの不安定要因を増幅するため、
Kimi K3 は3つの機構で対処する:
    1. RMSNorm (Normalized LatentMoE, §2.3.1) : W_up 適用前に集約表現 u を正規化
    2. SiTU-GLU (§2.3.2, sec:situ)            : 活性化爆発を抑える有界な GLU 変種
    3. Quantile Balancing (§2.3.3, sec:qb)     : 補助損失なしの負荷分散をquantileで解く

式番号:
    Eq.(latentmoe)        : y = shared(x) + W_up RMSNorm(u),  u = sum_i p_i E_i(W_down x)
    Eq.(situglu)           : SiTU-GLU(x) = [β1 tanh(Wg x/β1) ⊙ Sigmoid(Wg x)] ⊙ [β2 tanh(Wu x/β2)]
    Eq.(situglu-bound)     : ||SiTU-GLU(x)||_inf <= β1*β2 = 100  (β1=4, β2=25)
    Eq.(moe-routing)       : s_i = Sigmoid(W_r x_i), T_i = argtop_k(s_i + b)
    Eq.(qb-update)         : b_j <- -quantile_{1-k/n}(s_{:,j} - alpha) , then mean-centered
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
        return self.weight * x.to(dtype)


class SiTUGLU(nn.Module):
    """Eq.(situglu): Sigmoid Tanh Unit GLU。

    SwiGLU = Sigmoid(Wg x) ⊙ (Wg x) ⊙ (Wu x) の両方の乗数を softcap(z,β)=β tanh(z/β) で
    有界化したもの。β1,β2 -> ∞ で SwiGLU に一致し (Eq直後の局所展開)、原点近傍では
    SwiGLU と同じ振る舞いをする (Appendix §app:situglu)。

    入力  : gate, up  それぞれ (..., d_ffn)  (KimiBlockSparseMLP 側で w1(x), w3(x) を渡す)
    出力  : (..., d_ffn)
    """

    def __init__(self, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.float()
        up = up.float()
        capped_gate = self.beta1 * torch.tanh(gate / self.beta1)
        situ_gate = capped_gate * torch.sigmoid(gate)  # softcap(Wg x) ⊙ Sigmoid(Wg x)
        capped_up = self.beta2 * torch.tanh(up / self.beta2)  # softcap(Wu x)
        return (situ_gate * capped_up).to(gate.dtype)


class RoutedExpertMLP(nn.Module):
    """潜在空間 (次元 ℓ) 内で動作する1つのルーテッドエキスパート E_i^routed。"""

    def __init__(self, latent_dim: int, hidden_dim: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w_gate = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.act = SiTUGLU(beta1, beta2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act(self.w_gate(z), self.w_up(z)))


class SharedExpertMLP(nn.Module):
    """フル幅 (次元 d) で動作する共有エキスパート E_j^shared。"""

    def __init__(self, hidden_size: int, ffn_dim: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.act = SiTUGLU(beta1, beta2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act(self.w_gate(x), self.w_up(x)))


def quantile_balancing_update(
    scores: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Eq.(qb-update): Quantile Balancing によるバイアス更新 (1回の forward で計算)。

    Appendix §app:qb-histogram は本番では全バッチにまたがるヒストグラム近似で quantile を
    求めると説明しているが (百万トークン規模で厳密な quantile 計算が非現実的なため)、
    このファイルは教育目的の小規模デモなので `torch.quantile` による厳密計算で代用する
    (出力の意味は同一、近似は純粋にスケーラビリティのための実装上の最適化であり、
    アルゴリズムの定義そのものではないため簡略化して問題ない)。

    Args:
        scores: (M, n) ルータースコア s_i = Sigmoid(W_r x_i)  (M=バッチ内トークン数, n=エキスパート数)
        bias:   (n,)   現在のバイアス b^(t)
        top_k:  各トークンが選択するエキスパート数 k
    Returns:
        new_bias: (n,) 次ステップで使うバイアス b^(t+1) (平均0に補正済み)
    """
    M, n = scores.shape
    biased = scores + bias.unsqueeze(0)  # (M, n)

    # Top-(k+1) を取り、(k+1)番目のスコアを各トークンのカットオフ alpha_i とする
    topk1_vals, _ = torch.topk(biased, k=top_k + 1, dim=-1)  # (M, k+1) 降順
    cutoff = topk1_vals[:, -1]  # alpha_i^(t): (M,)  = (k+1)番目に大きいbiasedスコア

    margin = scores - cutoff.unsqueeze(1)  # s_{i,j} - alpha_i^(t) : (M, n)

    quantile_level = 1.0 - top_k / n  # (1 - k/n)-分位点
    b_hat = -torch.quantile(margin, q=quantile_level, dim=0)  # (n,)  Eq: b_hat_j = -quantile(...)
    new_bias = b_hat - b_hat.mean()  # 共通オフセットを除去 (Top-k選択に影響しないよう平均0化)
    return new_bias


class StableLatentMoE(nn.Module):
    """Stable LatentMoE 層 (1層分)。

    形状の記法:
        B, T   : バッチ, シーケンス長  (以下では N = B*T としてフラット化して扱う)
        d      : モデル隠れ次元 (hidden_size)
        ell    : ルーテッドエキスパートの潜在次元 (routed_expert_hidden_size, K3 実値: 3584)
        n      : ルーテッドエキスパート総数 (K3 実値: 896)
        k      : トークンあたりの活性化エキスパート数 (K3 実値: 16)
        n_s    : 共有エキスパート数 (K3 実値: 2)
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        num_routed_experts: int,
        num_experts_per_token: int,
        num_shared_experts: int,
        routed_ffn_dim: int,
        shared_ffn_dim: int,
        situ_beta1: float = 4.0,
        situ_beta2: float = 25.0,
        use_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_routed_experts = num_routed_experts
        self.top_k = num_experts_per_token

        # ルーティング (Eq.moe-routing): s_i = Sigmoid(W_r x_i)
        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(num_routed_experts))  # b (学習ではなくQBで更新)

        # 潜在空間への down/up 射影 (Eq.latentmoe: W_down, W_up)
        self.w_down = nn.Linear(hidden_size, latent_dim, bias=False)
        self.w_up = nn.Linear(latent_dim, hidden_size, bias=False)
        self.use_norm = use_norm
        if use_norm:
            self.latent_norm = KimiRMSNorm(latent_dim)  # §2.3.1 Normalized LatentMoE

        self.routed_experts = nn.ModuleList(
            [
                RoutedExpertMLP(latent_dim, routed_ffn_dim, situ_beta1, situ_beta2)
                for _ in range(num_routed_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SharedExpertMLP(hidden_size, shared_ffn_dim, situ_beta1, situ_beta2)
                for _ in range(num_shared_experts)
            ]
        )

    def route(self, x: torch.Tensor):
        """Eq.(moe-routing): トークンごとの Top-k エキスパート選択と正規化重み。

        Args:
            x: (N, d)
        Returns:
            topk_idx:    (N, k)  選択されたエキスパート番号
            topk_weight: (N, k)  正規化済みの混合重み p_i
            raw_scores:  (N, n)  QB 更新に使う生スコア (Sigmoid 後、bias 加算前)
        """
        raw_scores = torch.sigmoid(self.router(x).float())  # (N, n)
        biased = raw_scores + self.routing_bias.unsqueeze(0)
        topk_biased, topk_idx = torch.topk(biased, k=self.top_k, dim=-1)  # (N, k)
        topk_weight = torch.gather(raw_scores, dim=-1, index=topk_idx)  # bias抜きの生スコアで混合 (Eq.moe-routing)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight, raw_scores

    @torch.no_grad()
    def update_routing_bias(self, raw_scores: torch.Tensor) -> None:
        """1 forward 分の raw_scores から Quantile Balancing (Eq.qb-update) でバイアスを更新する。

        論文の因果性の注記通り (§2.3.3 末尾) この更新は「次のステップから」有効になる
        ("a batch is never routed with a bias derived from itself") ため、学習ループ側で
        forward の後にこのメソッドを呼ぶ想定 (loss_computation.py 等では扱わない、
        アーキテクチャ内部の負荷分散ロジックであるためここに実装する)。
        """
        self.routing_bias.copy_(quantile_balancing_update(raw_scores, self.routing_bias, self.top_k))

    def forward(self, hidden_states: torch.Tensor, update_bias: bool = False) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, d)
            update_bias: True なら forward 内で QB のバイアス更新も行う (学習時のみ)
        Returns:
            (B, T, d)
        """
        B, T, d = hidden_states.shape
        x = hidden_states.view(-1, d)  # (N, d), N = B*T

        topk_idx, topk_weight, raw_scores = self.route(x)
        if update_bias and self.training:
            self.update_routing_bias(raw_scores.detach())

        # --- ルーテッドパス: x -> 潜在空間 z -> 選択エキスパートで変換 -> 重み付き集約 u ---
        z = self.w_down(x)  # (N, ell)
        N = x.shape[0]
        u = z.new_zeros(N, self.latent_dim)
        for expert_id in range(self.num_routed_experts):
            token_mask = (topk_idx == expert_id)  # (N, k) の bool
            if not token_mask.any():
                continue
            token_idx, slot_idx = token_mask.nonzero(as_tuple=True)
            expert_out = self.routed_experts[expert_id](z[token_idx])  # (n_i, ell)
            weight = topk_weight[token_idx, slot_idx].unsqueeze(-1)  # (n_i, 1)
            u.index_add_(0, token_idx, weight * expert_out)

        if self.use_norm:
            u = self.latent_norm(u)  # §2.3.1: RMSNorm(u)  (up-projection 前の正規化)
        routed_out = self.w_up(u)  # (N, d)

        # --- 共有パス: フル幅で全トークンに適用 ---
        shared_out = sum(expert(x) for expert in self.shared_experts)

        y = shared_out + routed_out  # Eq.(latentmoe)
        return y.view(B, T, d)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d, ell = 2, 6, 64, 32
    n, k, n_s = 12, 3, 2  # 小規模デモ (K3 実値は n=896, k=16, n_s=2, ell=3584)

    moe = StableLatentMoE(
        hidden_size=d,
        latent_dim=ell,
        num_routed_experts=n,
        num_experts_per_token=k,
        num_shared_experts=n_s,
        routed_ffn_dim=48,
        shared_ffn_dim=96,
    )
    x = torch.randn(B, T, d)
    y = moe(x)
    print("StableLatentMoE output shape:", y.shape)  # (2, 6, 64)
    assert y.shape == x.shape

    # --- Quantile Balancing: 数ステップで負荷が target load q=Mk/n に近づくことを確認 ---
    moe.train()
    M = 256
    for step in range(30):
        x_batch = torch.randn(M, d)
        topk_idx, _, raw_scores = moe.route(x_batch)
        moe.update_routing_bias(raw_scores.detach())

    counts = torch.zeros(n)
    for e in range(n):
        counts[e] = (topk_idx == e).sum()
    target = M * k / n
    print(f"target load per expert = {target:.1f}, load std after QB = {counts.std().item():.2f}")
    print("per-expert token counts:", counts.tolist())

    # --- SiTU-GLU の有界性を数値確認 (Eq.situglu-bound: |f(x)| <= beta1*beta2 = 100) ---
    act = SiTUGLU(beta1=4.0, beta2=25.0)
    big = torch.linspace(-1000, 1000, steps=2001)
    out = act(big, big)
    print("max |SiTU-GLU(x)| over huge inputs:", out.abs().max().item(), "(bound = 100)")
    assert out.abs().max().item() <= 100.0 + 1e-3
    print("Stable LatentMoE OK")
