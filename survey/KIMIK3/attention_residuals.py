"""Attention Residuals (AttnRes) -- Kimi K3 論文 §2.2 (sec:attnres) の実装。

通常の残差接続 (ResNet 型) は深さ方向の情報を単一のベクトル h_l に圧縮してしまう
(時間方向における RNN のボトルネックと同じ構造)。AttnRes はこれを「深さ方向への
アテンション」で置き換え、各層が全ての先行層の出力を選択的に読み出せるようにする
(§2.2 冒頭)。

* `FullAttentionResidual` : Eq.(attnres-qkv), Eq.(attnres-full) をそのまま実装。
  全 L 層の出力を保持するため O(L^2 d) の計算量・O(Ld) のメモリを要するが、
  数式に忠実で最も読みやすい (層数 L が小さい場合のみ実用的)。
* `BlockAttentionResidual` : Eq.(attnres-block) を実装。L 層を S 層ずつ N=L/S 個の
  ブロックに分割し、ブロック内の出力を総和 b_n に集約してからブロック間でのみ
  full attention を行う。メモリ/通信量が O(Ld) から O(Nd) に削減され、
  Kimi K3 では N≈8 (S=12) を採用 (§2.2 末尾)。この実装は Kimi-K3 公式コード
  (modeling_kimi_linear.py の `_apply_attn_res`) と数学的に同一の定式化を採用しており、
  カーネル phi(q,k) = exp(q^T RMSNorm(k)) を用いたソフトマックスアテンションとして
  計算する (RMSNorm が大振幅な層の出力がスコアを支配するのを防ぐ, §2.2)。
"""
from __future__ import annotations

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


class FullAttentionResidual(nn.Module):
    """Eq.(attnres-qkv), Eq.(attnres-full) の素朴な実装 (教育目的、L が小さい場合用)。

    形状の記法:
        N          : トークン数 (B*T をフラット化したもの)
        L          : 現在までに蓄積された層出力の数 (0 = embedding, 1..l-1 = 各層の出力)
        d          : モデル隠れ次元
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.k_norm = KimiRMSNorm(hidden_size)

    def forward(
        self,
        pseudo_query: torch.Tensor,     # (d,)   w_l: 層 l 固有の学習可能な擬似クエリ
        layer_outputs: torch.Tensor,    # (N, L, d)  k_i = v_i = [h_1(=embed), f_1(h_1), ..., f_{l-1}(h_{l-1})]
    ) -> torch.Tensor:
        """
        Returns:
            h_l: (N, d) -- Eq.(attnres-full) の重み付き和
        """
        k = self.k_norm(layer_outputs)                      # RMSNorm(k_i): (N, L, d)
        scores = torch.einsum("d,nld->nl", pseudo_query, k)  # phi の指数部 q^T RMSNorm(k_i): (N, L)
        alpha = torch.softmax(scores, dim=-1)                 # Eq: alpha_{i->l} (softmax kernel)
        h_l = torch.einsum("nl,nld->nd", alpha, layer_outputs)  # sum_i alpha_i * v_i
        return h_l


class BlockAttentionResidual(nn.Module):
    """Eq.(attnres-block) のブロック化実装。1層分のモジュールで、backbone 側が
    `block_residual` (ブロック代表ベクトル列) を管理しながら各層でこのモジュールを呼ぶ。

    公式実装 (`_apply_attn_res`) と同様、学習可能な擬似クエリ w_l は
    `proj: Linear(d, 1, bias=False)` の重みベクトルとして持たせ、RMSNorm の
    weight と要素積を取ることで `q^T RMSNorm(k)` を 1 回の重み付き和で計算する。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = KimiRMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, 1, bias=False)  # 行ベクトルが擬似クエリ w_l

    def forward(self, prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prefix_sum: (N, d) 直前までの層出力の部分和 (現在のブロック内での途中経過、
                Eq の b_n^{i-1} に相当。ブロック境界では空のブロックの部分和として embedding
                そのものが渡される)
            block_residual: (N, M, d) 過去に確定したブロック代表ベクトル [b_0, ..., b_{n-1}]
                (M = これまでに確定したブロック数)
        Returns:
            h: (N, d) -- Eq.(attnres-block) の V に対して Eq.(attnres-full) のアテンションを適用した結果
        """
        # V = [b_0, ..., b_{n-1}, prefix_sum] : (N, M+1, d)
        v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)

        # q^T RMSNorm(k_i) = q^T (norm.weight ⊙ normalize(k_i))
        #                  = (normalize(k_i)) ・ (norm.weight ⊙ q)
        # なので「正規化のみ (weightなし)」の k に対し、score_weight = norm.weight ⊙ proj.weight
        # を内積すれば良い (RMSNorm の weight を二重適用しないための書き方)。
        score_weight = self.norm.weight * self.proj.weight.squeeze(0)  # (d,)
        v_float = v.float()
        variance = v_float.pow(2).mean(-1, keepdim=True)
        k_unweighted = v_float * torch.rsqrt(variance + self.norm.eps)
        scores = (k_unweighted * score_weight.float()).sum(-1)  # (N, M+1)

        alpha = torch.softmax(scores, dim=-1).unsqueeze(1)  # (N, 1, M+1)
        h = torch.matmul(alpha, v_float).squeeze(1)          # (N, d)
        return h.to(v.dtype)


class BlockAttnResBackbone(nn.Module):
    """L 層のバックボーンに Block AttnRes を組み込んだラッパーの最小例。

    各層 `layer_fn(h) -> f_l(h)` の出力を Eq.(attnres-block) に従って集約する。
    `attn_res_block_size` = S ごとにブロックが確定し、ブロック代表ベクトル
    `b_n = sum_{j in block n} f_j(h_j)` が `block_residual` に追加される。
    """

    def __init__(self, hidden_size: int, num_layers: int, block_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.block_size = block_size
        self.blocks = nn.ModuleList(
            [BlockAttentionResidual(hidden_size) for _ in range(num_layers)]
        )
        self.output_block = BlockAttentionResidual(hidden_size)

    def forward(self, embedding: torch.Tensor, layer_fns: list) -> torch.Tensor:
        """
        Args:
            embedding: (N, d)  トークン埋め込み h_1 (= b_0, Eq: "b_0 = h_1")
            layer_fns: 各層の変換 f_l: (N, d) -> (N, d) を行う callable のリスト (len == num_layers)
        Returns:
            (N, d) -- 全ブロックを集約した最終出力
        """
        N, d = embedding.shape
        block_residual = embedding.new_zeros(N, 0, d)  # 確定済みブロック代表 [b_0, ...] (最初は空)
        prefix_sum = embedding  # b_0 = h_1 をまず prefix_sum として保持

        for layer_idx, (block, layer_fn) in enumerate(zip(self.blocks, layer_fns)):
            if block_residual.shape[1] > 0 or layer_idx > 0:
                h = block(prefix_sum, block_residual)
            else:
                h = prefix_sum  # 最初の層の直前は b_0 = embedding のみなので softmax は自明

            if layer_idx % self.block_size == 0:
                # ブロック境界: これまでの prefix_sum (embedding を含む) を新しいブロック代表として確定
                block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
                prefix_sum = layer_fn(h)
            else:
                prefix_sum = prefix_sum + layer_fn(h)

        out = self.output_block(prefix_sum, block_residual)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    N, d, L = 5, 32, 6

    # --- Full AttnRes: 数式通りの素朴な実装の動作確認 ---
    full = FullAttentionResidual(d)
    pseudo_query = torch.randn(d)
    layer_outputs = torch.randn(N, L, d)
    h_full = full(pseudo_query, layer_outputs)
    print("FullAttentionResidual output shape:", h_full.shape)  # (5, 32)

    # --- Block AttnRes: N=8 相当のブロック構造をバックボーンに組み込んだ例 ---
    num_layers = 12
    block_size = 3  # 4ブロック相当 (論文では S=12, N=8 だがここでは小規模デモ)
    backbone = BlockAttnResBackbone(d, num_layers, block_size)
    layer_fns = [nn.Linear(d, d) for _ in range(num_layers)]
    embedding = torch.randn(N, d)
    out = backbone(embedding, layer_fns)
    print("BlockAttnResBackbone output shape:", out.shape)  # (5, 32)
    assert out.shape == (N, d)
    print("AttentionResiduals OK")
