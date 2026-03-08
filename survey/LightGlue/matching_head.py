"""
LightGlue - Matching Head
=========================

LightGlueのマッチング予測ヘッドを疑似コードで示します。

主要コンポーネント:
1. Match Assignment (Double Softmax + Matchability)
2. Match Filtering (Mutual NN + Threshold)

論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)

Shape Convention:
    B: バッチサイズ
    M: Image A のキーポイント数
    N: Image B のキーポイント数
    C: 特徴記述子の埋め込み次元 (dim, default 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# Sigmoid Log Double Softmax
# ============================================================

def sigmoid_log_double_softmax(
    sim: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor
) -> torch.Tensor:
    """
    Double Softmax + Matchability による Assignment Matrix 計算

    ========================================
    SuperGlueとの比較
    ========================================

    SuperGlue (Sinkhorn Algorithm):
        - 最適輸送問題を解く
        - 100イテレーションの正規化
        - Dustbin で unmatchable を表現
        - 計算量大、メモリ大
        - Dustbin が全点の類似度に影響 (entangled)

    LightGlue (Double Softmax + Matchability):
        - 行・列両方向の softmax
        - 1回の計算で完了
        - Matchability で unmatchable を分離表現
        - 計算量小、メモリ小、勾配クリーン

    ========================================
    数学的表現
    ========================================

    Similarity Matrix: S ∈ R^{M×N}
        S_ij = proj(x_A^i)^T proj(x_B^j)

    Matchability Score: σ ∈ [0, 1]
        σ_i = sigmoid(linear(x_i))
        → 点iがマッチ可能かどうか

    Assignment Matrix: P ∈ R^{(M+1)×(N+1)}
        P_ij = σ_A^i × σ_B^j × softmax(S, row) × softmax(S, col)

        (log domain)
        log P_ij = log_sigmoid(z_A^i) + log_sigmoid(z_B^j)
                 + log_softmax(S, row)_ij
                 + log_softmax(S, col)_ij

    Unmatchable (dustbin equivalent):
        P_{i, M+1} = sigmoid(-z_A^i)  → A点iがunmatchable
        P_{N+1, j} = sigmoid(-z_B^j)  → B点jがunmatchable

    ========================================
    入力・出力
    ========================================

    入力:
        sim: (B, M, N) 類似度行列
        z0: (B, M, 1) Image A の matchability logits
        z1: (B, N, 1) Image B の matchability logits

    出力:
        scores: (B, M+1, N+1) log assignment matrix

    scores[b, i, j] = log P(point i in A matches point j in B)
    scores[b, i, -1] = log P(point i in A has no match in B)
    scores[b, -1, j] = log P(point j in B has no match in A)
    """
    # sim: (B, M, N)
    # z0: (B, M, 1)
    # z1: (B, N, 1)
    B, M, N = sim.shape

    # ========================================
    # Step 1: Matchability (certainty)
    # ========================================
    # 両点がマッチ可能であることの確信度
    # F.logsigmoid(z0): (B, M, 1) → 各A点のlog σ(z)
    # F.logsigmoid(z1): (B, N, 1) → transpose → (B, 1, N) → 各B点のlog σ(z)
    # certainties[b, i, j] = log σ(z_A^i) + log σ(z_B^j)
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    # certainties: (B, M, 1) + (B, 1, N) → broadcast → (B, M, N)

    # ========================================
    # Step 2: Double Softmax (similarity)
    # ========================================
    # Row-wise softmax: 各A点について、B点の中で最も類似
    # sim: (B, M, N) → dim=2でsoftmax → (B, M, N)
    scores0 = F.log_softmax(sim, dim=2)
    # scores0: (B, M, N) - 各行が確率分布 (log domain)

    # Column-wise softmax: 各B点について、A点の中で最も類似
    # sim: (B, M, N) → transpose → (B, N, M) → dim=2でsoftmax → (B, N, M) → transpose → (B, M, N)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), dim=2).transpose(-1, -2)
    # scores1: (B, M, N) - 各列が確率分布 (log domain)

    # ========================================
    # Step 3: Combined Assignment
    # ========================================
    # (M+1) × (N+1) の assignment matrix を構築
    # 最後の行・列は "unmatchable" を表す
    scores = sim.new_full((B, M + 1, N + 1), 0)
    # scores: (B, M+1, N+1) 初期値0

    # 通常のマッチング確率 (log domain での加算 = 確率の乗算)
    # scores0: (B, M, N) + scores1: (B, M, N) + certainties: (B, M, N) → (B, M, N)
    scores[:, :M, :N] = scores0 + scores1 + certainties
    # scores[:, :M, :N]: (B, M, N) = log(softmax_row × softmax_col × σ_A × σ_B)

    # Unmatchable scores
    # P(A点iがunmatchable) = 1 - σ(z_A^i) = σ(-z_A^i)
    # z0: (B, M, 1) → squeeze(-1) → (B, M) → logsigmoid(-z0) → (B, M)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    # scores[:, :-1, -1]: (B, M) = log σ(-z_A) = log(1 - σ(z_A))

    # P(B点jがunmatchable) = 1 - σ(z_B^j) = σ(-z_B^j)
    # z1: (B, N, 1) → squeeze(-1) → (B, N) → logsigmoid(-z1) → (B, N)
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    # scores[:, -1, :-1]: (B, N) = log σ(-z_B) = log(1 - σ(z_B))

    return scores
    # return: (B, M+1, N+1) log assignment matrix


# ============================================================
# Match Assignment Module
# ============================================================

class MatchAssignment(nn.Module):
    """
    マッチ予測ヘッド

    ========================================
    処理フロー
    ========================================

    1. 記述子埋め込みを投影 (類似度計算用)
    2. 類似度行列を計算
    3. Matchability score を予測
    4. Assignment matrix を構築

    ========================================
    なぜ Matchability を分離するか？
    ========================================

    SuperGlueのDustbin問題:
        - Dustbinは全点の類似度に影響
        - 類似度の高い点同士でも、他にもっと似た点があれば
          マッチ確率が下がる
        - 学習が不安定

    Matchabilityの利点:
        - 類似度と「マッチ可能性」を分離
        - 類似度: 「点iとjがどれだけ似ているか」
        - Matchability: 「点iがマッチ相手を持つか」
        - クリーンな勾配、安定した学習
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: 記述子埋め込みの次元 (C)

        パラメータ:
            matchability: Linear(C, 1) → 各点のmatchability logit
            final_proj: Linear(C, C) → 類似度計算用投影
        """
        super().__init__()
        self.dim = dim

        # Matchability predictor (unary)
        # 各点が「マッチ可能か」を予測
        # Linear(C, 1): (B, N, C) → (B, N, 1)
        self.matchability = nn.Linear(dim, 1, bias=True)

        # 類似度計算用の投影
        # Linear(C, C): (B, N, C) → (B, N, C)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assignment matrix を計算

        入力:
            desc0: (B, M, C) Image A の特徴記述子埋め込み
            desc1: (B, N, C) Image B の特徴記述子埋め込み

        出力:
            scores: (B, M+1, N+1) log assignment matrix
            sim: (B, M, N) 類似度行列 (debugging用)
        """
        # desc0: (B, M, C), desc1: (B, N, C)

        # ========================================
        # Step 1: 投影
        # ========================================
        # final_proj: Linear(C, C)
        mdesc0 = self.final_proj(desc0)
        # mdesc0: (B, M, C)
        mdesc1 = self.final_proj(desc1)
        # mdesc1: (B, N, C)

        # ========================================
        # Step 2: 正規化 (温度スケーリング)
        # ========================================
        # d^{-0.25} でスケーリング (d = C)
        # これは softmax の温度パラメータに相当
        # 通常のAttentionは d^{-0.5} だが、Double Softmaxでは2回softmaxするため d^{-0.25}
        _, _, d = mdesc0.shape  # d = C
        mdesc0 = mdesc0 / (d ** 0.25)
        # mdesc0: (B, M, C) - スケーリング済み
        mdesc1 = mdesc1 / (d ** 0.25)
        # mdesc1: (B, N, C) - スケーリング済み

        # ========================================
        # Step 3: 類似度行列
        # ========================================
        # S_ij = mdesc0_i^T @ mdesc1_j
        # einsum: (B, M, C) × (B, N, C) → (B, M, N)
        sim = torch.einsum('bmd, bnd -> bmn', mdesc0, mdesc1)
        # sim: (B, M, N)

        # ========================================
        # Step 4: Matchability logits
        # ========================================
        # matchability: Linear(C, 1)
        # desc0: (B, M, C) → (B, M, 1)
        z0 = self.matchability(desc0)
        # z0: (B, M, 1)

        # desc1: (B, N, C) → (B, N, 1)
        z1 = self.matchability(desc1)
        # z1: (B, N, 1)

        # ========================================
        # Step 5: Assignment matrix
        # ========================================
        # sim: (B, M, N), z0: (B, M, 1), z1: (B, N, 1) → (B, M+1, N+1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        # scores: (B, M+1, N+1)

        return scores, sim
        # scores: (B, M+1, N+1), sim: (B, M, N)

    def get_matchability(self, desc: torch.Tensor) -> torch.Tensor:
        """
        Matchability score の取得

        入力:
            desc: (B, N, C) 特徴記述子埋め込み

        処理:
            matchability: Linear(C, 1)
            desc: (B, N, C) → Linear → (B, N, 1) → sigmoid → (B, N, 1) → squeeze → (B, N)

        出力:
            matchability: (B, N) in [0, 1]

        用途:
            - Point pruning の判定
            - Confident + Low matchability → prune
        """
        # self.matchability(desc): (B, N, C) → (B, N, 1)
        # sigmoid: (B, N, 1) → (B, N, 1)
        # squeeze(-1): (B, N, 1) → (B, N)
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)
        # return: (B, N)


# ============================================================
# Match Filtering
# ============================================================

def filter_matches(
    scores: torch.Tensor,
    threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Log assignment matrix からマッチを抽出

    ========================================
    処理フロー
    ========================================

    1. 各行・列の最大値を取得
    2. Mutual Nearest Neighbor check
    3. 閾値でフィルタリング

    ========================================
    Mutual Nearest Neighbor (MNN)
    ========================================

    条件:
        - A点iの最近傍がB点j
        - かつ B点jの最近傍がA点i
        → (i, j) はマッチ

    これにより:
        - 一対一対応を強制
        - 曖昧なマッチを排除

    ========================================
    入力・出力
    ========================================

    入力:
        scores: (B, M+1, N+1) log assignment matrix
        threshold: マッチ判定閾値 (default: 0.1)

    出力:
        m0: (B, M) 各A点のマッチ先 (-1 = unmatched)
        m1: (B, N) 各B点のマッチ先 (-1 = unmatched)
        mscores0: (B, M) マッチスコア
        mscores1: (B, N) マッチスコア
    """
    # scores: (B, M+1, N+1)

    # Dustbin (最後の行・列) を除く
    valid_scores = scores[:, :-1, :-1]
    # valid_scores: (B, M, N)

    # ========================================
    # Step 1: 各行・列の最大値
    # ========================================
    # 各A点について、最も類似するB点
    # valid_scores: (B, M, N) → dim=2で最大値 → values: (B, M), indices: (B, M)
    max0 = valid_scores.max(dim=2)
    # max0.values: (B, M) - 各A点の最大スコア
    # max0.indices: (B, M) - 各A点の最近傍B点インデックス

    # 各B点について、最も類似するA点
    # valid_scores: (B, M, N) → dim=1で最大値 → values: (B, N), indices: (B, N)
    max1 = valid_scores.max(dim=1)
    # max1.values: (B, N) - 各B点の最大スコア
    # max1.indices: (B, N) - 各B点の最近傍A点インデックス

    argmax0 = max0.indices
    # argmax0: (B, M) - A点iの最近傍B点
    argmax1 = max1.indices
    # argmax1: (B, N) - B点jの最近傍A点

    # ========================================
    # Step 2: Mutual Nearest Neighbor Check
    # ========================================
    # A点のインデックス: [0, 1, ..., M-1]
    indices0 = torch.arange(argmax0.shape[1], device=argmax0.device).unsqueeze(0)
    # indices0: (1, M)
    # B点のインデックス: [0, 1, ..., N-1]
    indices1 = torch.arange(argmax1.shape[1], device=argmax1.device).unsqueeze(0)
    # indices1: (1, N)

    # A点iの最近傍がB点j、かつB点jの最近傍がA点i
    # argmax0: (B, M) → argmax1.gather(1, argmax0): (B, M) → 各A点iについて、
    #   argmax0[b,i]=j → argmax1[b,j] → A点iの最近傍B点jの最近傍A点
    # indices0 == ... → (B, M) bool: MNNが成立するか
    mutual0 = indices0 == argmax1.gather(1, argmax0)
    # mutual0: (B, M) bool - A点iがMNNか

    # 同様にB点側
    # argmax1: (B, N) → argmax0.gather(1, argmax1): (B, N) → 各B点jについて、
    #   argmax1[b,j]=i → argmax0[b,i] → B点jの最近傍A点iの最近傍B点
    mutual1 = indices1 == argmax0.gather(1, argmax1)
    # mutual1: (B, N) bool - B点jがMNNか

    # ========================================
    # Step 3: スコア計算
    # ========================================
    # exp で確率に変換 (log domain → probability)
    # max0.values: (B, M) → exp → (B, M)
    max0_prob = max0.values.exp()
    # max0_prob: (B, M) - 各A点の最大マッチ確率

    # Mutual でない場合は 0
    # mutual0: (B, M) bool, max0_prob: (B, M) → (B, M)
    mscores0 = torch.where(mutual0, max0_prob, torch.zeros_like(max0_prob))
    # mscores0: (B, M) - MNNでないA点は0

    # B点側のスコア: 対応するA点のスコアを取得
    # argmax1: (B, N) → mscores0.gather(1, argmax1): (B, N)
    mscores1 = torch.where(
        mutual1,
        mscores0.gather(1, argmax1),
        torch.zeros_like(max0_prob[:, :argmax1.shape[1]])
    )
    # mscores1: (B, N) - MNNでないB点は0

    # ========================================
    # Step 4: 閾値フィルタリング
    # ========================================
    # Mutual かつ スコアが閾値以上
    # mutual0: (B, M) & (mscores0 > threshold): (B, M) → (B, M) bool
    valid0 = mutual0 & (mscores0 > threshold)
    # valid0: (B, M) bool

    # B点側: A点の対応元もvalidである必要がある
    # argmax1: (B, N) → valid0.gather(1, argmax1): (B, N)
    valid1 = mutual1 & valid0.gather(1, argmax1)
    # valid1: (B, N) bool

    # 無効なマッチは -1
    m0 = torch.where(valid0, argmax0, torch.full_like(argmax0, -1))
    # m0: (B, M) - 各A点のマッチ先B点 (-1 = unmatched)
    m1 = torch.where(valid1, argmax1, torch.full_like(argmax1, -1))
    # m1: (B, N) - 各B点のマッチ先A点 (-1 = unmatched)

    return m0, m1, mscores0, mscores1
    # m0: (B, M), m1: (B, N), mscores0: (B, M), mscores1: (B, N)


# ============================================================
# 可視化用ユーティリティ
# ============================================================

def visualize_assignment(scores: torch.Tensor, title: str = "Assignment Matrix"):
    """
    Assignment matrix を可視化

    入力:
        scores: (M+1, N+1) log assignment matrix (single sample)
    """
    import matplotlib.pyplot as plt

    # 確率に変換
    # scores: (M+1, N+1) → exp → (M+1, N+1)
    probs = scores.exp().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full assignment matrix
    ax = axes[0]
    im = ax.imshow(probs, aspect='auto', cmap='viridis')
    ax.set_xlabel('Image B keypoints')
    ax.set_ylabel('Image A keypoints')
    ax.set_title(f'{title} (Full)')
    ax.axhline(probs.shape[0] - 1.5, color='red', linestyle='--', alpha=0.7)
    ax.axvline(probs.shape[1] - 1.5, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax)

    # Valid matches only (without dustbin)
    ax = axes[1]
    valid_probs = probs[:-1, :-1]
    im = ax.imshow(valid_probs, aspect='auto', cmap='viridis')
    ax.set_xlabel('Image B keypoints')
    ax.set_ylabel('Image A keypoints')
    ax.set_title(f'{title} (Valid matches)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def explain_matching_head():
    """
    Matching Head の動作説明

    Double Softmax + Matchability の直感的理解
    """
    print("=" * 60)
    print("LightGlue Matching Head の仕組み")
    print("=" * 60)

    print("""
    === 問題設定 ===

    Image A: M個のキーポイント (i = 1, ..., M)
    Image B: N個のキーポイント (j = 1, ..., N)

    目標: どの点がどの点に対応するかを予測

    === SuperGlue (最適輸送) ===

    Sinkhorn Algorithm:
        1. 類似度行列 S を計算
        2. 行正規化 → 列正規化 → 行正規化 → ... (100回)
        3. Dustbin (unmatchable) を追加

    問題:
        - 100イテレーション → 遅い
        - Dustbin が全体に影響 → 学習不安定

    === LightGlue (Double Softmax + Matchability) ===

    Step 1: 類似度計算
        S_ij = proj(x_A^i)^T @ proj(x_B^j)

    Step 2: Double Softmax
        row_softmax = softmax(S, dim=row)   → 「AのどれがBのjに最も似ているか」
        col_softmax = softmax(S, dim=col)   → 「BのどれがAのiに最も似ているか」
        combined = row_softmax × col_softmax

    Step 3: Matchability (分離された unmatchable 予測)
        σ_i = sigmoid(linear(x_i))  → 「点iがマッチ相手を持つ確率」

    Step 4: 最終 Assignment
        P_ij = σ_A^i × σ_B^j × combined_ij

    === 利点 ===

    1. 計算効率: 100回 → 1回
    2. クリーンな勾配: 分離されたmatchability
    3. 解釈性: similarity と matchability を分離

    === 例 ===

    A点1: 建物の角 (良い特徴点)
    A点2: 空の領域 (悪い特徴点、B画像に対応なし)
    B点1: 建物の角 (A点1と対応)
    B点2: 別の建物

    SuperGlue:
        A点2のdustbinスコアが高い → 全体の正規化に影響

    LightGlue:
        matchability(A点1) = 0.95  (高い)
        matchability(A点2) = 0.05  (低い)
        similarity(A点1, B点1) = 高い
        → P(A点1 ↔ B点1) = 0.95 × 0.9 × 高い × 高い = 高い確率
        → P(A点2 ↔ any) = 0.05 × ... = 低い確率
    """)


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Matching Head の使用例
    """
    B, M, N = 2, 64, 48
    dim = 256  # C

    # 入力
    desc0 = torch.randn(B, M, dim)  # (B, M, C) = (2, 64, 256)
    desc1 = torch.randn(B, N, dim)  # (B, N, C) = (2, 48, 256)

    # Matching Head
    match_head = MatchAssignment(dim)

    # ========================================
    # Forward
    # ========================================
    # desc0: (B, M, C) = (2, 64, 256)
    # desc1: (B, N, C) = (2, 48, 256)
    # → scores: (B, M+1, N+1) = (2, 65, 49)
    # → sim: (B, M, N) = (2, 64, 48)
    scores, sim = match_head(desc0, desc1)

    print("=== Match Assignment ===")
    print(f"Input desc0: {desc0.shape}")   # (2, 64, 256)
    print(f"Input desc1: {desc1.shape}")   # (2, 48, 256)
    print(f"Similarity: {sim.shape}")      # (2, 64, 48)
    print(f"Scores: {scores.shape}")       # (2, 65, 49)

    # ========================================
    # Matchability
    # ========================================
    # desc0: (B, M, C) = (2, 64, 256) → (B, M) = (2, 64)
    matchability0 = match_head.get_matchability(desc0)
    # desc1: (B, N, C) = (2, 48, 256) → (B, N) = (2, 48)
    matchability1 = match_head.get_matchability(desc1)

    print(f"\nMatchability A: {matchability0.shape}, mean={matchability0.mean():.3f}")
    # (2, 64)
    print(f"Matchability B: {matchability1.shape}, mean={matchability1.mean():.3f}")
    # (2, 48)

    # ========================================
    # Filter Matches
    # ========================================
    # scores: (B, M+1, N+1) = (2, 65, 49)
    # → m0: (B, M) = (2, 64), m1: (B, N) = (2, 48)
    # → mscores0: (B, M) = (2, 64), mscores1: (B, N) = (2, 48)
    m0, m1, mscores0, mscores1 = filter_matches(scores, threshold=0.1)

    print("\n=== Match Filtering ===")
    print(f"Matches (A→B): {m0.shape}")     # (2, 64)
    print(f"Scores (A): {mscores0.shape}")  # (2, 64)

    # 有効なマッチ数をカウント
    for b in range(B):
        n_matches = (m0[b] >= 0).sum().item()
        avg_score = mscores0[b][m0[b] >= 0].mean().item() if n_matches > 0 else 0
        print(f"Batch {b}: {n_matches} matches, avg_score={avg_score:.3f}")

    # ========================================
    # Score 分布の確認
    # ========================================
    print("\n=== Score Distribution ===")
    valid_scores = scores[:, :-1, :-1]  # (B, M, N) = (2, 64, 48) dustbin除く
    print(f"Valid scores range: [{valid_scores.min():.3f}, {valid_scores.max():.3f}]")

    # Dustbin scores
    unmatch_A = scores[:, :-1, -1]  # (B, M) = (2, 64) A点がunmatchable
    unmatch_B = scores[:, -1, :-1]  # (B, N) = (2, 48) B点がunmatchable
    print(f"Unmatch A mean: {unmatch_A.mean():.3f}")
    print(f"Unmatch B mean: {unmatch_B.mean():.3f}")


if __name__ == "__main__":
    explain_matching_head()
    print("\n" + "=" * 60 + "\n")
    example_usage()
