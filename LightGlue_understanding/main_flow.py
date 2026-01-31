"""
LightGlue - メインフロー
========================

LightGlueの全体処理フローを疑似コードで示します。

論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)
公式実装: https://github.com/cvg/LightGlue

処理の流れ:
1. 入力正規化 & 投影
2. Positional Encoding (Rotary)
3. Transformer Layers (Self + Cross Attention)
4. Adaptive Inference (Early Stop + Pruning)
5. Matching Head
6. Match Filtering

============================================================
Shape Convention
============================================================
B: バッチサイズ
M: Image 0 のキーポイント数
N: Image 1 のキーポイント数
D: 入力記述子次元 (SuperPoint: 256, DISK/ALIKED: 128)
C: 特徴記述子の埋め込み次元 (default: 256)
H: Attention head数 (default: 4)
head_dim: C // H = 64
L: Transformerレイヤー数 (default: 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================
# 設定
# ============================================================

DEFAULT_CONFIG = {
    'input_dim': 256,           # D: 入力記述子次元 (SuperPoint: 256, DISK: 128)
    'descriptor_dim': 256,      # C: 特徴記述子の埋め込み次元
    'n_layers': 9,              # L: Transformerレイヤー数
    'num_heads': 4,             # H: Attention heads
    'depth_confidence': 0.95,   # Early stopping閾値 (-1で無効)
    'width_confidence': 0.99,   # Point pruning閾値 (-1で無効)
    'filter_threshold': 0.1,    # マッチ閾値
    'flash': True,              # FlashAttention使用
    'add_scale_ori': False,     # SIFT用: scale/orientation追加
}

# 対応する特徴量とその設定
FEATURE_CONFIGS = {
    'superpoint': {'input_dim': 256, 'add_scale_ori': False},  # D=256
    'disk': {'input_dim': 128, 'add_scale_ori': False},        # D=128
    'aliked': {'input_dim': 128, 'add_scale_ori': False},      # D=128
    'sift': {'input_dim': 128, 'add_scale_ori': True},         # D=128, pos_dim=4
    'doghardnet': {'input_dim': 128, 'add_scale_ori': True},   # D=128, pos_dim=4
}


# ============================================================
# キーポイント正規化
# ============================================================

def normalize_keypoints(
    kpts: torch.Tensor,
    size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    キーポイントを[-1, 1]に正規化

    ========================================
    Shape
    ========================================
    入力:
        kpts: (B, N, 2)
            - B: バッチサイズ
            - N: キーポイント数
            - 2: [x, y] ピクセル座標
        size: (B, 2) or (2,) or None
            - [W, H] 画像サイズ

    出力:
        kpts_norm: (B, N, 2)
            - 正規化座標 [-1, 1]

    ========================================
    処理詳細
    ========================================
    1. 画像中心を原点に移動
    2. 最大辺でスケーリング（アスペクト比維持）

    数式:
        shift = size / 2                    # (B, 2) or (2,)
        scale = max(size) / 2               # (B,) or scalar
        kpts_norm = (kpts - shift) / scale  # (B, N, 2)
    """
    if size is None:
        # サイズ不明時は自動推定
        # kpts.max(-2): (B, 2), kpts.min(-2): (B, 2)
        size = 1 + kpts.max(-2).values - kpts.min(-2).values  # (B, 2)

    size = size.to(kpts.device, kpts.dtype)  # (B, 2) or (2,)
    shift = size / 2                          # (B, 2) or (2,) - 中心座標
    scale = size.max(-1).values / 2           # (B,) or scalar - 最大辺の半分

    # Broadcasting:
    # kpts: (B, N, 2)
    # shift[..., None, :]: (B, 1, 2) or (1, 2)
    # scale[..., None, None]: (B, 1, 1) or scalar
    kpts_norm = (kpts - shift[..., None, :]) / scale[..., None, None]
    # kpts_norm: (B, N, 2)

    return kpts_norm


# ============================================================
# LightGlue メインクラス
# ============================================================

class LightGlue(nn.Module):
    """
    LightGlue: 局所特徴量マッチングネットワーク

    ========================================
    入力 Shape
    ========================================
    data: dict
        'image0': dict
            'keypoints': (B, M, 2)      - キーポイント座標
            'descriptors': (B, M, D)    - 記述子 (D=256 for SuperPoint)
            'image_size': (2,)          - [W, H]
        'image1': dict
            'keypoints': (B, N, 2)
            'descriptors': (B, N, D)
            'image_size': (2,)

    ========================================
    出力 Shape
    ========================================
    dict
        'matches0': (B, M)              - 各点のマッチ先 (-1 = unmatched)
        'matches1': (B, N)              - 各点のマッチ先 (-1 = unmatched)
        'matching_scores0': (B, M)      - マッチ信頼度 [0, 1]
        'matching_scores1': (B, N)      - マッチ信頼度 [0, 1]
        'matches': List[(S_b, 2)]       - バッチごとのマッチペア (S_b: batch b のマッチ数)
        'scores': List[(S_b,)]          - マッチスコア
        'stop': int                     - 終了レイヤー [1, L]
        'prune0': (B, M)                - 各点のpruningレイヤー
        'prune1': (B, N)                - 各点のpruningレイヤー

    ========================================
    内部次元
    ========================================
    - D: input_dim = 256 (SuperPoint), 128 (DISK/ALIKED)
    - C: descriptor_dim = 256 (特徴記述子の埋め込み次元)
    - H: num_heads = 4
    - head_dim: C // H = 64
    - L: n_layers = 9
    """

    def __init__(self, features='superpoint', **config):
        super().__init__()

        # 設定をマージ
        self.config = {**DEFAULT_CONFIG, **config}
        if features in FEATURE_CONFIGS:
            self.config.update(FEATURE_CONFIGS[features])

        D = self.config['input_dim']       # D: 入力次元
        C = self.config['descriptor_dim']  # C: 内部次元
        L = self.config['n_layers']        # L: レイヤー数
        H = self.config['num_heads']       # H: ヘッド数
        head_dim = C // H                  # head_dim = 64

        # ========================================
        # 1. Input Projection: (B, N, D) -> (B, N, C)
        # ========================================
        # 異なる特徴量次元を内部次元Cに統一
        if D != C:
            self.input_proj = nn.Linear(D, C, bias=True)
            # weight: (C, D) = (256, D)
            # bias: (C,) = (256,)
        else:
            self.input_proj = nn.Identity()

        # ========================================
        # 2. Positional Encoding (Rotary)
        # ========================================
        # pos_dim: 2 (x, y) or 4 (x, y, scale, orientation)
        pos_dim = 2 + 2 * self.config['add_scale_ori']
        self.posenc = LearnableFourierPositionalEncoding(
            M=pos_dim,           # 入力次元: 2 or 4
            dim=head_dim,        # 出力次元: 64
            F_dim=head_dim       # Fourier次元: 64
        )
        # 出力: (2, B, N, head_dim) = (2, B, N, 64) for cos/sin

        # ========================================
        # 3. Transformer Layers
        # ========================================
        # 各レイヤー: Self-Attention + Cross-Attention
        self.transformers = nn.ModuleList([
            TransformerLayer(
                dim=C,           # C=256
                num_heads=H,     # H=4
                flash=self.config['flash']
            )
            for _ in range(L)  # L=9 レイヤー
        ])

        # ========================================
        # 4. Matching Heads (各レイヤー用)
        # ========================================
        # Deep Supervision: 全レイヤーでマッチング予測
        self.log_assignment = nn.ModuleList([
            MatchAssignment(dim=C)  # C=256
            for _ in range(L)       # L=9
        ])
        # 出力: (B, M+1, N+1) log assignment matrix

        # ========================================
        # 5. Confidence Classifiers (最終レイヤー以外)
        # ========================================
        # Early stopping判定用
        self.token_confidence = nn.ModuleList([
            TokenConfidence(dim=C)  # C=256
            for _ in range(L - 1)   # L-1=8
        ])
        # 出力: (B, M), (B, N) confidence scores

        # ========================================
        # 6. 層ごとの確信度閾値（キャッシュ）
        # ========================================
        self.register_buffer(
            'confidence_thresholds',
            torch.tensor([self.confidence_threshold(i) for i in range(L)])
        )
        # confidence_thresholds: (L,) = (9,)

    def forward(self, data: Dict) -> Dict:
        """
        メイン推論関数

        ========================================
        処理フロー & Shape変化
        ========================================
        1. 入力データ取得
           kpts0: (B, M, 2), desc0: (B, M, D)
           kpts1: (B, N, 2), desc1: (B, N, D)

        2. キーポイント正規化
           kpts0: (B, M, 2) -> (B, M, 2) in [-1, 1]

        3. 入力投影
           desc0: (B, M, D) -> (B, M, C)

        4. 位置エンコーディング
           kpts0: (B, M, 2) -> encoding0: (2, B, M, head_dim)

        5. Transformerレイヤー × L回
           desc0: (B, M, C), desc1: (B, N, C)
           -> desc0: (B, M', C), desc1: (B, N', C)  # Pruning後

        6. Matching Head
           desc0: (B, M', C), desc1: (B, N', C)
           -> scores: (B, M'+1, N'+1)

        7. Match Filtering
           scores: (B, M'+1, N'+1)
           -> matches0: (B, M), matches1: (B, N)
        """
        # ========================================
        # Step 1: 入力データ取得
        # ========================================
        data0, data1 = data['image0'], data['image1']

        kpts0 = data0['keypoints']   # (B, M, 2) - ピクセル座標
        kpts1 = data1['keypoints']   # (B, N, 2) - ピクセル座標
        desc0 = data0['descriptors'] # (B, M, D) - 記述子
        desc1 = data1['descriptors'] # (B, N, D) - 記述子

        B, M, _ = kpts0.shape  # B: batch, M: image0のキーポイント数
        _, N, _ = kpts1.shape  # N: image1のキーポイント数
        device = kpts0.device

        size0 = data0.get('image_size')  # (2,) or (B, 2) or None
        size1 = data1.get('image_size')  # (2,) or (B, 2) or None

        # ========================================
        # Step 2: キーポイント正規化
        # ========================================
        # (B, M, 2) -> (B, M, 2) in [-1, 1]
        kpts0 = normalize_keypoints(kpts0, size0)
        # (B, N, 2) -> (B, N, 2) in [-1, 1]
        kpts1 = normalize_keypoints(kpts1, size1)

        # SIFT用: scale/orientationを追加
        if self.config['add_scale_ori']:
            # scales: (B, M), oris: (B, M)
            # kpts0: (B, M, 2) -> (B, M, 4)
            kpts0 = torch.cat([
                kpts0,                              # (B, M, 2)
                data0['scales'].unsqueeze(-1),      # (B, M, 1)
                data0['oris'].unsqueeze(-1)         # (B, M, 1)
            ], dim=-1)  # -> (B, M, 4)
            kpts1 = torch.cat([
                kpts1,                              # (B, N, 2)
                data1['scales'].unsqueeze(-1),      # (B, N, 1)
                data1['oris'].unsqueeze(-1)         # (B, N, 1)
            ], dim=-1)  # -> (B, N, 4)

        # ========================================
        # Step 3: 入力投影
        # ========================================
        # desc0: (B, M, D) -> (B, M, C)
        # D=256 (SuperPoint), C=256 -> Identity
        # D=128 (DISK), C=256 -> Linear projection
        desc0 = self.input_proj(desc0.detach())  # (B, M, C) = (B, M, 256)
        desc1 = self.input_proj(desc1.detach())  # (B, N, C) = (B, N, 256)

        # ========================================
        # Step 4: 位置エンコーディング（キャッシュ）
        # ========================================
        # kpts0: (B, M, 2 or 4) -> encoding0: (2, B, M, head_dim)
        # 2: [cos, sin] for Rotary PE
        encoding0 = self.posenc(kpts0)  # (2, B, M, head_dim) = (2, B, M, 64)
        encoding1 = self.posenc(kpts1)  # (2, B, N, head_dim) = (2, B, N, 64)

        # ========================================
        # Step 5: Transformer Layers
        # ========================================
        do_early_stop = self.config['depth_confidence'] > 0
        do_point_pruning = self.config['width_confidence'] > 0

        # Pruning用のインデックス追跡
        if do_point_pruning:
            # ind0: 現在の点が元のどのインデックスに対応するか
            ind0 = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)  # (B, M)
            ind1 = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
            # prune0: 各点がいつpruneされたか
            prune0 = torch.ones(B, M, device=device, dtype=torch.long)  # (B, M)
            prune1 = torch.ones(B, N, device=device, dtype=torch.long)  # (B, N)

        token0, token1 = None, None
        L = self.config['n_layers']  # L=9

        for i in range(L):
            # 現在のキーポイント数
            # desc0: (B, M_i, C), desc1: (B, N_i, C)
            M_i, N_i = desc0.shape[1], desc1.shape[1]

            # キーポイントがない場合は早期終了
            if M_i == 0 or N_i == 0:
                break

            # ----- Transformer Layer -----
            # desc0: (B, M_i, C), desc1: (B, N_i, C)
            # encoding0: (2, B, M_i, head_dim), encoding1: (2, B, N_i, head_dim)
            # -> desc0: (B, M_i, C), desc1: (B, N_i, C)
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1
            )

            # 最終レイヤーでは適応的処理をスキップ
            if i == L - 1:
                continue

            # ----- Adaptive Depth: Early Stopping -----
            if do_early_stop:
                # token0: (B, M_i), token1: (B, N_i) - confidence scores
                token0, token1 = self.token_confidence[i](desc0, desc1)

                # M, Nは元のサイズ (pruning前)
                if self.check_if_stop(token0, token1, i, M + N):
                    break  # 早期終了!

            # ----- Adaptive Width: Point Pruning -----
            pruning_threshold = 1024  # 最小キーポイント数

            if do_point_pruning and M_i > pruning_threshold:
                # Matchability score: (B, M_i)
                scores0 = self.log_assignment[i].get_matchability(desc0)
                # Pruning mask: (B, M_i) bool
                keep_mask0 = self.get_pruning_mask(token0, scores0, i)
                # 保持する点のインデックス: (K0,) where K0 <= M_i
                keep0 = torch.where(keep_mask0.flatten())[0]  # (K0,)
                num_keep0 = keep_mask0.sum().item()

                if num_keep0 < M_i:
                    # インデックス更新
                    # ind0: (B, M_i) -> (B, num_keep0)
                    ind0 = ind0.flatten()[keep0].view(B, -1)
                    # desc0: (B, M_i, C) -> (B, num_keep0, C)
                    desc0 = desc0.flatten(0, 1)[keep0].view(B, -1, desc0.shape[-1])
                    # encoding0: (2, B, M_i, head_dim) -> (2, B, num_keep0, head_dim)
                    encoding0 = encoding0.flatten(1, 2)[:, keep0].view(2, B, -1, encoding0.shape[-1])
                    # pruneレイヤーを記録
                    prune0.flatten()[keep0] += 1

            if do_point_pruning and N_i > pruning_threshold:
                # scores1: (B, N_i)
                scores1 = self.log_assignment[i].get_matchability(desc1)
                # keep_mask1: (B, N_i)
                keep_mask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(keep_mask1.flatten())[0]  # (K1,)
                num_keep1 = keep_mask1.sum().item()

                if num_keep1 < N_i:
                    ind1 = ind1.flatten()[keep1].view(B, -1)
                    desc1 = desc1.flatten(0, 1)[keep1].view(B, -1, desc1.shape[-1])
                    encoding1 = encoding1.flatten(1, 2)[:, keep1].view(2, B, -1, encoding1.shape[-1])
                    prune1.flatten()[keep1] += 1

        # ========================================
        # Step 6: Matching Head
        # ========================================
        # 最終的なキーポイント数
        M_final, N_final = desc0.shape[1], desc1.shape[1]

        if M_final == 0 or N_final == 0:
            # マッチなし
            return self._empty_result(B, M, N, device, i + 1)

        # Assignment matrix
        # desc0: (B, M_final, C), desc1: (B, N_final, C)
        # -> scores: (B, M_final+1, N_final+1) log assignment matrix
        # -> sim: (B, M_final, N_final) similarity matrix (optional)
        scores, sim = self.log_assignment[i](desc0, desc1)

        # ========================================
        # Step 7: Match Filtering
        # ========================================
        # scores: (B, M_final+1, N_final+1)
        # -> m0: (B, M_final) 各点のマッチ先 (-1 = unmatched)
        # -> m1: (B, N_final)
        # -> mscores0: (B, M_final) マッチスコア
        # -> mscores1: (B, N_final)
        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.config['filter_threshold']
        )

        # マッチペアを構築
        matches, mscores = [], []
        for b in range(B):
            # valid: (M_final,) bool
            valid = m0[b] > -1
            # m_indices_0: (S_b,) where S_b = valid.sum()
            m_indices_0 = torch.where(valid)[0]
            # m_indices_1: (S_b,)
            m_indices_1 = m0[b][valid]

            # Pruning使用時: 元のインデックスに復元
            if do_point_pruning:
                m_indices_0 = ind0[b, m_indices_0]  # (S_b,)
                m_indices_1 = ind1[b, m_indices_1]  # (S_b,)

            # matches[b]: (S_b, 2)
            matches.append(torch.stack([m_indices_0, m_indices_1], dim=-1))
            # mscores[b]: (S_b,)
            mscores.append(mscores0[b][valid])

        # インデックスを元のサイズに復元
        if do_point_pruning:
            # m0_: (B, M), m1_: (B, N)
            # mscores0_: (B, M), mscores1_: (B, N)
            m0_, m1_, mscores0_, mscores1_ = self._restore_indices(
                m0, m1, mscores0, mscores1, ind0, ind1, M, N, B
            )
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones(B, M, device=device) * L  # (B, M)
            prune1 = torch.ones(B, N, device=device) * L  # (B, N)

        return {
            'matches0': m0,                 # (B, M) マッチ先 (-1=unmatched)
            'matches1': m1,                 # (B, N) マッチ先 (-1=unmatched)
            'matching_scores0': mscores0,   # (B, M) スコア [0, 1]
            'matching_scores1': mscores1,   # (B, N) スコア [0, 1]
            'matches': matches,             # List[(S_b, 2)] バッチごとのマッチペア
            'scores': mscores,              # List[(S_b,)] マッチスコア
            'stop': i + 1,                  # int: 終了レイヤー [1, L]
            'prune0': prune0,               # (B, M) 各点のpruningレイヤー
            'prune1': prune1,               # (B, N) 各点のpruningレイヤー
        }

    def confidence_threshold(self, layer_index: int) -> float:
        """
        層ごとの確信度閾値

        ========================================
        数式
        ========================================
        λ_l = 0.8 + 0.1 × exp(-4l/L)

        l: 現在のレイヤー [0, L-1]
        L: 総レイヤー数 (9)

        ========================================
        値の例 (L=9)
        ========================================
        Layer 0: λ_0 = 0.9
        Layer 4: λ_4 ≈ 0.82
        Layer 8: λ_8 ≈ 0.8
        """
        L = self.config['n_layers']
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / L)
        return np.clip(threshold, 0, 1)

    def check_if_stop(
        self,
        token0: torch.Tensor,
        token1: torch.Tensor,
        layer_index: int,
        num_points: int
    ) -> bool:
        """
        早期終了の判定

        ========================================
        Shape
        ========================================
        入力:
            token0: (B, M') - image0の確信度
            token1: (B, N') - image1の確信度
            layer_index: int - 現在のレイヤー
            num_points: int - 元のキーポイント総数 (M + N)

        出力:
            bool - 終了するか

        ========================================
        条件
        ========================================
        (確信度 > λ_l の点の割合) > α

        where:
            α = depth_confidence (default: 0.95)
            λ_l = confidence_threshold(l)
        """
        # confidences: (B, M' + N')
        confidences = torch.cat([token0, token1], dim=-1)

        # threshold: scalar
        threshold = self.confidence_thresholds[layer_index]

        # ratio_confident: 確信度が閾値を超える点の割合
        # (confidences < threshold): (B, M' + N') bool
        # sum(): scalar
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points

        return ratio_confident > self.config['depth_confidence']

    def get_pruning_mask(
        self,
        confidences: torch.Tensor,
        matchability: torch.Tensor,
        layer_index: int
    ) -> torch.Tensor:
        """
        Point Pruning のマスク取得

        ========================================
        Shape
        ========================================
        入力:
            confidences: (B, M') or None - 確信度
            matchability: (B, M') - マッチ可能性
            layer_index: int

        出力:
            keep: (B, M') bool - 保持する点

        ========================================
        ロジック
        ========================================
        保持する点:
            1. Matchability が高い (マッチ可能性あり)
               matchability > (1 - width_confidence) = 0.01
            2. または Confidence が低い (まだ不確実)
               confidences <= λ_l

        除外する点:
            Confidence が高く、かつ Matchability が低い
            → マッチ相手がいないと確信
        """
        # Matchability閾値: 1 - width_confidence = 0.01
        # keep: (B, M') bool
        keep = matchability > (1 - self.config['width_confidence'])

        # Low-confidence points are never pruned
        if confidences is not None:
            # λ_l: scalar
            # keep: (B, M') bool
            keep = keep | (confidences <= self.confidence_thresholds[layer_index])

        return keep

    def _empty_result(self, B: int, M: int, N: int, device, stop_layer: int) -> Dict:
        """
        マッチなしの結果を生成

        ========================================
        Shape
        ========================================
        B: バッチサイズ
        M: Image 0 のキーポイント数
        N: Image 1 のキーポイント数
        """
        L = self.config['n_layers']
        return {
            'matches0': torch.full((B, M), -1, dtype=torch.long, device=device),           # (B, M)
            'matches1': torch.full((B, N), -1, dtype=torch.long, device=device),           # (B, N)
            'matching_scores0': torch.zeros(B, M, device=device),                          # (B, M)
            'matching_scores1': torch.zeros(B, N, device=device),                          # (B, N)
            'matches': [torch.empty(0, 2, dtype=torch.long, device=device) for _ in range(B)],  # List[(0, 2)]
            'scores': [torch.empty(0, device=device) for _ in range(B)],                   # List[(0,)]
            'stop': stop_layer,                                                            # int
            'prune0': torch.ones(B, M, device=device) * L,                                 # (B, M)
            'prune1': torch.ones(B, N, device=device) * L,                                 # (B, N)
        }

    def _restore_indices(
        self,
        m0: torch.Tensor,
        m1: torch.Tensor,
        mscores0: torch.Tensor,
        mscores1: torch.Tensor,
        ind0: torch.Tensor,
        ind1: torch.Tensor,
        M: int,
        N: int,
        B: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pruning後のインデックスを元のサイズに復元

        ========================================
        Shape
        ========================================
        入力:
            m0: (B, M') - pruning後のマッチ
            m1: (B, N')
            mscores0: (B, M')
            mscores1: (B, N')
            ind0: (B, M') - 元インデックスへのマッピング
            ind1: (B, N')
            M: 元のM
            N: 元のN
            B: バッチサイズ

        出力:
            m0_: (B, M) - 元サイズのマッチ
            m1_: (B, N)
            mscores0_: (B, M)
            mscores1_: (B, N)
        """
        # 初期化: すべて-1 (unmatched)
        m0_ = torch.full((B, M), -1, device=m0.device, dtype=m0.dtype)  # (B, M)
        m1_ = torch.full((B, N), -1, device=m1.device, dtype=m1.dtype)  # (B, N)

        # インデックス復元
        # m0: (B, M') -> m0_: (B, M)
        m0_[:, ind0] = torch.where(
            m0 == -1,
            torch.tensor(-1, device=m0.device),
            ind1.gather(1, m0.clamp(min=0))
        )
        m1_[:, ind1] = torch.where(
            m1 == -1,
            torch.tensor(-1, device=m1.device),
            ind0.gather(1, m1.clamp(min=0))
        )

        # スコア復元
        mscores0_ = torch.zeros((B, M), device=mscores0.device)  # (B, M)
        mscores1_ = torch.zeros((B, N), device=mscores1.device)  # (B, N)
        mscores0_[:, ind0] = mscores0
        mscores1_[:, ind1] = mscores1

        return m0_, m1_, mscores0_, mscores1_


# ============================================================
# Match Filtering
# ============================================================

def filter_matches(
    scores: torch.Tensor,
    threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Log assignment matrixからマッチを抽出

    ========================================
    Shape
    ========================================
    入力:
        scores: (B, M+1, N+1)
            - log assignment matrix
            - 最後の行・列はdustbin (unmatchable)
        threshold: float
            - マッチ閾値 (default: 0.1)

    出力:
        m0: (B, M)
            - 各点のマッチ先 (-1 = unmatched)
        m1: (B, N)
            - 各点のマッチ先 (-1 = unmatched)
        mscores0: (B, M)
            - マッチスコア [0, 1]
        mscores1: (B, N)
            - マッチスコア [0, 1]

    ========================================
    処理
    ========================================
    1. 各行・列の最大値を取得
    2. Mutual nearest neighbor check
    3. 閾値でフィルタリング
    """
    B = scores.shape[0]
    M = scores.shape[1] - 1  # dustbin除く
    N = scores.shape[2] - 1  # dustbin除く

    # 最後の行・列はdustbin（unmatchable）
    # valid_scores: (B, M, N)
    valid_scores = scores[:, :-1, :-1]

    # 各行の最大値とインデックス
    # max0: (B, M), argmax0: (B, M)
    max0, argmax0 = valid_scores.max(dim=2)

    # 各列の最大値とインデックス
    # max1: (B, N), argmax1: (B, N)
    max1, argmax1 = valid_scores.max(dim=1)

    # Mutual nearest neighbor check
    # indices0: (1, M) -> (B, M) by broadcast
    indices0 = torch.arange(M, device=argmax0.device).unsqueeze(0)
    # indices1: (1, N) -> (B, N) by broadcast
    indices1 = torch.arange(N, device=argmax1.device).unsqueeze(0)

    # argmax1.gather(1, argmax0): 各点のマッチ先から逆引き
    # mutual0: (B, M) bool - mutual nearest neighbor?
    mutual0 = indices0 == argmax1.gather(1, argmax0)
    # mutual1: (B, N) bool
    mutual1 = indices1 == argmax0.gather(1, argmax1)

    # スコア計算（exp で確率に変換）
    # max0_exp: (B, M)
    max0_exp = max0.exp()

    # mscores0: (B, M) - mutualな点のスコア、それ以外は0
    mscores0 = torch.where(mutual0, max0_exp, torch.zeros_like(max0_exp))

    # mscores1: (B, N) - mutualな点のスコア
    mscores1 = torch.where(
        mutual1,
        mscores0.gather(1, argmax1),
        torch.zeros_like(max1)
    )

    # 閾値フィルタリング
    # valid0: (B, M) bool
    valid0 = mutual0 & (mscores0 > threshold)
    # valid1: (B, N) bool
    valid1 = mutual1 & valid0.gather(1, argmax1)

    # 無効なマッチは-1
    # m0: (B, M)
    m0 = torch.where(valid0, argmax0, torch.full_like(argmax0, -1))
    # m1: (B, N)
    m1 = torch.where(valid1, argmax1, torch.full_like(argmax1, -1))

    return m0, m1, mscores0, mscores1


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    LightGlueの使用例

    ========================================
    Shape Summary
    ========================================
    Input:
        image0.keypoints: (B, M, 2) = (2, 512, 2)
        image0.descriptors: (B, M, D) = (2, 512, 256)
        image1.keypoints: (B, N, 2) = (2, 480, 2)
        image1.descriptors: (B, N, D) = (2, 480, 256)

    Output:
        matches0: (B, M) = (2, 512)
        matches1: (B, N) = (2, 480)
        matching_scores0: (B, M) = (2, 512)
        matching_scores1: (B, N) = (2, 480)
        matches: List[(S_b, 2)] - 各バッチのマッチペア
        scores: List[(S_b,)] - マッチスコア
    """
    print("=== LightGlue Example Usage ===\n")

    # モデル初期化
    model = LightGlue(features='superpoint')
    model.eval()

    # ダミーデータ
    B = 2      # バッチサイズ
    M = 512    # Image 0 のキーポイント数
    N = 480    # Image 1 のキーポイント数
    D = 256    # SuperPointの記述子次元

    data = {
        'image0': {
            'keypoints': torch.rand(B, M, 2) * 640,     # (B, M, 2) ピクセル座標
            'descriptors': torch.randn(B, M, D),        # (B, M, D) 記述子
            'image_size': torch.tensor([640, 480])      # (2,) [W, H]
        },
        'image1': {
            'keypoints': torch.rand(B, N, 2) * 640,     # (B, N, 2)
            'descriptors': torch.randn(B, N, D),        # (B, N, D)
            'image_size': torch.tensor([640, 480])      # (2,)
        }
    }

    print(f"Input shapes:")
    print(f"  image0.keypoints: {data['image0']['keypoints'].shape}")      # (2, 512, 2)
    print(f"  image0.descriptors: {data['image0']['descriptors'].shape}")  # (2, 512, 256)
    print(f"  image1.keypoints: {data['image1']['keypoints'].shape}")      # (2, 480, 2)
    print(f"  image1.descriptors: {data['image1']['descriptors'].shape}")  # (2, 480, 256)
    print()

    # 推論
    with torch.no_grad():
        output = model(data)

    print(f"Output shapes:")
    print(f"  matches0: {output['matches0'].shape}")                 # (2, 512)
    print(f"  matches1: {output['matches1'].shape}")                 # (2, 480)
    print(f"  matching_scores0: {output['matching_scores0'].shape}") # (2, 512)
    print(f"  matching_scores1: {output['matching_scores1'].shape}") # (2, 480)
    print(f"  stop: {output['stop']}")                               # int
    print(f"  prune0: {output['prune0'].shape}")                     # (2, 512)
    print(f"  prune1: {output['prune1'].shape}")                     # (2, 480)
    print()

    # マッチの取り出し
    for b in range(B):
        matches_b = output['matches'][b]  # (S_b, 2)
        scores_b = output['scores'][b]    # (S_b,)
        print(f"Batch {b}:")
        print(f"  matches shape: {matches_b.shape}")  # (S_b, 2)
        print(f"  scores shape: {scores_b.shape}")    # (S_b,)
        if len(scores_b) > 0:
            print(f"  avg score: {scores_b.mean():.3f}")


# ============================================================
# ダミークラス（実装は他ファイル参照）
# ============================================================

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding

    詳細は transformer_blocks.py 参照

    ========================================
    Shape
    ========================================
    入力:
        x: (B, N, pos_dim)
            - pos_dim: 2 (x, y) or 4 (x, y, scale, ori)

    出力:
        emb: (2, B, N, head_dim)
            - 2: [cos, sin]
            - head_dim: 64
    """
    def __init__(self, M: int, dim: int, F_dim: int = None):
        """
        Args:
            M: pos_dim - 入力次元 (2 or 4)
            dim: head_dim - 出力次元 (64)
            F_dim: Fourier次元 (64)
        """
        super().__init__()
        F_dim = F_dim or dim
        # Wr: (pos_dim, F_dim // 2) = (2, 32)
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, pos_dim)

        Returns:
            emb: (2, B, N, head_dim)
        """
        # projected: (B, N, F_dim // 2) = (B, N, 32)
        projected = self.Wr(x)

        # cosines, sines: (B, N, 32)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)

        # emb: (2, B, N, 32)
        emb = torch.stack([cosines, sines], dim=0)

        # emb: (2, 1, B, N, 32) -> (2, B, N, 32)
        emb = emb.unsqueeze(-3)

        # repeat_interleave: (2, B, N, 32) -> (2, B, N, 64)
        emb = emb.repeat_interleave(2, dim=-1)

        return emb  # (2, B, N, head_dim)


class TransformerLayer(nn.Module):
    """
    Transformer Layer - 詳細は transformer_blocks.py 参照

    ========================================
    Shape
    ========================================
    入力:
        desc0: (B, M, C) = (B, M, 256)
        desc1: (B, N, C) = (B, N, 256)
        encoding0: (2, B, M, head_dim) = (2, B, M, 64)
        encoding1: (2, B, N, head_dim) = (2, B, N, 64)

    出力:
        desc0: (B, M, C) = (B, M, 256)
        desc1: (B, N, C) = (B, N, 256)
    """
    def __init__(self, dim: int, num_heads: int, flash: bool = False):
        super().__init__()
        self.self_attn = nn.Identity()  # placeholder
        self.cross_attn = nn.Identity()  # placeholder

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return desc0, desc1  # placeholder


class MatchAssignment(nn.Module):
    """
    Matching Head - 詳細は matching_head.py 参照

    ========================================
    Shape
    ========================================
    入力:
        desc0: (B, M, C) = (B, M, 256)
        desc1: (B, N, C) = (B, N, 256)

    出力:
        scores: (B, M+1, N+1) - log assignment matrix
        sim: (B, M, N) - similarity matrix (optional)

    get_matchability:
        入力: desc: (B, M, C)
        出力: matchability: (B, M) in [0, 1]
    """
    def __init__(self, dim: int):
        super().__init__()
        # matchability: (C,) -> (1,)
        self.matchability = nn.Linear(dim, 1)
        # final_proj: (C,) -> (C,)
        self.final_proj = nn.Linear(dim, dim)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            desc0: (B, M, C)
            desc1: (B, N, C)

        Returns:
            scores: (B, M+1, N+1)
            sim: None (placeholder)
        """
        B, M, C = desc0.shape
        _, N, _ = desc1.shape
        return torch.zeros(B, M + 1, N + 1, device=desc0.device), None

    def get_matchability(self, desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            desc: (B, M, C)

        Returns:
            matchability: (B, M) in [0, 1]
        """
        # self.matchability(desc): (B, M, 1)
        # squeeze(-1): (B, M)
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


class TokenConfidence(nn.Module):
    """
    Confidence Classifier - 詳細は adaptive_inference.py 参照

    ========================================
    Shape
    ========================================
    入力:
        desc0: (B, M, C) = (B, M, 256)
        desc1: (B, N, C) = (B, N, 256)

    出力:
        conf0: (B, M) - 確信度 [0, 1]
        conf1: (B, N) - 確信度 [0, 1]
    """
    def __init__(self, dim: int):
        super().__init__()
        # token: (C,) -> (1,) -> sigmoid
        self.token = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            desc0: (B, M, C)
            desc1: (B, N, C)

        Returns:
            conf0: (B, M)
            conf1: (B, N)
        """
        # self.token(desc0.detach()): (B, M, 1)
        # squeeze(-1): (B, M)
        conf0 = self.token(desc0.detach()).squeeze(-1)
        conf1 = self.token(desc1.detach()).squeeze(-1)
        return conf0, conf1


if __name__ == "__main__":
    example_usage()
