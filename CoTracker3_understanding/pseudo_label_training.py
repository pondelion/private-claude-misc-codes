"""
CoTracker3 - Pseudo-Label訓練パイプライン
論文: https://arxiv.org/abs/2410.11831

【このファイルの概要】
CoTracker3の最大の技術的貢献: Pseudo-Labelingによる半教師あり学習。
合成データのみで訓練された複数の教師モデルから、
ラベルなし実動画のPseudo-Labelを生成し、それで生徒モデルをfine-tuneする。

【公式コードの対応箇所】
- 論文 Section 4: Semi-Supervised Training
- 論文 Algorithm 1: Training with Pseudo-Labels

========================================
Pseudo-Label訓練の全体像
========================================

Phase 1: 教師モデルの訓練 (合成データ)
  ┌─────────────────────────────────────┐
  │ KubricMOVi-f (合成データ, GT付き)   │
  │ → CoTracker3 Online (教師1)        │
  │ → CoTracker3 Offline (教師2)       │
  │ → CoTracker2 (教師3)               │
  │ → TAPIR (教師4)                    │
  └─────────────────────────────────────┘

Phase 2: Pseudo-Label生成 (ラベルなし実動画)
  ┌─────────────────────────────────────┐
  │ 実動画 (ラベルなし)                  │
  │ + SIFT検出器でクエリ点サンプリング    │
  │                                     │
  │ → 4教師モデルで推論                  │
  │ → 信頼度ベースの集約                 │
  │ → Pseudo-Label (座標 + 可視性)      │
  └─────────────────────────────────────┘

Phase 3: 生徒モデルのfine-tune (Pseudo-Label + 合成データ)
  ┌─────────────────────────────────────┐
  │ バッチ = 50% Pseudo-Label           │
  │        + 50% 合成データ             │
  │                                     │
  │ → L_coord (Huber) のみ更新          │
  │ → vis_conf_head はフリーズ          │
  └─────────────────────────────────────┘

========================================
なぜPseudo-Labelが効果的か
========================================

問題:
- 合成データ (Kubric) と実動画の間のドメインギャップ
- 実動画の密なPoint Trackingアノテーションは非常に高コスト
- TAP-Vidなどの評価セットは点数が少なく訓練には不十分

解決策:
- 合成データで訓練した教師モデルは、ある程度の実動画追跡能力を持つ
- 複数の教師モデルの「合意」は、単一モデルより信頼性が高い
- SIFTベースのサンプリングで追跡しやすい点を選択
  → Pseudo-Labelの品質向上

結果 (論文 Table 1-3):
- TAP-Vid-DAVIS AJ: 62.4 → 66.4 (+4.0)
- TAP-Vid-Kinetics AJ: 60.3 → 65.3 (+5.0)
- Dynamic Replica: 大幅改善
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# ========================================
# SIFTベースのクエリ点サンプリング
# ========================================
class SIFTQuerySampler:
    """
    SIFTキーポイントによるクエリ点サンプリング

    ========================================
    概要 (論文 Section 4.1)
    ========================================
    ランダムサンプリングの代わりに、SIFTキーポイントを使用して
    「追跡しやすい」点をサンプリングする。

    利点:
    - テクスチャのある領域を優先的にサンプリング
    - 平坦な領域 (追跡困難) を回避
    - Pseudo-Labelの品質が向上

    ========================================
    処理フロー
    ========================================
    1. 動画のランダムフレームをSIFT検出器に入力
    2. キーポイント検出 (通常数百〜数千点)
    3. N点をランダムに選択
    4. queries = [frame_idx, x, y] を構築

    ========================================
    実装ノート
    ========================================
    - OpenCV の SIFT 実装を使用
    - cv2.SIFT_create()
    - detectAndCompute() でキーポイント検出
    - 検出数 < N の場合はランダム点で補完
    """

    def __init__(self, n_points: int = 2048):
        self.n_points = n_points
        # 実際の実装では cv2.SIFT_create() を使用
        # import cv2
        # self.sift = cv2.SIFT_create()

    def sample_queries(
        self,
        video: torch.Tensor,  # (B, T, C, H, W)
    ) -> torch.Tensor:
        """
        SIFTキーポイントからクエリ点をサンプリング

        ========================================
        Shape
        ========================================
        入力:  video: (B, T, 3, H, W)
        出力:  queries: (B, N, 3) = [frame_idx, x, y]

        ========================================
        処理
        ========================================
        1. ランダムフレーム選択
        2. SIFT キーポイント検出
        3. N点サンプリング (足りなければランダム補完)
        4. queries構築
        """
        B, T, C, H, W = video.shape
        device = video.device
        queries = torch.zeros(B, self.n_points, 3, device=device)

        for b in range(B):
            # ランダムフレーム選択
            frame_idx = torch.randint(0, T, (1,)).item()

            # SIFTキーポイント検出 (擬似実装)
            # 実際: frame = video[b, frame_idx].permute(1,2,0).cpu().numpy()
            # kps = sift.detect(frame_uint8, None)
            # 擬似: ランダムに追跡しやすい点を生成
            n_sift = min(self.n_points * 2, 5000)  # 通常は十分な数が検出される
            sift_x = torch.randint(0, W, (n_sift,), device=device).float()
            sift_y = torch.randint(0, H, (n_sift,), device=device).float()

            # N点をランダム選択
            indices = torch.randperm(n_sift, device=device)[:self.n_points]
            queries[b, :, 0] = frame_idx
            queries[b, :, 1] = sift_x[indices]
            queries[b, :, 2] = sift_y[indices]

        return queries


# ========================================
# 教師モデルアンサンブル
# ========================================
class TeacherEnsemble:
    """
    複数の教師モデルからPseudo-Labelを集約

    ========================================
    教師モデルの構成 (論文 Section 4.2)
    ========================================
    1. CoTracker3 Online  (合成データで訓練)
    2. CoTracker3 Offline (合成データで訓練)
    3. CoTracker2         (合成データで訓練)
    4. TAPIR              (合成データで訓練)

    ========================================
    集約戦略
    ========================================
    各教師モデルの予測:
      tracks_i:     (B, T, N, 2)  座標
      visibility_i: (B, T, N)     可視性
      confidence_i: (B, T, N)     信頼度

    信頼度ベースの集約:
      1. 各教師の信頼度で重み付け
      2. 最も信頼度の高い教師の予測を採用
      3. 可視性: 全教師の可視性のAND (厳密モード)
         または信頼度加重平均

    ========================================
    Pseudo-Label の品質保証
    ========================================
    - 複数モデルの合意 → ノイズ削減
    - 異なるアーキテクチャ (CoTracker vs TAPIR) → 多様性
    - 信頼度フィルタリング → 低品質ラベルを除外
    """

    def __init__(self):
        self.teachers = {}
        # 実際の実装ではここで各教師モデルをロード
        # self.teachers['cotracker3_online'] = load_model(...)
        # self.teachers['cotracker3_offline'] = load_model(...)
        # self.teachers['cotracker2'] = load_model(...)
        # self.teachers['tapir'] = load_model(...)

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        video: torch.Tensor,      # (B, T, 3, H, W) [0-255]
        queries: torch.Tensor,     # (B, N, 3)
    ) -> Dict[str, torch.Tensor]:
        """
        全教師モデルで推論 → Pseudo-Label集約

        ========================================
        出力
        ========================================
        {
          'tracks':     (B, T, N, 2)  Pseudo-Label座標
          'visibility': (B, T, N)     Pseudo-Label可視性
          'valid':      (B, T, N)     有効マスク (信頼度 > しきい値)
        }

        ========================================
        処理フロー
        ========================================
        1. 各教師モデルで推論 (勾配なし)
        2. 信頼度ベースで最良の教師を選択 (点×フレームごと)
        3. 低信頼度の点をマスク (valid=0)
        """
        B, T, C, H, W = video.shape
        N = queries.shape[1]
        device = video.device

        # 各教師の予測を収集
        all_tracks = []
        all_vis = []
        all_conf = []

        for name, model in self.teachers.items():
            tracks, vis, conf, _ = model(video, queries, iters=4, is_train=False)
            all_tracks.append(tracks)    # (B, T, N, 2)
            all_vis.append(vis)          # (B, T, N)
            all_conf.append(conf)        # (B, T, N)

        if len(all_tracks) == 0:
            # 擬似実装: 教師モデルがない場合のダミー出力
            return {
                'tracks': torch.randn(B, T, N, 2, device=device) * 100,
                'visibility': (torch.rand(B, T, N, device=device) > 0.2).float(),
                'valid': torch.ones(B, T, N, device=device),
            }

        # 教師予測をスタック
        all_tracks = torch.stack(all_tracks, dim=0)   # (K, B, T, N, 2)
        all_vis = torch.stack(all_vis, dim=0)         # (K, B, T, N)
        all_conf = torch.stack(all_conf, dim=0)       # (K, B, T, N)

        # === 信頼度ベースの選択 ===
        # 各点×各フレームで最も信頼度の高い教師の予測を採用
        best_teacher = all_conf.argmax(dim=0)  # (B, T, N)

        # gather で最良教師の予測を取得
        B_idx = torch.arange(B, device=device)[:, None, None].expand_as(best_teacher)
        T_idx = torch.arange(T, device=device)[None, :, None].expand_as(best_teacher)
        N_idx = torch.arange(N, device=device)[None, None, :].expand_as(best_teacher)

        pseudo_tracks = all_tracks[best_teacher, B_idx, T_idx, N_idx]     # (B, T, N, 2)
        pseudo_vis = all_vis[best_teacher, B_idx, T_idx, N_idx]           # (B, T, N)
        best_conf = all_conf[best_teacher, B_idx, T_idx, N_idx]           # (B, T, N)

        # === 低信頼度フィルタリング ===
        conf_threshold = 0.5
        valid = (best_conf > conf_threshold).float()

        return {
            'tracks': pseudo_tracks,
            'visibility': pseudo_vis,
            'valid': valid,
        }


# ========================================
# Pseudo-Label 訓練ループ
# ========================================
class PseudoLabelTrainer:
    """
    Pseudo-Label訓練のメインクラス

    ========================================
    訓練設定 (論文 Section 4.3)
    ========================================
    - バッチ構成: 50% Pseudo-Label (実動画) + 50% 合成データ (Kubric)
    - 学習率: 5e-5 (合成データのみの訓練より低い)
    - 反復数: 50,000 steps
    - vis_conf_head: フリーズ (合成データの精度を維持)
    - flow_head + updateformer + fnet: 更新

    ========================================
    vis_conf_head フリーズの理由
    ========================================
    - Pseudo-Labelには可視性・信頼度のGTが含まれない
      (教師モデルの予測は正確ではない)
    - フリーズしないと、可視性・信頼度予測が劣化
    - linear_layer_for_vis_conf=True で分離してあるため、
      flow_head だけ学習可能

    ========================================
    合成データ混合の理由
    ========================================
    - 100% Pseudo-Labelだと過学習のリスク
    - 合成データは完全なGT付き → アンカーとして機能
    - 50%:50% が最適 (論文のアブレーション)

    ========================================
    データセット
    ========================================
    合成:
    - KubricMOVi-f: 24フレーム, 256×256, 物理シミュレーション

    実動画:
    - LibriSV (自然動画コレクション)
    - TAP-Vid-Kubricの実動画部分
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-5,
        mixed_ratio: float = 0.5,  # Pseudo-Labelの割合
    ):
        self.model = model
        self.mixed_ratio = mixed_ratio

        # === vis_conf_head をフリーズ ===
        self._freeze_vis_conf_head()

        # === オプティマイザ (フリーズされていないパラメータのみ) ===
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # 教師アンサンブル
        self.teacher_ensemble = TeacherEnsemble()

        # クエリサンプラー
        self.query_sampler = SIFTQuerySampler(n_points=2048)

    def _freeze_vis_conf_head(self):
        """
        vis_conf_head のパラメータをフリーズ

        ========================================
        フリーズ対象
        ========================================
        model.updateformer.vis_conf_head.weight
        model.updateformer.vis_conf_head.bias

        linear_layer_for_vis_conf=True の場合:
        - flow_head: delta_x, delta_y → 学習可能
        - vis_conf_head: delta_vis, delta_conf → フリーズ

        ========================================
        実装
        ========================================
        for name, param in model.named_parameters():
            if 'vis_conf_head' in name:
                param.requires_grad = False
        """
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if 'vis_conf_head' in name:
                param.requires_grad = False
                frozen_count += param.numel()
        print(f"[PseudoLabelTrainer] Froze vis_conf_head: {frozen_count} params")

    def train_step(
        self,
        real_video: torch.Tensor,       # (B/2, T, 3, H, W) ラベルなし実動画
        synthetic_video: torch.Tensor,   # (B/2, T, 3, H, W) 合成動画
        synthetic_gt_tracks: torch.Tensor,     # (B/2, T, N, 2) GT座標
        synthetic_gt_vis: torch.Tensor,        # (B/2, T, N) GT可視性
        synthetic_gt_valid: torch.Tensor,      # (B/2, T, N) 有効マスク
    ) -> Dict[str, float]:
        """
        1ステップの訓練

        ========================================
        処理フロー
        ========================================

        1. Pseudo-Label生成 (実動画)
           - SIFTでクエリ点サンプリング
           - 教師アンサンブルで推論
           - 信頼度ベースで集約

        2. 合成データの準備
           - 既存のGTを使用

        3. バッチ結合
           - [pseudo_label_batch, synthetic_batch]

        4. モデル推論 (is_train=True)
           - 全反復の予測を取得

        5. 損失計算
           - Pseudo-Label: L_coord のみ (vis/conf損失なし)
           - 合成データ: L_coord + L_vis + L_conf

        6. 逆伝播 + パラメータ更新

        ========================================
        損失関数の詳細
        ========================================

        Pseudo-Label損失:
          L_pseudo = sequence_loss(pred, pseudo_gt, pseudo_valid,
                                    add_huber_loss=True, gamma=0.8)
          ※ vis/conf損失は計算しない (vis_conf_headがフリーズのため不要)

        合成データ損失:
          L_synthetic = (
              sequence_loss(pred, gt, valid, add_huber_loss=True)
              + λ_vis * sequence_BCE_loss(vis_pred, vis_gt)
              + λ_conf * sequence_prob_loss(pred, conf, gt, vis_gt)
          )

        全体損失:
          L_total = L_pseudo + L_synthetic
        """
        self.model.train()

        # === 1. Pseudo-Label生成 (勾配なし) ===
        with torch.no_grad():
            pseudo_queries = self.query_sampler.sample_queries(real_video)
            pseudo_labels = self.teacher_ensemble.generate_pseudo_labels(
                real_video, pseudo_queries
            )
        pseudo_tracks_gt = pseudo_labels['tracks']       # (B/2, T, N, 2)
        pseudo_vis_gt = pseudo_labels['visibility']       # (B/2, T, N)
        pseudo_valid = pseudo_labels['valid']             # (B/2, T, N)

        # === 2. 合成データのクエリ構築 ===
        B_syn = synthetic_video.shape[0]
        N = synthetic_gt_tracks.shape[2]
        synthetic_queries = torch.zeros(B_syn, N, 3, device=synthetic_video.device)
        synthetic_queries[:, :, 0] = 0  # フレーム0でクエリ
        synthetic_queries[:, :, 1] = synthetic_gt_tracks[:, 0, :, 0]  # x
        synthetic_queries[:, :, 2] = synthetic_gt_tracks[:, 0, :, 1]  # y

        # === 3. Pseudo-Label バッチの推論 ===
        pred_tracks_p, pred_vis_p, pred_conf_p, train_data_p = self.model(
            real_video, pseudo_queries, iters=4, is_train=True
        )

        # === 4. 合成データ バッチの推論 ===
        pred_tracks_s, pred_vis_s, pred_conf_s, train_data_s = self.model(
            synthetic_video, synthetic_queries, iters=4, is_train=True
        )

        # === 5. 損失計算 ===
        loss_dict = {}

        # Pseudo-Label損失: 座標のみ
        if train_data_p is not None:
            all_coords_p, all_vis_p, all_conf_p, valid_mask_p = train_data_p
            loss_pseudo_coord = self._sequence_loss(
                all_coords_p, [pseudo_tracks_gt], [pseudo_valid],
                gamma=0.8, add_huber_loss=True,
            )
            loss_dict['pseudo_coord'] = loss_pseudo_coord

        # 合成データ損失: 座標 + 可視性 + 信頼度
        if train_data_s is not None:
            all_coords_s, all_vis_s, all_conf_s, valid_mask_s = train_data_s
            loss_syn_coord = self._sequence_loss(
                all_coords_s, [synthetic_gt_tracks], [synthetic_gt_valid],
                gamma=0.8, add_huber_loss=True,
            )
            loss_syn_vis = self._sequence_BCE_loss(
                all_vis_s, [synthetic_gt_vis]
            )
            loss_syn_conf = self._sequence_prob_loss(
                all_coords_s, all_conf_s,
                [synthetic_gt_tracks], [synthetic_gt_vis],
            )
            loss_dict['syn_coord'] = loss_syn_coord
            loss_dict['syn_vis'] = loss_syn_vis
            loss_dict['syn_conf'] = loss_syn_conf.mean()

        # 全体損失
        total_loss = sum(loss_dict.values())

        # === 6. 逆伝播 + 更新 ===
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def _sequence_loss(self, preds, gts, valids, gamma=0.8, add_huber_loss=True):
        """座標損失 (簡略版)"""
        total = 0.0
        for j in range(len(gts)):
            M = len(preds[j])
            for i in range(M):
                w = gamma ** (M - i - 1)
                if add_huber_loss:
                    loss = self._huber(preds[j][i], gts[j], delta=6.0)
                else:
                    loss = (preds[j][i] - gts[j]).abs()
                loss = loss.mean(dim=-1)  # (B, S, N)
                loss = (loss * valids[j]).sum() / valids[j].sum().clamp(min=1)
                total += w * loss
            total /= M
        return total / max(len(gts), 1)

    def _huber(self, x, y, delta=6.0):
        diff = (x - y).abs()
        flag = (diff <= delta).float()
        return flag * 0.5 * (x - y) ** 2 + (1 - flag) * delta * (diff - 0.5 * delta)

    def _sequence_BCE_loss(self, vis_preds, vis_gts):
        """可視性BCE損失 (簡略版)"""
        total = 0.0
        for j in range(len(vis_gts)):
            M = len(vis_preds[j])
            for i in range(M):
                total += F.binary_cross_entropy(vis_preds[j][i], vis_gts[j])
            total /= M
        return total / max(len(vis_gts), 1)

    def _sequence_prob_loss(self, tracks, confidence, targets, visibility, thresh=12.0):
        """信頼度損失 (簡略版)"""
        total = 0.0
        for j in range(len(targets)):
            M = len(tracks[j])
            for i in range(M):
                err = ((tracks[j][i].detach() - targets[j]) ** 2).sum(dim=-1)
                valid = (err <= thresh ** 2).float()
                loss = F.binary_cross_entropy(confidence[j][i], valid, reduction="none")
                loss = (loss * visibility[j]).mean(dim=[1, 2])
                total += loss
            total /= M
        return total / max(len(targets), 1)


# ========================================
# 訓練設定の比較
# ========================================
def print_training_configs():
    """
    Phase 1 (合成のみ) vs Phase 2 (Pseudo-Label) の訓練設定比較

    ========================================
    Phase 1: 合成データのみの訓練
    ========================================
    データ:      KubricMOVi-f (256×256, 24フレーム)
    バッチサイズ: 32
    学習率:      5e-4
    スケジューラ: OneCycleLR
    反復数:      200,000 steps
    解像度:      384×512 (リサイズ)
    vis_conf_head: 学習あり
    損失:        L_coord + L_vis + L_conf

    ========================================
    Phase 2: Pseudo-Label訓練
    ========================================
    データ:      50% Pseudo-Label (実動画) + 50% KubricMOVi-f
    バッチサイズ: 8
    学習率:      5e-5 (Phase 1の1/10)
    スケジューラ: OneCycleLR
    反復数:      50,000 steps
    解像度:      384×512
    vis_conf_head: フリーズ
    損失:
      Pseudo-Label: L_coord のみ
      合成データ:   L_coord + L_vis + L_conf
    """
    print("=" * 70)
    print("CoTracker3 訓練設定比較")
    print("=" * 70)

    configs = {
        "Phase 1 (合成のみ)": {
            "データ": "KubricMOVi-f (256×256, 24fps)",
            "バッチサイズ": "32",
            "学習率": "5e-4",
            "反復数": "200,000 steps",
            "vis_conf_head": "学習あり",
            "クエリサンプリング": "ランダム",
            "損失": "L_coord + L_vis + L_conf",
        },
        "Phase 2 (Pseudo-Label)": {
            "データ": "50% Pseudo-Label + 50% Kubric",
            "バッチサイズ": "8",
            "学習率": "5e-5",
            "反復数": "50,000 steps",
            "vis_conf_head": "フリーズ",
            "クエリサンプリング": "SIFT",
            "損失": "L_coord (Pseudo) + L_coord+vis+conf (Kubric)",
        },
    }

    for phase, config in configs.items():
        print(f"\n{phase}")
        print("-" * 40)
        for key, value in config.items():
            print(f"  {key:20s}: {value}")


# ========================================
# アブレーション結果
# ========================================
def print_ablation_results():
    """
    論文のアブレーション結果 (Table 4-6)

    ========================================
    Table 4: 教師モデルの組み合わせ
    ========================================
    教師モデル                            AJ (DAVIS)
    -------------------------------------------
    なし (合成のみ)                        62.4
    CoTracker3 Online のみ                64.8
    CoTracker3 + CoTracker2               65.5
    CoTracker3 + CoTracker2 + TAPIR       66.4  ← 最良

    → 多様な教師モデルの組み合わせが効果的

    ========================================
    Table 5: Pseudo-Label vs 合成データ比率
    ========================================
    Pseudo-Label %    AJ (DAVIS)
    -------------------------------------------
    0% (合成のみ)     62.4
    25%               64.8
    50%               66.4  ← 最良
    75%               65.7
    100%              64.2

    → 50:50 が最適バランス

    ========================================
    Table 6: クエリサンプリング手法
    ========================================
    手法                AJ (DAVIS)
    -------------------------------------------
    ランダム            64.8
    SIFT                66.4  ← +1.6

    → SIFTで追跡しやすい点を選ぶことで品質向上
    """
    print("\n" + "=" * 70)
    print("CoTracker3 Pseudo-Label アブレーション結果")
    print("=" * 70)

    print("\n[教師モデルの組み合わせ (TAP-Vid-DAVIS AJ)]")
    teachers = [
        ("合成のみ (ベースライン)", 62.4),
        ("CoTracker3 Online", 64.8),
        ("CT3 Online + CT3 Offline", 65.2),
        ("CT3 + CT2", 65.5),
        ("CT3 + CT2 + TAPIR", 66.4),
    ]
    for name, aj in teachers:
        bar = "█" * int(aj - 60)
        print(f"  {name:40s} {aj:.1f}  {bar}")

    print("\n[Pseudo-Label比率 (TAP-Vid-DAVIS AJ)]")
    ratios = [
        ("0% (合成のみ)", 62.4),
        ("25%", 64.8),
        ("50%", 66.4),
        ("75%", 65.7),
        ("100% (PLのみ)", 64.2),
    ]
    for name, aj in ratios:
        bar = "█" * int(aj - 60)
        print(f"  {name:25s} {aj:.1f}  {bar}")

    print("\n[クエリサンプリング手法 (TAP-Vid-DAVIS AJ)]")
    samplings = [
        ("ランダム", 64.8),
        ("SIFT", 66.4),
    ]
    for name, aj in samplings:
        bar = "█" * int(aj - 60)
        print(f"  {name:15s} {aj:.1f}  {bar}")


# ========================================
# デモ
# ========================================
def demo_pseudo_label_training():
    """
    Pseudo-Label訓練パイプラインのデモ
    """
    print("=" * 70)
    print("CoTracker3 Pseudo-Label 訓練パイプライン デモ")
    print("=" * 70)

    # === SIFTサンプリング ===
    print("\n[1. SIFTクエリサンプリング]")
    sampler = SIFTQuerySampler(n_points=2048)
    dummy_video = torch.randint(0, 256, (2, 24, 3, 384, 512), dtype=torch.float32)
    queries = sampler.sample_queries(dummy_video)
    print(f"  入力動画:   {dummy_video.shape}")
    print(f"  クエリ点:   {queries.shape}  [frame_idx, x, y]")
    print(f"  サンプル:   frame={queries[0, 0, 0].item():.0f}, "
          f"x={queries[0, 0, 1].item():.1f}, y={queries[0, 0, 2].item():.1f}")

    # === 教師アンサンブル ===
    print("\n[2. 教師アンサンブルによるPseudo-Label生成]")
    ensemble = TeacherEnsemble()
    pseudo_labels = ensemble.generate_pseudo_labels(dummy_video, queries)
    print(f"  Pseudo tracks:     {pseudo_labels['tracks'].shape}")
    print(f"  Pseudo visibility: {pseudo_labels['visibility'].shape}")
    print(f"  Pseudo valid:      {pseudo_labels['valid'].shape}")
    print(f"  有効率: {pseudo_labels['valid'].mean().item():.2%}")

    # === 訓練設定 ===
    print_training_configs()

    # === アブレーション ===
    print_ablation_results()

    # === 全体パイプラインのまとめ ===
    print("\n" + "=" * 70)
    print("Pseudo-Label訓練の全体フロー")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 1: 合成データのみで訓練                              │
    │   KubricMOVi-f → CoTracker3 (教師モデル候補)              │
    │   + CoTracker2, TAPIR (別途訓練済み)                      │
    └─────────────────────────┬───────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 2: Pseudo-Label生成                                │
    │   実動画 → SIFTサンプリング → 4教師で推論                  │
    │   → 信頼度ベース集約 → Pseudo-Label                      │
    └─────────────────────────┬───────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 3: 生徒モデルfine-tune                             │
    │   バッチ = 50% Pseudo-Label + 50% 合成データ              │
    │   vis_conf_head フリーズ                                  │
    │   L_coord (Huber) で座標予測を改善                         │
    │   → ドメインギャップを縮小 → 実動画での精度向上             │
    └─────────────────────────────────────────────────────────┘

    結果: TAP-Vid-DAVIS AJ: 62.4 → 66.4 (+4.0 ポイント改善)
    """)


if __name__ == "__main__":
    demo_pseudo_label_training()
