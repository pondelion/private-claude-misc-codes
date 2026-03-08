"""
SAM3 Loss Computation - 簡略化疑似コード
======================================

学習時のロス計算とマッチング戦略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class Sam3LossWrapper(nn.Module):
    """
    SAM3 メインロスラッパー

    処理フロー:
    1. ハンガリアンマッチング (予測とGTの対応付け)
    2. 各ロス関数の適用 (分類、ボックス、マスク)
    3. One-to-Many (O2M) ロスの追加
    4. Deep Supervision (各Decoderレイヤーでロス計算)
    """

    def __init__(
        self,
        loss_fns: List[nn.Module],          # ロス関数のリスト
        matcher: nn.Module,                  # O2O (One-to-One) マッチャー
        o2m_matcher: Optional[nn.Module] = None,  # O2M (One-to-Many) マッチャー
        o2m_weight: float = 1.0,             # O2Mロスの重み
        normalization: str = "global"        # "global", "local", "none"
    ):
        super().__init__()

        self.loss_fns = nn.ModuleList(loss_fns)
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        self.normalization = normalization


    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        ロス計算のメインフォワードパス

        入力:
            outputs: モデル出力 Dict {
                'pred_logits': (B, Q, 1) - 分類スコア
                    B: バッチサイズ
                    Q: クエリ数 (200)
                    1: バイナリ分類

                'pred_boxes': (B, Q, 4) - ボックス予測 (cx, cy, w, h) 正規化
                'pred_boxes_xyxy': (B, Q, 4) - ボックス予測 (x1, y1, x2, y2)
                'pred_masks': (B, Q, H, W) - マスク予測 (optional)

                'aux_outputs': List[Dict] - 中間レイヤー出力 (Deep Supervision用)
                'pred_logits_o2m': (B, Q', 1) - O2Mクエリ出力 (optional)
            }

            targets: Ground Truth Dict {
                'boxes': (N_total, 4) - 全GTボックス (バッチ全体をflatten)
                    N_total: バッチ内の全GTボックス数
                    4: (cx, cy, w, h) 正規化座標

                'boxes_xyxy': (N_total, 4) - GTボックス (x1, y1, x2, y2)
                'masks': (N_total, H, W) - GTマスク (optional)
                'num_boxes': (B,) - 各画像のGT数
                'is_exhaustive': (B,) - 完全アノテーション済みフラグ
            }

        出力:
            losses: Dict {
                'core_loss': Tensor - バックプロパゲーション用の総ロス
                'loss_ce': Tensor - 分類ロス
                'loss_bbox': Tensor - ボックスL1ロス
                'loss_giou': Tensor - GIoUロス
                'loss_mask': Tensor - マスクFocalロス
                'loss_dice': Tensor - Diceロス
                ... (aux_outputs用に _aux_0, _aux_1 等のサフィックス付き)
            }
        """

        # ========================================
        # Step 1: 正規化項の計算
        # ========================================

        num_boxes = self._get_num_boxes(targets)
        # num_boxes: スカラー - 正規化に使用するボックス数


        # ========================================
        # Step 2: メイン出力のロス計算 (O2O)
        # ========================================

        # ハンガリアンマッチングで予測とGTの対応付け
        indices = self.matcher(outputs, targets)
        # indices: List[Tuple(src_idx, tgt_idx)] 長さB
        #   src_idx: (M,) - マッチした予測のインデックス
        #   tgt_idx: (M,) - マッチしたGTのインデックス

        # 各ロス関数を適用
        losses = {}
        total_loss = 0.0

        for loss_fn in self.loss_fns:
            loss_dict = loss_fn(outputs, targets, indices, num_boxes)
            # loss_dict: Dict {loss_name: Tensor}

            losses.update(loss_dict)
            for loss_name, loss_value in loss_dict.items():
                total_loss += loss_value


        # ========================================
        # Step 3: O2M (One-to-Many) ロスの追加
        # ========================================

        if self.o2m_matcher is not None and 'pred_logits_o2m' in outputs:
            # O2Mマッチング (複数予測を1つのGTにマッチ可能)
            indices_o2m = self.o2m_matcher(outputs, targets)

            # O2M出力を使用してロス計算
            outputs_o2m = {
                'pred_logits': outputs['pred_logits_o2m'],
                'pred_boxes': outputs.get('pred_boxes_o2m', outputs['pred_boxes']),
                'pred_boxes_xyxy': outputs.get('pred_boxes_xyxy_o2m', outputs['pred_boxes_xyxy']),
            }

            for loss_fn in self.loss_fns:
                loss_dict_o2m = loss_fn(outputs_o2m, targets, indices_o2m, num_boxes)

                # O2Mロスを重み付けして追加
                for loss_name, loss_value in loss_dict_o2m.items():
                    o2m_loss_name = loss_name + '_o2m'
                    losses[o2m_loss_name] = loss_value
                    total_loss += loss_value * self.o2m_weight


        # ========================================
        # Step 4: Deep Supervision (補助出力のロス)
        # ========================================

        if 'aux_outputs' in outputs:
            for aux_idx, aux_outputs in enumerate(outputs['aux_outputs']):
                # 各中間レイヤーでマッチングとロス計算
                indices_aux = self.matcher(aux_outputs, targets)

                for loss_fn in self.loss_fns:
                    loss_dict_aux = loss_fn(aux_outputs, targets, indices_aux, num_boxes)

                    # サフィックスを追加
                    for loss_name, loss_value in loss_dict_aux.items():
                        aux_loss_name = f"{loss_name}_aux_{aux_idx}"
                        losses[aux_loss_name] = loss_value
                        total_loss += loss_value


        # ========================================
        # Step 5: 総ロスの返却
        # ========================================

        losses['core_loss'] = total_loss

        return losses


    def _get_num_boxes(self, targets: Dict) -> torch.Tensor:
        """
        正規化用のボックス数を計算

        入力:
            targets: GT辞書

        出力:
            num_boxes: スカラー Tensor
        """

        num_boxes = targets['num_boxes'].sum()  # バッチ内の総GT数

        if self.normalization == "global":
            # 分散学習時は全GPUで平均
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            num_boxes = num_boxes / world_size
        elif self.normalization == "local":
            # GPU毎に正規化
            pass
        elif self.normalization == "none":
            num_boxes = torch.tensor(1.0)

        # 最小値1にクリップ (0除算回避)
        num_boxes = torch.clamp(num_boxes, min=1.0)

        return num_boxes


# ============================================
# マッチング戦略
# ============================================

class BinaryHungarianMatcher(nn.Module):
    """
    ハンガリアンアルゴリズムによる二部マッチング

    予測とGround Truthの最適な1対1対応を見つける
    """

    def __init__(
        self,
        cost_class: float = 2.0,   # 分類コストの重み
        cost_bbox: float = 5.0,    # ボックスL1コストの重み
        cost_giou: float = 2.0,    # GIoUコストの重み
        alpha: float = 0.25,       # Focal Loss alpha
        gamma: float = 2.0,        # Focal Loss gamma
        focal: bool = True         # Focal Lossを使用
    ):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.focal = focal


    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ハンガリアンマッチングの実行

        入力:
            outputs: 予測 Dict
            targets: GT Dict

        出力:
            indices: List[Tuple(src_idx, tgt_idx)] 長さB
                各画像について:
                    src_idx: (M,) - マッチした予測のインデックス
                    tgt_idx: (M,) - マッチしたGTのインデックス
        """

        B, Q = outputs['pred_logits'].shape[:2]  # バッチサイズ、クエリ数

        # 予測を平坦化
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()  # (B*Q, 1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (B*Q, 4)
        out_bbox_xyxy = outputs['pred_boxes_xyxy'].flatten(0, 1)  # (B*Q, 4)

        # GTを取得
        tgt_bbox = targets['boxes']  # (N_total, 4)
        tgt_bbox_xyxy = targets['boxes_xyxy']  # (N_total, 4)

        # ========================================
        # コスト行列の計算
        # ========================================

        # 1. 分類コスト (Focal Loss風)
        if self.focal:
            # Negative cost: (1 - alpha) * (p^gamma) * log(p)
            # Positive cost: alpha * ((1-p)^gamma) * log(1-p)
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * \
                             (-out_prob.log() + 1e-8)

            # GTラベルは全て1 (オブジェクト存在)
            cost_class = pos_cost_class  # (B*Q, 1)
        else:
            # BCE cost
            cost_class = -out_prob  # (B*Q, 1)

        cost_class = cost_class.squeeze(1)  # (B*Q,)

        # 2. ボックスL1コスト
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (B*Q, N_total)

        # 3. GIoUコスト
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)  # (B*Q, N_total)

        # 総コスト
        C = self.cost_class * cost_class[:, None] + \
            self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou
        # C: (B*Q, N_total) - コスト行列


        # ========================================
        # バッチごとにハンガリアンアルゴリズム適用
        # ========================================

        C = C.view(B, Q, -1).cpu()  # (B, Q, N_total)

        indices = []
        offset = 0  # GTのオフセット

        for i in range(B):
            num_gt = targets['num_boxes'][i].item()  # 画像iのGT数

            if num_gt == 0:
                # GTなし
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            # 画像iのコスト行列
            c = C[i, :, offset:offset+num_gt]  # (Q, num_gt)

            # ハンガリアンアルゴリズム (scipy使用)
            src_idx, tgt_idx = linear_sum_assignment(c.numpy())

            # Tensorに変換
            src_idx = torch.as_tensor(src_idx, dtype=torch.long)
            tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long)

            indices.append((src_idx, tgt_idx))

            offset += num_gt

        return indices


class BinaryOneToManyMatcher(nn.Module):
    """
    One-to-Many マッチャー

    1つのGTに複数の予測をマッチ可能 (補助出力用)
    """

    def __init__(
        self,
        alpha: float = 0.3,      # スコア閾値用のalpha
        threshold: float = 0.4,  # IoU閾値
        topk: int = 4            # 各GTに最大何個マッチするか
    ):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        貪欲マッチング (スコア × IoU でソート)

        1つのGTに対して、スコアとIoUが高い上位k個をマッチ
        """

        B, Q = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].sigmoid()  # (B, Q, 1)
        out_bbox = outputs['pred_boxes_xyxy']  # (B, Q, 4)

        indices = []
        offset = 0

        for i in range(B):
            num_gt = targets['num_boxes'][i].item()

            if num_gt == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long)
                ))
                continue

            tgt_bbox = targets['boxes_xyxy'][offset:offset+num_gt]  # (num_gt, 4)

            # IoU計算
            iou = box_iou(out_bbox[i], tgt_bbox)  # (Q, num_gt)

            # スコア計算
            score = out_prob[i].squeeze(-1)  # (Q,)

            # 各GTに対してtopk予測を選択
            src_indices = []
            tgt_indices = []

            for gt_idx in range(num_gt):
                # GT gt_idxとのスコア
                gt_iou = iou[:, gt_idx]  # (Q,)
                gt_score = score * gt_iou  # (Q,) - スコア × IoU

                # 閾値以上のみ
                valid_mask = gt_iou > self.threshold
                gt_score = gt_score * valid_mask.float()

                # Top-k選択
                topk_values, topk_indices = torch.topk(
                    gt_score,
                    k=min(self.topk, valid_mask.sum().item()),
                    largest=True
                )

                # 追加
                for idx in topk_indices:
                    if gt_score[idx] > 0:
                        src_indices.append(idx.item())
                        tgt_indices.append(gt_idx)

            indices.append((
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long)
            ))

            offset += num_gt

        return indices


# ============================================
# 個別ロス関数
# ============================================

class ClassificationLoss(nn.Module):
    """
    分類ロス (Focal Loss + BCE)

    バイナリ分類: オブジェクトが存在するか
    """

    def __init__(
        self,
        weight_dict: Dict[str, float],
        pos_weight: float = 10.0,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()

        self.weight_dict = weight_dict
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma


    def forward(
        self,
        outputs: Dict,
        targets: Dict,
        indices: List[Tuple],
        num_boxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分類ロスの計算

        入力:
            outputs: 予測
            targets: GT
            indices: マッチング結果
            num_boxes: 正規化項

        出力:
            Dict {'loss_ce': Tensor}
        """

        pred_logits = outputs['pred_logits']  # (B, Q, 1)
        B, Q, _ = pred_logits.shape

        # ========================================
        # ターゲットラベルの構築
        # ========================================

        # 初期化: 全て負例 (0)
        target_classes = torch.zeros_like(pred_logits)  # (B, Q, 1)

        # マッチした予測を正例 (1) に設定
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx, 0] = 1.0


        # ========================================
        # Focal Lossの計算
        # ========================================

        # Sigmoid確率
        pred_prob = pred_logits.sigmoid()  # (B, Q, 1)

        # Focal Loss
        # loss = -alpha * (1-p)^gamma * log(p)  (正例)
        #        -(1-alpha) * p^gamma * log(1-p)  (負例)

        # 正例のロス
        pos_mask = (target_classes == 1.0)
        pos_loss = -self.alpha * ((1 - pred_prob) ** self.gamma) * \
                   torch.log(pred_prob + 1e-8)
        pos_loss = (pos_loss * pos_mask).sum()

        # 負例のロス
        neg_mask = (target_classes == 0.0)
        neg_loss = -(1 - self.alpha) * (pred_prob ** self.gamma) * \
                   torch.log(1 - pred_prob + 1e-8)
        neg_loss = (neg_loss * neg_mask).sum()

        # 総ロス
        loss_ce = (pos_loss * self.pos_weight + neg_loss) / num_boxes

        # 重み適用
        loss_ce = loss_ce * self.weight_dict.get('loss_ce', 1.0)

        return {'loss_ce': loss_ce}


class BoxLoss(nn.Module):
    """
    ボックスロス (L1 + GIoU)
    """

    def __init__(self, weight_dict: Dict[str, float]):
        super().__init__()
        self.weight_dict = weight_dict


    def forward(self, outputs, targets, indices, num_boxes):
        """
        ボックスロスの計算

        出力:
            Dict {'loss_bbox': Tensor, 'loss_giou': Tensor}
        """

        # マッチした予測とGTを抽出
        src_boxes = []
        tgt_boxes = []
        src_boxes_xyxy = []
        tgt_boxes_xyxy = []

        offset = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            # 予測
            src_boxes.append(outputs['pred_boxes'][i, src_idx])
            src_boxes_xyxy.append(outputs['pred_boxes_xyxy'][i, src_idx])

            # GT
            num_gt = targets['num_boxes'][i].item()
            tgt_boxes.append(targets['boxes'][offset:offset+num_gt][tgt_idx])
            tgt_boxes_xyxy.append(targets['boxes_xyxy'][offset:offset+num_gt][tgt_idx])

            offset += num_gt

        if len(src_boxes) == 0:
            # マッチなし
            device = outputs['pred_boxes'].device
            return {
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }

        src_boxes = torch.cat(src_boxes, dim=0)  # (M, 4)
        tgt_boxes = torch.cat(tgt_boxes, dim=0)  # (M, 4)
        src_boxes_xyxy = torch.cat(src_boxes_xyxy, dim=0)  # (M, 4)
        tgt_boxes_xyxy = torch.cat(tgt_boxes_xyxy, dim=0)  # (M, 4)


        # ========================================
        # L1 Loss
        # ========================================

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='sum')
        loss_bbox = loss_bbox / num_boxes
        loss_bbox = loss_bbox * self.weight_dict.get('loss_bbox', 1.0)


        # ========================================
        # GIoU Loss
        # ========================================

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy))
        loss_giou = loss_giou.sum() / num_boxes
        loss_giou = loss_giou * self.weight_dict.get('loss_giou', 1.0)


        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }


class MaskLoss(nn.Module):
    """
    マスクロス (Sigmoid Focal Loss + Dice Loss)
    """

    def __init__(
        self,
        weight_dict: Dict[str, float],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_sample_points: int = 12544  # サンプリングポイント数
    ):
        super().__init__()

        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_sample_points = num_sample_points


    def forward(self, outputs, targets, indices, num_boxes):
        """
        マスクロスの計算

        出力:
            Dict {'loss_mask': Tensor, 'loss_dice': Tensor}
        """

        if 'pred_masks' not in outputs:
            device = outputs['pred_logits'].device
            return {
                'loss_mask': torch.tensor(0.0, device=device),
                'loss_dice': torch.tensor(0.0, device=device)
            }

        # マッチした予測とGTマスクを抽出
        src_masks = []
        tgt_masks = []

        offset = 0
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            src_masks.append(outputs['pred_masks'][i, src_idx])

            num_gt = targets['num_boxes'][i].item()
            tgt_masks.append(targets['masks'][offset:offset+num_gt][tgt_idx])

            offset += num_gt

        if len(src_masks) == 0:
            device = outputs['pred_masks'].device
            return {
                'loss_mask': torch.tensor(0.0, device=device),
                'loss_dice': torch.tensor(0.0, device=device)
            }

        src_masks = torch.cat(src_masks, dim=0)  # (M, H, W)
        tgt_masks = torch.cat(tgt_masks, dim=0)  # (M, H, W)


        # ========================================
        # ポイントサンプリング (効率化)
        # ========================================

        # ランダム + 不確実性ベースのサンプリング
        src_masks_sampled, tgt_masks_sampled = self._sample_points(
            src_masks, tgt_masks, self.num_sample_points
        )
        # src_masks_sampled, tgt_masks_sampled: (M, num_sample_points)


        # ========================================
        # Sigmoid Focal Loss
        # ========================================

        loss_mask = sigmoid_focal_loss(
            src_masks_sampled,
            tgt_masks_sampled,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction='sum'
        )
        loss_mask = loss_mask / num_boxes
        loss_mask = loss_mask * self.weight_dict.get('loss_mask', 1.0)


        # ========================================
        # Dice Loss
        # ========================================

        loss_dice = dice_loss(
            src_masks_sampled,
            tgt_masks_sampled,
            reduction='sum'
        )
        loss_dice = loss_dice / num_boxes
        loss_dice = loss_dice * self.weight_dict.get('loss_dice', 1.0)


        return {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice
        }


    def _sample_points(self, src_masks, tgt_masks, num_points):
        """
        不確実性ベースのポイントサンプリング

        入力:
            src_masks: (M, H, W) - 予測マスク
            tgt_masks: (M, H, W) - GTマスク
            num_points: サンプリング数

        出力:
            src_sampled, tgt_sampled: (M, num_points)
        """

        M, H, W = src_masks.shape

        # 簡略化: ランダムサンプリング
        # 実際には不確実性(sigmoidに近い0.5)を優先的にサンプリング

        # ランダムな座標
        with torch.no_grad():
            point_coords = torch.rand(M, num_points, 2, device=src_masks.device)
            # point_coords: (M, num_points, 2) in [0, 1]

        # グリッドサンプリング
        src_sampled = self._point_sample(src_masks, point_coords)
        tgt_sampled = self._point_sample(tgt_masks, point_coords)

        return src_sampled, tgt_sampled


    def _point_sample(self, masks, point_coords):
        """
        座標でマスクをサンプリング

        入力:
            masks: (M, H, W)
            point_coords: (M, N, 2) - 正規化座標 [0, 1]

        出力:
            samples: (M, N)
        """

        M, H, W = masks.shape
        N = point_coords.shape[1]

        # 正規化座標を [-1, 1] に変換 (grid_sample用)
        point_coords = point_coords * 2 - 1  # [0, 1] -> [-1, 1]
        point_coords = point_coords.unsqueeze(1)  # (M, 1, N, 2)

        # マスクを (M, 1, H, W) に reshape
        masks = masks.unsqueeze(1)  # (M, 1, H, W)

        # サンプリング
        samples = F.grid_sample(
            masks,
            point_coords,
            mode='bilinear',
            align_corners=False
        )  # (M, 1, 1, N)

        samples = samples.squeeze(1).squeeze(1)  # (M, N)

        return samples


# ============================================
# 補助関数
# ============================================

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU (GIoU)

    入力:
        boxes1: (N, 4) - (x1, y1, x2, y2)
        boxes2: (M, 4) - (x1, y1, x2, y2)

    出力:
        giou: (N, M) - GIoU行列
    """

    # IoU
    iou = box_iou(boxes1, boxes2)  # (N, M)

    # 最小包含ボックス
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Union面積
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    union = area1[:, None] + area2 - iou * area1[:, None]  # (N, M)

    # GIoU
    giou = iou - (area_c - union) / (area_c + 1e-7)

    return giou


def box_iou(boxes1, boxes2):
    """
    IoU計算

    入力:
        boxes1: (N, 4) - (x1, y1, x2, y2)
        boxes2: (M, 4) - (x1, y1, x2, y2)

    出力:
        iou: (N, M)
    """

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-7)

    return iou


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Sigmoid Focal Loss

    入力:
        inputs: (N,) - ロジット
        targets: (N,) - ターゲット (0 or 1)
        alpha: 正例の重み
        gamma: フォーカスパラメータ

    出力:
        loss: スカラー
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def dice_loss(inputs, targets, reduction='mean'):
    """
    Dice Loss

    入力:
        inputs: (N,) - ロジット
        targets: (N,) - ターゲット (0 or 1)

    出力:
        loss: スカラー
    """

    inputs = torch.sigmoid(inputs)

    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()

    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss


# ============================================
# 使用例
# ============================================

def example_loss_computation():
    """ロス計算の使用例"""

    # マッチャーの初期化
    matcher = BinaryHungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal=True
    )

    # O2Mマッチャー
    o2m_matcher = BinaryOneToManyMatcher(
        topk=4,
        threshold=0.4
    )

    # ロス関数の定義
    loss_fns = [
        ClassificationLoss(
            weight_dict={'loss_ce': 20.0},
            pos_weight=10.0
        ),
        BoxLoss(
            weight_dict={'loss_bbox': 5.0, 'loss_giou': 2.0}
        ),
        MaskLoss(
            weight_dict={'loss_mask': 200.0, 'loss_dice': 10.0}
        )
    ]

    # ロスラッパー
    loss_wrapper = Sam3LossWrapper(
        loss_fns=loss_fns,
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0
    )

    # ダミーデータ
    B, Q = 2, 200
    outputs = {
        'pred_logits': torch.randn(B, Q, 1),
        'pred_boxes': torch.rand(B, Q, 4),
        'pred_boxes_xyxy': torch.rand(B, Q, 4),
        'pred_masks': torch.randn(B, Q, 256, 256),
    }

    targets = {
        'boxes': torch.rand(5, 4),  # 5個のGT
        'boxes_xyxy': torch.rand(5, 4),
        'masks': torch.rand(5, 256, 256),
        'num_boxes': torch.tensor([3, 2]),  # 画像0に3個、画像1に2個
    }

    # ロス計算
    losses = loss_wrapper(outputs, targets)

    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")


if __name__ == "__main__":
    example_loss_computation()
