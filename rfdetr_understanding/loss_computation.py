"""
RF-DETR Loss Computation - 簡略化疑似コード
==========================================

RF-DETRの損失計算
- IA-BCE Loss (IoU-Aware Binary Cross-Entropy) - KEY INNOVATION
- Hungarian Matching (検出タスク)
- Deep Supervision (全デコーダレイヤー)
- Point Sampling (セグメンテーション)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class RFDETRLossWrapper(nn.Module):
    """
    RF-DETR 損失計算ラッパー

    特徴:
    - IA-BCE Loss: IoU情報を統合したBinary Cross-Entropy
    - Deep Supervision: 全デコーダレイヤーで損失計算
    - One-to-One Matching: Hungarian algorithm
    - Auxiliary losses: bbox, giou
    """

    def __init__(
        self,
        num_classes: int = 80,
        matcher: str = 'HungarianMatcher',
        loss_weights: Dict[str, float] = None
    ):
        super().__init__()

        self.num_classes = num_classes

        # デフォルト損失重み
        if loss_weights is None:
            loss_weights = {
                'loss_cls': 2.0,      # IA-BCE Loss
                'loss_bbox': 5.0,     # L1 Loss
                'loss_giou': 2.0,     # GIoU Loss
                'loss_mask': 2.0,     # Mask Loss (セグメンテーション時)
                'loss_dice': 5.0      # Dice Loss (セグメンテーション時)
            }
        self.loss_weights = loss_weights

        # Hungarian Matcher
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )

        # 個別損失関数
        self.iabce_loss = IABCELoss(num_classes=num_classes)
        self.bbox_loss = BBoxLoss()
        self.giou_loss = GIoULoss()


    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        損失計算

        入力:
            outputs: {
                'pred_logits': (B, num_queries, num_classes)
                'pred_boxes': (B, num_queries, 4)
                'pred_masks': (B, num_queries, H, W) [optional]
                'aux_outputs': List[dict] - 中間レイヤー出力 (Deep Supervision用)
            }

            targets: List of dicts, length = B
                Each dict: {
                    'labels': (num_objects,) - クラスラベル
                    'boxes': (num_objects, 4) - [cx, cy, w, h] 正規化
                    'masks': (num_objects, H, W) [optional]
                }

        出力:
            losses: {
                'loss_cls': scalar
                'loss_bbox': scalar
                'loss_giou': scalar
                'loss_mask': scalar [optional]
                'loss_dice': scalar [optional]
            }
        """

        # ========================================
        # 最終レイヤーの損失
        # ========================================
        losses = self._compute_loss_single_layer(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            outputs.get('pred_masks', None),
            targets,
            layer_name='final'
        )

        # ========================================
        # Deep Supervision: 中間レイヤーの損失
        # ========================================
        if 'aux_outputs' in outputs:
            for i, aux_output in enumerate(outputs['aux_outputs']):
                aux_losses = self._compute_loss_single_layer(
                    aux_output['pred_logits'],
                    aux_output['pred_boxes'],
                    aux_output.get('pred_masks', None),
                    targets,
                    layer_name=f'aux_{i}'
                )

                # 補助損失を追加
                for k, v in aux_losses.items():
                    losses[k + f'_aux_{i}'] = v

        return losses


    def _compute_loss_single_layer(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        layer_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        単一レイヤーの損失計算

        入力:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4)
            pred_masks: (B, num_queries, H, W) or None
            targets: List[dict]

        出力:
            losses: {
                'loss_cls': scalar
                'loss_bbox': scalar
                'loss_giou': scalar
            }
        """

        # ========================================
        # Step 1: Hungarian Matching
        # ========================================
        # 予測とGTをマッチング
        indices = self.matcher(
            pred_logits,
            pred_boxes,
            targets
        )
        # indices: List[(query_indices, target_indices)] length = B

        # ========================================
        # Step 2: マッチング結果を抽出
        # ========================================
        # マッチしたクエリとターゲットを取得
        matched_pred_logits = []
        matched_pred_boxes = []
        matched_target_labels = []
        matched_target_boxes = []

        for i, (query_idx, target_idx) in enumerate(indices):
            matched_pred_logits.append(pred_logits[i, query_idx])
            matched_pred_boxes.append(pred_boxes[i, query_idx])
            matched_target_labels.append(targets[i]['labels'][target_idx])
            matched_target_boxes.append(targets[i]['boxes'][target_idx])

        # 連結
        matched_pred_logits = torch.cat(matched_pred_logits, dim=0)    # (N_matched, num_classes)
        matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)      # (N_matched, 4)
        matched_target_labels = torch.cat(matched_target_labels, dim=0) # (N_matched,)
        matched_target_boxes = torch.cat(matched_target_boxes, dim=0)   # (N_matched, 4)


        # ========================================
        # Step 3: 損失計算
        # ========================================

        losses = {}

        # (1) Classification Loss: IA-BCE
        loss_cls = self.iabce_loss(
            matched_pred_logits,
            matched_target_labels,
            matched_pred_boxes,
            matched_target_boxes
        )
        losses['loss_cls'] = loss_cls * self.loss_weights['loss_cls']

        # (2) BBox L1 Loss
        loss_bbox = self.bbox_loss(
            matched_pred_boxes,
            matched_target_boxes
        )
        losses['loss_bbox'] = loss_bbox * self.loss_weights['loss_bbox']

        # (3) GIoU Loss
        loss_giou = self.giou_loss(
            matched_pred_boxes,
            matched_target_boxes
        )
        losses['loss_giou'] = loss_giou * self.loss_weights['loss_giou']

        # (4) Mask Losses (セグメンテーション時)
        if pred_masks is not None:
            matched_pred_masks = []
            matched_target_masks = []

            for i, (query_idx, target_idx) in enumerate(indices):
                matched_pred_masks.append(pred_masks[i, query_idx])
                matched_target_masks.append(targets[i]['masks'][target_idx])

            matched_pred_masks = torch.stack(matched_pred_masks, dim=0)    # (N_matched, H, W)
            matched_target_masks = torch.stack(matched_target_masks, dim=0)  # (N_matched, H, W)

            # Point Sampling で効率化
            from segmentation import get_uncertain_point_coords_with_randomness, point_sample

            point_coords = get_uncertain_point_coords_with_randomness(
                matched_pred_masks.unsqueeze(0),  # (1, N_matched, H, W)
                num_points=12544
            ).squeeze(0)  # (N_matched, num_points, 2)

            # サンプリング
            sampled_pred = point_sample(
                matched_pred_masks.unsqueeze(0),
                point_coords.unsqueeze(0)
            ).squeeze(0)  # (N_matched, num_points)

            sampled_target = point_sample(
                matched_target_masks.unsqueeze(0),
                point_coords.unsqueeze(0)
            ).squeeze(0)  # (N_matched, num_points)

            # BCE Loss
            loss_mask = F.binary_cross_entropy_with_logits(
                sampled_pred,
                sampled_target.float()
            )
            losses['loss_mask'] = loss_mask * self.loss_weights['loss_mask']

            # Dice Loss
            loss_dice = dice_loss(
                sampled_pred.sigmoid(),
                sampled_target.float()
            )
            losses['loss_dice'] = loss_dice * self.loss_weights['loss_dice']

        return losses


class IABCELoss(nn.Module):
    """
    IA-BCE Loss (IoU-Aware Binary Cross-Entropy)

    RF-DETRのキー・イノベーション:
    - 通常のBCEにIoU情報を統合
    - クラス確率とBBox品質を同時に学習
    - より正確な検出スコアリング

    従来のBCE:
        BCE(p, y) = -[y * log(p) + (1-y) * log(1-p)]

    IA-BCE:
        IA-BCE(p, y, iou) = -[y * iou * log(p) + (1-y) * log(1-p)]

    ターゲットラベルをIoUで重み付け:
    - 高IoU → 強い正例シグナル
    - 低IoU → 弱い正例シグナル
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes


    def forward(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        IA-BCE Loss計算

        入力:
            pred_logits: (N, num_classes) - 予測ロジット
            target_labels: (N,) - ターゲットクラスID
            pred_boxes: (N, 4) - 予測BBox [cx, cy, w, h]
            target_boxes: (N, 4) - ターゲットBBox [cx, cy, w, h]

        出力:
            loss: scalar
        """

        # ========================================
        # Step 1: IoU計算
        # ========================================
        # BBox IoUを計算
        ious = compute_box_iou(pred_boxes, target_boxes)  # (N,)

        # ========================================
        # Step 2: ターゲットラベルベクトル作成
        # ========================================
        # One-hotエンコーディング
        target_classes_onehot = F.one_hot(
            target_labels,
            num_classes=self.num_classes
        ).float()  # (N, num_classes)

        # ========================================
        # Step 3: IoUで重み付け
        # ========================================
        # ターゲットをIoUで重み付け
        # target = 1.0 * iou (正例) or 0.0 (負例)
        target_classes_weighted = target_classes_onehot * ious.unsqueeze(1)
        # (N, num_classes)

        # ========================================
        # Step 4: Binary Cross Entropy
        # ========================================
        # Sigmoid + BCE
        pred_probs = pred_logits.sigmoid()

        # BCE計算
        loss = F.binary_cross_entropy(
            pred_probs,
            target_classes_weighted,
            reduction='mean'
        )

        return loss


class HungarianMatcher(nn.Module):
    """
    Hungarian Matching for Object Detection

    DETR標準のマッチング:
    - 予測クエリとGTオブジェクトを最適割り当て
    - コスト行列: classification + bbox + giou
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou


    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Hungarian Matching

        入力:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4)
            targets: List[dict] length = B

        出力:
            indices: List[(query_indices, target_indices)] length = B
        """

        B, num_queries, _ = pred_logits.shape

        # Sigmoid確率
        pred_probs = pred_logits.sigmoid()  # (B, num_queries, num_classes)

        indices = []

        for i in range(B):
            # バッチ内の各画像について
            target_labels = targets[i]['labels']  # (num_objects,)
            target_boxes = targets[i]['boxes']    # (num_objects, 4)
            num_objects = target_labels.shape[0]

            if num_objects == 0:
                # オブジェクトなし
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            # ========================================
            # Step 1: Classification Cost
            # ========================================
            # 各クエリと各ターゲットクラスのコスト
            cost_class = -pred_probs[i, :, target_labels]  # (num_queries, num_objects)

            # ========================================
            # Step 2: BBox L1 Cost
            # ========================================
            cost_bbox = torch.cdist(
                pred_boxes[i],
                target_boxes,
                p=1
            )  # (num_queries, num_objects)

            # ========================================
            # Step 3: GIoU Cost
            # ========================================
            cost_giou = -compute_giou(
                pred_boxes[i].unsqueeze(1),
                target_boxes.unsqueeze(0)
            )  # (num_queries, num_objects)

            # ========================================
            # Step 4: 総コスト
            # ========================================
            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )  # (num_queries, num_objects)

            # ========================================
            # Step 5: Hungarian Algorithm
            # ========================================
            # scipy.optimize.linear_sum_assignment を使用
            from scipy.optimize import linear_sum_assignment

            query_idx, target_idx = linear_sum_assignment(
                cost_matrix.cpu().numpy()
            )

            indices.append((
                torch.as_tensor(query_idx, dtype=torch.int64),
                torch.as_tensor(target_idx, dtype=torch.int64)
            ))

        return indices


class BBoxLoss(nn.Module):
    """BBox L1 Loss"""

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        入力:
            pred_boxes: (N, 4)
            target_boxes: (N, 4)
        """
        return F.l1_loss(pred_boxes, target_boxes, reduction='mean')


class GIoULoss(nn.Module):
    """Generalized IoU Loss"""

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        入力:
            pred_boxes: (N, 4)
            target_boxes: (N, 4)
        """
        giou = compute_giou(
            pred_boxes.unsqueeze(1),
            target_boxes.unsqueeze(1)
        ).squeeze()  # (N,)

        loss = 1 - giou.mean()
        return loss


# ============================================
# ユーティリティ関数
# ============================================

def compute_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    BBox IoU計算

    入力:
        boxes1: (N, 4) [cx, cy, w, h]
        boxes2: (N, 4) [cx, cy, w, h]

    出力:
        ious: (N,)
    """
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    # 交差領域
    lt = torch.max(boxes1_xyxy[:, :2], boxes2_xyxy[:, :2])
    rb = torch.min(boxes1_xyxy[:, 2:], boxes2_xyxy[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    # 面積
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # IoU
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    return iou


def compute_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Generalized IoU計算

    入力:
        boxes1: (N, M, 4) or (N, 4) [cx, cy, w, h]
        boxes2: (N, M, 4) or (N, 4) [cx, cy, w, h]

    出力:
        giou: same shape as input
    """
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    # IoU計算
    lt = torch.max(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    rb = torch.min(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]

    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    # 包含ボックス
    lt_enclosing = torch.min(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    rb_enclosing = torch.max(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])

    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[..., 0] * wh_enclosing[..., 1]

    # GIoU
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)

    return giou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    [cx, cy, w, h] -> [x1, y1, x2, y2]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Dice Loss

    入力:
        pred: (N, K) - 確率 [0, 1]
        target: (N, K) - ターゲット {0, 1}
    """
    smooth = 1.0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1 - dice

    return loss


# ============================================
# 使用例
# ============================================

def example_loss_computation():
    """損失計算の使用例"""

    # 損失ラッパー
    loss_wrapper = RFDETRLossWrapper(
        num_classes=80,
        loss_weights={
            'loss_cls': 2.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_mask': 2.0,
            'loss_dice': 5.0
        }
    )

    # ダミー出力
    outputs = {
        'pred_logits': torch.randn(2, 300, 80),
        'pred_boxes': torch.rand(2, 300, 4),
        'pred_masks': torch.randn(2, 300, 108, 108),
        'aux_outputs': [
            {
                'pred_logits': torch.randn(2, 300, 80),
                'pred_boxes': torch.rand(2, 300, 4),
                'pred_masks': torch.randn(2, 300, 108, 108)
            }
            for _ in range(5)  # 6レイヤーデコーダ
        ]
    }

    # ダミーターゲット
    targets = [
        {
            'labels': torch.tensor([0, 15, 62]),  # 3オブジェクト
            'boxes': torch.rand(3, 4),
            'masks': torch.randint(0, 2, (3, 108, 108)).float()
        },
        {
            'labels': torch.tensor([5, 23]),  # 2オブジェクト
            'boxes': torch.rand(2, 4),
            'masks': torch.randint(0, 2, (2, 108, 108)).float()
        }
    ]

    # 損失計算
    losses = loss_wrapper(outputs, targets)

    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # 総損失
    total_loss = sum(losses.values())
    print(f"\nTotal Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    example_loss_computation()
