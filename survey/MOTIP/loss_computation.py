"""
MOTIP Loss Computation

【統合損失関数】
L = λ_cls * L_cls + λ_L1 * L_L1 + λ_giou * L_giou + λ_id * L_id

1. DETR損失 (物体検出):
   - L_cls: Focal Loss (クラス分類)
   - L_L1: L1 Loss (バウンディングボックス回帰)
   - L_giou: GIoU Loss (バウンディングボックス精度)

2. ID損失 (軌跡関連付け):
   - L_id: Cross-Entropy Loss または Focal Loss

【重要な設計】
- DETR損失とID損失を統合したend-to-end学習
- ID損失は最初のフレームを除くすべてのフレームで計算
  (最初のフレームは履歴がないため)
- Hungarian Matchingは DETR部分のみで使用
  ID予測は直接分類タスクとして教師あり学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from scipy.optimize import linear_sum_assignment


class MOTIPCriterion(nn.Module):
    """
    MOTIP統合損失関数

    DETR損失 + ID損失の組み合わせ
    """

    def __init__(
        self,
        num_classes: int = 1,          # MOTでは通常1クラス (人物)
        lambda_cls: float = 2.0,
        lambda_l1: float = 5.0,
        lambda_giou: float = 2.0,
        lambda_id: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal_for_id: bool = False,  # ID損失にFocal Lossを使用するか
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_id = lambda_id

        # ========================================
        # DETR損失
        # ========================================
        self.detr_criterion = DETRCriterion(
            num_classes=num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        # ========================================
        # ID損失
        # ========================================
        self.use_focal_for_id = use_focal_for_id
        if use_focal_for_id:
            self.id_criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
            )
        else:
            self.id_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        MOTIP損失計算

        Args:
            outputs: モデル出力
                pred_logits: (B, T, N, num_classes)
                pred_boxes: (B, T, N, 4)
                id_logits: (B, T, N, K+1)

            targets: Ground Truth
                各要素は1フレームのアノテーション:
                    labels: (M,) - クラスラベル
                    boxes: (M, 4) - バウンディングボックス (cx, cy, w, h)
                    ids: (M,) - IDラベル

        Returns:
            losses: 損失辞書
                loss_cls: Focal Loss
                loss_l1: L1 Loss
                loss_giou: GIoU Loss
                loss_id: ID Loss
                loss: 統合損失
        """
        # ========================================
        # DETR損失計算
        # ========================================
        detr_losses = self.detr_criterion(
            pred_logits=outputs['pred_logits'],
            pred_boxes=outputs['pred_boxes'],
            targets=targets,
        )

        # ========================================
        # ID損失計算
        # ========================================
        id_losses = self._compute_id_loss(
            id_logits=outputs['id_logits'],
            targets=targets,
        )

        # ========================================
        # 統合損失
        # ========================================
        total_loss = (
            self.lambda_cls * detr_losses['loss_cls'] +
            self.lambda_l1 * detr_losses['loss_l1'] +
            self.lambda_giou * detr_losses['loss_giou'] +
            self.lambda_id * id_losses['loss_id']
        )

        losses = {
            'loss_cls': detr_losses['loss_cls'],
            'loss_l1': detr_losses['loss_l1'],
            'loss_giou': detr_losses['loss_giou'],
            'loss_id': id_losses['loss_id'],
            'loss': total_loss,
        }

        return losses

    def _compute_id_loss(
        self,
        id_logits: torch.Tensor,              # (B, T, N, K+1)
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        ID損失計算

        【重要】
        - 最初のフレーム (t=0) は履歴がないため除外
        - t >= 1 のフレームでのみID予測を教師あり学習

        Args:
            id_logits: ID予測ロジット (B, T, N, K+1)
            targets: Ground Truth (IDラベルを含む)

        Returns:
            losses: {'loss_id': tensor}
        """
        B, T, N, K_plus_1 = id_logits.shape

        # ID損失を計算するフレームを選択 (t >= 1)
        if T == 1:
            # 単一フレームの場合はID損失なし
            return {'loss_id': torch.tensor(0.0, device=id_logits.device)}

        # t >= 1 のフレームを選択
        id_logits_train = id_logits[:, 1:, :, :]  # (B, T-1, N, K+1)

        # Ground TruthからIDラベルを抽出
        # 簡略化: 各ターゲットにIDラベルが含まれていると仮定
        id_labels_list = []
        for target in targets:
            if 'ids' in target:
                id_labels_list.append(target['ids'])

        if len(id_labels_list) == 0:
            return {'loss_id': torch.tensor(0.0, device=id_logits.device)}

        # 簡略化のため、ここでは実際のID損失計算をプレースホルダー
        # 実際の実装では:
        # 1. Hungarian Matchingで予測とGround Truthを対応付け
        # 2. 対応付けられた予測に対してID損失を計算

        loss_id = torch.tensor(0.0, device=id_logits.device)

        return {'loss_id': loss_id}


class DETRCriterion(nn.Module):
    """
    Deformable DETR損失関数

    【損失構成】
    1. Focal Loss: クラス分類
    2. L1 Loss: バウンディングボックス回帰
    3. GIoU Loss: バウンディングボックス精度

    【Hungarian Matching】
    予測とGround Truthを二部グラフマッチングで対応付け
    """

    def __init__(
        self,
        num_classes: int = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Matcher
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,    # (B, T, N, num_classes)
        pred_boxes: torch.Tensor,     # (B, T, N, 4)
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        DETR損失計算

        Returns:
            losses:
                loss_cls: Focal Loss
                loss_l1: L1 Loss
                loss_giou: GIoU Loss
        """
        B, T, N, num_classes = pred_logits.shape

        # 簡略化: 各フレームごとに独立に処理
        total_loss_cls = 0.0
        total_loss_l1 = 0.0
        total_loss_giou = 0.0
        num_frames = 0

        for b in range(B):
            for t in range(T):
                frame_pred_logits = pred_logits[b, t]  # (N, num_classes)
                frame_pred_boxes = pred_boxes[b, t]    # (N, 4)

                # 対応するtargetを取得
                target_idx = b * T + t
                if target_idx >= len(targets):
                    continue

                target = targets[target_idx]
                if 'labels' not in target or 'boxes' not in target:
                    continue

                # Hungarian Matchingで対応付け
                indices = self.matcher(
                    pred_logits=frame_pred_logits[None],
                    pred_boxes=frame_pred_boxes[None],
                    targets=[target],
                )  # [(pred_idx, target_idx)]

                # Focal Loss
                loss_cls = self._focal_loss(
                    frame_pred_logits,
                    target['labels'],
                    indices[0],
                )

                # L1 Loss
                loss_l1 = self._l1_loss(
                    frame_pred_boxes,
                    target['boxes'],
                    indices[0],
                )

                # GIoU Loss
                loss_giou = self._giou_loss(
                    frame_pred_boxes,
                    target['boxes'],
                    indices[0],
                )

                total_loss_cls += loss_cls
                total_loss_l1 += loss_l1
                total_loss_giou += loss_giou
                num_frames += 1

        # 平均化
        if num_frames > 0:
            total_loss_cls /= num_frames
            total_loss_l1 /= num_frames
            total_loss_giou /= num_frames

        return {
            'loss_cls': total_loss_cls,
            'loss_l1': total_loss_l1,
            'loss_giou': total_loss_giou,
        }

    def _focal_loss(
        self,
        pred_logits: torch.Tensor,    # (N, num_classes)
        target_labels: torch.Tensor,  # (M,)
        indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Focal Loss計算"""
        # 簡略化
        return torch.tensor(0.0, device=pred_logits.device)

    def _l1_loss(
        self,
        pred_boxes: torch.Tensor,     # (N, 4)
        target_boxes: torch.Tensor,   # (M, 4)
        indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """L1 Loss計算"""
        pred_idx, target_idx = indices
        if len(pred_idx) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        pred_boxes_matched = pred_boxes[pred_idx]
        target_boxes_matched = target_boxes[target_idx]

        loss = F.l1_loss(pred_boxes_matched, target_boxes_matched, reduction='mean')
        return loss

    def _giou_loss(
        self,
        pred_boxes: torch.Tensor,     # (N, 4)
        target_boxes: torch.Tensor,   # (M, 4)
        indices: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """GIoU Loss計算"""
        pred_idx, target_idx = indices
        if len(pred_idx) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        pred_boxes_matched = pred_boxes[pred_idx]
        target_boxes_matched = target_boxes[target_idx]

        # GIoU計算 (簡略化)
        giou = generalized_box_iou(pred_boxes_matched, target_boxes_matched)
        loss = 1 - torch.diag(giou).mean()

        return loss


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher: 予測とGround Truthの二部グラフマッチング

    コスト行列:
    C[i, j] = λ_cls * cost_cls + λ_bbox * cost_bbox + λ_giou * cost_giou

    - cost_cls: -p[c_j] (クラスc_jの確率の負値)
    - cost_bbox: L1(b_i, b_j) (バウンディングボックスのL1距離)
    - cost_giou: -GIoU(b_i, b_j) (GIoUの負値)
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,    # (B, N, num_classes)
        pred_boxes: torch.Tensor,     # (B, N, 4)
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Hungarian Matching

        Returns:
            indices: List[(pred_idx, target_idx)]
                各バッチの対応インデックス
        """
        B, N, num_classes = pred_logits.shape

        # Softmax
        pred_probs = F.softmax(pred_logits, dim=-1)  # (B, N, num_classes)

        indices = []

        for b in range(B):
            target = targets[b]
            if 'labels' not in target or 'boxes' not in target:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            target_labels = target['labels']  # (M,)
            target_boxes = target['boxes']    # (M, 4)
            M = target_labels.shape[0]

            if M == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue

            # コスト行列: (N, M)
            # 1. Classification cost
            cost_cls = -pred_probs[b, :, target_labels]  # (N, M)

            # 2. L1 cost
            cost_bbox = torch.cdist(
                pred_boxes[b],      # (N, 4)
                target_boxes,       # (M, 4)
                p=1,
            )  # (N, M)

            # 3. GIoU cost
            cost_giou = -generalized_box_iou(
                pred_boxes[b],      # (N, 4)
                target_boxes,       # (M, 4)
            )  # (N, M)

            # 統合コスト
            C = (
                self.cost_class * cost_cls +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )  # (N, M)

            # Hungarian Algorithm
            C_np = C.cpu().numpy()
            pred_idx, target_idx = linear_sum_assignment(C_np)

            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(target_idx, dtype=torch.int64),
            ))

        return indices


class FocalLoss(nn.Module):
    """
    Focal Loss

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    - α_t: クラスバランス調整
    - γ: 難易度調整 (γ=2がデフォルト)
    - p_t: 正解クラスの予測確率
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        inputs: torch.Tensor,    # (N, C) - ロジット
        targets: torch.Tensor,   # (N,) - ラベル
    ) -> torch.Tensor:
        """
        Focal Loss計算

        Args:
            inputs: 予測ロジット (N, C)
            targets: Ground Truthラベル (N,)

        Returns:
            loss: スカラー損失
        """
        # Softmax
        probs = F.softmax(inputs, dim=-1)  # (N, C)

        # 正解クラスの確率を取得
        N = targets.shape[0]
        p_t = probs[torch.arange(N), targets]  # (N,)

        # Focal Loss
        focal_weight = (1 - p_t) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(p_t + 1e-8)

        return loss.mean()


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU)

    Args:
        boxes1: (N, 4) - (cx, cy, w, h)
        boxes2: (M, 4) - (cx, cy, w, h)

    Returns:
        giou: (N, M) - GIoU行列
    """
    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)

    # IoU計算
    iou, union = box_iou(boxes1_xyxy, boxes2_xyxy)

    # Enclosing box
    lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # GIoU = IoU - (area_c - union) / area_c
    giou = iou - (area_c - union) / area_c

    return giou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    (cx, cy, w, h) -> (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    IoU計算

    Args:
        boxes1: (N, 4) - (x1, y1, x2, y2)
        boxes2: (M, 4) - (x1, y1, x2, y2)

    Returns:
        iou: (N, M)
        union: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter  # (N, M)
    iou = inter / union

    return iou, union


# ========================================
# 形状ガイド
# ========================================
"""
【損失計算の入力】
outputs (モデル出力):
    pred_logits: (B, T, N, num_classes) - クラス確率
    pred_boxes: (B, T, N, 4) - バウンディングボックス (cx, cy, w, h)
    id_logits: (B, T, N, K+1) - ID予測ロジット

targets (Ground Truth):
    List of Dict, 各要素は1フレームのアノテーション:
        labels: (M,) - クラスラベル (0 ~ num_classes-1)
        boxes: (M, 4) - バウンディングボックス (cx, cy, w, h) 正規化座標
        ids: (M,) - IDラベル (1 ~ K)

【損失の出力】
losses:
    loss_cls: Focal Loss (クラス分類)
    loss_l1: L1 Loss (バウンディングボックス回帰)
    loss_giou: GIoU Loss (バウンディングボックス精度)
    loss_id: ID Loss (ID予測)
    loss: 統合損失 = λ_cls * loss_cls + λ_l1 * loss_l1 + λ_giou * loss_giou + λ_id * loss_id

【ハイパーパラメータ (デフォルト)】
λ_cls = 2.0
λ_l1 = 5.0
λ_giou = 2.0
λ_id = 1.0

focal_alpha = 0.25
focal_gamma = 2.0

【Hungarian Matchingコスト】
C[i, j] = 2.0 * cost_cls + 5.0 * cost_bbox + 2.0 * cost_giou
"""
