from typing import List, Optional, Dict
import numpy as np
import torch
from torchvision.ops import box_iou


def evaluate_precision_recall(
    preds_boxes: List[np.ndarray],
    preds_scores: List[np.ndarray],
    preds_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    conf_thresholds: List[float],
    iou_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    Evaluate precision and recall for multiple confidence thresholds and IoU thresholds.

    Returns:
        Dict: Nested dict of the form result[conf][iou] = {"precision": ..., "recall": ...}

    Notes:
        - This function computes **exact precision and recall at fixed confidence thresholds**.
        - Precision = TP / (TP + FP)
        - Recall    = TP / (TP + FN)
        - Matching is done using **1-to-1 greedy matching** between predictions and GT boxes,
          respecting both IoU thresholds and class labels.
        - Multiple confidence thresholds and IoU thresholds are supported.

        Difference from MeanAveragePrecision (torchmetrics):
        - MeanAveragePrecision internally **sweeps over all confidence thresholds**
          and computes COCO-style **interpolated precision**, which is optimistic.
        - MAP returns precision sampled at 101 recall points with upper-envelope interpolation,
          while this function returns **true precision/recall at specified confidence thresholds**.
        - This function is therefore suitable when you need the precision/recall **exactly at specific conf thresholds**,
          rather than the averaged/interpolated AP used in MAP.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    n_images = len(preds_boxes)
    assert all(len(lst) == n_images for lst in [preds_scores, preds_labels, gt_boxes, gt_labels])

    results = {}
    for conf_th in conf_thresholds:
        results[conf_th] = {}
        for iou_th in iou_thresholds:
            TP = 0
            FP = 0
            FN = 0
            for pb, ps, pl, gb, gl in zip(preds_boxes, preds_scores, preds_labels, gt_boxes, gt_labels):
                # conf閾値でフィルタ
                mask = ps >= conf_th
                pb_filtered = torch.tensor(pb[mask], dtype=torch.float32)
                pl_filtered = torch.tensor(pl[mask], dtype=torch.int64)

                gb_t = torch.tensor(gb, dtype=torch.float32)
                gl_t = torch.tensor(gl, dtype=torch.int64)

                if len(pb_filtered) == 0:
                    TP += 0
                    FP += 0
                    FN += len(gb)
                    continue
                if len(gb_t) == 0:
                    TP += 0
                    FP += len(pb_filtered)
                    FN += 0
                    continue

                # IoU計算
                ious = box_iou(pb_filtered, gb_t)  # [num_pred, num_gt]
                # classごとにマスク
                class_mask = (pl_filtered[:, None] == gl_t[None, :])
                ious = ious * class_mask.float()

                # greedy matching
                matched_pred = set()
                matched_gt = set()
                for pred_idx in range(ious.shape[0]):
                    gt_idx = torch.argmax(ious[pred_idx])
                    if ious[pred_idx, gt_idx] >= iou_th:
                        if gt_idx.item() not in matched_gt:
                            matched_pred.add(pred_idx)
                            matched_gt.add(gt_idx.item())
                TP += len(matched_pred)
                FP += len(pb_filtered) - len(matched_pred)
                FN += len(gb) - len(matched_gt)

            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            results[conf_th][iou_th] = {"precision": precision, "recall": recall}

    return results


# 使い方例
preds_boxes = [np.array([[10, 20, 50, 60], [15, 25, 55, 65]]),
               np.array([[5, 5, 30, 30]])]
preds_scores = [np.array([0.9, 0.6]), np.array([0.8])]
preds_labels = [np.array([1, 2]), np.array([1])]

gt_boxes = [np.array([[12, 22, 48, 58]]), np.array([[5, 5, 28, 28]])]
gt_labels = [np.array([1]), np.array([1])]

conf_thresholds = [0.5, 0.7]
iou_thresholds = [0.5, 0.75]

metrics = evaluate_precision_recall(
    preds_boxes, preds_scores, preds_labels,
    gt_boxes, gt_labels,
    conf_thresholds, iou_thresholds
)

import pprint
pprint.pprint(metrics)
