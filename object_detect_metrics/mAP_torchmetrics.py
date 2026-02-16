from typing import List, Optional
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate_map(
    preds_boxes: List[np.ndarray],
    preds_scores: List[np.ndarray],
    preds_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None
) -> float:
    """
    Compute mean Average Precision (mAP) for object detection using torchmetrics,
    with numpy array inputs.

    Args:
        preds_boxes (List[np.ndarray]): List of predicted boxes per image,
            each of shape [num_boxes, 4] in (x1, y1, x2, y2) format.
        preds_scores (List[np.ndarray]): List of predicted scores per image, shape [num_boxes].
        preds_labels (List[np.ndarray]): List of predicted class labels per image, shape [num_boxes].
        gt_boxes (List[np.ndarray]): List of ground truth boxes per image, shape [num_boxes, 4].
        gt_labels (List[np.ndarray]): List of ground truth labels per image, shape [num_boxes].
        iou_thresholds (Optional[List[float]]): List of IoU thresholds for mAP calculation.
            Defaults to COCO-style 0.5:0.95 step 0.05 if None.

    Returns:
        float: Mean Average Precision (mAP) over all classes and IoU thresholds.
    """
    n_images = len(preds_boxes)
    assert all(len(lst) == n_images for lst in [preds_scores, preds_labels, gt_boxes, gt_labels]), \
        "All input lists must have the same length equal to number of images"

    # Convert numpy inputs to torchmetrics-friendly format
    preds_tm = []
    gts_tm = []

    for pb, ps, pl, gb, gl in zip(preds_boxes, preds_scores, preds_labels, gt_boxes, gt_labels):
        preds_tm.append({
            'boxes': torch.tensor(pb, dtype=torch.float32),
            'scores': torch.tensor(ps, dtype=torch.float32),
            'labels': torch.tensor(pl, dtype=torch.int64)
        })
        gts_tm.append({
            'boxes': torch.tensor(gb, dtype=torch.float32),
            'labels': torch.tensor(gl, dtype=torch.int64)
        })

    # Initialize MeanAveragePrecision metric
    if iou_thresholds is not None:
        map_metric = MeanAveragePrecision(iou_thresholds=torch.tensor(iou_thresholds, dtype=torch.float32))
    else:
        map_metric = MeanAveragePrecision()  # defaults to COCO-style 0.5:0.95

    # Update metric
    map_metric.update(preds_tm, gts_tm)

    # Compute final mAP
    metrics = map_metric.compute()
    return metrics['map'].item()


# 2画像の例
preds_boxes = [np.array([[10, 20, 50, 60], [15, 25, 55, 65]]),
               np.array([[5, 5, 30, 30]])]
preds_scores = [np.array([0.9, 0.6]), np.array([0.8])]
preds_labels = [np.array([1, 2]), np.array([1])]

gt_boxes = [np.array([[12, 22, 48, 58]]), np.array([[5, 5, 28, 28]])]
gt_labels = [np.array([1]), np.array([1])]

map_val = evaluate_map(preds_boxes, preds_scores, preds_labels, gt_boxes, gt_labels)
print("mAP:", map_val)
# mAP: 0.699999988079071
