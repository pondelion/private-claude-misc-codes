from typing import List, Optional
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import json
import os


def evaluate_map_coco(
    preds_boxes: List[np.ndarray],
    preds_scores: List[np.ndarray],
    preds_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None,
) -> float:
    """
    Compute COCO-style mAP using pycocotools.

    Args:
        preds_boxes: list of [N_i, 4] xyxy numpy arrays
        preds_scores: list of [N_i]
        preds_labels: list of [N_i]
        gt_boxes: list of [M_i, 4] xyxy numpy arrays
        gt_labels: list of [M_i]
        iou_thresholds: optional list of IoU thresholds

    Returns:
        float: mAP
    """

    n_images = len(preds_boxes)
    assert all(len(lst) == n_images for lst in
               [preds_scores, preds_labels, gt_boxes, gt_labels]), \
        "All inputs must have same number of images"

    # ---- Build COCO GT format ----
    images = []
    annotations = []
    categories = set()

    ann_id = 1

    for img_id in range(n_images):
        images.append({
            "id": img_id
        })

        for box, label in zip(gt_boxes[img_id], gt_labels[img_id]):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, w, h],  # COCO is xywh
                "area": float(w * h),
                "iscrowd": 0,
            })
            categories.add(int(label))
            ann_id += 1

    categories = [{"id": cid} for cid in sorted(categories)]

    coco_gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # 一時ファイルに保存（pycocotoolsはファイル要求）
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_gt_dict, f)
        tmp_gt_path = f.name

    coco_gt = COCO(tmp_gt_path)

    # ---- Build predictions format ----
    coco_results = []

    for img_id in range(n_images):
        for box, score, label in zip(
            preds_boxes[img_id],
            preds_scores[img_id],
            preds_labels[img_id],
        ):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            coco_results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            })

    coco_dt = coco_gt.loadRes(coco_results)

    # ---- Evaluation ----
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.remove(tmp_gt_path)

    # mAP 0.5:0.95
    return float(coco_eval.stats[0])


# 2画像の例
preds_boxes = [np.array([[10, 20, 50, 60], [15, 25, 55, 65]]),
               np.array([[5, 5, 30, 30]])]
preds_scores = [np.array([0.9, 0.6]), np.array([0.8])]
preds_labels = [np.array([1, 2]), np.array([1])]

gt_boxes = [np.array([[12, 22, 48, 58]]), np.array([[5, 5, 28, 28]])]
gt_labels = [np.array([1]), np.array([1])]

map_val = evaluate_map_coco(preds_boxes, preds_scores, preds_labels, gt_boxes, gt_labels)
print("mAP:", map_val)

# loading annotations into memory...
# Done (t=0.00s)
# creating index...
# index created!
# Loading and preparing results...
# DONE (t=0.00s)
# creating index...
# index created!
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=0.01s).
# Accumulating evaluation results...
# DONE (t=0.01s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.700
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.700
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.700
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.700
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.700
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.700
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.700
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.700
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
# mAP: 0.7
