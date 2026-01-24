import mlflow
import numpy as np

def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union

with mlflow.start_run():
    score = iou(pred_mask, gt_mask)
    mlflow.log_metric("iou", score)
