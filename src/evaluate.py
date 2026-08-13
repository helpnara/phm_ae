"""평가지표 산출 (라벨이 있을 때). 불균형 데이터이므로 정확도는 참고용."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray, errors: np.ndarray, predictions: np.ndarray
) -> Dict[str, Any]:
    """혼동행렬, precision/recall/F1, ROC-AUC, PR-AUC를 계산한다.

    y_true: 0=정상, 1=이상
    errors: 재구성 오차(연속 점수, AUC용)
    predictions: 0/1 (임계값 적용 결과)
    """
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, accuracy_score,
    )

    y_true = np.asarray(y_true).astype(int)
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["precision"] = float(precision_score(y_true, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, predictions, zero_division=0))

    # AUC는 양/음 클래스가 모두 있어야 계산 가능
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, errors))
        metrics["pr_auc"] = float(average_precision_score(y_true, errors))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    return metrics


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Optional[Dict[str, Any]]:
    """F1을 최대화하는 임계값을 PR 곡선에서 찾아 추천한다(라벨 필요)."""
    from sklearn.metrics import precision_recall_curve
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall 길이 = thresholds + 1 → 마지막 원소 제외
    p, r = precision[:-1], recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    if len(f1) == 0:
        return None
    idx = int(np.argmax(f1))
    return {"threshold": float(thresholds[idx]), "f1": float(f1[idx]),
            "precision": float(p[idx]), "recall": float(r[idx])}


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> Optional[Dict[str, Any]]:
    """ROC 커브 좌표(fpr, tpr)와 AUC. 양·음 클래스가 모두 있어야 계산."""
    from sklearn.metrics import roc_curve, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_true, scores))}


def pr_curve_points(y_true: np.ndarray, scores: np.ndarray) -> Optional[Dict[str, Any]]:
    """Precision-Recall 커브 좌표와 AP(=PR-AUC). 불균형에 강건한 평가."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {"precision": precision.tolist(), "recall": recall.tolist(),
            "ap": float(average_precision_score(y_true, scores))}
