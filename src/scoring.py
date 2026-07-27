"""재구성 오차 계산 및 임계값 산출."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def per_feature_squared_error(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """피처별 제곱 오차 (n_samples, n_features). 원인 진단(설명가능성)에 사용."""
    return np.square(X - X_hat)


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """샘플별 재구성 오차(피처 평균 MSE) (n_samples,)."""
    return per_feature_squared_error(X, X_hat).mean(axis=1)


def seq_window_errors(Xw: np.ndarray, Xw_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """시퀀스 재구성 오차. (n_windows,) 윈도우 오차와 (n_windows, F) 피처별 오차."""
    se = np.square(Xw - Xw_hat)          # (nw, W, F)
    win_err = se.mean(axis=(1, 2))       # (nw,)
    per_feature = se.mean(axis=1)        # (nw, F)
    return win_err, per_feature


def threshold_for_far(normal_errors: np.ndarray, target_far: float) -> float:
    """허용 오경보율(FAR)을 만족하는 임계값을 정상 오차 분포에서 역산한다.

    target_far=0.01 이면 정상의 1%만 이상으로 판정되는 지점(=99 백분위수).
    """
    e = np.asarray(normal_errors, dtype=float)
    if len(e) == 0:
        return 0.0
    q = float(np.clip(1.0 - target_far, 0.0, 1.0)) * 100
    return float(np.percentile(e, q))


def far_for_threshold(normal_errors: np.ndarray, threshold: float) -> float:
    """임계값이 주어졌을 때 정상 데이터에서의 오경보율."""
    e = np.asarray(normal_errors, dtype=float)
    return float((e >= threshold).mean()) if len(e) else 0.0


def cost_optimal_threshold(y_true: np.ndarray, errors: np.ndarray,
                           cost_fn: float = 10.0, cost_fp: float = 1.0) -> Dict[str, Any]:
    """미탐(FN)·오탐(FP) 비용을 반영해 총비용을 최소화하는 임계값을 찾는다.

    cost_fn: 이상을 놓쳤을 때 비용(설비 손상·다운타임), cost_fp: 헛알람 비용(점검 공수).
    반환: 최적 임계값과 그때의 FN/FP/총비용, 그리고 임계값별 비용 곡선.
    """
    y = np.asarray(y_true).astype(int)
    e = np.asarray(errors, dtype=float)
    if len(np.unique(y)) < 2 or len(e) == 0:
        return {}
    # 후보 임계값: 오차 분위수 200개
    cands = np.unique(np.percentile(e, np.linspace(0, 100, 200)))
    rows = []
    for t in cands:
        pred = (e >= t).astype(int)
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        rows.append((float(t), fn, fp, fn * float(cost_fn) + fp * float(cost_fp)))
    curve = np.array([[r[0], r[3]] for r in rows], dtype=float)
    best = min(rows, key=lambda r: r[3])
    return {"threshold": best[0], "fn": best[1], "fp": best[2], "total_cost": best[3],
            "curve_thresholds": curve[:, 0].tolist(), "curve_costs": curve[:, 1].tolist(),
            "cost_fn": float(cost_fn), "cost_fp": float(cost_fp)}


def compute_threshold(errors: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """정상 데이터 재구성 오차 분포로부터 임계값을 산출한다.

    method:
      - percentile: 지정 백분위수(기본 99)
      - mean_std:   mean + k*std
    """
    t_cfg = cfg["threshold"]
    method = t_cfg.get("method", "percentile")
    if method == "percentile":
        p = float(t_cfg.get("percentile", 99.0))
        value = float(np.percentile(errors, p))
        detail = {"method": method, "percentile": p}
    elif method == "mean_std":
        k = float(t_cfg.get("k_std", 3.0))
        value = float(errors.mean() + k * errors.std())
        detail = {"method": method, "k_std": k}
    else:
        raise ValueError(f"알 수 없는 threshold.method: {method}")

    detail.update({
        "value": value,
        "train_error_mean": float(errors.mean()),
        "train_error_std": float(errors.std()),
        "train_error_max": float(errors.max()),
    })
    return detail
