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
