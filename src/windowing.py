"""시계열 윈도잉 유틸 (LSTM-AE용).

행 단위 배열을 슬라이딩 윈도우 시퀀스로 변환하고, 윈도우 단위 결과를
원래 행 인덱스에 매핑한다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_windows(X: np.ndarray, window: int) -> np.ndarray:
    """(n, F) → (n-window+1, window, F) 슬라이딩 윈도우(stride=1)."""
    n = X.shape[0]
    if n < window:
        raise ValueError(f"데이터 행 수({n})가 윈도우 크기({window})보다 작습니다.")
    idx = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    return X[idx]


def map_windows_to_rows(win_values: np.ndarray, n_rows: int, window: int) -> np.ndarray:
    """윈도우 단위 값(길이 n-window+1)을 행 단위(길이 n)로 매핑.

    k번째 윈도우는 행 (window-1+k)에서 끝나므로 그 행에 값을 할당한다.
    앞쪽 window-1개 행은 첫 윈도우 값으로 채운다(경계 처리).
    """
    win_values = np.asarray(win_values)
    if win_values.ndim == 1:
        out = np.empty(n_rows, dtype=float)
        out[window - 1:] = win_values
        out[: window - 1] = win_values[0]
    else:  # (n_windows, F) → (n_rows, F)
        f = win_values.shape[1]
        out = np.empty((n_rows, f), dtype=float)
        out[window - 1:] = win_values
        out[: window - 1] = win_values[0]
    return out
