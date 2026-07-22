"""성능 모니터링 계산 유틸.

운영(신규) 데이터의 재구성 오차를 시간/구간별로 집계하고, 학습 기준(baseline)
대비 분포 이동(드리프트)을 평가한다. 라벨이 필요 없다(비지도 드리프트).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def windowed_metrics(errors: np.ndarray, predictions: np.ndarray,
                     timestamps: Optional[pd.Series] = None,
                     n_windows: int = 30) -> pd.DataFrame:
    """구간별 평균 오차·이상률·건수를 집계한다.

    timestamps가 유효하면 시간 범위를 n_windows개 등간격으로 나누고,
    아니면 행 순서를 n_windows개 균등 청크로 나눈다.
    반환 컬럼: label, mean_error, anomaly_rate(0~1), count
    """
    df = pd.DataFrame({"error": np.asarray(errors, dtype=float),
                       "pred": np.asarray(predictions, dtype=int)})
    use_time = False
    if timestamps is not None:
        t = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="coerce")
        if t.notna().sum() > 1 and t.nunique() > 1:
            df["t"] = t.values
            df = df.sort_values("t").reset_index(drop=True)
            use_time = True

    n = len(df)
    bins = max(min(int(n_windows), n), 1)
    if use_time:
        ticks = df["t"].astype("int64")
        df["win"] = pd.cut(ticks, bins=bins, labels=False, include_lowest=True)
        agg = df.groupby("win", dropna=True).agg(
            mean_error=("error", "mean"), anomaly_rate=("pred", "mean"),
            count=("error", "size"), t=("t", "first")).reset_index()
        agg["label"] = pd.to_datetime(agg["t"]).dt.strftime("%m-%d %H:%M")
    else:
        df["win"] = (np.arange(n) * bins // max(n, 1))
        agg = df.groupby("win").agg(
            mean_error=("error", "mean"), anomaly_rate=("pred", "mean"),
            count=("error", "size")).reset_index()
        agg["label"] = "구간 " + (agg["win"] + 1).astype(str)
    return agg[["label", "mean_error", "anomaly_rate", "count"]]


def drift_summary(errors: np.ndarray, base_mean: Optional[float],
                  base_std: Optional[float], k: float = 3.0) -> Dict[str, Any]:
    """현재 오차 분포가 학습 기준 대비 얼마나 이동했는지 요약한다.

    z = (현재 평균 - 학습 평균) / 학습 표준편차.
    상태: 경고(현재 평균 > 학습평균 + kσ) / 주의(z ≥ 0.6k) / 정상.
    """
    cur_mean = float(np.mean(errors)) if len(errors) else 0.0
    bm = float(base_mean) if base_mean is not None else cur_mean
    bs = float(base_std) if (base_std is not None and base_std > 0) else 0.0
    z = (cur_mean - bm) / bs if bs > 0 else 0.0
    drift_line = bm + k * bs if bs > 0 else bm
    if bs > 0 and cur_mean > drift_line:
        status = "경고"
    elif bs > 0 and z >= 0.6 * k:
        status = "주의"
    else:
        status = "정상"
    return {"cur_mean": cur_mean, "base_mean": bm, "base_std": bs,
            "z": z, "drift_line": drift_line, "status": status}
