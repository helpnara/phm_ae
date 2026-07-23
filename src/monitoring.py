"""성능 모니터링 계산 유틸.

운영(신규) 데이터의 재구성 오차를 시간/구간별로 집계하고, 학습 기준(baseline)
대비 분포 이동(드리프트)을 평가한다. 라벨이 필요 없다(비지도 드리프트).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


_FREQ_FMT = {"h": "%m-%d %H시", "d": "%m-%d", "w": "%m-%d주"}


def windowed_metrics(errors: np.ndarray, predictions: np.ndarray,
                     timestamps: Optional[pd.Series] = None,
                     n_windows: int = 30, freq: Optional[str] = None) -> pd.DataFrame:
    """구간별 평균 오차·이상률·건수를 집계한다.

    freq(h/d/w)가 주어지고 timestamps가 유효하면 시간 주기로 리샘플하고,
    아니면 timestamps 유효 시 등간격 n_windows개, 그것도 없으면 순서 청크로 나눈다.
    반환 컬럼: label, mean_error, anomaly_rate(0~1), count
    """
    df = pd.DataFrame({"error": np.asarray(errors, dtype=float),
                       "pred": np.asarray(predictions, dtype=int)})
    t = None
    if timestamps is not None:
        tt = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="coerce")
        if tt.notna().sum() > 1 and tt.nunique() > 1:
            t = tt

    n = len(df)
    if t is not None and freq in _FREQ_FMT:
        df["t"] = t.values
        g = df.set_index("t").resample(freq).agg(
            mean_error=("error", "mean"), anomaly_rate=("pred", "mean"), count=("error", "size"))
        agg = g[g["count"] > 0].reset_index()
        agg["label"] = agg["t"].dt.strftime(_FREQ_FMT[freq])
        return agg[["label", "mean_error", "anomaly_rate", "count"]]

    bins = max(min(int(n_windows), n), 1)
    if t is not None:
        df["t"] = t.values
        df = df.sort_values("t").reset_index(drop=True)
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


def baseline_from_errors(errors: np.ndarray) -> Dict[str, float]:
    """기준 데이터의 재구성 오차로부터 baseline(평균·표준편차)을 산출한다."""
    e = np.asarray(errors, dtype=float)
    return {"mean": float(e.mean()) if len(e) else 0.0,
            "std": float(e.std()) if len(e) else 0.0}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. expected=baseline, actual=현재. 클수록 분포 이동 큼."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if len(e) == 0 or len(a) == 0:
        return 0.0
    qs = np.quantile(e, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    e_perc = np.histogram(e, qs)[0] / len(e)
    a_perc = np.histogram(a, qs)[0] / len(a)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def ks_stat(expected: np.ndarray, actual: np.ndarray) -> tuple:
    """두 표본 KS 검정. (통계량, p-value). p<유의수준이면 분포 상이."""
    from scipy.stats import ks_2samp
    r = ks_2samp(np.asarray(expected, dtype=float), np.asarray(actual, dtype=float))
    return float(r.statistic), float(r.pvalue)


def assess_drift(method: str, cur_errors: np.ndarray, base_mean: Optional[float],
                 base_std: Optional[float], baseline_errors: Optional[np.ndarray] = None,
                 k: float = 3.0) -> Dict[str, Any]:
    """선택한 방식으로 드리프트를 평가한다.

    method: 'zscore' | 'psi' | 'ks'. PSI/KS는 baseline_errors(기준 분포)가 필요하다.
    반환: {method, metric, value, status, text, unavailable?}
    """
    cur = np.asarray(cur_errors, dtype=float)
    if method == "psi":
        if baseline_errors is None or len(baseline_errors) == 0:
            return {"unavailable": True}
        v = psi(baseline_errors, cur)
        status = "경고" if v >= 0.25 else ("주의" if v >= 0.1 else "정상")
        return {"method": "PSI", "metric": "PSI", "value": v, "status": status,
                "text": f"PSI={v:.3f} (0.1↑ 주의 · 0.25↑ 경고)"}
    if method == "ks":
        if baseline_errors is None or len(baseline_errors) == 0:
            return {"unavailable": True}
        stat, pval = ks_stat(baseline_errors, cur)
        status = "경고" if pval < 0.05 else ("주의" if pval < 0.1 else "정상")
        return {"method": "KS", "metric": "KS 통계량", "value": stat, "pval": pval,
                "status": status, "text": f"KS={stat:.3f}, p={pval:.3g} (p<0.05 → 분포 상이)"}
    # zscore (기본)
    bm = float(base_mean) if base_mean is not None else float(cur.mean())
    bs = float(base_std) if (base_std is not None and base_std > 0) else 0.0
    cur_mean = float(cur.mean()) if len(cur) else 0.0
    z = (cur_mean - bm) / bs if bs > 0 else 0.0
    status = "경고" if (bs > 0 and cur_mean > bm + k * bs) else (
        "주의" if (bs > 0 and z >= 0.6 * k) else "정상")
    return {"method": "z-score", "metric": "z-점수", "value": z, "status": status,
            "text": f"평균 오차 z={z:.1f}σ"}


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
