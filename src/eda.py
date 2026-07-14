"""탐색적 데이터 분석(EDA) 계산 유틸.

streamlit/plotly에 의존하지 않는 순수 pandas/numpy 함수 모음.
앱(app/app.py)이 이 결과를 받아 표·차트로 렌더링한다.
불균형·결측·상수컬럼·이상치 등 이상탐지 품질에 직결되는 항목을 우선 제공한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def basic_info(df: pd.DataFrame, feature_cols: List[str],
               label_col: Optional[str], timestamp_col: Optional[str]) -> Dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_features": int(len(feature_cols)),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "n_duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "has_label": bool(label_col) and label_col in df.columns,
        "has_timestamp": bool(timestamp_col) and timestamp_col in df.columns,
    }


def dtypes_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
    })


def missing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """결측이 있는 컬럼만 반환(count, %)."""
    m = df.isna().sum()
    out = pd.DataFrame({
        "column": df.columns,
        "missing": m.values.astype(int),
        "missing_pct": (m.values / max(len(df), 1) * 100).round(2),
    })
    return out[out["missing"] > 0].sort_values("missing", ascending=False).reset_index(drop=True)


def describe_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """수치형 피처 기초 통계량(전치)."""
    if not feature_cols:
        return pd.DataFrame()
    d = df[feature_cols].describe().T
    d.insert(0, "feature", d.index)
    # 왜도/첨도 추가(분포 형태 파악)
    d["skew"] = [float(df[c].skew()) for c in feature_cols]
    d["kurtosis"] = [float(df[c].kurtosis()) for c in feature_cols]
    return d.reset_index(drop=True)


def constant_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """분산 0(상수) 컬럼. 학습 시 제거되므로 사전 경고용."""
    return [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]


def outlier_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """IQR 기준 이상치 개수/비율(피처별). 데이터 품질·오염 점검."""
    rows = []
    for c in feature_cols:
        s = df[c].dropna()
        if len(s) == 0:
            rows.append({"feature": c, "outliers": 0, "outlier_pct": 0.0})
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "feature": c,
            "outliers": n_out,
            "outlier_pct": round(n_out / len(s) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def label_balance(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """클래스 분포(불균형) 정보. 정확도 평가가 부적절함을 뒷받침."""
    vc = df[label_col].value_counts(dropna=False)
    total = int(len(df))
    frame = pd.DataFrame({
        "label": [str(k) for k in vc.index],
        "count": vc.values.astype(int),
        "pct": (vc.values / max(total, 1) * 100).round(2),
    })
    n_normal = int((df[label_col] == 0).sum())
    n_anom = int((df[label_col] == 1).sum())
    ratio = (n_normal / n_anom) if n_anom > 0 else float("inf")
    return {
        "frame": frame,
        "n_normal": n_normal,
        "n_anomaly": n_anom,
        "imbalance_ratio": ratio,  # 정상:이상 = ratio:1
    }


def correlation(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    if len(feature_cols) < 2:
        return pd.DataFrame()
    return df[feature_cols].corr()


def contamination_warnings(df: pd.DataFrame, feature_cols: List[str],
                           label_col: Optional[str], pct_threshold: float = 8.0
                           ) -> List[str]:
    """학습에 쓸 '정상' 데이터의 오염 가능성을 점검한다.

    정상(label=0, 없으면 전체)에서 IQR 이상치 비율이 임계(기본 8%)를 넘는 피처를
    경고로 반환한다. 정상 순수성 원칙(학습 데이터에 이상 혼입 방지)을 지원.
    """
    if label_col and label_col in df.columns:
        normal = df[df[label_col] == 0]
    else:
        normal = df
    warns: List[str] = []
    for c in feature_cols:
        s = normal[c].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct = ((s < lo) | (s > hi)).mean() * 100
        if pct >= pct_threshold:
            warns.append(f"{c}: 정상 데이터 내 이상치 {pct:.1f}%")
    return warns


def sample_for_plot(df: pd.DataFrame, max_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """대용량 대비 시각화용 다운샘플(통계량은 전체 사용, 플롯만 샘플)."""
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=seed).sort_index()
