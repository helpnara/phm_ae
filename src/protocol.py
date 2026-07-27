"""평가 프로토콜 — 시간 기반 분할 · 에피소드 검출률 · 탐지 지연.

샘플 단위 F1만으로는 PHM의 실제 유용성을 알기 어렵다. 현장에서 중요한 것은
"이상 구간(에피소드)을 놓치지 않고, 얼마나 빨리 잡는가"이므로 아래를 함께 본다.

- time_based_split : 과거로 학습 → 미래로 평가(운영과 같은 순서). 무작위 분할은 낙관적.
- episodes         : 연속된 이상 구간(에피소드) 추출
- episode_metrics  : 에피소드 검출률 + 탐지 지연(샘플/시간)
- per_type_recall  : 이상 유형별 검출률(유형 컬럼이 있을 때)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def time_based_split(df: pd.DataFrame, timestamp_col: Optional[str],
                     test_frac: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시간 순서로 앞부분(과거)=학습, 뒷부분(미래)=평가로 나눈다.

    타임스탬프가 없으면 행 순서를 시간 순으로 간주한다.
    """
    work = df
    if timestamp_col and timestamp_col in df.columns:
        t = pd.to_datetime(df[timestamp_col], errors="coerce")
        if t.notna().sum() > 1:
            work = df.assign(_t=t).sort_values("_t").drop(columns=["_t"])
    n = len(work)
    n_test = max(int(n * float(test_frac)), 1)
    n_train = n - n_test
    if n_train <= 0:
        raise ValueError("학습 구간이 비었습니다. test_frac을 줄이세요.")
    return work.iloc[:n_train].reset_index(drop=True), work.iloc[n_train:].reset_index(drop=True)


def episodes(y: np.ndarray) -> List[Tuple[int, int]]:
    """연속된 이상(label=1) 구간을 (시작, 끝) 인덱스 목록으로 반환한다."""
    y = np.asarray(y).astype(int)
    out: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(y):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(y) - 1))
    return out


def episode_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
    """에피소드 단위 검출률과 탐지 지연을 계산한다.

    - detected: 에피소드 내에서 한 번이라도 이상으로 판정되면 '검출'
    - latency : 에피소드 시작부터 첫 검출까지 걸린 샘플 수(및 시간)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    eps = episodes(y_true)
    t = None
    if timestamps is not None:
        tt = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="coerce")
        if tt.notna().sum() > 1:
            t = tt

    rows = []
    for i, (s, e) in enumerate(eps, start=1):
        seg = y_pred[s:e + 1]
        hit = np.where(seg == 1)[0]
        detected = len(hit) > 0
        lat = int(hit[0]) if detected else None
        row = {"에피소드": i, "시작": s, "끝": e, "길이": e - s + 1,
               "검출": "✅" if detected else "❌ 미검출",
               "지연(샘플)": lat if detected else None,
               "구간내 검출율(%)": round(float(seg.mean()) * 100, 1)}
        if t is not None and detected:
            row["지연(시간)"] = str(t.iloc[s + lat] - t.iloc[s])
        rows.append(row)

    frame = pd.DataFrame(rows)
    n_ep = len(eps)
    n_det = int((frame["검출"] == "✅").sum()) if n_ep else 0
    lats = [r["지연(샘플)"] for r in rows if r["지연(샘플)"] is not None]
    # 정상 구간에서 발생한 오경보(샘플 수)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    n_normal = int((y_true == 0).sum())
    return {
        "frame": frame,
        "n_episodes": n_ep,
        "n_detected": n_det,
        "episode_recall": (n_det / n_ep) if n_ep else 0.0,
        "median_latency": float(np.median(lats)) if lats else None,
        "max_latency": float(np.max(lats)) if lats else None,
        "false_alarm_rate": (fp / n_normal) if n_normal else 0.0,
        "n_false_alarms": fp,
    }


def per_type_recall(y_true: np.ndarray, y_pred: np.ndarray,
                    fault_type: pd.Series) -> pd.DataFrame:
    """이상 유형별 검출률(재현율). fault_type이 비어있는 정상 행은 제외한다."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    ft = pd.Series(fault_type).reset_index(drop=True).fillna("")
    rows = []
    for name in [v for v in ft.unique() if str(v) not in ("", "nan", "정상")]:
        m = (ft == name).to_numpy() & (y_true == 1)
        if m.sum() == 0:
            continue
        rows.append({"이상 유형": name, "건수": int(m.sum()),
                     "검출": int(y_pred[m].sum()),
                     "검출률(%)": round(float(y_pred[m].mean()) * 100, 1)})
    return pd.DataFrame(rows).sort_values("검출률(%)") if rows else pd.DataFrame()
