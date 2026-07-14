"""테스트용 합성 데이터 생성기.

실제 CSV가 없을 때 파이프라인 검증용 데이터를 만든다.
- normal.csv: 정상 데이터만 (학습용). timestamp 포함, label 없음.
- test.csv:   정상 + 이상 혼합 (평가용). timestamp, label(0/1) 포함.

가상의 설비 센서 5종을 상관관계 있게 생성하고, 이상은 평균 이동/스파이크로 주입한다.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]

# 정상 운전 평균/표준편차 (센서별 물리적으로 그럴듯한 범위)
NORMAL_MEAN = np.array([75.0, 2.0, 5.0, 1500.0, 12.0])
NORMAL_STD = np.array([3.0, 0.3, 0.4, 40.0, 0.8])


def _correlated_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """센서 간 약한 상관을 가진 정상 데이터 생성."""
    base = rng.standard_normal((n, 1))  # 공통 부하 요인
    noise = rng.standard_normal((n, len(SENSORS)))
    corr = 0.5 * base + 0.5 * noise
    return NORMAL_MEAN + NORMAL_STD * corr


def generate(n_normal: int, n_test: int, anomaly_ratio: float, out_dir: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 학습용 정상 데이터 ----
    normal = _correlated_normal(n_normal, rng)
    ts = pd.date_range("2026-01-01", periods=n_normal, freq="min")
    df_normal = pd.DataFrame(normal, columns=SENSORS)
    df_normal.insert(0, "timestamp", ts)
    normal_path = os.path.join(out_dir, "normal.csv")
    df_normal.to_csv(normal_path, index=False)
    print(f"[gen] 정상 학습 데이터: {normal_path} ({n_normal} rows)")

    # ---- 평가용 테스트 데이터 (정상 + 이상) ----
    n_anom = int(n_test * anomaly_ratio)
    n_norm = n_test - n_anom

    test_normal = _correlated_normal(n_norm, rng)

    # 이상 유형: (a) 평균 이동, (b) 스파이크
    anom = _correlated_normal(n_anom, rng)
    for i in range(n_anom):
        if i % 2 == 0:
            # 평균 이동: 온도/진동 상승
            anom[i, 0] += rng.uniform(15, 25)   # temperature
            anom[i, 1] += rng.uniform(1.5, 3.0)  # vibration
        else:
            # 스파이크: 무작위 센서 급변
            j = rng.integers(0, len(SENSORS))
            anom[i, j] += rng.choice([-1, 1]) * NORMAL_STD[j] * rng.uniform(6, 10)

    X_test = np.vstack([test_normal, anom])
    y_test = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])

    # 셔플
    perm = rng.permutation(n_test)
    X_test, y_test = X_test[perm], y_test[perm]

    ts_test = pd.date_range("2026-02-01", periods=n_test, freq="min")
    df_test = pd.DataFrame(X_test, columns=SENSORS)
    df_test.insert(0, "timestamp", ts_test)
    df_test["label"] = y_test
    test_path = os.path.join(out_dir, "test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"[gen] 테스트 데이터: {test_path} ({n_test} rows, 이상 {n_anom}개)")


def generate_timeseries(n: int, anomaly_ratio: float, out_path: str, seed: int) -> None:
    """시간 의존성(AR(1) 자기상관)이 강한 시계열 데이터를 생성한다(라벨 포함).

    이상은 여러 구간의 '지속적 수준 이동'(contextual anomaly)으로 주입한다 →
    시계열 윈도우 모델(LSTM-AE)이 행 단위 모델보다 유리한 데이터.
    """
    rng = np.random.default_rng(seed)
    phi = np.array([0.92, 0.90, 0.88, 0.90, 0.85])  # 자기상관 계수(1에 가까울수록 강함)

    def ar1(m: int) -> np.ndarray:
        x = np.zeros((m, len(SENSORS)))
        x[0] = NORMAL_MEAN
        eps = rng.standard_normal((m, len(SENSORS))) * NORMAL_STD * np.sqrt(1 - phi ** 2)
        for t in range(1, m):
            x[t] = NORMAL_MEAN + phi * (x[t - 1] - NORMAL_MEAN) + eps[t]
        return x

    X = ar1(n)
    y = np.zeros(n, dtype=int)

    # 이상 구간 주입(지속적 수준 이동)
    target = int(n * anomaly_ratio)
    injected = 0
    while injected < target:
        L = int(rng.integers(5, 16))
        start = int(rng.integers(0, n - L))
        j = int(rng.integers(0, len(SENSORS)))
        X[start:start + L, j] += rng.choice([-1, 1]) * NORMAL_STD[j] * rng.uniform(4, 7)
        y[start:start + L] = 1
        injected = int(y.sum())

    ts = pd.date_range("2026-03-01", periods=n, freq="min")
    df = pd.DataFrame(X, columns=SENSORS)
    df.insert(0, "timestamp", ts)
    df["label"] = y
    df.to_csv(out_path, index=False)
    print(f"[gen] 시계열 데이터: {out_path} ({n} rows, 이상 {int(y.sum())})")


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 PHM 데이터 생성")
    ap.add_argument("--n-normal", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--anomaly-ratio", type=float, default=0.1)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeseries", metavar="PATH",
                    help="시계열(AR1) 데이터를 지정 경로에 생성")
    ap.add_argument("--n", type=int, default=800, help="--timeseries 행 수")
    args = ap.parse_args()
    if args.timeseries:
        generate_timeseries(args.n, args.anomaly_ratio, args.timeseries, args.seed)
    else:
        generate(args.n_normal, args.n_test, args.anomaly_ratio, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
