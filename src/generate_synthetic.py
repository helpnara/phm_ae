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


def build_tabular_df(n: int, anomaly_ratio: float, seed: int) -> pd.DataFrame:
    """행 단위(정상+이상+label) 라벨 데이터프레임을 메모리에서 생성한다(웹 샘플용)."""
    rng = np.random.default_rng(seed)
    n_anom = int(n * anomaly_ratio)
    n_norm = n - n_anom
    normal = _correlated_normal(n_norm, rng)
    anom = _correlated_normal(n_anom, rng)
    for i in range(n_anom):
        if i % 2 == 0:
            anom[i, 0] += rng.uniform(15, 25)    # temperature
            anom[i, 1] += rng.uniform(1.5, 3.0)  # vibration
        else:
            j = rng.integers(0, len(SENSORS))
            anom[i, j] += rng.choice([-1, 1]) * NORMAL_STD[j] * rng.uniform(6, 10)
    X = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    ts = pd.date_range("2026-02-01", periods=n, freq="min")
    df = pd.DataFrame(X, columns=SENSORS)
    df.insert(0, "timestamp", ts)
    df["label"] = y
    return df


#: 난이도별 이상 크기(정상 σ 배수). 값이 작을수록 정상과 겹쳐 탐지가 어렵다.
DIFFICULTY = {
    "미세 (어려움)": (1.5, 3.0),
    "보통": (3.0, 6.0),
    "뚜렷 (쉬움)": (6.0, 10.0),
}


def build_scenario_df(n: int, anomaly_ratio: float, seed: int,
                      difficulty: str = "보통", n_modes: int = 1,
                      degradation: bool = False, missing_rate: float = 0.0) -> pd.DataFrame:
    """현실적인 PoC용 데이터를 생성한다(난이도·운전모드·열화·결측 조절).

    - difficulty: 이상 크기(σ 배수) — '미세'일수록 정상 분포와 겹쳐 탐지가 어렵다.
    - n_modes: 정상 운전 모드 수(가동/부하 조건 등). 2 이상이면 정상이 다봉 분포가 된다.
    - degradation: 후반부에 점진적 열화(서서히 증가하는 오프셋)를 주입한다.
    - missing_rate: 무작위 결측 비율(센서 결측 상황 재현).
    """
    rng = np.random.default_rng(seed)
    lo, hi = DIFFICULTY.get(difficulty, DIFFICULTY["보통"])
    n_modes = max(int(n_modes), 1)

    # ---- 정상: 운전 모드별로 평균이 다른 다봉 분포 ----
    mode_id = rng.integers(0, n_modes, n)
    # 모드별 오프셋(σ의 ±0~4배 범위에서 결정적으로 배치)
    mode_offsets = np.linspace(-2.0, 2.0, n_modes) if n_modes > 1 else np.array([0.0])
    base = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, len(SENSORS)))
    corr = 0.5 * base + 0.5 * noise
    X = NORMAL_MEAN + NORMAL_STD * (corr + mode_offsets[mode_id][:, None])
    y = np.zeros(n, dtype=int)

    # ---- 점진적 열화: 후반 40% 구간에서 서서히 증가하는 오프셋 ----
    if degradation:
        start = int(n * 0.6)
        ramp = np.zeros(n)
        ramp[start:] = np.linspace(0, hi * 0.7, n - start)
        X[:, 0] += NORMAL_STD[0] * ramp          # temperature 서서히 상승
        X[:, 1] += NORMAL_STD[1] * ramp * 0.8    # vibration 동반 상승

    # ---- 이상 주입: 난이도에 따른 크기 · 유형 라벨 기록 ----
    ftype = np.array([""] * n, dtype=object)
    n_anom = int(n * anomaly_ratio)
    if n_anom > 0:
        idx = rng.choice(n, n_anom, replace=False)
        for i, r in enumerate(idx):
            mag = rng.uniform(lo, hi)
            if i % 3 == 0:      # 단일 센서 스파이크
                j = int(rng.integers(0, len(SENSORS)))
                X[r, j] += rng.choice([-1, 1]) * NORMAL_STD[j] * mag
                ftype[r] = "스파이크"
            elif i % 3 == 1:    # 온도·진동 동반 상승(과열)
                X[r, 0] += NORMAL_STD[0] * mag
                X[r, 1] += NORMAL_STD[1] * mag * 0.9
                ftype[r] = "과열(온도·진동)"
            else:               # 상관 구조 붕괴(rpm↑ current↓)
                X[r, 3] += NORMAL_STD[3] * mag * 0.8
                X[r, 4] -= NORMAL_STD[4] * mag * 0.8
                ftype[r] = "상관구조 붕괴"
            y[r] = 1

    ts = pd.date_range("2026-04-01", periods=n, freq="min")
    df = pd.DataFrame(X, columns=SENSORS)
    df.insert(0, "timestamp", ts)
    if n_modes > 1:
        df["mode"] = mode_id
    df["label"] = y
    df["fault_type"] = ftype

    # ---- 결측 주입 ----
    if missing_rate > 0:
        for c in SENSORS:
            m = rng.random(n) < missing_rate
            df.loc[m, c] = np.nan
    return df


def build_timeseries_df(n: int, anomaly_ratio: float, seed: int) -> pd.DataFrame:
    """시계열(AR(1) 자기상관 강함) 라벨 데이터프레임을 생성한다(LSTM 데모용)."""
    rng = np.random.default_rng(seed)
    phi = np.array([0.92, 0.90, 0.88, 0.90, 0.85])

    def ar1(m: int) -> np.ndarray:
        x = np.zeros((m, len(SENSORS)))
        x[0] = NORMAL_MEAN
        eps = rng.standard_normal((m, len(SENSORS))) * NORMAL_STD * np.sqrt(1 - phi ** 2)
        for t in range(1, m):
            x[t] = NORMAL_MEAN + phi * (x[t - 1] - NORMAL_MEAN) + eps[t]
        return x

    X = ar1(n)
    y = np.zeros(n, dtype=int)
    target = int(n * anomaly_ratio)
    injected = 0
    while injected < target:
        L = int(rng.integers(5, 16))
        start = int(rng.integers(0, max(n - L, 1)))
        j = int(rng.integers(0, len(SENSORS)))
        X[start:start + L, j] += rng.choice([-1, 1]) * NORMAL_STD[j] * rng.uniform(4, 7)
        y[start:start + L] = 1
        injected = int(y.sum())
    ts = pd.date_range("2026-03-01", periods=n, freq="min")
    df = pd.DataFrame(X, columns=SENSORS)
    df.insert(0, "timestamp", ts)
    df["label"] = y
    return df


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
