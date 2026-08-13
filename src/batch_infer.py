"""배치 추론 (사내 이식용): Oracle에서 데이터 조회 → 판정 → 결과를 Oracle에 적재.

실행 예:
    python -m src.batch_infer \
        --sql "SELECT * FROM phm_sensor WHERE processed = 0" \
        --table PHM_ANOMALY_RESULT

스케줄러(cron/사내 스케줄러/Airflow 등)에 등록해 주기적으로 실행한다.
결과 테이블 스키마는 docs/migration_notes.md §6 참고.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import data as D
from src.config import load_config
from src.detector import AnomalyDetector


def build_result_frame(df: pd.DataFrame, detector: AnomalyDetector,
                       model_version: str) -> pd.DataFrame:
    """판정 결과를 결과 테이블 적재용 DataFrame으로 변환한다."""
    result = detector.predict(df)
    ts_col = detector.schema.timestamp_column
    out = result.summary_frame(df, ts_col)
    out["model_version"] = model_version
    # 원본 식별자 컬럼이 있으면 함께 적재(예: equipment_id)
    for key in ("id", "equipment_id", "eq_id"):
        if key in df.columns:
            out[key] = df[key].values
    return out


def run(sql: str, table: str, config_path: str | None = None,
        if_exists: str = "append") -> int:
    from src import db  # 지연 import (DB 의존성은 선택)

    cfg = load_config(config_path)
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    detector = AnomalyDetector(artifacts_dir)
    model_version = D.load_meta(artifacts_dir).get("created_at", "unknown")

    df = db.read_sql(sql)
    if df.empty:
        print("[batch] 조회 결과 없음. 종료.")
        return 0

    out = build_result_frame(df, detector, model_version)
    n = db.write_results(out, table, if_exists=if_exists)
    n_anom = int(out["prediction"].sum())
    print(f"[batch] {n}건 판정·적재 완료 (이상 {n_anom}건) → {table}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle 배치 이상탐지 추론")
    ap.add_argument("--sql", required=True, help="추론 대상 데이터 조회 SQL")
    ap.add_argument("--table", required=True, help="결과 적재 테이블명")
    ap.add_argument("--config", default=None, help="설정 YAML 경로")
    args = ap.parse_args()
    run(args.sql, args.table, args.config)


if __name__ == "__main__":
    main()
