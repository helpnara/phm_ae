"""데이터 로딩, 스키마 검증, 전처리, 아티팩트 입출력.

핵심 설계 원칙(CLAUDE.md):
- 스케일러는 정상 학습 데이터에서만 fit → 저장. 추론 시 transform만.
- 학습 시 피처 스키마를 저장하고 추론 시 업로드 CSV를 이와 대조한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCHEMA_FILE = "schema.json"
SCALER_FILE = "scaler.pkl"
THRESHOLD_FILE = "threshold.json"
META_FILE = "meta.json"
MODEL_FILE = "model.keras"


@dataclass
class Schema:
    """학습 시점의 피처 스키마. 추론 시 검증 기준이 된다."""
    feature_columns: List[str]
    label_column: Optional[str]
    timestamp_column: Optional[str]
    impute_strategy: str
    impute_values: Dict[str, float]  # median/mean 대치값(피처별)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Schema":
        return Schema(**d)


# ----------------------------- 로딩 -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    """CSV를 DataFrame으로 로드한다."""
    return pd.read_csv(path)


def load_dataframe(csv_path: Optional[str] = None, sql: Optional[str] = None) -> pd.DataFrame:
    """데이터 소스 추상화: CSV 경로 또는 Oracle SQL 중 하나로 DataFrame을 얻는다.

    사내 이식 시 CSV 대신 sql을 넘기면 DB에서 조회한다(src/db.py 사용).
    이후 전처리/모델/추론 코드는 DataFrame 형태만 맞으면 수정 불필요.
    """
    if csv_path and sql:
        raise ValueError("csv_path와 sql 중 하나만 지정하세요.")
    if csv_path:
        return load_csv(csv_path)
    if sql:
        from src.db import read_sql  # 지연 import (DB 의존성은 선택)
        return read_sql(sql)
    raise ValueError("csv_path 또는 sql 중 하나는 반드시 필요합니다.")


def infer_feature_columns(df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
    """제외 컬럼을 뺀 수치형 컬럼을 피처로 추론한다."""
    features = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


# ----------------------------- 전처리 (학습) -----------------------------

def fit_preprocess(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Schema, StandardScaler]:
    """학습용 정상 데이터에 대해 스키마/스케일러를 fit하고 배열을 반환한다."""
    data_cfg = cfg["data"]
    exclude = list(data_cfg.get("exclude_columns", []))
    features = infer_feature_columns(df, exclude)

    if not features:
        raise ValueError("피처로 사용할 수치형 컬럼이 없습니다. exclude_columns 설정을 확인하세요.")

    work = df[features].copy()

    # 결측 대치값 계산(피처별)
    strategy = data_cfg.get("impute_strategy", "median")
    impute_values: Dict[str, float] = {}
    if strategy in ("median", "mean"):
        for col in features:
            val = work[col].median() if strategy == "median" else work[col].mean()
            impute_values[col] = float(val) if pd.notna(val) else 0.0
            work[col] = work[col].fillna(impute_values[col])
    elif strategy == "drop":
        work = work.dropna()
    else:
        raise ValueError(f"알 수 없는 impute_strategy: {strategy}")

    # 상수(분산 0) 컬럼 제거
    if data_cfg.get("drop_constant_columns", True):
        non_constant = [c for c in features if work[c].nunique(dropna=True) > 1]
        dropped = sorted(set(features) - set(non_constant))
        if dropped:
            print(f"[data] 상수 컬럼 제거: {dropped}")
        features = non_constant
        work = work[features]
        impute_values = {c: impute_values[c] for c in features if c in impute_values}

    scaler = StandardScaler()
    X = scaler.fit_transform(work.values.astype("float64"))

    schema = Schema(
        feature_columns=features,
        label_column=data_cfg.get("label_column"),
        timestamp_column=data_cfg.get("timestamp_column"),
        impute_strategy=strategy,
        impute_values=impute_values,
    )
    return X, schema, scaler


# ----------------------------- 전처리 (추론) -----------------------------

class SchemaValidationError(ValueError):
    """업로드 CSV가 저장된 스키마와 불일치할 때 발생."""


def validate_and_transform(
    df: pd.DataFrame, schema: Schema, scaler: StandardScaler
) -> np.ndarray:
    """추론 입력 DataFrame을 스키마로 검증하고 스케일러로 변환한다.

    스케일러는 다시 fit하지 않는다(데이터 누수 방지) — transform만 수행.
    """
    missing = [c for c in schema.feature_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"필수 피처 컬럼이 누락되었습니다: {missing}\n"
            f"학습 스키마 피처: {schema.feature_columns}"
        )

    work = df[schema.feature_columns].copy()

    # 수치형 강제 변환(문자열 등은 NaN 처리 후 대치)
    for col in schema.feature_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # 학습 때 계산한 대치값으로 결측 대치(추론에서 통계 재산출 금지)
    if schema.impute_strategy in ("median", "mean"):
        for col in schema.feature_columns:
            work[col] = work[col].fillna(schema.impute_values.get(col, 0.0))
    else:
        work = work.fillna(0.0)

    X = scaler.transform(work.values.astype("float64"))
    return X


# ----------------------------- 아티팩트 입출력 -----------------------------

def save_artifacts(
    artifacts_dir: str,
    schema: Schema,
    scaler: StandardScaler,
    threshold: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, SCHEMA_FILE), "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, ensure_ascii=False, indent=2)
    joblib.dump(scaler, os.path.join(artifacts_dir, SCALER_FILE))
    with open(os.path.join(artifacts_dir, THRESHOLD_FILE), "w", encoding="utf-8") as f:
        json.dump(threshold, f, ensure_ascii=False, indent=2)
    with open(os.path.join(artifacts_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_schema(artifacts_dir: str) -> Schema:
    with open(os.path.join(artifacts_dir, SCHEMA_FILE), "r", encoding="utf-8") as f:
        return Schema.from_dict(json.load(f))


def load_scaler(artifacts_dir: str) -> StandardScaler:
    return joblib.load(os.path.join(artifacts_dir, SCALER_FILE))


def load_threshold(artifacts_dir: str) -> Dict[str, Any]:
    with open(os.path.join(artifacts_dir, THRESHOLD_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(artifacts_dir: str) -> Dict[str, Any]:
    with open(os.path.join(artifacts_dir, META_FILE), "r", encoding="utf-8") as f:
        return json.load(f)
