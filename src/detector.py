"""추론용 이상탐지기. 저장된 아티팩트를 로드해 CSV/DataFrame을 판정한다.

CLI와 Streamlit 웹앱에서 공통으로 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src import data as D
from src.scoring import per_feature_squared_error, reconstruction_error


@dataclass
class DetectionResult:
    errors: np.ndarray               # 샘플별 재구성 오차 (n,)
    predictions: np.ndarray          # 0=정상, 1=이상 (n,)
    per_feature_error: np.ndarray    # 피처별 제곱오차 (n, n_features)
    feature_columns: List[str]
    threshold: float
    threshold_detail: Dict[str, Any]

    def summary_frame(self, df: pd.DataFrame, timestamp_column: Optional[str]) -> pd.DataFrame:
        """샘플별 결과 표를 구성한다."""
        out = pd.DataFrame()
        if timestamp_column and timestamp_column in df.columns:
            out[timestamp_column] = df[timestamp_column].values
        out["reconstruction_error"] = self.errors
        out["threshold"] = self.threshold
        out["anomaly_score"] = self.errors / self.threshold  # 1.0 초과 = 이상
        out["prediction"] = self.predictions
        out["status"] = np.where(self.predictions == 1, "이상(anomaly)", "정상(normal)")
        # 최다 기여 피처(원인 진단)
        top_idx = self.per_feature_error.argmax(axis=1)
        out["top_feature"] = [self.feature_columns[i] for i in top_idx]
        return out

    def top_features(self, k: int = 5) -> pd.DataFrame:
        """이상 샘플들에서 평균 기여도가 높은 피처 상위 k개."""
        mask = self.predictions == 1
        if mask.sum() == 0:
            base = self.per_feature_error.mean(axis=0)
        else:
            base = self.per_feature_error[mask].mean(axis=0)
        order = np.argsort(base)[::-1][:k]
        return pd.DataFrame({
            "feature": [self.feature_columns[i] for i in order],
            "mean_squared_error": base[order],
        })


class AnomalyDetector:
    """아티팩트 로드 + 추론."""

    def __init__(self, artifacts_dir: str):
        # 지연 import: TF 로딩 비용을 detector 생성 시로 한정
        from tensorflow import keras
        import os

        self.artifacts_dir = artifacts_dir
        self.schema = D.load_schema(artifacts_dir)
        self.scaler = D.load_scaler(artifacts_dir)
        self.threshold_detail = D.load_threshold(artifacts_dir)
        self.threshold = float(self.threshold_detail["value"])
        self.model = keras.models.load_model(os.path.join(artifacts_dir, D.MODEL_FILE))

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> DetectionResult:
        """DataFrame을 판정한다. threshold를 주면 저장값 대신 사용(인터랙티브)."""
        X = D.validate_and_transform(df, self.schema, self.scaler)
        X_hat = self.model.predict(X, verbose=0)
        errors = reconstruction_error(X, X_hat)
        pfe = per_feature_squared_error(X, X_hat)
        thr = float(threshold) if threshold is not None else self.threshold
        preds = (errors >= thr).astype(int)
        return DetectionResult(
            errors=errors,
            predictions=preds,
            per_feature_error=pfe,
            feature_columns=self.schema.feature_columns,
            threshold=thr,
            threshold_detail=self.threshold_detail,
        )
