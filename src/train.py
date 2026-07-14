"""오토인코더 학습 스크립트.

사용법:
    python -m src.train --data data/normal.csv --config config/default.yaml

정상 데이터만으로 AE를 학습하고, 재구성 오차 임계값을 산출한 뒤
아티팩트(모델/스케일러/스키마/임계값/메타)를 artifacts/에 저장한다.
"""
from __future__ import annotations

import argparse
import os
import platform
from datetime import datetime, timezone

import numpy as np

from src import data as D
from src.config import load_config
from src.model import build_autoencoder
from src.scoring import compute_threshold, reconstruction_error


def set_seeds(seed: int) -> None:
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_from_dataframe(df, cfg: dict, artifacts_dir: str | None = None,
                         extra_callbacks: list | None = None, verbose: int = 2,
                         data_source: str = "dataframe",
                         model_type: str = "dense", window: int | None = None) -> str:
    """DataFrame(정상 데이터)으로 AE를 학습하고 아티팩트를 저장한다.

    model_type: "dense"(행 단위) 또는 "lstm"(시계열 윈도우). lstm이면 window 필요.
    CLI와 웹앱이 공통으로 사용한다.
    """
    import tensorflow as tf
    from tensorflow import keras
    from src.model import build_lstm_autoencoder
    from src.scoring import seq_window_errors
    from src.windowing import make_windows

    seed = int(cfg.get("seed", 42))
    set_seeds(seed)
    t_cfg = cfg["train"]

    # 1) 전처리(스케일러/스키마 fit) — 정상 데이터에서만
    X, schema, scaler = D.fit_preprocess(df, cfg)
    n_features = X.shape[1]
    if verbose:
        print(f"[train] 샘플 {X.shape[0]}개, 피처 {n_features}개, model={model_type}")

    # 2~4) 모델 구성/학습/임계값 — dense vs lstm
    if model_type == "lstm":
        window = int(window or 20)
        Xw = make_windows(X, window)                     # (nw, W, F)
        model = build_lstm_autoencoder(n_features, window, cfg)
        fit_x = Xw
    else:
        model = build_autoencoder(n_features, cfg)
        fit_x = X

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(t_cfg["learning_rate"])),
        loss="mse",
    )
    if verbose:
        model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(t_cfg["early_stopping_patience"]),
            restore_best_weights=True,
        )
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    history = model.fit(
        fit_x, fit_x,
        validation_split=float(t_cfg["validation_split"]),
        epochs=int(t_cfg["epochs"]),
        batch_size=int(t_cfg["batch_size"]),
        callbacks=callbacks,
        verbose=verbose,
    )

    # 임계값 산출(정상 데이터 재구성 오차 분포)
    if model_type == "lstm":
        Xw_hat = model.predict(fit_x, verbose=0)
        errors, _ = seq_window_errors(fit_x, Xw_hat)
    else:
        X_hat = model.predict(X, verbose=0)
        errors = reconstruction_error(X, X_hat)
    threshold = compute_threshold(errors, cfg)
    if verbose:
        print(f"[train] 임계값({threshold['method']}) = {threshold['value']:.6f}")

    # 5) 아티팩트 저장
    artifacts_dir = artifacts_dir or cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    model.save(os.path.join(artifacts_dir, D.MODEL_FILE))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "model_type": model_type,
        "window": int(window) if model_type == "lstm" else None,
        "n_samples": int(X.shape[0]),
        "n_features": int(n_features),
        "seed": seed,
        "config": cfg,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [float("nan")])[-1]),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
    }
    D.save_artifacts(artifacts_dir, schema, scaler, threshold, meta)

    # 학습 이력 저장(에폭별 손실곡선 시각화용)
    import json
    with open(os.path.join(artifacts_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        }, f)

    if verbose:
        print(f"[train] 아티팩트 저장 완료: {os.path.abspath(artifacts_dir)}")
    return artifacts_dir


def train(data_path: str | None = None, config_path: str | None = None,
          sql: str | None = None) -> str:
    cfg = load_config(config_path)
    # CSV(--data) 또는 Oracle 조회(--sql) 중 하나. 정상 데이터만 사용해야 함.
    df = D.load_dataframe(csv_path=data_path, sql=sql)
    source = os.path.abspath(data_path) if data_path else f"SQL: {sql}"
    return train_from_dataframe(df, cfg, data_source=source, verbose=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="PHM 오토인코더 학습")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="정상 데이터 CSV 경로")
    src.add_argument("--sql", help="정상 데이터 조회 SQL (Oracle). --data 대신 사용")
    ap.add_argument("--config", default=None, help="설정 YAML 경로")
    args = ap.parse_args()
    train(data_path=args.data, config_path=args.config, sql=args.sql)


if __name__ == "__main__":
    main()
