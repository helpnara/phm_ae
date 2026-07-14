"""PHM 오토인코더 이상탐지 — Streamlit 웹 테스트 앱.

실행:
    streamlit run app/app.py

흐름:
  ① CSV 업로드 → ② 'EDA'로 사전 검토(선택) / '학습' 설정 조정(선택)
  → ③ '학습' 버튼(진행률 표시) → ④ 학습 품질·이상 판정·개별 진단 확인
  - 초기 로딩 시에는 학습하지 않는다(업로드 후 버튼으로만 학습 시작).
  - 라벨(label) 컬럼이 있으면 정상(label=0)만 학습하고 전체를 판정·평가한다.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 프로젝트 루트를 import 경로에 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.detector import AnomalyDetector
from src.data import infer_feature_columns
from src import eda as EDA

st.set_page_config(page_title="PHM 이상탐지", page_icon="🔧", layout="wide")

NORMAL_C = "#2c7fb8"
ANOM_C = "#d7301f"


def _note(text: str) -> None:
    """각 결과에 대한 해석 가이드(5줄 이내)를 눈에 띄게 표시한다."""
    st.markdown(
        f"<div style='background:rgba(44,127,184,0.08);border-left:3px solid {NORMAL_C};"
        f"padding:8px 12px;border-radius:4px;font-size:0.86rem;color:inherit;margin:2px 0 10px'>"
        f"💡 <b>해석 가이드</b><br>{text}</div>",
        unsafe_allow_html=True,
    )


def _reset_if_new_upload(uploaded) -> None:
    """업로드 파일이 바뀌면(또는 제거되면) 이전 학습/EDA 상태를 초기화한다."""
    sig = None if uploaded is None else (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:
        st.session_state["upload_sig"] = sig
        for k in ("detector", "train_note", "show_eda"):
            st.session_state.pop(k, None)


def _zip_dir(path: str) -> bytes:
    """아티팩트 폴더를 zip 바이트로 묶는다(사내 이관용 다운로드)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(path)):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                z.write(fp, arcname=name)
    buf.seek(0)
    return buf.getvalue()


def train_and_store(df: pd.DataFrame, cfg: dict, artifacts_dir: str,
                    label_col: str | None, settings: dict) -> bool:
    """업로드 데이터로 학습하고 진행률을 표시한 뒤 detector를 세션에 저장한다."""
    from tensorflow import keras
    from src.train import train_from_dataframe

    # 설정 UI 값 반영(원본 cfg는 보존)
    cfg = copy.deepcopy(cfg)
    cfg["train"]["epochs"] = int(settings["epochs"])
    cfg["train"]["learning_rate"] = float(settings["lr"])
    cfg["model"]["bottleneck"] = int(settings["bottleneck"])
    cfg["threshold"]["method"] = "percentile"
    cfg["threshold"]["percentile"] = float(settings["percentile"])

    has_label = bool(label_col) and label_col in df.columns
    if has_label:
        train_df = df[df[label_col] == 0]
        note = (f"라벨 감지 → 정상(label=0) {len(train_df):,}건으로 학습, "
                f"전체 {len(df):,}건을 판정·평가")
    else:
        train_df = df
        note = f"라벨 없음 → 업로드 {len(df):,}건을 정상으로 간주해 학습 후 판정"

    if len(train_df) < 20:
        st.error(f"학습에 사용할 정상 데이터가 부족합니다(현재 {len(train_df)}건, 최소 20건 필요).")
        return False

    st.caption(note + f" · 설정: epoch {cfg['train']['epochs']}, 병목 {cfg['model']['bottleneck']}, "
                      f"임계 {cfg['threshold']['percentile']}%")
    epochs = max(int(cfg["train"]["epochs"]), 1)
    bar = st.progress(0.0, text="학습 준비 중...")

    class _Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            frac = min((epoch + 1) / epochs, 1.0)
            logs = logs or {}
            bar.progress(
                frac,
                text=(f"학습 중... {epoch + 1}/{epochs} epoch · "
                      f"loss={logs.get('loss', 0.0):.5f} · "
                      f"val_loss={logs.get('val_loss', 0.0):.5f}"),
            )

    try:
        train_from_dataframe(train_df, cfg, artifacts_dir=artifacts_dir,
                             extra_callbacks=[_Progress()], verbose=0,
                             data_source="streamlit upload")
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"학습 실패: {e}")
        return False

    bar.progress(1.0, text="학습 완료 ✓")
    st.session_state["detector"] = AnomalyDetector(artifacts_dir)
    st.session_state["train_note"] = note
    return True


# ----------------------------- EDA -----------------------------

def render_eda(df: pd.DataFrame, cfg: dict) -> None:
    """업로드 데이터의 통계량·EDA 결과를 해석 가이드와 함께 렌더링한다."""
    data_cfg = cfg["data"]
    exclude = list(data_cfg.get("exclude_columns", []))
    label_col = data_cfg.get("label_column")
    ts_col = data_cfg.get("timestamp_column")
    feature_cols = infer_feature_columns(df, exclude)

    info = EDA.basic_info(df, feature_cols, label_col, ts_col)
    plot_df = EDA.sample_for_plot(df)
    has_label = info["has_label"]
    n_anom_label = int((df[label_col] == 1).sum()) if has_label else 0

    st.header("🔍 EDA · 데이터 분석 결과")

    # 1) 데이터 개요
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("행(row)", f"{info['n_rows']:,}")
    c2.metric("열(col)", f"{info['n_cols']:,}")
    c3.metric("피처 수", f"{info['n_features']:,}")
    c4.metric("총 결측", f"{info['total_missing']:,}")
    c5.metric("중복 행", f"{info['n_duplicates']:,}")
    st.caption(f"메모리 ≈ {info['memory_mb']:.2f} MB · "
               f"피처: {', '.join(feature_cols) if feature_cols else '없음'}")
    _note("데이터 규모와 품질을 한눈에 봅니다. 피처 수 = 오토인코더 입력 차원.<br>"
          "총 결측·중복이 0이 아니면 아래 결측·품질 항목에서 원인을 확인하세요.")

    if not feature_cols:
        st.error("수치형 피처 컬럼이 없습니다. exclude_columns 설정 또는 데이터를 확인하세요.")
        return

    # 2) 클래스 분포(라벨 있을 때) — 불균형 확인
    if has_label:
        st.subheader("클래스 분포 (label)")
        bal = EDA.label_balance(df, label_col)
        lc1, lc2, lc3 = st.columns([1, 1, 2])
        lc1.metric("정상(0)", f"{bal['n_normal']:,}")
        lc2.metric("이상(1)", f"{bal['n_anomaly']:,}")
        ratio = bal["imbalance_ratio"]
        ratio_txt = "∞ : 1" if ratio == float("inf") else f"{ratio:.1f} : 1"
        lc3.metric("불균형비 (정상:이상)", ratio_txt)
        fig = px.bar(bal["frame"], x="label", y="count", text="count",
                     color="label", color_discrete_map={"0": NORMAL_C, "1": ANOM_C})
        fig.update_layout(height=260, showlegend=False, xaxis_title="label", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)
        _note(f"정상:이상 = {ratio_txt} 로 이상이 희소합니다. 이때 정확도(accuracy)는 무의미.<br>"
              "→ F1·PR-AUC로 평가하며, 본 모델은 '정상만 학습'해 이 불균형을 우회합니다.")

    # 3) 결측치
    st.subheader("결측치 점검")
    miss = EDA.missing_frame(df)
    if len(miss) == 0:
        st.success("결측치가 없습니다.")
    else:
        st.warning(f"결측이 있는 컬럼 {len(miss)}개 — 학습 시 설정된 대치 전략으로 처리됩니다.")
        st.dataframe(miss, use_container_width=True, hide_index=True)
    _note("결측은 학습 시 median 등으로 대치되고, 추론에선 학습 때 값을 재사용합니다(누수 방지).<br>"
          "특정 피처 결측이 과도하면 센서 결함 가능성 → 제외를 검토하세요.")

    # 4) 기초 통계량
    st.subheader("기초 통계량")
    desc = EDA.describe_frame(df, feature_cols)
    st.dataframe(desc.style.format(precision=3), use_container_width=True, hide_index=True)
    _note("mean/std로 정상 범위를, min/max로 이상 후보 범위를 가늠합니다.<br>"
          "|skew|가 크면 분포가 치우침, kurtosis가 크면 극단값이 많음을 의미합니다.<br>"
          "피처 간 스케일 차이가 커도 학습 파이프라인이 정규화로 자동 보정합니다.")

    # 5) 데이터 품질 — 상수 컬럼 / 이상치(IQR) / 오염 경고
    st.subheader("데이터 품질 점검")
    const_cols = EDA.constant_columns(df, feature_cols)
    if const_cols:
        st.warning(f"상수(분산 0) 컬럼: {const_cols} → 학습 시 자동 제거됩니다.")
    else:
        st.caption("상수 컬럼 없음.")
    out = EDA.outlier_frame(df, feature_cols)
    st.dataframe(out, use_container_width=True, hide_index=True)
    warns = EDA.contamination_warnings(df, feature_cols, label_col)
    if warns:
        st.error("⚠️ 학습 데이터 오염 경고 — 정상으로 학습할 데이터에 이상치가 많은 피처:\n\n- "
                 + "\n- ".join(warns)
                 + "\n\n정상 데이터에 이상이 섞이면 모델이 이상을 정상으로 학습할 수 있습니다.")
    _note("IQR 이상치 비율이 높은 피처는 데이터 오염 또는 실제 센서 이상 신호일 수 있습니다.<br>"
          "정상으로 학습할 데이터에 이상치가 과다하면 위에 경고가 표시됩니다(정상 순수성 점검).<br>"
          "상수 컬럼은 정보가 없어 학습에서 자동 제거됩니다.")

    # 6) 분포 (히스토그램)
    st.subheader("피처 분포")
    show_feats = feature_cols[:12]
    if len(feature_cols) > 12:
        st.caption(f"피처가 많아 상위 12개만 표시합니다(전체 {len(feature_cols)}개).")
    _hist_grid(plot_df, show_feats)
    _note("봉우리가 2개 이상(다봉)이면 운전 모드가 여러 개일 수 있습니다.<br>"
          "한쪽으로 긴 꼬리는 이상치/드리프트 신호일 수 있습니다.")

    # 6-1) 정상 vs 이상 분포 비교 (라벨 있을 때 — EDA 심화)
    if has_label and n_anom_label > 0:
        st.subheader("정상 vs 이상 — 피처 분포 비교")
        pdf = plot_df.copy()
        pdf["_클래스"] = pdf[label_col].map({0: "정상", 1: "이상"}).astype(str)
        ncol = 3
        for i in range(0, len(show_feats), ncol):
            cols = st.columns(ncol)
            for j, f in enumerate(show_feats[i:i + ncol]):
                with cols[j]:
                    h = px.histogram(pdf, x=f, color="_클래스", barmode="overlay",
                                     histnorm="probability density", nbins=40,
                                     color_discrete_map={"정상": NORMAL_C, "이상": ANOM_C})
                    h.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                                    showlegend=(i == 0 and j == 0),
                                    title=dict(text=f, font=dict(size=13)),
                                    legend=dict(orientation="h"))
                    st.plotly_chart(h, use_container_width=True)
        _note("정상(파랑)/이상(빨강) 분포가 많이 겹치면 그 피처만으로 구분이 어렵고,<br>"
              "확연히 분리되면 해당 센서가 이상 진단의 핵심 단서입니다.")

    # 7) 상관관계 히트맵
    corr = EDA.correlation(df, feature_cols)
    if not corr.empty:
        st.subheader("피처 상관관계")
        heat = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", aspect="auto")
        heat.update_layout(height=420)
        st.plotly_chart(heat, use_container_width=True)
        _note("|상관|이 0.8 이상인 피처쌍은 중복 정보 — 하나로 줄이면 병목이 간결해집니다.<br>"
              "정상에서 유지되던 상관 구조가 깨진 샘플은 이상일 가능성이 큽니다.")

    # 8) 시계열 추이 (timestamp 있을 때)
    if info["has_timestamp"]:
        st.subheader("시계열 추이")
        sel = st.selectbox("피처 선택", feature_cols, key="eda_ts_feature")
        tdf = plot_df[[ts_col, sel]].sort_values(ts_col)
        line = px.line(tdf, x=ts_col, y=sel)
        line.update_traces(line_color=NORMAL_C)
        line.update_layout(height=320)
        st.plotly_chart(line, use_container_width=True)
        if len(df) > len(plot_df):
            st.caption(f"대용량 대비 {len(plot_df):,}건으로 샘플링해 표시(통계량은 전체 기준).")
        _note("추세(우상향/우하향)는 마모·드리프트를, 급변 스파이크는 순간 이상을 시사합니다.<br>"
              "정상 구간만 학습해야 하므로 이상/정비 구간이 섞였는지 확인하세요.")


def _hist_grid(plot_df: pd.DataFrame, feats: list) -> None:
    ncol = 3
    for i in range(0, len(feats), ncol):
        cols = st.columns(ncol)
        for j, f in enumerate(feats[i:i + ncol]):
            with cols[j]:
                h = px.histogram(plot_df, x=f, nbins=40, color_discrete_sequence=[NORMAL_C])
                h.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                                showlegend=False, title=dict(text=f, font=dict(size=13)))
                st.plotly_chart(h, use_container_width=True)


# ----------------------------- 학습 품질 -----------------------------

def render_training_quality(detector: AnomalyDetector, df: pd.DataFrame,
                            label_col: str | None) -> None:
    """손실곡선·오차분포·ROC/PR 커브로 학습·임계값 품질을 보여준다."""
    st.subheader("학습 품질")
    has_label = bool(label_col) and label_col in df.columns
    colA, colB = st.columns(2)

    # 손실곡선
    hp = os.path.join(detector.artifacts_dir, "history.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            hist = json.load(f)
        ep = list(range(1, len(hist.get("loss", [])) + 1))
        lf = go.Figure()
        lf.add_trace(go.Scatter(x=ep, y=hist.get("loss", []), name="train loss",
                                line=dict(color=NORMAL_C)))
        if hist.get("val_loss"):
            lf.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="val loss",
                                    line=dict(color=ANOM_C)))
        lf.update_layout(height=320, title="에폭별 손실(MSE)", xaxis_title="epoch",
                         yaxis_title="loss", legend=dict(orientation="h"))
        colA.plotly_chart(lf, use_container_width=True)

    # 오차 분포 + 임계값
    res = detector.predict(df)
    errs = res.errors
    ef = go.Figure()
    if has_label:
        y = df[label_col].values
        ef.add_trace(go.Histogram(x=errs[y == 0], name="정상", marker_color=NORMAL_C, opacity=0.7))
        ef.add_trace(go.Histogram(x=errs[y == 1], name="이상", marker_color=ANOM_C, opacity=0.7))
        ef.update_layout(barmode="overlay")
    else:
        ef.add_trace(go.Histogram(x=errs, name="오차", marker_color=NORMAL_C))
    ef.add_vline(x=detector.threshold, line_dash="dash", line_color="#e6550d",
                 annotation_text="임계값")
    ef.update_layout(height=320, title="재구성 오차 분포 + 임계값",
                     xaxis_title="재구성 오차", yaxis_title="빈도", legend=dict(orientation="h"))
    colB.plotly_chart(ef, use_container_width=True)
    _note("손실이 감소 후 평탄해지면 수렴, train/val 격차가 크면 과적합입니다.<br>"
          "오른쪽 분포에서 정상(파랑)은 임계값 왼쪽, 이상(빨강)은 오른쪽에 몰릴수록 분리가 좋습니다.")

    # ROC / PR 커브
    if has_label:
        from src.evaluate import roc_curve_points, pr_curve_points
        roc = roc_curve_points(df[label_col].values, errs)
        pr = pr_curve_points(df[label_col].values, errs)
        r1, r2 = st.columns(2)
        if roc:
            f = go.Figure()
            f.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], fill="tozeroy",
                                   line=dict(color=NORMAL_C), name="ROC"))
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                                   name="랜덤"))
            f.update_layout(height=320, title=f"ROC 커브 (AUC={roc['auc']:.3f})",
                            xaxis_title="FPR", yaxis_title="TPR", legend=dict(orientation="h"))
            r1.plotly_chart(f, use_container_width=True)
        if pr:
            f = go.Figure()
            f.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], fill="tozeroy",
                                   line=dict(color=ANOM_C), name="PR"))
            f.update_layout(height=320, title=f"PR 커브 (AP={pr['ap']:.3f})",
                            xaxis_title="Recall", yaxis_title="Precision",
                            legend=dict(orientation="h"))
            r2.plotly_chart(f, use_container_width=True)
        _note("AUC/AP가 1에 가까울수록 정상·이상 분리력이 좋습니다.<br>"
              "불균형에서는 PR 커브(AP)가 더 신뢰할 만하며, Precision-Recall 균형점으로 임계값을 정합니다.")


# ----------------------------- 판정 결과 -----------------------------

def render_results(detector: AnomalyDetector, df: pd.DataFrame,
                   label_col: str | None) -> None:
    """판정 결과·시각화·평가지표·개별 진단·아티팩트 다운로드를 렌더링한다."""
    default_thr = detector.threshold
    thr = st.slider(
        "판정 임계값 (조정 시 민감도 변화)",
        min_value=float(default_thr) * 0.1,
        max_value=float(default_thr) * 3.0,
        value=float(default_thr),
        step=float(default_thr) * 0.01,
    )

    try:
        result = detector.predict(df, threshold=thr)
    except ValueError as e:
        st.error(f"스키마 검증 실패:\n\n{e}")
        return

    ts_col = detector.schema.timestamp_column
    summary = result.summary_frame(df, ts_col)
    n = len(summary)
    n_anom = int(result.predictions.sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 샘플", f"{n:,}")
    c2.metric("이상 탐지", f"{n_anom:,}", f"{n_anom / n * 100:.1f}%")
    c3.metric("적용 임계값", f"{thr:.5f}")

    # 재구성 오차 산점도
    st.subheader("재구성 오차")
    x_axis = summary[ts_col] if (ts_col and ts_col in summary.columns) else summary.index
    fig = go.Figure()
    normal_mask = result.predictions == 0
    fig.add_trace(go.Scatter(x=x_axis[normal_mask], y=result.errors[normal_mask],
                             mode="markers", name="정상", marker=dict(color=NORMAL_C, size=5)))
    fig.add_trace(go.Scatter(x=x_axis[~normal_mask], y=result.errors[~normal_mask],
                             mode="markers", name="이상",
                             marker=dict(color=ANOM_C, size=7, symbol="x")))
    fig.add_hline(y=thr, line_dash="dash", line_color="#e6550d",
                  annotation_text="임계값", annotation_position="top left")
    fig.update_layout(height=400, xaxis_title="샘플", yaxis_title="재구성 오차",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # 피처별 기여도
    st.subheader("이상 원인 — 피처별 평균 기여도 (상위)")
    top = result.top_features(k=len(result.feature_columns))
    barfig = px.bar(top, x="mean_squared_error", y="feature", orientation="h")
    barfig.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(barfig, use_container_width=True)

    # 평가지표 (라벨 있을 때)
    if label_col and label_col in df.columns:
        st.subheader("평가지표 (label 컬럼 감지됨)")
        from src.evaluate import compute_metrics
        metrics = compute_metrics(df[label_col].values, result.errors, result.predictions)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics['precision']:.3f}")
        m2.metric("Recall", f"{metrics['recall']:.3f}")
        m3.metric("F1", f"{metrics['f1']:.3f}")
        pr = metrics["pr_auc"]
        m4.metric("PR-AUC", f"{pr:.3f}" if pr is not None else "N/A")
        cm = metrics["confusion_matrix"]
        st.caption("혼동행렬 [[TN, FP], [FN, TP]]")
        st.table(pd.DataFrame(cm, index=["실제정상", "실제이상"], columns=["예측정상", "예측이상"]))

    # 개별 샘플 진단 (설명가능성)
    st.subheader("개별 샘플 진단")
    only_anom = st.checkbox("이상 샘플만 선택", value=(n_anom > 0))
    pool = list(np.where(result.predictions == 1)[0]) if (only_anom and n_anom > 0) else list(range(n))
    if not pool:
        pool = list(range(n))
    sel_idx = st.selectbox("샘플 인덱스", pool, index=0, key="drill_idx")
    pfe = result.per_feature_error[sel_idx]
    dd = pd.DataFrame({"feature": result.feature_columns, "squared_error": pfe}) \
        .sort_values("squared_error", ascending=False)
    dc1, dc2 = st.columns([1, 2])
    dc1.metric("상태", "이상" if result.predictions[sel_idx] == 1 else "정상")
    dc1.metric("이상 점수", f"{result.errors[sel_idx] / thr:.2f}")
    dc1.caption(f"재구성 오차 {result.errors[sel_idx]:.6f} / 임계값 {thr:.6f}")
    ddfig = px.bar(dd, x="squared_error", y="feature", orientation="h",
                   color_discrete_sequence=[ANOM_C])
    ddfig.update_layout(height=300, yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=10, t=20, b=10))
    dc2.plotly_chart(ddfig, use_container_width=True)
    _note("선택 샘플에서 재구성 오차가 큰 피처가 이상 원인일 가능성이 높습니다.<br>"
          "상위 피처에 해당하는 센서를 우선 점검하세요(설명가능성).")

    # 결과 표 & 다운로드
    st.subheader("샘플별 판정 결과")
    st.dataframe(summary, use_container_width=True, height=300)
    dl1, dl2 = st.columns(2)
    dl1.download_button("결과 CSV 다운로드",
                        data=summary.to_csv(index=False).encode("utf-8-sig"),
                        file_name="anomaly_result.csv", mime="text/csv",
                        use_container_width=True)
    dl2.download_button("학습 아티팩트(zip) 다운로드",
                        data=_zip_dir(detector.artifacts_dir),
                        file_name="phm_artifacts.zip", mime="application/zip",
                        use_container_width=True)
    st.caption("아티팩트 zip = 모델·스케일러·임계값·스키마·메타·이력. 사내 서버로 옮겨 그대로 사용 가능합니다.")


# ----------------------------- 메인 -----------------------------

def main() -> None:
    st.title("🔧 PHM 오토인코더 기반 이상탐지")
    st.caption("정상 데이터로 오토인코더를 학습하고, 재구성 오차가 임계값을 넘으면 이상으로 판정합니다. "
               "CSV 업로드 → (EDA 검토·학습 설정) → '학습' 순으로 진행하세요.")

    cfg = load_config()
    label_col = cfg["data"].get("label_column")

    if "artifacts_dir" not in st.session_state:
        st.session_state["artifacts_dir"] = tempfile.mkdtemp(prefix="phm_art_")
    artifacts_dir = st.session_state["artifacts_dir"]

    # 사이드바: 샘플 CSV 다운로드
    with st.sidebar:
        st.header("샘플 데이터")
        sample_path = os.path.join(ROOT, "samples", "sample_sensor.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as fp:
                st.download_button("샘플 CSV 내려받기", fp.read(),
                                   file_name="sample_sensor.csv", mime="text/csv",
                                   use_container_width=True)
            st.caption("내려받아 업로드 후 'EDA'→'학습' 순으로 체험해보세요. (정상+이상+label 포함)")

    uploaded = st.file_uploader("CSV 업로드 (학습·판정용)", type=["csv"])
    _reset_if_new_upload(uploaded)

    if uploaded is None:
        st.info("① CSV를 업로드하면 → ② 오른쪽에 'EDA'·'학습' 버튼이 나타납니다. "
                "EDA로 먼저 검토한 뒤 '학습'을 누르면 학습(진행률 표시) 후 판정을 진행합니다.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    # 미리보기(좌) + EDA·학습 버튼(우)
    left, right = st.columns([4, 1.4])
    with left:
        st.subheader("입력 데이터 미리보기")
        st.dataframe(df.head(), use_container_width=True)
    with right:
        st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        eda_clicked = bcol1.button("🔍 EDA", use_container_width=True)
        train_clicked = bcol2.button("🚀 학습", type="primary", use_container_width=True)
        st.caption(f"행 {len(df):,} · 열 {df.shape[1]}\n\nEDA로 먼저 검토 후 학습을 권장합니다.")

    # 학습 설정(선택)
    tcfg = cfg["train"]
    with st.expander("⚙️ 학습 설정 (선택) — 값 조정 후 '학습'을 누르면 반영됩니다"):
        s1, s2, s3, s4 = st.columns(4)
        set_epochs = s1.number_input("epoch 수", 10, 500, int(tcfg["epochs"]), step=10)
        set_bottleneck = s2.number_input("병목 차원(bottleneck)", 2, 64,
                                         int(cfg["model"]["bottleneck"]))
        set_pct = s3.number_input("임계값 백분위수(%)", 90.0, 99.9,
                                  float(cfg["threshold"].get("percentile", 99.0)), step=0.5)
        set_lr = s4.number_input("learning rate", 0.0001, 0.1,
                                 float(tcfg["learning_rate"]), step=0.0001, format="%.4f")
        st.caption("병목이 너무 크면 이상치까지 복원해 탐지가 약해집니다. "
                   "백분위수를 높이면 이상 판정이 더 보수적이 됩니다.")
    settings = {"epochs": set_epochs, "bottleneck": set_bottleneck,
                "percentile": set_pct, "lr": set_lr}

    if eda_clicked:
        st.session_state["show_eda"] = True
    if train_clicked:
        train_and_store(df, cfg, artifacts_dir, label_col, settings)

    # EDA 결과
    if st.session_state.get("show_eda"):
        st.divider()
        render_eda(df, cfg)

    # 학습 결과
    detector = st.session_state.get("detector")
    if detector is None:
        msg = ("EDA 검토 후 **'학습'** 버튼을 눌러 학습을 시작하세요."
               if st.session_state.get("show_eda")
               else "**'EDA'** 로 데이터를 먼저 검토하거나, **'학습'** 버튼으로 바로 학습을 시작하세요.")
        st.info(msg)
        st.stop()

    st.divider()
    if st.session_state.get("train_note"):
        st.success("학습 완료 · " + st.session_state["train_note"])
    render_training_quality(detector, df, label_col)
    st.divider()
    render_results(detector, df, label_col)


if __name__ == "__main__":
    main()
