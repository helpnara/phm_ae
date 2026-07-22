"""PHM 오토인코더 이상탐지 — Streamlit 웹 테스트 앱.

실행:
    streamlit run app/app.py

화면 구성(사용자 편의성 개편):
  ┌ 상단 컨트롤 프레임 : 업로드 · 미리보기 · [EDA][학습] 버튼 · 학습 설정 (고정)
  ├ EDA 결과 프레임    : 통계량·분포·상관·시간의존성 등 (내부 스크롤)
  └ 학습 결과 프레임    : 학습 품질·이상 판정·개별 진단·배치·리포트 (내부 스크롤)
모든 버튼을 상단 프레임에 모아 결과가 갱신돼도 컨트롤 위치가 움직이지 않는다.

시계열 윈도우 모델(LSTM-AE): EDA의 시간 의존성 진단으로 필요 여부를 판단하고,
필요 시 '학습' 클릭 때 팝업으로 사용자가 모델을 선택한다.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_config
from src.detector import AnomalyDetector
from src.data import infer_feature_columns, load_meta, load_schema
from src import eda as EDA
from src import registry as REG

st.set_page_config(page_title="PHM 이상탐지", page_icon="🔧", layout="wide")

NORMAL_C = "#2c7fb8"
ANOM_C = "#d7301f"


@st.cache_resource(show_spinner="모델 로딩 중...")
def get_detector(path: str) -> AnomalyDetector:
    """레지스트리 경로의 모델을 로드(캐시). 경로가 키이므로 모델별로 캐시된다."""
    return AnomalyDetector(path)


def _note(text: str) -> None:
    st.markdown(
        f"<div style='background:rgba(44,127,184,0.08);border-left:3px solid {NORMAL_C};"
        f"padding:8px 12px;border-radius:4px;font-size:0.86rem;color:inherit;margin:2px 0 10px'>"
        f"💡 <b>해석 가이드</b><br>{text}</div>", unsafe_allow_html=True)


def _reset_if_new_upload(uploaded) -> None:
    """새 업로드 시 EDA/학습 진행 상태만 초기화한다(선택된 모델은 유지)."""
    sig = None if uploaded is None else (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:
        st.session_state["upload_sig"] = sig
        for k in ("show_eda", "pending_train", "thr_slider", "cmp_result", "cmp_select"):
            st.session_state.pop(k, None)


@st.cache_data(show_spinner=False)
def _model_features(path: str, _mtime: float):
    """모델 스키마의 피처 컬럼(TF 로드 없이 schema.json만 읽어 캐시)."""
    try:
        return load_schema(path).feature_columns
    except Exception:  # noqa: BLE001
        return None


def _compat(path: str, df_columns) -> tuple[bool, list]:
    """모델과 업로드 데이터의 스키마 호환성. (호환여부, 누락 피처)."""
    schema_path = os.path.join(path, "schema.json")
    mtime = os.path.getmtime(schema_path) if os.path.exists(schema_path) else 0.0
    fc = _model_features(path, mtime)
    if fc is None:
        return False, ["스키마 없음"]
    missing = [c for c in fc if c not in df_columns]
    return (len(missing) == 0), missing


def _zip_dir(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(path)):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                z.write(fp, arcname=name)
    buf.seek(0)
    return buf.getvalue()


def train_and_store(df, cfg, models_dir, label_col, settings, model_type, window,
                    model_name) -> str | None:
    """학습 후 레지스트리에 새 모델을 저장하고 그 경로를 반환한다."""
    from tensorflow import keras
    from src.train import train_from_dataframe
    from src.evaluate import compute_metrics

    cfg = copy.deepcopy(cfg)
    cfg["train"]["epochs"] = int(settings["epochs"])
    cfg["train"]["learning_rate"] = float(settings["lr"])
    cfg["model"]["bottleneck"] = int(settings["bottleneck"])
    cfg["threshold"]["method"] = "percentile"
    cfg["threshold"]["percentile"] = float(settings["percentile"])

    has_label = bool(label_col) and label_col in df.columns
    train_df = df[df[label_col] == 0] if has_label else df
    kind = "시계열 윈도우(LSTM-AE)" if model_type == "lstm" else "행 단위(Dense AE)"
    if has_label:
        note = f"{kind} · 정상(label=0) {len(train_df):,}건 학습 → 전체 {len(df):,}건 판정"
    else:
        note = f"{kind} · 업로드 {len(df):,}건을 정상으로 간주해 학습 후 판정"

    min_rows = (int(window or 20) + 20) if model_type == "lstm" else 20
    if len(train_df) < min_rows:
        st.error(f"학습 데이터가 부족합니다(현재 {len(train_df)}건, 최소 {min_rows}건).")
        return None

    entry_dir = REG.new_entry_dir(models_dir, model_type, model_name)
    st.caption(note + f" · epoch {cfg['train']['epochs']} · 임계 {cfg['threshold']['percentile']}%")
    epochs = max(int(cfg["train"]["epochs"]), 1)
    bar = st.progress(0.0, text="학습 준비 중...")

    class _P(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            bar.progress(min((epoch + 1) / epochs, 1.0),
                         text=(f"학습 중... {epoch + 1}/{epochs} epoch · "
                               f"loss={logs.get('loss', 0.0):.5f} · "
                               f"val_loss={logs.get('val_loss', 0.0):.5f}"))

    try:
        train_from_dataframe(train_df, cfg, artifacts_dir=entry_dir,
                             extra_callbacks=[_P()], verbose=0,
                             data_source="streamlit upload",
                             model_type=model_type, window=window)
    except Exception as e:  # noqa: BLE001
        bar.empty()
        REG.delete_model(entry_dir)
        st.error(f"학습 실패: {e}")
        return None

    bar.progress(1.0, text="학습 완료 ✓")

    # 레지스트리 항목 정보(이름 + 평가지표) 저장
    metrics = None
    if has_label:
        det = AnomalyDetector(entry_dir)
        res = det.predict(df)
        mm = compute_metrics(df[label_col].values, res.errors, res.predictions)
        metrics = {k: mm[k] for k in ("f1", "pr_auc", "precision", "recall")}
    default_name = os.path.basename(entry_dir)
    REG.write_entry_info(entry_dir, {"name": (model_name or default_name), "metrics": metrics})

    st.session_state["train_note"] = note
    return entry_dir


@st.dialog("⏱️ 시계열 윈도우 모델 선택")
def ts_dialog(default_window: int) -> None:
    st.write("EDA 진단 결과 **시간 의존성이 높게** 나타났습니다. 어떤 모델로 학습할까요?")
    choice = st.radio(
        "모델 선택",
        ["시계열 윈도우 (LSTM-AE) — 시간 패턴 반영, 권장", "행 단위 (Dense AE) — 빠름"],
        index=0,
    )
    win = st.number_input("윈도우 크기(시계열 모델)", 5, 100, int(default_window),
                          help="연속된 몇 개의 타임스텝을 하나의 시퀀스로 볼지")
    st.caption("시계열 모델은 학습에 조금 더 오래 걸릴 수 있습니다.")
    if st.button("이 설정으로 학습 시작", type="primary", use_container_width=True):
        st.session_state["pending_train"] = {
            "model_type": "lstm" if "LSTM" in choice else "dense",
            "window": int(win),
        }
        st.rerun()


# ----------------------------- EDA -----------------------------

def _hist_grid(plot_df, feats):
    for i in range(0, len(feats), 3):
        cols = st.columns(3)
        for j, f in enumerate(feats[i:i + 3]):
            with cols[j]:
                h = px.histogram(plot_df, x=f, nbins=40, color_discrete_sequence=[NORMAL_C])
                h.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10),
                                showlegend=False, title=dict(text=f, font=dict(size=13)))
                st.plotly_chart(h, use_container_width=True)


def render_eda(df, cfg, td):
    data_cfg = cfg["data"]
    exclude = list(data_cfg.get("exclude_columns", []))
    label_col = data_cfg.get("label_column")
    ts_col = data_cfg.get("timestamp_column")
    feature_cols = infer_feature_columns(df, exclude)
    info = EDA.basic_info(df, feature_cols, label_col, ts_col)
    plot_df = EDA.sample_for_plot(df)
    has_label = info["has_label"]
    n_anom_label = int((df[label_col] == 1).sum()) if has_label else 0

    # 개요
    c = st.columns(5)
    c[0].metric("행", f"{info['n_rows']:,}"); c[1].metric("열", f"{info['n_cols']:,}")
    c[2].metric("피처", f"{info['n_features']:,}"); c[3].metric("총 결측", f"{info['total_missing']:,}")
    c[4].metric("중복 행", f"{info['n_duplicates']:,}")
    st.caption(f"메모리 ≈ {info['memory_mb']:.2f} MB · 피처: {', '.join(feature_cols) or '없음'}")
    _note("데이터 규모·품질을 한눈에 봅니다. 피처 수 = 오토인코더 입력 차원.<br>"
          "총 결측·중복이 0이 아니면 아래 항목에서 원인을 확인하세요.")
    if not feature_cols:
        st.error("수치형 피처가 없습니다. exclude_columns/데이터를 확인하세요.")
        return

    # 시간 의존성 진단
    if td is not None:
        st.subheader("⏱️ 시간 의존성 진단")
        verdict = "높음 → 시계열 윈도우(LSTM-AE) 권장" if td["recommend"] else "낮음 → 행 단위(Dense) 적합"
        (st.warning if td["recommend"] else st.success)(
            f"평균 |lag-1 자기상관| = {td['mean_abs_acf1']:.3f} (임계 {td['threshold']}) → **{verdict}**")
        st.dataframe(td["frame"].round(3), use_container_width=True, hide_index=True)
        _note("자기상관이 크면 값이 시간에 걸쳐 이어져(관성) 시계열 구조가 중요합니다.<br>"
              "이 경우 LSTM-AE가 유리하며, '학습' 시 팝업에서 선택할 수 있습니다.<br>"
              "자기상관이 작으면 각 행이 독립적이라 행 단위 Dense AE로 충분합니다.")

    # 클래스 분포
    if has_label:
        st.subheader("클래스 분포 (label)")
        bal = EDA.label_balance(df, label_col)
        lc = st.columns([1, 1, 2])
        lc[0].metric("정상(0)", f"{bal['n_normal']:,}"); lc[1].metric("이상(1)", f"{bal['n_anomaly']:,}")
        ratio = bal["imbalance_ratio"]
        rtxt = "∞ : 1" if ratio == float("inf") else f"{ratio:.1f} : 1"
        lc[2].metric("불균형비 (정상:이상)", rtxt)
        fig = px.bar(bal["frame"], x="label", y="count", text="count", color="label",
                     color_discrete_map={"0": NORMAL_C, "1": ANOM_C})
        fig.update_layout(height=250, showlegend=False, yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)
        _note(f"정상:이상 = {rtxt} 로 이상이 희소 → 정확도(accuracy)는 무의미.<br>"
              "F1·PR-AUC로 평가하며, 본 모델은 '정상만 학습'해 불균형을 우회합니다.")

    # 결측치
    st.subheader("결측치 점검")
    miss = EDA.missing_frame(df)
    if len(miss) == 0:
        st.success("결측치가 없습니다.")
    else:
        st.warning(f"결측 컬럼 {len(miss)}개 — 학습 시 대치됩니다.")
        st.dataframe(miss, use_container_width=True, hide_index=True)
    _note("결측은 학습 시 median 등으로 대치, 추론에선 학습 때 값을 재사용(누수 방지).<br>"
          "특정 피처 결측이 과도하면 센서 결함 가능성 → 제외 검토.")

    # 기초 통계량
    st.subheader("기초 통계량")
    st.dataframe(EDA.describe_frame(df, feature_cols).style.format(precision=3),
                 use_container_width=True, hide_index=True)
    _note("mean/std로 정상 범위, min/max로 이상 후보를 가늠합니다.<br>"
          "|skew| 큼=치우침, kurtosis 큼=극단값 많음. 스케일 차이는 정규화로 자동 보정됩니다.")

    # 데이터 품질
    st.subheader("데이터 품질 점검")
    const_cols = EDA.constant_columns(df, feature_cols)
    if const_cols:
        st.warning(f"상수 컬럼: {const_cols} → 학습 시 자동 제거.")
    st.dataframe(EDA.outlier_frame(df, feature_cols), use_container_width=True, hide_index=True)
    warns = EDA.contamination_warnings(df, feature_cols, label_col)
    if warns:
        st.error("⚠️ 학습 데이터 오염 경고 — 정상 데이터에 이상치가 많은 피처:\n\n- " + "\n- ".join(warns))
    _note("IQR 이상치 비율이 높으면 데이터 오염/센서 이상 신호일 수 있습니다.<br>"
          "정상 학습 데이터에 이상치가 과다하면 위 경고 표시(정상 순수성 점검). 상수 컬럼은 자동 제거.")

    # 분포
    st.subheader("피처 분포")
    show = feature_cols[:12]
    _hist_grid(plot_df, show)
    _note("다봉(봉우리 2개+)은 운전 모드가 여럿일 수 있음. 긴 꼬리는 이상치/드리프트 신호.")

    # 정상 vs 이상
    if has_label and n_anom_label > 0:
        st.subheader("정상 vs 이상 — 피처 분포 비교")
        pdf = plot_df.copy()
        pdf["_클래스"] = pdf[label_col].map({0: "정상", 1: "이상"}).astype(str)
        for i in range(0, len(show), 3):
            cols = st.columns(3)
            for j, f in enumerate(show[i:i + 3]):
                with cols[j]:
                    h = px.histogram(pdf, x=f, color="_클래스", barmode="overlay",
                                     histnorm="probability density", nbins=40,
                                     color_discrete_map={"정상": NORMAL_C, "이상": ANOM_C})
                    h.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                                    showlegend=(i == 0 and j == 0),
                                    title=dict(text=f, font=dict(size=13)),
                                    legend=dict(orientation="h"))
                    st.plotly_chart(h, use_container_width=True)
        _note("정상/이상 분포가 겹치면 그 피처만으론 구분 어렵고, 분리되면 이상 진단의 핵심 단서.")

    # 상관관계
    corr = EDA.correlation(df, feature_cols)
    if not corr.empty:
        st.subheader("피처 상관관계")
        heat = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", aspect="auto")
        heat.update_layout(height=400)
        st.plotly_chart(heat, use_container_width=True)
        _note("|상관|≥0.8 피처쌍은 중복 정보 — 줄이면 병목이 간결. 상관 구조가 깨진 샘플은 이상 가능성 큼.")

    # 시계열 추이
    if info["has_timestamp"]:
        st.subheader("시계열 추이")
        sel = st.selectbox("피처 선택", feature_cols, key="eda_ts_feature")
        tdf = plot_df[[ts_col, sel]].sort_values(ts_col)
        line = px.line(tdf, x=ts_col, y=sel)
        line.update_traces(line_color=NORMAL_C)
        line.update_layout(height=300)
        st.plotly_chart(line, use_container_width=True)
        _note("추세는 마모·드리프트, 급변 스파이크는 순간 이상을 시사. 이상/정비 구간 혼입 여부 확인.")


# ----------------------------- 학습 품질 -----------------------------

def render_training_quality(detector, df, label_col):
    st.subheader("학습 품질")
    has_label = bool(label_col) and label_col in df.columns
    colA, colB = st.columns(2)
    hp = os.path.join(detector.artifacts_dir, "history.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            hist = json.load(f)
        ep = list(range(1, len(hist.get("loss", [])) + 1))
        lf = go.Figure()
        lf.add_trace(go.Scatter(x=ep, y=hist.get("loss", []), name="train loss", line=dict(color=NORMAL_C)))
        if hist.get("val_loss"):
            lf.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="val loss", line=dict(color=ANOM_C)))
        lf.update_layout(height=300, title="에폭별 손실(MSE)", xaxis_title="epoch",
                         yaxis_title="loss", legend=dict(orientation="h"))
        colA.plotly_chart(lf, use_container_width=True)

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
    ef.add_vline(x=detector.threshold, line_dash="dash", line_color="#e6550d", annotation_text="임계값")
    ef.update_layout(height=300, title="재구성 오차 분포 + 임계값",
                     xaxis_title="재구성 오차", yaxis_title="빈도", legend=dict(orientation="h"))
    colB.plotly_chart(ef, use_container_width=True)
    _note("손실이 감소 후 평탄=수렴, train/val 격차 크면 과적합.<br>"
          "오른쪽 분포에서 정상(파랑)은 임계값 왼쪽, 이상(빨강)은 오른쪽에 몰릴수록 분리 좋음.")

    if has_label:
        from src.evaluate import roc_curve_points, pr_curve_points
        roc = roc_curve_points(df[label_col].values, errs)
        pr = pr_curve_points(df[label_col].values, errs)
        r = st.columns(2)
        if roc:
            f = go.Figure()
            f.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], fill="tozeroy", line=dict(color=NORMAL_C), name="ROC"))
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="랜덤"))
            f.update_layout(height=300, title=f"ROC (AUC={roc['auc']:.3f})",
                            xaxis_title="FPR", yaxis_title="TPR", legend=dict(orientation="h"))
            r[0].plotly_chart(f, use_container_width=True)
        if pr:
            f = go.Figure()
            f.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], fill="tozeroy", line=dict(color=ANOM_C), name="PR"))
            f.update_layout(height=300, title=f"PR (AP={pr['ap']:.3f})",
                            xaxis_title="Recall", yaxis_title="Precision", legend=dict(orientation="h"))
            r[1].plotly_chart(f, use_container_width=True)
        _note("AUC/AP가 1에 가까울수록 분리력 좋음. 불균형에선 PR(AP)이 더 신뢰할 만함.")


# ----------------------------- 판정 결과 -----------------------------

def render_results(detector, df, label_col):
    from src.evaluate import compute_metrics, best_f1_threshold
    default_thr = float(detector.threshold)
    has_label = bool(label_col) and label_col in df.columns

    res0 = detector.predict(df)          # 점수(임계값 무관) 확보
    rec = best_f1_threshold(df[label_col].values, res0.errors) if has_label else None

    smin, smax = default_thr * 0.1, default_thr * 3
    if rec:
        smax = max(smax, rec["threshold"] * 1.2)
    st.session_state.setdefault("thr_slider", default_thr)

    sc1, sc2 = st.columns([3, 1.4])
    with sc1:
        thr = st.slider("판정 임계값", float(smin), float(smax), key="thr_slider",
                        step=float((smax - smin) / 100))
    with sc2:
        if rec:
            st.caption(f"F1 최대 추천 임계값\n\n**{rec['threshold']:.5f}** (F1={rec['f1']:.3f})")
            if st.button("추천값 적용", use_container_width=True):
                st.session_state["thr_slider"] = float(np.clip(rec["threshold"], smin, smax))
                st.rerun()
        else:
            st.caption("라벨이 있으면 F1 최대 임계값을 추천합니다.")

    result = detector.predict(df, threshold=thr)
    ts_col = detector.schema.timestamp_column
    summary = result.summary_frame(df, ts_col)
    n, n_anom = len(summary), int(result.predictions.sum())

    m = st.columns(3)
    m[0].metric("전체 샘플", f"{n:,}")
    m[1].metric("이상 탐지", f"{n_anom:,}", f"{n_anom / n * 100:.1f}%")
    m[2].metric("적용 임계값", f"{thr:.5f}")

    # 재구성 오차
    st.subheader("재구성 오차")
    x_axis = summary[ts_col] if (ts_col and ts_col in summary.columns) else summary.index
    fig = go.Figure()
    nm = result.predictions == 0
    fig.add_trace(go.Scatter(x=x_axis[nm], y=result.errors[nm], mode="markers", name="정상",
                             marker=dict(color=NORMAL_C, size=5)))
    fig.add_trace(go.Scatter(x=x_axis[~nm], y=result.errors[~nm], mode="markers", name="이상",
                             marker=dict(color=ANOM_C, size=7, symbol="x")))
    fig.add_hline(y=thr, line_dash="dash", line_color="#e6550d", annotation_text="임계값")
    fig.update_layout(height=360, xaxis_title="샘플", yaxis_title="재구성 오차", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # 피처별 기여도
    st.subheader("이상 원인 — 피처별 평균 기여도")
    top = result.top_features(k=len(result.feature_columns))
    bfig = px.bar(top, x="mean_squared_error", y="feature", orientation="h")
    bfig.update_layout(height=320, yaxis=dict(autorange="reversed"))
    st.plotly_chart(bfig, use_container_width=True)

    metrics = None
    if has_label:
        st.subheader("평가지표")
        metrics = compute_metrics(df[label_col].values, result.errors, result.predictions)
        mm = st.columns(4)
        mm[0].metric("Precision", f"{metrics['precision']:.3f}")
        mm[1].metric("Recall", f"{metrics['recall']:.3f}")
        mm[2].metric("F1", f"{metrics['f1']:.3f}")
        pr = metrics["pr_auc"]
        mm[3].metric("PR-AUC", f"{pr:.3f}" if pr is not None else "N/A")
        cm = metrics["confusion_matrix"]
        st.table(pd.DataFrame(cm, index=["실제정상", "실제이상"], columns=["예측정상", "예측이상"]))

    # 개별 샘플 진단
    st.subheader("개별 샘플 진단")
    only_anom = st.checkbox("이상 샘플만 선택", value=(n_anom > 0))
    pool = list(np.where(result.predictions == 1)[0]) if (only_anom and n_anom > 0) else list(range(n))
    sel = st.selectbox("샘플 인덱스", pool or list(range(n)), key="drill_idx")
    dd = pd.DataFrame({"feature": result.feature_columns, "squared_error": result.per_feature_error[sel]}) \
        .sort_values("squared_error", ascending=False)
    d1, d2 = st.columns([1, 2])
    d1.metric("상태", "이상" if result.predictions[sel] == 1 else "정상")
    d1.metric("이상 점수", f"{result.errors[sel] / thr:.2f}")
    dfig = px.bar(dd, x="squared_error", y="feature", orientation="h", color_discrete_sequence=[ANOM_C])
    dfig.update_layout(height=280, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
    d2.plotly_chart(dfig, use_container_width=True)
    _note("선택 샘플에서 오차가 큰 피처가 이상 원인일 가능성이 큽니다. 해당 센서를 우선 점검하세요.")

    # 결과 표 & 다운로드 (CSV / 아티팩트 zip / HTML 리포트)
    st.subheader("샘플별 판정 결과")
    st.dataframe(summary, use_container_width=True, height=280)
    from src.report import build_html_report
    html = build_html_report(summary, load_meta(detector.artifacts_dir), thr, n_anom, metrics, top)
    g = st.columns(3)
    g[0].download_button("결과 CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                         "anomaly_result.csv", "text/csv", use_container_width=True)
    g[1].download_button("HTML 리포트", html.encode("utf-8"),
                         "phm_report.html", "text/html", use_container_width=True)
    g[2].download_button("학습 아티팩트(zip)", _zip_dir(detector.artifacts_dir),
                         "phm_artifacts.zip", "application/zip", use_container_width=True)
    st.caption("HTML 리포트는 브라우저 인쇄로 PDF 저장 가능 · 아티팩트 zip은 사내 서버 이관용.")

    # 다중 CSV 배치 판정
    st.subheader("다중 CSV 배치 판정")
    st.caption("학습된 모델로 여러 CSV를 한 번에 판정합니다.")
    batch = st.file_uploader("여러 CSV 업로드", type=["csv"], accept_multiple_files=True, key="batch")
    if batch:
        rows = []
        for f in batch:
            try:
                bdf = pd.read_csv(f)
                br = detector.predict(bdf, threshold=thr)
                na = int(br.predictions.sum())
                rows.append({"파일": f.name, "행": len(bdf), "이상": na,
                             "이상%": round(na / max(len(bdf), 1) * 100, 1), "비고": ""})
            except Exception as e:  # noqa: BLE001
                rows.append({"파일": f.name, "행": 0, "이상": 0, "이상%": 0.0, "비고": f"오류: {e}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------- 모델 비교 -----------------------------

def _run_comparison(paths, df, label_col):
    """선택 모델들을 현재 업로드 데이터로 실행해 비교 결과를 세션에 저장한다."""
    from src.evaluate import compute_metrics, roc_curve_points, pr_curve_points
    has_label = bool(label_col) and label_col in df.columns
    rows, curves = [], []
    for path in paths:
        det = get_detector(path)
        res = det.predict(df)
        name = REG.read_entry_info(path).get("name", os.path.basename(path))
        na = int(res.predictions.sum())
        row = {"모델": name, "종류": det.model_type,
               "이상건수": na, "이상%": round(na / max(len(df), 1) * 100, 1)}
        entry = {"name": name}
        if has_label:
            m = compute_metrics(df[label_col].values, res.errors, res.predictions)
            row.update({"Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3),
                        "F1": round(m["f1"], 3),
                        "PR-AUC": round(m["pr_auc"], 3) if m["pr_auc"] is not None else None,
                        "ROC-AUC": round(m["roc_auc"], 3) if m["roc_auc"] is not None else None})
            entry["roc"] = roc_curve_points(df[label_col].values, res.errors)
            entry["pr"] = pr_curve_points(df[label_col].values, res.errors)
        rows.append(row)
        curves.append(entry)
    st.session_state["cmp_result"] = {"rows": rows, "curves": curves,
                                      "has_label": has_label, "paths": list(paths)}


def _render_comparison_result():
    r = st.session_state.get("cmp_result")
    if not r:
        return
    st.markdown("**비교 결과** (현재 업로드 데이터 기준)")
    tbl = pd.DataFrame(r["rows"]).set_index("모델")
    # 최고 성능 강조(F1 있으면 F1, 없으면 이상% 최소)
    if r["has_label"] and "F1" in tbl.columns:
        st.dataframe(tbl.style.highlight_max(subset=["F1"], color="#c7e9c0")
                     .format(precision=3), use_container_width=True)
    else:
        st.dataframe(tbl, use_container_width=True)

    palette = ["#2c7fb8", "#d7301f", "#238b45", "#e6550d", "#6a51a3", "#08519c"]
    if r["has_label"]:
        cc = st.columns(2)
        roc = go.Figure()
        pr = go.Figure()
        for i, e in enumerate(r["curves"]):
            col = palette[i % len(palette)]
            if e.get("roc"):
                roc.add_trace(go.Scatter(x=e["roc"]["fpr"], y=e["roc"]["tpr"], name=e["name"],
                                         line=dict(color=col)))
            if e.get("pr"):
                pr.add_trace(go.Scatter(x=e["pr"]["recall"], y=e["pr"]["precision"], name=e["name"],
                                        line=dict(color=col)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        roc.update_layout(height=340, title="ROC 비교", xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(orientation="h"))
        pr.update_layout(height=340, title="PR 비교", xaxis_title="Recall", yaxis_title="Precision",
                         legend=dict(orientation="h"))
        cc[0].plotly_chart(roc, use_container_width=True)
        cc[1].plotly_chart(pr, use_container_width=True)
        _note("동일 데이터로 각 모델을 실행한 결과입니다. F1·PR-AUC가 높고 ROC/PR 곡선이 "
              "좌상단(ROC)·우상단(PR)에 가까운 모델이 우수합니다.")
    else:
        bar = px.bar(pd.DataFrame(r["rows"]), x="모델", y="이상%", color="모델",
                     color_discrete_sequence=palette)
        bar.update_layout(height=340, showlegend=False, yaxis_title="이상 탐지 비율(%)")
        st.plotly_chart(bar, use_container_width=True)
        _note("라벨이 없어 지표 비교는 생략됩니다. 모델별 이상 탐지 비율을 비교합니다. "
              "라벨 컬럼이 있으면 F1·ROC/PR로 정량 비교됩니다.")

    # 비교 결과에서 최고/원하는 모델을 바로 활성으로 지정
    paths = r.get("paths", [])
    names = [row["모델"] for row in r["rows"]]
    if paths:
        best_idx = 0
        if r["has_label"]:
            best_idx = int(np.argmax([(row.get("F1") or 0) for row in r["rows"]]))
        a1, a2 = st.columns([3, 1])
        pick = a1.selectbox("활성으로 지정할 모델 (기본=최고 성능)", list(range(len(paths))),
                            index=best_idx, format_func=lambda i: names[i], key="cmp_activate_pick")
        if a2.button("✅ 활성으로 지정", use_container_width=True):
            st.session_state["active_model_dir"] = paths[pick]
            st.session_state["model_select"] = paths[pick]
            st.session_state.pop("thr_slider", None)
            st.rerun()
        st.caption("지정하면 상단 '학습 결과'·'단건 수동 판정'과 사이드바 선택이 그 모델로 바뀝니다.")


def render_comparison(models_dir, df, label_col):
    """저장된 모델들을 현재 데이터로 비교한다(호환 모델만)."""
    models = REG.list_models(models_dir)
    if len(models) < 2:
        st.info("저장된 모델이 2개 이상일 때 비교할 수 있습니다. (좌측에서 학습을 더 진행하세요)")
        return
    compat = []
    for m in models:
        ok, _ = _compat(m["path"], df.columns)
        if ok:
            compat.append(m)
    st.caption(f"전체 {len(models)}개 중 현재 데이터와 **호환 {len(compat)}개**. 호환 모델만 비교 대상입니다.")
    if len(compat) < 2:
        st.warning("현재 업로드 데이터와 호환되는 모델이 2개 미만입니다. "
                   "같은 컬럼 구성의 데이터를 올리거나 해당 데이터로 학습하세요.")
        return
    paths = [m["path"] for m in compat]
    labels = {m["path"]: _model_label(m) for m in compat}
    sel = st.multiselect("비교할 모델 선택", paths, default=paths[:min(3, len(paths))],
                         format_func=lambda p: labels[p], key="cmp_select")
    if st.button("📊 비교 실행", disabled=len(sel) < 2, type="primary"):
        with st.spinner("선택 모델을 현재 데이터로 실행 중..."):
            _run_comparison(sel, df, label_col)
    _render_comparison_result()


# ----------------------------- 단건 수동 판정 -----------------------------

def render_manual(detector):
    """센서 값을 직접 입력해 즉시 정상/이상을 판정한다(CSV 불필요)."""
    st.caption("활성 모델로 센서 값을 직접 입력해 즉시 판정합니다. 기본값은 정상 학습데이터 평균입니다.")
    if detector.model_type == "lstm":
        st.info("시계열(LSTM) 모델은 단건 입력을 지원하지 않습니다(연속 구간이 필요). "
                "행 단위(Dense) 모델을 선택하세요.")
        return

    feats = detector.schema.feature_columns
    means = getattr(detector.scaler, "mean_", None)
    scales = getattr(detector.scaler, "scale_", None)
    ncol = min(len(feats), 4)
    cols = st.columns(ncol)
    vals = {}
    for i, f in enumerate(feats):
        default = float(means[i]) if means is not None else 0.0
        step = float(scales[i] / 10) if scales is not None else 0.1
        vals[f] = cols[i % ncol].number_input(f, value=round(default, 4),
                                               step=round(step, 4), format="%.4f",
                                               key=f"manual_{f}")

    if st.button("🔎 판정", type="primary"):
        row = pd.DataFrame([vals])
        try:
            res = detector.predict(row)
        except ValueError as e:
            st.error(f"판정 실패: {e}")
            return
        err = float(res.errors[0])
        pred = int(res.predictions[0])
        thr = float(detector.threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("판정", "이상 ⚠️" if pred == 1 else "정상 ✅")
        m2.metric("이상 점수", f"{err / thr:.2f}", help="1.0 초과 = 이상")
        m3.metric("재구성 오차", f"{err:.5f}", help=f"임계값 {thr:.5f}")
        pfe = res.per_feature_error[0]
        dd = pd.DataFrame({"feature": feats, "squared_error": pfe}) \
            .sort_values("squared_error", ascending=False)
        fig = px.bar(dd, x="squared_error", y="feature", orientation="h",
                     color_discrete_sequence=[ANOM_C if pred else NORMAL_C])
        fig.update_layout(height=260, yaxis=dict(autorange="reversed"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        _note("오차가 큰 피처가 정상 패턴에서 벗어난 센서입니다. 이상 판정 시 해당 센서를 우선 점검하세요.")


# ----------------------------- 메인 -----------------------------

def _model_label(m: dict) -> str:
    t = "LSTM" if m["model_type"] == "lstm" else "Dense"
    s = f"{m['name']} · {t}"
    if m.get("metrics") and m["metrics"].get("f1") is not None:
        s += f" · F1 {m['metrics']['f1']:.2f}"
    if m.get("tags"):
        s += " · 🏷 " + ",".join(m["tags"])
    return s


def main():
    st.title("🔧 PHM 오토인코더 기반 이상탐지")

    cfg = load_config()
    label_col = cfg["data"].get("label_column")
    ts_col = cfg["data"].get("timestamp_column")
    exclude = list(cfg["data"].get("exclude_columns", []))
    models_dir = os.path.join(ROOT, cfg["paths"].get("models_dir", "models"))
    os.makedirs(models_dir, exist_ok=True)
    REG.prune_incomplete(models_dir)  # 중단된 학습의 빈 항목 정리

    # ---- 업로드(메인 상단) ----
    uploaded = st.file_uploader("CSV 업로드 (학습·판정용)", type=["csv"])
    _reset_if_new_upload(uploaded)
    df, feature_cols, td, recommend_ts = None, [], None, False
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV 읽기 실패: {e}")
            df = None
    if df is not None:
        feature_cols = infer_feature_columns(df, exclude)
        td = EDA.time_dependency(df, feature_cols, ts_col) if (ts_col in df.columns) else None
        recommend_ts = bool(td and td["recommend"])

    # ---- 사이드바: 실행 버튼(항상 상단·좌측) + 모델 레지스트리 + 샘플 ----
    with st.sidebar:
        st.header("▶ 실행")
        disabled = df is None
        eda_clicked = st.button("🔍 EDA", use_container_width=True, disabled=disabled)
        train_clicked = st.button("🚀 학습", type="primary", use_container_width=True, disabled=disabled)
        if disabled:
            st.caption("먼저 CSV를 업로드하세요.")
        elif td is not None:
            (st.warning if recommend_ts else st.success)(
                f"⏱️ 시간 의존성 {'높음' if recommend_ts else '낮음'} (|acf1|={td['mean_abs_acf1']:.2f})")
        st.text_input("새 모델 이름(선택)", key="model_name_input", placeholder="예: 2월_정상라인A")
        st.divider()

        st.header("📁 저장된 모델")
        models = REG.list_models(models_dir)
        if not models:
            st.caption("아직 없음 — 학습하면 여기에 저장됩니다.")
        else:
            paths = [m["path"] for m in models]
            labels = {m["path"]: _model_label(m) for m in models}
            if st.session_state.get("model_select") not in paths:
                act = st.session_state.get("active_model_dir")
                st.session_state["model_select"] = act if act in paths else paths[0]
            chosen = st.selectbox("사용할 모델", paths, key="model_select",
                                  format_func=lambda p: labels[p])
            st.session_state["active_model_dir"] = chosen
            msel = next(m for m in models if m["path"] == chosen)
            cap = (f"학습 {msel['created_at'][:19]} · 피처 {msel['n_features']} · "
                   f"샘플 {(msel['n_samples'] or 0):,}")
            if msel["model_type"] == "lstm" and msel.get("window"):
                cap += f" · 윈도우 {msel['window']}"
            st.caption(cap)
            if msel.get("tags"):
                st.caption("🏷 " + ", ".join(msel["tags"]))
            if msel.get("memo"):
                st.caption("📝 " + msel["memo"])
            if df is not None:
                ok, missing = _compat(chosen, df.columns)
                if ok:
                    st.caption("✅ 업로드 데이터와 **호환**")
                else:
                    more = "…" if len(missing) > 3 else ""
                    st.caption(f"⚠️ 데이터 **불일치** (누락: {', '.join(missing[:3])}{more})")

            mid = os.path.basename(chosen)
            with st.expander("✏️ 이름·메모·태그 편집"):
                en = st.text_input("이름", value=msel["name"], key=f"edit_name_{mid}")
                em = st.text_area("메모", value=msel.get("memo", ""),
                                  key=f"edit_memo_{mid}", height=68)
                et = st.text_input("태그(쉼표로 구분)", value=", ".join(msel.get("tags", [])),
                                   key=f"edit_tags_{mid}")
                if st.button("💾 저장", use_container_width=True, key=f"edit_save_{mid}"):
                    tags = [t.strip() for t in et.split(",") if t.strip()]
                    REG.update_entry(chosen, name=(en.strip() or msel["name"]),
                                     memo=em.strip(), tags=tags)
                    st.success("저장되었습니다.")
                    st.rerun()

            if st.button("🗑 선택 모델 삭제", use_container_width=True):
                REG.delete_model(chosen)
                for k in ("model_select", "active_model_dir", "_last_active"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.divider()

        st.header("📄 샘플 데이터")
        for fname, capd in [("sample_sensor.csv", "행 단위(정상+이상+label)"),
                            ("sample_timeseries.csv", "시계열(자기상관 강함) → LSTM 데모")]:
            p = os.path.join(ROOT, "samples", fname)
            if os.path.exists(p):
                with open(p, "rb") as fp:
                    st.download_button(f"⬇ {fname}", fp.read(), file_name=fname,
                                       mime="text/csv", use_container_width=True)
                st.caption(capd)

    if df is None:
        st.info("샘플을 내려받아 업로드하거나 보유 CSV를 올리세요. "
                "업로드 후 **좌측 상단 'EDA'·'학습'** 버튼으로 실행합니다. "
                "좌측 **'저장된 모델'** 에서 과거 학습 모델을 선택할 수도 있습니다.")
        st.stop()

    # ---- 메인 컨트롤 프레임: 미리보기 + 학습 설정 ----
    with st.container(border=True):
        pv, meta_col = st.columns([3, 1])
        pv.dataframe(df.head(), use_container_width=True, height=180)
        meta_col.metric("행", f"{len(df):,}")
        meta_col.metric("열", f"{df.shape[1]:,}")
        tcfg = cfg["train"]
        with st.expander("⚙️ 학습 설정 (선택)"):
            s = st.columns(4)
            set_epochs = s[0].number_input("epoch", 10, 500, int(tcfg["epochs"]), step=10)
            set_bottleneck = s[1].number_input("병목 차원", 2, 64, int(cfg["model"]["bottleneck"]))
            set_pct = s[2].number_input("임계값 백분위수(%)", 90.0, 99.9,
                                        float(cfg["threshold"].get("percentile", 99.0)), step=0.5)
            set_lr = s[3].number_input("learning rate", 0.0001, 0.1,
                                       float(tcfg["learning_rate"]), step=0.0001, format="%.4f")
            st.caption("병목이 크면 이상까지 복원해 탐지가 약해집니다. 백분위수↑ = 보수적 판정.")
        settings = {"epochs": set_epochs, "bottleneck": set_bottleneck,
                    "percentile": set_pct, "lr": set_lr}

    # 클릭 처리 / 팝업 / 학습 예약
    if eda_clicked:
        st.session_state["show_eda"] = True
    if train_clicked:
        if recommend_ts:
            ts_dialog(default_window=20)
        else:
            st.session_state["pending_train"] = {"model_type": "dense", "window": None}

    # ===== EDA 결과 프레임 =====
    st.markdown("#### 🔍 EDA 결과")
    with st.container(border=True, height=560):
        if st.session_state.get("show_eda"):
            render_eda(df, cfg, td)
        else:
            st.info("좌측 상단 **'EDA'** 버튼을 누르면 이 영역에 데이터 분석 결과가 표시됩니다.")

    # ===== 학습 결과 프레임 =====
    st.markdown("#### 🚀 학습 결과 및 분석")
    with st.container(border=True, height=720):
        pend = st.session_state.pop("pending_train", None)
        if pend:
            entry = train_and_store(df, cfg, models_dir, label_col, settings,
                                    pend["model_type"], pend["window"],
                                    st.session_state.get("model_name_input", ""))
            if entry:
                st.session_state["active_model_dir"] = entry
                st.session_state["model_select"] = entry
                st.rerun()

        active = st.session_state.get("active_model_dir")
        if st.session_state.get("_last_active") != active:
            st.session_state.pop("thr_slider", None)
            st.session_state["_last_active"] = active
        detector = get_detector(active) if (active and os.path.isdir(active)) else None

        if detector is None:
            st.info("좌측 상단 **'학습'** 으로 새 모델을 만들거나, **'저장된 모델'** 에서 선택하세요.")
        else:
            meta = load_meta(active)
            name = REG.read_entry_info(active).get("name", os.path.basename(active))
            kind = "LSTM-AE(시계열)" if detector.model_type == "lstm" else "Dense AE(행 단위)"
            st.success(f"현재 모델: **{name}** · {kind} · 학습시각 {meta.get('created_at', '')[:19]}")
            try:
                render_training_quality(detector, df, label_col)
                st.divider()
                render_results(detector, df, label_col)
            except ValueError as e:
                st.error(f"선택한 모델과 업로드 데이터의 스키마가 맞지 않습니다.\n\n{e}\n\n"
                         f"같은 컬럼 구성의 CSV를 올리거나 새로 학습하세요.")

    # ===== 단건 수동 판정 프레임 =====
    st.markdown("#### ⌨️ 단건 수동 판정")
    with st.container(border=True):
        _active = st.session_state.get("active_model_dir")
        _det = get_detector(_active) if (_active and os.path.isdir(_active)) else None
        if _det is None:
            st.info("먼저 학습하거나 좌측 **'저장된 모델'** 에서 선택하면 단건 판정이 가능합니다.")
        else:
            render_manual(_det)

    # ===== 모델 비교 프레임 =====
    st.markdown("#### 📊 모델 비교")
    with st.container(border=True, height=620):
        render_comparison(models_dir, df, label_col)


if __name__ == "__main__":
    main()
