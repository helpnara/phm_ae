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
import tempfile
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
from src.data import infer_feature_columns, load_meta
from src import eda as EDA

st.set_page_config(page_title="PHM 이상탐지", page_icon="🔧", layout="wide")

NORMAL_C = "#2c7fb8"
ANOM_C = "#d7301f"


def _note(text: str) -> None:
    st.markdown(
        f"<div style='background:rgba(44,127,184,0.08);border-left:3px solid {NORMAL_C};"
        f"padding:8px 12px;border-radius:4px;font-size:0.86rem;color:inherit;margin:2px 0 10px'>"
        f"💡 <b>해석 가이드</b><br>{text}</div>", unsafe_allow_html=True)


def _reset_if_new_upload(uploaded) -> None:
    sig = None if uploaded is None else (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:
        st.session_state["upload_sig"] = sig
        for k in ("detector", "train_note", "show_eda", "pending_train", "thr_slider"):
            st.session_state.pop(k, None)


def _zip_dir(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(path)):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                z.write(fp, arcname=name)
    buf.seek(0)
    return buf.getvalue()


def train_and_store(df, cfg, artifacts_dir, label_col, settings,
                    model_type="dense", window=None) -> bool:
    from tensorflow import keras
    from src.train import train_from_dataframe

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

    min_rows = int(window or 20) + 20 if model_type == "lstm" else 20
    if len(train_df) < min_rows:
        st.error(f"학습 데이터가 부족합니다(현재 {len(train_df)}건, 최소 {min_rows}건).")
        return False

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
        train_from_dataframe(train_df, cfg, artifacts_dir=artifacts_dir,
                             extra_callbacks=[_P()], verbose=0,
                             data_source="streamlit upload",
                             model_type=model_type, window=window)
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"학습 실패: {e}")
        return False

    bar.progress(1.0, text="학습 완료 ✓")
    det = AnomalyDetector(artifacts_dir)
    st.session_state["detector"] = det
    st.session_state["train_note"] = note
    st.session_state["thr_slider"] = float(det.threshold)
    return True


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


# ----------------------------- 메인 -----------------------------

def main():
    st.title("🔧 PHM 오토인코더 기반 이상탐지")

    cfg = load_config()
    label_col = cfg["data"].get("label_column")
    ts_col = cfg["data"].get("timestamp_column")
    exclude = list(cfg["data"].get("exclude_columns", []))
    if "artifacts_dir" not in st.session_state:
        st.session_state["artifacts_dir"] = tempfile.mkdtemp(prefix="phm_art_")
    artifacts_dir = st.session_state["artifacts_dir"]

    with st.sidebar:
        st.header("샘플 데이터")
        for fname, cap in [("sample_sensor.csv", "행 단위(정상+이상+label)"),
                           ("sample_timeseries.csv", "시계열(자기상관 강함) → LSTM 데모")]:
            p = os.path.join(ROOT, "samples", fname)
            if os.path.exists(p):
                with open(p, "rb") as fp:
                    st.download_button(f"⬇ {fname}", fp.read(), file_name=fname,
                                       mime="text/csv", use_container_width=True)
                st.caption(cap)
        st.divider()
        st.caption("업로드 → (EDA 검토) → 학습 → 결과 확인 순으로 진행하세요.")

    # ===== 상단 컨트롤 프레임 =====
    with st.container(border=True):
        st.markdown("#### 1️⃣ 데이터 업로드 & 실행")
        uploaded = st.file_uploader("CSV 업로드 (학습·판정용)", type=["csv"])
        _reset_if_new_upload(uploaded)
        if uploaded is None:
            st.info("샘플을 내려받아 업로드하거나, 보유 CSV를 올리세요. "
                    "업로드하면 미리보기와 'EDA'·'학습' 버튼이 나타납니다.")
            st.stop()
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV 읽기 실패: {e}")
            st.stop()

        feature_cols = infer_feature_columns(df, exclude)
        td = EDA.time_dependency(df, feature_cols, ts_col) if (ts_col in df.columns) else None
        recommend_ts = bool(td and td["recommend"])

        pv, ctrl = st.columns([3, 1.5])
        with pv:
            st.dataframe(df.head(), use_container_width=True, height=180)
        with ctrl:
            b1, b2 = st.columns(2)
            eda_clicked = b1.button("🔍 EDA", use_container_width=True)
            train_clicked = b2.button("🚀 학습", type="primary", use_container_width=True)
            if td is not None:
                if recommend_ts:
                    st.warning(f"⏱️ 시간 의존성 높음(|acf1|={td['mean_abs_acf1']:.2f}) → 학습 시 모델 선택")
                else:
                    st.success(f"⏱️ 시간 의존성 낮음(|acf1|={td['mean_abs_acf1']:.2f}) → Dense 적합")
            st.caption(f"행 {len(df):,} · 열 {df.shape[1]}")

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

    # 버튼 처리 / 팝업 / 학습 예약
    if eda_clicked:
        st.session_state["show_eda"] = True
    if train_clicked:
        if recommend_ts:
            ts_dialog(default_window=20)          # 팝업 → pending_train 설정 후 rerun
        else:
            st.session_state["pending_train"] = {"model_type": "dense", "window": None}

    # ===== EDA 결과 프레임 =====
    st.markdown("#### 2️⃣ EDA 결과")
    with st.container(border=True, height=560):
        if st.session_state.get("show_eda"):
            render_eda(df, cfg, td)
        else:
            st.info("상단 **'EDA'** 버튼을 누르면 이 영역에 데이터 분석 결과가 표시됩니다.")

    # ===== 학습 결과 프레임 =====
    st.markdown("#### 3️⃣ 학습 결과 및 분석")
    with st.container(border=True, height=720):
        pend = st.session_state.pop("pending_train", None)
        if pend:
            train_and_store(df, cfg, artifacts_dir, label_col, settings,
                            model_type=pend["model_type"], window=pend["window"])
        detector = st.session_state.get("detector")
        if detector is None:
            st.info("상단 **'학습'** 버튼을 누르면 이 영역에 학습 품질·판정 결과가 표시됩니다.")
        else:
            if st.session_state.get("train_note"):
                st.success("학습 완료 · " + st.session_state["train_note"])
            render_training_quality(detector, df, label_col)
            st.divider()
            render_results(detector, df, label_col)


if __name__ == "__main__":
    main()
