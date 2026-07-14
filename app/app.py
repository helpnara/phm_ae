"""PHM 오토인코더 이상탐지 — Streamlit 웹 테스트 앱.

실행:
    streamlit run app/app.py

흐름:
  ① CSV 업로드 → ② 오른쪽 '학습' 버튼 클릭(진행률 표시) → ③ 학습 후 이상 판정
  - 초기 로딩 시에는 학습하지 않는다(업로드 후 버튼으로만 학습 시작).
  - 라벨(label) 컬럼이 있으면 정상(label=0)만 학습하고 전체를 판정·평가한다.
  - 라벨이 없으면 업로드 데이터를 정상으로 간주해 학습 후 동일 데이터를 판정한다.
"""
from __future__ import annotations

import os
import sys
import tempfile

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


def _reset_if_new_upload(uploaded) -> None:
    """업로드 파일이 바뀌면(또는 제거되면) 이전 학습 상태를 초기화한다."""
    sig = None if uploaded is None else (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:
        st.session_state["upload_sig"] = sig
        st.session_state.pop("detector", None)
        st.session_state.pop("train_note", None)
        st.session_state.pop("show_eda", None)


def train_and_store(df: pd.DataFrame, cfg: dict, artifacts_dir: str,
                    label_col: str | None) -> bool:
    """업로드 데이터로 학습하고 진행률을 표시한 뒤 detector를 세션에 저장한다."""
    from tensorflow import keras
    from src.train import train_from_dataframe

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

    st.caption(note)
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


def render_results(detector: AnomalyDetector, df: pd.DataFrame,
                   label_col: str | None) -> None:
    """학습 완료 후 판정 결과·시각화·평가지표·결과표를 렌더링한다."""
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

    # 요약 지표
    n = len(summary)
    n_anom = int(result.predictions.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 샘플", f"{n:,}")
    c2.metric("이상 탐지", f"{n_anom:,}", f"{n_anom / n * 100:.1f}%")
    c3.metric("적용 임계값", f"{thr:.5f}")

    # 재구성 오차 그래프
    st.subheader("재구성 오차")
    x_axis = summary[ts_col] if (ts_col and ts_col in summary.columns) else summary.index
    fig = go.Figure()
    normal_mask = result.predictions == 0
    fig.add_trace(go.Scatter(
        x=x_axis[normal_mask], y=result.errors[normal_mask],
        mode="markers", name="정상", marker=dict(color="#2c7fb8", size=5),
    ))
    fig.add_trace(go.Scatter(
        x=x_axis[~normal_mask], y=result.errors[~normal_mask],
        mode="markers", name="이상", marker=dict(color="#d7301f", size=7, symbol="x"),
    ))
    fig.add_hline(y=thr, line_dash="dash", line_color="#e6550d",
                  annotation_text="임계값", annotation_position="top left")
    fig.update_layout(height=400, xaxis_title="샘플", yaxis_title="재구성 오차",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # 피처별 기여도 (원인 진단)
    st.subheader("이상 원인 — 피처별 평균 기여도 (상위)")
    top = result.top_features(k=len(result.feature_columns))
    bar = px.bar(top, x="mean_squared_error", y="feature", orientation="h")
    bar.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, use_container_width=True)

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

    # 결과 표 & 다운로드
    st.subheader("샘플별 판정 결과")
    st.dataframe(summary, use_container_width=True, height=300)
    st.download_button(
        "결과 CSV 다운로드",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="anomaly_result.csv",
        mime="text/csv",
    )


def render_eda(df: pd.DataFrame, cfg: dict) -> None:
    """업로드 데이터의 통계량·EDA 결과를 렌더링한다(학습 전 사전 검토용)."""
    data_cfg = cfg["data"]
    exclude = list(data_cfg.get("exclude_columns", []))
    label_col = data_cfg.get("label_column")
    ts_col = data_cfg.get("timestamp_column")
    feature_cols = infer_feature_columns(df, exclude)

    info = EDA.basic_info(df, feature_cols, label_col, ts_col)
    plot_df = EDA.sample_for_plot(df)

    st.header("🔍 EDA · 데이터 분석 결과")

    # 1) 데이터 개요
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("행(row)", f"{info['n_rows']:,}")
    c2.metric("열(col)", f"{info['n_cols']:,}")
    c3.metric("피처 수", f"{info['n_features']:,}")
    c4.metric("총 결측", f"{info['total_missing']:,}")
    c5.metric("중복 행", f"{info['n_duplicates']:,}")
    st.caption(f"메모리 사용량 ≈ {info['memory_mb']:.2f} MB · "
               f"피처: {', '.join(feature_cols) if feature_cols else '없음'}")

    if not feature_cols:
        st.error("수치형 피처 컬럼이 없습니다. exclude_columns 설정 또는 데이터를 확인하세요.")
        return

    # 2) 클래스 분포(라벨 있을 때) — 불균형 확인
    if info["has_label"]:
        st.subheader("클래스 분포 (label)")
        bal = EDA.label_balance(df, label_col)
        lc1, lc2, lc3 = st.columns([1, 1, 2])
        lc1.metric("정상(0)", f"{bal['n_normal']:,}")
        lc2.metric("이상(1)", f"{bal['n_anomaly']:,}")
        ratio = bal["imbalance_ratio"]
        lc3.metric("불균형비 (정상:이상)",
                   "∞ : 1" if ratio == float("inf") else f"{ratio:.1f} : 1")
        st.caption("불균형 데이터이므로 정확도(accuracy)가 아닌 F1·PR-AUC로 평가합니다.")
        fig = px.bar(bal["frame"], x="label", y="count", text="count",
                     color="label", color_discrete_map={"0": "#2c7fb8", "1": "#d7301f"})
        fig.update_layout(height=280, showlegend=False, xaxis_title="label", yaxis_title="건수")
        st.plotly_chart(fig, use_container_width=True)

    # 3) 결측치
    st.subheader("결측치 점검")
    miss = EDA.missing_frame(df)
    if len(miss) == 0:
        st.success("결측치가 없습니다.")
    else:
        st.warning(f"결측이 있는 컬럼 {len(miss)}개 — 학습 시 설정된 대치 전략으로 처리됩니다.")
        st.dataframe(miss, use_container_width=True, hide_index=True)

    # 4) 기초 통계량
    st.subheader("기초 통계량")
    desc = EDA.describe_frame(df, feature_cols)
    st.dataframe(desc.style.format(precision=3), use_container_width=True, hide_index=True)

    # 5) 데이터 품질 — 상수 컬럼 / 이상치(IQR)
    st.subheader("데이터 품질 점검")
    const_cols = EDA.constant_columns(df, feature_cols)
    if const_cols:
        st.warning(f"상수(분산 0) 컬럼: {const_cols} → 학습 시 자동 제거됩니다.")
    else:
        st.caption("상수 컬럼 없음.")
    out = EDA.outlier_frame(df, feature_cols)
    st.caption("IQR(1.5×) 기준 피처별 이상치 비율 — 값이 높으면 데이터 오염/센서 이상 가능성.")
    st.dataframe(out, use_container_width=True, hide_index=True)

    # 6) 분포 (히스토그램)
    st.subheader("피처 분포")
    show_feats = feature_cols[:12]
    if len(feature_cols) > 12:
        st.caption(f"피처가 많아 상위 12개만 표시합니다(전체 {len(feature_cols)}개).")
    ncol = 3
    for i in range(0, len(show_feats), ncol):
        cols = st.columns(ncol)
        for j, f in enumerate(show_feats[i:i + ncol]):
            with cols[j]:
                h = px.histogram(plot_df, x=f, nbins=40, color_discrete_sequence=["#2c7fb8"])
                h.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                                showlegend=False, title=dict(text=f, font=dict(size=13)))
                st.plotly_chart(h, use_container_width=True)

    # 7) 상관관계 히트맵
    corr = EDA.correlation(df, feature_cols)
    if not corr.empty:
        st.subheader("피처 상관관계")
        heat = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                         color_continuous_scale="RdBu_r", aspect="auto")
        heat.update_layout(height=420)
        st.plotly_chart(heat, use_container_width=True)

    # 8) 시계열 추이 (timestamp 있을 때)
    if info["has_timestamp"]:
        st.subheader("시계열 추이")
        sel = st.selectbox("피처 선택", feature_cols, key="eda_ts_feature")
        tdf = plot_df[[ts_col, sel]].sort_values(ts_col)
        line = px.line(tdf, x=ts_col, y=sel)
        line.update_traces(line_color="#2c7fb8")
        line.update_layout(height=320)
        st.plotly_chart(line, use_container_width=True)
        if len(df) > len(plot_df):
            st.caption(f"대용량 대비 {len(plot_df):,}건으로 샘플링해 표시(통계량은 전체 기준).")


def main() -> None:
    st.title("🔧 PHM 오토인코더 기반 이상탐지")
    st.caption(
        "정상 데이터로 오토인코더를 학습하고, 재구성 오차가 임계값을 넘으면 이상으로 판정합니다. "
        "CSV 업로드 → 오른쪽 '학습' 버튼 순으로 진행하세요."
    )

    cfg = load_config()
    label_col = cfg["data"].get("label_column")

    # 세션별 아티팩트 저장 위치(멀티 유저 격리)
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
            st.caption("내려받아 아래 업로드 칸에 넣고 '학습'을 눌러보세요. "
                       "(정상+이상+label 포함)")

    # 파일 업로드
    uploaded = st.file_uploader("CSV 업로드 (학습·판정용)", type=["csv"])
    _reset_if_new_upload(uploaded)

    if uploaded is None:
        st.info("① CSV를 업로드하면 → ② 오른쪽에 '학습' 버튼이 나타납니다. "
                "버튼을 누르면 학습(진행률 표시) 후 이상 판정을 진행합니다.")
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

    # EDA는 한 번 켜지면 유지(재실행에도 표시)
    if eda_clicked:
        st.session_state["show_eda"] = True
    # 학습 실행
    if train_clicked:
        train_and_store(df, cfg, artifacts_dir, label_col)

    # ---- EDA 결과 ----
    if st.session_state.get("show_eda"):
        st.divider()
        render_eda(df, cfg)

    # ---- 학습 결과 ----
    detector = st.session_state.get("detector")
    if detector is None:
        if not st.session_state.get("show_eda"):
            st.info("**'EDA'** 로 데이터를 먼저 검토하거나, **'학습'** 버튼으로 바로 학습을 시작하세요.")
        else:
            st.info("EDA 검토 후 **'학습'** 버튼을 눌러 학습을 시작하세요.")
        st.stop()

    st.divider()
    if st.session_state.get("train_note"):
        st.success("학습 완료 · " + st.session_state["train_note"])
    render_results(detector, df, label_col)


if __name__ == "__main__":
    main()
