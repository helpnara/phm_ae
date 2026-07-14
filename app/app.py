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

st.set_page_config(page_title="PHM 이상탐지", page_icon="🔧", layout="wide")


def _reset_if_new_upload(uploaded) -> None:
    """업로드 파일이 바뀌면(또는 제거되면) 이전 학습 상태를 초기화한다."""
    sig = None if uploaded is None else (uploaded.name, uploaded.size)
    if st.session_state.get("upload_sig") != sig:
        st.session_state["upload_sig"] = sig
        st.session_state.pop("detector", None)
        st.session_state.pop("train_note", None)


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

    # 미리보기(좌) + 학습 버튼(우)
    left, right = st.columns([4, 1])
    with left:
        st.subheader("입력 데이터 미리보기")
        st.dataframe(df.head(), use_container_width=True)
    with right:
        st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
        train_clicked = st.button("🚀 학습", type="primary", use_container_width=True)
        st.caption(f"행 {len(df):,} · 열 {df.shape[1]}")

    # 학습 실행
    if train_clicked:
        train_and_store(df, cfg, artifacts_dir, label_col)

    # 학습 완료 시 결과 표시
    detector = st.session_state.get("detector")
    if detector is None:
        st.info("오른쪽 **'학습'** 버튼을 눌러 학습을 시작하세요.")
        st.stop()

    st.divider()
    if st.session_state.get("train_note"):
        st.success("학습 완료 · " + st.session_state["train_note"])
    render_results(detector, df, label_col)


if __name__ == "__main__":
    main()
