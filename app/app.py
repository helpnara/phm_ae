"""PHM 오토인코더 이상탐지 — Streamlit 웹 테스트 앱.

실행:
    streamlit run app/app.py

CSV를 업로드하면 저장된 아티팩트(모델/스케일러/임계값/스키마)로 이상 여부를 판정하고,
재구성 오차 시각화·피처별 기여도·평가지표(라벨 있을 때)를 보여준다.
"""
from __future__ import annotations

import os
import sys

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


@st.cache_resource(show_spinner="모델 아티팩트 로딩 중...")
def get_detector(artifacts_dir: str) -> AnomalyDetector:
    """아티팩트가 없으면 데모용으로 합성 데이터를 생성·학습한 뒤 로드한다.

    이 자동 부트스트랩 덕분에 저장소만으로도(모델 파일 커밋 없이) 어디서나 실행/배포된다.
    사내 실데이터 학습 시에는 `python -m src.train`으로 미리 아티팩트를 만들어 두면 된다.
    """
    if not os.path.exists(os.path.join(artifacts_dir, "model.keras")):
        from src.generate_synthetic import generate
        from src.train import train
        with st.spinner("최초 실행: 데모용 합성 데이터로 모델을 학습 중입니다(1~2분 소요)..."):
            os.makedirs("data", exist_ok=True)
            generate(n_normal=3000, n_test=600, anomaly_ratio=0.1, out_dir="data", seed=42)
            train(data_path="data/normal.csv")
    return AnomalyDetector(artifacts_dir)


def main() -> None:
    st.title("🔧 PHM 오토인코더 기반 이상탐지")
    st.caption(
        "정상 데이터로 학습된 오토인코더의 재구성 오차로 이상을 탐지합니다. "
        "오차가 임계값을 넘으면 이상으로 판정합니다."
    )

    cfg = load_config()
    artifacts_dir = cfg["paths"]["artifacts_dir"]

    # 아티팩트가 없으면 데모용 합성 데이터로 자동 학습 후 로드
    detector = get_detector(artifacts_dir)

    # ---- 사이드바 ----
    with st.sidebar:
        st.header("설정")
        # 데모: 업로드할 샘플 CSV 다운로드 (자기 파일이 없는 방문자용)
        sample_path = os.path.join("data", "test.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as fp:
                st.download_button("샘플 test.csv 내려받기", data=fp.read(),
                                   file_name="sample_test.csv", mime="text/csv")
            st.caption("내려받은 파일을 아래 업로드 칸에 넣어 바로 테스트할 수 있습니다.")
            st.divider()
        default_thr = detector.threshold
        st.write(f"저장된 임계값: **{default_thr:.6f}**")
        st.caption(f"방식: {detector.threshold_detail.get('method')}")
        thr = st.slider(
            "판정 임계값 (조정 시 민감도 변화)",
            min_value=float(default_thr) * 0.1,
            max_value=float(default_thr) * 3.0,
            value=float(default_thr),
            step=float(default_thr) * 0.01,
        )
        st.divider()
        st.caption("학습 스키마 피처:")
        st.code("\n".join(detector.schema.feature_columns))

    # ---- 파일 업로드 ----
    uploaded = st.file_uploader("판정할 CSV 업로드", type=["csv"])
    if uploaded is None:
        st.info("CSV 파일을 업로드하면 판정 결과가 표시됩니다.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    st.subheader("입력 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

    # ---- 판정 ----
    try:
        result = detector.predict(df, threshold=thr)
    except ValueError as e:
        st.error(f"스키마 검증 실패:\n\n{e}")
        st.stop()

    ts_col = detector.schema.timestamp_column
    summary = result.summary_frame(df, ts_col)

    # ---- 요약 지표 ----
    n = len(summary)
    n_anom = int(result.predictions.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 샘플", f"{n:,}")
    c2.metric("이상 탐지", f"{n_anom:,}", f"{n_anom / n * 100:.1f}%")
    c3.metric("적용 임계값", f"{thr:.5f}")

    # ---- 재구성 오차 그래프 ----
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

    # ---- 피처별 기여도 (원인 진단) ----
    st.subheader("이상 원인 — 피처별 평균 기여도 (상위)")
    top = result.top_features(k=len(result.feature_columns))
    bar = px.bar(top, x="mean_squared_error", y="feature", orientation="h")
    bar.update_layout(height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(bar, use_container_width=True)

    # ---- 평가지표 (라벨 있을 때) ----
    label_col = detector.schema.label_column
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

    # ---- 결과 표 & 다운로드 ----
    st.subheader("샘플별 판정 결과")
    st.dataframe(summary, use_container_width=True, height=300)
    st.download_button(
        "결과 CSV 다운로드",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="anomaly_result.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
