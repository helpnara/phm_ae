"""판정 결과 HTML 리포트 생성.

브라우저에서 열거나 인쇄(PDF 저장) 가능한 자기완결형 HTML을 만든다.
한글은 CSS 폰트 스택으로 처리하므로 별도 폰트 설치가 필요 없다.
(matplotlib 미사용 → 폰트 깨짐 이슈 없음)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

_FONT = ("-apple-system, 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', "
         "'맑은 고딕', sans-serif")


def build_html_report(summary: pd.DataFrame, meta: Dict[str, Any],
                      threshold: float, n_anom: int,
                      metrics: Optional[Dict[str, Any]] = None,
                      top_features: Optional[pd.DataFrame] = None) -> str:
    """판정 요약을 담은 HTML 문자열을 반환한다."""
    n = len(summary)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_type = meta.get("model_type", "dense")

    metric_html = ""
    if metrics:
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        metric_html = f"""
        <h2>평가지표</h2>
        <table>
          <tr><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
          <tr><td>{metrics['precision']:.3f}</td><td>{metrics['recall']:.3f}</td>
              <td>{metrics['f1']:.3f}</td>
              <td>{'-' if metrics['roc_auc'] is None else f"{metrics['roc_auc']:.3f}"}</td>
              <td>{'-' if metrics['pr_auc'] is None else f"{metrics['pr_auc']:.3f}"}</td></tr>
        </table>
        <p class="cap">혼동행렬 [[TN, FP], [FN, TP]] = {cm}</p>
        """

    topf_html = ""
    if top_features is not None and len(top_features) > 0:
        rows = "".join(
            f"<tr><td>{r.feature}</td><td>{r.mean_squared_error:.6f}</td></tr>"
            for r in top_features.itertuples()
        )
        topf_html = f"""
        <h2>이상 원인 — 피처별 평균 기여도</h2>
        <table><tr><th>feature</th><th>mean squared error</th></tr>{rows}</table>
        """

    table_html = summary.head(200).to_html(index=False, border=0, justify="center")

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>PHM 이상탐지 리포트</title>
<style>
  body {{ font-family: {_FONT}; margin: 32px; color: #222; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-top: 24px; border-bottom: 2px solid #2c7fb8; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; }}
  th {{ background: #f0f6fb; }}
  .kpi {{ display: inline-block; margin-right: 28px; }}
  .kpi b {{ font-size: 22px; color: #2c7fb8; display: block; }}
  .cap {{ color: #777; font-size: 12px; }}
</style></head><body>
  <h1>🔧 PHM 오토인코더 이상탐지 리포트</h1>
  <p class="cap">생성 {now} · 모델 {model_type} · 학습시각 {meta.get('created_at', '-')}</p>
  <div>
    <span class="kpi">전체 샘플<b>{n:,}</b></span>
    <span class="kpi">이상 탐지<b>{n_anom:,} ({n_anom / max(n,1) * 100:.1f}%)</b></span>
    <span class="kpi">적용 임계값<b>{threshold:.5f}</b></span>
  </div>
  {metric_html}
  {topf_html}
  <h2>샘플별 판정 결과 (상위 200행)</h2>
  {table_html}
  <p class="cap">전체 결과는 CSV 다운로드를 이용하세요. 이 페이지는 브라우저 인쇄로 PDF 저장이 가능합니다.</p>
</body></html>"""
