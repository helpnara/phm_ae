# 변경 이력 (CHANGELOG)

이 문서는 프로젝트의 주요 변경 이력을 날짜별로 정리한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/) 스타일을 따른다.

---

## [2026-07-14] 초기 구축 ~ 웹앱 고도화

PHM 오토인코더 기반 이상탐지 시스템의 문서·파이프라인·웹앱을 하루에 걸쳐 구축하고,
사용자 요청에 따라 단계적으로 고도화했다. (커밋 `da206d0` ~ `acf655e`, 총 11개)

### 1. 문서/설계 (`da206d0`, `232c495`)
- **요구사항 정의서**(`docs/requirements.md`) 작성 — 배경·목표, 추가 고려사항,
  기능/비기능 요구사항, 데이터·모델 설계, 아키텍처, 평가지표, 마일스톤, 리스크.
- **프로젝트 가이드**(`CLAUDE.md`) 작성 — 핵심 설계 원칙(데이터 누수 방지, 정상 순수성,
  임계값 아티팩트화, 과용량 경계, 불균형 평가, 스키마 검증, 설명가능성).
- 기술 스택 확정: **행 단위(tabular) Dense AE · Streamlit · TensorFlow/Keras**.

### 2. 핵심 파이프라인 + 웹앱 구현 (`ea883c3`)
- 합성 데이터 생성(`src/generate_synthetic.py`).
- 전처리·스키마/스케일러 아티팩트·누수 방지 변환(`src/data.py`).
- Dense Autoencoder(`src/model.py`), 학습+임계값+아티팩트 저장(`src/train.py`),
  재구성 오차·임계값(`src/scoring.py`), 추론기(`src/detector.py`).
- 불균형 평가지표 F1/PR-AUC/ROC-AUC/혼동행렬(`src/evaluate.py`).
- Streamlit 웹앱(업로드·시각화·다운로드), 설정 파일, README, .gitignore.
- 검증: 합성 데이터 기준 recall 1.0 / F1 0.92 / PR-AUC 0.999.

### 3. 사내 Oracle DB 연동 준비 (`93333f6`)
- Oracle 어댑터(`src/db.py`, python-oracledb + SQLAlchemy, 환경변수 접속).
- CSV/Oracle 소스 추상화(`load_dataframe`), `train.py --sql`, 배치 추론(`src/batch_infer.py`).
- 사내 이식 노트(`docs/migration_notes.md`)를 Oracle 기준으로 작성
  (DSN·thin/thick·결과 테이블 DDL·오프라인 설치·재학습 체크리스트).

### 4. 배포 준비 (`3567974`, `c0f0de8`, `f35e57c`)
- 실행 화면 스크린샷을 `docs/screenshots/`에 추가하고 README에 연결.
- 앱 자동 부트스트랩(아티팩트 없으면 최초 실행 시 학습) + Streamlit 배포 설정
  (`.streamlit/config.toml`) + 원클릭 배포 배지.
- **Streamlit Cloud 의존성 오류 해결**: `tensorflow-cpu` 사용, 버전 상한 핀 제거,
  Oracle 의존성을 `requirements-db.txt`로 분리. 클린 venv에서 설치·실행 검증.

### 5. 웹 흐름 개편 — 업로드 후 학습 (`93bbc65`)
- 초기 로딩 시 자동 학습 제거. **CSV 업로드 → 우측 '학습' 버튼 → 진행률 바** 흐름.
- 라벨 인식(정상 label=0만 학습, 전체 판정), 세션별 아티팩트 격리, 샘플 CSV 커밋.

### 6. EDA 기능 추가 (`c167452`)
- 업로드 후 **'EDA' 버튼**(학습과 분리). 데이터 개요·클래스 불균형·결측·기초 통계량·
  데이터 품질(상수/IQR 이상치)·분포·상관 히트맵·시계열 추이(`src/eda.py`).

### 7. 웹앱 고도화 4종 + 해석 가이드 (`0e98a0f`)
- **학습 품질 시각화**: 손실곡선, 오차분포+임계값, ROC/PR 커브.
- **EDA 심화**: 정상 vs 이상 피처 분포 비교, 학습 데이터 오염 경고.
- **설정 UI**: epoch·병목·임계 백분위수·learning rate 조정.
- **설명가능성·이관**: 개별 샘플 드릴다운, 학습 아티팩트 zip 다운로드.
- 각 EDA 단계에 **해석 가이드**(5줄 이내, 동적) 추가.

### 8. 추가 4기능 + 시계열 모델 + 레이아웃 개편 (`acf655e`)
- **임계값 자동 추천**: F1 최대 임계값(PR 곡선) 계산 및 원클릭 적용.
- **다중 CSV 배치 판정**: 여러 파일 일괄 판정 요약.
- **HTML 리포트 내보내기**(`src/report.py`): 자기완결형, 한글 폰트, PDF 인쇄 가능.
- **시계열 윈도우 모델 LSTM-AE**(`src/model.py`, `src/windowing.py`,
  `src/scoring.seq_window_errors`): EDA 시간 의존성 진단(`src/eda.time_dependency`)으로
  필요 여부 판단 → **학습 시 팝업**으로 LSTM vs Dense 선택. AR(1) 시계열 샘플 추가.
- **레이아웃 개편**: 상단 고정 컨트롤 프레임 + EDA 결과 프레임 + 학습 결과 프레임(3단).

### 검증 방식
- 각 단계마다 클린 환경/Playwright로 실제 앱을 구동해 렌더링·동작을 확인하고,
  백엔드 함수는 단위 테스트로 검증한 뒤 커밋했다.

### 다음 작업(예정)
- 레이아웃 대안(탭 방식) 검토, 추가 개선 작업은 익일부터 진행.

---

> 표기 규칙: 각 항목 끝의 괄호는 관련 커밋 해시. 상세 diff는 `git show <해시>` 참고.
