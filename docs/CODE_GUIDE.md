# 코드 가이드 (개발자 문서)

이 프로젝트의 **소스코드 구조와 동작 원리**를 설명합니다.
코드를 직접 읽으며 공부하거나, 기능을 추가·수정할 때 참고하세요.

> 이 파일이 **원본**입니다. 앱의 `🧩 코드 가이드` 메뉴는 이 파일을 그대로 읽어 보여줍니다.
> 코드를 고치면 이 문서도 함께 갱신하세요.

**읽는 순서 추천**: ① 아키텍처 → ② 실행 흐름 → ③ 모듈 레퍼런스(관심 모듈만) → ④ 데이터 계약

---

## 1. 아키텍처 개요

### 설계 원칙
1. **`src/`는 UI를 모른다** — 모든 로직은 Streamlit 없이 동작합니다. `app/`만 화면을 그립니다.
   → CLI·배치·다른 UI에서도 그대로 재사용 가능하고, 테스트가 쉽습니다.
2. **아티팩트가 계약이다** — 학습 결과(모델·스케일러·임계값·스키마)를 파일로 저장하고,
   추론은 그 파일만 읽습니다. 학습 코드와 추론 코드가 분리됩니다.
3. **설정은 YAML로** — 하이퍼파라미터·컬럼 규칙을 `config/default.yaml`에서 읽습니다.

### 레이어 구조

```
┌──────────────────────────────────────────────────────────┐
│  app/          화면 (Streamlit)                           │
│    app.py         메뉴 라우팅 + 5개 페이지 + 렌더 함수     │
│    help_page.py   도움말                                  │
│    code_page.py   이 문서를 렌더링                        │
└───────────────────────┬──────────────────────────────────┘
                        │ import (한 방향)
┌───────────────────────▼──────────────────────────────────┐
│  src/          로직 (UI 의존 없음)                        │
│                                                           │
│   [데이터]   data.py · generate_synthetic.py · eda.py     │
│   [모델]     model.py · train.py · windowing.py           │
│   [판정]     detector.py · scoring.py                     │
│   [평가]     evaluate.py · protocol.py                    │
│   [운영]     monitoring.py · registry.py · report.py      │
│   [연동]     db.py · batch_infer.py (사내 Oracle)         │
│   [공통]     config.py                                    │
└───────────────────────┬──────────────────────────────────┘
                        │ 파일 입출력
┌───────────────────────▼──────────────────────────────────┐
│  artifacts / models/<id>/   모델·스케일러·임계값·스키마    │
└──────────────────────────────────────────────────────────┘
```

### 모듈 의존 관계 (핵심만)

```
train.py ──> data.py (전처리·저장)
         ──> model.py (AE 구성)
         ──> scoring.py (오차·임계값)
         ──> windowing.py (LSTM일 때)

detector.py ──> data.py (아티팩트 로드·검증)
            ──> scoring.py (오차 계산)
            ──> windowing.py (LSTM일 때)

app.py ──> 위 전부 + eda / protocol / monitoring / registry / report
```

> **원칙**: `src` 안에서는 상위(app)를 절대 import하지 않습니다. 순환 참조를 막습니다.

---

## 2. 실행 흐름 따라가기

코드를 처음 읽는다면 이 순서로 따라가면 전체가 보입니다.

### A. 학습 (모델 생성)

```
[화면] app.py: page_train()  →  train_and_store()
   │
   └─> src/train.py: train_from_dataframe(df, cfg, ...)
         │
         ├─ 1. data.fit_preprocess(df, cfg)
         │      · 수치형 피처 선별(infer_feature_columns)
         │      · 결측 대치값 계산 → Schema에 저장
         │      · 상수(분산 0) 컬럼 제거
         │      · StandardScaler.fit_transform  ← 여기서만 fit!
         │      → (X, schema, scaler)
         │
         ├─ 2. 학습/보정 분리  ★중요
         │      calib_frac = cfg.threshold.calibration_split (기본 0.2)
         │      fit_x = 앞 80% / cal_x = 뒤 20%
         │      (LSTM은 행을 먼저 나눈 뒤 각각 windowing → 시퀀스 누수 방지)
         │
         ├─ 3. model.build_autoencoder() 또는 build_lstm_autoencoder()
         │      model.fit(fit_x, fit_x)   ← 입력=출력 (자기 자신을 복원)
         │
         ├─ 4. 임계값 산출  ★중요
         │      cal_x로 예측 → scoring.reconstruction_error()
         │      → scoring.compute_threshold(errors, cfg)
         │      (학습에 쓰지 않은 데이터라 낙관적 편향이 없음)
         │
         └─ 5. data.save_artifacts(...) + history.json + baseline_errors.json
                → models/<id>/ 에 저장
```

### B. 판정 (추론)

```
[화면] app.py: render_results() 등
   │
   └─> src/detector.py: AnomalyDetector(artifacts_dir)      ← 아티팩트 로드
         └─ .predict(df)
              │
              ├─ 1. data.validate_and_transform(df, schema, scaler)
              │      · 스키마 검증(피처 컬럼 존재 여부) → 없으면 SchemaValidationError
              │      · 학습 때 저장한 대치값으로 결측 채움
              │      · scaler.transform  ← fit 금지! (데이터 누수 방지)
              │
              ├─ 2. model.predict(X) → 복원값
              │
              ├─ 3. scoring.reconstruction_error(X, X_hat)   샘플별 오차
              │      scoring.per_feature_squared_error(...)   피처별 오차(원인 진단용)
              │      (LSTM이면 seq_window_errors + windowing.map_windows_to_rows)
              │
              └─ 4. predictions = (errors >= threshold)
                    → DetectionResult
```

**이 두 흐름만 이해하면 나머지는 부가 기능입니다.**

---

## 3. 모듈 레퍼런스

### 📁 공통

#### `src/config.py`
```python
load_config(path=None) -> dict      # config/default.yaml 로드
```
모든 모듈이 이 dict를 받아 동작합니다. 하드코딩된 값이 없도록 하는 것이 목적입니다.

---

### 📁 데이터

#### `src/data.py` — 전처리와 아티팩트 입출력 (**가장 중요한 모듈**)

```python
@dataclass
class Schema:                       # 학습 시점의 데이터 규격 = 추론 시 검증 기준
    feature_columns: List[str]      # 실제 학습에 쓴 피처 (상수 컬럼 제거 후)
    label_column: Optional[str]
    timestamp_column: Optional[str]
    impute_strategy: str            # "median" | "mean" | "drop"
    impute_values: Dict[str, float] # 피처별 대치값 ← 추론에서 재사용(누수 방지)

infer_feature_columns(df, exclude_columns) -> List[str]
    # exclude를 뺀 '수치형' 컬럼만 피처로. 문자열 컬럼은 자동 제외됨.

fit_preprocess(df, cfg) -> (X, Schema, StandardScaler)
    # 학습 전용. 여기서만 scaler.fit / 대치값 계산 / 상수 컬럼 제거.

validate_and_transform(df, schema, scaler) -> np.ndarray
    # 추론 전용. 스키마 검증 → 저장된 대치값 적용 → scaler.transform(fit 아님).
    # 누락 컬럼이 있으면 SchemaValidationError.

save_artifacts(dir, schema, scaler, threshold, meta)
load_schema/load_scaler/load_threshold/load_meta(dir)
```

**왜 이렇게?** — 학습에서 계산한 통계(평균·표준편차·대치값)를 추론에서 다시 계산하면
**데이터 누수**입니다. 그래서 계산은 `fit_preprocess`에만, 사용은 `validate_and_transform`에만 둡니다.

#### `src/generate_synthetic.py` — 테스트 데이터 생성
```python
build_tabular_df(n, anomaly_ratio, seed)        # 단순 행 단위
build_timeseries_df(n, anomaly_ratio, seed)     # AR(1) 자기상관 강함 → LSTM 데모용
build_scenario_df(n, anomaly_ratio, seed,       # ★현실 시나리오
                  difficulty, n_modes, degradation, missing_rate)
    # difficulty: 이상 크기(σ 배수). DIFFICULTY 딕셔너리 참조
    # n_modes: 운전 모드 수(다봉 정상 분포)
    # degradation: 후반부 점진적 열화
    # fault_type 컬럼으로 이상 유형(스파이크/과열/상관구조 붕괴) 기록
generate(...) / generate_timeseries(...)        # CLI용(파일로 저장)
```

#### `src/eda.py` — 탐색적 분석 계산 (순수 pandas, 차트 없음)
```python
basic_info / missing_frame / describe_frame / constant_columns
outlier_frame(df, cols)                 # IQR 1.5배 기준 이상치 비율
label_balance(df, label_col)            # 클래스 불균형비
correlation(df, cols)
contamination_warnings(df, cols, label) # ★정상 데이터 오염 경고
time_dependency(df, cols, ts_col)       # ★자기상관 → LSTM 권장 여부 판단
sample_for_plot(df, max_rows=5000)      # 대용량 플롯용 다운샘플
```
> 계산과 렌더링을 분리했기 때문에, 같은 함수를 배치 리포트에서도 쓸 수 있습니다.

---

### 📁 모델

#### `src/model.py` — 신경망 구조 정의
```python
build_autoencoder(n_features, cfg) -> keras.Model
    # 대칭 Dense AE: input → hidden[] → bottleneck → reversed(hidden[]) → output
    # cfg.model: hidden_layers, bottleneck, activation, dropout, l2

build_lstm_autoencoder(n_features, window, cfg) -> keras.Model
    # 입력 (window, n_features)
    # LSTM(hidden) → LSTM(latent) → RepeatVector(window) →
    #   LSTM(latent) → LSTM(hidden) → TimeDistributed(Dense)
```
**병목(bottleneck)이 핵심 하이퍼파라미터**입니다. 크면 이상까지 잘 복원해 탐지가 약해집니다.

#### `src/windowing.py` — 시계열 윈도우 변환
```python
make_windows(X, window) -> (n-window+1, window, F)   # 슬라이딩(stride=1)
map_windows_to_rows(win_values, n_rows, window)      # 윈도우 결과 → 행 단위 매핑
    # k번째 윈도우는 행 (window-1+k)에서 끝나므로 그 행에 할당.
    # 앞쪽 window-1개 행은 첫 윈도우 값으로 채움(경계 처리).
```

#### `src/train.py` — 학습 파이프라인
```python
train_from_dataframe(df, cfg, artifacts_dir=None, extra_callbacks=None,
                     verbose=2, data_source="dataframe",
                     model_type="dense", window=None) -> str
    # CLI와 웹앱이 공유하는 핵심 함수.
    # extra_callbacks: Keras 콜백 주입(웹앱의 진행률 바가 이걸 사용)
    # 반환: 아티팩트 저장 경로

train(data_path=None, config_path=None, sql=None) -> str   # CLI 진입점
set_seeds(seed)                                            # 재현성
```
**저장물**: `model.keras`, `scaler.pkl`, `schema.json`, `threshold.json`, `meta.json`,
`history.json`(손실곡선), `baseline_errors.json`(드리프트 PSI/KS용 기준 분포)

---

### 📁 판정

#### `src/scoring.py` — 오차·임계값 계산
```python
reconstruction_error(X, X_hat) -> (n,)             # 샘플별 평균 제곱오차
per_feature_squared_error(X, X_hat) -> (n, F)      # 피처별(원인 진단)
seq_window_errors(Xw, Xw_hat) -> (win_err, per_feature)   # LSTM용

compute_threshold(errors, cfg) -> dict             # percentile | mean_std
threshold_for_far(normal_errors, target_far)       # 오경보율 → 임계값 역산
far_for_threshold(normal_errors, threshold)        # 임계값 → 오경보율
cost_optimal_threshold(y_true, errors, cost_fn, cost_fp)  # 비용 최소 임계값 + 곡선
```

#### `src/detector.py` — 추론기
```python
@dataclass
class DetectionResult:
    errors: np.ndarray              # (n,)   샘플별 재구성 오차
    predictions: np.ndarray         # (n,)   0=정상, 1=이상
    per_feature_error: np.ndarray   # (n, F) 피처별 오차
    feature_columns / threshold / threshold_detail
    .summary_frame(df, ts_col)      # 결과 표(오차·점수·판정·top_feature)
    .top_features(k)                # 이상 샘플의 평균 기여도 상위 피처

class AnomalyDetector:
    __init__(artifacts_dir)         # 모델·스케일러·스키마·임계값·baseline 로드
    .predict(df, threshold=None) -> DetectionResult
    .model_type / .window / .baseline_errors
```
`threshold=None`이면 저장된 임계값을, 값을 주면 그 값으로 판정합니다(화면 슬라이더용).

---

### 📁 평가

#### `src/evaluate.py` — 표준 지표
```python
compute_metrics(y_true, errors, predictions) -> dict
    # precision/recall/f1/roc_auc/pr_auc/accuracy/confusion_matrix
best_f1_threshold(y_true, scores)      # PR 곡선에서 F1 최대점
roc_curve_points / pr_curve_points     # 곡선 좌표 + AUC/AP
```

#### `src/protocol.py` — 평가 프로토콜 (**현장 관점 지표**)
```python
time_based_split(df, ts_col, test_frac=0.3) -> (train_df, test_df)
    # 시간순 정렬 후 앞=과거(학습) / 뒤=미래(평가). 무작위 분할보다 보수적.

episodes(y) -> [(start, end), ...]     # 연속된 이상 구간
episode_metrics(y_true, y_pred, timestamps) -> dict
    # episode_recall(구간을 잡았는가), median/max latency(얼마나 빨리),
    # false_alarm_rate(정상 구간 오경보)
per_type_recall(y_true, y_pred, fault_type) -> DataFrame
```
> 샘플 단위 F1만으로는 "고장을 놓쳤는지"를 알 수 없어 만든 모듈입니다.

---

### 📁 운영

#### `src/monitoring.py` — 드리프트 감시
```python
windowed_metrics(errors, predictions, timestamps, n_windows, freq)
    # freq: "h"/"d"/"w" 리샘플, 없으면 등간격/순서 구간
assess_drift(method, cur_errors, base_mean, base_std, baseline_errors, k)
    # method: "zscore" | "psi" | "ks"  → {metric, value, status(정상/주의/경고)}
psi(expected, actual)          # 0.1↑ 주의, 0.25↑ 경고
ks_stat(expected, actual)      # scipy ks_2samp → (통계량, p값)
alarm_events(predictions, k=3, m=5)   # K/M 디바운싱 → 알람 발생 횟수
baseline_from_errors(errors)          # 기준선 재설정
```

#### `src/registry.py` — 모델 레지스트리
```python
new_entry_dir(models_dir, model_type, name)   # models/<타임스탬프_타입_이름_uuid>/
list_models(models_dir) -> [ {id, path, name, created_at, model_type,
                              n_features, metrics, memo, tags}, ... ]
read_entry_info / write_entry_info / update_entry   # entry.json (이름·메모·태그·지표)
delete_model / prune_incomplete                     # 중단된 빈 항목 정리
export_all(models_dir) -> bytes                     # 전체 zip (사내 이관)
import_zip(models_dir, data) -> int                 # 경로 탈출 방지 포함
```

#### `src/report.py`
```python
build_html_report(summary, meta, threshold, n_anom, metrics, top_features) -> str
    # 자기완결형 HTML(한글 폰트 스택). matplotlib 미사용 → 폰트 깨짐 없음.
```

---

### 📁 사내 연동 (스켈레톤)

#### `src/db.py` — Oracle 어댑터
```python
get_dsn()      # 환경변수 PHM_DB_USER / PHM_DB_PASSWORD / PHM_DB_DSN
get_engine()   # SQLAlchemy oracle+oracledb (thin 모드 기본)
read_sql(query, params)          # 바인드 변수 사용(인젝션 방지)
write_results(df, table)
```
#### `src/batch_infer.py` — 배치 추론
```python
run(sql, table, config_path)   # DB 조회 → AnomalyDetector.predict → 결과 적재
```
> 실제 접속은 사내 DB 확정 후. 자세한 내용은 `docs/migration_notes.md`.

---

### 📁 화면 (`app/app.py`)

Streamlit 특성상 **한 파일 = 한 앱**이라 길지만, 구조는 단순합니다.

```python
main()                      # 사이드바 메뉴 → 페이지 라우팅
├─ page_eda()               # 1. 데이터 준비 + EDA
├─ page_train()             # 2. 학습
├─ page_eval()              # 3. 평가 (5가지 모드)
├─ page_monitor()           # 4. 모니터링
└─ help_page / code_page    # 도움말 · 코드 가이드

# 재사용 렌더 함수
render_eda / render_training_quality / render_results /
render_comparison / render_protocol_metrics / render_time_split_validation /
render_threshold_policy / render_manual / render_model_registry
```

**Streamlit을 읽을 때 알아둘 점**
- 위젯을 조작하면 **스크립트 전체가 위에서 다시 실행**됩니다.
- 값을 유지하려면 `st.session_state`에 저장합니다
  (`data_df`, `active_model_dir`, `cmp_result` 등).
- 무거운 로드는 `@st.cache_resource`로 캐시합니다(`get_detector`).
- `@st.dialog`은 팝업(LSTM 선택), `st.data_editor`는 편집 가능한 표입니다.

---

## 4. 데이터 계약 (아티팩트 구조)

`models/<id>/` 한 폴더가 **모델 하나의 완전한 세트**입니다.

| 파일 | 내용 | 만든 곳 → 쓰는 곳 |
|---|---|---|
| `model.keras` | 학습된 신경망 | train → detector |
| `scaler.pkl` | StandardScaler(평균·표준편차) | data.fit_preprocess → validate_and_transform |
| `schema.json` | 피처 컬럼·대치값·라벨/시간 컬럼명 | data → detector(검증) · 호환성 배지 |
| `threshold.json` | 임계값 + 산출 방식 + `calibrated` 여부 | scoring → detector |
| `meta.json` | 학습 시각·모델 타입·윈도우·샘플 수·설정 스냅샷 | train → 화면 표시 |
| `history.json` | 에폭별 loss/val_loss | train → 손실곡선 |
| `baseline_errors.json` | 보정 데이터의 오차 분포(최대 2000개) | train → PSI/KS 드리프트 |
| `entry.json` | 이름·메모·태그·평가지표 | registry → 목록 표시 |

> **이 폴더만 복사하면 다른 서버에서 그대로 판정할 수 있습니다.**
> 화면의 "학습 아티팩트 zip 다운로드"가 이 폴더를 묶은 것입니다.

---

## 5. 설계 결정과 이유

코드를 읽다가 "왜 이렇게 했지?" 싶은 부분들입니다.

| 결정 | 이유 |
|---|---|
| **임계값을 보정용 데이터에서 산출** | 학습 데이터로 산출하면 오차가 낙관적으로 작아 임계값이 과소 설정되고, 운영에서 오경보가 급증합니다. (실측 개선: 2.9% → 1.4%) |
| **스케일러를 아티팩트로 저장** | 추론에서 다시 fit하면 데이터 누수. 학습 시점의 기준을 그대로 써야 판정이 일관됩니다. |
| **스키마 검증** | 컬럼이 다른 CSV가 들어오면 조용히 잘못된 결과를 내는 대신, 명시적으로 오류를 냅니다. |
| **피처별 오차를 항상 계산** | "이상이다"만으로는 현장에서 못 씁니다. 어느 센서인지 알려줘야 점검이 가능합니다. |
| **LSTM은 행을 먼저 분리 후 윈도잉** | 윈도우를 먼저 만들면 학습·보정 구간이 겹쳐 누수가 생깁니다. |
| **`src`가 UI를 모름** | 배치·CLI·다른 UI 재사용 + 단위 테스트 가능. |
| **에피소드 지표 별도 구현** | 샘플 F1은 "고장 구간을 놓쳤는지"를 알려주지 못합니다. |
| **모델을 폴더 단위로 저장** | 모델·전처리·임계값이 따로 놀면 재현이 깨집니다. 한 세트로 묶어 버전 관리합니다. |

---

## 6. 확장 가이드

### 새 드리프트 판정 방식 추가
1. `src/monitoring.py`에 계산 함수 작성 (예: `wasserstein(...)`)
2. `assess_drift()`에 `method` 분기 추가 — 반환 dict 형태(`method/metric/value/status/text`) 유지
3. `app/app.py`의 `page_monitor()` 안 `method_map`에 항목 추가

### 새 모델 구조 추가 (예: Conv1D-AE)
1. `src/model.py`에 `build_conv_autoencoder(...)` 작성
2. `src/train.py`의 `train_from_dataframe()`에서 `model_type` 분기 추가
   (필요하면 입력 변환도 — `windowing.py` 참고)
3. `src/detector.py`의 `predict()`에 대응 분기 추가
4. `meta.json`에 `model_type`이 저장되므로 detector가 자동으로 알아봅니다

### 새 평가지표 추가
- 표준 지표 → `src/evaluate.py`
- 현장 관점(구간·지연) → `src/protocol.py`
- 화면 표시 → `app/app.py`의 `render_protocol_metrics()` 등

### 새 페이지(메뉴) 추가
1. `app/app.py` 상단 `MENU_*` 상수 추가
2. `main()`의 `st.radio(...)` 목록과 라우팅 분기에 추가
3. 내용이 길면 `app/새페이지.py`로 분리하고 import (도움말·코드 가이드처럼)

### 설정값 추가
`config/default.yaml`에 키를 넣고 `cfg["섹션"]["키"]`로 읽습니다. **코드에 상수를 박지 마세요.**

---

## 7. 개발·테스트

```bash
pip install -r requirements.txt          # 기본
pip install -r requirements-db.txt       # 사내 Oracle 연동 시 추가

streamlit run app/app.py                 # 웹앱
python -m src.generate_synthetic         # 샘플 데이터 생성
python -m src.train --data data/normal.csv   # CLI 학습
```

**로직만 빠르게 확인하기** (UI 없이 — `src`가 독립적이라 가능):
```python
import pandas as pd
from src.config import load_config
from src.train import train_from_dataframe
from src.detector import AnomalyDetector
from src.evaluate import compute_metrics

cfg = load_config()
df = pd.read_csv("samples/sample_sensor.csv")
path = train_from_dataframe(df[df.label == 0], cfg, artifacts_dir="/tmp/m1", verbose=0)

det = AnomalyDetector(path)
res = det.predict(df)
print(compute_metrics(df["label"].values, res.errors, res.predictions))
```

**코딩 규칙**
- 한글 주석(주변 코드와 동일한 밀도), 타입 힌트 사용
- `src`에 `streamlit` import 금지
- 새 파일은 모듈 상단에 **"이 파일이 무엇을 하는가"** docstring

---

## 8. 관련 문서

| 문서 | 내용 |
|---|---|
| `README.md` | 프로젝트 개요·실행 |
| `docs/requirements.md` | 요구사항 정의서(왜 이 시스템인가) |
| `docs/TESTING.md` | 기능 테스트 가이드 |
| `docs/migration_notes.md` | 사내 서버·Oracle 이관 |
| `CHANGELOG.md` / `ROADMAP.md` | 변경 이력 / 진행 현황·계획 |
| `CLAUDE.md` | 개발 시 지켜야 할 핵심 원칙 |
