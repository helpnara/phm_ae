# 코드 가이드 (개발자 문서)

이 프로젝트의 **소스코드 구조와 동작 원리**를 설명합니다.
코드를 직접 읽으며 공부하거나, 기능을 추가·수정할 때 참고하세요.

> 이 파일이 **원본**입니다. 앱의 `🧩 코드 가이드` 메뉴는 이 파일을 그대로 읽어 보여줍니다.
> 코드를 고치면 이 문서도 함께 갱신하세요.

**읽는 순서 추천**
- **처음 코드를 볼 때**: ① 아키텍처 → ② 실행 흐름 → ④ 데이터 계약
- **원리가 궁금할 때**: ⑤ 오토인코더 이론과 수식 (수식 ↔ 코드 대응 표시)
- **화면 코드를 볼 때**: ⑥ Streamlit 동작 방식 (재실행 모델·session_state·캐시)
- **고칠 때**: ③ 모듈 레퍼런스 → ⑦ 설계 이유 → ⑧ 확장 가이드

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

## 5. 오토인코더 이론과 수식

코드가 **무엇을 계산하는지** 수식으로 정리하고, 각 수식이 어느 코드에 대응하는지 표시합니다.

### 5.1 오토인코더의 정의

오토인코더는 **인코더** $f_\theta$ 와 **디코더** $g_\phi$ 로 이루어진 신경망입니다.

$$z = f_\theta(x), \qquad \hat{x} = g_\phi(z) = g_\phi(f_\theta(x))$$

- $x \in \mathbb{R}^d$ : 입력 (센서 $d$개의 한 시점 값)
- $z \in \mathbb{R}^k$ : **잠재 표현(latent)**, $k \ll d$ ← 이 $k$가 **병목(bottleneck)**
- $\hat{x} \in \mathbb{R}^d$ : 복원값

목표는 **자기 자신을 복원**하는 것입니다. 학습 손실은 평균제곱오차(MSE):

$$\mathcal{L}(\theta,\phi) = \frac{1}{N}\sum_{i=1}^{N} \lVert x_i - g_\phi(f_\theta(x_i)) \rVert_2^2$$

> **코드**: `src/model.py: build_autoencoder()` 가 $f,g$ 구조를 만들고,
> `src/train.py` 의 `model.compile(loss="mse")` + `model.fit(fit_x, fit_x)` 가 위 손실을 최소화합니다.
> `fit(x, x)` 처럼 **입력과 정답이 같은 것**이 오토인코더의 특징입니다.

이 프로젝트의 기본 구조 (`config/default.yaml`: `hidden_layers=[32,16]`, `bottleneck=8`):

```
x(d=5) → Dense32 → Dense16 → Dense8(z) → Dense16 → Dense32 → x̂(5)
         └────── encoder f ──────┘      └────── decoder g ──────┘
```

### 5.2 왜 이것이 이상 탐지가 되는가

핵심은 **병목이 정보를 통과시키지 못한다**는 점입니다. $k < d$ 이므로 AE는 입력을 그대로
복사할 수 없고, **정상 데이터에서 반복되는 구조**(센서 간 상관, 전형적 범위)만 압축해 학습합니다.

기하학적으로 보면, AE는 정상 데이터가 놓인 **저차원 다양체(manifold)** $\mathcal{M}$ 를 근사하고,
복원은 그 위로의 **사영(projection)** 과 비슷하게 동작합니다.

$$\hat{x} \approx \text{proj}_{\mathcal{M}}(x)$$

- 정상 $x$ 는 이미 $\mathcal{M}$ 위에 있으므로 $\lVert x-\hat{x}\rVert$ 이 작습니다.
- 이상 $x$ 는 $\mathcal{M}$ 에서 떨어져 있어 사영 거리가 크고 → 오차가 큽니다.

즉 재구성 오차는 **"정상 패턴으로부터의 거리"** 를 근사한 값입니다.
고장을 배운 적이 없어도 탐지가 되는 이유가 여기 있습니다.

### 5.3 재구성 오차와 판정

**샘플별 오차** (피처 방향 평균):

$$e_i = \frac{1}{d}\sum_{j=1}^{d}\left(x_{ij} - \hat{x}_{ij}\right)^2$$

**피처별 오차** (원인 진단용):

$$e_{ij} = \left(x_{ij} - \hat{x}_{ij}\right)^2$$

**판정**: $\hat{y}_i = \mathbb{1}[\, e_i \ge \tau \,]$ , **이상 점수** $= e_i / \tau$ (1을 넘으면 이상)

> **코드**: `src/scoring.py: reconstruction_error()` = $e_i$,
> `per_feature_squared_error()` = $e_{ij}$, 판정은 `src/detector.py: predict()`.
> 화면의 "피처별 기여도" 막대가 $e_{ij}$ 를 정렬한 것입니다.

### 5.4 임계값 $\tau$ 와 오경보율

정상 데이터의 오차 분포를 $F_{\text{normal}}$ 이라 하면, 백분위수 방식은

$$\tau = F^{-1}_{\text{normal}}(1-\alpha)$$

여기서 $\alpha$ 가 **허용 오경보율(FAR)** 입니다. `percentile: 99.0` 은 $\alpha=0.01$ 을 뜻하고,
"정상의 1%는 이상으로 오판한다"는 **설계상의 약속**입니다.

$$\text{FAR}(\tau) = P(e \ge \tau \mid \text{정상}) \;\approx\; \frac{1}{N}\sum_i \mathbb{1}[e_i \ge \tau]$$

> **코드**: `compute_threshold()`(산출), `threshold_for_far()`($\alpha \to \tau$ 역산),
> `far_for_threshold()`($\tau \to \alpha$).

**⚠️ 여기가 이 프로젝트에서 가장 중요한 이론적 함정입니다.**
$F_{\text{normal}}$ 을 **학습에 사용한 데이터**로 추정하면, 모델이 그 데이터를 이미 잘 복원하므로
$e_i$ 가 낙관적으로 작습니다. 그러면 $\tau$ 가 과소 설정되고 **실제 FAR $\gg \alpha$** 가 됩니다.

$$\mathbb{E}[e \mid \text{학습에 쓴 정상}] \;<\; \mathbb{E}[e \mid \text{처음 보는 정상}]$$

그래서 정상 데이터를 학습용/**보정용**으로 나누고 보정용으로만 $\tau$ 를 추정합니다
(`threshold.calibration_split`). 실측으로 FAR이 2.9% → 1.4%로 목표 1%에 근접했습니다.

**비용 기반 임계값**은 미탐/오탐 비용을 반영해 총비용을 최소화합니다:

$$\tau^{*} = \arg\min_{\tau}\; \left[\, c_{FN}\cdot \text{FN}(\tau) + c_{FP}\cdot \text{FP}(\tau) \,\right]$$

> **코드**: `cost_optimal_threshold()`. $c_{FN}$ 이 커지면 $\tau^{*}$ 가 낮아집니다(민감해짐).

### 5.5 정규화가 필수인 이유

MSE는 **스케일에 민감**합니다. rpm(≈1500)과 vibration(≈2)을 그대로 쓰면
rpm의 오차가 손실을 지배해 진동 이상을 놓칩니다. 그래서 표준화합니다:

$$\tilde{x}_{j} = \frac{x_{j}-\mu_j}{\sigma_j}$$

$\mu_j, \sigma_j$ 는 **학습 정상 데이터에서만** 추정하고 저장합니다(`scaler.pkl`).
추론에서 다시 추정하면 **데이터 누수**이며, 판정 기준이 데이터마다 달라져 일관성이 깨집니다.

> **코드**: `data.fit_preprocess()`(추정+변환) / `data.validate_and_transform()`(변환만).

### 5.6 병목 크기의 trade-off

| 병목 $k$ | 결과 |
|---|---|
| 너무 작음 | 정상조차 복원 못 함 → 정상 오차↑ → **오경보 증가** |
| 적정 | 정상만 잘 복원, 이상은 복원 실패 → **분리 최대** |
| 너무 큼 | 항등함수 $g\circ f \approx I$ 에 가까워져 **이상까지 복원** → 탐지 실패 |

$k \ge d$ 이면 이론적으로 완전한 항등 사상이 가능해 탐지 능력이 사라집니다.
정규화(L2·Dropout)는 이를 완화하는 보조 수단입니다.

> **코드**: `config/default.yaml: model.bottleneck`, 화면의 "병목 차원" 설정.

### 5.7 LSTM 오토인코더 (시계열)

행 단위 AE는 각 시점을 **독립적으로** 봅니다. 값이 시간에 걸쳐 이어지는 데이터에서는
"값 자체는 정상 범위인데 **변화 패턴**이 이상한" 경우를 놓칩니다.

LSTM-AE는 **윈도우 시퀀스**를 입력으로 받습니다.
$X_t = (x_{t-W+1},\dots,x_t) \in \mathbb{R}^{W\times d}$

$$z = \text{LSTM}_{enc}(X_t), \qquad \hat{X}_t = \text{LSTM}_{dec}(\text{RepeatVector}_W(z))$$

$$e_t = \frac{1}{W d}\sum_{w=1}^{W}\sum_{j=1}^{d}\left(x_{t-W+w,j}-\hat{x}_{t-W+w,j}\right)^2$$

`RepeatVector`는 하나의 잠재벡터 $z$ 를 $W$번 복제해 디코더가 시퀀스를 재생성하게 합니다.

> **코드**: `src/model.py: build_lstm_autoencoder()`,
> `src/windowing.py: make_windows()`(시퀀스 생성) / `map_windows_to_rows()`(윈도우 결과 → 행 매핑),
> `src/scoring.py: seq_window_errors()` = 위 $e_t$.

**시간 의존성 판단**은 자기상관(ACF)으로 합니다:

$$\rho_k = \frac{\sum_t (x_t-\bar{x})(x_{t+k}-\bar{x})}{\sum_t (x_t-\bar{x})^2}$$

피처별 $|\rho_1|$ 의 평균이 임계(0.3) 이상이면 LSTM을 권장합니다.
> **코드**: `src/eda.py: time_dependency()`

### 5.8 드리프트 판정 수식

**z-score** — 평균 이동:
$$z = \frac{\bar{e}_{\text{현재}} - \mu_{\text{기준}}}{\sigma_{\text{기준}}}$$

**PSI** (Population Stability Index) — 분포 이동. 기준 분포의 분위수로 $B$개 구간을 나눈 뒤,
$$\text{PSI} = \sum_{b=1}^{B}\left(a_b - e_b\right)\ln\frac{a_b}{e_b}$$
($e_b$: 기준 비율, $a_b$: 현재 비율). 0.1↑ 주의, 0.25↑ 경고가 통용되는 기준입니다.

**KS 검정** — 두 경험적 분포함수의 최대 거리:
$$D = \sup_x \left| F_{\text{기준}}(x) - F_{\text{현재}}(x) \right|$$
$p<0.05$ 이면 "분포가 다르다"고 판단합니다.

> **코드**: `src/monitoring.py: assess_drift()` / `psi()` / `ks_stat()`

### 5.9 평가지표 수식

$$\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad
F_1 = 2\cdot\frac{P\cdot R}{P+R}$$

불균형 데이터에서 정확도 $\frac{TP+TN}{N}$ 는 무의미합니다(이상 1%면 전부 정상이라 해도 99%).
**PR-AUC**(=Average Precision)가 더 신뢰할 만합니다.

**에피소드 지표** — 연속 이상 구간 $E=[s,e]$ 에 대해
- 검출: $\exists\, t\in E,\ \hat{y}_t=1$
- 지연: $\min\{t\in E:\hat{y}_t=1\} - s$

> **코드**: `src/evaluate.py`(표준), `src/protocol.py: episode_metrics()`(에피소드·지연)

### 5.10 이 방식의 이론적 한계

- **다봉 정상**: 정상이 여러 운전 모드로 나뉘면 단일 다양체 가정이 깨집니다.
  → 모드별 모델 또는 모드를 피처로 투입.
- **일반화의 역설**: AE가 너무 잘 일반화하면 **처음 보는 이상도 복원**해버립니다(병목 문제와 동일).
- **미세 이상**: $\lVert x-\text{proj}_\mathcal{M}(x)\rVert$ 이 정상 오차 분포와 겹치면
  임계값으로 분리할 수 없습니다. (실측: 난이도 '미세'에서 F1 0.73)
- **오차는 방향을 모른다**: $e_i$ 는 크기만 알려주므로 "어떤 고장인지"는 판단하지 못합니다.

---

## 6. Streamlit 동작 방식 (화면 코드를 읽기 위한 배경)

`app/app.py`를 읽을 때 **Streamlit의 실행 모델**을 모르면 코드가 이상해 보입니다.
꼭 알아야 할 것만 정리합니다.

### 6.1 실행 모델 — "스크립트가 통째로 다시 실행된다"

일반 웹 프레임워크와 달리 Streamlit에는 이벤트 핸들러가 없습니다.
**위젯을 건드릴 때마다 `app.py`가 처음부터 끝까지 다시 실행**됩니다(rerun).

```
사용자가 슬라이더 이동
   → app.py 전체 재실행 (import부터 다시)
   → st.slider(...) 가 '새 값'을 반환
   → 그 값으로 아래 코드가 다시 그려짐
```

그래서 이런 코드가 성립합니다:
```python
thr = st.slider("판정 임계값", ...)   # 매 실행마다 현재 값을 "읽어오는" 것
result = detector.predict(df, threshold=thr)   # 그 값으로 다시 계산
```
**버튼도 마찬가지**입니다. `st.button()`은 "눌렸는가"를 **그 실행에서만** True로 반환합니다.
다음 rerun에서는 다시 False가 됩니다.

```python
if st.button("🚀 학습 시작"):    # 누른 그 순간의 실행에서만 True
    ...
```

### 6.2 `st.session_state` — 재실행을 넘어 값을 유지

스크립트가 매번 재실행되므로 **일반 변수는 사라집니다.** 유지하려면 `session_state`에 넣습니다.
브라우저 탭(세션)마다 독립적입니다.

이 프로젝트에서 실제로 쓰는 키:

| 키 | 용도 |
|---|---|
| `data_df` / `data_name` | EDA에서 준비한 데이터(메뉴를 옮겨도 유지) |
| `active_model_dir` / `model_select` | 현재 선택된 모델 경로 |
| `eda_ran` | EDA를 실행했는지(버튼 상태 기억) |
| `pending_train` | 학습 예약 — 팝업에서 선택한 모델 타입/윈도우 |
| `cmp_result` / `tsplit` | 모델 비교·시간분할 검증 결과(재실행해도 유지) |
| `thr_slider` | 임계값 슬라이더 값(모델 바뀌면 자동 보정) |
| `upload_sig` / `_eda_up_sig` | 업로드 파일이 바뀌었는지 감지(파일명+크기) |
| `onboard_hide` / `train_note` | 온보딩 배너 닫기 · 학습 결과 메시지 |

**위젯의 `key`도 session_state에 저장됩니다.** `st.slider(..., key="thr_slider")`로 만들면
`st.session_state["thr_slider"]`로 읽고 **쓸 수도** 있습니다.

```python
# 추천 임계값을 슬라이더에 반영하는 실제 코드 패턴
st.session_state["thr_slider"] = float(rec["threshold"])
st.rerun()      # 즉시 다시 그려서 반영
```

> ⚠️ `key`를 가진 위젯은 같은 `key`가 화면에 두 번 있으면 **DuplicateWidgetID 오류**가 납니다.
> 그래서 모델 편집 UI는 `key=f"edit_name_{mid}"`처럼 **모델 ID를 붙여** 유일성을 확보합니다.

### 6.3 `st.rerun()` — 즉시 재실행

상태를 바꾼 뒤 화면을 곧바로 갱신하려면 호출합니다(이 프로젝트에 9곳).
호출하면 그 지점에서 **실행이 중단되고** 처음부터 다시 시작합니다.

전형적 사용처: 모델 삭제 후 목록 갱신, 학습 완료 후 활성 모델 전환, 팝업 선택 반영.

### 6.4 캐시 — `cache_data` vs `cache_resource`

매번 재실행되므로 무거운 작업은 반드시 캐시해야 합니다.

| 데코레이터 | 대상 | 동작 | 이 프로젝트 사용처 |
|---|---|---|---|
| `@st.cache_resource` | 모델·커넥션 등 **공유 객체** | 원본 객체를 그대로 재사용 | `get_detector()` — TF 모델 로드(수 초) |
| `@st.cache_data` | DataFrame·문자열 등 **데이터** | 복사본 반환(변경해도 안전) | `_model_features()`, `_load_guide()` |

**캐시 키는 함수 인자**입니다. 그래서 파일 갱신을 반영하려면 mtime을 인자로 넘깁니다:

```python
@st.cache_data(show_spinner=False)
def _model_features(path: str, _mtime: float):   # mtime이 바뀌면 캐시 무효화
    return load_schema(path).feature_columns
```
> 인자 이름 앞의 `_`는 "해시하지 말라"는 뜻이 아니라 여기서는 단순 관례입니다.
> (Streamlit에서 `_`로 시작하는 인자는 **해시 대상에서 제외**되므로 주의해서 사용하세요.)

### 6.5 팝업(`st.dialog`)과 "예약" 패턴

`@st.dialog`로 만든 함수를 호출하면 모달이 뜹니다. 하지만 **모달 안에서 바로 학습을 시작하면
안 됩니다** — 모달이 닫히면서 실행이 끊길 수 있고, 진행률 표시도 모달에 갇힙니다.

그래서 이 프로젝트는 **예약 → 재실행 → 본문에서 실행** 패턴을 씁니다:

```python
@st.dialog("⏱️ 시계열 윈도우 모델 선택")
def ts_dialog(default_window):
    ...
    if st.button("이 설정으로 학습 시작"):
        st.session_state["pending_train"] = {"model_type": ..., "window": ...}  # 예약만
        st.rerun()                                                              # 모달 닫고 재실행

# 본문(page_train)에서
pend = st.session_state.pop("pending_train", None)
if pend:
    train_and_store(...)      # 여기서 실제 학습 + 진행률 표시
```

### 6.6 진행률 바 — Keras 콜백과 연결

학습은 `src/train.py`에서 돌지만 진행률은 화면에 그려야 합니다.
`src`가 Streamlit을 모르므로, **콜백을 주입**해 연결합니다.

```python
bar = st.progress(0.0, text="학습 준비 중...")

class _P(keras.callbacks.Callback):          # app 쪽에서 정의
    def on_epoch_end(self, epoch, logs=None):
        bar.progress((epoch+1)/epochs, text=f"학습 중... {epoch+1}/{epochs} epoch ...")

train_from_dataframe(..., extra_callbacks=[_P()])   # src는 콜백만 받아 넘김
```
> 이것이 "`src`는 UI를 모른다" 원칙을 지키면서도 화면과 연동하는 방법입니다.

### 6.7 파일 업로더

`st.file_uploader()`는 업로드된 파일 객체를 **재실행마다 계속 반환**합니다.
그래서 "새로 올렸는지"를 직접 판별해야 중복 처리를 막습니다:

```python
sig = (up.name, up.size)
if st.session_state.get("_eda_up_sig") != sig:   # 파일이 바뀐 경우에만
    st.session_state["_eda_up_sig"] = sig
    _set_data(pd.read_csv(up), up.name)
```
> ⚠️ 한 페이지에 업로더가 여러 개면 DOM 순서가 헷갈립니다.
> (테스트 코드에서 `input[type=file]`을 `.last`로 지정하는 이유)

### 6.8 레이아웃 API (이 프로젝트에서 쓰는 것)

| API | 용도 | 사용처 |
|---|---|---|
| `st.sidebar` | 좌측 메뉴 | 메뉴 라디오·진행 상태 |
| `st.columns([3,1])` | 가로 분할(비율) | KPI·컨트롤 배치 |
| `st.tabs([...])` | 탭 | 도움말 7탭, EDA 업로드/생성 |
| `st.expander()` | 접이식 | 학습 설정, 상세 표 |
| `st.container(border=True, height=..)` | 테두리·내부 스크롤 박스 | 결과 프레임 |
| `st.metric(label, value, delta)` | 큰 숫자 지표 | F1·드리프트 상태 등 |
| `st.data_editor()` | 편집 가능한 표 | 다중행 수동 판정 |
| `st.plotly_chart(fig, use_container_width=True)` | 차트 | 전 화면 |

### 6.9 흔한 함정과 이 프로젝트의 대응

| 함정 | 증상 | 대응 |
|---|---|---|
| 슬라이더 범위 밖 값이 session_state에 남음 | 모델 바꾸면 예외 발생 | `render_results()`에서 범위 벗어나면 기본값으로 자동 보정 |
| 매 재실행마다 모델 로드 | 화면이 느려짐 | `@st.cache_resource`로 캐시 |
| 같은 `key` 중복 | DuplicateWidgetID 오류 | 모델 ID 등 접두어로 유일화 |
| 무거운 계산이 반복 실행 | 조작할 때마다 느림 | 결과를 `session_state`에 저장(`cmp_result`, `tsplit`) |
| 큰 DataFrame을 통째로 플롯 | 렌더링 지연 | `eda.sample_for_plot()`으로 5000행 다운샘플 |

### 6.10 디버깅 팁

```bash
streamlit run app/app.py --logger.level=debug     # 상세 로그
```
- 화면에 값 찍기: `st.write(변수)` (dict·DataFrame도 예쁘게 출력)
- 현재 상태 확인: `st.write(st.session_state)`
- **로직 버그는 UI 없이 잡는 게 빠릅니다** — `src`는 독립적이므로
  7장의 스니펫처럼 파이썬으로 직접 호출해 확인하세요.
- 코드 저장 시 우상단 **Rerun** 또는 `Always rerun` 설정으로 자동 반영됩니다.

---

## 7. 설계 결정과 이유

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

## 8. 확장 가이드

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

## 9. 개발·테스트

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

## 10. 관련 문서

| 문서 | 내용 |
|---|---|
| `README.md` | 프로젝트 개요·실행 |
| `docs/requirements.md` | 요구사항 정의서(왜 이 시스템인가) |
| `docs/TESTING.md` | 기능 테스트 가이드 |
| `docs/migration_notes.md` | 사내 서버·Oracle 이관 |
| `CHANGELOG.md` / `ROADMAP.md` | 변경 이력 / 진행 현황·계획 |
| `CLAUDE.md` | 개발 시 지켜야 할 핵심 원칙 |
