# 사내 서버 이식 · Oracle DB 연동 수정사항 노트

> 이 문서는 현재의 **로컬/CSV 기반 프로토타입**을 **사내 서버 + Oracle DB 연동**
> 환경으로 옮길 때 수정해야 할 부분을 정리한 체크리스트다. 이식 작업 시 이 문서를
> 기준으로 항목별로 진행한다. (작성일 2026-07-14, 사내 DB = **Oracle** 전제)
>
> 참고: Oracle 연동을 위한 **스켈레톤 코드가 이미 포함**되어 있다.
> - `src/db.py` — Oracle 접속/조회/적재 어댑터 (환경변수 기반)
> - `src/batch_infer.py` — DB 조회 → 판정 → 결과 DB 적재 배치
> - `src/train.py --sql "..."` — CSV 대신 Oracle 조회로 학습
> - `config/default.yaml`의 `database:` 섹션, `.env.example`
>
> 이 스켈레톤은 **실제 Oracle 없이 로컬 검증 불가**하므로, 사내 이식 시 접속·쿼리·
> 테이블명을 실제 환경에 맞춰 확정해야 한다.

---

## 0. 현재 프로토타입 전제
- 입력: 로컬 CSV 파일 (합성 정상 데이터 `data/normal.csv`로 학습, `data/test.csv`로 테스트)
- 실행: 로컬 Python + Streamlit (`streamlit run app/app.py`)
- 아티팩트: 로컬 `artifacts/` 폴더에 모델/스케일러/임계값/스키마 저장
- 인증/권한/모니터링 없음

이식 목표: **DB에서 데이터를 읽어 학습·추론**하고, 결과를 **DB에 기록**, 사내망에서
**웹으로 접근**. 아래 항목을 순서대로 처리한다.

---

## 1. 데이터 소스: CSV → Oracle 연동 ⭐(최우선)

CSV 읽기 지점은 이미 `src/db.py`(Oracle 어댑터)로 대체 가능하도록 추상화되어 있다.

| 위치 | 현재 | 이식 후 |
|------|------|---------|
| `src/data.py :: load_dataframe()` | `csv_path` 사용 | `sql=` 인자로 Oracle 조회 (내부에서 `src/db.py` 호출) |
| `src/train.py` | `--data <csv>` | `--sql "SELECT ... WHERE STATUS='NORMAL'"` |
| `src/batch_infer.py` | (신규) | Oracle 조회 → 판정 → 결과 Oracle 적재 |
| `app/app.py` | `st.file_uploader` | 업로드 유지 + (선택) "DB에서 최근 N건 조회" 버튼 추가 |

**남은 작업(사내에서 확정)**
1. `config/default.yaml`의 `database.normal_query` / `result_table`을 **실제 테이블·컬럼명**으로 수정.
2. Oracle 접속정보를 환경변수로 주입(§2). `src/db.py`는 `PHM_DB_USER/PASSWORD/DSN`을 읽는다.
3. `python -m src.train --sql "<정상 데이터 조회>"`로 학습.
4. **DataFrame 형태(컬럼=센서, 행=관측)만 맞추면** 이후 전처리/모델/추론 코드는 수정 불필요.

**Oracle 관련 주의**
- **DSN 형식**: `host:1521/service_name` 또는 `tnsnames.ora` 별칭. `PHM_DB_DSN`에 지정.
- **thin vs thick 모드**: `python-oracledb`는 기본 **thin 모드**로 Instant Client 없이 동작한다.
  구형 서버/특수 인증(예: 외부 Kerberos)로 thick가 필요하면 `src/db.py`의
  `oracledb.init_oracle_client(lib_dir=...)` 주석을 해제하고 Instant Client 경로 지정.
- **대소문자**: Oracle 식별자는 기본 대문자. 조회 결과 컬럼명이 대문자로 올 수 있으므로
  `config`의 `exclude_columns`/`label_column`/`timestamp_column`을 실제 반환 컬럼명과 일치시킨다.
- **바인드 변수**: 기간/설비ID 필터는 `db.read_sql(sql, {"eid": ...})` 바인드로(인젝션 방지).
- **정상 구간만 조회**: 이상/정비 구간 제외 조건을 쿼리에 반드시 포함(학습 데이터 오염 방지 — 핵심 원칙).
- DB 컬럼명 = 실제 센서명. 합성 센서명(`temperature, vibration, ...`)과 다르므로 **실데이터 재학습 필수**(§4).

---

## 2. 설정 · 시크릿 관리 (하드코딩 금지)

DB 접속정보, 경로 등은 코드가 아닌 **환경변수/시크릿**으로 주입한다.

- Oracle 접속: 환경변수 `PHM_DB_USER`, `PHM_DB_PASSWORD`, `PHM_DB_DSN` (`.env.example` 참고).
  - `src/db.py`가 이 세 변수를 읽어 SQLAlchemy 엔진(`oracle+oracledb://...`)을 구성한다.
  - 실값이 담긴 `.env`/`secrets.toml`은 `.gitignore`에 등록됨. **절대 커밋 금지.**
- `config/default.yaml`은 하이퍼파라미터/스키마/쿼리 설정용으로 유지, **접속정보(비번)는 넣지 않는다**.
- 사내 배포용 `config/prod.yaml`을 별도로 두고 `--config`로 지정하는 방식 권장.

---

## 3. 의존성 설치 (사내망/오프라인 대응)

사내 PC는 외부 인터넷이 막혀 있을 수 있다.

- 사내 **PyPI 미러**가 있으면 `pip install -i <내부 인덱스> -r requirements.txt`.
- 완전 폐쇄망이면 인터넷 되는 PC에서 `pip download -r requirements.txt -d wheels/` 후
  `wheels/`를 옮겨 `pip install --no-index --find-links wheels/ -r requirements.txt`.
- **TensorFlow**가 가장 큰 의존성. 사내 PC 사양 확인:
  - CPU만 있으면 CPU 빌드로 충분(추론은 가볍다).
  - Python/OS에 맞는 wheel 확보 필요(버전 핀은 `requirements.txt` 참고).
  - TF 설치가 어려우면 추론 전용으로 **ONNX/TFLite 변환** 또는 PyTorch 대체 검토(설계 변경 필요).
- **Oracle 드라이버**: `oracledb`(python-oracledb) + `SQLAlchemy`. **`requirements-db.txt`로
  분리**되어 있음(클라우드 데모 의존성과 격리). 사내 이식 시:
  `pip install -r requirements.txt -r requirements-db.txt`.
  thin 모드는 순수 파이썬에 가까워 오프라인 설치도 용이. Instant Client는 thick 모드에서만 필요(§1).
- **TensorFlow**: 기본 `requirements.txt`는 `tensorflow-cpu`를 사용(추론/소형 AE에 충분, 설치 가벼움).
  사내에 GPU가 있고 학습 가속이 필요하면 `tensorflow[and-cuda]`로 교체.

---

## 4. 실데이터 재학습 (필수)

프로토타입 모델은 합성 데이터 기준이므로 **사내 실데이터로 다시 학습**해야 한다.

1. `config/default.yaml`의 `data.exclude_columns`, `label_column`, `timestamp_column`을
   실제 컬럼명에 맞게 수정.
2. 정상 데이터로 학습 재실행 — Oracle: `python -m src.train --sql "SELECT ... WHERE STATUS='NORMAL'"`
   (로컬 검증은 `--data <csv>`도 가능).
3. 재학습 시 스키마(`artifacts/schema.json`)가 실제 센서명으로 갱신됨 → 웹 검증이 실데이터에 맞음.
4. 피처 수/분포에 따라 `model.hidden_layers`, `bottleneck`, `threshold.percentile` 튜닝.

---

## 5. 아티팩트 저장 위치

- 현재: 로컬 `artifacts/`.
- 이식: 서버의 **영속 경로**(예: `/opt/phm/artifacts`) 또는 공유 스토리지/오브젝트 스토리지.
  - `config`의 `paths.artifacts_dir`만 바꾸면 됨.
- 버전 관리: 재학습마다 `artifacts_YYYYMMDD/`로 분리 저장하고 심볼릭 링크로 current 지정 권장.

---

## 6. 추론 결과의 DB 기록 (배치 서빙)

웹 테스트 외에 **주기적 자동 추론 → 결과 DB 적재** 파이프라인이 필요할 수 있다.
이 흐름은 `src/batch_infer.py`에 이미 구현되어 있다(조회→판정→적재).

```bash
python -m src.batch_infer \
    --sql "SELECT * FROM PHM_SENSOR WHERE PROCESSED = 0" \
    --table PHM_ANOMALY_RESULT
```
- 스케줄링: cron / 사내 스케줄러 / Airflow 등에 위 명령 등록.
- 결과 테이블은 **사전에 생성**되어 있어야 한다. Oracle DDL 예시:

```sql
CREATE TABLE PHM_ANOMALY_RESULT (
    TIMESTAMP              TIMESTAMP,
    EQUIPMENT_ID          VARCHAR2(64),
    RECONSTRUCTION_ERROR  BINARY_DOUBLE,
    THRESHOLD             BINARY_DOUBLE,
    ANOMALY_SCORE         BINARY_DOUBLE,
    PREDICTION            NUMBER(1),
    STATUS                VARCHAR2(32),
    TOP_FEATURE           VARCHAR2(128),
    MODEL_VERSION         VARCHAR2(64)
);
```
- `batch_infer.py`는 원본에 `id`/`equipment_id`/`eq_id` 컬럼이 있으면 결과에 함께 적재한다.
  적재 컬럼명이 위 DDL과 일치하도록 조회/컬럼명을 맞춘다(Oracle 대문자 주의).

---

## 7. 웹 배포 · 접근 제어

- 실행: `streamlit run app/app.py --server.port <포트> --server.address 0.0.0.0`.
- 서비스화: `systemd` 유닛 또는 컨테이너로 상시 구동.
- 리버스 프록시(Nginx 등) 뒤에 두고 사내 인증(SSO/Basic Auth) 연동 권장.
  - Streamlit 자체 인증은 제한적 → 프록시 레벨 인증 또는 사내 게이트웨이 활용.
- 방화벽/포트 정책 확인.

---

## 8. 운영 · 모니터링

- 로깅: 추론 요청/오류/판정 통계를 파일 또는 사내 로깅 시스템으로.
- **Concept drift 모니터링**: 정상 데이터의 재구성 오차 평균이 시간에 따라 상승하면
  정상 운전 상태가 변한 것 → 재학습 트리거. 임계 초과 알림 설계.
- 재학습 주기 정의(예: 월 1회 또는 오차 분포 이탈 시).

---

## 9. 시간대 · 데이터 타입

- DB의 `timestamp` 타임존/포맷 확인 → 시각화 x축, 결과 적재 시 일관성 유지.
- 수치 컬럼에 문자열/NULL이 섞이면 `validate_and_transform`가 대치하지만,
  실데이터의 결측/이상값 정책을 §4 재학습 전에 확정.

---

## 10. 이식 체크리스트 요약
- [x] `src/db.py` Oracle 어댑터 (스켈레톤 포함) — 실쿼리/테이블명만 확정
- [x] `load_dataframe` 로 CSV/Oracle 추상화, `train.py --sql` 지원
- [x] `oracledb` + `SQLAlchemy` 드라이버 `requirements-db.txt`로 분리 등록
- [ ] Oracle 접속정보 환경변수화 (`PHM_DB_USER/PASSWORD/DSN`, 커밋 금지)
- [ ] `config.database.normal_query` / `result_table`를 실제 테이블·컬럼으로 수정
- [ ] thin/thick 모드 결정 (필요 시 Instant Client 경로 설정)
- [ ] 사내망 pip 설치 방식 확정(미러/오프라인 wheel)
- [ ] TensorFlow 설치 검증(CPU/오프라인)
- [ ] `config`의 컬럼 설정을 실데이터에 맞게 수정 후 **실데이터 재학습**
- [ ] `artifacts_dir` 영속 경로로 변경
- [ ] 결과 테이블 DDL 생성(§6) + `batch_infer.py` 스케줄러 등록
- [ ] Streamlit 서비스화 + 리버스 프록시 인증
- [ ] 로깅/드리프트 모니터링/재학습 주기 수립
