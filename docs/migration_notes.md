# 사내 서버 이식 · DB 연동 수정사항 노트

> 이 문서는 현재의 **로컬/CSV 기반 프로토타입**을 **사내 서버 + DB 연동** 환경으로
> 옮길 때 수정해야 할 부분을 정리한 체크리스트다. 이식 작업 시 이 문서를 기준으로
> 항목별로 진행한다. (작성일 2026-07-14)

---

## 0. 현재 프로토타입 전제
- 입력: 로컬 CSV 파일 (합성 정상 데이터 `data/normal.csv`로 학습, `data/test.csv`로 테스트)
- 실행: 로컬 Python + Streamlit (`streamlit run app/app.py`)
- 아티팩트: 로컬 `artifacts/` 폴더에 모델/스케일러/임계값/스키마 저장
- 인증/권한/모니터링 없음

이식 목표: **DB에서 데이터를 읽어 학습·추론**하고, 결과를 **DB에 기록**, 사내망에서
**웹으로 접근**. 아래 항목을 순서대로 처리한다.

---

## 1. 데이터 소스: CSV → DB 연동 ⭐(최우선)

현재 CSV를 읽는 지점을 DB 어댑터로 교체한다.

| 위치 | 현재 | 이식 후 |
|------|------|---------|
| `src/data.py :: load_csv()` | `pd.read_csv(path)` | DB 쿼리 결과를 DataFrame으로 반환하는 함수 추가 |
| `src/train.py` (`--data` 인자) | CSV 경로 | 쿼리/기간 파라미터(예: 설비ID, 기간)로 정상 데이터 조회 |
| `app/app.py` (파일 업로드) | `st.file_uploader` | 업로드 유지 + "DB에서 최근 N건 조회" 옵션 추가 |

**작업 지침**
1. `src/db.py`(신규) 작성 — 커넥션 풀 + 조회/기록 함수.
   - 권장: SQLAlchemy 엔진 사용(예: `create_engine(DB_URL)`), `pd.read_sql(query, engine)`.
   - 지원 DB에 맞는 드라이버를 `requirements.txt`에 추가
     (예: PostgreSQL `psycopg2-binary`, MSSQL `pyodbc`, Oracle `oracledb`, MySQL `PyMySQL`).
2. `load_csv` 호출부를 `load_dataframe(source)`로 추상화해 CSV/DB 양쪽을 지원.
3. **DataFrame 형태(컬럼=센서, 행=관측)만 맞추면** 이후 전처리/모델/추론 코드는 수정 불필요.

**주의**
- DB 컬럼명 = 실제 센서명. 프로토타입의 합성 센서명(`temperature, vibration, ...`)과
  다르므로 반드시 **실데이터로 재학습**해야 한다(§4).
- 정상 데이터 조회 시 "정상 운전 구간"만 필터링(이상/정비 구간 제외) — 학습 데이터
  오염 방지(요구사항 핵심 원칙).

---

## 2. 설정 · 시크릿 관리 (하드코딩 금지)

DB 접속정보, 경로 등은 코드가 아닌 **환경변수/시크릿**으로 주입한다.

- DB URL/계정: 환경변수(예: `PHM_DB_URL`) 또는 Streamlit `secrets.toml` 사용.
  - `.gitignore`에 이미 `.streamlit/secrets.toml` 등록됨. **절대 커밋 금지.**
- `config/default.yaml`은 하이퍼파라미터/스키마 설정용으로 유지, 접속정보는 넣지 않는다.
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

---

## 4. 실데이터 재학습 (필수)

프로토타입 모델은 합성 데이터 기준이므로 **사내 실데이터로 다시 학습**해야 한다.

1. `config/default.yaml`의 `data.exclude_columns`, `label_column`, `timestamp_column`을
   실제 컬럼명에 맞게 수정.
2. 정상 데이터로 `python -m src.train --data <정상 조회>` 재실행.
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

- `src/batch_infer.py`(신규) 작성:
  1. DB에서 최근 미처리 데이터 조회
  2. `AnomalyDetector.predict()` 실행
  3. 결과(오차, 판정, top_feature, 시각)를 결과 테이블에 INSERT
- 스케줄링: cron / 사내 스케줄러 / Airflow 등.
- 결과 테이블 스키마(예): `timestamp, equipment_id, reconstruction_error, threshold, prediction, top_feature, model_version`.

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
- [ ] `src/db.py` 작성, `load_csv` → DB 조회 추상화
- [ ] DB 드라이버 `requirements.txt` 추가
- [ ] 접속정보 환경변수/시크릿화 (커밋 금지)
- [ ] 사내망 pip 설치 방식 확정(미러/오프라인 wheel)
- [ ] TensorFlow 설치 검증(CPU/오프라인)
- [ ] `config`의 컬럼 설정을 실데이터에 맞게 수정
- [ ] 실데이터로 재학습 및 임계값 튜닝
- [ ] `artifacts_dir` 영속 경로로 변경
- [ ] (필요 시) `batch_infer.py` + 결과 테이블 + 스케줄러
- [ ] Streamlit 서비스화 + 리버스 프록시 인증
- [ ] 로깅/드리프트 모니터링/재학습 주기 수립
