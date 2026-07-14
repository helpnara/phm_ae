"""Oracle DB 어댑터 (사내 이식용).

접속정보는 코드가 아닌 **환경변수**로 주입한다(하드코딩/커밋 금지):
    PHM_DB_USER      : 계정
    PHM_DB_PASSWORD  : 비밀번호
    PHM_DB_DSN       : "host:1521/service_name" 또는 TNS 별칭

드라이버는 python-oracledb(`oracledb`)를 사용한다. 기본 **thin 모드**는 Oracle
Instant Client 설치 없이 동작한다. 구버전 서버 등으로 thick 모드가 필요하면
`oracledb.init_oracle_client(lib_dir=...)`를 호출한다(아래 주석 참고).

의존성은 requirements.txt의 "사내 DB(선택)" 항목 참조. 사내망 오프라인 설치는
docs/migration_notes.md §3 참고.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd

_ENGINE = None  # 프로세스 단위 캐시


def get_dsn() -> tuple[str, str, str]:
    """환경변수에서 접속정보를 읽는다. 누락 시 명확한 오류."""
    user = os.environ.get("PHM_DB_USER")
    pwd = os.environ.get("PHM_DB_PASSWORD")
    dsn = os.environ.get("PHM_DB_DSN")
    missing = [k for k, v in
               {"PHM_DB_USER": user, "PHM_DB_PASSWORD": pwd, "PHM_DB_DSN": dsn}.items()
               if not v]
    if missing:
        raise RuntimeError(
            f"DB 접속 환경변수가 설정되지 않았습니다: {missing}\n"
            f"예) export PHM_DB_USER=phm PHM_DB_PASSWORD=*** PHM_DB_DSN=host:1521/ORCLPDB1"
        )
    return user, pwd, dsn  # type: ignore[return-value]


def get_engine():
    """SQLAlchemy 엔진(oracle+oracledb)을 생성/캐시해 반환한다.

    thick 모드가 필요하면 아래 init_oracle_client 주석을 해제한다.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # 지연 import: 사내 환경에서만 설치되는 의존성
    from sqlalchemy import create_engine
    # import oracledb
    # oracledb.init_oracle_client(lib_dir=r"/opt/oracle/instantclient_21_13")  # thick 모드 필요 시

    user, pwd, dsn = get_dsn()
    # oracle+oracledb 방언 사용. DSN은 "host:port/service" 형태를 그대로 지원.
    url = f"oracle+oracledb://{user}:{pwd}@{dsn}"
    _ENGINE = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _ENGINE


def read_sql(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """SQL 조회 결과를 DataFrame으로 반환한다.

    바인드 변수는 params로 전달한다(SQL 인젝션 방지). 예:
        read_sql("SELECT * FROM t WHERE eq_id = :eid", {"eid": "A01"})
    """
    engine = get_engine()
    return pd.read_sql(query, engine, params=params)


def write_results(df: pd.DataFrame, table: str, if_exists: str = "append") -> int:
    """추론 결과 DataFrame을 결과 테이블에 적재한다. 적재 행 수를 반환한다.

    table은 사전에 생성되어 있어야 한다(스키마는 migration_notes §6 참고).
    컬럼명은 대소문자에 유의(Oracle 기본 대문자).
    """
    engine = get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    return len(df)
