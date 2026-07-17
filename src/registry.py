"""학습 모델 레지스트리.

각 학습 결과를 models_dir 아래 개별 폴더로 영구 저장하고, 웹앱이 과거 모델을
목록·선택·삭제할 수 있게 한다. 각 항목은 아티팩트(모델/스케일러/임계값/스키마/메타/이력)
+ entry.json(표시 이름·평가지표 등)로 구성된다.

주의: Streamlit Community Cloud 등 임시 파일시스템에서는 재부팅 시 초기화된다.
영구 보관은 사내 서버의 영속 경로(또는 오브젝트 스토리지/DB)로 이관해 사용한다.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

from src.data import MODEL_FILE, load_meta

ENTRY_FILE = "entry.json"


def new_entry_dir(models_dir: str, model_type: str, name: Optional[str] = None) -> str:
    """새 모델 저장 폴더를 만들고 경로를 반환한다(고유 ID 보장)."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", " ")).strip()
    safe = safe.replace(" ", "-")
    entry_id = f"{ts}_{model_type}" + (f"_{safe}" if safe else "") + f"_{uuid.uuid4().hex[:4]}"
    path = os.path.join(models_dir, entry_id)
    os.makedirs(path, exist_ok=True)
    return path


def write_entry_info(entry_dir: str, info: Dict[str, Any]) -> None:
    with open(os.path.join(entry_dir, ENTRY_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def read_entry_info(entry_dir: str) -> Dict[str, Any]:
    p = os.path.join(entry_dir, ENTRY_FILE)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def list_models(models_dir: str) -> List[Dict[str, Any]]:
    """저장된 모델 목록을 최신순으로 반환한다."""
    if not os.path.isdir(models_dir):
        return []
    out: List[Dict[str, Any]] = []
    for name in os.listdir(models_dir):
        d = os.path.join(models_dir, name)
        if not os.path.isfile(os.path.join(d, MODEL_FILE)):
            continue
        try:
            meta = load_meta(d)
        except Exception:  # noqa: BLE001
            meta = {}
        info = read_entry_info(d)
        out.append({
            "id": name,
            "path": d,
            "name": info.get("name") or name,
            "created_at": meta.get("created_at", ""),
            "model_type": meta.get("model_type", "dense"),
            "window": meta.get("window"),
            "n_features": meta.get("n_features"),
            "n_samples": meta.get("n_samples"),
            "metrics": info.get("metrics"),
        })
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


def delete_model(entry_dir: str) -> None:
    shutil.rmtree(entry_dir, ignore_errors=True)


def prune_incomplete(models_dir: str) -> int:
    """학습이 중단돼 model.keras가 없는 빈/불완전 항목을 정리한다."""
    if not os.path.isdir(models_dir):
        return 0
    removed = 0
    for name in os.listdir(models_dir):
        d = os.path.join(models_dir, name)
        if os.path.isdir(d) and not os.path.isfile(os.path.join(d, MODEL_FILE)):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed
