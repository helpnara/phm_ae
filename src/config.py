"""설정 로딩 유틸리티."""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "default.yaml",
)


def load_config(path: str | None = None) -> Dict[str, Any]:
    """YAML 설정 파일을 로드한다. path가 없으면 config/default.yaml."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
