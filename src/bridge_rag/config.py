"""YAML config. Secrets are environment variable names, never values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bridge_rag.schemas import EngineConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def engine_config_from_mapping(data: dict[str, Any]) -> EngineConfig:
    block = data.get("engine") or data
    return EngineConfig.model_validate(block)
