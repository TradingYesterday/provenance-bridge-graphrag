"""Append-only JSONL trace. Secret-like fields are redacted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SECRET_PARTS = ("api_key", "authorization", "secret", "password", "token")


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SECRET_PARTS):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = scrub(item)
        return cleaned
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


class TraceLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, event: str, **fields: Any) -> dict[str, Any]:
        row = scrub({"event": event, "i": len(self.events), **fields})
        self.events.append(row)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
