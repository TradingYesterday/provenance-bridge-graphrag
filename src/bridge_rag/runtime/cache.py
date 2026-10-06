"""Candidate cache. Bindings are part of the key."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(parts: dict[str, Any]) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CandidateCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.keys_seen: list[str] = []

    def get(self, key: str) -> Any | None:
        self.keys_seen.append(key)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value
