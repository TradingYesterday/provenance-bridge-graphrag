"""Diagnostic summary. This is not an answer-EM report."""

from __future__ import annotations

import json
from pathlib import Path


def write_diagnostic_report(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "kind": "diagnostic_not_qa_score",
        "passed": summary["passed"],
        "failed": summary["failed"],
        "cases": summary["cases"],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
