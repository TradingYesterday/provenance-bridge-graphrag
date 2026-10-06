"""Diagnostic and paired method reports. These are not a QA leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

from bridge_rag.evaluation.cost import cost_view
from bridge_rag.evaluation.evidence import score_case
from bridge_rag.evaluation.gold import MetricGold
from bridge_rag.schemas import RunResult


def write_diagnostic_report(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "kind": "diagnostic_not_qa_score",
        "passed": summary["passed"],
        "failed": summary["failed"],
        "cases": summary["cases"],
    }
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def paired_report(rows: list[dict], baseline: str, other: str, field: str) -> dict:
    left = {row["question_id"]: row for row in rows if row["method"] == baseline}
    right = {row["question_id"]: row for row in rows if row["method"] == other}
    shared = sorted(set(left) & set(right))
    deltas = []
    for question_id in shared:
        a = left[question_id].get(field)
        b = right[question_id].get(field)
        if isinstance(a, bool):
            a = float(a)
        if isinstance(b, bool):
            b = float(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas.append(b - a)
    return {
        "baseline": baseline,
        "other": other,
        "field": field,
        "n": len(shared),
        "mean_delta": _mean(deltas),
    }


def method_report(scored_rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    failures: dict[str, int] = {}
    for row in scored_rows:
        groups.setdefault(row["method"], []).append(row)
        if row["status"] != "COMPLETE":
            failures[row["status"]] = failures.get(row["status"], 0) + 1
    overall = {}
    for method, items in groups.items():
        f1_values = [item["token_f1"] for item in items if isinstance(item["token_f1"], float)]
        successor = [1.0 if item["successor_acquired"] else 0.0 for item in items]
        precision_values = [item["proof_precision"] for item in items if isinstance(item["proof_precision"], float)]
        overall[method] = {
            "n": len(items),
            "mean_f1": _mean(f1_values),
            "successor_rate": _mean(successor),
            "mean_proof_precision": _mean(precision_values),
            "precision_coverage": sum(1 for item in items if item["precision_coverage"]) / len(items),
        }
    return {
        "kind": "paired_method_report_not_a_leaderboard",
        "overall": overall,
        "failure_counts": failures,
        "paired": [
            paired_report(scored_rows, "core00", "core11", "successor_acquired"),
            paired_report(scored_rows, "core11", "simple_bind", "successor_acquired"),
            paired_report(scored_rows, "core11", "cog_controlled", "successor_acquired"),
            paired_report(scored_rows, "core11", "iterative_text", "successor_acquired"),
        ],
        "rows": scored_rows,
    }


def score_methods(results: dict[str, RunResult], gold: MetricGold) -> dict:
    rows = []
    for method, result in results.items():
        row = score_case(result, gold)
        row["method"] = method
        row["cost"] = cost_view(result)
        rows.append(row)
    return method_report(rows)
