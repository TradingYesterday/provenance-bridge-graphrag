import json
from pathlib import Path

from bridge_rag.diagnostics.catalog import all_cases
from bridge_rag.diagnostics.runner import run_catalog

ROOT = Path(__file__).resolve().parents[2]


def test_catalog_has_the_planned_coverage():
    ids = [case.case_id for case in all_cases()]
    assert len(ids) >= 24
    for required in (
        "normal_missing_edge",
        "no_reentry",
        "conflict_negation",
        "unlinkable",
        "parent_revoke",
        "budget_exhausted",
        "plan_cycle",
        "literal_not_entity",
    ):
        assert required in ids


def test_diagnostic_catalog(tmp_path: Path):
    summary = run_catalog(tmp_path / "traces")
    failed = [row for row in summary["cases"] if not row["ok"]]
    assert not failed, json.dumps(failed, ensure_ascii=False, indent=2)
    assert summary["passed"] == len(summary["cases"])
    normal = next(row for row in summary["cases"] if row["case_id"] == "normal_missing_edge")
    assert normal["answer"] == "临江市"
    assert (tmp_path / "traces" / "normal_missing_edge.jsonl").exists()
