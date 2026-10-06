"""Command line. Batch 1 does not call a model API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridge_rag import __version__
from bridge_rag.adapters.csrag import PINNED_COMMIT, synthetic_localization_cohort, upstream_command
from bridge_rag.baselines import UNAVAILABLE
from bridge_rag.config import load_yaml
from bridge_rag.diagnostics.runner import run_catalog
from bridge_rag.evaluation.report import write_diagnostic_report


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    out = Path(args.out)
    summary = run_catalog(out / "traces")
    write_diagnostic_report(summary, out / "summary.json")
    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"], "out": str(out)}, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    from bridge_rag.adapters.csrag import require_path

    config = load_yaml(args.config)
    upstream = config.get("upstream") or {}
    missing = []
    for key in ("query_json", "phase1_json", "phase2_json", "kg_dir", "raw_json"):
        value = upstream.get(key)
        if not value:
            missing.append(key)
            continue
        try:
            require_path(value)
        except FileNotFoundError as exc:
            print(str(exc))
            return 2
    report = {
        "version": __version__,
        "pinned_commit": PINNED_COMMIT,
        "configured_paths": {key: upstream.get(key) for key in ("query_json", "phase1_json", "phase2_json", "kg_dir", "raw_json")},
        "unset_paths": missing,
        "baselines_unavailable": list(UNAVAILABLE),
        "note": "Unset paths are not guessed. planned_queries is not rewritten to planner_queries.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_cohort(args: argparse.Namespace) -> int:
    rows = synthetic_localization_cohort(args.n)
    bad = [row for row in rows if not row["ok"] or row["kg_triple_id"] != row["expected_kg_triple_id"] or row["gold_leaked"]]
    payload = {"n": len(rows), "ok": not bad, "source": "synthetic_not_2wiki", "rows": rows}
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": payload["n"], "ok": payload["ok"], "source": payload["source"]}, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def _cmd_pilot(_args: argparse.Namespace) -> int:
    print(
        "Pilot gate is closed. Batch 1 has no API runner. "
        "Finish input audit, diagnostics, oracle re-entry, cache isolation, and frozen prompts before a 100-200 question run."
    )
    return 2


def _cmd_upstream(args: argparse.Namespace) -> int:
    print(" ".join(upstream_command(args.config)))
    print("This command is not executed. Upstream modules load models at import time.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge-rag")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("run-diagnostics")
    diag.add_argument("--out", default="outputs/diagnostics")
    diag.set_defaults(func=_cmd_diagnostics)
    validate = sub.add_parser("validate-inputs")
    validate.add_argument("--config", default="configs/2wiki_pilot.yaml")
    validate.set_defaults(func=_cmd_validate)
    cohort = sub.add_parser("audit-synthetic")
    cohort.add_argument("--n", type=int, default=20)
    cohort.add_argument("--out", default="")
    cohort.set_defaults(func=_cmd_cohort)
    pilot = sub.add_parser("run-pilot")
    pilot.set_defaults(func=_cmd_pilot)
    upstream = sub.add_parser("upstream-command")
    upstream.add_argument("--config", default="configs/2wiki_pilot.yaml")
    upstream.set_defaults(func=_cmd_upstream)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
