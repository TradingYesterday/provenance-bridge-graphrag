"""Run the diagnostic catalog and write a JSONL trace plus a summary."""

from __future__ import annotations

import json
from pathlib import Path

from bridge_rag.diagnostics.build import compile_world
from bridge_rag.diagnostics.catalog import DiagnosticCase, all_cases
from bridge_rag.diagnostics.check import evaluate
from bridge_rag.execution.engine import RunInput, execute
from bridge_rag.planning import compile_plan


def run_case(case: DiagnosticCase, trace_dir: Path | None = None) -> dict:
    if case.compile_only:
        plan = compile_plan(case.plan_triples, "diag")
        producer = case.expect.get("producer") or {}
        errors = []
        if not plan.supported:
            errors.append(f"plan failed: {plan.reason_code}")
        for variable, constraint_id in producer.items():
            if plan.producers.get(variable) != constraint_id:
                errors.append(f"producer {variable}={plan.producers.get(variable)} != {constraint_id}")
        terminal_index = case.expect.get("terminal_index")
        if terminal_index is not None:
            terminals = [item.query_triple_index for item in plan.constraints if item.is_terminal]
            if terminals != [terminal_index]:
                errors.append(f"terminals {terminals} != [{terminal_index}]")
        return {"case_id": case.case_id, "ok": not errors, "errors": errors, "status": "COMPILE"}
    world = compile_world(question_id=case.case_id, **case.world_args)
    before = world.graph.fingerprint()
    trace_path = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = str(trace_dir / f"{case.case_id}.jsonl")
    result = execute(
        RunInput(
            question_id=case.case_id,
            plan=world.plan,
            graph=world.graph,
            texts=world.texts,
            scripted=world.scripted,
            forced_graph_ids=world.forced,
            config=world.config,
            revoke_variable=world.revoke_variable,
            trace_path=trace_path,
        )
    )
    errors = evaluate(result, world.graph, case.expect, before)
    return {
        "case_id": case.case_id,
        "ok": not errors,
        "errors": errors,
        "status": result.status.value,
        "answer": result.answer,
        "uncertain": result.uncertain,
        "answers": result.answers,
        "reason_codes": result.reason_codes,
        "result": result,
    }


def run_catalog(trace_dir: Path | None = None) -> dict:
    rows = [run_case(case, trace_dir) for case in all_cases()]
    public = []
    for row in rows:
        public.append({key: value for key, value in row.items() if key != "result"})
    return {"passed": sum(1 for row in public if row["ok"]), "failed": sum(1 for row in public if not row["ok"]), "cases": public}


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
