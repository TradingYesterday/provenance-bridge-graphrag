"""Compare an engine result with a diagnostic expectation."""

from __future__ import annotations

from bridge_rag.backends.graph import GraphStore
from bridge_rag.execution.proof import active_chain
from bridge_rag.schemas import BranchStatus, RunResult


def _chain(result: RunResult) -> list[dict]:
    rows = []
    for branch in result.branches:
        if "FORKED" in branch.reason_codes:
            continue
        for proof in active_chain(branch.accepted_evidence):
            rows.append(
                {
                    "branch_id": branch.branch_id,
                    "kind": proof.kind.value,
                    "relation": proof.relation,
                    "head": proof.head_surface,
                    "tail": proof.tail_surface,
                    "head_id": proof.head_entity_id,
                    "tail_id": proof.tail_entity_id,
                    "literal": proof.tail_literal,
                    "sent_index": proof.source.sent_index if proof.source else None,
                    "invalidated": proof.invalidated,
                }
            )
    return rows


def _decisions(result: RunResult) -> list[str]:
    found = []
    for event in result.trace:
        if event.get("event") == "verify" and event.get("decision"):
            found.append(event["decision"])
        if event.get("event") == "link":
            found.extend(event.get("notes") or [])
    return found


def evaluate(result: RunResult, graph: GraphStore, expect: dict, fingerprint_before: str) -> list[str]:
    errors: list[str] = []
    if "status" in expect and result.status.value != expect["status"]:
        errors.append(f"status {result.status.value} != {expect['status']}")
    if "answer" in expect and result.answer != expect["answer"]:
        errors.append(f"answer {result.answer!r} != {expect['answer']!r}")
    if "uncertain" in expect and result.uncertain != expect["uncertain"]:
        errors.append(f"uncertain {result.uncertain} != {expect['uncertain']}")
    if "answers" in expect and sorted(result.answers) != sorted(expect["answers"]):
        errors.append(f"answers {result.answers} != {expect['answers']}")
    for code in expect.get("reasons", []):
        if code not in result.reason_codes and code not in _decisions(result):
            errors.append(f"missing reason {code}; have {result.reason_codes}")
    for code in expect.get("forbid_reasons", []):
        if code in result.reason_codes:
            errors.append(f"forbidden reason {code}")
    if expect.get("graph_unchanged", True):
        if graph.fingerprint() != fingerprint_before:
            errors.append("graph fingerprint changed")
        if result.graph_writes != 0:
            errors.append(f"graph writes {result.graph_writes}")
    absent = set(expect.get("graph_relations_absent") or [])
    present = {triple.relation for triple in graph.triples.values()}
    overlap = sorted(absent & present)
    if overlap:
        errors.append(f"graph still has relations {overlap}")
    decisions = _decisions(result)
    for decision in expect.get("decisions", []):
        if decision not in decisions:
            errors.append(f"missing decision {decision}; have {decisions}")
    if "match_log_excludes" in expect:
        for surface in expect["match_log_excludes"]:
            if surface in graph.match_log:
                errors.append(f"linker looked up literal/surface {surface}")
    chains = _chain(result)
    if "chain_contains" in expect:
        for needed in expect["chain_contains"]:
            if not any(all(row.get(key) == value for key, value in needed.items()) for row in chains):
                errors.append(f"chain missing {needed}; have {chains}")
    if "chain_absent" in expect:
        for banned in expect["chain_absent"]:
            if any(all(row.get(key) == value for key, value in banned.items()) for row in chains):
                errors.append(f"chain unexpectedly has {banned}")
    if "no_mixed" in expect:
        for branch in result.branches:
            surfaces = []
            for proof in active_chain(branch.accepted_evidence):
                surfaces.extend([proof.head_surface, proof.tail_surface])
            hits = [name for name in expect["no_mixed"] if name in surfaces]
            if len(hits) > 1:
                errors.append(f"branch {branch.branch_id} mixed {hits}")
    if "bindings" in expect:
        live = {}
        for branch in result.branches:
            if "FORKED" in branch.reason_codes or branch.status is BranchStatus.INVALIDATED:
                continue
            for name, binding in branch.bindings.items():
                live.setdefault(name, set()).add(binding.canonical_name)
        for name, expected in expect["bindings"].items():
            if expected not in live.get(name, set()):
                errors.append(f"binding {name}={expected} not in {live}")
    if "bindings_absent" in expect:
        for branch in result.branches:
            for name in expect["bindings_absent"]:
                if name in branch.bindings:
                    errors.append(f"{branch.branch_id} still binds {name}")
    if expect.get("proofs_invalidated"):
        lingering = [proof.proof_id for proof in result.proofs if not proof.invalidated and proof.produces_variable]
        if lingering:
            errors.append(f"active produced proofs remain: {lingering}")
    return errors
