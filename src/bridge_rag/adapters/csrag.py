"""Read CS-RAG JSON products. Do not import the upstream package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bridge_rag.adapters.dataset_2wiki import example_id, iter_sentences, public_fields, split_gold
from bridge_rag.adapters.provenance import locate_in_example
from bridge_rag.backends.graph import EntityRecord, GraphStore, TripleRecord
from bridge_rag.planning import compile_plan
from bridge_rag.schemas import CompiledPlan, GoldRecord, PlanTriple, QuestionCase, SourceLocation

PINNED_COMMIT = "c84a081c728dd45c284a7165031c853eef4e8f24"
KG_FILES = ("entities.jsonl", "triples.jsonl", "title2entities.jsonl", "title2triples.jsonl")


class InputContractError(ValueError):
    """A required upstream field or path is missing or inconsistent."""


def require_file(path: Path) -> Path:
    if path.is_file():
        return path
    hint = ""
    parts = path.parts
    if "planned_queries" in parts:
        index = parts.index("planned_queries")
        sibling = Path(*parts[:index], "planner_queries", *parts[index + 1 :])
        if sibling.is_file():
            hint = f" A file exists at {sibling}. Pass that path explicitly; planned_queries is not rewritten."
        else:
            hint = " Upstream configs say planned_queries/, while the pinned tree uses planner_queries/."
    raise InputContractError(f"required file does not exist: {path}.{hint}")


def require_path(path: str | Path) -> Path:
    try:
        return require_file(Path(path))
    except InputContractError as exc:
        raise FileNotFoundError(str(exc)) from exc


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputContractError(f"{path}:{line_no} is not JSON") from exc
    return rows


def load_kg_directory(kg_dir: Path, *, namespace: str, version: str) -> tuple[GraphStore, list[str]]:
    if not kg_dir.is_dir():
        raise InputContractError(f"KG directory does not exist: {kg_dir}")
    missing = [name for name in KG_FILES if not (kg_dir / name).is_file()]
    if missing:
        raise InputContractError(f"{kg_dir} missing {missing}")
    notes: list[str] = []
    entity_rows = load_jsonl(kg_dir / "entities.jsonl")
    triple_rows = load_jsonl(kg_dir / "triples.jsonl")
    graph = GraphStore(namespace=namespace, version=version)
    index_to_entity: dict[int, str] = {}
    for position, row in enumerate(entity_rows):
        if not isinstance(row, dict) or "id" not in row or "name" not in row:
            raise InputContractError(f"entity row {position} requires id and name")
        upstream_id = int(row["id"])
        if upstream_id != position:
            notes.append(f"entity id {upstream_id} differs from row position {position}; id field kept")
        entity_id = f"{namespace}:{upstream_id}"
        graph.add_entity(
            EntityRecord(
                entity_id=entity_id,
                namespace=namespace,
                canonical_name=str(row["name"]),
                aliases=[str(item) for item in row.get("aliases") or [] if isinstance(item, str)],
                upstream_id=upstream_id,
                entity_type=str(row["type"]) if row.get("type") else None,
            )
        )
        index_to_entity[upstream_id] = entity_id
    for position, row in enumerate(triple_rows):
        if not isinstance(row, dict):
            raise InputContractError(f"triple row {position} must be an object")
        head = row.get("h", row.get("head"))
        tail = row.get("t", row.get("tail"))
        relation = row.get("r", row.get("relation"))
        if head is None or relation is None or tail is None:
            raise InputContractError(f"triple row {position} requires h/r/t")
        head_id = index_to_entity.get(int(head))
        tail_id = index_to_entity.get(int(tail))
        if head_id is None or tail_id is None:
            raise InputContractError(f"triple row {position} points outside the entity table")
        title = str(row["title"]) if row.get("title") else None
        evidence = str(row["evidence"]) if row.get("evidence") else None
        source = SourceLocation(
            source_id=f"{version}:triple:{position}",
            title=title,
            quote=evidence,
            unit_kind="kg_evidence",
            ambiguous=not bool(evidence and title),
        )
        graph.add_triple(
            TripleRecord(
                triple_id=str(position),
                upstream_index=position,
                head_id=head_id,
                relation=str(relation),
                tail_id=tail_id,
                title=title,
                sources=[source],
            )
        )
    for row in load_jsonl(kg_dir / "title2triples.jsonl"):
        if not isinstance(row, dict) or "title" not in row or "triple_idxs" not in row:
            raise InputContractError("title2triples row requires title and triple_idxs")
        ids = []
        for raw_index in row["triple_idxs"]:
            triple_id = str(int(raw_index))
            if triple_id not in graph.triples:
                raise InputContractError(f"title2triples points at missing triple {triple_id}")
            ids.append(triple_id)
        graph.title_to_triple_ids[str(row["title"])] = ids
    for row in load_jsonl(kg_dir / "title2entities.jsonl"):
        if not isinstance(row, dict) or "title" not in row or "entity_ids" not in row:
            raise InputContractError("title2entities row requires title and entity_ids")
        ids = []
        for raw_id in row["entity_ids"]:
            entity_id = index_to_entity.get(int(raw_id))
            if entity_id is None:
                raise InputContractError(f"title2entities points at missing entity {raw_id}")
            ids.append(entity_id)
        graph.title_to_entity_ids[str(row["title"])] = ids
    return graph, notes


def _plan_from_query(query_plan: list[Any], namespace: str) -> CompiledPlan:
    triples: list[PlanTriple] = []
    for item in query_plan:
        if not isinstance(item, dict):
            raise InputContractError("query_plan entries must be objects")
        relation = str(item.get("relation") or "")
        variants = [str(value) for value in item.get("relation_variants") or [] if isinstance(value, str)]
        if not relation and variants:
            relation = variants[0]
        if not relation:
            raise InputContractError("query triple is missing relation")
        triples.append(
            PlanTriple(
                head=str(item.get("head") or ""),
                relation=relation,
                relation_variants=variants,
                tail=str(item.get("tail") or ""),
            )
        )
    return compile_plan(triples, namespace)


def assemble_case(
    phase1_item: dict[str, Any],
    *,
    dataset: str,
    data_version: str,
    namespace: str,
    raw_example: dict[str, Any] | None = None,
) -> tuple[QuestionCase, GoldRecord, CompiledPlan]:
    if "id" not in phase1_item and "_id" not in phase1_item:
        raise InputContractError("phase1 item requires id")
    if not isinstance(phase1_item.get("query_plan"), list):
        raise InputContractError("phase1 item requires query_plan")
    question_id = example_id(phase1_item)
    plan = _plan_from_query(phase1_item["query_plan"], namespace)
    gold_rows = [phase1_item]
    if raw_example is not None:
        gold_rows.append(raw_example)
    gold = split_gold(*gold_rows)
    gold.question_id = question_id
    case = QuestionCase(
        question_id=question_id,
        question=str(phase1_item.get("question") or (raw_example or {}).get("question") or ""),
        dataset=dataset,
        data_version=data_version,
        graph_namespace=namespace,
        constraints=plan.constraints if plan.supported else [],
    )
    return case, gold, plan


def localization_issues(
    *,
    phase1_item: dict[str, Any],
    phase2_item: dict[str, Any] | None,
    raw_example: dict[str, Any] | None,
    graph: GraphStore,
    data_version: str,
) -> dict[str, Any]:
    question_id = example_id(phase1_item)
    hard: list[str] = []
    ambiguities: list[str] = []
    query_plan = phase1_item.get("query_plan") or []
    if not isinstance(query_plan, list):
        hard.append("query_plan missing")
        query_plan = []
    evidence = phase1_item.get("evidence_triples") or []
    if not isinstance(evidence, list):
        hard.append("evidence_triples missing")
        evidence = []
    for item in evidence:
        if not isinstance(item, dict):
            hard.append("evidence triple is not an object")
            continue
        triple_id = item.get("kg_triple_id")
        record = graph.triples.get(str(triple_id))
        if record is None:
            hard.append(f"kg_triple_id {triple_id} missing")
            continue
        head_id = item.get("head_entity_id")
        tail_id = item.get("tail_entity_id")
        if head_id is not None and f"{graph.namespace}:{int(head_id)}" != record.head_id:
            hard.append(f"triple {triple_id} head mismatch")
        if tail_id is not None and f"{graph.namespace}:{int(tail_id)}" != record.tail_id:
            hard.append(f"triple {triple_id} tail mismatch")
        if str(item.get("relation")) != record.relation:
            hard.append(f"triple {triple_id} relation mismatch")
    support = phase1_item.get("support") or []
    for item in support if isinstance(support, list) else []:
        if not isinstance(item, dict):
            hard.append("support entry is not an object")
            continue
        query_index = item.get("query_triple_index")
        if not isinstance(query_index, int) or query_index < 0 or query_index >= len(query_plan):
            hard.append(f"support query_triple_index {query_index} out of range")
        for evidence_index in item.get("evidence_indices") or []:
            if not isinstance(evidence_index, int) or evidence_index < 0 or evidence_index >= len(evidence):
                hard.append(f"evidence index {evidence_index} out of range")
    for entry in (phase1_item.get("debug") or {}).get("entity_map") or []:
        if not isinstance(entry, dict):
            continue
        for entity_id in entry.get("kg_entity_ids") or []:
            if f"{graph.namespace}:{int(entity_id)}" not in graph.entities:
                hard.append(f"entity_map id {entity_id} missing")
    if phase2_item is not None:
        if example_id(phase2_item) != question_id:
            hard.append("phase2 id does not match phase1")
        rows = phase2_item.get("triples_evidence") or []
        if len(rows) != len(query_plan):
            hard.append(f"phase2 triple count {len(rows)} != query_plan {len(query_plan)}")
        for row in rows if isinstance(rows, list) else []:
            query_index = row.get("query_triple_index") if isinstance(row, dict) else None
            if not isinstance(query_index, int) or query_index >= len(query_plan):
                hard.append(f"phase2 query_triple_index {query_index} out of range")
                continue
            for candidate in row.get("candidate_evidences") or []:
                if not isinstance(candidate, dict) or raw_example is None:
                    continue
                title = candidate.get("title")
                quote = candidate.get("evidence") or candidate.get("context")
                doc_index = candidate.get("paragraph_idx")
                sent_index = candidate.get("sent_idx")
                located = locate_in_example(
                    raw_example,
                    data_version=data_version,
                    question_id=question_id,
                    title=str(title) if title else None,
                    quote=str(quote) if quote else None,
                    doc_index=int(doc_index) if isinstance(doc_index, int) else None,
                    sent_index=int(sent_index) if isinstance(sent_index, int) else None,
                )
                if isinstance(sent_index, int) and not located.locations:
                    hard.append(f"sentence index {sent_index} does not locate a quote for query triple {query_index}")
                elif located.ambiguous:
                    ambiguities.append(f"q{question_id} triple {query_index}: {located.reason}")
                elif isinstance(sent_index, int):
                    sentence = located.locations[0]
                    raw = _sentence_at(raw_example, sentence.doc_index or 0, sentence.sent_index or 0)
                    expected = candidate.get("evidence")
                    if expected is not None and expected not in raw and raw != expected:
                        hard.append(f"sentence mismatch at {sentence.source_id}")
    case, _gold, _plan = assemble_case(
        phase1_item,
        dataset="2wiki",
        data_version=data_version,
        namespace=graph.namespace,
        raw_example=raw_example,
    )
    leaked = [key for key in case.model_dump().keys() if key in {"answer", "ground_truth_answer", "supporting_facts"}]
    if leaked:
        hard.append(f"gold leaked into QuestionCase: {leaked}")
    return {
        "question_id": question_id,
        "hard_failures": hard,
        "ambiguities": ambiguities,
        "public_question": public_fields(phase1_item).get("question"),
    }


def _sentence_at(example: dict, doc_index: int, sent_index: int) -> str:
    for raw_doc, _title, raw_sent, sentence in iter_sentences(example):
        if raw_doc == doc_index and raw_sent == sent_index:
            return sentence
    return ""


def audit_records(
    phase1_items: list[dict[str, Any]],
    phase2_items: list[dict[str, Any]] | None,
    raw_by_id: dict[str, dict[str, Any]],
    graph: GraphStore,
    *,
    data_version: str,
    source_label: str,
) -> dict[str, Any]:
    phase2_by_id = {}
    if phase2_items is not None:
        phase2_by_id = {example_id(item): item for item in phase2_items}
    reports = []
    for item in phase1_items:
        question_id = example_id(item)
        reports.append(
            localization_issues(
                phase1_item=item,
                phase2_item=phase2_by_id.get(question_id),
                raw_example=raw_by_id.get(question_id),
                graph=graph,
                data_version=data_version,
            )
        )
    hard = [issue for report in reports for issue in report["hard_failures"]]
    ambiguities = [issue for report in reports for issue in report["ambiguities"]]
    return {
        "source": source_label,
        "pinned_commit": PINNED_COMMIT,
        "questions": len(reports),
        "hard_failures": hard,
        "ambiguity_count": len(ambiguities),
        "ambiguities": ambiguities,
        "passed": not hard,
    }


def upstream_command(config_path: str) -> list[str]:
    return [
        "python",
        "run.py",
        "--config",
        config_path,
        "#",
        f"pinned=myz12138/CS-RAG@{PINNED_COMMIT}",
        "#",
        "do not import components.phase1.legacy_impl; it loads a reranker at import time",
    ]


def synthetic_localization_cohort(n: int = 20) -> list[dict[str, Any]]:
    """20 schema-level questions. Not a 2Wiki result."""
    graph = GraphStore(namespace="cs", version="synthetic-v1")
    for index in range(n):
        graph.add_entity(EntityRecord(entity_id=f"cs:{index}", namespace="cs", canonical_name=f"Person {index}", upstream_id=index))
        graph.add_entity(EntityRecord(entity_id=f"cs:{1000 + index}", namespace="cs", canonical_name=f"Org {index}", upstream_id=1000 + index))
    for index in range(n):
        quote = f"Person {index} is a member of Org {index}."
        graph.add_triple(
            TripleRecord(
                triple_id=str(index * 2),
                upstream_index=index * 2,
                head_id=f"cs:{index}",
                relation="decoy",
                tail_id=f"cs:{1000 + index}",
                title=f"Doc {index}",
                sources=[SourceLocation(source_id=f"synthetic:decoy:{index}", title=f"Doc {index}", quote="decoy", ambiguous=True)],
            )
        )
        graph.add_triple(
            TripleRecord(
                triple_id=str(index * 2 + 1),
                upstream_index=index * 2 + 1,
                head_id=f"cs:{index}",
                relation="member of",
                tail_id=f"cs:{1000 + index}",
                title=f"Doc {index}",
                sources=[SourceLocation(source_id=f"synthetic:real:{index}", title=f"Doc {index}", quote=quote)],
            )
        )
    reports = []
    for index in range(n):
        quote = f"Person {index} is a member of Org {index}."
        phase1 = {
            "id": f"q{index}",
            "question": f"Which organization includes Person {index}?",
            "ground_truth_answer": f"SECRET_{index}",
            "query_plan": [{"head": f"Person {index}", "relation": "member of", "relation_variants": ["member of"], "tail": "?org"}],
            "evidence_triples": [
                {
                    "kg_triple_id": index * 2 + 1,
                    "head_entity_id": index,
                    "tail_entity_id": 1000 + index,
                    "head": f"Person {index}",
                    "relation": "member of",
                    "tail": f"Org {index}",
                }
            ],
            "support": [{"query_triple_index": 0, "evidence_indices": [0]}],
            "debug": {"variable_candidates": {"?org": {"primary_ids": [1000 + index]}}, "entity_map": []},
        }
        raw = {
            "_id": f"q{index}",
            "question": phase1["question"],
            "answer": f"SECRET_{index}",
            "supporting_facts": [[f"Doc {index}", 0]],
            "context": [[f"Doc {index}", [quote]]],
        }
        phase2 = {
            "id": f"q{index}",
            "triples_evidence": [
                {"query_triple_index": 0, "candidate_evidences": [{"title": f"Doc {index}", "sent_idx": 0, "evidence": quote}]}
            ],
        }
        case, gold, _plan = assemble_case(phase1, dataset="2wiki", data_version="synthetic-v1", namespace="cs", raw_example=raw)
        issues = localization_issues(phase1_item=phase1, phase2_item=phase2, raw_example=raw, graph=graph, data_version="synthetic-v1")
        reports.append(
            {
                "question_id": case.question_id,
                "ok": not issues["hard_failures"] and not issues["ambiguities"],
                "gold_leaked": bool(gold.answer and gold.answer in case.model_dump_json()),
                "kg_triple_id": str(phase1["evidence_triples"][0]["kg_triple_id"]),
                "expected_kg_triple_id": str(index * 2 + 1),
                "phase1_candidates_are_bindings": False,
                "hard_failures": issues["hard_failures"],
                "ambiguities": issues["ambiguities"],
            }
        )
    return reports


class AdaptedQuestion:
    def __init__(self, case: QuestionCase, gold: GoldRecord, plan: CompiledPlan, hints: dict[str, Any], audit: dict[str, Any]) -> None:
        self.case = case
        self.gold = gold
        self.plan = plan
        self.hints = hints
        self.audit = audit


def adapt_question(
    phase1_item: dict[str, Any],
    phase2_item: dict[str, Any] | None,
    raw_example: dict[str, Any] | None,
    graph: GraphStore,
    *,
    data_version: str,
) -> AdaptedQuestion:
    case, gold, plan = assemble_case(
        phase1_item,
        dataset="2wiki",
        data_version=data_version,
        namespace=graph.namespace,
        raw_example=raw_example,
    )
    issues = localization_issues(
        phase1_item=phase1_item,
        phase2_item=phase2_item,
        raw_example=raw_example,
        graph=graph,
        data_version=data_version,
    )
    candidates = []
    for index, item in enumerate(phase1_item.get("evidence_triples") or []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "kg_triple_id": str(item.get("kg_triple_id")),
                "compact_index": index,
                "head_entity_id": item.get("head_entity_id"),
                "tail_entity_id": item.get("tail_entity_id"),
                "relation": item.get("relation"),
            }
        )
    leaked = bool(gold.answer and gold.answer in case.model_dump_json())
    audit = {
        "gold_leaked": leaked,
        "misaligned": len(issues["hard_failures"]),
        "ambiguous": len(issues["ambiguities"]),
        "ok": not leaked and not issues["hard_failures"],
        "hard_failures": issues["hard_failures"],
        "ambiguities": issues["ambiguities"],
    }
    hints = {"accept_phase1_bindings": False, "resolved_graph_candidates": candidates}
    return AdaptedQuestion(case, gold, plan, hints, audit)


def audit_bundle(bundle: AdaptedQuestion) -> dict[str, Any]:
    return dict(bundle.audit)
