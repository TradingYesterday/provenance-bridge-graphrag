"""Build an in-memory graph, text pool, and oracle candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from bridge_rag.backends.graph import EntityRecord, GraphStore, TripleRecord
from bridge_rag.backends.text import TextStore, TextUnit
from bridge_rag.planning import compile_plan
from bridge_rag.schemas import (
    CompiledPlan,
    EngineConfig,
    PlanTriple,
    RecoveryCandidate,
    SourceLocation,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class World:
    graph: GraphStore
    texts: TextStore
    plan: CompiledPlan
    scripted: dict[str, list[RecoveryCandidate]] = field(default_factory=dict)
    forced: dict[str, list[str]] = field(default_factory=dict)
    config: EngineConfig = field(default_factory=EngineConfig)
    revoke_variable: str | None = None


def make_graph(entities: list[dict], triples: list[dict], titles: dict[str, list[str]] | None = None) -> GraphStore:
    graph = GraphStore(namespace="diag", version="diag-v1")
    for index, item in enumerate(entities):
        graph.add_entity(
            EntityRecord(
                entity_id=item["entity_id"],
                namespace="diag",
                canonical_name=item["name"],
                aliases=list(item.get("aliases") or []),
                upstream_id=item.get("upstream_id", index),
                entity_type=item.get("type"),
            )
        )
    for item in triples:
        sources = []
        if item.get("quote") or item.get("title"):
            sources.append(
                SourceLocation(
                    source_id=f"graph:{item['triple_id']}",
                    title=item.get("title"),
                    doc_index=item.get("doc_index"),
                    sent_index=item.get("sent_index"),
                    quote=item.get("quote"),
                    unit_kind="sentence",
                )
            )
        graph.add_triple(
            TripleRecord(
                triple_id=item["triple_id"],
                upstream_index=item["index"],
                head_id=item["head"],
                relation=item["relation"],
                tail_id=item.get("tail"),
                tail_literal=item.get("literal"),
                title=item.get("title"),
                sources=sources,
            )
        )
    for title, triple_ids in (titles or {}).items():
        graph.title_to_triple_ids[title] = list(triple_ids)
    return graph


def make_texts(question_id: str, units: list[dict]) -> TextStore:
    store = TextStore()
    for item in units:
        store.units.append(
            TextUnit(
                passage_id=item["passage_id"],
                question_id=question_id,
                data_version="diag-v1",
                doc_index=item["doc_index"],
                sent_index=item["sent_index"],
                title=item["title"],
                raw_text=item["text"],
                unit_kind=item.get("unit_kind", "sentence"),
                visible=item.get("visible", True),
            )
        )
    return store


def make_candidate(
    *,
    constraint_index: int,
    unit: dict,
    quote: str,
    head: str,
    predicate: str,
    tail: str,
    score: float | None = None,
    bad_span: bool = False,
    rank: int = 0,
) -> RecoveryCandidate:
    if bad_span:
        start, end = 0, min(1, len(unit["text"]))
    else:
        start = unit["text"].index(quote)
        end = start + len(quote)
    return RecoveryCandidate(
        candidate_id=f"cand-c{constraint_index}-{rank}",
        constraint_id=f"c{constraint_index}",
        branch_id="pending",
        head_surface=head,
        predicate_surface=predicate,
        tail_surface=tail,
        source_doc_id=f"diag-v1:q:{unit['doc_index']}",
        title=unit["title"],
        passage_id=unit["passage_id"],
        char_start=start,
        char_end=end,
        quote=quote,
        extractor_model="oracle-scripted",
        prompt_hash=_digest("oracle-scripted"),
        raw_response_hash=_digest(quote),
        retrieval_rank=rank,
        reranker_score=score,
    )


def compile_world(
    triples: list[PlanTriple],
    entities: list[dict],
    graph_triples: list[dict],
    units: list[dict],
    scripted: list[RecoveryCandidate] | None = None,
    forced: dict[int, list[str]] | None = None,
    config: dict | None = None,
    revoke_variable: str | None = None,
    question_id: str = "q",
) -> World:
    grouped: dict[str, list[RecoveryCandidate]] = {}
    for candidate in scripted or []:
        grouped.setdefault(candidate.constraint_id, []).append(candidate)
    return World(
        graph=make_graph(entities, graph_triples),
        texts=make_texts(question_id, units),
        plan=compile_plan(triples, "diag"),
        scripted=grouped,
        forced={f"c{index}": ids for index, ids in (forced or {}).items()},
        config=EngineConfig.model_validate(config or {}),
        revoke_variable=revoke_variable,
    )
