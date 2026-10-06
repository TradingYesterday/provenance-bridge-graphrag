"""One-hop candidate enumeration. Direction is enforced by the caller."""

from __future__ import annotations

from bridge_rag.backends.graph import GraphStore, TripleRecord


def enumerate_side(graph: GraphStore, entity_id: str, bound_on_head: bool, forced_ids: list[str] | None) -> list[TripleRecord]:
    if forced_ids is not None:
        records = []
        for triple_id in forced_ids:
            record = graph.triples.get(triple_id)
            if record is not None and record.active:
                records.append(record)
        return records
    pool = graph.outgoing(entity_id) if bound_on_head else graph.incoming(entity_id)
    return sorted(pool, key=lambda triple: (triple.upstream_index, triple.triple_id))
