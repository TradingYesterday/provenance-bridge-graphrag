"""Graph store with stable IDs. Masking an edge does not renumber anything."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from bridge_rag.schemas import SourceLocation


def norm_name(value: str) -> str:
    return " ".join(value.casefold().strip().split())


@dataclass
class EntityRecord:
    entity_id: str
    namespace: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    upstream_id: int | None = None
    entity_type: str | None = None

    def names(self) -> list[str]:
        return [self.canonical_name, *self.aliases]


@dataclass
class TripleRecord:
    triple_id: str
    upstream_index: int
    head_id: str
    relation: str
    tail_id: str | None
    tail_literal: str | None = None
    active: bool = True
    title: str | None = None
    sources: list[SourceLocation] = field(default_factory=list)


@dataclass
class GraphStore:
    namespace: str
    version: str
    entities: dict[str, EntityRecord] = field(default_factory=dict)
    triples: dict[str, TripleRecord] = field(default_factory=dict)
    title_to_triple_ids: dict[str, list[str]] = field(default_factory=dict)
    title_to_entity_ids: dict[str, list[str]] = field(default_factory=dict)
    writes: int = 0
    lookup_count: int = 0
    match_log: list[str] = field(default_factory=list)

    def add_entity(self, record: EntityRecord) -> None:
        if record.namespace != self.namespace:
            raise ValueError("cross-namespace entity insert is rejected")
        if record.entity_id in self.entities:
            raise ValueError(f"duplicate entity id {record.entity_id}")
        self.entities[record.entity_id] = record
        self.writes += 1

    def add_triple(self, record: TripleRecord) -> None:
        if record.triple_id in self.triples:
            raise ValueError(f"duplicate triple id {record.triple_id}")
        if record.head_id not in self.entities:
            raise ValueError(f"dangling head {record.head_id}")
        if record.tail_id is not None and record.tail_id not in self.entities:
            raise ValueError(f"dangling tail {record.tail_id}")
        self.triples[record.triple_id] = record
        self.writes += 1

    def mask_triple(self, triple_id: str) -> None:
        record = self.triples[triple_id]
        record.active = False

    def entity(self, entity_id: str) -> EntityRecord | None:
        self.lookup_count += 1
        return self.entities.get(entity_id)

    def match_surface(self, surface: str) -> list[EntityRecord]:
        self.lookup_count += 1
        self.match_log.append(surface)
        key = norm_name(surface)
        hits = [ent for ent in self.entities.values() if any(norm_name(name) == key for name in ent.names())]
        return sorted(hits, key=lambda ent: ent.entity_id)

    def outgoing(self, entity_id: str) -> list[TripleRecord]:
        return [
            triple
            for triple in self.triples.values()
            if triple.active and triple.head_id == entity_id
        ]

    def incoming(self, entity_id: str) -> list[TripleRecord]:
        return [
            triple
            for triple in self.triples.values()
            if triple.active and triple.tail_id == entity_id
        ]

    def active_title_triples(self, title: str) -> list[str]:
        return [tid for tid in self.title_to_triple_ids.get(title, []) if self.triples[tid].active]

    def fingerprint(self) -> str:
        payload = {
            "namespace": self.namespace,
            "version": self.version,
            "entities": [
                {
                    "id": ent.entity_id,
                    "name": ent.canonical_name,
                    "aliases": ent.aliases,
                }
                for ent in sorted(self.entities.values(), key=lambda item: item.entity_id)
            ],
            "triples": [
                {
                    "id": triple.triple_id,
                    "index": triple.upstream_index,
                    "head": triple.head_id,
                    "rel": triple.relation,
                    "tail": triple.tail_id,
                    "literal": triple.tail_literal,
                    "active": triple.active,
                }
                for triple in sorted(self.triples.values(), key=lambda item: item.upstream_index)
            ],
            "title2triples": {key: list(ids) for key, ids in sorted(self.title_to_triple_ids.items())},
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def snapshot_ids(self) -> dict[str, int]:
        return {triple.triple_id: triple.upstream_index for triple in self.triples.values()}
