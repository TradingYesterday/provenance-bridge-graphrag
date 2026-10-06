"""Link a surface to the current graph only. Never invent an entity id."""

from __future__ import annotations

from bridge_rag.backends.graph import GraphStore
from bridge_rag.schemas import LinkedBinding


def link_surface(
    graph: GraphStore,
    surface: str,
    variable: str,
    *,
    binding_id: str,
    limit: int,
) -> tuple[list[LinkedBinding], list[str]]:
    matches = graph.match_surface(surface)
    notes: list[str] = []
    if len(matches) > limit:
        notes.append(f"LINK_TRUNCATED:{len(matches)}->{limit}")
        matches = matches[:limit]
    if not matches:
        return (
            [
                LinkedBinding(
                    binding_id=binding_id,
                    variable=variable,
                    entity_id=None,
                    canonical_name=surface,
                    link_score=0.0,
                    membership_in_current_graph=False,
                    addressable_for_successor=False,
                    graph_namespace=graph.namespace,
                )
            ],
            ["UNLINKABLE"],
        )
    if len(matches) > 1:
        bindings = [
            LinkedBinding(
                binding_id=f"{binding_id}:{index}",
                variable=variable,
                entity_id=entity.entity_id,
                canonical_name=entity.canonical_name,
                aliases=list(entity.aliases),
                link_score=None,
                membership_in_current_graph=True,
                addressable_for_successor=False,
                graph_namespace=graph.namespace,
            )
            for index, entity in enumerate(matches)
        ]
        return bindings, ["AMBIGUOUS_ENTITY"]
    entity = matches[0]
    alias_hit = surface != entity.canonical_name
    return (
        [
            LinkedBinding(
                binding_id=binding_id,
                variable=variable,
                entity_id=entity.entity_id,
                canonical_name=entity.canonical_name,
                aliases=list(entity.aliases),
                link_score=1.0,
                membership_in_current_graph=True,
                addressable_for_successor=True,
                graph_namespace=graph.namespace,
            )
        ],
        ["ALIAS_UNIQUE"] if alias_hit else ["EXACT_NAME"],
    )
