"""Scripted extractor. It can only return candidates whose passage was retrieved."""

from __future__ import annotations

from bridge_rag.schemas import RecoveryCandidate


def extract_scripted(
    scripted: dict[str, list[RecoveryCandidate]],
    constraint_id: str,
    allowed_passage_ids: set[str],
) -> list[RecoveryCandidate]:
    selected: list[RecoveryCandidate] = []
    for candidate in scripted.get(constraint_id, []):
        if candidate.passage_id in allowed_passage_ids:
            selected.append(candidate)
    return selected
