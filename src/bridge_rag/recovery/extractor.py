"""Extract relation mentions from retrieved passages only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bridge_rag.backends.llm import LLMClient, ModelCallError
from bridge_rag.backends.text import TextUnit
from bridge_rag.runtime.budget import BudgetLedger
from bridge_rag.schemas import Constraint, RecoveryCandidate


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


def _prompt_text(filename: str, fallback: str) -> str:
    path = Path(__file__).resolve().parents[3] / "prompts" / filename
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extraction_prompt(constraint: Constraint, passages: list[TextUnit], bound_name: str) -> tuple[str, str]:
    system = _prompt_text(
        "recover_relation.txt",
        "Extract relations only from the supplied passages. Return JSON.",
    )
    blocks = [f"[{unit.passage_id}] {unit.raw_text}" for unit in passages]
    user = (
        f"constraint_id: {constraint.constraint_id}\n"
        f"relation: {constraint.predicate_text}\n"
        f"bound_endpoint: {bound_name}\n"
        "passages:\n"
        + "\n".join(blocks)
        + '\nReturn {"relations":[{"passage_id":"","quote":"","head":"","predicate":"","tail":""}]}'
    )
    return system, user


def candidates_from_content(
    content: str,
    *,
    constraint: Constraint,
    passages: list[TextUnit],
    model: str,
    prompt_hash_value: str,
    raw_hash: str,
) -> tuple[list[RecoveryCandidate], list[str]]:
    """Parse one structured response. Invalid JSON yields no candidates and a parse note."""
    by_id = {unit.passage_id: unit for unit in passages}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return [], ["PARSE_ERROR"]
    rows = payload.get("relations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return [], ["PARSE_ERROR"]
    notes: list[str] = []
    selected: list[RecoveryCandidate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            notes.append("DROPPED_ROW")
            continue
        passage_id = str(row.get("passage_id") or "")
        unit = by_id.get(passage_id)
        quote = str(row.get("quote") or "")
        if unit is None or not quote:
            notes.append("DROPPED_PASSAGE")
            continue
        start = unit.raw_text.find(quote)
        if start < 0:
            start, end = 0, min(1, len(unit.raw_text))
        else:
            end = start + len(quote)
        selected.append(
            RecoveryCandidate(
                candidate_id=f"model-{constraint.constraint_id}-{index}",
                constraint_id=constraint.constraint_id,
                branch_id="model",
                head_surface=str(row.get("head") or ""),
                predicate_surface=str(row.get("predicate") or constraint.predicate_text),
                tail_surface=str(row.get("tail") or ""),
                source_doc_id=unit.source_id(),
                title=unit.title,
                passage_id=passage_id,
                char_start=start,
                char_end=end,
                quote=quote,
                extractor_model=model,
                prompt_hash=prompt_hash_value,
                raw_response_hash=raw_hash,
            )
        )
    return selected, notes


def extract_with_model(
    client: LLMClient,
    constraint: Constraint,
    passages: list[TextUnit],
    bound_name: str,
    ledger: BudgetLedger | None = None,
) -> tuple[list[RecoveryCandidate], list[str], str]:
    system, user = extraction_prompt(constraint, passages, bound_name)
    try:
        response = client.complete_json(system=system, user=user, purpose="extract", ledger=ledger)
    except ModelCallError:
        raise
    selected, notes = candidates_from_content(
        response.content,
        constraint=constraint,
        passages=passages,
        model=response.model,
        prompt_hash_value=response.prompt_hash,
        raw_hash=response.raw_hash,
    )
    return selected, notes, response.model
