"""Two-layer checks. Layer 1 is always on. Layer 2 is the text/graph semantic switch."""

from __future__ import annotations

from bridge_rag.backends.graph import EntityRecord, TripleRecord, norm_name
from bridge_rag.backends.text import TextUnit
from bridge_rag.schemas import (
    EngineConfig,
    RecoveryCandidate,
    Term,
    VerificationDecision,
    VerificationResult,
)

_NEGATIONS = ("没有", "并未", "并非", "不是", "从未", "不曾", "never", "not ")
_QUALIFIERS = ("预科", "短期交流", "访问学者", "进修", "荣誉学位", "交换生")


def span_ok(unit: TextUnit, quote: str, start: int, end: int) -> bool:
    if unit is None or start < 0 or end > len(unit.raw_text) or start >= end:
        return False
    return unit.raw_text[start:end] == quote


def relation_matches(relation: str, variants: list[str]) -> bool:
    rel = norm_name(relation)
    for variant in variants:
        key = norm_name(variant)
        if not key:
            continue
        if key == rel or key in rel or rel in key:
            return True
    return False


def cue_in_text(text: str, variants: list[str]) -> bool:
    return any(variant and variant in text for variant in variants)


def negation_before_cue(quote: str, variants: list[str]) -> bool:
    for variant in variants:
        index = quote.find(variant)
        if index < 0:
            continue
        window = quote[max(0, index - 8) : index]
        if any(token in window for token in _NEGATIONS):
            return True
    return False


def qualifier_conflict(quote: str, predicate: str) -> bool:
    for token in _QUALIFIERS:
        if token in quote and token not in predicate:
            return True
    return False


def _result(
    config: EngineConfig,
    decision: VerificationDecision,
    *,
    direction_ok: bool,
    qualifier_ok: bool,
    reason_code: str,
    semantic_checked: bool,
    identity_score: float | None = None,
) -> VerificationResult:
    return VerificationResult(
        decision=decision,
        entailment_score=None,
        identity_score=identity_score,
        direction_ok=direction_ok,
        qualifier_ok=qualifier_ok,
        reason_code=reason_code,
        verifier_model=config.verifier_model,
        prompt_hash=config.prompt_hash,
        semantic_checked=semantic_checked,
    )


def names_match(surface: str, names: list[str]) -> bool:
    key = norm_name(surface)
    return any(norm_name(name) == key for name in names if name)


def verify_text(
    candidate: RecoveryCandidate,
    unit: TextUnit | None,
    *,
    bound_names: list[str],
    bound_on_head: bool,
    variants: list[str],
    predicate: str,
    config: EngineConfig,
) -> VerificationResult:
    if unit is None or not span_ok(unit, candidate.quote, candidate.char_start, candidate.char_end):
        return _result(
            config,
            VerificationDecision.INVALID_SPAN,
            direction_ok=False,
            qualifier_ok=False,
            reason_code="INVALID_SPAN",
            semantic_checked=False,
        )
    if bound_on_head:
        direction_ok = names_match(candidate.head_surface, bound_names)
    else:
        direction_ok = names_match(candidate.tail_surface, bound_names)
    if not direction_ok:
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=False,
            qualifier_ok=True,
            reason_code="DIRECTION_MISMATCH",
            semantic_checked=False,
        )
    if not cue_in_text(candidate.quote, variants):
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=True,
            qualifier_ok=True,
            reason_code="COOCCURRENCE",
            semantic_checked=False,
        )
    if config.verify_text and negation_before_cue(candidate.quote, variants):
        return _result(
            config,
            VerificationDecision.CONFLICT,
            direction_ok=True,
            qualifier_ok=False,
            reason_code="NEGATION",
            semantic_checked=True,
        )
    if config.verify_text and qualifier_conflict(candidate.quote, predicate):
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=True,
            qualifier_ok=False,
            reason_code="QUALIFIER_MISMATCH",
            semantic_checked=True,
        )
    reason = "TEXT_SUPPORTED" if config.verify_text else "TEXT_PROGRAMMATIC_ACCEPT"
    return _result(
        config,
        VerificationDecision.SUPPORTED,
        direction_ok=True,
        qualifier_ok=True,
        reason_code=reason,
        semantic_checked=config.verify_text,
        identity_score=None,
    )


def verify_graph(
    triple: TripleRecord,
    *,
    bound_entity_id: str,
    bound_on_head: bool,
    variants: list[str],
    config: EngineConfig,
    forced: bool,
) -> VerificationResult:
    if bound_on_head:
        direction_ok = triple.head_id == bound_entity_id
    else:
        direction_ok = triple.tail_id == bound_entity_id
    if not direction_ok:
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=False,
            qualifier_ok=True,
            reason_code="DIRECTION_MISMATCH",
            semantic_checked=False,
        )
    relation_ok = relation_matches(triple.relation, variants)
    source_text = " ".join(source.quote or "" for source in triple.sources)
    located = bool(triple.sources) and any((source.quote or source.title) for source in triple.sources)
    if not config.verify_graph:
        if relation_ok or forced:
            return _result(
                config,
                VerificationDecision.SUPPORTED,
                direction_ok=True,
                qualifier_ok=True,
                reason_code="GRAPH_UNVERIFIED_ACCEPT",
                semantic_checked=False,
            )
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=True,
            qualifier_ok=True,
            reason_code="RELATION_MISMATCH",
            semantic_checked=False,
        )
    if not located:
        return _result(
            config,
            VerificationDecision.UNKNOWN,
            direction_ok=True,
            qualifier_ok=True,
            reason_code="SOURCE_UNLOCATABLE",
            semantic_checked=True,
        )
    cue = cue_in_text(source_text, variants)
    negated = negation_before_cue(source_text, variants)
    if relation_ok and cue and not negated:
        return _result(
            config,
            VerificationDecision.SUPPORTED,
            direction_ok=True,
            qualifier_ok=True,
            reason_code="GRAPH_SOURCE_OK",
            semantic_checked=True,
        )
    if negated or not relation_ok or not cue:
        return _result(
            config,
            VerificationDecision.CONFLICT,
            direction_ok=True,
            qualifier_ok=not negated,
            reason_code="GRAPH_SOURCE_REJECT",
            semantic_checked=True,
        )
    return _result(
        config,
        VerificationDecision.UNKNOWN,
        direction_ok=True,
        qualifier_ok=True,
        reason_code="GRAPH_SOURCE_UNKNOWN",
        semantic_checked=True,
    )


def bound_names_for(entity: EntityRecord | None, term: Term, extra: list[str] | None = None) -> list[str]:
    names = list(extra or [])
    names.append(term.surface)
    if entity is not None:
        names.extend(entity.names())
    return names
