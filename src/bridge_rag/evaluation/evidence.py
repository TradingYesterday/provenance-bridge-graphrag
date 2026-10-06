"""Evidence, binding, proof, and answer metrics.

An empty prediction does not receive precision 1. Precision is null and coverage is reported.
"""

from __future__ import annotations

from bridge_rag.evaluation.gold import MetricGold, RelationGold
from bridge_rag.execution.proof import active_chain
from bridge_rag.schemas import RunResult


def _tokens(text: str) -> list[str]:
    return [char for char in text.casefold() if char.isalnum() or "\u4e00" <= char <= "\u9fff"]


def exact_match(predicted: str | None, gold: str | None) -> bool | None:
    if gold is None:
        return None
    if predicted is None:
        return False
    return _tokens(predicted) == _tokens(gold)


def token_f1(predicted: str | None, gold: str | None) -> float | None:
    if gold is None:
        return None
    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return None
    if predicted is None:
        return 0.0
    pred_tokens = _tokens(predicted)
    if not pred_tokens:
        return 0.0
    overlap = 0
    pool = list(pred_tokens)
    for token in gold_tokens:
        if token in pool:
            overlap += 1
            pool.remove(token)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _pair(relation: RelationGold) -> tuple[str, str, str]:
    return (relation.head, relation.relation, relation.tail)


def _answer_proofs(result: RunResult):
    forked = {branch.branch_id for branch in result.branches if "FORKED" in branch.reason_codes}
    return [proof for proof in active_chain(result.proofs) if proof.branch_id not in forked]


def _predicted_pairs(result: RunResult) -> list[tuple[str, str, str]]:
    return [(proof.head_surface, proof.relation, proof.tail_surface) for proof in _answer_proofs(result)]


def _has_pair(result: RunResult, relation: RelationGold | None) -> bool:
    if relation is None:
        return False
    wanted = _pair(relation)
    return wanted in _predicted_pairs(result)


def proof_precision(result: RunResult, gold: MetricGold) -> float | None:
    predicted = _predicted_pairs(result)
    if not predicted:
        return None
    allowed = set()
    if gold.bridge is not None:
        allowed.add(_pair(gold.bridge))
    if gold.successor is not None:
        allowed.add(_pair(gold.successor))
    if not allowed:
        return None
    hits = sum(1 for pair in predicted if pair in allowed)
    return hits / len(predicted)


def complete_support(result: RunResult, gold: MetricGold) -> bool:
    quotes = [proof.source.quote for proof in _answer_proofs(result) if proof.source and proof.source.quote]
    text = " ".join(quotes)
    for quote in gold.support_quotes:
        if quote not in text:
            return False
    if not gold.support_triple_ids:
        return True
    triple_ids = {
        str(event["triple_id"])
        for event in result.trace
        if event.get("event") == "commit" and event.get("channel") == "graph" and event.get("triple_id")
    }
    return all(triple_id in triple_ids for triple_id in gold.support_triple_ids)


def score_case(result: RunResult, gold: MetricGold) -> dict:
    predicted = _predicted_pairs(result)
    precision = proof_precision(result, gold)
    return {
        "question_id": gold.question_id,
        "group": gold.group,
        "status": result.status.value,
        "answer": result.answer,
        "exact_match": exact_match(result.answer, gold.answer),
        "token_f1": token_f1(result.answer, gold.answer),
        "bridge_acquired": _has_pair(result, gold.bridge),
        "successor_acquired": _has_pair(result, gold.successor),
        "complete_support": complete_support(result, gold),
        "proof_precision": precision,
        "proof_count": len(predicted),
        "precision_coverage": precision is not None,
        "graph_writes": result.graph_writes,
        "uncertain": result.uncertain,
    }
