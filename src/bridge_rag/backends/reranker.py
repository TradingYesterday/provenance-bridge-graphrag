"""Deterministic lexical stand-in.

This is not the upstream cross-encoder. Calls are labeled deterministic-lexical-v1.
"""

from __future__ import annotations


def _tokens(text: str) -> set[str]:
    return {part for part in text.casefold().split() if part}


def lexical_scores(pairs: list[tuple[str, str]]) -> list[float]:
    scores: list[float] = []
    for query, passage in pairs:
        left = _tokens(query)
        right = _tokens(passage)
        if not left or not right:
            scores.append(0.0)
            continue
        scores.append(len(left & right) / len(left | right))
    return scores
