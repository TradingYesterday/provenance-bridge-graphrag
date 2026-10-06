"""Locate a quote in raw sentences. Title alone is not a unique id."""

from __future__ import annotations

from dataclasses import dataclass

from bridge_rag.adapters.dataset_2wiki import iter_sentences
from bridge_rag.backends.text import TextUnit
from bridge_rag.schemas import SourceLocation


@dataclass
class LocatedQuote:
    locations: list[SourceLocation]
    ambiguous: bool
    reason: str


def locate_quote(units: list[TextUnit], title: str | None, quote: str | None) -> list[TextUnit]:
    hits: list[TextUnit] = []
    for unit in units:
        if title and unit.title != title:
            continue
        if quote is not None and quote not in unit.raw_text:
            continue
        hits.append(unit)
    return hits


def source_from_hits(hits: list[TextUnit], *, quote: str | None, fallback_title: str, fallback_id: str) -> SourceLocation:
    if len(hits) != 1:
        return SourceLocation(
            source_id=fallback_id,
            title=fallback_title,
            quote=quote,
            unit_kind="sentence",
            ambiguous=True,
        )
    unit = hits[0]
    start = unit.raw_text.find(quote) if quote else None
    end = None if start is None or start < 0 else start + len(quote)
    return SourceLocation(
        source_id=unit.source_id(),
        title=unit.title,
        doc_index=unit.doc_index,
        sent_index=unit.sent_index,
        char_start=start,
        char_end=end,
        quote=quote,
        unit_kind=unit.unit_kind,
        ambiguous=False,
    )


def locate_in_example(
    example: dict,
    *,
    data_version: str,
    question_id: str,
    title: str | None,
    quote: str | None,
    doc_index: int | None = None,
    sent_index: int | None = None,
) -> LocatedQuote:
    hits: list[SourceLocation] = []
    for raw_doc, raw_title, raw_sent, sentence in iter_sentences(example):
        if doc_index is not None and raw_doc != doc_index:
            continue
        if sent_index is not None and raw_sent != sent_index:
            continue
        if title and raw_title != title:
            continue
        if quote:
            start = sentence.find(quote)
            if start < 0:
                continue
            end = start + len(quote)
        else:
            start = None
            end = None
        hits.append(
            SourceLocation(
                source_id=f"{data_version}:{question_id}:{raw_doc}:{raw_sent}",
                title=raw_title,
                doc_index=raw_doc,
                sent_index=raw_sent,
                char_start=start,
                char_end=end,
                quote=quote,
                unit_kind="sentence",
                ambiguous=False,
            )
        )
    if len(hits) == 1:
        return LocatedQuote(hits, False, "unique")
    if not hits:
        return LocatedQuote([], True, "unlocated")
    for hit in hits:
        hit.ambiguous = True
    return LocatedQuote(hits, True, "multiple_spans")
