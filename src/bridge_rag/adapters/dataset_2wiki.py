"""2Wiki-shaped records. Gold stays in GoldRecord."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from bridge_rag.schemas import GOLD_FIELD_NAMES, GoldRecord


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def example_id(row: dict[str, Any]) -> str:
    value = row.get("id", row.get("_id", row.get("qid")))
    if value is None:
        raise ValueError("example is missing id/_id/qid")
    return str(value)


def iter_sentences(row: dict[str, Any]) -> Iterator[tuple[int, str, int, str]]:
    context = row.get("context") or row.get("contexts") or []
    if not isinstance(context, list):
        return
    for doc_index, item in enumerate(context):
        title = ""
        sentences: list[str] = []
        if isinstance(item, list) and len(item) >= 2:
            title = str(item[0] or "").strip()
            body = item[1]
            if isinstance(body, list):
                sentences = [str(part) for part in body]
            elif str(body or "").strip():
                sentences = [str(body)]
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            body = item.get("sentences") or item.get("sents") or item.get("text") or []
            if isinstance(body, list):
                sentences = [str(part) for part in body]
            elif str(body or "").strip():
                sentences = [str(body)]
        else:
            continue
        for sent_index, sentence in enumerate(sentences):
            yield doc_index, title, sent_index, sentence


def split_gold(*rows: dict[str, Any]) -> GoldRecord:
    merged: dict[str, Any] = {}
    question_id = ""
    for row in rows:
        if not question_id:
            try:
                question_id = example_id(row)
            except ValueError:
                question_id = str(row.get("id") or "")
        for key, value in row.items():
            if key in GOLD_FIELD_NAMES or key in {"answer", "ground_truth_answer"}:
                merged[key] = value
    return GoldRecord(
        question_id=question_id,
        answer=_as_text(merged.get("answer", merged.get("ground_truth_answer"))),
        supporting_facts=list(merged.get("supporting_facts") or []),
        gold_links=list(merged.get("gold_links") or []),
        missing_edges=list(merged.get("missing_edges") or merged.get("missing_edge") or []),
        extra_gold={key: value for key, value in merged.items() if key not in {"answer", "ground_truth_answer", "supporting_facts", "gold_links", "missing_edges", "missing_edge"}},
    )


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def public_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in GOLD_FIELD_NAMES}


def question_view(row: dict[str, Any]) -> dict[str, Any]:
    return public_fields(row)


def gold_record(row: dict[str, Any]) -> GoldRecord:
    return split_gold(row)


def assert_no_gold_keys(row: dict[str, Any]) -> None:
    leaked = sorted(GOLD_FIELD_NAMES.intersection(row))
    if leaked:
        raise ValueError(f"gold fields leaked into the inference view: {leaked}")
