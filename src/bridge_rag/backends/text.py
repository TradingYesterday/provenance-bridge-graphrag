"""Text units. Character offsets refer to the raw sentence, not a title-joined string."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextUnit:
    passage_id: str
    question_id: str
    data_version: str
    doc_index: int
    sent_index: int
    title: str
    raw_text: str
    unit_kind: str = "sentence"
    visible: bool = True

    def source_id(self) -> str:
        return f"{self.data_version}:{self.question_id}:{self.doc_index}:{self.sent_index}"


@dataclass
class TextStore:
    units: list[TextUnit] = field(default_factory=list)

    def by_id(self, passage_id: str) -> TextUnit | None:
        for unit in self.units:
            if unit.passage_id == passage_id:
                return unit
        return None

    def visible_units(self) -> list[TextUnit]:
        return [unit for unit in self.units if unit.visible]

    def locate_by_title(self, title: str, quote: str | None = None) -> list[TextUnit]:
        hits = [unit for unit in self.units if unit.title == title]
        if quote is None:
            return hits
        return [unit for unit in hits if quote in unit.raw_text]
