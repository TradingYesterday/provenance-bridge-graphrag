"""Offline gold used only by the scorer. The executor never receives this object."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RelationGold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    head: str
    relation: str
    tail: str


class MetricGold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    answer: str | None = None
    group: str = "synthetic"
    bridge: RelationGold | None = None
    successor: RelationGold | None = None
    support_quotes: list[str] = Field(default_factory=list)
    support_triple_ids: list[str] = Field(default_factory=list)
