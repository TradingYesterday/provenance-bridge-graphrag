"""Strict data contracts. Unknown extras may be retained; missing core fields fail."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

GOLD_FIELD_NAMES = frozenset(
    {
        "answer",
        "ground_truth_answer",
        "supporting_facts",
        "gold_links",
        "missing_edge",
        "missing_edges",
        "bridge_label",
    }
)


class TermKind(str, Enum):
    ENTITY = "ENTITY"
    VARIABLE = "VARIABLE"
    LITERAL = "LITERAL"


class BranchStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    AMBIGUOUS = "AMBIGUOUS"
    EXHAUSTED = "EXHAUSTED"
    INVALIDATED = "INVALIDATED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    PLAN_UNSUPPORTED = "PLAN_UNSUPPORTED"


class VerificationDecision(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"
    INVALID_SPAN = "INVALID_SPAN"
    UNLINKABLE = "UNLINKABLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ProofKind(str, Enum):
    GRAPH = "GRAPH"
    TEXT = "TEXT"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Term(StrictModel):
    kind: TermKind
    surface: str
    variable_name: str | None = None
    value_type: TermKind = TermKind.ENTITY
    entity_id: str | None = None
    graph_namespace: str | None = None
    literal_value: str | None = None

    @model_validator(mode="after")
    def _core(self) -> "Term":
        if not self.surface:
            raise ValueError("term.surface is required")
        if self.kind is TermKind.VARIABLE and not self.variable_name:
            raise ValueError("VARIABLE term requires variable_name")
        if self.kind is TermKind.LITERAL and not self.literal_value:
            raise ValueError("LITERAL term requires literal_value")
        if self.entity_id and not self.graph_namespace:
            raise ValueError("entity_id requires graph_namespace")
        return self


class Constraint(StrictModel):
    constraint_id: str
    query_triple_index: int
    head_term: Term
    predicate_text: str
    predicate_variants: list[str] = Field(default_factory=list)
    tail_term: Term
    depends_on: list[str] = Field(default_factory=list)
    is_terminal: bool

    @model_validator(mode="after")
    def _core(self) -> "Constraint":
        if not self.constraint_id:
            raise ValueError("constraint_id is required")
        if not self.predicate_text:
            raise ValueError("predicate_text is required")
        if self.predicate_text not in self.predicate_variants:
            self.predicate_variants = [self.predicate_text, *self.predicate_variants]
        if self.head_term.kind is TermKind.LITERAL and self.tail_term.kind is TermKind.LITERAL:
            raise ValueError("a constraint cannot have two literal endpoints")
        return self


class SourceLocation(StrictModel):
    source_id: str
    title: str | None = None
    doc_index: int | None = None
    sent_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    quote: str | None = None
    unit_kind: str = "unknown"
    ambiguous: bool = False


class RecoveryCandidate(StrictModel):
    candidate_id: str
    constraint_id: str
    branch_id: str
    head_surface: str
    predicate_surface: str
    tail_surface: str
    source_doc_id: str
    title: str | None = None
    passage_id: str
    char_start: int
    char_end: int
    quote: str
    extractor_model: str
    prompt_hash: str
    raw_response_hash: str
    retrieval_rank: int = 0
    reranker_score: float | None = None


class LinkedBinding(StrictModel):
    binding_id: str
    variable: str
    entity_id: str | None = None
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    link_score: float | None = None
    membership_in_current_graph: bool
    addressable_for_successor: bool
    is_literal: bool = False
    graph_namespace: str | None = None


class VerificationResult(StrictModel):
    decision: VerificationDecision
    entailment_score: float | None = None
    identity_score: float | None = None
    direction_ok: bool
    qualifier_ok: bool
    reason_code: str
    verifier_model: str
    prompt_hash: str
    semantic_checked: bool = False


class ProofItem(StrictModel):
    proof_id: str
    branch_id: str
    kind: ProofKind
    constraint_id: str
    query_triple_index: int
    head_surface: str
    relation: str
    tail_surface: str
    head_entity_id: str | None = None
    tail_entity_id: str | None = None
    tail_literal: str | None = None
    source: SourceLocation | None = None
    parent_binding_id: str | None = None
    produces_variable: str | None = None
    produces_entity_id: str | None = None
    acquisition_order: int
    verification_status: VerificationDecision
    invalidated: bool = False
    reason_code: str | None = None


class BudgetView(StrictModel):
    max_steps: int
    steps_used: int
    max_active_branches: int
    recoveries_used: int = 0


class BranchState(StrictModel):
    question_id: str
    branch_id: str
    parent_branch_id: str | None = None
    bindings: dict[str, LinkedBinding] = Field(default_factory=dict)
    pending_constraints: list[str] = Field(default_factory=list)
    accepted_evidence: list[ProofItem] = Field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)
    depth: int = 0
    budget: BudgetView
    status: BranchStatus = BranchStatus.ACTIVE
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _core(self) -> "BranchState":
        if not self.question_id or not self.branch_id:
            raise ValueError("branch requires question_id and branch_id")
        return self


class QuestionCase(StrictModel):
    question_id: str
    question: str
    dataset: str
    data_version: str
    graph_namespace: str
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_gold(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaked = sorted(GOLD_FIELD_NAMES.intersection(data))
            if leaked:
                raise ValueError(f"QuestionCase cannot carry gold fields: {leaked}")
        return data


class GoldRecord(StrictModel):
    """Evaluation-only record. Never pass this object into the executor."""

    question_id: str
    answer: str | None = None
    supporting_facts: list[Any] = Field(default_factory=list)
    gold_links: list[Any] = Field(default_factory=list)
    missing_edges: list[Any] = Field(default_factory=list)
    extra_gold: dict[str, Any] = Field(default_factory=dict)


class PlanTriple(StrictModel):
    head: str
    relation: str
    relation_variants: list[str] = Field(default_factory=list)
    tail: str
    head_kind: TermKind | None = None
    tail_kind: TermKind | None = None
    head_namespace: str | None = None
    tail_namespace: str | None = None


class CompiledPlan(StrictModel):
    constraints: list[Constraint]
    producers: dict[str, str] = Field(default_factory=dict)
    reads: dict[str, list[str]] = Field(default_factory=dict)
    writes: dict[str, list[str]] = Field(default_factory=dict)
    answer_variable: str | None = None
    supported: bool = True
    reason_code: str | None = None


class EngineConfig(StrictModel):
    allow_text_binding: bool = True
    allow_successor_reentry: bool = True
    verify_text: bool = True
    verify_graph: bool = True
    max_active_branches: int = 3
    max_recoveries_per_key: int = 1
    text_top_k: int = 8
    max_relations_to_verify: int = 3
    max_link_candidates: int = 3
    max_steps: int = 32
    verifier_model: str = "programmatic-v1"
    prompt_hash: str = "programmatic-v1"


class RunResult(StrictModel):
    question_id: str
    status: BranchStatus
    uncertain: bool = False
    answer: str | None = None
    answers: list[str] = Field(default_factory=list)
    branches: list[BranchState] = Field(default_factory=list)
    proofs: list[ProofItem] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    graph_fingerprint_before: str = ""
    graph_fingerprint_after: str = ""
    graph_writes: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    cache_keys: list[str] = Field(default_factory=list)
