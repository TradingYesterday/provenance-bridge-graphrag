"""Deterministic branch engine. Oracle candidates in, no model calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from bridge_rag.backends.graph import EntityRecord, GraphStore, TripleRecord
from bridge_rag.backends.text import TextStore, TextUnit
from bridge_rag.recovery.commit import commit_text_binding, dependent_variables, revoke_variable
from bridge_rag.recovery.extractor import extract_scripted
from bridge_rag.recovery.linker import link_surface
from bridge_rag.recovery.scheduler import attempt_key, recovery_candidates, topo_ranks
from bridge_rag.recovery.verifier import verify_graph, verify_text
from bridge_rag.runtime.budget import BudgetLedger
from bridge_rag.runtime.cache import cache_key
from bridge_rag.runtime.cache import CandidateCache
from bridge_rag.runtime.trace import TraceLog
from bridge_rag.execution.proof import active_chain
from bridge_rag.execution.successor import enumerate_side
from bridge_rag.schemas import (
    BranchState,
    BranchStatus,
    BudgetView,
    CompiledPlan,
    Constraint,
    EngineConfig,
    LinkedBinding,
    ProofItem,
    ProofKind,
    RecoveryCandidate,
    RunResult,
    SourceLocation,
    Term,
    TermKind,
    VerificationDecision,
)


@dataclass
class Shape:
    kind: str
    bound_entity: EntityRecord | None = None
    bound_on_head: bool = True
    free_term: Term | None = None
    free_variable: str | None = None
    other_entity: EntityRecord | None = None
    literal_value: str | None = None
    bound_names: list[str] = field(default_factory=list)


@dataclass
class RunInput:
    question_id: str
    plan: CompiledPlan
    graph: GraphStore
    texts: TextStore
    scripted: dict[str, list[RecoveryCandidate]]
    forced_graph_ids: dict[str, list[str]] = field(default_factory=dict)
    config: EngineConfig = field(default_factory=EngineConfig)
    data_version: str = "diag-v1"
    revoke_variable: str | None = None
    trace_path: str | None = None


def _active(branch: BranchState) -> list[ProofItem]:
    return [proof for proof in branch.accepted_evidence if not proof.invalidated]


def _satisfied(branch: BranchState, constraint: Constraint) -> bool:
    return any(proof.constraint_id == constraint.constraint_id for proof in _active(branch))


def _resolve(term: Term, branch: BranchState, graph: GraphStore):
    if term.kind is TermKind.LITERAL:
        return ("literal", term.literal_value)
    if term.kind is TermKind.VARIABLE and term.variable_name:
        binding = branch.bindings.get(term.variable_name)
        if binding is None:
            return ("unbound", term)
        if binding.is_literal:
            return ("literal_binding", binding)
        entity = graph.entity(binding.entity_id or "")
        if entity is None:
            return ("missing", term.surface)
        return ("entity_binding", entity)
    if term.entity_id:
        entity = graph.entity(term.entity_id)
        return ("entity", entity) if entity else ("missing", term.surface)
    matches = graph.match_surface(term.surface)
    if len(matches) == 1:
        return ("entity", matches[0])
    if not matches:
        return ("missing", term.surface)
    return ("ambiguous", matches)


def _entity_of(resolved) -> EntityRecord | None:
    if resolved[0] in ("entity", "entity_binding"):
        return resolved[1]
    return None


def classify(constraint: Constraint, branch: BranchState, graph: GraphStore) -> Shape:
    left = _resolve(constraint.head_term, branch, graph)
    right = _resolve(constraint.tail_term, branch, graph)
    if left[0] == "ambiguous" or right[0] == "ambiguous":
        return Shape(kind="ambiguous")
    left_ent = _entity_of(left)
    right_ent = _entity_of(right)
    if left[0] == "unbound" and right_ent is not None:
        term = left[1]
        return Shape(
            kind="one_entity_one_var",
            bound_entity=right_ent,
            bound_on_head=False,
            free_term=term,
            free_variable=term.variable_name,
            bound_names=right_ent.names(),
        )
    if right[0] == "unbound" and left_ent is not None:
        term = right[1]
        return Shape(
            kind="one_entity_one_var",
            bound_entity=left_ent,
            bound_on_head=True,
            free_term=term,
            free_variable=term.variable_name,
            bound_names=left_ent.names(),
        )
    if left_ent is not None and right_ent is not None:
        return Shape(kind="both_entity", bound_entity=left_ent, other_entity=right_ent, bound_on_head=True)
    if left_ent is not None and right[0] == "literal":
        return Shape(kind="entity_literal", bound_entity=left_ent, bound_on_head=True, literal_value=right[1])
    if right_ent is not None and left[0] == "literal":
        return Shape(kind="entity_literal", bound_entity=right_ent, bound_on_head=False, literal_value=left[1])
    return Shape(kind="wait")


def _parent_binding_id(branch: BranchState, plan: CompiledPlan, constraint: Constraint) -> str | None:
    for variable in plan.reads.get(constraint.constraint_id, []):
        binding = branch.bindings.get(variable)
        if binding is not None:
            return binding.binding_id
    return None


def _budget_view(ledger: BudgetLedger) -> BudgetView:
    raw = ledger.view()
    return BudgetView(
        max_steps=raw["max_steps"],
        steps_used=raw["steps_used"],
        max_active_branches=raw["max_active_branches"],
        recoveries_used=raw["recoveries_used"],
    )


def _refresh(branch: BranchState, plan: CompiledPlan, ledger: BudgetLedger) -> None:
    branch.pending_constraints = [
        constraint.constraint_id
        for constraint in plan.constraints
        if not _satisfied(branch, constraint)
    ]
    branch.budget = _budget_view(ledger)


def _consistent(branch: BranchState) -> bool:
    for proof in _active(branch):
        variable = proof.produces_variable
        if not variable:
            continue
        binding = branch.bindings.get(variable)
        if binding is None:
            return False
        if binding.is_literal:
            if proof.tail_literal != binding.canonical_name:
                return False
        elif proof.produces_entity_id != binding.entity_id:
            return False
    return True


def _complete(branch: BranchState, plan: CompiledPlan) -> bool:
    if not plan.supported or not plan.constraints:
        return False
    if any(not _satisfied(branch, constraint) for constraint in plan.constraints):
        return False
    if plan.answer_variable and plan.answer_variable not in branch.bindings:
        return False
    return _consistent(branch)


def _answer(branch: BranchState, plan: CompiledPlan) -> str | None:
    if not plan.answer_variable:
        return None
    binding = branch.bindings.get(plan.answer_variable)
    if binding is None:
        return None
    return binding.canonical_name


class _Ids:
    def __init__(self) -> None:
        self.branch = 0
        self.proof = 0
        self.binding = 0

    def next_branch(self) -> str:
        value = f"b{self.branch}"
        self.branch += 1
        return value

    def next_proof(self) -> str:
        value = f"p{self.proof}"
        self.proof += 1
        return value

    def next_binding(self, variable: str) -> str:
        value = f"bind-{variable}-{self.binding}"
        self.binding += 1
        return value


def _add_graph_proof(
    branch: BranchState,
    *,
    ids: _Ids,
    order: int,
    constraint: Constraint,
    triple: TripleRecord,
    graph: GraphStore,
    produces: str | None,
    produces_entity: str | None,
    literal: str | None,
    parent_binding_id: str | None,
    reason: str,
) -> None:
    head = graph.entity(triple.head_id)
    tail = graph.entity(triple.tail_id) if triple.tail_id else None
    source = triple.sources[0] if triple.sources else None
    branch.accepted_evidence.append(
        ProofItem(
            proof_id=ids.next_proof(),
            branch_id=branch.branch_id,
            kind=ProofKind.GRAPH,
            constraint_id=constraint.constraint_id,
            query_triple_index=constraint.query_triple_index,
            head_surface=head.canonical_name if head else triple.head_id,
            relation=triple.relation,
            tail_surface=literal or (tail.canonical_name if tail else (triple.tail_literal or "")),
            head_entity_id=triple.head_id,
            tail_entity_id=None if literal else triple.tail_id,
            tail_literal=literal,
            source=source,
            parent_binding_id=parent_binding_id,
            produces_variable=produces,
            produces_entity_id=produces_entity,
            acquisition_order=order,
            verification_status=VerificationDecision.SUPPORTED,
            reason_code=reason,
        )
    )


def _bind_entity(ids: _Ids, variable: str, entity: EntityRecord) -> LinkedBinding:
    return LinkedBinding(
        binding_id=ids.next_binding(variable),
        variable=variable,
        entity_id=entity.entity_id,
        canonical_name=entity.canonical_name,
        aliases=list(entity.aliases),
        link_score=1.0,
        membership_in_current_graph=True,
        addressable_for_successor=True,
        graph_namespace=entity.namespace,
    )


def _bind_literal(ids: _Ids, variable: str, value: str, namespace: str) -> LinkedBinding:
    return LinkedBinding(
        binding_id=ids.next_binding(variable),
        variable=variable,
        entity_id=None,
        canonical_name=value,
        link_score=None,
        membership_in_current_graph=False,
        addressable_for_successor=False,
        is_literal=True,
        graph_namespace=namespace,
    )


def graph_closure(
    branch: BranchState,
    run: RunInput,
    ids: _Ids,
    order_box: list[int],
    trace: TraceLog,
) -> None:
    ranks = topo_ranks(run.plan)
    ordered = sorted(run.plan.constraints, key=lambda item: (ranks[item.constraint_id], item.query_triple_index))
    progressed = True
    while progressed:
        progressed = False
        for constraint in ordered:
            if _satisfied(branch, constraint):
                continue
            shape = classify(constraint, branch, run.graph)
            if shape.kind == "ambiguous":
                if "AMBIGUOUS_ENTITY" not in branch.reason_codes:
                    branch.reason_codes.append("AMBIGUOUS_ENTITY")
                continue
            if shape.kind not in ("one_entity_one_var", "both_entity", "entity_literal"):
                continue
            forced = run.forced_graph_ids.get(constraint.constraint_id)
            if shape.kind == "both_entity":
                assert shape.bound_entity and shape.other_entity
                pool = [
                    triple
                    for triple in run.graph.outgoing(shape.bound_entity.entity_id)
                    if triple.tail_id == shape.other_entity.entity_id
                ]
                forced_flag = False
            elif shape.kind == "entity_literal":
                assert shape.bound_entity and shape.literal_value is not None
                pool = [
                    triple
                    for triple in (
                        run.graph.outgoing(shape.bound_entity.entity_id)
                        if shape.bound_on_head
                        else run.graph.incoming(shape.bound_entity.entity_id)
                    )
                    if triple.tail_literal == shape.literal_value
                ]
                forced_flag = False
            else:
                assert shape.bound_entity
                use_forced = forced if shape.free_term and shape.free_term.value_type is TermKind.ENTITY else None
                pool = enumerate_side(run.graph, shape.bound_entity.entity_id, shape.bound_on_head, use_forced)
                forced_flag = use_forced is not None
            trace.add(
                "graph_access",
                branch_id=branch.branch_id,
                constraint_id=constraint.constraint_id,
                candidate_ids=[triple.triple_id for triple in pool],
            )
            accepted: TripleRecord | None = None
            accept_reason = ""
            for triple in pool:
                verdict = verify_graph(
                    triple,
                    bound_entity_id=shape.bound_entity.entity_id,
                    bound_on_head=shape.bound_on_head if shape.kind != "both_entity" else True,
                    variants=constraint.predicate_variants,
                    config=run.config,
                    forced=forced_flag,
                )
                trace.add(
                    "verify",
                    channel="graph",
                    branch_id=branch.branch_id,
                    constraint_id=constraint.constraint_id,
                    triple_id=triple.triple_id,
                    decision=verdict.decision.value,
                    reason_code=verdict.reason_code,
                )
                if verdict.decision is VerificationDecision.SUPPORTED:
                    accepted = triple
                    accept_reason = verdict.reason_code
                    break
                branch.rejected_candidates.append(
                    {
                        "constraint_id": constraint.constraint_id,
                        "triple_id": triple.triple_id,
                        "decision": verdict.decision.value,
                        "reason_code": verdict.reason_code,
                    }
                )
            if accepted is None:
                continue
            produces = None
            produces_entity = None
            literal = None
            if shape.kind == "one_entity_one_var" and shape.free_variable and shape.free_term:
                if shape.free_term.value_type is TermKind.LITERAL:
                    literal = accepted.tail_literal
                    if not literal:
                        continue
                    branch.bindings[shape.free_variable] = _bind_literal(
                        ids, shape.free_variable, literal, run.graph.namespace
                    )
                    produces = shape.free_variable
                else:
                    free_id = accepted.tail_id if shape.bound_on_head else accepted.head_id
                    entity = run.graph.entity(free_id or "")
                    if entity is None:
                        continue
                    branch.bindings[shape.free_variable] = _bind_entity(ids, shape.free_variable, entity)
                    produces = shape.free_variable
                    produces_entity = entity.entity_id
            elif shape.kind == "entity_literal":
                literal = shape.literal_value
            order_box[0] += 1
            _add_graph_proof(
                branch,
                ids=ids,
                order=order_box[0],
                constraint=constraint,
                triple=accepted,
                graph=run.graph,
                produces=produces,
                produces_entity=produces_entity,
                literal=literal,
                parent_binding_id=_parent_binding_id(branch, run.plan, constraint),
                reason=accept_reason,
            )
            trace.add(
                "commit",
                channel="graph",
                branch_id=branch.branch_id,
                constraint_id=constraint.constraint_id,
                produces=produces,
                triple_id=accepted.triple_id,
            )
            progressed = True


def _retrieve(run: RunInput, branch: BranchState, constraint: Constraint, cache: CandidateCache, ledger: BudgetLedger, trace: TraceLog) -> list[TextUnit]:
    signature = {
        name: binding.entity_id or binding.canonical_name for name, binding in sorted(branch.bindings.items())
    }
    key = cache_key(
        {
            "data_version": run.data_version,
            "question_id": run.question_id,
            "constraint_id": constraint.constraint_id,
            "predicate": constraint.predicate_variants,
            "bindings": signature,
            "model": "oracle-scripted",
            "prompt_hash": run.config.prompt_hash,
            "top_k": run.config.text_top_k,
        }
    )
    cached = cache.get(key)
    if cached is None:
        units = sorted(run.texts.visible_units(), key=lambda unit: (unit.doc_index, unit.sent_index, unit.passage_id))
        cached = [unit.passage_id for unit in units[: run.config.text_top_k]]
        cache.put(key, cached)
    else:
        ledger.note_cache_hit()
    trace.add("retrieve", branch_id=branch.branch_id, constraint_id=constraint.constraint_id, cache_key=key, passage_ids=cached)
    return [unit for passage_id in cached if (unit := run.texts.by_id(passage_id)) is not None]


def _recover(
    branch: BranchState,
    run: RunInput,
    ids: _Ids,
    order_box: list[int],
    ledger: BudgetLedger,
    cache: CandidateCache,
    trace: TraceLog,
    room: int,
) -> list[BranchState]:
    ready = recovery_candidates(branch, run.plan, ledger.recoveries, run.config.max_recoveries_per_key)
    actionable: list[tuple[Constraint, Shape]] = []
    for constraint in ready:
        shape = classify(constraint, branch, run.graph)
        if shape.kind == "ambiguous":
            ledger.allow_recovery(attempt_key(branch, constraint))
            if "AMBIGUOUS_ENTITY" not in branch.reason_codes:
                branch.reason_codes.append("AMBIGUOUS_ENTITY")
            continue
        if (
            shape.kind != "one_entity_one_var"
            or shape.free_term is None
            or shape.free_term.value_type is not TermKind.ENTITY
            or not shape.free_variable
        ):
            ledger.allow_recovery(attempt_key(branch, constraint))
            if "NOT_RECOVERABLE" not in branch.reason_codes:
                branch.reason_codes.append("NOT_RECOVERABLE")
            continue
        actionable.append((constraint, shape))
    if not actionable:
        return []
    constraint, shape = actionable[0]
    if not ledger.allow_recovery(attempt_key(branch, constraint)):
        return []
    passages = _retrieve(run, branch, constraint, cache, ledger, trace)
    allowed = {unit.passage_id for unit in passages}
    extracted = extract_scripted(run.scripted, constraint.constraint_id, allowed)
    trace.add(
        "extract",
        branch_id=branch.branch_id,
        constraint_id=constraint.constraint_id,
        count=len(extracted),
        extractor_model="oracle-scripted",
    )
    if len(extracted) > run.config.max_relations_to_verify:
        trace.add(
            "truncate",
            reason_code="RELATION_TRUNCATED",
            kept=run.config.max_relations_to_verify,
            dropped=len(extracted) - run.config.max_relations_to_verify,
        )
        extracted = extracted[: run.config.max_relations_to_verify]
    if not extracted and "NO_TEXT_CANDIDATE" not in branch.reason_codes:
        branch.reason_codes.append("NO_TEXT_CANDIDATE")
    children: list[BranchState] = []
    assert shape.bound_entity is not None and shape.free_variable is not None
    for candidate in extracted:
        unit = run.texts.by_id(candidate.passage_id)
        verdict = verify_text(
            candidate,
            unit,
            bound_names=shape.bound_names,
            bound_on_head=shape.bound_on_head,
            variants=constraint.predicate_variants,
            predicate=constraint.predicate_text,
            config=run.config,
        )
        trace.add(
            "verify",
            channel="text",
            branch_id=branch.branch_id,
            constraint_id=constraint.constraint_id,
            candidate_id=candidate.candidate_id,
            decision=verdict.decision.value,
            reason_code=verdict.reason_code,
            semantic_checked=verdict.semantic_checked,
        )
        if verdict.decision is not VerificationDecision.SUPPORTED:
            branch.rejected_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "decision": verdict.decision.value,
                    "reason_code": verdict.reason_code,
                    "reranker_score": candidate.reranker_score,
                }
            )
            continue
        surface = candidate.tail_surface if shape.bound_on_head else candidate.head_surface
        links, notes = link_surface(
            run.graph,
            surface,
            shape.free_variable,
            binding_id=ids.next_binding(shape.free_variable),
            limit=run.config.max_link_candidates,
        )
        trace.add(
            "link",
            branch_id=branch.branch_id,
            constraint_id=constraint.constraint_id,
            surface=surface,
            notes=notes,
            entity_ids=[link.entity_id for link in links],
        )
        for note in notes:
            if note not in branch.reason_codes:
                branch.reason_codes.append(note)
        link = links[0]
        if "AMBIGUOUS_ENTITY" in notes or "UNLINKABLE" in notes or not link.addressable_for_successor:
            decision = "UNLINKABLE" if "UNLINKABLE" in notes else "UNKNOWN"
            branch.rejected_candidates.append(
                {"candidate_id": candidate.candidate_id, "decision": decision, "reason_code": notes[0]}
            )
            continue
        if not run.config.allow_text_binding:
            trace.add("skip_commit", reason_code="TEXT_BINDING_DISABLED", candidate_id=candidate.candidate_id)
            if "TEXT_BINDING_DISABLED" not in branch.reason_codes:
                branch.reason_codes.append("TEXT_BINDING_DISABLED")
            continue
        if room <= 0 or len(children) >= room:
            trace.add("truncate", reason_code="FORK_TRUNCATED", branch_id=branch.branch_id)
            if "FORK_TRUNCATED" not in branch.reason_codes:
                branch.reason_codes.append("FORK_TRUNCATED")
            break
        order_box[0] += 1
        child = commit_text_binding(
            branch,
            branch_id=ids.next_branch(),
            constraint=constraint,
            binding=link,
            candidate=candidate,
            proof_id=ids.next_proof(),
            order=order_box[0],
            parent_binding_id=_parent_binding_id(branch, run.plan, constraint),
        )
        proof = child.accepted_evidence[-1]
        if unit is not None:
            proof.source = SourceLocation(
                source_id=unit.source_id(),
                title=unit.title,
                doc_index=unit.doc_index,
                sent_index=unit.sent_index,
                char_start=candidate.char_start,
                char_end=candidate.char_end,
                quote=candidate.quote,
                unit_kind=unit.unit_kind,
            )
        if shape.bound_on_head:
            proof.head_entity_id = shape.bound_entity.entity_id
            proof.tail_entity_id = link.entity_id
        else:
            proof.tail_entity_id = shape.bound_entity.entity_id
            proof.head_entity_id = link.entity_id
        proof.produces_entity_id = link.entity_id
        if not run.config.allow_successor_reentry:
            if "REENTRY_DISABLED" not in child.reason_codes:
                child.reason_codes.append("REENTRY_DISABLED")
        trace.add(
            "commit",
            channel="text",
            branch_id=child.branch_id,
            parent_branch_id=branch.branch_id,
            constraint_id=constraint.constraint_id,
            entity_id=link.entity_id,
            reranker_score=candidate.reranker_score,
        )
        children.append(child)
    return children


def _note_successors(branch: BranchState, run: RunInput) -> None:
    if not run.config.allow_successor_reentry:
        return
    for constraint in run.plan.constraints:
        if _satisfied(branch, constraint):
            continue
        reads = set(run.plan.reads.get(constraint.constraint_id, []))
        if not reads & set(branch.bindings):
            continue
        shape = classify(constraint, branch, run.graph)
        if shape.kind == "one_entity_one_var" and "NO_SUCCESSOR_EVIDENCE" not in branch.reason_codes:
            branch.reason_codes.append("NO_SUCCESSOR_EVIDENCE")


def _aggregate(question_id: str, branches: list[BranchState], plan: CompiledPlan, budget_hit: bool) -> tuple[BranchStatus, bool, str | None, list[str]]:
    usable = [branch for branch in branches if "FORKED" not in branch.reason_codes]
    complete = [branch for branch in usable if branch.status is BranchStatus.COMPLETE and _complete(branch, plan)]
    answers: list[str] = []
    for branch in complete:
        value = _answer(branch, plan)
        if value is not None and value not in answers:
            answers.append(value)
    live: dict[str, set[str]] = {}
    for branch in usable:
        if branch.status is BranchStatus.INVALIDATED:
            continue
        for name, binding in branch.bindings.items():
            live.setdefault(name, set()).add(binding.entity_id or f"lit:{binding.canonical_name}")
    competing = any(len(values) > 1 for values in live.values())
    unfinished = any(branch.status is BranchStatus.BUDGET_EXHAUSTED for branch in usable)
    if complete and plan.answer_variable is None:
        return BranchStatus.COMPLETE, False, None, []
    if len(answers) > 1:
        return BranchStatus.AMBIGUOUS, False, None, answers
    if len(answers) == 1:
        uncertain = unfinished or budget_hit or competing
        return BranchStatus.COMPLETE, uncertain, answers[0], answers
    if any(branch.status is BranchStatus.INVALIDATED for branch in usable):
        return BranchStatus.INVALIDATED, False, None, []
    if unfinished or budget_hit:
        return BranchStatus.BUDGET_EXHAUSTED, False, None, []
    return BranchStatus.EXHAUSTED, False, None, []


def execute(run: RunInput) -> RunResult:
    trace = TraceLog(run.trace_path)
    fingerprint_before = run.graph.fingerprint()
    writes_before = run.graph.writes
    if not run.plan.supported:
        trace.add("plan", supported=False, reason_code=run.plan.reason_code)
        return RunResult(
            question_id=run.question_id,
            status=BranchStatus.PLAN_UNSUPPORTED,
            reason_codes=[run.plan.reason_code or "PLAN_UNSUPPORTED"],
            graph_fingerprint_before=fingerprint_before,
            graph_fingerprint_after=run.graph.fingerprint(),
            graph_writes=run.graph.writes - writes_before,
            trace=trace.events,
        )
    ledger = BudgetLedger(
        max_steps=run.config.max_steps,
        max_active_branches=run.config.max_active_branches,
        max_recoveries_per_key=run.config.max_recoveries_per_key,
    )
    cache = CandidateCache()
    ids = _Ids()
    order_box = [0]
    root = BranchState(
        question_id=run.question_id,
        branch_id=ids.next_branch(),
        budget=_budget_view(ledger),
    )
    active = [root]
    finished: list[BranchState] = []
    budget_hit = False
    while active:
        branch = active.pop(0)
        if not ledger.charge_step():
            budget_hit = True
            for pending in [branch, *active]:
                pending.status = BranchStatus.BUDGET_EXHAUSTED
                if "BUDGET_EXHAUSTED" not in pending.reason_codes:
                    pending.reason_codes.append("BUDGET_EXHAUSTED")
                _refresh(pending, run.plan, ledger)
                finished.append(pending)
            active = []
            trace.add("budget", reason_code="BUDGET_EXHAUSTED")
            break
        if branch.parent_branch_id is None or run.config.allow_successor_reentry:
            graph_closure(branch, run, ids, order_box, trace)
        if branch.parent_branch_id is not None:
            _note_successors(branch, run)
        if _complete(branch, run.plan):
            branch.status = BranchStatus.COMPLETE
            _refresh(branch, run.plan, ledger)
            finished.append(branch)
            trace.add("branch", branch_id=branch.branch_id, status="COMPLETE")
            continue
        if not ledger.charge_step():
            budget_hit = True
            branch.status = BranchStatus.BUDGET_EXHAUSTED
            if "BUDGET_EXHAUSTED" not in branch.reason_codes:
                branch.reason_codes.append("BUDGET_EXHAUSTED")
            _refresh(branch, run.plan, ledger)
            finished.append(branch)
            for pending in active:
                pending.status = BranchStatus.BUDGET_EXHAUSTED
                if "BUDGET_EXHAUSTED" not in pending.reason_codes:
                    pending.reason_codes.append("BUDGET_EXHAUSTED")
                _refresh(pending, run.plan, ledger)
                finished.append(pending)
            active = []
            trace.add("budget", reason_code="BUDGET_EXHAUSTED", branch_id=branch.branch_id)
            break
        room = run.config.max_active_branches - len(active)
        children = _recover(branch, run, ids, order_box, ledger, cache, trace, room)
        if children:
            branch.status = BranchStatus.EXHAUSTED
            if "FORKED" not in branch.reason_codes:
                branch.reason_codes.append("FORKED")
            _refresh(branch, run.plan, ledger)
            finished.append(branch)
            active.extend(children)
            continue
        ready_left = recovery_candidates(branch, run.plan, ledger.recoveries, run.config.max_recoveries_per_key)
        if ready_left:
            active.append(branch)
            continue
        branch.status = BranchStatus.EXHAUSTED
        _refresh(branch, run.plan, ledger)
        finished.append(branch)
        trace.add("branch", branch_id=branch.branch_id, status="EXHAUSTED", reasons=list(branch.reason_codes))
    if run.revoke_variable:
        for branch in finished:
            if run.revoke_variable in branch.bindings or any(
                proof.produces_variable == run.revoke_variable for proof in branch.accepted_evidence
            ):
                revoke_variable(branch, run.revoke_variable, run.plan)
                trace.add("revoke", branch_id=branch.branch_id, variable=run.revoke_variable, doomed=sorted(dependent_variables(run.revoke_variable, run.plan)))
                _refresh(branch, run.plan, ledger)
    status, uncertain, answer, answers = _aggregate(run.question_id, finished, run.plan, budget_hit)
    answer_proofs = [
        proof
        for branch in finished
        if "FORKED" not in branch.reason_codes and branch.status is not BranchStatus.INVALIDATED
        for proof in branch.accepted_evidence
    ]
    trace.add(
        "replay",
        proof_ids=[proof.proof_id for proof in active_chain(answer_proofs)],
        status=status.value,
        uncertain=uncertain,
        answer=answer,
    )
    proofs = [proof for branch in finished for proof in branch.accepted_evidence]
    reason_codes: list[str] = []
    for branch in finished:
        for code in branch.reason_codes:
            if code not in reason_codes:
                reason_codes.append(code)
    for event in trace.events:
        code = event.get("reason_code")
        if isinstance(code, str) and code not in reason_codes:
            reason_codes.append(code)
    return RunResult(
        question_id=run.question_id,
        status=status,
        uncertain=uncertain,
        answer=answer,
        answers=answers,
        branches=finished,
        proofs=proofs,
        reason_codes=reason_codes,
        graph_fingerprint_before=fingerprint_before,
        graph_fingerprint_after=run.graph.fingerprint(),
        graph_writes=run.graph.writes - writes_before,
        trace=trace.events,
        cache_keys=list(cache.keys_seen),
    )
