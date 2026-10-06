"""Copy-on-write commit and cascading revoke. The KG is not updated."""

from __future__ import annotations

from bridge_rag.schemas import (
    BranchState,
    BranchStatus,
    CompiledPlan,
    Constraint,
    LinkedBinding,
    ProofItem,
    ProofKind,
    RecoveryCandidate,
    SourceLocation,
    VerificationDecision,
)


def fork_branch(parent: BranchState, branch_id: str) -> BranchState:
    child = parent.model_copy(deep=True)
    child.branch_id = branch_id
    child.parent_branch_id = parent.branch_id
    child.depth = parent.depth + 1
    child.status = BranchStatus.ACTIVE
    child.reason_codes = [code for code in child.reason_codes if code != "FORKED"]
    for proof in child.accepted_evidence:
        proof.branch_id = branch_id
    return child


def commit_text_binding(
    parent: BranchState,
    *,
    branch_id: str,
    constraint: Constraint,
    binding: LinkedBinding,
    candidate: RecoveryCandidate,
    proof_id: str,
    order: int,
    parent_binding_id: str | None,
) -> BranchState:
    if not binding.membership_in_current_graph or not binding.addressable_for_successor:
        raise ValueError("text commit requires an addressable graph member")
    if binding.entity_id is None:
        raise ValueError("text commit cannot invent an entity id")
    child = fork_branch(parent, branch_id)
    child.bindings[binding.variable] = binding
    child.accepted_evidence.append(
        ProofItem(
            proof_id=proof_id,
            branch_id=branch_id,
            kind=ProofKind.TEXT,
            constraint_id=constraint.constraint_id,
            query_triple_index=constraint.query_triple_index,
            head_surface=candidate.head_surface,
            relation=candidate.predicate_surface,
            tail_surface=candidate.tail_surface,
            head_entity_id=binding.entity_id if constraint.head_term.variable_name == binding.variable else None,
            tail_entity_id=binding.entity_id if constraint.tail_term.variable_name == binding.variable else None,
            source=SourceLocation(
                source_id=candidate.passage_id,
                title=candidate.title,
                char_start=candidate.char_start,
                char_end=candidate.char_end,
                quote=candidate.quote,
                unit_kind="sentence",
            ),
            parent_binding_id=parent_binding_id,
            produces_variable=binding.variable,
            produces_entity_id=binding.entity_id,
            acquisition_order=order,
            verification_status=VerificationDecision.SUPPORTED,
            reason_code="TEXT_COMMIT",
        )
    )
    return child


def dependent_variables(root: str, plan: CompiledPlan) -> set[str]:
    doomed = {root}
    changed = True
    while changed:
        changed = False
        for constraint in plan.constraints:
            reads = set(plan.reads.get(constraint.constraint_id, []))
            writes = set(plan.writes.get(constraint.constraint_id, []))
            extra = writes - doomed
            if reads & doomed and extra:
                doomed |= extra
                changed = True
    return doomed


def revoke_variable(branch: BranchState, variable: str, plan: CompiledPlan) -> BranchState:
    doomed = dependent_variables(variable, plan)
    doomed_binding_ids = {
        binding.binding_id for name, binding in branch.bindings.items() if name in doomed
    }
    for proof in branch.accepted_evidence:
        if proof.produces_variable in doomed or (
            proof.parent_binding_id and proof.parent_binding_id in doomed_binding_ids
        ):
            proof.invalidated = True
            proof.reason_code = "REVOKED"
    branch.bindings = {name: binding for name, binding in branch.bindings.items() if name not in doomed}
    branch.status = BranchStatus.INVALIDATED
    if "REVOKED" not in branch.reason_codes:
        branch.reason_codes.append("REVOKED")
    return branch
