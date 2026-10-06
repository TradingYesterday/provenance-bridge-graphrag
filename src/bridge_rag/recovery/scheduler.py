"""Recovery eligibility. Topology first, then query triple index."""

from __future__ import annotations

from bridge_rag.schemas import BranchState, CompiledPlan, Constraint, TermKind


def topo_ranks(plan: CompiledPlan) -> dict[str, int]:
    ranks = {constraint.constraint_id: 0 for constraint in plan.constraints}
    changed = True
    while changed:
        changed = False
        for constraint in plan.constraints:
            if not constraint.depends_on:
                continue
            rank = 1 + max(ranks[item] for item in constraint.depends_on)
            if rank != ranks[constraint.constraint_id]:
                ranks[constraint.constraint_id] = rank
                changed = True
    return ranks


def _active_ids(branch: BranchState) -> set[str]:
    return {proof.constraint_id for proof in branch.accepted_evidence if not proof.invalidated}


def has_successor(constraint: Constraint, plan: CompiledPlan) -> bool:
    return any(constraint.constraint_id in other.depends_on for other in plan.constraints)


def recovery_candidates(
    branch: BranchState,
    plan: CompiledPlan,
    attempts: dict[str, int],
    limit: int,
    *,
    include_terminal: bool = False,
) -> list[Constraint]:
    done = _active_ids(branch)
    ranks = topo_ranks(plan)
    ready: list[Constraint] = []
    for constraint in plan.constraints:
        if constraint.constraint_id in done:
            continue
        if constraint.is_terminal and not include_terminal:
            continue
        if not constraint.is_terminal and not has_successor(constraint, plan):
            continue
        if not all(dep in done for dep in constraint.depends_on):
            continue
        key = attempt_key(branch, constraint)
        if attempts.get(key, 0) >= limit:
            continue
        ready.append(constraint)
    ready.sort(key=lambda item: (ranks[item.constraint_id], item.query_triple_index, item.constraint_id))
    return ready


def attempt_key(branch: BranchState, constraint: Constraint) -> str:
    signature = tuple(
        sorted(
            (name, binding.entity_id or binding.canonical_name)
            for name, binding in branch.bindings.items()
        )
    )
    return f"{constraint.constraint_id}|{signature}"


def writes_entity_variable(plan: CompiledPlan, constraint: Constraint) -> str | None:
    for variable in plan.writes.get(constraint.constraint_id, []):
        for term in (constraint.head_term, constraint.tail_term):
            if term.variable_name == variable and term.value_type is TermKind.ENTITY:
                return variable
    return None
