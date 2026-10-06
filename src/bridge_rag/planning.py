"""Constraint compilation from variable read/write structure.

Array order is not a dependency. Cycles and two writers for one variable
are PLAN_UNSUPPORTED.
"""

from __future__ import annotations

from bridge_rag.schemas import CompiledPlan, Constraint, PlanTriple, Term, TermKind


def _term(
    surface: str,
    kind: TermKind | None,
    namespace: str,
    explicit_namespace: str | None,
) -> Term:
    ns = explicit_namespace or namespace
    if kind is TermKind.LITERAL or (kind is None and surface.startswith("lit:")):
        value = surface[4:] if surface.startswith("lit:") else surface
        return Term(kind=TermKind.LITERAL, surface=surface, literal_value=value, value_type=TermKind.LITERAL)
    if kind is TermKind.VARIABLE or (kind is None and surface.startswith("?")):
        value_type = TermKind.LITERAL if kind is TermKind.LITERAL else TermKind.ENTITY
        if kind is TermKind.VARIABLE and surface.startswith("lit?"):
            value_type = TermKind.LITERAL
        return Term(
            kind=TermKind.VARIABLE,
            surface=surface,
            variable_name=surface,
            value_type=value_type,
            graph_namespace=ns,
        )
    return Term(kind=TermKind.ENTITY, surface=surface, graph_namespace=ns, value_type=TermKind.ENTITY)


def parse_term(surface: str, kind: TermKind | None, namespace: str, explicit_namespace: str | None) -> Term:
    if kind is TermKind.LITERAL and surface.startswith("?"):
        return Term(
            kind=TermKind.VARIABLE,
            surface=surface,
            variable_name=surface,
            value_type=TermKind.LITERAL,
            graph_namespace=explicit_namespace or namespace,
        )
    return _term(surface, kind, namespace, explicit_namespace)


def _endpoint_vars(term: Term) -> list[str]:
    if term.kind is TermKind.VARIABLE and term.variable_name:
        return [term.variable_name]
    return []


def compile_plan(triples: list[PlanTriple], namespace: str) -> CompiledPlan:
    constraints: list[Constraint] = []
    for index, triple in enumerate(triples):
        variants = list(triple.relation_variants)
        if triple.relation and triple.relation not in variants:
            variants = [triple.relation, *variants]
        constraints.append(
            Constraint(
                constraint_id=f"c{index}",
                query_triple_index=index,
                head_term=parse_term(triple.head, triple.head_kind, namespace, triple.head_namespace),
                predicate_text=triple.relation,
                predicate_variants=variants,
                tail_term=parse_term(triple.tail, triple.tail_kind, namespace, triple.tail_namespace),
                depends_on=[],
                is_terminal=False,
            )
        )
    if not constraints:
        return CompiledPlan(constraints=[], supported=False, reason_code="PLAN_EMPTY")

    produced: dict[str, str] = {}
    depends: dict[str, list[str]] = {c.constraint_id: [] for c in constraints}
    settled: set[str] = set()

    def grounded(term: Term) -> bool:
        if term.kind in (TermKind.ENTITY, TermKind.LITERAL):
            return True
        return bool(term.variable_name and term.variable_name in produced)

    while True:
        proposals: list[tuple[str, str, list[str]]] = []
        check_settles: list[tuple[str, list[str]]] = []
        for constraint in constraints:
            if constraint.constraint_id in settled:
                continue
            ends = [constraint.head_term, constraint.tail_term]
            ungrounded = [t for t in ends if t.kind is TermKind.VARIABLE and t.variable_name not in produced]
            if len(ungrounded) > 1:
                continue
            if len(ungrounded) == 1:
                variable = ungrounded[0].variable_name or ""
                other = ends[0] if ends[1] is ungrounded[0] else ends[1]
                if not grounded(other):
                    continue
                deps: list[str] = []
                if other.kind is TermKind.VARIABLE and other.variable_name in produced:
                    deps.append(produced[other.variable_name])
                proposals.append((variable, constraint.constraint_id, deps))
                continue
            if all(grounded(t) for t in ends):
                deps = []
                for term in ends:
                    if term.kind is TermKind.VARIABLE and term.variable_name in produced:
                        deps.append(produced[term.variable_name])
                check_settles.append((constraint.constraint_id, deps))
        if not proposals and not check_settles:
            break
        by_var: dict[str, list[tuple[str, list[str]]]] = {}
        for variable, cid, deps in proposals:
            by_var.setdefault(variable, []).append((cid, deps))
        for variable, group in by_var.items():
            writers = {cid for cid, _ in group}
            if variable in produced or len(writers) > 1:
                return CompiledPlan(
                    constraints=constraints,
                    supported=False,
                    reason_code="PLAN_AMBIGUOUS_WRITER",
                )
        for variable, group in by_var.items():
            cid, deps = group[0]
            produced[variable] = cid
            depends[cid] = sorted(set(deps))
            settled.add(cid)
        for cid, deps in check_settles:
            depends[cid] = sorted(set(deps))
            settled.add(cid)

    if len(settled) != len(constraints):
        remaining = [c for c in constraints if c.constraint_id not in settled]
        has_constant = False
        for constraint in remaining:
            for term in (constraint.head_term, constraint.tail_term):
                if term.kind in (TermKind.ENTITY, TermKind.LITERAL):
                    has_constant = True
        reason = "PLAN_UNGROUNDED" if has_constant else "PLAN_CYCLE"
        # A component with only variables is a cycle or an unanchored plan.
        if has_constant:
            reason = "PLAN_UNGROUNDED"
        else:
            reason = "PLAN_CYCLE"
        return CompiledPlan(constraints=constraints, supported=False, reason_code=reason)

    reads: dict[str, list[str]] = {}
    writes: dict[str, list[str]] = {}
    for constraint in constraints:
        constraint.depends_on = list(depends[constraint.constraint_id])
        written = [var for var, cid in produced.items() if cid == constraint.constraint_id]
        read: list[str] = []
        for term in (constraint.head_term, constraint.tail_term):
            for var in _endpoint_vars(term):
                if var not in written and var in produced:
                    read.append(var)
        reads[constraint.constraint_id] = sorted(set(read))
        writes[constraint.constraint_id] = sorted(set(written))

    read_somewhere: set[str] = set()
    for values in reads.values():
        read_somewhere.update(values)
    answer_variable = None
    sinks = [var for var in produced if var not in read_somewhere]
    if len(sinks) == 1:
        answer_variable = sinks[0]
    for constraint in constraints:
        written = writes[constraint.constraint_id]
        if not written:
            constraint.is_terminal = True
        else:
            constraint.is_terminal = all(var not in read_somewhere for var in written)

    return CompiledPlan(
        constraints=constraints,
        producers=produced,
        reads=reads,
        writes=writes,
        answer_variable=answer_variable,
        supported=True,
    )
