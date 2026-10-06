"""Hand-checkable diagnostic cases. These are not 2Wiki scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from bridge_rag.diagnostics.build import make_candidate
from bridge_rag.schemas import PlanTriple, TermKind

BOOK = "diag:book"
LIN = "diag:lin"
LIN2 = "diag:lin2"
UNI = "diag:uni"
CITY = "diag:city"
HAI = "diag:hai"
HAI_CITY = "diag:haicity"

AUTHOR_Q = "《星河》的作者是林岚。"
GRAD_Q = "林岚毕业于明川大学，后来在海川大学任教。"
LOC_Q = "明川大学位于临江市。"
TEACH_Q = "林岚在海川大学任教。"
HAI_LOC_Q = "海川大学位于海川市。"
HAI_GRAD_Q = "林岚毕业于海川大学。"


def _ent(entity_id: str, name: str, aliases: list[str] | None = None, upstream_id: int = 0) -> dict:
    return {"entity_id": entity_id, "name": name, "aliases": aliases or [], "upstream_id": upstream_id}


def _people() -> list[dict]:
    return [
        _ent(BOOK, "星河", ["《星河》", "小说《星河》"], 1),
        _ent(LIN, "林岚", [], 2),
        _ent(UNI, "明川大学", ["明川大"], 3),
        _ent(CITY, "临江市", [], 4),
        _ent(HAI, "海川大学", [], 5),
        _ent(HAI_CITY, "海川市", [], 6),
    ]


def _edge(triple_id: str, index: int, head: str, tail: str | None, relation: str, quote: str, title: str, literal: str | None = None) -> dict:
    return {
        "triple_id": triple_id,
        "index": index,
        "head": head,
        "tail": tail,
        "relation": relation,
        "quote": quote,
        "title": title,
        "literal": literal,
        "doc_index": index,
        "sent_index": 0,
    }


def _base_edges(extra: list[dict] | None = None, drop: set[str] | None = None) -> list[dict]:
    rows = [
        _edge("t_author", 0, BOOK, LIN, "作者", AUTHOR_Q, "星河"),
        _edge("t_loc", 1, UNI, CITY, "位于", LOC_Q, "明川大学"),
        _edge("t_teach", 2, LIN, HAI, "任教", TEACH_Q, "林岚"),
        _edge("t_hai_loc", 3, HAI, HAI_CITY, "位于", HAI_LOC_Q, "海川大学"),
    ]
    blocked = drop or set()
    rows = [row for row in rows if row["triple_id"] not in blocked]
    rows.extend(extra or [])
    return rows


def _unit(passage_id: str, title: str, doc: int, sent: int, text: str, visible: bool = True) -> dict:
    return {
        "passage_id": passage_id,
        "title": title,
        "doc_index": doc,
        "sent_index": sent,
        "text": text,
        "visible": visible,
    }


def _plan() -> list[PlanTriple]:
    return [
        PlanTriple(head="《星河》", relation="作者", relation_variants=["作者", "作者是"], tail="?author"),
        PlanTriple(head="?author", relation="毕业于", relation_variants=["毕业于", "毕业"], tail="?university"),
        PlanTriple(head="?university", relation="位于", relation_variants=["位于"], tail="?city"),
    ]


def _grad(unit: dict, quote: str, tail: str, rank: int = 0, score: float | None = None, bad_span: bool = False, head: str = "林岚") -> object:
    return make_candidate(
        constraint_index=1,
        unit=unit,
        quote=quote,
        head=head,
        predicate="毕业于",
        tail=tail,
        score=score,
        bad_span=bad_span,
        rank=rank,
    )


@dataclass
class DiagnosticCase:
    case_id: str
    question: str
    world_args: dict
    expect: dict
    notes: str = ""
    compile_only: bool = False
    plan_triples: list[PlanTriple] = field(default_factory=list)


def _case(case_id: str, question: str, expect: dict, **world) -> DiagnosticCase:
    return DiagnosticCase(case_id=case_id, question=question, world_args=world, expect=expect, plan_triples=world.get("triples") or [])


def all_cases() -> list[DiagnosticCase]:
    d2 = _unit("d2", "林岚", 2, 0, GRAD_Q)
    d2_hidden = _unit("d2", "林岚", 2, 0, GRAD_Q, visible=False)
    d3 = _unit("d3", "明川大学", 1, 0, LOC_Q, visible=False)
    author_unit = _unit("d1", "星河", 0, 0, AUTHOR_Q)
    neg = _unit("d2n", "林岚", 2, 0, "林岚没有毕业于明川大学。")
    co = _unit("d2c", "林岚", 2, 0, "名单上有林岚、明川大学。")
    qual = _unit("d2q", "林岚", 2, 0, "林岚毕业于明川大学预科班。")
    hai_only = _unit("d2h", "林岚", 2, 0, HAI_GRAD_Q)
    alias = _unit("d2a", "林岚", 2, 0, "林岚毕业于明川大。")
    dup0 = _unit("dup0", "林岚", 4, 0, "林岚毕业于明川大学。")
    dup1 = _unit("dup1", "林岚", 4, 1, "林岚毕业于明川大学。")
    question = "小说《星河》的作者毕业的大学位于哪个城市？"
    common_units = [author_unit, d2, d3]
    base = dict(entities=_people(), graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}), units=common_units, triples=_plan())
    cases = [
        _case(
            "normal_missing_edge",
            question,
            {
                "status": "COMPLETE",
                "answer": "临江市",
                "uncertain": False,
                "graph_relations_absent": ["毕业于"],
                "chain_contains": [
                    {"kind": "GRAPH", "relation": "作者"},
                    {"kind": "TEXT", "relation": "毕业于", "tail": "明川大学"},
                    {"kind": "GRAPH", "relation": "位于", "head": "明川大学", "tail": "临江市"},
                ],
            },
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            **base,
        ),
        _case(
            "text_binding_disabled",
            question,
            {
                "status": "EXHAUSTED",
                "answer": None,
                "reasons": ["TEXT_BINDING_DISABLED"],
                "bindings_absent": ["?university", "?city"],
                "chain_absent": [{"kind": "TEXT"}, {"relation": "位于"}],
            },
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            config={"allow_text_binding": False},
            **base,
        ),
        _case(
            "no_reentry",
            question,
            {
                "status": "EXHAUSTED",
                "answer": None,
                "reasons": ["REENTRY_DISABLED"],
                "bindings": {"?university": "明川大学"},
                "chain_contains": [{"kind": "TEXT", "relation": "毕业于"}],
                "chain_absent": [{"relation": "位于"}],
                "graph_relations_absent": ["毕业于"],
            },
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            config={"allow_successor_reentry": False},
            **base,
        ),
        _case(
            "relation_confusion",
            question,
            {
                "status": "COMPLETE",
                "answer": "临江市",
                "decisions": ["CONFLICT", "SUPPORTED"],
                "reasons": ["GRAPH_SOURCE_REJECT"],
                "bindings": {"?university": "明川大学"},
                "chain_absent": [{"tail": "海川大学"}],
            },
            entities=_people(),
            graph_triples=_base_edges(),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            forced={1: ["t_teach"]},
        ),
        _case(
            "relation_confusion_no_graph_check",
            question,
            {
                "status": "COMPLETE",
                "answer": "海川市",
                "uncertain": False,
                "chain_contains": [{"kind": "GRAPH", "relation": "任教", "tail": "海川大学"}],
                "chain_absent": [{"kind": "TEXT", "relation": "毕业于"}],
            },
            entities=_people(),
            graph_triples=_base_edges(),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            forced={1: ["t_teach"]},
            config={"verify_graph": False},
        ),
        _case(
            "conflict_negation",
            question,
            {"status": "EXHAUSTED", "answer": None, "decisions": ["CONFLICT"], "bindings_absent": ["?university"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, neg, d3],
            triples=_plan(),
            scripted=[_grad(neg, "林岚没有毕业于明川大学。", "明川大学")],
        ),
        _case(
            "negation_without_text_check",
            question,
            {"status": "COMPLETE", "answer": "临江市", "decisions": ["SUPPORTED"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, neg, d3],
            triples=_plan(),
            scripted=[_grad(neg, "林岚没有毕业于明川大学。", "明川大学")],
            config={"verify_text": False},
        ),
        _case(
            "name_cooccurrence",
            question,
            {"status": "EXHAUSTED", "answer": None, "decisions": ["UNKNOWN"], "reasons": ["COOCCURRENCE"], "bindings_absent": ["?university"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, co, d3],
            triples=_plan(),
            scripted=[_grad(co, "名单上有林岚、明川大学。", "明川大学")],
        ),
        _case(
            "same_name_ambiguous",
            "哪位林岚毕业于明川大学？",
            {"status": "EXHAUSTED", "answer": None, "decisions": ["AMBIGUOUS_ENTITY"], "bindings_absent": ["?person"]},
            entities=[
                _ent(LIN, "林岚", [], 2),
                _ent(LIN2, "林岚", [], 7),
                _ent(UNI, "明川大学", [], 3),
                _ent(CITY, "临江市", [], 4),
            ],
            graph_triples=[_edge("t_loc", 1, UNI, CITY, "位于", LOC_Q, "明川大学")],
            units=[_unit("d2s", "林岚", 2, 0, GRAD_Q)],
            triples=[
                PlanTriple(head="?person", relation="毕业于", relation_variants=["毕业于"], tail="明川大学"),
                PlanTriple(head="?person", relation="居住", relation_variants=["居住"], tail="?city"),
            ],
            scripted=[
                make_candidate(
                    constraint_index=0,
                    unit=_unit("d2s", "林岚", 2, 0, GRAD_Q),
                    quote=GRAD_Q,
                    head="林岚",
                    predicate="毕业于",
                    tail="明川大学",
                )
            ],
        ),
        _case(
            "unlinkable",
            question,
            {"status": "EXHAUSTED", "answer": None, "decisions": ["UNLINKABLE"], "bindings_absent": ["?university"]},
            entities=[_ent(BOOK, "星河", ["《星河》"], 1), _ent(LIN, "林岚", [], 2), _ent(CITY, "临江市", [], 4)],
            graph_triples=[_edge("t_author", 0, BOOK, LIN, "作者", AUTHOR_Q, "星河")],
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
        ),
        _case(
            "successor_missing_edge",
            question,
            {
                "status": "EXHAUSTED",
                "answer": None,
                "reasons": ["NO_SUCCESSOR_EVIDENCE"],
                "bindings": {"?university": "明川大学"},
                "chain_contains": [{"kind": "TEXT", "relation": "毕业于"}],
                "chain_absent": [{"relation": "位于"}],
            },
            entities=_people(),
            graph_triples=_base_edges(drop={"t_loc", "t_teach", "t_hai_loc"}),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
        ),
        _case(
            "bridge_text_clipped",
            question,
            {"status": "EXHAUSTED", "answer": None, "reasons": ["NO_TEXT_CANDIDATE"], "chain_absent": [{"kind": "TEXT"}]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, d2_hidden, d3],
            triples=_plan(),
            scripted=[_grad(d2_hidden, GRAD_Q, "明川大学")],
        ),
        _case(
            "branch_contradiction",
            question,
            {
                "status": "AMBIGUOUS",
                "answer": None,
                "answers": ["临江市", "海川市"],
                "no_mixed": ["临江市", "海川市"],
            },
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach"}),
            units=[author_unit, d2, hai_only, d3],
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学", rank=0), _grad(hai_only, HAI_GRAD_Q, "海川大学", rank=1)],
        ),
        _case(
            "incompatible_partial",
            question,
            {
                "status": "COMPLETE",
                "answer": "临江市",
                "uncertain": True,
                "no_mixed": ["临江市", "海川大学"],
                "reasons": ["NO_SUCCESSOR_EVIDENCE"],
            },
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, d2, hai_only, d3],
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学", rank=0), _grad(hai_only, HAI_GRAD_Q, "海川大学", rank=1)],
        ),
        _case(
            "parent_revoke",
            question,
            {
                "status": "INVALIDATED",
                "answer": None,
                "reasons": ["REVOKED"],
                "proofs_invalidated": True,
                "bindings_absent": ["?author", "?university", "?city"],
            },
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            revoke_variable="?author",
            **base,
        ),
        _case(
            "budget_exhausted",
            question,
            {
                "status": "BUDGET_EXHAUSTED",
                "answer": None,
                "reasons": ["BUDGET_EXHAUSTED"],
                "chain_contains": [{"kind": "GRAPH", "relation": "作者"}],
                "chain_absent": [{"kind": "TEXT"}],
            },
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
            config={"max_steps": 1},
            **base,
        ),
        _case(
            "competing_branch_unfinished",
            question,
            {"status": "COMPLETE", "answer": "临江市", "uncertain": True, "reasons": ["BUDGET_EXHAUSTED"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach"}),
            units=[author_unit, d2, hai_only, d3],
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学", rank=0), _grad(hai_only, HAI_GRAD_Q, "海川大学", rank=1)],
            config={"max_steps": 3},
        ),
        _case(
            "reverse_direction",
            question,
            {"status": "EXHAUSTED", "answer": None, "decisions": ["UNKNOWN"], "reasons": ["DIRECTION_MISMATCH"], "bindings_absent": ["?university"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "林岚", head="明川大学")],
        ),
        _case(
            "qualifier_mismatch",
            question,
            {"status": "EXHAUSTED", "answer": None, "reasons": ["QUALIFIER_MISMATCH"], "bindings_absent": ["?university"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, qual, d3],
            triples=_plan(),
            scripted=[_grad(qual, "林岚毕业于明川大学预科班。", "明川大学")],
        ),
        _case(
            "invalid_span",
            question,
            {"status": "EXHAUSTED", "answer": None, "decisions": ["INVALID_SPAN"], "bindings_absent": ["?university"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学", bad_span=True)],
        ),
        _case(
            "alias_unique",
            question,
            {"status": "COMPLETE", "answer": "临江市", "decisions": ["ALIAS_UNIQUE"], "bindings": {"?university": "明川大学"}},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, alias, d3],
            triples=_plan(),
            scripted=[_grad(alias, "林岚毕业于明川大。", "明川大")],
        ),
        _case(
            "duplicate_sentence_position",
            question,
            {"status": "COMPLETE", "answer": "临江市", "chain_contains": [{"kind": "TEXT", "sent_index": 1}]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, dup0, dup1, d3],
            triples=_plan(),
            scripted=[_grad(dup1, "林岚毕业于明川大学。", "明川大学")],
        ),
        _case(
            "short_wrong_path",
            question,
            {
                "status": "COMPLETE",
                "answer": "临江市",
                "chain_contains": [{"relation": "位于", "head": "明川大学", "tail": "临江市"}],
                "chain_absent": [{"relation": "位于", "head": "星河"}],
            },
            entities=_people(),
            graph_triples=_base_edges(
                drop={"t_teach", "t_hai_loc"},
                extra=[_edge("t_short", 9, BOOK, CITY, "位于", "星河位于临江市。", "星河")],
            ),
            units=common_units,
            triples=_plan(),
            scripted=[_grad(d2, GRAD_Q, "明川大学")],
        ),
        _case(
            "score_does_not_override_verification",
            question,
            {
                "status": "COMPLETE",
                "answer": "临江市",
                "decisions": ["CONFLICT", "SUPPORTED"],
                "bindings": {"?university": "明川大学"},
            },
            entities=_people(),
            graph_triples=_base_edges(drop={"t_teach", "t_hai_loc"}),
            units=[author_unit, neg, d2, d3],
            triples=_plan(),
            scripted=[
                _grad(neg, "林岚没有毕业于明川大学。", "海川大学", rank=0, score=0.99),
                _grad(d2, GRAD_Q, "明川大学", rank=1, score=0.01),
            ],
        ),
        _case(
            "both_endpoints_known",
            "《星河》的作者是谁？",
            {"status": "COMPLETE", "answer": None, "chain_contains": [{"kind": "GRAPH", "relation": "作者"}], "bindings_absent": ["?author"]},
            entities=_people(),
            graph_triples=_base_edges(drop={"t_loc", "t_teach", "t_hai_loc"}),
            units=[author_unit],
            triples=[PlanTriple(head="《星河》", relation="作者", relation_variants=["作者"], tail="林岚")],
            scripted=[],
        ),
        _case(
            "literal_not_entity",
            "《星河》的作者出生年份是哪一年？",
            {
                "status": "COMPLETE",
                "answer": "1980",
                "match_log_excludes": ["1980"],
                "chain_contains": [{"relation": "出生年份", "literal": "1980", "tail_id": None}],
            },
            entities=_people(),
            graph_triples=_base_edges(drop={"t_loc", "t_teach", "t_hai_loc"})
            + [_edge("t_year", 8, LIN, None, "出生年份", "林岚出生年份是1980。", "林岚", literal="1980")],
            units=[author_unit],
            triples=[
                PlanTriple(head="《星河》", relation="作者", relation_variants=["作者"], tail="?author"),
                PlanTriple(
                    head="?author",
                    relation="出生年份",
                    relation_variants=["出生年份"],
                    tail="?year",
                    tail_kind=TermKind.LITERAL,
                ),
            ],
            scripted=[],
        ),
        _case(
            "branch_cap_truncation",
            question,
            {"status": "AMBIGUOUS", "reasons": ["FORK_TRUNCATED"], "answer": None},
            entities=_people()
            + [_ent("diag:u3", "北泽大学", [], 8), _ent("diag:c3", "北泽市", [], 9), _ent("diag:u4", "南泽大学", [], 10), _ent("diag:c4", "南泽市", [], 11)],
            graph_triples=_base_edges(drop={"t_teach"})
            + [
                _edge("t_u3", 5, "diag:u3", "diag:c3", "位于", "北泽大学位于北泽市。", "北泽大学"),
                _edge("t_u4", 6, "diag:u4", "diag:c4", "位于", "南泽大学位于南泽市。", "南泽大学"),
            ],
            units=[
                author_unit,
                d2,
                hai_only,
                _unit("d2u3", "林岚", 5, 0, "林岚毕业于北泽大学。"),
                _unit("d2u4", "林岚", 6, 0, "林岚毕业于南泽大学。"),
            ],
            triples=_plan(),
            scripted=[
                _grad(d2, GRAD_Q, "明川大学", rank=0),
                _grad(hai_only, HAI_GRAD_Q, "海川大学", rank=1),
                _grad(_unit("d2u3", "林岚", 5, 0, "林岚毕业于北泽大学。"), "林岚毕业于北泽大学。", "北泽大学", rank=2),
                _grad(_unit("d2u4", "林岚", 6, 0, "林岚毕业于南泽大学。"), "林岚毕业于南泽大学。", "南泽大学", rank=3),
            ],
            config={"max_active_branches": 3, "max_relations_to_verify": 4},
        ),
    ]
    cases.extend(_plan_cases())
    return cases


def _plan_cases() -> list[DiagnosticCase]:
    empty = dict(entities=[], graph_triples=[], units=[], scripted=[])
    specs = [
        (
            "plan_cycle",
            [
                PlanTriple(head="?a", relation="关联", tail="?b"),
                PlanTriple(head="?b", relation="关联", tail="?a"),
            ],
            "PLAN_CYCLE",
        ),
        (
            "both_endpoints_unknown",
            [PlanTriple(head="?a", relation="关联", tail="?b")],
            "PLAN_CYCLE",
        ),
        (
            "plan_ambiguous_writer",
            [
                PlanTriple(head="《星河》", relation="作者", tail="?author"),
                PlanTriple(head="林岚", relation="笔名", tail="?author"),
            ],
            "PLAN_AMBIGUOUS_WRITER",
        ),
    ]
    built = []
    for case_id, triples, reason in specs:
        built.append(
            _case(
                case_id,
                case_id,
                {"status": "PLAN_UNSUPPORTED", "answer": None, "reasons": [reason], "graph_unchanged": True},
                triples=triples,
                **empty,
            )
        )
    order = [
        PlanTriple(head="?university", relation="位于", tail="?city"),
        PlanTriple(head="?author", relation="毕业于", tail="?university"),
        PlanTriple(head="《星河》", relation="作者", tail="?author"),
    ]
    built.append(
        DiagnosticCase(
            case_id="plan_order_independent",
            question="order",
            world_args={},
            expect={"producer": {"?author": "c2"}, "terminal_index": 0},
            compile_only=True,
            plan_triples=order,
        )
    )
    return built
