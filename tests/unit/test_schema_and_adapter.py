import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bridge_rag.adapters.csrag import (
    assemble_case,
    load_kg_directory,
    localization_issues,
    require_path,
    upstream_command,
)
from bridge_rag.adapters.dataset_2wiki import public_fields, split_gold
from bridge_rag.backends.llm import LLMClient, ModelUnavailable
from bridge_rag.planning import compile_plan
from bridge_rag.recovery.verifier import span_ok
from bridge_rag.runtime.cache import CandidateCache, cache_key
from bridge_rag.runtime.trace import scrub
from bridge_rag.schemas import BranchState, BudgetView, PlanTriple, QuestionCase, Term, TermKind
from bridge_rag.backends.text import TextUnit


def test_schema_rejects_missing_core_and_gold_on_question_case():
    with pytest.raises(ValidationError):
        Term(kind=TermKind.VARIABLE, surface="?x")
    with pytest.raises(ValidationError):
        QuestionCase(
            question_id="q",
            question="q",
            dataset="2wiki",
            data_version="v",
            graph_namespace="ns",
            answer="secret",
        )
    branch = BranchState(question_id="q", branch_id="b0", budget=BudgetView(max_steps=1, steps_used=0, max_active_branches=3))
    assert branch.status.value == "ACTIVE"


def test_span_direction_and_literal_plan():
    unit = TextUnit("p", "q", "v", 0, 0, "t", "林岚毕业于明川大学。")
    assert span_ok(unit, "毕业于", 2, 5)
    assert not span_ok(unit, "毕业于", 0, 3)
    plan = compile_plan(
        [
            PlanTriple(head="《星河》", relation="作者", tail="?author"),
            PlanTriple(head="?author", relation="出生年份", tail="?year", tail_kind=TermKind.LITERAL),
        ],
        "diag",
    )
    assert plan.supported
    year = plan.constraints[1].tail_term
    assert year.value_type is TermKind.LITERAL
    assert year.kind is TermKind.VARIABLE


def test_cache_key_changes_with_bindings_and_hit_is_not_free():
    left = cache_key({"bindings": {"?author": "diag:lin"}, "constraint": "c1"})
    right = cache_key({"bindings": {"?author": "diag:lin2"}, "constraint": "c1"})
    assert left != right
    cache = CandidateCache()
    assert cache.get(left) is None
    cache.put(left, ["p"])
    assert cache.get(left) == ["p"]
    assert cache.hits == 1


def test_trace_redacts_secrets():
    assert scrub({"api_key": "sk-test", "nested": {"Authorization": "bearer"}})["api_key"] == "[redacted]"


def test_llm_refuses_without_inventing_an_answer(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelUnavailable):
        LLMClient().complete("question", purpose="extract")


def test_graph_mask_keeps_ids_and_title_map(tmp_path: Path):
    root = tmp_path / "kg"
    root.mkdir()
    (root / "entities.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": 0, "name": "Alpha", "type": "PERSON"}),
                json.dumps({"id": 1, "name": "Beta", "type": "ORG"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "triples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"h": 0, "t": 1, "r": "decoy", "evidence": "Alpha decoy Beta.", "title": "Alpha"}),
                json.dumps({"h": 0, "t": 1, "r": "member of", "evidence": "Alpha is a member of Beta.", "title": "Alpha"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "title2entities.jsonl").write_text(json.dumps({"title": "Alpha", "entity_ids": [0, 1]}) + "\n", encoding="utf-8")
    (root / "title2triples.jsonl").write_text(json.dumps({"title": "Alpha", "triple_idxs": [0, 1]}) + "\n", encoding="utf-8")
    graph, _notes = load_kg_directory(root, namespace="cs", version="v")
    assert graph.snapshot_ids() == {"0": 0, "1": 1}
    graph.mask_triple("0")
    assert graph.snapshot_ids() == {"0": 0, "1": 1}
    assert graph.active_title_triples("Alpha") == ["1"]
    assert graph.title_to_triple_ids["Alpha"] == ["0", "1"]
    assert graph.triples["1"].upstream_index == 1


def test_adapter_uses_kg_id_not_compact_index_and_hides_gold(tmp_path: Path):
    root = tmp_path / "kg"
    root.mkdir()
    (root / "entities.jsonl").write_text(
        "\n".join([json.dumps({"id": 0, "name": "Alpha"}), json.dumps({"id": 1, "name": "Beta"})]) + "\n",
        encoding="utf-8",
    )
    (root / "triples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"h": 0, "t": 1, "r": "decoy", "evidence": "ignore", "title": "Alpha"}),
                json.dumps({"h": 0, "t": 1, "r": "member of", "evidence": "Alpha is a member of Beta.", "title": "Alpha"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "title2entities.jsonl").write_text(json.dumps({"title": "Alpha", "entity_ids": [0, 1]}) + "\n", encoding="utf-8")
    (root / "title2triples.jsonl").write_text(json.dumps({"title": "Alpha", "triple_idxs": [0, 1]}) + "\n", encoding="utf-8")
    graph, _notes = load_kg_directory(root, namespace="cs", version="v1")
    phase1 = {
        "id": "q1",
        "question": "Who is related to Alpha?",
        "ground_truth_answer": "SECRET_ANSWER",
        "query_plan": [{"head": "Alpha", "relation": "member of", "relation_variants": ["member of"], "tail": "?org"}],
        "evidence_triples": [
            {"kg_triple_id": 1, "head": "Alpha", "relation": "member of", "tail": "Beta", "head_entity_id": 0, "tail_entity_id": 1}
        ],
        "support": [{"query_triple_index": 0, "evidence_indices": [0]}],
        "debug": {"variable_candidates": {"?org": {"primary_ids": [1]}}, "entity_map": []},
    }
    phase2 = {
        "id": "q1",
        "ground_truth_answer": "SECRET_ANSWER",
        "triples_evidence": [
            {
                "query_triple_index": 0,
                "triple_status": "resolved",
                "candidate_evidences": [{"title": "Alpha", "context": "Alpha is a member of Beta."}],
            }
        ],
    }
    raw = {
        "_id": "q1",
        "question": "Who is related to Alpha?",
        "answer": "SECRET_ANSWER",
        "supporting_facts": [["Alpha", 0]],
        "context": [["Alpha", ["Alpha is a member of Beta."]], ["Alpha", ["Alpha is a member of Beta."]]],
    }
    view = public_fields(raw)
    assert "answer" not in view and "supporting_facts" not in view
    assert split_gold(raw).answer == "SECRET_ANSWER"
    case, gold, _plan = assemble_case(phase1, dataset="2wiki", data_version="v1", namespace="cs", raw_example=raw)
    issues = localization_issues(phase1_item=phase1, phase2_item=phase2, raw_example=raw, graph=graph, data_version="v1")
    assert gold.answer == "SECRET_ANSWER"
    assert "SECRET_ANSWER" not in case.model_dump_json()
    assert issues["hard_failures"] == []
    assert issues["ambiguities"]
    phase2["triples_evidence"][0]["candidate_evidences"] = [{"title": "Alpha", "sent_idx": 3, "evidence": "missing sentence"}]
    raw_unique = dict(raw)
    raw_unique["context"] = [["Alpha", ["Alpha is a member of Beta."]]]
    unique = localization_issues(phase1_item=phase1, phase2_item=phase2, raw_example=raw_unique, graph=graph, data_version="v1")
    assert unique["hard_failures"]


def test_require_path_does_not_rewrite_planned_queries(tmp_path: Path):
    real = tmp_path / "planner_queries" / "2wiki_data"
    real.mkdir(parents=True)
    target = real / "query_graph_v8_2wiki.json"
    target.write_text("[]", encoding="utf-8")
    missing = tmp_path / "planned_queries" / "2wiki_data" / "query_graph_v8_2wiki.json"
    with pytest.raises(Exception) as caught:
        require_path(missing)
    assert "planner_queries" in str(caught.value)
    assert require_path(target) == target


def test_upstream_command_does_not_import_models():
    command = upstream_command("configs/robustness_runner.yaml")
    assert "legacy_impl" in " ".join(command)
    assert "c84a081c728dd45c284a7165031c853eef4e8f24" in " ".join(command)


def test_synthetic_cohort_keeps_stable_ids_and_hides_gold():
    from bridge_rag.adapters.csrag import synthetic_localization_cohort

    rows = synthetic_localization_cohort(20)
    assert len(rows) == 20
    assert all(row["ok"] and not row["gold_leaked"] for row in rows)
    assert all(row["kg_triple_id"] == row["expected_kg_triple_id"] for row in rows)
    assert all(row["phase1_candidates_are_bindings"] is False for row in rows)
    assert rows[0]["kg_triple_id"] == "1"


def test_fork_does_not_share_bindings():
    from bridge_rag.execution.branch import fork_branch
    from bridge_rag.schemas import LinkedBinding

    parent = BranchState(question_id="q", branch_id="b0", budget=BudgetView(max_steps=2, steps_used=0, max_active_branches=3))
    parent.bindings["?author"] = LinkedBinding(
        binding_id="bind-author",
        variable="?author",
        entity_id="diag:lin",
        canonical_name="林岚",
        membership_in_current_graph=True,
        addressable_for_successor=True,
        graph_namespace="diag",
    )
    child = fork_branch(parent, "b1")
    child.bindings["?university"] = LinkedBinding(
        binding_id="bind-uni",
        variable="?university",
        entity_id="diag:uni",
        canonical_name="明川大学",
        membership_in_current_graph=True,
        addressable_for_successor=True,
        graph_namespace="diag",
    )
    assert "?university" not in parent.bindings
    assert child.bindings["?author"].entity_id == "diag:lin"


def test_title_collision_is_not_resolved_by_first_doc():
    from bridge_rag.adapters.provenance import locate_in_example

    example = {"context": [["Same", ["Shared sentence."]], ["Same", ["Shared sentence."]]]}
    located = locate_in_example(example, data_version="v", question_id="q", title="Same", quote="Shared sentence.")
    assert located.ambiguous is True
    assert len(located.locations) == 2
    assert {item.doc_index for item in located.locations} == {0, 1}
