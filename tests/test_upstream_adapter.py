import json

import pytest

from bridge_rag.adapters.csrag import InputContractError, audit_records, load_kg_directory, require_file
from bridge_rag.adapters.dataset_2wiki import public_fields
from bridge_rag.schemas import QuestionCase


def _write_kg(root, entities, triples):
    kg = root / "KG"
    kg.mkdir()
    (kg / "entities.jsonl").write_text(
        "".join(json.dumps({"id": index, "name": name, "type": "OTHER"}, ensure_ascii=False) + "\n" for index, name in enumerate(entities)),
        encoding="utf-8",
    )
    lines = []
    title_triples: dict[str, list[int]] = {}
    title_entities: dict[str, set[int]] = {}
    for index, (head, relation, tail, title, evidence) in enumerate(triples):
        lines.append(json.dumps({"h": head, "r": relation, "t": tail, "title": title, "evidence": evidence}, ensure_ascii=False))
        title_triples.setdefault(title, []).append(index)
        title_entities.setdefault(title, set()).update((head, tail))
    (kg / "triples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (kg / "title2triples.jsonl").write_text(
        "".join(json.dumps({"title": title, "triple_idxs": ids}, ensure_ascii=False) + "\n" for title, ids in title_triples.items()),
        encoding="utf-8",
    )
    (kg / "title2entities.jsonl").write_text(
        "".join(
            json.dumps({"title": title, "entity_ids": sorted(ids)}, ensure_ascii=False) + "\n"
            for title, ids in title_entities.items()
        ),
        encoding="utf-8",
    )
    return kg


def _bundle(root):
    entities = [f"实体{i}" for i in range(40)]
    triples = []
    for index in range(20):
        triples.append((index, "相关", index + 20, f"页{index}", f"{entities[index]}相关{entities[index + 20]}。"))
    kg = _write_kg(root, entities, triples)
    graph, notes = load_kg_directory(kg, namespace="wiki", version="fixture-v1")
    phase1 = []
    phase2 = []
    raw = {}
    for index in range(20):
        question_id = f"q{index}"
        title = f"页{index}"
        sentence = f"{entities[index]}相关{entities[index + 20]}。"
        repeated = "重复句。"
        if index == 18:
            context = [[title, [repeated]], [title, [repeated]]]
            evidences = [{"title": title, "context": repeated}]
        elif index == 19:
            context = [[title, [sentence]]]
            evidences = [{"title": title, "context": "这句话不在原文里。"}]
        elif index == 17:
            context = [[title, [sentence]]]
            evidences = [{"title": title, "context": sentence}]
        else:
            context = [[title, [sentence, "另一句。"]], ["其他", ["无关。"]]]
            evidences = [{"title": title, "paragraph_idx": 0, "sent_idx": 0, "evidence": sentence, "context": f"{title} {sentence}"}]
        raw_row = {
            "_id": question_id,
            "question": f"问题{index}？",
            "answer": f"答案{index}",
            "supporting_facts": [[title, 0]],
            "context": context,
        }
        raw[question_id] = raw_row
        phase1.append(
            {
                "id": question_id,
                "question": raw_row["question"],
                "ground_truth_answer": raw_row["answer"],
                "query_plan": [
                    {"head": entities[index], "relation": "相关", "relation_variants": ["相关"], "tail": "?x"}
                ],
                "evidence_triples": [
                    {
                        "kg_triple_id": index,
                        "head_entity_id": index,
                        "tail_entity_id": index + 20,
                        "head": entities[index],
                        "relation": "相关",
                        "tail": entities[index + 20],
                    }
                ],
                "support": [{"query_triple_index": 0, "evidence_indices": [0]}],
                "debug": {"entity_map": [{"query_entity": entities[index], "kg_entity_ids": [index]}]},
            }
        )
        phase2.append(
            {
                "id": question_id,
                "question": raw_row["question"],
                "ground_truth_answer": raw_row["answer"],
                "triples_evidence": [
                    {
                        "query_triple_index": 0,
                        "query_triple": phase1[-1]["query_plan"][0],
                        "triple_status": "resolved" if index >= 17 else "unresolved",
                        "candidate_evidences": evidences,
                        "kg_triples": phase1[-1]["evidence_triples"],
                    }
                ],
            }
        )
    return graph, notes, phase1, phase2, raw


def test_synthetic_twenty_question_localization(tmp_path):
    graph, notes, phase1, phase2, raw = _bundle(tmp_path)
    assert notes == []
    report = audit_records(phase1, phase2, raw, graph, data_version="fixture-v1", source_label="synthetic-fixture")
    assert report["questions"] == 20
    assert report["hard_failures"] == []
    assert report["passed"] is True
    assert report["ambiguity_count"] == 2
    assert any("multiple_spans" in item for item in report["ambiguities"])
    assert any("unlocated" in item for item in report["ambiguities"])

    before = graph.snapshot_ids()
    graph.mask_triple("3")
    assert graph.snapshot_ids() == before
    assert "3" in graph.title_to_triple_ids["页3"]
    assert "3" not in graph.active_title_triples("页3")

    original = phase1[0]
    stripped = public_fields(original)
    from bridge_rag.adapters.csrag import assemble_case

    left, _gold, _plan = assemble_case(original, dataset="2wiki", data_version="fixture-v1", namespace="wiki", raw_example=raw["q0"])
    right, _gold2, _plan2 = assemble_case(stripped, dataset="2wiki", data_version="fixture-v1", namespace="wiki")
    assert isinstance(left, QuestionCase)
    dumped = left.model_dump()
    assert "answer" not in dumped
    assert "ground_truth_answer" not in dumped
    assert "supporting_facts" not in dumped
    assert left.model_dump() == right.model_dump()


def test_planned_queries_path_is_not_silently_rewritten(tmp_path):
    real = tmp_path / "planner_queries" / "2wiki_data"
    real.mkdir(parents=True)
    (real / "query_graph_v8_2wiki.json").write_text("[]", encoding="utf-8")
    configured = tmp_path / "planned_queries" / "2wiki_data" / "query_graph_v8_2wiki.json"
    with pytest.raises(InputContractError) as caught:
        require_file(configured)
    assert "planner_queries" in str(caught.value)
    assert require_file(real / "query_graph_v8_2wiki.json").is_file()
