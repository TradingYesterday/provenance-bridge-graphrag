from bridge_rag.backends.neural import RankedUnit, resolve_device
from bridge_rag.diagnostics.catalog import all_cases
from bridge_rag.diagnostics.build import compile_world
from bridge_rag.execution.engine import RunInput, execute


class _FakeRetriever:
    embedding_model = "fake-embed"
    reranker_model = "fake-rerank"
    device = "cpu"

    def rank_units(self, query: str, units: list, top_k: int) -> list[RankedUnit]:
        ordered = list(reversed(units))[:top_k]
        return [RankedUnit(unit=unit, coarse_score=0.0, rerank_score=float(index)) for index, unit in enumerate(ordered)]


def test_engine_uses_attached_retriever_order_without_loading_models():
    case = next(item for item in all_cases() if item.case_id == "normal_missing_edge")
    world = compile_world(question_id=case.case_id, **case.world_args)
    result = execute(
        RunInput(
            question_id=case.case_id,
            plan=world.plan,
            graph=world.graph,
            texts=world.texts,
            scripted=world.scripted,
            retriever=_FakeRetriever(),
        )
    )
    retrieve = next(event for event in result.trace if event.get("event") == "retrieve")
    visible = [unit.passage_id for unit in world.texts.visible_units()]
    assert retrieve["passage_ids"][0] == list(reversed(visible))[0]
    assert any(event.get("event") == "rerank" and event.get("device") == "cpu" for event in result.trace)
    assert result.answer == "临江市"


def test_auto_device_stays_on_cpu_while_another_job_holds_the_gpu(monkeypatch):
    monkeypatch.setattr(
        "bridge_rag.backends.neural.gpu_status",
        lambda: {"free_mb": 17000, "compute_pids": [11558]},
    )
    assert resolve_device("auto") == "cpu"
