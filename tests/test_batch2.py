"""Batch 2: mock model I/O, shared-engine baselines, and paired metrics."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from bridge_rag.backends.llm import LLMClient, ModelCallError, ModelUnavailable
from bridge_rag.diagnostics.build import compile_world
from bridge_rag.diagnostics.catalog import all_cases
from bridge_rag.evaluation.evidence import proof_precision, score_case
from bridge_rag.evaluation.gold import MetricGold, RelationGold
from bridge_rag.evaluation.report import score_methods
from bridge_rag.execution.engine import RunInput, execute
from bridge_rag.execution.methods import run_method
from bridge_rag.execution.proof import active_chain
from bridge_rag.schemas import EngineConfig, RecoveryCandidate


def _case(case_id: str):
    return next(case for case in all_cases() if case.case_id == case_id)


def _run_input(case_id: str) -> RunInput:
    case = _case(case_id)
    world = compile_world(question_id=case.case_id, **case.world_args)
    return RunInput(
        question_id=case.case_id,
        plan=world.plan,
        graph=world.graph,
        texts=world.texts,
        scripted=world.scripted,
        forced_graph_ids=world.forced,
        config=world.config,
    )


def _gold() -> MetricGold:
    return MetricGold(
        question_id="normal_missing_edge",
        answer="临江市",
        group="missing_edge",
        bridge=RelationGold(head="林岚", relation="毕业于", tail="明川大学"),
        successor=RelationGold(head="明川大学", relation="位于", tail="临江市"),
        support_quotes=["林岚毕业于明川大学，后来在海川大学任教。", "明川大学位于临江市。"],
    )


class _ScriptedTransport:
    def __init__(self, content: str, *, usage: dict | None = None, failures: int = 0, status: int = 500) -> None:
        self.content = content
        self.usage = usage
        self.failures = failures
        self.status = status
        self.calls = 0
        self.payloads = []
        self.headers = []

    def post(self, url: str, payload: dict, headers: dict):
        self.calls += 1
        self.payloads.append(payload)
        self.headers.append(headers)
        if self.calls <= self.failures:
            raise ModelCallError(self.status, "busy")
        body = {
            "model": "mock-model",
            "choices": [{"message": {"content": self.content}}],
        }
        if self.usage is not None:
            body["usage"] = self.usage
        return 200, body


def test_core_matrix_and_baselines_share_one_executor():
    base = _run_input("normal_missing_edge")
    city = next(unit for unit in base.texts.units if unit.passage_id == "d3")
    base.scripted["c2"] = [
        RecoveryCandidate(
            candidate_id="city",
            constraint_id="c2",
            branch_id="pending",
            head_surface="明川大学",
            predicate_surface="位于",
            tail_surface="临江市",
            source_doc_id=city.source_id(),
            title=city.title,
            passage_id=city.passage_id,
            char_start=0,
            char_end=len(city.raw_text),
            quote=city.raw_text,
            extractor_model="oracle-scripted",
            prompt_hash="oracle",
            raw_response_hash="oracle",
        )
    ]
    results = {name: run_method(name, base) for name in ("core00", "core01", "core10", "core11", "no_reentry", "iterative_text")}
    assert results["core00"].answer is None
    assert results["core01"].answer is None
    assert results["core10"].answer == "临江市"
    assert results["core11"].answer == "临江市"
    assert results["no_reentry"].answer is None
    assert results["core00"].graph_writes == 0
    iterative_location = [
        proof
        for proof in active_chain(results["iterative_text"].proofs)
        if proof.relation == "位于"
    ]
    assert iterative_location
    assert {proof.kind.value for proof in iterative_location} == {"TEXT"}
    assert results["iterative_text"].answer == "临江市"
    scored = score_methods(results, _gold())
    assert scored["overall"]["core00"]["successor_rate"] == 0.0
    assert scored["overall"]["core11"]["successor_rate"] == 1.0
    paired = next(item for item in scored["paired"] if item["baseline"] == "core00" and item["other"] == "core11")
    assert paired["mean_delta"] == 1.0
    assert proof_precision(results["core00"].model_copy(update={"proofs": [], "branches": []}), _gold()) is None


def test_simple_bind_accepts_cooccurrence_that_relation_commit_rejects():
    base = _run_input("name_cooccurrence")
    strict = run_method("core11", base)
    simple = run_method("simple_bind", base)
    assert strict.answer is None
    assert simple.answer == "临江市"
    assert simple.graph_writes == 0
    assert any(proof.reason_code == "SIMPLE_SLOT" for proof in simple.proofs)


def test_cog_notebook_does_not_use_the_relation_commit_gate():
    base = _run_input("conflict_negation")
    strict = run_method("core11", base)
    cog = run_method("cog_controlled", base)
    assert strict.answer is None
    assert cog.answer == "临江市"
    kinds = {event.get("kind") for event in cog.trace if event.get("event") == "notebook"}
    assert {"fact", "followup_query", "judgment", "analysis"} <= kinds
    assert any(proof.reason_code == "COG_NOTEBOOK" for proof in cog.proofs)


def test_model_extract_then_rule_verify_reenters_graph():
    base = _run_input("normal_missing_edge")
    base.scripted = {}
    transport = _ScriptedTransport(
        json.dumps(
            {
                "relations": [
                    {
                        "passage_id": "d2",
                        "quote": "林岚毕业于明川大学",
                        "head": "林岚",
                        "predicate": "毕业于",
                        "tail": "明川大学",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        usage={"prompt_tokens": 20, "completion_tokens": 8},
    )
    base.model = LLMClient(transport=transport, model="mock-model", base_url="http://mock")
    result = execute(base)
    assert result.answer == "临江市"
    assert result.graph_writes == 0
    assert transport.headers[0]["X-Bridge-Purpose"] == "extract"
    assert "SECRET" not in transport.payloads[0]["messages"][1]["content"]
    assert transport.payloads[0]["temperature"] == 0
    row = score_case(result, _gold())
    assert row["successor_acquired"] is True
    assert row["exact_match"] is True
    assert row["bridge_acquired"] is True


def test_separate_verifier_call_can_reject_without_inventing_a_binding(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    base = _run_input("normal_missing_edge")
    base.scripted = {}
    transport = _ScriptedTransport(
        json.dumps({"decision": "CONFLICT", "reason": "MODEL_CONFLICT"}, ensure_ascii=False),
        usage={"prompt_tokens": 5, "completion_tokens": 2},
    )
    original = transport.post

    def post(url, payload, headers):
        if headers["X-Bridge-Purpose"] == "extract":
            transport.calls += 1
            body = {
                "model": "mock-model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "relations": [
                                        {
                                            "passage_id": "d2",
                                            "quote": "林岚毕业于明川大学",
                                            "head": "林岚",
                                            "predicate": "毕业于",
                                            "tail": "明川大学",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            }
            return 200, body
        return original(url, payload, headers)

    transport.post = post
    client = LLMClient(transport=transport, model="mock-model", base_url="http://mock")
    base.model = client
    base.config = EngineConfig(semantic_backend="llm")
    result = execute(base)
    assert result.answer is None
    purposes = [call.purpose for call in client.calls]
    assert purposes == ["extract", "verify"]
    assert client.calls[0].prompt_hash != client.calls[1].prompt_hash
    assert any(event.get("process_isolation") is True for event in result.trace if event.get("event") == "model_verify")
    assert "reranker_score" not in client.calls[1].content


def test_missing_usage_is_estimated_and_not_zero():
    transport = _ScriptedTransport('{"relations": []}', usage=None)
    client = LLMClient(transport=transport, model="mock-model", base_url="http://mock")
    response = client.complete_json(system="system", user="user", purpose="extract")
    assert response.usage_estimated is True
    assert response.prompt_tokens >= 1
    assert response.completion_tokens >= 1


def test_retries_then_auth_failure_does_not_return_an_answer():
    flaky = _ScriptedTransport(
        json.dumps({"relations": []}),
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        failures=2,
    )
    client = LLMClient(transport=flaky, model="mock-model", base_url="http://mock")
    response = client.complete_json(system="s", user="u", purpose="extract")
    assert response.attempts == 3
    denied = _ScriptedTransport("{}", failures=1, status=401)
    client = LLMClient(transport=denied, model="mock-model", base_url="http://mock")
    with pytest.raises(ModelCallError) as caught:
        client.complete_json(system="s", user="u", purpose="extract")
    assert caught.value.status == 401


def test_garbage_json_does_not_invent_a_relation():
    base = _run_input("normal_missing_edge")
    base.scripted = {}
    base.model = LLMClient(
        transport=_ScriptedTransport("not-json", usage={"prompt_tokens": 2, "completion_tokens": 1}),
        model="mock-model",
        base_url="http://mock",
    )
    result = execute(base)
    assert result.answer is None
    assert any("PARSE_ERROR" in (event.get("notes") or []) for event in result.trace if event.get("event") == "extract")


def test_no_credential_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelUnavailable):
        LLMClient().complete_json(system="s", user="u", purpose="extract")


def test_http_transport_reads_structured_usage(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert body["temperature"] == 0
            assert self.headers["X-Bridge-Purpose"] == "extract"
            raw = json.dumps(
                {
                    "model": "srv",
                    "choices": [{"message": {"content": "{\"relations\": []}"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://{host}:{port}")
        monkeypatch.setenv("OPENAI_MODEL", "srv")
        response = LLMClient().complete_json(system="sys", user="usr", purpose="extract")
    finally:
        server.shutdown()
    assert response.usage_estimated is False
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 1
    assert response.model == "srv"
