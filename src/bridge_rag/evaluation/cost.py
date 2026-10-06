"""Call and token cost read from the trace. Estimated usage is kept separate."""

from __future__ import annotations

from bridge_rag.schemas import RunResult


def cost_view(result: RunResult) -> dict:
    api_calls = 0
    input_tokens = 0
    output_tokens = 0
    estimated_input = 0
    estimated_output = 0
    graph_accesses = 0
    extracts = 0
    for event in result.trace:
        name = event.get("event")
        if name == "graph_access":
            graph_accesses += 1
        elif name == "extract":
            extracts += 1
        elif name == "model_verify" or (name == "extract" and event.get("extractor_model") not in {None, "oracle-scripted"}):
            pass
        if event.get("usage_estimated") is True:
            estimated_input += int(event.get("prompt_tokens") or 0)
            estimated_output += int(event.get("completion_tokens") or 0)
            api_calls += 1
        elif event.get("prompt_tokens") is not None and name == "model_verify":
            input_tokens += int(event["prompt_tokens"])
            output_tokens += int(event.get("completion_tokens") or 0)
            api_calls += 1
    return {
        "api_calls_in_trace": api_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "graph_accesses": graph_accesses,
        "extract_steps": extracts,
        "cache_keys": len(result.cache_keys),
    }
