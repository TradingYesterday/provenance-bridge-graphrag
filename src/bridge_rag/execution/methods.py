"""Named runs on the shared executor. Each method gets its own graph copy."""

from __future__ import annotations

import copy

from bridge_rag.execution.engine import RunInput, execute
from bridge_rag.schemas import EngineConfig, RunResult

PRESETS: dict[str, dict] = {
    "core00": {
        "method": "relation_commit",
        "allow_text_binding": False,
        "verify_text": False,
        "allow_successor_reentry": False,
        "verify_graph": True,
    },
    "core01": {
        "method": "relation_commit",
        "allow_text_binding": False,
        "verify_text": True,
        "allow_successor_reentry": False,
        "verify_graph": True,
    },
    "core10": {
        "method": "relation_commit",
        "allow_text_binding": True,
        "verify_text": False,
        "allow_successor_reentry": True,
        "verify_graph": True,
    },
    "core11": {
        "method": "relation_commit",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": True,
        "verify_graph": True,
    },
    "simple_bind": {
        "method": "simple_slot",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": True,
        "verify_graph": True,
    },
    "cog_controlled": {
        "method": "cog_notebook",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": True,
        "verify_graph": False,
    },
    "iterative_text": {
        "method": "iterative_text",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": False,
        "verify_graph": True,
    },
    "no_reentry": {
        "method": "relation_commit",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": False,
        "verify_graph": True,
    },
    "verify_graph_off": {
        "method": "relation_commit",
        "allow_text_binding": True,
        "verify_text": True,
        "allow_successor_reentry": True,
        "verify_graph": False,
    },
}


def run_method(name: str, run: RunInput, *, config: EngineConfig | None = None) -> RunResult:
    if name not in PRESETS:
        raise KeyError(name)
    cloned = copy.deepcopy(run)
    base = config or cloned.config
    cloned.config = base.model_copy(update=PRESETS[name])
    cloned.graph.match_log.clear()
    return execute(cloned)
