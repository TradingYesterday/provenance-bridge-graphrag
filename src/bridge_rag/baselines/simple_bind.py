"""Slot filling on the shared executor. It does not use B's relation-commit gate."""

from bridge_rag.execution.engine import RunInput
from bridge_rag.execution.methods import run_method
from bridge_rag.schemas import RunResult


def run(run_input: RunInput) -> RunResult:
    return run_method("simple_bind", run_input)
