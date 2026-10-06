"""Follow-up text retrieval without graph re-entry. Same executor and budget knobs."""

from bridge_rag.execution.engine import RunInput
from bridge_rag.execution.methods import run_method
from bridge_rag.schemas import RunResult


def run(run_input: RunInput) -> RunResult:
    return run_method("iterative_text", run_input)
