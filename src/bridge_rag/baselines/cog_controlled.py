"""Controlled CoG-style notebook on the same graph and text pool.

This is not a reproduction of the original global-KG CoG setup. It keeps facts,
clues, judgments, analysis, and follow-up queries, and it does not use B's
relation-commit rule.
"""

from bridge_rag.execution.engine import RunInput
from bridge_rag.execution.methods import run_method
from bridge_rag.schemas import RunResult


class BaselineUnavailable(RuntimeError):
    pass


def run(run_input: RunInput) -> RunResult:
    return run_method("cog_controlled", run_input)


def upstream_csrag(*_args, **_kwargs):
    raise BaselineUnavailable(
        "The pinned CS-RAG process is not executed in this repository. "
        "See docs/UPSTREAM_AUDIT.md for the wrapper command."
    )
