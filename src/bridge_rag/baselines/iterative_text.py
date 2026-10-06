"""Iterative text baseline. Not available in batch 1."""

from bridge_rag.baselines.cog_controlled import BaselineUnavailable


def run(*_args, **_kwargs):
    raise BaselineUnavailable("iterative_text is not implemented in batch 1.")
