"""Simple text-bind baseline. Not available in batch 1."""

from bridge_rag.baselines.cog_controlled import BaselineUnavailable


def run(*_args, **_kwargs):
    raise BaselineUnavailable("simple_bind is not implemented in batch 1.")
