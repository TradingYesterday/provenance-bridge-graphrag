"""Controlled baselines on the shared executor.

The upstream CS-RAG process is still not executed here.
"""

UNAVAILABLE = ("csrag_upstream_runner",)
AVAILABLE = ("core00", "core01", "core10", "core11", "simple_bind", "cog_controlled", "iterative_text")
