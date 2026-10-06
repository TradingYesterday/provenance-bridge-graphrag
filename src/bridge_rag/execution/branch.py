"""Branch copy boundary. Child branches do not share binding dicts with the parent."""

from bridge_rag.recovery.commit import fork_branch

__all__ = ["fork_branch"]
