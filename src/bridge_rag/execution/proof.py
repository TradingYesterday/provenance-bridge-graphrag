"""Proof replay. Invalidated items stay in the log but drop out of the active chain."""

from __future__ import annotations

from bridge_rag.schemas import ProofItem


def active_chain(proofs: list[ProofItem]) -> list[ProofItem]:
    return sorted((proof for proof in proofs if not proof.invalidated), key=lambda proof: proof.acquisition_order)


def covers(proofs: list[ProofItem], constraint_ids: list[str]) -> bool:
    seen = {proof.constraint_id for proof in active_chain(proofs)}
    return all(constraint_id in seen for constraint_id in constraint_ids)
