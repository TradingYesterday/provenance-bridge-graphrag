"""OpenAI-compatible client hook. Batch 1 does not call it.

If the runtime reaches this client without an explicit transport, it fails.
It does not invent an answer.
"""

from __future__ import annotations

import os


class ModelUnavailable(RuntimeError):
    """The configured model API cannot be used."""


class LLMClient:
    def __init__(self, api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL", model_env: str = "OPENAI_MODEL") -> None:
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model_env = model_env

    def complete(self, prompt: str, *, purpose: str) -> dict:
        _ = prompt
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ModelUnavailable(
                f"No credentials in {self.api_key_env}. Refusing to substitute a hardcoded answer for {purpose}."
            )
        raise ModelUnavailable(
            "Batch 1 has no API transport. Refusing to substitute a hardcoded answer for "
            f"{purpose}."
        )
