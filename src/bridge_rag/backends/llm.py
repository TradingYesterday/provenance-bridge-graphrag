"""OpenAI-compatible chat client.

A missing credential or a failed call raises. The client never fills in an answer.
Token usage is taken from the response. If the payload omits usage, the client
records a conservative character upper bound and marks it estimated. It does not
store zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from bridge_rag.runtime.budget import BudgetLedger


class ModelUnavailable(RuntimeError):
    """No credential or no transport. Do not replace the call with a stored answer."""


class ModelCallError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"model call failed status={status}")


@dataclass
class ModelResponse:
    content: str
    model: str
    purpose: str
    prompt_hash: str
    raw_hash: str
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    attempts: int


class Transport(Protocol):
    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        ...


class UrllibTransport:
    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelCallError(exc.code, body) from exc


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _upper_bound(text: str) -> int:
    return max(1, len(text))


class LLMClient:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        model_env: str = "OPENAI_MODEL",
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.transport = transport
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model_env = model_env
        self._model = model
        self._base_url = base_url
        self.calls: list[ModelResponse] = []

    def _resolve(self) -> tuple[str, str, str]:
        key = os.environ.get(self.api_key_env, "")
        if self.transport is None and not key:
            raise ModelUnavailable(
                f"No credentials in {self.api_key_env}. Refusing to substitute a hardcoded answer."
            )
        if self.transport is None and not self._base_url and not os.environ.get(self.base_url_env, ""):
            raise ModelUnavailable("OPENAI_BASE_URL is empty. Refusing to substitute a hardcoded answer.")
        model = self._model or os.environ.get(self.model_env, "")
        if not model:
            raise ModelUnavailable("OPENAI_MODEL is empty. Refusing to substitute a hardcoded answer.")
        base = (self._base_url or os.environ.get(self.base_url_env, "")).rstrip("/")
        return key, model, base

    def complete(self, prompt: str, *, purpose: str) -> dict:
        response = self.complete_json(system="", user=prompt, purpose=purpose)
        return {"content": response.content, "usage_estimated": response.usage_estimated}

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        purpose: str,
        ledger: BudgetLedger | None = None,
    ) -> ModelResponse:
        key, model, base = self._resolve()
        prompt = system + "\n" + user
        prompt_hash = _digest(prompt)
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "X-Bridge-Purpose": purpose,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        transport = self.transport or UrllibTransport()
        url = f"{base}/chat/completions" if base else ""
        if self.transport is None and not url:
            raise ModelUnavailable("No API URL. Refusing to substitute a hardcoded answer.")
        last_error: ModelCallError | None = None
        for attempt in range(1, 4):
            if ledger is not None and not ledger.allow_api_call():
                raise ModelCallError(0, "API budget exhausted")
            try:
                _status, body = transport.post(url or "mock://chat", payload, headers)
            except ModelCallError as exc:
                last_error = exc
                if exc.status in {400, 401, 403}:
                    raise
                continue
            content = _message_content(body)
            prompt_tokens, completion_tokens, estimated = _usage(body, prompt, content)
            response = ModelResponse(
                content=content,
                model=str(body.get("model") or model),
                purpose=purpose,
                prompt_hash=prompt_hash,
                raw_hash=_digest(json.dumps(body, ensure_ascii=False, sort_keys=True)),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_estimated=estimated,
                attempts=attempt,
            )
            if ledger is not None:
                ledger.note_usage(prompt_tokens, completion_tokens, estimated)
            self.calls.append(response)
            return response
        assert last_error is not None
        raise last_error


def _message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ModelCallError(200, "response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ModelCallError(200, "response content is empty")
    return content


def _usage(body: dict[str, Any], prompt: str, content: str) -> tuple[int, int, bool]:
    usage = body.get("usage")
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
        return int(usage["prompt_tokens"]), int(usage["completion_tokens"]), False
    return _upper_bound(prompt), _upper_bound(content), True
