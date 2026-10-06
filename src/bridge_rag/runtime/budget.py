"""Shared budget ledger. Cached repeats still record a logical charge."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetLedger:
    max_steps: int
    max_active_branches: int
    max_recoveries_per_key: int
    steps_used: int = 0
    logical_calls: int = 0
    paid_calls: int = 0
    max_api_calls: int = 16
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    recoveries: dict[str, int] = field(default_factory=dict)

    def charge_step(self) -> bool:
        if self.steps_used >= self.max_steps:
            return False
        self.steps_used += 1
        self.logical_calls += 1
        return True

    def allow_recovery(self, key: str) -> bool:
        used = self.recoveries.get(key, 0)
        if used >= self.max_recoveries_per_key:
            return False
        self.recoveries[key] = used + 1
        self.logical_calls += 1
        return True

    def note_cache_hit(self) -> None:
        self.logical_calls += 1

    def note_paid_call(self) -> None:
        self.paid_calls += 1
        self.logical_calls += 1

    def allow_api_call(self) -> bool:
        if self.api_calls >= self.max_api_calls:
            return False
        self.api_calls += 1
        self.paid_calls += 1
        self.logical_calls += 1
        return True

    def note_usage(self, prompt_tokens: int, completion_tokens: int, estimated: bool) -> None:
        if estimated:
            self.estimated_input_tokens += prompt_tokens
            self.estimated_output_tokens += completion_tokens
        else:
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens

    def view(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "steps_used": self.steps_used,
            "max_active_branches": self.max_active_branches,
            "recoveries_used": sum(self.recoveries.values()),
            "logical_calls": self.logical_calls,
            "paid_calls": self.paid_calls,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
        }
