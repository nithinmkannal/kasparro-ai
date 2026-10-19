"""
Schema types for agent output.  Pure dataclasses – no LLM coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClaimPrediction:
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str          # supported | contradicted | unknown
    fact_ids: list[str]
    reason: str


@dataclass
class AnswerResult:
    answer_id: str
    version: str        # v1 | v2
    split: str          # development | check
    status: str         # ok | error
    claims: list[ClaimPrediction] = field(default_factory=list)
    error: str | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str | None = None
    model_turns: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
