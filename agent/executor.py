"""
Core agent executor.

Handles:
- Multi-turn conversation with tool-calling (max 3 model turns)
- get_brand_facts tool dispatch with brand validation
- 15-second per-request timeout + 1 retry for transient failures
- JSON parsing and full output validation
- Langfuse tracing (optional – no-ops if LANGFUSE_* not set)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import openai

from .facts import KNOWN_BRANDS, GET_BRAND_FACTS_TOOL, get_brand_facts
from .prompts import SYSTEM_V1, SYSTEM_V2, user_prompt_v1, user_prompt_v2
from .schema import AnswerResult, ClaimPrediction

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #
MAX_TURNS = 3
REQUEST_TIMEOUT = 15.0          # seconds per provider request
RETRY_DELAY = 2.0               # seconds between retry attempts
VALID_LABELS = frozenset({"supported", "contradicted", "unknown"})

# Transient error codes / messages worth retrying
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
#  Langfuse helpers (gracefully absent)                                        #
# --------------------------------------------------------------------------- #
def _make_langfuse() -> Any | None:
    try:
        from langfuse import Langfuse  # type: ignore
        key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        pk  = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if key and pk:
            return Langfuse(secret_key=key, public_key=pk, host=host)
    except ImportError:
        pass
    return None


_lf = _make_langfuse()


def _lf_trace(run_id: str, answer_id: str, split: str, version: str, **kwargs) -> Any:
    if _lf is None:
        return None
    return _lf.trace(
        name=f"claim-extraction-{version}",
        metadata={
            "run_id": run_id,
            "answer_id": answer_id,
            "split": split,
            "version": version,
            **kwargs,
        },
    )


# --------------------------------------------------------------------------- #
#  Provider request with timeout + retry                                       #
# --------------------------------------------------------------------------- #
async def _chat_request(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    tools: list[dict],
    model: str,
) -> openai.types.chat.ChatCompletion:
    """Single provider call with 15-second timeout.  Raises on failure."""
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        ),
        timeout=REQUEST_TIMEOUT,
    )


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    return False


async def _chat_with_retry(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    tools: list[dict],
    model: str,
) -> openai.types.chat.ChatCompletion:
    """Up to 2 attempts (1 retry) for transient failures."""
    for attempt in range(2):
        try:
            return await _chat_request(client, messages, tools, model)
        except Exception as exc:
            if attempt == 0 and _is_transient(exc):
                logger.warning("Transient failure (%s), retrying in %.1fs…", exc, RETRY_DELAY)
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise


# --------------------------------------------------------------------------- #
#  Output validation                                                           #
# --------------------------------------------------------------------------- #
def _validate_and_parse(
    raw: str,
    answer_id: str,
    ai_answer: str,
    fact_ids_known: set[str],
) -> list[ClaimPrediction]:
    """
    Parse JSON, validate every field and compute/verify character spans.
    Raises ValueError with a descriptive message on any violation.
    """
    # Extract JSON even if model wraps it in markdown fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    payload = json.loads(match.group())

    if payload.get("answer_id") != answer_id:
        raise ValueError(
            f"answer_id mismatch: expected {answer_id!r}, got {payload.get('answer_id')!r}"
        )

    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("'claims' must be a list")

    claims: list[ClaimPrediction] = []
    seen_ids: set[str] = set()
    prev_end = -1

    for idx, c in enumerate(raw_claims):
        # Required fields
        for f in ("claim_id", "start_char", "end_char", "quote", "label", "fact_ids", "reason"):
            if f not in c:
                raise ValueError(f"Claim #{idx} missing field {f!r}")

        start = int(c["start_char"])
        end   = int(c["end_char"])
        quote = c["quote"]
        label = c["label"]
        cid   = c["claim_id"]
        fact_ids = c["fact_ids"]
        reason   = c["reason"]

        # Span bounds
        if start < 0 or end > len(ai_answer) or start >= end:
            raise ValueError(f"Claim {cid!r}: invalid span [{start}:{end}]")

        # Quote equality
        expected_quote = ai_answer[start:end]
        if quote != expected_quote:
            raise ValueError(
                f"Claim {cid!r}: quote mismatch\n"
                f"  model:    {quote!r}\n"
                f"  actual:   {expected_quote!r}"
            )

        # Leading/trailing whitespace
        if quote != quote.strip():
            raise ValueError(f"Claim {cid!r}: quote has leading/trailing whitespace")

        # claim_id format
        expected_cid = f"{answer_id}:{start}:{end}"
        if cid != expected_cid:
            raise ValueError(f"Claim {cid!r}: expected id {expected_cid!r}")

        # Unique IDs
        if cid in seen_ids:
            raise ValueError(f"Duplicate claim_id {cid!r}")
        seen_ids.add(cid)

        # No overlaps (claims must be ordered)
        if start < prev_end:
            raise ValueError(f"Claim {cid!r}: overlaps previous claim (prev_end={prev_end})")
        prev_end = end

        # Label
        if label not in VALID_LABELS:
            raise ValueError(f"Claim {cid!r}: invalid label {label!r}")

        # fact_ids
        if not isinstance(fact_ids, list):
            raise ValueError(f"Claim {cid!r}: fact_ids must be a list")
        for fid in fact_ids:
            if fid not in fact_ids_known:
                raise ValueError(f"Claim {cid!r}: unknown fact_id {fid!r}")

        claims.append(ClaimPrediction(
            claim_id=cid,
            start_char=start,
            end_char=end,
            quote=quote,
            label=label,
            fact_ids=list(fact_ids),
            reason=str(reason),
        ))

    return claims


# --------------------------------------------------------------------------- #
#  Main agent entry point                                                      #
# --------------------------------------------------------------------------- #
async def run_agent(
    answer_id: str,
    brand_id: str,
    question: str,
    ai_answer: str,
    split: str,
    version: str,           # "v1" or "v2"
    run_id: str,
    model: str = "gpt-4o",
    client: openai.AsyncOpenAI | None = None,
) -> AnswerResult:
    if client is None:
        client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    system_prompt = SYSTEM_V1 if version == "v1" else SYSTEM_V2
    user_prompt   = (user_prompt_v1 if version == "v1" else user_prompt_v2)(
        answer_id, brand_id, question, ai_answer
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    tools = [GET_BRAND_FACTS_TOOL]

    # All known fact IDs (for validation)
    from .facts import _catalogue
    all_fact_ids = {row["fact_id"] for rows in _catalogue.values() for row in rows}

    trace = _lf_trace(
        run_id=run_id,
        answer_id=answer_id,
        split=split,
        version=version,
        brand_id=brand_id,
        model=model,
    )

    start_time = time.monotonic()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    turns = 0

    try:
        while turns < MAX_TURNS:
            turns += 1

            # Provider call with retry
            response = await _chat_with_retry(client, messages, tools, model)
            msg = response.choices[0].message

            total_prompt_tokens     += response.usage.prompt_tokens if response.usage else 0
            total_completion_tokens += response.usage.completion_tokens if response.usage else 0

            # Langfuse generation observation
            if trace:
                trace.generation(
                    name=f"turn-{turns}",
                    model=model,
                    input=messages,
                    output=msg.content or "",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                    metadata={"turn": turns, "finish_reason": response.choices[0].finish_reason},
                )

            # Handle tool calls
            if msg.tool_calls:
                # Append assistant message with tool_calls
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name

                    # Security: only allow get_brand_facts
                    if tool_name != "get_brand_facts":
                        err_msg = f"Illegal tool requested: {tool_name!r}"
                        logger.warning(err_msg)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": err_msg}),
                        })
                        continue

                    try:
                        args = json.loads(tc.function.arguments)
                        requested_brand = args.get("brand_id", "")

                        # Security: only known brands
                        if requested_brand not in KNOWN_BRANDS:
                            raise ValueError(f"Unknown brand_id: {requested_brand!r}")

                        result = get_brand_facts(requested_brand)
                        tool_content = json.dumps(result)

                    except (ValueError, KeyError) as exc:
                        tool_content = json.dumps({"error": str(exc)})

                    if trace:
                        trace.span(
                            name="get_brand_facts",
                            input=args if "args" in dir() else {},
                            output=tool_content[:500],
                            metadata={"brand_id": requested_brand},
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

                continue  # next turn to get model's response after tool results

            # No tool calls – expect final JSON answer
            raw_content = msg.content or ""
            claims = _validate_and_parse(raw_content, answer_id, ai_answer, all_fact_ids)
            latency = time.monotonic() - start_time

            if trace:
                # Attach extraction metrics metadata (scores computed by harness later)
                trace.update(
                    output={
                        "status": "ok",
                        "n_claims": len(claims),
                        "spans": [
                            {"claim_id": c.claim_id, "label": c.label} for c in claims
                        ],
                    }
                )

            return AnswerResult(
                answer_id=answer_id,
                version=version,
                split=split,
                status="ok",
                claims=claims,
                latency_seconds=latency,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                trace_id=trace.id if trace and hasattr(trace, "id") else None,
                model_turns=turns,
            )

        # Exhausted turns without a final answer
        raise RuntimeError(f"Turn limit ({MAX_TURNS}) exhausted without final answer")

    except Exception as exc:
        latency = time.monotonic() - start_time
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.error("answer %s error: %s", answer_id, err_msg)

        if trace:
            trace.update(
                output={"status": "error", "error": err_msg},
                level="ERROR",
            )

        return AnswerResult(
            answer_id=answer_id,
            version=version,
            split=split,
            status="error",
            error=err_msg,
            latency_seconds=latency,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            trace_id=trace.id if trace and hasattr(trace, "id") else None,
            model_turns=turns,
        )
