"""
Core agent executor.

Handles:
- Multi-turn conversation with tool-calling (max 3 model turns)
- get_brand_facts tool dispatch with tool-name and brand validation
  (an illegal tool or unknown brand is an answer-level error)
- 15-second per-request timeout + 1 retry for transient failures
- JSON parsing and full output validation
- Python-side offset resolution: the model returns only the quote text;
  Python locates the substring in ai_answer and computes start/end offsets.
  This avoids off-by-N counting errors from the model.
- Langfuse tracing (v5 API; optional – no-ops if LANGFUSE_* not set; never fatal)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import openai

load_dotenv()

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


class ToolError(Exception):
    """Illegal tool name or invalid tool arguments requested by the model."""


def _sanitize_error(msg: str) -> str:
    """Strip anything key-shaped from an error message and bound its length."""
    msg = re.sub(r"\b(?:sk|pk)-[A-Za-z0-9_\-]{6,}", "[REDACTED]", msg)
    msg = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", msg)
    return msg[:500]


# --------------------------------------------------------------------------- #
#  Langfuse helpers – v5 API (gracefully absent, never fatal)                 #
# --------------------------------------------------------------------------- #
def _make_langfuse() -> Any | None:
    try:
        from langfuse import Langfuse  # type: ignore
        key  = os.environ.get("LANGFUSE_SECRET_KEY", "")
        pk   = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        # Treat placeholder values (containing "...") as unset
        real_key = key and "..." not in key
        real_pk  = pk  and "..." not in pk
        if real_key and real_pk:
            return Langfuse(secret_key=key, public_key=pk, host=host)
    except Exception as exc:
        logger.warning("Langfuse disabled: %s", type(exc).__name__)
    return None


_lf = _make_langfuse()


def _lf_start_trace(
    run_id: str, answer_id: str, split: str, version: str, **kwargs
) -> Any:
    """
    Create a root span (trace) for one answer using the Langfuse v5 API.
    Returns the span object, or None if tracing is unavailable.
    """
    if _lf is None:
        return None
    try:
        # v5: create a root span that acts as the trace container
        trace_id = _lf.create_trace_id(seed=f"{run_id}:{answer_id}")
        span = _lf.start_observation(
            trace_context={"trace_id": trace_id},
            name=f"claim-extraction-{version}",
            as_type="span",
            metadata={
                "run_id": run_id,
                "answer_id": answer_id,
                "split": split,
                "version": version,
                **kwargs,
            },
        )
        return span
    except Exception as exc:
        logger.warning("Langfuse trace creation failed: %s", type(exc).__name__)
        return None


def _trace_generation(trace: Any, name: str, **kwargs) -> Any:
    """Add a generation child observation; returns the child or None."""
    if trace is None:
        return None
    try:
        return trace.start_observation(name=name, as_type="generation", **kwargs)
    except Exception as exc:
        logger.warning("Langfuse generation failed: %s", type(exc).__name__)
        return None


def _trace_tool(trace: Any, name: str, **kwargs) -> Any:
    """Add a tool child observation; returns the child or None."""
    if trace is None:
        return None
    try:
        return trace.start_observation(name=name, as_type="tool", **kwargs)
    except Exception as exc:
        logger.warning("Langfuse tool span failed: %s", type(exc).__name__)
        return None


def _obs_end(obs: Any, output: Any = None, level: str | None = None) -> None:
    """End an observation, updating output/level first if provided."""
    if obs is None:
        return
    try:
        kw: dict[str, Any] = {}
        if output is not None:
            kw["output"] = output
        if level is not None:
            kw["level"] = level
        if kw:
            obs.update(**kw)
        obs.end()
    except Exception as exc:
        logger.warning("Langfuse end failed: %s", type(exc).__name__)


def _trace_update(trace: Any, **kwargs) -> None:
    """Update a span's metadata/output; tracing problems must never fail."""
    if trace is None:
        return
    try:
        trace.update(**kwargs)
    except Exception as exc:
        logger.warning("Langfuse update failed: %s", type(exc).__name__)


def _lf_score(
    trace: Any,
    name: str,
    value: float,
    comment: str | None = None,
    observation_id: str | None = None,
) -> None:
    """Attach a numeric score to a trace (or observation) via the v5 API."""
    if _lf is None or trace is None:
        return
    try:
        tid = getattr(trace, "trace_id", None)
        if tid is None:
            return
        _lf.create_score(
            trace_id=tid,
            observation_id=observation_id,
            name=name,
            value=value,
            data_type="NUMERIC",
            comment=comment,
        )
    except Exception as exc:
        logger.warning("Langfuse score failed: %s", type(exc).__name__)


def flush_traces() -> None:
    if _lf is None:
        return
    try:
        _lf.flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", type(exc).__name__)


def get_trace_id(trace: Any) -> str | None:
    """Return the trace_id string from a span, or None."""
    if trace is None:
        return None
    try:
        return getattr(trace, "trace_id", None)
    except Exception:
        return None


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
    if isinstance(exc, openai.APIConnectionError):   # includes APITimeoutError
        return True
    return False


def _retry_delay(exc: Exception) -> float:
    """
    Return how long to wait before retrying.
    For rate-limit errors, Groq and Gemini embed wait times in the message
    e.g. "Please try again in 7.7475s" or "Please retry in 24.698s" — parse if
    present, else fall back to RETRY_DELAY.
    """
    if isinstance(exc, openai.RateLimitError):
        match = re.search(r"(?:try again|retry)\s+in\s+([\d.]+)s", str(exc), re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 0.5, 65.0)  # add 0.5s buffer, cap at 65s
    return RETRY_DELAY


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
                delay = _retry_delay(exc)
                logger.warning("Transient failure (%s), retrying in %.1fs…", type(exc).__name__, delay)
                await asyncio.sleep(delay)
            else:
                raise


# --------------------------------------------------------------------------- #
#  Tool execution                                                              #
# --------------------------------------------------------------------------- #
def _execute_tool_call(tc: Any) -> tuple[str, dict]:
    """
    Validate and execute one model-requested tool call.
    Returns (json_content, parsed_args).  Raises ToolError on any violation.
    """
    name = tc.function.name
    if name != "get_brand_facts":
        raise ToolError(f"Illegal tool requested: {name!r}")

    try:
        args = json.loads(tc.function.arguments)
    except (TypeError, ValueError) as exc:
        raise ToolError("get_brand_facts arguments are not valid JSON") from exc
    if not isinstance(args, dict):
        raise ToolError("get_brand_facts arguments must be a JSON object")

    brand = args.get("brand_id")
    if not isinstance(brand, str) or brand not in KNOWN_BRANDS:
        raise ToolError(f"Unknown brand_id: {brand!r}")

    return json.dumps(get_brand_facts(brand)), args


# --------------------------------------------------------------------------- #
#  Python-side offset resolution                                               #
# --------------------------------------------------------------------------- #
def _find_quote_offset(
    quote: str,
    ai_answer: str,
    context_before: str | None = None,
    context_after: str | None = None,
) -> tuple[int, int]:
    """
    Locate `quote` in `ai_answer` and return (start_char, end_char).
    Uses context clues to disambiguate repeated occurrences.
    Raises ValueError if the quote cannot be found or is ambiguous without context.

    The spec says: "Do not rely on the model to count characters."
    Python locates the exact Unicode code-point offsets from the model's
    exact quoted text.
    """
    q = quote.strip()
    if not q:
        raise ValueError("Empty quote")

    # Find all non-overlapping occurrences
    occurrences: list[int] = []
    start = 0
    while True:
        idx = ai_answer.find(q, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + 1

    if not occurrences:
        raise ValueError(f"Quote not found in ai_answer: {q[:80]!r}")

    if len(occurrences) == 1:
        s = occurrences[0]
        return s, s + len(q)

    # Multiple occurrences: use context to disambiguate
    if context_before or context_after:
        for s in occurrences:
            e = s + len(q)
            before_ok = True
            after_ok  = True
            if context_before:
                cb = context_before.strip()
                # The text just before this occurrence should end with context_before
                pre = ai_answer[max(0, s - len(cb) - 20):s]
                before_ok = cb in pre
            if context_after:
                ca = context_after.strip()
                post = ai_answer[e:e + len(ca) + 20]
                after_ok = ca in post
            if before_ok and after_ok:
                return s, s + len(q)

    # Fall back to first occurrence and warn
    logger.warning(
        "Ambiguous quote (found %d times); using first occurrence: %r",
        len(occurrences), q[:60],
    )
    s = occurrences[0]
    return s, s + len(q)


# --------------------------------------------------------------------------- #
#  Output validation                                                           #
# --------------------------------------------------------------------------- #
def _strict_int(value: Any, field: str, cid: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Claim {cid!r}: {field} must be an integer, got {value!r}")
    return value


def _validate_and_parse(
    raw: str,
    answer_id: str,
    ai_answer: str,
    fact_ids_known: set[str],
) -> list[ClaimPrediction]:
    """
    Parse JSON, resolve character offsets from quotes (Python-side), validate
    all fields, and return sorted, non-overlapping ClaimPredictions.
    Raises ValueError with a descriptive message on any violation.
    """
    # Extract JSON even if model wraps it in markdown fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    payload = json.loads(match.group())

    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object")
    if payload.get("answer_id") != answer_id:
        raise ValueError(
            f"answer_id mismatch: expected {answer_id!r}, got {payload.get('answer_id')!r}"
        )

    if "claims" not in payload:
        raise ValueError("Model output is missing the 'claims' list")
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise ValueError("'claims' must be a list")

    claims_raw: list[dict] = []

    for idx, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            raise ValueError(f"Claim #{idx} must be an object")

        # Required fields
        for f in ("quote", "label", "fact_ids", "reason"):
            if f not in c:
                raise ValueError(f"Claim #{idx} missing field {f!r}")

        quote    = c["quote"]
        label    = c["label"]
        fact_ids = c["fact_ids"]
        reason   = c["reason"]

        if not isinstance(quote, str):
            raise ValueError(f"Claim #{idx}: quote must be a string, got {quote!r}")
        if not isinstance(reason, str):
            raise ValueError(f"Claim #{idx}: reason must be a string")

        # Leading/trailing whitespace
        if quote != quote.strip():
            raise ValueError(f"Claim #{idx}: quote has leading/trailing whitespace")
        if not quote:
            raise ValueError(f"Claim #{idx}: quote must not be empty")

        # Python locates the offsets from the quote text
        context_before = c.get("context_before")
        context_after  = c.get("context_after")
        start, end = _find_quote_offset(quote, ai_answer, context_before, context_after)

        # Label
        if not isinstance(label, str) or label not in VALID_LABELS:
            raise ValueError(f"Claim #{idx}: invalid label {label!r}")

        # fact_ids
        if not isinstance(fact_ids, list):
            raise ValueError(f"Claim #{idx}: fact_ids must be a list")
        for fid in fact_ids:
            if not isinstance(fid, str) or fid not in fact_ids_known:
                raise ValueError(f"Claim #{idx}: unknown fact_id {fid!r}")

        cid = f"{answer_id}:{start}:{end}"
        claims_raw.append({
            "claim_id": cid,
            "start_char": start,
            "end_char": end,
            "quote": quote,
            "label": label,
            "fact_ids": list(fact_ids),
            "reason": reason,
        })

    # Sort by start_char and enforce no overlaps / no duplicates
    claims_raw.sort(key=lambda x: x["start_char"])

    seen_ids: set[str] = set()
    prev_end = -1
    claims: list[ClaimPrediction] = []

    for c in claims_raw:
        cid   = c["claim_id"]
        start = c["start_char"]
        end   = c["end_char"]

        if cid in seen_ids:
            raise ValueError(f"Duplicate claim_id {cid!r}")
        seen_ids.add(cid)

        if start < prev_end:
            raise ValueError(f"Claim {cid!r}: overlaps previous claim (prev_end={prev_end})")
        prev_end = end

        claims.append(ClaimPrediction(
            claim_id=cid,
            start_char=start,
            end_char=end,
            quote=c["quote"],
            label=c["label"],
            fact_ids=c["fact_ids"],
            reason=c["reason"],
        ))

    return claims


# --------------------------------------------------------------------------- #
#  Provider client factory                                                     #
# --------------------------------------------------------------------------- #
# Provider base URLs (all OpenAI-compatible)
_GROQ_BASE_URL    = "https://api.groq.com/openai/v1"
_GEMINI_BASE_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/"

_PROVIDER_CONFIGS: dict[str, dict] = {
    "openai": {"env_key": "OPENAI_API_KEY",  "base_url": None},
    "groq":   {"env_key": "GROQ_API_KEY",    "base_url": _GROQ_BASE_URL},
    "gemini": {"env_key": "GEMINI_API_KEY",  "base_url": _GEMINI_BASE_URL},
}


def _make_client(provider: str) -> openai.AsyncOpenAI:
    """Build an AsyncOpenAI-compatible client for the given provider."""
    cfg = _PROVIDER_CONFIGS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider {provider!r}. Choices: {list(_PROVIDER_CONFIGS)}")
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Environment variable {cfg['env_key']} is not set for provider {provider!r}"
        )
    kwargs: dict = {"api_key": api_key, "max_retries": 0}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return openai.AsyncOpenAI(**kwargs)


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
    provider: str = "openai",
) -> AnswerResult:
    if client is None:
        # max_retries=0: the SDK must not add hidden retries on top of ours
        client = _make_client(provider)

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

    # Langfuse v5: root span for this answer
    trace = _lf_start_trace(
        run_id=run_id,
        answer_id=answer_id,
        split=split,
        version=version,
        brand_id=brand_id,
        model=model,
        provider=provider,
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

            pt = response.usage.prompt_tokens     if response.usage else 0
            ct = response.usage.completion_tokens if response.usage else 0
            total_prompt_tokens     += pt
            total_completion_tokens += ct

            # Langfuse v5: generation child observation
            gen_obs = _trace_generation(
                trace,
                name=f"turn-{turns}",
                model=model,
                input=messages,
                usage_details={"input": pt, "output": ct},
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
                    try:
                        tool_content, args = _execute_tool_call(tc)
                    except ToolError as exc:
                        tool_obs = _trace_tool(
                            gen_obs,
                            name="get_brand_facts",
                            input={"tool": str(tc.function.name)},
                        )
                        _obs_end(tool_obs, output=str(exc), level="ERROR")
                        _obs_end(gen_obs, output=msg.content or "")
                        raise  # illegal tool / unknown brand => answer-level error

                    tool_obs = _trace_tool(
                        gen_obs,
                        name="get_brand_facts",
                        input=args,
                        metadata={"brand_id": args["brand_id"]},
                    )
                    _obs_end(tool_obs, output=tool_content[:500])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

                _obs_end(gen_obs, output=msg.content or "")
                continue  # next turn to get model's response after tool results

            # No tool calls – expect final JSON answer
            raw_content = msg.content or ""
            claims = _validate_and_parse(raw_content, answer_id, ai_answer, all_fact_ids)
            latency = time.monotonic() - start_time

            _obs_end(gen_obs, output=raw_content[:500])
            _trace_update(
                trace,
                output={
                    "status": "ok",
                    "n_claims": len(claims),
                    "spans": [
                        {"claim_id": c.claim_id, "label": c.label} for c in claims
                    ],
                },
            )
            _obs_end(trace)

            return AnswerResult(
                answer_id=answer_id,
                version=version,
                split=split,
                status="ok",
                claims=claims,
                latency_seconds=latency,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                trace_id=get_trace_id(trace),
                model_turns=turns,
            )

        # Exhausted turns without a final answer
        raise RuntimeError(f"Turn limit ({MAX_TURNS}) exhausted without final answer")

    except Exception as exc:
        latency = time.monotonic() - start_time
        err_msg = _sanitize_error(f"{type(exc).__name__}: {exc}")
        logger.error("answer %s error: %s", answer_id, err_msg)

        _trace_update(trace, output={"status": "error", "error": err_msg}, level="ERROR")
        _obs_end(trace, level="ERROR")

        return AnswerResult(
            answer_id=answer_id,
            version=version,
            split=split,
            status="error",
            error=err_msg,
            latency_seconds=latency,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            trace_id=get_trace_id(trace),
            model_turns=turns,
        )


# --------------------------------------------------------------------------- #
#  Post-scoring Langfuse scores (called by run.py after harness scoring)      #
# --------------------------------------------------------------------------- #
def attach_langfuse_scores(
    answer_id: str,
    trace_id: str | None,
    extraction_f1: float | None,
    joint_f1: float | None,
    claim_correctness: list[dict] | None = None,
    missing_ids: list[str] | None = None,
    extra_ids: list[str] | None = None,
) -> None:
    """
    Attach deterministic per-answer Langfuse scores after gold matching.
    Safe to call even when Langfuse is unavailable.

    claim_correctness: list of {"claim_id": str, "label_correct": bool}
    """
    if _lf is None or trace_id is None:
        return
    try:
        if extraction_f1 is not None:
            _lf.create_score(
                trace_id=trace_id,
                name="extraction_f1",
                value=extraction_f1,
                data_type="NUMERIC",
                comment=f"answer={answer_id}",
            )
        if joint_f1 is not None:
            _lf.create_score(
                trace_id=trace_id,
                name="joint_f1",
                value=joint_f1,
                data_type="NUMERIC",
                comment=f"answer={answer_id}",
            )
        for cc in (claim_correctness or []):
            _lf.create_score(
                trace_id=trace_id,
                name="label_correct",
                value=1.0 if cc["label_correct"] else 0.0,
                data_type="NUMERIC",
                comment=f"claim={cc['claim_id']}",
            )
    except Exception as exc:
        logger.warning("Langfuse attach_scores failed: %s", type(exc).__name__)
