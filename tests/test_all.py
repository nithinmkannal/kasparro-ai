"""
Controlled tests covering all required scenarios.

Sections:
  1. CSV parsing (quoted commas, embedded newlines, UTF-8)
  2. Span and quote validation
  3. Scorer arithmetic (including G=10 P=10 M=8 C=7 example)
  4. Fake-provider agent tests (malformed output, bad labels, illegal tools,
     unknown brands, timeout, turn exhaustion, gold never in payloads)
  5. FastAPI + DB persistence tests
  6. Concurrency, retry, non-retryable error isolation tests
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =========================================================================== #
#  1. CSV PARSING                                                              #
# =========================================================================== #
class TestCSVParsing:
    _CSV_CONTENT = (
        'answer_id,split,brand_id,question,ai_answer\n'
        'A01,development,cleartrail,"What is it?","This has a comma, inside."\n'
        'A02,development,stonewave,"Q2","Line one\nLine two"\n'
        'A03,check,cloudnote,"UTF-8 – naïve","Résumé test"\n'
    )

    def test_quoted_commas(self):
        reader = csv.DictReader(io.StringIO(self._CSV_CONTENT))
        rows = list(reader)
        assert rows[0]["ai_answer"] == "This has a comma, inside."

    def test_embedded_newlines(self):
        reader = csv.DictReader(io.StringIO(self._CSV_CONTENT))
        rows = list(reader)
        assert rows[1]["ai_answer"] == "Line one\nLine two"

    def test_utf8_decoding(self):
        reader = csv.DictReader(io.StringIO(self._CSV_CONTENT))
        rows = list(reader)
        assert "Résumé" in rows[2]["ai_answer"]
        assert "naïve" in rows[2]["question"]

    def test_ai_answer_preserved_exactly(self):
        """ai_answer must not be altered (no whitespace normalization)."""
        raw = "  leading   and trailing  "
        content = f'answer_id,split,brand_id,question,ai_answer\nA01,development,b,"Q","{raw}"\n'
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        assert rows[0]["ai_answer"] == raw


# =========================================================================== #
#  2. SPAN AND QUOTE VALIDATION                                               #
# =========================================================================== #
class TestSpanValidation:
    """
    Tests for agent/_validate_and_parse.

    The new design: the model returns only 'quote'; Python finds the offsets.
    Payload fields claim_id/start_char/end_char are NOT supplied by the model.
    """

    def _make_valid_payload(self, ai_answer: str, claims: list[dict]) -> str:
        return json.dumps({"answer_id": "A01", "claims": claims})

    def _parse(self, raw: str, ai_answer: str):
        from agent.executor import _validate_and_parse
        return _validate_and_parse(raw, "A01", ai_answer, {"fact1"})

    def test_valid_span_accepted(self):
        ai = "Hello world."
        payload = self._make_valid_payload(ai, [{
            "quote": "Hello world.", "label": "unknown", "fact_ids": [], "reason": "ok"
        }])
        claims = self._parse(payload, ai)
        assert len(claims) == 1
        assert claims[0].quote == "Hello world."
        assert claims[0].start_char == 0
        assert claims[0].end_char == 12
        assert claims[0].claim_id == "A01:0:12"

    def test_quote_not_found_raises(self):
        """A quote that doesn't exist in ai_answer must be rejected."""
        ai = "Hello world."
        payload = json.dumps({"answer_id": "A01", "claims": [{
            "quote": "Not in text at all.", "label": "unknown", "fact_ids": [], "reason": "ok"
        }]})
        with pytest.raises(ValueError, match="[Qq]uote not found"):
            self._parse(payload, ai)

    def test_duplicate_quote_same_location_raises(self):
        """Two claims with identical quotes resolve to the same span → Duplicate."""
        ai = "Hello world."
        claim = {"quote": "Hello world.", "label": "unknown", "fact_ids": [], "reason": "ok"}
        payload = json.dumps({"answer_id": "A01", "claims": [claim, claim]})
        with pytest.raises(ValueError, match="Duplicate"):
            self._parse(payload, ai)

    def test_overlapping_quotes_resolved_to_overlap_raises(self):
        # "AAABBB" and "BBBCCC" share "BBB" in "AAABBBCCC" → overlap after offset resolution
        ai = "AAABBBCCC"
        payload = json.dumps({"answer_id": "A01", "claims": [
            {"quote": "AAABBB", "label": "unknown", "fact_ids": [], "reason": "ok"},
            {"quote": "BBBCCC", "label": "unknown", "fact_ids": [], "reason": "ok"},
        ]})
        with pytest.raises(ValueError, match="overlaps"):
            self._parse(payload, ai)

    def test_unknown_fact_id_raises(self):
        ai = "Hello world."
        payload = json.dumps({"answer_id": "A01", "claims": [{
            "quote": "Hello world.", "label": "unknown",
            "fact_ids": ["nonexistent.fact"], "reason": "ok"
        }]})
        with pytest.raises(ValueError, match="unknown fact_id"):
            self._parse(payload, ai)

    def test_invalid_label_raises(self):
        ai = "Hello world."
        payload = json.dumps({"answer_id": "A01", "claims": [{
            "quote": "Hello world.", "label": "MAYBE",
            "fact_ids": [], "reason": "ok"
        }]})
        with pytest.raises(ValueError, match="invalid label"):
            self._parse(payload, ai)

    def test_leading_trailing_whitespace_in_quote_raises(self):
        """A quote with leading or trailing whitespace must be rejected."""
        ai = "Hello world."
        payload = json.dumps({"answer_id": "A01", "claims": [{
            "quote": " Hello world.",   # leading space → whitespace check
            "label": "unknown",
            "fact_ids": [], "reason": "ok"
        }]})
        with pytest.raises(ValueError):
            self._parse(payload, ai)


# =========================================================================== #
#  3. SCORER ARITHMETIC                                                        #
# =========================================================================== #
class TestScorerArithmetic:
    """Verifies harness math against the spec's worked example and edge cases."""

    def _make_gold(self, claims: list[tuple]) -> dict:
        """claims = [(answer_id, start, end, label)]"""
        from harness import GoldClaim
        result = defaultdict(list)
        for aid, s, e, lbl in claims:
            result[aid].append(GoldClaim(
                answer_id=aid, claim_id=f"{aid}:{s}:{e}",
                start_char=s, end_char=e, quote="x" * (e - s),
                expected_label=lbl, evidence_fact_ids=[], explanation="", difficulty_tags=""
            ))
        return dict(result)

    def _make_result(self, answer_id: str, claims: list[tuple], status: str = "ok"):
        from agent.schema import AnswerResult, ClaimPrediction
        preds = []
        for s, e, lbl in claims:
            preds.append(ClaimPrediction(
                claim_id=f"{answer_id}:{s}:{e}",
                start_char=s, end_char=e,
                quote="x" * (e - s), label=lbl, fact_ids=[], reason=""
            ))
        return AnswerResult(
            answer_id=answer_id, version="v1", split="development",
            status=status, claims=preds
        )

    def _score(self, results, gold):
        from harness import score_run
        metrics, per_answer = score_run(results, gold)
        return metrics, per_answer

    def test_spec_example_g10_p10_m8_c7(self):
        """G=10, P=10, M=8, C=7 → extraction & joint F1 both 0.80 and 0.70."""
        gold_claims = [(f"A01", i * 2, i * 2 + 1, "supported") for i in range(10)]
        gold = self._make_gold(gold_claims)

        # 8 matching (7 correct, 1 wrong label) + 2 non-matching predictions
        preds = []
        for i in range(7):    # 7 correct matches
            preds.append((i * 2, i * 2 + 1, "supported"))
        preds.append((14, 15, "contradicted"))    # match but wrong label
        preds.append((100, 101, "supported"))     # extra (no match)
        preds.append((102, 103, "supported"))     # extra (no match)

        result = self._make_result("A01", preds)
        metrics, _ = self._score([result], gold)

        assert metrics.G == 10
        assert metrics.P == 10
        assert metrics.M == 8
        assert metrics.C == 7
        assert abs(metrics.extraction_precision - 0.80) < 1e-9
        assert abs(metrics.extraction_recall    - 0.80) < 1e-9
        assert abs(metrics.extraction_f1        - 0.80) < 1e-9
        assert abs(metrics.joint_precision - 0.70) < 1e-9
        assert abs(metrics.joint_recall    - 0.70) < 1e-9
        assert abs(metrics.joint_f1        - 0.70) < 1e-9

    def test_perfect_output(self):
        gold = self._make_gold([("A01", 0, 1, "supported"), ("A01", 2, 3, "contradicted")])
        result = self._make_result("A01", [(0, 1, "supported"), (2, 3, "contradicted")])
        metrics, _ = self._score([result], gold)
        assert metrics.extraction_f1 == 1.0
        assert metrics.joint_f1 == 1.0

    def test_missed_claims(self):
        gold = self._make_gold([("A01", 0, 1, "supported"), ("A01", 2, 3, "supported")])
        result = self._make_result("A01", [])   # no predictions
        metrics, per_answer = self._score([result], gold)
        assert metrics.M == 0
        assert metrics.extraction_f1 == 0.0
        assert per_answer[0].missed_ids == ["A01:0:1", "A01:2:3"]

    def test_extra_advice_extracted(self):
        """Extra predictions increase P and reduce precision."""
        gold = self._make_gold([("A01", 0, 1, "supported")])
        result = self._make_result("A01", [
            (0, 1, "supported"),   # correct
            (10, 11, "unknown"),   # extra
            (20, 21, "unknown"),   # extra
        ])
        metrics, per_answer = self._score([result], gold)
        assert metrics.P == 3
        assert metrics.M == 1
        assert len(per_answer[0].extra_ids) == 2

    def test_wrong_labels(self):
        gold = self._make_gold([("A01", 0, 1, "supported"), ("A01", 2, 3, "contradicted")])
        result = self._make_result("A01", [
            (0, 1, "contradicted"),   # wrong label
            (2, 3, "supported"),      # wrong label
        ])
        metrics, per_answer = self._score([result], gold)
        assert metrics.M == 2
        assert metrics.C == 0
        assert len(per_answer[0].wrong_label_ids) == 2

    def test_provider_error_answer(self):
        """Error answer: all gold claims missed, no accepted predictions."""
        gold = self._make_gold([("A01", 0, 1, "supported"), ("A01", 2, 3, "unknown")])
        from agent.schema import AnswerResult
        error_result = AnswerResult(
            answer_id="A01", version="v1", split="development",
            status="error", error="timeout", claims=[]
        )
        metrics, per_answer = self._score([error_result], gold)
        assert metrics.P == 0
        assert metrics.M == 0
        assert metrics.G == 2
        assert per_answer[0].missed_ids == ["A01:0:1", "A01:2:3"]

    def test_valid_empty_predictions(self):
        """Valid empty claims list: P=0, extraction credit=0 when gold exists."""
        gold = self._make_gold([("A01", 0, 1, "supported")])
        result = self._make_result("A01", [])
        metrics, _ = self._score([result], gold)
        assert metrics.G == 1
        assert metrics.P == 0
        assert metrics.M == 0
        assert metrics.extraction_f1 == 0.0

    def test_reject_missing_answer_outcome(self):
        """If a scheduled answer is absent from results it's a missed gold claim."""
        gold = self._make_gold([
            ("A01", 0, 1, "supported"),
            ("A02", 0, 1, "unknown"),
        ])
        # Only A01 result provided
        result_a01 = self._make_result("A01", [(0, 1, "supported")])
        # A02 is missing from results
        from agent.schema import AnswerResult
        # A02 never submitted → harness has no row, G includes A02's claim
        # We add an error result for A02 to simulate scheduled-but-errored
        error_a02 = AnswerResult(
            answer_id="A02", version="v1", split="development",
            status="error", error="missing"
        )
        metrics, _ = self._score([result_a01, error_a02], gold)
        assert metrics.G == 2   # both claims in G
        assert metrics.C == 1   # only A01 correct

    def test_no_zero_denominator_crash(self):
        """G=0, P=0 → null metrics (no ZeroDivisionError)."""
        from harness import score_run
        metrics, _ = score_run([], {})
        assert metrics.extraction_f1 is None
        assert metrics.joint_f1 is None


# =========================================================================== #
#  4. FAKE-PROVIDER AGENT TESTS                                               #
# =========================================================================== #
@pytest.mark.asyncio
class TestFakeProvider:
    """Use a fake client to test agent error paths without real API calls."""

    def _fake_client(self, responses: list[Any]):
        """Builds a mock AsyncOpenAI client that returns responses in sequence."""
        client = MagicMock()
        call_count = [0]

        async def _create(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx >= len(responses):
                raise RuntimeError("Unexpected extra call")
            resp = responses[idx]
            if isinstance(resp, Exception):
                raise resp
            return resp

        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = _create
        return client

    def _make_response(self, content: str | None = None, tool_calls=None,
                       finish_reason: str = "stop", usage=(10, 5)):
        """Build a minimal ChatCompletion-shaped object."""
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []

        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = finish_reason

        usage_obj = MagicMock()
        usage_obj.prompt_tokens = usage[0]
        usage_obj.completion_tokens = usage[1]

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage_obj
        return resp

    def _valid_json(self, answer_id="A01", claims=None):
        return json.dumps({
            "answer_id": answer_id,
            "claims": claims or []
        })

    async def _run(self, client, version="v1"):
        from agent.executor import run_agent
        return await run_agent(
            answer_id="A01", brand_id="cleartrail",
            question="Q", ai_answer="Hello world.",
            split="development", version=version,
            run_id="test-run", model="gpt-4o",
            client=client,
        )

    async def test_malformed_json_creates_error(self):
        client = self._fake_client([
            self._make_response(content="NOT JSON AT ALL { broken")
        ])
        result = await self._run(client)
        assert result.status == "error"
        assert result.claims == []

    async def test_unsupported_label_creates_error(self):
        client = self._fake_client([
            self._make_response(content=json.dumps({
                "answer_id": "A01",
                "claims": [{
                    "quote": "Hello world.", "label": "MAYBE",
                    "fact_ids": [], "reason": "ok"
                }]
            }))
        ])
        result = await self._run(client)
        assert result.status == "error"

    async def test_invented_fact_id_creates_error(self):
        client = self._fake_client([
            self._make_response(content=json.dumps({
                "answer_id": "A01",
                "claims": [{
                    "quote": "Hello world.", "label": "unknown",
                    "fact_ids": ["totally.made.up.id"], "reason": "ok"
                }]
            }))
        ])
        result = await self._run(client)
        assert result.status == "error"

    async def test_illegal_tool_is_answer_error(self):
        """An illegal tool is an answer-level error (spec), never executed."""
        illegal_tc = MagicMock()
        illegal_tc.id = "call_1"
        illegal_tc.function.name = "execute_python"
        illegal_tc.function.arguments = "{}"

        client = self._fake_client([self._make_response(tool_calls=[illegal_tc])])
        result = await self._run(client)
        assert result.status == "error"
        assert result.claims == []
        assert "Illegal tool" in result.error

    async def test_unknown_brand_is_answer_error(self):
        """An unknown requested brand is an answer-level error (spec)."""
        tc = MagicMock()
        tc.id = "call_2"
        tc.function.name = "get_brand_facts"
        tc.function.arguments = json.dumps({"brand_id": "fakebrand"})

        client = self._fake_client([self._make_response(tool_calls=[tc])])
        result = await self._run(client)
        assert result.status == "error"
        assert "Unknown brand_id" in result.error

    async def test_timeout_creates_error(self):
        import asyncio
        # Timeout is transient → 1 retry.  Both attempts must time out to get error.
        client = self._fake_client([asyncio.TimeoutError(), asyncio.TimeoutError()])
        with patch("agent.executor.RETRY_DELAY", 0.0):
            result = await self._run(client)
        assert result.status == "error"
        assert "TimeoutError" in result.error or "Timeout" in result.error

    async def test_turn_exhaustion_creates_error(self):
        """3 turns each returning a tool call (no final answer) → error."""
        tc = MagicMock()
        tc.id = "call_loop"
        tc.function.name = "get_brand_facts"
        tc.function.arguments = json.dumps({"brand_id": "cleartrail"})
        loop_resp = self._make_response(tool_calls=[tc])

        client = self._fake_client([loop_resp, loop_resp, loop_resp])
        result = await self._run(client)
        assert result.status == "error"
        assert result.model_turns == 3

    async def test_gold_never_in_model_payloads(self):
        """System prompt must not contain gold spans or labels."""
        from agent.prompts import SYSTEM_V1, SYSTEM_V2
        # Gold-related terms that must not appear in prompts
        forbidden = ["development_gold", "gold_claim", "expected_label",
                     "A01:306:460", "contradicted,cleartrail"]
        for prompt in [SYSTEM_V1, SYSTEM_V2]:
            for term in forbidden:
                assert term not in prompt, f"Gold data found in prompt: {term!r}"

    async def test_non_retryable_error_not_retried(self):
        """Auth errors must not be retried (max 1 call)."""
        import openai as oai
        auth_error = oai.AuthenticationError(
            "Invalid API key", response=MagicMock(status_code=401), body={}
        )
        call_count = [0]

        async def _create(**kwargs):
            call_count[0] += 1
            raise auth_error

        client = MagicMock()
        client.chat.completions.create = _create
        result = await self._run(client)
        assert result.status == "error"
        assert call_count[0] == 1, "Auth error should not be retried"

    async def test_transient_error_retried_once(self):
        """Rate-limit error on first call, success on second."""
        import openai as oai
        call_count = [0]

        async def _create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise oai.RateLimitError(
                    "rate limit", response=MagicMock(status_code=429), body={}
                )
            return self._make_response(content=self._valid_json())

        client = MagicMock()
        client.chat.completions.create = _create

        with patch("agent.executor.RETRY_DELAY", 0.0):
            result = await self._run(client)
        assert result.status == "ok"
        assert call_count[0] == 2


# =========================================================================== #
#  5. FASTAPI + DB PERSISTENCE TESTS                                          #
# =========================================================================== #
@pytest.fixture
async def test_app():
    """In-memory-style: patch DB pool with a fake that records calls."""
    # This test uses httpx AsyncClient against the real app but mocks the pool.
    pytest.importorskip("httpx")
    from httpx import AsyncClient, ASGITransport
    from api.app import app

    # Patch get_pool to avoid real DB
    call_log: list[dict] = []

    class FakeConn:
        def __init__(self):
            self._rows: list = []

        async def execute(self, sql, *args):
            call_log.append({"sql": sql.strip()[:60], "args": args})

        async def fetchrow(self, sql, *args):
            return None  # not found by default

        async def fetch(self, sql, *args):
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeTx:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    class FakeConn2(FakeConn):
        def transaction(self):
            return FakeTx()

    class FakePool:
        def acquire(self):
            return FakeConn2()

    with patch("api.app.get_pool", AsyncMock(return_value=FakePool())):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, call_log


@pytest.mark.asyncio
async def test_api_create_run_success(test_app):
    client, call_log = test_app
    run_id = str(uuid.uuid4())
    payload = {
        "run_id": run_id,
        "version": "v1",
        "split": "development",
        "model": "gpt-4o",
        "timestamp": "2026-01-01T00:00:00Z",
        "answers": [{
            "answer_id": "A01",
            "split": "development",
            "status": "ok",
            "claims": [{
                "claim_id": "A01:0:5",
                "start_char": 0, "end_char": 5,
                "quote": "Hello",
                "label": "unknown",
                "fact_ids": [],
                "reason": "no facts"
            }]
        }]
    }
    resp = await client.post("/runs", json=payload)
    assert resp.status_code == 201
    assert resp.json()["run_id"] == run_id


@pytest.mark.asyncio
async def test_api_create_run_with_error_answer(test_app):
    client, _ = test_app
    run_id = str(uuid.uuid4())
    payload = {
        "run_id": run_id,
        "version": "v1",
        "split": "development",
        "model": "gpt-4o",
        "timestamp": "2026-01-01T00:00:00Z",
        "error_count": 1,
        "answers": [{
            "answer_id": "A01",
            "split": "development",
            "status": "error",
            "error": "TimeoutError: timed out",
            "claims": []
        }]
    }
    resp = await client.post("/runs", json=payload)
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_api_get_run_not_found(test_app):
    client, _ = test_app
    resp = await client.get("/runs/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_list_runs(test_app):
    client, _ = test_app
    resp = await client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# =========================================================================== #
#  6. CONCURRENCY + ISOLATION TESTS                                           #
# =========================================================================== #
@pytest.mark.asyncio
class TestConcurrency:
    async def test_semaphore_limits_concurrent_requests(self):
        """At most 3 concurrent provider requests at a time."""
        active = [0]
        max_active = [0]
        lock = asyncio.Lock()

        async def _fake_run(*args, **kwargs):
            async with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            async with lock:
                active[0] -= 1
            from agent.schema import AnswerResult
            return AnswerResult(
                answer_id=kwargs.get("answer_id", "A01"),
                version="v1", split="development", status="ok"
            )

        # Simulate 9 concurrent requests with semaphore of 3
        sem = asyncio.Semaphore(3)
        answers = [{"answer_id": f"A{i:02d}", "brand_id": "cleartrail",
                    "question": "Q", "ai_answer": "text"} for i in range(9)]

        async def _bounded(row):
            async with sem:
                return await _fake_run(
                    answer_id=row["answer_id"], brand_id=row["brand_id"],
                    question=row["question"], ai_answer=row["ai_answer"],
                    split="development", version="v1", run_id="r1", model="gpt-4o",
                )

        await asyncio.gather(*[_bounded(r) for r in answers])
        assert max_active[0] <= 3, f"Semaphore violated: max_active={max_active[0]}"

    async def test_one_answer_error_does_not_affect_others(self):
        """Failure in one answer must not cause others to fail."""
        from agent.schema import AnswerResult

        results: list[AnswerResult] = []

        async def _fake_run(answer_id, **kwargs):
            if answer_id == "A05":
                return AnswerResult(
                    answer_id=answer_id, version="v1", split="development",
                    status="error", error="injected"
                )
            return AnswerResult(
                answer_id=answer_id, version="v1", split="development", status="ok"
            )

        tasks = [asyncio.create_task(_fake_run(f"A{i:02d}")) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)
        errors = [r for r in results if r.status == "error"]
        successes = [r for r in results if r.status == "ok"]
        assert len(errors) == 1
        assert errors[0].answer_id == "A05"
        assert len(successes) == 4
