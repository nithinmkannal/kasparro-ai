"""
FastAPI application – thin routes over PostgreSQL.

Tables:
  runs            – one row per run
  answer_results  – one row per (run_id, answer_id)
  claim_results   – one row per accepted prediction

Schema enforces:
  UNIQUE (run_id, answer_id)
  UNIQUE (run_id, answer_id, start_char, end_char)
  Foreign keys: answer_results.run_id → runs.run_id
                claim_results.run_id, claim_results.answer_id →
                    answer_results(run_id, answer_id)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel




# --------------------------------------------------------------------------- #
#  DB connection                                                               #
# --------------------------------------------------------------------------- #
_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          TEXT PRIMARY KEY,
                version         TEXT NOT NULL,
                split           TEXT NOT NULL,
                model           TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                dataset_version TEXT,
                fact_version    TEXT,
                evaluator_version TEXT,
                prompt_version  TEXT,
                file_hashes     JSONB,
                metrics         JSONB,
                settings        JSONB,
                error_count     INT DEFAULT 0,
                wall_time_seconds FLOAT
            );

            CREATE TABLE IF NOT EXISTS answer_results (
                run_id          TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                answer_id       TEXT NOT NULL,
                split           TEXT NOT NULL,
                status          TEXT NOT NULL,
                error           TEXT,
                latency_seconds FLOAT,
                prompt_tokens   INT DEFAULT 0,
                completion_tokens INT DEFAULT 0,
                model_turns     INT DEFAULT 0,
                trace_id        TEXT,
                per_answer_metrics JSONB,
                PRIMARY KEY (run_id, answer_id)
            );

            CREATE TABLE IF NOT EXISTS claim_results (
                run_id      TEXT NOT NULL,
                answer_id   TEXT NOT NULL,
                claim_id    TEXT NOT NULL,
                start_char  INT NOT NULL,
                end_char    INT NOT NULL,
                quote       TEXT NOT NULL,
                label       TEXT NOT NULL,
                fact_ids    JSONB,
                reason      TEXT,
                match_gold  BOOLEAN,
                label_correct BOOLEAN,
                PRIMARY KEY (run_id, answer_id, start_char, end_char),
                FOREIGN KEY (run_id, answer_id)
                    REFERENCES answer_results (run_id, answer_id)
                    ON DELETE CASCADE
            );
        """)
    yield
    # Shutdown
    if _pool:
        await _pool.close()


app = FastAPI(title="Kasparro Claim Extractor", version="1.0.0", lifespan=lifespan)



# --------------------------------------------------------------------------- #
#  Pydantic models                                                             #
# --------------------------------------------------------------------------- #
class ClaimIn(BaseModel):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str]
    reason: str
    match_gold: bool | None = None
    label_correct: bool | None = None


class AnswerResultIn(BaseModel):
    answer_id: str
    split: str
    status: str
    error: str | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_turns: int = 0
    trace_id: str | None = None
    per_answer_metrics: dict | None = None
    claims: list[ClaimIn] = []


class RunIn(BaseModel):
    run_id: str
    version: str
    split: str
    model: str
    timestamp: str
    dataset_version: str | None = None
    fact_version: str | None = None
    evaluator_version: str | None = None
    prompt_version: str | None = None
    file_hashes: dict | None = None
    metrics: dict | None = None
    settings: dict | None = None
    error_count: int = 0
    wall_time_seconds: float = 0.0
    answers: list[AnswerResultIn] = []


# --------------------------------------------------------------------------- #
#  Routes                                                                      #
# --------------------------------------------------------------------------- #
@app.post("/runs", status_code=201)
async def create_run(body: RunIn) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id, version, split, model, timestamp,
                    dataset_version, fact_version, evaluator_version, prompt_version,
                    file_hashes, metrics, settings, error_count, wall_time_seconds
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT (run_id) DO NOTHING
                """,
                body.run_id, body.version, body.split, body.model, body.timestamp,
                body.dataset_version, body.fact_version, body.evaluator_version,
                body.prompt_version,
                json.dumps(body.file_hashes) if body.file_hashes else None,
                json.dumps(body.metrics) if body.metrics else None,
                json.dumps(body.settings) if body.settings else None,
                body.error_count, body.wall_time_seconds,
            )

            for ans in body.answers:
                await conn.execute(
                    """
                    INSERT INTO answer_results (
                        run_id, answer_id, split, status, error,
                        latency_seconds, prompt_tokens, completion_tokens,
                        model_turns, trace_id, per_answer_metrics
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    ON CONFLICT (run_id, answer_id) DO NOTHING
                    """,
                    body.run_id, ans.answer_id, ans.split, ans.status, ans.error,
                    ans.latency_seconds, ans.prompt_tokens, ans.completion_tokens,
                    ans.model_turns, ans.trace_id,
                    json.dumps(ans.per_answer_metrics) if ans.per_answer_metrics else None,
                )

                for c in ans.claims:
                    await conn.execute(
                        """
                        INSERT INTO claim_results (
                            run_id, answer_id, claim_id, start_char, end_char,
                            quote, label, fact_ids, reason, match_gold, label_correct
                        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                        ON CONFLICT (run_id, answer_id, start_char, end_char) DO NOTHING
                        """,
                        body.run_id, ans.answer_id, c.claim_id,
                        c.start_char, c.end_char, c.quote, c.label,
                        json.dumps(c.fact_ids), c.reason,
                        c.match_gold, c.label_correct,
                    )

    return {"run_id": body.run_id, "status": "stored"}


@app.get("/runs")
async def list_runs() -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT run_id, version, split, model, timestamp, error_count, "
            "wall_time_seconds FROM runs ORDER BY timestamp DESC"
        )
    return [dict(r) for r in rows]


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        answers = await conn.fetch(
            "SELECT * FROM answer_results WHERE run_id = $1 ORDER BY answer_id",
            run_id,
        )
        result: dict[str, Any] = dict(row)
        result["answers"] = [dict(a) for a in answers]
    return result


@app.get("/runs/{run_id}/answers/{answer_id}")
async def get_answer(run_id: str, answer_id: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM answer_results WHERE run_id=$1 AND answer_id=$2",
            run_id, answer_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Answer not found")
        claims = await conn.fetch(
            "SELECT * FROM claim_results "
            "WHERE run_id=$1 AND answer_id=$2 ORDER BY start_char",
            run_id, answer_id,
        )
        result = dict(row)
        result["claims"] = [dict(c) for c in claims]
    return result
