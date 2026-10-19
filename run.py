"""
Async CLI runner.

Usage:
    python run.py --version v1 --split development
    python run.py --version v2 --split development
    python run.py --version v1 --split check
    python run.py --version v2 --split check

Produces a JSON run report in runs/<version>_<split>_<timestamp>.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
import openai

load_dotenv()

from agent.executor import attach_langfuse_scores, flush_traces, run_agent
from agent.schema import AnswerResult
from harness import load_gold, score_run, compare_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path("data")
RUNS_DIR = pathlib.Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# Default concurrency cap; Groq free tier (8000 TPM) needs 1 to stay under the limit.
# Override with --concurrency if needed.
MAX_CONCURRENT = 3


def _file_hash(path: pathlib.Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_revision() -> str | None:
    """Return the current HEAD commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_answers(split: str) -> list[dict]:
    rows = []
    with open(DATA_DIR / "answers.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                rows.append(row)
    return rows


async def run_all(
    answers: list[dict],
    version: str,
    split: str,
    run_id: str,
    model: str,
    client: openai.AsyncOpenAI | None = None,
    provider: str = "openai",
    concurrency: int = MAX_CONCURRENT,
) -> list[AnswerResult]:
    if client is None:
        from agent.executor import _make_client
        client = _make_client(provider)
    semaphore = asyncio.Semaphore(concurrency)

    # For Groq free tier (8000 TPM) and Gemini free tier (15 RPM) stagger task creation to avoid burst
    inter_delay = 12.0 if (provider == "groq" and concurrency == 1) else (
        5.0 if (provider == "gemini" and concurrency == 1) else 0.0
    )

    async def _run_one(row: dict, idx: int) -> AnswerResult:
        if inter_delay:
            await asyncio.sleep(idx * inter_delay)
        async with semaphore:
            return await run_agent(
                answer_id=row["answer_id"],
                brand_id=row["brand_id"],
                question=row["question"],
                ai_answer=row["ai_answer"],
                split=split,
                version=version,
                run_id=run_id,
                model=model,
                client=client,
                provider=provider,
            )

    tasks = [asyncio.create_task(_run_one(row, i)) for i, row in enumerate(answers)]
    results = await asyncio.gather(*tasks)
    flush_traces()
    return sorted(results, key=lambda r: r.answer_id)


def _result_to_dict(r: AnswerResult) -> dict:
    return {
        "answer_id": r.answer_id,
        "version": r.version,
        "split": r.split,
        "status": r.status,
        "error": r.error,
        "latency_seconds": round(r.latency_seconds, 3),
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "model_turns": r.model_turns,
        "trace_id": r.trace_id,
        "claims": [
            {
                "claim_id": c.claim_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "quote": c.quote,
                "label": c.label,
                "fact_ids": c.fact_ids,
                "reason": c.reason,
            }
            for c in r.claims
        ],
    }


def _save_report(path: pathlib.Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved run report → %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",     required=True, choices=["v1", "v2"])
    parser.add_argument("--split",       required=True, choices=["development", "check"])
    parser.add_argument("--model",       default=None,
                        help="Model name. Defaults: openai→gpt-4o, groq→qwen/qwen3.8-27b")
    parser.add_argument("--provider",    default="openai", choices=["openai", "groq", "gemini"],
                        help="Inference provider (default: openai)")
    parser.add_argument("--concurrency", default=None, type=int,
                        help="Max concurrent requests (default: 3 openai/gemini, 1 groq)")
    parser.add_argument("--compare",     action="store_true",
                        help="Also load the other version's run and print comparison")
    args = parser.parse_args()

    # Defaults per provider
    _default_models = {
        "openai": "gpt-4o",
        "groq":   "qwen/qwen3.8-27b",
        "gemini": "gemini-3.5-flash-lite",
    }
    if args.model is None:
        args.model = _default_models.get(args.provider, "gpt-4o")
    # Free tiers (Groq 8000 ITPM, Gemini 15 RPM) — run serially to avoid rate limits
    concurrency = args.concurrency if args.concurrency is not None else (
        1 if args.provider in ("groq", "gemini") else MAX_CONCURRENT
    )

    run_id     = str(uuid.uuid4())
    answers    = _load_answers(args.split)
    source_rev = _git_revision()

    logger.info("Loaded %d answers for split=%s version=%s provider=%s model=%s concurrency=%d",
                len(answers), args.split, args.version, args.provider, args.model, concurrency)

    # ------------------------------------------------------------------ #
    #  Settings captured for this run                                      #
    # ------------------------------------------------------------------ #
    settings = {
        "provider":         args.provider,
        "model":            args.model,
        "temperature":      0.0,
        "timeout_seconds":  15,
        "max_retries":      1,
        "max_concurrent":   concurrency,
        "max_turns":        3,
    }

    # ------------------------------------------------------------------ #
    #  Build a partial report before running so we can save on interrupt   #
    # ------------------------------------------------------------------ #
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RUNS_DIR / f"{args.version}_{args.split}_{ts}.json"

    partial_report: dict = {
        "run_id":             run_id,
        "run_status":         "running",      # will be updated to completed/failed
        "version":            args.version,
        "split":              args.split,
        "model":              args.model,
        "provider":           args.provider,
        "timestamp":          ts,
        "source_revision":    source_rev,
        "agent_version":      args.version,
        "prompt_version":     args.version,
        "dataset_version":    "long-brand-answers-50-v4",
        "evaluator_version":  "sentence-spans-and-labels-v4",
        "fact_version":       "speaker-catalogue-v4",
        "settings":           settings,
        "file_hashes": {
            "answers.csv":          _file_hash(DATA_DIR / "answers.csv"),
            "brand_facts.csv":      _file_hash(DATA_DIR / "brand_facts.csv"),
            "development_gold.csv": _file_hash(DATA_DIR / "development_gold.csv"),
        },
        "hash_algorithm":     "sha256",
        "wall_time_seconds":  None,
        "error_count":        None,
        "metrics":            None,
        "per_answer":         None,
        "answers":            [],
    }
    _save_report(report_path, partial_report)   # preserve even if interrupted

    t0 = time.monotonic()
    run_failed = False
    results: list[AnswerResult] = []

    try:
        results = asyncio.run(
            run_all(answers, args.version, args.split, run_id, args.model,
                    provider=args.provider, concurrency=concurrency)
        )
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user – partial evidence preserved in %s", report_path)
        run_failed = True
    except Exception as exc:
        logger.error("Run failed with exception: %s", exc)
        run_failed = True

    wall_time = time.monotonic() - t0

    # ------------------------------------------------------------------ #
    #  Quality metrics (development only)                                 #
    # ------------------------------------------------------------------ #
    metrics_dict: dict | None = None
    per_answer_list: list | None = None
    # trace_id lookup for Langfuse score attachment
    trace_by_aid: dict[str, str | None] = {r.answer_id: r.trace_id for r in results}

    if results and args.split == "development":
        gold = load_gold()
        metrics, per_answer = score_run(
            results, gold, split=args.split,
            scheduled_ids=[a["answer_id"] for a in answers],
        )
        metrics_dict = {
            "G": metrics.G, "P": metrics.P, "M": metrics.M, "C": metrics.C,
            "extraction_precision": metrics.extraction_precision,
            "extraction_recall":    metrics.extraction_recall,
            "extraction_f1":        metrics.extraction_f1,
            "joint_precision":      metrics.joint_precision,
            "joint_recall":         metrics.joint_recall,
            "joint_f1":             metrics.joint_f1,
            "macro_f1":             metrics.macro_f1,
            "per_label":            metrics.per_label,
            "confusion_matrix":     metrics.confusion_matrix,
        }

        per_answer_list = []
        for a in per_answer:
            # Determine label_correct per matched claim for Langfuse scores
            claim_correctness: list[dict] = []
            # Find the result for this answer to identify matched predictions
            result_for_a = next((r for r in results if r.answer_id == a.answer_id), None)
            if result_for_a and result_for_a.status == "ok":
                gold_map = {
                    (g.start_char, g.end_char): g
                    for g in gold.get(a.answer_id, [])
                }
                for pred in result_for_a.claims:
                    key = (pred.start_char, pred.end_char)
                    g = gold_map.get(key)
                    if g is not None and pred.quote == g.quote:
                        claim_correctness.append({
                            "claim_id":     pred.claim_id,
                            "label_correct": pred.label == g.expected_label,
                        })

            # Attach Langfuse scores for this answer
            attach_langfuse_scores(
                answer_id=a.answer_id,
                trace_id=trace_by_aid.get(a.answer_id),
                extraction_f1=a.extraction_f1,
                joint_f1=a.joint_f1,
                claim_correctness=claim_correctness,
                missing_ids=a.missed_ids,
                extra_ids=a.extra_ids,
            )

            per_answer_list.append({
                "answer_id":      a.answer_id,
                "gold_count":     a.gold_count,
                "pred_count":     a.pred_count,
                "match_count":    a.match_count,
                "correct_count":  a.correct_count,
                "missed_ids":     a.missed_ids,
                "extra_ids":      a.extra_ids,
                "wrong_label_ids": a.wrong_label_ids,
                "extraction_f1":  a.extraction_f1,
                "joint_f1":       a.joint_f1,
            })

        logger.info(
            "RESULTS  extraction_f1=%.3f  joint_f1=%.3f  macro_f1=%.3f  "
            "G=%d P=%d M=%d C=%d",
            metrics.extraction_f1 or 0, metrics.joint_f1 or 0, metrics.macro_f1 or 0,
            metrics.G, metrics.P, metrics.M, metrics.C,
        )

    # ------------------------------------------------------------------ #
    #  Save final run report (overwrite partial)                          #
    # ------------------------------------------------------------------ #
    run_status = "failed" if run_failed else "completed"
    report: dict = {
        **partial_report,
        "run_status":         run_status,
        "wall_time_seconds":  round(wall_time, 2),
        "error_count":        sum(1 for r in results if r.status == "error"),
        "metrics":            metrics_dict,
        "per_answer":         per_answer_list,
        "answers":            [_result_to_dict(r) for r in results],
    }
    _save_report(report_path, report)

    # ------------------------------------------------------------------ #
    #  Optional comparison                                                 #
    # ------------------------------------------------------------------ #
    if args.compare and args.split == "development" and not run_failed:
        other_version = "v2" if args.version == "v1" else "v1"
        existing = sorted(RUNS_DIR.glob(f"{other_version}_{args.split}_*.json"))
        if existing:
            other_report = json.loads(existing[-1].read_text(encoding="utf-8"))
            om = other_report.get("metrics", {})
            from harness import AnswerMetrics, Metrics
            other_metrics = Metrics(
                G=om["G"], P=om["P"], M=om["M"], C=om["C"],
                extraction_f1=om["extraction_f1"],
                joint_f1=om["joint_f1"],
                macro_f1=om["macro_f1"],
            )
            other_per_answer = [
                AnswerMetrics(
                    answer_id=a["answer_id"], gold_count=a["gold_count"],
                    pred_count=a["pred_count"], match_count=a["match_count"],
                    correct_count=a["correct_count"], missed_ids=a["missed_ids"],
                    extra_ids=a["extra_ids"], wrong_label_ids=a["wrong_label_ids"],
                    extraction_f1=a["extraction_f1"], joint_f1=a["joint_f1"],
                )
                for a in other_report.get("per_answer", [])
            ]
            # compare_runs(v1, v2): always pass baseline first, candidate second
            if args.version == "v2":
                comp = compare_runs(other_metrics, metrics, other_per_answer, per_answer)
            else:
                comp = compare_runs(metrics, other_metrics, per_answer, other_per_answer)
            comp_path = RUNS_DIR / f"comparison_v1_vs_v2_{ts}.json"
            comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
            logger.info("Comparison saved → %s", comp_path)
            logger.info(
                "Delta joint_f1 (v2 - v1): %.4f  (v1 %.4f → v2 %.4f)",
                comp["deltas"]["joint_f1"] or 0,
                comp["v1"]["joint_f1"] or 0,
                comp["v2"]["joint_f1"] or 0,
            )
        else:
            logger.warning("No existing %s run found for comparison.", other_version)


if __name__ == "__main__":
    main()
