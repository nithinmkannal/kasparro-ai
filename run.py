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
import time
import uuid
from datetime import datetime, timezone

import openai

from agent.executor import run_agent
from agent.schema import AnswerResult
from harness import load_gold, score_run, compare_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path("data")
RUNS_DIR = pathlib.Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT = 3   # semaphore: at most 3 active provider requests


def _file_hash(path: pathlib.Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


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
) -> list[AnswerResult]:
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _run_one(row: dict) -> AnswerResult:
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
            )

    tasks = [asyncio.create_task(_run_one(row)) for row in answers]
    results = await asyncio.gather(*tasks)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=["v1", "v2"])
    parser.add_argument("--split",   required=True, choices=["development", "check"])
    parser.add_argument("--model",   default="gpt-4o")
    parser.add_argument("--compare", action="store_true",
                        help="Also load the other version's run and print comparison")
    args = parser.parse_args()

    run_id = str(uuid.uuid4())
    answers = _load_answers(args.split)
    logger.info("Loaded %d answers for split=%s", len(answers), args.split)

    t0 = time.monotonic()
    results = asyncio.run(run_all(answers, args.version, args.split, run_id, args.model))
    wall_time = time.monotonic() - t0

    # ------------------------------------------------------------------ #
    #  Quality metrics (development only)                                 #
    # ------------------------------------------------------------------ #
    metrics_dict: dict | None = None
    per_answer_list: list | None = None

    if args.split == "development":
        gold = load_gold()
        metrics, per_answer = score_run(results, gold, split=args.split)
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
        per_answer_list = [
            {
                "answer_id": a.answer_id,
                "gold_count": a.gold_count,
                "pred_count": a.pred_count,
                "match_count": a.match_count,
                "correct_count": a.correct_count,
                "missed_ids": a.missed_ids,
                "extra_ids": a.extra_ids,
                "wrong_label_ids": a.wrong_label_ids,
                "extraction_f1": a.extraction_f1,
                "joint_f1": a.joint_f1,
            }
            for a in per_answer
        ]

        logger.info(
            "RESULTS  extraction_f1=%.3f  joint_f1=%.3f  macro_f1=%.3f  "
            "G=%d P=%d M=%d C=%d",
            metrics.extraction_f1 or 0, metrics.joint_f1 or 0, metrics.macro_f1 or 0,
            metrics.G, metrics.P, metrics.M, metrics.C,
        )

    # ------------------------------------------------------------------ #
    #  Save run report                                                     #
    # ------------------------------------------------------------------ #
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RUNS_DIR / f"{args.version}_{args.split}_{ts}.json"

    report = {
        "run_id": run_id,
        "version": args.version,
        "split": args.split,
        "model": args.model,
        "timestamp": ts,
        "wall_time_seconds": round(wall_time, 2),
        "agent_version": args.version,
        "prompt_version": args.version,
        "dataset_version": "long-brand-answers-50-v4",
        "evaluator_version": "sentence-spans-and-labels-v4",
        "fact_version": "speaker-catalogue-v4",
        "file_hashes": {
            "answers.csv":          _file_hash(DATA_DIR / "answers.csv"),
            "brand_facts.csv":      _file_hash(DATA_DIR / "brand_facts.csv"),
            "development_gold.csv": _file_hash(DATA_DIR / "development_gold.csv"),
        },
        "error_count": sum(1 for r in results if r.status == "error"),
        "metrics": metrics_dict,
        "per_answer": per_answer_list,
        "answers": [_result_to_dict(r) for r in results],
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved run report → %s", report_path)

    # ------------------------------------------------------------------ #
    #  Optional comparison                                                 #
    # ------------------------------------------------------------------ #
    if args.compare and args.split == "development":
        other_version = "v2" if args.version == "v1" else "v1"
        # Find the most recent run for other version
        existing = sorted(RUNS_DIR.glob(f"{other_version}_{args.split}_*.json"))
        if existing:
            other_report = json.loads(existing[-1].read_text(encoding="utf-8"))
            # Reconstruct metrics from saved data
            om = other_report.get("metrics", {})
            from harness import Metrics
            other_metrics = Metrics(
                G=om["G"], P=om["P"], M=om["M"], C=om["C"],
                extraction_f1=om["extraction_f1"],
                joint_f1=om["joint_f1"],
                macro_f1=om["macro_f1"],
            )
            comp = compare_runs(other_metrics, metrics, [], per_answer)
            comp_path = RUNS_DIR / f"comparison_{other_version}_vs_{args.version}_{ts}.json"
            comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
            logger.info("Comparison saved → %s", comp_path)
            logger.info(
                "Delta joint_f1: %.4f  (was %.4f → now %.4f)",
                comp["deltas"]["joint_f1"] or 0,
                comp[other_version]["joint_f1"] or 0,
                comp[args.version]["joint_f1"] or 0,
            )
        else:
            logger.warning("No existing %s run found for comparison.", other_version)


if __name__ == "__main__":
    main()
