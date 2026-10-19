"""
Evaluation harness.

Implements:
- Deterministic span matching (exact answer_id + start_char + end_char)
- Extraction P/R/F1
- Joint (extraction + label) P/R/F1
- Per-label P/R/F1 and macro-F1
- Confusion matrix
- Per-answer missed / extra / wrong-label IDs
- Run comparison report
"""
from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from agent.schema import AnswerResult, ClaimPrediction

_GOLD_PATH = pathlib.Path(__file__).parent.parent / "data" / "development_gold.csv"
_LABELS = ("supported", "contradicted", "unknown")


# --------------------------------------------------------------------------- #
#  Gold loading                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: list[str]
    explanation: str
    difficulty_tags: str


def load_gold() -> dict[str, list[GoldClaim]]:
    """Returns gold claims keyed by answer_id (development only)."""
    gold: dict[str, list[GoldClaim]] = defaultdict(list)
    with open(_GOLD_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gold[row["answer_id"]].append(
                GoldClaim(
                    answer_id=row["answer_id"],
                    claim_id=row["claim_id"],
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    quote=row["quote"],
                    expected_label=row["expected_label"],
                    evidence_fact_ids=(
                        row["evidence_fact_ids"].split("|")
                        if row["evidence_fact_ids"]
                        else []
                    ),
                    explanation=row["explanation"],
                    difficulty_tags=row["difficulty_tags"],
                )
            )
    return dict(gold)


# --------------------------------------------------------------------------- #
#  Metrics dataclasses                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    G: int = 0   # gold claims
    P: int = 0   # accepted predictions
    M: int = 0   # span-matched
    C: int = 0   # span-matched AND label correct

    extraction_precision: float | None = None
    extraction_recall:    float | None = None
    extraction_f1:        float | None = None
    joint_precision:      float | None = None
    joint_recall:         float | None = None
    joint_f1:             float | None = None

    per_label: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_f1:  float | None = None

    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class AnswerMetrics:
    answer_id: str
    gold_count: int
    pred_count: int
    match_count: int
    correct_count: int
    missed_ids: list[str]
    extra_ids: list[str]
    wrong_label_ids: list[str]
    extraction_f1: float | None
    joint_f1: float | None


# --------------------------------------------------------------------------- #
#  Core scorer                                                                 #
# --------------------------------------------------------------------------- #
def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_run(
    results: list[AnswerResult],
    gold_by_answer: dict[str, list[GoldClaim]],
    split: str = "development",
) -> tuple[Metrics, list[AnswerMetrics]]:
    """
    Score a full run against gold.  Returns aggregate Metrics and per-answer details.
    Only development answers have gold; check answers get null quality metrics.
    """
    G_total = 0
    P_total = 0
    M_total = 0
    C_total = 0

    # Per-label counters: TP, FP, FN
    lbl_tp: dict[str, int] = {l: 0 for l in _LABELS}
    lbl_fp: dict[str, int] = {l: 0 for l in _LABELS}
    lbl_fn: dict[str, int] = {l: 0 for l in _LABELS}

    # Confusion matrix: rows=expected (supported/contradicted/unknown/no_gold)
    #                   cols=predicted (supported/contradicted/unknown/missing_or_error)
    cm_rows = list(_LABELS) + ["no_gold"]
    cm_cols = list(_LABELS) + ["missing_or_error"]
    cm: dict[str, dict[str, int]] = {r: {c: 0 for c in cm_cols} for r in cm_rows}

    per_answer: list[AnswerMetrics] = []

    # Collect all scheduled answer_ids (both errors and successes)
    scheduled_ids = {r.answer_id for r in results}

    for result in sorted(results, key=lambda r: r.answer_id):
        aid = result.answer_id
        gold_claims = gold_by_answer.get(aid, [])

        G = len(gold_claims)
        G_total += G

        if result.status == "error":
            # Error answer: 0 accepted predictions, all gold claims are missed
            P = 0
            M = 0
            C = 0
            missed_ids = [g.claim_id for g in gold_claims]
            extra_ids: list[str] = []
            wrong_label_ids: list[str] = []

            for g in gold_claims:
                lbl_fn[g.expected_label] += 1
                cm[g.expected_label]["missing_or_error"] += 1
        else:
            preds = result.claims
            P = len(preds)

            # Build lookup: (start_char, end_char) -> prediction
            pred_map = {(p.start_char, p.end_char): p for p in preds}
            gold_map = {(g.start_char, g.end_char): g for g in gold_claims}

            M = 0
            C = 0
            missed_ids = []
            extra_ids = []
            wrong_label_ids = []

            matched_gold_keys: set[tuple[int, int]] = set()

            for g in gold_claims:
                key = (g.start_char, g.end_char)
                if key in pred_map:
                    pred = pred_map[key]
                    M += 1
                    matched_gold_keys.add(key)
                    if pred.label == g.expected_label:
                        C += 1
                        lbl_tp[g.expected_label] += 1
                        cm[g.expected_label][g.expected_label] += 1
                    else:
                        wrong_label_ids.append(g.claim_id)
                        lbl_fp[pred.label] += 1
                        lbl_fn[g.expected_label] += 1
                        cm[g.expected_label][pred.label] += 1
                else:
                    missed_ids.append(g.claim_id)
                    lbl_fn[g.expected_label] += 1
                    cm[g.expected_label]["missing_or_error"] += 1

            # Unmatched predictions → extra
            for p in preds:
                key = (p.start_char, p.end_char)
                if key not in {(g.start_char, g.end_char) for g in gold_claims}:
                    extra_ids.append(p.claim_id)
                    lbl_fp[p.label] += 1
                    cm["no_gold"][p.label] += 1

            P_total += P
            M_total += M
            C_total += C

        # Per-answer metrics
        def _ans_f1(m_: int, g_: int, p_: int) -> float | None:
            if g_ == 0 and p_ == 0:
                return None
            prec = m_ / p_ if p_ > 0 else 0.0
            rec  = m_ / g_ if g_ > 0 else 0.0
            return _f1(prec, rec)

        per_answer.append(AnswerMetrics(
            answer_id=aid,
            gold_count=G,
            pred_count=P if result.status != "error" else 0,
            match_count=M,
            correct_count=C,
            missed_ids=missed_ids,
            extra_ids=extra_ids,
            wrong_label_ids=wrong_label_ids,
            extraction_f1=_ans_f1(M, G, P if result.status != "error" else 0),
            joint_f1=_ans_f1(C, G, P if result.status != "error" else 0),
        ))

    # Aggregate extraction metrics
    metrics = Metrics(G=G_total, P=P_total, M=M_total, C=C_total)

    if G_total > 0 or P_total > 0:
        ep = M_total / P_total if P_total > 0 else 0.0
        er = M_total / G_total if G_total > 0 else 0.0
        metrics.extraction_precision = ep
        metrics.extraction_recall    = er
        metrics.extraction_f1        = _f1(ep, er)

        jp = C_total / P_total if P_total > 0 else 0.0
        jr = C_total / G_total if G_total > 0 else 0.0
        metrics.joint_precision = jp
        metrics.joint_recall    = jr
        metrics.joint_f1        = _f1(jp, jr)

    # Per-label metrics
    per_label: dict[str, dict[str, float]] = {}
    macro_f1_sum = 0.0
    for lbl in _LABELS:
        tp = lbl_tp[lbl]
        fp = lbl_fp[lbl]
        fn = lbl_fn[lbl]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = _f1(prec, rec)
        per_label[lbl] = {"precision": prec, "recall": rec, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn}
        macro_f1_sum += f1
    metrics.per_label = per_label
    metrics.macro_f1  = macro_f1_sum / len(_LABELS) if _LABELS else None
    metrics.confusion_matrix = cm

    return metrics, per_answer


# --------------------------------------------------------------------------- #
#  Run comparison                                                              #
# --------------------------------------------------------------------------- #
def compare_runs(
    v1_metrics: Metrics,
    v2_metrics: Metrics,
    v1_per_answer: list[AnswerMetrics],
    v2_per_answer: list[AnswerMetrics],
) -> dict[str, Any]:
    """Return a comparison dict highlighting improvements and regressions."""
    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return round(b - a, 4)

    v1a = {m.answer_id: m for m in v1_per_answer}
    v2a = {m.answer_id: m for m in v2_per_answer}

    regressions = []
    improvements = []
    for aid in sorted(v1a):
        d = _delta(v1a[aid].joint_f1, v2a[aid].joint_f1 if aid in v2a else None)
        if d is not None and d < 0:
            regressions.append({"answer_id": aid, "joint_f1_delta": d})
        elif d is not None and d > 0:
            improvements.append({"answer_id": aid, "joint_f1_delta": d})

    return {
        "v1": {
            "extraction_f1": v1_metrics.extraction_f1,
            "joint_f1": v1_metrics.joint_f1,
            "macro_f1": v1_metrics.macro_f1,
            "G": v1_metrics.G, "P": v1_metrics.P,
            "M": v1_metrics.M, "C": v1_metrics.C,
        },
        "v2": {
            "extraction_f1": v2_metrics.extraction_f1,
            "joint_f1": v2_metrics.joint_f1,
            "macro_f1": v2_metrics.macro_f1,
            "G": v2_metrics.G, "P": v2_metrics.P,
            "M": v2_metrics.M, "C": v2_metrics.C,
        },
        "deltas": {
            "extraction_f1": _delta(v1_metrics.extraction_f1, v2_metrics.extraction_f1),
            "joint_f1":      _delta(v1_metrics.joint_f1,      v2_metrics.joint_f1),
            "macro_f1":      _delta(v1_metrics.macro_f1,      v2_metrics.macro_f1),
        },
        "answer_improvements": improvements,
        "answer_regressions": regressions,
    }
