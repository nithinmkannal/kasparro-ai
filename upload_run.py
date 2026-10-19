"""
Script to upload a saved run JSON report to the FastAPI + PostgreSQL service.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import httpx

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


def upload_run(report_path: str) -> None:
    path = pathlib.Path(report_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"Uploading run_id={report['run_id']} split={report['split']} version={report['version']}…")

    resp = httpx.post(f"{API_BASE}/runs", json=report, timeout=30)
    resp.raise_for_status()
    print(f"Stored: {resp.json()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_run.py <path/to/run.json>")
        sys.exit(1)
    upload_run(sys.argv[1])
