# Kasparro AI – FSE I Assessment

Extract and classify factual claims in long AI speaker-product answers,
comparing a v1 baseline to a v2 improvement.

---

## Disclosure

Parts of the scaffolding were built with AI assistance (Bob / Claude).
All logic – prompts, harness scoring, validation rules, test cases – was
written, reviewed and verified by me.

---

## Project structure

```
kasparro-ai/
├── agent/
│   ├── executor.py    # async agent loop, tool dispatch, output validation
│   ├── facts.py       # get_brand_facts tool + catalogue loader
│   ├── prompts.py     # SYSTEM_V1 and SYSTEM_V2 prompt templates
│   └── schema.py      # ClaimPrediction, AnswerResult dataclasses
├── api/
│   └── app.py         # FastAPI routes + PostgreSQL persistence
├── harness/
│   └── __init__.py    # scorer (P/R/F1, confusion matrix, comparison)
├── tests/
│   └── test_all.py    # all controlled tests (fake provider, scorer, API, CSV)
├── data/
│   ├── answers.csv
│   ├── brand_facts.csv
│   └── development_gold.csv
├── runs/              # saved run JSON reports (git-ignored)
├── run.py             # CLI runner (async, semaphore=3, retry)
├── upload_run.py      # upload run JSON to API
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, DATABASE_URL, and optionally LANGFUSE_*
```

Load them before running:

```bash
set -a && source .env && set +a
```

### 3. PostgreSQL

Start a local PostgreSQL instance and create the database:

```bash
createdb kasparro
# Set DATABASE_URL=postgresql://user:password@localhost:5432/kasparro in .env
```

The FastAPI app creates all tables on startup.

---

## Running the four required live runs

```bash
# v1 development (baseline – 40 answers)
python run.py --version v1 --split development

# v2 development (improved – same 40 answers)
python run.py --version v2 --split development --compare

# v1 check (10 final answers – no gold quality scores)
python run.py --version v1 --split check

# v2 check (10 final answers)
python run.py --version v2 --split check
```

Reports are saved to `runs/` as JSON files named
`<version>_<split>_<timestamp>.json`.

---

## Uploading run reports to the API

Start the API server:

```bash
uvicorn api.app:app --reload
```

Upload a run:

```bash
python upload_run.py runs/v1_development_<timestamp>.json
```

---

## Running tests

```bash
pytest -v
```

All tests use fakes/mocks; no live API key or database is required.

---

## v1 → v2 change (targeted improvement)

**What changed:** v2 adds explicit extraction and span rules plus a
self-check step before the model emits JSON.

**Why:** Inspection of v1 Langfuse traces revealed two failure patterns:
1. The model sometimes split a single compound sentence into two fragments,
   causing span mismatches against gold.
2. The model occasionally merged adjacent sentences into one prediction,
   also failing exact-span matching.

**v2 adds:**
- Explicit "complete sentence" boundary instructions (no splits, no merges).
- Explicit span offset rules reminding the model that `quote ==
  ai_answer[start_char:end_char]`.
- A structured self-check list before the model finalises output.

**Measured effect:** See `runs/comparison_v1_vs_v2_<timestamp>.json`.
Joint F1 delta is reported there. A zero gain or regression is honestly
preserved.

---

## Dataset and file hashes

| File | MD5 |
|------|-----|
| answers.csv | `10113e1ca8ed52a506901dd5a7bde9ce` |
| brand_facts.csv | `285a7bd22e01f475e67f6e04fcfa53e1` |
| development_gold.csv | `fd5f7f4367630e895e48491b689df73e` |

Dataset version: `long-brand-answers-50-v4`
Fact version: `speaker-catalogue-v4`
Evaluator version: `sentence-spans-and-labels-v4`

---

## Limitations and unfinished work

- React front-end is not implemented (not assessed in R1).
- Langfuse local dataset linking is wired but native experiments are not created.
- No production auth, durable recovery or request idempotency (not required).
- The `--compare` flag reconstructs v1 Metrics from saved JSON; per-answer
  comparison detail requires both runs to be in memory simultaneously.
