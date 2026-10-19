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
# Edit .env and fill in your API key for the chosen provider:
# - GEMINI_API_KEY (for --provider gemini)
# - GROQ_API_KEY (for --provider groq)
# - OPENAI_API_KEY (for --provider openai)
# Optional: DATABASE_URL, LANGFUSE_*
```

Environment variables are loaded automatically from `.env` via `python-dotenv`.

### 3. PostgreSQL

Start a local PostgreSQL instance and create the database:

```bash
createdb kasparro
# Set DATABASE_URL=postgresql://user:password@localhost:5432/kasparro in .env
```

The FastAPI app creates all tables on startup.

---

## Running the live runs

Run with your chosen provider (`gemini`, `groq`, or `openai`):

```bash
# v1 development (baseline – 40 answers)
python run.py --provider gemini --version v1 --split development

# v2 development (improved – same 40 answers)
python run.py --provider gemini --version v2 --split development --compare

# v1 check (10 final answers – no gold quality scores)
python run.py --provider gemini --version v1 --split check

# v2 check (10 final answers)
python run.py --provider gemini --version v2 --split check
```

Supported providers and default models:
- `--provider gemini` (default model: `gemini-3.5-flash-lite`)
- `--provider groq` (default model: `qwen/qwen3.8-27b`)
- `--provider openai` (default model: `gpt-4o`)

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

## Benchmark & Experimental Results (Development Split)

Tested live with **Google Gemini (`gemini-3.5-flash-lite`)** across all 40 development answers (376 reference gold claims):

| Metric | Baseline (v1) | Improved (v2) | Delta ($\Delta$) |
|---|---|---|---|
| **Gold Claims ($G$)** | 376 | 376 | — |
| **Predicted Claims ($P$)** | 308 | 356 | +48 |
| **Exact Span Matches ($M$)** | 307 | 346 | +39 |
| **Correct Label Matches ($C$)** | 298 | 342 | +44 |
| **Extraction Precision** | 99.68% | 97.19% | -2.49% |
| **Extraction Recall** | 81.65% | 92.02% | **+10.37%** |
| **Extraction F1** | 89.77% | **94.54%** | **+4.77%** |
| **Joint Precision** | 96.75% | 96.07% | -0.68% |
| **Joint Recall** | 79.26% | 90.96% | **+11.70%** |
| **Joint F1 (Primary Metric)** | 87.13% | **93.44%** | **+6.31%** |
| **Macro F1** | 86.68% | **93.54%** | **+6.86%** |

### Per-Label Performance Breakdown (v2):
- **Supported**: Precision 96.52% | Recall 92.50% | **F1: 94.47%**
- **Contradicted**: Precision 98.21% | Recall 91.67% | **F1: 94.83%**
- **Unknown**: Precision 93.80% | Recall 88.97% | **F1: 91.32%**

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
