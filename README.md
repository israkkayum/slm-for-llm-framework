# SLM for LLM Framework (Hallucination Firewall)

A lightweight, layered framework to reduce LLM hallucinations by combining:

- **Retrieval grounding** (FAISS + sentence-transformer embeddings)
- **SLM-based fact verification** (Yes/No probability scoring)
- **Optional neuro-symbolic reasoning** for post-verification correction/rejection
- **Evaluation and plotting** utilities for analysis-ready outputs

The repository includes end-to-end scripts for indexing documents, running the pipeline, generating experiment runs, and producing publication-style figures.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Run the API](#run-the-api)
- [Run Experiments and Generate Plots](#run-experiments-and-generate-plots)
- [Outputs](#outputs)
- [Current Dataset in Repo](#current-dataset-in-repo)
- [Troubleshooting](#troubleshooting)

---

## Overview

The framework is designed to answer user queries with grounded evidence and then verify factual claims before trusting the final response.

Core goals:

1. Retrieve relevant evidence from local documents.
2. Generate an answer constrained by retrieved context.
3. Decompose the answer into atomic claims.
4. Verify each claim with a small language model (SLM).
5. Optionally reason over claim/context pairs for correction or rejection.
6. Log outputs for reproducible evaluation.

---

## How It Works

### Layer 1 — Retrieval

- Documents are loaded from `data/raw/**` (`.pdf`, `.txt`, `.md`)
- Text is chunked and embedded
- Embeddings are indexed with FAISS
- Query-time search returns top-k candidate chunks

Main files:
- `scripts/build_index.py`
- `src/retrieval/loaders.py`
- `src/retrieval/chunker.py`
- `src/retrieval/embedder.py`
- `src/retrieval/index_builder.py`
- `src/retrieval/vector_store.py`

### Layer 2 — SLM Gate (Claim Verification)

Each atomic claim is checked against retrieved context using a strict YES/NO prompt:

- Computes `p_yes`, `p_no`, `sim_max`, and `uncertainty`
- Produces one label:
  - `SUPPORTED`
  - `CONTRADICTED`
  - `UNDER_SPECIFIED`

Main files:
- `src/slm_gate/slm_loader.py`
- `src/slm_gate/gate.py`

### Layer 3 — Neuro-Symbolic Reasoning (Optional)

- Applies simple rule signals
- Uses LLM reasoning over claim + context + rule output
- Returns structured verdict JSON as a string

Main files:
- `src/reasoning/rules.py`
- `src/reasoning/neurosymbolic_reasoner.py`

### Orchestration

The end-to-end pipeline is implemented in:

- `scripts/run_demo.py` (pipeline entrypoint)
- `scripts/app_cli.py` (interactive CLI)
- `src/app/api.py` (FastAPI wrapper)

---

## Repository Structure

```text
slm-for-llm-framework/
├── data/
│   ├── raw/                  # Input documents
│   └── index/                # Generated FAISS index + metadata
├── outputs/
│   ├── runs/                 # JSON logs per run
│   └── figures/              # Generated plots + CSV summaries
├── scripts/
│   ├── build_index.py
│   ├── run_demo.py
│   ├── app_cli.py
│   ├── run_experiments.py
│   └── make_plots.py
├── src/
│   ├── app/
│   ├── llm/
│   ├── reasoning/
│   ├── retrieval/
│   ├── slm_gate/
│   └── evaluation/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python **3.10+**
- Enough RAM/CPU (or CUDA GPU) for:
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - SLM gate model: `Qwen/Qwen2.5-1.5B-Instruct`
- OpenRouter-compatible API key for generation/reasoning calls

---

## Installation

```bash
cd /home/runner/work/slm-for-llm-framework/slm-for-llm-framework
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional but recommended for full functionality:

```bash
pip install pypdf seaborn
```

---

## Configuration

Create a `.env` file in repository root:

```env
OPENAI_API_KEY=your_openrouter_or_openai_compatible_key
OPENAI_MODEL=openai/gpt-4o-mini
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
SLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
INDEX_DIR=data/index
TOP_K=5
```

Config is loaded from `src/config.py`.

---

## Quick Start

### 1) Build retrieval index

```bash
python scripts/build_index.py
```

This scans `data/raw/`, chunks documents, embeds them, and writes:

- `data/index/faiss.index`
- `data/index/meta.json`

### 2) Run one pipeline example

```bash
python scripts/run_demo.py
```

It generates a run log in `outputs/runs/run_*.json`.

### 3) Use interactive CLI

```bash
python scripts/app_cli.py
```

Type queries and inspect:
- final grounded answer
- trust score
- supported / under-specified / contradicted fact lists

---

## Run the API

Start server:

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

Request example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the working day?"}'
```

---

## Run Experiments and Generate Plots

### Generate benchmark runs

```bash
python scripts/run_experiments.py --n_runs 30 --facts_per_run 8 --k 5 --seed 42
```

Useful flags:
- `--domain university|company|medical`
- `--variant full` (for ablation tracking)
- `--include_reasoner` (if reasoner branch is enabled)

### Build figures and CSV summaries

```bash
python scripts/make_plots.py
```

This produces plots such as:
- predicted label distributions
- confusion matrix
- ROC / PR curves
- threshold sensitivity
- calibration curve
- latency breakdown
- ablation macro-F1

---

## Outputs

### Run Logs

`outputs/runs/run_*.json` contains:

- `query`
- `variant`
- `timings`
- `llm_answer`
- `atomic_facts`
- `per_fact[]` with:
  - `fact`
  - `gate_label`
  - `p_yes`, `p_no`, `sim_max`, `uncertainty`
  - `retrieved`
  - `reasoning`
  - `gold_label` (if available)

### Figures and Metrics

`outputs/figures/` includes:
- PNG figures
- `summary_metrics.csv`
- `classification_report.csv`

---

## Current Dataset in Repo

The included sample corpus is currently under:

- `data/raw/university/class_routine.pdf`

You can add more documents in domain folders (for example `data/raw/company/`, `data/raw/medical/`) and rebuild the index.

---

## Troubleshooting

- **`meta.json not found`**  
  Run `python scripts/build_index.py` first.

- **Slow or high-memory verification**  
  Reduce retrieval breadth (`k`) and/or context size in gate settings.

- **Model download/startup delays**  
  First run may take time due to Hugging Face model downloads.

- **Empty/limited plots**  
  Some plots require `gold_label`, `variant`, or timing fields in run logs.

---
