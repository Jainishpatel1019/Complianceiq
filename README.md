---
title: ComplianceIQ
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ComplianceIQ

Automated regulatory intelligence for banking compliance teams. Detects changes in US federal banking regulations, quantifies their financial impact using causal inference, and surfaces the results through an interactive dashboard.

**Live demo:** [huggingface.co/spaces/Jainish1019/complianceiq](https://huggingface.co/spaces/Jainish1019/complianceiq) *(demo data: Jan–Jun 2024)*

---

## What it does

US regulators publish ~40 rule changes per week across the Federal Register, FDIC, and CFPB. ComplianceIQ ingests all of them daily and does three things:

1. **Detects meaningful change** — three complementary measures (semantic drift, JSD, Wasserstein distance) catch changes in meaning, vocabulary, and structure respectively. Anything above the 0.15 drift threshold triggers the agent pipeline.
2. **Quantifies financial impact** — Difference-in-Differences on FDIC call report data, Synthetic Control for system-wide regulations, and Basel III ΔRWA math with Monte Carlo uncertainty propagation.
3. **Reasons about impact** — a LangGraph agent runs eight tools (RAG, knowledge graph, causal estimation, Bayesian network scoring) and produces a structured report: *summary, impact score, ΔRWA median + 90% CI, affected business lines, citations, reasoning trace*.

---

## Stack

| Layer | Technology |
|---|---|
| Ingestion | Apache Airflow 2.9, Federal Register API, FDIC BankFind |
| Storage | PostgreSQL 16 + TimescaleDB, ChromaDB, pgvector |
| ML | PyTorch Geometric (GAT), pgmpy (Bayesian network), econml (causal), RAGAS (eval) |
| Agent | LangGraph 0.2, Ollama (mistral:7b, nomic-embed-text, llama3.2:3b) |
| API | FastAPI, WebSockets |
| Frontend | React 18, D3.js, Tailwind CSS |
| Tracking | MLflow |
| Infra | Docker Compose, Alembic, DVC |

All models run locally via Ollama. No OpenAI or other API keys required.

---

## Quickstart

```bash
git clone https://github.com/Jainish1019/complianceiq
cd complianceiq
make setup    # copies .env.example → .env, builds images, pulls Ollama models (~5 min)
make dev      # starts all 11 services
```

Services after `make dev`:

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8081/docs |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Flower | http://localhost:5555 |

```bash
make test     # runs the full test suite (178 tests)
make lint     # ruff + mypy
make seed-db  # load sample data without running ingestion DAGs
make down     # stop everything
```

---

## Architecture

```
Federal Register API ──┐
FDIC BankFind API    ──┤──→ Airflow DAGs ──→ PostgreSQL/TimescaleDB
CFPB API             ──┘         │
                                  ├──→ Embedding pipeline ──→ ChromaDB
                                  ├──→ Change detection   ──→ Drift scores
                                  ├──→ Causal inference   ──→ ΔRWA estimates
                                  └──→ LangGraph agent    ──→ Impact reports
                                                                    │
                                              FastAPI ──────────────┘
                                                 │
                                           React dashboard
```

Nine Airflow DAGs run on schedule:

| DAG | Schedule | Purpose |
|---|---|---|
| `ingest_sources` | Daily 02:00 UTC | Federal Register + FDIC fetch |
| `embed_and_index` | Daily 03:00 UTC | Chunk + embed + ChromaDB upsert |
| `change_detection` | Daily 04:00 UTC | Drift/JSD/Wasserstein scores |
| `causal_estimation` | Weekly | DiD + Synthetic Control updates |
| `impact_agent` | Daily 09:00 UTC | LangGraph agent run on flagged docs |
| `graph_update` | Daily 10:00 UTC | Knowledge graph rebuild + PageRank |
| `alert_dispatch` | Daily 11:00 UTC | Slack/email for P(High) ≥ 0.8 |
| `model_registry` | Weekly | MLflow model promotion |
| `evaluate_pipeline` | Weekly Sat | RAGAS + ablation + calibration |

---

## Evaluation

Run on a 300-document human-labelled test set + 500 RAG queries:

| Metric | Score |
|---|---|
| Change detection F1 (all three measures) | 0.84 |
| Change detection F1 (drift only, ablation) | 0.71 |
| RAGAS faithfulness | 0.82 |
| RAGAS context recall | 0.76 |
| Drift threshold (95% CI) | 0.15 [0.12, 0.18] |

---

## Hugging Face deployment

```bash
# Build pre-seeded demo data locally first (run after ingestion DAGs complete)
make seed-export   # exports pg_dump + ChromaDB snapshot to data/seed/

# Then deploy to HF Spaces (slimmed 5-service compose, no Airflow/MLflow/Redis)
docker compose -f docker-compose.hf.yml up
```

Set these in HF Space secrets: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `API_SECRET_KEY`.

---

## Project layout

```
complianceiq/
├── api/                    FastAPI app + route handlers
├── backend/
│   ├── agents/             LangGraph impact agent
│   ├── models/             Change detection, causal, Bayesian network, graph
│   └── pipelines/          Ingestion, embedding, evaluation
├── airflow/dags/           9 Airflow DAG files
├── db/                     SQLAlchemy models + Alembic migrations
├── frontend/src/           React dashboard
├── tests/                  178 unit tests (pytest)
├── docs/                   math_explainer.md, diagrams
├── data/seed/              Pre-seeded demo data for HF Space
├── docker-compose.yml      Full local dev (11 services)
├── docker-compose.hf.yml   HF Spaces slim deploy (5 services)
└── Makefile
```

---

## Technical write-up

See [`docs/math_explainer.md`](docs/math_explainer.md) for the full explanation of the change detection math, causal inference methods, Bayesian network architecture, and evaluation framework.
