# ComplianceIQ — Makefile
#
# docker-compose.yml      = HuggingFace Spaces (5 services, used by HF auto-build)
# docker-compose.dev.yml  = Full local dev (11 services: Airflow, MLflow, Flower, etc.)
#
# Usage:
#   make setup   →  first-time local setup
#   make dev     →  start all 11 local services
#   make test    →  run test suite
#   make down    →  stop local services

.PHONY: setup dev down test lint migrate seed-db pull-models seed-export hf-build hf-up hf-down

DEV_COMPOSE := docker compose -f docker-compose.dev.yml

# ── First-time setup ──────────────────────────────────────────────────────────
setup:
	@echo "→ Copying .env.example to .env (edit before running)"
	cp -n .env.example .env || true
	@echo "→ Building Docker images"
	$(DEV_COMPOSE) build
	@echo "→ Initialising Airflow DB"
	$(DEV_COMPOSE) run --rm airflow-webserver airflow db init
	@echo "→ Creating Airflow admin user"
	$(DEV_COMPOSE) run --rm airflow-webserver airflow users create \
		--username $${AIRFLOW_ADMIN_USER:-admin} \
		--password $${AIRFLOW_ADMIN_PASSWORD:-admin} \
		--firstname Admin --lastname User \
		--role Admin --email admin@complianceiq.local
	@echo "→ Running Alembic migrations"
	$(DEV_COMPOSE) run --rm api alembic upgrade head
	@echo "→ Pulling Ollama models (this takes a few minutes)"
	$(MAKE) pull-models
	@echo "✓ Setup complete. Run 'make dev' to start."

# ── Start all 11 local services ───────────────────────────────────────────────
dev:
	$(DEV_COMPOSE) up -d
	@echo "✓ Services started:"
	@echo "  Airflow UI    → http://localhost:8080"
	@echo "  FastAPI docs  → http://localhost:8081/docs"
	@echo "  Frontend      → http://localhost:3000"
	@echo "  MLflow        → http://localhost:5000"
	@echo "  Flower        → http://localhost:5555"

# ── Stop all local services ───────────────────────────────────────────────────
down:
	$(DEV_COMPOSE) down

# ── Pull Ollama models ────────────────────────────────────────────────────────
pull-models:
	$(DEV_COMPOSE) exec ollama ollama pull nomic-embed-text
	$(DEV_COMPOSE) exec ollama ollama pull mistral:7b
	$(DEV_COMPOSE) exec ollama ollama pull llama3.2:3b

# ── Run tests ─────────────────────────────────────────────────────────────────
test:
	$(DEV_COMPOSE) run --rm api pytest -v --tb=short

# ── Run linter + type checker ─────────────────────────────────────────────────
lint:
	$(DEV_COMPOSE) run --rm api ruff check .
	$(DEV_COMPOSE) run --rm api mypy backend/ api/

# ── Apply DB migrations ───────────────────────────────────────────────────────
migrate:
	$(DEV_COMPOSE) run --rm api alembic upgrade head

# ── Seed DB with sample data (for demo / HF Space) ───────────────────────────
seed-db:
	$(DEV_COMPOSE) run --rm api python -m backend.pipelines.seed

# ── Logs shortcut ─────────────────────────────────────────────────────────────
logs:
	$(DEV_COMPOSE) logs -f --tail=100

logs-api:
	$(DEV_COMPOSE) logs -f api --tail=100

logs-airflow:
	$(DEV_COMPOSE) logs -f airflow-webserver airflow-scheduler --tail=100

# ── Export seed data for HF Space ─────────────────────────────────────────────
# Run AFTER the ingestion DAGs have completed at least one full cycle.
# Exports pg_dump + ChromaDB snapshot → data/seed/
seed-export:
	@echo "→ Creating data/seed directories"
	mkdir -p data/seed/init data/seed/chromadb_snapshot
	@echo "→ Exporting PostgreSQL seed dump"
	$(DEV_COMPOSE) exec postgres pg_dump \
		-U $${POSTGRES_USER:-complianceiq} \
		-d $${POSTGRES_DB:-complianceiq} \
		--no-owner --no-acl \
		| gzip > data/seed/init/01_complianceiq_seed.sql.gz
	@echo "→ Copying ChromaDB snapshot"
	$(DEV_COMPOSE) cp chromadb:/chroma/chroma/. data/seed/chromadb_snapshot/
	@echo "✓ Seed export complete → data/seed/"

# ── HuggingFace Space (uses docker-compose.yml = slim 5-service) ──────────────
hf-build:
	@echo "→ Building React frontend"
	cd frontend && npm ci && npm run build
	@echo "→ Building HF Docker images"
	docker compose build

hf-up:
	docker compose up -d
	@echo "✓ HF demo running:"
	@echo "  API   → http://localhost:7860/docs"
	@echo "  UI    → http://localhost:3000"

hf-down:
	docker compose down
