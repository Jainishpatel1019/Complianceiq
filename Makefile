# ComplianceIQ — Makefile
# Usage: make setup  →  first-time setup
#        make dev    →  start all 11 services
#        make test   →  run test suite
#        make down   →  stop all services

.PHONY: setup dev down test lint migrate seed-db pull-models seed-export hf-build hf-up hf-down

# ── First-time setup ──────────────────────────────────────────────────────────
setup:
	@echo "→ Copying .env.example to .env (edit before running)"
	cp -n .env.example .env || true
	@echo "→ Building Docker images"
	docker compose build
	@echo "→ Initialising Airflow DB"
	docker compose run --rm airflow-webserver airflow db init
	@echo "→ Creating Airflow admin user"
	docker compose run --rm airflow-webserver airflow users create \
		--username $${AIRFLOW_ADMIN_USER:-admin} \
		--password $${AIRFLOW_ADMIN_PASSWORD:-admin} \
		--firstname Admin --lastname User \
		--role Admin --email admin@complianceiq.local
	@echo "→ Running Alembic migrations"
	docker compose run --rm api alembic upgrade head
	@echo "→ Pulling Ollama models (this takes a few minutes)"
	$(MAKE) pull-models
	@echo "✓ Setup complete. Run 'make dev' to start."

# ── Start all services ────────────────────────────────────────────────────────
dev:
	docker compose up -d
	@echo "✓ Services started:"
	@echo "  Airflow UI    → http://localhost:8080"
	@echo "  FastAPI docs  → http://localhost:8081/docs"
	@echo "  Frontend      → http://localhost:3000"
	@echo "  MLflow        → http://localhost:5000"
	@echo "  Flower        → http://localhost:5555"

# ── Stop all services ─────────────────────────────────────────────────────────
down:
	docker compose down

# ── Pull Ollama models ────────────────────────────────────────────────────────
pull-models:
	docker compose exec ollama ollama pull nomic-embed-text
	docker compose exec ollama ollama pull mistral:7b
	docker compose exec ollama ollama pull llama3.2:3b

# ── Run tests ─────────────────────────────────────────────────────────────────
test:
	docker compose run --rm api pytest -v --tb=short

# ── Run linter + type checker ─────────────────────────────────────────────────
lint:
	docker compose run --rm api ruff check .
	docker compose run --rm api mypy backend/ api/

# ── Apply DB migrations ───────────────────────────────────────────────────────
migrate:
	docker compose run --rm api alembic upgrade head

# ── Seed DB with sample data (for demo / HF Space) ───────────────────────────
seed-db:
	docker compose run --rm api python -m backend.pipelines.seed

# ── Logs shortcut ─────────────────────────────────────────────────────────────
logs:
	docker compose logs -f --tail=100

logs-api:
	docker compose logs -f api --tail=100

logs-airflow:
	docker compose logs -f airflow-webserver airflow-scheduler --tail=100

# ── Export seed data for HF Space ─────────────────────────────────────────────
# Run AFTER the ingestion DAGs have completed at least one full cycle.
# Exports pg_dump + ChromaDB snapshot → data/seed/
seed-export:
	@echo "→ Creating data/seed directories"
	mkdir -p data/seed/init data/seed/chromadb_snapshot
	@echo "→ Exporting PostgreSQL seed dump"
	docker compose exec postgres pg_dump \
		-U $${POSTGRES_USER:-complianceiq} \
		-d $${POSTGRES_DB:-complianceiq} \
		--no-owner --no-acl \
		| gzip > data/seed/init/01_complianceiq_seed.sql.gz
	@echo "→ Copying ChromaDB snapshot"
	docker compose cp chromadb:/chroma/chroma/. data/seed/chromadb_snapshot/
	@echo "✓ Seed export complete → data/seed/"

# ── Hugging Face Space (slimmed 5-service) ────────────────────────────────────
hf-build:
	@echo "→ Building React frontend"
	cd frontend && npm ci && npm run build
	@echo "→ Building HF Docker images"
	docker compose -f docker-compose.hf.yml build

hf-up:
	docker compose -f docker-compose.hf.yml up -d
	@echo "✓ HF demo running:"
	@echo "  API   → http://localhost:7860/docs"
	@echo "  UI    → http://localhost:3000"

hf-down:
	docker compose -f docker-compose.hf.yml down
