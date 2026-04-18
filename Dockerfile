# ComplianceIQ — HuggingFace Spaces Dockerfile
#
# Single-container deployment for HF free tier (16 GB RAM, 2 vCPU).
# Runs: PostgreSQL 16 + ChromaDB (in-process) + FastAPI on port 7860.
# Ollama/LLM calls are disabled in DEMO_MODE=true (pre-computed responses).
#
# Multi-container local dev:  docker compose -f docker-compose.dev.yml up
# HF slim compose (optional): docker compose up  (uses docker-compose.yml)

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl gnupg \
    postgresql postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir --upgrade "sqlalchemy>=2.0,<3" "alembic>=1.13"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment — PostgreSQL ──────────────────────────────────────────────────
ENV POSTGRES_HOST=localhost
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=complianceiq
ENV POSTGRES_USER=complianceiq
ENV POSTGRES_PASSWORD=complianceiq_demo

# ── Environment — ChromaDB (runs as subprocess on port 8001) ─────────────────
ENV CHROMADB_HOST=localhost
ENV CHROMADB_PORT=8001

# ── Environment — Ollama (disabled in demo mode) ─────────────────────────────
ENV OLLAMA_BASE_URL=http://localhost:11434

# ── Environment — App ────────────────────────────────────────────────────────
ENV DEMO_MODE=true
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── PostgreSQL: allow local connections without password ─────────────────────
RUN echo "host all all 127.0.0.1/32 trust" >> /etc/postgresql/15/main/pg_hba.conf 2>/dev/null || \
    echo "host all all 127.0.0.1/32 trust" >> /etc/postgresql/16/main/pg_hba.conf 2>/dev/null || true

# ── Startup script ────────────────────────────────────────────────────────────
RUN chmod +x /app/deployment/start.sh

EXPOSE 7860

CMD ["/app/deployment/start.sh"]
