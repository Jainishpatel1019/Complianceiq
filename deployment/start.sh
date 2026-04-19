#!/bin/bash
# ComplianceIQ — HuggingFace Space startup script
# Runs inside the single-container HF deployment.
# Order: postgres → chromadb → migrations → seed → uvicorn
set -e

echo "==> Starting PostgreSQL..."
service postgresql start

echo "==> Waiting for PostgreSQL to be ready..."
until pg_isready -h localhost -U postgres -q; do sleep 1; done

echo "==> Creating database user and database (idempotent)..."
su -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='complianceiq'\" | grep -q 1 || \
       psql -c \"CREATE USER complianceiq WITH SUPERUSER PASSWORD '${POSTGRES_PASSWORD:-complianceiq_demo}';\"" postgres
su -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='complianceiq'\" | grep -q 1 || \
       createdb -O complianceiq complianceiq" postgres

echo "==> Running Alembic migrations..."
cd /app && alembic upgrade head

echo "==> Starting ChromaDB (background)..."
# Copy snapshot to writable location (snapshot is read-only in repo)
mkdir -p /var/lib/chromadb
if [ -d "/app/data/seed/chromadb_snapshot" ] && [ "$(ls -A /app/data/seed/chromadb_snapshot 2>/dev/null)" ]; then
    cp -rn /app/data/seed/chromadb_snapshot/. /var/lib/chromadb/ 2>/dev/null || true
fi
IS_PERSISTENT=TRUE ANONYMIZED_TELEMETRY=FALSE \
    python -m chromadb.app --path /var/lib/chromadb --port 8001 &
CHROMA_PID=$!

echo "==> Waiting for ChromaDB..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
        echo "    ChromaDB ready."
        break
    fi
    sleep 2
done

echo "==> Seeding database (idempotent)..."
CHROMADB_HOST=localhost CHROMADB_PORT=8001 \
    python -m backend.pipelines.seed

echo "==> Starting ComplianceIQ API on port 7860..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 2 \
    --log-level info
