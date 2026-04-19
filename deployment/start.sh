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

echo "==> Seeding database with core 10 regulations (idempotent)..."
CHROMADB_HOST=localhost CHROMADB_PORT=8001 \
    python -m backend.pipelines.seed || echo "Core seed warning (non-fatal)"

echo "==> Launching full 3500-regulation seed in background..."
# api/seed.py is self-contained (no sklearn, no ChromaDB).
# The API's _auto_seed_if_empty() will also run this after startup
# if fewer than 1000 records exist — belt-and-suspenders approach.
(
    sleep 5
    python -c "
import asyncio, os, sys
sys.path.insert(0, '/app')
os.environ.setdefault('DATABASE_URL', 'postgresql://complianceiq:${POSTGRES_PASSWORD:-complianceiq_demo}@localhost/complianceiq')

async def run():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from api.seed import seed_db
    db_url = os.environ['DATABASE_URL'].replace('postgresql://', 'postgresql+asyncpg://', 1)
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        n = await seed_db(session, target=3500)
    await engine.dispose()
    print(f'Background seed complete: {n} records inserted')

asyncio.run(run())
" >> /tmp/seed_api.log 2>&1 \
    && echo "API seed complete" >> /tmp/seed_api.log \
    || echo "API seed error — check /tmp/seed_api.log"
) &

echo "==> Starting ComplianceIQ API on port 7860..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 2 \
    --log-level info
