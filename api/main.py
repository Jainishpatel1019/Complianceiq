"""ComplianceIQ FastAPI application entry point.

Why FastAPI over Flask/Django:
- Native async/await support — all DB queries are async (asyncpg), so we
  need an async framework. Flask requires hacks; Django's async support is
  partial. FastAPI is async-first.
- Automatic OpenAPI docs at /docs — interviewers can try the API live.
- Pydantic v2 integration — request/response validation is zero boilerplate.
- WebSocket support built-in — needed for live agent reasoning trace streaming
  to the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import regulations, reports, health, change_scores, causal, graph, refresh
from api.websockets import router as ws_router

# ── Structured logging setup ─────────────────────────────────────────────────
# Why structlog over stdlib logging: structlog produces machine-readable JSON
# logs. Every log line includes regulation_id, dag_run_id, etc. as structured
# fields — searchable in any log aggregator without regex parsing.
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log = structlog.get_logger()


async def _auto_seed_if_empty() -> None:
    """Self-healing seed — runs inside the API's own async event loop.

    Uses api/seed.py which has ZERO external dependencies (no sklearn,
    no ChromaDB, no subprocess). Inserts directly via AsyncSession.
    Triggered 3 seconds after startup.

    Threshold: if fewer than 1000 regulations exist, run the full seed.
    This covers the case where start.sh core-seed inserted only ~10 records
    and the background bulk-seed failed — so we don't skip when count > 0.
    """
    await asyncio.sleep(3)
    try:
        from sqlalchemy import text as sqla_text
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        from db import get_engine
        from api.seed import seed_db

        engine = get_engine()
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with engine.connect() as conn:
            row = await conn.execute(sqla_text("SELECT COUNT(*) FROM regulations"))
            count = row.scalar() or 0

        if count >= 1000:
            log.info("auto_seed_skip", existing=count, reason="already>=1000")
            return

        log.info("auto_seed_start", existing=count, reason="below 1000 — running full seed")
        async with async_session() as session:
            inserted = await seed_db(session, target=3500)
        log.info("auto_seed_complete", inserted=inserted)

    except Exception as exc:
        log.error("auto_seed_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic for the FastAPI app.

    Why lifespan over @app.on_event: on_event is deprecated in FastAPI 0.95+.
    Lifespan uses async context manager — cleaner resource management.
    """
    log.info("ComplianceIQ API starting up")
    # Verify DB connection on startup — fail fast rather than serving 500s
    from db import get_engine
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    log.info("Database connection verified")

    # Self-healing: seed the DB if it's empty (catches start.sh seed failures)
    asyncio.create_task(_auto_seed_if_empty())

    yield

    log.info("ComplianceIQ API shutting down")
    await get_engine().dispose()


app = FastAPI(
    title="ComplianceIQ API",
    description=(
        "Regulatory Intelligence Platform — automated financial compliance analysis "
        "with causal inference and mathematical confidence intervals."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — built from env so the same image works locally and on HF Spaces.
# FRONTEND_URL defaults to localhost for local dev; override in .env for prod.
_cors_origins = [
    "http://localhost:3000",
    "http://localhost:8081",
    "https://jainish1019-complianceiq.hf.space",   # old URL (kept for safety)
    "https://jainishp1019-complianceiq.hf.space",  # actual HF Space URL
]
_extra = os.environ.get("FRONTEND_URL", "").strip()
if _extra and _extra not in _cors_origins:
    _cors_origins.append(_extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(regulations.router, prefix="/api/v1/regulations", tags=["regulations"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(change_scores.router, prefix="/api/v1/change-scores", tags=["change-scores"])
app.include_router(causal.router, prefix="/api/v1/causal", tags=["causal"])
app.include_router(graph.router,  prefix="/api/v1/graph",  tags=["graph"])
app.include_router(ws_router, prefix="/ws", tags=["websockets"])
app.include_router(refresh.router, prefix="/api/v1/refresh", tags=["refresh"])

# ── Static files ──────────────────────────────────────────────────────────────
# 1. Landing page dashboard (api/static/) — always present in the image
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# 2. React frontend dist (frontend/dist/) — present if built during Docker build
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/app", StaticFiles(directory=_frontend_dist, html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the landing page dashboard."""
    from fastapi.responses import FileResponse
    landing = os.path.join(_static_dir, "index.html")
    if os.path.isfile(landing):
        return FileResponse(landing, media_type="text/html")
    return RedirectResponse(url="/docs")
