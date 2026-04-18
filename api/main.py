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

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import regulations, reports, health, change_scores, causal, graph
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
    "https://jainish1019-complianceiq.hf.space",  # HF Space URL
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
