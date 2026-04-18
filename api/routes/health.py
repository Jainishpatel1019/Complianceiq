"""Health check endpoints.

Kubernetes/Docker Compose health checks call GET /health/live and /health/ready.
'live' = process is running. 'ready' = process can serve traffic (DB connected).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db_session

router = APIRouter()
log = structlog.get_logger()


class HealthResponse(BaseModel):
    status: str
    service: str = "complianceiq-api"


@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness probe — returns 200 if process is alive."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def readiness(db: AsyncSession = Depends(get_db_session)) -> HealthResponse:
    """Readiness probe — returns 200 only if DB is reachable.

    Args:
        db: Injected async DB session (FastAPI dependency).

    Returns:
        HealthResponse with status='ok'.

    Raises:
        HTTPException 503 if DB is unreachable (handled by FastAPI default error handler).
    """
    await db.execute(text("SELECT 1"))
    return HealthResponse(status="ok")
