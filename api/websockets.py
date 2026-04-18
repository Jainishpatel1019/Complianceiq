"""WebSocket endpoint for live agent reasoning trace streaming.

Why WebSocket over polling: the LangGraph agent produces a reasoning trace
step-by-step (fetch → RAG → graph → causal → report). With polling, the
dashboard would check every N seconds and miss intermediate steps. WebSocket
pushes each step the moment it's written to PostgreSQL, giving a live trace
that interviewers find impressive.

The connection lifecycle:
  1. Client connects to /ws/agent-trace/{regulation_id}
  2. Server streams any existing trace steps immediately (catch-up)
  3. Server polls for new steps every 1 second and pushes them
  4. Client disconnects when the report is complete (or on page close)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select, desc

from db import get_session_factory
from db.models import AgentReport

router = APIRouter()
log = structlog.get_logger()

POLL_INTERVAL_SECONDS = 1.0


@router.websocket("/agent-trace/{regulation_id}")
async def agent_trace_stream(websocket: WebSocket, regulation_id: str) -> None:
    """Stream the LangGraph agent reasoning trace for a regulation in real-time.

    Sends JSON messages of the form:
      {"type": "step", "index": 0, "content": {"node": "fetch_classify", ...}}
      {"type": "complete", "report_id": "uuid"}
      {"type": "error", "message": "..."}

    Args:
        websocket: FastAPI WebSocket connection.
        regulation_id: UUID of the regulation being analysed.
    """
    await websocket.accept()
    log.info("WebSocket connected", regulation_id=regulation_id)

    try:
        reg_uuid = uuid.UUID(regulation_id)
    except ValueError:
        await websocket.send_json({"type": "error", "message": "Invalid regulation ID"})
        await websocket.close()
        return

    factory = get_session_factory()
    sent_step_count = 0

    try:
        while True:
            async with factory() as session:
                result = await session.execute(
                    select(AgentReport)
                    .where(AgentReport.regulation_id == reg_uuid)
                    .order_by(desc(AgentReport.created_at))
                    .limit(1)
                )
                report = result.scalar_one_or_none()

            if report is None:
                # No report yet — agent may not have started
                await websocket.send_json({"type": "waiting", "message": "Agent not yet started"})
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            trace: list[dict[str, Any]] = report.agent_reasoning_trace or []

            # Push any new steps since last poll
            new_steps = trace[sent_step_count:]
            for i, step in enumerate(new_steps):
                await websocket.send_json({
                    "type": "step",
                    "index": sent_step_count + i,
                    "content": step,
                })
            sent_step_count += len(new_steps)

            # Check if report is complete (impact_score_high is populated)
            if report.impact_score_high is not None and sent_step_count > 0:
                await websocket.send_json({
                    "type": "complete",
                    "report_id": str(report.id),
                    "p_high": report.impact_score_high,
                    "delta_rwa_median_m": report.delta_rwa_median_m,
                })
                break

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected", regulation_id=regulation_id)
    except Exception as exc:
        log.exception("WebSocket error", regulation_id=regulation_id, error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            pass  # Connection already closed
