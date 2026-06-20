"""
src/api/server.py — Lightweight FastAPI REST layer for violation data.

Endpoints:
    GET /health                                        — liveness check
    GET /api/sessions                                  — list all sessions
    GET /api/sessions/{session}/cameras                — list cameras for a session
    GET /api/violations?session=&limit=&camera_id=     — list violation events
    GET /api/violations/{session}                      — same, keyed by session path param
    GET /api/summary/{session}                         — aggregated stats

Run standalone:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
Or via:
    python -m src.api.server
"""

from __future__ import annotations

import time
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.database.repository import ViolationRepository

app = FastAPI(
    title="PPE Compliance Monitor API",
    description="Query violation events recorded by the PPE Compliance Monitor.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_repo: ViolationRepository | None = None


def _get_repo() -> ViolationRepository:
    global _repo
    if _repo is None:
        _repo = ViolationRepository()
        _repo.init()
    return _repo


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/sessions", tags=["data"])
def list_sessions() -> dict:
    return {"sessions": _get_repo().list_sessions()}


@app.get("/api/sessions/{session}/cameras", tags=["data"])
def list_cameras(session: str) -> dict:
    return {"session": session, "cameras": _get_repo().list_cameras(session)}


@app.get("/api/violations", tags=["data"])
def get_violations(
    session:   str          = Query(..., description="Session identifier"),
    limit:     int          = Query(500, ge=1, le=5000, description="Max rows to return"),
    camera_id: str | None   = Query(None, description="Filter by camera ID"),
) -> dict:
    events = _get_repo().get_violations(session, limit=limit, camera_id=camera_id)
    return {"session": session, "count": len(events), "events": events}


@app.get("/api/violations/{session}", tags=["data"])
def get_violations_by_path(
    session:   str,
    limit:     int        = Query(500, ge=1, le=5000),
    camera_id: str | None = Query(None),
) -> dict:
    events = _get_repo().get_violations(session, limit=limit, camera_id=camera_id)
    return {"session": session, "count": len(events), "events": events}


@app.get("/api/summary/{session}", tags=["data"])
def get_summary(session: str) -> dict:
    summary = _get_repo().get_session_summary(session)
    if summary["total_events"] == 0:
        raise HTTPException(status_code=404, detail=f"No data for session {session!r}")
    return {"session": session, **summary}


if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
