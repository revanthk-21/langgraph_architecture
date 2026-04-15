"""
api/main.py
───────────
FastAPI gateway. Three endpoints — one per agent.
All stream LangGraph state updates via Server-Sent Events (SSE).

Running:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

from agents.graph import compiled_graph

app = FastAPI(title="Agentic AI Framework", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # Next.js dev
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────

class RagRequest(BaseModel):
    query: str = Field(..., description="User's question for RAG retrieval")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class DFMEARequest(BaseModel):
    component:   str
    subsystem:   str
    case:        Literal["new", "new_env", "design_change"]
    import_path: Optional[str] = None   # required for new_env / design_change
    session_id:  str = Field(default_factory=lambda: str(uuid.uuid4()))


class OptimizeRequest(BaseModel):
    k_bounds:         tuple[float, float] = (5000.0, 50000.0)
    ode_params:       dict = {}           # ms, mu, c, kt, road_profile, t_end, dt
    max_iterations:   int  = 80
    convergence_tol:  float = 1e-4
    session_id:       str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── SSE streaming helper ───────────────────────────────────────────────────────

async def stream_graph(initial_state: dict, session_id: str) -> AsyncGenerator[str, None]:
    """
    Runs the compiled LangGraph and yields each state update as an SSE event.
    Frontend receives: data: {"node": "...", "state_delta": {...}}\n\n
    """
    config = {"configurable": {"thread_id": session_id}}

    async for chunk in compiled_graph.astream(initial_state, config=config):
        for node_name, state_delta in chunk.items():
            event = json.dumps({
                "node":        node_name,
                "state_delta": state_delta,
            })
            yield f"data: {event}\n\n"
        await asyncio.sleep(0)   # yield control to event loop

    yield "data: {\"node\": \"__end__\", \"state_delta\": {}}\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/rag")
async def rag_endpoint(req: RagRequest) -> StreamingResponse:
    initial_state = {
        "task_type":    "rag",
        "rag_query":    req.query,
        "rag_confidence": 1.0,
        "rag_retrieved_docs": [],
        "rag_sources":  [],
    }
    return StreamingResponse(
        stream_graph(initial_state, req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/dfmea")
async def dfmea_endpoint(req: DFMEARequest) -> StreamingResponse:
    if req.case in ("new_env", "design_change") and not req.import_path:
        raise HTTPException(400, "import_path required for new_env and design_change cases")

    initial_state = {
        "task_type":         "dfmea",
        "dfmea_case":        req.case,
        "dfmea_component":   req.component,
        "dfmea_subsystem":   req.subsystem,
        "dfmea_import_path": req.import_path,
        "dfmea_noise_factors":   [],
        "dfmea_failure_modes":   [],
        "dfmea_failure_causes":  [],
    }
    return StreamingResponse(
        stream_graph(initial_state, req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/optimize")
async def optimize_endpoint(req: OptimizeRequest) -> StreamingResponse:
    initial_state = {
        "task_type":          "optimize",
        "opt_k_bounds":       req.k_bounds,
        "opt_ode_params":     req.ode_params,
        "opt_max_iterations": req.max_iterations,
        "opt_convergence_tol": req.convergence_tol,
        "opt_history":        [],
    }
    return StreamingResponse(
        stream_graph(initial_state, req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}
