"""
core/state.py
─────────────
Shared AgentState TypedDict that flows through the entire LangGraph.
Every node reads from and writes to this object — it is the single
source of truth across all agents.

Rule: never import from agents/ here. Only primitives and typing.
"""

from __future__ import annotations
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
import operator


# ── Reducer helpers ──────────────────────────────────────────────────────────
# operator.add is used for list fields so multiple nodes can append
# without overwriting each other (LangGraph fan-out safe).

class AgentState(TypedDict, total=False):
    # ── Routing ──────────────────────────────────────────────────────────────
    task_type: Literal["rag", "dfmea", "optimize"]
    error: Optional[str]                          # set by any node on failure

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_query: str
    rag_retrieved_docs: Annotated[list[str], operator.add]   # appended by retriever
    rag_answer: str
    rag_sources: Annotated[list[str], operator.add]
    rag_confidence: float                         # 0-1, triggers re-retrieval if low

    # ── DFMEA ────────────────────────────────────────────────────────────────
    dfmea_case: Literal["new", "new_env", "design_change"]
    dfmea_component: str                          # e.g. "front suspension spring"
    dfmea_subsystem: str                          # e.g. "chassis"
    dfmea_import_path: Optional[str]              # xlsx path for import cases
    dfmea_noise_factors: Annotated[list[str], operator.add]
    dfmea_rag_context: str                        # RAG output injected into DFMEA prompts
    dfmea_elements: list[dict]                    # Step 1 output
    dfmea_functions: list[dict]                   # Step 4 output
    dfmea_failure_modes: Annotated[list[dict], operator.add]
    dfmea_failure_causes: Annotated[list[dict], operator.add]
    dfmea_risk_ratings: list[dict]
    dfmea_output: dict                            # Final assembled DFMEA
    dfmea_export_path: Optional[str]              # Path to generated xlsx

    # ── Optimization ─────────────────────────────────────────────────────────
    opt_spring_k: float                           # Current k under evaluation (N/m)
    opt_k_bounds: tuple[float, float]             # (k_min, k_max)
    opt_ode_params: dict                          # ms, mu, c, kt, road_profile
    opt_rms_acceleration: float                   # Objective value for current k
    opt_iteration: int
    opt_max_iterations: int
    opt_convergence_tol: float
    opt_converged: bool
    opt_history: Annotated[list[dict], operator.add]   # [{k, rms, iter}]
    opt_best_k: float
    opt_best_rms: float
    opt_summary: str                              # LLM-generated engineering summary
