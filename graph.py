"""
agents/graph.py
───────────────
Main LangGraph graph definition. This is the single file that wires
every node and edge together. Reading this file gives you the complete
picture of how the system executes.

Graph structure:
                        ┌──────────────────────────────────┐
                        │           SUPERVISOR             │
                        └──┬───────────┬──────────────┬───┘
                           │           │              │
                        [rag]       [dfmea]       [optimize]
                           │           │              │
                    ┌──────┘    ┌──────┘       ┌─────┘
                    ▼           ▼               ▼
              RAG sub-graph  DFMEA sub-graph  OPT sub-graph
                    │           │               │  (cyclic)
                    └───────────┴───────────────┘
                                │
                               END
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver   # pip install langgraph-checkpoint-redis

from core.state import AgentState
from core.tool_registry import TOOL_REGISTRY

# ── Redis checkpointer (enables resume on failure + human-in-loop) ───────────
# Replace with MemorySaver() for local dev without Redis:
#   from langgraph.checkpoint.memory import MemorySaver
#   checkpointer = MemorySaver()

REDIS_URL   = "redis://localhost:6379"
checkpointer = RedisSaver.from_conn_string(REDIS_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions (used as conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def route_task(state: AgentState) -> Literal["rag", "dfmea", "optimize", "error"]:
    """Entry router — reads task_type and dispatches to the correct sub-graph."""
    if state.get("error"):
        return "error"
    return state.get("task_type", "error")


def route_rag_confidence(state: AgentState) -> Literal["rewrite_query", "generate_answer"]:
    """
    After grade_relevance: if confidence < 0.7 AND we haven't already
    rewritten (to prevent infinite loops), rewrite. Otherwise, generate.
    """
    confidence = state.get("rag_confidence", 1.0)
    query      = state.get("rag_query", "")
    # Detect rewrite by checking if query was already rewritten (simple heuristic)
    if confidence < 0.7 and not query.startswith("[rewritten]"):
        return "rewrite_query"
    return "generate_answer"


def route_dfmea_case(state: AgentState) -> Literal["parse_import", "dfmea_rag_context"]:
    """
    After case_router: import cases (new_env, design_change) parse the xlsx first.
    New design goes straight to RAG context retrieval.
    """
    if state.get("error"):
        return END
    case = state.get("dfmea_case", "new")
    return "parse_import" if case in ("new_env", "design_change") else "dfmea_rag_context"


def route_convergence(state: AgentState) -> Literal["solve_ode", "summarize_opt"]:
    """After check_convergence: loop back to ODE or proceed to summary."""
    return "summarize_opt" if state.get("opt_converged", False) else "solve_ode"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    reg = TOOL_REGISTRY
    g   = StateGraph(AgentState)

    # ── Error sink ────────────────────────────────────────────────────────────
    async def error_node(state: AgentState) -> dict:
        # In production: log to Sentry/CloudWatch here
        print(f"[ERROR] {state.get('error')}")
        return {}

    g.add_node("error", error_node)

    # ── RAG nodes ─────────────────────────────────────────────────────────────
    g.add_node("embed_query",     reg["embed_query"].as_node())
    g.add_node("retrieve_docs",   reg["retrieve_docs"].as_node())
    g.add_node("grade_relevance", reg["grade_relevance"].as_node())
    g.add_node("rewrite_query",   reg["rewrite_query"].as_node())
    g.add_node("generate_answer", reg["generate_answer"].as_node())

    # ── DFMEA nodes ───────────────────────────────────────────────────────────
    g.add_node("case_router",         reg["case_router"].as_node())
    g.add_node("parse_import",        reg["parse_import"].as_node())
    g.add_node("dfmea_rag_context",   reg["dfmea_rag_context"].as_node())
    g.add_node("generate_elements",   reg["generate_elements"].as_node())
    g.add_node("generate_functions",  reg["generate_functions"].as_node())
    g.add_node("generate_failures",   reg["generate_failures"].as_node())
    g.add_node("generate_causes",     reg["generate_causes"].as_node())
    g.add_node("rate_risks",          reg["rate_risks"].as_node())
    g.add_node("assemble_output",     reg["assemble_output"].as_node())
    g.add_node("export_xlsx",         reg["export_xlsx"].as_node())

    # ── Optimizer nodes ───────────────────────────────────────────────────────
    g.add_node("init_opt",          reg["init_opt"].as_node())
    g.add_node("solve_ode",         reg["solve_ode"].as_node())
    g.add_node("compute_rms",       reg["compute_rms"].as_node())
    g.add_node("propose_k",         reg["propose_k"].as_node())
    g.add_node("check_convergence", reg["check_convergence"].as_node())
    g.add_node("summarize_opt",     reg["summarize_opt"].as_node())

    # ─────────────────────────────────────────────────────────────────────────
    # Edges — Entry
    # ─────────────────────────────────────────────────────────────────────────
    g.set_entry_point("__supervisor__")
    g.add_node("__supervisor__", lambda s: {})   # passthrough — routing happens on edge

    g.add_conditional_edges(
        "__supervisor__",
        route_task,
        {
            "rag":      "embed_query",
            "dfmea":    "case_router",
            "optimize": "init_opt",
            "error":    "error",
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # RAG edges
    # ─────────────────────────────────────────────────────────────────────────
    g.add_edge("embed_query",     "retrieve_docs")
    g.add_edge("retrieve_docs",   "grade_relevance")

    g.add_conditional_edges(
        "grade_relevance",
        route_rag_confidence,
        {
            "rewrite_query":  "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )

    # Rewrite cycles back to retrieval (multi-hop RAG)
    g.add_edge("rewrite_query",   "retrieve_docs")
    g.add_edge("generate_answer", END)

    # ─────────────────────────────────────────────────────────────────────────
    # DFMEA edges
    # ─────────────────────────────────────────────────────────────────────────
    g.add_conditional_edges(
        "case_router",
        route_dfmea_case,
        {
            "parse_import":      "parse_import",
            "dfmea_rag_context": "dfmea_rag_context",
        },
    )

    # parse_import → rag context (both import cases still get RAG enrichment)
    g.add_edge("parse_import",       "dfmea_rag_context")

    # Linear wizard pipeline (RAG context feeds every LLM step via state)
    g.add_edge("dfmea_rag_context",  "generate_elements")
    g.add_edge("generate_elements",  "generate_functions")
    g.add_edge("generate_functions", "generate_failures")
    g.add_edge("generate_failures",  "generate_causes")
    g.add_edge("generate_causes",    "rate_risks")
    g.add_edge("rate_risks",         "assemble_output")
    g.add_edge("assemble_output",    "export_xlsx")
    g.add_edge("export_xlsx",        END)

    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer edges  (cyclic)
    # ─────────────────────────────────────────────────────────────────────────
    g.add_edge("init_opt",          "solve_ode")
    g.add_edge("solve_ode",         "compute_rms")
    g.add_edge("compute_rms",       "propose_k")
    g.add_edge("propose_k",         "check_convergence")

    g.add_conditional_edges(
        "check_convergence",
        route_convergence,
        {
            "solve_ode":    "solve_ode",      # ← cycle
            "summarize_opt": "summarize_opt",
        },
    )

    g.add_edge("summarize_opt", END)

    # ─────────────────────────────────────────────────────────────────────────
    # Error sink
    # ─────────────────────────────────────────────────────────────────────────
    g.add_edge("error", END)

    return g.compile(checkpointer=checkpointer)


# Compiled graph — import this everywhere
compiled_graph = build_graph()
