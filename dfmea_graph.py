"""
dfmea_graph.py
==============
LangGraph pipeline for the DFMEA Generator.

Graph topology
--------------

                ┌─────────────┐
                │  START      │
                └──────┬──────┘
                       │
              ┌────────▼────────┐
              │  route_import   │  (conditional — skipped for Case 1)
              └────────┬────────┘
          Case 2/3     │  Case 1
         ┌─────────────┴──────────────┐
         ▼                            ▼
   ┌──────────┐               ┌──────────────┐
   │parse_node│               │ b_diagram_   │
   └────┬─────┘               │ checkpoint   │  ← human-in-the-loop
        │                     └──────┬───────┘
        └──────────┬──────────────────┘
                   ▼
           ┌───────────────┐
           │failure_mode   │  (generates modes per selected element+function)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │mode_select    │  ← human-in-the-loop
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │failure_cause  │  (generates causes per selected mode)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │cause_select   │  ← human-in-the-loop
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │risk_rating    │  (maps answers → O / D / S / RPN / AP)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │ifmea_generate │  (generates interface failure modes)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │ifmea_select   │  ← human-in-the-loop (mode selection + matrix)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │assemble_rows  │  (joins DFMEA + IFMEA into final rows)
           └──────┬────────┘
                  │
           ┌──────▼────────┐
           │export_xlsx    │  (writes .xlsx to disk)
           └──────┬────────┘
                  │
                ┌─▼──┐
                │ END │
                └─────┘

Human-in-the-loop nodes use LangGraph's interrupt() so the FastAPI layer
can pause, send state to the frontend, wait for user input, then resume.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from dfmea_state import (
    DFMEAState,
    ImportCase,
    TaskStatus,
    IFMEAInterface,
    IFMEAModeRecord,
    FailureModeRecord,
    FailureCause,
    DFMEARow,
    selected_failure_modes,
    selected_failure_causes,
    focus_element,
    higher_elements,
    lower_elements,
)
from llm_client import bedrock_invoke  # your existing AWS Bedrock wrapper
from dfmea_universal_parser import parse_dfmea_file  # your existing parser
from routers.risk_rating import (
    occurrence_from_answer,
    detection_from_answer,
    severity_from_effect,
    compute_ap,
)
from routers.export import build_xlsx  # your existing xlsx builder


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _status(state: DFMEAState, field: str) -> dict:
    """Return a patch that sets <field>_status = RUNNING."""
    return {f"{field}_status": TaskStatus.RUNNING}


# ─────────────────────────────────────────────────────────────────────────────
# Node: route_import
# Decides whether to run the parser or jump straight to b_diagram_checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

def route_import(state: DFMEAState) -> str:
    """Conditional edge function — returns the name of the next node."""
    if state.get("import_case") in (ImportCase.NEW_ENVIRONMENT, ImportCase.DESIGN_CHANGE):
        return "parse_node"
    return "b_diagram_checkpoint"


# ─────────────────────────────────────────────────────────────────────────────
# Node: parse_node  (Case 2 & 3)
# Calls the universal xlsx parser and pre-populates state.
# ─────────────────────────────────────────────────────────────────────────────

def parse_node(state: DFMEAState) -> dict:
    file_path = state.get("uploaded_file_path")
    if not file_path:
        return {
            "parse_status": TaskStatus.FAILED,
            "parse_error": "No uploaded file path in state.",
        }

    try:
        # The parser needs the element hierarchy the user supplied up-front.
        # For Case 2/3 we expect elements to already be in state (from the
        # launcher screen) or we run a lightweight header-scan pass first.
        elements_input = {
            "focus":   next((e["name"] for e in state.get("elements", []) if e.get("level") == "focus"), ""),
            "higher":  [e["name"] for e in higher_elements(state)],
            "lower":   [e["name"] for e in lower_elements(state)],
        }

        result = parse_dfmea_file(
            file_path=file_path,
            elements=elements_input,
        )

        # Map parser output back into DFMEAState fields
        return {
            "parsed_import":      result,
            "b_connections":      result.get("connections", []),
            "noise_factors":      result.get("noise_factors", []),
            "dfmea_rows":         result.get("dfmea_rows", []),
            "parse_status":       TaskStatus.DONE,
            "parse_error":        None,
        }

    except Exception as exc:
        return {
            "parse_status": TaskStatus.FAILED,
            "parse_error":  str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node: b_diagram_checkpoint  (human-in-the-loop)
# Pauses so the user can draw / confirm the B-Diagram in the frontend.
# Resumes with updated elements and b_connections in state.
# ─────────────────────────────────────────────────────────────────────────────

def b_diagram_checkpoint(state: DFMEAState) -> dict:
    """
    Interrupt and hand control to the frontend.
    The frontend will call graph.update_state(thread_id, {...}) with:
      - elements:      list[ElementDef]
      - b_connections: list[BConnection]
      - b_diagram_svg: str (optional)
    Then resume with graph.invoke(Command(resume={}), config).
    """
    user_input = interrupt({
        "step":    "b_diagram",
        "message": "Draw or confirm the B-Diagram, then continue.",
        "state":   {
            "elements":      state.get("elements", []),
            "b_connections": state.get("b_connections", []),
        },
    })
    # user_input is the dict the frontend sends back
    return {
        "elements":      user_input.get("elements",      state.get("elements", [])),
        "b_connections": user_input.get("b_connections", state.get("b_connections", [])),
        "b_diagram_svg": user_input.get("b_diagram_svg"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: ifmea_generate
# For each B-connection, calls the LLM to generate IFMEA failure modes.
# ─────────────────────────────────────────────────────────────────────────────

def ifmea_generate(state: DFMEAState) -> dict:
    connections = state.get("b_connections", [])
    if not connections:
        return {"ifmea_status": TaskStatus.SKIPPED, "ifmea_interfaces": []}

    # Build element name lookup
    elem_map = {e["key"]: e["name"] for e in state.get("elements", [])}

    interfaces: list[IFMEAInterface] = []

    for conn in connections:
        from_name = elem_map.get(conn.get("from_key", ""), conn.get("from_key", ""))
        to_name   = elem_map.get(conn.get("to_key", ""),   conn.get("to_key", ""))
        conn_type = conn.get("conn_type", "Physical")

        prompt = f"""You are an FMEA expert. Given an interface between two elements:
- From element: {from_name}
- To element: {to_name}
- Interface type: {conn_type}

Generate 4-6 realistic interface failure modes. Each failure mode should describe
how the intended transfer across this interface could fail.

Return ONLY a JSON array of strings, no markdown, no preamble.
Example: ["No signal transfer", "Signal degraded", "Intermittent contact"]
"""
        try:
            raw = bedrock_invoke(prompt)
            import json
            modes_raw = json.loads(raw)
        except Exception:
            modes_raw = [f"{conn_type} transfer failure", "Intermittent failure", "Complete loss of transfer"]

        interfaces.append(IFMEAInterface(
            conn_id=conn.get("id", _uid()),
            from_element=from_name,
            to_element=to_name,
            conn_type=conn_type,
            nominal_transfer=conn.get("label", ""),
            failure_modes=[
                IFMEAModeRecord(id=_uid(), mode=m, selected=False)
                for m in modes_raw
            ],
            causes=[],
            modes_loading=False,
            modes_generated=True,
        ))

    return {
        "ifmea_interfaces": interfaces,
        "ifmea_status":     TaskStatus.RUNNING,  # still needs user selection
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: ifmea_select  (human-in-the-loop)
# User selects which IFMEA modes to keep and fills in the matrix ratings.
# ─────────────────────────────────────────────────────────────────────────────

def ifmea_select(state: DFMEAState) -> dict:
    user_input = interrupt({
        "step":    "ifmea_select",
        "message": "Select IFMEA failure modes and complete the interface matrix.",
        "state":   {
            "ifmea_interfaces": state.get("ifmea_interfaces", []),
        },
    })
    return {
        "ifmea_interfaces": user_input.get("ifmea_interfaces", state.get("ifmea_interfaces", [])),
        "ifmea_matrix":     user_input.get("ifmea_matrix"),
        "ifmea_status":     TaskStatus.DONE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: failure_mode_node
# Generates DFMEA failure modes for each selected element × function pair.
# ─────────────────────────────────────────────────────────────────────────────

def failure_mode_node(state: DFMEAState) -> dict:
    elements   = state.get("elements", [])
    noise_facs = [nf.get("factor", "") for nf in state.get("noise_factors", [])]
    noise_str  = ", ".join(noise_facs) if noise_facs else "general operating conditions"

    all_modes: list[FailureModeRecord] = []

    for elem in elements:
        for fn in elem.get("functions", []):
            prompt = f"""You are an FMEA expert. Generate 4-6 failure modes for:
- Element: {elem.get('name')}
- Function: {fn}
- Noise factors: {noise_str}

A failure mode is the way in which this function could fail to perform as intended.
Return ONLY a JSON array of strings (the failure modes). No markdown, no preamble.
"""
            try:
                import json
                raw   = bedrock_invoke(prompt)
                modes = json.loads(raw)
            except Exception:
                modes = [f"{fn} — complete loss", f"{fn} — degraded performance", f"{fn} — intermittent failure"]

            for m in modes:
                all_modes.append(FailureModeRecord(
                    id=_uid(),
                    mode=m,
                    element=elem.get("name", ""),
                    selected=False,
                ))

    return {
        "failure_modes":        all_modes,
        "failure_mode_status":  TaskStatus.RUNNING,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: mode_select  (human-in-the-loop)
# User reviews and selects which failure modes to analyse further.
# ─────────────────────────────────────────────────────────────────────────────

def mode_select(state: DFMEAState) -> dict:
    user_input = interrupt({
        "step":    "mode_select",
        "message": "Select the failure modes you want to analyse.",
        "state":   {"failure_modes": state.get("failure_modes", [])},
    })
    return {
        "failure_modes":       user_input.get("failure_modes", state.get("failure_modes", [])),
        "failure_mode_status": TaskStatus.DONE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: failure_cause_node
# Generates causes for each selected failure mode.
# Case 1 note: ALL causes are noise-driven by definition (no design defects).
# ─────────────────────────────────────────────────────────────────────────────

def failure_cause_node(state: DFMEAState) -> dict:
    import_case  = state.get("import_case", ImportCase.NEW_DESIGN)
    noise_factors = state.get("noise_factors", [])
    modes         = selected_failure_modes(state)

    case1_note = (
        "IMPORTANT: This is a new design. All failure causes must be noise-driven "
        "(customer usage variation, manufacturing variation, environmental factors). "
        "Do NOT generate design defect causes." if import_case == ImportCase.NEW_DESIGN else ""
    )

    all_causes: list[FailureCause] = []

    for mode in modes:
        noise_str = "; ".join(
            f"{nf.get('category','')}: {nf.get('factor','')}"
            for nf in noise_factors
        ) or "typical operating conditions"

        prompt = f"""You are an FMEA expert. Generate 3-5 failure causes for:
- Element: {mode.get('element')}
- Failure mode: {mode.get('mode')}
- Noise factors to consider: {noise_str}
{case1_note}

Return ONLY a JSON array of objects with keys:
  cause, noise_category, noise_factor
No markdown, no preamble.
"""
        try:
            import json
            raw    = bedrock_invoke(prompt)
            causes = json.loads(raw)
        except Exception:
            causes = [{"cause": f"Noise-induced {mode.get('mode','failure')}", "noise_category": "Customer usage", "noise_factor": "Variation"}]

        for c in causes:
            all_causes.append(FailureCause(
                id=_uid(),
                failure_mode_id=mode.get("id", ""),
                cause=c.get("cause", ""),
                noise_category=c.get("noise_category", ""),
                noise_factor=c.get("noise_factor", ""),
                selected=False,
                prevention_methods="",
                detection_methods="",
                occurrence_answer="",
                detection_answer="",
            ))

    return {
        "failure_causes":       all_causes,
        "failure_cause_status": TaskStatus.RUNNING,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: cause_select  (human-in-the-loop)
# User selects causes, enters prevention/detection methods, and answers
# the occurrence/detection rating questions.
# ─────────────────────────────────────────────────────────────────────────────

def cause_select(state: DFMEAState) -> dict:
    user_input = interrupt({
        "step":    "cause_select",
        "message": "Select causes and enter prevention/detection methods and ratings.",
        "state":   {"failure_causes": state.get("failure_causes", [])},
    })
    return {
        "failure_causes":       user_input.get("failure_causes", state.get("failure_causes", [])),
        "failure_cause_status": TaskStatus.DONE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: risk_rating_node
# Maps user answers → O / D / S / RPN / AP for each selected cause.
# ─────────────────────────────────────────────────────────────────────────────

def risk_rating_node(state: DFMEAState) -> dict:
    causes = selected_failure_causes(state)
    modes  = {m["id"]: m for m in state.get("failure_modes", [])}

    updated: list[FailureCause] = []
    for cause in causes:
        mode_rec = modes.get(cause.get("failure_mode_id", ""), {})

        occ  = occurrence_from_answer(cause.get("occurrence_answer", ""))
        det  = detection_from_answer(cause.get("detection_answer", ""))
        sev  = severity_from_effect(mode_rec.get("mode", ""), cause.get("cause", ""))
        rpn  = occ * det * sev
        ap   = compute_ap(sev, occ, det)

        updated.append({
            **cause,
            "occurrence": occ,
            "detection":  det,
            "severity":   sev,
            "rpn":        rpn,
            "action_priority": ap,
        })

    # Write back into state replacing the old cause list
    all_causes = [
        c if c.get("id") not in {u["id"] for u in updated} else
        next(u for u in updated if u["id"] == c.get("id"))
        for c in state.get("failure_causes", [])
    ]

    return {
        "failure_causes":     all_causes,
        "risk_rating_status": TaskStatus.DONE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: assemble_rows
# Joins elements, functions, modes, causes, and ratings into flat DFMEARow list.
# ─────────────────────────────────────────────────────────────────────────────

def assemble_rows(state: DFMEAState) -> dict:
    elem_fn_map: dict[str, str] = {}
    for elem in state.get("elements", []):
        for fn in elem.get("functions", []):
            elem_fn_map[fn] = elem.get("name", "")

    mode_map = {m["id"]: m for m in state.get("failure_modes", [])}

    rows: list[DFMEARow] = []

    for cause in selected_failure_causes(state):
        mode = mode_map.get(cause.get("failure_mode_id", ""), {})
        element_name = mode.get("element", "")

        # Best-effort function lookup
        fn = next(
            (f for e in state.get("elements", [])
             if e.get("name") == element_name
             for f in e.get("functions", [])),
            "",
        )

        rows.append(DFMEARow(
            id=_uid(),
            element=element_name,
            function=fn,
            failure_mode=mode.get("mode", ""),
            failure_effect="",               # user may have typed this; extend as needed
            failure_cause=cause.get("cause", ""),
            noise_category=cause.get("noise_category", ""),
            noise_factor=cause.get("noise_factor", ""),
            prevention_controls=cause.get("prevention_methods", ""),
            detection_controls=cause.get("detection_methods", ""),
            severity=cause.get("severity", 1),
            occurrence=cause.get("occurrence", 1),
            detection=cause.get("detection", 1),
            rpn=cause.get("rpn", 1),
            action_priority=cause.get("action_priority", "L"),
            recommended_action=cause.get("recommended_action", ""),
        ))

    # Assemble IFMEA rows from selected interface modes
    ifmea_rows = []
    for iface in state.get("ifmea_interfaces", []):
        for mode in iface.get("failure_modes", []):
            if mode.get("selected"):
                ifmea_rows.append({
                    "interface": f"{iface['from_element']} → {iface['to_element']}",
                    "conn_type": iface.get("conn_type", ""),
                    "failure_mode": mode.get("mode", ""),
                })

    return {
        "dfmea_rows": rows,
        "ifmea_rows": ifmea_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: export_node
# Writes the final DFMEA (and optionally IFMEA) to .xlsx.
# ─────────────────────────────────────────────────────────────────────────────

def export_node(state: DFMEAState) -> dict:
    try:
        path = build_xlsx(
            dfmea_rows=state.get("dfmea_rows", []),
            ifmea_rows=state.get("ifmea_rows", []),
            project_name=state.get("project_name", "DFMEA"),
        )
        return {
            "export_path":   path,
            "export_status": TaskStatus.DONE,
            "export_error":  None,
        }
    except Exception as exc:
        return {
            "export_status": TaskStatus.FAILED,
            "export_error":  str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the DFMEA StateGraph.

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        Defaults to MemorySaver (in-process). For production, pass a
        PostgresSaver or RedisSaver so state survives across restarts.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph graph, ready to invoke or stream.

    Usage
    -----
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Initial invocation (runs until first interrupt)
    result = graph.invoke(initial_state(...), config)

    # Resume after user input at an interrupt checkpoint
    graph.update_state(config, {"elements": [...], "b_connections": [...]})
    result = graph.invoke(Command(resume={}), config)
    """

    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(DFMEAState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("parse_node",           parse_node)
    builder.add_node("b_diagram_checkpoint", b_diagram_checkpoint)
    builder.add_node("ifmea_generate",       ifmea_generate)
    builder.add_node("ifmea_select",         ifmea_select)
    builder.add_node("failure_mode_node",    failure_mode_node)
    builder.add_node("mode_select",          mode_select)
    builder.add_node("failure_cause_node",   failure_cause_node)
    builder.add_node("cause_select",         cause_select)
    builder.add_node("risk_rating_node",     risk_rating_node)
    builder.add_node("assemble_rows",        assemble_rows)
    builder.add_node("export_node",          export_node)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry: conditional on import case
    builder.add_conditional_edges(
        START,
        route_import,
        {
            "parse_node":           "parse_node",
            "b_diagram_checkpoint": "b_diagram_checkpoint",
        },
    )

    # parse_node → b_diagram (user reviews / adjusts the pre-populated diagram)
    builder.add_edge("parse_node", "b_diagram_checkpoint")

    # B-Diagram confirmed → failure modes (IFMEA runs after risk rating now)
    builder.add_edge("b_diagram_checkpoint", "failure_mode_node")

    # Failure modes generated → user selection
    builder.add_edge("failure_mode_node", "mode_select")

    # Modes selected → causes
    builder.add_edge("mode_select", "failure_cause_node")

    # Causes generated → user selection + rating input
    builder.add_edge("failure_cause_node", "cause_select")

    # Causes selected → compute risk ratings
    builder.add_edge("cause_select", "risk_rating_node")

    # Risk rated → assemble flat rows
    # Risk rated → IFMEA (runs last, after all DFMEA work is done)
    builder.add_edge("risk_rating_node", "ifmea_generate")

    # IFMEA generate → user selection
    builder.add_edge("ifmea_generate", "ifmea_select")

    # IFMEA selected → assemble flat rows
    builder.add_edge("ifmea_select", "assemble_rows")

    # Rows assembled → export
    builder.add_edge("assemble_rows", "export_node")

    # Done
    builder.add_edge("export_node", END)

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_graph() -> Any:
    """
    Singleton graph instance for FastAPI dependency injection.

    In production, replace MemorySaver with a persistent checkpointer:

        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver.from_conn_string(settings.POSTGRES_URI)
        return build_graph(checkpointer)
    """
    if not hasattr(get_graph, "_instance"):
        get_graph._instance = build_graph()
    return get_graph._instance


def thread_config(session_id: str) -> dict:
    """Return the LangGraph config dict for a given session/thread."""
    return {"configurable": {"thread_id": session_id}}


def get_current_interrupt(graph, session_id: str) -> dict | None:
    """
    Return the current interrupt payload if the graph is paused,
    or None if the graph is still running / already finished.
    Used by the FastAPI /status endpoint to push state to the frontend.
    """
    config = thread_config(session_id)
    state  = graph.get_state(config)

    if state.next:
        # Graph is paused at an interrupt — return the pending tasks payload
        return {
            "paused_at": list(state.next),
            "state":     state.values,
        }
    return None
