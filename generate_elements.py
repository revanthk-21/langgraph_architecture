"""
tools/dfmea/generate_elements.py  (and subsequent DFMEA generation tools)
──────────────────────────────────
All DFMEA generation tools follow the same pattern:
  1. Build a prompt using RAG context + component info
  2. Call Claude Sonnet with structured JSON output
  3. Parse and write to state

RAG context (dfmea_rag_context) is threaded through every prompt —
this is the RAG-into-DFMEA integration.
"""

from __future__ import annotations
from core.tool_base import BaseTool, ToolResult
from core.state import AgentState
from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
import json


# ── Shared JSON extraction helper ─────────────────────────────────────────────

def _extract_json(text: str) -> list | dict:
    """Strip markdown fences and parse JSON from LLM output."""
    clean = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(clean)


# ── 1. Generate Elements (Step 1 of wizard) ───────────────────────────────────

ELEMENTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AIAG-VDA 2019 DFMEA expert.
Generate a structured list of system elements for the given component.
Each element must have: id, name, type (system|subsystem|component), parent_id.
Follow the boundary diagram (B-Diagram) hierarchy.

Historical context from engineering documents:
{rag_context}

Output ONLY a JSON array. No preamble."""),
    ("human", "Component: {component}\nSubsystem: {subsystem}\nCase: {case}"),
])


class GenerateElementsTool(BaseTool):
    name        = "generate_elements"
    description = "Generate DFMEA system elements hierarchy (Step 1). Uses RAG context."
    reads       = ["dfmea_component", "dfmea_subsystem", "dfmea_case", "dfmea_rag_context"]
    writes      = ["dfmea_elements"]

    async def arun(self, state: AgentState) -> ToolResult:
        llm   = get_llm(temperature=0.0)
        chain = ELEMENTS_PROMPT | llm
        resp  = await chain.ainvoke({
            "component":   state.get("dfmea_component", ""),
            "subsystem":   state.get("dfmea_subsystem", ""),
            "case":        state.get("dfmea_case", "new"),
            "rag_context": state.get("dfmea_rag_context", "No historical context available."),
        })
        return {"dfmea_elements": _extract_json(resp.content)}


# ── 2. Generate Functions (Step 4) ────────────────────────────────────────────

FUNCTIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AIAG-VDA 2019 DFMEA expert.
For each system element, define its functions using verb + noun format.
Each function must have: id, element_id, description, requirement.

Historical context:
{rag_context}

Output ONLY a JSON array."""),
    ("human", "Elements:\n{elements}"),
])


class GenerateFunctionsTool(BaseTool):
    name        = "generate_functions"
    description = "Generate functions for each DFMEA element (Step 4). Uses RAG context."
    reads       = ["dfmea_elements", "dfmea_rag_context"]
    writes      = ["dfmea_functions"]

    async def arun(self, state: AgentState) -> ToolResult:
        llm   = get_llm(temperature=0.0)
        chain = FUNCTIONS_PROMPT | llm
        resp  = await chain.ainvoke({
            "elements":    json.dumps(state.get("dfmea_elements", []), indent=2),
            "rag_context": state.get("dfmea_rag_context", ""),
        })
        return {"dfmea_functions": _extract_json(resp.content)}


# ── 3. Generate Failure Modes (Step 7) ───────────────────────────────────────

FAILURES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AIAG-VDA 2019 DFMEA expert.
For each function, generate potential failure modes (FM), their effects (FE),
and severity (S) ratings 1-10 per AIAG-VDA criteria.

Each failure mode: {{id, function_id, failure_mode, failure_effect, severity_s}}

IMPORTANT — prioritise these historically documented failure modes from engineering records:
{rag_context}

Output ONLY a JSON array."""),
    ("human", "Functions:\n{functions}"),
])


class GenerateFailuresTool(BaseTool):
    name        = "generate_failures"
    description = "Generate failure modes and effects with severity ratings. RAG-informed."
    reads       = ["dfmea_functions", "dfmea_rag_context"]
    writes      = ["dfmea_failure_modes"]

    async def arun(self, state: AgentState) -> ToolResult:
        llm   = get_llm(temperature=0.0)
        chain = FAILURES_PROMPT | llm
        resp  = await chain.ainvoke({
            "functions":   json.dumps(state.get("dfmea_functions", []), indent=2),
            "rag_context": state.get("dfmea_rag_context", ""),
        })
        return {"dfmea_failure_modes": _extract_json(resp.content)}


# ── 4. Generate Failure Causes (Step 8) ──────────────────────────────────────

CAUSES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AIAG-VDA 2019 DFMEA expert.
For each failure mode, generate failure causes (FC) and prevention controls (PC).
Include occurrence rating (O) 1-10 and detection rating (D) 1-10.

Each cause: {{id, failure_mode_id, cause, prevention_control, occurrence_o, detection_d}}

Use these historically documented causes as reference:
{rag_context}

Output ONLY a JSON array."""),
    ("human", "Failure modes:\n{failure_modes}\nNoise factors:\n{noise_factors}"),
])


class GenerateCausesTool(BaseTool):
    name        = "generate_causes"
    description = "Generate failure causes with O and D ratings. RAG-informed."
    reads       = ["dfmea_failure_modes", "dfmea_noise_factors", "dfmea_rag_context"]
    writes      = ["dfmea_failure_causes"]

    async def arun(self, state: AgentState) -> ToolResult:
        llm   = get_llm(temperature=0.0)
        chain = CAUSES_PROMPT | llm
        resp  = await chain.ainvoke({
            "failure_modes":  json.dumps(state.get("dfmea_failure_modes", []), indent=2),
            "noise_factors":  json.dumps(state.get("dfmea_noise_factors", []), indent=2),
            "rag_context":    state.get("dfmea_rag_context", ""),
        })
        return {"dfmea_failure_causes": _extract_json(resp.content)}


# ── 5. Rate Risks (Step 9) ────────────────────────────────────────────────────

class RateRisksTool(BaseTool):
    name        = "rate_risks"
    description = "Compute AP (Action Priority) from S/O/D per AIAG-VDA 2019 table."
    reads       = ["dfmea_failure_modes", "dfmea_failure_causes"]
    writes      = ["dfmea_risk_ratings"]

    def run(self, state: AgentState) -> ToolResult:
        """
        AIAG-VDA 2019 uses Action Priority (AP: H/M/L) not RPN.
        This implements the lookup table logic deterministically.
        """
        causes = state.get("dfmea_failure_causes", [])
        modes  = {m["id"]: m for m in state.get("dfmea_failure_modes", [])}
        ratings = []

        for cause in causes:
            fm_id = cause.get("failure_mode_id")
            fm    = modes.get(fm_id, {})
            S     = int(fm.get("severity_s", 5))
            O     = int(cause.get("occurrence_o", 5))
            D     = int(cause.get("detection_d", 5))
            AP    = _compute_action_priority(S, O, D)
            ratings.append({
                "cause_id": cause["id"],
                "S": S, "O": O, "D": D,
                "AP": AP,
                "rpn_legacy": S * O * D,   # for backwards compatibility with Ford/GM format
            })

        return {"dfmea_risk_ratings": ratings}


def _compute_action_priority(S: int, O: int, D: int) -> str:
    """AIAG-VDA 2019 AP lookup (simplified — replace with full table if needed)."""
    if S >= 9:
        return "H"
    if S >= 7 and O >= 4:
        return "H"
    if S >= 5 and O >= 4 and D >= 7:
        return "H"
    if S >= 7 or (O >= 4 and D >= 4):
        return "M"
    return "L"


# ── 6. Assemble Output ────────────────────────────────────────────────────────

class AssembleOutputTool(BaseTool):
    name        = "assemble_output"
    description = "Assemble all DFMEA fields into a single structured output dict."
    reads       = ["dfmea_elements", "dfmea_functions", "dfmea_failure_modes",
                   "dfmea_failure_causes", "dfmea_risk_ratings",
                   "dfmea_component", "dfmea_subsystem", "dfmea_case"]
    writes      = ["dfmea_output"]

    def run(self, state: AgentState) -> ToolResult:
        output = {
            "metadata": {
                "component": state.get("dfmea_component"),
                "subsystem": state.get("dfmea_subsystem"),
                "case":      state.get("dfmea_case"),
                "standard":  "AIAG-VDA 2019",
            },
            "elements":       state.get("dfmea_elements", []),
            "functions":      state.get("dfmea_functions", []),
            "failure_modes":  state.get("dfmea_failure_modes", []),
            "failure_causes": state.get("dfmea_failure_causes", []),
            "risk_ratings":   state.get("dfmea_risk_ratings", []),
        }
        return {"dfmea_output": output}


# ── 7. Export XLSX ────────────────────────────────────────────────────────────

class ExportXlsxTool(BaseTool):
    name        = "export_xlsx"
    description = "Export the assembled DFMEA output to a formatted xlsx file."
    reads       = ["dfmea_output"]
    writes      = ["dfmea_export_path"]

    def run(self, state: AgentState) -> ToolResult:
        # Delegate to your existing exporter — no rewrite needed
        from routers_ifmea import export_dfmea_to_xlsx   # your existing module
        path = export_dfmea_to_xlsx(state["dfmea_output"])
        return {"dfmea_export_path": path}
