"""
dfmea_state.py
==============
Typed state schema for the DFMEA LangGraph pipeline.

Each field maps to a wizard step or cross-cutting concern.
All fields are Optional so nodes only mutate what they own.

Wizard step → node mapping:
  Step 0  B-Diagram          → b_diagram_node
  Step 1  IFMEA              → ifmea_generate_node  → ifmea_select_node
  Step 2  Functions          → (user input only; no LLM node)
  Step 3  P-Diagram          → (user input only; no LLM node)
  Step 4  Failure Modes      → failure_mode_node
  Step 5  Failure Causes     → failure_cause_node
  Step 6  Risk Rating        → risk_rating_node
  Step 7  XLSX Export        → export_node
  Import  Universal Parser   → parse_node
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

import operator


# ─────────────────────────────────────────────────────────────────────────────
# Enums / Literals
# ─────────────────────────────────────────────────────────────────────────────

class ImportCase(str, Enum):
    NEW_DESIGN       = "case1"   # New design — no existing DFMEA
    NEW_ENVIRONMENT  = "case2"   # Same design, new environment / use case
    DESIGN_CHANGE    = "case3"   # Changes in an existing design


class ConnType(str, Enum):
    PHYSICAL     = "Physical"
    ENERGY       = "Energy"
    INFORMATION  = "Information"
    MATERIAL     = "Material"


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-schemas (reusable typed dicts)
# ─────────────────────────────────────────────────────────────────────────────

class BConnection(TypedDict, total=False):
    """One typed connection in the B-Diagram."""
    id:           str          # uuid
    from_key:     str          # element key, e.g. "focus" | "lower_0" | "higher_1"
    to_key:       str
    conn_type:    ConnType
    label:        str          # optional human label


class ElementDef(TypedDict, total=False):
    """One element (focus / higher / lower) in the system."""
    key:       str             # "focus" | "lower_0" | "higher_1" etc.
    name:      str
    level:     Literal["focus", "higher", "lower"]
    functions: list[str]       # user-defined functions for this element


class NoiseFactor(TypedDict, total=False):
    id:       str
    category: str              # Customer usage / Manufacturing variation / etc.
    factor:   str              # Free-text noise factor description


class IFMEAModeRecord(TypedDict, total=False):
    id:       str
    mode:     str
    selected: bool


class IFMEAInterface(TypedDict, total=False):
    """One interface from a B-diagram connection, analysed by IFMEA."""
    conn_id:          str
    from_element:     str
    to_element:       str
    conn_type:        ConnType
    nominal_transfer: str      # user describes what should flow across
    failure_modes:    list[IFMEAModeRecord]
    causes:           list["IFMEACause"]
    modes_loading:    bool
    modes_generated:  bool


class IFMEACause(TypedDict, total=False):
    id:                  str
    cause:               str
    noise_category:      str
    noise_factor:        str
    selected:            bool
    prevention_methods:  str
    detection_methods:   str
    occurrence_answer:   str
    detection_answer:    str
    occurrence:          Optional[int]
    detection:           Optional[int]
    rpn:                 Optional[int]
    action_priority:     Optional[str]


class FailureModeRecord(TypedDict, total=False):
    id:       str
    mode:     str
    element:  str              # which element owns this mode
    selected: bool


class FailureCause(TypedDict, total=False):
    id:                  str
    failure_mode_id:     str
    cause:               str
    noise_category:      str
    noise_factor:        str
    selected:            bool
    prevention_methods:  str
    detection_methods:   str
    occurrence_answer:   str
    detection_answer:    str
    occurrence:          Optional[int]
    detection:           Optional[int]
    severity:            Optional[int]
    rpn:                 Optional[int]
    action_priority:     Optional[str]
    recommended_action:  str


class DFMEARow(TypedDict, total=False):
    """One finalised DFMEA row ready for export."""
    id:                  str
    element:             str
    function:            str
    failure_mode:        str
    failure_effect:      str
    failure_cause:       str
    noise_category:      str
    noise_factor:        str
    prevention_controls: str
    detection_controls:  str
    severity:            int
    occurrence:          int
    detection:           int
    rpn:                 int
    action_priority:     str
    recommended_action:  str


class ParsedImport(TypedDict, total=False):
    """Output of the universal DFMEA xlsx parser."""
    format_detected:   str     # "aiag_vda_2019" | "legacy_ford_gm_chrysler"
    elements:          list[ElementDef]
    connections:       list[BConnection]
    noise_factors:     list[NoiseFactor]
    dfmea_rows:        list[DFMEARow]
    raw_header_map:    dict[str, str]  # original col → normalised col


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Graph State
# ─────────────────────────────────────────────────────────────────────────────

class DFMEAState(TypedDict, total=False):
    """
    Master state object threaded through every node in the LangGraph pipeline.

    Convention
    ----------
    - Fields named `*_status` are TaskStatus literals for UI progress.
    - Fields named `*_error` carry human-readable error strings.
    - List fields use `Annotated[list, operator.add]` so parallel nodes can
      append without clobbering each other.
    - Everything is Optional so nodes only declare what they touch.
    """

    # ── Session metadata ───────────────────────────────────────────────────
    session_id:    str
    import_case:   ImportCase           # which of the 3 DFMEA cases
    project_name:  str
    created_at:    str                  # ISO datetime

    # ── Import / Parse (Case 2 & 3 only) ─────────────────────────────────
    uploaded_file_path:  Optional[str]
    parsed_import:       Optional[ParsedImport]
    parse_status:        TaskStatus
    parse_error:         Optional[str]

    # ── Step 0: B-Diagram ─────────────────────────────────────────────────
    elements:            list[ElementDef]   # all elements in the system
    b_connections:       list[BConnection]  # typed connections between elements
    b_diagram_svg:       Optional[str]      # serialised SVG for export

    # ── Step 1: IFMEA ──────────────────────────────────────────────────────
    ifmea_interfaces:    list[IFMEAInterface]
    ifmea_status:        TaskStatus
    ifmea_error:         Optional[str]
    ifmea_matrix:        Optional[list[dict]]   # n×n rating matrix (−2 to +2)

    # ── Step 2: Functions ─────────────────────────────────────────────────
    # Functions are stored inside each ElementDef.functions — no separate field.

    # ── Step 3: P-Diagram (noise factors) ────────────────────────────────
    noise_factors:       list[NoiseFactor]

    # ── Step 4: Failure Modes ─────────────────────────────────────────────
    failure_modes:       list[FailureModeRecord]
    failure_mode_status: TaskStatus
    failure_mode_error:  Optional[str]

    # ── Step 5: Failure Causes ────────────────────────────────────────────
    failure_causes:      list[FailureCause]
    failure_cause_status: TaskStatus
    failure_cause_error:  Optional[str]

    # ── Step 6: Risk Rating ───────────────────────────────────────────────
    # Risk fields live inside each FailureCause after the rating node runs.
    risk_rating_status:  TaskStatus
    risk_rating_error:   Optional[str]

    # ── Step 7: Finalised rows & Export ───────────────────────────────────
    dfmea_rows:          list[DFMEARow]
    ifmea_rows:          list[dict]         # finalised IFMEA rows
    export_path:         Optional[str]      # path to generated .xlsx
    export_status:       TaskStatus
    export_error:        Optional[str]

    # ── Global error / interrupt ───────────────────────────────────────────
    error:               Optional[str]
    interrupted:         bool               # set True on human-in-the-loop pause


# ─────────────────────────────────────────────────────────────────────────────
# Reducers for list fields (used by parallel fan-out nodes)
# ─────────────────────────────────────────────────────────────────────────────
# When two nodes write to the same list concurrently, LangGraph needs a
# reducer. We expose pre-built Annotated aliases for convenience.

FailureModeList  = Annotated[list[FailureModeRecord], operator.add]
FailureCauseList = Annotated[list[FailureCause],      operator.add]
DFMEARowList     = Annotated[list[DFMEARow],          operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def initial_state(
    session_id: str,
    import_case: ImportCase = ImportCase.NEW_DESIGN,
    project_name: str = "Untitled Project",
) -> DFMEAState:
    """Return a blank DFMEAState for a new session."""
    from datetime import datetime, timezone
    return DFMEAState(
        session_id=session_id,
        import_case=import_case,
        project_name=project_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        elements=[],
        b_connections=[],
        ifmea_interfaces=[],
        noise_factors=[],
        failure_modes=[],
        failure_causes=[],
        dfmea_rows=[],
        ifmea_rows=[],
        parse_status=TaskStatus.PENDING,
        ifmea_status=TaskStatus.PENDING,
        failure_mode_status=TaskStatus.PENDING,
        failure_cause_status=TaskStatus.PENDING,
        risk_rating_status=TaskStatus.PENDING,
        export_status=TaskStatus.PENDING,
        interrupted=False,
        error=None,
    )


def focus_element(state: DFMEAState) -> Optional[ElementDef]:
    """Return the focus element from state, or None."""
    return next(
        (e for e in state.get("elements", []) if e.get("level") == "focus"),
        None,
    )


def higher_elements(state: DFMEAState) -> list[ElementDef]:
    return [e for e in state.get("elements", []) if e.get("level") == "higher"]


def lower_elements(state: DFMEAState) -> list[ElementDef]:
    return [e for e in state.get("elements", []) if e.get("level") == "lower"]


def selected_failure_modes(state: DFMEAState) -> list[FailureModeRecord]:
    return [m for m in state.get("failure_modes", []) if m.get("selected")]


def selected_failure_causes(state: DFMEAState) -> list[FailureCause]:
    return [c for c in state.get("failure_causes", []) if c.get("selected")]
