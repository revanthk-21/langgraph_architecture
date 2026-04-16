""" dfmea_universal_parser.py
=========================
Universal DFMEA xlsx parser.  Supports two formats automatically detected by
header fingerprinting:

  FORMAT A — AIAG-VDA 2019  (IIT Madras / Ashok Leyland new format)
    Explicit higher / focus / lower element columns per row.
    Function columns present for all three levels.
    LLM used for: resolving per-row element names (batch fuzzy+LLM match) +
                  deriving missing functions (batch) +
                  noise classification (per-row).

  FORMAT B — Legacy Ford/GM/Chrysler  (Ashok Leyland production format)
    Single "Item / Function" column.  No element columns.
    LLM used for (2 calls per row — no batch to avoid timeouts):
      Call 1: Parse focus / lower / higher element names from row text,
              then match each to user-supplied candidate lists.
      Call 2: Derive focus_fn / higher_fn / lower_fn, preserving causal chain.
      Call 3: Classify noise factors from failure cause text.

WORKFLOW:
  1. User provides: lower_elements list, focus_element, higher_elements list
  2. File format is detected via header fingerprinting
  3. Legacy: per-row element parsing + matching (2 LLM calls) then noise (1 LLM call)
     AIAG:   batch element matching + batch function derivation + per-row noise
  4. Output: elements at each level with their functions + noise factors

IMPORTANT: Element names are NOT inferred by the parser.
They must be provided by the caller:
  {
    focus_element:   str         — single focus system name
    higher_elements: list[str]   — one or more higher system names
    lower_elements:  list[str]   — one or more lower component names
  }

Output schema:
  {
    focus_element:    str,
    higher_elements:  [{ name, functions[] }],
    lower_elements:   [{ name, functions[] }],
    focus_functions:  str[],
    connections: [
      { lower_element, lower_fn, focus_fn, higher_element, higher_fn, row_index }
    ],
    failure_modes: [{
      focus_fn, failure_mode, failure_effect, severity, classification,
      causes: [{
        lower_element, lower_fn, higher_element, higher_fn, cause_text,
        noise_driven, noise_category, noise_factor,
        occurrence, detection, rpn, ap,
        prevention_controls, detection_controls,
      }]
    }],
    noise_factors: { pieceTopiece, changeOverTime, customerUsage,
                     externalEnvironment, systemInteractions }
  }
"""

from __future__ import annotations

import re
import json
import uuid
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable, List

import pandas as pd
import os
import requests

# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────────────

_BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
_BEDROCK_MODEL  = os.environ.get("BEDROCK_MODEL",  "anthropic.claude-3-sonnet-20240229-v1:0")
_BEDROCK_KEY    = os.environ.get("BEDROCK_API_KEY", "")
_BEDROCK_URL    = (
    f"https://bedrock-runtime.{_BEDROCK_REGION}.amazonaws.com"
    f"/model/{_BEDROCK_MODEL}/invoke"
)

_SYSTEM = (
    "You are an expert DFMEA engineer following AIAG-VDA methodology. "
    "Return ONLY what is asked — no preamble, no markdown fences, no explanation."
)


def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _llm(prompt: str, function: str, max_tokens: int = 1024) -> str:
    if not _BEDROCK_KEY:
        return _llm_stub(prompt)
    headers = {
        "Authorization": f"Bearer {_BEDROCK_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(_BEDROCK_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    stop_reason = body.get("stop_reason", "")
    if stop_reason == "max_tokens":
        print(f"[llm WARNING] '{function}' hit max_tokens={max_tokens} — TRUNCATED.")
    text = body["content"][0]["text"].strip()
    print(
        f"[llm] {function} ({max_tokens} tok, stop={stop_reason!r}): "
        f"{text[:120]}{'…' if len(text) > 120 else ''}"
    )
    return text


def _llm_stub(prompt: str) -> str:
    """Offline stub — deterministic placeholders for unit testing."""
    p = prompt.lower()
    if "match" in p and ("lower element" in p or "higher element" in p):
        arr = re.findall(r'"([^"]+)"', prompt)
        return json.dumps({"matched_element": arr[0] if arr else "Unknown"})
    if "batch" in p and "function" in p:
        return json.dumps([
            {
                "focus_fn":  "Perform intended function",
                "higher_fn": "Maintain system operation",
                "lower_fn":  "Perform sub-component function",
            }
        ])
    if "noise" in p:
        return json.dumps(
            {"noise_driven": False, "noise_category": None, "noise_factor": None}
        )
    return "{}"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _s(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _strip_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    try:
        return json.loads(_strip_json(text))
    except Exception:
        return {}


def _parse_json_array(text: str) -> list:
    try:
        result = json.loads(_strip_json(text))
        if isinstance(result, list):
            return result
    except Exception:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_AIAG_FINGERPRINTS = {
    "next higher level", "focus element", "next lower level",
    "failure effect (fe)", "failure mode (fm)", "failure cause (fc)",
}
_LEGACY_FINGERPRINTS = {
    "item / function", "item/function", "requirement",
    "potential failure mode", "potential effect", "potential cause",
    "controls prevention", "controls detection",
}


def detect_format(header_values: list[str]) -> str:
    lowered = {h.lower().strip() for h in header_values if h}
    aiag_hits   = sum(1 for f in _AIAG_FINGERPRINTS   if any(f in h for h in lowered))
    legacy_hits = sum(1 for f in _LEGACY_FINGERPRINTS if any(f in h for h in lowered))
    return "aiag_vda" if aiag_hits >= legacy_hits else "legacy"


def find_header_row(df: pd.DataFrame) -> int:
    for i in range(min(20, len(df))):
        row_text = " ".join(_s(v).lower() for v in df.iloc[i].tolist())
        hits = sum(
            1 for kw in (
                "item", "function", "failure mode", "severity", "occurrence",
                "detection", "cause", "effect", "requirement", "next higher",
            )
            if kw in row_text
        )
        if hits >= 3:
            return i
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPER
# ─────────────────────────────────────────────────────────────────────────────

_AIAG_SEMANTIC: dict[str, list[str]] = {
    # Per-row element name columns (AIAG-VDA has these explicitly)
    "higher_element_col": ["next higher level element", "next higher level", "higher level element"],
    "lower_element_col":  ["next lower level element",  "next lower level",  "lower level element"],
    # Function columns
    "higher_fn":      ["next higher level function", "higher level function"],
    "focus_fn":       ["focus element function", "focus function"],
    "lower_fn":       ["next lower level function", "lower level function"],
    # Failure analysis
    "failure_effect": ["failure effect (fe)", "failure effect to the next higher"],
    "severity":       ["severity (s) of failure", "severity (s)"],
    "classification": ["classification"],
    "failure_mode":   ["failure mode (fm)", "failure mode of the focus"],
    "failure_cause":  ["failure cause (fc)", "failure cause of the next lower"],
    "prevention":     ["current prevention controls (pc)", "current prevention controls"],
    "occurrence":     ["occurrence (o) of fc", "occurrence (o)"],
    "detection_ctrl": ["current detection controls (dc)", "current detection controls"],
    "detection":      ["detection (d) of fc", "detection (d)"],
    "ap":             ["ap"],
}

_LEGACY_SEMANTIC: dict[str, list[str]] = {
    "item_function":  ["item / function", "item/function", "item", "function"],
    "requirement":    ["requirement"],
    "failure_mode":   ["potential failure mode", "failure mode"],
    "failure_effect": ["potential effect(s) of failure", "potential effect", "potential effects"],
    "severity":       ["severity", "sev"],
    "classification": ["classification", "class"],
    "failure_cause":  ["potential cause(s) of failure", "potential causes", "potential cause"],
    "prevention":     ["controls prevention", "prevention control", "current prevention"],
    "occurrence":     ["occurrence", "occ"],
    "detection_ctrl": ["controls detection", "detection control", "current detection"],
    "detection":      ["detection", "det"],
    "rpn":            ["rpn"],
}


def map_columns(header_row: list[str], fmt: str) -> dict[str, int]:
    semantic_map = _AIAG_SEMANTIC if fmt == "aiag_vda" else _LEGACY_SEMANTIC
    result: dict[str, int] = {}
    for semantic, keywords in semantic_map.items():
        for ci, raw_h in enumerate(header_row):
            h = _s(raw_h).lower().strip()
            if not h:
                continue
            if any(kw in h for kw in keywords) and semantic not in result:
                result[semantic] = ci
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT MATCHING  — resolve per-row element from user-supplied candidate list
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_match(cell: str, candidates: list[str]) -> str | None:
    """Fast substring / token overlap match; returns None if ambiguous."""
    if not cell or not candidates:
        return None
    cl = cell.lower()
    # Exact
    for c in candidates:
        if c.lower() == cl:
            return c
    # Substring
    for c in candidates:
        if c.lower() in cl or cl in c.lower():
            return c
    # Token overlap (≥1 common token)
    cell_toks = set(re.split(r"[\s\-_/]+", cl))
    best, best_score = None, 0
    for c in candidates:
        score = len(cell_toks & set(re.split(r"[\s\-_/]+", c.lower())))
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= 1 else None


def llm_batch_match_elements(
    cell_values: list[str],
    candidates: list[str],
    level: str = "lower",
) -> dict[str, str]:
    """
    Map each unique cell value to one of the user-supplied candidate element names.
    Fast-path: fuzzy match; LLM called only for unresolved values.
    Returns { cell_value: matched_element_name }.
    """
    result: dict[str, str] = {}
    needs_llm: list[str] = []

    for cv in cell_values:
        m = _fuzzy_match(cv, candidates)
        if m is not None:
            result[cv] = m
        else:
            needs_llm.append(cv)

    if not needs_llm:
        return result

    rows_text = "\n".join(f'{i+1}. "{cv}"' for i, cv in enumerate(needs_llm))
    prompt = f"""Match each DFMEA {level}-level element cell value to the closest
entry in the user-supplied element list below.

Valid {level}-level element names:
{json.dumps(candidates)}

Cell values (one per line):
{rows_text}

Output ONLY a valid JSON array with one object per cell value, in order:
[
  {{"cell": "<cell value>", "matched_element": "<element from list>"}},
  ...
]"""

    token_budget = min(4096, max(256, len(needs_llm) * 60 + 256))
    raw = _llm(prompt, f"llm_batch_match_elements_{level}", max_tokens=token_budget)
    parsed = _parse_json_array(raw)

    for entry in parsed:
        cv      = entry.get("cell", "")
        matched = entry.get("matched_element", "")
        if cv in needs_llm and matched in candidates:
            result[cv] = matched

    # Final fallback: fuzzy or first candidate
    for cv in needs_llm:
        if cv not in result:
            result[cv] = _fuzzy_match(cv, candidates) or candidates[0]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY PER-ROW LLM CALLS  (2 calls per row — no batching to avoid timeouts)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_legacy_parse_elements(
    row: dict,
    focus_element: str,
    higher_candidates: list[str],
    lower_candidates: list[str],
) -> dict:
    """
    LLM Call 1 of 2 per legacy row.

    Parses: focus element name, lower element name, higher element name
    from the raw row text (item/function, failure_cause, failure_effect).
    Returns {focus_element, lower_element, higher_element} matched to
    user-supplied candidate lists.
    """
    # Fast-path fuzzy match first — only call LLM if needed
    lower_fuzzy  = _fuzzy_match(row.get("failure_cause",  ""), lower_candidates)
    higher_fuzzy = _fuzzy_match(row.get("failure_effect", ""), higher_candidates)

    if lower_fuzzy and higher_fuzzy:
        return {
            "focus_element":  focus_element,
            "lower_element":  lower_fuzzy,
            "higher_element": higher_fuzzy,
        }

    prompt = f"""You are analysing one row of a legacy DFMEA spreadsheet.

User-supplied element names:
  Focus element   : "{focus_element}"
  Higher elements : {json.dumps(higher_candidates)}
  Lower elements  : {json.dumps(lower_candidates)}

Row data:
  Item / Function : "{row.get('_item', '')}"
  Requirement     : "{row.get('requirement', '')}"
  Failure mode    : "{row.get('failure_mode', '')}"
  Failure effect  : "{row.get('failure_effect', '')}"
  Failure cause   : "{row.get('failure_cause', '')}"

Tasks:
1. Identify which HIGHER element (from the list) is impacted by the failure effect.
2. Identify which LOWER element (from the list) is the source of the failure cause.
3. The focus element is always "{focus_element}".

Return ONLY valid JSON — no extra text:
{{
  "focus_element":  "{focus_element}",
  "higher_element": "<exact name from higher elements list>",
  "lower_element":  "<exact name from lower elements list>"
}}"""

    raw  = _llm(prompt, "_llm_legacy_parse_elements", max_tokens=150)
    data = _parse_json(raw)

    # Validate and fallback
    he = data.get("higher_element", "")
    le = data.get("lower_element",  "")

    if he not in higher_candidates:
        he = higher_fuzzy or _fuzzy_match(he, higher_candidates) or higher_candidates[0]
    if le not in lower_candidates:
        le = lower_fuzzy or _fuzzy_match(le, lower_candidates) or lower_candidates[0]

    return {
        "focus_element":  focus_element,
        "higher_element": he,
        "lower_element":  le,
    }


def _llm_legacy_parse_functions(
    row: dict,
    focus_element: str,
    higher_element: str,
    lower_element: str,
) -> dict:
    """
    LLM Call 2 of 2 per legacy row.

    Derives function statements for focus, higher, and lower elements
    from the row context, preserving connections between them.
    Returns {focus_fn, higher_fn, lower_fn}.
    """
    # If requirement cell already looks like a function statement, use it as focus_fn hint
    req = row.get("requirement", "")
    existing_focus_fn = req if req and len(req) > 5 else ""

    if existing_focus_fn:
        fields = ['  "higher_fn": "<function of higher element>"',
                  '  "lower_fn":  "<function of lower element>"']
        existing_note = f'\n  existing_focus_fn (echo unchanged): "{existing_focus_fn}"'
    else:
        fields = ['  "focus_fn":  "<function of focus element>"',
                  '  "higher_fn": "<function of higher element>"',
                  '  "lower_fn":  "<function of lower element>"']
        existing_note = ""

    prompt = f"""Derive DFMEA function statements for one row.

Elements:
  Higher : "{higher_element}"
  Focus  : "{focus_element}"
  Lower  : "{lower_element}"{existing_note}

Row data:
  Item / Function : "{row.get('_item', '')}"
  Requirement     : "{req}"
  Failure mode    : "{row.get('failure_mode', '')}"
  Failure effect  : "{row.get('failure_effect', '')}"
  Failure cause   : "{row.get('failure_cause', '')}"

Rules:
- focus_fn  : positive function of "{focus_element}" that the failure_mode violates
- higher_fn : positive function of "{higher_element}" impaired by the failure_effect
- lower_fn  : positive function of "{lower_element}" negated by the failure_cause
- Format: verb + object, ≤12 words (e.g. "Transmit braking torque to wheel hub")
- Preserve the causal chain: lower_fn → focus_fn → higher_fn

Return ONLY valid JSON:
{{
{chr(10).join(fields)}{(',' + chr(10) + '  "focus_fn": "' + existing_focus_fn + '"') if existing_focus_fn else ''}
}}"""

    raw  = _llm(prompt, "_llm_legacy_parse_functions", max_tokens=200)
    data = _parse_json(raw)

    return {
        "focus_fn":  data.get("focus_fn",  existing_focus_fn),
        "higher_fn": data.get("higher_fn", ""),
        "lower_fn":  data.get("lower_fn",  ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NOISE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_RE = re.compile(
    r"overload|overloading|excess.*load|temperature|\+\d+.*deg|-\d+.*deg|"
    r"salt|corrosive|wading|water ingress|mining|paved road|river sand|"
    r"molten tar|dust|vibration.*road|extreme load|operating in|operating on|"
    r"operating with|environmental|humidity|altitude",
    re.IGNORECASE,
)

_NOISE_CATS = [
    "pieceTopiece", "changeOverTime", "customerUsage",
    "externalEnvironment", "systemInteractions",
]

_CAT_KEYWORDS: dict[str, list[str]] = {
    "pieceTopiece":        ["dimension", "tolerance", "variation", "manufacturing", "assembly", "material property"],
    "changeOverTime":      ["wear", "fatigue", "corrosion", "age", "degrade", "creep", "drift", "life", "cycle"],
    "customerUsage":       ["overload", "load", "road", "speed", "duty cycle", "misuse", "abuse", "operator"],
    "externalEnvironment": ["temperature", "humidity", "salt", "dust", "vibration", "altitude", "water", "molten", "deg"],
    "systemInteractions":  ["interaction", "adjacent", "system", "interface", "cross-talk", "shared"],
}


def classify_cause_noise(cause_text: str, use_llm: bool = True) -> dict:
    if not cause_text.strip():
        return {"noise_driven": False, "noise_category": None, "noise_factor": None}
    if not _NOISE_RE.search(cause_text):
        return {"noise_driven": False, "noise_category": None, "noise_factor": None}

    if not use_llm:
        tl = cause_text.lower()
        scores = {cat: sum(1 for kw in kws if kw in tl) for cat, kws in _CAT_KEYWORDS.items()}
        cat = max(scores, key=lambda k: scores[k])
        m = re.search(r"due to (.+?)(?:\.|$)", cause_text, re.IGNORECASE)
        factor = m.group(1).strip()[:80] if m else cause_text[:60]
        return {"noise_driven": True, "noise_category": cat, "noise_factor": factor}

    prompt = f"""Analyse this DFMEA failure cause. Assume it is noise-driven.

Failure cause: "{cause_text}"

Output ONLY valid JSON:
{{
  "noise_driven": true,
  "noise_category": <"pieceTopiece"|"changeOverTime"|"customerUsage"|"externalEnvironment"|"systemInteractions"|null>,
  "noise_factor": <"concise label e.g. '+52 deg C ambient', 'salt spray corrosion'"|null>
}}"""
    raw = _llm(prompt, "classify_cause_noise", max_tokens=150)
    r   = _parse_json(raw)
    return {
        "noise_driven":   bool(r.get("noise_driven", True)),
        "noise_category": r.get("noise_category"),
        "noise_factor":   r.get("noise_factor"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROW-LEVEL FUNCTION DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def llm_batch_derive_row_functions(rows_data: list[dict]) -> list[dict | None]:
    """
    Derive focus_fn / higher_fn / lower_fn for every row in one batched LLM call.

    Each entry in rows_data must have:
      focus_element, higher_element, lower_element  — ACTUAL per-row element names
      failure_mode, failure_effect, failure_cause, requirement
      existing_focus_fn, existing_higher_fn, existing_lower_fn

    Returns list[{focus_fn, higher_fn, lower_fn}], same length as input.
    Rows with all three functions already filled are short-circuited (no LLM call).
    """
    if not rows_data:
        return []

    results: list[dict | None] = [None] * len(rows_data)
    need_llm: list[int] = []

    for i, rd in enumerate(rows_data):
        if rd.get("existing_focus_fn") and rd.get("existing_higher_fn") and rd.get("existing_lower_fn"):
            results[i] = {
                "focus_fn":  rd["existing_focus_fn"],
                "higher_fn": rd["existing_higher_fn"],
                "lower_fn":  rd["existing_lower_fn"],
            }
        else:
            need_llm.append(i)

    if not need_llm:
        return results  # type: ignore[return-value]

    rows_text = ""
    for seq, i in enumerate(need_llm, 1):
        rd = rows_data[i]
        rows_text += f"""
Row {seq}:
  higher_element    : "{rd.get('higher_element', '')}"
  focus_element     : "{rd.get('focus_element', '')}"
  lower_element     : "{rd.get('lower_element', '')}"
  failure_mode      : "{rd.get('failure_mode', '')}"
  failure_effect    : "{rd.get('failure_effect', '')}"
  failure_cause     : "{rd.get('failure_cause', '')}"
  requirement       : "{rd.get('requirement', '')}"
  existing_focus_fn : "{rd.get('existing_focus_fn', '')}"
  existing_higher_fn: "{rd.get('existing_higher_fn', '')}"
  existing_lower_fn : "{rd.get('existing_lower_fn', '')}"
"""

    prompt = f"""Write DFMEA function statements for these rows. Each row has its own
element names — use those exact names, not generic labels.

Rules:
- focus_fn  : positive function of focus_element that the failure_mode violates
- higher_fn : positive function of higher_element impaired by failure_effect
- lower_fn  : positive function of lower_element negated by failure_cause
- Format: verb + object, ≤12 words (e.g. "Transmit braking torque")
- If existing_*_fn is non-empty, echo it back unchanged

Return a JSON ARRAY of {len(need_llm)} objects, one per row, in order:
[
  {{"focus_fn": "...", "higher_fn": "...", "lower_fn": "..."}},
  ...
]

Rows:
{rows_text}

Output ONLY valid JSON array."""

    token_budget = min(8192, max(512, len(need_llm) * 130 + 256))
    raw = _llm(prompt, "llm_batch_derive_row_functions", max_tokens=token_budget)

    try:
        parsed_list = json.loads(_strip_json(raw))
        if isinstance(parsed_list, list):
            for seq, i in enumerate(need_llm):
                if seq < len(parsed_list):
                    results[i] = {
                        "focus_fn":  parsed_list[seq].get("focus_fn", ""),
                        "higher_fn": parsed_list[seq].get("higher_fn", ""),
                        "lower_fn":  parsed_list[seq].get("lower_fn", ""),
                    }
                else:
                    results[i] = _llm_derive_single(rows_data[i])
            return results  # type: ignore[return-value]
    except Exception:
        pass

    print("[llm WARNING] llm_batch_derive_row_functions: parse failed, per-row fallback.")
    for i in need_llm:
        results[i] = _llm_derive_single(rows_data[i])
    return results  # type: ignore[return-value]


def _llm_derive_single(rd: dict) -> dict:
    """Single-row fallback for function derivation."""
    focus_el  = rd.get("focus_element",  "focus element")
    higher_el = rd.get("higher_element", "higher element")
    lower_el  = rd.get("lower_element",  "lower element")

    fields: list[str] = []
    if not rd.get("existing_focus_fn"):
        fields.append(f'  "focus_fn": "<function of {focus_el}>"')
    if not rd.get("existing_higher_fn"):
        fields.append(f'  "higher_fn": "<function of {higher_el}>"')
    if not rd.get("existing_lower_fn"):
        fields.append(f'  "lower_fn": "<function of {lower_el}>"')

    if not fields:
        return {
            "focus_fn":  rd["existing_focus_fn"],
            "higher_fn": rd["existing_higher_fn"],
            "lower_fn":  rd["existing_lower_fn"],
        }

    prompt = f"""Derive DFMEA function statements.

Higher: "{higher_el}" | Focus: "{focus_el}" | Lower: "{lower_el}"
Failure mode  : "{rd.get('failure_mode', '')}"
Failure effect: "{rd.get('failure_effect', '')}"
Failure cause : "{rd.get('failure_cause', '')}"
Requirement   : "{rd.get('requirement', '')}"

Output ONLY valid JSON:
{{
{chr(10).join(fields)}
}}"""
    raw    = _llm(prompt, "_llm_derive_single", max_tokens=len(fields) * 80 + 100)
    parsed = _parse_json(raw)
    return {
        "focus_fn":  parsed.get("focus_fn",  rd.get("existing_focus_fn",  "")),
        "higher_fn": parsed.get("higher_fn", rd.get("existing_higher_fn", "")),
        "lower_fn":  parsed.get("lower_fn",  rd.get("existing_lower_fn",  "")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT A PARSER — AIAG-VDA 2019
# ─────────────────────────────────────────────────────────────────────────────

def parse_aiag_vda(df: pd.DataFrame, col: dict[str, int], elements: dict) -> dict:
    """
    Parse AIAG-VDA 2019 format.

    Per-row element resolution:
      - focus_element  : always user-supplied single string
      - higher_element : read from 'next higher level element' column → matched to
                         elements['higher_elements'] list via fuzzy + LLM batch
      - lower_element  : read from 'next lower level element' column  → matched to
                         elements['lower_elements']  list via fuzzy + LLM batch

    Functions read from explicit columns if present, else LLM-derived (batch).
    Noise factors derived per-row (individual LLM call per cause).
    Connections built from row co-occurrence with fully qualified element names.
    """
    focus_element     = elements["focus_element"]
    higher_candidates = elements["higher_elements"]
    lower_candidates  = elements["lower_elements"]

    has_higher_col = "higher_element_col" in col
    has_lower_col  = "lower_element_col"  in col

    # ── Pass 1: Collect raw rows ──────────────────────────────────────────────
    raw_rows: list[dict] = []
    for _, row in df.iterrows():
        higher_cell       = _s(row.get(col.get("higher_element_col", -1), "")) if has_higher_col else ""
        lower_cell        = _s(row.get(col.get("lower_element_col",  -1), "")) if has_lower_col  else ""
        higher_fn_exist   = _s(row.get(col.get("higher_fn",          -1), ""))
        focus_fn_exist    = _s(row.get(col.get("focus_fn",           -1), ""))
        lower_fn_exist    = _s(row.get(col.get("lower_fn",           -1), ""))
        fe   = _s(row.get(col.get("failure_effect",  -1), ""))
        sev  = _int(row.get(col.get("severity",      -1), ""))
        fm   = _s(row.get(col.get("failure_mode",    -1), ""))
        fc   = _s(row.get(col.get("failure_cause",   -1), ""))
        prev = _s(row.get(col.get("prevention",      -1), ""))
        occ  = _int(row.get(col.get("occurrence",    -1), ""))
        dc   = _s(row.get(col.get("detection_ctrl",  -1), ""))
        det  = _int(row.get(col.get("detection",     -1), ""))
        cls  = _s(row.get(col.get("classification",  -1), ""))
        ap   = _s(row.get(col.get("ap",              -1), ""))

        if not fm and not fc:
            continue

        raw_rows.append({
            "_higher_cell":       higher_cell,
            "_lower_cell":        lower_cell,
            "existing_focus_fn":  focus_fn_exist,
            "existing_higher_fn": higher_fn_exist,
            "existing_lower_fn":  lower_fn_exist,
            "failure_mode":       fm,
            "failure_effect":     fe,
            "failure_cause":      fc,
            "requirement":        "",
            "_severity":          sev,
            "_classification":    cls,
            "_prevention":        prev,
            "_occurrence":        occ,
            "_det_ctrl":          dc,
            "_detection":         det,
            "_ap":                ap,
        })

    if not raw_rows:
        return {"error": "No data rows found"}

    # ── Pass 2: Batch-resolve element names ───────────────────────────────────
    unique_higher = list({r["_higher_cell"] for r in raw_rows if r["_higher_cell"]})
    unique_lower  = list({r["_lower_cell"]  for r in raw_rows if r["_lower_cell"]})

    higher_map = llm_batch_match_elements(unique_higher, higher_candidates, level="higher") if unique_higher and higher_candidates else {}
    lower_map  = llm_batch_match_elements(unique_lower,  lower_candidates,  level="lower")  if unique_lower  and lower_candidates  else {}

    # ── Pass 3: Build LLM function-derivation inputs ──────────────────────────
    llm_inputs: list[dict] = []
    for r in raw_rows:
        he = higher_map.get(r["_higher_cell"], higher_candidates[0] if higher_candidates else "Higher System")
        le = lower_map.get( r["_lower_cell"],  lower_candidates[0]  if lower_candidates  else "Lower Component")
        r["_higher_element"] = he
        r["_lower_element"]  = le
        llm_inputs.append({
            "focus_element":      focus_element,
            "higher_element":     he,
            "lower_element":      le,
            "failure_mode":       r["failure_mode"],
            "failure_effect":     r["failure_effect"],
            "failure_cause":      r["failure_cause"],
            "requirement":        r["requirement"],
            "existing_focus_fn":  r["existing_focus_fn"],
            "existing_higher_fn": r["existing_higher_fn"],
            "existing_lower_fn":  r["existing_lower_fn"],
        })

    # ── Pass 4: Batch derive functions ────────────────────────────────────────
    fn_results = llm_batch_derive_row_functions(llm_inputs)

    # ── Pass 5: Build output rows + connections ────────────────────────────────
    rows_out:    list[dict] = []
    connections: list[dict] = []
    noise_acc:   dict[str, set] = {c: set() for c in _NOISE_CATS}

    for row_idx, (r, fns) in enumerate(zip(raw_rows, fn_results)):
        fns = fns or {"focus_fn": "", "higher_fn": "", "lower_fn": ""}
        fc  = r["failure_cause"]
        ni  = classify_cause_noise(fc, use_llm=bool(_BEDROCK_KEY))
        if ni["noise_driven"] and ni["noise_category"] and ni["noise_factor"]:
            noise_acc[ni["noise_category"]].add(ni["noise_factor"])

        sev = r["_severity"]
        occ = r["_occurrence"]
        det = r["_detection"]
        rpn = (sev or 0) * (occ or 0) * (det or 0) or None

        rows_out.append({
            "id":                  _uid(),
            "focus_element":       focus_element,
            "higher_element":      r["_higher_element"],
            "lower_element":       r["_lower_element"],
            "higher_fn":           fns["higher_fn"],
            "focus_fn":            fns["focus_fn"],
            "lower_fn":            fns["lower_fn"],
            "failure_effect":      r["failure_effect"],
            "severity":            sev,
            "classification":      r["_classification"],
            "failure_mode":        r["failure_mode"],
            "failure_cause":       fc,
            "prevention_controls": r["_prevention"],
            "occurrence":          occ,
            "detection_controls":  r["_det_ctrl"],
            "detection":           det,
            "rpn":                 rpn,
            "ap":                  r["_ap"],
            "noise_driven":        ni["noise_driven"],
            "noise_category":      ni["noise_category"],
            "noise_factor":        ni["noise_factor"],
            "row_index":           row_idx,
        })

        if fns["lower_fn"] and fns["focus_fn"] and fns["higher_fn"]:
            connections.append({
                "lower_element":  r["_lower_element"],
                "lower_fn":       fns["lower_fn"],
                "focus_fn":       fns["focus_fn"],
                "higher_element": r["_higher_element"],
                "higher_fn":      fns["higher_fn"],
                "row_index":      row_idx,
            })

    return _assemble_output(rows_out, noise_acc, connections, focus_element, fmt="aiag_vda")


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT B PARSER — Legacy Ford/GM/Chrysler
# ─────────────────────────────────────────────────────────────────────────────

def parse_legacy(df: pd.DataFrame, col: dict[str, int], elements: dict) -> dict:
    """
    Parse legacy Ford/GM/Chrysler format.  No explicit element columns.

    Per-row element resolution (2 LLM calls per row — no batching):
      Call 1: Parse focus / lower / higher element names from row text,
              then match each to user-supplied candidate lists.
      Call 2: Derive focus_fn / lower_fn / higher_fn, preserving connections.
      Call 3: Classify noise factors from failure cause.

    Output: elements at each level with their functions, plus noise factors.
    """
    focus_element     = elements["focus_element"]
    higher_candidates = elements["higher_elements"]
    lower_candidates  = elements["lower_elements"]

    # ── Pass 1: Forward-fill item / requirement columns ───────────────────────
    item_col_idx = col.get("item_function", 0)
    req_col_idx  = col.get("requirement",   1)
    fe_col_idx   = col.get("failure_effect", -1)

    items_filled: list[str] = []
    cur = ""
    for v in df.iloc[:, item_col_idx].apply(_s):
        if v: cur = v
        items_filled.append(cur)

    reqs_filled: list[str] = []
    cur = ""
    for v in df.iloc[:, req_col_idx].apply(_s):
        if v: cur = v
        reqs_filled.append(cur)

    # ── Pass 2: Collect raw rows ──────────────────────────────────────────────
    raw_rows: list[dict] = []
    for ri in range(len(df)):
        fm  = _s(df.iloc[ri, col["failure_mode"]])    if "failure_mode"   in col else ""
        fe  = _s(df.iloc[ri, fe_col_idx])             if fe_col_idx >= 0   else ""
        sev = df.iloc[ri, col["severity"]]            if "severity"        in col else None
        fc  = _s(df.iloc[ri, col["failure_cause"]])   if "failure_cause"   in col else ""
        prv = _s(df.iloc[ri, col["prevention"]])      if "prevention"      in col else ""
        occ = df.iloc[ri, col["occurrence"]]          if "occurrence"      in col else None
        dc  = _s(df.iloc[ri, col["detection_ctrl"]])  if "detection_ctrl"  in col else ""
        det = df.iloc[ri, col["detection"]]           if "detection"       in col else None
        rpn = df.iloc[ri, col["rpn"]]                 if "rpn"             in col else None
        cls = _s(df.iloc[ri, col["classification"]])  if "classification"  in col else ""
        item = items_filled[ri]
        req  = reqs_filled[ri]

        if not item or (not fm and not fc):
            continue

        si = _int(sev); oi = _int(occ); di = _int(det)
        ri_ = _int(rpn) or ((si or 0) * (oi or 0) * (di or 0) or None)

        raw_rows.append({
            "failure_mode":    fm,
            "failure_effect":  fe,
            "failure_cause":   fc,
            "requirement":     req,
            "_item":           item,
            "_severity":       si,
            "_classification": cls,
            "_prevention":     prv,
            "_occurrence":     oi,
            "_det_ctrl":       dc,
            "_detection":      di,
            "_rpn":            ri_,
            "_row_index":      ri,
        })

    if not raw_rows:
        return {"error": "No data rows found"}

    # ── Pass 3: Per-row: LLM call 1 (element names) + LLM call 2 (functions) ──
    # Two separate LLM calls per row — no batching — to avoid timeout issues.
    rows_out:    list[dict] = []
    connections: list[dict] = []
    noise_acc:   dict[str, set] = {c: set() for c in _NOISE_CATS}

    for row_idx, r in enumerate(raw_rows):
        # ── LLM Call 1: Parse + match element names ───────────────────────────
        el_result = _llm_legacy_parse_elements(
            row=r,
            focus_element=focus_element,
            higher_candidates=higher_candidates,
            lower_candidates=lower_candidates,
        )
        he = el_result["higher_element"]
        le = el_result["lower_element"]
        r["_higher_element"] = he
        r["_lower_element"]  = le

        # ── LLM Call 2: Derive function statements ────────────────────────────
        fns = _llm_legacy_parse_functions(
            row=r,
            focus_element=focus_element,
            higher_element=he,
            lower_element=le,
        )

        # ── LLM Call 3: Classify noise factors ───────────────────────────────
        fc = r["failure_cause"]
        ni = classify_cause_noise(fc, use_llm=bool(_BEDROCK_KEY))
        if ni["noise_driven"] and ni["noise_category"] and ni["noise_factor"]:
            noise_acc[ni["noise_category"]].add(ni["noise_factor"])

        rows_out.append({
            "id":                  _uid(),
            "focus_element":       focus_element,
            "higher_element":      he,
            "lower_element":       le,
            "higher_fn":           fns["higher_fn"],
            "focus_fn":            fns["focus_fn"],
            "lower_fn":            fns["lower_fn"],
            "failure_effect":      r["failure_effect"],
            "severity":            r["_severity"],
            "classification":      r["_classification"],
            "failure_mode":        r["failure_mode"],
            "failure_cause":       fc,
            "prevention_controls": r["_prevention"],
            "occurrence":          r["_occurrence"],
            "detection_controls":  r["_det_ctrl"],
            "detection":           r["_detection"],
            "rpn":                 r["_rpn"],
            "ap":                  None,
            "noise_driven":        ni["noise_driven"],
            "noise_category":      ni["noise_category"],
            "noise_factor":        ni["noise_factor"],
            "row_index":           r["_row_index"],
        })

        if fns["lower_fn"] and fns["focus_fn"] and fns["higher_fn"]:
            connections.append({
                "lower_element":  le,
                "lower_fn":       fns["lower_fn"],
                "focus_fn":       fns["focus_fn"],
                "higher_element": he,
                "higher_fn":      fns["higher_fn"],
                "row_index":      r["_row_index"],
            })

        print(
            f"[legacy] row {row_idx + 1}/{len(raw_rows)}  "
            f"lower={le!r}  higher={he!r}  "
            f"focus_fn={fns['focus_fn'][:40]!r}"
        )

    return _assemble_output(rows_out, noise_acc, connections, focus_element, fmt="legacy")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_connections(connections: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out:  list[dict] = []
    for c in connections:
        key = (c["lower_element"], c["lower_fn"], c["focus_fn"], c["higher_element"], c["higher_fn"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _assemble_output(
    rows: list[dict],
    noise_acc: dict[str, set],
    connections: list[dict],
    focus_element: str,
    fmt: str,
) -> dict:
    if not rows:
        return {"error": "No data rows found"}

    # ── Focus functions ───────────────────────────────────────────────────────
    focus_functions = sorted(set(r["focus_fn"] for r in rows if r["focus_fn"]))

    # ── Higher elements → functions (one entry per unique element name) ────────
    higher_map: dict[str, set] = defaultdict(set)
    for r in rows:
        he = r["higher_element"]
        if he:
            if r["higher_fn"]:
                higher_map[he].add(r["higher_fn"])
            else:
                higher_map[he]           # ensure key exists even without functions

    higher_elements = [
        {"name": el, "functions": sorted(fns)}
        for el, fns in sorted(higher_map.items())
    ]

    # ── Lower elements → functions ─────────────────────────────────────────────
    lower_map: dict[str, set] = defaultdict(set)
    for r in rows:
        le = r["lower_element"]
        if le:
            if r["lower_fn"]:
                lower_map[le].add(r["lower_fn"])
            else:
                lower_map[le]

    lower_elements = [
        {"name": el, "functions": sorted(fns)}
        for el, fns in sorted(lower_map.items())
    ]

    # ── Connections (fully qualified with element names) ───────────────────────
    unique_connections = _dedup_connections(connections)

    # ── Failure modes grouped by (focus_fn, failure_mode) ─────────────────────
    mode_map: dict[tuple, dict] = {}
    for r in rows:
        key = (r["focus_fn"], r["failure_mode"])
        if key not in mode_map:
            mode_map[key] = {
                "id":             _uid(),
                "focus_fn":       r["focus_fn"],
                "failure_mode":   r["failure_mode"],
                "failure_effect": r["failure_effect"],
                "severity":       r["severity"],
                "classification": r["classification"],
                "causes":         [],
            }
        ce = {
            "id":                  r["id"],
            "lower_element":       r["lower_element"],
            "lower_fn":            r["lower_fn"],
            "higher_element":      r["higher_element"],
            "higher_fn":           r["higher_fn"],
            "cause_text":          r["failure_cause"],
            "prevention_controls": r["prevention_controls"],
            "detection_controls":  r["detection_controls"],
            "occurrence":          r["occurrence"],
            "detection":           r["detection"],
            "rpn":                 r["rpn"],
            "ap":                  r["ap"],
            "noise_driven":        r["noise_driven"],
            "noise_category":      r["noise_category"],
            "noise_factor":        r["noise_factor"],
        }
        if ce["cause_text"] not in {c["cause_text"] for c in mode_map[key]["causes"]}:
            mode_map[key]["causes"].append(ce)

    failure_modes = list(mode_map.values())

    # ── Noise factors ─────────────────────────────────────────────────────────
    noise_factors = {cat: sorted(items) for cat, items in noise_acc.items()}

    # ── S/O/D summary per lower element ───────────────────────────────────────
    el_sod: dict[str, dict] = defaultdict(lambda: {"sev": [], "occ": [], "det": [], "rpn": []})
    for r in rows:
        el = r["lower_element"] or focus_element
        if r["severity"]  is not None: el_sod[el]["sev"].append(r["severity"])
        if r["occurrence"] is not None: el_sod[el]["occ"].append(r["occurrence"])
        if r["detection"]  is not None: el_sod[el]["det"].append(r["detection"])
        if r["rpn"]        is not None: el_sod[el]["rpn"].append(r["rpn"])

    sod_by_element = {
        el: {
            "max_severity":   max(v["sev"]) if v["sev"] else None,
            "avg_occurrence": round(sum(v["occ"]) / len(v["occ"]), 1) if v["occ"] else None,
            "avg_detection":  round(sum(v["det"]) / len(v["det"]), 1) if v["det"] else None,
            "max_rpn":        max(v["rpn"]) if v["rpn"] else None,
        }
        for el, v in el_sod.items()
    }

    return {
        "format_detected":  fmt,
        "focus_element":    focus_element,
        "higher_elements":  higher_elements,    # [{name, functions[]}]
        "lower_elements":   lower_elements,     # [{name, functions[]}]
        "focus_functions":  focus_functions,
        "connections":      unique_connections, # [{lower_element, lower_fn, focus_fn, higher_element, higher_fn, row_index}]
        "failure_modes":    failure_modes,
        "noise_factors":    noise_factors,
        "sod_by_element":   sod_by_element,
        "raw_rows":         rows,
        "stats": {
            "total_rows":            len(rows),
            "failure_mode_count":    len(failure_modes),
            "focus_function_count":  len(focus_functions),
            "higher_element_count":  len(higher_elements),
            "lower_element_count":   len(lower_elements),
            "connection_count":      len(unique_connections),
            "noise_driven_count":    sum(1 for r in rows if r["noise_driven"]),
            "design_cause_count":    sum(1 for r in rows if not r["noise_driven"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHEET FINDER
# ─────────────────────────────────────────────────────────────────────────────

_DFMEA_SHEET_KEYWORDS = ["dfmea", "worksheet", "fmea", "failure mode"]


def find_dfmea_sheet(xl: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    for name, df in xl.items():
        if any(kw in name.lower() for kw in _DFMEA_SHEET_KEYWORDS):
            return name, df
    return max(xl.items(), key=lambda kv: len(kv[1]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_dfmea_file(
    path: str,
    elements: dict,
    sheet_name: str | None = None,
) -> dict:
    """
    Universal parser entry point.

    Parameters
    ----------
    path     : path to the xlsx file
    elements : {
                  focus_element:   str        — single focus system name
                  higher_elements: list[str]  — one or more higher system names
                  lower_elements:  list[str]  — one or more lower component names
               }
               ALL values are user-supplied before calling this function.
               The parser resolves which element applies to each row.
    sheet_name : optional — if None, auto-detected
    """
    required = ("focus_element", "higher_elements", "lower_elements")
    if not elements or not all(k in elements for k in required):
        raise ValueError(
            "elements dict must have: focus_element (str), "
            "higher_elements (list[str]), lower_elements (list[str])"
        )
    if not isinstance(elements["higher_elements"], list) or not elements["higher_elements"]:
        raise ValueError("elements['higher_elements'] must be a non-empty list.")
    if not isinstance(elements["lower_elements"], list) or not elements["lower_elements"]:
        raise ValueError("elements['lower_elements'] must be a non-empty list.")

    xl = pd.read_excel(path, sheet_name=None, header=None)
    if sheet_name and sheet_name in xl:
        df_raw = xl[sheet_name]
    else:
        sheet_name, df_raw = find_dfmea_sheet(xl)

    print(f"[parser] Sheet: '{sheet_name}'  shape={df_raw.shape}")
    print(f"[parser] Focus   : {elements['focus_element']}")
    print(f"[parser] Higher  : {elements['higher_elements']}")
    print(f"[parser] Lower   : {elements['lower_elements']}")

    hdr_idx       = find_header_row(df_raw)
    header_values = [_s(v) for v in df_raw.iloc[hdr_idx].tolist()]
    data_df       = df_raw.iloc[hdr_idx + 1:].reset_index(drop=True)
    data_df.columns = range(data_df.shape[1])
    data_df       = data_df.dropna(how="all").reset_index(drop=True)

    fmt = detect_format(header_values)
    col = map_columns(header_values, fmt)
    print(f"[parser] Format={fmt}  Columns={col}")

    result = (parse_aiag_vda if fmt == "aiag_vda" else parse_legacy)(data_df, col, elements)
    result["source_file"] = Path(path).name
    result["sheet_name"]  = sheet_name
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CASE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_case1_new_conditions(parsed: dict) -> dict:
    """
    Case 1 — same design, new operating environment.
    Retains all elements, functions, and connections.
    Strips noise-driven causes (replaced in P-Diagram step).
    Clears O/D ratings; retains S ratings.
    """
    rows = parsed.get("raw_rows", [])
    seen: set[tuple] = set()
    modes_out = []
    for r in rows:
        key = (r["focus_fn"], r["failure_mode"])
        seen.add(key)
        is_noise = r.get("noise_driven", True)
        modes_out.append({
            "focus_fn":       r["focus_fn"],
            "failure_mode":   r["failure_mode"],
            "failure_effect": r["failure_effect"],
            "severity":       r["severity"],
            "cause":          None if is_noise else r["failure_cause"],
            "lower_element":  r["lower_element"],
            "lower_fn":       r["lower_fn"],
            "higher_element": r["higher_element"],
            "higher_fn":      r["higher_fn"],
            "occurrence":     None,
            "detection":      None,
            "noise_driven":   is_noise,
            "noise_label":    r.get("noise_factor"),
        })

    stripped = sorted(set(
        r["noise_factor"] for r in rows if r.get("noise_driven") and r.get("noise_factor")
    ))

    return {
        "case":            "new_conditions",
        "focus_element":   parsed["focus_element"],
        "higher_elements": parsed["higher_elements"],
        "lower_elements":  parsed["lower_elements"],
        "focus_functions": parsed["focus_functions"],
        "connections":     parsed["connections"],
        "failure_modes":   modes_out,
        "noise_stub": {
            "stripped_factors": stripped,
            "instructions": (
                "Replace these with your new operating-environment noise factors "
                "in the P-Diagram step."
            ),
        },
        "stats": {
            "modes":           len(seen),
            "connections":     len(parsed["connections"]),
            "higher_elements": len(parsed["higher_elements"]),
            "lower_elements":  len(parsed["lower_elements"]),
            "causes_retained": sum(1 for r in rows if not r.get("noise_driven")),
            "causes_stripped": sum(1 for r in rows if r.get("noise_driven")),
        },
    }


def build_case2_modified_design(parsed: dict, modified_elements: list[str] | None = None) -> dict:
    """
    Case 2 — same environment, modified component(s).
    Strips design-mechanism causes for modified lower elements.
    Noise-driven causes always retained.
    User must re-define modified elements before LLM re-generation.
    """
    rows = parsed.get("raw_rows", [])
    mod_set = set(modified_elements or [])

    modes_out = []
    for r in rows:
        is_mod   = bool(mod_set) and r["lower_element"] in mod_set
        is_noise = r.get("noise_driven", False)
        retain   = is_noise or not is_mod
        modes_out.append({
            "focus_fn":         r["focus_fn"],
            "failure_mode":     r["failure_mode"],
            "failure_effect":   r["failure_effect"],
            "severity":         r["severity"],
            "cause":            r["failure_cause"] if retain else None,
            "lower_element":    r["lower_element"],
            "lower_fn":         r["lower_fn"],
            "higher_element":   r["higher_element"],
            "higher_fn":        r["higher_fn"],
            "occurrence":       r["occurrence"] if retain else None,
            "detection":        r["detection"]  if retain else None,
            "rpn":              r["rpn"]         if retain else None,
            "noise_driven":     is_noise,
            "retained":         retain,
            "needs_regen":      not retain,
            "modified_element": is_mod,
        })

    return {
        "case":             "modified_design",
        "focus_element":    parsed["focus_element"],
        "higher_elements":  parsed["higher_elements"],
        "lower_elements":   parsed["lower_elements"],
        "focus_functions":  parsed["focus_functions"],
        "connections":      parsed["connections"],
        "noise_factors":    parsed["noise_factors"],
        "failure_modes":    modes_out,
        "modified_elements": list(mod_set) or "user_to_specify",
        "user_action_required": (
            "Re-define modified elements and their functions before "
            "regenerating failure causes."
        ),
        "stats": {
            "total_rows":          len(rows),
            "connections":         len(parsed["connections"]),
            "higher_elements":     len(parsed["higher_elements"]),
            "lower_elements":      len(parsed["lower_elements"]),
            "causes_retained":     sum(1 for m in modes_out if m["retained"]),
            "causes_stripped":     sum(1 for m in modes_out if not m["retained"]),
            "modes_needing_regen": sum(1 for m in modes_out if m["needs_regen"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(parsed: dict) -> None:
    print("\n" + "=" * 72)
    print(f"  UNIVERSAL DFMEA PARSER REPORT")
    print(f"  Source : {parsed.get('source_file', '—')}  |  Sheet: {parsed.get('sheet_name','—')}")
    print(f"  Format : {parsed.get('format_detected', '—').upper()}")
    print("=" * 72)

    print(f"\n  Focus: {parsed['focus_element']}")

    print(f"\n  Higher elements ({len(parsed['higher_elements'])}):")
    for he in parsed["higher_elements"]:
        print(f"    · {he['name']}")
        for fn in he["functions"][:3]:
            print(f"        – {fn[:80]}")

    print(f"\n  Lower elements ({len(parsed['lower_elements'])}):")
    for le in parsed["lower_elements"]:
        sod = parsed.get("sod_by_element", {}).get(le["name"], {})
        print(f"    · {le['name']}  "
              f"[S={sod.get('max_severity')} O={sod.get('avg_occurrence')} "
              f"D={sod.get('avg_detection')} RPN={sod.get('max_rpn')}]")
        for fn in le["functions"][:2]:
            print(f"        – {fn[:80]}")

    print(f"\n  Focus functions ({len(parsed['focus_functions'])}):")
    for fn in parsed["focus_functions"][:5]:
        print(f"    · {fn[:80]}")

    print(f"\n  Connections [{len(parsed['connections'])} unique]:")
    for c in parsed["connections"][:5]:
        print(f"    [row {c['row_index']}]  {c['lower_element']} → focus → {c['higher_element']}")
        print(f"      {c['lower_fn'][:55]}")
        print(f"      → {c['focus_fn'][:55]}")
        print(f"      → {c['higher_fn'][:55]}")
    if len(parsed["connections"]) > 5:
        print(f"    … {len(parsed['connections']) - 5} more")

    print(f"\n  Failure modes ({len(parsed['failure_modes'])}):")
    for m in parsed["failure_modes"][:5]:
        cs = m["causes"]
        print(f"    · {m['failure_mode'][:60]}  (S={m['severity']})")
        print(f"      {len(cs)} causes  ({sum(1 for c in cs if c.get('noise_driven'))} noise)")
    if len(parsed["failure_modes"]) > 5:
        print(f"    … {len(parsed['failure_modes']) - 5} more")

    print(f"\n  Noise factors:")
    for cat, facs in parsed["noise_factors"].items():
        if facs:
            print(f"    [{cat}]: {', '.join(facs[:4])}")

    s = parsed["stats"]
    print(
        f"\n  {s['total_rows']} rows | {s['failure_mode_count']} modes | "
        f"{s['focus_function_count']} focus fns | "
        f"{s['higher_element_count']} higher els | {s['lower_element_count']} lower els | "
        f"{s['connection_count']} connections | "
        f"{s['noise_driven_count']} noise / {s['design_cause_count']} design causes"
    )
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTER  (uncomment and register in main.py)
# ─────────────────────────────────────────────────────────────────────────────

# from fastapi import APIRouter, UploadFile, File, Form
# import tempfile, shutil
#
# router = APIRouter()
#
# @router.post("/api/dfmea/import/parse")
# async def import_parse(
#     file:              UploadFile = File(...),
#     focus_element:     str = Form(...),   # wizard Step 1 — single string
#     higher_elements:   str = Form(...),   # comma-separated, e.g. "Chassis,Frame"
#     lower_elements:    str = Form(...),   # comma-separated, e.g. "Differential,Hub,Axle Shaft"
#     case:              str = Form("new_conditions"),
#     modified_elements: str = Form(""),    # comma-separated subset of lower_elements
#     sheet_name:        str = Form(""),
# ):
#     elements = {
#         "focus_element":   focus_element,
#         "higher_elements": [e.strip() for e in higher_elements.split(",") if e.strip()],
#         "lower_elements":  [e.strip() for e in lower_elements.split(",")  if e.strip()],
#     }
#     with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name
#     parsed  = parse_dfmea_file(tmp_path, elements=elements, sheet_name=sheet_name or None)
#     mod_els = [e.strip() for e in modified_elements.split(",") if e.strip()]
#     return (build_case2_modified_design(parsed, mod_els or None)
#             if case == "modified_design" else build_case1_new_conditions(parsed))


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path  = sys.argv[1] if len(sys.argv) > 1 else (
        "/mnt/user-data/uploads/1-Chassis_System-Axle_System_DFMEA_v4_Final__3_.xlsx"
    )
    sheet = sys.argv[2] if len(sys.argv) > 2 else None

    print("\n[parser] Collect element names (from wizard Step 1 in production)\n")
    focus_el  = input("  Focus element name                         : ").strip() or "Focus System"
    higher_raw = input("  Higher element names (comma-separated)     : ").strip()
    lower_raw  = input("  Lower element names  (comma-separated)     : ").strip()

    elements = {
        "focus_element":   focus_el,
        "higher_elements": [e.strip() for e in higher_raw.split(",") if e.strip()] or ["Higher System"],
        "lower_elements":  [e.strip() for e in lower_raw.split(",")  if e.strip()] or ["Lower Component"],
    }

    parsed = parse_dfmea_file(path, elements=elements, sheet_name=sheet)
    print_report(parsed)

    out = Path("/outputs")
    out.mkdir(exist_ok=True)
    (out / "universal_parsed_full.json").write_text(
        json.dumps(parsed, indent=2, default=str), encoding="utf-8"
    )
    (out / "universal_case1_new_conditions.json").write_text(
        json.dumps(build_case1_new_conditions(parsed), indent=2, default=str), encoding="utf-8"
    )
    (out / "universal_case2_modified_design.json").write_text(
        json.dumps(build_case2_modified_design(parsed), indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[parser] Outputs written to {out}")
