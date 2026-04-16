""" dfmea_universal_parser.py
=========================
Universal DFMEA xlsx parser.  Supports two formats automatically detected by
header fingerprinting:

  FORMAT A — AIAG-VDA 2019  (IIT Madras / Ashok Leyland new format)
    Explicit higher / focus / lower element columns.
    Function columns present for all three levels.
    LLM used for: deriving functions from row data + noise factor classification.

  FORMAT B — Legacy Ford/GM/Chrysler  (Ashok Leyland production format)
    Single "Item / Function" column — focus system + numbered sub-items.
    "Requirement" column = function statement.
    No higher-level element column.
    LLM used for:
      - Derive focus function from failure mode / requirement text
      - Extract lower-level function from cause description
      - Derive higher-level function from failure effect text
      - Classify noise factors from cause text

IMPORTANT: Element names (focus, higher, lower) are NOT inferred by the parser.
They must be provided by the caller via the `elements` parameter to
`parse_dfmea_file()`. The parser derives FUNCTIONS and CONNECTIONS from row data.

Both formats produce the same output schema:
  {
    focus_element:    str,
    higher_element:   str,
    lower_element:    str,
    focus_functions:  str[],
    higher_functions: str[],
    lower_functions:  str[],
    connections: [
      {
        lower_fn:     str,   # what lower function…
        focus_fn:     str,   # …affects what focus function…
        higher_fn:    str,   # …which affects what higher function
        row_index:    int,   # source row for traceability
      }
    ],
    failure_modes: [{
      focus_fn, failure_mode, failure_effect,
      severity, occurrence, detection, rpn,
      classification, prevention_controls, detection_controls,
      causes: [{
        lower_fn, cause_text,
        noise_driven, noise_category, noise_factor,
        occurrence, detection
      }]
    }],
    noise_factors: {
      pieceTopiece, changeOverTime, customerUsage,
      externalEnvironment, systemInteractions
    }
  }

Usage:
  python dfmea_universal_parser.py <path_to_xlsx> [sheet_name]
  # Elements must be passed programmatically via parse_dfmea_file(path, elements={...})
"""

from __future__ import annotations

import re
import json
import uuid
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LLM CLIENT  (same interface as existing llm_client.py)
# ─────────────────────────────────────────────────────────────────────────────

import os
import requests

_BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
_BEDROCK_MODEL  = os.environ.get("BEDROCK_MODEL",  "anthropic.claude-3-sonnet-20240229-v1:0")
_BEDROCK_KEY    = os.environ.get("BEDROCK_API_KEY", "")
_BEDROCK_URL    = f"https://bedrock-runtime.{_BEDROCK_REGION}.amazonaws.com/model/{_BEDROCK_MODEL}/invoke"

_SYSTEM = (
    "You are an expert DFMEA engineer following AIAG-VDA methodology. "
    "Return ONLY what is asked — no preamble, no markdown fences, no explanation."
)


from typing import Iterable, List

def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _llm(prompt: str, function: str, max_tokens: int = 1024) -> str:
    """Call AWS Bedrock. Falls back to stub if no key is set (for testing)."""
    if not _BEDROCK_KEY:
        return _llm_stub(prompt)
    headers = {"Authorization": f"Bearer {_BEDROCK_KEY}", "Content-Type": "application/json"}
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "system":  _SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(_BEDROCK_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()

    stop_reason = body.get("stop_reason", "")
    if stop_reason == "max_tokens":
        print(
            f"[llm WARNING] '{function}' hit max_tokens={max_tokens} — "
            "output was TRUNCATED. Increase max_tokens for this call."
        )

    text = body["content"][0]["text"].strip()
    print(f"[llm] {function} ({max_tokens} tok cap, stop={stop_reason!r}): {text[:120]}{'…' if len(text) > 120 else ''}")
    return text


def _llm_stub(prompt: str) -> str:
    """Offline stub — returns deterministic placeholder JSON."""
    if "focus function" in prompt.lower() and "failure mode" in prompt.lower():
        return "Perform intended function"
    if "lower" in prompt.lower() and "function" in prompt.lower() and "cause" in prompt.lower():
        return json.dumps({
            "lower_fn": "Perform sub-component function",
        })
    if "higher" in prompt.lower() and "function" in prompt.lower() and "effect" in prompt.lower():
        return json.dumps({
            "higher_fn": "Maintain system-level operation",
        })
    if "noise" in prompt.lower():
        return json.dumps({
            "noise_driven": False,
            "noise_category": None,
            "noise_factor": None,
        })
    if "functions" in prompt.lower() and "batch" in prompt.lower():
        return json.dumps([
            {"focus_fn": "Perform intended function", "higher_fn": "Maintain system operation", "lower_fn": "Perform sub-function"}
        ])
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
        hits = sum(1 for kw in (
            "item", "function", "failure mode", "severity", "occurrence",
            "detection", "cause", "effect", "requirement", "next higher",
        ) if kw in row_text)
        if hits >= 3:
            return i
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPER
# ─────────────────────────────────────────────────────────────────────────────

_AIAG_SEMANTIC = {
    "higher_fn":      ["next higher level function", "higher level function"],
    "focus_fn":       ["focus element function", "focus function"],
    "lower_fn":       ["next lower level function", "lower level function"],
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

_LEGACY_SEMANTIC = {
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
            if any(kw in h for kw in keywords):
                if semantic not in result:
                    result[semantic] = ci
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NOISE CLASSIFICATION  (LLM-assisted)
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_RE = re.compile(
    r"overload|overloading|excess.*load|temperature|\+\d+.*deg|-\d+.*deg|"
    r"salt|corrosive|wading|water ingress|mining|paved road|river sand|"
    r"molten tar|dust|vibration.*road|extreme load|operating in|operating on|"
    r"operating with|environmental|humidity|altitude",
    re.IGNORECASE,
)

_NOISE_CATS = ["pieceTopiece", "changeOverTime", "customerUsage", "externalEnvironment", "systemInteractions"]

_CAT_KEYWORDS = {
    "pieceTopiece":        ["dimension", "tolerance", "variation", "manufacturing", "assembly", "material property"],
    "changeOverTime":      ["wear", "fatigue", "corrosion", "age", "degrade", "creep", "drift", "life", "cycle"],
    "customerUsage":       ["overload", "load", "road", "speed", "duty cycle", "misuse", "abuse", "operator"],
    "externalEnvironment": ["temperature", "humidity", "salt", "dust", "vibration", "altitude", "water", "molten", "deg"],
    "systemInteractions":  ["interaction", "adjacent", "system", "interface", "cross-talk", "shared"],
}


def classify_cause_noise(cause_text: str, use_llm: bool = True) -> dict:
    """
    Given a failure cause string, determine:
      - noise_driven:   bool
      - noise_category: str (one of 5 P-diagram categories, or None)
      - noise_factor:   str (concise label, or None)
    """
    if not cause_text.strip():
        return {"noise_driven": False, "noise_category": None, "noise_factor": None}

    if not _NOISE_RE.search(cause_text):
        return {"noise_driven": False, "noise_category": None, "noise_factor": None}

    if not use_llm:
        cat = _classify_cat_by_keywords(cause_text)
        factor = _extract_factor_regex(cause_text)
        return {"noise_driven": True, "noise_category": cat, "noise_factor": factor}

    prompt = f"""Analyse this DFMEA failure cause and assume it is driven by a noise factor (external condition):

Failure cause: "{cause_text}"

A noise factor is an external condition that the design cannot control, such as:
- Operating environment (temperature, humidity, salt, dust, wading)  
- Customer usage pattern (overloading, road type, duty cycle)
- Material variation / piece-to-piece variation
- System-level interactions from adjacent systems
- Change over time (wear, fatigue, corrosion)

Output ONLY valid JSON, no markdown:
{{
  "noise_driven": true,
  "noise_category": <"pieceTopiece"|"changeOverTime"|"customerUsage"|"externalEnvironment"|"systemInteractions"|null>,
  "noise_factor": <"short label for the specific factor e.g. '+52°C high temperature', 'salt pan corrosion', 'overloading 150%'"|null>
}}"""

    raw = _llm(prompt, "classify_cause_noise", max_tokens=150)
    result = _parse_json(raw)
    return {
        "noise_driven":   bool(result.get("noise_driven", True)),
        "noise_category": result.get("noise_category"),
        "noise_factor":   result.get("noise_factor"),
    }


def _classify_cat_by_keywords(text: str) -> str:
    text_l = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_l) for cat, kws in _CAT_KEYWORDS.items()}
    return max(scores, key=lambda k: scores[k])


def _extract_factor_regex(text: str) -> str | None:
    m = re.search(r"due to (.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:80]
    return text[:60]


# ─────────────────────────────────────────────────────────────────────────────
# ROW-LEVEL FUNCTION DERIVATION  (LLM-assisted)
# ─────────────────────────────────────────────────────────────────────────────

def llm_derive_row_functions(
    focus_element: str,
    higher_element: str,
    lower_element: str,
    failure_mode: str,
    failure_effect: str,
    failure_cause: str,
    requirement: str = "",
    existing_focus_fn: str = "",
    existing_higher_fn: str = "",
    existing_lower_fn: str = "",
) -> dict:
    """
    Given a single DFMEA row's data and the three element names (user-supplied),
    derive the function statements for each level:
      - focus_fn:  what the focus element does
      - higher_fn: what the higher element does (that is affected by focus failures)
      - lower_fn:  what the lower element does (that causes focus failures)

    If functions are already present in the spreadsheet (explicit columns),
    those are passed in as existing_* and returned as-is — LLM is only called
    for the missing ones.

    Returns: { focus_fn: str, higher_fn: str, lower_fn: str }
    """
    result = {
        "focus_fn":  existing_focus_fn,
        "higher_fn": existing_higher_fn,
        "lower_fn":  existing_lower_fn,
    }

    need_focus  = not existing_focus_fn.strip()
    need_higher = not existing_higher_fn.strip()
    need_lower  = not existing_lower_fn.strip()

    if not need_focus and not need_higher and not need_lower:
        return result  # All already present — skip LLM call

    req_ctx = f"\nRequirement / function statement in sheet: \"{requirement}\"" if requirement else ""
    cause_ctx = f"\nFailure cause (what the lower element did wrong): \"{failure_cause}\"" if failure_cause else ""
    effect_ctx = f"\nFailure effect (what went wrong at higher level): \"{failure_effect}\"" if failure_effect else ""

    needed_fields = []
    if need_focus:
        needed_fields.append(
            f'  "focus_fn": "<positive function of {focus_element} that this failure mode violates>"'
        )
    if need_higher:
        needed_fields.append(
            f'  "higher_fn": "<positive function of {higher_element} that is impaired when focus element fails>"'
        )
    if need_lower:
        needed_fields.append(
            f'  "lower_fn": "<positive function of {lower_element} that contributes to the focus function when working correctly>"'
        )

    fields_str = ",\n".join(needed_fields)

    prompt = f"""You are writing a DFMEA function statement for each level of a 3-level hierarchy.

Elements:
  Higher-level element : "{higher_element}"
  Focus element        : "{focus_element}"
  Lower-level element  : "{lower_element}"

Row data from spreadsheet:
  Failure mode  (focus level):  "{failure_mode}"{req_ctx}{effect_ctx}{cause_ctx}

Rules:
- Each function is a positive statement: verb + object (e.g. "Transmit braking torque")
- Focus function: what {focus_element} is supposed to do (negation of its failure mode)
- Higher function: what {higher_element} is supposed to achieve (impaired by the failure effect)
- Lower function: what {lower_element} contributes to the focus function (negated by the failure cause)
- Keep each function concise (≤12 words)

Output ONLY valid JSON with exactly these keys:
{{
{fields_str}
}}"""

    token_budget = 80 * len(needed_fields) + 128
    raw = _llm(prompt, "llm_derive_row_functions", max_tokens=token_budget)
    parsed = _parse_json(raw)

    if need_focus  and parsed.get("focus_fn"):  result["focus_fn"]  = parsed["focus_fn"]
    if need_higher and parsed.get("higher_fn"): result["higher_fn"] = parsed["higher_fn"]
    if need_lower  and parsed.get("lower_fn"):  result["lower_fn"]  = parsed["lower_fn"]

    return result


def llm_batch_derive_row_functions(
    rows_data: list[dict],
    focus_element: str,
    higher_element: str,
    lower_element: str,
) -> list[dict]:
    """
    Batch version of llm_derive_row_functions — processes multiple rows in one LLM call.
    Each entry in rows_data must have: failure_mode, failure_effect, failure_cause,
    requirement (optional), existing_focus_fn, existing_higher_fn, existing_lower_fn.

    Returns a list of { focus_fn, higher_fn, lower_fn } dicts, same length as rows_data.
    """
    if not rows_data:
        return []

    # Pre-fill rows that already have all three functions
    results = [None] * len(rows_data)
    indices_needing_llm = []

    for i, rd in enumerate(rows_data):
        if (rd.get("existing_focus_fn") and
                rd.get("existing_higher_fn") and
                rd.get("existing_lower_fn")):
            results[i] = {
                "focus_fn":  rd["existing_focus_fn"],
                "higher_fn": rd["existing_higher_fn"],
                "lower_fn":  rd["existing_lower_fn"],
            }
        else:
            indices_needing_llm.append(i)

    if not indices_needing_llm:
        return results

    # Build batch prompt for rows that need LLM
    rows_text = ""
    for seq, i in enumerate(indices_needing_llm, start=1):
        rd = rows_data[i]
        rows_text += f"""
Row {seq}:
  failure_mode  : "{rd.get('failure_mode', '')}"
  failure_effect: "{rd.get('failure_effect', '')}"
  failure_cause : "{rd.get('failure_cause', '')}"
  requirement   : "{rd.get('requirement', '')}"
  existing_focus_fn : "{rd.get('existing_focus_fn', '')}"
  existing_higher_fn: "{rd.get('existing_higher_fn', '')}"
  existing_lower_fn : "{rd.get('existing_lower_fn', '')}"
"""

    prompt = f"""You are writing DFMEA function statements for a 3-level element hierarchy.

Elements:
  Higher-level element : "{higher_element}"
  Focus element        : "{focus_element}"
  Lower-level element  : "{lower_element}"

Rules for function statements:
- Positive verb + object (e.g. "Transmit braking torque", "Maintain suspension geometry")
- focus_fn : what {focus_element} is supposed to do (negation of its failure mode)
- higher_fn: what {higher_element} must achieve (impaired when focus element fails)
- lower_fn : what {lower_element} contributes (negated by the failure cause)
- If an existing_*_fn is already provided in the row, echo it back unchanged
- Keep each function concise (≤12 words)

For each row below, output one JSON object with keys: focus_fn, higher_fn, lower_fn.
Return a JSON ARRAY of {len(indices_needing_llm)} objects, one per row, in order.

Rows:
{rows_text}

Output ONLY valid JSON array, no markdown:
[
  {{"focus_fn": "...", "higher_fn": "...", "lower_fn": "..."}},
  ...
]"""

    token_budget = min(8192, max(512, len(indices_needing_llm) * 120 + 256))
    raw = _llm(prompt, "llm_batch_derive_row_functions", max_tokens=token_budget)

    try:
        parsed_list = json.loads(_strip_json(raw))
        if isinstance(parsed_list, list) and len(parsed_list) == len(indices_needing_llm):
            for seq, i in enumerate(indices_needing_llm):
                results[i] = {
                    "focus_fn":  parsed_list[seq].get("focus_fn", ""),
                    "higher_fn": parsed_list[seq].get("higher_fn", ""),
                    "lower_fn":  parsed_list[seq].get("lower_fn", ""),
                }
            return results
        # Partial result — fall back per-row for remainder
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            for seq, i in enumerate(indices_needing_llm[:len(parsed_list)]):
                results[i] = {
                    "focus_fn":  parsed_list[seq].get("focus_fn", ""),
                    "higher_fn": parsed_list[seq].get("higher_fn", ""),
                    "lower_fn":  parsed_list[seq].get("lower_fn", ""),
                }
            for i in indices_needing_llm[len(parsed_list):]:
                rd = rows_data[i]
                results[i] = llm_derive_row_functions(
                    focus_element, higher_element, lower_element,
                    rd.get("failure_mode", ""), rd.get("failure_effect", ""),
                    rd.get("failure_cause", ""), rd.get("requirement", ""),
                    rd.get("existing_focus_fn", ""), rd.get("existing_higher_fn", ""),
                    rd.get("existing_lower_fn", ""),
                )
            return results
    except Exception:
        pass

    # Full fallback — one call per row
    print("[llm WARNING] llm_batch_derive_row_functions: JSON parse failed, falling back to per-row calls.")
    for i in indices_needing_llm:
        rd = rows_data[i]
        results[i] = llm_derive_row_functions(
            focus_element, higher_element, lower_element,
            rd.get("failure_mode", ""), rd.get("failure_effect", ""),
            rd.get("failure_cause", ""), rd.get("requirement", ""),
            rd.get("existing_focus_fn", ""), rd.get("existing_higher_fn", ""),
            rd.get("existing_lower_fn", ""),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT A PARSER — AIAG-VDA 2019
# ─────────────────────────────────────────────────────────────────────────────

def parse_aiag_vda(df: pd.DataFrame, col: dict[str, int], elements: dict) -> dict:
    """
    Parse AIAG-VDA 2019 format DataFrame.

    `elements` must contain: focus_element, higher_element, lower_element
    (user-supplied at wizard step, not inferred from data).

    For each row, functions are:
      - Read from explicit function columns if present
      - Otherwise derived via LLM from failure_mode / failure_effect / failure_cause
    Connections are built from co-occurrence in the same row.
    """
    focus_element  = elements.get("focus_element",  "Focus System")
    higher_element = elements.get("higher_element", "Higher System")
    lower_element  = elements.get("lower_element",  "Lower Component")

    # ── Collect raw row data ──────────────────────────────────────────────────
    raw_rows_input: list[dict] = []
    for _, row in df.iterrows():
        higher_fn_existing = _s(row.get(col.get("higher_fn",  -1), ""))
        focus_fn_existing  = _s(row.get(col.get("focus_fn",   -1), ""))
        lower_fn_existing  = _s(row.get(col.get("lower_fn",   -1), ""))
        fe  = _s(row.get(col.get("failure_effect", -1), ""))
        sev = _int(row.get(col.get("severity",     -1), ""))
        fm  = _s(row.get(col.get("failure_mode",   -1), ""))
        fc  = _s(row.get(col.get("failure_cause",  -1), ""))
        prev    = _s(row.get(col.get("prevention",     -1), ""))
        occ     = _int(row.get(col.get("occurrence",   -1), ""))
        det_ctrl = _s(row.get(col.get("detection_ctrl",-1), ""))
        det     = _int(row.get(col.get("detection",    -1), ""))
        cls     = _s(row.get(col.get("classification", -1), ""))
        ap      = _s(row.get(col.get("ap",            -1), ""))

        if not fm and not fc:
            continue

        raw_rows_input.append({
            "failure_mode":       fm,
            "failure_effect":     fe,
            "failure_cause":      fc,
            "requirement":        "",
            "existing_focus_fn":  focus_fn_existing,
            "existing_higher_fn": higher_fn_existing,
            "existing_lower_fn":  lower_fn_existing,
            # carry through for output
            "_severity":    sev,
            "_classification": cls,
            "_prevention":  prev,
            "_occurrence":  occ,
            "_det_ctrl":    det_ctrl,
            "_detection":   det,
            "_ap":          ap,
        })

    # ── Batch derive functions for all rows ───────────────────────────────────
    fn_results = llm_batch_derive_row_functions(
        raw_rows_input, focus_element, higher_element, lower_element
    )

    # ── Build output rows and connections ──────────────────────────────────────
    rows_out: list[dict] = []
    connections: list[dict] = []
    noise_accumulator: dict[str, set] = {c: set() for c in _NOISE_CATS}

    for row_idx, (rd, fns) in enumerate(zip(raw_rows_input, fn_results)):
        if fns is None:
            fns = {"focus_fn": "", "higher_fn": "", "lower_fn": ""}

        fc = rd["failure_cause"]
        noise_info = classify_cause_noise(fc, use_llm=bool(_BEDROCK_KEY))
        if noise_info["noise_driven"] and noise_info["noise_category"] and noise_info["noise_factor"]:
            noise_accumulator[noise_info["noise_category"]].add(noise_info["noise_factor"])

        sev = rd["_severity"]
        occ = rd["_occurrence"]
        det = rd["_detection"]
        rpn = (sev or 0) * (occ or 0) * (det or 0) or None

        rows_out.append({
            "id":                  _uid(),
            "focus_element":       focus_element,
            "higher_element":      higher_element,
            "lower_element":       lower_element,
            "higher_fn":           fns["higher_fn"],
            "focus_fn":            fns["focus_fn"],
            "lower_fn":            fns["lower_fn"],
            "failure_effect":      rd["failure_effect"],
            "severity":            sev,
            "classification":      rd["_classification"],
            "failure_mode":        rd["failure_mode"],
            "failure_cause":       fc,
            "prevention_controls": rd["_prevention"],
            "occurrence":          occ,
            "detection_controls":  rd["_det_ctrl"],
            "detection":           det,
            "rpn":                 rpn,
            "ap":                  rd["_ap"],
            "noise_driven":        noise_info["noise_driven"],
            "noise_category":      noise_info["noise_category"],
            "noise_factor":        noise_info["noise_factor"],
            "row_index":           row_idx,
        })

        # Connection derived from same row: lower_fn → focus_fn → higher_fn
        if fns["lower_fn"] and fns["focus_fn"] and fns["higher_fn"]:
            connections.append({
                "lower_fn":  fns["lower_fn"],
                "focus_fn":  fns["focus_fn"],
                "higher_fn": fns["higher_fn"],
                "row_index": row_idx,
            })

    return _assemble_output(
        rows_out, noise_accumulator, connections,
        focus_element, higher_element, lower_element,
        fmt="aiag_vda"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT B PARSER — Legacy Ford/GM/Chrysler
# ─────────────────────────────────────────────────────────────────────────────

def parse_legacy(df: pd.DataFrame, col: dict[str, int], elements: dict) -> dict:
    """
    Parse legacy Ford/GM/Chrysler format.

    `elements` must contain: focus_element, higher_element, lower_element.
    Functions and connections are derived row-by-row via LLM from:
      - Failure mode  → focus_fn (negation)
      - Failure effect → higher_fn (what higher element was trying to achieve)
      - Failure cause → lower_fn (what lower element was supposed to do)
    """
    focus_element  = elements.get("focus_element",  "Focus System")
    higher_element = elements.get("higher_element", "Higher System")
    lower_element  = elements.get("lower_element",  "Lower Component")

    # ── Pass 1: Forward-fill Item/Function and Requirement columns ─────────────
    item_col_idx = col.get("item_function", 0)
    req_col_idx  = col.get("requirement",   1)

    item_col = df.iloc[:, item_col_idx].apply(_s)
    req_col  = df.iloc[:, req_col_idx].apply(_s)

    current_item = ""
    items_filled: list[str] = []
    for v in item_col:
        if v:
            current_item = v
        items_filled.append(current_item)

    current_req = ""
    reqs_filled: list[str] = []
    for v in req_col:
        if v:
            current_req = v
        reqs_filled.append(current_req)

    # ── Pass 2: Collect raw rows ───────────────────────────────────────────────
    fe_col_idx = col.get("failure_effect", -1)
    raw_rows_input: list[dict] = []

    for ri in range(len(df)):
        fm_val  = _s(df.iloc[ri, col["failure_mode"]])   if "failure_mode"   in col else ""
        fe_val  = _s(df.iloc[ri, fe_col_idx])            if fe_col_idx >= 0   else ""
        sev_val = df.iloc[ri, col["severity"]]           if "severity"        in col else None
        fc_val  = _s(df.iloc[ri, col["failure_cause"]])  if "failure_cause"   in col else ""
        prev_val = _s(df.iloc[ri, col["prevention"]])    if "prevention"      in col else ""
        occ_val  = df.iloc[ri, col["occurrence"]]        if "occurrence"      in col else None
        dc_val   = _s(df.iloc[ri, col["detection_ctrl"]]) if "detection_ctrl" in col else ""
        det_val  = df.iloc[ri, col["detection"]]         if "detection"       in col else None
        rpn_val  = df.iloc[ri, col["rpn"]]               if "rpn"             in col else None
        cls_val  = _s(df.iloc[ri, col["classification"]]) if "classification" in col else ""

        item_val = items_filled[ri]
        req_val  = reqs_filled[ri]

        if not item_val or (not fm_val and not fc_val):
            continue

        sev_int = _int(sev_val)
        occ_int = _int(occ_val)
        det_int = _int(det_val)
        rpn_int = _int(rpn_val) or ((sev_int or 0) * (occ_int or 0) * (det_int or 0) or None)

        # In legacy format, function columns are absent — all three are derived
        raw_rows_input.append({
            "failure_mode":       fm_val,
            "failure_effect":     fe_val,
            "failure_cause":      fc_val,
            "requirement":        req_val,
            # Functions are always derived in legacy format
            "existing_focus_fn":  req_val if req_val and len(req_val) > 5 else "",
            "existing_higher_fn": "",
            "existing_lower_fn":  "",
            # carry through
            "_item":      item_val,
            "_severity":  sev_int,
            "_classification": cls_val,
            "_prevention": prev_val,
            "_occurrence": occ_int,
            "_det_ctrl":  dc_val,
            "_detection": det_int,
            "_rpn":       rpn_int,
            "_row_index": ri,
        })

    # ── Batch derive functions for all rows ───────────────────────────────────
    fn_results = llm_batch_derive_row_functions(
        raw_rows_input, focus_element, higher_element, lower_element
    )

    # ── Build output rows and connections ──────────────────────────────────────
    rows_out: list[dict] = []
    connections: list[dict] = []
    noise_accumulator: dict[str, set] = {c: set() for c in _NOISE_CATS}

    for row_idx, (rd, fns) in enumerate(zip(raw_rows_input, fn_results)):
        if fns is None:
            fns = {"focus_fn": "", "higher_fn": "", "lower_fn": ""}

        fc = rd["failure_cause"]
        noise_info = classify_cause_noise(fc, use_llm=bool(_BEDROCK_KEY))
        if noise_info["noise_driven"] and noise_info["noise_category"] and noise_info["noise_factor"]:
            noise_accumulator[noise_info["noise_category"]].add(noise_info["noise_factor"])

        rows_out.append({
            "id":                  _uid(),
            "focus_element":       focus_element,
            "higher_element":      higher_element,
            "lower_element":       lower_element,
            "higher_fn":           fns["higher_fn"],
            "focus_fn":            fns["focus_fn"],
            "lower_fn":            fns["lower_fn"],
            "failure_effect":      rd["failure_effect"],
            "severity":            rd["_severity"],
            "classification":      rd["_classification"],
            "failure_mode":        rd["failure_mode"],
            "failure_cause":       fc,
            "prevention_controls": rd["_prevention"],
            "occurrence":          rd["_occurrence"],
            "detection_controls":  rd["_det_ctrl"],
            "detection":           rd["_detection"],
            "rpn":                 rd["_rpn"],
            "ap":                  None,
            "noise_driven":        noise_info["noise_driven"],
            "noise_category":      noise_info["noise_category"],
            "noise_factor":        noise_info["noise_factor"],
            "row_index":           rd["_row_index"],
        })

        # Connection from same row: lower_fn → focus_fn → higher_fn
        if fns["lower_fn"] and fns["focus_fn"] and fns["higher_fn"]:
            connections.append({
                "lower_fn":  fns["lower_fn"],
                "focus_fn":  fns["focus_fn"],
                "higher_fn": fns["higher_fn"],
                "row_index": rd["_row_index"],
            })

    return _assemble_output(
        rows_out, noise_accumulator, connections,
        focus_element, higher_element, lower_element,
        fmt="legacy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT ASSEMBLER  — same schema for both formats
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_connections(connections: list[dict]) -> list[dict]:
    """
    Remove duplicate connections (same lower_fn → focus_fn → higher_fn triplet).
    Keeps first occurrence (lowest row_index).
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in connections:
        key = (c["lower_fn"], c["focus_fn"], c["higher_fn"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _assemble_output(
    rows: list[dict],
    noise_acc: dict[str, set],
    connections: list[dict],
    focus_element: str,
    higher_element: str,
    lower_element: str,
    fmt: str,
) -> dict:
    """
    Collapse raw rows into the canonical output schema.
    Now includes user-supplied element names and derived connections.
    """
    if not rows:
        return {"error": "No data rows found"}

    # ── Focus functions (deduplicated) ────────────────────────────────────────
    focus_functions = sorted(set(r["focus_fn"] for r in rows if r["focus_fn"]))

    # ── Higher functions (deduplicated) ───────────────────────────────────────
    higher_functions = sorted(set(r["higher_fn"] for r in rows if r["higher_fn"]))

    # ── Lower functions (deduplicated) ────────────────────────────────────────
    lower_functions = sorted(set(r["lower_fn"] for r in rows if r["lower_fn"]))

    # ── Deduplicated connections ───────────────────────────────────────────────
    unique_connections = _deduplicate_connections(connections)

    # ── Failure modes grouped by (focus_fn, failure_mode) ────────────────────
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
        cause_entry = {
            "id":                  r["id"],
            "lower_fn":            r["lower_fn"],
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
        existing_causes = {c["cause_text"] for c in mode_map[key]["causes"]}
        if cause_entry["cause_text"] not in existing_causes:
            mode_map[key]["causes"].append(cause_entry)

    failure_modes = list(mode_map.values())

    # ── Noise factors ─────────────────────────────────────────────────────────
    noise_factors = {cat: sorted(items) for cat, items in noise_acc.items()}

    # ── S/O/D summary ─────────────────────────────────────────────────────────
    sod: dict = {"severities": [], "occurrences": [], "detections": [], "rpns": []}
    for r in rows:
        if r["severity"]  is not None: sod["severities"].append(r["severity"])
        if r["occurrence"] is not None: sod["occurrences"].append(r["occurrence"])
        if r["detection"]  is not None: sod["detections"].append(r["detection"])
        if r["rpn"]        is not None: sod["rpns"].append(r["rpn"])

    sod_summary = {
        "max_severity":   max(sod["severities"])  if sod["severities"]  else None,
        "avg_occurrence": round(sum(sod["occurrences"]) / len(sod["occurrences"]), 1) if sod["occurrences"] else None,
        "avg_detection":  round(sum(sod["detections"])  / len(sod["detections"]),  1) if sod["detections"]  else None,
        "max_rpn":        max(sod["rpns"]) if sod["rpns"] else None,
    }

    return {
        "format_detected":   fmt,
        # ── User-supplied element names ──
        "focus_element":     focus_element,
        "higher_element":    higher_element,
        "lower_element":     lower_element,
        # ── Derived functions per level ──
        "focus_functions":   focus_functions,
        "higher_functions":  higher_functions,
        "lower_functions":   lower_functions,
        # ── Connections (lower_fn → focus_fn → higher_fn) per row ──
        "connections":       unique_connections,
        # ── Failure analysis ──
        "failure_modes":     failure_modes,
        "noise_factors":     noise_factors,
        "sod_summary":       sod_summary,
        "raw_rows":          rows,
        "stats": {
            "total_rows":          len(rows),
            "failure_mode_count":  len(failure_modes),
            "focus_function_count": len(focus_functions),
            "higher_function_count": len(higher_functions),
            "lower_function_count":  len(lower_functions),
            "connection_count":    len(unique_connections),
            "noise_driven_count":  sum(1 for r in rows if r["noise_driven"]),
            "design_cause_count":  sum(1 for r in rows if not r["noise_driven"]),
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
    best = max(xl.items(), key=lambda kv: len(kv[1]))
    return best


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
    path        : path to the xlsx file
    elements    : dict with keys:
                    focus_element  — name of the focus system (user-provided)
                    higher_element — name of the next higher system (user-provided)
                    lower_element  — name of the next lower component (user-provided)
    sheet_name  : optional — if None, auto-detected

    Returns
    -------
    Canonical output dict (see module docstring for schema).

    NOTE: Element names are NOT inferred from the spreadsheet.
    They must be collected from the user before calling this function
    (e.g. via the wizard's Element step in the frontend).
    """
    if not elements or not all(k in elements for k in ("focus_element", "higher_element", "lower_element")):
        raise ValueError(
            "parse_dfmea_file() requires an 'elements' dict with keys: "
            "focus_element, higher_element, lower_element. "
            "These must be collected from the user before parsing."
        )

    xl = pd.read_excel(path, sheet_name=None, header=None)

    if sheet_name and sheet_name in xl:
        df_raw = xl[sheet_name]
    else:
        sheet_name, df_raw = find_dfmea_sheet(xl)

    print(f"[parser] Using sheet: '{sheet_name}'  shape={df_raw.shape}")
    print(f"[parser] Elements — focus: '{elements['focus_element']}' | "
          f"higher: '{elements['higher_element']}' | "
          f"lower: '{elements['lower_element']}'")

    header_row_idx = find_header_row(df_raw)
    print(f"[parser] Header row detected at index {header_row_idx}")

    header_values = [_s(v) for v in df_raw.iloc[header_row_idx].tolist()]
    data_df = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
    data_df.columns = range(data_df.shape[1])

    fmt = detect_format(header_values)
    print(f"[parser] Format detected: {fmt}")

    col = map_columns(header_values, fmt)
    print(f"[parser] Columns mapped: {col}")

    data_df = _wrap_df(data_df, col)

    if fmt == "aiag_vda":
        result = parse_aiag_vda(data_df, col, elements)
    else:
        result = parse_legacy(data_df, col, elements)

    result["source_file"] = Path(path).name
    result["sheet_name"]  = sheet_name
    return result


def _wrap_df(df: pd.DataFrame, col: dict[str, int]) -> pd.DataFrame:
    df = df.dropna(how="all").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CASE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_case1_new_conditions(parsed: dict) -> dict:
    """
    Case 1: Same design, new operating environment.
    - Keep: elements, functions, connections, failure modes
    - Strip: noise-driven causes (replace with new noise factors from P-Diagram)
    - Clear: O and D ratings (environment-dependent)
    - Keep: S ratings (severity is system-level, independent of environment)
    - Always treats noise factors as present (new environment assumption)
    """
    rows = parsed.get("raw_rows", [])

    modes_out = []
    seen: set[tuple] = set()
    for r in rows:
        key = (r["focus_fn"], r["failure_mode"])
        if key not in seen:
            seen.add(key)
        is_noise = r.get("noise_driven", True)  # default True for new env
        modes_out.append({
            "focus_fn":      r["focus_fn"],
            "failure_mode":  r["failure_mode"],
            "failure_effect": r["failure_effect"],
            "severity":      r["severity"],
            "cause":         None if is_noise else r["failure_cause"],
            "lower_fn":      r["lower_fn"],
            "occurrence":    None,
            "detection":     None,
            "noise_driven":  is_noise,
            "noise_label":   r.get("noise_factor"),
        })

    noise_stripped = sorted(set(
        r["noise_factor"] for r in rows
        if r.get("noise_driven") and r.get("noise_factor")
    ))

    return {
        "case":            "new_conditions",
        "focus_element":   parsed["focus_element"],
        "higher_element":  parsed["higher_element"],
        "lower_element":   parsed["lower_element"],
        "focus_functions":  parsed["focus_functions"],
        "higher_functions": parsed["higher_functions"],
        "lower_functions":  parsed["lower_functions"],
        "connections":     parsed["connections"],
        "failure_modes":   modes_out,
        "noise_stub": {
            "stripped_factors": noise_stripped,
            "instructions": (
                "These noise factors came from the original operating environment. "
                "Replace with your new conditions in the P-Diagram step. "
                "Noise-driven causes will be re-generated from new noise factors."
            ),
        },
        "stats": {
            "modes":             len(seen),
            "connections":       len(parsed["connections"]),
            "causes_retained":   sum(1 for r in rows if not r.get("noise_driven")),
            "causes_stripped":   sum(1 for r in rows if r.get("noise_driven")),
        },
    }


def build_case2_modified_design(parsed: dict, modified_elements: list[str] | None = None) -> dict:
    """
    Case 2: Same environment, modified component(s).
    - Keep: elements, functions, connections, noise-driven causes (with S/O/D)
    - Strip: design-mechanism causes for modified elements only
    - Re-generate: causes for modified elements via LLM
    - User must re-define modified elements and their functions before re-gen
    """
    rows = parsed.get("raw_rows", [])
    modified_set = set(modified_elements or [])

    modes_out = []
    for r in rows:
        is_modified = bool(modified_set) and (r["lower_element"] in modified_set)
        is_noise    = r.get("noise_driven", False)
        retain = is_noise or not is_modified
        modes_out.append({
            "focus_fn":       r["focus_fn"],
            "failure_mode":   r["failure_mode"],
            "failure_effect": r["failure_effect"],
            "severity":       r["severity"],
            "cause":          r["failure_cause"] if retain else None,
            "lower_fn":       r["lower_fn"],
            "occurrence":     r["occurrence"] if retain else None,
            "detection":      r["detection"]  if retain else None,
            "rpn":            r["rpn"]         if retain else None,
            "noise_driven":   is_noise,
            "retained":       retain,
            "needs_regen":    not retain,
            "modified_element": is_modified,
        })

    return {
        "case":             "modified_design",
        "focus_element":    parsed["focus_element"],
        "higher_element":   parsed["higher_element"],
        "lower_element":    parsed["lower_element"],
        "focus_functions":  parsed["focus_functions"],
        "higher_functions": parsed["higher_functions"],
        "lower_functions":  parsed["lower_functions"],
        "connections":      parsed["connections"],
        "noise_factors":    parsed["noise_factors"],
        "failure_modes":    modes_out,
        "modified_elements": list(modified_set) or "user_to_specify",
        "user_action_required": (
            "Please re-define the modified elements and their functions "
            "before regenerating failure causes for those components."
        ),
        "stats": {
            "total_rows":          len(rows),
            "connections":         len(parsed["connections"]),
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
    print(f"  Source : {parsed.get('source_file', '—')}")
    print(f"  Sheet  : {parsed.get('sheet_name',  '—')}")
    print(f"  Format : {parsed.get('format_detected', '—').upper()}")
    print("=" * 72)

    print(f"\n  Element hierarchy (user-supplied):")
    print(f"    Higher : {parsed['higher_element']}")
    print(f"    Focus  : {parsed['focus_element']}")
    print(f"    Lower  : {parsed['lower_element']}")

    print(f"\n  Higher functions ({len(parsed['higher_functions'])}):")
    for fn in parsed["higher_functions"][:5]:
        print(f"    · {fn[:80]}")

    print(f"\n  Focus functions ({len(parsed['focus_functions'])}):")
    for fn in parsed["focus_functions"][:5]:
        print(f"    · {fn[:80]}")

    print(f"\n  Lower functions ({len(parsed['lower_functions'])}):")
    for fn in parsed["lower_functions"][:5]:
        print(f"    · {fn[:80]}")

    print(f"\n  Connections (lower_fn → focus_fn → higher_fn) [{len(parsed['connections'])} unique]:")
    for c in parsed["connections"][:5]:
        print(f"    [row {c['row_index']}]")
        print(f"      {c['lower_fn'][:50]}")
        print(f"      → {c['focus_fn'][:50]}")
        print(f"      → {c['higher_fn'][:50]}")
    if len(parsed["connections"]) > 5:
        print(f"    … and {len(parsed['connections']) - 5} more connections")

    print(f"\n  Failure modes ({len(parsed['failure_modes'])}):")
    for mode in parsed["failure_modes"][:5]:
        causes = mode.get("causes", [])
        print(f"    · [{mode['focus_fn'][:40]}]  →  {mode['failure_mode'][:60]}")
        print(f"      S={mode['severity']}  effect: {mode['failure_effect'][:60]}")
        print(f"      Causes: {len(causes)} total  ({sum(1 for c in causes if c.get('noise_driven'))} noise-driven)")
    if len(parsed["failure_modes"]) > 5:
        print(f"    … and {len(parsed['failure_modes']) - 5} more modes")

    print(f"\n  Noise factors extracted:")
    for cat, factors in parsed["noise_factors"].items():
        if factors:
            print(f"    [{cat}]")
            for f in factors:
                print(f"        · {f}")

    s = parsed["stats"]
    print(f"\n  Stats: {s['total_rows']} rows | {s['failure_mode_count']} modes | "
          f"{s['focus_function_count']} focus fns | {s['lower_function_count']} lower fns | "
          f"{s['connection_count']} connections | "
          f"{s['noise_driven_count']} noise causes | {s['design_cause_count']} design causes")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTER  (drop into routers/ folder, register in main.py)
# ─────────────────────────────────────────────────────────────────────────────

# from fastapi import APIRouter, UploadFile, File, Form
# import tempfile, shutil
#
# router = APIRouter()
#
# @router.post("/api/dfmea/import/parse")
# async def import_parse(
#     file: UploadFile = File(...),
#     focus_element: str = Form(...),      # REQUIRED — collected from wizard Step 1
#     higher_element: str = Form(...),     # REQUIRED — collected from wizard Step 1
#     lower_element: str = Form(...),      # REQUIRED — collected from wizard Step 1
#     case: str = Form("new_conditions"),  # "new_conditions" | "modified_design"
#     modified_elements: str = Form(""),   # comma-separated list
#     sheet_name: str = Form(""),
# ):
#     elements = {
#         "focus_element":  focus_element,
#         "higher_element": higher_element,
#         "lower_element":  lower_element,
#     }
#     with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name
#
#     parsed = parse_dfmea_file(tmp_path, elements=elements, sheet_name=sheet_name or None)
#     mod_els = [e.strip() for e in modified_elements.split(",") if e.strip()]
#
#     if case == "modified_design":
#         return build_case2_modified_design(parsed, mod_els or None)
#     else:
#         return build_case1_new_conditions(parsed)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/mnt/user-data/uploads/1-Chassis_System-Axle_System_DFMEA_v4_Final__3_.xlsx"
    )
    sheet = sys.argv[2] if len(sys.argv) > 2 else None

    # ── Elements MUST be provided by user — collect from CLI for testing ───────
    print("\n[parser] Element names must be provided by the user.")
    print("         In production these come from the wizard's Element step.\n")

    focus_element  = input("  Focus element name  (e.g. 'Axle System')          : ").strip()
    higher_element = input("  Higher element name (e.g. 'Chassis')               : ").strip()
    lower_element  = input("  Lower element name  (e.g. 'Differential Assembly') : ").strip()

    elements = {
        "focus_element":  focus_element  or "Focus System",
        "higher_element": higher_element or "Higher System",
        "lower_element":  lower_element  or "Lower Component",
    }

    print(f"\n[parser] Parsing: {path}")
    parsed = parse_dfmea_file(path, elements=elements, sheet_name=sheet)

    print_report(parsed)

    out_dir = Path("/outputs")
    out_dir.mkdir(exist_ok=True)

    (out_dir / "universal_parsed_full.json").write_text(
        json.dumps(parsed, indent=2, default=str), encoding="utf-8"
    )

    case1 = build_case1_new_conditions(parsed)
    (out_dir / "universal_case1_new_conditions.json").write_text(
        json.dumps(case1, indent=2, default=str), encoding="utf-8"
    )

    case2 = build_case2_modified_design(parsed, modified_elements=None)
    (out_dir / "universal_case2_modified_design.json").write_text(
        json.dumps(case2, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[parser] Outputs written to {out_dir}:")
    print("  · universal_parsed_full.json")
    print("  · universal_case1_new_conditions.json")
    print("  · universal_case2_modified_design.json")
