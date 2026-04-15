"""
tools/optimizer/
────────────────
Spring stiffness optimization tools.

The execution cycle in LangGraph:
  init_opt → solve_ode → compute_rms → propose_k → check_convergence
               ↑___________________________|  (loop if not converged)
                                           └→ summarize_opt (if converged)

The quarter-car 2-DOF model:
  Sprung mass ms   (cabin)   : x1, v1
  Unsprung mass mu (axle)    : x2, v2
  Spring stiffness k         : optimisation variable
  Damping coefficient c      : fixed parameter
  Tyre stiffness kt          : fixed parameter
  Road excitation w(t)       : defined by road_profile in ode_params
"""

from __future__ import annotations
from core.tool_base import BaseTool, ToolResult
from core.state import AgentState
from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

import numpy as np
from scipy.integrate import solve_ivp
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── 1. Initialize ─────────────────────────────────────────────────────────────

class InitializeOptTool(BaseTool):
    name        = "init_opt"
    description = "Initialise optimisation state: set k to midpoint of bounds, reset counters."
    reads       = ["opt_k_bounds", "opt_ode_params"]
    writes      = ["opt_spring_k", "opt_iteration", "opt_converged",
                   "opt_best_k", "opt_best_rms", "opt_history"]

    def run(self, state: AgentState) -> ToolResult:
        k_min, k_max = state.get("opt_k_bounds", (5000.0, 50000.0))
        k_init       = (k_min + k_max) / 2.0

        # Create an Optuna study and store its name globally for reuse
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        _OPTUNA_STUDY_CACHE["study"] = study
        _OPTUNA_STUDY_CACHE["k_bounds"] = (k_min, k_max)

        return {
            "opt_spring_k":   k_init,
            "opt_iteration":  0,
            "opt_converged":  False,
            "opt_best_k":     k_init,
            "opt_best_rms":   float("inf"),
            "opt_history":    [],
        }


_OPTUNA_STUDY_CACHE: dict = {}


# ── 2. Solve ODE ──────────────────────────────────────────────────────────────

class SolveOdeTool(BaseTool):
    name        = "solve_ode"
    description = "Integrate quarter-car 2-DOF ODEs for current spring stiffness k."
    reads       = ["opt_spring_k", "opt_ode_params"]
    writes      = ["_ode_solution"]   # stored in module cache, not state (large array)

    def run(self, state: AgentState) -> ToolResult:
        k      = state["opt_spring_k"]
        params = state.get("opt_ode_params", {})

        ms  = params.get("ms",  400.0)    # sprung mass kg
        mu  = params.get("mu",  45.0)     # unsprung mass kg
        c   = params.get("c",   1500.0)   # damping N·s/m
        kt  = params.get("kt",  150000.0) # tyre stiffness N/m
        t_end = params.get("t_end", 10.0) # simulation time s
        dt    = params.get("dt",   0.001) # time step s

        road_profile = params.get("road_profile", "white_noise")
        t_eval = np.arange(0, t_end, dt)

        # Road excitation
        rng = np.random.default_rng(seed=0)  # deterministic road for fair comparison
        if road_profile == "white_noise":
            amplitude = params.get("road_amplitude", 0.01)  # m
            road_vals = amplitude * rng.standard_normal(len(t_eval))
        elif road_profile == "bump":
            road_vals = np.zeros(len(t_eval))
            bump_idx  = int(0.3 * len(t_eval))
            road_vals[bump_idx:bump_idx+50] = 0.05   # 5cm bump
        else:
            road_vals = np.zeros(len(t_eval))

        road_fn = lambda t: float(np.interp(t, t_eval, road_vals))

        def quarter_car(t, y):
            x1, v1, x2, v2 = y
            # Sprung mass (cabin)
            dv1 = (-k * (x1 - x2) - c * (v1 - v2)) / ms
            # Unsprung mass (axle)
            dv2 = (k * (x1 - x2) + c * (v1 - v2) - kt * (x2 - road_fn(t))) / mu
            return [v1, dv1, v2, dv2]

        sol = solve_ivp(
            quarter_car,
            t_span=(0, t_end),
            y0=[0.0, 0.0, 0.0, 0.0],
            method="RK45",
            t_eval=t_eval,
            rtol=1e-6, atol=1e-8,
        )

        _ODE_SOLUTION_CACHE["sol"] = sol
        _ODE_SOLUTION_CACHE["dt"]  = dt
        return {}   # state written by compute_rms

_ODE_SOLUTION_CACHE: dict = {}


# ── 3. Compute RMS Acceleration ───────────────────────────────────────────────

class ComputeRmsTool(BaseTool):
    name        = "compute_rms"
    description = "Compute RMS cabin acceleration from ODE solution. Update best if improved."
    reads       = ["opt_spring_k", "opt_iteration", "opt_best_rms"]
    writes      = ["opt_rms_acceleration", "opt_best_k", "opt_best_rms",
                   "opt_history", "opt_iteration"]

    def run(self, state: AgentState) -> ToolResult:
        sol = _ODE_SOLUTION_CACHE.get("sol")
        dt  = _ODE_SOLUTION_CACHE.get("dt", 0.001)

        if sol is None or not sol.success:
            return {"opt_rms_acceleration": float("inf")}

        # Cabin acceleration = numerical derivative of cabin velocity (y[1])
        v_cabin = sol.y[1]
        a_cabin = np.diff(v_cabin) / dt
        rms     = float(np.sqrt(np.mean(a_cabin ** 2)))

        k    = state["opt_spring_k"]
        best = state.get("opt_best_rms", float("inf"))
        it   = state.get("opt_iteration", 0)

        new_best_k   = k    if rms < best else state.get("opt_best_k", k)
        new_best_rms = rms  if rms < best else best

        # Tell Optuna about this trial result
        study = _OPTUNA_STUDY_CACHE.get("study")
        if study:
            trial = study.ask()
            study.tell(trial, rms)

        return {
            "opt_rms_acceleration": rms,
            "opt_best_k":           new_best_k,
            "opt_best_rms":         new_best_rms,
            "opt_iteration":        it + 1,
            "opt_history":          [{"iteration": it + 1, "k": k, "rms": rms}],
        }


# ── 4. Propose Next k ────────────────────────────────────────────────────────

class ProposeKTool(BaseTool):
    name        = "propose_k"
    description = "Ask Optuna TPE sampler for the next spring stiffness k to evaluate."
    reads       = ["opt_k_bounds"]
    writes      = ["opt_spring_k"]

    def run(self, state: AgentState) -> ToolResult:
        study    = _OPTUNA_STUDY_CACHE.get("study")
        k_min, k_max = _OPTUNA_STUDY_CACHE.get("k_bounds", (5000.0, 50000.0))

        if study is None:
            # Fallback: random sample
            return {"opt_spring_k": float(np.random.uniform(k_min, k_max))}

        trial = study.ask()
        k_new = trial.suggest_float("k", k_min, k_max)
        return {"opt_spring_k": k_new}


# ── 5. Check Convergence ─────────────────────────────────────────────────────

class CheckConvergenceTool(BaseTool):
    name        = "check_convergence"
    description = "Check if optimisation has converged. Sets opt_converged for graph routing."
    reads       = ["opt_history", "opt_iteration", "opt_max_iterations", "opt_convergence_tol"]
    writes      = ["opt_converged"]

    def run(self, state: AgentState) -> ToolResult:
        history  = state.get("opt_history", [])
        it       = state.get("opt_iteration", 0)
        max_it   = state.get("opt_max_iterations", 80)
        tol      = state.get("opt_convergence_tol", 1e-4)

        if it >= max_it:
            return {"opt_converged": True}

        if len(history) >= 5:
            recent_rms = [h["rms"] for h in history[-5:]]
            delta = max(recent_rms) - min(recent_rms)
            if delta < tol:
                return {"opt_converged": True}

        return {"opt_converged": False}


# ── 6. Summarize Result ───────────────────────────────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an automotive NVH (Noise Vibration Harshness) engineer.
Interpret the spring stiffness optimisation result in plain engineering language.
Include: optimal k value, achieved RMS acceleration, ride comfort implication,
and whether the value is within typical automotive suspension ranges.
Be concise — 3-4 sentences."""),
    ("human", "Optimal k: {best_k} N/m\nRMS acceleration: {best_rms:.4f} m/s²\nIterations: {iterations}\nHistory summary: {history_summary}"),
])


class SummarizeOptTool(BaseTool):
    name        = "summarize_opt"
    description = "Generate LLM engineering summary of the optimisation result."
    reads       = ["opt_best_k", "opt_best_rms", "opt_iteration", "opt_history"]
    writes      = ["opt_summary"]

    async def arun(self, state: AgentState) -> ToolResult:
        llm   = get_llm(temperature=0.3)
        chain = SUMMARY_PROMPT | llm

        history = state.get("opt_history", [])
        history_summary = f"{len(history)} evaluations, RMS range: " \
                          f"{min(h['rms'] for h in history):.4f} – {max(h['rms'] for h in history):.4f} m/s²" \
                          if history else "no history"

        resp = await chain.ainvoke({
            "best_k":          state.get("opt_best_k", 0),
            "best_rms":        state.get("opt_best_rms", 0),
            "iterations":      state.get("opt_iteration", 0),
            "history_summary": history_summary,
        })

        return {"opt_summary": resp.content}
