"""
Module 5: Results Analyzer & Visualizer
=========================================
Decodes the raw bitstring measurement histogram into a human-readable
optimization result and produces visualisation artifacts.

Responsibilities:
  - Rank bitstrings by probability and decode them to asset selections
  - Check each candidate against the problem's hard constraints
  - Compute the approximation ratio vs. a classical brute-force optimum
  - Generate two matplotlib figures:
      1. Bitstring probability histogram (top-10 outcomes)
      2. COBYLA convergence curve (energy vs. iteration)
"""


from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit compatibility
import matplotlib.pyplot as plt
import numpy as np

from agents.states import AgentState, ParsedProblem

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLASSICAL BRUTE-FORCE (for small problems only, n ≤ 20 assets)
# ──────────────────────────────────────────────────────────────────────────────

def _brute_force_optimum(problem: ParsedProblem) -> tuple[float, list[str]]:
    """
    Try all 2^n possible asset combinations and return the best feasible one.
    Only practical for n ≤ 20 (2^20 ≈ 1 million combinations).
    """
    assets = problem.get("assets", [])
    weights = problem.get("objective_weights", {a: 1.0 for a in assets})
    objective = problem.get("objective", "maximize")
    num_select = problem.get("num_select", 0)
    hard_constraints = problem.get("hard_constraints", [])
    n = len(assets)

    # For large problems, brute force is computationally infeasible
    if n > 20:
        return float("-inf") if objective == "maximize" else float("inf"), []

    best_val = float("-inf") if objective == "maximize" else float("inf")
    best_selection: list[str] = []

    # Iterate over all 2^n binary patterns using bit manipulation
    # mask=0b00101 means assets at position 0 and 2 are selected
    for mask in range(1 << n):
        selection = [assets[i] for i in range(n) if (mask >> i) & 1]

        if num_select > 0 and len(selection) != num_select:
            continue

        satisfied = True
        for constraint in hard_constraints:
            ctype = constraint.get("type", "")
            if ctype == "budget":
                cost_weights = constraint.get("weights", {})
                limit = constraint.get("limit", float("inf"))
                total_cost = sum(cost_weights.get(a, 1.0) for a in selection)
                if total_cost > limit:
                    satisfied = False
                    break

        if not satisfied:
            continue

        obj_val = sum(weights.get(a, 1.0) for a in selection)

        if objective == "maximize" and obj_val > best_val:
            best_val = obj_val
            best_selection = selection
        elif objective == "minimize" and obj_val < best_val:
            best_val = obj_val
            best_selection = selection

    return best_val, best_selection


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRAINT CHECKER
# ──────────────────────────────────────────────────────────────────────────────

def _check_constraints(selection: list[str], problem: ParsedProblem) -> bool:
    """
    Return True if the given asset selection satisfies ALL hard constraints.

    Checks:
      - Cardinality: did we select exactly num_select assets?
      - Budget: is the total cost within the budget limit?
    """
    num_select = problem.get("num_select", 0)
    hard_constraints = problem.get("hard_constraints", [])

    if num_select > 0 and len(selection) != num_select:
        return False

    for constraint in hard_constraints:
        ctype = constraint.get("type", "")
        if ctype == "budget":
            cost_weights = constraint.get("weights", {})
            limit = constraint.get("limit", float("inf"))
            total_cost = sum(cost_weights.get(a, 1.0) for a in selection)
            if total_cost > limit:
                return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# BITSTRING DECODER
# ──────────────────────────────────────────────────────────────────────────────

def _decode_bitstring(bitstring: str, assets: list[str]) -> list[str]:
    """
    Convert a measurement bitstring like "10110" into asset names.

    IMPORTANT — Qiskit's bit ordering:
      The RIGHTMOST bit in the string corresponds to qubit 0 (the first asset).
      So we read the string RIGHT TO LEFT when mapping to assets.

    Example: bitstring="10110", assets=["AAPL","GOOGL","MSFT","AMZN","TSLA"]
      Position:  4    3    2    1    0    (qubit index, read right-to-left)
      Bit:       1    0    1    1    0
      Asset:  TSLA AMZN MSFT GOOGL AAPL
      Bits set (=1): TSLA, MSFT, GOOGL → these are selected
    """
    n = len(assets)

    # zfill pads short strings with leading zeros; [-n:] truncates from the right
    bits = bitstring.zfill(n)[-n:]

    # bits[-(i+1)] gives the bit for qubit i (rightmost = qubit 0)
    return [assets[i] for i in range(n) if bits[-(i + 1)] == "1"]


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _plot_histogram(counts: dict[str, int], top_n: int = 10) -> plt.Figure:
    """
    Create a bar chart showing the probability of the top-N measurement outcomes.

    Each bar = one bitstring. Taller bar = more likely that combination was measured.
    The most probable bitstring is usually the best solution.
    """
    if not counts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("No measurement data available")
        return fig

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [item[0] for item in sorted_counts]
    values = [item[1] for item in sorted_counts]
    total = sum(counts.values())
    probabilities = [v / total for v in values]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, probabilities, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Bitstring (measurement outcome)", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("QAOA Measurement Distribution (Top States)", fontsize=13)
    ax.set_ylim(0, max(probabilities) * 1.2 if probabilities else 1.0)
    plt.xticks(rotation=45, ha="right")

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{prob:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    return fig


def _plot_convergence(convergence_history: list[float]) -> plt.Figure:
    """
    Create a line chart showing how the energy decreased during COBYLA optimization.

    x-axis = COBYLA iteration number
    y-axis = ⟨H⟩ (energy / expectation value)

    A downward-trending line means the optimizer is finding better solutions.
    When the line flattens, the optimizer has converged.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(convergence_history) + 1),
        convergence_history,
        color="#DD8452",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("COBYLA Iteration", fontsize=11)
    ax.set_ylabel("⟨H⟩ (Expectation Value)", fontsize=11)
    ax.set_title("Classical-Quantum Hybrid Optimisation Convergence", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def analyze(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 5 — Results Analyzer & Visualizer.

    Reads:   state['bitstring_counts'], state['parsed_problem'],
             state['convergence_history'], state['optimal_energy']
    Writes:  state['final_solution'], state['approximation_ratio'],
             state['constraint_satisfied'], state['logs']
    """
    bitstring_counts: dict[str, int] = state.get("bitstring_counts", {})
    parsed_problem: ParsedProblem = state.get("parsed_problem", {})
    convergence_history: list[float] = state.get("convergence_history", [])
    optimal_energy: float = state.get("optimal_energy", 0.0)
    logs: list[str] = list(state.get("logs", []))

    if not bitstring_counts:
        return {**state, "error": "[results_analyzer] No bitstring counts found.", "logs": logs}

    assets: list[str] = parsed_problem.get("assets", [])
    objective_weights = parsed_problem.get("objective_weights", {a: 1.0 for a in assets})
    objective_dir = parsed_problem.get("objective", "maximize")

    # ── Step 1: Decode every bitstring into asset names ───────────────────────
    total_shots = sum(bitstring_counts.values())
    candidates = []

    for bitstring, count in sorted(bitstring_counts.items(), key=lambda x: x[1], reverse=True):
        selection = _decode_bitstring(bitstring, assets)
        obj_val = sum(objective_weights.get(a, 1.0) for a in selection)
        satisfied = _check_constraints(selection, parsed_problem)

        candidates.append({
            "bitstring": bitstring,
            "selection": selection,
            "probability": count / total_shots,
            "objective_value": obj_val,
            "constraint_satisfied": satisfied,
        })

    # ── Step 2: Pick the best feasible solution ───────────────────────────────
    feasible = [c for c in candidates if c["constraint_satisfied"]]

    if not feasible:
        # No solution satisfied the constraints → use the most probable one anyway
        best_candidate = candidates[0]
        logs.append("[results_analyzer] WARNING: No constraint-satisfying solution found.")
    else:
        if objective_dir == "maximize":
            best_candidate = max(feasible, key=lambda c: c["objective_value"])
        else:
            best_candidate = min(feasible, key=lambda c: c["objective_value"])

    best_selection = best_candidate["selection"]
    best_obj_val = best_candidate["objective_value"]
    constraint_satisfied = best_candidate["constraint_satisfied"]

    # ── Step 3: Compute the approximation ratio ───────────────────────────────
    classical_optimum, classical_selection = _brute_force_optimum(parsed_problem)

    # approximation_ratio = quantum_result / classical_best
    # 1.0 = perfect, 0.9 = 90% of optimal
    if classical_optimum not in (float("inf"), float("-inf")) and classical_optimum != 0:
        approximation_ratio = best_obj_val / classical_optimum
    else:
        approximation_ratio = 1.0   # Can't compute → default to 1

    logs.append(
        f"[results_analyzer] Best solution: {best_selection}, "
        f"obj_val={best_obj_val:.4f}, "
        f"approx_ratio={approximation_ratio:.4f}, "
        f"constraint_satisfied={constraint_satisfied}"
    )
    if classical_selection:
        logs.append(
            f"[results_analyzer] Classical optimum: {classical_selection}, val={classical_optimum:.4f}"
        )

    # ── Step 4: Build the final result dictionary ─────────────────────────────
    final_solution = {
        "selected_assets": best_selection,
        "objective_value": best_obj_val,
        "objective_direction": objective_dir,
        "constraint_satisfied": constraint_satisfied,
        "approximation_ratio": approximation_ratio,
        "classical_optimum_selection": classical_selection,
        "classical_optimum_value": classical_optimum if isinstance(classical_optimum, float) else None,
        "optimal_energy": optimal_energy,
        "top_candidates": candidates[:5],
        "total_shots": total_shots,
    }

    # ── Step 5: Generate the visualization charts ─────────────────────────────
    # We store matplotlib Figure objects directly in the dict.
    # The Streamlit UI (app.py) will call st.pyplot() on them.
    # Keys starting with "_" are excluded from JSON serialization.
    try:
        hist_fig = _plot_histogram(bitstring_counts)
        conv_fig = _plot_convergence(convergence_history) if convergence_history else None
        final_solution["_histogram_fig"] = hist_fig
        final_solution["_convergence_fig"] = conv_fig
    except Exception as exc:
        logger.warning(f"[results_analyzer] Visualisation error: {exc}")

    return {
        **state,
        "final_solution": final_solution,
        "approximation_ratio": approximation_ratio,
        "constraint_satisfied": constraint_satisfied,
        "logs": logs,
        "error": None,
    }
