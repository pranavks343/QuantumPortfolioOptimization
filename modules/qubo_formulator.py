"""
Module 2: Automated QUBO Formulator
=====================================
Converts the structured problem representation produced by Module 1
into a mathematical QUBO/Ising model that can drive a QAOA circuit.

Pipeline:
  ParsedProblem
    → qiskit_optimization.QuadraticProgram (QP)
    → InequalityToEquality (slack variables)
    → IntegerToBinary
    → QuadraticProgramToQubo
    → SparsePauliOp (Ising Hamiltonian) via QubitConverter

Key concepts:
  QUBO: minimise  x^T Q x + c   subject to  x ∈ {0,1}^n
  Ising: minimise <ψ|H|ψ>  where H is a sum of Pauli-Z products
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is Step 2 of the pipeline. It takes the structured problem from Step 1
# and converts it into the mathematical language quantum computers understand.
#
# Conversion chain:
#   ParsedProblem (Python dict)
#       ↓
#   QuadraticProgram  ← like writing equations with variables and constraints
#       ↓
#   QUBO matrix       ← the problem as a grid of numbers (x^T * Q * x)
#       ↓
#   Ising Hamiltonian ← quantum operators that the QAOA circuit minimizes
#
# WHY QUBO?
#   Quantum computers can't directly solve "pick the best 3 stocks."
#   We rewrite it as: minimize x^T * Q * x, where x is a 0/1 vector.
#
# WHY ISING?
#   QAOA circuits are built to minimize Ising Hamiltonians (sums of Pauli-Z).
#   QUBO → Ising is a standard mathematical substitution.
#
# HOW ARE CONSTRAINTS HANDLED?
#   Constraints like "select exactly 3" become PENALTY TERMS in the objective.
#   Violating the constraint adds a large energy cost, so the optimizer avoids it.
#
#   "Select exactly 3 from 5 assets" becomes:
#       penalty * (x1 + x2 + x3 + x4 + x5 - 3)^2
#   This equals 0 when exactly 3 are selected, and is positive otherwise.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agents.states import AgentState, ParsedProblem

logger = logging.getLogger(__name__)

# How much to penalize constraint violations.
# 10.0 means: violating a rule is "10 times worse" than ignoring the objective.
# Too low → optimizer ignores rules. Too high → optimizer ignores the goal.
_DEFAULT_PENALTY = 10.0


# ──────────────────────────────────────────────────────────────────────────────
# STEP A: Build a Quadratic Program
# ──────────────────────────────────────────────────────────────────────────────
def _build_quadratic_program(problem: ParsedProblem):
    """Build a qiskit_optimization QuadraticProgram from the parsed problem."""
    from qiskit_optimization.problems import QuadraticProgram

    # Unpack fields from the parsed problem dictionary
    assets: list[str] = problem.get("assets", [])
    objective_weights: dict[str, float] = problem.get("objective_weights", {})
    objective_dir: str = problem.get("objective", "maximize")
    num_select: int = problem.get("num_select", 0)
    hard_constraints: list[dict] = problem.get("hard_constraints", [])

    if not assets:
        raise ValueError("No assets (decision variables) found in parsed problem.")

    # Create an empty optimization model
    qp = QuadraticProgram(name="PortfolioOptimization")

    # Add one binary decision variable per asset: 0 = not selected, 1 = selected
    for asset in assets:
        qp.binary_var(name=asset)

    # ── Set the objective function ────────────────────────────────────────────
    # QUBO always minimizes, so we flip the sign for maximization problems.
    # "Maximize 0.9*AAPL" → "Minimize -0.9*AAPL"
    linear_terms = {}
    for asset in assets:
        weight = objective_weights.get(asset, 1.0)  # Default weight = 1.0
        linear_terms[asset] = -weight if objective_dir == "maximize" else weight

    qp.minimize(linear=linear_terms)

    # ── Add the cardinality constraint ────────────────────────────────────────
    # "Select exactly k" → AAPL + GOOGL + MSFT + AMZN + TSLA = k
    if num_select > 0:
        qp.linear_constraint(
            linear={a: 1 for a in assets},  # All assets contribute weight 1
            sense="==",                      # Must equal exactly
            rhs=num_select,                  # The required count k
            name="cardinality",
        )

    # ── Add any other hard constraints from the parsed problem ────────────────
    for i, constraint in enumerate(hard_constraints):
        ctype = constraint.get("type", "")

        if ctype == "cardinality":
            continue  # Already handled via num_select above

        elif ctype == "budget":
            # "Total cost must not exceed limit"
            # e.g. 30*Laptop + 50*Phone + 20*Tablet ≤ 100
            weights_map: dict = constraint.get("weights", {})
            limit: float = constraint.get("limit", 0)
            qp.linear_constraint(
                linear={a: weights_map.get(a, 1.0) for a in assets},
                sense="<=",       # Less-than-or-equal (budget cap)
                rhs=limit,        # The budget limit
                name=f"budget_{i}",
            )

    return qp


# ──────────────────────────────────────────────────────────────────────────────
# STEP B: Convert QuadraticProgram → QUBO → Ising Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────
def _qubo_to_ising(qp):
    """Convert a QUBO QuadraticProgram to an Ising SparsePauliOp."""
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising

    # QuadraticProgramToQubo:
    #   1. Converts inequality constraints (≤) to equality by adding slack variables
    #   2. Moves ALL constraints into the objective as penalty terms
    # After this step, the problem is truly unconstrained (no explicit rules).
    converter = QuadraticProgramToQubo(penalty=_DEFAULT_PENALTY)
    qubo = converter.convert(qp)

    # Convert QUBO (binary variables) → Ising (±1 spin variables → Pauli-Z operators)
    # This is the format QAOA circuits are designed to minimize.
    # Returns the operator and a constant energy offset.
    ising_op, offset = to_ising(qubo)

    return qubo, ising_op, offset


# ──────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def build_qubo(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 2 — Automated QUBO Formulator.

    Reads:   state['parsed_problem']
    Writes:  state['qubo_matrix'], state['qubo_offset'],
             state['ising_hamiltonian'], state['ising_offset'], state['logs']
    """
    parsed_problem: ParsedProblem = state.get("parsed_problem", {})
    logs: list[str] = list(state.get("logs", []))

    # Safety check: if Step 1 flagged clarification is needed, don't proceed
    if parsed_problem.get("clarification_needed"):
        return {
            **state,
            "error": f"Clarification required: {parsed_problem.get('clarification_question')}",
            "logs": logs,
        }

    try:
        # ── Step A: Build the optimization model ─────────────────────────────
        logs.append("[qubo_formulator] Building QuadraticProgram...")
        qp = _build_quadratic_program(parsed_problem)
        logs.append(
            f"[qubo_formulator] QP: {qp.get_num_vars()} vars, "
            f"{qp.get_num_linear_constraints()} linear constraints"
        )

        # ── Step B: Convert to QUBO and then Ising Hamiltonian ───────────────
        logs.append("[qubo_formulator] Converting to QUBO / Ising Hamiltonian...")
        qubo, ising_op, offset = _qubo_to_ising(qp)

        # ── Extract the QUBO matrix as a 2D list for the UI heatmap ──────────
        # Note: the converter may have added slack variables, so the matrix
        # may be larger than the number of original assets.
        n = qubo.get_num_vars()
        Q = np.zeros((n, n))  # Start with all zeros

        # Fill quadratic (off-diagonal) terms — these represent interactions
        # use_name=False → keys are integer indices (0, 1, 2...) not variable names
        for (i, j), val in qubo.objective.quadratic.to_dict(use_name=False).items():
            Q[int(i)][int(j)] = val
            if i != j:
                Q[int(j)][int(i)] = val  # QUBO matrix is symmetric

        # Fill linear (diagonal) terms — these represent individual asset costs
        for i, val in qubo.objective.linear.to_dict(use_name=False).items():
            Q[int(i)][int(i)] += val

        # The constant offset doesn't affect which solution is best,
        # but it's needed to compute the true energy value
        qubo_offset = qubo.objective.constant

        logs.append(
            f"[qubo_formulator] Ising Hamiltonian: {ising_op.num_qubits} qubits, "
            f"{len(ising_op)} Pauli terms, offset={offset:.4f}"
        )

        # Return the updated state with all quantum math fields filled in
        return {
            **state,
            "qubo_matrix": Q.tolist(),           # 2D list for the heatmap chart
            "qubo_offset": float(qubo_offset),   # Constant from QUBO formulation
            "ising_hamiltonian": ising_op,        # The quantum operator object
            "ising_offset": float(offset),        # Constant from Ising conversion
            "num_qubits": ising_op.num_qubits,    # Used in Step 3 for circuit building
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        msg = f"[qubo_formulator] Error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
