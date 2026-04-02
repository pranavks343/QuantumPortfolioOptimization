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

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agents.states import AgentState, ParsedProblem

logger = logging.getLogger(__name__)

# Default penalty coefficient for constraint violations
_DEFAULT_PENALTY = 10.0


def _build_quadratic_program(problem: ParsedProblem):
    """Build a qiskit_optimization QuadraticProgram from the parsed problem."""
    from qiskit_optimization.problems import QuadraticProgram

    assets: list[str] = problem.get("assets", [])
    objective_weights: dict[str, float] = problem.get("objective_weights", {})
    objective_dir: str = problem.get("objective", "maximize")
    num_select: int = problem.get("num_select", 0)
    hard_constraints: list[dict] = problem.get("hard_constraints", [])

    if not assets:
        raise ValueError("No assets (decision variables) found in parsed problem.")

    qp = QuadraticProgram(name="PortfolioOptimization")

    # Add one binary variable per asset
    for asset in assets:
        qp.binary_var(name=asset)

    # Objective: maximise return ↔ minimise negative return
    linear_terms = {}
    for asset in assets:
        weight = objective_weights.get(asset, 1.0)
        linear_terms[asset] = -weight if objective_dir == "maximize" else weight

    qp.minimize(linear=linear_terms)

    # Cardinality constraint: sum(x_i) == k  (hard)
    if num_select > 0:
        qp.linear_constraint(
            linear={a: 1 for a in assets},
            sense="==",
            rhs=num_select,
            name="cardinality",
        )

    # Additional hard constraints from parsed problem
    for i, constraint in enumerate(hard_constraints):
        ctype = constraint.get("type", "")
        if ctype == "cardinality":
            # Already handled above via num_select
            continue
        elif ctype == "budget":
            weights_map: dict = constraint.get("weights", {})
            limit: float = constraint.get("limit", 0)
            qp.linear_constraint(
                linear={a: weights_map.get(a, 1.0) for a in assets},
                sense="<=",
                rhs=limit,
                name=f"budget_{i}",
            )

    return qp


def _qubo_to_ising(qp):
    """Convert a QUBO QuadraticProgram to an Ising SparsePauliOp."""
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising

    converter = QuadraticProgramToQubo(penalty=_DEFAULT_PENALTY)
    qubo = converter.convert(qp)
    ising_op, offset = to_ising(qubo)
    return qubo, ising_op, offset


def build_qubo(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 2 — Automated QUBO Formulator.

    Reads:   state['parsed_problem']
    Writes:  state['qubo_matrix'], state['qubo_offset'],
             state['ising_hamiltonian'], state['ising_offset'], state['logs']
    """
    parsed_problem: ParsedProblem = state.get("parsed_problem", {})
    logs: list[str] = list(state.get("logs", []))

    if parsed_problem.get("clarification_needed"):
        return {
            **state,
            "error": f"Clarification required: {parsed_problem.get('clarification_question')}",
            "logs": logs,
        }

    try:
        logs.append("[qubo_formulator] Building QuadraticProgram...")
        qp = _build_quadratic_program(parsed_problem)
        logs.append(f"[qubo_formulator] QP: {qp.get_num_vars()} vars, "
                    f"{qp.get_num_linear_constraints()} linear constraints")

        # Convert inequalities to equalities (slack vars), then to QUBO
        logs.append("[qubo_formulator] Converting to QUBO / Ising Hamiltonian...")
        qubo, ising_op, offset = _qubo_to_ising(qp)

        # Extract QUBO matrix for display
        # use_name=False ensures keys are integer indices, not variable names
        n = qubo.get_num_vars()
        Q = np.zeros((n, n))
        for (i, j), val in qubo.objective.quadratic.to_dict(use_name=False).items():
            Q[int(i)][int(j)] = val
            if i != j:
                Q[int(j)][int(i)] = val
        for i, val in qubo.objective.linear.to_dict(use_name=False).items():
            Q[int(i)][int(i)] += val
        qubo_offset = qubo.objective.constant

        logs.append(
            f"[qubo_formulator] Ising Hamiltonian: {ising_op.num_qubits} qubits, "
            f"{len(ising_op)} Pauli terms, offset={offset:.4f}"
        )

        return {
            **state,
            "qubo_matrix": Q.tolist(),
            "qubo_offset": float(qubo_offset),
            "ising_hamiltonian": ising_op,
            "ising_offset": float(offset),
            "num_qubits": ising_op.num_qubits,
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        msg = f"[qubo_formulator] Error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
