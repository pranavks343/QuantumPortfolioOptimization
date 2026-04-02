"""
LangGraph State Schema for the Agentic Quantum Optimization Copilot.

The AgentState TypedDict is the single shared state object that flows through
every node in the LangGraph StateGraph. Each node reads fields it needs and
updates the fields it produces.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ParsedProblem(TypedDict, total=False):
    """Structured output produced by Module 1 (intake_agent)."""
    assets: list[str]                   # Decision variable names (binary x_i)
    objective: str                      # "maximize" or "minimize"
    objective_weights: dict[str, float] # {asset_name: weight/return}
    num_select: int                     # How many assets to select (k)
    hard_constraints: list[dict]        # Must-satisfy constraints
    soft_constraints: list[dict]        # Penalised in objective
    clarification_needed: bool          # True if problem is under-defined
    clarification_question: str         # Question to ask the user


class AgentState(TypedDict, total=False):
    """
    Shared state that flows between all LangGraph nodes.

    Fields are populated progressively as the graph executes:
      intake → formulate → construct_circuit → execute → analyze → END
    """

    # ── Input ──────────────────────────────────────────────────────────────
    user_input: str                    # Raw natural language problem statement
    backend_name: str                  # "aer_simulator" or IBM Quantum backend

    # ── Module 1: Problem Intake & Reasoning Agent ─────────────────────────
    parsed_problem: ParsedProblem      # Structured extraction of the problem

    # ── Module 2: QUBO Formulator ──────────────────────────────────────────
    qubo_matrix: list[list[float]]     # QUBO Q-matrix as nested list (JSON-serialisable)
    qubo_offset: float                 # Constant offset from QUBO conversion
    ising_hamiltonian: Any             # SparsePauliOp (Qiskit object)
    ising_offset: float                # Constant offset from Ising conversion

    # ── Module 3: Quantum Circuit Constructor ──────────────────────────────
    circuit_qasm: str                  # OpenQASM 3 string of the parameterised QAOA circuit
    num_qubits: int                    # Number of qubits in the circuit
    circuit_depth: int                 # Depth of the transpiled circuit
    num_parameters: int                # Number of variational parameters (2 * reps)
    transpilation_feasible: bool       # Did transpilation check pass?

    # ── Module 4: Hybrid Execution Manager ────────────────────────────────
    initial_params: list[float]        # Starting gamma/beta values
    optimal_params: list[float]        # Optimised gamma/beta values
    optimal_energy: float              # Final expectation value
    convergence_history: list[float]   # Energy at each COBYLA iteration
    bitstring_counts: dict[str, int]   # Measurement outcome histogram

    # ── Module 5: Results Analyzer ────────────────────────────────────────
    final_solution: dict[str, Any]     # Human-readable result dict
    approximation_ratio: float         # Solution quality vs classical optimum
    constraint_satisfied: bool         # Did the best solution satisfy hard constraints?

    # ── Control & Logging ─────────────────────────────────────────────────
    logs: list[str]                    # Append-only experiment log
    error: str | None                  # Set by any node on failure; triggers error handler
    retry_count: int                   # Number of retries attempted
