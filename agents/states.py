"""
LangGraph State Schema for the Agentic Quantum Optimization Copilot.

The AgentState TypedDict is the single shared state object that flows through
every node in the LangGraph StateGraph. Each node reads fields it needs and
updates the fields it produces.
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# Think of this file as defining the "notepad" that gets passed between every
# step in our pipeline.
#
# Step 1 (intake)   reads: user_input       writes: parsed_problem
# Step 2 (formulate) reads: parsed_problem  writes: qubo_matrix, ising_hamiltonian
# Step 3 (circuit)  reads: ising_hamiltonian writes: circuit_qasm
# Step 4 (execute)  reads: circuit_qasm     writes: bitstring_counts
# Step 5 (analyze)  reads: bitstring_counts writes: final_solution
#
# Every step gets the FULL notepad, picks out the fields it needs,
# does its work, and hands the updated notepad to the next step.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# ParsedProblem
# ──────────────────────────────────────────────────────────────────────────────
# This is what Step 1 (the LLM intake agent) produces.
# It's a structured version of whatever the user typed in plain English.
#
# Example: User typed  →  "Pick 3 from AAPL, GOOGL, MSFT with returns 0.9, 0.8, 0.75"
#          ParsedProblem →  { assets: ["AAPL","GOOGL","MSFT"],
#                             objective: "maximize",
#                             objective_weights: {"AAPL":0.9, ...},
#                             num_select: 3, ... }
# ──────────────────────────────────────────────────────────────────────────────
class ParsedProblem(TypedDict, total=False):
    """Structured output produced by Module 1 (intake_agent)."""

    # List of asset/variable names the user mentioned  e.g. ["AAPL", "GOOGL"]
    assets: list[str]

    # Whether we want to maximize (returns) or minimize (cost/risk)
    objective: str  # "maximize" or "minimize"

    # How much each asset is worth  e.g. {"AAPL": 0.9, "GOOGL": 0.8}
    objective_weights: dict[str, float]

    # How many assets the user wants to select  e.g. 3
    num_select: int

    # Rules that MUST be followed  e.g. [{"type": "cardinality", "value": 3}]
    # Violating these gives a big penalty in the quantum math
    hard_constraints: list[dict]

    # Rules that are nice-to-have but not mandatory (smaller penalty)
    soft_constraints: list[dict]

    # If the LLM couldn't understand the problem, this is True
    # and the pipeline stops to ask the user a follow-up question
    clarification_needed: bool

    # The actual follow-up question to show the user
    clarification_question: str


# ──────────────────────────────────────────────────────────────────────────────
# AgentState
# ──────────────────────────────────────────────────────────────────────────────
# This is the FULL notepad passed between all 5 pipeline steps.
# Fields are filled in progressively as the pipeline runs.
#
# Fields are grouped by which step fills them in:
#   INPUT      → you fill this before calling the pipeline
#   MODULE 1   → intake_agent.py fills this
#   MODULE 2   → qubo_formulator.py fills this
#   MODULE 3   → circuit_constructor.py fills this
#   MODULE 4   → execution_manager.py fills this
#   MODULE 5   → results_analyzer.py fills this
#   CONTROL    → any module can update these (errors, logs, retries)
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """
    Shared state that flows between all LangGraph nodes.

    Fields are populated progressively as the graph executes:
      intake → formulate → construct_circuit → execute → analyze → END
    """

    # ── INPUT ─────────────────────────────────────────────────────────────────
    # What the user typed in the text box  e.g. "Pick 3 stocks from..."
    user_input: str

    # Which quantum computer to run on
    # "aer_simulator" = local CPU simulation (free, fast)
    # "ibm_brisbane"  = real IBM quantum hardware (requires token, slower)
    backend_name: str

    # ── MODULE 1 OUTPUT ───────────────────────────────────────────────────────
    # The LLM's structured understanding of the user's plain English problem
    parsed_problem: ParsedProblem

    # ── MODULE 2 OUTPUT ───────────────────────────────────────────────────────
    # The QUBO matrix — a 2D grid of numbers that encodes the problem mathematically
    # Used only for the heatmap visualization in the UI
    qubo_matrix: list[list[float]]

    # A constant number that gets added to the QUBO energy (offset from reformulation)
    qubo_offset: float

    # The Ising Hamiltonian — the quantum version of the problem
    # This is a Qiskit SparsePauliOp object (a sum of Pauli operators like Z, ZZ, etc.)
    # The quantum circuit is built specifically to minimize this
    ising_hamiltonian: Any

    # Constant energy offset from converting QUBO → Ising
    ising_offset: float

    # ── MODULE 3 OUTPUT ───────────────────────────────────────────────────────
    # The quantum circuit written as text (OpenQASM 3 format)
    # Like "source code" for the quantum computer
    circuit_qasm: str

    # How many qubits the circuit uses (one per asset, usually)
    num_qubits: int

    # How many layers deep the circuit is after compilation for hardware
    # Deeper = slower and more error-prone on real hardware
    circuit_depth: int

    # How many adjustable parameters (knobs) the circuit has
    # For QAOA with reps=1: 2 params (γ and β)
    # For QAOA with reps=2: 4 params (γ₁, β₁, γ₂, β₂)
    num_parameters: int

    # True = circuit will fit on the hardware, False = too big
    transpilation_feasible: bool

    # ── MODULE 4 OUTPUT ───────────────────────────────────────────────────────
    # The random starting values for γ and β before optimization begins
    initial_params: list[float]

    # The best γ and β values found after COBYLA optimization
    optimal_params: list[float]

    # The lowest energy the optimizer achieved  (more negative = better solution)
    optimal_energy: float

    # Energy value at each COBYLA iteration — used to draw the convergence graph
    convergence_history: list[float]

    # How many times each bitstring was measured
    # e.g. {"101": 312, "110": 289, "011": 198, ...}  (1024 total shots)
    bitstring_counts: dict[str, int]

    # ── MODULE 5 OUTPUT ───────────────────────────────────────────────────────
    # Human-readable result — selected assets, objective value, charts, etc.
    final_solution: dict[str, Any]

    # How close the quantum answer is to the best possible classical answer
    # 1.0 = perfect, 0.9 = 90% of optimal
    approximation_ratio: float

    # True if the best solution follows all the hard rules (e.g. exactly k assets)
    constraint_satisfied: bool

    # ── CONTROL & LOGGING ─────────────────────────────────────────────────────
    # A running list of messages from every step (like a detailed log file)
    logs: list[str]

    # If any step fails, it puts an error message here
    # The graph router sees this and sends execution to the error handler
    error: str | None

    # How many times we've retried after an error (max 2 retries allowed)
    retry_count: int
