"""
Module 3: Quantum Circuit Constructor (The "Coder")
====================================================
Takes the Ising Hamiltonian produced by Module 2 and generates the
corresponding QAOA ansatz (parameterised quantum circuit).

Responsibilities:
  - Build the QAOAAnsatz with a configurable number of QAOA layers (reps / p)
  - Run a Transpilation Check against the target backend to validate feasibility
    (qubit count ≤ backend capacity, circuit depth within coherence limits)
  - Return the transpiled circuit as both a Qiskit QuantumCircuit object and
    an OpenQASM string for serialisation to the MCP server
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is Step 3 of the pipeline. It takes the Ising Hamiltonian from Step 2
# and turns it into an actual quantum circuit (the QAOAAnsatz).
#
# WHAT IS A QAOA CIRCUIT?
#   QAOA = Quantum Approximate Optimization Algorithm.
#   It's a circuit with adjustable "knobs" called gamma (γ) and beta (β).
#   The circuit alternates between two layers:
#     - Cost layer: encodes the problem (using the Hamiltonian)
#     - Mixer layer: explores different solutions
#   Each pair of (cost + mixer) is one "rep". More reps = better solution
#   but deeper circuit (harder to run on real hardware).
#
# WHAT IS TRANSPILATION?
#   Real quantum hardware can't run arbitrary gates — it only supports a
#   specific set of native gates (like CX, RZ, SX on IBM machines).
#   Transpilation = rewriting the circuit using only those native gates,
#   and mapping logical qubits to physical ones on the chip's topology.
#   This file runs the transpiler and checks if the result is feasible:
#     - Does it fit within the qubit count? (n_qubits ≤ backend capacity)
#     - Is it shallow enough? (depth ≤ 200, beyond which errors dominate)
#
# OUTPUT: The circuit is serialized to OpenQASM 3 (text format) so it can
#   be sent over the MCP protocol to mcp_server.py for execution.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from typing import Any

from agents.states import AgentState

logger = logging.getLogger(__name__)

# A circuit with depth > 200 will accumulate too many errors on real quantum hardware.
# At this depth, the signal gets buried in noise and results become unreliable.
_MAX_SAFE_DEPTH = 200

# Default number of QAOA alternating layers (p).
# More layers = better approximation but exponentially deeper circuit.
_DEFAULT_REPS = 1


def _runtime_channel() -> str:
    """Normalize legacy channel names to the current Qiskit Runtime values."""
    channel = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform").strip()
    if channel == "ibm_quantum":
        return "ibm_quantum_platform"
    return channel or "ibm_quantum_platform"


def _get_backend(backend_name: str) -> Any:
    """Return a Qiskit backend object for the given name."""
    from qiskit_aer import AerSimulator

    # If the name is "aer_simulator" (or empty), use the local CPU simulator
    if backend_name in ("aer_simulator", "local", "", None):
        return AerSimulator()

    # Otherwise try to connect to IBM Quantum hardware using the token from .env
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        token = os.getenv("IBM_QUANTUM_TOKEN")
        if not token:
            raise RuntimeError("IBM_QUANTUM_TOKEN is not configured.")

        service = QiskitRuntimeService(
            channel=_runtime_channel(),
            token=token,
        )
        return service.backend(backend_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load IBM Quantum backend '{backend_name}': {exc}"
        ) from exc


def _choose_reps(num_qubits: int) -> int:
    """
    Pick the number of QAOA layers (reps) based on the problem size.

    More qubits = more gates per layer = deeper circuit.
    So we use fewer layers for larger problems to keep depth manageable.
    """
    if num_qubits <= 5:
        return 2   # Small problem: 2 layers gives better quality
    elif num_qubits <= 10:
        return 1   # Medium problem: 1 layer keeps it feasible
    else:
        return 1   # Large problem: stick to 1 layer to stay within hardware limits


def _to_qasm(ansatz) -> str:
    """Decompose a QAOA ansatz enough for OpenQASM 3 export."""
    decomposed = ansatz
    for _ in range(3):
        decomposed = decomposed.decompose()

    from qiskit.qasm3 import dumps as qasm3_dumps
    return qasm3_dumps(decomposed)


def build_qaoa_circuit(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 3 — Quantum Circuit Constructor.

    Reads:   state['ising_hamiltonian'], state['num_qubits'], state['backend_name']
    Writes:  state['circuit_qasm'], state['circuit_depth'], state['num_parameters'],
             state['transpilation_feasible'], state['logs']
    """
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    ising_hamiltonian = state.get("ising_hamiltonian")
    num_qubits: int = state.get("num_qubits", 0)
    backend_name: str = state.get("backend_name", "aer_simulator")
    logs: list[str] = list(state.get("logs", []))

    # Safety check: we can't build a circuit without a Hamiltonian
    if ising_hamiltonian is None:
        return {**state, "error": "[circuit_constructor] No Ising Hamiltonian in state.", "logs": logs}

    try:
        # ── Build the QAOA Ansatz ─────────────────────────────────────────────
        # QAOAAnsatz takes the Hamiltonian and creates a parameterized circuit.
        # It has "reps" pairs of (cost layer + mixer layer).
        # Each rep adds 2 parameters: one gamma (γ) and one beta (β).
        preferred_reps = _choose_reps(num_qubits)
        logs.append(
            f"[circuit_constructor] Building QAOAAnsatz: {num_qubits} qubits, "
            f"preferred_reps={preferred_reps}"
        )

        # ── Transpilation Check ───────────────────────────────────────────────
        # Get the backend object so we know its qubit count and native gate set
        backend = _get_backend(backend_name)

        # How many physical qubits does this backend have?
        backend_num_qubits = (
            backend.num_qubits
            if hasattr(backend, "num_qubits")
            else 127  # Default assumption for unknown IBM backends
        )

        # Check if the problem is too big for this hardware
        if num_qubits > backend_num_qubits:
            msg = (
                f"[circuit_constructor] Problem requires {num_qubits} qubits but "
                f"backend '{backend_name}' only has {backend_num_qubits}. "
                "Problem is too large for available hardware."
            )
            logs.append(msg)
            return {
                **state,
                "transpilation_feasible": False,
                "error": msg,
                "logs": logs,
            }

        chosen_ansatz = None
        chosen_depth = None
        chosen_num_parameters = None
        best_failed_depth = None

        for reps in range(preferred_reps, 0, -1):
            ansatz = QAOAAnsatz(cost_operator=ising_hamiltonian, reps=reps)
            num_parameters = ansatz.num_parameters
            logs.append(
                f"[circuit_constructor] Trying reps={reps} "
                f"({num_parameters} variational parameters)"
            )

            # Use a stronger optimization pass when checking hardware feasibility.
            pm = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend,
            )
            transpiled = pm.run(ansatz)
            circuit_depth = transpiled.depth()
            gate_count = sum(transpiled.count_ops().values())

            logs.append(
                f"[circuit_constructor] Transpiled reps={reps}: "
                f"depth={circuit_depth}, gate_count={gate_count}"
            )

            if circuit_depth <= _MAX_SAFE_DEPTH:
                chosen_ansatz = ansatz
                chosen_depth = circuit_depth
                chosen_num_parameters = num_parameters
                if reps != preferred_reps:
                    logs.append(
                        f"[circuit_constructor] Reduced reps from {preferred_reps} to {reps} "
                        f"to satisfy depth limit {_MAX_SAFE_DEPTH}."
                    )
                break

            best_failed_depth = circuit_depth

        if chosen_ansatz is None:
            msg = (
                f"[circuit_constructor] No feasible QAOA circuit found for backend "
                f"'{backend_name}'. Even reps=1 exceeded the safe depth limit "
                f"({_MAX_SAFE_DEPTH}); last observed depth={best_failed_depth}."
            )
            logs.append(msg)
            return {
                **state,
                "transpilation_feasible": False,
                "error": msg,
                "logs": logs,
            }

        logs.append(
            f"[circuit_constructor] Transpilation OK: depth={chosen_depth}, "
            f"num_parameters={chosen_num_parameters}"
        )

        # ── Serialize to OpenQASM 3 ───────────────────────────────────────────
        # We use the ORIGINAL ansatz (not transpiled) for the QASM export,
        # because the MCP server will retranspile it for the actual backend.
        circuit_qasm = _to_qasm(chosen_ansatz)

        return {
            **state,
            "circuit_qasm": circuit_qasm,                    # Text representation of the circuit
            "circuit_depth": chosen_depth,                   # Depth after transpilation
            "num_parameters": chosen_num_parameters,         # How many knobs COBYLA will tune
            "transpilation_feasible": True,                  # Pass/fail flag for the UI
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        msg = f"[circuit_constructor] Error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
