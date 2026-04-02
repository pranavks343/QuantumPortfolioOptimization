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

from __future__ import annotations

import logging
import os
from typing import Any

from agents.states import AgentState

logger = logging.getLogger(__name__)

# Depth limit considered "safe" for near-term hardware
_MAX_SAFE_DEPTH = 200
# Default number of QAOA layers (p)
_DEFAULT_REPS = 1


def _get_backend(backend_name: str) -> Any:
    """Return a Qiskit backend object for the given name."""
    from qiskit_aer import AerSimulator

    if backend_name in ("aer_simulator", "local", "", None):
        return AerSimulator()

    # Try IBM Quantum
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
            token=os.getenv("IBM_QUANTUM_TOKEN"),
        )
        return service.backend(backend_name)
    except Exception as exc:
        logger.warning(
            f"Could not load IBM Quantum backend '{backend_name}': {exc}. "
            "Falling back to AerSimulator."
        )
        return AerSimulator()


def _choose_reps(num_qubits: int) -> int:
    """
    Heuristically choose the number of QAOA layers.
    More qubits → fewer layers to keep depth manageable.
    """
    if num_qubits <= 5:
        return 2
    elif num_qubits <= 10:
        return 1
    else:
        return 1


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

    if ising_hamiltonian is None:
        return {**state, "error": "[circuit_constructor] No Ising Hamiltonian in state.", "logs": logs}

    try:
        reps = _choose_reps(num_qubits)
        logs.append(f"[circuit_constructor] Building QAOAAnsatz: {num_qubits} qubits, reps={reps}")

        # Build the parameterised QAOA ansatz
        ansatz = QAOAAnsatz(cost_operator=ising_hamiltonian, reps=reps)
        num_parameters = ansatz.num_parameters
        logs.append(f"[circuit_constructor] Ansatz has {num_parameters} variational parameters")

        # ── Transpilation Check ────────────────────────────────────────────
        backend = _get_backend(backend_name)
        backend_num_qubits = (
            backend.num_qubits
            if hasattr(backend, "num_qubits")
            else 127  # Safe default for unknown backends
        )

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

        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=backend,
        )
        transpiled = pm.run(ansatz)

        circuit_depth = transpiled.depth()
        transpilation_feasible = circuit_depth <= _MAX_SAFE_DEPTH

        if not transpilation_feasible:
            logs.append(
                f"[circuit_constructor] WARNING: transpiled depth={circuit_depth} "
                f"exceeds safe limit ({_MAX_SAFE_DEPTH}). Consider reducing reps."
            )
        else:
            logs.append(
                f"[circuit_constructor] Transpilation OK: depth={circuit_depth}, "
                f"gate_count={sum(transpiled.count_ops().values())}"
            )

        # Decompose high-level gates (e.g. QAOA) to standard gates before QASM export
        decomposed = ansatz
        for _ in range(3):
            decomposed = decomposed.decompose()

        # Serialise to OpenQASM 3 for transport via MCP
        from qiskit.qasm3 import dumps as qasm3_dumps
        circuit_qasm = qasm3_dumps(decomposed)

        return {
            **state,
            "circuit_qasm": circuit_qasm,
            "circuit_depth": circuit_depth,
            "num_parameters": num_parameters,
            "transpilation_feasible": transpilation_feasible,
            "logs": logs,
            "error": None,
        }

    except Exception as exc:
        msg = f"[circuit_constructor] Error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
