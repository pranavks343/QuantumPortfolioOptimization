"""
Qiskit Utility Functions
=========================
Standalone helpers used across modules to avoid code duplication.
These do NOT depend on the MCP server; they use Qiskit directly.
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# A collection of small utility functions shared across the pipeline modules.
# Each function does exactly one simple thing and has no side effects.
#
# These are separate from the modules to avoid copy-pasting the same logic
# in multiple places. If we need to change how circuits are serialized,
# we change it here and everywhere that uses it updates automatically.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def circuit_to_qasm(circuit) -> str:
    """
    Convert a Qiskit QuantumCircuit object into an OpenQASM 3 text string.

    OpenQASM 3 is a standard text format for describing quantum circuits.
    We use it to send circuits over the MCP protocol (which only handles text).

    Example output (simplified):
        OPENQASM 3.0;
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
    """
    from qiskit.qasm3 import dumps
    return dumps(circuit)


def qasm_to_circuit(qasm_str: str):
    """
    Parse an OpenQASM 3 text string back into a Qiskit QuantumCircuit object.
    This is the reverse of circuit_to_qasm().
    """
    from qiskit.qasm3 import loads
    return loads(qasm_str)


def hamiltonian_to_json(sparse_pauli_op) -> str:
    """
    Convert a Qiskit SparsePauliOp (the Ising Hamiltonian) into a JSON string.

    A SparsePauliOp is a sum of Pauli operator terms, e.g.:
        H = -0.5 * ZZ + 0.25 * ZI + 0.25 * IZ

    This gets serialized as:
        '[["ZZ", -0.5], ["ZI", 0.25], ["IZ", 0.25]]'

    This format is required by the MCP server's run_estimator tool,
    which needs to receive the Hamiltonian as plain text (not a Python object).
    """
    pairs = []
    for pauli, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
        # str(pauli) → "ZZ" or "ZZII" etc.
        # np.real(coeff) → strip imaginary part (should be 0 for valid Hamiltonians)
        pairs.append([str(pauli), float(np.real(coeff))])
    return json.dumps(pairs)


def json_to_hamiltonian(json_str: str):
    """
    Reconstruct a SparsePauliOp from a JSON string (the reverse of hamiltonian_to_json).

    Input:  '[["ZZ", -0.5], ["ZI", 0.25], ["IZ", 0.25]]'
    Output: SparsePauliOp object
    """
    from qiskit.quantum_info import SparsePauliOp
    pairs = json.loads(json_str)
    paulis = [p[0] for p in pairs]   # ["ZZ", "ZI", "IZ"]
    coeffs = [p[1] for p in pairs]   # [-0.5, 0.25, 0.25]
    return SparsePauliOp(paulis, coeffs=coeffs)


def most_probable_bitstring(counts: dict[str, int]) -> str:
    """
    Return the bitstring that was measured most often.

    Example: {"101": 312, "110": 289, "011": 198}  →  "101"
    """
    return max(counts, key=counts.get)


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """
    Convert raw measurement counts to probabilities (fractions that sum to 1).

    Example: {"101": 512, "110": 512}  →  {"101": 0.5, "110": 0.5}
    """
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def compute_approximation_ratio(
    quantum_value: float,
    classical_value: float,
    objective: str = "maximize",
) -> float:
    """
    Compute how close the quantum solution is to the best possible classical solution.

    For maximization: ratio = quantum_value / classical_value
      - 1.0 = quantum matched the best possible answer
      - 0.9 = quantum got 90% of the optimal value

    For minimization: ratio = classical_value / quantum_value
      - 1.0 = quantum matched the minimum possible cost
      - Values < 1.0 mean the quantum solution was worse (higher cost)
    """
    if classical_value == 0:
        return 1.0   # Avoid division by zero

    if objective == "maximize":
        return quantum_value / classical_value
    else:
        return classical_value / quantum_value
