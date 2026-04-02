"""
Qiskit Utility Functions
=========================
Standalone helpers used across modules to avoid code duplication.
These do NOT depend on the MCP server; they use Qiskit directly.
"""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def circuit_to_qasm(circuit) -> str:
    """Serialise a QuantumCircuit to an OpenQASM 3 string."""
    from qiskit.qasm3 import dumps
    return dumps(circuit)


def qasm_to_circuit(qasm_str: str):
    """Parse an OpenQASM 3 string back to a QuantumCircuit."""
    from qiskit.qasm3 import loads
    return loads(qasm_str)


def hamiltonian_to_json(sparse_pauli_op) -> str:
    """
    Serialise a SparsePauliOp to a JSON list of [pauli_str, coeff] pairs.
    Compatible with the MCP server's observables format.
    """
    pairs = []
    for pauli, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
        pairs.append([str(pauli), float(np.real(coeff))])
    return json.dumps(pairs)


def json_to_hamiltonian(json_str: str):
    """Reconstruct a SparsePauliOp from a JSON list of [pauli_str, coeff] pairs."""
    from qiskit.quantum_info import SparsePauliOp
    pairs = json.loads(json_str)
    paulis = [p[0] for p in pairs]
    coeffs = [p[1] for p in pairs]
    return SparsePauliOp(paulis, coeffs=coeffs)


def most_probable_bitstring(counts: dict[str, int]) -> str:
    """Return the bitstring with the highest measurement count."""
    return max(counts, key=counts.get)


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalise measurement counts to probabilities."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def compute_approximation_ratio(
    quantum_value: float,
    classical_value: float,
    objective: str = "maximize",
) -> float:
    """
    Compute the QAOA approximation ratio.
    For maximisation: ratio = quantum_value / classical_value  (ideal = 1.0)
    For minimisation: ratio = classical_value / quantum_value  (ideal = 1.0)
    """
    if classical_value == 0:
        return 1.0
    if objective == "maximize":
        return quantum_value / classical_value
    else:
        return classical_value / quantum_value
