"""
Module 4: Hybrid Execution Manager via MCP
==========================================
This is the bridge between the AI agent and the quantum computer.

It implements the Variational Quantum Eigensolver (VQE) / QAOA hybrid loop:
  1. Start with random initial gamma/beta parameters
  2. Classical COBYLA optimiser proposes new parameters
  3. MCP Estimator tool evaluates <H> (expectation value) on the quantum backend
  4. COBYLA repeats until convergence
  5. Final parameters are used with the MCP Sampler to get the bitstring distribution

Error Handling / Fallback:
  - Retries transient execution failures (up to 3 attempts)
  - Preserves the requested backend instead of silently falling back

Logging:
  - Every COBYLA iteration logs (circuit_depth, shots, current params, energy)
  - Written to both state['logs'] and a JSON file under logs/
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from agents.states import AgentState

logger = logging.getLogger(__name__)

# Number of times to retry a quantum call before giving up
_MAX_RETRIES = 3

# Number of times to measure the quantum circuit in the final sampling phase.
# More shots → more accurate probability distribution, but takes longer.
_DEFAULT_SHOTS = 1024

# Maximum number of optimization rounds COBYLA is allowed to run.
# More iterations → closer to optimal, but takes more time.
_COBYLA_MAX_ITER = 200

_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))


def _serialise_hamiltonian(ising_hamiltonian) -> str:
    """
    Convert a Qiskit SparsePauliOp into a JSON string.

    The MCP server can't receive Python objects — only text.
    So we convert the Hamiltonian to a list of [pauli_string, coefficient] pairs.

    Example output: '[["ZZ", -0.5], ["ZI", 0.25], ["IZ", 0.25]]'
    """
    pairs = []
    for pauli, coeff in zip(ising_hamiltonian.paulis, ising_hamiltonian.coeffs):
        # str(pauli) → something like "ZZII" or "IZZI"
        # np.real(coeff) → strip the imaginary part (should be 0 for valid Hamiltonians)
        pairs.append([str(pauli), float(np.real(coeff))])
    return json.dumps(pairs)


def _call_estimator(
    circuit_qasm: str,
    observables_json: str,
    params: list[float],
    backend_name: str,
) -> float:
    """
    Ask the MCP server to run the quantum circuit and compute ⟨H⟩ (the energy).

    This is called by COBYLA at each optimization iteration.
    Lower energy = better parameter settings = better solution to the problem.

    Returns a single float: the expectation value ⟨ψ(θ)|H|ψ(θ)⟩
    """
    from tools.mcp_client import _call_mcp_tool

    result = _call_mcp_tool("run_estimator", {
        "circuit_qasm": circuit_qasm,
        "observables": observables_json,
        "params": list(params),
        "backend_name": backend_name,
    })

    if "error" in result:
        raise RuntimeError(f"Estimator MCP error: {result['error']}")

    return float(result["expectation_value"])


def _call_sampler(
    circuit_qasm: str,
    params: list[float],
    shots: int,
    backend_name: str,
) -> dict[str, int]:
    """
    Ask the MCP server to run the circuit many times and count the results.

    This is the final measurement phase after optimization is complete.
    We use the best parameters found by COBYLA.

    Returns a dictionary like: {"101": 312, "110": 289, "011": 198, ...}
    where the keys are bitstrings and values are how many times each appeared.
    """
    from tools.mcp_client import _call_mcp_tool

    result = _call_mcp_tool("run_qaoa_sampler", {
        "circuit_qasm": circuit_qasm,
        "params": list(params),
        "shots": shots,
        "backend_name": backend_name,
    })

    if "error" in result:
        raise RuntimeError(f"Sampler MCP error: {result['error']}")

    return result.get("counts", {})


def _save_experiment_log(log_data: dict) -> None:
    """
    Save the full experiment record to a JSON file in the logs/ directory.

    Each run gets its own timestamped file so you can review experiments later.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOG_DIR / f"experiment_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Experiment log saved to {log_path}")


def run_hybrid_loop(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 4 — Hybrid Execution Manager via MCP.

    Reads:   state['circuit_qasm'], state['ising_hamiltonian'],
             state['num_parameters'], state['backend_name'],
             state['transpilation_feasible']
    Writes:  state['initial_params'], state['optimal_params'],
             state['optimal_energy'], state['convergence_history'],
             state['bitstring_counts'], state['logs']
    """
    circuit_qasm: str = state.get("circuit_qasm", "")
    ising_hamiltonian = state.get("ising_hamiltonian")
    num_parameters: int = state.get("num_parameters", 2)
    backend_name: str = state.get("backend_name", "aer_simulator")
    transpilation_feasible: bool = state.get("transpilation_feasible", True)
    logs: list[str] = list(state.get("logs", []))
    retry_count: int = state.get("retry_count", 0)

    if not circuit_qasm:
        return {**state, "error": "[execution_manager] No circuit QASM in state.", "logs": logs}

    if ising_hamiltonian is None:
        return {**state, "error": "[execution_manager] No Ising Hamiltonian in state.", "logs": logs}

    if not transpilation_feasible:
        return {**state, "error": "[execution_manager] Circuit failed transpilation check.", "logs": logs}

    # Running every COBYLA evaluation on a real QPU makes the UI appear stuck
    # for a very long time because each objective call becomes its own hardware
    # job. Use the simulator for the iterative optimiser and reserve the chosen
    # backend for the final sampling pass that feeds the frontend results.
    requested_backend = backend_name
    estimator_backend = "aer_simulator" if requested_backend != "aer_simulator" else requested_backend
    sampler_backend = requested_backend

    # Convert the Hamiltonian Python object to a JSON string for MCP transport
    observables_json = _serialise_hamiltonian(ising_hamiltonian)

    # ── Initialize random starting parameters ─────────────────────────────────
    rng = np.random.default_rng(seed=42)
    initial_params = rng.uniform(0, np.pi, size=num_parameters).tolist()

    convergence_history: list[float] = []

    logs.append(
        f"[execution_manager] Starting COBYLA optimisation on backend='{estimator_backend}' "
        f"with {num_parameters} parameters, max_iter={_COBYLA_MAX_ITER}"
    )
    if sampler_backend != estimator_backend:
        logs.append(
            f"[execution_manager] Requested IBM backend '{sampler_backend}'. "
            f"Using '{estimator_backend}' for COBYLA objective evaluations and "
            f"'{sampler_backend}' for the final sampling pass."
        )

    iteration_logs = []

    # ── Define the objective function for COBYLA ──────────────────────────────
    def objective(params_array: np.ndarray) -> float:
        params_list = params_array.tolist()

        # Try up to _MAX_RETRIES times in case of transient quantum hardware errors
        for attempt in range(_MAX_RETRIES):
            try:
                ev = _call_estimator(circuit_qasm, observables_json, params_list, estimator_backend)

                convergence_history.append(ev)
                iteration_logs.append({
                    "iteration": len(convergence_history),
                    "params": params_list,
                    "energy": ev,
                    "backend": estimator_backend,
                })

                # Log progress every 10 iterations to avoid flooding the output
                if len(convergence_history) % 10 == 0:
                    logger.info(f"  Iteration {len(convergence_history)}: energy={ev:.6f}")

                return ev

            except RuntimeError as exc:
                logger.warning(f"  Attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}")
                if attempt == _MAX_RETRIES - 1:
                    raise

        return float("inf")  # Should never reach here, but satisfies the type checker

    # ── Run COBYLA optimization ───────────────────────────────────────────────
    try:
        opt_result = minimize(
            objective,
            x0=np.array(initial_params),
            method="COBYLA",
            options={"maxiter": _COBYLA_MAX_ITER, "rhobeg": 0.5},
        )
        optimal_params = opt_result.x.tolist()
        optimal_energy = float(opt_result.fun)
        logs.append(
            f"[execution_manager] COBYLA converged in {len(convergence_history)} iterations. "
            f"Optimal energy={optimal_energy:.6f}"
        )
    except Exception as exc:
        msg = f"[execution_manager] COBYLA optimisation failed: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}

    # ── Sample the optimized circuit ──────────────────────────────────────────
    shots = _DEFAULT_SHOTS
    logs.append(
        f"[execution_manager] Sampling optimised circuit with {shots} shots on "
        f"backend='{sampler_backend}'..."
    )
    try:
        bitstring_counts = _call_sampler(circuit_qasm, optimal_params, shots, sampler_backend)

        if bitstring_counts:
            best_bitstring = max(bitstring_counts, key=bitstring_counts.get)
            logs.append(
                f"[execution_manager] Sampling complete. "
                f"Top outcome: {best_bitstring}"
            )
        else:
            logs.append("[execution_manager] Sampling returned empty counts.")
            return {**state, "error": "[execution_manager] Sampling returned no measurements.", "logs": logs}

    except Exception as exc:
        msg = f"[execution_manager] Sampling failed: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}

    # ── Save the experiment record to disk ────────────────────────────────────
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "requested_backend": requested_backend,
        "estimator_backend": estimator_backend,
        "sampler_backend": sampler_backend,
        "num_qubits": state.get("num_qubits"),
        "circuit_depth": state.get("circuit_depth"),
        "num_parameters": num_parameters,
        "shots": shots,
        "initial_params": initial_params,
        "optimal_params": optimal_params,
        "optimal_energy": optimal_energy,
        "convergence_history": convergence_history,
        "iteration_logs": iteration_logs,
    }
    try:
        _save_experiment_log(experiment_log)
    except Exception as exc:
        # Don't fail the whole pipeline just because saving the log failed
        logger.warning(f"Could not save experiment log: {exc}")

    return {
        **state,
        "initial_params": initial_params,
        "estimator_backend": estimator_backend,
        "sampler_backend": sampler_backend,
        "optimal_params": optimal_params,
        "optimal_energy": optimal_energy,
        "convergence_history": convergence_history,
        "bitstring_counts": bitstring_counts,
        "backend_name": sampler_backend,
        "logs": logs,
        "error": None,
    }
