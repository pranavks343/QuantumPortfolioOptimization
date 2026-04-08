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
  - If IBM Quantum execution fails, automatically fall back to AerSimulator
  - Retry logic (up to 3 attempts) before propagating an error

Logging:
  - Every COBYLA iteration logs (circuit_depth, shots, current params, energy)
  - Written to both state['logs'] and a JSON file under logs/
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is Step 4 — the heart of QAOA. It runs the hybrid classical-quantum loop:
#
#   PHASE 1: OPTIMIZATION
#   ─────────────────────
#   We have a quantum circuit with adjustable knobs (γ, β parameters).
#   We need to find the best knob settings that minimize the energy ⟨H⟩.
#
#   The trick: we use a CLASSICAL optimizer (COBYLA) to tune the knobs,
#   while using a QUANTUM circuit to evaluate the energy for each setting.
#
#   Round 1:  COBYLA proposes γ=0.3, β=1.2
#             → send to MCP server → run quantum circuit → get energy = -2.1
#   Round 2:  COBYLA proposes γ=0.4, β=1.1
#             → send to MCP server → run quantum circuit → get energy = -2.4 ✓
#   ... repeat up to 200 times until energy stops improving ...
#
#   PHASE 2: SAMPLING
#   ─────────────────
#   Once we have the best knob settings, we run the circuit 1024 times.
#   Each run collapses to a binary string (e.g. "10110").
#   We count how often each string appears → the probability distribution.
#   The most frequent string = most likely best solution.
#
# HOW ARE QUANTUM CALLS MADE?
#   Via MCP (Model Context Protocol) — we call tools on mcp_server.py,
#   which runs the actual Qiskit code and returns results as JSON.
#   This keeps the quantum execution isolated in the MCP server process.
# ──────────────────────────────────────────────────────────────────────────────

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

# Where to save the JSON experiment log files
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
        "circuit_qasm": circuit_qasm,          # The circuit as text
        "observables": observables_json,        # The Hamiltonian as JSON text
        "params": list(params),                 # Current γ, β values to try
        "backend_name": backend_name,           # Which quantum backend to use
    })

    if "error" in result:
        raise RuntimeError(f"Estimator MCP error: {result['error']}")

    return float(result["expectation_value"])   # The energy ⟨H⟩


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
        "params": list(params),     # The optimized parameters
        "shots": shots,             # How many times to measure (1024)
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
    # Pull everything we need from the shared state
    circuit_qasm: str = state.get("circuit_qasm", "")
    ising_hamiltonian = state.get("ising_hamiltonian")
    num_parameters: int = state.get("num_parameters", 2)
    backend_name: str = state.get("backend_name", "aer_simulator")
    transpilation_feasible: bool = state.get("transpilation_feasible", True)
    logs: list[str] = list(state.get("logs", []))
    retry_count: int = state.get("retry_count", 0)

    # Safety checks before we start the expensive optimization loop
    if not circuit_qasm:
        return {**state, "error": "[execution_manager] No circuit QASM in state.", "logs": logs}

    if ising_hamiltonian is None:
        return {**state, "error": "[execution_manager] No Ising Hamiltonian in state.", "logs": logs}

    if not transpilation_feasible:
        return {**state, "error": "[execution_manager] Circuit failed transpilation check.", "logs": logs}

    # Track which backend is actively being used (may switch during the loop)
    active_backend = backend_name

    # Convert the Hamiltonian Python object to a JSON string for MCP transport
    observables_json = _serialise_hamiltonian(ising_hamiltonian)

    # ── Initialize random starting parameters ─────────────────────────────────
    # We use a fixed seed (42) so the starting point is reproducible.
    # Parameters are in the range [0, π] which covers a full rotation.
    rng = np.random.default_rng(seed=42)
    initial_params = rng.uniform(0, np.pi, size=num_parameters).tolist()

    # This list will grow with one entry per COBYLA iteration
    convergence_history: list[float] = []

    logs.append(
        f"[execution_manager] Starting COBYLA optimisation on backend='{active_backend}' "
        f"with {num_parameters} parameters, max_iter={_COBYLA_MAX_ITER}"
    )

    # Detailed per-iteration log (saved to JSON file at the end)
    iteration_logs = []

    # ── Define the objective function for COBYLA ──────────────────────────────
    # COBYLA repeatedly calls this function with different parameter values.
    # Each call runs the quantum circuit and returns the energy.
    # COBYLA tries to find parameters that minimize this energy.
    def objective(params_array: np.ndarray) -> float:
        nonlocal active_backend   # Allow this inner function to update the backend variable
        params_list = params_array.tolist()

        # Try up to _MAX_RETRIES times in case of transient quantum hardware errors
        for attempt in range(_MAX_RETRIES):
            try:
                # Run the circuit on the quantum backend and get the energy
                ev = _call_estimator(circuit_qasm, observables_json, params_list, active_backend)

                # Record the energy for the convergence plot
                convergence_history.append(ev)
                iteration_logs.append({
                    "iteration": len(convergence_history),
                    "params": params_list,
                    "energy": ev,
                    "backend": active_backend,
                })

                # Log progress every 10 iterations to avoid flooding the output
                if len(convergence_history) % 10 == 0:
                    logger.info(f"  Iteration {len(convergence_history)}: energy={ev:.6f}")

                return ev  # Return the energy to COBYLA

            except RuntimeError as exc:
                logger.warning(f"  Attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}")

                # If IBM Quantum failed, switch to the local simulator for the retry
                if active_backend != "aer_simulator":
                    logger.warning("  Falling back to AerSimulator.")
                    active_backend = "aer_simulator"
                elif attempt == _MAX_RETRIES - 1:
                    # All retries exhausted on the simulator too — re-raise the error
                    raise

        return float("inf")  # Should never reach here, but satisfies the type checker

    # ── Run COBYLA optimization ───────────────────────────────────────────────
    # scipy.optimize.minimize with method="COBYLA":
    #   - x0 = starting parameter values
    #   - rhobeg = initial step size for exploring parameter space
    #   - maxiter = stop after this many function evaluations
    try:
        opt_result = minimize(
            objective,                           # The function to minimize
            x0=np.array(initial_params),         # Starting parameter values
            method="COBYLA",
            options={"maxiter": _COBYLA_MAX_ITER, "rhobeg": 0.5},
        )
        optimal_params = opt_result.x.tolist()   # Best parameters found
        optimal_energy = float(opt_result.fun)    # Best (lowest) energy achieved
        logs.append(
            f"[execution_manager] COBYLA converged in {len(convergence_history)} iterations. "
            f"Optimal energy={optimal_energy:.6f}"
        )
    except Exception as exc:
        msg = f"[execution_manager] COBYLA optimisation failed: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}

    # ── Sample the optimized circuit ──────────────────────────────────────────
    # Now that we have the best parameters, run the circuit 1024 times.
    # This gives us a probability distribution over all possible solutions.
    shots = _DEFAULT_SHOTS
    logs.append(f"[execution_manager] Sampling optimised circuit with {shots} shots...")
    try:
        bitstring_counts = _call_sampler(circuit_qasm, optimal_params, shots, active_backend)

        if bitstring_counts:
            # Log the most frequently observed bitstring
            best_bitstring = max(bitstring_counts, key=bitstring_counts.get)
            logs.append(
                f"[execution_manager] Sampling complete. "
                f"Top outcome: {best_bitstring}"
            )
        else:
            # Empty counts means something went wrong with the measurement
            logs.append("[execution_manager] Sampling returned empty counts.")
            return {**state, "error": "[execution_manager] Sampling returned no measurements.", "logs": logs}

    except Exception as exc:
        msg = f"[execution_manager] Sampling failed: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}

    # ── Save the experiment record to disk ────────────────────────────────────
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "backend": active_backend,
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

    # Return all results in the updated state
    return {
        **state,
        "initial_params": initial_params,          # Starting knob values
        "optimal_params": optimal_params,          # Best knob values found
        "optimal_energy": optimal_energy,          # Lowest energy achieved
        "convergence_history": convergence_history, # Energy at each iteration (for chart)
        "bitstring_counts": bitstring_counts,       # Measurement histogram
        "backend_name": active_backend,             # May have changed to "aer_simulator"
        "logs": logs,
        "error": None,
    }
