"""
MCP Server: Quantum Tool Interface
====================================
Exposes Qiskit quantum execution capabilities as MCP tools that the
LangGraph orchestrator can call via the Model Context Protocol.

Tools exposed:
  - run_qaoa_sampler   : Run a parameterised QAOA circuit and return bitstring counts
  - run_estimator      : Compute expectation value of an observable (for COBYLA optimisation)
  - list_backends      : List available IBM Quantum or local simulator backends
  - validate_circuit   : Check circuit depth / gate count / qubit feasibility

Run with:
  python mcp_server.py

The server speaks stdio-based MCP by default (suitable for subprocess transport).
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This file is a standalone server that exposes 4 quantum tools over the
# MCP (Model Context Protocol). It runs as a SUBPROCESS — meaning the main app
# launches it in the background and communicates with it via stdin/stdout pipes.
#
# WHY A SEPARATE SERVER?
#   - Keeps quantum execution isolated from the main Python process
#   - MCP is a standard protocol, so other AI tools can also call these tools
#   - The subprocess can be restarted independently if it crashes
#
# HOW DOES COMMUNICATION WORK?
#   main app (execution_manager.py)
#       ↓  calls _call_mcp_tool("run_estimator", {...})
#   mcp_client.py
#       ↓  launches this script as a subprocess
#       ↓  sends JSON-RPC message over stdin pipe
#   mcp_server.py  ← THIS FILE
#       ↓  receives the message, runs Qiskit code
#       ↓  sends JSON result back over stdout pipe
#   mcp_client.py
#       ↓  returns the parsed result to execution_manager.py
#
# THE 4 TOOLS:
#   1. run_qaoa_sampler  — measure the circuit many times → bitstring histogram
#   2. run_estimator     — compute ⟨H⟩ (energy) for one parameter setting
#   3. list_backends     — list available quantum backends
#   4. validate_circuit  — check if a circuit fits on a given backend
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import logging
import os

import numpy as np
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create the MCP server instance with a name identifier
app = Server("quantum-copilot")


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_aer_backend():
    """Return a local Qiskit AerSimulator (runs on your CPU, no account needed)."""
    from qiskit_aer import AerSimulator
    return AerSimulator()


def _runtime_channel() -> str:
    """Normalize legacy channel names to the current Qiskit Runtime values."""
    channel = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform").strip()
    if channel == "ibm_quantum":
        return "ibm_quantum_platform"
    return channel or "ibm_quantum_platform"


def _get_ibm_backend(name: str):
    """Connect to a real IBM Quantum machine using the token from the .env file."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError("IBM_QUANTUM_TOKEN is not configured.")
    service = QiskitRuntimeService(
        channel=_runtime_channel(),
        token=token,
    )
    return service.backend(name)


def _resolve_backend(backend_name: str):
    """
    Return the appropriate Qiskit backend object and a label ("local" or "ibm").

    If the name is "aer_simulator" (or empty), use the local simulator.
    If the name is an IBM backend (e.g. "ibm_brisbane"), require a real IBM backend.
    """
    if backend_name in ("aer_simulator", "local", "", None):
        return _get_aer_backend(), "local"
    return _get_ibm_backend(backend_name), "ibm"


def _extract_counts(pub_result) -> dict[str, int]:
    """
    Robustly extract a {bitstring: count} dict from a SamplerV2 pub result.

    After transpilation the classical register might be named 'meas', 'c', 'c0',
    or something else depending on the Qiskit version and circuit structure.
    This helper tries common names and then falls back to scanning all attributes.
    """
    data = pub_result.data
    # Try the most common classical-register names first
    for reg_name in ("meas", "c", "c0", "cr", "m"):
        bit_array = getattr(data, reg_name, None)
        if bit_array is not None and hasattr(bit_array, "get_counts"):
            return {k: int(v) for k, v in bit_array.get_counts().items()}
    # Fallback: scan every attribute on DataBin for one that has get_counts()
    for attr_name in dir(data):
        if attr_name.startswith("_"):
            continue
        attr = getattr(data, attr_name, None)
        if attr is not None and hasattr(attr, "get_counts"):
            return {k: int(v) for k, v in attr.get_counts().items()}
    raise RuntimeError(
        f"Could not find a measurement register in SamplerV2 result. "
        f"Available data fields: {dir(data)}"
    )


def _load_circuit(circuit_qasm: str):
    """
    Parse an OpenQASM 3 text string into a Qiskit QuantumCircuit object.

    OpenQASM 3 is a text format for quantum circuits — like Python source code
    for quantum programs. Qiskit's qasm3.loads() compiles it back into an object.
    """
    from qiskit.qasm3 import loads as qasm3_loads
    return qasm3_loads(circuit_qasm)


def _bind_params(circuit, params: list[float]):
    """
    Plug concrete numbers into the circuit's parameter "slots" (γ and β values).

    A QAOA circuit has named parameters like θ[0], θ[1], etc.
    This function replaces those placeholders with actual float values.
    Sort by name to ensure consistent ordering.
    """
    param_dict = dict(zip(sorted(circuit.parameters, key=lambda p: p.name), params))
    return circuit.assign_parameters(param_dict)


# ──────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
# This function is called when the MCP client asks "what tools do you have?"
# It returns a list of Tool objects that describe each tool's name, purpose,
# and what inputs it accepts (as JSON Schema).
# ──────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_qaoa_sampler",
            description=(
                "Execute a parameterised QAOA circuit using the Sampler primitive and "
                "return a dictionary of bitstring → count. "
                "Use this after classical optimisation converges to get the final solution distribution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "circuit_qasm": {"type": "string", "description": "OpenQASM 3 string of the QAOA circuit"},
                    "params": {"type": "array", "items": {"type": "number"}, "description": "Variational parameter values (gamma, beta)"},
                    "shots": {"type": "integer", "default": 1024, "description": "Number of measurement shots"},
                    "backend_name": {"type": "string", "default": "aer_simulator", "description": "Backend name"},
                },
                "required": ["circuit_qasm", "params"],
            },
        ),
        Tool(
            name="run_estimator",
            description=(
                "Compute the expectation value <ψ(θ)|H|ψ(θ)> for given parameters θ. "
                "Used by the classical COBYLA optimiser as the objective function to minimise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "circuit_qasm": {"type": "string", "description": "OpenQASM 3 string of the QAOA ansatz"},
                    "observables": {"type": "string", "description": "JSON-serialised SparsePauliOp (list of [pauli_str, coeff])"},
                    "params": {"type": "array", "items": {"type": "number"}, "description": "Variational parameter values"},
                    "backend_name": {"type": "string", "default": "aer_simulator"},
                },
                "required": ["circuit_qasm", "observables", "params"],
            },
        ),
        Tool(
            name="list_backends",
            description="List available IBM Quantum backends with their qubit counts and operational status.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="validate_circuit",
            description=(
                "Transpile and validate a circuit against a target backend. "
                "Returns depth, gate count, qubit count, and a feasibility flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "circuit_qasm": {"type": "string"},
                    "backend_name": {"type": "string", "default": "aer_simulator"},
                },
                "required": ["circuit_qasm"],
            },
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# TOOL DISPATCH
# ──────────────────────────────────────────────────────────────────────────────
# When the MCP client calls a tool, this function receives the tool name
# and its arguments, then routes to the appropriate implementation below.
# ──────────────────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "run_qaoa_sampler":
        return await _run_qaoa_sampler(**arguments)
    elif name == "run_estimator":
        return await _run_estimator(**arguments)
    elif name == "list_backends":
        return await _list_backends()
    elif name == "validate_circuit":
        return await _validate_circuit(**arguments)
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 1: run_qaoa_sampler
# ──────────────────────────────────────────────────────────────────────────────
# This is the FINAL measurement step.
# After COBYLA found the best parameters, we run the circuit many times
# and count how often each bitstring appears.
# ──────────────────────────────────────────────────────────────────────────────

async def _run_qaoa_sampler(
    circuit_qasm: str,
    params: list[float],
    shots: int = 1024,
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler

        # Parse the OpenQASM text back into a Qiskit circuit object
        circuit = _load_circuit(circuit_qasm)

        # Plug the optimized parameters into the circuit's parameter slots
        bound = _bind_params(circuit, params)

        # Add measurement gates at the end if not already present
        # (We need measurements to get classical output from the quantum circuit)
        if not any(bound.count_ops().get(gate, 0) for gate in ("measure",)):
            bound.measure_all()

        backend, kind = _resolve_backend(backend_name)

        if kind == "local":
            # Run on the local AerSimulator (fast, no network required)
            sampler = AerSampler()
            job = sampler.run([bound], shots=shots)
            result = job.result()
            counts = _extract_counts(result[0])
        else:
            # IBM hardware requires the circuit to be transpiled into its native
            # gate set (cx, rz, sx, x) before submission. Generic gates like 'u'
            # have not been accepted since March 2024.
            from qiskit_ibm_runtime import SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            transpiled = pm.run(bound)
            sampler = SamplerV2(mode=backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            counts = _extract_counts(result[0])

        # Return results as a JSON string wrapped in TextContent (MCP format)
        return [TextContent(type="text", text=json.dumps({
            "counts": counts,       # The measurement histogram
            "shots": shots,         # How many measurements were taken
            "backend": backend_name,
        }))]

    except Exception as exc:
        logger.error(f"run_qaoa_sampler error: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 2: run_estimator
# ──────────────────────────────────────────────────────────────────────────────
# This is called HUNDREDS of times during the COBYLA optimization loop.
# Each call evaluates the energy ⟨H⟩ for one specific set of parameters.
# ──────────────────────────────────────────────────────────────────────────────

async def _run_estimator(
    circuit_qasm: str,
    observables: str,
    params: list[float],
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        # Parse the circuit from text
        circuit = _load_circuit(circuit_qasm)

        # Reconstruct the Hamiltonian from its JSON representation
        # observables is a JSON string like: '[["ZZ", -0.5], ["ZI", 0.25]]'
        obs_data = json.loads(observables)
        paulis = [item[0] for item in obs_data]   # ["ZZ", "ZI", ...]
        coeffs = [item[1] for item in obs_data]   # [-0.5, 0.25, ...]
        observable = SparsePauliOp(paulis, coeffs=coeffs)

        backend, kind = _resolve_backend(backend_name)

        if kind == "local":
            estimator = AerEstimator()
            # A "pub" (Primitive Unified Bloc) bundles circuit + observable + params
            pub = (circuit, observable, params)
            job = estimator.run([pub])
            result = job.result()
            # .evs = expectation values (a single float for one observable)
            ev = float(result[0].data.evs)
        else:
            # IBM hardware requires native-gate transpilation before submission.
            # After transpilation the qubit layout changes, so we must remap the
            # observable via apply_layout() so it still matches the right qubits.
            from qiskit_ibm_runtime import EstimatorV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            transpiled = pm.run(circuit)
            mapped_observable = observable.apply_layout(transpiled.layout)
            estimator = EstimatorV2(mode=backend)
            pub = (transpiled, mapped_observable, params)
            job = estimator.run([pub])
            result = job.result()
            ev = float(result[0].data.evs)

        return [TextContent(type="text", text=json.dumps({
            "expectation_value": ev,   # The energy ⟨H⟩ — what COBYLA is minimizing
            "backend": backend_name,
        }))]

    except Exception as exc:
        logger.error(f"run_estimator error: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 3: list_backends
# ──────────────────────────────────────────────────────────────────────────────
# Returns a list of available quantum backends.
# Always includes the local AerSimulator.
# Adds IBM Quantum machines if IBM_QUANTUM_TOKEN is set in the .env file.
# ──────────────────────────────────────────────────────────────────────────────

async def _list_backends() -> list[TextContent]:
    # Start with the local simulator (always available)
    backends_info = [{"name": "aer_simulator", "qubits": 32, "status": "available", "type": "local"}]

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.getenv("IBM_QUANTUM_TOKEN")
        if token:
            # Connect to IBM Quantum and fetch all operational machines
            service = QiskitRuntimeService(
                channel=_runtime_channel(),
                token=token,
            )
            for backend in service.backends(operational=True):
                backends_info.append({
                    "name": backend.name,
                    "qubits": backend.num_qubits,
                    "status": "operational",
                    "type": "ibm_quantum",
                })
    except Exception as exc:
        # If IBM is unreachable, just return the local simulator
        logger.warning(f"Could not fetch IBM Quantum backends: {exc}")

    return [TextContent(type="text", text=json.dumps({"backends": backends_info}))]


# ──────────────────────────────────────────────────────────────────────────────
# TOOL 4: validate_circuit
# ──────────────────────────────────────────────────────────────────────────────
# Compiles the circuit for the target backend and reports how big it is.
# Used by circuit_constructor.py to check feasibility before execution.
# ──────────────────────────────────────────────────────────────────────────────

async def _validate_circuit(
    circuit_qasm: str,
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Parse the circuit and compile it for the target backend
        circuit = _load_circuit(circuit_qasm)
        backend, _ = _resolve_backend(backend_name)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        transpiled = pm.run(circuit)

        # Measure the compiled circuit
        depth = transpiled.depth()                           # Number of sequential gate layers
        gate_count = sum(transpiled.count_ops().values())    # Total number of gates
        num_qubits = transpiled.num_qubits                   # Qubits used
        backend_qubits = backend.num_qubits if hasattr(backend, "num_qubits") else 127

        # Feasible = fits on the hardware AND is shallow enough to run reliably
        feasible = (num_qubits <= backend_qubits) and (depth <= 200)

        return [TextContent(type="text", text=json.dumps({
            "depth": depth,
            "gate_count": gate_count,
            "num_qubits": num_qubits,
            "backend_qubits": backend_qubits,
            "feasible": feasible,
            "backend": backend_name,
        }))]
    except Exception as exc:
        logger.error(f"validate_circuit error: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
# When this script is launched as a subprocess, it starts listening for
# MCP messages on stdin and sends responses on stdout.
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    # stdio_server() sets up the stdin/stdout communication channels
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
