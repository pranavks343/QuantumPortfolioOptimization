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

app = Server("quantum-copilot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_aer_backend():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def _get_ibm_backend(name: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
        token=os.getenv("IBM_QUANTUM_TOKEN"),
    )
    return service.backend(name)


def _resolve_backend(backend_name: str):
    if backend_name in ("aer_simulator", "local", "", None):
        return _get_aer_backend(), "local"
    try:
        return _get_ibm_backend(backend_name), "ibm"
    except Exception as exc:
        logger.warning(f"IBM backend '{backend_name}' unavailable ({exc}), using AerSimulator.")
        return _get_aer_backend(), "local"


def _load_circuit(circuit_qasm: str):
    """Parse an OpenQASM 3 string into a QuantumCircuit."""
    from qiskit.qasm3 import loads as qasm3_loads
    return qasm3_loads(circuit_qasm)


def _bind_params(circuit, params: list[float]):
    """Bind a flat list of parameter values to the circuit's ParameterVector."""
    param_dict = dict(zip(sorted(circuit.parameters, key=lambda p: p.name), params))
    return circuit.assign_parameters(param_dict)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

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


async def _run_qaoa_sampler(
    circuit_qasm: str,
    params: list[float],
    shots: int = 1024,
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler

        circuit = _load_circuit(circuit_qasm)
        bound = _bind_params(circuit, params)
        # Only add measurements if the circuit doesn't already have them
        if not any(bound.count_ops().get(gate, 0) for gate in ("measure",)):
            bound.measure_all()

        backend, kind = _resolve_backend(backend_name)

        if kind == "local":
            sampler = AerSampler()
            job = sampler.run([bound], shots=shots)
            result = job.result()
            # AerSampler V2 result handling
            counts_raw = result[0].data.meas.get_counts()
            counts = {k: int(v) for k, v in counts_raw.items()}
        else:
            from qiskit_ibm_runtime import SamplerV2, Session
            with Session(backend=backend) as session:
                sampler = SamplerV2(session=session)
                job = sampler.run([bound], shots=shots)
                result = job.result()
                counts_raw = result[0].data.meas.get_counts()
                counts = {k: int(v) for k, v in counts_raw.items()}

        return [TextContent(type="text", text=json.dumps({"counts": counts, "shots": shots, "backend": backend_name}))]

    except Exception as exc:
        logger.error(f"run_qaoa_sampler error: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


async def _run_estimator(
    circuit_qasm: str,
    observables: str,
    params: list[float],
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        circuit = _load_circuit(circuit_qasm)

        # Deserialise the observable
        obs_data = json.loads(observables)
        paulis = [item[0] for item in obs_data]
        coeffs = [item[1] for item in obs_data]
        observable = SparsePauliOp(paulis, coeffs=coeffs)

        backend, kind = _resolve_backend(backend_name)

        if kind == "local":
            estimator = AerEstimator()
            pub = (circuit, observable, params)
            job = estimator.run([pub])
            result = job.result()
            ev = float(result[0].data.evs)
        else:
            from qiskit_ibm_runtime import EstimatorV2, Session
            with Session(backend=backend) as session:
                estimator = EstimatorV2(session=session)
                pub = (circuit, observable, params)
                job = estimator.run([pub])
                result = job.result()
                ev = float(result[0].data.evs)

        return [TextContent(type="text", text=json.dumps({"expectation_value": ev, "backend": backend_name}))]

    except Exception as exc:
        logger.error(f"run_estimator error: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


async def _list_backends() -> list[TextContent]:
    backends_info = [{"name": "aer_simulator", "qubits": 32, "status": "available", "type": "local"}]
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.getenv("IBM_QUANTUM_TOKEN")
        if token:
            service = QiskitRuntimeService(
                channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
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
        logger.warning(f"Could not fetch IBM Quantum backends: {exc}")

    return [TextContent(type="text", text=json.dumps({"backends": backends_info}))]


async def _validate_circuit(
    circuit_qasm: str,
    backend_name: str = "aer_simulator",
) -> list[TextContent]:
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        circuit = _load_circuit(circuit_qasm)
        backend, _ = _resolve_backend(backend_name)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        transpiled = pm.run(circuit)

        depth = transpiled.depth()
        gate_count = sum(transpiled.count_ops().values())
        num_qubits = transpiled.num_qubits
        backend_qubits = backend.num_qubits if hasattr(backend, "num_qubits") else 127
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
