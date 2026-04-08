"""
MCP Client — LangChain Tool Wrappers
======================================
Wraps the four MCP server tools as LangChain BaseTool subclasses so that
the LangGraph orchestrator can call them as standard agent tools.

Each tool launches the MCP server as a subprocess (stdio transport) and
communicates with it via the MCP Python client.
"""

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This file is the BRIDGE between the main pipeline (execution_manager.py)
# and the MCP server (mcp_server.py).
#
# It has two parts:
#
# PART 1: _call_mcp_tool() — the raw communication function
#   This function:
#   1. Launches mcp_server.py as a child subprocess
#   2. Sends it a JSON-RPC message over stdin: "call tool X with args Y"
#   3. Reads the JSON response from stdout
#   4. Returns the parsed result as a Python dict
#
# PART 2: LangChain Tool classes (QAOASamplerTool, EstimatorTool, etc.)
#   These wrap _call_mcp_tool() in LangChain's BaseTool format.
#   This lets the LangGraph orchestrator discover and call them as
#   standard "agent tools" if it ever needs to use them directly.
#
# In practice, execution_manager.py calls _call_mcp_tool() directly
# (bypassing the LangChain wrapper) for simplicity.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Absolute path to the MCP server script so we can launch it as a subprocess
# __file__ = this file (mcp_client.py), .parent.parent = project root
_SERVER_PATH = Path(__file__).parent.parent / "mcp_server.py"


def _call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Launch the MCP server and call one tool on it. Returns the result as a dict.

    HOW IT WORKS:
      1. Spawn mcp_server.py as a child process using sys.executable (same Python)
      2. Connect to it via MCP's stdio client (stdin/stdout pipes)
      3. Send a "call_tool" request: { name: tool_name, arguments: {...} }
      4. Read the JSON response from the server
      5. Return the parsed Python dict

    TRICKY PART — Async vs. Sync:
      MCP uses async Python (asyncio). But Streamlit runs its own event loop.
      If we call asyncio.run() while Streamlit's loop is running → RuntimeError.
      Solution: detect if a loop is already running, and if so, use a background
      thread (ThreadPoolExecutor) to run our async code in a separate loop.
    """
    try:
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Tell MCP how to start the server: run "python mcp_server.py"
        server_params = StdioServerParameters(
            command=sys.executable,     # The current Python interpreter
            args=[str(_SERVER_PATH)],   # Path to mcp_server.py
        )

        async def _run():
            """The actual async communication logic."""
            async with stdio_client(server_params) as (read, write):
                # Establish an MCP session over the stdin/stdout pipes
                async with ClientSession(read, write) as session:
                    await session.initialize()   # MCP handshake

                    # Call the tool and wait for the response
                    result = await session.call_tool(tool_name, arguments)

                    # result.content is a list of TextContent objects
                    # We take the first one and parse its JSON text
                    if result.content:
                        return json.loads(result.content[0].text)
                    return {}

        # Detect if we're already inside a running event loop (e.g. Streamlit)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None   # No running loop — we can safely call asyncio.run()

        if loop and loop.is_running():
            # We're inside Streamlit or another async framework.
            # Run our coroutine in a separate thread with its own event loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result(timeout=300)  # 5-minute timeout for IBM Quantum jobs
        else:
            # Normal case: no existing event loop — just use asyncio.run()
            return asyncio.run(_run())

    except Exception as exc:
        logger.error(f"MCP tool call '{tool_name}' failed: {exc}")
        return {"error": str(exc)}   # Return error dict instead of raising


# ---------------------------------------------------------------------------
# Tool 1: QAOA Sampler
# ---------------------------------------------------------------------------

class QAOASamplerInput(BaseModel):
    circuit_qasm: str = Field(description="OpenQASM 3 string of the QAOA circuit")
    params: list[float] = Field(description="Variational parameter values (gamma and beta alternating)")
    shots: int = Field(default=1024, description="Number of measurement shots")
    backend_name: str = Field(default="aer_simulator", description="Backend name")


class QAOASamplerTool(BaseTool):
    name: str = "qaoa_sampler"
    description: str = (
        "Run a parameterised QAOA circuit and return a bitstring count histogram. "
        "Use this after the classical optimiser has converged to obtain the final solution."
    )
    args_schema: Type[BaseModel] = QAOASamplerInput

    def _run(
        self,
        circuit_qasm: str,
        params: list[float],
        shots: int = 1024,
        backend_name: str = "aer_simulator",
    ) -> str:
        result = _call_mcp_tool("run_qaoa_sampler", {
            "circuit_qasm": circuit_qasm,
            "params": params,
            "shots": shots,
            "backend_name": backend_name,
        })
        return json.dumps(result)

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Use synchronous _run instead.")


# ---------------------------------------------------------------------------
# Tool 2: Estimator (expectation value for COBYLA)
# ---------------------------------------------------------------------------

class EstimatorInput(BaseModel):
    circuit_qasm: str = Field(description="OpenQASM 3 string of the QAOA ansatz")
    observables: str = Field(description="JSON list of [pauli_str, coeff] pairs representing the Hamiltonian")
    params: list[float] = Field(description="Variational parameter values")
    backend_name: str = Field(default="aer_simulator")


class EstimatorTool(BaseTool):
    name: str = "estimator"
    description: str = (
        "Compute the expectation value <ψ(θ)|H|ψ(θ)> for variational parameters θ. "
        "This is the objective function minimised by the classical COBYLA optimiser."
    )
    args_schema: Type[BaseModel] = EstimatorInput

    def _run(
        self,
        circuit_qasm: str,
        observables: str,
        params: list[float],
        backend_name: str = "aer_simulator",
    ) -> str:
        result = _call_mcp_tool("run_estimator", {
            "circuit_qasm": circuit_qasm,
            "observables": observables,
            "params": params,
            "backend_name": backend_name,
        })
        return json.dumps(result)

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 3: Backend List
# ---------------------------------------------------------------------------

class BackendListTool(BaseTool):
    name: str = "list_backends"
    description: str = (
        "List available quantum backends (IBM Quantum QPUs and local simulators) "
        "with their qubit counts and operational status."
    )

    def _run(self) -> str:
        result = _call_mcp_tool("list_backends", {})
        return json.dumps(result)

    async def _arun(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 4: Circuit Validator
# ---------------------------------------------------------------------------

class CircuitValidatorInput(BaseModel):
    circuit_qasm: str = Field(description="OpenQASM 3 circuit string to validate")
    backend_name: str = Field(default="aer_simulator", description="Target backend for transpilation check")


class CircuitValidatorTool(BaseTool):
    name: str = "validate_circuit"
    description: str = (
        "Transpile and validate a quantum circuit against a target backend. "
        "Returns depth, gate count, qubit count, and a feasibility flag. "
        "Use this to check if the circuit fits within hardware constraints before execution."
    )
    args_schema: Type[BaseModel] = CircuitValidatorInput

    def _run(self, circuit_qasm: str, backend_name: str = "aer_simulator") -> str:
        result = _call_mcp_tool("validate_circuit", {
            "circuit_qasm": circuit_qasm,
            "backend_name": backend_name,
        })
        return json.dumps(result)

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Convenience: get all tools as a list
# ---------------------------------------------------------------------------

def get_mcp_tools() -> list[BaseTool]:
    """Return all MCP-backed LangChain tools for use in the LangGraph agent."""
    return [
        QAOASamplerTool(),
        EstimatorTool(),
        BackendListTool(),
        CircuitValidatorTool(),
    ]
