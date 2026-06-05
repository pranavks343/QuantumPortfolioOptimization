"""
FastAPI REST Interface — Quantum Alpha Copilot
=============================================================
A thin HTTP layer over the LangGraph pipeline. Lets any client run the
full natural-language → QUBO → QAOA → analysis pipeline over REST instead
of the Streamlit UI.

Endpoints:
  GET  /health    — liveness check
  GET  /backends  — list available quantum backends (Aer + IBM if token set)
  POST /optimize  — run the full optimization pipeline on a problem statement

Run it:
  uvicorn api:app --reload --port 8000
  # then POST to http://localhost:8000/optimize

The pipeline (agents.graph.build_graph) is identical to the one the
Streamlit app drives — this module only translates HTTP ↔ AgentState and
serializes the result to JSON (dropping the in-memory Qiskit Hamiltonian
object and the matplotlib figures, which aren't JSON-serializable).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.graph import build_graph
from tools.mcp_client import BackendListTool

load_dotenv()

app = FastAPI(
    title="Quantum Alpha Copilot API",
    description="REST interface to the natural-language → QUBO/Ising → QAOA pipeline.",
    version="1.0.0",
)

# Compile the LangGraph pipeline once at import time and reuse it across requests.
_PIPELINE = build_graph()


# ──────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    """Body for POST /optimize."""

    problem: str = Field(
        ...,
        min_length=1,
        description="Natural-language problem statement, e.g. "
                    "'Select 2 assets from AAPL, GOOGL, MSFT to maximize return.'",
        examples=["Select 2 assets from [A, B, C, D] to maximize return. "
                  "Returns are A=0.9, B=0.6, C=0.8, D=0.4."],
    )
    backend_name: str = Field(
        default="aer_simulator",
        description="Quantum backend: 'aer_simulator' (local) or an IBM backend "
                    "name like 'ibm_brisbane' (requires IBM_QUANTUM_TOKEN).",
    )


# ──────────────────────────────────────────────────────────────────────────────
# SERIALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Keys in AgentState that hold non-JSON-serializable objects (Qiskit operators).
_DROP_STATE_KEYS = {"ising_hamiltonian"}


def _sanitize(value: Any) -> Any:
    """
    Recursively convert a pipeline result into JSON-safe primitives.

    - Drops dict keys starting with "_" (matplotlib Figure handles).
    - Replaces non-finite floats (inf/nan) with None — JSON can't encode them,
      and brute-force optima can legitimately be ±inf for large problems.
    """
    if isinstance(value, dict):
        return {
            k: _sanitize(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Turn the raw AgentState into a JSON-safe response payload."""
    clean = {k: v for k, v in result.items() if k not in _DROP_STATE_KEYS}
    return _sanitize(clean)


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/backends")
def list_backends() -> dict[str, Any]:
    """List quantum backends the server can target (Aer always; IBM if token set)."""
    try:
        raw = BackendListTool()._run()  # returns a JSON string
        return json.loads(raw)
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=502, detail=f"Backend lookup failed: {exc}")


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    """
    Run the full quantum optimization pipeline on a natural-language problem.

    Mirrors the Streamlit flow: builds an AgentState, invokes the compiled
    LangGraph, and returns the JSON-safe result (parsed problem, QUBO,
    circuit metadata, convergence history, and the final solution).
    """
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY.",
        )

    try:
        result = _PIPELINE.invoke({
            "user_input": req.problem.strip(),
            "backend_name": req.backend_name,
            "logs": [],
            "retry_count": 0,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Pipeline-level error (e.g. retries exhausted) → 422 with the message + logs.
    if result.get("error"):
        raise HTTPException(
            status_code=422,
            detail={"error": result["error"], "logs": result.get("logs", [])},
        )

    # The LLM asked for clarification rather than producing a solution.
    parsed = result.get("parsed_problem", {})
    if parsed.get("clarification_needed"):
        return {
            "status": "clarification_needed",
            "clarification_question": parsed.get("clarification_question", ""),
            "logs": result.get("logs", []),
        }

    payload = _serialize_result(result)
    payload["status"] = "ok"
    return payload
