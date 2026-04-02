"""
LangGraph StateGraph — Agentic Quantum Optimization Copilot
=============================================================
Defines the directed graph of agent nodes that orchestrates the full
quantum optimization pipeline:

  [intake] → [formulate] → [construct_circuit] → [execute] → [analyze] → END
                                                      ↑
                                               [handle_error] (on failure)

Nodes:
  intake           — Module 1: parse natural language input
  formulate        — Module 2: build QUBO / Ising Hamiltonian
  construct_circuit — Module 3: QAOA ansatz + transpilation check
  execute          — Module 4: hybrid COBYLA + MCP quantum execution
  analyze          — Module 5: decode bitstrings, compute approximation ratio
  handle_error     — Retry logic / IBM → Aer fallback / graceful error report

Conditional routing:
  After each node, if state['error'] is set the graph routes to handle_error.
  handle_error resets the error and increments retry_count; if retry_count
  exceeds the limit it routes to END, otherwise back to the failing node.
"""

from __future__ import annotations

import logging
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.states import AgentState
from modules.circuit_constructor import build_qaoa_circuit
from modules.execution_manager import run_hybrid_loop
from modules.intake_agent import parse_problem
from modules.qubo_formulator import build_qubo
from modules.results_analyzer import analyze

load_dotenv()
logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Error handler node
# ---------------------------------------------------------------------------

def handle_error(state: AgentState) -> AgentState:
    """
    Route to this node when any upstream node sets state['error'].
    Attempts:
      1. If backend is IBM Quantum and error is execution-related → switch to aer_simulator
      2. Increment retry counter
      3. If retries exhausted → leave error set and route to END
    """
    error: str = state.get("error", "Unknown error")
    retry_count: int = state.get("retry_count", 0)
    logs: list[str] = list(state.get("logs", []))
    backend_name: str = state.get("backend_name", "aer_simulator")

    logs.append(f"[handle_error] Error encountered (retry {retry_count + 1}/{_MAX_RETRIES}): {error}")

    if retry_count >= _MAX_RETRIES:
        logs.append("[handle_error] Max retries exceeded. Terminating with error.")
        return {**state, "logs": logs}  # Keep error set → routes to END

    # IBM → Aer fallback
    if backend_name != "aer_simulator" and "execution" in error.lower():
        logs.append("[handle_error] Switching backend to aer_simulator for retry.")
        backend_name = "aer_simulator"

    return {
        **state,
        "error": None,
        "retry_count": retry_count + 1,
        "backend_name": backend_name,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def _route_after_intake(state: AgentState) -> Literal["handle_error", "formulate", "__end__"]:
    if state.get("error"):
        return "handle_error"
    if state.get("parsed_problem", {}).get("clarification_needed"):
        # Surface the clarification question; stop the graph
        return "__end__"
    return "formulate"


def _route_after_formulate(state: AgentState) -> Literal["handle_error", "construct_circuit"]:
    return "handle_error" if state.get("error") else "construct_circuit"


def _route_after_construct(state: AgentState) -> Literal["handle_error", "execute"]:
    return "handle_error" if state.get("error") else "execute"


def _route_after_execute(state: AgentState) -> Literal["handle_error", "analyze"]:
    return "handle_error" if state.get("error") else "analyze"


def _route_after_analyze(state: AgentState) -> Literal["__end__"]:
    return "__end__"


def _route_after_error(state: AgentState) -> Literal["intake", "formulate", "construct_circuit", "execute", "__end__"]:
    """After handle_error, if error is cleared retry the last meaningful stage."""
    if state.get("error"):
        # Max retries hit; give up
        return "__end__"

    # Determine which stage to retry based on what's missing in state
    if state.get("bitstring_counts") is None and state.get("circuit_qasm"):
        return "execute"
    if state.get("circuit_qasm") is None and state.get("ising_hamiltonian") is not None:
        return "construct_circuit"
    if state.get("ising_hamiltonian") is None and state.get("parsed_problem"):
        return "formulate"
    return "intake"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intake", parse_problem)
    graph.add_node("formulate", build_qubo)
    graph.add_node("construct_circuit", build_qaoa_circuit)
    graph.add_node("execute", run_hybrid_loop)
    graph.add_node("analyze", analyze)
    graph.add_node("handle_error", handle_error)

    # Entry point
    graph.set_entry_point("intake")

    # Conditional edges after each processing node
    graph.add_conditional_edges("intake", _route_after_intake, {
        "handle_error": "handle_error",
        "formulate": "formulate",
        "__end__": END,
    })
    graph.add_conditional_edges("formulate", _route_after_formulate, {
        "handle_error": "handle_error",
        "construct_circuit": "construct_circuit",
    })
    graph.add_conditional_edges("construct_circuit", _route_after_construct, {
        "handle_error": "handle_error",
        "execute": "execute",
    })
    graph.add_conditional_edges("execute", _route_after_execute, {
        "handle_error": "handle_error",
        "analyze": "analyze",
    })
    graph.add_conditional_edges("analyze", _route_after_analyze, {
        "__end__": END,
    })

    # Error handler routing
    graph.add_conditional_edges("handle_error", _route_after_error, {
        "intake": "intake",
        "formulate": "formulate",
        "construct_circuit": "construct_circuit",
        "execute": "execute",
        "__end__": END,
    })

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: run from CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pprint import pprint

    user_input = (
        " ".join(sys.argv[1:])
        or "Select 2 assets from [A, B, C, D] to maximize return. "
           "Returns are A=0.9, B=0.6, C=0.8, D=0.4."
    )
    print(f"\nUser Input: {user_input}\n")

    compiled = build_graph()
    result = compiled.invoke({
        "user_input": user_input,
        "backend_name": "aer_simulator",
        "logs": [],
        "retry_count": 0,
    })

    print("\n=== Final Solution ===")
    sol = result.get("final_solution", {})
    for key, val in sol.items():
        if not key.startswith("_"):  # Skip matplotlib figures
            print(f"  {key}: {val}")

    print("\n=== Logs ===")
    for line in result.get("logs", []):
        print(" ", line)
