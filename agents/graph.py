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

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This file is the "traffic controller" for the whole pipeline.
# It decides:
#   1. Which step runs next (the happy path)
#   2. What to do when a step fails (error handling + retry logic)
#
# It uses LangGraph's StateGraph — think of it like a flowchart where:
#   - Nodes  = the actual work (each module)
#   - Edges  = the arrows between nodes
#   - Conditional edges = "if error → go here, else → go there"
# ──────────────────────────────────────────────────────────────────────────────

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

load_dotenv()  # Load API keys from the .env file
logger = logging.getLogger(__name__)

# Maximum number of times we'll retry a failed step before giving up
_MAX_RETRIES = 2


# ──────────────────────────────────────────────────────────────────────────────
# ERROR HANDLER NODE
# ──────────────────────────────────────────────────────────────────────────────
# This node runs whenever any other node sets state["error"].
# It tries to recover by:
#   1. Switching from IBM Quantum → local simulator (if hardware-related failure)
#   2. Incrementing the retry counter
#   3. Giving up if we've already retried _MAX_RETRIES times
# ──────────────────────────────────────────────────────────────────────────────

def handle_error(state: AgentState) -> AgentState:
    """
    Route to this node when any upstream node sets state['error'].
    Attempts:
      1. Preserve the requested backend
      2. Increment retry counter
      3. If retries exhausted → leave error set and route to END
    """
    # Read the error message set by the failing node
    error: str = state.get("error", "Unknown error")

    # How many times have we already retried?
    retry_count: int = state.get("retry_count", 0)

    # Make a mutable copy of the logs list
    logs: list[str] = list(state.get("logs", []))

    # Which backend are we currently trying to use?
    backend_name: str = state.get("backend_name", "aer_simulator")

    logs.append(f"[handle_error] Error encountered (retry {retry_count + 1}/{_MAX_RETRIES}): {error}")

    # If we've already used all our retries, give up permanently
    # Leaving "error" set causes the router below to send us to END
    if retry_count >= _MAX_RETRIES:
        logs.append("[handle_error] Max retries exceeded. Terminating with error.")
        return {**state, "logs": logs}  # error stays set → routes to END

    logs.append(f"[handle_error] Retrying on requested backend '{backend_name}'.")

    # Clear the error and increment retry_count so routing continues normally
    return {
        **state,
        "error": None,                   # Clear the error
        "retry_count": retry_count + 1,  # Count this retry
        "backend_name": backend_name,    # Possibly switched to local simulator
        "logs": logs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS (Conditional Edges)
# ──────────────────────────────────────────────────────────────────────────────
# After each node runs, LangGraph calls one of these functions to decide
# which node to run next.
#
# Pattern: if there's an error → go to handle_error
#          otherwise           → go to the next step
# ──────────────────────────────────────────────────────────────────────────────

def _route_after_intake(state: AgentState) -> Literal["handle_error", "formulate", "__end__"]:
    """After Step 1 (intake), decide where to go next."""
    if state.get("error"):
        return "handle_error"  # Something went wrong

    # Special case: LLM needs more info from the user → stop and show the question
    if state.get("parsed_problem", {}).get("clarification_needed"):
        return "__end__"  # Stop here; UI will display the clarification question

    return "formulate"  # All good → go to Step 2


def _route_after_formulate(state: AgentState) -> Literal["handle_error", "construct_circuit"]:
    """After Step 2 (QUBO formulation), decide where to go next."""
    return "handle_error" if state.get("error") else "construct_circuit"


def _route_after_construct(state: AgentState) -> Literal["handle_error", "execute"]:
    """After Step 3 (circuit construction), decide where to go next."""
    return "handle_error" if state.get("error") else "execute"


def _route_after_execute(state: AgentState) -> Literal["handle_error", "analyze"]:
    """After Step 4 (quantum execution), decide where to go next."""
    return "handle_error" if state.get("error") else "analyze"


def _route_after_analyze(state: AgentState) -> Literal["__end__"]:
    """After Step 5 (analysis), we're always done — no errors possible here."""
    return "__end__"


def _route_after_error(state: AgentState) -> Literal["intake", "formulate", "construct_circuit", "execute", "__end__"]:
    """
    After handle_error clears the error, figure out which step to retry.

    Instead of always restarting from Step 1, we look at what data is
    missing in state to figure out exactly which step failed, and retry
    only from that point forward.
    """
    # If error is still set, max retries were hit → give up
    if state.get("error"):
        return "__end__"

    # No measurements yet but circuit exists → execute failed → retry execute
    if state.get("bitstring_counts") is None and state.get("circuit_qasm"):
        return "execute"

    # No circuit but we have a Hamiltonian → circuit construction failed → retry there
    if state.get("circuit_qasm") is None and state.get("ising_hamiltonian") is not None:
        return "construct_circuit"

    # No Hamiltonian but parsed problem exists → QUBO formulation failed → retry there
    if state.get("ising_hamiltonian") is None and state.get("parsed_problem"):
        return "formulate"

    # Otherwise, restart from the very beginning
    return "intake"


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ──────────────────────────────────────────────────────────────────────────────
# This function assembles all nodes and edges into a compiled graph
# that LangGraph can execute.
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""

    # Create an empty graph using AgentState as its shared data structure
    graph = StateGraph(AgentState)

    # ── Add all the worker nodes ──────────────────────────────────────────────
    # Each entry is: graph.add_node("label", function_to_call)
    # The function receives the full AgentState dict and returns an updated one
    graph.add_node("intake", parse_problem)                  # Step 1
    graph.add_node("formulate", build_qubo)                  # Step 2
    graph.add_node("construct_circuit", build_qaoa_circuit)  # Step 3
    graph.add_node("execute", run_hybrid_loop)               # Step 4
    graph.add_node("analyze", analyze)                       # Step 5
    graph.add_node("handle_error", handle_error)             # Error recovery

    # ── Set the entry point ───────────────────────────────────────────────────
    graph.set_entry_point("intake")  # Always start from Step 1

    # ── Wire up conditional edges ─────────────────────────────────────────────
    # Each call defines:
    #   - which node just finished
    #   - a routing function that returns a string key
    #   - a dict mapping string keys → actual node names

    graph.add_conditional_edges("intake", _route_after_intake, {
        "handle_error": "handle_error",
        "formulate": "formulate",
        "__end__": END,          # END is LangGraph's built-in terminal node
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

    # After the error handler runs, decide where to retry
    graph.add_conditional_edges("handle_error", _route_after_error, {
        "intake": "intake",
        "formulate": "formulate",
        "construct_circuit": "construct_circuit",
        "execute": "execute",
        "__end__": END,
    })

    # Compile and return the executable graph
    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND LINE USAGE
# ──────────────────────────────────────────────────────────────────────────────
# Run this file directly to test the pipeline without the Streamlit UI:
#   python agents/graph.py "Select 2 assets from A,B,C. Returns A=0.9, B=0.6, C=0.8"
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pprint import pprint

    # Use command-line argument if provided, otherwise use a default example
    user_input = (
        " ".join(sys.argv[1:])
        or "Select 2 assets from [A, B, C, D] to maximize return. "
           "Returns are A=0.9, B=0.6, C=0.8, D=0.4."
    )
    print(f"\nUser Input: {user_input}\n")

    # Build the graph and run the full pipeline
    compiled = build_graph()
    result = compiled.invoke({
        "user_input": user_input,
        "backend_name": "aer_simulator",  # Use local simulator for CLI testing
        "logs": [],
        "retry_count": 0,
    })

    # Print the final solution (skip matplotlib figures — not printable as text)
    print("\n=== Final Solution ===")
    sol = result.get("final_solution", {})
    for key, val in sol.items():
        if not key.startswith("_"):  # Keys starting with "_" are matplotlib figures
            print(f"  {key}: {val}")

    # Print every log line from every step
    print("\n=== Logs ===")
    for line in result.get("logs", []):
        print(" ", line)
