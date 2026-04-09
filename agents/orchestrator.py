"""
Orchestrator Agent (The Brain)
================================
An optional LLM-based orchestrator that can decompose complex user requests
into sub-tasks and summarise the final results in natural language.

This module is separate from the deterministic LangGraph pipeline (graph.py)
and is used for:
  1. High-level reasoning about the problem BEFORE sending it to the pipeline
  2. Generating a natural language summary of results AFTER the pipeline runs
  3. Suggesting better initial parameters for "closed-loop tuning" (Phase 4)

The orchestrator uses LangChain with the Claude model and has access to
all four MCP tools (via tools/mcp_client.py).
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = """You are the Quantum Alpha Copilot.
You have just executed a QAOA optimization run on a quantum computer (or simulator).
Given the optimization results, provide a clear, concise, human-readable summary that:
1. States what problem was solved
2. Reports the best solution found and its objective value
3. Mentions whether all constraints were satisfied
4. Reports the approximation ratio compared to the classical optimum
5. Gives a brief interpretation (e.g., "Asset A and C give the highest return of 1.7")

Keep the summary to 3-4 sentences. Use plain English suitable for a business user.
"""

_TUNING_SYSTEM_PROMPT = """You are an expert in variational quantum algorithms (VQA).
Given the convergence history of a previous QAOA run (energy values per iteration),
suggest better initial parameter values (gamma and beta) for the next run to improve convergence.

Return ONLY a JSON object: {"suggested_params": [gamma1, beta1, gamma2, beta2, ...]}
Base your suggestion on the shape of the convergence curve (e.g. if it plateaued, suggest
randomising more; if it converged smoothly, suggest a small perturbation of the optimal params).
"""


def _build_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0.3)
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-opus-4-6", api_key=anthropic_key, temperature=0.3)
    raise EnvironmentError("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.")


def generate_result_summary(result: dict) -> str:
    """
    Generate a natural language summary of the optimization results.

    Args:
        result: The final AgentState dict after graph execution.

    Returns:
        A human-readable summary string.
    """
    sol = result.get("final_solution", {})
    if not sol:
        return "No solution was produced."

    context = (
        f"Problem: {result.get('user_input', 'Unknown')}\n"
        f"Selected assets: {sol.get('selected_assets', [])}\n"
        f"Objective value: {sol.get('objective_value', 0):.4f}\n"
        f"Objective direction: {sol.get('objective_direction', 'maximize')}\n"
        f"Constraints satisfied: {sol.get('constraint_satisfied', False)}\n"
        f"Approximation ratio: {sol.get('approximation_ratio', 0):.4f}\n"
        f"Classical optimum: {sol.get('classical_optimum_selection', [])} "
        f"(value={sol.get('classical_optimum_value', 'N/A')})\n"
        f"Optimal energy: {sol.get('optimal_energy', 0):.6f}\n"
        f"Backend used: {result.get('backend_name', 'aer_simulator')}\n"
    )

    try:
        llm = _build_llm()
        response = llm.invoke([
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning(f"Could not generate LLM summary: {exc}")
        selected = sol.get("selected_assets", [])
        obj = sol.get("objective_value", 0)
        return (
            f"Optimization complete. Selected assets: {', '.join(selected)}. "
            f"Objective value: {obj:.4f}. "
            f"Approximation ratio: {sol.get('approximation_ratio', 0):.3f}."
        )


def suggest_better_params(
    convergence_history: list[float],
    current_optimal_params: list[float],
) -> list[float]:
    """
    Use the LLM to suggest improved initial parameters for the next run
    based on the convergence curve shape (closed-loop tuning).

    Args:
        convergence_history: Energy values from each COBYLA iteration.
        current_optimal_params: The gamma/beta values from the last run.

    Returns:
        A list of suggested initial parameter values.
    """
    import json
    import numpy as np

    context = (
        f"Previous optimal params: {current_optimal_params}\n"
        f"Convergence history (last 20 values): {convergence_history[-20:]}\n"
        f"Final energy: {convergence_history[-1] if convergence_history else 'N/A'}\n"
        f"Number of parameters: {len(current_optimal_params)}\n"
    )

    try:
        llm = _build_llm()
        response = llm.invoke([
            SystemMessage(content=_TUNING_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        suggested = data.get("suggested_params", current_optimal_params)
        # Ensure same length as current params
        if len(suggested) != len(current_optimal_params):
            suggested = current_optimal_params
        return [float(p) for p in suggested]
    except Exception as exc:
        logger.warning(f"Could not generate suggested params: {exc}. Using small perturbation.")
        rng = np.random.default_rng()
        return [p + rng.uniform(-0.1, 0.1) for p in current_optimal_params]
