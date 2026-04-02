"""
Module 1: Problem Intake & Reasoning Agent
==========================================
Parses a natural language optimization problem into a structured representation
that downstream modules (QUBO formulator, circuit constructor, etc.) can consume.

Key responsibilities:
  - Extract decision variables (binary asset names)
  - Identify the objective (maximize return / minimize risk)
  - Separate hard constraints (must satisfy) from soft constraints (penalised)
  - Request clarification from the user if the problem is under-defined
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from agents.states import AgentState, ParsedProblem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt that instructs the LLM on how to parse the problem
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert in combinatorial optimization and quantum computing.
Your task is to parse a natural language optimization problem into a strict JSON structure.

Return ONLY a JSON object with these fields:
{
  "assets": ["A", "B", "C", ...],          // Names of the decision variables (binary x_i)
  "objective": "maximize" or "minimize",
  "objective_weights": {"A": 0.8, "B": 0.5, ...},  // Return/cost per asset (default 1.0 if not specified)
  "num_select": 2,                          // How many to select (k); use 0 if not specified
  "hard_constraints": [
    {"type": "cardinality", "value": 2}     // Example: must select exactly 2 assets
  ],
  "soft_constraints": [
    {"type": "budget", "limit": 100, "weights": {"A": 30, "B": 50, ...}}
  ],
  "clarification_needed": false,
  "clarification_question": ""             // Non-empty only if clarification_needed is true
}

Rules:
1. "hard_constraints" are must-satisfy (enforced as QUBO penalties with high penalty P).
2. "soft_constraints" are desirable but can be penalised (lower penalty).
3. If the user specifies "select k assets", set num_select = k and add a cardinality hard constraint.
4. If critical information is missing (e.g. no assets listed, no objective), set clarification_needed=true
   and write a specific clarification_question.
5. Do NOT include markdown, code fences, or any text outside the JSON object.
"""


def _build_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0)
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-opus-4-6", api_key=anthropic_key, temperature=0)
    raise EnvironmentError("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.")


def parse_problem(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 1 — Problem Intake & Reasoning Agent.

    Reads:   state['user_input']
    Writes:  state['parsed_problem'], state['logs'], state['error']
    """
    user_input: str = state.get("user_input", "").strip()
    logs: list[str] = list(state.get("logs", []))

    if not user_input:
        return {**state, "error": "No user input provided.", "logs": logs}

    logs.append(f"[intake_agent] Received input: {user_input[:120]}...")

    try:
        llm = _build_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip code fences if the LLM wraps in them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed: ParsedProblem = json.loads(raw)

        if parsed.get("clarification_needed"):
            logs.append(f"[intake_agent] Clarification needed: {parsed.get('clarification_question')}")
        else:
            logs.append(
                f"[intake_agent] Parsed: {len(parsed.get('assets', []))} assets, "
                f"objective={parsed.get('objective')}, "
                f"num_select={parsed.get('num_select')}"
            )

        return {**state, "parsed_problem": parsed, "logs": logs, "error": None}

    except json.JSONDecodeError as exc:
        msg = f"[intake_agent] Failed to parse LLM response as JSON: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
    except Exception as exc:
        msg = f"[intake_agent] Unexpected error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
