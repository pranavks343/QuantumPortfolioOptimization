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

# ──────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is Step 1 of the pipeline. It takes whatever the user typed in plain
# English and converts it into a clean, structured Python dictionary.
#
# HOW? It sends the user's text to an LLM (Claude or GPT-4) along with very
# specific instructions that say: "Return ONLY a JSON object with these exact
# fields." The LLM acts as a smart parser.
#
# INPUT:  "I have 5 assets: AAPL, GOOGL, MSFT, AMZN, TSLA.
#          Returns: 0.9, 0.8, 0.75, 0.85, 0.7. Select exactly 3."
#
# OUTPUT: {
#   "assets": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
#   "objective": "maximize",
#   "objective_weights": {"AAPL": 0.9, "GOOGL": 0.8, ...},
#   "num_select": 3,
#   "hard_constraints": [{"type": "cardinality", "value": 3}],
#   "clarification_needed": false
# }
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from agents.states import AgentState, ParsedProblem

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────
# This is the instruction we send to the LLM BEFORE the user's message.
# It tells the LLM exactly what format to respond in.
#
# Think of it like giving a translator a template:
#   "Translate the following text, and put your answer in THIS exact format."
#
# We use temperature=0 (set below) so the LLM always gives a deterministic,
# structured response instead of a creative one.
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# LLM FACTORY
# ──────────────────────────────────────────────────────────────────────────────
# Builds whichever LLM client is available based on environment variables.
# Priority: OpenAI first, then Anthropic (Claude).
# ──────────────────────────────────────────────────────────────────────────────
def _build_llm():
    """Return a LangChain LLM client based on which API key is available."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        from langchain_openai import ChatOpenAI
        # temperature=0 → deterministic output, no randomness (we need exact JSON)
        return ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0)

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-opus-4-6", api_key=anthropic_key, temperature=0)

    raise EnvironmentError("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
# This is the function that graph.py registers as the "intake" node.
# LangGraph calls it with the full AgentState dict and expects an updated
# AgentState dict back.
# ──────────────────────────────────────────────────────────────────────────────
def parse_problem(state: AgentState) -> AgentState:
    """
    LangGraph node: Module 1 — Problem Intake & Reasoning Agent.

    Reads:   state['user_input']
    Writes:  state['parsed_problem'], state['logs'], state['error']
    """
    # Pull the user's text from the shared state dictionary
    user_input: str = state.get("user_input", "").strip()

    # Make a mutable copy of the logs list (we'll append to it)
    logs: list[str] = list(state.get("logs", []))

    # Guard: if the user didn't type anything, set an error and return early
    if not user_input:
        return {**state, "error": "No user input provided.", "logs": logs}

    # Log that we received the input (truncated to 120 chars for readability)
    logs.append(f"[intake_agent] Received input: {user_input[:120]}...")

    try:
        # Build the LLM client (Claude or GPT-4, depending on .env)
        llm = _build_llm()

        # Construct the message list:
        #   1. SystemMessage = the "rules" we give the LLM
        #   2. HumanMessage  = the user's actual problem text
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        # Call the LLM and get its response
        response = llm.invoke(messages)
        raw = response.content.strip()

        # ── Defensive parsing: strip code fences if LLM added them ──────────
        # Despite instructions, some LLM responses look like:
        #   ```json
        #   { "assets": [...] }
        #   ```
        # We strip those fences before parsing.
        if raw.startswith("```"):
            raw = raw.split("```")[1]      # Get the content between the fences
            if raw.startswith("json"):
                raw = raw[4:]              # Remove the "json" language tag

        # Parse the JSON string into a Python dictionary
        parsed: ParsedProblem = json.loads(raw)

        # Log different messages depending on whether clarification is needed
        if parsed.get("clarification_needed"):
            # The LLM couldn't fully understand the problem → pipeline will stop here
            # and the UI will show the clarification question
            logs.append(f"[intake_agent] Clarification needed: {parsed.get('clarification_question')}")
        else:
            logs.append(
                f"[intake_agent] Parsed: {len(parsed.get('assets', []))} assets, "
                f"objective={parsed.get('objective')}, "
                f"num_select={parsed.get('num_select')}"
            )

        # Return updated state: parsed_problem filled in, error cleared
        return {**state, "parsed_problem": parsed, "logs": logs, "error": None}

    except json.JSONDecodeError as exc:
        # The LLM returned something that wasn't valid JSON
        msg = f"[intake_agent] Failed to parse LLM response as JSON: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}

    except Exception as exc:
        # Catch-all for any other unexpected errors (network issues, API errors, etc.)
        msg = f"[intake_agent] Unexpected error: {exc}"
        logger.error(msg)
        return {**state, "error": msg, "logs": logs}
