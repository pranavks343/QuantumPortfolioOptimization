"""
Streamlit UI — Agentic Quantum Optimization Copilot
=====================================================
Interactive dashboard for:
  1. Inputting a natural language optimization problem
  2. Selecting the quantum backend
  3. Running the full LangGraph pipeline
  4. Visualising: parsed problem, QUBO matrix heatmap, QAOA circuit,
     convergence curve, bitstring histogram, and the final solution

Run with:
  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Quantum Optimization Copilot",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Agentic Quantum Optimization Copilot")

# ── Sidebar — Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    backend_name = st.selectbox(
        "Quantum Backend",
        options=["aer_simulator", "ibm_brisbane", "ibm_kyoto", "ibm_osaka"],
        index=0,
        help="Select 'aer_simulator' for local simulation or an IBM Quantum backend.",
    )


    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This copilot translates natural language optimization problems into "
        "quantum circuits and executes them using QAOA. The pipeline is "
        "orchestrated by a LangGraph agent that calls Qiskit tools via MCP."
    )

# ── Example problems ─────────────────────────────────────────────────────────
EXAMPLES = {
    "Portfolio Selection (5 assets)": (
        "I have 5 assets: AAPL, GOOGL, MSFT, AMZN, TSLA. "
        "Their expected returns are 0.9, 0.8, 0.75, 0.85, 0.7 respectively. "
        "Select exactly 3 assets to maximize total return."
    ),
    "Knapsack (4 items)": (
        "I have 4 items: Laptop, Phone, Tablet, Camera. "
        "Their values are 10, 6, 5, 4 and weights are 5, 3, 2, 1. "
        "Select items to maximize value with total weight at most 6. "
        "Treat weight constraint as hard."
    ),
    "Simple 3-asset selection": (
        "Choose 2 assets from [Alpha, Beta, Gamma] to maximize return. "
        "Alpha returns 0.8, Beta returns 0.6, Gamma returns 0.9."
    ),
}

# ── Problem input ─────────────────────────────────────────────────────────────
st.header("📝 Problem Statement")

col1, col2 = st.columns([3, 1])
with col2:
    example_choice = st.selectbox("Load an example", ["— Custom —"] + list(EXAMPLES.keys()))

with col1:
    default_text = EXAMPLES.get(example_choice, "") if example_choice != "— Custom —" else ""
    user_input = st.text_area(
        "Describe your optimization problem in plain English",
        value=default_text,
        height=150,
        placeholder="e.g. Select 2 assets from [A, B, C, D] to maximize return ...",
    )

run_button = st.button("🚀 Run Quantum Optimization", type="primary", use_container_width=True)

# ── Run pipeline ─────────────────────────────────────────────────────────────
if run_button:
    if not user_input.strip():
        st.error("Please enter a problem statement.")
    elif not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide an API Key (OpenAI or Anthropic) in the sidebar.")
    else:
        with st.spinner("Running the Agentic Quantum Optimization pipeline..."):
            try:
                from agents.graph import build_graph

                graph = build_graph()
                result = graph.invoke({
                    "user_input": user_input.strip(),
                    "backend_name": backend_name,
                    "logs": [],
                    "retry_count": 0,
                })
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        # ── Check for errors ──────────────────────────────────────────────
        if result.get("error"):
            st.error(f"⚠️ Pipeline stopped with error: {result['error']}")

        # ── Clarification needed ──────────────────────────────────────────
        parsed = result.get("parsed_problem", {})
        if parsed.get("clarification_needed"):
            st.warning(f"🤔 Clarification needed: {parsed.get('clarification_question')}")
            st.stop()

        st.success("✅ Optimization complete!")

        # ════════════════════════════════════════════════════════════════════
        # Section 1: Parsed Problem
        # ════════════════════════════════════════════════════════════════════
        st.header("1️⃣ Parsed Problem")
        if parsed:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Assets", len(parsed.get("assets", [])))
            col_b.metric("Objective", parsed.get("objective", "—").capitalize())
            col_c.metric("Select", parsed.get("num_select", "—"))

            with st.expander("View full parsed problem"):
                st.json({k: v for k, v in parsed.items() if k != "clarification_question"})

        # ════════════════════════════════════════════════════════════════════
        # Section 2: QUBO Matrix
        # ════════════════════════════════════════════════════════════════════
        st.header("2️⃣ QUBO Matrix")
        qubo_matrix = result.get("qubo_matrix")
        if qubo_matrix:
            Q = np.array(qubo_matrix)
            assets = parsed.get("assets", [str(i) for i in range(Q.shape[0])])
            fig_qubo, ax = plt.subplots(figsize=(min(len(assets), 8), min(len(assets), 8)))
            im = ax.imshow(Q, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(assets)))
            ax.set_yticks(range(len(assets)))
            ax.set_xticklabels(assets, rotation=45, ha="right")
            ax.set_yticklabels(assets)
            plt.colorbar(im, ax=ax, label="Q coefficient")
            ax.set_title("QUBO Matrix Heatmap", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig_qubo)

            col_qa, col_qb = st.columns(2)
            col_qa.metric("QUBO Variables", Q.shape[0])
            col_qb.metric("Constant Offset", f"{result.get('qubo_offset', 0.0):.4f}")

        # ════════════════════════════════════════════════════════════════════
        # Section 3: QAOA Circuit
        # ════════════════════════════════════════════════════════════════════
        st.header("3️⃣ QAOA Circuit")
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Qubits", result.get("num_qubits", "—"))
        col_c2.metric("Circuit Depth", result.get("circuit_depth", "—"))
        col_c3.metric("Parameters", result.get("num_parameters", "—"))

        feasible = result.get("transpilation_feasible")
        if feasible is True:
            st.success("Transpilation check passed ✔")
        elif feasible is False:
            st.error("Transpilation check failed — circuit exceeds hardware limits.")

        if result.get("circuit_qasm"):
            with st.expander("View OpenQASM source"):
                st.code(result["circuit_qasm"], language="qasm")

        # ════════════════════════════════════════════════════════════════════
        # Section 4: Hybrid Execution
        # ════════════════════════════════════════════════════════════════════
        st.header("4️⃣ Hybrid Execution")
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Optimal Energy ⟨H⟩", f"{result.get('optimal_energy', 0.0):.6f}")
        col_e2.metric("Backend", result.get("backend_name", "—"))

        conv = result.get("convergence_history", [])
        sol = result.get("final_solution", {})
        conv_fig = sol.get("_convergence_fig") if sol else None
        if conv_fig:
            st.pyplot(conv_fig)
        elif conv:
            fig_conv, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(conv) + 1), conv, color="#DD8452", linewidth=2)
            ax.set_xlabel("COBYLA Iteration")
            ax.set_ylabel("⟨H⟩ (Expectation Value)")
            ax.set_title("Optimisation Convergence")
            ax.grid(alpha=0.3)
            st.pyplot(fig_conv)

        # ════════════════════════════════════════════════════════════════════
        # Section 5: Results
        # ════════════════════════════════════════════════════════════════════
        st.header("5️⃣ Optimization Results")

        if sol:
            selected = sol.get("selected_assets", [])
            obj_val = sol.get("objective_value", 0)
            constraint_ok = sol.get("constraint_satisfied", False)
            approx_ratio = sol.get("approximation_ratio", 0)

            st.subheader("🏆 Best Solution")
            result_cols = st.columns(4)
            result_cols[0].metric("Selected Assets", ", ".join(selected) or "None")
            result_cols[1].metric("Objective Value", f"{obj_val:.4f}")
            result_cols[2].metric("Approximation Ratio", f"{approx_ratio:.3f}")
            result_cols[3].metric(
                "Constraints",
                "✅ Satisfied" if constraint_ok else "❌ Violated",
            )

            classical = sol.get("classical_optimum_selection", [])
            classical_val = sol.get("classical_optimum_value")
            if classical:
                val_str = f"{classical_val:.4f}" if classical_val is not None else "N/A"
                st.info(
                    f"Classical brute-force optimum: **{', '.join(classical)}** "
                    f"(value = {val_str})"
                )

            # Bitstring histogram
            hist_fig = sol.get("_histogram_fig")
            if hist_fig:
                st.subheader("📊 Measurement Distribution")
                st.pyplot(hist_fig)

            # Top candidates table
            candidates = sol.get("top_candidates", [])
            if candidates:
                st.subheader("Top Measurement Outcomes")
                table_data = []
                for c in candidates:
                    table_data.append({
                        "Bitstring": c["bitstring"],
                        "Selected Assets": ", ".join(c["selection"]),
                        "Probability": f"{c['probability']:.4f}",
                        "Objective": f"{c['objective_value']:.4f}",
                        "Constraints": "✅" if c["constraint_satisfied"] else "❌",
                    })
                st.table(table_data)

        # ════════════════════════════════════════════════════════════════════
        # Section 6: Experiment Log
        # ════════════════════════════════════════════════════════════════════
        st.header("📋 Experiment Log")
        logs = result.get("logs", [])
        log_text = "\n".join(logs)
        with st.expander(f"Show logs ({len(logs)} entries)"):
            st.text(log_text)

        log_json = json.dumps(
            {k: v for k, v in result.items() if k not in ("ising_hamiltonian", "final_solution")},
            default=str,
            indent=2,
        )
        st.download_button(
            label="⬇️ Download Experiment Log (JSON)",
            data=log_json,
            file_name="quantum_experiment_log.json",
            mime="application/json",
        )
