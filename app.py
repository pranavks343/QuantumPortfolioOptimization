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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

matplotlib.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.6,
})

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Portfolio Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #020817;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* ── Main content area ── */
.block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* ── Page title ── */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #475569;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1e293b;
}
.section-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    flex-shrink: 0;
}

/* ── Cards ── */
.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #818cf8;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    gap: 1rem;
}
.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-value.accent { color: #818cf8; }
.metric-value.green  { color: #34d399; }
.metric-value.amber  { color: #fbbf24; }
.metric-value.red    { color: #f87171; }

/* ── Result hero card ── */
.result-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1a1040 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.result-hero-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.5rem;
}
.result-hero-assets {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.result-hero-value {
    font-size: 1rem;
    color: #64748b;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-green  { background: #052e16; color: #34d399; border: 1px solid #166534; }
.badge-red    { background: #450a0a; color: #f87171; border: 1px solid #991b1b; }
.badge-blue   { background: #0c1a4a; color: #818cf8; border: 1px solid #3730a3; }
.badge-amber  { background: #3d1f00; color: #fbbf24; border: 1px solid #92400e; }

/* ── Table ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.styled-table th {
    background: #1e293b;
    color: #64748b;
    font-weight: 600;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid #334155;
}
.styled-table td {
    padding: 0.65rem 1rem;
    color: #cbd5e1;
    border-bottom: 1px solid #1e293b;
}
.styled-table tr:hover td { background: #0f172a; }
.styled-table tr:last-child td { border-bottom: none; }

/* ── Input styling ── */
.stTextArea textarea {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.1) !important;
}
.stTextArea label { color: #64748b !important; font-size: 0.75rem !important; }

.stSelectbox > div > div {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: #0a0f1e !important;
    border: 1px solid #1e293b !important;
    border-top: none !important;
}

/* ── Code block ── */
.stCode, code {
    background: #0a0f1e !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Alerts ── */
.stSuccess { background: #052e16 !important; border: 1px solid #166534 !important; border-radius: 10px !important; }
.stError   { background: #450a0a !important; border: 1px solid #991b1b !important; border-radius: 10px !important; }
.stWarning { background: #3d1f00 !important; border: 1px solid #92400e !important; border-radius: 10px !important; }
.stInfo    { background: #0c1a4a !important; border: 1px solid #3730a3 !important; border-radius: 10px !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #0f172a !important;
    color: #818cf8 !important;
    border: 1px solid #312e81 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #1a1040 !important;
    border-color: #818cf8 !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #020817; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render a section header ──────────────────────────────────────────
def section(title: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot"></div>
        {title}
    </div>""", unsafe_allow_html=True)


# ── Helper: render metric cards in a row ─────────────────────────────────────
def metric_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        color_class = m.get("color", "")
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{m['label']}</div>
            <div class="metric-value {color_class}">{m['value']}</div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em;">
            ⚛ Quantum Optimizer
        </div>
        <div style="font-size:0.72rem; color:#475569; margin-top:0.2rem;">
            Powered by QAOA + LangGraph
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:0.5rem;">Configuration</div>', unsafe_allow_html=True)

    backend_name = st.selectbox(
        "Quantum Backend",
        options=["aer_simulator", "ibm_brisbane", "ibm_kyoto", "ibm_osaka"],
        index=0,
        help="aer_simulator runs locally. IBM backends require a token in .env",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:0.75rem;">Pipeline Steps</div>', unsafe_allow_html=True)

    steps = [
        ("01", "NL → Structured Problem", "#818cf8"),
        ("02", "QUBO Formulation",        "#38bdf8"),
        ("03", "QAOA Circuit Build",       "#34d399"),
        ("04", "Hybrid Execution",         "#fbbf24"),
        ("05", "Results Analysis",         "#f472b6"),
    ]
    for num, label, color in steps:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
            <div style="width:20px; height:20px; border-radius:5px; background:{color}22;
                        border:1px solid {color}44; display:flex; align-items:center;
                        justify-content:center; font-size:0.55rem; font-weight:700; color:{color}; flex-shrink:0;">
                {num}
            </div>
            <div style="font-size:0.75rem; color:#64748b;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a0f1e; border:1px solid #1e293b; border-radius:8px; padding:0.85rem; font-size:0.72rem; color:#475569; line-height:1.6;">
        Translates natural language into quantum circuits and solves them via QAOA.
        Classical COBYLA tunes the variational parameters.
    </div>""", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Agentic Quantum Optimization Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Describe a combinatorial optimization problem in plain English — the agent formulates it as QUBO, builds a QAOA circuit, and finds the optimal solution.</div>', unsafe_allow_html=True)

# ── Example problems ──────────────────────────────────────────────────────────
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
section("Problem Statement")

col_input, col_example = st.columns([3, 1])
with col_example:
    example_choice = st.selectbox("Load example", ["— Custom —"] + list(EXAMPLES.keys()), label_visibility="collapsed")
with col_input:
    default_text = EXAMPLES.get(example_choice, "") if example_choice != "— Custom —" else ""
    user_input = st.text_area(
        "problem",
        value=default_text,
        height=130,
        placeholder="e.g.  Select 3 from AAPL, GOOGL, MSFT, AMZN, TSLA to maximize return. Returns: 0.9, 0.8, 0.75, 0.85, 0.7",
        label_visibility="collapsed",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
run_button = st.button("Run Quantum Optimization", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_button:
    if not user_input.strip():
        st.error("Please enter a problem statement.")
        st.stop()

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        st.error("No API key found. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file.")
        st.stop()

    with st.spinner("Running the quantum optimization pipeline..."):
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

    if result.get("error"):
        st.error(f"Pipeline stopped: {result['error']}")
        st.stop()

    parsed = result.get("parsed_problem", {})
    if parsed.get("clarification_needed"):
        st.warning(f"Clarification needed: {parsed.get('clarification_question')}")
        st.stop()

    st.success("Optimization complete")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Parsed Problem
    # ════════════════════════════════════════════════════════════════════════════
    section("01  ·  Parsed Problem")
    if parsed:
        assets_list = parsed.get("assets", [])
        metric_row([
            {"label": "Total Assets",  "value": len(assets_list),                          "color": "accent"},
            {"label": "Objective",     "value": parsed.get("objective", "—").capitalize(),  "color": ""},
            {"label": "Select k",      "value": parsed.get("num_select", "—"),              "color": ""},
            {"label": "Constraints",   "value": len(parsed.get("hard_constraints", [])),    "color": ""},
        ])

        if assets_list:
            weights = parsed.get("objective_weights", {})
            sorted_assets = sorted(assets_list, key=lambda a: weights.get(a, 0), reverse=True)
            pills_html = " ".join(
                f'<span style="background:#0c1a4a; border:1px solid #3730a3; color:#818cf8; '
                f'border-radius:6px; padding:0.2rem 0.6rem; font-size:0.75rem; font-weight:500;">'
                f'{a} <span style="color:#475569;">{weights.get(a, "")}</span></span>'
                for a in sorted_assets
            )
            st.markdown(f'<div style="margin-top:0.75rem; display:flex; flex-wrap:wrap; gap:0.4rem;">{pills_html}</div>', unsafe_allow_html=True)

        with st.expander("View structured JSON"):
            st.json({k: v for k, v in parsed.items() if k != "clarification_question"})

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — QUBO Matrix
    # ════════════════════════════════════════════════════════════════════════════
    section("02  ·  QUBO Matrix")
    qubo_matrix = result.get("qubo_matrix")
    if qubo_matrix:
        Q = np.array(qubo_matrix)
        n = Q.shape[0]
        display_assets = parsed.get("assets", [str(i) for i in range(n)])

        col_heatmap, col_qubo_stats = st.columns([3, 1])
        with col_heatmap:
            sz = min(max(n, 4), 9)
            fig_qubo, ax = plt.subplots(figsize=(sz, sz))
            im = ax.imshow(Q, cmap="RdYlBu_r", aspect="auto")
            ax.set_xticks(range(len(display_assets)))
            ax.set_yticks(range(len(display_assets)))
            ax.set_xticklabels(display_assets, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(display_assets, fontsize=9)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8, colors="#64748b")
            cb.outline.set_edgecolor("#334155")
            ax.set_title("QUBO Coefficient Matrix", fontsize=11, pad=12, color="#94a3b8")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_qubo, use_container_width=True)

        with col_qubo_stats:
            st.markdown("<br>", unsafe_allow_html=True)
            for lbl, val in [
                ("Variables", Q.shape[0]),
                ("Non-zero", int(np.count_nonzero(Q))),
                ("Offset", f"{result.get('qubo_offset', 0.0):.3f}"),
                ("Qubits", result.get("num_qubits", "—")),
            ]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.6rem;">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="font-size:1.2rem;">{val}</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — QAOA Circuit
    # ════════════════════════════════════════════════════════════════════════════
    section("03  ·  QAOA Circuit")
    feasible = result.get("transpilation_feasible")
    metric_row([
        {"label": "Qubits",         "value": result.get("num_qubits", "—"),   "color": "accent"},
        {"label": "Circuit Depth",  "value": result.get("circuit_depth", "—"), "color": "amber" if (result.get("circuit_depth") or 0) > 100 else "green"},
        {"label": "Parameters",     "value": result.get("num_parameters", "—"), "color": ""},
        {"label": "Transpilation",  "value": "Pass" if feasible else "Fail",   "color": "green" if feasible else "red"},
    ])

    if result.get("circuit_qasm"):
        with st.expander("View OpenQASM 3 source"):
            st.code(result["circuit_qasm"], language="text")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Hybrid Execution
    # ════════════════════════════════════════════════════════════════════════════
    section("04  ·  Hybrid Execution  ·  COBYLA + QAOA")
    metric_row([
        {"label": "Optimal Energy ⟨H⟩", "value": f"{result.get('optimal_energy', 0.0):.5f}", "color": "accent"},
        {"label": "Backend",             "value": result.get("backend_name", "—"),             "color": ""},
        {"label": "Iterations",          "value": len(result.get("convergence_history", [])),  "color": ""},
    ])

    conv = result.get("convergence_history", [])
    sol = result.get("final_solution", {})
    conv_fig = sol.get("_convergence_fig") if sol else None

    if conv_fig:
        # Re-style the convergence figure to match the dark theme
        conv_fig.set_facecolor("#0f172a")
        for ax in conv_fig.axes:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#64748b")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            ax.title.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            for line in ax.get_lines():
                line.set_color("#818cf8")
                line.set_linewidth(2)
        st.pyplot(conv_fig, use_container_width=True)
    elif conv:
        fig_conv, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(range(1, len(conv) + 1), conv, color="#818cf8", linewidth=2, alpha=0.9)
        ax.fill_between(range(1, len(conv) + 1), conv, alpha=0.08, color="#818cf8")
        ax.set_xlabel("COBYLA Iteration")
        ax.set_ylabel("⟨H⟩  Energy")
        ax.set_title("Optimization Convergence", pad=12)
        ax.grid(True, alpha=0.15)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        plt.tight_layout()
        st.pyplot(fig_conv, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Results
    # ════════════════════════════════════════════════════════════════════════════
    section("05  ·  Optimization Results")

    if sol:
        selected      = sol.get("selected_assets", [])
        obj_val       = sol.get("objective_value", 0)
        constraint_ok = sol.get("constraint_satisfied", False)
        approx_ratio  = sol.get("approximation_ratio", 0)
        classical     = sol.get("classical_optimum_selection", [])
        classical_val = sol.get("classical_optimum_value")

        # Result hero card
        constraint_badge = '<span class="badge badge-green">Constraints satisfied</span>' if constraint_ok else '<span class="badge badge-red">Constraints violated</span>'
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-hero-label">Optimal Portfolio</div>
            <div class="result-hero-assets">{" · ".join(selected) if selected else "No solution"}</div>
            <div class="result-hero-value" style="margin-bottom:0.75rem;">Objective value: <strong style="color:#f1f5f9;">{obj_val:.4f}</strong></div>
            {constraint_badge}
        </div>""", unsafe_allow_html=True)

        # Metrics row
        ratio_color = "green" if approx_ratio >= 0.95 else "amber" if approx_ratio >= 0.8 else "red"
        metric_row([
            {"label": "Objective Value",     "value": f"{obj_val:.4f}",            "color": "accent"},
            {"label": "Approximation Ratio", "value": f"{approx_ratio:.3f}",       "color": ratio_color},
            {"label": "Total Shots",         "value": sol.get("total_shots", 1024), "color": ""},
            {"label": "Classical Optimum",   "value": f"{classical_val:.4f}" if classical_val else "—", "color": ""},
        ])

        if classical:
            st.markdown(f"""
            <div style="background:#0c1a4a; border:1px solid #3730a3; border-radius:8px;
                        padding:0.75rem 1rem; margin-top:0.75rem; font-size:0.8rem; color:#94a3b8;">
                Classical brute-force optimum:
                <strong style="color:#818cf8;">{", ".join(classical)}</strong>
                (value = {f"{classical_val:.4f}" if classical_val else "N/A"})
            </div>""", unsafe_allow_html=True)

        # Histogram
        hist_fig = sol.get("_histogram_fig")
        if hist_fig:
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            hist_fig.set_facecolor("#0f172a")
            for ax in hist_fig.axes:
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#64748b")
                ax.xaxis.label.set_color("#94a3b8")
                ax.yaxis.label.set_color("#94a3b8")
                ax.title.set_color("#e2e8f0")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")
                for bar in ax.patches:
                    bar.set_facecolor("#6366f1")
                    bar.set_edgecolor("#4f46e5")
                    bar.set_alpha(0.85)
            st.pyplot(hist_fig, use_container_width=True)

        # Top candidates table
        candidates = sol.get("top_candidates", [])
        if candidates:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569; margin-bottom:0.5rem;">Top Measurement Outcomes</div>', unsafe_allow_html=True)
            rows = ""
            for c in candidates:
                sat = '<span class="badge badge-green">✓</span>' if c["constraint_satisfied"] else '<span class="badge badge-red">✗</span>'
                rows += f"""<tr>
                    <td><code style="background:#0a0f1e; padding:0.15rem 0.4rem; border-radius:4px; font-size:0.75rem; color:#818cf8;">{c['bitstring']}</code></td>
                    <td>{", ".join(c['selection']) or "—"}</td>
                    <td>{c['probability']:.4f}</td>
                    <td>{c['objective_value']:.4f}</td>
                    <td>{sat}</td>
                </tr>"""
            st.markdown(f"""
            <div class="card" style="padding:0; overflow:hidden;">
            <table class="styled-table">
                <thead><tr>
                    <th>Bitstring</th><th>Selected Assets</th>
                    <th>Probability</th><th>Objective</th><th>Constraints</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Experiment Log
    # ════════════════════════════════════════════════════════════════════════════
    section("06  ·  Experiment Log")
    logs = result.get("logs", [])
    with st.expander(f"{len(logs)} log entries"):
        log_html = "".join(
            f'<div style="font-size:0.73rem; color:#{"34d399" if "[intake" in l else "38bdf8" if "[qubo" in l else "818cf8" if "[circuit" in l else "fbbf24" if "[execution" in l else "f472b6" if "[results" in l else "64748b"}; '
            f'padding:0.15rem 0; font-family:\'JetBrains Mono\', monospace; border-bottom:1px solid #0a0f1e;">{l}</div>'
            for l in logs
        )
        st.markdown(f'<div style="background:#0a0f1e; border-radius:8px; padding:0.75rem; max-height:300px; overflow-y:auto;">{log_html}</div>', unsafe_allow_html=True)

    log_json = json.dumps(
        {k: v for k, v in result.items() if k not in ("ising_hamiltonian", "final_solution")},
        default=str,
        indent=2,
    )
    st.download_button(
        label="Download Experiment Log  (JSON)",
        data=log_json,
        file_name="quantum_experiment_log.json",
        mime="application/json",
        use_container_width=True,
    )
