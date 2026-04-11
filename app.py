"""
Streamlit UI — Quantum Alpha Copilot
====================================
Run with:  streamlit run app.py
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

# ── Matplotlib global style (matches light purple theme) ─────────────────────
matplotlib.rcParams.update({
    "figure.facecolor": "#f8f7ff",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#ddd6fe",
    "axes.labelcolor":  "#5b5470",
    "axes.titlecolor":  "#1e1535",
    "xtick.color":      "#7a7391",
    "ytick.color":      "#7a7391",
    "text.color":       "#1e1535",
    "grid.color":       "#ede9fe",
    "grid.linewidth":   0.5,
    "font.family":      "sans-serif",
})

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Alpha Copilot",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }

/* Background */
[data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(ellipse at 10% 0%, rgba(109,40,217,0.08) 0%, transparent 45%),
                linear-gradient(180deg, #f8f7ff 0%, #f3f0ff 100%);
}
[data-testid="stAppViewContainer"] { background: #f8f7ff; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ddd6fe !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Content */
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

/* Page title */
.page-title {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #6d28d9;
    line-height: 1.1;
    margin-top: 1.75rem;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #7a7391;
    margin-bottom: 2rem;
    max-width: 700px;
    line-height: 1.6;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #9d8fc0;
    margin: 2.25rem 0 1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #e9e4ff;
}
.section-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #7c3aed;
    flex-shrink: 0;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    padding: 1rem 1.2rem 1.1rem;
    transition: border-color .18s, box-shadow .18s;
    box-shadow: 0 2px 8px rgba(109,40,217,0.05);
}
.metric-card:hover {
    border-color: #c4b5fd;
    box-shadow: 0 6px 20px rgba(109,40,217,0.1);
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #9d8fc0;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1e1535;
    line-height: 1;
}
.metric-value.accent { color: #6d28d9; }
.metric-value.green  { color: #059669; }
.metric-value.amber  { color: #b45309; }
.metric-value.red    { color: #dc2626; }

/* Result hero */
.result-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
    border: 1px solid #c4b5fd;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(109,40,217,0.1);
}
.result-hero-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.45rem;
}
.result-hero-assets {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #4c1d95;
    margin-bottom: 0.25rem;
}
.result-hero-sub { font-size: 0.9rem; color: #7a7391; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-green  { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.badge-red    { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.badge-purple { background: #f5f3ff; color: #5b21b6; border: 1px solid #ddd6fe; }
.badge-amber  { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }

/* Styled table */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.styled-table th {
    background: #f5f3ff;
    color: #9d8fc0;
    font-weight: 600;
    font-size: 0.63rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding: 0.55rem 1rem;
    text-align: left;
    border-bottom: 1px solid #ddd6fe;
}
.styled-table td {
    padding: 0.6rem 1rem;
    color: #1e1535;
    border-bottom: 1px solid #ede9fe;
    vertical-align: middle;
}
.styled-table tr:hover td { background: #faf8ff; }
.styled-table tr:last-child td { border-bottom: none; }

/* Inputs */
.stTextArea textarea {
    border: 1px solid #ddd6fe !important;
    border-radius: 10px !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    transition: border-color .15s, box-shadow .15s !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #6d28d9 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(109,40,217,0.3) !important;
    transition: all .18s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #5b21b6 !important;
    box-shadow: 0 6px 20px rgba(109,40,217,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    background: #ffffff !important;
    color: #6d28d9 !important;
    border: 1px solid #c4b5fd !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #f5f3ff !important;
    border-color: #7c3aed !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #ddd6fe !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #5b5470 !important;
    font-weight: 500 !important;
}

/* Code */
[data-testid="stCode"] {
    border: 1px solid #ddd6fe !important;
    border-radius: 8px !important;
    background: #faf8ff !important;
}
code { font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #7c3aed !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f3ff; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a78bfa; }

/* Sidebar brand */
.sidebar-brand { padding: 0.5rem 0 1.5rem; }
.sidebar-brand-title {
    font-size: 1.05rem; font-weight: 700;
    color: #4c1d95; letter-spacing: -0.01em;
}
.sidebar-brand-sub { font-size: 0.7rem; color: #9d8fc0; margin-top: 0.15rem; }

/* Step pill */
.step-pill {
    display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.45rem;
}
.step-num {
    width: 22px; height: 22px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.58rem; font-weight: 700;
    flex-shrink: 0;
}
.step-label { font-size: 0.74rem; color: #5b5470; }

/* Info box */
.info-box {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: #5b5470;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(
        f'<div class="section-header"><div class="section-dot"></div>{title}</div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{m["label"]}</div>'
            f'<div class="metric-value {m.get("color","")}">{m["value"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def style_figure(fig, *, accent="#6d28d9", bar_color="#8b5cf6"):
    """Apply the app colour palette to a matplotlib figure in-place."""
    fig.set_facecolor("#f8f7ff")
    for ax in fig.axes:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#ddd6fe")
        ax.tick_params(colors="#7a7391", labelsize=8)
        ax.xaxis.label.set_color("#5b5470")
        ax.yaxis.label.set_color("#5b5470")
        ax.title.set_color("#1e1535")
        ax.grid(True, color="#ede9fe", linewidth=0.5, alpha=0.8)
        for line in ax.get_lines():
            line.set_color(accent)
            line.set_linewidth(2)
        for patch in ax.patches:
            patch.set_facecolor(bar_color)
            patch.set_edgecolor("#7c3aed")
            patch.set_alpha(0.85)
        for coll in ax.collections:
            try:
                coll.set_facecolor(accent)
                coll.set_alpha(0.1)
            except Exception:
                pass
    fig.tight_layout()


@st.cache_data(show_spinner=False)
def _ibm_backends() -> list[str]:
    token = os.getenv("IBM_QUANTUM_TOKEN", "").strip()
    if not token:
        return []
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        ch = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform").strip()
        if ch == "ibm_quantum":
            ch = "ibm_quantum_platform"
        svc = QiskitRuntimeService(channel=ch, token=token)
        return sorted(b.name for b in svc.backends(operational=True))
    except Exception:
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⚛ Quantum Alpha Copilot</div>
        <div class="sidebar-brand-sub">QAOA · LangGraph · MCP</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:.68rem;font-weight:600;letter-spacing:.1em;'
        'text-transform:uppercase;color:#9d8fc0;margin-bottom:.5rem;">Backend</div>',
        unsafe_allow_html=True,
    )

    ibm = _ibm_backends()
    backend_options = ["aer_simulator"] + ibm
    backend_name = st.selectbox(
        "Backend",
        options=backend_options,
        index=0,
        help="aer_simulator runs locally (free). IBM backends require IBM_QUANTUM_TOKEN in .env",
        label_visibility="collapsed",
    )

    if backend_name == "aer_simulator":
        st.markdown(
            '<div style="font-size:.72rem;color:#059669;margin-top:.25rem;">'
            '● Local simulator active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:.72rem;color:#7c3aed;margin-top:.25rem;">'
            f'● IBM Quantum — {backend_name}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.68rem;font-weight:600;letter-spacing:.1em;'
        'text-transform:uppercase;color:#9d8fc0;margin-bottom:.6rem;">Pipeline</div>',
        unsafe_allow_html=True,
    )

    _steps = [
        ("01", "NL → Structured Problem", "#6d28d9"),
        ("02", "QUBO Formulation",        "#7c3aed"),
        ("03", "QAOA Circuit Build",      "#8b5cf6"),
        ("04", "Hybrid Execution",        "#a78bfa"),
        ("05", "Results Analysis",        "#c4b5fd"),
    ]
    for num, label, color in _steps:
        st.markdown(
            f'<div class="step-pill">'
            f'<div class="step-num" style="background:{color}22;border:1px solid {color}55;color:{color};">{num}</div>'
            f'<div class="step-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Translates a plain-English problem into a QUBO, '
        'builds a QAOA circuit, and finds the optimal solution via classical-quantum '
        'hybrid optimization.</div>',
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Quantum Alpha Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Describe a combinatorial optimization problem in plain English. '
    'The agent formulates it as QUBO, builds a QAOA circuit, and finds the optimal solution.</div>',
    unsafe_allow_html=True,
)

# ── Examples ──────────────────────────────────────────────────────────────────
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

col_txt, col_ex = st.columns([3, 1])
with col_ex:
    ex_choice = st.selectbox(
        "Example", ["— Custom —"] + list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
with col_txt:
    user_input = st.text_area(
        "problem",
        value=EXAMPLES.get(ex_choice, "") if ex_choice != "— Custom —" else "",
        height=128,
        placeholder="e.g.  Select 3 from AAPL, GOOGL, MSFT, AMZN, TSLA to maximize return…",
        label_visibility="collapsed",
    )

st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
run_btn = st.button("Run Quantum Optimization", type="primary", use_container_width=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if not user_input.strip():
        st.error("Please enter a problem statement.")
        st.stop()
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        st.error("No LLM API key found — add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file.")
        st.stop()

    # Clear any previous result before running
    st.session_state.pop("_result", None)

    with st.spinner("Running the quantum optimization pipeline…"):
        try:
            from agents.graph import build_graph
            result = build_graph().invoke({
                "user_input": user_input.strip(),
                "backend_name": backend_name,
                "logs": [],
                "retry_count": 0,
            })
            # Persist result in session_state so it survives Streamlit reruns
            st.session_state["_result"] = result
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

# ── Render results from session state (survives page reruns) ──────────────────
if "_result" not in st.session_state:
    st.stop()

result = st.session_state["_result"]

if result.get("error"):
    st.error(f"Pipeline stopped: {result['error']}")
    st.stop()

parsed = result.get("parsed_problem", {})
if parsed.get("clarification_needed"):
    st.warning(f"Clarification needed — {parsed.get('clarification_question')}")
    st.stop()

st.success("Optimization complete")

# ── Section 1: Parsed Problem ─────────────────────────────────────────────────
section("01  ·  Parsed Problem")
assets_list = parsed.get("assets", [])
metric_row([
    {"label": "Assets",      "value": len(assets_list),                         "color": "accent"},
    {"label": "Objective",   "value": parsed.get("objective","—").capitalize(),  "color": ""},
    {"label": "Select k",    "value": parsed.get("num_select", "—"),             "color": ""},
    {"label": "Constraints", "value": len(parsed.get("hard_constraints", [])),   "color": ""},
])

if assets_list:
    weights = parsed.get("objective_weights", {})
    pills = " ".join(
        f'<span style="display:inline-flex;align-items:center;gap:.3rem;'
        f'background:#f5f3ff;border:1px solid #ddd6fe;color:#5b21b6;'
        f'border-radius:7px;padding:.2rem .65rem;font-size:.76rem;font-weight:500;">'
        f'{a}<span style="color:#9d8fc0;font-weight:400;">{weights.get(a,"")}</span></span>'
        for a in sorted(assets_list, key=lambda x: weights.get(x, 0), reverse=True)
    )
    st.markdown(
        f'<div style="margin-top:.75rem;display:flex;flex-wrap:wrap;gap:.4rem;">{pills}</div>',
        unsafe_allow_html=True,
    )

with st.expander("View structured JSON"):
    st.json({k: v for k, v in parsed.items() if k != "clarification_question"})

# ── Section 2: QUBO Matrix ────────────────────────────────────────────────────
section("02  ·  QUBO Matrix")
qubo_matrix = result.get("qubo_matrix")
if qubo_matrix:
    Q = np.array(qubo_matrix)
    n = Q.shape[0]
    labels = parsed.get("assets", [str(i) for i in range(n)])

    col_heat, col_stats = st.columns([3, 1])
    with col_heat:
        sz = min(max(n, 4), 9)
        fig_q, ax_q = plt.subplots(figsize=(sz, sz))
        im = ax_q.imshow(Q, cmap="PuRd", aspect="auto")
        ax_q.set_xticks(range(len(labels)))
        ax_q.set_yticks(range(len(labels)))
        ax_q.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax_q.set_yticklabels(labels, fontsize=9)
        cb = plt.colorbar(im, ax=ax_q, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8, colors="#7a7391")
        cb.outline.set_edgecolor("#ddd6fe")
        ax_q.set_title("QUBO Coefficient Matrix", fontsize=11, pad=10)
        for sp in ax_q.spines.values():
            sp.set_edgecolor("#ddd6fe")
        fig_q.tight_layout()
        st.pyplot(fig_q, use_container_width=True)

    with col_stats:
        st.markdown("<br>", unsafe_allow_html=True)
        for lbl, val in [
            ("Variables", Q.shape[0]),
            ("Non-zero",  int(np.count_nonzero(Q))),
            ("Offset",    f"{result.get('qubo_offset', 0.0):.3f}"),
            ("Qubits",    result.get("num_qubits", "—")),
        ]:
            st.markdown(
                f'<div class="metric-card" style="margin-bottom:.55rem;">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value" style="font-size:1.15rem;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Section 3: QAOA Circuit ───────────────────────────────────────────────────
section("03  ·  QAOA Circuit")
feasible = result.get("transpilation_feasible")
depth    = result.get("circuit_depth") or 0
metric_row([
    {"label": "Qubits",        "value": result.get("num_qubits", "—"),   "color": "accent"},
    {"label": "Circuit Depth", "value": depth,                            "color": "amber" if depth > 100 else "green"},
    {"label": "Parameters",    "value": result.get("num_parameters","—"), "color": ""},
    {"label": "Transpilation", "value": "Pass" if feasible else "Fail",   "color": "green" if feasible else "red"},
])
if result.get("circuit_qasm"):
    with st.expander("View OpenQASM 3 source"):
        st.code(result["circuit_qasm"], language="text")

# ── Section 4: Hybrid Execution ───────────────────────────────────────────────
section("04  ·  Hybrid Execution  ·  COBYLA + QAOA")
conv = result.get("convergence_history", [])
estimator_backend = result.get("estimator_backend", result.get("backend_name", "—"))
sampler_backend = result.get("sampler_backend", result.get("backend_name", "—"))
metric_row([
    {"label": "Optimal Energy ⟨H⟩", "value": f"{result.get('optimal_energy', 0.0):.5f}", "color": "accent"},
    {"label": "Optimizer Backend",   "value": estimator_backend,                            "color": ""},
    {"label": "Sampling Backend",    "value": sampler_backend,                              "color": ""},
    {"label": "COBYLA Iterations",   "value": len(conv),                                   "color": ""},
])

if estimator_backend != sampler_backend:
    st.markdown(
        '<div class="info-box" style="margin-top:.75rem;">'
        'Objective evaluations are run on the local simulator for fast convergence. '
        'The final shot-based distribution is sampled on the selected IBM backend so the '
        'frontend can render hardware results without waiting on hundreds of QPU jobs.'
        '</div>',
        unsafe_allow_html=True,
    )

sol      = result.get("final_solution", {}) or {}
conv_fig = sol.get("_convergence_fig")

if conv_fig:
    style_figure(conv_fig, accent="#6d28d9")
    st.pyplot(conv_fig, use_container_width=True)
elif conv:
    fig_c, ax_c = plt.subplots(figsize=(10, 3.2))
    ax_c.plot(range(1, len(conv) + 1), conv, color="#6d28d9", linewidth=2)
    ax_c.fill_between(range(1, len(conv) + 1), conv, alpha=0.08, color="#6d28d9")
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("⟨H⟩")
    ax_c.set_title("Optimization Convergence")
    ax_c.grid(True, color="#ede9fe", linewidth=0.5)
    for sp in ax_c.spines.values():
        sp.set_edgecolor("#ddd6fe")
    fig_c.tight_layout()
    st.pyplot(fig_c, use_container_width=True)

# ── Section 5: Results ────────────────────────────────────────────────────────
section("05  ·  Optimization Results")
if sol:
    selected      = sol.get("selected_assets", [])
    obj_val       = sol.get("objective_value", 0)
    constraint_ok = sol.get("constraint_satisfied", False)
    approx_ratio  = sol.get("approximation_ratio", 0)
    classical     = sol.get("classical_optimum_selection", [])
    classical_val = sol.get("classical_optimum_value")

    badge = (
        '<span class="badge badge-green">Constraints satisfied</span>'
        if constraint_ok else
        '<span class="badge badge-red">Constraints violated</span>'
    )
    st.markdown(
        f'<div class="result-hero">'
        f'<div class="result-hero-label">Optimal Portfolio</div>'
        f'<div class="result-hero-assets">{" · ".join(selected) if selected else "No solution found"}</div>'
        f'<div class="result-hero-sub" style="margin-bottom:.7rem;">'
        f'Objective value: <strong style="color:#4c1d95;">{obj_val:.4f}</strong></div>'
        f'{badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    ratio_color = "green" if approx_ratio >= 0.95 else "amber" if approx_ratio >= 0.8 else "red"
    metric_row([
        {"label": "Objective Value",     "value": f"{obj_val:.4f}",                                         "color": "accent"},
        {"label": "Approximation Ratio", "value": f"{approx_ratio:.3f}",                                    "color": ratio_color},
        {"label": "Total Shots",         "value": sol.get("total_shots", 1024),                             "color": ""},
        {"label": "Classical Optimum",   "value": f"{classical_val:.4f}" if classical_val else "—",         "color": ""},
    ])

    if classical:
        st.markdown(
            f'<div class="info-box" style="margin-top:.75rem;">'
            f'Classical brute-force optimum: '
            f'<strong style="color:#4c1d95;">{", ".join(classical)}</strong>'
            f' (value = {f"{classical_val:.4f}" if classical_val else "N/A"})'
            f'</div>',
            unsafe_allow_html=True,
        )

    hist_fig = sol.get("_histogram_fig")
    if hist_fig:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        style_figure(hist_fig, accent="#6d28d9", bar_color="#8b5cf6")
        st.pyplot(hist_fig, use_container_width=True)

    candidates = sol.get("top_candidates", [])
    if candidates:
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        _ok_badge  = "<span class='badge badge-green'>✓</span>"
        _fail_badge = "<span class='badge badge-red'>✗</span>"
        rows = "".join(
            f'<tr>'
            f'<td><code style="background:#f5f3ff;padding:.15rem .4rem;border-radius:5px;'
            f'font-size:.74rem;color:#5b21b6;">{c["bitstring"]}</code></td>'
            f'<td style="font-weight:500;">{", ".join(c["selection"]) or "—"}</td>'
            f'<td>{c["probability"]:.4f}</td>'
            f'<td>{c["objective_value"]:.4f}</td>'
            f'<td>{_ok_badge if c["constraint_satisfied"] else _fail_badge}</td>'
            f'</tr>'
            for c in candidates
        )
        st.markdown(
            f'<div style="border:1px solid #ddd6fe;border-radius:12px;overflow:hidden;margin-top:.5rem;">'
            f'<table class="styled-table">'
            f'<thead><tr><th>Bitstring</th><th>Selected Assets</th>'
            f'<th>Probability</th><th>Objective</th><th>Constraints</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

# ── Section 6: Log ────────────────────────────────────────────────────────────
section("06  ·  Experiment Log")
logs = result.get("logs", [])
with st.expander(f"{len(logs)} log entries"):
    _colors = {
        "[intake":    "#6d28d9",
        "[qubo":      "#7c3aed",
        "[circuit":   "#0369a1",
        "[execution": "#b45309",
        "[results":   "#065f46",
        "[handle":    "#dc2626",
    }
    rows_html = "".join(
        f'<div style="font-size:.72rem;color:{next((v for k,v in _colors.items() if k in l),"#7a7391")};'
        f'padding:.12rem 0;border-bottom:1px solid #f3f0ff;'
        f'font-family:\'JetBrains Mono\',monospace;">{l}</div>'
        for l in logs
    )
    st.markdown(
        f'<div style="background:#faf8ff;border:1px solid #ede9fe;border-radius:10px;'
        f'padding:.75rem;max-height:280px;overflow-y:auto;">{rows_html}</div>',
        unsafe_allow_html=True,
    )

log_json = json.dumps(
    {k: v for k, v in result.items() if k not in ("ising_hamiltonian", "final_solution")},
    default=str, indent=2,
)
st.download_button(
    "Download Experiment Log (JSON)",
    data=log_json,
    file_name="quantum_experiment_log.json",
    mime="application/json",
    use_container_width=True,
)
