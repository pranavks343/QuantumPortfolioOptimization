# Quantum Portfolio Optimization Copilot

An agentic quantum optimization system that translates natural-language portfolio problems into quantum circuits and solves them using QAOA. The pipeline is orchestrated by a LangGraph agent graph, with quantum operations exposed through an MCP (Model Context Protocol) server backed by Qiskit.

## How It Works

```
Natural Language Problem
        │
        ▼
┌──────────────┐     LLM parses problem into
│  Intake Agent │──▶  structured JSON (assets,
└──────┬───────┘     returns, constraints)
       │
       ▼
┌──────────────────┐  Builds QuadraticProgram,
│ QUBO Formulator  │──▶ converts to Ising
└──────┬───────────┘  Hamiltonian
       │
       ▼
┌────────────────────┐  Generates parameterized
│ Circuit Constructor│──▶ QAOA ansatz, checks
└──────┬─────────────┘  transpilation feasibility
       │
       ▼
┌────────────────────┐  Hybrid classical-quantum
│ Execution Manager  │──▶ loop (COBYLA + QAOA)
└──────┬─────────────┘  via MCP tools
       │
       ▼
┌──────────────────┐  Decodes bitstrings, checks
│ Results Analyzer │──▶ constraints, computes
└──────────────────┘  approximation ratio
```

The entire pipeline runs as a LangGraph state graph with automatic error handling, retry logic, and fallback from IBM Quantum hardware to a local Aer simulator.

## Features

- **Natural language input** — describe your optimization problem in plain English
- **LLM-powered parsing** — Claude or GPT-4 extracts assets, returns, constraints, and objectives
- **QUBO/Ising formulation** — automatic conversion with penalty-based constraint handling
- **QAOA execution** — parameterized quantum circuits optimized via classical COBYLA
- **MCP tool server** — quantum operations (sampling, estimation, validation) exposed as MCP tools
- **IBM Quantum support** — run on real hardware (ibm_brisbane, ibm_kyoto, ibm_osaka) or simulate locally
- **Interactive dashboard** — Streamlit UI with QUBO heatmaps, convergence plots, and measurement histograms
- **Classical benchmarking** — brute-force comparison and approximation ratio computation

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── mcp_server.py           # MCP server (stdio) exposing Qiskit tools
├── agents/
│   ├── graph.py            # LangGraph state graph (pipeline orchestration)
│   ├── orchestrator.py     # Optional LLM orchestrator for reasoning/summaries
│   └── states.py           # Shared state schema (AgentState, ParsedProblem)
├── modules/
│   ├── intake_agent.py     # NL → structured problem parsing
│   ├── qubo_formulator.py  # Problem → QUBO/Ising Hamiltonian
│   ├── circuit_constructor.py  # Hamiltonian → QAOA circuit
│   ├── execution_manager.py    # Hybrid optimization loop
│   └── results_analyzer.py     # Bitstring decoding & visualization
├── tools/
│   ├── mcp_client.py       # LangChain tool wrappers for MCP server
│   └── qiskit_tools.py     # Utility functions (serialization, metrics)
├── requirements.txt
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10+
- An LLM API key (Anthropic or OpenAI)
- *(Optional)* IBM Quantum account for real hardware execution

### Installation

```bash
git clone https://github.com/pranavks343/QuantumPortfolioOptimization.git
cd QuantumPortfolioOptimization

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | LLM API key for the intake agent |
| `IBM_QUANTUM_TOKEN` | *(Optional)* IBM Quantum API token for hardware backends |

### Running

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The MCP server starts automatically as a subprocess when the pipeline runs — no separate launch step is needed.

## Usage

1. Open the dashboard in your browser (default: `http://localhost:8501`)
2. Select a quantum backend from the sidebar (`aer_simulator` for local testing)
3. Type an optimization problem or pick an example:
   - *"I have 5 assets: AAPL, GOOGL, MSFT, AMZN, TSLA. Their expected returns are 0.9, 0.8, 0.75, 0.85, 0.7. Select exactly 3 to maximize total return."*
4. Click **Run Quantum Optimization**
5. View results: parsed problem, QUBO matrix, QAOA circuit, convergence curve, and the optimal portfolio

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agent orchestration and state management
- **[Qiskit](https://qiskit.org/)** — quantum circuit construction, simulation, and IBM Quantum access
- **[MCP](https://modelcontextprotocol.io/)** — tool server protocol for quantum operations
- **[Streamlit](https://streamlit.io/)** — interactive web dashboard
- **[LangChain](https://www.langchain.com/)** — LLM integration (Claude / GPT-4)

## License

This project is provided as-is for educational and research purposes.
