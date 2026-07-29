# Sponsor Match

*[Deutsche Version](README.md)*

Sponsor Match is a Streamlit app that uses a [LangGraph](https://www.langchain.com/langgraph) agent to evaluate how well a company fits as a sponsor for a sports club. The agent researches the company on the web, compares it against the club's profile, and returns a fit score (0.0–1.0) with reasoning, plus either a draft outreach message or a rejection explanation depending on the result.

## Features

- **Research & fit evaluation**: web research on the company (Tavily), LLM-based scoring with pros/cons bullet points and a recommendation.
- **Competitor analysis (portfolio plugin)**: analyzes, generically for any company, its own sponsorship portfolio (categories, active sponsorships, target audience), audience match with the club, and market saturation in that sport — with a dynamic score adjustment.
- **Size matching**: compares club size (static per club) and company size (estimated via web search + LLM: Small/Medium/Large) and adjusts the score using a size-match matrix.
- **Budget estimator & external sponsorship DB**: data-driven budget estimate and historical sponsorship examples from a fictional external database.
- **Case study knowledge base**: RAG search over fictional sponsorship case studies as additional context for the evaluation.
- **Human-in-the-loop (HITL)**: for uncertain scores (0.45–0.55), the user can agree, disagree, or request more information — the agent learns from this for future requests about the same company.
- **Manual score correction**: after negative feedback, the user can set the correct score via a slider; this is stored as ground truth and used the next time for the same company+club combination.
- **Agent learning**: feedback (positive/negative) on similar companies slightly influences future scores (±0.05 to ±0.10).
- **Score caching**: identical company+club requests return a stable, consistent score instead of drifting on every repeat.
- **Analysis history & agent evaluation**: filterable history of all analyses, a score-consistency metric, and an LLM-judge-based RAGAs-style quality report (relevance, faithfulness, answer relevance, trend over time).
- **Plugin system**: every optional capability (competitor analysis, budget estimator, sponsorship DB, size matching) can be toggled individually in the Plugin Manager — changes apply immediately to the display without re-running the analysis.
- **Multilingual**: German, English, French (UI text and agent responses).
- **User management**: login/registration, preferred language & favorite clubs, account deletion, session timeout.
- **LangSmith tracing**: optional trace link per analysis for debugging/traceability.
- **MCP server**: `mcp_server.py` exposes research, competitor analysis, fit evaluation, and size matching as MCP tools (e.g. for Claude Desktop) — see [README_MCP.md](README_MCP.md).

## Tech stack

- **UI**: [Streamlit](https://streamlit.io/) (dark theme, custom CSS)
- **Agent**: [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/) via [OpenRouter](https://openrouter.ai/) (swappable LLM)
- **Web search**: [Tavily](https://tavily.com/)
- **Persistence**: SQLite (`data/sponsor_match.db` for analyses, `data/users.db` for user accounts)
- **Tracing**: [LangSmith](https://smith.langchain.com/) (optional)
- **MCP server**: [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (`mcp>=1.9,<2`)
- **Package management**: [uv](https://docs.astral.sh/uv/)

## Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

```bash
uv sync
```

### Environment variables

Create a `.env` file in the project root:

```bash
TAVILY_API_KEY=...          # required – web search
OPENROUTER_API_KEY=...      # required – LLM access via OpenRouter
LANGSMITH_API_KEY=...       # optional – enables tracing/trace links
LANGSMITH_PROJECT=sponsor-match   # optional, default: sponsor-match
```

### Run

```bash
uv run streamlit run main.py
```

The app is then available at `http://localhost:8501`. On first use, create an account via "Register".

## Project structure

```
main.py                        # Streamlit UI: auth, sidebar, form, result cards
mcp_server.py                  # FastMCP server: research/competitor analysis/fit/size matching as MCP tools
mcp_config.json                # Ready-made Claude Desktop config entry
src/agent.py                   # LangGraph agent, scoring logic, SQLite persistence, plugins
src/tools.py                   # Tavily search tool
data/clubs.json                # Club profiles (sport, fanbase, values, size, ...)
data/case_studies.json         # Fictional case study knowledge base
data/sponsorship_database.json # Fictional external sponsorship database
data/available_plugins.json    # Plugin configuration (on/off, required plugins)
.streamlit/config.toml         # Color scheme (dark mode)
```

`data/sponsor_match.db`, `data/users.db`, `data/feedback.jsonl`, `data/security_log.jsonl`, and `logs/` are generated at runtime and are intentionally not part of the repo (see `.gitignore`).

## Notes

- All case studies, the external sponsorship DB, and company-size/competitor-analysis values are AI estimates or fictional demo data — not verified market data.
- Question detection in the company input field (protection against prompt-injection-like input) currently checks primarily for German question words/"?".
