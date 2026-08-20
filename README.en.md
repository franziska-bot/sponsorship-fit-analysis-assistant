# Sponsor Match

*[Deutsche Version](README.md)*

Sponsor Match is a Streamlit app that uses a [LangGraph](https://www.langchain.com/langgraph) agent to evaluate how well a company fits as a sponsor for a sports club. The agent researches the company on the web, compares it against the club's profile, and returns a fit score (0.0–1.0) with reasoning, plus either a draft outreach message or a rejection explanation depending on the result.

⚠️ **Showcase**: [Project link coming soon](#) *(TODO: add after uploading to [showcase.turingcollege.com](https://showcase.turingcollege.com/))*

## Problem & solution

**Situation:** Sports clubs rely on sponsorship for part of their budget. Before a club even approaches a company, someone has to judge whether that company is a good fit as a sponsor — values, audience, budget, sport relevance.

**Complication:** That judgment needs research across several independent dimensions — the company's public reputation, its financial capacity (often buried in annual reports), whether it's already saturated with sponsorships in that sport, how well its audience matches the fanbase. Doing this by hand is slow, inconsistent (different people weigh these factors differently), and usually ends in an unexplained gut call — with no way to later trace why a company was rated a good or bad fit, or how confident that rating actually was.

**Resolution:** Sponsor Match automates this vetting with a multi-stage LangGraph pipeline of three specialist agents (research, financial analysis, fit evaluation): parallel web research with source-credibility scoring, real financial-metric extraction from annual reports found online, a transparent 8-factor weighted evaluation instead of a single black-box number, and explicit data-quality/confidence metrics so it's clear when to trust the result versus do more manual digging. A human-in-the-loop step for uncertain cases and a learning loop from past feedback turn a one-off gut feeling into a traceable, continuously improving process.

## Features

- **Research & fit evaluation**: web research on the company (Tavily), LLM-based scoring with pros/cons bullet points and a recommendation.
- **Competitor analysis (portfolio plugin)**: analyzes, generically for any company, its own sponsorship portfolio (categories, active sponsorships, target audience), audience match with the club, and market saturation in that sport — with a dynamic score adjustment.
- **Size matching**: compares club size (static per club) and company size (estimated via web search + LLM: Small/Medium/Large) and adjusts the score using a size-match matrix.
- **Budget estimator & external sponsorship DB**: data-driven budget estimate and historical sponsorship examples from a fictional external database, complemented by real financial metrics (revenue, EBITDA, profit, cash, marketing spend) extracted best-effort from annual reports (PDF) found online.
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
- **MCP server**: `mcp_server.py` exposes research, competitor analysis, fit evaluation, and size matching as MCP tools (e.g. for Claude Desktop) — setup in the documentation below.
- **Observability**: detailed agent tracing (`logs/agent_trace.log`), per-agent performance metrics (timing breakdown), and data-quality metrics (source credibility, PDF-extraction confidence, per-factor confidence breakdown) shown directly in the results view.
- **PDF cache & error handling**: discovered annual reports are cached locally (30 days for PDFs, 60 days for extracted metrics), downloads run with exponential-backoff retry (1s/2s/4s) and fail gracefully instead of aborting the whole analysis.
- **Security**: 70+ attack-pattern detection (prompt injection, jailbreaks, instruction chaining, context confusion, obfuscation via Base64/ROT13/hex/leetspeak/Cyrillic lookalikes, technical exploits) in both German and English, two-tier rate limiting (session- and IP-based), and attack-specific banning (temporary ban after 3 attacks/minute, permanent block + admin alert after 20/hour).
- **Health-check endpoint**: optional standalone monitoring server (`src/health_server.py`) for DevOps/ops.

## Tech stack

- **UI**: [Streamlit](https://streamlit.io/) (dark theme, custom CSS)
- **Agent**: [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/) via [OpenRouter](https://openrouter.ai/) (swappable LLM)
- **Web search**: [Tavily](https://tavily.com/)
- **Persistence**: SQLite (`data/sponsor_match.db` for analyses, `data/users.db` for user accounts)
- **Tracing**: [LangSmith](https://smith.langchain.com/) (optional)
- **MCP server**: [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (`mcp>=1.9,<2`)
- **PDF extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber) (text + tables from annual reports)
- **Tests**: [pytest](https://pytest.org/)
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
src/orchestrator.py            # Manager: LangGraph pipeline, SQLite persistence, RAGAs-style eval
src/search_agent.py            # Specialist: web research, case-study RAG, sponsorship DB lookup
src/analysis_agent.py          # Specialist: sponsorship portfolio, company size, budget estimate, PDF extraction
src/fit_agent.py               # Specialist: fit scoring, outreach draft, rejection reason
src/security_validator.py      # Input security: 70+ attack-pattern categories + rate limiting/banning
src/logger.py                  # AgentLogger: structured tracing + security audit log
src/pdf_cache.py               # PDFCache: two-tier cache for annual-report PDFs
src/error_handler.py           # ErrorHandler: retry with exponential backoff, graceful degradation
src/performance_monitor.py     # PerformanceMonitor: per-agent timing tracking
src/health_server.py           # Standalone health-check server (monitoring, optional)
src/tools.py                   # Shared infra: Tavily search tool, LLM factory, state schema, plugins
tests/test_security_injection.py  # 122 security tests (attack patterns + legitimate-input cases)
data/clubs.json                # Club profiles (sport, fanbase, values, size, ...)
data/case_studies.json         # Fictional case study knowledge base
data/sponsorship_database.json # Fictional external sponsorship database
data/available_plugins.json    # Plugin configuration (on/off, required plugins)
.streamlit/config.toml         # Color scheme (dark mode)
```

`data/sponsor_match.db`, `data/users.db`, `data/feedback.jsonl`, `data/security_log.jsonl`, `data/admin_alerts.jsonl`, `data/pdf_cache/`, and `logs/` are generated at runtime and are intentionally not part of the repo (see `.gitignore`).

## Further documentation

- [README_AGENTS.md](README_AGENTS.md) — agent architecture in detail: data flow, logging, caching, performance, error handling.
- [MULTI_AGENT_ARCHITECTURE.md](MULTI_AGENT_ARCHITECTURE.md) — LangGraph state machine, module dependencies, full state schema.
- [QUALITY_METRICS.md](QUALITY_METRICS.md) — what the confidence scores mean and when to trust them.
- [DEPLOYMENT.md](DEPLOYMENT.md) — setup, monitoring, performance tuning, troubleshooting.
- [README_MCP.md](README_MCP.md) — MCP server setup (e.g. for Claude Desktop).

## Notes

- All case studies, the external sponsorship DB, and company-size/competitor-analysis values are AI estimates or fictional demo data — not verified market data.
- Question detection in the company input field (protection against prompt-injection-like input) currently checks primarily for German question words/"?".
