# Agent Architecture

Companion to [README.md](README.md) (product-level overview) — this file documents the
multi-agent pipeline itself: what each agent does, how they hand off state, and the
operational layers added on top (logging, caching, error handling, performance/quality
metrics) as part of the Phase 3 "Polish & Production Readiness" work.

## Architecture overview

```
                          OrchestratingAgent (src/orchestrator.py)
                                       │
                validates input (security_validator.py), builds +
                runs a LangGraph pipeline, times every node, persists
                to SQLite, returns the accumulated state
                                       │
        ┌──────────────────┬──────────┴──────────┬────────────────────┐
        ▼                  ▼                      ▼                    ▼
  research_company   analyze_financials      evaluate_fit      draft_outreach /
  (SearchAgent)       (AnalysisAgent)          (FitAgent)       explain_rejection
                                                                    (FitAgent)
```

Concretely, `OrchestratingAgent._build_graph()` wires up a LangGraph `StateGraph`:

```
START → research_company → analyze_financials → evaluate_fit ─┬→ draft_outreach → END
                                                                └→ explain_rejection → END
```

Every node reads and writes the same shared `SponsorMatchState` (a `TypedDict`, defined
in `src/tools.py`) — there's no message-passing between agents, just state accumulation.
`OrchestratingAgent.invoke()` runs the graph via `.stream(stream_mode="updates")` rather
than a plain `.invoke()`, specifically so it can log each node's own delta and time each
node's wall-clock duration without a second, wasted pipeline run.

## What each agent does

### SearchAgent — `src/search_agent.py`

- `research_company()`: runs 5 Tavily queries in parallel (sponsorship, budget, values,
  audience, timeline history), scores every result's domain for credibility (0.0–1.0,
  `.gov`/`.edu`/major outlets score highest, social platforms lowest), deduplicates by
  URL, and feeds the top sources into a single LLM call that returns a structured
  summary + sentiment + timeline + confidence — one call instead of three, to keep token
  cost down.
- `search_case_studies()` / `query_sponsorship_db()`: RAG lookups into the fictional
  internal case-study knowledge base and external sponsorship database (`data/`), scoped
  strictly to the club's own sport (a case study in the wrong sport is never shown, even
  as a fallback).

### AnalysisAgent — `src/analysis_agent.py`

- `_analyze_company_sponsorship_portfolio()` (competitor_analysis plugin): what the
  company itself already sponsors — categories, active deal count, saturation in the
  club's sport, audience overlap with the club.
- `_estimate_company_size()` (size_matching plugin): Small/Medium/Large classification
  from a web search, compared against the club's own size for a score adjustment.
- `_research_financial_pdfs()` (budget_estimator plugin, Phase 2): finds annual
  report/10-K PDFs via 4 parallel Tavily queries, downloads them (retried through
  `ErrorHandler`, see below), extracts text + tables via `pdfplumber`, and regex-matches
  five financial metrics (revenue, EBITDA, profit, cash, marketing spend). Best-effort by
  design — most companies have no directly findable report PDF, and that's the expected
  normal case, not an error.

### FitAgent — `src/fit_agent.py`

- `evaluate_fit()`: one LLM call scores 8 independent factors (values fit, audience fit,
  sport relevance, historical precedent, budget fit, strategic potential, brand safety,
  market timing), which Python then combines via fixed weights
  (20/20/15/15/10/10/5/5%) into the base score — the LLM never invents the final number
  itself. The base score is then adjusted (in order): agent-learning feedback pattern
  (±0.05–0.10), competitor-portfolio saturation impact, HITL ground truth for uncertain
  scores, and size-match compatibility.
- **Score caching**: an identical company+club pair reuses its previous score instead of
  re-scoring (and re-applying adjustments) on every repeat request — consistency over
  novelty.
- `draft_outreach()` / `explain_rejection()`: routed by `route_by_fit()` on a 0.6 score
  threshold.

### OrchestratingAgent — `src/orchestrator.py`

Builds and runs the LangGraph pipeline above, validates the company name against
`security_validator.py` before anything reaches an LLM prompt, persists every analysis to
SQLite (`data/sponsor_match.db`), manages user accounts (`data/users.db`), exposes
LangSmith trace URLs, and runs the RAGAs-style LLM-judge evaluation
(`evaluate_with_ragas()`).

## Data flow between agents

`SponsorMatchState` fields written by each stage (abbreviated):

| Stage | Writes |
|---|---|
| `research_company` | `research_findings` (markdown), `research_quality` (dict) |
| `analyze_financials` | `competitor_analysis`, `size_compatibility`, `pdf_financials`, `financial_data` (dict) |
| `evaluate_fit` | `fit_score`, `fit_reasoning`, `fit_agent_factors` (dict), `agent_confidence`, `is_uncertain` |
| `draft_outreach` / `explain_rejection` | `outreach_draft` or `rejection_reason` |

`token_usage` is the one field every node appends to (`Annotated[list, operator.add]` in
the state schema) — LangGraph merges it automatically across nodes.

## Logging

`src/logger.py`'s `AgentLogger` — one instance per module (`search_agent`,
`analysis_agent`, `fit_agent`, `orchestrator`), each logging to the same file:

```
logs/agent_trace.log     # full DEBUG-level trace: every log_agent_step() call
                          # (parallel query dispatch, PDF cache hits, retries, ...)
```

Console output is INFO-level only (start/result/error), so an interactive
`streamlit run` doesn't get flooded with per-step detail — the file always has the full
trace. Format: `START <task>`, `STEP <detail>`, `DONE <task> (<elapsed>s) | {context}`,
`ERROR <task>: <error> | {context}`.

## Cache

`src/pdf_cache.py`'s `PDFCache`, used only by the PDF financial-extraction pipeline:

```
data/pdf_cache/pdf/<md5(url)>.pdf          # raw PDF bytes, 30-day TTL
data/pdf_cache/extraction/<md5(url)>.json  # extracted metrics + latest-year metadata, 60-day TTL
```

A repeat run against the same report URL skips download + parse + regex-extraction
entirely on an extraction-cache hit. Measured locally against a real report PDF: ~1.8s
cold (download + parse + extract) vs. ~0ms on a cache hit.

## Performance

Timed via `src/performance_monitor.py`'s `PerformanceMonitor`, one bucket per agent
(`search_agent`, `analysis_agent`, `fit_agent` — `draft_outreach`/`explain_rejection`
also count toward `fit_agent`, since both live in `fit_agent.py`). Exposed in
`state["performance_metrics"]` and shown in main.py's "Performance" card.

These are real numbers from live runs during development (Nike Inc / FC Nordlicht),
**not fixed targets** — actual timing depends heavily on Tavily search latency and how
many PDF reports get found on a given run:

| Run | Total | SearchAgent | AnalysisAgent | FitAgent |
|---|---|---|---|---|
| 1st run (cold, PDFs downloaded fresh) | ~23.4s | ~5.7s | ~17.7s | ~2ms* |
| 2nd run (PDF cache hit) | ~6.4s | ~3.6s | ~2.8s | ~3ms* |

\* FitAgent's own `evaluate_fit()` step is near-instant on a **score cache hit** — most
of the pipeline's wall time is SearchAgent's 5 parallel web searches and AnalysisAgent's
PDF download/parse work, not LLM latency. Run `get_slowest_agent()` to see which agent
was the bottleneck for a specific analysis.

## Error handling

`src/error_handler.py`'s `ErrorHandler`, used for the PDF download/parse pipeline only
(the one part of the pipeline that hits arbitrary third-party URLs):

- **Retries**: `retry_on_failure()` retries a failing call up to 3 times with exponential
  backoff (1s → 2s → 4s), triggered by either a raised exception or a falsy return value
  (this codebase's I/O helpers signal failure by returning `None`, not by raising).
- **Graceful degradation**: once retries are exhausted, `handle_pdf_download_error()` /
  `handle_extraction_error()` log the failure and the caller skips that one PDF —
  the rest of the batch (other PDFs, the rest of the pipeline) continues unaffected. A
  single broken/paywalled/403'd report never aborts the whole analysis.

## Quality metrics

See [QUALITY_METRICS.md](QUALITY_METRICS.md) for the full explanation of what each
confidence score means and when to trust it vs. do manual research. Short version: every
stage reports its own confidence (`research_quality.data_quality`,
`financial_data.data_quality`, per-factor confidence in `fit_agent_factors`) so a low
overall fit score and a low *confidence* in that score are visibly different situations.
