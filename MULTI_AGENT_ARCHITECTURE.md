# Multi-Agent Architecture

Technical reference for how the pipeline is actually wired together — the LangGraph
state machine, the module dependency graph, and the full state schema. For a narrative
walkthrough of what each agent does and why, see [README_AGENTS.md](README_AGENTS.md);
for confidence-score semantics see [QUALITY_METRICS.md](QUALITY_METRICS.md).

## LangGraph state machine

Built in `OrchestratingAgent._build_graph()` (`src/orchestrator.py`):

```
START
  │
  ▼
research_company ──────────────────────► SearchAgent (src/search_agent.py)
  │
  ▼
analyze_financials ─────────────────────► AnalysisAgent (src/analysis_agent.py)
  │
  ▼
evaluate_fit ────────────────────────────► FitAgent (src/fit_agent.py)
  │
  ▼
route_by_fit(state) — conditional edge
  │
  ├── fit_score >= 0.6 ──► draft_outreach ──────► END
  │
  └── fit_score <  0.6 ──► explain_rejection ───► END
```

Linear through the first three nodes, then a single conditional fork. No cycles, no
sub-graphs, no tool-calling loop inside a node — each node is one Python function that
reads `SponsorMatchState`, does its work (LLM calls, web search, PDF processing), and
returns a partial-state dict that LangGraph merges into the accumulated state.

`OrchestratingAgent.invoke()` runs this via `self._graph.stream(state,
stream_mode="updates")` rather than `.invoke()` — the stream yields one delta per
completed node, which is what makes it possible to log each node's own output keys and
measure its wall-clock duration (`PerformanceMonitor`) without a second, duplicate run of
the graph.

## Module dependency graph

```
main.py ─────────────┐
mcp_server.py ────────┤
                       ▼
              src/orchestrator.py  (manager: builds/runs the graph, SQLite, auth, LangSmith)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 src/fit_agent.py  src/analysis_   src/search_agent.py
        │           agent.py            │
        │              │                │
        └──────────────┴────────────────┘
                       │
                       ▼
                src/tools.py   (LLM factory, search tool, SponsorMatchState, plugin config)

Cross-cutting leaves (imported by orchestrator + analysis_agent, no upward dependencies):
  src/logger.py             — AgentLogger, one instance per module
  src/pdf_cache.py          — PDFCache, used only by analysis_agent's PDF pipeline
  src/error_handler.py      — ErrorHandler, used only by analysis_agent's PDF pipeline
  src/performance_monitor.py — PerformanceMonitor, used only by orchestrator
  src/security_validator.py — validate_input() + rate_limit_check(), used by main.py,
                               mcp_server.py, and orchestrator.py (defense-in-depth: the
                               same input is validated at the UI layer AND again inside
                               OrchestratingAgent.invoke() for any caller that bypasses
                               main.py's form, e.g. a future script or MCP tool)
```

`fit_agent.py` imports from `analysis_agent.py` (budget/PDF-financials formatting
helpers) and `search_agent.py` (case-study RAG, sponsorship DB, credibility scoring) —
the one non-obvious edge is that `fit_agent.py` also needs `orchestrator.py`'s
persistence functions (`get_exact_previous_analysis`, `save_analysis`, ...), but imports
them **locally inside `evaluate_fit()`**, not at module level, specifically to avoid a
`orchestrator → fit_agent → orchestrator` circular import (orchestrator imports
`evaluate_fit` at module level to build the graph; it doesn't need fit_agent's own
imports to resolve orchestrator first).

## State schema

`SponsorMatchState` (`src/tools.py`, a `TypedDict`) — every field, which stage writes it,
and its shape:

| Field | Written by | Type |
|---|---|---|
| `club_profile`, `company_name`, `user_id`, `selected_model`, `language` | caller (input) | — |
| `research_findings` | `research_company` | markdown `str` |
| `research_quality` | `research_company` | dict (see QUALITY_METRICS.md) |
| `competitor_analysis` | `analyze_financials`, adjusted again in `evaluate_fit` | dict |
| `size_compatibility` | `analyze_financials`, adjusted again in `evaluate_fit` | dict |
| `pdf_financials` | `analyze_financials` | dict (raw metrics, for the UI budget section) |
| `financial_data` | `analyze_financials` | dict (quality summary, see QUALITY_METRICS.md) |
| `fit_score`, `fit_reasoning` | `evaluate_fit` | `float`, `str` |
| `fit_agent_factors` | `evaluate_fit` | dict, `{}` on a score-cache hit |
| `used_case_studies`, `used_sponsorship_matches` | `evaluate_fit` | `list[dict]` |
| `budget_estimate` | `evaluate_fit` | `str` |
| `analysis_id` | `evaluate_fit` (SQLite row id) | `int` |
| `learning_applied`, `is_uncertain`, `agent_confidence`, `hitl_resolved_count` | `evaluate_fit` | — |
| `outreach_draft` | `draft_outreach` (only if routed there) | `str` |
| `rejection_reason` | `explain_rejection` (only if routed there) | `str` |
| `token_usage` | every LLM-calling node, appended | `Annotated[list, operator.add]` — LangGraph's reducer merges this automatically instead of overwriting |
| `performance_metrics` | `OrchestratingAgent.invoke()` itself, after the stream completes | dict (see README_AGENTS.md) |

`token_usage` is the only field with a non-default reducer — every other field is a plain
overwrite (last writer wins), which is why nodes that adjust an earlier node's dict
(`competitor_analysis`, `size_compatibility` getting `score_before/after_adjustment`
fields added in `evaluate_fit`) explicitly copy-then-mutate rather than relying on any
merge behavior.

## Request sequence

What actually happens for one `OrchestratingAgent.invoke(state)` call, in order:

1. **Input validation** (`main.py`, before the graph even runs): company-name pattern
   checks (`security_validator.validate_input`), "looks like a question" heuristic,
   session-scoped rate limit, IP-scoped rate limit (`security_validator.rate_limit_check`).
2. **Defense-in-depth re-validation** (`OrchestratingAgent.invoke()`): re-runs
   `validate_input()` so any caller that bypasses `main.py`'s form (a script, an MCP
   tool) still gets the same protection.
3. **`research_company`**: 5 parallel Tavily queries → dedupe by URL → credibility score
   → one LLM synthesis call → `research_findings` + `research_quality`.
4. **`analyze_financials`**: competitor-portfolio search (optional plugin) + company-size
   estimate (optional plugin) + PDF financial extraction (optional plugin — find PDFs,
   check `PDFCache`, download via `ErrorHandler.retry_on_failure` on a cache miss, parse,
   regex-extract, cache the result) → `competitor_analysis` + `size_compatibility` +
   `pdf_financials` + `financial_data`.
5. **`evaluate_fit`**: RAG lookups (case studies, sponsorship DB) → score-cache check
   (`get_exact_previous_analysis`) → if no cache hit, one LLM call scoring 8 factors →
   weighted combination → feedback/portfolio/HITL/size adjustments → `save_analysis()` to
   SQLite → `fit_score` + `fit_reasoning` + `fit_agent_factors`.
6. **`route_by_fit`**: conditional edge on `fit_score >= 0.6`.
7. **`draft_outreach`** or **`explain_rejection`**: one more LLM call (outreach) or none
   (rejection, pure string formatting).
8. **Back in `OrchestratingAgent.invoke()`**: accumulate `performance_metrics` from the
   per-node timing collected during the stream, log the pipeline result, return the full
   accumulated state to the caller (`main.py`, which renders it into UI cards).

## Key architectural decisions

- **Functions + module-level singletons, not agent classes.** Every "agent" is a plain
  function (`research_company`, `analyze_financials`, `evaluate_fit`) taking and
  returning a dict — there's no `SearchAgent`/`AnalysisAgent`/`FitAgent` class hierarchy.
  Cross-cutting concerns added in Phase 3 (`AgentLogger`, `PDFCache`, `ErrorHandler`,
  `PerformanceMonitor`) are implemented as classes but instantiated **once per module**
  at import time (e.g. `_logger = AgentLogger("analysis_agent")`), the same pattern
  already established by `src/tools.py`'s `@lru_cache`-wrapped LLM factory — this fits a
  LangGraph node function's calling convention (`state -> dict`) better than threading a
  `self` through every node.
- **LangGraph over a hand-rolled loop.** `.stream(stream_mode="updates")` gives per-node
  deltas for free, which both the logging and performance-timing layers depend on;
  building that manually on top of plain function calls would have meant re-implementing
  what LangGraph already provides.
- **Score caching over always-fresh scoring.** Identical company+club requests reuse the
  prior score rather than re-running the LLM evaluation — deliberately prioritizing
  consistency (the same input always gives the same answer) over the possibility that a
  fresh call might produce a marginally different score. This is also why
  `fit_agent_factors` is empty on a cache hit: there's no fresh LLM call to have
  per-factor detail from.
- **Best-effort, not fail-fast, for anything hitting the open web.** PDF discovery/
  download/parsing (`analyze_financials`) is the one part of the pipeline reaching
  arbitrary third-party URLs outside Tavily's own API — it's built to degrade gracefully
  at every step (`ErrorHandler` retries + skips, `{"found": False}` is a normal return
  value, not an exception) rather than let one broken PDF fail the whole analysis.
