# Deployment

## Requirements

- Python 3.12 (pinned in `pyproject.toml`: `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — this project is
  managed with `uv`/`pyproject.toml` + `uv.lock`, not `pip`/`requirements.txt`

```bash
uv sync
```

## Environment setup

Create a `.env` file in the project root:

```bash
TAVILY_API_KEY=...               # required — web search
OPENROUTER_API_KEY=...           # required — LLM access via OpenRouter
LANGSMITH_API_KEY=...            # optional — enables tracing/trace links
LANGSMITH_PROJECT=sponsor-match  # optional, default: sponsor-match
HEALTH_CHECK_PORT=8600           # optional, default: 8600 (see Monitoring below)
HEALTH_CHECK_HOST=127.0.0.1      # optional, default: 127.0.0.1
```

Without `TAVILY_API_KEY` / `OPENROUTER_API_KEY` the app will raise on the first
search/LLM call — there is no offline/mock mode.

## Running

```bash
uv run streamlit run main.py
```

The app runs at `http://localhost:8501`. Create an account via "Register" on first run.

## Monitoring

**Logs** — `logs/agent_trace.log` (created automatically): full DEBUG-level trace of
every agent step (search queries dispatched, PDF cache hits/misses, retries, errors).
Console output during `streamlit run` is INFO-level only (start/result/error per stage),
so tail the file for full detail:

```bash
tail -f logs/agent_trace.log
```

**Health endpoint (optional)** — Streamlit has no Flask-style `@app.route`, so a small
standalone process (`src/health_server.py`, stdlib `http.server` only, no extra
dependency) serves a JSON health payload as a *second* process alongside Streamlit:

```bash
uv run python -m src.health_server
curl http://127.0.0.1:8600/health
```

```json
{
  "status": "ok",
  "cache_size": {"files": 32, "bytes": 57731808},
  "latest_analysis": {"company_name": "...", "club_name": "...", "timestamp": "...", "fit_score": 0.81},
  "log_size": 84823
}
```

Point a load balancer / container health-check probe at this instead of Streamlit's own
port if you need liveness plus a signal that the pipeline is actually producing results
(not just that the process is up).

## Performance tuning

- **PDF download retries** (`src/error_handler.py`): `ErrorHandler(max_retries=3, ...)` in
  `src/analysis_agent.py`'s module-level `_error_handler`. Backoff is `2 ** (attempt - 1)`
  seconds (1s/2s/4s for the default 3 retries) — raising `max_retries` adds one more
  doubling each time, so pick a number with the worst-case wait in mind (3 retries ≈ 7s
  worst case per failing URL, run in parallel across URLs via `ThreadPoolExecutor`, not
  serially).
- **PDF cache TTL** (`src/pdf_cache.py`): `_PDF_TTL_SECONDS` (raw PDF bytes, default 30
  days) and `_EXTRACTION_TTL_SECONDS` (extracted metrics, default 60 days) are module
  constants — lower them if you expect companies to republish reports frequently, raise
  them to cut Tavily/download load further at the cost of staler `data_freshness`.
- **Rate limits** (`src/security_validator.py` + `main.py`): two independent layers —
  `main.py`'s `RATE_LIMIT_SHORT`/`RATE_LIMIT_LONG` (session-scoped, `st.session_state`)
  and `security_validator.py`'s `_IP_RATE_LIMIT_SHORT`/`_IP_RATE_LIMIT_LONG` (IP-scoped,
  in-memory, process-lifetime). Both default to 10/short-window, 100/long-window; adjust
  independently depending on whether you're more worried about one browser tab hammering
  the app or one IP doing so across many tabs/sessions.
- **PDF cache is in-memory-adjacent but file-based** (`data/pdf_cache/`) — safe across
  Streamlit reruns and process restarts. The IP rate limiter is purely in-memory
  (`security_validator.py`'s module-level dict) — it resets on restart and does **not**
  share state across multiple worker processes/instances behind a load balancer. If you
  scale to multiple instances, swap that dict for Redis (the module docstring notes this
  explicitly as the intended extension point).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'TAVILY_API_KEY'` / `'OPENROUTER_API_KEY'` on startup | Missing `.env` | Create `.env` per the Environment setup section above |
| Every analysis returns "no financial data found" | Normal — most companies have no directly Tavily-findable report PDF (best-effort by design, see README_AGENTS.md) | Not a bug; check `logs/agent_trace.log` for `pdf_urls_found`/`pdf_no_metrics_found` to confirm URLs were tried |
| PDF downloads consistently fail with `403 Forbidden` for one domain | That server blocks the request's User-Agent/IP (not this app's error) | Check `logs/agent_trace.log` for the `download_pdf:<url>` ERROR line; `ErrorHandler` already retried 3x and gracefully skipped it — other PDFs in the batch are unaffected |
| `sqlite3.OperationalError: database is locked` | A cached `sqlite3.Connection` (via `st.cache_resource`) survived a Streamlit rerun that was interrupted mid-write | Restart the app; `src/orchestrator.py`'s `_get_connection()` opens a fresh short-lived connection per call unless `configure_shared_connection()` was explicitly wired up — don't reintroduce a cached long-lived connection without care (this bit the project once already) |
| 2nd run of the same company isn't noticeably faster | PDF cache TTL expired, or Tavily found different/no report URLs than the first run (search results aren't guaranteed stable) | Check `logs/agent_trace.log` for `cache_hit`/`pdf_downloaded` lines to see whether the cache was actually consulted |
| Rate-limit warning appears immediately on a fresh session | IP rate limiter (`security_validator.py`) is shared across *all* Streamlit sessions in the same process — a shared dev machine or NAT'd office network can trip it for everyone at once | Expected trade-off of IP-based limiting (see README_AGENTS.md); increase `_IP_RATE_LIMIT_SHORT`/`_IP_RATE_LIMIT_LONG` if this happens legitimately often |
| `ModuleNotFoundError: No module named 'src'` | Running a script directly instead of via `uv run` from the project root | Always run from the project root: `uv run streamlit run main.py` / `uv run python -m src.health_server` |
