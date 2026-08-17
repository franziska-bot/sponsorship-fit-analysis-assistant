# Quality Metrics

Every pipeline stage reports its own confidence alongside its actual output. This
matters because a **low fit score** and a **low-confidence fit score** are different
situations: the first says "this sponsor probably isn't a good match," the second says
"we don't have enough to say either way — verify manually before acting on this."
This document explains each metric, where it comes from, and how to read it.

## Data quality tiers

Both `research_quality.data_quality` and `financial_data.data_quality`
(`src/search_agent.py`'s `_classify_research_quality()` and
`src/analysis_agent.py`'s `_classify_financial_quality()`) use the same three tiers:

| Tier | Threshold | Meaning |
|---|---|---|
| **High** | ≥ 80% | Multiple sources agree, recent data, official/high-credibility sources |
| **Medium** | 60–80% | Some metrics missing, older data, or a thinner source set |
| **Low** | < 60% | Very few sources, contradictory data, or almost nothing found |

## SearchAgent: `research_quality`

| Field | Range | What it means |
|---|---|---|
| `sources_found` | integer | Distinct web sources (deduplicated by URL) that fed the research summary |
| `average_credibility` | 0.0–1.0 | Mean per-source credibility score — `.gov`/`.edu`/major outlets (Reuters, Bloomberg, FT, ...) score 0.9, Wikipedia 0.8, social/forum platforms (LinkedIn, Reddit, Twitter/X, ...) score 0.3, everything else 0.5 |
| `sentiment_score` | -1.0 to 1.0 | Direction (positive/negative/neutral) × the LLM's own stated confidence in that sentiment reading; 0.0 for neutral or unparseable |
| `timeline_completeness` | 0.0–1.0 | Dated sponsorship-history entries found, relative to a target of 3 — 3+ entries is treated as "complete" |
| `data_quality` | high/medium/low | `0.5 × min(sources_found/10, 1.0) + 0.35 × average_credibility + 0.15 × timeline_completeness`, against the tiers above |

**How to read it**: a low `sources_found` with high `average_credibility` (a handful of
official sources) is more trustworthy than the reverse (many low-credibility social
posts). `sentiment_score` and the fit score are independent — a company can have
strongly positive public sentiment and still be a poor sponsorship fit for a specific
club (wrong audience, wrong budget tier), and vice versa.

## AnalysisAgent: `financial_data`

| Field | Range | What it means |
|---|---|---|
| `pdfs_found` | integer | Report/filing PDF URLs Tavily surfaced across 4 query variants |
| `pdfs_successfully_parsed` | integer | Of those, how many yielded extractable text *and* at least one recognized financial metric |
| `metrics_extracted` | list | Which of revenue/EBITDA/profit/cash/marketing-spend were actually found |
| `metrics_missing` | list | The complement — metrics this run does *not* have data for |
| `extraction_confidence` | 0.0–1.0 | `0.4 × agreement + 0.35 × avg_credibility + 0.25 × coverage` — agreement rewards multiple PDFs confirming the same number, coverage rewards having found more of the 5 target metrics |
| `data_freshness` | year string or `"unknown"` | The most recent-looking 4-digit year mentioned anywhere in the parsed PDF text, capped at the current year |
| `data_quality` | high/medium/low | `extraction_confidence` against the tiers above |

**Important caveat**: this pipeline is genuinely best-effort. Most companies have no
directly Tavily-findable report PDF at all — `pdfs_found: 0` is the expected normal case
for most inputs, not a failure. Even when PDFs are found and parsed, the regex-based
extraction occasionally misreads a plausible-looking but wrong number (a year mentioned
near a dollar sign, a number from an unrelated table). **Treat every PDF-derived
financial number as a lead to verify, not a citable fact** — the UI shows an explicit
disclaimer alongside these figures for the same reason. `data_freshness` is a rough
signal only: it's the newest year mentioned *anywhere* in the document, not necessarily
the report's actual "as of" date.

`pdfs_successfully_parsed` being lower than `pdfs_found` is normal and not itself a
quality problem — some of Tavily's PDF-looking search hits turn out to be scanned images
(no extractable text), academic papers that happen to match the search terms, or 403/404
on download. `ErrorHandler` retries transient failures 3x before giving up on a URL (see
[README_AGENTS.md](README_AGENTS.md)); the ones it does give up on don't block the rest.

## FitAgent: `fit_agent_factors`

Eight independent 0.0–1.0 confidence scores, one per weighted factor in the final score
(`src/fit_agent.py`'s `_FACTOR_WEIGHTS`):

| Factor | Weight | What it captures |
|---|---|---|
| Values fit | 20% | Alignment between the company's stated brand values and the club's |
| Audience fit | 20% | Overlap between the company's target audience and the club's fanbase |
| Sport relevance | 15% | How naturally this company's category fits this sport |
| Historical precedent | 15% | Grounded in case studies / sponsorship-database history, not just LLM intuition |
| Budget fit | 10% | Explicitly asked to account for sensitivity to budget assumptions |
| Strategic potential | 10% | Longer-term strategic upside beyond an immediate deal |
| Brand safety | 5% | **Inverted** — 1.0 means low risk, 0.0 means high risk |
| Market timing | 5% | Whether now is a sensible moment for this partnership |

**These are not separately verified** — they're the LLM's own stated per-factor
confidence from the same call that produces the final score, exposed for transparency
rather than re-derived independently. A factor score of 0.9 doesn't mean "90% likely to
be correct," it means "the model was quite confident in this particular sub-judgment."

**Empty on a score-cache hit**: if this exact company+club pair was already analyzed,
`evaluate_fit()` reuses the cached score instead of making a new LLM call — so there are
no fresh factor scores to show. `fit_agent_factors` is `{}` in that case, and the UI
shows an explanatory note instead of empty progress bars. This is intentional: a cached
score stays numerically stable across repeat requests (see README_AGENTS.md's note on
score caching), which would be undermined if it also carried "fresh" per-factor detail
that wasn't actually recomputed.

## When to trust vs. when to do manual research

**Trust the output more when:**
- `research_quality.data_quality` is `high` *and* `financial_data.data_quality` is at
  least `medium` — the recommendation is grounded in enough independent, credible signal
- `fit_agent_factors` shows high confidence specifically on the factors that matter most
  for your decision (e.g. values fit and audience fit, the two highest-weighted factors)
- `is_uncertain` is `False` (final score outside the 0.45–0.55 band) — see
  `agent_confidence` for a combined signal that also accounts for prior human-reviewed
  decisions about this same company

**Do manual research before acting when:**
- `research_quality.sources_found` is very low (a handful of low-credibility sources) —
  the summary may be thin regardless of what the fit score says
- `financial_data.pdfs_found` is 0 or `metrics_missing` covers most of the five target
  metrics — budget-fit reasoning has little real financial grounding behind it
- `is_uncertain` is `True` — this is exactly what the Human-in-the-Loop review flow in
  the UI is for; use it rather than taking the raw score at face value
- Any `financial_data` numbers look surprising — cross-check against the actual filing
  before including them in anything external-facing (see the caveat above)
