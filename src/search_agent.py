"""Search specialist: web research + internal knowledge-base retrieval
(case-study RAG, external sponsorship database).
"""

import json as _json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.parse import urlparse

from langsmith import traceable

from src.logger import AgentLogger
from src.tools import (
    SponsorMatchState,
    _get_llm,
    _track_tokens,
    get_search_tool,
    is_plugin_enabled,
)

_logger = AgentLogger("search_agent")

with open("data/case_studies.json", "r", encoding="utf-8") as _f:
    _CASE_STUDIES = _json.load(_f)


def search_case_studies(company_name: str, sport: str, top_k: int = 3) -> list[dict]:
    """Keyword-basierte RAG-Suche über die Case-Study-Wissensbasis.

    Case-Studies, deren Sportart nicht zum Verein passt, werden NIE
    zurückgegeben (auch nicht als Fallback) – eine andere Sportart "gehört
    nicht zum Verein", egal wie gut die Firma sonst passt.

    Zwei Prioritätsstufen, beide auf die Sportart des Vereins beschränkt:
    1. "company_sport": exakt gleiche Firma UND gleiche Sportart (bester Fit).
    2. "sport": andere Firma, aber gleiche Sportart – reine Sportart-Analogie.
    """
    company_terms = [t for t in company_name.lower().split() if t]
    sport_lower = sport.lower()

    exact_matches = []
    sport_only_matches = []

    for case in _CASE_STUDIES:
        if case["sport"].lower() != sport_lower:
            continue  # falsche Sportart -> gehört nicht zum Verein, nie anzeigen

        company_score = sum(1 for term in company_terms if term in case["company"].lower())
        if company_score > 0:
            exact_matches.append((company_score, case))
        else:
            sport_only_matches.append(case)

    if exact_matches:
        exact_matches.sort(key=lambda item: item[0], reverse=True)
        return [{**case, "match_type": "company_sport"} for _, case in exact_matches[:top_k]]

    return [{**case, "match_type": "sport"} for case in sport_only_matches[:top_k]]


# --- Externe Sponsorship-Datenbank (data/sponsorship_database.json) ---

SPONSORSHIP_DB_PATH = "data/sponsorship_database.json"


@lru_cache(maxsize=1)
def _load_sponsorship_db() -> list[dict]:
    """Lädt die externe Sponsorship-Datenbank einmalig (lru_cache memoiziert
    den einzigen no-arg-Aufruf, die Datei wird also nur einmal gelesen).

    Graceful Fallback: gibt eine leere Liste zurück, falls die Datei fehlt
    oder kein gültiges JSON enthält, statt die App abstürzen zu lassen.
    """
    try:
        with open(SPONSORSHIP_DB_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return []


@lru_cache(maxsize=256)
def query_sponsorship_db(company_name: str, sport: str) -> list[dict]:
    """Sucht in der externen Sponsorship-Datenbank nach Fällen zur Firma
    (Teilstring, case-insensitive) mit exakt passender Sportart.

    Ergebnis wird pro (company_name, sport)-Kombination gecacht.
    """
    company_lower = company_name.lower()
    sport_lower = sport.lower()
    return [
        case
        for case in _load_sponsorship_db()
        if company_lower in case["company"].lower() and case["sport"].lower() == sport_lower
    ]


def _build_sponsorship_context(matches: list[dict], language: str) -> str:
    """Baut den Kontext aus der externen Sponsorship-DB für den Bewertungs-Prompt."""
    if not matches:
        return {
            "de": "Keine historischen Daten in der externen Sponsorship-Datenbank gefunden.",
            "en": "No historical data found in the external sponsorship database.",
            "fr": "Aucune donnée historique trouvée dans la base de sponsoring externe.",
        }[language]

    lines = [
        f"- {m['company']} → {m['athlete_or_team']} ({m['start_year']}-{m['end_year']}): "
        f"{m['success_metric']} (brand fit: {m['brand_fit']})"
        for m in matches
    ]
    header = {
        "de": "Historische Sponsorship-Erfolge dieser Firma (externe Datenbank, fiktive Beispieldaten):",
        "en": "Historical sponsorship track record for this company (external database, fictional sample data):",
        "fr": "Historique de sponsoring de cette entreprise (base de données externe, données fictives) :",
    }[language]
    return header + "\n" + "\n".join(lines)


# --- Multi-Query Research (Phase 1): 5 parallele Tavily-Suchen statt einer,
# credibility-gewichtet, per URL dedupliziert, in EINEM LLM-Call zu Summary +
# Sentiment + Timeline + Confidence synthetisiert (ein Call statt drei, um die
# Token-Kosten nicht zu verdreifachen – siehe Token-Verbrauch-Karte in main.py).

_RESEARCH_QUERY_TEMPLATES = {
    "sponsorship": "{company} sponsorship deals partnerships sports",
    "budget": "{company} marketing budget sponsorship investment spending",
    "values": "{company} brand values mission corporate culture",
    "audience": "{company} target audience customer demographics",
    "timeline": "{company} sponsorship history timeline past partnerships",
}

_HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com", "bloomberg.com", "forbes.com", "ft.com", "wsj.com",
    "bbc.com", "cnbc.com",
}
_LOW_CREDIBILITY_DOMAINS = {
    "linkedin.com", "reddit.com", "medium.com", "quora.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com",
}


def _score_source_credibility(url: str) -> float:
    """Heuristisches Credibility-Scoring pro Quelle (0.0-1.0), rein
    domain-basiert – bewusst kein ML-Modell: Tavilys eigener 'score' misst nur
    Relevanz zur Suchanfrage, nicht Vertrauenswürdigkeit der Quelle (ein
    LinkedIn-Post und ein Reuters-Artikel können beide hoch relevant sein)."""
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    if domain.endswith((".gov", ".edu")):
        return 0.9
    if domain in _HIGH_CREDIBILITY_DOMAINS:
        return 0.9
    if domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
        return 0.8
    if domain in _LOW_CREDIBILITY_DOMAINS:
        return 0.3
    return 0.5


def _deduplicate_findings(results: list[dict]) -> list[dict]:
    """Merged Ergebnisse über die 5 Queries hinweg per URL (dieselbe Quelle
    taucht bei unterschiedlich formulierten Queries oft mehrfach auf) – behält
    pro URL das Vorkommen mit dem höchsten Tavily-Relevanz-Score. Wirklich
    widersprüchliche Quellen (z.B. zwei verschiedene Budgetzahlen) werden hier
    NICHT algorithmisch aufgelöst, sondern der LLM-Synthese überlassen – darin
    sind LLMs tatsächlich gut, im Gegensatz zu fragilen Heuristiken."""
    best_by_url: dict[str, dict] = {}
    for result in results:
        url = result.get("url", "")
        if not url:
            continue
        existing = best_by_url.get(url)
        if existing is None or result.get("score", 0) > existing.get("score", 0):
            best_by_url[url] = result
    return list(best_by_url.values())


def _run_research_queries(company_name: str) -> list[dict]:
    """Führt die 5 Research-Queries parallel aus (ThreadPoolExecutor, da die
    Tavily-Calls I/O-bound sind – kein Grund, den Graph-Node dafür auf async
    umzustellen) und gibt die deduplizierten, credibility-gescorten Treffer
    absteigend nach Credibility sortiert zurück."""
    search_tool = get_search_tool()
    queries = [
        (key, template.format(company=company_name)) for key, template in _RESEARCH_QUERY_TEMPLATES.items()
    ]

    def run_query(labeled_query: tuple[str, str]) -> dict:
        key, query = labeled_query
        _logger.log_agent_step(f"research_query:{key}", query=query)
        try:
            return search_tool.invoke(query)
        except Exception as exc:
            _logger.log_error(f"research_query:{key}", exc, query=query)
            return {"results": []}

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        raw_responses = list(executor.map(run_query, queries))

    all_results = [result for response in raw_responses for result in response.get("results", [])]
    deduped = _deduplicate_findings(all_results)
    for result in deduped:
        result["credibility"] = _score_source_credibility(result.get("url", ""))
    deduped.sort(key=lambda r: r["credibility"], reverse=True)
    _logger.log_agent_step("research_queries_deduped", raw_count=len(all_results), deduped_count=len(deduped))
    return deduped


def _build_sources_context(sources: list[dict], limit: int = 12) -> str:
    """Baut den Quellen-Kontext für den Synthese-Prompt, sortiert nach
    Credibility (höchste zuerst) mit Credibility-Tag pro Quelle, damit das LLM
    seine CONFIDENCE-Einschätzung tatsächlich darauf stützen kann statt sie zu
    erfinden. Auf die Top `limit` Quellen begrenzt, um den Prompt kompakt zu
    halten (25 Rohtreffer aus 5 Queries wären unnötig viel Kontext)."""
    lines = [
        f"- [credibility={r['credibility']:.1f}] {r.get('title', '')}: {r.get('content', '')[:400]}"
        for r in sources[:limit]
    ]
    return "\n".join(lines)


_SENTIMENT_LABELS = {
    "de": {"positive": "positiv", "negative": "negativ", "neutral": "neutral"},
    "en": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    "fr": {"positive": "positif", "negative": "négatif", "neutral": "neutre"},
}

_RESEARCH_SECTION_HEADERS = {
    "de": {"sentiment": "Sponsoring-Sentiment", "timeline": "Sponsoring-Historie"},
    "en": {"sentiment": "Sponsorship sentiment", "timeline": "Sponsorship history"},
    "fr": {"sentiment": "Sentiment de sponsoring", "timeline": "Historique de sponsoring"},
}


def _analyze_sentiment(raw_line: str, language: str) -> tuple[str, float | None]:
    """Parst die SENTIMENT-Zeile der LLM-Antwort (Format
    '<label>|<confidence>|<reason>') zu einem formatierten Markdown-Absatz.
    Gibt ("", None) zurück, wenn die Zeile fehlt/unparsebar ist, statt einen
    falschen Default vorzugeben.

    Liefert zusätzlich einen numerischen sentiment_score (-1.0..1.0, für
    research_quality/Phase 3e): Richtung (positive=+, negative=-,
    neutral=0) skaliert mit der vom LLM angegebenen Konfidenz."""
    label, _, rest = raw_line.partition("|")
    confidence_str, _, reason = rest.partition("|")
    label = label.strip().lower()
    labels = _SENTIMENT_LABELS[language]
    if label not in labels:
        return "", None
    try:
        confidence = max(0.0, min(1.0, float(confidence_str.strip())))
    except ValueError:
        confidence = 0.5
    header = _RESEARCH_SECTION_HEADERS[language]["sentiment"]
    text = f"**{header}:** {labels[label]} ({confidence:.0%}) — {reason.strip()}"
    direction = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[label]
    return text, round(direction * confidence, 2)


# Ziel-Anzahl an Timeline-Einträgen, ab der eine Recherche als "vollständig"
# gilt (timeline_completeness=1.0) – 3 datierte Sponsoring-Meilensteine gelten
# hier als guter Überblick, mehr bringt für die Qualitätsanzeige wenig Zusatzwert.
_TIMELINE_TARGET_ENTRIES = 3


def _extract_timeline(raw_line: str, language: str) -> tuple[str, float]:
    """Parst die TIMELINE-Zeile (Einträge getrennt durch ';;', je
    '<Zeitraum>|<Beschreibung>') zu einer Markdown-Liste. Leerer String, wenn
    keine Einträge geliefert wurden (z.B. keine bekannte Sponsoring-Historie).

    Liefert zusätzlich timeline_completeness (0.0-1.0, für
    research_quality/Phase 3e): Anzahl gefundener Einträge relativ zu
    _TIMELINE_TARGET_ENTRIES, gedeckelt bei 1.0."""
    entries = []
    for raw_entry in raw_line.split(";;"):
        period, _, description = raw_entry.partition("|")
        period, description = period.strip(), description.strip()
        if period and description:
            entries.append(f"- **{period}:** {description}")
    if not entries:
        return "", 0.0
    header = _RESEARCH_SECTION_HEADERS[language]["timeline"]
    completeness = min(len(entries) / _TIMELINE_TARGET_ENTRIES, 1.0)
    return f"**{header}:**\n" + "\n".join(entries), round(completeness, 2)


def _classify_research_quality(sources_found: int, average_credibility: float, timeline_completeness: float) -> str:
    """Heuristische Gesamt-Einordnung ('high'/'medium'/'low') für die
    Data-Quality-Anzeige (Phase 3e/QUALITY_METRICS.md): gewichtet aus
    Quellenanzahl (gedeckelt bei 10, mehr bringt kaum Zusatzvertrauen),
    durchschnittlicher Quellen-Credibility und Timeline-Vollständigkeit –
    dieselbe Gewichtungs-Philosophie wie _calculate_confidence in
    analysis_agent.py (mehrere Signale kombiniert statt einem einzelnen
    Schwellenwert). Tier-Grenzen (80%/60%) siehe QUALITY_METRICS.md."""
    score = 0.5 * min(sources_found / 10, 1.0) + 0.35 * average_credibility + 0.15 * timeline_completeness
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _format_confidence_section(confidence: float, source_count: int, language: str) -> str:
    """Baut die abschließende Research-Konfidenz-Zeile, inkl. Anzahl der
    tatsächlich ausgewerteten (deduplizierten) Quellen als Transparenz-Signal."""
    templates = {
        "de": f"**Research-Konfidenz:** {confidence:.0%} ({source_count} Quellen)",
        "en": f"**Research confidence:** {confidence:.0%} ({source_count} sources)",
        "fr": f"**Confiance de la recherche :** {confidence:.0%} ({source_count} sources)",
    }
    return templates[language]


@traceable(run_type="chain", name="research_company")
def research_company(state: SponsorMatchState) -> dict:
    """Recherchiert die angegebene Firma über 5 parallele Tavily-Queries
    (Sponsoring, Budget, Werte, Zielgruppe, Historie), gewichtet die Quellen
    nach Glaubwürdigkeit, dedupliziert sie per URL und synthetisiert Summary,
    Sentiment, Timeline und Research-Konfidenz in einem einzigen LLM-Call."""
    company_name = state["company_name"]
    llm = _get_llm(state["selected_model"])
    language = state["language"]
    _logger.log_agent_start("research_company", company=company_name)

    # Plugin-Gating: web_search ist "required", d.h. im UI nicht abschaltbar –
    # dieser Zweig ist defensive Absicherung, falls die Config trotzdem einmal
    # enabled=false enthält. Früher Return statt eines LLM-Calls über einen
    # "Plugin deaktiviert"-Platzhaltertext, der ohnehin nichts zu sagen hätte.
    if not is_plugin_enabled("web_search"):
        fallback = {
            "de": "(Web-Search-Plugin deaktiviert – keine externen Rechercheergebnisse verfügbar.)",
            "en": "(Web search plugin disabled – no external research results available.)",
            "fr": "(Plugin de recherche web désactivé – aucun résultat de recherche externe disponible.)",
        }[language]
        _logger.log_agent_result("research_company", skipped=True, reason="web_search_disabled")
        empty_quality = {
            "sources_found": 0,
            "average_credibility": 0.0,
            "sentiment_score": None,
            "timeline_completeness": 0.0,
            "data_quality": "low",
        }
        return {"research_findings": fallback, "research_quality": empty_quality, "token_usage": []}

    sources = _run_research_queries(company_name)
    sources_context = _build_sources_context(sources)

    if language == "en":
        prompt = (
            f"Based on the following credibility-tagged web search snippets about "
            f"'{company_name}' (higher credibility = more trustworthy source), "
            f"respond in exactly this format, nothing before or after (keep labels "
            f"in English, write content in English):\n"
            f"SUMMARY: <4-5 sentence synthesis covering industry, past sponsorship "
            f"activities, target audience, and brand values>\n"
            f"SENTIMENT: <positive|negative|neutral>|<0.0-1.0 confidence>|<short reason "
            f"about sponsorship/brand perception, max 15 words>\n"
            f"TIMELINE: <period>|<description>;;<period>|<description>;;... (known "
            f"sponsorship history with approximate dates, leave empty if none found)\n"
            f"CONFIDENCE: <0.0-1.0, your overall confidence in this research given the "
            f"number and credibility of sources>\n\n"
            f"Sources:\n{sources_context}"
        )
    elif language == "fr":
        prompt = (
            f"À partir des extraits de recherche web suivants sur '{company_name}' "
            f"(étiquetés par crédibilité — plus élevé = source plus fiable), répondez "
            f"exactement dans ce format, rien avant ni après (gardez les étiquettes en "
            f"anglais, rédigez le contenu en français) :\n"
            f"SUMMARY: <synthèse de 4-5 phrases couvrant le secteur, les activités de "
            f"parrainage passées, le public cible et les valeurs de marque>\n"
            f"SENTIMENT: <positive|negative|neutral>|<confiance 0.0-1.0>|<raison courte "
            f"sur la perception de la marque/du sponsoring, max 15 mots>\n"
            f"TIMELINE: <période>|<description>;;<période>|<description>;;... (historique "
            f"de sponsoring connu avec dates approximatives, laissez vide si aucun trouvé)\n"
            f"CONFIDENCE: <0.0-1.0, votre confiance globale dans cette recherche selon le "
            f"nombre et la crédibilité des sources>\n\n"
            f"Sources :\n{sources_context}"
        )
    else:
        prompt = (
            f"Basierend auf den folgenden, nach Glaubwürdigkeit markierten "
            f"Web-Suchausschnitten zu '{company_name}' (höher = vertrauenswürdigere "
            f"Quelle), antworte in genau diesem Format, nichts davor oder danach "
            f"(behalte die Labels auf Englisch, schreibe den Inhalt auf Deutsch):\n"
            f"SUMMARY: <4-5 Sätze zu Branche, bisherigen Sponsoring-Aktivitäten, "
            f"Zielgruppe und Markenwerten>\n"
            f"SENTIMENT: <positive|negative|neutral>|<Konfidenz 0.0-1.0>|<kurze "
            f"Begründung zur Sponsoring-/Markenwahrnehmung, max. 15 Wörter>\n"
            f"TIMELINE: <Zeitraum>|<Beschreibung>;;<Zeitraum>|<Beschreibung>;;... "
            f"(bekannte Sponsoring-Historie mit ungefähren Daten, leer lassen falls "
            f"keine gefunden)\n"
            f"CONFIDENCE: <0.0-1.0, deine Gesamt-Konfidenz in diese Recherche basierend "
            f"auf Anzahl und Glaubwürdigkeit der Quellen>\n\n"
            f"Quellen:\n{sources_context}"
        )

    response = llm.invoke(prompt)
    content = response.content.strip()
    token_usage = [_track_tokens("research_company", response)]

    summary = ""
    sentiment_raw = ""
    timeline_raw = ""
    confidence_raw = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "", 1).strip()
        elif line.startswith("SENTIMENT:"):
            sentiment_raw = line.replace("SENTIMENT:", "", 1).strip()
        elif line.startswith("TIMELINE:"):
            timeline_raw = line.replace("TIMELINE:", "", 1).strip()
        elif line.startswith("CONFIDENCE:"):
            confidence_raw = line.replace("CONFIDENCE:", "", 1).strip()

    findings = summary or content

    sentiment_section, sentiment_score = _analyze_sentiment(sentiment_raw, language)
    if sentiment_section:
        findings += "\n\n" + sentiment_section

    timeline_section, timeline_completeness = _extract_timeline(timeline_raw, language)
    if timeline_section:
        findings += "\n\n" + timeline_section

    research_confidence = None
    try:
        research_confidence = max(0.0, min(1.0, float(confidence_raw)))
        findings += "\n\n" + _format_confidence_section(research_confidence, len(sources), language)
    except ValueError:
        pass

    average_credibility = round(sum(r["credibility"] for r in sources) / len(sources), 2) if sources else 0.0
    research_quality = {
        "sources_found": len(sources),
        "average_credibility": average_credibility,
        # Fallback 0.0 (neutral) statt None, wenn die SENTIMENT-Zeile
        # unparsebar war – dieselbe neutrale-Default-Philosophie wie
        # _analyze_sentiment's confidence=0.5-Fallback oben.
        "sentiment_score": sentiment_score if sentiment_score is not None else 0.0,
        "timeline_completeness": timeline_completeness,
        "data_quality": _classify_research_quality(len(sources), average_credibility, timeline_completeness),
    }

    _logger.log_agent_result(
        "research_company", sources=len(sources), confidence=research_confidence,
        data_quality=research_quality["data_quality"],
    )
    return {
        "research_findings": findings,
        "research_quality": research_quality,
        "token_usage": token_usage,
    }
