"""Final fit evaluation specialist: combines research, financial analysis,
case-study RAG, and long-term memory into a fit score + reasoning, then
drafts outreach or explains a rejection.
"""

import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.logger import AgentLogger
from src.tools import SponsorMatchState, _get_llm, _track_tokens, is_plugin_enabled
from src.search_agent import search_case_studies, query_sponsorship_db, _build_sponsorship_context
from src.analysis_agent import _build_budget_estimate, _format_pdf_financials

_logger = AgentLogger("fit_agent")

_LANGUAGE_NAMES = {"de": "German", "en": "English", "fr": "French"}


def _translate_case_studies(case_studies: list[dict], language: str, llm: ChatOpenAI):
    """Übersetzt die Case-Study-Summaries via LLM, falls language != 'de'.

    Gibt (übersetzte_case_studies, token_usage_entry_oder_None) zurück.
    """
    if language == "de" or not case_studies:
        return case_studies, None

    numbered = "\n".join(f"{i + 1}. {c['summary']}" for i, c in enumerate(case_studies))
    prompt = (
        f"Translate the following numbered case study summaries into "
        f"{_LANGUAGE_NAMES[language]}. Keep the exact same numbering, one "
        f"translated summary per line, no extra commentary before or after.\n\n{numbered}"
    )
    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]

    translated = []
    for i, case in enumerate(case_studies):
        prefix = f"{i + 1}."
        match = next((line[len(prefix):].strip() for line in lines if line.startswith(prefix)), None)
        translated.append({**case, "summary": match or case["summary"]})

    return translated, _track_tokens("translate_case_studies", response)


def _build_rag_context(case_studies: list[dict], company_name: str, sport: str, language: str) -> str:
    """Baut den Case-Study-Kontext für den Bewertungs-Prompt, sprachabhängig formatiert."""
    if not case_studies:
        return {
            "de": "Keine relevanten Case-Studies gefunden.",
            "en": "No relevant case studies found.",
            "fr": "Aucune étude de cas pertinente trouvée.",
        }[language]

    match_type = case_studies[0]["match_type"]

    if language == "en":
        status = lambda success: "success" if success else "no success"
        bullets = "\n".join(
            f"- {c['company']} ({c['sport']}, {status(c['success'])}): {c['summary']}" for c in case_studies
        )
        if match_type == "company_sport":
            return bullets
        return (
            f"No case study found for '{company_name}' itself. For context, here are "
            f"examples from the same sport ({sport}) with other companies:\n{bullets}"
        )
    if language == "fr":
        status = lambda success: "succès" if success else "échec"
        bullets = "\n".join(
            f"- {c['company']} ({c['sport']}, {status(c['success'])}): {c['summary']}" for c in case_studies
        )
        if match_type == "company_sport":
            return bullets
        return (
            f"Aucune étude de cas trouvée pour '{company_name}' elle-même. À titre indicatif, "
            f"voici des exemples du même sport ({sport}) avec d'autres entreprises :\n{bullets}"
        )

    status = lambda success: "Erfolg" if success else "kein Erfolg"
    bullets = "\n".join(
        f"- {c['company']} ({c['sport']}, {status(c['success'])}): {c['summary']}" for c in case_studies
    )
    if match_type == "company_sport":
        return bullets
    return (
        f"Keine Case-Study zu '{company_name}' selbst gefunden. Hier zur Einordnung "
        f"Beispiele aus derselben Sportart ({sport}) mit anderen Firmen:\n{bullets}"
    )


# --- Agent Learning: Feedback-Muster ähnlicher Sponsoren in die Bewertung einfließen lassen ---

_FEEDBACK_LABELS = {
    "de": {"positive": "positiv", "negative": "negativ", "none": "kein Feedback"},
    "en": {"positive": "positive", "negative": "negative", "none": "no feedback"},
    "fr": {"positive": "positif", "negative": "négatif", "none": "pas de feedback"},
}


def _build_history_context(past_analyses: list[dict], language: str) -> str:
    """Baut den Kontext aus ähnlichen bisherigen Analysen für den Bewertungs-Prompt."""
    if not past_analyses:
        return {
            "de": "Keine früheren Analysen zu dieser Firma gefunden.",
            "en": "No previous analyses found for this company.",
            "fr": "Aucune analyse précédente trouvée pour cette entreprise.",
        }[language]

    feedback_labels = _FEEDBACK_LABELS[language]
    lines = []
    for a in past_analyses:
        date = a["timestamp"][:10]
        fb = feedback_labels.get(a["feedback"], a["feedback"])
        if language == "en":
            lines.append(f"- {a['club_name']}, {date}: score {a['fit_score']:.2f}, feedback: {fb}")
        elif language == "fr":
            lines.append(f"- {a['club_name']}, {date} : score {a['fit_score']:.2f}, feedback : {fb}")
        else:
            lines.append(f"- {a['club_name']}, {date}: Score {a['fit_score']:.2f}, Feedback: {fb}")

    header = {
        "de": "Hier sind ähnliche bisherige Analysen zu dieser Firma:",
        "en": "Here are similar previous analyses of this company:",
        "fr": "Voici des analyses précédentes similaires de cette entreprise :",
    }[language]
    return header + "\n" + "\n".join(lines)


def _build_feedback_pattern_text(feedback_rows: list[dict], language: str) -> str:
    """Baut den Hinweistext zu vergangenen Feedback-Mustern für den Bewertungs-Prompt.

    Gibt einen leeren String zurück, wenn keine bewerteten (positive/negative)
    früheren Analysen vorliegen.
    """
    if not feedback_rows:
        return ""

    positive_count = sum(1 for r in feedback_rows if r["feedback"] == "positive")
    negative_count = sum(1 for r in feedback_rows if r["feedback"] == "negative")
    companies = ", ".join(sorted({r["company_name"] for r in feedback_rows}))

    if language == "en":
        return (
            f"Note: For similar sponsors ({companies}), past feedback patterns were "
            f"{positive_count}x positive, {negative_count}x negative."
        )
    if language == "fr":
        return (
            f"Remarque : pour des sponsors similaires ({companies}), les retours passés "
            f"étaient {positive_count}x positifs, {negative_count}x négatifs."
        )
    return (
        f"Beachte: Bei ähnlichen Sponsoren ({companies}) gab es folgende Feedback-Muster: "
        f"{positive_count}x positiv, {negative_count}x negativ."
    )


def compute_feedback_adjustment(feedback_rows: list[dict]) -> float:
    """Berechnet die Score-Anpassung aus früherem Feedback zu ähnlichen Sponsoren.

    Bewusst subtil gehalten (±0.05 bis ±0.10, statt der ursprünglichen ±0.2):
    der Basis-Score des LLM soll stabil bleiben, Learning ist nur eine kleine
    Justierung, kein Ersatz fürs eigene Denken des Agents.
    """
    positive_count = sum(1 for r in feedback_rows if r["feedback"] == "positive")
    negative_count = sum(1 for r in feedback_rows if r["feedback"] == "negative")
    net = positive_count - negative_count
    if net == 0:
        return 0.0
    magnitude = min(0.05 + 0.01 * (abs(net) - 1), 0.10)
    return magnitude if net > 0 else -magnitude


def compute_hitl_adjustment(hitl_rows: list[dict]) -> float:
    """Berechnet die Score-Anpassung aus per Human-in-the-Loop-Review aufgelösten
    Ground-Truth-Entscheidungen zu unsicheren Scores (0.45-0.55) derselben Firma.

    'agree' bestätigt die bisherige grenzwertige Einschätzung (Score bleibt eher niedrig),
    'disagree' korrigiert sie nach oben. Etwas stärker gewichtet als reguläres
    Feedback (siehe compute_feedback_adjustment), da es eine explizite Human-Review-
    Entscheidung ist statt beiläufiges Daumen-hoch/-runter.
    """
    agree_count = sum(1 for r in hitl_rows if r["hitl_decision"] == "agree")
    disagree_count = sum(1 for r in hitl_rows if r["hitl_decision"] == "disagree")
    net = disagree_count - agree_count
    if net == 0:
        return 0.0
    magnitude = min(0.08 + 0.02 * (abs(net) - 1), 0.20)
    return magnitude if net > 0 else -magnitude


def _compute_data_quality(research_findings: str, case_studies: list[dict], sponsorship_matches: list[dict]) -> float:
    """Heuristisches Datenqualitäts-Signal (0.0-1.0) für die Confidence-Anzeige:
    dünne/leere Recherche bzw. fehlende Case-Studies/Sponsorship-Daten senken die
    Konfidenz spürbar, da der Agent dann auf weniger fundierten Signalen aufbaut."""
    quality = 1.0
    if not research_findings or len(research_findings) < 100:
        quality -= 0.3
    if not case_studies:
        quality -= 0.2
    if not sponsorship_matches:
        quality -= 0.1
    return max(0.0, quality)


def compute_agent_confidence(fit_score: float, resolved_hitl_count: int, data_quality: float = 1.0) -> int:
    """Agent-Confidence in Prozent für die Uncertainty-Anzeige bei Scores im unsicheren
    Band (0.45-0.55): Basiswert ist der Score selbst (nahe 50% => 'unsicher'), zusätzlich
    erhöht jede bereits per Human Review aufgelöste Entscheidung zu dieser Firma die
    Konfidenz spürbar, da der Agent dann auf bestätigter Ground Truth statt nur der
    eigenen Einschätzung aufbaut.

    data_quality (0.0-1.0, siehe _compute_data_quality) dämpft den Basiswert leicht,
    wenn Recherche/Case-Studies/Sponsorship-Daten dünn waren – Default 1.0, damit
    bestehende Aufrufer (main.py's Confidence-Neuberechnung nach einer HITL-
    Entscheidung, wo diese Signale nicht vorliegen) unverändert funktionieren."""
    base = fit_score * 100 * (0.85 + 0.15 * data_quality)
    boost = min(resolved_hitl_count * 27, 50)
    return int(round(min(100, base + boost)))


_FIT_SECTION_LABELS = {
    "de": {"pros": "Was passt gut", "cons": "Was passt weniger", "recommendation": "Empfehlung"},
    "en": {"pros": "What fits well", "cons": "What fits less", "recommendation": "Recommendation"},
    "fr": {"pros": "Ce qui correspond bien", "cons": "Ce qui correspond moins", "recommendation": "Recommandation"},
}


# --- Multi-Faktor-Scoring: LLM liefert 8 unabhängige Faktor-Scores + Begründung,
# Python kombiniert sie über feste, im Code definierte Gewichte zum Basis-Score –
# analog zu _compute_portfolio_score_impact/_compute_size_match_adjustment, wo das
# LLM ebenfalls nur Rohsignale liefert und die Gewichtung/Berechnung deterministisch
# in Python passiert, nicht vom LLM selbst erfunden wird.

_FACTOR_WEIGHTS = {
    "VALUES_FIT": 0.20,
    "AUDIENCE_FIT": 0.20,
    "SPORT_RELEVANCE": 0.15,
    "HISTORICAL_PRECEDENT": 0.15,
    "BUDGET_FIT": 0.10,
    "STRATEGIC_POTENTIAL": 0.10,
    "BRAND_SAFETY": 0.05,
    "MARKET_TIMING": 0.05,
}

_FACTOR_LABELS = {
    "de": {
        "VALUES_FIT": "Werte-Fit",
        "AUDIENCE_FIT": "Zielgruppen-Fit",
        "SPORT_RELEVANCE": "Sportart-Passung",
        "HISTORICAL_PRECEDENT": "Historische Präzedenzfälle",
        "BUDGET_FIT": "Budget-Fit",
        "STRATEGIC_POTENTIAL": "Strategisches Potenzial",
        "BRAND_SAFETY": "Markensicherheit",
        "MARKET_TIMING": "Markttiming",
    },
    "en": {
        "VALUES_FIT": "Values fit",
        "AUDIENCE_FIT": "Audience fit",
        "SPORT_RELEVANCE": "Sport relevance",
        "HISTORICAL_PRECEDENT": "Historical precedent",
        "BUDGET_FIT": "Budget fit",
        "STRATEGIC_POTENTIAL": "Strategic potential",
        "BRAND_SAFETY": "Brand safety",
        "MARKET_TIMING": "Market timing",
    },
    "fr": {
        "VALUES_FIT": "Adéquation des valeurs",
        "AUDIENCE_FIT": "Adéquation du public",
        "SPORT_RELEVANCE": "Pertinence sportive",
        "HISTORICAL_PRECEDENT": "Précédents historiques",
        "BUDGET_FIT": "Adéquation budgétaire",
        "STRATEGIC_POTENTIAL": "Potentiel stratégique",
        "BRAND_SAFETY": "Sécurité de marque",
        "MARKET_TIMING": "Timing du marché",
    },
}

_FACTOR_BREAKDOWN_HEADER = {
    "de": "Bewertungsfaktoren",
    "en": "Evaluation factors",
    "fr": "Facteurs d'évaluation",
}


def _compute_weighted_score(factor_scores: dict[str, float]) -> float:
    """Kombiniert die 8 Einzelfaktor-Scores über die festen _FACTOR_WEIGHTS zu einem
    Basis-Score (0.0-1.0). Fehlende Faktoren (z.B. bei einer unvollständig
    geparsten LLM-Antwort) fallen auf 0.5 zurück, dieselbe neutrale Default-
    Philosophie wie der bisherige direkte SCORE-Parse-Fallback."""
    weighted_sum = sum(factor_scores.get(key, 0.5) * weight for key, weight in _FACTOR_WEIGHTS.items())
    return max(0.0, min(1.0, weighted_sum))


def _format_factor_breakdown(factor_scores: dict[str, float], factor_notes: dict[str, str], language: str) -> str:
    """Baut die Faktoren-Aufschlüsselung als eigene Markdown-Sektion, angehängt an
    die Fit-Begründung – landet damit automatisch im gecachten fit_reasoning-String
    und wird bei einem Score-Cache-Treffer ohne separate Caching-Logik mit angezeigt."""
    labels = _FACTOR_LABELS[language]
    lines = []
    for key, weight in _FACTOR_WEIGHTS.items():
        score = factor_scores.get(key, 0.5)
        note = factor_notes.get(key, "")
        suffix = f" — {note}" if note else ""
        lines.append(f"- {labels[key]} ({int(weight * 100)}%): {score:.2f}{suffix}")
    header = _FACTOR_BREAKDOWN_HEADER[language]
    return f"**{header}:**\n" + "\n".join(lines)


def _format_fit_reasoning(pros: list[str], cons: list[str], recommendation: str, language: str) -> str:
    """Baut die kompakte, strukturierte Fit-Begründung (Bullets statt Fließtext,
    keine technischen Details wie Score-Caching/Datum – die stehen ggf. separat
    in der eigenen Konkurrenzanalyse-Sektion)."""
    section_labels = _FIT_SECTION_LABELS[language]
    parts = []
    if pros:
        parts.append(f"**{section_labels['pros']}:**\n" + "\n".join(f"- {p}" for p in pros))
    if cons:
        parts.append(f"**{section_labels['cons']}:**\n" + "\n".join(f"- {c}" for c in cons))
    if recommendation:
        parts.append(f"**{section_labels['recommendation']}:** {recommendation}")
    return "\n\n".join(parts)


_LEGACY_SCORE_NOTE_RE = re.compile(r"\s*\([^)]*\bScore\b[^)]*\)", re.IGNORECASE)


def _strip_legacy_score_notes(text: str) -> str:
    """Entfernt technische Alt-Hinweise ("(Score ... angepasst ...)", "(Score aus
    identischer vorheriger Analyse vom ...)"), die vor der Fit-Bewertungs-Vereinfachung
    in gecachte fit_reasoning-Strings in der DB geschrieben wurden und sonst bei jedem
    Cache-Hit weiter angezeigt (und dabei teils mehrfach akkumuliert) würden."""
    return _LEGACY_SCORE_NOTE_RE.sub("", text).strip()


@traceable(run_type="chain", name="evaluate_fit")
def evaluate_fit(state: SponsorMatchState) -> dict:
    """Bewertet den Markenfit zwischen Firma und Verein basierend auf den Recherche-Ergebnissen."""
    # Lokaler Import statt Modul-Top-Level, um den Zyklus orchestrator -> fit_agent
    # (Graph-Aufbau) <-> fit_agent -> orchestrator (Persistenz-Zugriff) zu vermeiden:
    # orchestrator.py importiert evaluate_fit, um den Graphen zu bauen, braucht dafür
    # aber selbst keine Persistenz-Funktionen von hier – der Zyklus entsteht nur, wenn
    # dieser Import auf Modulebene stünde.
    from src.orchestrator import (
        get_exact_previous_analysis,
        get_similar_analyses,
        get_resolved_hitl_decisions,
        save_analysis,
    )

    club = state["club_profile"]
    findings = state["research_findings"]
    company_name = state["company_name"]
    language = state["language"]
    llm = _get_llm(state["selected_model"])
    _logger.log_agent_start("evaluate_fit", company=company_name, club=club["name"])

    # RAG: relevante Case-Studies zu Firma + Sportart aus der Wissensbasis holen
    # (läuft immer, auch bei Score-Cache-Treffer, da für die Anzeige gebraucht)
    # Plugin-Gating case_study_db: "required", UI kann es nicht abschalten –
    # dieser Zweig greift nur defensiv, falls die Config es trotzdem tut.
    token_usage = []
    if is_plugin_enabled("case_study_db"):
        case_studies = search_case_studies(company_name, club["sport"])
        case_studies, translate_token_entry = _translate_case_studies(case_studies, language, llm)
        if translate_token_entry:
            token_usage.append(translate_token_entry)
    else:
        case_studies = []

    # Externe Sponsorship-DB: historische Sponsoring-Fälle zu Firma + Sportart
    # (ebenfalls unabhängig vom Score-Cache, da für die Anzeige gebraucht)
    # Plugin-Gating sponsorship_db: nicht required, kann im Plugin Manager
    # abgeschaltet werden – dann wird die externe DB gar nicht erst abgefragt.
    sponsorship_matches = query_sponsorship_db(company_name, club["sport"]) if is_plugin_enabled(
        "sponsorship_db"
    ) else []

    # Score-Konsistenz: exakt gleiche Firma+Verein-Kombination schon einmal
    # analysiert? Dann Score cachen statt neu (und ggf. anders) zu berechnen.
    cached = get_exact_previous_analysis(company_name, club["name"])

    # Portfolio + Size Compatibility kommen bereits fertig berechnet vom
    # analyze_financials-Node (analysis_agent.py), der vor diesem Node läuft –
    # kopieren, da unten score_before/after_adjustment ergänzt werden.
    portfolio = dict(state["competitor_analysis"])
    size_compatibility = dict(state["size_compatibility"])

    # Budget-Schätzung für die eigene UI-Sektion (main.py), ebenfalls unabhängig
    # vom Score-Cache-Zweig, da rein datenbasiert (kein LLM-Call) und für die
    # Anzeige gebraucht. Ergänzt um evtl. per PDF-Extraktion gefundene echte
    # Finanzkennzahlen (analyze_financials-Node, Phase 2) – state.get() statt
    # bracket-Zugriff, damit ein Aufrufer, der pdf_financials vergisst zu
    # setzen (z.B. ein künftiger MCP-Tool-Pfad), einfach keine PDF-Sektion
    # bekommt statt eines KeyError.
    budget_estimate_text = (
        _build_budget_estimate(sponsorship_matches, language) if is_plugin_enabled("budget_estimator") else ""
    )
    pdf_financials_section = _format_pdf_financials(state.get("pdf_financials", {}), language)
    if pdf_financials_section:
        budget_estimate_text = (
            f"{budget_estimate_text}\n\n{pdf_financials_section}" if budget_estimate_text else pdf_financials_section
        )

    # Human-in-the-Loop: bereits per Human Review aufgelöste Ground-Truth-Entscheidungen
    # zu dieser Firma holen (unabhängig vom Score-Cache-Zweig, wird für die
    # Agent-Confidence-Anzeige unten in jedem Fall gebraucht).
    resolved_hitl_rows = get_resolved_hitl_decisions(company_name)

    # Phase 3e: pro-Faktor-Konfidenz für die Quality-Metrics-Anzeige (main.py).
    # Bleibt bei einem Score-Cache-Treffer leer, da dort kein LLM-Call läuft und
    # somit keine frischen Faktor-Scores existieren (siehe cached-Zweig unten) –
    # main.py behandelt eine leere fit_agent_factors-Dict als "kein Breakdown".
    factor_scores: dict[str, float] = {}

    if cached is not None:
        score = cached["fit_score"]
        reasoning = _strip_legacy_score_notes((cached.get("fit_reasoning") or "").strip())
        learning_applied = False
        # Score aus dem Cache wird bewusst NICHT nochmal per Portfolio-Analyse
        # angepasst (sonst würde sich ein gecachter Score bei jeder Wiederholung
        # der gleichen Anfrage weiter verschieben).
        portfolio["score_before_adjustment"] = score
        portfolio["score_after_adjustment"] = score
        size_compatibility["score_before_adjustment"] = score
        size_compatibility["score_after_adjustment"] = score
    else:
        rag_context = _build_rag_context(case_studies, company_name, club["sport"], language)
        sponsorship_context = _build_sponsorship_context(sponsorship_matches, language)

        # Long-term memory: haben wir diese Firma schon einmal analysiert?
        past_analyses = get_similar_analyses(company_name, limit=10)
        history_context = _build_history_context(past_analyses[:3], language)

        # Agent Learning: Feedback-Muster aus früheren Analysen ähnlicher Sponsoren
        feedback_rows = [a for a in past_analyses if a["feedback"] in ("positive", "negative")]
        feedback_pattern_text = _build_feedback_pattern_text(feedback_rows, language)
        memory_context = history_context
        if feedback_pattern_text:
            memory_context += "\n\n" + feedback_pattern_text

        # Optionale Plugins: nur aktiv, wenn im Plugin Manager eingeschaltet
        # (competitor_analysis macht einen echten zusätzlichen Tool-Call).
        if portfolio["found"]:
            memory_context += (
                "\n\nCompany's own sponsorship portfolio: "
                f"categories={', '.join(portfolio['categories'])}, "
                f"active_sponsorships={portfolio['active_count']}, "
                f"audience={portfolio['audience']}, "
                f"sponsorships_in_this_sport={portfolio['same_sport_count']} "
                f"(saturation: {portfolio['saturation_level']}), "
                f"audience_fit_with_club={portfolio['audience_fit']} ({portfolio['match_percent']}%)"
            )
        if budget_estimate_text:
            memory_context += "\n\n" + budget_estimate_text

        if language == "en":
            prompt = (
                f"Evaluate how well the following company fits as a sponsor for the club "
                f"'{club['name']}'.\n\n"
                f"Club profile:\n"
                f"- Sport: {club['sport']}\n"
                f"- Fanbase: {club['fanbase']}\n"
                f"- Values: {club['values']}\n"
                f"- Sponsorship gaps sought: {club['sponsorship_gaps']}\n\n"
                f"Research on the company:\n{findings}\n\n"
                f"Here are relevant case studies (fictional sample data from an internal "
                f"knowledge base, not verified real facts – use them only as a rough "
                f"indication of patterns, not as documented history):\n{rag_context}\n\n"
                f"{memory_context}\n\n"
                f"{sponsorship_context}\n\n"
                f"Respond in exactly this format, nothing before or after (keep the labels "
                f"in English, write the content itself in English). Score each factor "
                f"independently from 0.0 (poor) to 1.0 (excellent) — note that BRAND_SAFETY "
                f"is inverted: 1.0 means low risk/safe, 0.0 means high risk. Keep each "
                f"justification to about 10 words. Be concise and avoid repetition — the "
                f"whole response must stay under 280 words:\n"
                f"VALUES_FIT: <0.0-1.0>|<short justification>\n"
                f"AUDIENCE_FIT: <0.0-1.0>|<short justification>\n"
                f"SPORT_RELEVANCE: <0.0-1.0>|<short justification>\n"
                f"HISTORICAL_PRECEDENT: <0.0-1.0>|<short justification, grounded in the case "
                f"studies/sponsorship history above>\n"
                f"BUDGET_FIT: <0.0-1.0>|<short justification, mention sensitivity to budget "
                f"assumptions>\n"
                f"STRATEGIC_POTENTIAL: <0.0-1.0>|<short justification>\n"
                f"BRAND_SAFETY: <0.0-1.0, 1.0=low risk>|<short justification>\n"
                f"MARKET_TIMING: <0.0-1.0>|<short justification>\n"
                f"PROS: <3-4 short bullet points, separated by |>\n"
                f"CONS: <2-3 short bullet points, separated by |>\n"
                f"RECOMMENDATION: <exactly one sentence>"
            )
        elif language == "fr":
            prompt = (
                f"Évaluez dans quelle mesure l'entreprise suivante convient comme sponsor "
                f"pour le club '{club['name']}'.\n\n"
                f"Profil du club :\n"
                f"- Sport : {club['sport']}\n"
                f"- Base de fans : {club['fanbase']}\n"
                f"- Valeurs : {club['values']}\n"
                f"- Domaines de sponsoring recherchés : {club['sponsorship_gaps']}\n\n"
                f"Recherche sur l'entreprise :\n{findings}\n\n"
                f"Voici des études de cas pertinentes (données fictives d'une base de "
                f"connaissances interne, pas des faits réels vérifiés – utilisez-les "
                f"uniquement comme indication approximative de tendances, pas comme "
                f"historique avéré) :\n{rag_context}\n\n"
                f"{memory_context}\n\n"
                f"{sponsorship_context}\n\n"
                f"Répondez exactement dans ce format, rien avant ni après (gardez les "
                f"étiquettes en anglais, rédigez le contenu lui-même en français). Évaluez "
                f"chaque facteur indépendamment de 0.0 (faible) à 1.0 (excellent) — notez que "
                f"BRAND_SAFETY est inversé : 1.0 signifie un risque faible/sûr, 0.0 un risque "
                f"élevé. Limitez chaque justification à environ 10 mots. Soyez concis et "
                f"évitez les répétitions — la réponse entière doit rester sous 280 mots :\n"
                f"VALUES_FIT: <0.0-1.0>|<justification courte>\n"
                f"AUDIENCE_FIT: <0.0-1.0>|<justification courte>\n"
                f"SPORT_RELEVANCE: <0.0-1.0>|<justification courte>\n"
                f"HISTORICAL_PRECEDENT: <0.0-1.0>|<justification courte, basée sur les études "
                f"de cas/l'historique de sponsoring ci-dessus>\n"
                f"BUDGET_FIT: <0.0-1.0>|<justification courte, mentionnez la sensibilité aux "
                f"hypothèses budgétaires>\n"
                f"STRATEGIC_POTENTIAL: <0.0-1.0>|<justification courte>\n"
                f"BRAND_SAFETY: <0.0-1.0, 1.0=risque faible>|<justification courte>\n"
                f"MARKET_TIMING: <0.0-1.0>|<justification courte>\n"
                f"PROS: <3-4 puces courtes, séparées par |>\n"
                f"CONS: <2-3 puces courtes, séparées par |>\n"
                f"RECOMMENDATION: <exactement une phrase>"
            )
        else:
            prompt = (
                f"Bewerte, wie gut folgende Firma als Sponsor für den Verein '{club['name']}' passt.\n\n"
                f"Vereinsprofil:\n"
                f"- Sportart: {club['sport']}\n"
                f"- Fanbase: {club['fanbase']}\n"
                f"- Werte: {club['values']}\n"
                f"- Gesuchte Sponsoring-Bereiche: {club['sponsorship_gaps']}\n\n"
                f"Recherche zur Firma:\n{findings}\n\n"
                f"Hier sind relevante Case-Studies (fiktive Beispieldaten aus einer internen "
                f"Wissensbasis, keine verifizierten realen Fakten – nutze sie nur als groben "
                f"Anhaltspunkt für Muster, nicht als belegte Historie):\n{rag_context}\n\n"
                f"{memory_context}\n\n"
                f"{sponsorship_context}\n\n"
                f"Antworte in genau diesem Format, nichts davor oder danach (behalte die "
                f"Labels auf Englisch, schreibe den Inhalt selbst auf Deutsch). Bewerte jeden "
                f"Faktor unabhängig von 0.0 (schlecht) bis 1.0 (exzellent) — BRAND_SAFETY ist "
                f"invertiert: 1.0 bedeutet geringes Risiko/sicher, 0.0 bedeutet hohes Risiko. "
                f"Halte jede Begründung auf ca. 10 Wörter. Sei prägnant und vermeide "
                f"Wiederholungen — die gesamte Antwort muss unter 280 Wörtern bleiben:\n"
                f"VALUES_FIT: <0.0-1.0>|<kurze Begründung>\n"
                f"AUDIENCE_FIT: <0.0-1.0>|<kurze Begründung>\n"
                f"SPORT_RELEVANCE: <0.0-1.0>|<kurze Begründung>\n"
                f"HISTORICAL_PRECEDENT: <0.0-1.0>|<kurze Begründung, gestützt auf die "
                f"Case-Studies/Sponsorship-Historie oben>\n"
                f"BUDGET_FIT: <0.0-1.0>|<kurze Begründung, erwähne Sensitivität gegenüber "
                f"Budget-Annahmen>\n"
                f"STRATEGIC_POTENTIAL: <0.0-1.0>|<kurze Begründung>\n"
                f"BRAND_SAFETY: <0.0-1.0, 1.0=geringes Risiko>|<kurze Begründung>\n"
                f"MARKET_TIMING: <0.0-1.0>|<kurze Begründung>\n"
                f"PROS: <3-4 kurze Stichpunkte, getrennt durch |>\n"
                f"CONS: <2-3 kurze Stichpunkte, getrennt durch |>\n"
                f"RECOMMENDATION: <genau ein Satz>"
            )

        response = llm.invoke(prompt)
        content = response.content.strip()
        token_usage.append(_track_tokens("evaluate_fit", response))

        factor_notes: dict[str, str] = {}
        pros: list[str] = []
        cons: list[str] = []
        recommendation = ""

        for line in content.split("\n"):
            line = line.strip()
            matched_factor = False
            for key in _FACTOR_WEIGHTS:
                prefix = f"{key}:"
                if line.startswith(prefix):
                    raw_score, _, note = line[len(prefix):].strip().partition("|")
                    try:
                        factor_scores[key] = max(0.0, min(1.0, float(raw_score.strip())))
                    except ValueError:
                        pass
                    factor_notes[key] = note.strip()
                    matched_factor = True
                    _logger.log_agent_step(
                        f"evaluation_factor:{key}", score=factor_scores.get(key), note=note.strip()
                    )
                    break
            if matched_factor:
                continue
            if line.startswith("PROS:"):
                pros = [p.strip() for p in line.replace("PROS:", "", 1).split("|") if p.strip()]
            elif line.startswith("CONS:"):
                cons = [c.strip() for c in line.replace("CONS:", "", 1).split("|") if c.strip()]
            elif line.startswith("RECOMMENDATION:"):
                recommendation = line.replace("RECOMMENDATION:", "", 1).strip()

        # Basis-Score: gewichtete Kombination der 8 Einzelfaktoren statt eines
        # einzigen vom LLM direkt gelieferten Werts (siehe _compute_weighted_score).
        score = _compute_weighted_score(factor_scores)
        reasoning = _format_fit_reasoning(pros, cons, recommendation, language) or content
        reasoning += "\n\n" + _format_factor_breakdown(factor_scores, factor_notes, language)

        # Agent Learning: Score anhand früherer Feedback-Muster leicht anpassen (max ±0.10)
        adjustment = compute_feedback_adjustment(feedback_rows)
        learning_applied = adjustment != 0.0
        if learning_applied:
            score = max(0.0, min(1.0, score + adjustment))

        # Konkurrenzanalyse-Impact: Score zusätzlich anpassen, abhängig vom individuellen
        # Zielgruppen-Match und der individuellen Sättigung DIESER Firma in DIESER
        # Sportart (dynamisch, kein fixer Wert – siehe _compute_portfolio_score_impact).
        # Die Begründung dazu steht nicht mehr im Fließtext, sondern separat in der
        # eigenen Konkurrenzanalyse-Sektion (main.py card_competitor_analysis).
        portfolio["score_before_adjustment"] = score
        portfolio_adjustment = portfolio["score_adjustment"]
        if portfolio["found"] and portfolio_adjustment != 0.0:
            score = max(0.0, min(1.0, score + portfolio_adjustment))
        portfolio["score_after_adjustment"] = score

        # Human-in-the-Loop: bei unsicheren Scores (0.45-0.55) fließen bereits vom Menschen
        # aufgelöste Agree/Disagree-Entscheidungen zu ähnlichen Anfragen dieser Firma als
        # Ground Truth in den Score ein (siehe compute_hitl_adjustment). Wie beim Score-
        # Cache bewusst NICHT im cached-Zweig angewendet, damit ein gecachter Score stabil
        # bleibt statt sich bei jeder Wiederholung weiter zu verschieben.
        if 0.45 <= round(score, 2) <= 0.55 and resolved_hitl_rows:
            hitl_adjustment = compute_hitl_adjustment(resolved_hitl_rows)
            if hitl_adjustment != 0.0:
                score = max(0.0, min(1.0, score + hitl_adjustment))

        # Size Compatibility: Score zusätzlich anpassen, abhängig vom Größen-Match
        # zwischen Club und Company (sport-agnostisch, siehe _compute_size_match_adjustment).
        size_compatibility["score_before_adjustment"] = score
        size_adjustment = size_compatibility["score_adjustment"]
        if size_adjustment != 0.0:
            score = max(0.0, min(1.0, score + size_adjustment))
        size_compatibility["score_after_adjustment"] = score

    # round(): die Summe mehrerer Score-Anpassungen (z.B. 0.5 - 0.05) erzeugt gelegentlich
    # Gleitkomma-Artefakte wie 0.44999999999999996 statt exakt 0.45 – ohne Rundung würde
    # ein für den User sichtbarer Score von "0.45" fälschlich NICHT als unsicher gelten
    # (HITL-Review würde dann nicht auslösen, obwohl die Anzeige "0.45" zeigt).
    score = round(score, 2)

    # Human-in-the-Loop-Anzeige: unsicher, wenn der finale Score im Grenzband liegt;
    # Agent-Confidence berücksichtigt bereits vorhandene Human-Review-Ground-Truth.
    is_uncertain = 0.45 <= score <= 0.55
    data_quality = _compute_data_quality(findings, case_studies, sponsorship_matches)
    agent_confidence = compute_agent_confidence(score, len(resolved_hitl_rows), data_quality)

    # Long-term memory: neue Analyse speichern (Feedback wird später per Update nachgetragen)
    analysis_id = save_analysis(
        club_name=club["name"],
        company_name=company_name,
        fit_score=score,
        language=language,
        selected_model=state["selected_model"],
        research_summary=findings,
        fit_reasoning=reasoning,
        case_studies=case_studies,
        learning_applied=learning_applied,
        score_cached=cached is not None,
        user_id=state.get("user_id"),
    )

    # Phase 3e: localized-label -> 0.0-1.0 Konfidenz pro Faktor, direkt in main.py's
    # st.progress(confidence, text=f"{factor}: {confidence:.0%}")-Schleife nutzbar.
    # Leer bei einem Score-Cache-Treffer (siehe factor_scores-Deklaration oben).
    fit_agent_factors = {
        _FACTOR_LABELS[language][key]: factor_scores[key] for key in _FACTOR_WEIGHTS if key in factor_scores
    }

    _logger.log_agent_result(
        "evaluate_fit", score=score, agent_confidence=agent_confidence, is_uncertain=is_uncertain,
        score_cached=cached is not None,
    )
    return {
        "fit_score": score,
        "fit_reasoning": reasoning,
        "used_case_studies": case_studies,
        "used_sponsorship_matches": sponsorship_matches,
        "competitor_analysis": portfolio,
        "budget_estimate": budget_estimate_text,
        "size_compatibility": size_compatibility,
        "analysis_id": analysis_id,
        "learning_applied": learning_applied,
        "is_uncertain": is_uncertain,
        "agent_confidence": agent_confidence,
        "fit_agent_factors": fit_agent_factors,
        "hitl_resolved_count": len(resolved_hitl_rows),
        "token_usage": token_usage,
    }


def route_by_fit(state: SponsorMatchState) -> Literal["draft_outreach", "explain_rejection"]:
    """Entscheidet basierend auf dem Fit-Score, ob eine Ansprache entworfen oder abgelehnt wird."""
    if state["fit_score"] >= 0.6:
        return "draft_outreach"
    return "explain_rejection"


@traceable(run_type="chain", name="draft_outreach")
def draft_outreach(state: SponsorMatchState) -> dict:
    """Entwirft eine Erstansprache an die Firma, basierend auf Recherche und Fit-Bewertung."""
    club = state["club_profile"]
    company_name = state["company_name"]
    reasoning = state["fit_reasoning"]
    language = state["language"]
    llm = _get_llm(state["selected_model"])

    if language == "en":
        prompt = (
            f"Write a short, professional initial outreach message (email style, "
            f"4-5 sentences) from '{club['name']}' to '{company_name}', proposing a "
            f"sponsorship partnership. Use this reasoning as the basis for why the fit "
            f"is good: {reasoning}\n\n"
            f"Tone: professional but enthusiastic. No greeting/closing needed, just "
            f"the main body text. Write in English."
        )
    elif language == "fr":
        prompt = (
            f"Rédigez un court message de prise de contact professionnel (style "
            f"e-mail, 4-5 phrases) de '{club['name']}' à '{company_name}', proposant "
            f"un partenariat de sponsoring. Utilisez ce raisonnement comme base pour "
            f"expliquer pourquoi l'adéquation est bonne : {reasoning}\n\n"
            f"Ton : professionnel mais enthousiaste. Pas besoin de formule d'appel ou "
            f"de politesse, juste le texte principal. Répondez en français."
        )
    else:
        prompt = (
            f"Schreib eine kurze, professionelle Erstansprache (E-Mail-Stil, 4-5 Sätze) "
            f"von '{club['name']}' an '{company_name}', um eine Sponsoring-Partnerschaft "
            f"vorzuschlagen. Nutze diese Begründung als Grundlage, warum der Fit gut ist: "
            f"{reasoning}\n\n"
            f"Ton: professionell, aber enthusiastisch. Keine Anrede/Grußformel nötig, "
            f"nur der Haupttext. Antworte auf Deutsch."
        )
    response = llm.invoke(prompt)
    return {
        "outreach_draft": response.content.strip(),
        "token_usage": [_track_tokens("draft_outreach", response)],
    }


@traceable(run_type="chain", name="explain_rejection")
def explain_rejection(state: SponsorMatchState) -> dict:
    """Erklärt kurz und sachlich, warum die Firma kein guter Sponsoring-Fit ist."""
    reasoning = state["fit_reasoning"]
    language = state["language"]
    if language == "en":
        rejection_reason = f"Insufficient fit (score < 0.6): {reasoning}"
    elif language == "fr":
        rejection_reason = f"Adéquation insuffisante (score < 0.6) : {reasoning}"
    else:
        rejection_reason = f"Kein ausreichender Fit (Score < 0.6): {reasoning}"
    return {"rejection_reason": rejection_reason}
