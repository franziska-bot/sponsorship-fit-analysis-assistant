import json
import os
import re
import time
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.tracers.context import collect_runs
from src.agent import (
    app,
    get_langsmith_trace_url,
    get_analysis_history,
    update_analysis_feedback,
    get_feedback_confidence,
    evaluate_with_ragas,
    get_score_consistency,
    create_user,
    verify_user,
    get_user_settings,
    save_user_settings,
    delete_user_account,
)

# --- Security Guard: Rate Limiting, Input Validation, Security-Logging ---

SECURITY_LOG_FILE = "data/security_log.jsonl"
COMPANY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9\s\-\.äöüßÄÖÜ]*$")
QUESTION_PATTERN = re.compile(r"\?|\bwas\b|\bwer\b|\bwie\b", re.IGNORECASE)
RATE_LIMIT_SHORT = (10, 5 * 60)      # max. 10 Analysen pro 5 Minuten
RATE_LIMIT_LONG = (100, 24 * 60 * 60)  # max. 100 Analysen pro 24h


def looks_like_question(company_name: str) -> bool:
    """Erkennt Fragen ("?", "was", "wer", "wie") statt Firmennamen im Input-Feld."""
    return bool(QUESTION_PATTERN.search(company_name))


def validate_company_input(company_name: str) -> bool:
    """Min. 2 / Max. 100 Zeichen, nur alphanumerisch + Leerzeichen/-.äöüßÄÖÜ.

    Verhindert, dass beliebiger Text (z.B. Prompt-Injection-Versuche oder
    SQL-Metazeichen) über das Company-Feld in Prompts/Queries gelangt.
    Hinweis: unsere DB-Queries sind ohnehin bereits parametrisiert (kein
    String-Interpolation-Risiko) – diese Validierung ist primär Schutz vor
    Prompt-Injection und allgemeiner Input-Hygiene, nicht die einzige
    SQL-Absicherung.
    """
    stripped = company_name.strip()
    if not (2 <= len(stripped) <= 100):
        return False
    return bool(COMPANY_NAME_PATTERN.fullmatch(company_name))


def check_rate_limit() -> tuple[bool, float]:
    """Rate Limiting pro Session (st.session_state, nicht global/über Nutzer hinweg).

    Gibt (erlaubt, wartezeit_in_minuten) zurück.
    """
    now = time.time()
    timestamps = st.session_state.setdefault("request_timestamps", [])
    # Alte Einträge jenseits des längeren Fensters aufräumen, damit die Liste
    # nicht unbegrenzt wächst.
    timestamps[:] = [t for t in timestamps if now - t < RATE_LIMIT_LONG[1]]

    short_limit, short_window = RATE_LIMIT_SHORT
    recent_short = [t for t in timestamps if now - t < short_window]
    if len(recent_short) >= short_limit:
        wait_seconds = short_window - (now - min(recent_short))
        return False, max(wait_seconds / 60, 0.1)

    long_limit, long_window = RATE_LIMIT_LONG
    if len(timestamps) >= long_limit:
        wait_seconds = long_window - (now - min(timestamps))
        return False, max(wait_seconds / 60, 0.1)

    return True, 0.0


def record_request() -> None:
    st.session_state.setdefault("request_timestamps", []).append(time.time())


def log_security_event(company_name: str, status: str) -> None:
    """Silentes Security-Log: keine UI-Ausgabe, nur Persistenz für spätere Analyse."""
    os.makedirs(os.path.dirname(SECURITY_LOG_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "company_name": company_name,
        "session_id": st.session_state.get("session_id"),
        "status": status,
    }
    with open(SECURITY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# --- Plugin System: Agent-Fähigkeiten dynamisch an/abschaltbar ---

PLUGINS_FILE = "data/available_plugins.json"


def load_plugins() -> list[dict]:
    try:
        with open(PLUGINS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_plugins(plugins: list[dict]) -> None:
    os.makedirs(os.path.dirname(PLUGINS_FILE), exist_ok=True)
    with open(PLUGINS_FILE, "w", encoding="utf-8") as f:
        json.dump(plugins, f, ensure_ascii=False, indent=2)


def is_plugin_enabled_for_display(plugin_id: str) -> bool:
    """Steuert, ob eine Result-Sektion aktuell angezeigt wird.

    Quelle der Wahrheit ist st.session_state["enabled_plugins"] (befüllt vom
    Plugin Manager), nicht der Stand des Plugins zum Analyse-Zeitpunkt – so
    verschwindet eine Sektion sofort per Toggle, auch für bereits angezeigte
    Ergebnisse.
    """
    return st.session_state.get("enabled_plugins", {}).get(plugin_id, True)

# Bewusst KEINE geteilte/gecachte DB-Connection (kein @st.cache_resource +
# configure_shared_connection): eine über Streamlit-Reruns hinweg geteilte
# sqlite3.Connection kann in einen "readonly"/gesperrten Zustand geraten, wenn
# Streamlit einen Rerun mitten in einem Schreibzugriff abbricht (z.B. durch
# schnell aufeinanderfolgende Interaktionen). agent.py öffnet daher pro
# Funktionsaufruf weiterhin eine frische, kurzlebige Connection – für lokales
# SQLite ist das ohnehin im Mikrosekundenbereich und damit kein spürbarer
# Performance-Nachteil.

FEEDBACK_FILE = "data/feedback.jsonl"

st.set_page_config(page_title="Sponsor Match", initial_sidebar_state="expanded")

# Minimales CSS statt vieler einzelner Inline-Styles: eine Handvoll globaler
# Regeln, die über data-testid/st-key-Selektoren ALLE Karten/Buttons auf
# einmal stylen, statt pro Widget eigenes Markup zu brauchen. Karten nutzen
# st.container(key="card_...", border=True) – der Rahmen/die Rundung kommt
# nativ von Streamlit, hier wird nur der dezente graue Hintergrund ergänzt.
CUSTOM_CSS = """
<style>
div[class*="st-key-card_"] {
    background: #2B2B2B;
    color: #FAFAFA;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.25rem;
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}
.sm-sidebar-group-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #AAAAAA;
    margin: 0.2rem 0 0.4rem 0;
}
/* Required-Plugin-Toggles sind absichtlich nicht klickbar (disabled=True), zeigen
   aber standardmäßig einen "not-allowed"-Cursor (Verbotssymbol) an. Rein visuell
   auf einen neutralen Cursor zurücksetzen – Toggle bleibt weiterhin deaktiviert. */
div[class*="st-key-plugin_toggle_"] label[data-disabled="true"] {
    cursor: default;
}
/* Streamlits Default-Styling für deaktivierte, aber aktive (checked) Toggles nutzt
   ein halbtransparentes Weiß (rgba(250,250,250,0.2)) für die Schiene und einen sehr
   dunklen Knopf – im Dark Theme dadurch praktisch unsichtbar, Required-Plugins wirken
   fälschlich "aus". Rein visuell auf die gleiche rote "An"-Farbe wie bei normalen
   Toggles zurücksetzen; der Toggle bleibt weiterhin nicht klickbar (disabled=True).*/
div[class*="st-key-plugin_toggle_"] label[data-disabled="true"][data-selected="true"] > div:first-of-type {
    background-color: rgba(255, 75, 75, 0.7) !important;
}
div[class*="st-key-plugin_toggle_"] label[data-disabled="true"][data-selected="true"] > div:first-of-type > div {
    background-color: #FAFAFA !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_group_label(text: str) -> None:
    st.markdown(f'<p class="sm-sidebar-group-label">{text}</p>', unsafe_allow_html=True)


def render_help_popover(help_text: str) -> None:
    """Kompaktes '?'-Hover-Tooltip (kein Klick/Dropdown) neben Feature-Namen, die kein
    natives help= unterstützen (z.B. st.expander-Header). Wird direkt über dem
    jeweiligen Feature platziert."""
    _, help_col = st.columns([0.86, 0.14])
    with help_col:
        st.caption("", help=help_text)


# --- User Authentication (Login/Register) ---
# Login-Bildschirm ist bewusst nicht mehrsprachig (fest Deutsch) – die
# Sprachauswahl selbst lebt in der Sidebar der Hauptapp, die vor dem Login
# noch nicht sichtbar ist (Henne-Ei-Problem), daher hier kein LABELS-Bezug.

SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 Min. Inaktivität -> automatischer Logout


def render_auth_ui() -> None:
    st.title("Sponsor Match")
    st.caption("Bitte melde dich an, um fortzufahren.")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Passwort", type="password")
            login_submitted = st.form_submit_button("Login", type="primary")
        if login_submitted:
            user = verify_user(username.strip(), password)
            if user:
                st.session_state["user"] = user["username"]
                st.session_state["user_id"] = user["id"]
                st.session_state["last_activity"] = time.time()
                st.rerun()
            else:
                st.error("Ungültiger Username oder Passwort.")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("E-Mail", key="reg_email")
            new_password = st.text_input("Passwort", type="password", key="reg_password")
            confirm_password = st.text_input(
                "Passwort bestätigen", type="password", key="reg_confirm"
            )
            register_submitted = st.form_submit_button("Registrieren", type="primary")
        if register_submitted:
            if not new_username.strip() or len(new_username.strip()) < 3:
                st.error("Username muss mindestens 3 Zeichen haben.")
            elif len(new_password) < 8:
                st.error("Passwort muss mindestens 8 Zeichen haben.")
            elif new_password != confirm_password:
                st.error("Passwörter stimmen nicht überein.")
            else:
                success, error_code = create_user(
                    new_username.strip(), new_password, new_email.strip()
                )
                if success:
                    st.success("Account erstellt! Du kannst dich jetzt einloggen.")
                elif error_code == "username_taken":
                    st.error("Dieser Username ist bereits vergeben.")
                else:
                    st.error("Registrierung fehlgeschlagen.")


if "user" not in st.session_state:
    render_auth_ui()
    st.stop()

if time.time() - st.session_state.get("last_activity", time.time()) > SESSION_TIMEOUT_SECONDS:
    for _key in ("user", "user_id", "last_activity"):
        st.session_state.pop(_key, None)
    st.warning("Session wegen Inaktivität abgelaufen. Bitte erneut anmelden.")
    render_auth_ui()
    st.stop()

st.session_state["last_activity"] = time.time()

# OpenRouter-Preise in $ pro 1K Tokens
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "openai/gpt-4o": {"input": 0.005, "output": 0.015},
    "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "anthropic/claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
}

LANGUAGE_MAP = {"Deutsch": "de", "English": "en", "Français": "fr"}

LABELS = {
    "de": {
        "title": "Sponsor Match",
        "caption": "Sponsoring-Fit-Analyse für Sportvereine",
        "settings": "Einstellungen",
        "model_label": "LLM Modell",
        "model_label_help": "Wählt das LLM-Modell für Recherche & Bewertung; beeinflusst Qualität und Kosten.",
        "feedback_stats": "Feedback-Statistik",
        "feedback_stats_help": "Zeigt Agent-Performance: Accuracy, gesammeltes Feedback und Agent Confidence.",
        "total_analyses": "Gesamte Analysen",
        "positive_feedback": "Positive Feedbacks",
        "negative_feedback": "Negative Feedbacks",
        "accuracy": "Accuracy",
        "agent_confidence": "Agent Confidence",
        "agent_confidence_caption": "basierend auf {count} Feedbacks",
        "club_select": "Verein auswählen",
        "company_input": "Firma, die geprüft werden soll",
        "company_placeholder": "z.B. Red Bull, Nike, Adidas",
        "company_hint": "Diese App analysiert Sponsoring-Fits. Gib einen Firmennamen ein (z.B. 'Red Bull', 'Nike', 'Adidas').",
        "start_button": "Analyse starten",
        "warning_no_company": "Bitte gib eine Firma ein.",
        "warning_is_question": "Das ist eine Frage, keine Firma. Gib einen Firmennamen ein.",
        "warning_invalid_company": "Ungültiger Firmenname. Nur alphanumerische Zeichen erlaubt.",
        "warning_rate_limit": "Rate-Limit erreicht. Versuche es in {minutes:.0f} Minuten erneut.",
        "developer_settings_header": "Developer Settings",
        "developer_settings_help": "Erweiterte Einstellungen wie Modellwahl – für Poweruser gedacht.",
        "plugin_manager_header": "Plugin Manager",
        "plugin_manager_info": "Toggle Plugins ein/aus, um die Agent-Performance zu customizen.",
        "plugin_manager_active_count": "Aktive Plugins: {active}/{total}",
        "plugin_required_badge": "Required",
        "user_settings_header": "Einstellungen",
        "user_settings_help": "Bevorzugte Sprache, Lieblingsvereine und Account-Löschung verwalten.",
        "preferred_language_label": "Bevorzugte Sprache",
        "favorite_clubs_label": "Lieblingsvereine",
        "delete_account_confirm_checkbox": "Ich bin sicher, dass ich meinen Account unwiderruflich löschen möchte.",
        "delete_account_button": "Account löschen",
        "spinner_text": "Analysiere...",
        "research_header": "Recherche-Ergebnisse",
        "research_help": "Web-Recherche zum Unternehmen – Branche, Zielgruppe, Markenwerte.",
        "research_more_btn": "Mehr anzeigen",
        "research_less_btn": "Weniger anzeigen",
        "company_intel_header": "Company Intelligence (via Companies House API)",
        "company_intel_help": "Offizielle Firmendaten von Companies House (nur UK-Firmen).",
        "company_intel_disclaimer": "Deckt nur UK-registrierte Firmen ab (Datenquelle: UK Companies House).",
        "company_intel_none": "{message}",
        "company_intel_number": "Company Number",
        "company_intel_status": "Status",
        "company_intel_founded": "Founded",
        "company_intel_address": "Address",
        "company_intel_type": "Company Type",
        "competitor_analysis_header": "Konkurrenzanalyse",
        "competitor_analysis_help": "Analysiert das bestehende Sponsoring-Portfolio DIESER Firma (nicht ihrer Konkurrenten) und den Zielgruppen-Fit zum Verein – generisch für jede Firma, fließt in die Fit-Bewertung ein.",
        "competitor_analysis_none": "Keine Konkurrenzanalyse verfügbar (Plugin war deaktiviert oder keine Web-Daten gefunden).",
        "portfolio_disclaimer": "KI-geschätzte Werte (Web-Suche + LLM-Interpretation) – keine verifizierten Marktdaten.",
        "portfolio_categories_label": "Top-Kategorien",
        "portfolio_active_count_label": "Aktive Sponsorings (geschätzt)",
        "portfolio_audience_label": "Typische Zielgruppe",
        "audience_fit_header": "Vergleich mit {club}",
        "audience_fit_label": "Passt Zielgruppe?",
        "audience_fit_yes": "Ja",
        "audience_fit_no": "Nein",
        "audience_fit_partial": "Teilweise",
        "match_percent_label": "Match-Prozent",
        "saturation_same_sport_label": "Sponsorings in {sport}",
        "saturation_level_label": "Sättigungslevel",
        "saturation_level_low": "Niedrig",
        "saturation_level_medium": "Mittel",
        "saturation_level_high": "Hoch",
        "saturation_level_extreme": "Extrem",
        "competitor_impact_header": "Impact auf Fit-Score",
        "competitor_impact_help": "Score-Anpassung basierend auf Zielgruppen-Match UND Sättigung dieser Firma in dieser Sportart – variiert dynamisch je nach Firma, kein fixer Wert.",
        "competitor_impact_adjusted": "Score angepasst von {before:.2f} → {after:.2f}: {match_percent}% Zielgruppen-Match und {saturation} Sättigung von {company} in {sport}.",
        "competitor_impact_none": "Kein Score-Impact (neutraler Match/Sättigung oder Score aus Cache übernommen).",
        "market_saturation_header": "Market Saturation",
        "market_saturation_interpretation_low": "Niedrig saturiert – viel Freiraum für neue Sponsorings in dieser Sportart.",
        "market_saturation_interpretation_medium": "Mäßig saturiert – noch Raum für weitere Sponsorings.",
        "market_saturation_interpretation_high": "Hoch saturiert – wenig Platz für weitere Sponsorings in dieser Sportart.",
        "market_saturation_interpretation_extreme": "Extrem saturiert – Firma ist bereits stark in dieser Sportart engagiert.",
        "case_studies_header": "Relevante Case-Studies aus KB",
        "case_studies_help": "Ähnliche, frühere Sponsoring-Fälle aus der Wissensdatenbank zum Vergleich.",
        "case_studies_disclaimer": "Fiktive Beispieldaten zu Demozwecken – keine realen Sponsoring-Fälle.",
        "case_studies_none": "Keine passenden Case-Studies in der Wissensbasis gefunden.",
        "case_studies_fallback": "Keine Case-Study zu '{company}' gefunden – hier stattdessen Beispiele aus derselben Sportart:",
        "case_study_success": "Erfolgreich",
        "case_study_no_success": "Kein Erfolg",
        "brand_fit_high": "Hoch",
        "brand_fit_medium": "Mittel",
        "brand_fit_low": "Niedrig",
        "sponsorship_db_header": "Externe Sponsorship DB",
        "sponsorship_db_help": "Historische Sponsoring-Deals aus einer externen Datenbank als Referenz.",
        "sponsorship_db_disclaimer": "Fiktive, aber realistische Beispieldaten – keine realen Sponsoring-Fälle.",
        "sponsorship_db_none": "Keine historischen Daten zu dieser Company in der DB gefunden",
        "sponsorship_db_entry": "{company} sponserte {team} ({year}) - {metric}",
        "budget_estimator_header": "Budget Estimator",
        "budget_estimator_help": "Grobe Budget-Schätzung, datenbasiert aus ähnlichen Fällen der externen Sponsorship-DB – kein LLM-Call.",
        "budget_estimator_none": "Keine Budget-Schätzung verfügbar (Plugin war bei dieser Analyse deaktiviert).",
        "fit_header": "Fit-Bewertung",
        "fit_help": "Bewertet, wie gut Firma und Verein zusammenpassen (Score 0–1) mit Begründung.",
        "score_label": "Score",
        "outreach_header": "Entwurf: Erstansprache",
        "outreach_help": "Vom Agenten formulierter Vorschlag für die erste Kontaktaufnahme.",
        "rejection_header": "Ablehnung",
        "rejection_help": "Begründung, warum der Fit für ein Sponsoring nicht ausreicht.",
        "tokens_header": "Token-Verbrauch & Kosten",
        "tokens_help": "Zeigt verbrauchte Tokens und geschätzte Kosten dieser Analyse.",
        "total_tokens": "Gesamt Tokens",
        "estimated_cost": "Geschätzte Kosten",
        "details_expander": "Details pro Schritt",
        "model_caption": "Modell: {model}",
        "feedback_question": "War diese Analyse hilfreich?",
        "feedback_question_help": "Agent lernt aus deinem Feedback und verbessert sich.",
        "feedback_positive_btn": "Guter Fit - Agent hat gut bewertet",
        "feedback_negative_btn": "Nicht hilfreich - Agent hat falsch bewertet",
        "feedback_thanks_positive": "Dank für Feedback! Agent lernt und wird besser.",
        "feedback_thanks_negative": "Dank für Feedback! Agent passt sich an.",
        "learning_badge": "Agent lernt aus Feedback",
        "trace_header": "Agent Trace & Debugging",
        "trace_none": "Kein Trace verfügbar – LangSmith-Tracing lief für diesen Lauf nicht mit.",
        "trace_link": "LangSmith Trace ansehen",
        "trace_fallback": (
            "Trace-Link konnte nicht erzeugt werden (LANGSMITH_API_KEY fehlt/ungültig "
            "oder Trace ist noch nicht verarbeitet). Run-ID: `{run_id}` – schau im "
            "Projekt **{project}** auf https://smith.langchain.com nach."
        ),
        "help_header": "Hilfe & Guide",
        "help_header_help": "FAQ und Chat-Hilfe zur Nutzung der App.",
        "chat_subheader": "Hast du eine andere Frage?",
        "chat_placeholder": "Hast du eine Frage?",
        "chat_fallback": "Gute Frage! Schau dir die FAQ oben an oder kontaktiere den Support.",
        "history_header": "Analyseverlauf",
        "history_header_help": "Frühere Analysen durchsuchen und filtern (Verein, Firma, Feedback).",
        "history_filter_club": "Verein",
        "history_filter_company": "Firma",
        "history_filter_feedback": "Feedback",
        "history_filter_all": "Alle",
        "history_empty": "Noch keine Analysen vorhanden.",
        "history_detail_research": "Recherche-Zusammenfassung:",
        "history_detail_meta": "Sprache: {language} · Modell: {model} · {timestamp}",
        "agent_eval_header": "Agent Evaluation",
        "agent_eval_header_help": "Automatische Qualitätsprüfung: Konsistenz der Agent-Scores bei wiederholten Anfragen.",
        "generate_report_btn": "RAGAs Report generieren",
        "eval_spinner": "Bewerte vergangene Analysen...",
        "eval_no_data": "Noch keine Analysen zum Bewerten vorhanden.",
        "eval_num_analyzed": "{count} Analysen bewertet",
        "eval_overall_relevance": "Relevance",
        "eval_overall_faithfulness": "Faithfulness",
        "eval_overall_answer_relevance": "Answer Relevance",
        "eval_trend_header": "Trend über Zeit",
        "eval_trend_improving": "Verbessert sich (+{delta:.1%})",
        "eval_trend_declining": "Verschlechtert sich ({delta:.1%})",
        "eval_trend_stable": "Stabil ({delta:+.1%})",
        "eval_trend_insufficient": "Noch zu wenige Analysen für einen Trend (mind. 4 nötig).",
        "eval_chart_header": "Agent vs. Zufalls-Baseline",
        "eval_cost_caption": "Kosten dieser Bewertung: ${cost:.4f} ({tokens} Tokens, Modell {model})",
        "eval_download_json": "Report als JSON",
        "eval_download_csv": "Report als CSV",
        "eval_note": "Hinweis: nutzt eine eigene LLM-Judge-Bewertung (siehe Code-Kommentar) statt der ragas-Bibliothek, da diese mit der installierten langchain-community-Version nicht kompatibel ist.",
        "score_consistency": "Score Consistency",
        "score_consistency_caption": "basierend auf {count} wiederholten Firma+Verein-Analysen",
        "score_consistency_none": "Noch keine wiederholten Analysen zur selben Firma+Verein-Kombination.",
        "sidebar_group_account": "Account",
        "sidebar_group_settings": "Einstellungen",
        "sidebar_group_actions": "Aktionen",
        "sidebar_group_analytics": "Analytics",
        "logged_in_as": "Angemeldet als:",
        "logout_button": "Abmelden",
        "clear_chat_button": "Chat löschen",
        "show_analytics_checkbox": "Analytics Dashboard anzeigen",
        "sidebar_group_history": "Verlauf & Hilfe",
        "quicklinks_header": "Quick Links",
        "quicklinks_help": "Direktlink zum LangSmith-Trace für Debugging der Analyse.",
    },
    "en": {
        "title": "Sponsor Match",
        "caption": "Sponsorship fit analysis for sports clubs",
        "settings": "Settings",
        "model_label": "LLM Model",
        "model_label_help": "Selects the LLM model for research & evaluation; affects quality and cost.",
        "feedback_stats": "Feedback Stats",
        "feedback_stats_help": "Shows agent performance: accuracy, collected feedback, and agent confidence.",
        "total_analyses": "Total Analyses",
        "positive_feedback": "Positive Feedback",
        "negative_feedback": "Negative Feedback",
        "accuracy": "Accuracy",
        "agent_confidence": "Agent Confidence",
        "agent_confidence_caption": "based on {count} feedback entries",
        "club_select": "Select Club",
        "company_input": "Company to evaluate",
        "company_placeholder": "e.g. Red Bull, Nike, Adidas",
        "company_hint": "This app analyzes sponsorship fits. Enter a company name (e.g. 'Red Bull', 'Nike', 'Adidas').",
        "start_button": "Start Analysis",
        "warning_no_company": "Please enter a company.",
        "warning_is_question": "That's a question, not a company. Please enter a company name.",
        "warning_invalid_company": "Invalid company name. Only alphanumeric characters allowed.",
        "warning_rate_limit": "Rate limit exceeded. Try again in {minutes:.0f} minutes.",
        "developer_settings_header": "Developer Settings",
        "developer_settings_help": "Advanced settings like model choice – for power users.",
        "plugin_manager_header": "Plugin Manager",
        "plugin_manager_info": "Toggle plugins on/off to customize agent performance.",
        "plugin_manager_active_count": "Active plugins: {active}/{total}",
        "plugin_required_badge": "Required",
        "user_settings_header": "Settings",
        "user_settings_help": "Manage preferred language, favorite clubs, and account deletion.",
        "preferred_language_label": "Preferred language",
        "favorite_clubs_label": "Favorite clubs",
        "delete_account_confirm_checkbox": "I'm sure I want to permanently delete my account.",
        "delete_account_button": "Delete account",
        "spinner_text": "Analyzing...",
        "research_header": "Research Findings",
        "research_help": "Web research on the company – industry, target audience, brand values.",
        "research_more_btn": "Show more",
        "research_less_btn": "Show less",
        "company_intel_header": "Company Intelligence (via Companies House API)",
        "company_intel_help": "Official company data from Companies House (UK companies only).",
        "company_intel_disclaimer": "Covers UK-registered companies only (data source: UK Companies House).",
        "company_intel_none": "{message}",
        "company_intel_number": "Company Number",
        "company_intel_status": "Status",
        "company_intel_founded": "Founded",
        "company_intel_address": "Address",
        "company_intel_type": "Company Type",
        "competitor_analysis_header": "Competitor Analysis",
        "competitor_analysis_help": "Analyzes this company's OWN existing sponsorship portfolio (not its competitors') and its audience fit with the club – generic for any company, factored into the fit evaluation.",
        "competitor_analysis_none": "No competitor analysis available (plugin was disabled or no web data found).",
        "portfolio_disclaimer": "AI-estimated values (web search + LLM interpretation) – not verified market data.",
        "portfolio_categories_label": "Top categories",
        "portfolio_active_count_label": "Active sponsorships (estimated)",
        "portfolio_audience_label": "Typical target audience",
        "audience_fit_header": "Comparison with {club}",
        "audience_fit_label": "Does the audience fit?",
        "audience_fit_yes": "Yes",
        "audience_fit_no": "No",
        "audience_fit_partial": "Partial",
        "match_percent_label": "Match percentage",
        "saturation_same_sport_label": "Sponsorships in {sport}",
        "saturation_level_label": "Saturation level",
        "saturation_level_low": "Low",
        "saturation_level_medium": "Medium",
        "saturation_level_high": "High",
        "saturation_level_extreme": "Extreme",
        "competitor_impact_header": "Impact on Fit Score",
        "competitor_impact_help": "Score adjustment based on audience match AND this company's saturation in this sport – varies dynamically per company, not a fixed value.",
        "competitor_impact_adjusted": "Score adjusted from {before:.2f} to {after:.2f}: {match_percent}% audience match and {saturation} saturation of {company} in {sport}.",
        "competitor_impact_none": "No score impact (neutral match/saturation or score reused from cache).",
        "market_saturation_header": "Market Saturation",
        "market_saturation_interpretation_low": "Low saturation – plenty of room for new sponsorships in this sport.",
        "market_saturation_interpretation_medium": "Moderately saturated – still room for more sponsorships.",
        "market_saturation_interpretation_high": "Highly saturated – little room for further sponsorships in this sport.",
        "market_saturation_interpretation_extreme": "Extremely saturated – company is already heavily engaged in this sport.",
        "case_studies_header": "Relevant Case Studies from KB",
        "case_studies_help": "Similar past sponsorship cases from the knowledge base for comparison.",
        "case_studies_disclaimer": "Fictional sample data for demo purposes – not real sponsorship cases.",
        "case_studies_none": "No matching case studies found in the knowledge base.",
        "case_studies_fallback": "No case study found for '{company}' – showing examples from the same sport instead:",
        "case_study_success": "Successful",
        "case_study_no_success": "Not successful",
        "brand_fit_high": "High",
        "brand_fit_medium": "Medium",
        "brand_fit_low": "Low",
        "sponsorship_db_header": "External Sponsorship DB",
        "sponsorship_db_help": "Historical sponsorship deals from an external database as reference.",
        "sponsorship_db_disclaimer": "Fictional but realistic sample data – not real sponsorship cases.",
        "sponsorship_db_none": "No historical data for this company found in the DB",
        "sponsorship_db_entry": "{company} sponsored {team} ({year}) - {metric}",
        "budget_estimator_header": "Budget Estimator",
        "budget_estimator_help": "Rough budget estimate, data-based from similar cases in the external sponsorship DB – no LLM call.",
        "budget_estimator_none": "No budget estimate available (plugin was disabled for this analysis).",
        "fit_header": "Fit Evaluation",
        "fit_help": "Rates how well company and club match (score 0–1) with reasoning.",
        "score_label": "Score",
        "outreach_header": "Draft: Initial Outreach",
        "outreach_help": "Agent-drafted suggestion for the initial outreach message.",
        "rejection_header": "Rejection",
        "rejection_help": "Explanation of why the fit isn't sufficient for a sponsorship.",
        "tokens_header": "Token Usage & Cost",
        "tokens_help": "Shows tokens consumed and estimated cost of this analysis.",
        "total_tokens": "Total Tokens",
        "estimated_cost": "Estimated Cost",
        "details_expander": "Details per step",
        "model_caption": "Model: {model}",
        "feedback_question": "Was this analysis helpful?",
        "feedback_question_help": "The agent learns from your feedback and improves.",
        "feedback_positive_btn": "Good Fit - Agent evaluated well",
        "feedback_negative_btn": "Not helpful - Agent evaluated incorrectly",
        "feedback_thanks_positive": "Thanks for the feedback! The agent is learning and improving.",
        "feedback_thanks_negative": "Thanks for the feedback! The agent is adapting.",
        "learning_badge": "Agent learns from feedback",
        "trace_header": "Agent Trace & Debugging",
        "trace_none": "No trace available – LangSmith tracing did not run for this analysis.",
        "trace_link": "View LangSmith Trace",
        "trace_fallback": (
            "Trace link could not be generated (LANGSMITH_API_KEY missing/invalid or "
            "trace not yet processed). Run ID: `{run_id}` – check project "
            "**{project}** at https://smith.langchain.com."
        ),
        "help_header": "Help & Guide",
        "help_header_help": "FAQ and chat help for using the app.",
        "chat_subheader": "Have another question?",
        "chat_placeholder": "Have a question?",
        "chat_fallback": "Good question! Check the FAQ above or contact support.",
        "history_header": "Analysis History",
        "history_header_help": "Browse and filter past analyses (club, company, feedback).",
        "history_filter_club": "Club",
        "history_filter_company": "Company",
        "history_filter_feedback": "Feedback",
        "history_filter_all": "All",
        "history_empty": "No analyses yet.",
        "history_detail_research": "Research summary:",
        "history_detail_meta": "Language: {language} · Model: {model} · {timestamp}",
        "agent_eval_header": "Agent Evaluation",
        "agent_eval_header_help": "Automatic quality check: how consistent the agent scores on repeated requests.",
        "generate_report_btn": "Generate RAGAs Report",
        "eval_spinner": "Evaluating past analyses...",
        "eval_no_data": "No analyses to evaluate yet.",
        "eval_num_analyzed": "{count} analyses evaluated",
        "eval_overall_relevance": "Relevance",
        "eval_overall_faithfulness": "Faithfulness",
        "eval_overall_answer_relevance": "Answer Relevance",
        "eval_trend_header": "Trend over time",
        "eval_trend_improving": "Improving (+{delta:.1%})",
        "eval_trend_declining": "Declining ({delta:.1%})",
        "eval_trend_stable": "Stable ({delta:+.1%})",
        "eval_trend_insufficient": "Not enough analyses yet for a trend (at least 4 needed).",
        "eval_chart_header": "Agent vs. Random Baseline",
        "eval_cost_caption": "Cost of this evaluation: ${cost:.4f} ({tokens} tokens, model {model})",
        "eval_download_json": "Report as JSON",
        "eval_download_csv": "Report as CSV",
        "eval_note": "Note: uses a custom LLM-judge implementation (see code comment) instead of the ragas library, which is incompatible with the installed langchain-community version.",
        "score_consistency": "Score Consistency",
        "score_consistency_caption": "based on {count} repeated company+club analyses",
        "score_consistency_none": "No repeated analyses of the same company+club combination yet.",
        "sidebar_group_account": "Account",
        "sidebar_group_settings": "Settings",
        "sidebar_group_actions": "Actions",
        "sidebar_group_analytics": "Analytics",
        "logged_in_as": "Logged in as:",
        "logout_button": "Logout",
        "clear_chat_button": "Clear chat",
        "show_analytics_checkbox": "Show Analytics Dashboard",
        "sidebar_group_history": "History & Help",
        "quicklinks_header": "Quick Links",
        "quicklinks_help": "Direct link to the LangSmith trace for debugging the analysis.",
    },
    "fr": {
        "title": "Sponsor Match",
        "caption": "Analyse de compatibilité de sponsoring pour clubs sportifs",
        "settings": "Paramètres",
        "model_label": "Modèle LLM",
        "model_label_help": "Sélectionne le modèle LLM pour la recherche et l'évaluation ; influence qualité et coût.",
        "feedback_stats": "Statistiques de feedback",
        "feedback_stats_help": "Montre la performance de l'agent : précision, feedback collecté et confiance de l'agent.",
        "total_analyses": "Analyses totales",
        "positive_feedback": "Feedbacks positifs",
        "negative_feedback": "Feedbacks négatifs",
        "accuracy": "Précision",
        "agent_confidence": "Confiance de l'agent",
        "agent_confidence_caption": "basé sur {count} retours",
        "club_select": "Choisir un club",
        "company_input": "Entreprise à évaluer",
        "company_placeholder": "p.ex. Red Bull, Nike, Adidas",
        "company_hint": "Cette application analyse l'adéquation des sponsorings. Saisissez un nom d'entreprise (p.ex. 'Red Bull', 'Nike', 'Adidas').",
        "start_button": "Démarrer l'analyse",
        "warning_no_company": "Veuillez saisir une entreprise.",
        "warning_is_question": "Ceci est une question, pas une entreprise. Veuillez saisir un nom d'entreprise.",
        "warning_invalid_company": "Nom d'entreprise invalide. Seuls les caractères alphanumériques sont autorisés.",
        "warning_rate_limit": "Limite de requêtes atteinte. Réessayez dans {minutes:.0f} minutes.",
        "developer_settings_header": "Paramètres développeur",
        "developer_settings_help": "Paramètres avancés comme le choix du modèle – pour utilisateurs avancés.",
        "plugin_manager_header": "Gestionnaire de plugins",
        "plugin_manager_info": "Activez/désactivez des plugins pour personnaliser la performance de l'agent.",
        "plugin_manager_active_count": "Plugins actifs : {active}/{total}",
        "plugin_required_badge": "Requis",
        "user_settings_header": "Paramètres",
        "user_settings_help": "Gérez la langue préférée, les clubs favoris et la suppression du compte.",
        "preferred_language_label": "Langue préférée",
        "favorite_clubs_label": "Clubs favoris",
        "delete_account_confirm_checkbox": "Je suis sûr(e) de vouloir supprimer définitivement mon compte.",
        "delete_account_button": "Supprimer le compte",
        "spinner_text": "Analyse en cours...",
        "research_header": "Résultats de recherche",
        "research_help": "Recherche web sur l'entreprise – secteur, public cible, valeurs de marque.",
        "research_more_btn": "Afficher plus",
        "research_less_btn": "Afficher moins",
        "company_intel_header": "Company Intelligence (via l'API Companies House)",
        "company_intel_help": "Données officielles de l'entreprise via Companies House (entreprises UK uniquement).",
        "company_intel_disclaimer": "Couvre uniquement les entreprises enregistrées au Royaume-Uni (source : UK Companies House).",
        "company_intel_none": "{message}",
        "company_intel_number": "Numéro d'entreprise",
        "company_intel_status": "Statut",
        "company_intel_founded": "Fondée en",
        "company_intel_address": "Adresse",
        "company_intel_type": "Forme juridique",
        "competitor_analysis_header": "Analyse de la concurrence",
        "competitor_analysis_help": "Analyse le portefeuille de sponsoring existant DE CETTE entreprise (pas de ses concurrents) et l'adéquation d'audience avec le club – générique pour toute entreprise, pris en compte dans l'évaluation du fit.",
        "competitor_analysis_none": "Aucune analyse de la concurrence disponible (plugin désactivé ou aucune donnée web trouvée).",
        "portfolio_disclaimer": "Valeurs estimées par IA (recherche web + interprétation LLM) – données de marché non vérifiées.",
        "portfolio_categories_label": "Catégories principales",
        "portfolio_active_count_label": "Sponsorings actifs (estimé)",
        "portfolio_audience_label": "Public cible typique",
        "audience_fit_header": "Comparaison avec {club}",
        "audience_fit_label": "Le public cible correspond-il ?",
        "audience_fit_yes": "Oui",
        "audience_fit_no": "Non",
        "audience_fit_partial": "Partiellement",
        "match_percent_label": "Pourcentage de correspondance",
        "saturation_same_sport_label": "Sponsorings dans {sport}",
        "saturation_level_label": "Niveau de saturation",
        "saturation_level_low": "Faible",
        "saturation_level_medium": "Moyen",
        "saturation_level_high": "Élevé",
        "saturation_level_extreme": "Extrême",
        "competitor_impact_header": "Impact sur le Fit Score",
        "competitor_impact_help": "Ajustement du score basé sur la correspondance d'audience ET la saturation de cette entreprise dans ce sport – varie dynamiquement selon l'entreprise, pas une valeur fixe.",
        "competitor_impact_adjusted": "Score ajusté de {before:.2f} à {after:.2f} : {match_percent}% de correspondance d'audience et saturation {saturation} de {company} dans {sport}.",
        "competitor_impact_none": "Aucun impact sur le score (correspondance/saturation neutre ou score repris du cache).",
        "market_saturation_header": "Saturation du marché",
        "market_saturation_interpretation_low": "Faible saturation – beaucoup de place pour de nouveaux sponsorings dans ce sport.",
        "market_saturation_interpretation_medium": "Saturation modérée – encore de la place pour d'autres sponsorings.",
        "market_saturation_interpretation_high": "Forte saturation – peu de place pour d'autres sponsorings dans ce sport.",
        "market_saturation_interpretation_extreme": "Saturation extrême – l'entreprise est déjà fortement engagée dans ce sport.",
        "case_studies_header": "Études de cas pertinentes (base de connaissances)",
        "case_studies_help": "Cas de sponsoring similaires issus de la base de connaissances, à titre de comparaison.",
        "case_studies_disclaimer": "Données fictives à des fins de démonstration – pas de cas de sponsoring réels.",
        "case_studies_none": "Aucune étude de cas correspondante trouvée dans la base de connaissances.",
        "case_studies_fallback": "Aucune étude de cas trouvée pour '{company}' – voici à la place des exemples du même sport :",
        "case_study_success": "Réussi",
        "case_study_no_success": "Pas de réussite",
        "brand_fit_high": "Élevé",
        "brand_fit_medium": "Moyen",
        "brand_fit_low": "Faible",
        "sponsorship_db_header": "Base de sponsoring externe",
        "sponsorship_db_help": "Deals de sponsoring historiques issus d'une base de données externe, à titre de référence.",
        "sponsorship_db_disclaimer": "Données fictives mais réalistes – pas de cas de sponsoring réels.",
        "sponsorship_db_none": "Aucune donnée historique trouvée pour cette entreprise dans la base",
        "sponsorship_db_entry": "{company} a sponsorisé {team} ({year}) - {metric}",
        "budget_estimator_header": "Budget Estimator",
        "budget_estimator_help": "Estimation budgétaire approximative, basée sur des cas similaires de la base de sponsoring externe – sans appel LLM.",
        "budget_estimator_none": "Aucune estimation de budget disponible (plugin désactivé pour cette analyse).",
        "fit_header": "Évaluation du fit",
        "fit_help": "Évalue à quel point l'entreprise et le club correspondent (score 0–1) avec justification.",
        "score_label": "Score",
        "outreach_header": "Brouillon : prise de contact initiale",
        "outreach_help": "Proposition rédigée par l'agent pour le premier message de prise de contact.",
        "rejection_header": "Rejet",
        "rejection_help": "Explique pourquoi l'adéquation est insuffisante pour un sponsoring.",
        "tokens_header": "Utilisation des tokens & coûts",
        "tokens_help": "Montre les tokens consommés et le coût estimé de cette analyse.",
        "total_tokens": "Total des tokens",
        "estimated_cost": "Coût estimé",
        "details_expander": "Détails par étape",
        "model_caption": "Modèle : {model}",
        "feedback_question": "Cette analyse vous a-t-elle été utile ?",
        "feedback_question_help": "L'agent apprend de votre feedback et s'améliore.",
        "feedback_positive_btn": "Bon fit - L'agent a bien évalué",
        "feedback_negative_btn": "Pas utile - L'agent s'est trompé",
        "feedback_thanks_positive": "Merci pour votre retour ! L'agent apprend et s'améliore.",
        "feedback_thanks_negative": "Merci pour votre retour ! L'agent s'adapte.",
        "learning_badge": "L'agent apprend du feedback",
        "trace_header": "Trace de l'agent & débogage",
        "trace_none": "Aucune trace disponible – le tracing LangSmith n'a pas fonctionné pour cette analyse.",
        "trace_link": "Voir la trace LangSmith",
        "trace_fallback": (
            "Impossible de générer le lien de trace (LANGSMITH_API_KEY manquant/invalide "
            "ou trace pas encore traitée). ID de run : `{run_id}` – consultez le projet "
            "**{project}** sur https://smith.langchain.com."
        ),
        "help_header": "Aide & Guide",
        "help_header_help": "FAQ et aide par chat pour l'utilisation de l'application.",
        "chat_subheader": "Vous avez une autre question ?",
        "chat_placeholder": "Vous avez une question ?",
        "chat_fallback": "Bonne question ! Consultez la FAQ ci-dessus ou contactez le support.",
        "history_header": "Historique des analyses",
        "history_header_help": "Parcourez et filtrez les analyses précédentes (club, entreprise, feedback).",
        "history_filter_club": "Club",
        "history_filter_company": "Entreprise",
        "history_filter_feedback": "Feedback",
        "history_filter_all": "Tous",
        "history_empty": "Aucune analyse pour l'instant.",
        "history_detail_research": "Résumé de recherche :",
        "history_detail_meta": "Langue : {language} · Modèle : {model} · {timestamp}",
        "agent_eval_header": "Agent Evaluation",
        "agent_eval_header_help": "Contrôle qualité automatique : cohérence des scores de l'agent sur des demandes répétées.",
        "generate_report_btn": "Générer le rapport RAGAs",
        "eval_spinner": "Évaluation des analyses passées...",
        "eval_no_data": "Aucune analyse à évaluer pour l'instant.",
        "eval_num_analyzed": "{count} analyses évaluées",
        "eval_overall_relevance": "Relevance",
        "eval_overall_faithfulness": "Faithfulness",
        "eval_overall_answer_relevance": "Answer Relevance",
        "eval_trend_header": "Tendance dans le temps",
        "eval_trend_improving": "S'améliore (+{delta:.1%})",
        "eval_trend_declining": "Se dégrade ({delta:.1%})",
        "eval_trend_stable": "Stable ({delta:+.1%})",
        "eval_trend_insufficient": "Pas encore assez d'analyses pour une tendance (4 minimum).",
        "eval_chart_header": "Agent vs. Baseline aléatoire",
        "eval_cost_caption": "Coût de cette évaluation : ${cost:.4f} ({tokens} tokens, modèle {model})",
        "eval_download_json": "Rapport en JSON",
        "eval_download_csv": "Rapport en CSV",
        "eval_note": "Remarque : utilise une évaluation LLM-judge maison (voir commentaire dans le code) au lieu de la bibliothèque ragas, incompatible avec la version installée de langchain-community.",
        "score_consistency": "Score Consistency",
        "score_consistency_caption": "basé sur {count} analyses répétées entreprise+club",
        "score_consistency_none": "Pas encore d'analyses répétées pour la même combinaison entreprise+club.",
        "sidebar_group_account": "Compte",
        "sidebar_group_settings": "Paramètres",
        "sidebar_group_actions": "Actions",
        "sidebar_group_analytics": "Analytics",
        "logged_in_as": "Connecté en tant que :",
        "logout_button": "Déconnexion",
        "clear_chat_button": "Effacer le chat",
        "show_analytics_checkbox": "Afficher le tableau de bord Analytics",
        "sidebar_group_history": "Historique & Aide",
        "quicklinks_header": "Liens rapides",
        "quicklinks_help": "Lien direct vers la trace LangSmith pour déboguer l'analyse.",
    },
}

FEEDBACK_DISPLAY = {
    "de": {"positive": "Positiv", "negative": "Negativ", "none": "Kein Feedback"},
    "en": {"positive": "Positive", "negative": "Negative", "none": "No feedback"},
    "fr": {"positive": "Positif", "negative": "Négatif", "none": "Pas de feedback"},
}


def format_relative_time(timestamp: str, language: str) -> str:
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    delta_days = (datetime.now() - ts).days
    if delta_days <= 0:
        return {"de": "heute", "en": "today", "fr": "aujourd'hui"}[language]
    if delta_days == 1:
        return {"de": "gestern", "en": "yesterday", "fr": "hier"}[language]
    template = {"de": "vor {n} Tagen", "en": "{n} days ago", "fr": "il y a {n} jours"}[language]
    return template.format(n=delta_days)

FAQ = {
    "de": {
        "how_it_works": {
            "question": "Wie funktioniert Sponsor Match?",
            "answer": (
                "Sponsor Match recherchiert die eingegebene Firma automatisiert im Web, "
                "bewertet mit einem LLM den Marken-Fit anhand des Vereinsprofils und "
                "ähnlicher Case-Studies, und erstellt je nach Fit-Score entweder einen "
                "Ansprache-Entwurf oder eine Ablehnungsbegründung."
            ),
            "keywords": ["funktioniert", "ablauf", "sponsor match", "prozess", "wie"],
        },
        "fit_score": {
            "question": "Was bedeutet der Fit-Score?",
            "answer": (
                "Der Fit-Score (0.0–1.0) zeigt, wie gut eine Firma als Sponsor zum Verein "
                "passt – basierend auf Werten, Zielgruppe und gesuchten Sponsoring-Bereichen. "
                "Ab 0.6 wird automatisch ein Ansprache-Entwurf erstellt, darunter eine "
                "Ablehnungsbegründung."
            ),
            "keywords": ["fit-score", "score", "bewertung", "punktzahl"],
        },
        "case_studies": {
            "question": "Wie werden die Case-Studies genutzt?",
            "answer": (
                "Vor der Bewertung durchsucht der Agent eine kleine Wissensbasis mit "
                "fiktiven Sponsoring-Case-Studies nach Beispielen zur gesuchten Firma oder "
                "Sportart und gibt sie dem LLM als Zusatzkontext. Es sind Demodaten, keine "
                "echten Fakten."
            ),
            "keywords": ["case-studies", "case studies", "wissensbasis", "kb", "rag"],
        },
        "rejection": {
            "question": "Warum wurde meine Firma abgelehnt?",
            "answer": (
                "Eine Firma wird abgelehnt, wenn der Fit-Score unter 0.6 liegt – meist weil "
                "Zielgruppe, Werte oder Sponsoring-Bereiche nicht gut zum Verein passen. Die "
                "genaue Begründung steht im Abschnitt 'Fit-Bewertung'."
            ),
            "keywords": ["abgelehnt", "ablehnung", "rejection"],
        },
        "feedback_change": {
            "question": "Kann ich mein Feedback ändern?",
            "answer": (
                "Nein, pro Analyse kannst du nur einmal Feedback geben (positiv oder negativ) – "
                "danach sind die Buttons deaktiviert. Starte eine neue Analyse, um erneut "
                "Feedback zu geben."
            ),
            "keywords": ["feedback", "ändern", "zurücknehmen"],
        },
        "which_llm": {
            "question": "Welches LLM soll ich nutzen?",
            "answer": (
                "gpt-4o-mini ist die günstigste Wahl für schnelle Tests. gpt-4o und "
                "gpt-4-turbo liefern differenziertere Bewertungen, kosten aber deutlich mehr. "
                "claude-3-5-sonnet ist eine gute Alternative mit starker Textqualität zu "
                "mittlerem Preis."
            ),
            "keywords": ["llm", "modell", "gpt", "claude"],
        },
    },
    "en": {
        "how_it_works": {
            "question": "How does Sponsor Match work?",
            "answer": (
                "Sponsor Match automatically researches the entered company on the web, "
                "uses an LLM to evaluate the brand fit based on the club profile and "
                "similar case studies, and then generates either an outreach draft or a "
                "rejection reason depending on the fit score."
            ),
            "keywords": ["work", "works", "process", "sponsor match", "how"],
        },
        "fit_score": {
            "question": "What does the fit score mean?",
            "answer": (
                "The fit score (0.0–1.0) shows how well a company fits as a sponsor for the "
                "club, based on values, target audience, and sponsorship gaps. From 0.6 "
                "upward, an outreach draft is generated automatically; below that, a "
                "rejection reason."
            ),
            "keywords": ["fit score", "score", "evaluation", "rating"],
        },
        "case_studies": {
            "question": "How are the case studies used?",
            "answer": (
                "Before evaluating, the agent searches a small knowledge base of fictional "
                "sponsorship case studies for examples matching the company or sport, and "
                "passes them to the LLM as extra context. They are demo data, not real facts."
            ),
            "keywords": ["case studies", "case-studies", "knowledge base", "kb", "rag"],
        },
        "rejection": {
            "question": "Why was my company rejected?",
            "answer": (
                "A company is rejected when the fit score is below 0.6 – usually because "
                "target audience, values, or sponsorship gaps don't align well with the "
                "club. The exact reasoning is shown in the 'Fit Evaluation' section."
            ),
            "keywords": ["rejected", "rejection", "why"],
        },
        "feedback_change": {
            "question": "Can I change my feedback?",
            "answer": (
                "No, you can only give feedback once per analysis (positive or negative) – "
                "after that the buttons are disabled. Start a new analysis to give feedback "
                "again."
            ),
            "keywords": ["feedback", "change", "undo"],
        },
        "which_llm": {
            "question": "Which LLM should I use?",
            "answer": (
                "gpt-4o-mini is the cheapest choice for quick tests. gpt-4o and gpt-4-turbo "
                "give more nuanced evaluations but cost significantly more. "
                "claude-3-5-sonnet is a good alternative with strong text quality at a "
                "medium price."
            ),
            "keywords": ["llm", "model", "gpt", "claude"],
        },
    },
    "fr": {
        "how_it_works": {
            "question": "Comment fonctionne Sponsor Match ?",
            "answer": (
                "Sponsor Match recherche automatiquement l'entreprise saisie sur le web, "
                "utilise un LLM pour évaluer l'adéquation de marque à partir du profil du "
                "club et d'études de cas similaires, puis génère soit un brouillon de prise "
                "de contact, soit une raison de rejet selon le score de fit."
            ),
            "keywords": ["fonctionne", "processus", "sponsor match", "comment"],
        },
        "fit_score": {
            "question": "Que signifie le score de fit ?",
            "answer": (
                "Le score de fit (0.0–1.0) indique dans quelle mesure une entreprise "
                "convient comme sponsor pour le club, en fonction des valeurs, du public "
                "cible et des domaines de sponsoring recherchés. À partir de 0.6, un "
                "brouillon de prise de contact est généré automatiquement, en dessous une "
                "raison de rejet."
            ),
            "keywords": ["score de fit", "score", "évaluation"],
        },
        "case_studies": {
            "question": "Comment les études de cas sont-elles utilisées ?",
            "answer": (
                "Avant l'évaluation, l'agent recherche dans une petite base de connaissances "
                "d'études de cas fictives des exemples correspondant à l'entreprise ou au "
                "sport, et les transmet au LLM comme contexte supplémentaire. Ce sont des "
                "données de démonstration, pas des faits réels."
            ),
            "keywords": ["études de cas", "base de connaissances", "kb", "rag"],
        },
        "rejection": {
            "question": "Pourquoi mon entreprise a-t-elle été rejetée ?",
            "answer": (
                "Une entreprise est rejetée lorsque le score de fit est inférieur à 0.6 – "
                "généralement parce que le public cible, les valeurs ou les domaines de "
                "sponsoring ne correspondent pas bien au club. La raison exacte se trouve "
                "dans la section « Évaluation du fit »."
            ),
            "keywords": ["rejetée", "rejet", "pourquoi"],
        },
        "feedback_change": {
            "question": "Puis-je modifier mon feedback ?",
            "answer": (
                "Non, vous ne pouvez donner votre avis qu'une seule fois par analyse (positif "
                "ou négatif) – les boutons sont ensuite désactivés. Démarrez une nouvelle "
                "analyse pour redonner votre avis."
            ),
            "keywords": ["feedback", "modifier", "changer"],
        },
        "which_llm": {
            "question": "Quel LLM devrais-je utiliser ?",
            "answer": (
                "gpt-4o-mini est le choix le plus économique pour des tests rapides. gpt-4o "
                "et gpt-4-turbo offrent des évaluations plus nuancées mais coûtent "
                "nettement plus cher. claude-3-5-sonnet est une bonne alternative avec une "
                "qualité de texte élevée à prix moyen."
            ),
            "keywords": ["llm", "modèle", "gpt", "claude"],
        },
    },
}


def match_faq_answer(user_message: str, faq: dict) -> str | None:
    """Rein lokaler Keyword-Lookup gegen die FAQ – keine LLM-Calls."""
    message_lower = user_message.lower()
    best_key, best_score = None, 0
    for key, entry in faq.items():
        score = sum(1 for kw in entry["keywords"] if kw in message_lower)
        if score > best_score:
            best_key, best_score = key, score
    return faq[best_key]["answer"] if best_key else None


def load_feedback_stats():
    positive = 0
    negative = 0
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("feedback") == "positive":
                    positive += 1
                elif entry.get("feedback") == "negative":
                    negative += 1
    return positive, negative


def save_feedback(club: str, company: str, fit_score: float, feedback: str, selected_model: str):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "club": club,
        "company": company,
        "fit_score": fit_score,
        "feedback": feedback,
        "selected_model": selected_model,
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Vereins-Profile laden (schon hier, damit die Sidebar-Filter im Verlauf sie nutzen können)
with open("data/clubs.json", "r", encoding="utf-8") as f:
    clubs = json.load(f)

with st.sidebar:
    # --- Gruppe: Account (oben) ---
    # Die Sprachauswahl selbst wird erst weiter unten instanziiert, ist aber dank
    # des expliziten `key` bereits im session_state verfügbar (vom vorherigen Rerun),
    # sodass die Account-Sektion trotz Reihenfolge in der richtigen Sprache rendert.
    _lang_probe = LABELS[
        LANGUAGE_MAP.get(st.session_state.get("ui_language_select", "Deutsch"), "de")
    ]
    sidebar_group_label(_lang_probe["sidebar_group_account"])
    logout_col1, logout_col2 = st.columns([3, 1])
    logout_col1.markdown(f"{_lang_probe['logged_in_as']} **{st.session_state['user']}**")
    if logout_col2.button(_lang_probe["logout_button"], key="logout_button"):
        for _key in ("user", "user_id", "last_activity"):
            st.session_state.pop(_key, None)
        st.rerun()
    st.divider()

    # --- Gruppe: Einstellungen (Mitte) ---
    # Fester, mehrsprachiger Label-Text, damit die Sprachauswahl selbst nicht
    # von der Sprache abhängt, die sie gerade erst festlegt.
    selected_language = st.selectbox(
        "Sprache / Language / Langue",
        options=list(LANGUAGE_MAP.keys()),
        index=0,
        key="ui_language_select",
        help="Wähle UI-Sprache und Agent-Antworten. / Choose UI language and agent responses. / "
        "Choisissez la langue de l'interface et des réponses de l'agent.",
    )
    language = LANGUAGE_MAP[selected_language]
    labels = LABELS[language]

    sidebar_group_label(labels["sidebar_group_settings"])

    # Modellwahl ist eine Developer-Einstellung, keine reguläre Nutzerinteraktion
    # (Prompt-Injection-Vermeidung: normale User können nur den Company-Namen
    # eingeben, nicht die Prompts/Konfiguration selbst beeinflussen). Eingeklappt
    # standardmäßig, damit normale User es nicht sehen müssen.
    render_help_popover(labels["developer_settings_help"])
    with st.expander(labels["developer_settings_header"], expanded=False):
        selected_model = st.selectbox(
            labels["model_label"],
            options=list(MODEL_PRICING.keys()),
            index=0,
            help=labels["model_label_help"],
        )

    render_help_popover(labels["plugin_manager_info"])
    with st.expander(labels["plugin_manager_header"], expanded=False):
        st.caption(labels["plugin_manager_info"])
        plugins = load_plugins()
        # st.session_state["enabled_plugins"] ist die Quelle der Wahrheit für die
        # Chat-Result-Anzeige weiter unten – so verschwindet z.B. die Company-
        # Intelligence-Sektion sofort, wenn das Plugin ausgeschaltet wird, ohne
        # dass die Analyse neu laufen muss.
        st.session_state["enabled_plugins"] = {p["id"]: p["enabled"] for p in plugins}
        active_count = sum(1 for p in plugins if p["enabled"])
        st.caption(labels["plugin_manager_active_count"].format(active=active_count, total=len(plugins)))

        for plugin in plugins:
            plugin_col1, plugin_col2 = st.columns([4, 1])
            with plugin_col1:
                name_line = f"**{plugin['name']}**"
                if plugin["required"]:
                    name_line += f" {labels['plugin_required_badge']}"
                st.markdown(name_line)
                st.caption(plugin["description"])
            with plugin_col2:
                new_enabled = st.toggle(
                    plugin["name"],
                    value=plugin["enabled"],
                    disabled=plugin["required"],
                    key=f"plugin_toggle_{plugin['id']}",
                    label_visibility="collapsed",
                )
            if not plugin["required"] and new_enabled != plugin["enabled"]:
                plugin["enabled"] = new_enabled
                save_plugins(plugins)
                st.session_state["enabled_plugins"][plugin["id"]] = new_enabled
                st.rerun()

    render_help_popover(labels["user_settings_help"])
    with st.expander(labels["user_settings_header"], expanded=False):
        user_settings = get_user_settings(st.session_state["user_id"])

        preferred_language_display = next(
            (name for name, code in LANGUAGE_MAP.items() if code == user_settings["language"]),
            "Deutsch",
        )
        new_preferred_language = st.selectbox(
            labels["preferred_language_label"],
            options=list(LANGUAGE_MAP.keys()),
            index=list(LANGUAGE_MAP.keys()).index(preferred_language_display),
            key="preferred_language_select",
        )

        all_club_names = [c["name"] for c in clubs.values()]
        new_favorite_clubs = st.multiselect(
            labels["favorite_clubs_label"],
            options=all_club_names,
            default=[c for c in user_settings["favorite_clubs"] if c in all_club_names],
            key="favorite_clubs_select",
        )

        if (
            LANGUAGE_MAP[new_preferred_language] != user_settings["language"]
            or new_favorite_clubs != user_settings["favorite_clubs"]
        ):
            save_user_settings(
                st.session_state["user_id"],
                user_settings["theme"],
                LANGUAGE_MAP[new_preferred_language],
                new_favorite_clubs,
            )

        delete_confirmed = st.checkbox(labels["delete_account_confirm_checkbox"], key="delete_account_confirm")
        if st.button(labels["delete_account_button"], disabled=not delete_confirmed, key="delete_account_button"):
            delete_user_account(st.session_state["user_id"])
            for _key in ("user", "user_id", "last_activity"):
                st.session_state.pop(_key, None)
            st.rerun()

    st.divider()

    # --- Gruppe: Aktionen ---
    sidebar_group_label(labels["sidebar_group_actions"])
    if st.button(labels["clear_chat_button"], key="clear_chat_button"):
        for _key in ("result", "result_meta", "feedback_given", "feedback_type"):
            st.session_state.pop(_key, None)
        st.rerun()
    show_analytics = st.checkbox(labels["show_analytics_checkbox"], key="show_analytics_checkbox")
    st.divider()

    # --- Gruppe: Analytics (nur bei Bedarf sichtbar) ---
    if show_analytics:
        sidebar_group_label(labels["sidebar_group_analytics"])
        st.header(labels["feedback_stats"], help=labels["feedback_stats_help"])
        positive_count, negative_count = load_feedback_stats()
        total_feedback = positive_count + negative_count
        st.metric(labels["total_analyses"], total_feedback)
        st.metric(labels["positive_feedback"], positive_count)
        st.metric(labels["negative_feedback"], negative_count)
        if total_feedback > 0:
            accuracy = positive_count / total_feedback * 100
            st.metric(labels["accuracy"], f"{accuracy:.0f}%")

        db_positive, db_negative = get_feedback_confidence()
        db_total = db_positive + db_negative
        if db_total > 0:
            confidence = db_positive / db_total * 100
            st.metric(labels["agent_confidence"], f"{confidence:.0f}%")
            st.caption(labels["agent_confidence_caption"].format(count=db_total))

        st.divider()

        st.header(labels["history_header"], help=labels["history_header_help"])
        hist_col1, hist_col2 = st.columns(2)
        club_filter_options = [labels["history_filter_all"]] + [c["name"] for c in clubs.values()]
        selected_club_filter = hist_col1.selectbox(
            labels["history_filter_club"], club_filter_options, key="history_club_filter"
        )
        feedback_filter_options = [
            (labels["history_filter_all"], None),
            (FEEDBACK_DISPLAY[language]["positive"], "positive"),
            (FEEDBACK_DISPLAY[language]["negative"], "negative"),
            (FEEDBACK_DISPLAY[language]["none"], "none"),
        ]
        selected_feedback_label = hist_col2.selectbox(
            labels["history_filter_feedback"],
            [opt[0] for opt in feedback_filter_options],
            key="history_feedback_filter",
        )
        selected_feedback_filter = dict(feedback_filter_options)[selected_feedback_label]
        company_filter_text = st.text_input(
            labels["history_filter_company"],
            placeholder=labels["company_placeholder"],
            key="history_company_filter",
        )

        history_rows = get_analysis_history(
            limit=10,
            club_name=None if selected_club_filter == labels["history_filter_all"] else selected_club_filter,
            company_name=company_filter_text or None,
            feedback=selected_feedback_filter,
            user_id=st.session_state["user_id"],  # personalisiert: nur eigene Analysen
        )
        if not history_rows:
            st.caption(labels["history_empty"])
        else:
            for row in history_rows:
                fb_display = FEEDBACK_DISPLAY[language].get(row["feedback"], row["feedback"])
                rel_time = format_relative_time(row["timestamp"], language)
                entry_label = (
                    f"{row['company_name']} | {row['club_name']} | "
                    f"Score: {row['fit_score']:.2f} | {rel_time} | {fb_display}"
                )
                with st.expander(entry_label):
                    st.caption(
                        labels["history_detail_meta"].format(
                            language=row["language"], model=row["selected_model"], timestamp=row["timestamp"]
                        )
                    )
                    st.write(f"**{labels['history_detail_research']}**")
                    st.write(row["research_summary"])

        st.header(labels["agent_eval_header"], help=labels["agent_eval_header_help"])

        consistency_pct, consistency_pairs = get_score_consistency()
        if consistency_pct is None:
            st.caption(labels["score_consistency_none"])
        else:
            st.metric(labels["score_consistency"], f"{consistency_pct:.0f}%")
            st.caption(labels["score_consistency_caption"].format(count=consistency_pairs))

        if st.button(labels["generate_report_btn"], key="generate_ragas_report"):
            with st.spinner(labels["eval_spinner"]):
                st.session_state["ragas_report"] = evaluate_with_ragas(limit=20)

        ragas_report = st.session_state.get("ragas_report")
        if ragas_report:
            if ragas_report["num_analyses_evaluated"] == 0:
                st.caption(labels["eval_no_data"])
            else:
                st.caption(labels["eval_num_analyzed"].format(count=ragas_report["num_analyses_evaluated"]))

                eval_col1, eval_col2, eval_col3 = st.columns(3)
                eval_col1.metric(labels["eval_overall_relevance"], f"{ragas_report['overall_relevance'] * 100:.0f}%")
                eval_col2.metric(labels["eval_overall_faithfulness"], f"{ragas_report['overall_faithfulness'] * 100:.0f}%")
                eval_col3.metric(
                    labels["eval_overall_answer_relevance"], f"{ragas_report['overall_answer_relevance'] * 100:.0f}%"
                )

                st.markdown(f"**{labels['eval_trend_header']}**")
                trend = ragas_report.get("trend")
                if trend is None:
                    st.caption(labels["eval_trend_insufficient"])
                else:
                    st.write(labels[f"eval_trend_{trend['direction']}"].format(delta=trend["delta"]))

                st.markdown(f"**{labels['eval_chart_header']}**")
                baseline = ragas_report["baseline"]
                chart_df = pd.DataFrame(
                    {
                        "Agent": [
                            ragas_report["overall_relevance"],
                            ragas_report["overall_faithfulness"],
                            ragas_report["overall_answer_relevance"],
                        ],
                        "Baseline": [baseline["relevance"], baseline["faithfulness"], baseline["answer_relevance"]],
                    },
                    index=["Relevance", "Faithfulness", "Answer Relevance"],
                )
                st.bar_chart(chart_df, color=["#2a78d6", "#eb6834"])

                eval_pricing = MODEL_PRICING["openai/gpt-4o-mini"]
                eval_tokens_in = sum(e["input_tokens"] for e in ragas_report["token_usage"])
                eval_tokens_out = sum(e["output_tokens"] for e in ragas_report["token_usage"])
                eval_cost = (
                    eval_tokens_in / 1000 * eval_pricing["input"] + eval_tokens_out / 1000 * eval_pricing["output"]
                )
                st.caption(
                    labels["eval_cost_caption"].format(
                        cost=eval_cost, tokens=eval_tokens_in + eval_tokens_out, model="openai/gpt-4o-mini"
                    )
                )

                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(
                    labels["eval_download_json"],
                    data=json.dumps(ragas_report, ensure_ascii=False, indent=2),
                    file_name="evaluation_report.json",
                    mime="application/json",
                    key="download_eval_json",
                )
                dl_col2.download_button(
                    labels["eval_download_csv"],
                    data=pd.DataFrame(ragas_report["per_analysis"]).to_csv(index=False),
                    file_name="evaluation_report.csv",
                    mime="text/csv",
                    key="download_eval_csv",
                )

                st.caption(labels["eval_note"])

    render_help_popover(labels["help_header_help"])
    with st.expander(labels["help_header"]):
        for entry in FAQ[language].values():
            st.markdown(f"**{entry['question']}**")
            st.write(entry["answer"])
            st.divider()

        st.markdown(f"**{labels['chat_subheader']}**")
        user_question = st.chat_input(labels["chat_placeholder"], key="help_chat_input")
        if user_question:
            answer = match_faq_answer(user_question, FAQ[language])
            st.info(answer or labels["chat_fallback"])

st.title(labels["title"])
st.caption(labels["caption"])

# Anzeigenamen für das Dropdown vorbereiten
club_options = {club["name"]: key for key, club in clubs.items()}

# Mittlere Spalte (Formular + Ergebnis-Karten) + rechte Info-Spalte (Kosten,
# Feedback, Quick Links) – die Sidebar selbst ist die "linke Spalte". Beide
# Variablen werden unten sowohl vom Formular-Block als auch (nach einem
# Rerun) vom Ergebnis-Block wiederverwendet.
col_main, col_info = st.columns([2, 1])

with col_main:
    with st.container(key="card_form", border=True):
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            selected_club_name = st.selectbox(labels["club_select"], list(club_options.keys()))
        with form_col2:
            company_name = st.text_input(labels["company_input"], placeholder=labels["company_placeholder"])

        st.info(labels["company_hint"])

        if st.button(labels["start_button"], type="primary"):
            rate_limit_ok, rate_limit_wait_minutes = check_rate_limit()

            if not company_name.strip():
                st.warning(labels["warning_no_company"])
            elif looks_like_question(company_name):
                st.warning(labels["warning_is_question"])
            elif not validate_company_input(company_name):
                st.warning(labels["warning_invalid_company"])
                log_security_event(company_name, "blocked_invalid_input")
            elif not rate_limit_ok:
                st.warning(labels["warning_rate_limit"].format(minutes=rate_limit_wait_minutes))
                log_security_event(company_name, "blocked_rate_limit")
            else:
                record_request()
                log_security_event(company_name, "success")

                selected_key = club_options[selected_club_name]
                club_profile = clubs[selected_key]

                with st.spinner(labels["spinner_text"]):
                    with collect_runs() as run_collector:
                        result = app.invoke({
                            "club_profile": club_profile,
                            "company_name": company_name,
                            "user_id": st.session_state["user_id"],
                            "selected_model": selected_model,
                            "language": language,
                            "research_findings": "",
                            "fit_score": 0.0,
                            "fit_reasoning": "",
                            "outreach_draft": "",
                            "rejection_reason": "",
                            "used_case_studies": [],
                            "used_sponsorship_matches": [],
                            "company_intelligence": {},
                            "competitor_analysis": {},
                            "budget_estimate": "",
                            "analysis_id": 0,
                            "learning_applied": False,
                            "token_usage": [],
                        })
                    run_id = run_collector.traced_runs[0].id if run_collector.traced_runs else None

                st.session_state["result"] = result
                st.session_state["result_meta"] = {
                    "club": club_profile["name"],
                    "sport": club_profile["sport"],
                    "company": company_name,
                    "selected_model": selected_model,
                    "language": language,
                    "run_id": run_id,
                    "analysis_id": result.get("analysis_id"),
                }
                st.session_state["feedback_given"] = False
                # Rerun, damit die Sidebar (rendert vor diesem Block) sofort den frischen
                # DB-Stand zeigt (Analyseverlauf, Score Consistency, Agent Confidence).
                st.rerun()

if "result" in st.session_state:
    result = st.session_state["result"]
    meta = st.session_state["result_meta"]
    # Labels für die Ergebnisanzeige richten sich nach der Sprache, die zum
    # Zeitpunkt der Analyse gewählt war (nicht nach der aktuellen Sidebar-Auswahl).
    result_labels = LABELS[meta.get("language", language)]

    with col_main:
        if is_plugin_enabled_for_display("web_search"):
            with st.container(key="card_research", border=True):
                with st.expander(result_labels["research_header"], expanded=False):
                    findings = result["research_findings"]
                    show_full = st.session_state.get("research_show_full", False)
                    if len(findings) <= 200 or show_full:
                        st.write(findings)
                        if len(findings) > 200 and st.button(
                            result_labels["research_less_btn"], key="research_less_btn"
                        ):
                            st.session_state["research_show_full"] = False
                            st.rerun()
                    else:
                        st.write(findings[:200] + "…")
                        if st.button(result_labels["research_more_btn"], key="research_more_btn"):
                            st.session_state["research_show_full"] = True
                            st.rerun()

        if is_plugin_enabled_for_display("company_intelligence"):
            with st.container(key="card_company_intel", border=True):
                with st.expander(result_labels["company_intel_header"], expanded=False):
                    st.caption(result_labels["company_intel_disclaimer"])
                    company_intel = result.get("company_intelligence", {})
                    if not company_intel or "error" in company_intel:
                        message = company_intel.get("error", "") if company_intel else ""
                        st.caption(result_labels["company_intel_none"].format(message=message))
                    else:
                        st.markdown(f"**{company_intel.get('company_name') or meta['company']}**")
                        intel_col1, intel_col2, intel_col3 = st.columns(3)
                        intel_col1.metric(
                            result_labels["company_intel_number"], company_intel.get("company_number") or "—"
                        )
                        intel_col2.metric(
                            result_labels["company_intel_status"],
                            (company_intel.get("company_status") or "—").title(),
                        )
                        intel_col3.metric(
                            result_labels["company_intel_founded"], company_intel.get("date_of_creation") or "—"
                        )
                        st.write(
                            f"**{result_labels['company_intel_type']}:** {company_intel.get('company_type') or '—'}"
                        )
                        if company_intel.get("address"):
                            st.write(f"**{result_labels['company_intel_address']}:** {company_intel['address']}")

        if is_plugin_enabled_for_display("case_study_db"):
            with st.container(key="card_case_studies", border=True):
                with st.expander(result_labels["case_studies_header"], expanded=False):
                    st.caption(result_labels["case_studies_disclaimer"])
                    used_case_studies = result.get("used_case_studies", [])
                    if used_case_studies:
                        if used_case_studies[0].get("match_type") == "sport":
                            st.caption(result_labels["case_studies_fallback"].format(company=meta["company"]))
                        for case in used_case_studies:
                            status = (
                                result_labels["case_study_success"]
                                if case["success"]
                                else result_labels["case_study_no_success"]
                            )
                            st.markdown(f"**{case['company']} – {case['sport']} ({status})**")
                            st.caption(case["summary"])
                    else:
                        st.caption(result_labels["case_studies_none"])

        if is_plugin_enabled_for_display("sponsorship_db"):
            with st.container(key="card_sponsorship_db", border=True):
                with st.expander(result_labels["sponsorship_db_header"], expanded=False):
                    st.caption(result_labels["sponsorship_db_disclaimer"])
                    used_sponsorship_matches = result.get("used_sponsorship_matches", [])
                    if used_sponsorship_matches:
                        brand_fit_labels = {
                            "high": result_labels["brand_fit_high"],
                            "medium": result_labels["brand_fit_medium"],
                            "low": result_labels["brand_fit_low"],
                        }
                        for match in used_sponsorship_matches:
                            fit_text = brand_fit_labels.get(match["brand_fit"], result_labels["brand_fit_high"])
                            st.markdown(
                                f"**({fit_text})** "
                                + result_labels["sponsorship_db_entry"].format(
                                    company=match["company"],
                                    team=match["athlete_or_team"],
                                    year=match["start_year"],
                                    metric=match["success_metric"],
                                )
                            )
                    else:
                        st.caption(result_labels["sponsorship_db_none"])

        if is_plugin_enabled_for_display("budget_estimator"):
            with st.container(key="card_budget_estimator", border=True):
                with st.expander(result_labels["budget_estimator_header"], expanded=False):
                    budget_estimate = result.get("budget_estimate", "")
                    if budget_estimate:
                        st.write(budget_estimate)
                    else:
                        st.caption(result_labels["budget_estimator_none"])

        if is_plugin_enabled_for_display("competitor_analysis"):
            with st.container(key="card_competitor_analysis", border=True):
                with st.expander(result_labels["competitor_analysis_header"], expanded=False):
                    portfolio = result.get("competitor_analysis", {})

                    if not portfolio.get("found"):
                        st.caption(result_labels["competitor_analysis_none"])
                    else:
                        st.caption(result_labels["portfolio_disclaimer"])

                        # 1) Company's Sponsoring-Portfolio (generisch für jede Firma,
                        # da alle Werte aus company_name/sport abgeleitet werden)
                        st.markdown(
                            f"**{result_labels['portfolio_categories_label']}:** "
                            + (", ".join(portfolio["categories"]) or "—")
                        )
                        st.markdown(
                            f"**{result_labels['portfolio_active_count_label']}:** {portfolio['active_count']}"
                        )
                        st.markdown(f"**{result_labels['portfolio_audience_label']}:** {portfolio['audience']}")

                        # 2) Vergleich mit dem gewählten Verein
                        st.markdown(
                            f"**{result_labels['audience_fit_header'].format(club=meta['club'])}**"
                        )
                        fit_key = f"audience_fit_{portfolio['audience_fit']}"
                        fit_display = result_labels.get(fit_key, portfolio["audience_fit"] or "—")
                        st.write(f"{result_labels['audience_fit_label']} {fit_display}")
                        st.progress(portfolio["match_percent"] / 100)
                        st.caption(f"{result_labels['match_percent_label']}: {portfolio['match_percent']}%")

                        # 3) Market Saturation (Company-Perspektive: eigene Sponsorings
                        # dieser Firma in DIESER Sportart, nicht der Gesamtmarkt)
                        st.markdown(f"**{result_labels['market_saturation_header']}**")
                        st.write(
                            result_labels["saturation_same_sport_label"].format(sport=meta.get("sport", ""))
                            + f": {portfolio['same_sport_count']}"
                        )
                        saturation_level = portfolio["saturation_level"]
                        saturation_progress = {"low": 0.25, "medium": 0.5, "high": 0.75, "extreme": 1.0}[
                            saturation_level
                        ]
                        st.progress(saturation_progress)
                        st.caption(
                            f"{result_labels['saturation_level_label']}: "
                            f"{result_labels[f'saturation_level_{saturation_level}']}"
                        )
                        st.caption(result_labels[f"market_saturation_interpretation_{saturation_level}"])

                        # 4) Impact auf Fit-Score (dynamisch, siehe agent.py)
                        st.markdown(f"**{result_labels['competitor_impact_header']}**")
                        score_before = portfolio.get("score_before_adjustment", result["fit_score"])
                        score_after = portfolio.get("score_after_adjustment", result["fit_score"])
                        if score_before != score_after:
                            st.info(
                                result_labels["competitor_impact_adjusted"].format(
                                    before=score_before,
                                    after=score_after,
                                    match_percent=portfolio["match_percent"],
                                    saturation=result_labels[f"saturation_level_{saturation_level}"],
                                    company=meta["company"],
                                    sport=meta.get("sport", ""),
                                )
                            )
                        else:
                            st.caption(result_labels["competitor_impact_none"])

        with st.container(key="card_fit", border=True):
            with st.expander(result_labels["fit_header"], expanded=False):
                if result.get("learning_applied"):
                    st.badge(result_labels["learning_badge"], color="violet")
                st.metric(result_labels["score_label"], f"{result['fit_score']:.2f}")
                st.write(result["fit_reasoning"])

        with st.container(key="card_outreach", border=True):
            if result["outreach_draft"]:
                with st.expander(result_labels["outreach_header"], expanded=False):
                    st.success(result["outreach_draft"])
            else:
                with st.expander(result_labels["rejection_header"], expanded=False):
                    st.error(result["rejection_reason"])

    with col_info:
        with st.container(key="card_tokens", border=True):
            with st.expander(result_labels["tokens_header"], expanded=False):
                total_tokens = sum(entry["total_tokens"] for entry in result["token_usage"])
                total_input_tokens = sum(entry["input_tokens"] for entry in result["token_usage"])
                total_output_tokens = sum(entry["output_tokens"] for entry in result["token_usage"])

                pricing = MODEL_PRICING[meta["selected_model"]]
                cost_usd = (
                    total_input_tokens / 1000 * pricing["input"]
                    + total_output_tokens / 1000 * pricing["output"]
                )

                st.metric(result_labels["total_tokens"], total_tokens)
                st.metric(result_labels["estimated_cost"], f"${cost_usd:.4f}")

                # Kein verschachtelter st.expander hier (Streamlit erlaubt keine
                # geschachtelten Expander) – Checkbox übernimmt die Ein-/Ausklapp-Rolle.
                if st.checkbox(result_labels["details_expander"], key="tokens_details_toggle"):
                    st.caption(result_labels["model_caption"].format(model=meta["selected_model"]))
                    for entry in result["token_usage"]:
                        entry_cost = (
                            entry["input_tokens"] / 1000 * pricing["input"]
                            + entry["output_tokens"] / 1000 * pricing["output"]
                        )
                        st.write(
                            f"**{entry['node']}**: {entry['total_tokens']} Tokens "
                            f"(Input: {entry['input_tokens']}, Output: {entry['output_tokens']}) "
                            f"– ${entry_cost:.4f}"
                        )

        with st.container(key="card_feedback", border=True):
            with st.expander(result_labels["feedback_question"], expanded=False):
                feedback_given = st.session_state.get("feedback_given", False)

                if st.button(
                    result_labels["feedback_positive_btn"],
                    disabled=feedback_given,
                    key="feedback_positive",
                ):
                    save_feedback(
                        meta["club"], meta["company"], result["fit_score"], "positive", meta["selected_model"]
                    )
                    if meta.get("analysis_id"):
                        update_analysis_feedback(meta["analysis_id"], "positive")
                    st.session_state["feedback_given"] = True
                    st.session_state["feedback_type"] = "positive"
                    st.rerun()

                if st.button(
                    result_labels["feedback_negative_btn"],
                    disabled=feedback_given,
                    key="feedback_negative",
                ):
                    save_feedback(
                        meta["club"], meta["company"], result["fit_score"], "negative", meta["selected_model"]
                    )
                    if meta.get("analysis_id"):
                        update_analysis_feedback(meta["analysis_id"], "negative")
                    st.session_state["feedback_given"] = True
                    st.session_state["feedback_type"] = "negative"
                    st.rerun()

                if feedback_given:
                    feedback_type = st.session_state.get("feedback_type", "positive")
                    thanks_key = (
                        "feedback_thanks_positive" if feedback_type == "positive" else "feedback_thanks_negative"
                    )
                    st.success(result_labels[thanks_key])

        with st.container(key="card_quicklinks", border=True):
            with st.expander(labels["quicklinks_header"], expanded=False):
                st.caption(result_labels["trace_header"])
                run_id = meta.get("run_id")
                if not run_id:
                    st.caption(result_labels["trace_none"])
                else:
                    trace_url = get_langsmith_trace_url(run_id)
                    if trace_url:
                        st.markdown(f"[{result_labels['trace_link']}]({trace_url})")
                    else:
                        project = os.environ.get("LANGSMITH_PROJECT", "sponsor-match")
                        st.caption(result_labels["trace_fallback"].format(run_id=run_id, project=project))
