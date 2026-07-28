from dotenv import load_dotenv
load_dotenv()
import os
import re
import operator
import sqlite3
import datetime
import random
import threading
import bcrypt
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace
from typing_extensions import TypedDict
from typing import Literal, Annotated

os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "sponsor-match")

from langsmith import Client, traceable

_langsmith_client = None

def get_langsmith_trace_url(run_id) -> str | None:
    """Baut die LangSmith-Trace-URL für eine Run-ID, oder None falls nicht verfügbar
    (z.B. LANGSMITH_API_KEY fehlt oder die Anfrage schlägt fehl)."""
    global _langsmith_client
    if not os.environ.get("LANGSMITH_API_KEY") or run_id is None:
        return None
    try:
        if _langsmith_client is None:
            _langsmith_client = Client()
        return _langsmith_client.get_run_url(
            run=SimpleNamespace(id=run_id, session_id=None),
            project_name=os.environ.get("LANGSMITH_PROJECT", "sponsor-match"),
        )
    except Exception:
        return None

class SponsorMatchState(TypedDict):
    club_profile: dict       # gewähltes Vereins-Profil aus clubs.json
    company_name: str        # Nutzereingabe: Firma, die geprüft werden soll
    user_id: int | None      # ID des eingeloggten Users (für personalisierten Verlauf)
    selected_model: str       # gewähltes LLM-Modell (OpenRouter-Modell-ID)
    language: str             # Zielsprache für alle LLM-Ausgaben: "de" (default), "en", "fr"
    research_findings: str   # Ergebnis von research_company
    fit_score: float         # Ergebnis von evaluate_fit, z.B. 0.0–1.0
    fit_reasoning: str        # Begründung für den Score
    outreach_draft: str       # nur befüllt bei gutem Fit
    rejection_reason: str     # nur befüllt bei schlechtem Fit
    used_case_studies: list   # RAG-Treffer aus der Case-Study-Wissensbasis
    used_sponsorship_matches: list  # Treffer aus der externen Sponsorship-Datenbank
    company_intelligence: dict  # Ergebnis von get_company_intelligence (OpenCorporates)
    competitor_analysis: dict  # strukturierte Konkurrentenliste + Score-/Marktsättigungs-Impact (competitor_analysis-Plugin)
    budget_estimate: str  # Ergebnis von _build_budget_estimate (budget_estimator-Plugin)
    analysis_id: int          # ID des in der SQLite-DB gespeicherten Analyse-Datensatzes
    learning_applied: bool    # True, wenn frühere Feedback-Muster den Score angepasst haben
    token_usage: Annotated[list, operator.add]  # Tokenverbrauch pro LLM-Aufruf

from langchain_openai import ChatOpenAI
from src.tools import get_search_tool, get_company_intelligence

@lru_cache(maxsize=None)
def _get_llm(model: str) -> ChatOpenAI:
    """Erstellt (und cached) eine LLM-Instanz für das gewählte Modell.

    temperature=0 für deterministische, reproduzierbare Outputs – gilt für
    jeden Node (research, evaluate, draft, Case-Study-Übersetzung, RAGAs-Judge),
    da sie alle über diese Factory laufen.
    """
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

import json as _json

# --- Plugin System: schaltet Agent-Fähigkeiten dynamisch an/aus ---

PLUGINS_PATH = "data/available_plugins.json"

def _load_plugins() -> list[dict]:
    """Lädt die Plugin-Konfiguration live (kein Caching!) – der Plugin Manager
    in main.py kann sie jederzeit ändern, und die nächste Analyse soll die
    aktuellen Toggles sofort berücksichtigen.

    Graceful Fallback: leere Liste, falls die Datei fehlt/ungültig ist –
    dann greift der "enabled=True"-Default in is_plugin_enabled().
    """
    try:
        with open(PLUGINS_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return []

def is_plugin_enabled(plugin_id: str) -> bool:
    """Prüft, ob ein Plugin aktuell aktiviert ist. Fällt auf True zurück,
    falls die Config fehlt oder das Plugin dort nicht auftaucht, damit eine
    kaputte/fehlende Datei den Agent nicht lahmlegt."""
    for plugin in _load_plugins():
        if plugin["id"] == plugin_id:
            return plugin.get("enabled", True)
    return True

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

# --- Optionales Plugin: competitor_analysis (Tavily) ---
# Analysiert generisch das SPONSORING-PORTFOLIO DER EINGEGEBENEN FIRMA SELBST
# (nicht ihrer Konkurrenten) – funktioniert für jede beliebige Company, da
# Suchbegriff, Prompt und Score-Formel ausschließlich von company_name/club
# abhängen, nichts ist auf eine bestimmte Firma fest verdrahtet.

_PORTFOLIO_FIELD_PREFIXES = (
    "CATEGORIES",
    "ACTIVE_COUNT",
    "AUDIENCE",
    "SAME_SPORT_COUNT",
    "AUDIENCE_FIT",
    "MATCH_PERCENT",
)


def _to_nonneg_int(value: str, default: int = 0) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else default


def _analyze_company_sponsorship_portfolio(company_name: str, club: dict, language: str, llm) -> dict:
    """competitor_analysis-Plugin: sucht per Tavily nach dem bestehenden Sponsoring-
    Portfolio der EINGEGEBENEN Firma (nicht ihrer Konkurrenten) und lässt ein LLM
    daraus Kategorien, aktive Sponsoring-Anzahl, Zielgruppe, Sättigung in der
    Vereins-Sportart sowie den Zielgruppen-Fit zum konkreten Verein extrahieren.

    ACHTUNG: alle Zahlen/Einschätzungen sind KI-Schätzungen auf Basis der
    Websuche, keine verifizierten Marktdaten – main.py zeigt das mit Disclaimer an.
    """
    sport = club["sport"]
    try:
        results = get_search_tool().invoke(f"{company_name} sponsorships list")
    except Exception:
        results = ""

    if not results:
        return {"found": False, "token_usage": None}

    snippet = str(results)[:2000]
    output_language = {"de": "German", "en": "English", "fr": "French"}[language]
    prompt = (
        f"Based on this web search about {company_name}'s sponsorships:\n{snippet}\n\n"
        f"Analyze {company_name}'s OWN sponsorship portfolio (not its competitors'). "
        f"The club being evaluated as a sponsorship target has this profile:\n"
        f"- Sport: {sport}\n"
        f"- Fanbase: {club['fanbase']}\n"
        f"- Values: {club['values']}\n\n"
        f"Respond in exactly this format, nothing before or after:\n"
        f"CATEGORIES: <up to 3 main sports/categories {company_name} sponsors, comma-separated>\n"
        f"ACTIVE_COUNT: <your best estimate of the number of currently active sponsorship deals overall, "
        f"as a single integer>\n"
        f"AUDIENCE: <typical target audience of {company_name}'s sponsorship activities, a short phrase>\n"
        f"SAME_SPORT_COUNT: <your best estimate of how many of those active sponsorships are specifically "
        f"in {sport}, as a single integer, 0 if none>\n"
        f"AUDIENCE_FIT: <yes, no, or partial - does {company_name}'s typical sponsorship audience match "
        f"this club's fanbase>\n"
        f"MATCH_PERCENT: <integer 0-100, how well {company_name}'s audience overlaps with this club's fanbase>\n"
        f"Write all text values (except AUDIENCE_FIT and the numbers) in {output_language}. Use only real "
        f"information found in the search results; if uncertain, give a reasonable estimate rather than "
        f"leaving fields blank."
    )
    response = llm.invoke(prompt)
    token_entry = _track_tokens("competitor_analysis", response)

    fields = {}
    for line in response.content.split("\n"):
        line = line.strip()
        for key in _PORTFOLIO_FIELD_PREFIXES:
            prefix = f"{key}:"
            if line.upper().startswith(prefix):
                fields[key] = line[len(prefix):].strip()
                break

    # Normalisierung, da das LLM AUDIENCE_FIT trotz Anweisung gelegentlich in der
    # Zielsprache statt auf Englisch beantwortet (z.B. "ja" statt "yes").
    raw_fit = fields.get("AUDIENCE_FIT", "").strip().lower()
    if raw_fit.startswith(("yes", "ja", "oui")):
        audience_fit = "yes"
    elif raw_fit.startswith(("no", "non", "nein")):
        audience_fit = "no"
    else:
        audience_fit = "partial"

    return {
        "found": True,
        "categories": [c.strip() for c in fields.get("CATEGORIES", "").split(",") if c.strip()][:3],
        "active_count": _to_nonneg_int(fields.get("ACTIVE_COUNT", "")),
        "audience": fields.get("AUDIENCE", "—"),
        "same_sport_count": _to_nonneg_int(fields.get("SAME_SPORT_COUNT", "")),
        "audience_fit": audience_fit,
        "match_percent": max(0, min(100, _to_nonneg_int(fields.get("MATCH_PERCENT", ""), default=50))),
        "token_usage": token_entry,
    }


def _compute_saturation_level(same_sport_count: int) -> str:
    """Company-Perspektive: wie gesättigt ist die Firma bereits in DIESER Sportart
    (nicht wie viele Sponsoren der Markt insgesamt verträgt) – rein aus der Anzahl
    ihrer eigenen, bereits laufenden Sponsorings in dieser Sportart abgeleitet."""
    if same_sport_count >= 7:
        return "extreme"
    if same_sport_count >= 4:
        return "high"
    if same_sport_count >= 2:
        return "medium"
    return "low"


def _compute_portfolio_score_impact(match_percent: int, saturation_level: str) -> float:
    """Dynamische Score-Anpassung: hängt vom individuellen Zielgruppen-Match UND von
    der individuellen Sättigung DIESER Firma in DIESER Sportart ab – kein fixer
    Wert, unterschiedliche Firmen/Vereine ergeben unterschiedliche Anpassungen.
    """
    if match_percent >= 70:
        base = 0.10
    elif match_percent >= 40:
        base = 0.0
    else:
        base = -0.10

    saturation_penalty = {"low": 0.0, "medium": -0.03, "high": -0.08, "extreme": -0.15}[saturation_level]
    return round(max(-0.25, min(0.15, base + saturation_penalty)), 2)

def _build_budget_estimate(sponsorship_matches: list[dict], language: str) -> str:
    """budget_estimator-Plugin: leitet eine grobe Budget-Schätzung aus den
    investment_range-Werten passender externer Sponsorship-DB-Fälle ab.
    Rein datenbasiert, kein LLM-Call.
    """
    if not sponsorship_matches:
        return {
            "de": "Keine Budget-Schätzung möglich (keine passenden historischen Daten).",
            "en": "No budget estimate possible (no matching historical data).",
            "fr": "Aucune estimation de budget possible (pas de données historiques correspondantes).",
        }[language]

    ranges = sorted({m["investment_range"] for m in sponsorship_matches})
    label = {
        "de": "Geschätztes Sponsoring-Budget (basierend auf ähnlichen historischen Fällen)",
        "en": "Estimated sponsorship budget (based on similar historical cases)",
        "fr": "Budget de sponsoring estimé (basé sur des cas historiques similaires)",
    }[language]
    return f"{label}: {', '.join(ranges)}"

def _build_company_intelligence_context(company_data: dict, company_name: str, language: str) -> str:
    """company_intelligence-Plugin: baut den Prompt-Kontext aus den
    Companies-House-Registrierungsdaten (oder erklärt, warum keine da sind).
    Deckt naturgemäß nur UK-registrierte Firmen ab."""
    if "error" in company_data:
        return {
            "de": f"Company-Daten: Keine Companies-House-Daten zu '{company_name}' verfügbar ({company_data['error']}).",
            "en": f"Company data: No Companies House data available for '{company_name}' ({company_data['error']}).",
            "fr": f"Données d'entreprise : aucune donnée Companies House disponible pour '{company_name}' ({company_data['error']}).",
        }[language]

    if language == "en":
        return (
            f"Company data (UK Companies House registry): '{company_data['company_name']}', "
            f"UK company number: {company_data['company_number']}, status: {company_data['company_status']}, "
            f"legal type: {company_data['company_type']}"
            + (f", founded: {company_data['date_of_creation']}" if company_data.get("date_of_creation") else "")
            + (f", address: {company_data['address']}" if company_data.get("address") else "")
            + "."
        )
    if language == "fr":
        return (
            f"Données d'entreprise (registre UK Companies House) : '{company_data['company_name']}', "
            f"numéro d'entreprise UK : {company_data['company_number']}, statut : {company_data['company_status']}, "
            f"forme juridique : {company_data['company_type']}"
            + (f", fondée en : {company_data['date_of_creation']}" if company_data.get("date_of_creation") else "")
            + (f", adresse : {company_data['address']}" if company_data.get("address") else "")
            + "."
        )
    return (
        f"Company-Daten (UK Companies-House-Registry): '{company_data['company_name']}', "
        f"UK-Nummer: {company_data['company_number']}, Status: {company_data['company_status']}, "
        f"Rechtsform: {company_data['company_type']}"
        + (f", gegründet: {company_data['date_of_creation']}" if company_data.get("date_of_creation") else "")
        + (f", Adresse: {company_data['address']}" if company_data.get("address") else "")
        + "."
    )

# --- Long-term memory: SQLite-Verlauf vergangener Analysen ---

DB_PATH = "data/sponsor_match.db"
EVALUATION_REPORT_PATH = "data/evaluation_report.json"

_shared_connection: sqlite3.Connection | None = None
_db_lock = threading.Lock()

def configure_shared_connection(conn: sqlite3.Connection) -> None:
    """Registriert eine von außen gecachte, langlebige Connection (z.B. via
    Streamlits @st.cache_resource in main.py), die ab sofort für alle
    DB-Zugriffe wiederverwendet wird, statt bei jedem Aufruf neu zu öffnen.

    Optional: ohne diesen Aufruf öffnet jede Funktion weiterhin ihre eigene
    kurzlebige Connection wie bisher (z.B. im CLI-Testlauf ohne Streamlit).
    """
    global _shared_connection
    _shared_connection = conn

@contextmanager
def _get_connection():
    """Liefert eine SQLite-Connection als Context-Manager (committet bei
    Erfolg, rollt bei Fehlern zurück – wie zuvor `with sqlite3.Connection`).

    Nutzt die über `configure_shared_connection` registrierte Connection,
    falls vorhanden, sonst eine frische, kurzlebige. Ein Lock schützt vor
    gleichzeitigem Zugriff mehrerer Threads/Sessions auf dieselbe (gecachte)
    Connection – SQLite-Connections sind nicht für parallele Nutzung gedacht.
    """
    with _db_lock:
        if _shared_connection is not None:
            conn = _shared_connection
            owns_connection = False
        else:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            owns_connection = True
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

def _init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                club_name TEXT NOT NULL,
                company_name TEXT NOT NULL,
                fit_score REAL NOT NULL,
                feedback TEXT NOT NULL DEFAULT 'none',
                language TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                research_summary TEXT,
                learning_applied INTEGER NOT NULL DEFAULT 0,
                fit_reasoning TEXT,
                case_studies_summary TEXT,
                score_cached INTEGER NOT NULL DEFAULT 0,
                user_id INTEGER
            )
            """
        )
        # Migrationen für DBs, die vor Einführung dieser Spalten erstellt wurden.
        for migration in (
            "ALTER TABLE analyses ADD COLUMN learning_applied INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE analyses ADD COLUMN fit_reasoning TEXT",
            "ALTER TABLE analyses ADD COLUMN case_studies_summary TEXT",
            "ALTER TABLE analyses ADD COLUMN score_cached INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE analyses ADD COLUMN user_id INTEGER",
        ):
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # Spalte existiert bereits

_init_db()

def get_similar_analyses(company_name: str, limit: int = 3) -> list[dict]:
    """Sucht frühere Analysen zur selben Firma (parametrisierte Teilstring-Suche)."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE company_name LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{company_name}%", limit),
        ).fetchall()
    return [dict(row) for row in rows]

def save_analysis(
    club_name: str,
    company_name: str,
    fit_score: float,
    language: str,
    selected_model: str,
    research_summary: str,
    fit_reasoning: str = "",
    case_studies: list | None = None,
    feedback: str = "none",
    learning_applied: bool = False,
    score_cached: bool = False,
    user_id: int | None = None,
) -> int:
    """Speichert eine neue Analyse und gibt die vergebene ID zurück."""
    case_studies_summary = _json.dumps(
        [c["summary"] for c in case_studies] if case_studies else []
    )
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses
                (timestamp, club_name, company_name, fit_score, feedback, language,
                 selected_model, research_summary, learning_applied, fit_reasoning,
                 case_studies_summary, score_cached, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.datetime.now().isoformat(),
                club_name,
                company_name,
                fit_score,
                feedback,
                language,
                selected_model,
                research_summary,
                int(learning_applied),
                fit_reasoning,
                case_studies_summary,
                int(score_cached),
                user_id,
            ),
        )
        return cursor.lastrowid

def update_analysis_feedback(analysis_id: int, feedback: str) -> None:
    """Trägt nachträglich das Nutzer-Feedback (positive/negative) zu einer Analyse-ID ein."""
    with _get_connection() as conn:
        conn.execute("UPDATE analyses SET feedback = ? WHERE id = ?", (feedback, analysis_id))

def get_exact_previous_analysis(company_name: str, club_name: str) -> dict | None:
    """Sucht die letzte Analyse zu genau dieser Firma+Verein-Kombination (exakter
    Match, kein Teilstring) – Grundlage fürs Score-Caching zur Konsistenzsicherung.
    """
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE company_name = ? AND club_name = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (company_name, club_name),
        ).fetchone()
    return dict(row) if row else None

def get_score_consistency() -> tuple[float | None, int]:
    """Misst, wie oft der Agent bei identischer Firma+Verein-Kombination denselben
    Score liefert: Anteil übereinstimmender aufeinanderfolgender Score-Paare
    innerhalb jeder (Firma, Verein)-Gruppe.

    Gibt (Konsistenz_in_Prozent, Anzahl_verglichener_Paare) zurück;
    Konsistenz ist None, wenn es noch keine Wiederholungs-Analyse gibt.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT company_name, club_name, fit_score FROM analyses "
            "ORDER BY company_name, club_name, timestamp"
        ).fetchall()

    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((row["company_name"], row["club_name"]), []).append(row["fit_score"])

    total_pairs = 0
    matching_pairs = 0
    for scores in groups.values():
        for i in range(1, len(scores)):
            total_pairs += 1
            if scores[i] == scores[i - 1]:
                matching_pairs += 1

    if total_pairs == 0:
        return None, 0
    return matching_pairs / total_pairs * 100, total_pairs

def get_analysis_history(
    limit: int = 10,
    club_name: str | None = None,
    company_name: str | None = None,
    feedback: str | None = None,
    user_id: int | None = None,
) -> list[dict]:
    """Liest den Analyseverlauf, optional gefiltert nach Verein/Firma/Feedback-Status/User.

    user_id filtert den Verlauf auf die Analysen genau dieses Users (personalisierter
    Analyseverlauf) – ohne user_id werden Analysen aller User zurückgegeben.
    """
    query = "SELECT * FROM analyses WHERE 1=1"
    params: list = []
    if club_name:
        query += " AND club_name = ?"
        params.append(club_name)
    if company_name:
        query += " AND company_name LIKE ?"
        params.append(f"%{company_name}%")
    if feedback:
        query += " AND feedback = ?"
        params.append(feedback)
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def get_feedback_confidence() -> tuple[int, int]:
    """Zählt alle positiven/negativen Feedbacks über sämtliche Analysen (für die Sidebar-Statistik)."""
    with _get_connection() as conn:
        positive = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE feedback = 'positive'"
        ).fetchone()[0]
        negative = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE feedback = 'negative'"
        ).fetchone()[0]
    return positive, negative

# --- User Authentication + Personalisierung (eigene DB: data/users.db) ---
#
# Bewusst eine EIGENE, kurzlebige Connection pro Aufruf (kein geteiltes/
# gecachtes Connection-Objekt) – dieselbe Lehre wie bei sponsor_match.db:
# eine über Streamlit-Reruns hinweg geteilte sqlite3.Connection kann in einen
# "readonly"/gesperrten Zustand geraten, wenn ein Rerun mitten in einem
# Schreibzugriff abgebrochen wird.

USERS_DB_PATH = "data/users.db"

def _get_users_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(USERS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_users_db() -> None:
    conn = _get_users_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                theme TEXT NOT NULL DEFAULT 'light',
                language TEXT NOT NULL DEFAULT 'de',
                favorite_clubs TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

_init_users_db()

def create_user(username: str, password: str, email: str) -> tuple[bool, str]:
    """Legt einen neuen User + Default-Settings an. Passwort wird NIE im
    Klartext gespeichert, nur der bcrypt-Hash. Gibt (erfolg, fehlercode) zurück
    – fehlercode ist "" bei Erfolg, sonst z.B. "username_taken"."""
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    now = datetime.datetime.now().isoformat()

    conn = _get_users_connection()
    try:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, email, now),
            )
        except sqlite3.IntegrityError:
            return False, "username_taken"

        user_id = cursor.lastrowid
        conn.execute(
            """
            INSERT INTO user_settings (user_id, theme, language, favorite_clubs, created_at, updated_at)
            VALUES (?, 'light', 'de', '[]', ?, ?)
            """,
            (user_id, now, now),
        )
        conn.commit()
        return True, ""
    finally:
        conn.close()

def verify_user(username: str, password: str) -> dict | None:
    """Prüft Login-Daten gegen den gespeicherten bcrypt-Hash. Gibt bei Erfolg
    {"id", "username", "email"} zurück, sonst None (falscher User ODER falsches
    Passwort – bewusst keine Unterscheidung in der Rückgabe, um nicht zu
    verraten, ob ein Username existiert)."""
    conn = _get_users_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    if bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return {"id": row["id"], "username": row["username"], "email": row["email"]}
    return None

def get_user_settings(user_id: int) -> dict:
    conn = _get_users_connection()
    try:
        row = conn.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"theme": "light", "language": "de", "favorite_clubs": []}
    return {
        "theme": row["theme"],
        "language": row["language"],
        "favorite_clubs": _json.loads(row["favorite_clubs"]),
    }

def save_user_settings(user_id: int, theme: str, language: str, favorite_clubs: list[str]) -> None:
    now = datetime.datetime.now().isoformat()
    conn = _get_users_connection()
    try:
        conn.execute(
            """
            INSERT INTO user_settings (user_id, theme, language, favorite_clubs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                theme = excluded.theme,
                language = excluded.language,
                favorite_clubs = excluded.favorite_clubs,
                updated_at = excluded.updated_at
            """,
            (user_id, theme, language, _json.dumps(favorite_clubs), now, now),
        )
        conn.commit()
    finally:
        conn.close()

def delete_user_account(user_id: int) -> None:
    """Löscht den User + seine Settings unwiderruflich. Bereits gespeicherte
    Analysen bleiben erhalten (Business-Daten, kein Nutzerdatum), werden aber
    von der User-Zuordnung gelöst (user_id -> NULL in sponsor_match.db)."""
    with _get_connection() as conn:
        conn.execute("UPDATE analyses SET user_id = NULL WHERE user_id = ?", (user_id,))

    conn = _get_users_connection()
    try:
        conn.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()

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

# --- Agent Learning: Feedback-Muster ähnlicher Sponsoren in die Bewertung einfließen lassen ---

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

_FIT_SECTION_LABELS = {
    "de": {"pros": "Was passt gut", "cons": "Was passt weniger", "recommendation": "Empfehlung"},
    "en": {"pros": "What fits well", "cons": "What fits less", "recommendation": "Recommendation"},
    "fr": {"pros": "Ce qui correspond bien", "cons": "Ce qui correspond moins", "recommendation": "Recommandation"},
}


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

def _track_tokens(node_name: str, response) -> dict:
    """Extrahiert die Token-Nutzung einer LLM-Antwort für das Tracking."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "node": node_name,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

@traceable(run_type="chain", name="research_company")
def research_company(state: SponsorMatchState) -> dict:
    """Recherchiert die angegebene Firma über Tavily und fasst die Ergebnisse zusammen."""
    company_name = state["company_name"]
    llm = _get_llm(state["selected_model"])
    language = state["language"]

    # Plugin-Gating: web_search ist "required", d.h. im UI nicht abschaltbar –
    # dieser Zweig ist defensive Absicherung, falls die Config trotzdem einmal
    # enabled=false enthält.
    if is_plugin_enabled("web_search"):
        search_tool = get_search_tool()
        search_results = search_tool.invoke(
            f"{company_name} company sponsorship marketing target audience brand values"
        )
    else:
        search_results = {
            "de": "(Web-Search-Plugin deaktiviert – keine externen Rechercheergebnisse verfügbar.)",
            "en": "(Web search plugin disabled – no external research results available.)",
            "fr": "(Plugin de recherche web désactivé – aucun résultat de recherche externe disponible.)",
        }[language]
    if language == "en":
        summarize_prompt = (
            f"Summarize the following search results for '{company_name}', "
            f"focusing on: industry, past sponsorship activities, target audience, "
            f"and brand values. Keep it to 4-5 sentences. Answer in English.\n\n"
            f"Search results: {search_results}"
        )
    elif language == "fr":
        summarize_prompt = (
            f"Résumez les résultats de recherche suivants pour '{company_name}', "
            f"en vous concentrant sur : le secteur d'activité, les activités de parrainage "
            f"passées, le public cible et les valeurs de marque. Limitez-vous à 4-5 phrases. "
            f"Répondez en français.\n\n"
            f"Résultats de recherche : {search_results}"
        )
    else:
        summarize_prompt = (
            f"Fasse die folgenden Suchergebnisse zu '{company_name}' zusammen, "
            f"mit Fokus auf: Branche, bisherige Sponsoring-Aktivitäten, Zielgruppe, "
            f"Markenwerte. Halte es auf 4-5 Sätze. Antworte auf Deutsch.\n\n"
            f"Suchergebnisse: {search_results}"
        )
    response = llm.invoke(summarize_prompt)

    return {
        "research_findings": response.content.strip(),
        "token_usage": [_track_tokens("research_company", response)],
    }

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

@traceable(run_type="chain", name="evaluate_fit")
def evaluate_fit(state: SponsorMatchState) -> dict:
    """Bewertet den Markenfit zwischen Firma und Verein basierend auf den Recherche-Ergebnissen."""
    club = state["club_profile"]
    findings = state["research_findings"]
    company_name = state["company_name"]
    language = state["language"]
    llm = _get_llm(state["selected_model"])

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

    # Company Intelligence (Companies House, UK): läuft ebenfalls unabhängig
    # vom Score-Cache, da für die Anzeige gebraucht. Plugin-Gating: kann im
    # Plugin Manager abgeschaltet werden – dann entfällt der externe API-Call.
    company_intelligence = (
        get_company_intelligence.invoke(company_name)
        if is_plugin_enabled("company_intelligence")
        else {"error": "Company Intelligence Plugin deaktiviert"}
    )

    # Score-Konsistenz: exakt gleiche Firma+Verein-Kombination schon einmal
    # analysiert? Dann Score cachen statt neu (und ggf. anders) zu berechnen.
    cached = get_exact_previous_analysis(company_name, club["name"])

    # Strukturierte Konkurrenzanalyse (Sponsoring-Portfolio DER FIRMA SELBST, generisch
    # für jede Company) für die eigene UI-Sektion (main.py), unabhängig vom
    # Score-Cache-Zweig, damit die Karte auch bei gecachtem Score befüllt ist.
    if is_plugin_enabled("competitor_analysis"):
        portfolio = _analyze_company_sponsorship_portfolio(company_name, club, language, llm)
        if portfolio.get("token_usage"):
            token_usage.append(portfolio["token_usage"])
    else:
        portfolio = {"found": False}
    if portfolio["found"]:
        portfolio["saturation_level"] = _compute_saturation_level(portfolio["same_sport_count"])
        portfolio["score_adjustment"] = _compute_portfolio_score_impact(
            portfolio["match_percent"], portfolio["saturation_level"]
        )
    else:
        portfolio["saturation_level"] = "low"
        portfolio["score_adjustment"] = 0.0

    # Budget-Schätzung für die eigene UI-Sektion (main.py), ebenfalls unabhängig
    # vom Score-Cache-Zweig, da rein datenbasiert (kein LLM-Call) und für die
    # Anzeige gebraucht.
    budget_estimate_text = (
        _build_budget_estimate(sponsorship_matches, language) if is_plugin_enabled("budget_estimator") else ""
    )

    if cached is not None:
        score = cached["fit_score"]
        reasoning = _strip_legacy_score_notes((cached.get("fit_reasoning") or "").strip())
        learning_applied = False
        # Score aus dem Cache wird bewusst NICHT nochmal per Portfolio-Analyse
        # angepasst (sonst würde sich ein gecachter Score bei jeder Wiederholung
        # der gleichen Anfrage weiter verschieben).
        portfolio["score_before_adjustment"] = score
        portfolio["score_after_adjustment"] = score
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
        if is_plugin_enabled("company_intelligence"):
            memory_context += "\n\n" + _build_company_intelligence_context(
                company_intelligence, company_name, language
            )

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
                f"in English, write the content itself in English). Be concise and avoid "
                f"repetition — the whole response must stay under 150 words:\n"
                f"SCORE: <number between 0.0 and 1.0>\n"
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
                f"étiquettes en anglais, rédigez le contenu lui-même en français). Soyez "
                f"concis et évitez les répétitions — la réponse entière doit rester sous "
                f"150 mots :\n"
                f"SCORE: <nombre entre 0.0 et 1.0>\n"
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
                f"Labels auf Englisch, schreibe den Inhalt selbst auf Deutsch). Sei prägnant "
                f"und vermeide Wiederholungen — die gesamte Antwort muss unter 150 Wörtern "
                f"bleiben:\n"
                f"SCORE: <Zahl zwischen 0.0 und 1.0>\n"
                f"PROS: <3-4 kurze Stichpunkte, getrennt durch |>\n"
                f"CONS: <2-3 kurze Stichpunkte, getrennt durch |>\n"
                f"RECOMMENDATION: <genau ein Satz>"
            )

        response = llm.invoke(prompt)
        content = response.content.strip()
        token_usage.append(_track_tokens("evaluate_fit", response))

        score = 0.5
        pros: list[str] = []
        cons: list[str] = []
        recommendation = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("PROS:"):
                pros = [p.strip() for p in line.replace("PROS:", "", 1).split("|") if p.strip()]
            elif line.startswith("CONS:"):
                cons = [c.strip() for c in line.replace("CONS:", "", 1).split("|") if c.strip()]
            elif line.startswith("RECOMMENDATION:"):
                recommendation = line.replace("RECOMMENDATION:", "", 1).strip()

        reasoning = _format_fit_reasoning(pros, cons, recommendation, language) or content

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

    return {
        "fit_score": score,
        "fit_reasoning": reasoning,
        "used_case_studies": case_studies,
        "used_sponsorship_matches": sponsorship_matches,
        "company_intelligence": company_intelligence,
        "competitor_analysis": portfolio,
        "budget_estimate": budget_estimate_text,
        "analysis_id": analysis_id,
        "learning_applied": learning_applied,
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

# --- Agent Evaluation: eigene RAGAs-artige LLM-Judge-Metriken ---
#
# Das echte `ragas`-Paket (getestet: 0.4.3 und 0.2.15) importiert beim Laden
# hart `from langchain_community.chat_models.vertexai import ChatVertexAI`.
# Dieses Submodul existiert in der hier installierten langchain-community-
# Version (>=0.4.2, offiziell "sunset") nicht mehr, weshalb `import ragas`
# mit einem ModuleNotFoundError fehlschlägt – unabhängig vom Rest dieses
# Projekts. Die folgenden Funktionen bilden dieselben drei Bewertungs-
# dimensionen (relevance, faithfulness, answer_relevance) über einen
# eigenen, leichten LLM-Judge nach.

_EVAL_MODEL = "openai/gpt-4o-mini"

def _judge_analysis_quality(row: dict) -> tuple[dict, dict]:
    """Bewertet eine historische Analyse per LLM-Judge auf drei Dimensionen (0.0-1.0).

    Gibt (scores_dict, token_usage_entry) zurück.
    """
    llm = _get_llm(_EVAL_MODEL)

    try:
        case_study_summaries = _json.loads(row.get("case_studies_summary") or "[]")
    except ValueError:
        case_study_summaries = []
    case_studies_text = "\n".join(f"- {s}" for s in case_study_summaries) or "(no case studies used)"

    prompt = (
        "You are an evaluation judge for an AI sponsor-matching agent. Score the "
        "following completed analysis on three dimensions, each from 0.0 to 1.0:\n\n"
        "1. RELEVANCE: how relevant are the case studies (if any) to evaluating this "
        "specific company? If none were used, judge whether that is reasonable.\n"
        "2. FAITHFULNESS: is the reasoning grounded in and consistent with the "
        "research summary and case studies, with no unsupported or contradictory claims?\n"
        "3. ANSWER_RELEVANCE: does the reasoning specifically and directly justify the "
        "given fit score for this company/club pair (vs. generic boilerplate text)?\n\n"
        f"Company: {row['company_name']}\n"
        f"Club: {row['club_name']}\n"
        f"Research summary:\n{row.get('research_summary') or '(none)'}\n\n"
        f"Case studies used:\n{case_studies_text}\n\n"
        f"Fit score given: {row['fit_score']}\n"
        f"Reasoning given: {row.get('fit_reasoning') or '(none)'}\n\n"
        "Respond in exactly this format, nothing before or after:\n"
        "RELEVANCE: <number between 0.0 and 1.0>\n"
        "FAITHFULNESS: <number between 0.0 and 1.0>\n"
        "ANSWER_RELEVANCE: <number between 0.0 and 1.0>"
    )
    response = llm.invoke(prompt)
    content = response.content.strip()

    scores = {"relevance": 0.5, "faithfulness": 0.5, "answer_relevance": 0.5}
    prefixes = {
        "relevance": "RELEVANCE:",
        "faithfulness": "FAITHFULNESS:",
        "answer_relevance": "ANSWER_RELEVANCE:",
    }
    for line in content.split("\n"):
        for key, prefix in prefixes.items():
            if line.startswith(prefix):
                try:
                    scores[key] = max(0.0, min(1.0, float(line.replace(prefix, "").strip())))
                except ValueError:
                    pass

    return scores, _track_tokens("ragas_evaluation", response)

def _compute_trend(per_analysis: list[dict]) -> dict | None:
    """Vergleicht die ältere mit der neueren Hälfte der Batch (nach Zeitstempel
    sortiert), um grob zu erkennen, ob sich der Agent über die Zeit verbessert.
    Gibt None zurück, wenn die Datenbasis dafür zu klein ist.
    """
    if len(per_analysis) < 4:
        return None
    ordered = sorted(per_analysis, key=lambda r: r["timestamp"])
    mid = len(ordered) // 2
    older, newer = ordered[:mid], ordered[mid:]

    def avg_quality(rows):
        return sum((r["relevance"] + r["faithfulness"] + r["answer_relevance"]) / 3 for r in rows) / len(rows)

    older_avg, newer_avg = avg_quality(older), avg_quality(newer)
    delta = newer_avg - older_avg
    if delta > 0.02:
        direction = "improving"
    elif delta < -0.02:
        direction = "declining"
    else:
        direction = "stable"
    return {"older_avg": older_avg, "newer_avg": newer_avg, "delta": delta, "direction": direction}

def _generate_random_baseline(n: int) -> dict:
    """Naiver Zufalls-Baseline-Vergleich, um die Agent-Scores optisch einzuordnen."""
    if n == 0:
        return {"relevance": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0}
    samples = {"relevance": [], "faithfulness": [], "answer_relevance": []}
    for _ in range(n):
        for key in samples:
            samples[key].append(random.uniform(0.0, 1.0))
    return {key: sum(vals) / len(vals) for key, vals in samples.items()}

def evaluate_with_ragas(limit: int = 20) -> dict:
    """Bewertet die letzten `limit` Analysen entlang dreier RAGAs-artiger
    Dimensionen (relevance, faithfulness, answer_relevance) per LLM-Judge,
    aggregiert die Ergebnisse, schätzt einen Zeit-Trend und einen naiven
    Zufalls-Baseline-Vergleich, und schreibt den Report nach
    `data/evaluation_report.json`.
    """
    rows = get_analysis_history(limit=limit)
    if not rows:
        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "num_analyses_evaluated": 0,
            "overall_relevance": None,
            "overall_faithfulness": None,
            "overall_answer_relevance": None,
            "trend": None,
            "baseline": None,
            "per_analysis": [],
            "token_usage": [],
        }
        with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
            _json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    per_analysis = []
    token_usage = []
    for row in rows:
        scores, usage_entry = _judge_analysis_quality(row)
        token_usage.append(usage_entry)
        per_analysis.append({
            "analysis_id": row["id"],
            "company_name": row["company_name"],
            "club_name": row["club_name"],
            "timestamp": row["timestamp"],
            "fit_score": row["fit_score"],
            **scores,
        })

    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "num_analyses_evaluated": len(per_analysis),
        "overall_relevance": sum(r["relevance"] for r in per_analysis) / len(per_analysis),
        "overall_faithfulness": sum(r["faithfulness"] for r in per_analysis) / len(per_analysis),
        "overall_answer_relevance": sum(r["answer_relevance"] for r in per_analysis) / len(per_analysis),
        "trend": _compute_trend(per_analysis),
        "baseline": _generate_random_baseline(len(per_analysis)),
        "per_analysis": per_analysis,
        "token_usage": token_usage,
    }

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)

    return report

from langgraph.graph import StateGraph, START, END

graph = StateGraph(SponsorMatchState)

# Nodes registrieren
graph.add_node("research_company", research_company)
graph.add_node("evaluate_fit", evaluate_fit)
graph.add_node("draft_outreach", draft_outreach)
graph.add_node("explain_rejection", explain_rejection)

# Fester Ablauf: Start -> Recherche -> Bewertung
graph.add_edge(START, "research_company")
graph.add_edge("research_company", "evaluate_fit")

# Verzweigung nach der Bewertung
graph.add_conditional_edges(
    "evaluate_fit",
    route_by_fit,
    ["draft_outreach", "explain_rejection"],
)

# Beide Endpfade führen zum Ende
graph.add_edge("draft_outreach", END)
graph.add_edge("explain_rejection", END)

app = graph.compile()

if __name__ == "__main__":
    import json

    with open("data/clubs.json", "r", encoding="utf-8") as f:
        clubs = json.load(f)

    result = app.invoke({
        "club_profile": clubs["iron_fist_kickboxing"],
        "company_name": "Red Bull GmbH",
        "user_id": None,
        "selected_model": "openai/gpt-4o-mini",
        "language": "de",
        "research_findings": "",
        "fit_score": 0.0,
        "fit_reasoning": "",
        "outreach_draft": "",
        "rejection_reason": "",
        "used_case_studies": [],
        "used_sponsorship_matches": [],
        "company_intelligence": {},
        "analysis_id": 0,
        "learning_applied": False,
        "token_usage": [],
    })

    print("--- Research Findings ---")
    print(result["research_findings"])
    print("\n--- Case-Studies (RAG) ---")
    for case in result["used_case_studies"]:
        print(f"- {case['company']} ({case['sport']}): {case['summary']}")
    print("\n--- Fit Evaluation ---")
    print(f"Score: {result['fit_score']}")
    print(f"Begründung: {result['fit_reasoning']}")

    if result["outreach_draft"]:
        print("\n--- Outreach Draft ---")
        print(result["outreach_draft"])
    else:
        print("\n--- Rejection ---")
        print(result["rejection_reason"])

    print("\n--- Token-Verbrauch ---")
    for entry in result["token_usage"]:
        print(f"{entry['node']}: {entry['total_tokens']} Tokens "
              f"(Input: {entry['input_tokens']}, Output: {entry['output_tokens']})")
    total_tokens = sum(entry["total_tokens"] for entry in result["token_usage"])
    print(f"Gesamt: {total_tokens} Tokens")
