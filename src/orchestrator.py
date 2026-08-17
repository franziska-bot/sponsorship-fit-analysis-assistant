"""Manager agent: owns long-term memory (SQLite persistence, user auth),
observability (LangSmith trace URLs, RAGAs-style evaluation), and builds +
runs the LangGraph pipeline that delegates to the search/analysis/fit
specialists.
"""

import os
import sqlite3
import datetime
import random
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
import json as _json

import bcrypt

from src.logger import AgentLogger
from src.tools import SponsorMatchState, _get_llm, _track_tokens

_logger = AgentLogger("orchestrator")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "sponsor-match")

from langsmith import Client

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
                user_id INTEGER,
                hitl_decision TEXT,
                corrected_score REAL
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
            "ALTER TABLE analyses ADD COLUMN hitl_decision TEXT",
            "ALTER TABLE analyses ADD COLUMN corrected_score REAL",
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


def update_analysis_hitl_decision(analysis_id: int, decision: str) -> None:
    """Trägt die Human-in-the-Loop-Entscheidung ('agree', 'disagree' oder 'need_more_info')
    zu einer unsicheren Analyse (Score 0.45-0.55) nach. 'agree'/'disagree' gelten als Ground
    Truth für künftige Score-Anpassungen bei ähnlichen Anfragen derselben Firma (siehe
    compute_hitl_adjustment); 'need_more_info' fließt bewusst nicht ins Learning ein."""
    with _get_connection() as conn:
        conn.execute("UPDATE analyses SET hitl_decision = ? WHERE id = ?", (decision, analysis_id))


def get_resolved_hitl_decisions(company_name: str) -> list[dict]:
    """Holt alle bereits per Human Review aufgelösten Ground-Truth-Entscheidungen (agree/
    disagree, NICHT need_more_info) zu ähnlichen Sponsoring-Anfragen dieser Firma."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE company_name LIKE ? AND hitl_decision IN ('agree', 'disagree') "
            "ORDER BY timestamp DESC",
            (f"%{company_name}%",),
        ).fetchall()
    return [dict(row) for row in rows]


def update_analysis_corrected_score(analysis_id: int, corrected_score: float) -> None:
    """Speichert eine manuelle Score-Korrektur des Users als Ground Truth für diese
    Analyse (nach negativem Feedback). Überschreibt zusätzlich fit_score direkt auf
    dieser Zeile, damit der bestehende Score-Cache (get_exact_previous_analysis, exakter
    Company+Club-Match) bei der nächsten identischen Anfrage automatisch den
    korrigierten Wert liefert – ohne eigene neue Caching-Logik."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE analyses SET corrected_score = ?, fit_score = ? WHERE id = ?",
            (corrected_score, corrected_score, analysis_id),
        )


def count_score_corrections(company_name: str, club_name: str) -> int:
    """Zählt, wie oft der Score für exakt diese Firma+Verein-Kombination bereits manuell
    korrigiert wurde – Grundlage für die Confidence-Anzeige nach einer Korrektur."""
    with _get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE company_name = ? AND club_name = ? "
            "AND corrected_score IS NOT NULL",
            (company_name, club_name),
        ).fetchone()[0]


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


# --- Manager: delegiert an die Spezialisten-Agents via LangGraph ---

from langgraph.graph import StateGraph, START, END

from src.search_agent import research_company
from src.analysis_agent import analyze_financials
from src.fit_agent import evaluate_fit, draft_outreach, explain_rejection, route_by_fit
from src.performance_monitor import PerformanceMonitor
from src.security_validator import validate_input, validate_club_profile_values

# Graph-Node -> Performance-Bucket: draft_outreach/explain_rejection laufen
# beide in fit_agent.py (wie evaluate_fit) und zählen daher ebenfalls auf
# dessen Timing-Bucket, statt eigene Buckets für zwei sich gegenseitig
# ausschließende Nodes zu erzeugen (main.py fragt gezielt "*_avg" für
# search_agent/analysis_agent/fit_agent ab, siehe Phase 3d).
_NODE_TO_AGENT_BUCKET = {
    "research_company": "search_agent",
    "analyze_financials": "analysis_agent",
    "evaluate_fit": "fit_agent",
    "draft_outreach": "fit_agent",
    "explain_rejection": "fit_agent",
}


class SecurityValidationError(ValueError):
    """Eingabe hat mindestens ein Sicherheits-Pattern verletzt (siehe security_validator.py)."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__(f"Input blocked by security validator: {', '.join(violations)}")


class OrchestratingAgent:
    """Manager-Agent: validiert die Eingabe und delegiert dann an die Search-,
    Analysis- und Fit-Spezialisten über eine kompilierte LangGraph-Pipeline
    (research_company -> analyze_financials -> evaluate_fit ->
    draft_outreach | explain_rejection).
    """

    def __init__(self):
        self._graph = self._build_graph()

    @staticmethod
    def _build_graph():
        graph = StateGraph(SponsorMatchState)

        graph.add_node("research_company", research_company)
        graph.add_node("analyze_financials", analyze_financials)
        graph.add_node("evaluate_fit", evaluate_fit)
        graph.add_node("draft_outreach", draft_outreach)
        graph.add_node("explain_rejection", explain_rejection)

        graph.add_edge(START, "research_company")
        graph.add_edge("research_company", "analyze_financials")
        graph.add_edge("analyze_financials", "evaluate_fit")
        graph.add_conditional_edges(
            "evaluate_fit",
            route_by_fit,
            ["draft_outreach", "explain_rejection"],
        )
        graph.add_edge("draft_outreach", END)
        graph.add_edge("explain_rejection", END)

        return graph.compile()

    def invoke(self, state: SponsorMatchState) -> dict:
        # Defense-in-depth: main.py validiert Company-Namen bereits vor dem
        # Aufruf von app.invoke(...), dieser Check greift also normalerweise
        # nie – schützt aber jeden anderen Aufrufer von OrchestratingAgent
        # direkt (z.B. zukünftige Skripte), der main.py's UI-Validierung
        # nicht durchläuft.
        company_name = state["company_name"]
        _logger.log_agent_start("pipeline", company=company_name)

        is_safe, violations = validate_input(company_name)
        if not is_safe:
            _logger.log_error("pipeline", f"security validation failed: {violations}", company=company_name)
            raise SecurityValidationError(violations)

        # Security Update: Club-Profile kommen aktuell nur aus der statischen
        # data/clubs.json (nie aus Nutzereingabe), landen aber direkt in
        # LLM-Prompts (fit_agent.py) – reine Vorsorge für den Fall, dass
        # Club-Profile künftig einmal editierbar werden.
        club_is_safe, club_violations = validate_club_profile_values(state["club_profile"])
        if not club_is_safe:
            _logger.log_error(
                "pipeline", f"club profile security validation failed: {club_violations}", company=company_name
            )
            raise SecurityValidationError(club_violations)

        # .stream(stream_mode="updates") statt .invoke(): liefert nach jedem
        # Node nur dessen eigenen Delta-Output (nicht den kompletten State),
        # was echtes Logging des Orchestrierungs-Flows zwischen den
        # Spezialisten ermöglicht – ohne den Graphen ein zweites Mal
        # auszuführen (kein doppelter LLM-/API-Kostenaufwand für reines
        # Logging). Der volle State wird manuell nachgebildet, indem jedes
        # Node-Delta in `accumulated` gemerged wird: token_usage (das
        # einzige Feld mit Annotated[list, operator.add] in
        # SponsorMatchState) per Listen-Append statt Überschreiben – exakt
        # dasselbe Merge-Verhalten, das LangGraphs eigener Reducer bei einem
        # normalen .invoke()-Aufruf anwenden würde.
        accumulated: dict = dict(state)
        monitor = PerformanceMonitor()
        # Jeder .stream()-Tick liefert genau EIN abgeschlossenes Node-Delta, also
        # ist die seit dem vorherigen Tick vergangene Wall-Clock-Zeit exakt die
        # Laufzeit dieses einen Node – ohne die Pipeline (wie bei .invoke()) ein
        # zweites Mal auszuführen oder jeden Agenten einzeln zu timen.
        last_tick = time.monotonic()
        try:
            for update in self._graph.stream(state, stream_mode="updates"):
                for node_name, delta in update.items():
                    now = time.monotonic()
                    duration_ms = (now - last_tick) * 1000
                    last_tick = now
                    agent_bucket = _NODE_TO_AGENT_BUCKET.get(node_name)
                    if agent_bucket:
                        monitor.record_agent_execution(agent_bucket, duration_ms)
                    _logger.log_agent_step(node_name, keys=list(delta.keys()), duration_ms=round(duration_ms, 1))
                    for key, value in delta.items():
                        if key == "token_usage":
                            accumulated["token_usage"] = accumulated.get("token_usage", []) + value
                        else:
                            accumulated[key] = value
        except Exception as exc:
            _logger.log_error("pipeline", exc, company=company_name)
            raise

        accumulated["performance_metrics"] = monitor.to_dict()

        _logger.log_agent_result(
            "pipeline", company=company_name, fit_score=accumulated.get("fit_score"),
            agent_confidence=accumulated.get("agent_confidence"),
            performance_metrics=accumulated["performance_metrics"],
        )
        return accumulated

    run = invoke


app = OrchestratingAgent()


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
        "research_quality": {},
        "fit_score": 0.0,
        "fit_reasoning": "",
        "outreach_draft": "",
        "rejection_reason": "",
        "used_case_studies": [],
        "used_sponsorship_matches": [],
        "competitor_analysis": {},
        "budget_estimate": "",
        "size_compatibility": {},
        "pdf_financials": {},
        "financial_data": {},
        "analysis_id": 0,
        "learning_applied": False,
        "is_uncertain": False,
        "agent_confidence": 0,
        "fit_agent_factors": {},
        "hitl_resolved_count": 0,
        "token_usage": [],
        "performance_metrics": {},
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

    print("\n--- Performance ---")
    perf = result["performance_metrics"]
    for agent in ("search_agent", "analysis_agent", "fit_agent"):
        if f"{agent}_avg" in perf:
            print(f"{agent}: avg={perf[f'{agent}_avg']:.0f}ms total={perf[f'{agent}_total']:.0f}ms")
    print(f"Total: {perf.get('total_time', 0):.0f}ms | Slowest: {perf.get('slowest_agent')}")
