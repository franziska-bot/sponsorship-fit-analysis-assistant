"""FastMCP-Server für Sponsor Match.

Exponiert vier zustandslose Kern-Fähigkeiten der Sponsor-Match-Spezialisten-Agents
(src/search_agent.py, src/analysis_agent.py, src/fit_agent.py) als MCP-Tools, damit
sie z.B. aus Claude Desktop heraus aufgerufen werden können:

    - research_company   Web-Recherche zur Firma (Tavily + LLM-Zusammenfassung)
    - analyze_competitors Sponsoring-Portfolio-Analyse der Firma selbst
    - evaluate_fit        Recherche + Fit-Score/Begründung (nutzt denselben
                           Score-Cache wie main.py)
    - get_size_match       Club-Size vs. Company-Size Matching

Transport: standardmäßig stdio (das Format, das Claude Desktop über den
"command"/"args"-Eintrag in claude_desktop_config.json erwartet, siehe
mcp_config.json). Alternativ per Umgebungsvariable MCP_TRANSPORT=streamable-http
(oder "sse") als Netzwerk-Server auf MCP_PORT (Default 5000) – z.B. zum
eigenständigen Testen ohne Claude Desktop.

Kein eigenes .env-Handling nötig: src.tools ruft beim Import bereits
load_dotenv() auf, TAVILY_API_KEY/OPENROUTER_API_KEY etc. werden also
transitiv geladen, sobald dieses Modul einen der src.*-Spezialisten importiert.
"""

import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.tools import _get_llm
from src.search_agent import research_company as _research_company_node
from src.analysis_agent import (
    _analyze_company_sponsorship_portfolio,
    _build_size_explanation,
    _compute_portfolio_score_impact,
    _compute_saturation_level,
    _compute_size_match_adjustment,
    _compute_size_match_percent,
    _estimate_company_size,
    analyze_financials as _analyze_financials_node,
)
from src.fit_agent import evaluate_fit as _evaluate_fit_node
from src.security_validator import validate_input

BASE_DIR = Path(__file__).resolve().parent

# --- Logging: jeder Tool-Call landet in logs/mcp_server.log (Name, Inputs, Erfolg/Fehler) ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sponsor_match_mcp")

# --- Vereinsprofile laden (dieselbe Quelle wie main.py) ---
with open(BASE_DIR / "data" / "clubs.json", "r", encoding="utf-8") as f:
    _CLUBS = json.load(f)
_CLUBS_BY_NAME = {club["name"].strip().lower(): club for club in _CLUBS.values()}

# Feste Defaults, da die Tools stateless sind (keine Sprachauswahl/Modellwahl per
# Session wie in main.py) – Sprache/Modell könnten bei Bedarf zu echten
# Tool-Parametern gemacht werden, für den MCP-Einstieg reicht ein sinnvoller Default.
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_LANGUAGE = "de"

TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
PORT = int(os.environ.get("MCP_PORT", "5000"))

mcp = FastMCP("sponsor-match", port=PORT)


class ValidationError(ValueError):
    """Eingabefehler (fehlende/ungültige company_name oder club_profile)."""


def _validate_company_name(company_name: str) -> str:
    company_name = (company_name or "").strip()
    if not company_name:
        raise ValidationError("company_name darf nicht leer sein.")
    # MCP-Tool-Aufrufe durchlaufen nicht main.py's UI-Validierung (Streamlit-
    # Formular, Rate-Limiting) – ohne diesen Check würden Angriffsmuster
    # (Prompt-Injection, SQL-/Command-Injection, ...) hier ungefiltert bis in
    # die LLM-Prompts/Suchanfragen durchgereicht.
    is_safe, violations = validate_input(company_name)
    if not is_safe:
        raise ValidationError(f"company_name enthält unzulässige Muster: {', '.join(violations)}")
    return company_name


def _resolve_club(club_profile: str) -> dict:
    """club_profile ist der Vereinsname (z.B. 'FC Nordlicht'), siehe data/clubs.json."""
    club = _CLUBS_BY_NAME.get((club_profile or "").strip().lower())
    if club is None:
        valid = ", ".join(c["name"] for c in _CLUBS.values())
        raise ValidationError(f"Unbekannter Verein '{club_profile}'. Gültige Vereine: {valid}")
    return club


@mcp.tool()
def research_company(company_name: str, club_profile: str) -> dict:
    """Recherchiert eine Firma im Web (Tavily) und fasst Branche, bisherige
    Sponsoring-Aktivitäten, Zielgruppe und Markenwerte in 4-5 Sätzen zusammen.

    club_profile (Vereinsname, siehe data/clubs.json) wird nur validiert – die
    Recherche selbst ist unabhängig vom Verein.
    """
    logger.info("tool call: research_company(company_name=%r, club_profile=%r)", company_name, club_profile)
    try:
        company_name = _validate_company_name(company_name)
        _resolve_club(club_profile)
        state = {"company_name": company_name, "selected_model": DEFAULT_MODEL, "language": DEFAULT_LANGUAGE}
        result = _research_company_node(state)
        logger.info("research_company OK für %r", company_name)
        return {"success": True, "research_findings": result["research_findings"]}
    except ValidationError as exc:
        logger.warning("research_company Validierungsfehler: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("research_company fehlgeschlagen")
        return {"success": False, "error": f"Interner Fehler: {exc}"}


@mcp.tool()
def analyze_competitors(company_name: str, club_profile: str) -> dict:
    """Analysiert generisch für JEDE Firma deren eigenes Sponsoring-Portfolio
    (Top-Kategorien, aktive Sponsorings, Zielgruppe) sowie Marktsättigung und
    Zielgruppen-Match zum angegebenen Verein (club_profile = Vereinsname).
    """
    logger.info("tool call: analyze_competitors(company_name=%r, club_profile=%r)", company_name, club_profile)
    try:
        company_name = _validate_company_name(company_name)
        club = _resolve_club(club_profile)
        llm = _get_llm(DEFAULT_MODEL)
        portfolio = _analyze_company_sponsorship_portfolio(company_name, club, DEFAULT_LANGUAGE, llm)
        if portfolio.get("found"):
            portfolio["saturation_level"] = _compute_saturation_level(portfolio["same_sport_count"])
            portfolio["score_adjustment"] = _compute_portfolio_score_impact(
                portfolio["match_percent"], portfolio["saturation_level"]
            )
        else:
            portfolio["saturation_level"] = "low"
            portfolio["score_adjustment"] = 0.0
        portfolio.pop("token_usage", None)
        logger.info("analyze_competitors OK für %r", company_name)
        return {"success": True, **portfolio}
    except ValidationError as exc:
        logger.warning("analyze_competitors Validierungsfehler: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("analyze_competitors fehlgeschlagen")
        return {"success": False, "error": f"Interner Fehler: {exc}"}


@mcp.tool()
def evaluate_fit(company_name: str, club_profile: str) -> dict:
    """Führt Web-Recherche + Fit-Bewertung durch und liefert Score (0.0-1.0),
    strukturierte Begründung (Was passt gut/weniger, Empfehlung), sowie ob der
    Score als unsicher gilt (0.45-0.55) und die Agent-Confidence.

    Nutzt denselben Score-Cache wie main.py: identische company_name+club_profile-
    Anfragen liefern einen stabilen Score. Die Analyse wird wie in main.py in der
    SQLite-DB (data/sponsor_match.db) gespeichert und zählt zum Analyseverlauf.
    """
    logger.info("tool call: evaluate_fit(company_name=%r, club_profile=%r)", company_name, club_profile)
    try:
        company_name = _validate_company_name(company_name)
        club = _resolve_club(club_profile)

        research_state = {
            "company_name": company_name,
            "selected_model": DEFAULT_MODEL,
            "language": DEFAULT_LANGUAGE,
        }
        research_result = _research_company_node(research_state)

        financials_state = {
            "club_profile": club,
            "company_name": company_name,
            "selected_model": DEFAULT_MODEL,
            "language": DEFAULT_LANGUAGE,
        }
        financials_result = _analyze_financials_node(financials_state)

        eval_state = {
            "club_profile": club,
            "company_name": company_name,
            "user_id": None,
            "selected_model": DEFAULT_MODEL,
            "language": DEFAULT_LANGUAGE,
            "research_findings": research_result["research_findings"],
            "competitor_analysis": financials_result["competitor_analysis"],
            "size_compatibility": financials_result["size_compatibility"],
            "pdf_financials": financials_result.get("pdf_financials", {}),
        }
        result = _evaluate_fit_node(eval_state)
        logger.info("evaluate_fit OK für %r: score=%.2f", company_name, result["fit_score"])
        return {
            "success": True,
            "fit_score": result["fit_score"],
            "fit_reasoning": result["fit_reasoning"],
            "is_uncertain": result["is_uncertain"],
            "agent_confidence": result["agent_confidence"],
            "research_findings": research_result["research_findings"],
        }
    except ValidationError as exc:
        logger.warning("evaluate_fit Validierungsfehler: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("evaluate_fit fehlgeschlagen")
        return {"success": False, "error": f"Interner Fehler: {exc}"}


@mcp.tool()
def get_size_match(company_name: str, club_profile: str) -> dict:
    """Vergleicht Club-Größe (statisch je Verein, siehe data/clubs.json) und
    Company-Größe (per Web-Suche + LLM geschätzt: Small/Medium/Large) und
    berechnet den daraus resultierenden Score-Impact sowie einen Match-Prozentsatz.
    """
    logger.info("tool call: get_size_match(company_name=%r, club_profile=%r)", company_name, club_profile)
    try:
        company_name = _validate_company_name(company_name)
        club = _resolve_club(club_profile)
        llm = _get_llm(DEFAULT_MODEL)

        club_size = club.get("size", "Medium")
        company_size_result = _estimate_company_size(company_name, llm)
        adjustment = _compute_size_match_adjustment(club_size, company_size_result["size"])
        match_percent = _compute_size_match_percent(adjustment)
        explanation = _build_size_explanation(
            club_size, company_size_result["size"], match_percent, DEFAULT_LANGUAGE
        )
        logger.info("get_size_match OK für %r: %s/%s -> %d%%", company_name, club_size, company_size_result["size"], match_percent)
        return {
            "success": True,
            "club_size": club_size,
            "company_size": company_size_result["size"],
            "match_percent": match_percent,
            "score_adjustment": adjustment,
            "explanation": explanation,
        }
    except ValidationError as exc:
        logger.warning("get_size_match Validierungsfehler: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("get_size_match fehlgeschlagen")
        return {"success": False, "error": f"Interner Fehler: {exc}"}


if __name__ == "__main__":
    logger.info(
        "Sponsor Match MCP Server startet (transport=%s%s)",
        TRANSPORT,
        f", port={PORT}" if TRANSPORT != "stdio" else "",
    )
    mcp.run(transport=TRANSPORT)
