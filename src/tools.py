"""Shared leaf infra used by every specialist agent and the orchestrator.

Deliberately dependency-free w.r.t. the rest of `src/` so it can be imported
first by anyone (search_agent, analysis_agent, fit_agent, orchestrator,
mcp_server.py) without any circular-import risk.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json as _json
from functools import lru_cache
from typing_extensions import TypedDict
from typing import Annotated
import operator

# LANGSMITH_TRACING must be set before any @traceable-decorated function is
# defined in the specialist modules that import this one first.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "sponsor-match")

from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI


def get_search_tool():
    """Erstellt das Tavily-Such-Tool für die Firmenrecherche."""
    return TavilySearch(
        max_results=5,
        api_key=os.environ["TAVILY_API_KEY"],
    )


class SponsorMatchState(TypedDict):
    club_profile: dict       # gewähltes Vereins-Profil aus clubs.json
    company_name: str        # Nutzereingabe: Firma, die geprüft werden soll
    user_id: int | None      # ID des eingeloggten Users (für personalisierten Verlauf)
    selected_model: str       # gewähltes LLM-Modell (OpenRouter-Modell-ID)
    language: str             # Zielsprache für alle LLM-Ausgaben: "de" (default), "en", "fr"
    research_findings: str   # Ergebnis von research_company
    research_quality: dict   # Data-Quality-Metriken zur Recherche (sources_found, credibility, ...)
    fit_score: float         # Ergebnis von evaluate_fit, z.B. 0.0–1.0
    fit_reasoning: str        # Begründung für den Score
    outreach_draft: str       # nur befüllt bei gutem Fit
    rejection_reason: str     # nur befüllt bei schlechtem Fit
    used_case_studies: list   # RAG-Treffer aus der Case-Study-Wissensbasis
    used_sponsorship_matches: list  # Treffer aus der externen Sponsorship-Datenbank
    competitor_analysis: dict  # strukturierte Konkurrentenliste + Score-/Marktsättigungs-Impact (competitor_analysis-Plugin)
    budget_estimate: str  # Ergebnis von _build_budget_estimate (budget_estimator-Plugin)
    size_compatibility: dict  # Club-/Company-Size-Match + Score-Impact (size_matching-Plugin)
    pdf_financials: dict  # Aus PDF-Geschäftsberichten extrahierte Finanzkennzahlen (budget_estimator-Plugin, Phase 2)
    financial_data: dict  # Data-Quality-Auszug aus pdf_financials (pdfs_found/parsed, metrics_missing, ...)
    analysis_id: int          # ID des in der SQLite-DB gespeicherten Analyse-Datensatzes
    learning_applied: bool    # True, wenn frühere Feedback-Muster den Score angepasst haben
    is_uncertain: bool        # True, wenn fit_score im unsicheren Band 0.45-0.55 liegt (HITL-Trigger)
    agent_confidence: int     # Agent-Confidence in %, für die Human-in-the-Loop-Anzeige
    fit_agent_factors: dict   # localized-Label -> 0.0-1.0 Konfidenz pro Bewertungsfaktor (leer bei Score-Cache-Treffer)
    hitl_resolved_count: int  # Anzahl bereits per Human Review aufgelöster Entscheidungen zu dieser Firma
    token_usage: Annotated[list, operator.add]  # Tokenverbrauch pro LLM-Aufruf
    performance_metrics: dict  # Timing-Breakdown pro Agent (PerformanceMonitor.to_dict())


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


def _track_tokens(node_name: str, response) -> dict:
    """Extrahiert die Token-Nutzung einer LLM-Antwort für das Tracking."""
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "node": node_name,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


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
