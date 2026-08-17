# Sponsor Match

*[English version](README.en.md)*

Sponsor Match ist eine Streamlit-App, die mit einem [LangGraph](https://www.langchain.com/langgraph)-Agenten bewertet, wie gut eine Firma als Sponsor zu einem Sportverein passt. Der Agent recherchiert die Firma im Web, vergleicht sie mit dem Vereinsprofil und liefert einen Fit-Score (0.0–1.0) mit Begründung, sowie je nach Ergebnis einen Ansprache-Entwurf oder eine Ablehnungsbegründung.

## Features

- **Recherche & Fit-Bewertung**: Web-Recherche zur Firma (Tavily), LLM-basierte Bewertung mit Score, Pro-/Contra-Stichpunkten und einer Empfehlung.
- **Konkurrenzanalyse (Portfolio-Plugin)**: Analysiert generisch für jede Firma deren eigenes Sponsoring-Portfolio (Kategorien, aktive Sponsorings, Zielgruppe), Zielgruppen-Match zum Verein und Marktsättigung in der jeweiligen Sportart – mit dynamischer Score-Anpassung.
- **Size Matching**: Vergleicht Club-Größe (statisch je Verein) und Company-Größe (per Web-Suche + LLM geschätzt: Small/Medium/Large) und passt den Score anhand einer Größen-Match-Matrix an.
- **Budget Estimator & externe Sponsorship-DB**: Datenbasierte Budget-Schätzung und historische Sponsoring-Beispiele aus einer fiktiven externen Datenbank.
- **Case-Study-Wissensbasis**: RAG-Suche über fiktive Sponsoring-Case-Studies als zusätzlicher Kontext für die Bewertung.
- **Human-in-the-Loop (HITL)**: Bei unsicheren Scores (0.45–0.55) kann der Nutzer der Einschätzung zustimmen, widersprechen oder weitere Informationen anfordern – der Agent lernt daraus für künftige Anfragen zur selben Firma.
- **Manuelle Score-Korrektur**: Nach negativem Feedback kann der Nutzer per Slider den korrekten Score festlegen; dieser wird als Ground Truth gespeichert und beim nächsten Mal für dieselbe Firma+Verein-Kombination verwendet.
- **Agent Learning**: Feedback (positiv/negativ) zu ähnlichen Firmen beeinflusst künftige Scores leicht (±0.05 bis ±0.10).
- **Score-Caching**: Identische Firma+Verein-Anfragen liefern einen stabilen, konsistenten Score statt bei jeder Wiederholung neu zu schwanken.
- **Analyseverlauf & Agent Evaluation**: Filterbarer Verlauf aller Analysen, Score-Consistency-Metrik sowie ein LLM-Judge-basierter RAGAs-artiger Qualitätsreport (Relevance, Faithfulness, Answer Relevance, Trend über Zeit).
- **Plugin-System**: Jede optionale Fähigkeit (Konkurrenzanalyse, Budget Estimator, Sponsorship DB, Size Matching) lässt sich im Plugin Manager einzeln ein-/ausschalten – Änderungen wirken sofort auf die Anzeige, ohne die Analyse neu laufen zu lassen.
- **Mehrsprachigkeit**: Deutsch, Englisch, Französisch (UI-Texte und Agent-Antworten).
- **Nutzerverwaltung**: Login/Registrierung, bevorzugte Sprache & Lieblingsvereine, Account-Löschung, Session-Timeout.
- **LangSmith-Tracing**: Optionaler Trace-Link pro Analyse zur Fehlersuche/Nachvollziehbarkeit.
- **MCP Server**: `mcp_server.py` exponiert Recherche, Konkurrenzanalyse, Fit-Bewertung und Size Matching als MCP-Tools (z.B. für Claude Desktop) – siehe [README_MCP.md](README_MCP.md).

## Tech-Stack

- **UI**: [Streamlit](https://streamlit.io/) (Dark-Theme, eigenes CSS)
- **Agent**: [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/) über [OpenRouter](https://openrouter.ai/) (austauschbares LLM)
- **Web-Suche**: [Tavily](https://tavily.com/)
- **Persistenz**: SQLite (`data/sponsor_match.db` für Analysen, `data/users.db` für Nutzerkonten)
- **Tracing**: [LangSmith](https://smith.langchain.com/) (optional)
- **MCP-Server**: [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (`mcp>=1.9,<2`)
- **Package-Management**: [uv](https://docs.astral.sh/uv/)

## Setup

### Voraussetzungen

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

```bash
uv sync
```

### Umgebungsvariablen

Eine `.env`-Datei im Projekt-Root anlegen:

```bash
TAVILY_API_KEY=...          # erforderlich – Web-Suche
OPENROUTER_API_KEY=...      # erforderlich – LLM-Zugriff über OpenRouter
LANGSMITH_API_KEY=...       # optional – aktiviert Tracing/Trace-Links
LANGSMITH_PROJECT=sponsor-match   # optional, Default: sponsor-match
```

### Starten

```bash
uv run streamlit run main.py
```

Die App ist danach unter `http://localhost:8501` erreichbar. Beim ersten Start über "Register" einen Account anlegen.

## Projektstruktur

```
main.py                        # Streamlit-UI: Auth, Sidebar, Formular, Ergebnis-Karten
mcp_server.py                  # FastMCP-Server: Recherche/Konkurrenzanalyse/Fit/Size Matching als MCP-Tools
mcp_config.json                # Fertiger Claude-Desktop-Konfigurationseintrag
src/orchestrator.py            # Manager: LangGraph-Pipeline, SQLite-Persistenz, RAGAs-artiges Eval
src/search_agent.py            # Spezialist: Web-Recherche, Case-Study-RAG, Sponsorship-DB-Suche
src/analysis_agent.py          # Spezialist: Sponsoring-Portfolio, Company-Size, Budget-Schätzung
src/fit_agent.py                # Spezialist: Fit-Scoring, Outreach-Entwurf, Ablehnungsbegründung
src/security_validator.py      # Input-Security: 11 Angriffsmuster-Kategorien
src/tools.py                   # Shared Infra: Tavily-Such-Tool, LLM-Factory, State-Schema, Plugins
data/clubs.json                # Vereinsprofile (Sportart, Fanbase, Werte, Size, ...)
data/case_studies.json         # Fiktive Case-Study-Wissensbasis
data/sponsorship_database.json # Fiktive externe Sponsorship-Datenbank
data/available_plugins.json    # Plugin-Konfiguration (an/aus, Pflicht-Plugins)
.streamlit/config.toml         # Farbschema (Dark Mode)
```

`data/sponsor_match.db`, `data/users.db`, `data/feedback.jsonl`, `data/security_log.jsonl` und `logs/` werden zur Laufzeit erzeugt und sind bewusst nicht Teil des Repos (siehe `.gitignore`).

## Hinweise

- Alle Case-Studies, die externe Sponsorship-DB sowie Company-Size- und Konkurrenzanalyse-Werte sind KI-Schätzungen bzw. fiktive Demodaten – keine verifizierten Marktdaten.
- Die Frage-Erkennung im Firmen-Eingabefeld (Schutz vor Prompt-Injection-artigen Eingaben) prüft aktuell primär auf deutsche Fragewörter/​"?".
