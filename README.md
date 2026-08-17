# Sponsor Match

*[English version](README.en.md)*

Sponsor Match ist eine Streamlit-App, die mit einem [LangGraph](https://www.langchain.com/langgraph)-Agenten bewertet, wie gut eine Firma als Sponsor zu einem Sportverein passt. Der Agent recherchiert die Firma im Web, vergleicht sie mit dem Vereinsprofil und liefert einen Fit-Score (0.0–1.0) mit Begründung, sowie je nach Ergebnis einen Ansprache-Entwurf oder eine Ablehnungsbegründung.

## Features

- **Recherche & Fit-Bewertung**: Web-Recherche zur Firma (Tavily), LLM-basierte Bewertung mit Score, Pro-/Contra-Stichpunkten und einer Empfehlung.
- **Konkurrenzanalyse (Portfolio-Plugin)**: Analysiert generisch für jede Firma deren eigenes Sponsoring-Portfolio (Kategorien, aktive Sponsorings, Zielgruppe), Zielgruppen-Match zum Verein und Marktsättigung in der jeweiligen Sportart – mit dynamischer Score-Anpassung.
- **Size Matching**: Vergleicht Club-Größe (statisch je Verein) und Company-Größe (per Web-Suche + LLM geschätzt: Small/Medium/Large) und passt den Score anhand einer Größen-Match-Matrix an.
- **Budget Estimator & externe Sponsorship-DB**: Datenbasierte Budget-Schätzung und historische Sponsoring-Beispiele aus einer fiktiven externen Datenbank, ergänzt um echte Finanzkennzahlen (Umsatz, EBITDA, Gewinn, Cash, Marketingausgaben), die best-effort aus online gefundenen Geschäftsberichten (PDF) extrahiert werden.
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
- **MCP Server**: `mcp_server.py` exponiert Recherche, Konkurrenzanalyse, Fit-Bewertung und Size Matching als MCP-Tools (z.B. für Claude Desktop) – Setup siehe Dokumentation unten.
- **Observability**: Detailliertes Agent-Tracing (`logs/agent_trace.log`), Performance-Metriken pro Agent (Laufzeit-Breakdown) sowie Data-Quality-Metriken (Quellen-Glaubwürdigkeit, PDF-Extraktions-Konfidenz, Faktor-Breakdown pro Bewertungskriterium) direkt in der Ergebnis-Ansicht.
- **PDF-Cache & Error Handling**: Gefundene Geschäftsberichte werden lokal gecacht (30 Tage für PDFs, 60 Tage für extrahierte Kennzahlen), Downloads laufen mit Exponential-Backoff-Retry (1s/2s/4s) und schlagen bei Fehlern graceful fehl, statt die ganze Analyse abzubrechen.
- **Security**: 70+ Angriffsmuster-Erkennung (Prompt-Injection, Jailbreaks, Instruction-Chaining, Context-Confusion, Obfuskation via Base64/ROT13/Hex/Leetspeak/Cyrillic-Lookalikes, technische Exploits) auf Deutsch und Englisch, zweistufiges Rate Limiting (Session- und IP-basiert) sowie angriffs-spezifisches Banning (Temp-Ban nach 3 Angriffen/Minute, permanenter Block + Admin-Alert nach 20/Stunde).
- **Health-Check-Endpoint**: Optionaler eigenständiger Monitoring-Server (`src/health_server.py`) für DevOps/Betrieb.

## Tech-Stack

- **UI**: [Streamlit](https://streamlit.io/) (Dark-Theme, eigenes CSS)
- **Agent**: [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/) über [OpenRouter](https://openrouter.ai/) (austauschbares LLM)
- **Web-Suche**: [Tavily](https://tavily.com/)
- **Persistenz**: SQLite (`data/sponsor_match.db` für Analysen, `data/users.db` für Nutzerkonten)
- **Tracing**: [LangSmith](https://smith.langchain.com/) (optional)
- **MCP-Server**: [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (`mcp>=1.9,<2`)
- **PDF-Extraktion**: [pdfplumber](https://github.com/jsvine/pdfplumber) (Text + Tabellen aus Geschäftsberichten)
- **Tests**: [pytest](https://pytest.org/)
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
src/analysis_agent.py          # Spezialist: Sponsoring-Portfolio, Company-Size, Budget-Schätzung, PDF-Extraktion
src/fit_agent.py                # Spezialist: Fit-Scoring, Outreach-Entwurf, Ablehnungsbegründung
src/security_validator.py      # Input-Security: 70+ Angriffsmuster-Kategorien + Rate Limiting/Banning
src/logger.py                  # AgentLogger: strukturiertes Tracing + Security-Audit-Log
src/pdf_cache.py               # PDFCache: Zwei-Stufen-Cache für Geschäftsbericht-PDFs
src/error_handler.py           # ErrorHandler: Retry mit Exponential Backoff, Graceful Degradation
src/performance_monitor.py     # PerformanceMonitor: Laufzeit-Tracking pro Agent
src/health_server.py           # Eigenständiger Health-Check-Server (Monitoring, optional)
src/tools.py                   # Shared Infra: Tavily-Such-Tool, LLM-Factory, State-Schema, Plugins
tests/test_security_injection.py  # 122 Security-Tests (Angriffsmuster + Legitim-Input-Fälle)
data/clubs.json                # Vereinsprofile (Sportart, Fanbase, Werte, Size, ...)
data/case_studies.json         # Fiktive Case-Study-Wissensbasis
data/sponsorship_database.json # Fiktive externe Sponsorship-Datenbank
data/available_plugins.json    # Plugin-Konfiguration (an/aus, Pflicht-Plugins)
.streamlit/config.toml         # Farbschema (Dark Mode)
```

`data/sponsor_match.db`, `data/users.db`, `data/feedback.jsonl`, `data/security_log.jsonl`, `data/admin_alerts.jsonl`, `data/pdf_cache/` und `logs/` werden zur Laufzeit erzeugt und sind bewusst nicht Teil des Repos (siehe `.gitignore`).

## Weitere Dokumentation

- [README_AGENTS.md](README_AGENTS.md) – Agent-Architektur im Detail: Datenfluss, Logging, Caching, Performance, Fehlerbehandlung.
- [MULTI_AGENT_ARCHITECTURE.md](MULTI_AGENT_ARCHITECTURE.md) – LangGraph-State-Machine, Modul-Abhängigkeiten, vollständiges State-Schema.
- [QUALITY_METRICS.md](QUALITY_METRICS.md) – Was die Konfidenz-Werte bedeuten und wann man ihnen vertrauen sollte.
- [DEPLOYMENT.md](DEPLOYMENT.md) – Setup, Monitoring, Performance-Tuning, Troubleshooting.
- [README_MCP.md](README_MCP.md) – MCP-Server-Einrichtung (z.B. für Claude Desktop).

## Hinweise

- Alle Case-Studies, die externe Sponsorship-DB sowie Company-Size- und Konkurrenzanalyse-Werte sind KI-Schätzungen bzw. fiktive Demodaten – keine verifizierten Marktdaten.
- Die Frage-Erkennung im Firmen-Eingabefeld (Schutz vor Prompt-Injection-artigen Eingaben) prüft aktuell primär auf deutsche Fragewörter/​"?".
