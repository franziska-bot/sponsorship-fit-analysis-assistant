# Sponsor Match – MCP Server

`mcp_server.py` exponiert vier zustandslose Kern-Fähigkeiten des Sponsor-Match-Agents
(`src/agent.py`) als [MCP](https://modelcontextprotocol.io/)-Tools, gebaut mit
[FastMCP](https://github.com/modelcontextprotocol/python-sdk) (`mcp.server.fastmcp.FastMCP`).

## Tools

Alle vier Tools nehmen dieselben zwei Parameter entgegen:

| Parameter      | Typ    | Bedeutung                                                        |
|----------------|--------|--------------------------------------------------------------------|
| `company_name` | string | Name der zu prüfenden Firma, z.B. `"Nike"`                        |
| `club_profile` | string | Vereinsname aus `data/clubs.json` (`FC Nordlicht`, `Riverside Hawks`, `Iron Fist Kickboxing Club`) |

| Tool                 | Beschreibung |
|----------------------|--------------|
| `research_company`   | Web-Recherche zur Firma (Tavily) + LLM-Zusammenfassung (Branche, Sponsoring-Historie, Zielgruppe, Markenwerte). |
| `analyze_competitors` | Analysiert generisch das eigene Sponsoring-Portfolio der Firma (Kategorien, aktive Sponsorings, Zielgruppe) sowie Marktsättigung und Zielgruppen-Match zum Verein. |
| `evaluate_fit`        | Recherche + Fit-Bewertung: Score (0.0–1.0), strukturierte Begründung, Unsicherheits-Flag, Agent-Confidence. |
| `get_size_match`      | Vergleicht Club-Größe und (per Web-Suche geschätzte) Company-Größe, liefert Score-Impact + Match-Prozent. |

Jedes Tool gibt `{"success": true, ...}` oder bei ungültigen Eingaben/Fehlern
`{"success": false, "error": "..."}` zurück (siehe „Error Handling“ unten).

**Wichtig:** `evaluate_fit` (und intern auch die anderen Tools, sofern die jeweiligen
Plugins im Plugin Manager aktiv sind) schreibt wie die Streamlit-App in
`data/sponsor_match.db` – Aufrufe über den MCP-Server zählen also zum selben
Analyseverlauf/Score-Cache wie `main.py`.

## Setup

Abhängigkeit ist bereits über `uv add "mcp>=1.9,<2"` in `pyproject.toml`/`uv.lock`
eingetragen (Version bewusst auf die 1.x-Reihe gepinnt, da `mcp>=2.0` die Klasse
`FastMCP` in `MCPServer` umbenannt hat).

```bash
uv sync
```

`.env` mit `TAVILY_API_KEY` und `OPENROUTER_API_KEY` muss wie für `main.py` vorhanden
sein – `mcp_server.py` braucht keine eigene Env-Konfiguration, da `src/agent.py`
`load_dotenv()` bereits beim Import ausführt.

## Standalone starten & testen (ohne Claude Desktop)

```bash
uv run python mcp_server.py
```

Läuft per Default über **stdio** (das Transportformat, das Claude Desktop über
`command`/`args` erwartet – siehe unten). Für einen eigenständigen Test ohne Claude
Desktop lässt sich stattdessen ein Netzwerk-Transport wählen:

```bash
MCP_TRANSPORT=streamable-http MCP_PORT=5000 uv run python mcp_server.py
```

Die Tools lassen sich auch ganz ohne MCP-Client direkt als Python-Funktionen
aufrufen (z.B. zum schnellen Debuggen):

```bash
uv run python3 -c "
import mcp_server as s
print(s.evaluate_fit('Nike', 'FC Nordlicht'))
"
```

**Hinweis zur Aufgabenstellung:** Im Auftrag stand „Läuft auf Port 5000“ *und* die
klassische Claude-Desktop-`command`/`args`-Konfiguration – das sind zwei
unterschiedliche MCP-Transportarten. `command`/`args` (Abschnitt „Claude Desktop
Config“) funktioniert nur mit **stdio**, nicht mit einem Port. Da Claude Desktop
lokale Server ausschließlich per stdio startet, ist stdio der Default; der Port
(5000, per `MCP_PORT` änderbar) existiert nur im optionalen `streamable-http`/`sse`-Modus
für eigenständiges Testen.

## In Claude Desktop registrieren

Der fertige Eintrag liegt in [`mcp_config.json`](mcp_config.json):

```json
{
  "mcpServers": {
    "sponsor-match": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/franziskahol/sponsoring-fit-analysis-assistant",
        "run",
        "python",
        "mcp_server.py"
      ]
    }
  }
}
```

**Ich habe deine echte `claude_desktop_config.json`
(`~/Library/Application Support/Claude/claude_desktop_config.json`) bewusst nicht
automatisch verändert** – das ist eine dauerhafte, geräteweite Konfiguration. Um den
Server zu registrieren:

1. Öffne `~/Library/Application Support/Claude/claude_desktop_config.json`.
2. Füge den `"sponsor-match": {...}`-Eintrag aus `mcp_config.json` in das dortige
   `"mcpServers"`-Objekt ein (falls schon andere Server registriert sind, danebenschreiben,
   nicht ersetzen).
3. Claude Desktop komplett neu starten.

Sag Bescheid, falls ich das für dich direkt in die Datei eintragen soll.

## Testing

Nach dem Neustart von Claude Desktop sollten die vier Sponsor-Match-Tools im
Tool-Picker auftauchen. Ein Prompt wie

> Analysiere Nike für FC Nordlicht

sollte Claude dazu bringen, `research_company` und/oder `evaluate_fit` mit
`company_name="Nike"`, `club_profile="FC Nordlicht"` aufzurufen.

## Logging

Jeder Tool-Call (Name, Eingaben, Erfolg/Fehler inkl. Traceback) wird nach
`logs/mcp_server.log` geschrieben (Verzeichnis wird beim Start automatisch angelegt).

## Error Handling & Validierung

- `company_name` muss nicht-leer sein.
- `club_profile` muss exakt (case-insensitive) einem Vereinsnamen aus
  `data/clubs.json` entsprechen – bei Nichtübereinstimmung wird die Liste gültiger
  Vereine in der Fehlermeldung mit ausgegeben.
- Alle vier Tools fangen unerwartete Exceptions ab und geben
  `{"success": false, "error": "..."}` zurück, statt den Server abstürzen zu lassen.
