"""Standalone health-check HTTP server for monitoring/DevOps (Phase 3f,
optional). Streamlit has no Flask-style @app.route, so this runs as a small
separate process alongside `streamlit run main.py` rather than inside it —
stdlib http.server only, no new dependency for one optional endpoint.

Start:  python -m src.health_server   (default port 8600, override with
        HEALTH_CHECK_PORT; binds to 127.0.0.1 by default, override with
        HEALTH_CHECK_HOST for e.g. a container health-check probe)

GET /health -> {"status", "cache_size": {"files", "bytes"}, "latest_analysis", "log_size"}
"""

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.logger import LOG_FILE
from src.orchestrator import DB_PATH
from src.pdf_cache import PDFCache

DEFAULT_PORT = int(os.environ.get("HEALTH_CHECK_PORT", "8600"))
DEFAULT_HOST = os.environ.get("HEALTH_CHECK_HOST", "127.0.0.1")


def _cache_size() -> dict:
    cache = PDFCache()
    files = list(cache.pdf_dir.glob("*.pdf")) + list(cache.extraction_dir.glob("*.json"))
    return {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}


def _latest_analysis() -> dict | None:
    if not Path(DB_PATH).exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT company_name, club_name, timestamp, fit_score FROM analyses ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"company_name": row[0], "club_name": row[1], "timestamp": row[2], "fit_score": row[3]}


def _log_size() -> int:
    return LOG_FILE.stat().st_size if LOG_FILE.exists() else 0


def build_health_payload() -> dict:
    return {
        "status": "ok",
        "cache_size": _cache_size(),
        "latest_analysis": _latest_analysis(),
        "log_size": _log_size(),
    }


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(build_health_payload()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        pass  # kein stdout-Spam pro Monitoring-Poll (z.B. alle 10s vom Load Balancer)


def run(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> None:
    server = ThreadingHTTPServer((host, port), _HealthHandler)
    print(f"Health check server listening on http://{host}:{port}/health")
    server.serve_forever()


if __name__ == "__main__":
    run()
