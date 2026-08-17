"""Input security specialist: pattern-based detection of malicious or
malformed input (prompt injection, jailbreaks, injection attacks, control
character abuse, ...) before it reaches any LLM prompt, DB query, or search.
Also tracks IP-based rate limiting (Phase 3f) as a defense-in-depth layer.

validate_input() stays pure/side-effect-free so it can be imported from
main.py, mcp_server.py, and orchestrator.py without any circular-import
risk. rate_limit_check() is the one function in this module with side
effects (in-memory request counters) — see its docstring.

Scope note: this module only detects attack *patterns* and counts requests
per IP. UX-only checks ("this looks like a question, not a company name")
and the pre-existing *session*-scoped rate limiter (st.session_state-bound,
can't live here since this module has no Streamlit dependency) stay in
main.py.
"""

import re
import threading
import time

MIN_LENGTH = 2
MAX_LENGTH = 100

# Baseline allow-list: letters/digits/space/-.äöüßÄÖÜ, same charset the app
# has always restricted company names to.
_ALLOWED_CHARSET_RE = re.compile(r"^[a-zA-Z0-9\s\-\.äöüßÄÖÜ]*$")

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPECIAL_CHAR_FLOOD_RE = re.compile(r"[^\w\s]{10,}")

_PATTERN_CATEGORIES: list[tuple[str, re.Pattern]] = [
    (
        "prompt_injection",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
            r"|disregard\s+(all\s+)?(previous|prior|above)"
            r"|new\s+instructions\s*:"
            r"|forget\s+(everything|all\s+prior)"
            r"|system\s*prompt",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_roleplay",
        re.compile(
            r"\bact\s+as\b|\bpretend\s+(you\s+are|to\s+be)\b|\bDAN\s+mode\b"
            r"|\bdeveloper\s+mode\b|\bno\s+restrictions\b|\bjailbreak\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sql_injection",
        re.compile(
            r"--|;.*(drop|delete|update|insert)\b|'\s*or\s*'?1'?\s*=\s*'?1"
            r"|\bunion\s+select\b|\bdrop\s+table\b|\bxp_cmdshell\b",
            re.IGNORECASE,
        ),
    ),
    (
        "script_xss",
        re.compile(r"<script|javascript:|on(error|load)\s*=", re.IGNORECASE),
    ),
    (
        "command_injection",
        re.compile(r"`[^`]*`|\$\([^)]*\)|&&|\|\||;\s*rm\s+-rf"),
    ),
    (
        "path_traversal",
        re.compile(r"\.\./|\.\.\\|/etc/passwd|[a-zA-Z]:\\\\?windows", re.IGNORECASE),
    ),
    (
        "template_injection",
        re.compile(r"\{\{.*\}\}|\$\{.*\}|<%.*%>"),
    ),
    (
        "control_chars",
        _CONTROL_CHARS_RE,
    ),
    (
        "special_char_flood",
        _SPECIAL_CHAR_FLOOD_RE,
    ),
]


def validate_input(text: str, max_length: int = MAX_LENGTH) -> tuple[bool, list[str]]:
    """Prüft `text` gegen alle bekannten Angriffsmuster.

    Gibt (is_safe, violated_category_names) zurück. is_safe ist nur dann
    True, wenn kein einziges Muster (inkl. Charset-Whitelist und Längen-
    Grenzen) verletzt wurde.
    """
    stripped = text.strip()
    violations: list[str] = []

    if not (MIN_LENGTH <= len(stripped) <= max_length):
        violations.append("excessive_length")

    if not _ALLOWED_CHARSET_RE.fullmatch(text):
        violations.append("allowed_charset")

    for category, pattern in _PATTERN_CATEGORIES:
        if pattern.search(text):
            violations.append(category)

    return not violations, violations


# --- IP-basiertes Rate Limiting (Phase 3f) ---
#
# Zusätzliche Verteidigungsebene NEBEN main.py's bestehendem session-basierten
# check_rate_limit() (10/5min, 100/24h, in st.session_state) – ersetzt ihn
# nicht. Der Session-Limiter überlebt keinen neuen Tab/Inkognito-Fenster, ist
# aber pro Nutzer präzise; dieser IP-Limiter überlebt neue Tabs, ist aber pro
# NAT/Firmennetz gemeinsam genutzt und laut Streamlits eigener
# ip_address-Doku spoofbar – zusammen decken beide mehr Abuse-Muster ab als
# jeder für sich.
#
# In-Memory (Prozess-Lifetime, kein Redis) wie in der Task-Vorgabe als Option
# genannt: reicht für einen Single-Instance-Streamlit-Prozess; bei mehreren
# Worker-Prozessen/Instanzen hinter einem Load Balancer müsste dieser Zustand
# extern geteilt werden (z.B. Redis) – siehe DEPLOYMENT.md.

_IP_RATE_LIMIT_SHORT = (10, 60)  # max. 10 Requests pro 60s (pro IP)
_IP_RATE_LIMIT_LONG = (100, 60 * 60)  # max. 100 Requests pro 3600s (pro IP)

_ip_request_log: dict[str, list[float]] = {}
_ip_rate_limit_lock = threading.Lock()


def rate_limit_check(client_ip: str) -> tuple[bool, str]:
    """Prüft UND zählt (atomar, ein Aufruf statt check+record) einen Request
    für `client_ip` gegen die IP-Limits. Gibt (is_allowed, error_message)
    zurück – error_message ist "" wenn erlaubt.

    Zählt den Request nur bei Erfolg (is_allowed=True), damit bereits
    geblockte Anfragen das Limit nicht künstlich weiter ausschöpfen."""
    now = time.time()
    with _ip_rate_limit_lock:
        timestamps = _ip_request_log.setdefault(client_ip, [])
        timestamps[:] = [t for t in timestamps if now - t < _IP_RATE_LIMIT_LONG[1]]

        short_limit, short_window = _IP_RATE_LIMIT_SHORT
        recent_short = [t for t in timestamps if now - t < short_window]
        if len(recent_short) >= short_limit:
            wait_seconds = short_window - (now - min(recent_short))
            return False, f"max {short_limit} requests/minute exceeded, retry in {wait_seconds:.0f}s"

        long_limit, long_window = _IP_RATE_LIMIT_LONG
        if len(timestamps) >= long_limit:
            wait_seconds = long_window - (now - min(timestamps))
            return False, f"max {long_limit} requests/hour exceeded, retry in {wait_seconds / 60:.0f}min"

        timestamps.append(now)
        return True, ""
