"""Shared agent tracing: start/step/result/error logging across the pipeline.

Same stdlib-`logging` approach mcp_server.py already uses (file handler +
`logging.getLogger`), packaged as a small reusable class instead of one more
module-level `basicConfig` call — `basicConfig` is a one-time process-global
setup, so multiple modules calling it independently would fight each other.
"""

import logging
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_trace.log"

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")


class AgentLogger:
    """Tracing logger for one pipeline stage (e.g. "search_agent",
    "orchestrator"). File handler captures everything at DEBUG level; console
    handler stays at INFO so interactive `streamlit run` output isn't flooded
    with per-step detail — the file has the full trace either way.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"sponsor_match.{agent_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if not self.logger.handlers:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(_FORMATTER)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(_FORMATTER)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self._start_times: dict[str, float] = {}

    @staticmethod
    def _format_context(context: dict) -> str:
        return f" | {context}" if context else ""

    def log_agent_start(self, task: str, **context) -> None:
        """Marks the start of `task`, recording a timestamp so a matching
        log_agent_result() call can report elapsed time."""
        self._start_times[task] = time.monotonic()
        self.logger.info(f"START {task}{self._format_context(context)}")

    def log_agent_step(self, step: str, **context) -> None:
        """Logs an intermediate step (DEBUG-only, file trace)."""
        self.logger.debug(f"STEP {step}{self._format_context(context)}")

    def log_agent_result(self, task: str, **context) -> None:
        """Marks the completion of `task`, including elapsed time since the
        matching log_agent_start() call (0.0 if no matching start was recorded)."""
        elapsed = time.monotonic() - self._start_times.pop(task, time.monotonic())
        self.logger.info(f"DONE {task} ({elapsed:.2f}s){self._format_context(context)}")

    def log_error(
        self, task: str, error: Exception | str, *, error_type: str = "", url: str | None = None, **context
    ) -> None:
        """Logs a failure. Never raises — callers keep their existing
        graceful-fallback behavior, this only adds observability.

        error_type/url are pulled out as named (rather than left in
        **context) so error-reporting call sites read consistently, but
        both stay optional — existing call sites that only pass
        log_error(task, error, **context) are unaffected."""
        if url:
            context = {"url": url, **context}
        if error_type:
            context = {"error_type": error_type, **context}
        self.logger.error(f"ERROR {task}: {error}{self._format_context(context)}")

    def log_attack_attempt(self, attack_type: str, input_text: str, ip_address: str) -> None:
        """Security audit trail entry for a detected attack attempt
        (src/security_validator.py). Always logged at ERROR level — never
        DEBUG-only — so it's never silently lost to the file-only debug
        stream; still lands in the same logs/agent_trace.log as everything
        else, filterable by the 'ATTACK' prefix. input_text is truncated to
        200 chars so a single crafted payload can't bloat the log."""
        self.logger.error(f"ATTACK type={attack_type} ip={ip_address} input={input_text[:200]!r}")
