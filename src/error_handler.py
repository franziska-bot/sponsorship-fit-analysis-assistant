"""Retry-with-backoff and graceful degradation for the PDF pipeline's
per-URL I/O (download, parse/extract) in analysis_agent.py. A single URL's
failure must never abort the whole batch — the ThreadPoolExecutor loop in
_research_financial_pdfs already skips a URL when process_url() returns
None, so every path through this module ends in either a recovered result
or a logged, swallowed failure, never a raised exception.
"""

import time


class ErrorHandler:
    def __init__(self, max_retries: int = 3, logger=None):
        self.max_retries = max_retries
        self.logger = logger
        self.recovered_count = 0
        self.fatal_count = 0

    def retry_on_failure(self, func, *args, **kwargs):
        """Calls func(*args, **kwargs), retrying up to max_retries times
        with exponential backoff (1s, 2s, 4s, ...) whenever it raises or
        returns a falsy value — this codebase's I/O helpers (e.g.
        _download_pdf) signal failure by returning None rather than
        raising, so both cases must trigger a retry.

        Returns the first truthy result, or the last falsy/None result once
        retries are exhausted; it's up to the caller to decide that counts
        as fatal (see handle_pdf_download_error) and log it as such.
        """
        result = None
        last_error: object = None

        for attempt in range(1, self.max_retries + 2):  # 1 initial try + max_retries retries
            try:
                result = func(*args, **kwargs)
                if result:
                    if attempt > 1:
                        self.recovered_count += 1
                    return result
                last_error = "empty result"
            except Exception as exc:
                result = None
                last_error = exc

            if attempt <= self.max_retries:
                wait_seconds = 2 ** (attempt - 1)  # 1s, 2s, 4s
                if self.logger:
                    self.logger.log_agent_step(
                        f"Retry attempt {attempt}/{self.max_retries}",
                        func=getattr(func, "__name__", str(func)),
                        wait_seconds=wait_seconds,
                        reason=str(last_error),
                    )
                time.sleep(wait_seconds)

        return result

    def handle_pdf_download_error(self, url: str, error) -> None:
        """Graceful degradation once retry_on_failure has exhausted its
        attempts for a download: logs the final failure and counts it as
        fatal (unrecoverable for this URL). Caller skips this PDF and
        continues with the rest."""
        self.fatal_count += 1
        if self.logger:
            self.logger.log_error(f"download_pdf:{url}", error, error_type="download", url=url)

    def handle_extraction_error(self, url: str, error) -> None:
        """Graceful degradation for a parsing/extraction failure (not
        retried — a corrupt or unparseable PDF won't parse differently on a
        second attempt): logs it and counts it as fatal. Caller skips this
        PDF and continues with the rest."""
        self.fatal_count += 1
        if self.logger:
            self.logger.log_error(f"extract_pdf:{url}", error, error_type="extraction", url=url)
