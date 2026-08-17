"""Two-tier disk cache for the PDF financial-extraction pipeline
(analysis_agent.py): raw PDF bytes (30-day TTL) and the regex-extracted
metrics derived from them (60-day TTL), both keyed by MD5(url) so a repeat
run against the same report skips the download+parse+regex work entirely.
"""

import hashlib
import json
import time
from pathlib import Path

_PDF_TTL_SECONDS = 30 * 24 * 60 * 60
_EXTRACTION_TTL_SECONDS = 60 * 24 * 60 * 60


class PDFCache:
    """File-based cache, one subdirectory per artifact type so a PDF and its
    extraction can expire independently (a report's numbers don't change
    once extracted, even after the raw PDF bytes have aged out)."""

    def __init__(self, cache_dir: str = "data/pdf_cache"):
        self.cache_dir = Path(cache_dir)
        self.pdf_dir = self.cache_dir / "pdf"
        self.extraction_dir = self.cache_dir / "extraction"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.extraction_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _pdf_path(self, url: str) -> Path:
        return self.pdf_dir / f"{self._hash_url(url)}.pdf"

    def _extraction_path(self, url: str) -> Path:
        return self.extraction_dir / f"{self._hash_url(url)}.json"

    @staticmethod
    def _is_fresh(path: Path, ttl_seconds: int) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) <= ttl_seconds

    def get_cached_pdf(self, url: str) -> bytes | None:
        path = self._pdf_path(url)
        if not self._is_fresh(path, _PDF_TTL_SECONDS):
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def cache_pdf(self, url: str, pdf_content: bytes) -> None:
        self._pdf_path(url).write_bytes(pdf_content)

    def get_cached_extraction(self, url: str) -> dict | None:
        path = self._extraction_path(url)
        if not self._is_fresh(path, _EXTRACTION_TTL_SECONDS):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def cache_extraction(self, url: str, metrics: dict) -> None:
        with open(self._extraction_path(url), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False)
