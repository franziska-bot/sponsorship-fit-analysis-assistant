"""Financial data extraction specialist: the company's own sponsorship
portfolio, its estimated size/revenue tier, a budget estimate derived from
historical sponsorship deals, and (Phase 2) real financial metrics extracted
from PDFs (annual reports / 10-K filings) discovered via web search.
"""

import datetime
import io
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pdfplumber
import requests
from langsmith import traceable

from src.error_handler import ErrorHandler
from src.logger import AgentLogger
from src.pdf_cache import PDFCache
from src.tools import SponsorMatchState, _get_llm, _track_tokens, get_search_tool, is_plugin_enabled
from src.search_agent import _score_source_credibility

_logger = AgentLogger("analysis_agent")
_pdf_cache = PDFCache()
_error_handler = ErrorHandler(max_retries=3, logger=_logger)

# --- Optionales Plugin: competitor_analysis (Tavily) ---
# Analysiert generisch das SPONSORING-PORTFOLIO DER EINGEGEBENEN FIRMA SELBST
# (nicht ihrer Konkurrenten) – funktioniert für jede beliebige Company, da
# Suchbegriff, Prompt und Score-Formel ausschließlich von company_name/club
# abhängen, nichts ist auf eine bestimmte Firma fest verdrahtet.

_PORTFOLIO_FIELD_PREFIXES = (
    "CATEGORIES",
    "ACTIVE_COUNT",
    "AUDIENCE",
    "SAME_SPORT_COUNT",
    "AUDIENCE_FIT",
    "MATCH_PERCENT",
)


def _to_nonneg_int(value: str, default: int = 0) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else default


def _analyze_company_sponsorship_portfolio(company_name: str, club: dict, language: str, llm) -> dict:
    """competitor_analysis-Plugin: sucht per Tavily nach dem bestehenden Sponsoring-
    Portfolio der EINGEGEBENEN Firma (nicht ihrer Konkurrenten) und lässt ein LLM
    daraus Kategorien, aktive Sponsoring-Anzahl, Zielgruppe, Sättigung in der
    Vereins-Sportart sowie den Zielgruppen-Fit zum konkreten Verein extrahieren.

    ACHTUNG: alle Zahlen/Einschätzungen sind KI-Schätzungen auf Basis der
    Websuche, keine verifizierten Marktdaten – main.py zeigt das mit Disclaimer an.
    """
    sport = club["sport"]
    try:
        results = get_search_tool().invoke(f"{company_name} sponsorships list")
    except Exception:
        results = ""

    if not results:
        return {"found": False, "token_usage": None}

    snippet = str(results)[:2000]
    output_language = {"de": "German", "en": "English", "fr": "French"}[language]
    prompt = (
        f"Based on this web search about {company_name}'s sponsorships:\n{snippet}\n\n"
        f"Analyze {company_name}'s OWN sponsorship portfolio (not its competitors'). "
        f"The club being evaluated as a sponsorship target has this profile:\n"
        f"- Sport: {sport}\n"
        f"- Fanbase: {club['fanbase']}\n"
        f"- Values: {club['values']}\n\n"
        f"Respond in exactly this format, nothing before or after:\n"
        f"CATEGORIES: <up to 3 main sports/categories {company_name} sponsors, comma-separated>\n"
        f"ACTIVE_COUNT: <your best estimate of the number of currently active sponsorship deals overall, "
        f"as a single integer>\n"
        f"AUDIENCE: <typical target audience of {company_name}'s sponsorship activities, a short phrase>\n"
        f"SAME_SPORT_COUNT: <your best estimate of how many of those active sponsorships are specifically "
        f"in {sport}, as a single integer, 0 if none>\n"
        f"AUDIENCE_FIT: <yes, no, or partial - does {company_name}'s typical sponsorship audience match "
        f"this club's fanbase>\n"
        f"MATCH_PERCENT: <integer 0-100, how well {company_name}'s audience overlaps with this club's fanbase>\n"
        f"Write all text values (except AUDIENCE_FIT and the numbers) in {output_language}. Use only real "
        f"information found in the search results; if uncertain, give a reasonable estimate rather than "
        f"leaving fields blank."
    )
    response = llm.invoke(prompt)
    token_entry = _track_tokens("competitor_analysis", response)

    fields = {}
    for line in response.content.split("\n"):
        line = line.strip()
        for key in _PORTFOLIO_FIELD_PREFIXES:
            prefix = f"{key}:"
            if line.upper().startswith(prefix):
                fields[key] = line[len(prefix):].strip()
                break

    # Normalisierung, da das LLM AUDIENCE_FIT trotz Anweisung gelegentlich in der
    # Zielsprache statt auf Englisch beantwortet (z.B. "ja" statt "yes").
    raw_fit = fields.get("AUDIENCE_FIT", "").strip().lower()
    if raw_fit.startswith(("yes", "ja", "oui")):
        audience_fit = "yes"
    elif raw_fit.startswith(("no", "non", "nein")):
        audience_fit = "no"
    else:
        audience_fit = "partial"

    return {
        "found": True,
        "categories": [c.strip() for c in fields.get("CATEGORIES", "").split(",") if c.strip()][:3],
        "active_count": _to_nonneg_int(fields.get("ACTIVE_COUNT", "")),
        "audience": fields.get("AUDIENCE", "—"),
        "same_sport_count": _to_nonneg_int(fields.get("SAME_SPORT_COUNT", "")),
        "audience_fit": audience_fit,
        "match_percent": max(0, min(100, _to_nonneg_int(fields.get("MATCH_PERCENT", ""), default=50))),
        "token_usage": token_entry,
    }


def _compute_saturation_level(same_sport_count: int) -> str:
    """Company-Perspektive: wie gesättigt ist die Firma bereits in DIESER Sportart
    (nicht wie viele Sponsoren der Markt insgesamt verträgt) – rein aus der Anzahl
    ihrer eigenen, bereits laufenden Sponsorings in dieser Sportart abgeleitet."""
    if same_sport_count >= 7:
        return "extreme"
    if same_sport_count >= 4:
        return "high"
    if same_sport_count >= 2:
        return "medium"
    return "low"


def _compute_portfolio_score_impact(match_percent: int, saturation_level: str) -> float:
    """Dynamische Score-Anpassung: hängt vom individuellen Zielgruppen-Match UND von
    der individuellen Sättigung DIESER Firma in DIESER Sportart ab – kein fixer
    Wert, unterschiedliche Firmen/Vereine ergeben unterschiedliche Anpassungen.
    """
    if match_percent >= 70:
        base = 0.10
    elif match_percent >= 40:
        base = 0.0
    else:
        base = -0.10

    saturation_penalty = {"low": 0.0, "medium": -0.03, "high": -0.08, "extreme": -0.15}[saturation_level]
    return round(max(-0.25, min(0.15, base + saturation_penalty)), 2)


# --- Optionales Plugin: size_matching (Club-Größe vs. Company-Größe, sport-agnostisch) ---

_SIZE_MATCH_ADJUSTMENTS = {
    ("Small", "Small"): 0.15,
    ("Small", "Medium"): 0.05,
    ("Small", "Large"): -0.20,
    ("Medium", "Small"): 0.05,
    ("Medium", "Medium"): 0.10,
    ("Medium", "Large"): 0.0,
    ("Large", "Small"): -0.10,
    ("Large", "Medium"): 0.05,
    ("Large", "Large"): 0.05,
    # Elite+Small/Medium waren in der Spezifikation nicht explizit vorgegeben;
    # analog zu Large+Small (-0.10) bzw. verschärft, da Elite-Clubs eine noch
    # größere Reichweiten-/Budget-Lücke zu kleinen/mittleren Firmen haben.
    ("Elite", "Small"): -0.25,
    ("Elite", "Medium"): -0.10,
    ("Elite", "Large"): 0.10,
}


def _estimate_company_size(company_name: str, llm) -> dict:
    """size_matching-Plugin: recherchiert generisch (für JEDE Firma, unabhängig von der
    Branche) via Web-Suche nach Umsatz-/Reichweite-Signalen und lässt ein LLM daraus eine
    Company-Size-Kategorie (Small/Medium/Large) ableiten. Rein heuristische KI-Schätzung,
    keine verifizierten Finanzdaten – main.py zeigt das mit Disclaimer an."""
    try:
        results = get_search_tool().invoke(f"{company_name} annual revenue company size global reach")
    except Exception:
        results = ""

    if not results:
        return {"found": False, "size": "Medium", "token_usage": None}

    snippet = str(results)[:1500]
    prompt = (
        f"Based on this web search about {company_name}:\n{snippet}\n\n"
        f"Classify {company_name}'s company size into exactly one of these three "
        f"categories, based on estimated annual revenue and market reach:\n"
        f"SMALL: local/regional brand, revenue under 10M EUR\n"
        f"MEDIUM: national brand, revenue 10-500M EUR\n"
        f"LARGE: global brand, revenue over 500M EUR\n\n"
        f"Respond in exactly this format, nothing before or after:\n"
        f"SIZE: <Small, Medium, or Large>"
    )
    response = llm.invoke(prompt)
    token_entry = _track_tokens("size_matching", response)

    size = "Medium"
    for line in response.content.split("\n"):
        line = line.strip()
        if line.upper().startswith("SIZE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw.startswith("small"):
                size = "Small"
            elif raw.startswith("large"):
                size = "Large"
            else:
                size = "Medium"
            break

    return {"found": True, "size": size, "token_usage": token_entry}


def _compute_size_match_adjustment(club_size: str, company_size: str) -> float:
    """Score-Anpassung basierend auf dem Größen-Match zwischen Club und Company
    (sport-agnostisch, siehe _SIZE_MATCH_ADJUSTMENTS)."""
    return _SIZE_MATCH_ADJUSTMENTS.get((club_size, company_size), 0.0)


def _compute_size_match_percent(score_adjustment: float) -> int:
    """Bildet die Score-Anpassung (-0.25 bis +0.15) linear auf einen 0-100%-Match-Wert
    für die Progress-Bar-Anzeige ab."""
    percent = (score_adjustment + 0.25) / 0.40 * 100
    return int(round(max(0.0, min(100.0, percent))))


_SIZE_EXPLANATION_TIERS = {
    "de": {
        "good": "{club_size} Club und {company_size} Company passen von der Größe gut zusammen.",
        "moderate": (
            "{club_size} Club und {company_size} Company ergeben einen moderaten Fit – "
            "strategische Positionierung ist sinnvoll."
        ),
        "poor": (
            "{club_size} Club und {company_size} Company passen von der Größe kaum zusammen – "
            "andere Partner könnten besser passen."
        ),
    },
    "en": {
        "good": "{club_size} club and {company_size} company are a strong size match.",
        "moderate": (
            "{club_size} club and {company_size} company make a moderate fit – "
            "strategic positioning is recommended."
        ),
        "poor": (
            "{club_size} club and {company_size} company sizes are mismatched – "
            "consider alternative partners."
        ),
    },
    "fr": {
        "good": "Le club {club_size} et l'entreprise {company_size} sont bien assortis en termes de taille.",
        "moderate": (
            "Le club {club_size} et l'entreprise {company_size} forment un fit modéré – "
            "un positionnement stratégique est recommandé."
        ),
        "poor": (
            "Les tailles du club {club_size} et de l'entreprise {company_size} ne correspondent guère – "
            "envisagez d'autres partenaires."
        ),
    },
}


def _build_size_explanation(club_size: str, company_size: str, match_percent: int, language: str) -> str:
    """Baut den erklärenden Satz zur Size Compatibility, gestuft nach Match-Prozent
    (>70 gut, 40-70 moderat, <40 schwach) – dieselben Schwellen wie die Ampelfarben."""
    if match_percent > 70:
        tier = "good"
    elif match_percent >= 40:
        tier = "moderate"
    else:
        tier = "poor"
    return _SIZE_EXPLANATION_TIERS[language][tier].format(club_size=club_size, company_size=company_size)


def _build_budget_estimate(sponsorship_matches: list[dict], language: str) -> str:
    """budget_estimator-Plugin: leitet eine grobe Budget-Schätzung aus den
    investment_range-Werten passender externer Sponsorship-DB-Fälle ab.
    Rein datenbasiert, kein LLM-Call.
    """
    if not sponsorship_matches:
        return {
            "de": "Keine Budget-Schätzung möglich (keine passenden historischen Daten).",
            "en": "No budget estimate possible (no matching historical data).",
            "fr": "Aucune estimation de budget possible (pas de données historiques correspondantes).",
        }[language]

    ranges = sorted({m["investment_range"] for m in sponsorship_matches})
    label = {
        "de": "Geschätztes Sponsoring-Budget (basierend auf ähnlichen historischen Fällen)",
        "en": "Estimated sponsorship budget (based on similar historical cases)",
        "fr": "Budget de sponsoring estimé (basé sur des cas historiques similaires)",
    }[language]
    return f"{label}: {', '.join(ranges)}"


# --- Phase 2: PDF Financial Extraction (budget_estimator-Plugin) ---
#
# Findet Geschäftsberichte/10-K-Filings per Web-Suche, lädt sie herunter und
# extrahiert Finanzkennzahlen per Regex. Bewusst best-effort: die meisten
# Firmen haben keinen per Suche direkt auffindbaren PDF-Link, viele PDFs sind
# gescannte Bilder ohne extrahierbaren Text – ein leeres {"found": False} ist
# der erwartete Normalfall, kein Fehler, analog zu
# _analyze_company_sponsorship_portfolio/_estimate_company_size oben, die bei
# leerer Websuche ebenfalls sauber auf einen Fallback zurückfallen.

_PDF_SEARCH_QUERY_TEMPLATES = [
    "{company} investor relations annual report filetype:pdf",
    '"{company} annual report" PDF',
    "{company} 10-K 10-Q SEC filing",
    "{company} Geschäftsbericht PDF",
]

_PDF_DOWNLOAD_TIMEOUT = 10.0
_MAX_PDF_BYTES = 15_000_000  # 15 MB Deckel, gegen Speicher-/Zeitverbrauch durch riesige PDFs
_MAX_PDF_PAGES = 30  # reale Geschäftsberichte haben oft 100+ Seiten; Kennzahlen stehen meist früh
_PDF_MAGIC_BYTES = b"%PDF"

_PDF_HIGH_CREDIBILITY_DOMAINS = {"sec.gov", "annualreports.com"}

_FINANCIAL_METRIC_KEYWORDS = {
    "revenue": [r"total revenue", r"net sales", r"revenue", r"umsatz", r"nettoumsatz", r"konzernumsatz"],
    "ebitda": [r"ebitda"],
    "profit": [r"net profit", r"net income", r"profit for the year", r"jahresüberschuss", r"nettogewinn"],
    "cash": [r"cash and cash equivalents", r"total cash", r"liquide mittel", r"zahlungsmittel"],
    "marketing_spend": [r"marketing expenses", r"advertising expenses", r"marketing spend", r"marketingaufwendungen"],
}

_SCALE_MULTIPLIERS = {
    "k": 1_000, "tsd": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "mio": 1_000_000, "million": 1_000_000,
    "bn": 1_000_000_000, "mrd": 1_000_000_000, "billion": 1_000_000_000,
}
_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}

_AMOUNT_RE = re.compile(
    r"(?P<currency>[€$£]|EUR|USD|GBP)?\s*"
    r"(?P<number>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    # Längere/spezifischere Skalen-Alternativen zuerst (z.B. "million" vor "m",
    # "mio"/"mn" vor "m"), sonst würde die Alternation vorzeitig nur "m" matchen
    # und den Rest ("illion") unkonsumiert lassen. \b nach der Gruppe verhindert,
    # dass "m"/"k" als Präfix eines unabhängigen Folgeworts (z.B. "mentioned")
    # fälschlich als Skala erkannt werden.
    r"\s*(?P<scale>thousand|million|billion|mio\.?|mrd\.?|tsd\.?|bn|mn|m|k)?\b",
    re.IGNORECASE,
)


def _find_financial_pdfs(company_name: str, max_urls: int = 5) -> list[str]:
    """Sucht per Tavily nach PDF-Links zu Geschäftsberichten/Filings (4 Query-
    Varianten aus der Spezifikation, parallel ausgeführt wie in search_agent.py's
    _run_research_queries). Sprachunabhängig von der UI-Sprache – ein deutscher
    Geschäftsbericht kann auch dann relevant sein, wenn die App auf Englisch steht."""
    search_tool = get_search_tool()
    queries = [template.format(company=company_name) for template in _PDF_SEARCH_QUERY_TEMPLATES]

    def run_query(query: str) -> dict:
        try:
            return search_tool.invoke(query)
        except Exception:
            return {"results": []}

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        responses = list(executor.map(run_query, queries))

    urls: list[str] = []
    for response in responses:
        for result in response.get("results", []):
            url = result.get("url", "")
            if url.lower().endswith(".pdf") and url not in urls:
                urls.append(url)
    return urls[:max_urls]


def _download_pdf(url: str) -> bytes | None:
    """Lädt ein PDF herunter mit Sicherheits-Guards: nur http/https, Zeitlimit,
    Größenlimit (per Content-Length UND per laufender Summe beim Streamen
    durchgesetzt, falls der Header fehlt oder lügt), und ein Magic-Bytes-Check
    statt blindem Vertrauen auf den Content-Type-Header (der bei manchen
    Servern falsch gesetzt ist). Jeder Fehlerfall gibt None zurück statt zu
    werfen – ein einzelnes kaputtes PDF darf die restliche Pipeline nicht stoppen."""
    if not url.lower().startswith(("http://", "https://")):
        return None

    try:
        response = requests.get(
            url,
            timeout=_PDF_DOWNLOAD_TIMEOUT,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SponsorMatchBot/1.0)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        _logger.log_error(f"download_pdf:{url}", exc)
        return None

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > _MAX_PDF_BYTES:
        _logger.log_error(f"download_pdf:{url}", f"Content-Length {content_length} exceeds {_MAX_PDF_BYTES} byte cap")
        return None

    chunks = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > _MAX_PDF_BYTES:
                _logger.log_error(f"download_pdf:{url}", f"exceeded {_MAX_PDF_BYTES} byte cap while streaming")
                return None
            chunks.append(chunk)
    except requests.RequestException as exc:
        _logger.log_error(f"download_pdf:{url}", exc)
        return None

    content = b"".join(chunks)
    if not content.startswith(_PDF_MAGIC_BYTES):
        _logger.log_error(f"download_pdf:{url}", "downloaded content is not a PDF (magic bytes mismatch)")
        return None
    return content


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extrahiert Text + Tabellen aus den ersten _MAX_PDF_PAGES Seiten (Tabellen
    zeilenweise als '|'-getrennter Text angehängt, da Finanzkennzahlen oft in
    Tabellen statt Fließtext stehen). Wirft bei jedem Parse-Fehler (z.B. defektes
    oder passwortgeschütztes PDF), statt ihn zu verschlucken – der Aufruf läuft in
    process_url() innerhalb eines try/except, das den Fehler an
    ErrorHandler.handle_extraction_error() weiterreicht (graceful degradation:
    diese eine PDF wird übersprungen, die übrigen laufen weiter)."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[:_MAX_PDF_PAGES]:
            text_parts.append(page.extract_text() or "")
            for table in page.extract_tables():
                for row in table:
                    text_parts.append(" | ".join(cell or "" for cell in row))
    return "\n".join(text_parts)


def _parse_amount_number(raw: str) -> float | None:
    """Normalisiert eine Zahl in US- (1,234.56) oder DE-Format (1.234,56) zu
    einem float. Heuristik: bei beiden Trennzeichen vorhanden gewinnt das
    letzte als Dezimaltrenner; bei nur Komma vorhanden zählt es nur dann als
    Dezimaltrenner, wenn genau 1-2 Ziffern folgen (sonst Tausendertrennzeichen).
    Zahlenformat-Erkennung aus Freitext ist inhärent mehrdeutig – diese
    Heuristik deckt die gängigsten Fälle ab, ist aber keine Garantie."""
    raw = raw.strip()
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        integer_part, _, frac_part = raw.rpartition(",")
        if len(frac_part) in (1, 2):
            raw = integer_part.replace(".", "") + "." + frac_part
        else:
            raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_financial_metrics(text: str) -> dict:
    """Regex-basierte Extraktion von Revenue/EBITDA/Profit/Cash/Marketing-Spend:
    pro Metrik wird die erste Zeile gesucht, die einen bekannten DE/EN-
    Schlüsselbegriff enthält, und darin nach einem Betrag (Währung+Zahl+Skala)
    gesucht. Bewusst best-effort – keine Metrik wird erzwungen, wenn nichts
    Eindeutiges gefunden wird, statt ein Rateergebnis zu liefern. KEINE echte
    Währungsumrechnung (USD->EUR bräuchte einen Live-FX-Kurs, außerhalb des
    Scopes) – der Wert bleibt in seiner Original-Währung, die mit ausgegeben wird."""
    metrics: dict[str, dict] = {}
    lines = text.split("\n")
    for metric_key, keyword_patterns in _FINANCIAL_METRIC_KEYWORDS.items():
        for line in lines:
            if not any(re.search(kw, line, re.IGNORECASE) for kw in keyword_patterns):
                continue
            amount_match = _AMOUNT_RE.search(line)
            if not amount_match:
                continue
            number = _parse_amount_number(amount_match.group("number"))
            if number is None:
                continue
            scale = (amount_match.group("scale") or "").lower().rstrip(".")
            multiplier = _SCALE_MULTIPLIERS.get(scale, 1)
            currency_raw = amount_match.group("currency") or ""
            currency = _CURRENCY_SYMBOLS.get(currency_raw, currency_raw.upper()) or "EUR"
            metrics[metric_key] = {
                "value": number * multiplier,
                "currency": currency,
                "raw_line": line.strip()[:200],
            }
            break  # erste eindeutige Erwähnung reicht, nicht jede Zeile weiter durchsuchen
    return metrics


def _score_pdf_source_credibility(url: str) -> float:
    """Credibility-Scoring speziell für PDF-Quellen: ergänzt die allgemeine
    Domain-Heuristik aus search_agent.py (_score_source_credibility) um
    Finanzfilings-spezifische Top-Quellen (SEC EDGAR, annualreports.com)."""
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain in _PDF_HIGH_CREDIBILITY_DOMAINS:
        return 0.95
    return _score_source_credibility(url)


def _normalize_extracted_data(per_pdf_results: list[dict]) -> dict:
    """Merged die pro-PDF extrahierten Metriken: bei mehreren Quellen für
    dieselbe Metrik wird der credibility-gewichtete Mittelwert gebildet (offizielle
    Quellen zählen stärker als Drittanbieter-Zusammenfassungen), plus die Anzahl
    übereinstimmender Quellen pro Metrik (source_count) für die Konfidenz-Berechnung.
    Kein echtes Recency-Ranking ('most recent' aus der Spezifikation) – ein
    PDF-Veröffentlichungsdatum lässt sich nicht zuverlässig aus Freitext
    extrahieren, daher Credibility-Gewichtung statt Datum als Signal."""
    by_metric: dict[str, list[dict]] = {}
    for pdf_result in per_pdf_results:
        for metric_key, metric_data in pdf_result["metrics"].items():
            by_metric.setdefault(metric_key, []).append({**metric_data, "credibility": pdf_result["credibility"]})

    merged: dict[str, dict] = {}
    for metric_key, occurrences in by_metric.items():
        total_weight = sum(o["credibility"] for o in occurrences) or 1.0
        weighted_value = sum(o["value"] * o["credibility"] for o in occurrences) / total_weight
        best = max(occurrences, key=lambda o: o["credibility"])
        merged[metric_key] = {
            "value": weighted_value,
            "currency": best["currency"],
            "source_count": len(occurrences),
            "avg_credibility": total_weight / len(occurrences),
        }
    return merged


def _calculate_confidence(merged_metrics: dict) -> float:
    """Heuristische Gesamt-Konfidenz (0.0-1.0): gewichtet aus Quellen-Übereinstimmung
    (mehrere PDFs bestätigen dieselbe Metrik), durchschnittlicher Quellen-Credibility,
    und Abdeckung der 5 Zielmetriken. Kein Alters-Signal (siehe
    _normalize_extracted_data – PDF-Datum wird nicht zuverlässig extrahiert)."""
    if not merged_metrics:
        return 0.0
    avg_source_count = sum(m["source_count"] for m in merged_metrics.values()) / len(merged_metrics)
    agreement_factor = min(avg_source_count / 2, 1.0)  # sättigt ab durchschnittlich 2+ übereinstimmenden Quellen
    avg_credibility = sum(m["avg_credibility"] for m in merged_metrics.values()) / len(merged_metrics)
    coverage = len(merged_metrics) / len(_FINANCIAL_METRIC_KEYWORDS)
    return round(max(0.0, min(1.0, 0.4 * agreement_factor + 0.35 * avg_credibility + 0.25 * coverage)), 2)


_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _extract_latest_year(text: str) -> str | None:
    """Heuristisches 'data_freshness'-Signal (Phase 3e): die aktuellste
    4-stellige Jahreszahl irgendwo im PDF-Text, gedeckelt auf das laufende
    Jahr (Berichte referenzieren gelegentlich ein zukünftiges Zieljahr, das
    keine sinnvolle Freshness wäre). Best-effort wie der Rest der
    PDF-Pipeline – ein Berichts-Stichtag lässt sich nicht zuverlässig aus
    Freitext extrahieren (siehe _normalize_extracted_data's Kommentar zur
    selben Einschränkung bei einem echten Recency-Ranking)."""
    current_year = datetime.datetime.now().year
    years = [int(y) for y in _YEAR_RE.findall(text) if int(y) <= current_year]
    return str(max(years)) if years else None


def _classify_financial_quality(confidence: float) -> str:
    """Bildet extraction_confidence (bereits eine 0.0-1.0-Mischung aus
    Quellen-Übereinstimmung, Credibility und Metrik-Abdeckung, siehe
    _calculate_confidence) auf 'high'/'medium'/'low' ab, für die
    Data-Quality-Anzeige (Phase 3e). Tier-Grenzen (80%/60%) siehe
    QUALITY_METRICS.md."""
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _research_financial_pdfs(company_name: str) -> dict:
    """Orchestriert die komplette PDF-Pipeline: URLs finden -> parallel
    herunterladen+parsen+extrahieren -> über alle PDFs normalisieren ->
    Gesamt-Konfidenz berechnen. Liefert {"found": False}, wenn keine PDFs
    gefunden oder erfolgreich geparst wurden – der erwartete Normalfall für
    die meisten Firmen, kein Ausnahmefall (siehe Modul-Docstring-Kommentar oben)."""
    urls = _find_financial_pdfs(company_name)
    _logger.log_agent_step("pdf_urls_found", count=len(urls), urls=urls)

    def _empty_result(pdfs_parsed: int = 0) -> dict:
        """Gemeinsamer Rückgabepfad für jeden 'nichts gefunden'-Fall (keine
        URLs, keine erfolgreich geparste PDF, keine Metriken übrig) – pdfs_found
        bleibt dabei immer die tatsächliche URL-Anzahl, damit main.py z.B.
        '0/5 geparst' statt '0/0' anzeigen kann."""
        return {
            "found": False,
            "pdfs_found": len(urls),
            "pdfs_successfully_parsed": pdfs_parsed,
            "metrics_extracted": [],
            "metrics_missing": list(_FINANCIAL_METRIC_KEYWORDS),
            "extraction_confidence": 0.0,
            "data_freshness": "unknown",
            "data_quality": "low",
        }

    if not urls:
        return _empty_result()

    def process_url(url: str) -> dict | None:
        cached_metrics = _pdf_cache.get_cached_extraction(url)
        if cached_metrics is not None:
            _logger.log_agent_step("cache_hit", url=url)
            # "_meta" reist im selben JSON wie die Metriken mit (siehe unten,
            # wo es beim Cache-Schreiben angehängt wird), statt eine zweite
            # Cache-Datei/einen zweiten PDFCache-Aufruf nur für ein Jahr zu
            # brauchen. Kollidiert nicht mit echten Metrik-Keys (revenue/
            # ebitda/profit/cash/marketing_spend, siehe
            # _FINANCIAL_METRIC_KEYWORDS).
            cached_metrics = dict(cached_metrics)
            meta = cached_metrics.pop("_meta", {})
            return {
                "url": url,
                "metrics": cached_metrics,
                "year": meta.get("year"),
                "credibility": _score_pdf_source_credibility(url),
                "from_cache": True,
            }

        pdf_bytes = _pdf_cache.get_cached_pdf(url)
        if pdf_bytes is not None:
            _logger.log_agent_step("cache_hit_pdf", url=url)
        else:
            pdf_bytes = _error_handler.retry_on_failure(_download_pdf, url)
            if not pdf_bytes:
                _error_handler.handle_pdf_download_error(url, "download failed after retries")
                return None
            _pdf_cache.cache_pdf(url, pdf_bytes)
            _logger.log_agent_step("pdf_downloaded", url=url, bytes=len(pdf_bytes))

        try:
            text = _extract_pdf_text(pdf_bytes)
            if not text.strip():
                raise ValueError("no extractable text (scanned image or empty PDF)")
            metrics = _extract_financial_metrics(text)
        except Exception as exc:
            _error_handler.handle_extraction_error(url, exc)
            return None

        if not metrics:
            _logger.log_agent_step("pdf_no_metrics_found", url=url)
            return None
        _logger.log_agent_step("pdf_parsed", url=url, text_length=len(text))
        _logger.log_agent_step("pdf_metrics_extracted", url=url, metrics=list(metrics.keys()))

        year = _extract_latest_year(text)
        cache_payload = dict(metrics)
        if year:
            cache_payload["_meta"] = {"year": year}
        _pdf_cache.cache_extraction(url, cache_payload)

        return {
            "url": url,
            "metrics": metrics,
            "year": year,
            "credibility": _score_pdf_source_credibility(url),
            "from_cache": False,
        }

    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        per_pdf_results = [r for r in executor.map(process_url, urls) if r is not None]

    if not per_pdf_results:
        return _empty_result()

    cache_hits = sum(1 for r in per_pdf_results if r["from_cache"])
    fresh_downloads = len(per_pdf_results) - cache_hits
    _logger.log_agent_step("pdf_cache_summary", cache_hits=cache_hits, fresh_downloads=fresh_downloads)

    merged_metrics = _normalize_extracted_data(per_pdf_results)
    if not merged_metrics:
        return _empty_result(pdfs_parsed=len(per_pdf_results))

    confidence = _calculate_confidence(merged_metrics)
    years = [r["year"] for r in per_pdf_results if r.get("year")]

    return {
        "found": True,
        "metrics": merged_metrics,
        "confidence": confidence,
        "pdf_count": len(per_pdf_results),
        "pdfs_from_cache": cache_hits,
        "pdfs_downloaded_fresh": fresh_downloads,
        # Phase 3e: Data-Quality-Metriken, gespiegelt in financial_data (siehe
        # analyze_financials unten) für main.py's Quality-Metrics-Anzeige.
        "pdfs_found": len(urls),
        "pdfs_successfully_parsed": len(per_pdf_results),
        "metrics_extracted": list(merged_metrics.keys()),
        "metrics_missing": [k for k in _FINANCIAL_METRIC_KEYWORDS if k not in merged_metrics],
        "extraction_confidence": confidence,
        "data_freshness": max(years) if years else "unknown",
        "data_quality": _classify_financial_quality(confidence),
    }


_PDF_METRIC_LABELS = {
    "de": {
        "revenue": "Umsatz", "ebitda": "EBITDA", "profit": "Gewinn",
        "cash": "Liquide Mittel", "marketing_spend": "Marketing-Ausgaben",
    },
    "en": {
        "revenue": "Revenue", "ebitda": "EBITDA", "profit": "Profit",
        "cash": "Cash", "marketing_spend": "Marketing spend",
    },
    "fr": {
        "revenue": "Chiffre d'affaires", "ebitda": "EBITDA", "profit": "Bénéfice",
        "cash": "Trésorerie", "marketing_spend": "Dépenses marketing",
    },
}

_PDF_SECTION_TEXT = {
    "de": {
        "header": "Finanzdaten aus Geschäftsberichten (PDF-Extraktion)",
        "disclaimer": (
            "Unverifiziert: automatisch per Regex aus per Websuche gefundenen PDFs "
            "extrahiert. Kann falsche, veraltete oder komplett irrelevante Werte "
            "enthalten (z.B. Zahlen aus themenfremden Dokumenten) – keine geprüften "
            "Finanzdaten, nicht ungeprüft übernehmen."
        ),
        "source_suffix": lambda count, cred: f"{count} Quelle(n), Ø Credibility {cred:.1f}",
        "confidence": lambda conf, count: f"Konfidenz: {conf:.0%} (aus {count} PDF(s))",
    },
    "en": {
        "header": "Financial data from annual reports (PDF extraction)",
        "disclaimer": (
            "Unverified: automatically regex-extracted from PDFs found via web search. "
            "May contain wrong, outdated, or entirely irrelevant values (e.g. numbers "
            "from unrelated documents) – not audited financial data, do not rely on "
            "this without independent verification."
        ),
        "source_suffix": lambda count, cred: f"{count} source(s), avg. credibility {cred:.1f}",
        "confidence": lambda conf, count: f"Confidence: {conf:.0%} (from {count} PDF(s))",
    },
    "fr": {
        "header": "Données financières des rapports annuels (extraction PDF)",
        "disclaimer": (
            "Non vérifié : extrait automatiquement par regex de PDF trouvés via "
            "recherche web. Peut contenir des valeurs erronées, obsolètes ou "
            "totalement hors sujet (ex. chiffres issus de documents sans rapport) – "
            "pas des données financières auditées, à ne pas utiliser sans "
            "vérification indépendante."
        ),
        "source_suffix": lambda count, cred: f"{count} source(s), crédibilité moy. {cred:.1f}",
        "confidence": lambda conf, count: f"Confiance : {conf:.0%} (à partir de {count} PDF(s))",
    },
}


def _format_pdf_financials(pdf_data: dict, language: str) -> str:
    """Baut die PDF-Finanzdaten-Sektion als Markdown, angehängt an
    budget_estimate_text in fit_agent.py (gleiches Anhänge-Muster wie
    _format_factor_breakdown/die Phase-1-Recherche-Sektionen).

    Führt einen auffälligen Disclaimer voran: reale Tests zeigten, dass die
    Regex-Extraktion gelegentlich plausibel aussehende, aber falsche Werte
    liefert (z.B. eine Jahreszahl wie "2024" als Gewinn missinterpretiert,
    oder Zahlen aus einem komplett themenfremden PDF für eine erfundene
    Firma) – die simplen "kein Ergebnis"-Fallbacks fangen das nicht ab,
    daher hier eine explizite, nicht zu übersehende Warnung statt nur eines
    dezenten Hinweises wie bei den übrigen KI-Schätzungen in diesem Modul."""
    if not pdf_data.get("found"):
        return ""
    labels = _PDF_METRIC_LABELS[language]
    texts = _PDF_SECTION_TEXT[language]
    lines = [
        f"- {labels.get(key, key)}: {value['value']:,.0f} {value['currency']} "
        f"({texts['source_suffix'](value['source_count'], value['avg_credibility'])})"
        for key, value in pdf_data["metrics"].items()
    ]
    header = texts["header"]
    disclaimer = texts["disclaimer"]
    confidence_line = texts["confidence"](pdf_data["confidence"], pdf_data["pdf_count"])
    return f"**{header}:**\n*{disclaimer}*\n\n" + "\n".join(lines) + f"\n\n{confidence_line}"


@traceable(run_type="chain", name="analyze_financials")
def analyze_financials(state: SponsorMatchState) -> dict:
    """Extrahiert das eigene Sponsoring-Portfolio der Firma (competitor_analysis-Plugin),
    schätzt ihre Unternehmensgröße (size_matching-Plugin) inkl. der daraus abgeleiteten
    Score-Anpassungen, und versucht (budget_estimator-Plugin, Phase 2) echte
    Finanzkennzahlen aus per Websuche gefundenen PDF-Geschäftsberichten zu extrahieren.
    Läuft als eigener Graph-Node zwischen research_company und evaluate_fit, unabhängig
    vom Score-Cache in evaluate_fit, da die Rohdaten auch bei gecachtem Score für die
    eigenen UI-Sektionen (main.py) gebraucht werden.
    """
    company_name = state["company_name"]
    club = state["club_profile"]
    language = state["language"]
    llm = _get_llm(state["selected_model"])
    token_usage = []
    _logger.log_agent_start("analyze_financials", company=company_name)

    if is_plugin_enabled("competitor_analysis"):
        portfolio = _analyze_company_sponsorship_portfolio(company_name, club, language, llm)
        if portfolio.get("token_usage"):
            token_usage.append(portfolio["token_usage"])
    else:
        portfolio = {"found": False}
    if portfolio["found"]:
        portfolio["saturation_level"] = _compute_saturation_level(portfolio["same_sport_count"])
        portfolio["score_adjustment"] = _compute_portfolio_score_impact(
            portfolio["match_percent"], portfolio["saturation_level"]
        )
    else:
        portfolio["saturation_level"] = "low"
        portfolio["score_adjustment"] = 0.0

    club_size = club.get("size", "Medium")
    if is_plugin_enabled("size_matching"):
        company_size_result = _estimate_company_size(company_name, llm)
        if company_size_result.get("token_usage"):
            token_usage.append(company_size_result["token_usage"])
    else:
        company_size_result = {"found": False, "size": "Medium"}

    size_adjustment = _compute_size_match_adjustment(club_size, company_size_result["size"])
    size_match_percent = _compute_size_match_percent(size_adjustment)
    size_compatibility = {
        "found": company_size_result["found"],
        "club_size": club_size,
        "company_size": company_size_result["size"],
        "match_percent": size_match_percent,
        "score_adjustment": size_adjustment,
        "explanation": _build_size_explanation(club_size, company_size_result["size"], size_match_percent, language),
    }

    # PDF-Finanzdaten: nutzt denselben budget_estimator-Toggle wie
    # _build_budget_estimate (fit_agent.py) – thematisch dieselbe Kategorie
    # (Budget-relevante Finanzdaten), kein eigenes neues Plugin nötig.
    if is_plugin_enabled("budget_estimator"):
        pdf_financials = _research_financial_pdfs(company_name)
    else:
        pdf_financials = {"found": False}

    # Phase 3e: schlanker Data-Quality-Auszug aus pdf_financials, in der von
    # main.py erwarteten Feldbenennung. .get()-Defaults fangen sowohl den
    # Plugin-deaktiviert-Fall (pdf_financials nur {"found": False}) als auch
    # jeden "nichts gefunden"-Pfad in _research_financial_pdfs ab.
    financial_data = {
        "pdfs_found": pdf_financials.get("pdfs_found", 0),
        "pdfs_successfully_parsed": pdf_financials.get("pdfs_successfully_parsed", 0),
        "metrics_extracted": pdf_financials.get("metrics_extracted", []),
        "metrics_missing": pdf_financials.get("metrics_missing", list(_FINANCIAL_METRIC_KEYWORDS)),
        "extraction_confidence": pdf_financials.get("extraction_confidence", 0.0),
        "data_freshness": pdf_financials.get("data_freshness", "unknown"),
        "data_quality": pdf_financials.get("data_quality", "low"),
    }

    _logger.log_agent_result(
        "analyze_financials",
        portfolio_found=portfolio["found"],
        company_size=company_size_result["size"],
        pdf_financials_found=pdf_financials.get("found", False),
        pdfs_processed=pdf_financials.get("pdf_count", 0),
        pdfs_from_cache=pdf_financials.get("pdfs_from_cache", 0),
        pdfs_downloaded_fresh=pdf_financials.get("pdfs_downloaded_fresh", 0),
        # Läuft seit Prozessstart mit (module-level ErrorHandler-Singleton, analog
        # zu _logger/_pdf_cache), ist also ein kumulativer Session-Zähler über
        # mehrere Analysen hinweg, keine reine Pro-Request-Zahl.
        errors_recovered_session_total=_error_handler.recovered_count,
        errors_fatal_session_total=_error_handler.fatal_count,
    )
    return {
        "competitor_analysis": portfolio,
        "size_compatibility": size_compatibility,
        "pdf_financials": pdf_financials,
        "financial_data": financial_data,
        "token_usage": token_usage,
    }
