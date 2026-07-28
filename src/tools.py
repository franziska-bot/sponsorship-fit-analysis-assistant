import os
import time
import functools

import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

def get_search_tool():
    """Erstellt das Tavily-Such-Tool für die Firmenrecherche."""
    return TavilySearch(
        max_results=5,
        api_key=os.environ["TAVILY_API_KEY"],
    )

# --- Companies House: externe Company-Intelligence-API (UK) ---
#
# HINWEIS: Companies House verlangt einen API-Key per HTTP Basic Auth
# (Key als Username, leeres Passwort) – per Live-Test verifiziert: ohne
# Auth-Header kommt 401 "Empty Authorization header", mit einem
# Dummy-Key 401 "Invalid Authorization". Anders als bei Crunchbase/
# OpenCorporates ist die Registrierung hier aber wirklich kostenlos und
# unkompliziert (offizielle UK-Behörden-API): kostenloser Account unter
# https://developer.company-information.service.gov.uk/ → REST-API-Key
# erzeugen → als COMPANIES_HOUSE_API_KEY in .env eintragen. Ohne Key (oder
# bei API-Fehlern) greift der Graceful Fallback unten – der Agent läuft
# normal weiter, nur ohne diese Zusatzdaten. Deckt naturgemäß nur UK-
# registrierte Firmen ab.

COMPANIES_HOUSE_SEARCH_URL = "https://api.companieshouse.gov.uk/search/companies"

_COMPANY_TYPE_NAMES = {
    "ltd": "Private Limited Company",
    "plc": "Public Limited Company",
    "llp": "Limited Liability Partnership",
    "private-unlimited": "Private Unlimited Company",
    "old-public-company": "Old Public Company",
    "private-limited-guarant-nsc": "Private Limited by Guarantee",
}

@functools.lru_cache(maxsize=256)
def _fetch_companies_house_data(company_name: str, cache_bucket: int) -> dict:
    """Tatsächliche Implementierung, gecacht über lru_cache (siehe get_company_intelligence).

    cache_bucket wechselt einmal pro Woche (siehe Aufrufer) und sorgt so für
    ein ~7-Tage-TTL, ohne eine zusätzliche Cache-Bibliothek zu brauchen.
    """
    del cache_bucket  # nur Teil des Cache-Keys, wird hier nicht gebraucht

    api_key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not api_key:
        return {"error": "Companies House API nicht verfügbar (kein API-Key konfiguriert)."}

    try:
        response = requests.get(
            COMPANIES_HOUSE_SEARCH_URL,
            params={"q": company_name},
            headers={"Accept": "application/json"},
            auth=(api_key, ""),
            timeout=10,
        )
    except requests.exceptions.RequestException:
        return {"error": "Companies House API nicht verfügbar"}

    if response.status_code != 200:
        return {"error": "Companies House API nicht verfügbar"}

    try:
        items = response.json()["items"]
    except (ValueError, KeyError, TypeError):
        return {"error": "Companies House API nicht verfügbar"}

    if not items:
        return {"error": "Keine UK-Registrierung gefunden"}

    item = items[0]
    company_type_code = item.get("company_type") or ""

    return {
        "company_name": item.get("company_name") or item.get("title") or company_name,
        "company_number": item.get("company_number"),
        "address": item.get("address_snippet"),
        "company_status": item.get("company_status"),
        "date_of_creation": item.get("date_of_creation"),
        "company_type": _COMPANY_TYPE_NAMES.get(company_type_code, company_type_code.replace("-", " ").title()),
    }

@tool
def get_company_intelligence(company_name: str) -> dict:
    """Zieht UK-Firmenregistrierungsdaten (Company Number, Status, Gründungsdatum,
    Rechtsform, Adresse) von der Companies House API für die angegebene Firma.
    Deckt nur UK-registrierte Firmen ab. Gibt bei Fehlern/fehlendem Key ein
    Dict mit "error"-Key zurück statt zu werfen (Graceful Fallback)."""
    cache_bucket = int(time.time() // (7 * 86400))  # ~7-Tage-TTL: ein neuer Bucket pro Woche
    return _fetch_companies_house_data(company_name, cache_bucket)
