"""
League Registry
===============
Single source of truth for API-Football league ID → name mapping.

Resolution order:
  1. Static built-in table (works offline, covers ~60 major leagues)
  2. Local DB  (fixtures table — populated after first bootstrap)
  3. API call  (last resort, requires API key)
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static table — major leagues from api-football.com
# ---------------------------------------------------------------------------
_STATIC: dict[int, str] = {
    # UEFA
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Europa Conference League",
    # England
    39: "Premier League",
    40: "Championship",
    41: "League One",
    42: "League Two",
    45: "FA Cup",
    48: "Carabao Cup",
    # Italy
    135: "Serie A",
    136: "Serie B",
    137: "Coppa Italia",
    # Spain
    140: "La Liga",
    141: "La Liga 2",
    143: "Copa del Rey",
    # Germany
    78: "Bundesliga",
    79: "2. Bundesliga",
    81: "DFB Pokal",
    # France
    61: "Ligue 1",
    62: "Ligue 2",
    66: "Coupe de France",
    # Netherlands
    88: "Eredivisie",
    89: "Eerste Divisie",
    # Portugal
    94: "Primeira Liga",
    95: "Liga Portugal 2",
    96: "Taca de Portugal",
    # Belgium
    144: "Pro League",
    145: "Eerste Klasse B",
    # Turkey
    203: "Süper Lig",
    204: "TFF First League",
    # Russia
    235: "Premier League Russia",
    # Brazil
    71: "Serie A Brazil",
    72: "Serie B Brazil",
    # Argentina
    128: "Liga Profesional",
    # Mexico
    262: "Liga MX",
    # USA
    253: "MLS",
    # Scotland
    179: "Scottish Premiership",
    # Greece
    197: "Super League Greece",
    # Austria
    218: "Bundesliga Austria",
    # Switzerland
    207: "Super League Switzerland",
    # Denmark
    119: "Superliga Denmark",
    # Sweden
    113: "Allsvenskan",
    # Norway
    103: "Eliteserien",
    # Poland
    106: "Ekstraklasa",
    # Czech Republic
    345: "Czech First League",
    # Romania
    283: "Liga 1 Romania",
    # Croatia
    210: "HNL Croatia",
    # Serbia
    286: "Super Liga Serbia",
    # Japan
    98: "J1 League",
    # South Korea
    292: "K League 1",
    # China
    169: "Chinese Super League",
    # Saudi Arabia
    307: "Saudi Pro League",
}


def name(league_id: int, db_name: str | None = None, api_key: str | None = None) -> str:
    """Return the human-readable name for a league ID.

    Falls back through: static table → DB → API → "League {id}".
    """
    if league_id in _STATIC:
        return _STATIC[league_id]

    if db_name:
        result = _from_db(league_id, db_name)
        if result:
            return result

    if api_key:
        result = _from_api(league_id, api_key)
        if result:
            _STATIC[league_id] = result  # cache for this session
            return result

    return f"League {league_id}"


def all_known() -> dict[int, str]:
    """Return a copy of the full static registry."""
    return dict(_STATIC)


def enrich_from_db(db_name: str) -> None:
    """Load any league names present in the fixtures table into the in-memory cache."""
    try:
        conn = sqlite3.connect(db_name)
        rows = conn.execute(
            "SELECT DISTINCT league_id, league FROM fixtures "
            "WHERE league_id IS NOT NULL AND league IS NOT NULL AND league != ''"
        ).fetchall()
        conn.close()
        for lid, lname in rows:
            if lid not in _STATIC:
                _STATIC[lid] = lname
    except Exception as exc:
        logger.debug("LeagueRegistry.enrich_from_db failed: %s", exc)


def _from_db(league_id: int, db_name: str) -> str:
    try:
        conn = sqlite3.connect(db_name)
        row = conn.execute(
            "SELECT league FROM fixtures WHERE league_id=? AND league IS NOT NULL LIMIT 1",
            (league_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def _from_api(league_id: int, api_key: str) -> str:
    try:
        import requests

        resp = requests.get(
            "https://v3.football.api-sports.io/leagues",
            headers={"x-apisports-key": api_key},
            params={"id": league_id},
            timeout=10,
        )
        data = resp.json().get("response", [])
        if data:
            return data[0]["league"]["name"]
    except Exception as exc:
        logger.debug("LeagueRegistry API lookup failed for %d: %s", league_id, exc)
    return ""
