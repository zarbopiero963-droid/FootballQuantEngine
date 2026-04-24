"""
API-Football v3 collector with rate limiting, retry, and full endpoint coverage.

Rate limiting
-------------
API-Football enforces per-minute and per-day quotas that vary by plan:
  Free  : 100 req/day, no per-minute limit stated (be conservative: 10/min)
  Basic+: 3000 req/day, 300/min
  Pro+  : higher tiers

This collector uses a token-bucket rate limiter (default: 10 req/min) and
retries with exponential back-off on 429 / 5xx responses.

Usage
-----
    from data.api_football_collector import ApiFootballCollector

    collector = ApiFootballCollector(requests_per_minute=10)
    live      = collector.get_live_fixtures()
    finished  = collector.get_fixtures_by_status("FT")
    upcoming  = collector.get_next_fixtures(n=200)
    season    = collector.get_league_season(135, 2024)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config.constants import BASE_URL_API_FOOTBALL
from config.settings_manager import load_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants (API-Football v3)
# ---------------------------------------------------------------------------

STATUS_FINISHED = frozenset({"FT", "AET", "PEN"})
STATUS_LIVE = frozenset({"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"})
STATUS_SCHEDULED = frozenset({"NS", "TBD"})
STATUS_INVALID = frozenset({"PST", "CANC", "ABD", "AWD", "WO"})
STATUS_ALL = STATUS_FINISHED | STATUS_LIVE | STATUS_SCHEDULED | STATUS_INVALID

# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Thread-safe token bucket for per-minute rate limiting."""

    def __init__(self, rate_per_minute: float) -> None:
        self._rate = rate_per_minute / 60.0  # tokens per second
        self._capacity = max(1.0, rate_per_minute)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def _with_retry(fn, max_attempts: int = 4, base_delay: float = 2.0):
    """
    Call fn() with exponential back-off on transient failures.

    Handles:
      - requests.exceptions.RequestException (network errors)
      - HTTP 429 (rate limit exceeded) — uses Retry-After header if present
      - HTTP 5xx (server errors)
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = fn()
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", base_delay * attempt))
                logger.warning(
                    "API-Football 429 rate limit; waiting %.0fs (attempt %d/%d)",
                    retry_after,
                    attempt,
                    max_attempts,
                )
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "API-Football %d server error; retry in %.0fs (attempt %d/%d)",
                    resp.status_code,
                    delay,
                    attempt,
                    max_attempts,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            delay = base_delay * (2 ** (attempt - 1))
            if attempt == max_attempts:
                raise
            logger.warning(
                "API-Football network error: %s; retry in %.0fs (attempt %d/%d)",
                exc,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)
    raise RuntimeError("All retry attempts exhausted")  # unreachable but satisfies linters


# ---------------------------------------------------------------------------
# Fixture parser
# ---------------------------------------------------------------------------


def _parse_fixture(raw: Dict) -> Optional[Dict]:
    """
    Convert a raw API-Football fixture dict to a flat internal dict.

    Returns None if mandatory fields are missing.
    """
    try:
        fix = raw["fixture"]
        teams = raw["teams"]
        goals = raw.get("goals") or {}
        league = raw.get("league") or {}
        score = raw.get("score") or {}

        status_short = fix["status"]["short"]

        return {
            "fixture_id": int(fix["id"]),
            "league_id": int(league.get("id") or 0),
            "league": str(league.get("name") or ""),
            "season": int(league.get("season") or 0),
            "home": str(teams["home"]["name"]),
            "away": str(teams["away"]["name"]),
            "match_date": str(fix.get("date") or ""),
            "status": status_short,
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "venue": str((fix.get("venue") or {}).get("name") or ""),
            "referee": str(fix.get("referee") or ""),
            "elapsed": (fix.get("status") or {}).get("elapsed"),
            # Nested odds fields are populated separately by the odds poller
            "home_odds": None,
            "draw_odds": None,
            "away_odds": None,
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("_parse_fixture: skipped malformed record: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------


class ApiFootballCollector:
    """
    API-Football v3 collector with rate limiting and retry.

    Parameters
    ----------
    requests_per_minute : conservative per-minute request budget (default 10)
    max_retries         : number of retry attempts per request (default 4)
    timeout             : HTTP request timeout in seconds (default 30)
    """

    def __init__(
        self,
        requests_per_minute: float = 10.0,
        max_retries: int = 4,
        timeout: float = 30.0,
    ) -> None:
        self._settings = load_settings()
        self._bucket = _TokenBucket(requests_per_minute)
        self._max_retries = max_retries
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"x-apisports-key": self._settings.api_football_key}
        )

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a GET request against the API-Football v3 base URL.

        Returns the ``response`` list from the JSON payload.
        Applies rate limiting and retry automatically.
        """
        self._bucket.consume()
        url = f"{BASE_URL_API_FOOTBALL}/{endpoint.lstrip('/')}"
        params = params or {}

        def _call():
            return self._session.get(url, params=params, timeout=self._timeout)

        resp = _with_retry(_call, max_attempts=self._max_retries)
        data = resp.json()

        errors = data.get("errors")
        if errors and errors not in ({}, [], None):
            logger.warning("API-Football errors on %s %s: %s", endpoint, params, errors)

        results = data.get("response", [])
        remaining = resp.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None and int(remaining) < 10:
            logger.warning(
                "API-Football quota almost exhausted: %s requests remaining.", remaining
            )
        return results

    def _parse_fixtures(self, raw_list: List[Dict]) -> List[Dict]:
        """Parse a list of raw fixture dicts, dropping invalid ones."""
        out = []
        for raw in raw_list:
            parsed = _parse_fixture(raw)
            if parsed is not None:
                out.append(parsed)
        return out

    # ------------------------------------------------------------------
    # Fixture endpoints
    # ------------------------------------------------------------------

    def get_live_fixtures(self) -> List[Dict]:
        """
        Fetch all currently live fixtures across all leagues.

        Covers statuses: 1H, HT, 2H, ET, BT, P, INT, LIVE.
        """
        raw = self._get("fixtures", {"live": "all"})
        fixtures = self._parse_fixtures(raw)
        logger.info("get_live_fixtures: %d live fixtures fetched.", len(fixtures))
        return fixtures

    def get_fixtures_by_status(self, status: str) -> List[Dict]:
        """
        Fetch fixtures filtered by a single status string.

        status examples: "FT", "NS", "PST", "CANC"
        """
        raw = self._get("fixtures", {"status": status})
        fixtures = self._parse_fixtures(raw)
        logger.info(
            "get_fixtures_by_status(%s): %d fixtures fetched.", status, len(fixtures)
        )
        return fixtures

    def get_next_fixtures(self, n: int = 200) -> List[Dict]:
        """Fetch the next N scheduled fixtures across all leagues."""
        raw = self._get("fixtures", {"next": min(n, 500)})
        fixtures = self._parse_fixtures(raw)
        logger.info("get_next_fixtures(%d): %d fixtures fetched.", n, len(fixtures))
        return fixtures

    def get_fixtures_by_date(self, date_str: str) -> List[Dict]:
        """
        Fetch all fixtures for a specific date (YYYY-MM-DD).

        Useful for fetching yesterday's date to catch late-finishing matches.
        """
        raw = self._get("fixtures", {"date": date_str})
        fixtures = self._parse_fixtures(raw)
        logger.info(
            "get_fixtures_by_date(%s): %d fixtures fetched.", date_str, len(fixtures)
        )
        return fixtures

    def get_league_season(
        self, league_id: int, season: int
    ) -> List[Dict]:
        """
        Fetch all fixtures for a given league and season.

        Used by the historical downloader.
        """
        raw = self._get("fixtures", {"league": league_id, "season": season})
        fixtures = self._parse_fixtures(raw)
        logger.info(
            "get_league_season(league=%d, season=%d): %d fixtures fetched.",
            league_id,
            season,
            len(fixtures),
        )
        return fixtures

    def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict]:
        """Fetch a single fixture by its API-Football ID."""
        raw = self._get("fixtures", {"id": fixture_id})
        if not raw:
            return None
        return _parse_fixture(raw[0])

    # ------------------------------------------------------------------
    # Odds endpoints
    # ------------------------------------------------------------------

    def get_fixture_odds(
        self, fixture_id: int, bookmaker_id: int = 6
    ) -> Optional[Dict[str, float]]:
        """
        Fetch pre-match 1×2 odds for a fixture from one bookmaker.

        bookmaker_id=6 → Bet365 (commonly available on API-Football).

        Returns dict with home_odds, draw_odds, away_odds or None.
        """
        raw = self._get(
            "odds",
            {"fixture": fixture_id, "bookmaker": bookmaker_id, "bet": 1},
        )
        if not raw:
            return None
        try:
            bets = raw[0]["bookmakers"][0]["bets"][0]["values"]
            mapping = {v["value"]: float(v["odd"]) for v in bets}
            return {
                "home_odds": mapping.get("Home"),
                "draw_odds": mapping.get("Draw"),
                "away_odds": mapping.get("Away"),
            }
        except (IndexError, KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # League/season discovery
    # ------------------------------------------------------------------

    def get_available_seasons(self, league_id: int) -> List[int]:
        """Return list of available season years for a league."""
        raw = self._get("leagues", {"id": league_id})
        if not raw:
            return []
        try:
            seasons = raw[0]["seasons"]
            return sorted(int(s["year"]) for s in seasons if s.get("coverage"))
        except (IndexError, KeyError, TypeError):
            return []

    def get_remaining_quota(self) -> Dict[str, Any]:
        """
        Query the API status endpoint to get current quota usage.

        Returns dict with requests_limit, requests_remaining, requests_current.
        """
        raw = self._get("status")
        if not raw:
            return {}
        try:
            r = raw[0]["requests"]
            return {
                "requests_limit": r.get("limit_day"),
                "requests_remaining": r.get("current"),
                "plan": raw[0].get("subscription", {}).get("plan"),
            }
        except (IndexError, KeyError, TypeError):
            return {}
