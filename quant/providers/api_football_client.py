"""
API-Football client — full coverage:
  - All seasons for a league
  - Full fixture list per season (with halftime scores)
  - Fixture statistics (corners, shots, xG, cards, possession)
  - Pre-match odds for all markets (1X2, OU, BTTS, corners, HT, CS)
  - Live fixtures with current score and elapsed time
  - Upcoming fixtures
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

BASE_URL = "https://v3.football.api-sports.io"

logger = logging.getLogger(__name__)


class APIFootballClient:

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY missing — set in Settings or env var")
        self.headers = {"x-apisports-key": self.api_key}
        self._last_call = 0.0
        self._min_interval = 0.35  # ~3 req/s to stay under free-tier limits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> list[Any]:
        # Gentle rate-limiting — pause if called too quickly
        gap = time.monotonic() - self._last_call
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)
        self._last_call = time.monotonic()

        url = BASE_URL + path
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("API-Football request failed: %s — %s", url, exc)
            raise

        payload = resp.json()
        if "response" not in payload:
            raise RuntimeError(f"API-Football: missing 'response' in payload for {path}")
        return payload["response"]

    def _paginate(self, path: str, params: dict) -> list[Any]:
        """Fetch all pages for endpoints that support pagination (page param)."""
        results: list[Any] = []
        page = 1
        while True:
            params["page"] = page
            chunk = self._get(path, params)
            if not chunk:
                break
            results.extend(chunk)
            if len(chunk) < 20:
                break
            page += 1
        return results

    # ------------------------------------------------------------------
    # Seasons
    # ------------------------------------------------------------------

    def get_available_seasons(self, league_id: int) -> list[int]:
        """Return all seasons available for a league, oldest first."""
        data = self._get("/leagues", {"id": league_id})
        if not data:
            return []
        seasons = data[0].get("seasons", [])
        return sorted({int(s["year"]) for s in seasons})

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def get_fixtures(self, league_id: int, season: int) -> list[dict]:
        """All fixtures for a season (finished + upcoming)."""
        data = self._get("/fixtures", {"league": league_id, "season": season})
        return [self._parse_fixture(item) for item in data]

    def get_completed_matches(self, league_id: int, season: int) -> list[dict]:
        """Only FT fixtures with goals."""
        data = self._get("/fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT",
        })
        matches = []
        for item in data:
            f = self._parse_fixture(item)
            if f["home_goals"] is not None and f["away_goals"] is not None:
                matches.append(f)
        return matches

    def get_upcoming_matches(self, league_id: int, season: int) -> list[dict]:
        """Fixtures not yet played (NS = Not Started)."""
        data = self._get("/fixtures", {
            "league": league_id,
            "season": season,
            "status": "NS",
        })
        return [self._parse_fixture(item) for item in data]

    def get_live_fixtures(self) -> list[dict]:
        """All currently live fixtures across all leagues."""
        data = self._get("/fixtures", {"live": "all"})
        return [self._parse_fixture(item) for item in data]

    def _parse_fixture(self, item: dict) -> dict:
        fix = item.get("fixture", {})
        goals = item.get("goals", {})
        score = item.get("score", {})
        teams = item.get("teams", {})
        league = item.get("league", {})

        ht = score.get("halftime", {})
        return {
            "fixture_id":  str(fix.get("id", "")),
            "league_id":   league.get("id"),
            "league":      league.get("name", ""),
            "season":      league.get("season"),
            "home_team":   teams.get("home", {}).get("name", ""),
            "away_team":   teams.get("away", {}).get("name", ""),
            "match_date":  fix.get("date", ""),
            "status":      fix.get("status", {}).get("short", ""),
            "elapsed":     fix.get("status", {}).get("elapsed"),
            "venue":       fix.get("venue", {}).get("name", ""),
            "referee":     fix.get("referee", ""),
            "home_goals":  goals.get("home"),
            "away_goals":  goals.get("away"),
            "ht_home":     ht.get("home"),
            "ht_away":     ht.get("away"),
        }

    # ------------------------------------------------------------------
    # Fixture statistics (corners, shots, xG, cards, possession)
    # ------------------------------------------------------------------

    def get_fixture_stats(self, fixture_id: str | int) -> dict:
        """Return a flat dict of home/away stats for a finished fixture."""
        data = self._get("/fixtures/statistics", {"fixture": fixture_id})
        result: dict = {"fixture_id": str(fixture_id)}

        def _val(stats: list[dict], name: str) -> Any:
            for s in stats:
                if s.get("type", "").lower() == name.lower():
                    v = s.get("value")
                    return v if v not in (None, "null", "") else None
            return None

        for team_block in data:
            side = "home" if team_block.get("team", {}).get("name") else None
            # Determine home/away by order: first block = home, second = away
            pass  # handled below

        if len(data) >= 2:
            home_stats = data[0].get("statistics", [])
            away_stats = data[1].get("statistics", [])

            def safe_int(v: Any) -> int | None:
                try:
                    return int(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            def safe_float(v: Any) -> float | None:
                if v is None:
                    return None
                try:
                    s = str(v).replace("%", "").strip()
                    return float(s)
                except (TypeError, ValueError):
                    return None

            result.update({
                "home_corners":    safe_int(_val(home_stats, "Corner Kicks")),
                "away_corners":    safe_int(_val(away_stats, "Corner Kicks")),
                "home_shots":      safe_int(_val(home_stats, "Total Shots")),
                "away_shots":      safe_int(_val(away_stats, "Total Shots")),
                "home_shots_on":   safe_int(_val(home_stats, "Shots on Goal")),
                "away_shots_on":   safe_int(_val(away_stats, "Shots on Goal")),
                "home_possession": safe_float(_val(home_stats, "Ball Possession")),
                "away_possession": safe_float(_val(away_stats, "Ball Possession")),
                "home_yellow":     safe_int(_val(home_stats, "Yellow Cards")),
                "away_yellow":     safe_int(_val(away_stats, "Yellow Cards")),
                "home_red":        safe_int(_val(home_stats, "Red Cards")),
                "away_red":        safe_int(_val(away_stats, "Red Cards")),
                "home_xg":         safe_float(_val(home_stats, "expected_goals")),
                "away_xg":         safe_float(_val(away_stats, "expected_goals")),
            })

        return result

    # ------------------------------------------------------------------
    # Odds — pre-match, all markets
    # ------------------------------------------------------------------

    # Bet names returned by API-Football for each market
    _MARKET_NAMES = {
        "1x2":         "Match Winner",
        "ou25":        "Goals Over/Under",
        "ou15":        "Goals Over/Under",
        "ou35":        "Goals Over/Under",
        "btts":        "Both Teams Score",
        "ht_ou":       "Goals Over/Under First Half",
        "corners_ou":  "Corner Kicks Over/Under",
        "cs":          "Correct Score",
    }

    def get_prematch_odds(self, fixture_ids: list) -> dict[str, dict]:
        """
        Returns a dict keyed by fixture_id.
        Each value: {
            '1x2':  {'home': 2.10, 'draw': 3.40, 'away': 3.20},
            'ou25': {'over': 1.85, 'under': 1.95},
            'ou15': {'over': 1.25, 'under': 3.80},
            'ou35': {'over': 3.10, 'under': 1.35},
            'btts':  {'yes': 1.75, 'no': 1.95},
            'ht_ou': {'over': 2.20, 'under': 1.65},
            'corners_ou': {'over': 1.85, 'under': 1.95},
            'cs':   {'1-0': 7.00, '2-1': 9.00, ...},
        }
        """
        odds_map: dict[str, dict] = {}
        for fid in fixture_ids:
            try:
                data = self._get("/odds", {"fixture": fid})
                if not data:
                    continue
                odds_map[str(fid)] = self._parse_all_markets(data)
            except Exception as exc:
                logger.warning("Odds fetch failed for fixture %s: %s", fid, exc)
        return odds_map

    def _parse_all_markets(self, data: list) -> dict:
        result: dict = {}
        if not data:
            return result

        bookmakers = data[0].get("bookmakers", [])
        if not bookmakers:
            return result

        bets = bookmakers[0].get("bets", [])

        for bet in bets:
            name: str = bet.get("name", "")
            values: list[dict] = bet.get("values", [])

            if name == "Match Winner":
                result["1x2"] = self._extract_1x2(values)

            elif name == "Goals Over/Under":
                for line, key in [("2.5", "ou25"), ("1.5", "ou15"), ("3.5", "ou35")]:
                    entry = self._extract_ou(values, line)
                    if entry:
                        result[key] = entry

            elif name == "Both Teams Score":
                result["btts"] = self._extract_btts(values)

            elif name in ("Goals Over/Under First Half", "First Half - Goals Over/Under"):
                entry = self._extract_ou(values, "1.5")
                if not entry:
                    entry = self._extract_ou_first(values)
                if entry:
                    result["ht_ou15"] = entry

            elif "Corner" in name and "Over/Under" in name:
                for line in ["9.5", "10.5", "8.5"]:
                    entry = self._extract_ou(values, line)
                    if entry:
                        result[f"corners_{line}"] = entry
                        break

            elif name == "Correct Score":
                result["cs"] = self._extract_cs(values)

        return result

    def _extract_1x2(self, values: list) -> dict:
        out: dict = {}
        for v in values:
            label = v.get("value", "")
            odd = self._parse_odd(v.get("odd"))
            if odd is None:
                continue
            if label == "Home":
                out["home"] = odd
            elif label == "Draw":
                out["draw"] = odd
            elif label == "Away":
                out["away"] = odd
        return out

    def _extract_ou(self, values: list, line: str) -> dict:
        out: dict = {}
        for v in values:
            label: str = v.get("value", "")
            odd = self._parse_odd(v.get("odd"))
            if odd is None:
                continue
            if f"Over {line}" in label or label == f"Over({line})":
                out["over"] = odd
            elif f"Under {line}" in label or label == f"Under({line})":
                out["under"] = odd
        return out

    def _extract_ou_first(self, values: list) -> dict:
        """Fallback: take first Over/Under pair regardless of line."""
        out: dict = {}
        for v in values:
            label: str = v.get("value", "")
            odd = self._parse_odd(v.get("odd"))
            if odd is None:
                continue
            if "Over" in label and "over" not in out:
                out["over"] = odd
            elif "Under" in label and "under" not in out:
                out["under"] = odd
        return out

    def _extract_btts(self, values: list) -> dict:
        out: dict = {}
        for v in values:
            label = v.get("value", "").lower()
            odd = self._parse_odd(v.get("odd"))
            if odd is None:
                continue
            if label in ("yes", "si"):
                out["yes"] = odd
            elif label == "no":
                out["no"] = odd
        return out

    def _extract_cs(self, values: list) -> dict:
        out: dict = {}
        for v in values:
            label: str = v.get("value", "")
            odd = self._parse_odd(v.get("odd"))
            if odd is not None:
                out[label] = odd
        return out

    @staticmethod
    def _parse_odd(raw: Any) -> float | None:
        try:
            f = float(raw)
            return f if f > 1.0 else None
        except (TypeError, ValueError):
            return None
