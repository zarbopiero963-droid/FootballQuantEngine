"""
API-Football client — full endpoint coverage:

  Leagues/Meta   : leagues, seasons, countries, timezones, venues
  Fixtures       : all, completed, upcoming, live, statistics
  Teams          : info, season statistics
  Players        : season stats, top scorers/assists/cards, seasons
  Standings      : league table
  Odds           : pre-match (all markets), in-play
  Injuries       : squad injury list
  Sidelined      : long-term absences
  Predictions    : API-Football's own match prediction signal
  Coaches        : coach profile per team
  Transfers      : inbound/outbound transfer history
  Trophies       : title history for a player or coach
  Referee stats  : derived from completed fixtures
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
        """Fetch all pages for endpoints that support pagination."""
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
    # Leagues / seasons
    # ------------------------------------------------------------------

    def get_leagues(self) -> list:
        return self._get("/leagues")

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
        fix    = item.get("fixture", {})
        goals  = item.get("goals", {})
        score  = item.get("score", {})
        teams  = item.get("teams", {})
        league = item.get("league", {})
        ht     = score.get("halftime", {})

        return {
            "fixture_id": str(fix.get("id", "")),
            "league_id":  league.get("id"),
            "league":     league.get("name", ""),
            "season":     league.get("season"),
            "home_team":  teams.get("home", {}).get("name", ""),
            "away_team":  teams.get("away", {}).get("name", ""),
            "match_date": fix.get("date", ""),
            "status":     fix.get("status", {}).get("short", ""),
            "elapsed":    fix.get("status", {}).get("elapsed"),
            "venue":      fix.get("venue", {}).get("name", ""),
            "referee":    fix.get("referee", ""),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "ht_home":    ht.get("home"),
            "ht_away":    ht.get("away"),
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
                    return float(str(v).replace("%", "").strip())
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

    def get_fixture_statistics(self, fixture_ids: list) -> dict[str, dict]:
        """
        Returns per-team statistics keyed by fixture_id.
        {fixture_id: {team_name: {shots, shots_on_target, xg}}}
        """
        stats_map: dict = {}
        for fixture_id in fixture_ids:
            data = self._get("/fixtures/statistics", {"fixture": fixture_id})
            if not data or len(data) < 2:
                continue
            try:
                result: dict = {}
                for team_block in data:
                    team_name = team_block["team"]["name"]
                    raw = {
                        s["type"]: s["value"] for s in team_block.get("statistics", [])
                    }
                    result[team_name] = {
                        "shots": float(raw.get("Total Shots") or 0),
                        "shots_on_target": float(raw.get("Shots on Goal") or 0),
                        "xg": float(raw.get("Expected Goals") or 0),
                    }
                stats_map[str(fixture_id)] = result
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                logger.warning("Failed to parse stats for fixture %s: %s", fixture_id, exc)
        return stats_map

    def get_team_xg_averages(self, league: int, season: int) -> dict[str, dict]:
        """Derives per-team average xG from completed fixture statistics."""
        completed = self.get_completed_matches(league, season)
        fixture_ids = [m["fixture_id"] for m in completed]
        all_stats = self.get_fixture_statistics(fixture_ids)

        team_acc: dict[str, dict] = {}
        for match in completed:
            fid = match["fixture_id"]
            stats = all_stats.get(fid)
            if not stats:
                continue
            home = match["home_team"]
            away = match["away_team"]
            hs   = stats.get(home, {})
            as_  = stats.get(away, {})

            for team, scored, conceded in ((home, hs, as_), (away, as_, hs)):
                if team not in team_acc:
                    team_acc[team] = {
                        "xg_for": 0.0, "xg_against": 0.0,
                        "shots": 0.0, "shots_on_target": 0.0, "n": 0,
                    }
                team_acc[team]["xg_for"]          += scored.get("xg", 0.0)
                team_acc[team]["xg_against"]       += conceded.get("xg", 0.0)
                team_acc[team]["shots"]            += scored.get("shots", 0.0)
                team_acc[team]["shots_on_target"]  += scored.get("shots_on_target", 0.0)
                team_acc[team]["n"] += 1

        averages: dict[str, dict] = {}
        for team, acc in team_acc.items():
            n = max(acc["n"], 1)
            averages[team] = {
                "xg_for":          acc["xg_for"] / n,
                "xg_against":      acc["xg_against"] / n,
                "shots":           acc["shots"] / n,
                "shots_on_target": acc["shots_on_target"] / n,
            }
        return averages

    # ------------------------------------------------------------------
    # Odds — pre-match, all markets
    # ------------------------------------------------------------------

    def get_odds(self, fixture_ids) -> dict:
        """Alias for get_prematch_odds (backwards compatibility)."""
        return self.get_prematch_odds(list(fixture_ids))

    def get_prematch_odds(self, fixture_ids: list) -> dict[str, dict]:
        """
        Returns odds keyed by fixture_id, with sub-dicts per market:
          '1x2':  {'home': 2.10, 'draw': 3.40, 'away': 3.20}
          'ou25': {'over': 1.85, 'under': 1.95}
          'btts':  {'yes': 1.75, 'no': 1.95}
          'ht_ou15': {'over': 2.20, 'under': 1.65}
          'corners_9.5': {'over': 1.85, 'under': 1.95}
          'cs':   {'1-0': 7.00, ...}
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
            name:   str        = bet.get("name", "")
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
                entry = self._extract_ou(values, "1.5") or self._extract_ou_first(values)
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
            odd   = self._parse_odd(v.get("odd"))
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
            odd        = self._parse_odd(v.get("odd"))
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
            odd        = self._parse_odd(v.get("odd"))
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
            odd   = self._parse_odd(v.get("odd"))
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
            odd        = self._parse_odd(v.get("odd"))
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

    # ------------------------------------------------------------------
    # Countries
    # ------------------------------------------------------------------

    def get_countries(self) -> list[dict]:
        """All countries supported by API-Football."""
        data = self._get("/countries")
        return [
            {"name": c.get("name", ""), "code": c.get("code", ""), "flag": c.get("flag", "")}
            for c in data
        ]

    # ------------------------------------------------------------------
    # Timezones
    # ------------------------------------------------------------------

    def get_timezones(self) -> list[str]:
        """All valid timezone strings accepted by the API."""
        return self._get("/timezone")

    # ------------------------------------------------------------------
    # Venues
    # ------------------------------------------------------------------

    def get_venues(
        self,
        *,
        venue_id: int | None = None,
        name: str | None = None,
        city: str | None = None,
        country: str | None = None,
    ) -> list[dict]:
        """Search venues by id, name, city, or country."""
        params: dict = {}
        if venue_id is not None:
            params["id"] = venue_id
        if name:
            params["name"] = name
        if city:
            params["city"] = city
        if country:
            params["country"] = country
        data = self._get("/venues", params or None)
        return [
            {
                "id":       v.get("id"),
                "name":     v.get("name", ""),
                "city":     v.get("city", ""),
                "country":  v.get("country", ""),
                "capacity": v.get("capacity"),
                "surface":  v.get("surface", ""),
            }
            for v in data
        ]

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def get_teams(
        self,
        *,
        league: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        search: str | None = None,
    ) -> list[dict]:
        """Search/list teams. Use league+season for a full squad list."""
        params: dict = {}
        if league is not None:
            params["league"] = league
        if season is not None:
            params["season"] = season
        if team_id is not None:
            params["id"] = team_id
        if search:
            params["search"] = search
        data = self._get("/teams", params or None)
        return [
            {
                "id":      item.get("team", {}).get("id"),
                "name":    item.get("team", {}).get("name", ""),
                "code":    item.get("team", {}).get("code", ""),
                "country": item.get("team", {}).get("country", ""),
                "founded": item.get("team", {}).get("founded"),
                "venue":   item.get("venue", {}).get("name", ""),
            }
            for item in data
        ]

    def get_team_statistics(self, team: int, league: int, season: int) -> dict:
        """Season-level aggregated statistics for a single team."""
        data = self._get("/teams/statistics", {"team": team, "league": league, "season": season})
        if not data:
            return {}
        d = data[0] if isinstance(data, list) else data
        fixtures = d.get("fixtures", {})
        goals    = d.get("goals", {})
        cards    = d.get("cards", {})
        return {
            "team":            d.get("team", {}).get("name", ""),
            "league":          d.get("league", {}).get("name", ""),
            "season":          d.get("league", {}).get("season"),
            "played_home":     fixtures.get("played", {}).get("home", 0),
            "played_away":     fixtures.get("played", {}).get("away", 0),
            "wins_home":       fixtures.get("wins", {}).get("home", 0),
            "wins_away":       fixtures.get("wins", {}).get("away", 0),
            "draws_home":      fixtures.get("draws", {}).get("home", 0),
            "draws_away":      fixtures.get("draws", {}).get("away", 0),
            "loses_home":      fixtures.get("loses", {}).get("home", 0),
            "loses_away":      fixtures.get("loses", {}).get("away", 0),
            "goals_for_home":  goals.get("for", {}).get("total", {}).get("home", 0),
            "goals_for_away":  goals.get("for", {}).get("total", {}).get("away", 0),
            "goals_against_home": goals.get("against", {}).get("total", {}).get("home", 0),
            "goals_against_away": goals.get("against", {}).get("total", {}).get("away", 0),
            "yellow_cards":    sum(v.get("total", 0) or 0 for v in cards.get("yellow", {}).values()),
            "red_cards":       sum(v.get("total", 0) or 0 for v in cards.get("red", {}).values()),
        }

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def get_players(
        self,
        *,
        team: int | None = None,
        league: int | None = None,
        season: int | None = None,
        player_id: int | None = None,
        search: str | None = None,
    ) -> list[dict]:
        """Season stats for players. Paginated automatically."""
        params: dict = {}
        if team is not None:
            params["team"] = team
        if league is not None:
            params["league"] = league
        if season is not None:
            params["season"] = season
        if player_id is not None:
            params["id"] = player_id
        if search:
            params["search"] = search
        data = self._paginate("/players", params)
        return [self._parse_player(item) for item in data]

    def get_player_seasons(self, player_id: int) -> list[int]:
        """All seasons for which a player has statistics."""
        data = self._get("/players/seasons", {"player": player_id})
        return sorted(int(s) for s in data)

    def get_top_scorers(self, league: int, season: int) -> list[dict]:
        data = self._get("/players/topscorers", {"league": league, "season": season})
        return [self._parse_player(item) for item in data]

    def get_top_assists(self, league: int, season: int) -> list[dict]:
        data = self._get("/players/topassists", {"league": league, "season": season})
        return [self._parse_player(item) for item in data]

    def get_top_yellow_cards(self, league: int, season: int) -> list[dict]:
        data = self._get("/players/topyellowcards", {"league": league, "season": season})
        return [self._parse_player(item) for item in data]

    def get_top_red_cards(self, league: int, season: int) -> list[dict]:
        data = self._get("/players/topredcards", {"league": league, "season": season})
        return [self._parse_player(item) for item in data]

    def _parse_player(self, item: dict) -> dict:
        p    = item.get("player", {})
        stat = (item.get("statistics") or [{}])[0]
        team = stat.get("team", {})
        lg   = stat.get("league", {})
        gls  = stat.get("goals", {})
        drb  = stat.get("dribbles", {})
        pas  = stat.get("passes", {})
        sht  = stat.get("shots", {})
        crd  = stat.get("cards", {})
        sub  = stat.get("substitutes", {})
        return {
            "player_id":  p.get("id"),
            "name":       p.get("name", ""),
            "age":        p.get("age"),
            "nationality":p.get("nationality", ""),
            "team_id":    team.get("id"),
            "team":       team.get("name", ""),
            "league_id":  lg.get("id"),
            "league":     lg.get("name", ""),
            "season":     lg.get("season"),
            "appearances":stat.get("games", {}).get("appearences"),
            "minutes":    stat.get("games", {}).get("minutes"),
            "goals":      gls.get("total"),
            "assists":    gls.get("assists"),
            "shots":      sht.get("total"),
            "shots_on":   sht.get("on"),
            "passes":     pas.get("total"),
            "key_passes": pas.get("key"),
            "dribbles":   drb.get("attempts"),
            "dribbles_success": drb.get("success"),
            "yellow_cards": crd.get("yellow"),
            "red_cards":    crd.get("red"),
            "sub_in":     sub.get("in"),
            "sub_out":    sub.get("out"),
        }

    # ------------------------------------------------------------------
    # Coaches
    # ------------------------------------------------------------------

    def get_coach(self, team: int) -> dict:
        """Current head coach for a team."""
        data = self._get("/coachs", {"team": team})
        if not data:
            return {}
        c = data[0]
        career = c.get("career", [])
        return {
            "id":          c.get("id"),
            "name":        c.get("name", ""),
            "nationality": c.get("nationality", ""),
            "age":         c.get("age"),
            "team":        c.get("team", {}).get("name", ""),
            "career":      [
                {
                    "team":  e.get("team", {}).get("name", ""),
                    "start": e.get("start"),
                    "end":   e.get("end"),
                }
                for e in career
            ],
        }

    def search_coaches(self, search: str) -> list[dict]:
        data = self._get("/coachs", {"search": search})
        return [
            {
                "id":          c.get("id"),
                "name":        c.get("name", ""),
                "nationality": c.get("nationality", ""),
                "team":        c.get("team", {}).get("name", ""),
            }
            for c in data
        ]

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def get_transfers_by_player(self, player_id: int) -> list[dict]:
        """Transfer history for a player."""
        data = self._get("/transfers", {"player": player_id})
        return self._parse_transfers(data)

    def get_transfers_by_team(self, team_id: int) -> list[dict]:
        """Transfer history for a team."""
        data = self._get("/transfers", {"team": team_id})
        return self._parse_transfers(data)

    @staticmethod
    def _parse_transfers(data: list) -> list[dict]:
        out: list[dict] = []
        for item in data:
            player = item.get("player", {})
            for t in item.get("transfers", []):
                out.append({
                    "player_id":   player.get("id"),
                    "player":      player.get("name", ""),
                    "date":        t.get("date", ""),
                    "type":        t.get("type", ""),
                    "from_team":   t.get("teams", {}).get("out", {}).get("name", ""),
                    "to_team":     t.get("teams", {}).get("in", {}).get("name", ""),
                })
        return out

    # ------------------------------------------------------------------
    # Trophies
    # ------------------------------------------------------------------

    def get_trophies_by_player(self, player_id: int) -> list[dict]:
        data = self._get("/trophies", {"player": player_id})
        return self._parse_trophies(data)

    def get_trophies_by_coach(self, coach_id: int) -> list[dict]:
        data = self._get("/trophies", {"coach": coach_id})
        return self._parse_trophies(data)

    @staticmethod
    def _parse_trophies(data: list) -> list[dict]:
        return [
            {
                "league":  t.get("league", ""),
                "country": t.get("country", ""),
                "season":  t.get("season", ""),
                "place":   t.get("place", ""),
            }
            for t in data
        ]

    # ------------------------------------------------------------------
    # Sidelined
    # ------------------------------------------------------------------

    def get_sidelined_by_player(self, player_id: int) -> list[dict]:
        """Long-term absence history for a player."""
        data = self._get("/sidelined", {"player": player_id})
        return self._parse_sidelined(data)

    def get_sidelined_by_coach(self, coach_id: int) -> list[dict]:
        data = self._get("/sidelined", {"coach": coach_id})
        return self._parse_sidelined(data)

    @staticmethod
    def _parse_sidelined(data: list) -> list[dict]:
        return [
            {
                "type":  s.get("type", ""),
                "start": s.get("start", ""),
                "end":   s.get("end", ""),
            }
            for s in data
        ]

    # ------------------------------------------------------------------
    # API-Football Predictions
    # ------------------------------------------------------------------

    def get_prediction(self, fixture_id: int | str) -> dict:
        """
        API-Football's own ML prediction for a fixture.
        Returns a flat dict with win probabilities and recommended bet.
        """
        data = self._get("/predictions", {"fixture": fixture_id})
        if not data:
            return {}
        p = data[0]
        pred    = p.get("predictions", {})
        winner  = pred.get("winner", {})
        percent = pred.get("percent", {})
        teams   = p.get("teams", {})
        return {
            "fixture_id":    str(fixture_id),
            "winner_id":     winner.get("id"),
            "winner_name":   winner.get("name", ""),
            "winner_comment":winner.get("comment", ""),
            "home_pct":      percent.get("home", ""),
            "draw_pct":      percent.get("draws", ""),
            "away_pct":      percent.get("away", ""),
            "advice":        pred.get("advice", ""),
            "home_form":     teams.get("home", {}).get("last_5", {}).get("form", ""),
            "away_form":     teams.get("away", {}).get("last_5", {}).get("form", ""),
            "home_goals_for_avg":     teams.get("home", {}).get("last_5", {}).get("goals", {}).get("for", {}).get("average"),
            "away_goals_for_avg":     teams.get("away", {}).get("last_5", {}).get("goals", {}).get("for", {}).get("average"),
            "home_goals_against_avg": teams.get("home", {}).get("last_5", {}).get("goals", {}).get("against", {}).get("average"),
            "away_goals_against_avg": teams.get("away", {}).get("last_5", {}).get("goals", {}).get("against", {}).get("average"),
        }

    # ------------------------------------------------------------------
    # Live / in-play odds
    # ------------------------------------------------------------------

    def get_live_odds(
        self,
        *,
        fixture_id: int | str | None = None,
        league: int | None = None,
        bet: int | None = None,
    ) -> list[dict]:
        """
        In-play odds for live fixtures.
        At least one of fixture_id, league must be supplied.
        """
        params: dict = {}
        if fixture_id is not None:
            params["fixture"] = fixture_id
        if league is not None:
            params["league"] = league
        if bet is not None:
            params["bet"] = bet
        data = self._get("/odds/live", params or None)
        out: list[dict] = []
        for item in data:
            fid = item.get("fixture", {}).get("id")
            for bk in item.get("odds", []):
                bk_name = bk.get("name", "")
                for b in bk.get("bets", []):
                    out.append({
                        "fixture_id":  str(fid),
                        "bookmaker":   bk_name,
                        "market":      b.get("name", ""),
                        "values":      b.get("values", []),
                    })
        return out

    def get_live_odds_bets(self) -> list[dict]:
        """All available in-play bet type definitions."""
        data = self._get("/odds/live/bets")
        return [{"id": b.get("id"), "name": b.get("name", "")} for b in data]

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def get_standings(self, league: int, season: int) -> list[dict]:
        data = self._get("/standings", {"league": league, "season": season})
        if not data:
            return []
        try:
            rows = data[0]["league"]["standings"][0]
        except (IndexError, KeyError, TypeError):
            return []
        result: list[dict] = []
        for row in rows:
            result.append({
                "team":      row["team"]["name"],
                "position":  int(row["rank"]),
                "points":    int(row["points"]),
                "goal_diff": int(row["goalsDiff"]),
                "wins":      int(row["all"]["win"]),
                "losses":    int(row["all"]["lose"]),
            })
        return result

    # ------------------------------------------------------------------
    # Injuries
    # ------------------------------------------------------------------

    def get_injuries(self, league: int, season: int) -> dict[str, list[dict]]:
        data = self._get("/injuries", {"league": league, "season": season})
        injuries: dict[str, list[dict]] = {}
        for item in data:
            try:
                team     = item["team"]["name"]
                player   = item["player"]["name"]
                position = item["player"].get("position", "Midfielder")
                inj_type = item["injury"].get("type", "Unknown")
                injuries.setdefault(team, []).append(
                    {"player": player, "position": position, "type": inj_type}
                )
            except (KeyError, TypeError):
                continue
        return injuries

    # ------------------------------------------------------------------
    # Referee stats (derived from completed fixtures)
    # ------------------------------------------------------------------

    def get_referee_stats(self, league: int, season: int) -> dict[str, dict]:
        data = self._get(
            "/fixtures", {"league": league, "season": season, "status": "FT"}
        )
        ref_acc: dict[str, dict] = {}
        for item in data:
            ref = item["fixture"].get("referee") or ""
            if not ref:
                continue
            ref_acc.setdefault(ref, {"home_cards": 0, "away_cards": 0, "n": 0})
            ref_acc[ref]["n"] += 1

        stats: dict[str, dict] = {}
        for ref, acc in ref_acc.items():
            n          = max(acc["n"], 1)
            total      = acc["home_cards"] + acc["away_cards"]
            home_bias  = (acc["home_cards"] - acc["away_cards"]) / n
            stats[ref] = {
                "home_bias":  round(home_bias, 3),
                "strictness": round(total / n, 2),
                "games":      n,
            }
        return stats
