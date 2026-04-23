from __future__ import annotations

import logging
import os

import requests

BASE_URL = "https://v3.football.api-sports.io"
_log = logging.getLogger(__name__)


class APIFootballClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY missing")
        self.headers = {"x-apisports-key": self.api_key}

    def _get(self, path: str, params: dict | None = None) -> list:
        url = BASE_URL + path
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "response" not in payload:
            raise RuntimeError("API-Football response missing 'response' field")
        return payload["response"]

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def get_leagues(self) -> list:
        return self._get("/leagues")

    def get_fixtures(self, league, season) -> list[dict]:
        data = self._get("/fixtures", {"league": league, "season": season})
        return [
            {
                "fixture_id": str(item["fixture"]["id"]),
                "home_team": item["teams"]["home"]["name"],
                "away_team": item["teams"]["away"]["name"],
            }
            for item in data
        ]

    def get_completed_matches(self, league, season) -> list[dict]:
        data = self._get(
            "/fixtures", {"league": league, "season": season, "status": "FT"}
        )
        matches = []
        for item in data:
            hg = item.get("goals", {}).get("home")
            ag = item.get("goals", {}).get("away")
            if hg is None or ag is None:
                continue
            matches.append(
                {
                    "fixture_id": str(item["fixture"]["id"]),
                    "home_team": item["teams"]["home"]["name"],
                    "home_team_id": item["teams"]["home"]["id"],
                    "away_team": item["teams"]["away"]["name"],
                    "away_team_id": item["teams"]["away"]["id"],
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "match_date": item["fixture"].get("date", ""),
                    "referee": item["fixture"].get("referee") or "",
                }
            )
        return matches

    def get_upcoming_matches(self, league, season) -> list[dict]:
        data = self._get(
            "/fixtures", {"league": league, "season": season, "status": "NS"}
        )
        return [
            {
                "fixture_id": str(item["fixture"]["id"]),
                "home_team": item["teams"]["home"]["name"],
                "home_team_id": item["teams"]["home"]["id"],
                "away_team": item["teams"]["away"]["name"],
                "away_team_id": item["teams"]["away"]["id"],
                "match_date": item["fixture"].get("date", ""),
                "referee": item["fixture"].get("referee") or "",
            }
            for item in data
        ]

    # ------------------------------------------------------------------
    # Odds
    # ------------------------------------------------------------------

    def get_odds(self, fixture_ids) -> dict:
        odds_map: dict = {}
        for fixture_id in fixture_ids:
            data = self._get("/odds", {"fixture": fixture_id})
            if not data:
                continue
            try:
                bookmakers = data[0].get("bookmakers", [])
                if not bookmakers:
                    continue
                odds: dict = {}
                for bet in bookmakers[0].get("bets", []):
                    if bet.get("name") != "Match Winner":
                        continue
                    for value in bet.get("values", []):
                        label = value.get("value")
                        odd = value.get("odd")
                        if odd in (None, ""):
                            continue
                        if label == "Home":
                            odds["home"] = float(odd)
                        elif label == "Draw":
                            odds["draw"] = float(odd)
                        elif label == "Away":
                            odds["away"] = float(odd)
                if odds:
                    odds_map[str(fixture_id)] = odds
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                _log.warning("Failed to parse odds for fixture %s: %s", fixture_id, exc)
        return odds_map

    def get_prematch_odds(self, fixture_ids) -> dict:
        return self.get_odds(fixture_ids)

    # ------------------------------------------------------------------
    # Fixture statistics (xG / shots)
    # ------------------------------------------------------------------

    def get_fixture_statistics(self, fixture_ids: list) -> dict[str, dict]:
        """
        Returns per-team statistics for each fixture.
        {fixture_id: {home_team: {shots, shots_on_target, ...}, away_team: {...}}}
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
                _log.warning(
                    "Failed to parse stats for fixture %s: %s", fixture_id, exc
                )
        return stats_map

    def get_team_xg_averages(self, league, season) -> dict[str, dict]:
        """
        Derives per-team average xG from completed fixture statistics.
        Returns {team_name: {xg_for, xg_against, shots, shots_on_target}}
        """
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
            hs = stats.get(home, {})
            as_ = stats.get(away, {})

            for team, scored, conceded in (
                (home, hs, as_),
                (away, as_, hs),
            ):
                if team not in team_acc:
                    team_acc[team] = {
                        "xg_for": 0.0,
                        "xg_against": 0.0,
                        "shots": 0.0,
                        "shots_on_target": 0.0,
                        "n": 0,
                    }
                team_acc[team]["xg_for"] += scored.get("xg", 0.0)
                team_acc[team]["xg_against"] += conceded.get("xg", 0.0)
                team_acc[team]["shots"] += scored.get("shots", 0.0)
                team_acc[team]["shots_on_target"] += scored.get("shots_on_target", 0.0)
                team_acc[team]["n"] += 1

        averages: dict[str, dict] = {}
        for team, acc in team_acc.items():
            n = max(acc["n"], 1)
            averages[team] = {
                "xg_for": acc["xg_for"] / n,
                "xg_against": acc["xg_against"] / n,
                "shots": acc["shots"] / n,
                "shots_on_target": acc["shots_on_target"] / n,
            }
        return averages

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def get_standings(self, league, season) -> list[dict]:
        """
        Returns standings list: [{"team": name, "position": int, "points": int,
        "goal_diff": int, "wins": int, "losses": int}]
        """
        data = self._get("/standings", {"league": league, "season": season})
        if not data:
            return []
        try:
            rows = data[0]["league"]["standings"][0]
        except (IndexError, KeyError, TypeError):
            return []
        result: list[dict] = []
        for row in rows:
            result.append(
                {
                    "team": row["team"]["name"],
                    "position": int(row["rank"]),
                    "points": int(row["points"]),
                    "goal_diff": int(row["goalsDiff"]),
                    "wins": int(row["all"]["win"]),
                    "losses": int(row["all"]["lose"]),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Injuries
    # ------------------------------------------------------------------

    def get_injuries(self, league, season) -> dict[str, list[dict]]:
        """
        Returns {team_name: [{"player": str, "position": str, "type": str}]}
        """
        data = self._get("/injuries", {"league": league, "season": season})
        injuries: dict[str, list[dict]] = {}
        for item in data:
            try:
                team = item["team"]["name"]
                player = item["player"]["name"]
                position = item["player"].get("position", "Midfielder")
                inj_type = item["injury"].get("type", "Unknown")
                injuries.setdefault(team, []).append(
                    {"player": player, "position": position, "type": inj_type}
                )
            except (KeyError, TypeError):
                continue
        return injuries

    # ------------------------------------------------------------------
    # Referee stats (built from completed fixtures)
    # ------------------------------------------------------------------

    def get_referee_stats(self, league, season) -> dict[str, dict]:
        """
        Derives referee stats (home bias, strictness) from completed fixtures.
        Returns {referee_name: {home_bias, strictness, games}}
        """
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
            n = max(acc["n"], 1)
            total_cards = acc["home_cards"] + acc["away_cards"]
            home_bias = (acc["home_cards"] - acc["away_cards"]) / n if n else 0.0
            stats[ref] = {
                "home_bias": round(home_bias, 3),
                "strictness": round(total_cards / n, 2),
                "games": n,
            }
        return stats
