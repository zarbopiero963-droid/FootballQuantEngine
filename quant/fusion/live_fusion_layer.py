from __future__ import annotations

from quant.fusion.team_mapper import TeamMapper
from quant.fusion.understat_stats_adapter import UnderstatStatsAdapter


class LiveFusionLayer:

    def __init__(self, mapper: TeamMapper | None = None):
        self.mapper = mapper or TeamMapper()
        self.understat_adapter = UnderstatStatsAdapter(self.mapper)

    def fuse(
        self,
        fixtures: list[dict] | None,
        odds_map: dict | None,
        understat_stats: dict | None,
    ) -> list[dict]:
        fixtures = fixtures or []
        odds_map = odds_map or {}
        adapted_understat = self.understat_adapter.adapt(understat_stats)

        fused_rows = []

        for fixture in fixtures:
            fixture_id = str(fixture.get("fixture_id", ""))
            home_team = fixture.get("home_team", "")
            away_team = fixture.get("away_team", "")

            home_stats = self.understat_adapter.get_team_stats(
                adapted_understat, home_team
            )
            away_stats = self.understat_adapter.get_team_stats(
                adapted_understat, away_team
            )

            bookmaker_odds = odds_map.get(
                fixture_id,
                {
                    "home": 0.0,
                    "draw": 0.0,
                    "away": 0.0,
                },
            )

            fused_rows.append(
                {
                    "fixture_id": fixture_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_team_key": self.mapper.map_name(home_team),
                    "away_team_key": self.mapper.map_name(away_team),
                    "bookmaker_odds_home": float(
                        bookmaker_odds.get("home", 0.0) or 0.0
                    ),
                    "bookmaker_odds_draw": float(
                        bookmaker_odds.get("draw", 0.0) or 0.0
                    ),
                    "bookmaker_odds_away": float(
                        bookmaker_odds.get("away", 0.0) or 0.0
                    ),
                    "home_xg_for": home_stats["xg_for"],
                    "home_xg_against": home_stats["xg_against"],
                    "home_xpts": home_stats["xpts"],
                    "home_shots": home_stats["shots"],
                    "home_shots_on_target": home_stats["shots_on_target"],
                    "home_finishing_delta": home_stats["finishing_delta"],
                    "home_defensive_delta": home_stats["defensive_delta"],
                    "home_form_score": home_stats["form_score"],
                    "away_xg_for": away_stats["xg_for"],
                    "away_xg_against": away_stats["xg_against"],
                    "away_xpts": away_stats["xpts"],
                    "away_shots": away_stats["shots"],
                    "away_shots_on_target": away_stats["shots_on_target"],
                    "away_finishing_delta": away_stats["finishing_delta"],
                    "away_defensive_delta": away_stats["defensive_delta"],
                    "away_form_score": away_stats["form_score"],
                    "xg_diff": home_stats["xg_for"] - away_stats["xg_for"],
                    "xga_diff": away_stats["xg_against"] - home_stats["xg_against"],
                    "xpts_diff": home_stats["xpts"] - away_stats["xpts"],
                    "shots_diff": home_stats["shots"] - away_stats["shots"],
                    "shots_on_target_diff": (
                        home_stats["shots_on_target"] - away_stats["shots_on_target"]
                    ),
                    "form_score_diff": home_stats["form_score"]
                    - away_stats["form_score"],
                    "fusion_status": "ok",
                }
            )

        return fused_rows
