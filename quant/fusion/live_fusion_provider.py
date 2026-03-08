from __future__ import annotations

from quant.fusion.live_fusion_layer import LiveFusionLayer


class LiveFusionProvider:

    def __init__(self, fixtures_provider, odds_provider, understat_provider):
        self.fixtures_provider = fixtures_provider
        self.odds_provider = odds_provider
        self.understat_provider = understat_provider
        self.fusion = LiveFusionLayer()

    def build_fused_fixtures(self, league=None, season=None) -> list[dict]:
        fixtures = self.fixtures_provider.get_upcoming_matches(
            league=league,
            season=season,
        )

        fixture_ids = [str(item.get("fixture_id", "")) for item in fixtures]

        odds_map = self.odds_provider.get_prematch_odds(fixture_ids)
        understat_stats = self.understat_provider.get_team_advanced_stats(
            league=league,
            season=season,
        )

        return self.fusion.fuse(
            fixtures=fixtures,
            odds_map=odds_map,
            understat_stats=understat_stats,
        )
