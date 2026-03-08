from __future__ import annotations

from quant.fusion.live_fusion_exporter import LiveFusionExporter
from quant.fusion.live_fusion_provider import LiveFusionProvider
from quant.providers.sample_clients import (
    SampleAPIFootballClient,
    SampleUnderstatClient,
)


class LiveFusionRunner:

    def __init__(
        self, fixtures_provider=None, odds_provider=None, understat_provider=None
    ):
        fixtures_provider = fixtures_provider or SampleAPIFootballClient()
        odds_provider = odds_provider or SampleAPIFootballClient()
        understat_provider = understat_provider or SampleUnderstatClient()

        self.provider = LiveFusionProvider(
            fixtures_provider=fixtures_provider,
            odds_provider=odds_provider,
            understat_provider=understat_provider,
        )
        self.exporter = LiveFusionExporter()

    def run(self, league="Serie A", season=2024) -> dict:
        fused_rows = self.provider.build_fused_fixtures(
            league=league,
            season=season,
        )
        export_path = self.exporter.save_json(fused_rows)

        return {
            "rows": fused_rows,
            "count": len(fused_rows),
            "export_path": export_path,
        }
