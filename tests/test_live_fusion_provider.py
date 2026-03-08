from quant.fusion.live_fusion_provider import LiveFusionProvider
from quant.providers.sample_clients import (
    SampleAPIFootballClient,
    SampleUnderstatClient,
)


def test_live_fusion_provider_builds_rows():
    provider = LiveFusionProvider(
        fixtures_provider=SampleAPIFootballClient(),
        odds_provider=SampleAPIFootballClient(),
        understat_provider=SampleUnderstatClient(),
    )

    rows = provider.build_fused_fixtures(league="Serie A", season=2024)

    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "fixture_id" in rows[0]
    assert "xg_diff" in rows[0]
