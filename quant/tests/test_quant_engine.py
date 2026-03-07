from quant.providers.sample_clients import (
    SampleAPIFootballClient,
    SampleUnderstatClient,
)
from quant.services.quant_engine import QuantEngine


def test_quant_engine_predicts():
    engine = QuantEngine(
        fixtures_provider=SampleAPIFootballClient(),
        odds_provider=SampleAPIFootballClient(),
        advanced_provider=SampleUnderstatClient(),
    )

    engine.fit(league="Serie A", season=2024)
    results = engine.predict(league="Serie A", season=2024)

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].decision in ("BET", "WATCHLIST", "NO_BET")
