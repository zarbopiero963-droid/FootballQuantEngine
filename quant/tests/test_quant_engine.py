from quant.providers.sample_clients import SampleAPIFootballClient
from quant.services.quant_engine import QuantEngine


def test_quant_engine_predicts():
    engine = QuantEngine(
        fixtures_provider=SampleAPIFootballClient(),
        odds_provider=SampleAPIFootballClient(),
    )

    engine.fit(league=135, season=2024)
    results = engine.predict(league=135, season=2024)

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].decision in ("BET", "WATCHLIST", "NO_BET")


def test_quant_engine_details_contain_new_signals():
    engine = QuantEngine(
        fixtures_provider=SampleAPIFootballClient(),
        odds_provider=SampleAPIFootballClient(),
    )
    engine.fit(league=135, season=2024)
    results = engine.predict(league=135, season=2024)

    detail = results[0].details
    for key in ("h2h_diff", "momentum_diff", "motivation_diff", "rest_diff", "dc_rho"):
        assert key in detail, f"Missing detail key: {key}"
