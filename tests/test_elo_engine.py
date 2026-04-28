from quant.models.elo_engine import EloEngine


def test_elo_engine_updates():
    engine = EloEngine()

    engine.update_match("A", "B", 2, 0)

    assert engine.get_rating("A") != 1500.0
    assert engine.get_rating("B") != 1500.0
