from quant.value.kelly_engine import KellyEngine


def test_kelly_engine_positive_fraction():
    engine = KellyEngine()

    fraction = engine.kelly_fraction(0.55, 2.10)

    assert fraction >= 0.0
