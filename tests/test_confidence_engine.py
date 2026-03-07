from engine.confidence_engine import ConfidenceEngine


def test_confidence_engine_runs():

    score = ConfidenceEngine().score(
        probability=0.58,
        edge=0.07,
        agreement=0.70,
    )

    assert 0.0 <= score <= 1.0
