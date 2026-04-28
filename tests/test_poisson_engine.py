from quant.models.poisson_engine import PoissonEngine


def test_poisson_engine_probabilities():
    engine = PoissonEngine()

    engine.fit(
        [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
            {"home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1},
            {"home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 1},
            {"home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 1},
        ]
    )

    probs = engine.probabilities_1x2("A", "B")

    total = probs["home_win"] + probs["draw"] + probs["away_win"]

    assert 0.999 <= total <= 1.001
