import pytest

scipy = pytest.importorskip("scipy", reason="scipy not installed")

from models.poisson_model import PoissonModel


def test_poisson_probabilities_sum_close_to_one():

    model = PoissonModel()

    probs = model.predict(
        home_attack=1.4, home_defense=1.0, away_attack=1.2, away_defense=1.1
    )

    total = probs["home_win"] + probs["draw"] + probs["away_win"]

    assert 0.95 <= total <= 1.05
