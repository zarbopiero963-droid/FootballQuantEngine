import pytest

scipy = pytest.importorskip("scipy", reason="scipy not installed")

from models.bivariate_poisson import BivariatePoisson


def test_bivariate_probabilities_sum_to_one():

    model = BivariatePoisson()

    probs = model.predict(lambda_home=1.5, lambda_away=1.1, rho=0.05)

    total = probs["home_win"] + probs["draw"] + probs["away_win"]

    assert 0.999 <= total <= 1.001
