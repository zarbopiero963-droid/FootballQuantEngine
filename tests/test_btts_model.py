from quant.markets.btts_model import BTTSModel


def test_btts_model_probabilities_sum_to_one():
    model = BTTSModel()
    probs = model.probabilities(1.4, 1.1)

    total = probs["yes"] + probs["no"]

    assert 0.999 <= total <= 1.001
