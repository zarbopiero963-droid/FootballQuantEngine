from quant.markets.over_under_model import OverUnderModel


def test_over_under_model_probabilities_sum_to_one():
    model = OverUnderModel()
    probs = model.probabilities(1.5, 1.2, line=2.5)

    total = probs["over"] + probs["under"]

    assert 0.999 <= total <= 1.001
