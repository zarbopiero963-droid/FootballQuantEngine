from engine.model_agreement import ModelAgreement


def test_model_agreement_runs():

    agreement = ModelAgreement().calculate(
        [
            {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
            {"home_win": 0.52, "draw": 0.24, "away_win": 0.24},
        ]
    )

    assert 0.0 <= agreement <= 1.0
