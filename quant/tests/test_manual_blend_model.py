from quant.models.manual_blend_model import ManualBlendModel


def test_manual_blend_model_runs():
    model = ManualBlendModel()

    result = model.combine(
        dc_probs={"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
        elo_diff=55,
        form_diff=0.20,
        xg_diff=0.30,
        market_probs={"home_win": 0.48, "draw": 0.27, "away_win": 0.25},
    )

    total = result["home_win"] + result["draw"] + result["away_win"]

    assert 0.999 <= total <= 1.001
