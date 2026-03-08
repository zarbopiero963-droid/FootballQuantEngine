from app.market_expansion_controller import AppMarketExpansionController


def test_app_market_expansion_controller_runs():
    rows = AppMarketExpansionController().run()

    assert isinstance(rows, list)
    assert len(rows) > 0
