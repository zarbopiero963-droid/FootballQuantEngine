import os

import pytest

if not os.getenv("API_FOOTBALL_KEY"):
    pytest.skip("API_FOOTBALL_KEY missing", allow_module_level=True)

from app.quant_controller import AppQuantController


def test_app_quant_controller_runs():

    controller = AppQuantController()

    results = controller.run_quant_cycle()

    assert isinstance(results, list)
    assert len(results) > 0
