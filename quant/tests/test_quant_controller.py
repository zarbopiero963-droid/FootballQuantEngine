from quant.providers.sample_clients import SampleAPIFootballClient
from quant.services.quant_controller import QuantController


def test_quant_controller_returns_records():
    controller = QuantController(api_client=SampleAPIFootballClient())

    records = controller.run(league=135, season=2024)

    assert isinstance(records, list)
    assert len(records) > 0
    assert "fixture_id" in records[0]
    assert "decision" in records[0]
