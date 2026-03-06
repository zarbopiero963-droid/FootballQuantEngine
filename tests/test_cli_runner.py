from unittest.mock import patch

from engine.cli_runner import CliRunner


@patch("engine.cli_runner.JobRunner")
def test_cli_runner_class_exists(mock_job_runner):

    runner = CliRunner()

    assert isinstance(runner, CliRunner)
    mock_job_runner.assert_called_once()
