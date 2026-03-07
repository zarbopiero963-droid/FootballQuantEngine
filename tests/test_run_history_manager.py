from utils.run_history_manager import RunHistoryManager


def test_run_history_write_and_read():

    manager = RunHistoryManager("data/test_history.json")

    manager.add_run({"bankroll": 10, "roi": 5, "yield": 5, "hit_rate": 50})

    runs = manager.get_runs()

    assert isinstance(runs, list)

    assert len(runs) >= 1
