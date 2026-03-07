import pandas as pd

from utils.dashboard_metrics_loader import compute_dashboard_metrics


def test_compute_metrics():

    df = pd.DataFrame({"stake": [10, 10, 10], "profit": [5, -10, 3]})

    metrics = compute_dashboard_metrics(df)

    assert "roi" in metrics

    assert "hit_rate" in metrics
