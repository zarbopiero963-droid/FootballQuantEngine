import pandas as pd

from training.local_automl import LocalAutoML


def test_local_automl_runs():

    df = pd.DataFrame(
        [
            {"target_home_win": 1},
            {"target_home_win": 0},
        ]
    )

    result = LocalAutoML().run(df)

    assert "status" in result
    assert "available_models" in result
