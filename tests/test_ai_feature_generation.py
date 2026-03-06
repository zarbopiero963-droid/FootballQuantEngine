import pandas as pd

from analysis.ai_feature_generation import AIFeatureGenerator


def test_ai_feature_generation_runs():

    df = pd.DataFrame(
        [
            {
                "team": "TeamA",
                "weighted_form_score": 5.0,
                "goals_for_avg": 1.6,
                "clean_sheet_rate": 0.4,
                "btts_rate": 0.5,
            }
        ]
    )

    result = AIFeatureGenerator().generate(df)

    assert isinstance(result, pd.DataFrame)
    assert "ai_confidence_score" in result.columns
    assert "ai_defensive_index" in result.columns
