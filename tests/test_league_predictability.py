import pandas as pd

from analysis.league_predictability import LeaguePredictability


def test_league_predictability_runs():

    df = pd.DataFrame(
        [
            {
                "league": "Serie A",
                "home_goals": 2,
                "away_goals": 1,
            },
            {
                "league": "Serie A",
                "home_goals": 1,
                "away_goals": 1,
            },
            {
                "league": "Premier League",
                "home_goals": 0,
                "away_goals": 2,
            },
        ]
    )

    analyzer = LeaguePredictability()

    result = analyzer.score(df)

    assert isinstance(result, pd.DataFrame)
    assert "league" in result.columns
    assert "predictability_score" in result.columns
