import pandas as pd

from analytics.match_ranking import MatchRanking


def test_match_ranking_runs():

    value_bets = [
        {
            "match_id": "A_vs_B",
            "market": "home",
            "probability": 0.60,
            "odds": 2.0,
        },
        {
            "match_id": "C_vs_D",
            "market": "away",
            "probability": 0.55,
            "odds": 2.5,
        },
    ]

    ranking = MatchRanking().rank(value_bets)

    assert isinstance(ranking, pd.DataFrame)
    assert "edge" in ranking.columns
