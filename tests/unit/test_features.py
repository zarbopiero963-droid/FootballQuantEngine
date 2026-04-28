"""
Unit tests for features/ — elo_features and xg_features.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from features.elo_features import elo_diff
from features.xg_features import extract_xg

# ---------------------------------------------------------------------------
# elo_features.elo_diff
# ---------------------------------------------------------------------------


class TestEloDiff:
    @pytest.fixture
    def elo_df(self):
        return pd.DataFrame(
            {
                "Club": ["Juve", "Inter", "Milan"],
                "Elo": [1800.0, 1750.0, 1720.0],
            }
        )

    def test_positive_diff_home_stronger(self, elo_df) -> None:
        result = elo_diff("Juve", "Inter", elo_df)
        assert result == pytest.approx(50.0)

    def test_negative_diff_away_stronger(self, elo_df) -> None:
        result = elo_diff("Inter", "Juve", elo_df)
        assert result == pytest.approx(-50.0)

    def test_same_team_returns_zero(self, elo_df) -> None:
        assert elo_diff("Juve", "Juve", elo_df) == pytest.approx(0.0)

    def test_unknown_home_team_returns_zero(self, elo_df) -> None:
        assert elo_diff("Unknown", "Juve", elo_df) == 0.0

    def test_unknown_away_team_returns_zero(self, elo_df) -> None:
        assert elo_diff("Juve", "Unknown", elo_df) == 0.0

    def test_empty_df_returns_zero(self) -> None:
        empty = pd.DataFrame({"Club": [], "Elo": []})
        assert elo_diff("Juve", "Inter", empty) == 0.0

    def test_returns_float(self, elo_df) -> None:
        result = elo_diff("Juve", "Milan", elo_df)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# xg_features.extract_xg
# ---------------------------------------------------------------------------


class TestExtractXg:
    def _make_match(self, xg: float, xga: float, result: str = "W") -> dict:
        return {"datetime": "2024-01-01", "xG": xg, "xGA": xga, "result": result}

    def test_basic_shape(self) -> None:
        data = [self._make_match(1.5, 0.8), self._make_match(0.9, 1.2)]
        df = extract_xg(data)
        assert len(df) == 2
        assert set(df.columns) == {"date", "xg", "xga", "result"}

    def test_xg_values_correct(self) -> None:
        df = extract_xg([self._make_match(2.0, 0.5)])
        assert df.iloc[0]["xg"] == pytest.approx(2.0)
        assert df.iloc[0]["xga"] == pytest.approx(0.5)

    def test_missing_xg_defaults_to_zero(self) -> None:
        df = extract_xg([{"datetime": "2024-01-01", "result": "W"}])
        assert df.iloc[0]["xg"] == pytest.approx(0.0)
        assert df.iloc[0]["xga"] == pytest.approx(0.0)

    def test_empty_input_returns_empty_df(self) -> None:
        df = extract_xg([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_returns_dataframe(self) -> None:
        df = extract_xg([self._make_match(1.0, 1.0)])
        assert isinstance(df, pd.DataFrame)

    def test_result_column_preserved(self) -> None:
        df = extract_xg([self._make_match(1.0, 0.5, "L")])
        assert df.iloc[0]["result"] == "L"
