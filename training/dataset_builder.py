"""
Real rolling-window feature engineering for football match prediction.

For each completed fixture the builder computes features from ALL
information available strictly before kick-off (no look-ahead). Features
are computed in chronological order by replaying match history through
live state objects.

Feature set produced
--------------------
elo_diff              ELO rating gap (home - away, home-advantage adjusted)
form_home             Home team win-rate over last FORM_WINDOW matches
form_away             Away team win-rate over last FORM_WINDOW matches
form_diff             form_home - form_away
att_strength_home     Home team attack strength (home goals / league avg)
def_strength_home     Home team defense strength (conceded / league avg away goals)
att_strength_away     Away team attack strength (away goals / league avg away)
def_strength_away     Away team defense strength (conceded / league avg home goals)
lambda_home           Expected home goals (Dixon-Coles style)
lambda_away           Expected away goals
h2h_home_win_rate     Home team win-rate in past H2H meetings
h2h_draw_rate         Draw rate in past H2H meetings
h2h_n                 Number of H2H meetings
goals_scored_home     Rolling avg goals scored at home
goals_conceded_home   Rolling avg goals conceded at home
goals_scored_away     Rolling avg goals scored away
goals_conceded_away   Rolling avg goals conceded away
home_win_rate_season  Season win-rate (all venues)
away_win_rate_season  Season win-rate (all venues)
league_mu_home        League avg home goals at this point in season
league_mu_away        League avg away goals
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from analysis.betfair_leagues import is_betfair_league
from database.db_manager import connect

_FORM_WINDOW = 5
_GOAL_WINDOW = 8
_ELO_BASE = 1500.0
_ELO_K = 20.0
_ELO_HOME = 55.0


# ---------------------------------------------------------------------------
# Rolling state helpers (pure Python, no numpy required)
# ---------------------------------------------------------------------------


class _RollingWindow:
    def __init__(self, maxlen: int) -> None:
        self._buf: list[float] = []
        self._maxlen = maxlen

    def push(self, v: float) -> None:
        self._buf.append(v)
        if len(self._buf) > self._maxlen:
            self._buf.pop(0)

    def mean(self) -> float | None:
        return sum(self._buf) / len(self._buf) if self._buf else None

    def __len__(self) -> int:
        return len(self._buf)


class _EloState:
    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}

    def _r(self, team: str) -> float:
        return self._ratings.get(team, _ELO_BASE)

    def diff(self, home: str, away: str) -> float:
        return (self._r(home) + _ELO_HOME) - self._r(away)

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        rh = self._r(home) + _ELO_HOME
        ra = self._r(away)
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
        exp_a = 1.0 - exp_h
        act_h, act_a = (1.0, 0.0) if hg > ag else (0.0, 1.0) if ag > hg else (0.5, 0.5)
        self._ratings[home] = self._r(home) + _ELO_K * (act_h - exp_h)
        self._ratings[away] = self._r(away) + _ELO_K * (act_a - exp_a)


class _TeamState:
    def __init__(self) -> None:
        self.form_all = _RollingWindow(_FORM_WINDOW)
        self.goals_h = _RollingWindow(_GOAL_WINDOW)
        self.conc_h = _RollingWindow(_GOAL_WINDOW)
        self.goals_a = _RollingWindow(_GOAL_WINDOW)
        self.conc_a = _RollingWindow(_GOAL_WINDOW)
        self.season_w = 0
        self.season_n = 0

    def push_home(self, hg: int, ag: int) -> None:
        self.form_all.push(1.0 if hg > ag else 0.5 if hg == ag else 0.0)
        if hg > ag:
            self.season_w += 1
        self.goals_h.push(float(hg))
        self.conc_h.push(float(ag))
        self.season_n += 1

    def push_away(self, hg: int, ag: int) -> None:
        self.form_all.push(1.0 if ag > hg else 0.5 if ag == hg else 0.0)
        if ag > hg:
            self.season_w += 1
        self.goals_a.push(float(ag))
        self.conc_a.push(float(hg))
        self.season_n += 1

    def form(self) -> float:
        return self.form_all.mean() or 0.5

    def win_rate(self) -> float:
        return self.season_w / self.season_n if self.season_n else 0.5


class _H2HState:
    def __init__(self) -> None:
        self._rec: dict[frozenset, list[tuple]] = defaultdict(list)

    def stats(self, home: str, away: str) -> dict[str, float]:
        key = frozenset({home, away})
        recs = self._rec[key]
        if not recs:
            return {"h2h_home_win_rate": 0.5, "h2h_draw_rate": 0.33, "h2h_n": 0.0}
        wins = sum(1 for h, _, hg, ag in recs if h == home and hg > ag)
        wins += sum(1 for h, a, hg, ag in recs if a == home and ag > hg)
        draws = sum(1 for _, _, hg, ag in recs if hg == ag)
        n = len(recs)
        return {
            "h2h_home_win_rate": wins / n,
            "h2h_draw_rate": draws / n,
            "h2h_n": float(n),
        }

    def push(self, home: str, away: str, hg: int, ag: int) -> None:
        self._rec[frozenset({home, away})].append((home, away, hg, ag))


class _LeagueAvg:
    def __init__(self) -> None:
        self._h: list[float] = []
        self._a: list[float] = []

    def push(self, hg: float, ag: float) -> None:
        self._h.append(hg)
        self._a.append(ag)

    def mu_home(self) -> float:
        return sum(self._h) / len(self._h) if self._h else 1.35

    def mu_away(self) -> float:
        return sum(self._a) / len(self._a) if self._a else 1.10


# ---------------------------------------------------------------------------
# Feature extraction (pre-match snapshot)
# ---------------------------------------------------------------------------


def _extract(
    home: str,
    away: str,
    elo: _EloState,
    teams: dict[str, _TeamState],
    h2h: _H2HState,
    league: _LeagueAvg,
) -> dict[str, float]:
    ts_h = teams[home]
    ts_a = teams[away]

    mu_h = league.mu_home()
    mu_a = league.mu_away()

    gh_h = ts_h.goals_h.mean()  # home team goals scored at home
    gc_h = ts_h.conc_h.mean()  # home team goals conceded at home
    gh_a = ts_a.goals_a.mean()  # away team goals scored away
    gc_a = ts_a.conc_a.mean()  # away team goals conceded away

    att_h = (gh_h / mu_h) if (gh_h is not None and mu_h > 0) else 1.0
    def_h = (gc_h / mu_a) if (gc_h is not None and mu_a > 0) else 1.0
    att_a = (gh_a / mu_a) if (gh_a is not None and mu_a > 0) else 1.0
    def_a = (gc_a / mu_h) if (gc_a is not None and mu_h > 0) else 1.0

    lam_h = max(mu_h * att_h * def_a, 0.10)
    lam_a = max(mu_a * att_a * def_h, 0.10)

    feats: dict[str, float] = {
        "elo_diff": elo.diff(home, away),
        "form_home": ts_h.form(),
        "form_away": ts_a.form(),
        "form_diff": ts_h.form() - ts_a.form(),
        "att_strength_home": att_h,
        "def_strength_home": def_h,
        "att_strength_away": att_a,
        "def_strength_away": def_a,
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "goals_scored_home": gh_h if gh_h is not None else mu_h,
        "goals_conceded_home": gc_h if gc_h is not None else mu_a,
        "goals_scored_away": gh_a if gh_a is not None else mu_a,
        "goals_conceded_away": gc_a if gc_a is not None else mu_h,
        "home_win_rate_season": ts_h.win_rate(),
        "away_win_rate_season": ts_a.win_rate(),
        "league_mu_home": mu_h,
        "league_mu_away": mu_a,
    }
    feats.update(h2h.stats(home, away))
    return feats


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """
    Builds a labelled feature matrix from completed fixtures in the DB.

    Each row represents a single match with pre-match features (no look-ahead)
    and binary outcome targets.
    """

    FEATURE_COLS: list[str] = [
        "elo_diff",
        "form_home",
        "form_away",
        "form_diff",
        "att_strength_home",
        "def_strength_home",
        "att_strength_away",
        "def_strength_away",
        "lambda_home",
        "lambda_away",
        "goals_scored_home",
        "goals_conceded_home",
        "goals_scored_away",
        "goals_conceded_away",
        "home_win_rate_season",
        "away_win_rate_season",
        "league_mu_home",
        "league_mu_away",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_n",
    ]

    def load_fixtures(self) -> pd.DataFrame:
        conn = connect()
        df = pd.read_sql_query(
            """
            SELECT
                fixture_id, league, season, home, away,
                match_date, home_goals, away_goals, status
            FROM fixtures
            WHERE status = 'FT'
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY match_date ASC
            """,
            conn,
        )
        conn.close()
        return df

    def build_training_dataset(self, league_filter: bool = True) -> pd.DataFrame:
        """
        Build the full feature matrix + targets in one chronological pass.

        Parameters
        ----------
        league_filter : bool
            Restrict to Betfair-listed leagues (default True).

        Returns
        -------
        pd.DataFrame
            Columns: fixture_id, league, match_date, home, away,
            home_goals, away_goals, target_home_win, target_draw,
            target_away_win, <feature_cols>.
            Empty DataFrame when no completed fixtures found.
        """
        raw = self.load_fixtures()
        if raw.empty:
            return pd.DataFrame()

        if league_filter:
            raw = raw[raw["league"].apply(is_betfair_league)].copy()
            if raw.empty:
                return pd.DataFrame()

        raw = raw.sort_values("match_date").reset_index(drop=True)

        elo = _EloState()
        h2h = _H2HState()
        lg = _LeagueAvg()
        teams: dict[str, _TeamState] = defaultdict(_TeamState)

        rows: list[dict] = []

        for _, row in raw.iterrows():
            home = str(row["home"])
            away = str(row["away"])
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])

            # Pre-match features (current state, before updating)
            feats = _extract(home, away, elo, teams, h2h, lg)

            record: dict = {
                "fixture_id": row["fixture_id"],
                "league": row["league"],
                "match_date": row["match_date"],
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "target_home_win": int(hg > ag),
                "target_draw": int(hg == ag),
                "target_away_win": int(hg < ag),
            }
            record.update(feats)
            rows.append(record)

            # Update state AFTER extracting features (strict no look-ahead)
            elo.update(home, away, hg, ag)
            teams[home].push_home(hg, ag)
            teams[away].push_away(hg, ag)
            h2h.push(home, away, hg, ag)
            lg.push(float(hg), float(ag))

        return pd.DataFrame(rows)

    def feature_columns(self) -> list[str]:
        return list(self.FEATURE_COLS)
