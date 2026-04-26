"""
Dynamic cross-league strength normalization.

Uses three complementary approaches:

1. Static UEFA-style coefficients (immediate baseline, no data required)
2. ELO-based dynamic adjustment (fitted from cross-league historical data)
3. Goal-rate calibration (scales attack/defense rates across competitions)

The key insight is that team ratings computed from matches within League A
are not directly comparable with those from League B — a 1500 ELO in Serie A
represents a different absolute quality than 1500 ELO in Ligue 1. This module
provides conversion factors so that all team ratings live on the same scale.

Usage
-----
    norm = LeagueStrengthNormalizer()
    norm.fit(cross_league_matches)   # optional: refines static coefficients

    # Scale a team's attack strength to a universal scale
    adj_attack = norm.normalize_attack(raw_attack, league="Serie A")

    # Convert an ELO rating from one league to another
    elo_equiv = norm.transfer_elo(1520, from_league="Bundesliga",
                                  to_league="Premier League")
"""

from __future__ import annotations

from collections import defaultdict

# ---------------------------------------------------------------------------
# Static baseline coefficients (UEFA club competition performance, 5-year)
# Higher = stronger league, Premier League = 1.000 reference
# ---------------------------------------------------------------------------

_STATIC_COEFF: dict[str, float] = {
    "Premier League": 1.000,
    "La Liga": 0.970,
    "Bundesliga": 0.950,
    "Serie A": 0.940,
    "Ligue 1": 0.890,
    "Eredivisie": 0.800,
    "Primeira Liga": 0.790,
    "Liga NOS": 0.790,
    "Super Lig": 0.760,
    "Belgian Pro League": 0.750,
    "Scottish Premiership": 0.700,
    "Jupiler Pro League": 0.750,
    "Bundesliga 2": 0.820,
    "Serie B": 0.780,
    "Championship": 0.800,
    "EFL Championship": 0.800,
    "Segunda Division": 0.810,
    "Ligue 2": 0.760,
    "Russian Premier League": 0.740,
    "Ukrainian Premier League": 0.700,
    "MLS": 0.680,
}

_DEFAULT_COEFF: float = 0.750  # fallback for unknown leagues

# Goal inflation factors: how many goals per match each league averages
# relative to Premier League (which sets the scale = 1.000)
_GOAL_INFLATION: dict[str, float] = {
    "Premier League": 1.000,
    "Bundesliga": 1.060,  # historically highest-scoring major league
    "Serie A": 0.920,
    "La Liga": 0.960,
    "Ligue 1": 0.940,
    "Eredivisie": 1.020,
    "Primeira Liga": 0.900,
    "Liga NOS": 0.900,
}

_DEFAULT_INFLATION: float = 0.950


# ---------------------------------------------------------------------------
# ELO transfer model
# ---------------------------------------------------------------------------


class _EloTransferModel:
    """
    Estimates cross-league ELO transfer functions from historical inter-league
    matches (e.g., Champions League / Europa League group stages).

    Models: ELO_universal = ELO_local × coefficient + offset

    When no cross-league data is available, falls back to the static
    coefficient table.
    """

    def __init__(self) -> None:
        # {league: (scale, offset)} fitted parameters
        self._params: dict[str, tuple[float, float]] = {}

    def fit(self, cross_league_matches: list[dict]) -> "_EloTransferModel":
        """
        Estimate per-league scale factors from cross-league encounter data.

        Each match dict must contain:
          home_team, away_team, home_league, away_league,
          home_elo, away_elo, home_goals, away_goals.

        The method uses a simple iterative approach:
        1. Compute expected outcome from raw ELOs.
        2. Compare with actual outcome.
        3. Adjust league-level scale to minimise prediction error.
        """
        if not cross_league_matches:
            return self

        # Accumulate per-league ELO prediction errors
        errors: dict[str, list[float]] = defaultdict(list)

        for m in cross_league_matches:
            hl = str(m.get("home_league", ""))
            al = str(m.get("away_league", ""))
            if hl == al:
                continue  # same league, not useful for cross-league calibration

            elo_h = float(m.get("home_elo", 1500.0))
            elo_a = float(m.get("away_elo", 1500.0))
            hg = int(m.get("home_goals", 0))
            ag = int(m.get("away_goals", 0))

            # Adjust each league's ELO by its static coefficient
            c_h = _STATIC_COEFF.get(hl, _DEFAULT_COEFF)
            c_a = _STATIC_COEFF.get(al, _DEFAULT_COEFF)

            adjusted_h = elo_h * c_h
            adjusted_a = elo_a * c_a

            exp_h = 1.0 / (1.0 + 10.0 ** ((adjusted_a - adjusted_h) / 400.0))
            act_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0

            errors[hl].append(act_h - exp_h)
            errors[al].append((1.0 - act_h) - (1.0 - exp_h))

        # Derive correction factors: leagues that consistently under/over-predict
        for league, errs in errors.items():
            if len(errs) < 5:
                continue
            mean_err = sum(errs) / len(errs)
            # Convert prediction error to a multiplicative coefficient adjustment
            # A positive mean error means the league is stronger than its ELO suggests
            correction = 1.0 + mean_err * 0.5  # damped correction
            base = _STATIC_COEFF.get(league, _DEFAULT_COEFF)
            self._params[league] = (base * correction, 0.0)

        return self

    def scale(self, league: str) -> float:
        """Return the fitted (or static fallback) coefficient for a league."""
        if league in self._params:
            return self._params[league][0]
        return _STATIC_COEFF.get(league, _DEFAULT_COEFF)


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------


class LeagueStrengthNormalizer:
    """
    Normalises team strength metrics across leagues to a universal scale.

    All output values are expressed relative to the Premier League = 1.000.

    Parameters
    ----------
    reference_league : str
        The league whose coefficient is fixed at 1.000 (default: Premier League).
    """

    def __init__(self, reference_league: str = "Premier League") -> None:
        self._ref = reference_league
        self._elo_model = _EloTransferModel()
        self._ref_coeff = _STATIC_COEFF.get(reference_league, 1.000)

    def fit(self, cross_league_matches: list[dict]) -> "LeagueStrengthNormalizer":
        """
        Refine coefficients from cross-league match history.

        See _EloTransferModel.fit() for the expected dict format.
        """
        self._elo_model.fit(cross_league_matches)
        return self

    # ------------------------------------------------------------------
    # Coefficient accessors
    # ------------------------------------------------------------------

    def coefficient(self, league: str) -> float:
        """Universal strength coefficient for a league (reference = 1.000)."""
        raw = self._elo_model.scale(league)
        ref = self._elo_model.scale(self._ref) or self._ref_coeff
        return round(raw / ref, 5)

    def goal_inflation(self, league: str) -> float:
        """
        Goal-rate inflation factor for a league vs reference.
        Used to adjust xG / attack rates when comparing across competitions.
        """
        raw_inf = _GOAL_INFLATION.get(league, _DEFAULT_INFLATION)
        ref_inf = _GOAL_INFLATION.get(self._ref, 1.000)
        return round(raw_inf / ref_inf, 5)

    # ------------------------------------------------------------------
    # Normalisation functions
    # ------------------------------------------------------------------

    def normalize_attack(self, raw_attack: float, league: str) -> float:
        """
        Scale a team's attack rate from league-local to universal scale.

        Args
        ----
        raw_attack : attack rate (e.g. goals/match or normalised strength)
        league     : the league from which raw_attack was computed

        Returns
        -------
        Attack rate on the universal (reference-league) scale.
        """
        coeff = self.coefficient(league)
        goal_adj = self.goal_inflation(league)
        return round(raw_attack * coeff / goal_adj, 5)

    def normalize_defense(self, raw_defense: float, league: str) -> float:
        """
        Scale a team's defense rate from league-local to universal scale.

        A weaker league's defense rate is penalised (inflated) because the
        same goals-conceded figure means less vs weaker opponents.
        """
        coeff = self.coefficient(league)
        goal_adj = self.goal_inflation(league)
        return round(raw_defense / coeff * goal_adj, 5)

    def transfer_elo(
        self,
        elo: float,
        from_league: str,
        to_league: str = "Premier League",
    ) -> float:
        """
        Convert a team's ELO rating from one league to another league's scale.

        Useful when two teams from different leagues meet (e.g., UCL).

        The conversion preserves the team's *relative* quality:
          ELO_to = ELO_from × (coefficient_to / coefficient_from)
        """
        c_from = self.coefficient(from_league) or 1.0
        c_to = self.coefficient(to_league) or 1.0
        return round(elo * c_to / c_from, 1)

    def normalize_strength(self, league: str, value: float) -> float:
        """
        Simple strength normalisation (backwards-compatible scalar function).
        Multiplies `value` by the league's universal coefficient.
        """
        return round(value * self.coefficient(league), 5)

    # ------------------------------------------------------------------
    # League rankings
    # ------------------------------------------------------------------

    def rankings(self) -> list[dict]:
        """
        Return all known leagues ranked by universal coefficient (descending).
        """
        rows = []
        for league in sorted(
            set(list(_STATIC_COEFF.keys()) + list(_GOAL_INFLATION.keys()))
        ):
            rows.append(
                {
                    "league": league,
                    "coefficient": self.coefficient(league),
                    "goal_inflation": self.goal_inflation(league),
                }
            )
        rows.sort(key=lambda r: r["coefficient"], reverse=True)
        return rows

    def expected_goals_cross_league(
        self,
        home_lambda: float,
        home_league: str,
        away_lambda: float,
        away_league: str,
        neutral_venue: bool = True,
    ) -> tuple[float, float]:
        """
        Compute cross-league adjusted expected goals for a hypothetical match.

        Scales each team's expected goals to the universal scale before
        computing the head-to-head lambdas.

        Returns
        -------
        (lambda_home_adj, lambda_away_adj) on the universal scale.
        """
        lh_adj = self.normalize_attack(home_lambda, home_league)
        la_adj = self.normalize_attack(away_lambda, away_league)

        if not neutral_venue:
            # Apply a home-advantage correction (average across leagues)
            lh_adj *= 1.20

        return round(lh_adj, 4), round(la_adj, 4)


# ---------------------------------------------------------------------------
# Module-level convenience functions  (referenced in the Part 4 spec)
# ---------------------------------------------------------------------------

_default_normalizer = LeagueStrengthNormalizer()

LEAGUE_STRENGTH = {
    league: _default_normalizer.coefficient(league) for league in _STATIC_COEFF
}


def normalize_strength(league: str, value: float) -> float:
    """
    Scale `value` by the league's universal strength coefficient.

    Backward-compatible with the simple Part-4 reference API.
    """
    return _default_normalizer.normalize_strength(league, value)
