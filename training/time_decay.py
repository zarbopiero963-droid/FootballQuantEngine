"""
Exponential time-decay weighting for ML training rows.

Rationale
---------
Recent form is more predictive than historical form.  A team's stats from
2 years ago say little about their current quality.  Time-decay weights
let gradient-boosted models (XGBoost, LightGBM) and logistic regression
focus on recent data by assigning higher sample weights to newer matches.

Decay formula
-------------
  w(t) = exp(-λ × Δdays)

where:
  Δdays = (reference_date - match_date).days  (how old the match is)
  λ = ln(2) / half_life_days                   (decay rate)

A match at age = half_life_days has weight 0.5.
A match at age = 0 (today) has weight 1.0.

Default half-life: 90 days.  This means a match played 3 months ago
carries 50% the weight of a match played today, and a match from a year
ago carries ~5% weight.

Usage
-----
    from training.time_decay import TimeDecayWeighter

    weighter = TimeDecayWeighter(half_life_days=90)
    weights = weighter.compute(match_dates, reference_date=today)

    # In sklearn:
    clf.fit(X, y, sample_weight=weights)

    # With a list of match dicts:
    rows, weights = weighter.filter_and_weight(rows, date_col="match_date")
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple, Union

_DEFAULT_HALF_LIFE = 90  # days


class TimeDecayWeighter:
    """
    Computes exponential time-decay sample weights.

    Parameters
    ----------
    half_life_days : age (in days) at which a sample's weight is 0.5.
                     Default: 90 (3 months).
    min_weight     : floor weight applied to very old samples (default 0.01).
    """

    def __init__(
        self,
        half_life_days: float = _DEFAULT_HALF_LIFE,
        min_weight: float = 0.01,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days
        self.min_weight = max(0.0, min_weight)
        self._lambda = math.log(2.0) / half_life_days

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def weight(self, age_days: float) -> float:
        """Return the weight for a single match of the given age in days."""
        raw = math.exp(-self._lambda * max(0.0, age_days))
        return max(self.min_weight, raw)

    def compute(
        self,
        match_dates: List[Union[str, date, datetime]],
        reference_date: Optional[Union[str, date, datetime]] = None,
    ) -> List[float]:
        """
        Compute weights for a list of match dates.

        Parameters
        ----------
        match_dates    : list of date strings ("YYYY-MM-DD"), date, or datetime objects
        reference_date : the "today" anchor (default: current UTC date)

        Returns
        -------
        list of floats, one weight per match (same order as match_dates)
        """
        ref = _to_date(reference_date) if reference_date is not None else _today()
        weights = []
        for md in match_dates:
            match_date_obj = _to_date(md)
            age = max(0, (ref - match_date_obj).days)
            weights.append(self.weight(age))
        return weights

    def filter_and_weight(
        self,
        rows: List[dict],
        date_col: str = "match_date",
        reference_date: Optional[Union[str, date, datetime]] = None,
        max_age_days: Optional[float] = None,
    ) -> Tuple[List[dict], List[float]]:
        """
        Filter rows to at most max_age_days old and return (rows, weights).

        Parameters
        ----------
        rows          : list of match dicts (each must contain date_col)
        date_col      : key in each dict holding the match date string
        reference_date: anchor date (default: UTC today)
        max_age_days  : discard matches older than this (default: no limit)

        Returns
        -------
        (filtered_rows, weights) — same length; weights in [min_weight, 1.0]
        """
        ref = _to_date(reference_date) if reference_date is not None else _today()
        filtered = []
        weights = []

        for row in rows:
            raw_date = row.get(date_col)
            if raw_date is None:
                filtered.append(row)
                weights.append(1.0)
                continue

            try:
                match_date_obj = _to_date(raw_date)
            except Exception:  # noqa: BLE001
                filtered.append(row)
                weights.append(1.0)
                continue

            age = max(0, (ref - match_date_obj).days)

            if max_age_days is not None and age > max_age_days:
                continue

            filtered.append(row)
            weights.append(self.weight(age))

        return filtered, weights

    # ------------------------------------------------------------------
    # Convenience: normalise weights to mean=1 (preserves gradient scale)
    # ------------------------------------------------------------------

    @staticmethod
    def normalise(weights: List[float]) -> List[float]:
        """Rescale weights so their mean equals 1.0."""
        if not weights:
            return []
        mean = sum(weights) / len(weights)
        if mean <= 0:
            return weights
        return [w / mean for w in weights]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> str:
        ages = [0, 30, 60, 90, 180, 365]
        parts = [f"  age={a:4d}d → weight={self.weight(a):.3f}" for a in ages]
        return (
            f"TimeDecayWeighter(half_life={self.half_life_days}d, "
            f"λ={self._lambda:.5f}, min_weight={self.min_weight})\n" + "\n".join(parts)
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def apply_time_decay(
    rows: List[dict],
    date_col: str = "match_date",
    half_life_days: float = _DEFAULT_HALF_LIFE,
    reference_date: Optional[Union[str, date, datetime]] = None,
    max_age_days: Optional[float] = None,
    normalise: bool = True,
) -> Tuple[List[dict], List[float]]:
    """
    One-shot convenience: filter rows and return time-decay sample weights.

    Parameters
    ----------
    rows           : list of match dicts
    date_col       : dict key containing the match date string
    half_life_days : age at which weight = 0.5 (default 90 days)
    reference_date : anchor date (default: UTC today)
    max_age_days   : drop rows older than this (default: no limit)
    normalise      : rescale weights to mean=1 (default True)

    Returns
    -------
    (filtered_rows, weights)
    """
    weighter = TimeDecayWeighter(half_life_days=half_life_days)
    filtered, weights = weighter.filter_and_weight(
        rows,
        date_col=date_col,
        reference_date=reference_date,
        max_age_days=max_age_days,
    )
    if normalise:
        weights = TimeDecayWeighter.normalise(weights)
    return filtered, weights


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _to_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    # ISO string: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS..."
    return date.fromisoformat(str(d)[:10])


def _today() -> date:
    return datetime.now(timezone.utc).date()
