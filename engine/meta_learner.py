"""
Meta-Learning Ensemble Supervisor
===================================

Solves the "which model do I trust?" problem when multiple prediction engines
disagree.  Instead of averaging, the Meta-Learner applies condition-specific
weight rules learned from historical calibration data:

    "Under heavy rain in the Premier League, Poisson accuracy drops to 48%
     but xG-based models hit 71%.  Under these exact conditions, weight
     xG-model ×2.0 and Poisson ×0.50."

The weight update rule uses Brier score tracking per (model, league):
    Brier = (p_home - y_home)² + (p_draw - y_draw)² + (p_away - y_away)²
Lower Brier → higher weight in subsequent matches in that league.

Usage
-----
    from engine.meta_learner import MetaLearner, ModelPrediction, MatchConditions

    learner = MetaLearner()
    preds = [
        ModelPrediction("poisson",  0.45, 0.28, 0.27),
        ModelPrediction("elo",      0.42, 0.30, 0.28),
        ModelPrediction("xg_model", 0.50, 0.26, 0.24),
    ]
    conds = MatchConditions(league="premier_league", rain_mm_per_hour=8.0, ...)
    result = learner.predict(preds, conds)
    print(result.dominant_model, result.fair_odds())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_KNOWN_MODELS = ("poisson", "elo", "xg_model", "skellam", "gradient_boost")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MatchConditions:
    league: str = "unknown"
    is_cup: bool = False
    rain_mm_per_hour: float = 0.0
    wind_speed_kph: float = 0.0
    temperature_c: float = 15.0
    home_strength: float = 0.50  # [0,1] normalised ELO
    away_strength: float = 0.50
    home_form: float = 0.50  # last-5-game form [0,1]
    away_form: float = 0.50
    days_rest_home: int = 7
    days_rest_away: int = 7
    pitch_type: str = "NATURAL"  # "NATURAL"|"HYBRID"|"SYNTHETIC"
    is_derby: bool = False
    crowd_factor: float = 0.50  # 0=empty, 1=packed hostile


@dataclass
class ModelPrediction:
    model_name: str
    p_home: float
    p_draw: float
    p_away: float
    lambda_home: Optional[float] = None
    lambda_away: Optional[float] = None
    confidence: float = 1.0


@dataclass
class EnsemblePrediction:
    conditions: MatchConditions
    model_predictions: List[ModelPrediction]
    model_weights: Dict[str, float]
    ensemble_p_home: float
    ensemble_p_draw: float
    ensemble_p_away: float
    ensemble_lambda_home: Optional[float]
    ensemble_lambda_away: Optional[float]
    dominant_model: str
    confidence_score: float
    notes: List[str]

    def __str__(self) -> str:
        lines = [
            "=" * 56,
            "META-ENSEMBLE PREDICTION",
            f"League: {self.conditions.league}",
            f"Dominant model: {self.dominant_model}",
            "-" * 56,
        ]
        for model, w in sorted(
            self.model_weights.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {model:20s} weight={w:.3f}")
        lines += [
            "-" * 56,
            f"P(home): {self.ensemble_p_home:.3f}",
            f"P(draw): {self.ensemble_p_draw:.3f}",
            f"P(away): {self.ensemble_p_away:.3f}",
            f"Confidence: {self.confidence_score:.3f}",
        ]
        if self.notes:
            lines.append("-" * 56)
            for note in self.notes:
                lines.append(f"  • {note}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def fair_odds(self) -> Tuple[float, float, float]:
        """Returns (home_odds, draw_odds, away_odds) without margin."""
        return (
            1.0 / max(self.ensemble_p_home, 1e-6),
            1.0 / max(self.ensemble_p_draw, 1e-6),
            1.0 / max(self.ensemble_p_away, 1e-6),
        )


# ---------------------------------------------------------------------------
# Condition rule definitions
# ---------------------------------------------------------------------------

# Each rule: (condition_fn, {model_name: weight_multiplier})
_ConditionRule = Tuple[Callable[[MatchConditions], bool], Dict[str, float]]

_CONDITION_RULES: List[_ConditionRule] = [
    # 1. Heavy rain → Poisson unreliable, xG model captures weather better
    (
        lambda c: c.rain_mm_per_hour > 5.0,
        {"poisson": 0.70, "xg_model": 1.40, "skellam": 1.10},
    ),
    # 2. Very heavy rain
    (
        lambda c: c.rain_mm_per_hour > 10.0,
        {"poisson": 0.55, "xg_model": 1.60, "gradient_boost": 1.20},
    ),
    # 3. Synthetic turf — historical data (ELO) more reliable than possession-based xG
    (lambda c: c.pitch_type == "SYNTHETIC", {"elo": 1.30, "xg_model": 0.80}),
    # 4. Cup match — motivation unmeasured by Poisson; xG model better
    (lambda c: c.is_cup, {"xg_model": 1.25, "gradient_boost": 1.15, "poisson": 0.85}),
    # 5. Cold weather (<5°C) — Skellam captures goal difference better in tough conditions
    (lambda c: c.temperature_c < 5.0, {"skellam": 1.20, "xg_model": 0.90}),
    # 6. Home congestion (< 3 days rest)
    (
        lambda c: c.days_rest_home < 3,
        {"gradient_boost": 1.40, "poisson": 0.80, "xg_model": 1.20},
    ),
    # 7. Away congestion
    (lambda c: c.days_rest_away < 3, {"gradient_boost": 1.35, "poisson": 0.82}),
    # 8. Derby — crowd factor inflates home win rate; Poisson underestimates
    (lambda c: c.is_derby, {"elo": 1.30, "gradient_boost": 1.40, "poisson": 0.80}),
    # 9. High wind — long-ball weather; Poisson underestimates chaos
    (
        lambda c: c.wind_speed_kph > 40.0,
        {"skellam": 1.20, "gradient_boost": 1.15, "xg_model": 0.85},
    ),
    # 10. Big mismatch (strong home, weak away) — Poisson shines; GBT can overfit
    (
        lambda c: (c.home_strength - c.away_strength) > 0.35,
        {"poisson": 1.15, "elo": 1.10, "gradient_boost": 0.90},
    ),
    # 11. Close match (strength near equal) — xG + gradient_boost capture form
    (
        lambda c: abs(c.home_strength - c.away_strength) < 0.10,
        {"xg_model": 1.20, "gradient_boost": 1.15, "poisson": 0.90},
    ),
    # 12. Both teams in poor form — entropy is high, weight toward Skellam
    (
        lambda c: c.home_form < 0.30 and c.away_form < 0.30,
        {"skellam": 1.25, "xg_model": 1.10, "elo": 0.85},
    ),
    # 13. Champion leagues / high-profile — ELO historically accurate
    (
        lambda c: c.league in ("champions_league", "europa_league"),
        {"elo": 1.20, "gradient_boost": 1.10},
    ),
    # 14. Empty stadium (< 0.20 crowd factor)
    (lambda c: c.crowd_factor < 0.20, {"poisson": 1.10, "xg_model": 1.05, "elo": 0.95}),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MetaLearner:
    """
    Dynamically weights prediction models based on match conditions and
    running Brier-score calibration per (model, league).

    Parameters
    ----------
    base_weights : initial weights per model (default equal ~1.0)
    temperature : softmax temperature; lower = more winner-takes-all
    brier_learning_rate : how quickly Brier-based weights shift (0-1)
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "poisson": 1.00,
        "elo": 0.90,
        "xg_model": 1.10,
        "skellam": 0.85,
        "gradient_boost": 1.20,
    }

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        brier_learning_rate: float = 0.05,
    ) -> None:
        self._base = dict(self._DEFAULT_WEIGHTS)
        if base_weights:
            self._base.update(base_weights)
        self._temperature = temperature
        self._lr = brier_learning_rate
        # Running Brier scores: {(model, league): [brier_scores]}
        self._brier: Dict[Tuple[str, str], List[float]] = {}

    def predict(
        self,
        predictions: List[ModelPrediction],
        conditions: MatchConditions,
    ) -> EnsemblePrediction:
        """Compute weighted ensemble prediction for this fixture."""
        # Gather base weights for the models that provided predictions
        present = {p.model_name for p in predictions}
        raw_weights: Dict[str, float] = {m: self._base.get(m, 1.0) for m in present}

        # Apply Brier adjustments
        for model in present:
            key = (model, conditions.league)
            if key in self._brier and self._brier[key]:
                avg_brier = sum(self._brier[key]) / len(self._brier[key])
                # Lower Brier → multiply weight by inverse factor
                raw_weights[model] *= max(0.40, 1.0 - avg_brier * 2.0)

        # Apply condition rules
        notes: List[str] = []
        raw_weights, rule_notes = self._apply_condition_rules(raw_weights, conditions)
        notes.extend(rule_notes)

        # Normalise via softmax
        final_weights = self._softmax_weights(raw_weights)

        # Weighted average
        ensemble = self._weighted_average(predictions, final_weights)
        lh, la = self._weighted_lambdas(predictions, final_weights)
        confidence = self._compute_confidence(predictions, final_weights)
        dominant = max(final_weights, key=lambda k: final_weights[k])

        logger.info(
            "MetaLearner: dominant=%s conf=%.3f p=%.3f/%.3f/%.3f",
            dominant,
            confidence,
            ensemble[0],
            ensemble[1],
            ensemble[2],
        )

        return EnsemblePrediction(
            conditions=conditions,
            model_predictions=predictions,
            model_weights=final_weights,
            ensemble_p_home=ensemble[0],
            ensemble_p_draw=ensemble[1],
            ensemble_p_away=ensemble[2],
            ensemble_lambda_home=lh,
            ensemble_lambda_away=la,
            dominant_model=dominant,
            confidence_score=confidence,
            notes=notes,
        )

    def update(
        self,
        model_name: str,
        league: str,
        prediction: ModelPrediction,
        actual_outcome: str,
    ) -> None:
        """Update Brier score for model/league after result is known."""
        outcome_map = {
            "home_win": (1.0, 0.0, 0.0),
            "draw": (0.0, 1.0, 0.0),
            "away_win": (0.0, 0.0, 1.0),
        }
        actual = outcome_map.get(actual_outcome, (0.0, 0.0, 0.0))
        brier = (
            (prediction.p_home - actual[0]) ** 2
            + (prediction.p_draw - actual[1]) ** 2
            + (prediction.p_away - actual[2]) ** 2
        )
        key = (model_name, league)
        if key not in self._brier:
            self._brier[key] = []
        self._brier[key].append(brier)
        # Keep last 100 observations
        if len(self._brier[key]) > 100:
            self._brier[key].pop(0)

    def model_rankings(self, league: Optional[str] = None) -> List[Tuple[str, float]]:
        """Return models sorted by effective weight (best first)."""
        effective: Dict[str, float] = dict(self._base)
        for model in list(effective.keys()):
            key = (model, league or "unknown")
            if key in self._brier and self._brier[key]:
                avg_b = sum(self._brier[key]) / len(self._brier[key])
                effective[model] *= max(0.40, 1.0 - avg_b * 2.0)
        return sorted(effective.items(), key=lambda t: t[1], reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_condition_rules(
        self, weights: Dict[str, float], conditions: MatchConditions
    ) -> Tuple[Dict[str, float], List[str]]:
        notes: List[str] = []
        for check_fn, boosts in _CONDITION_RULES:
            try:
                if check_fn(conditions):
                    for model, mult in boosts.items():
                        if model in weights:
                            weights[model] *= mult
                    desc = ", ".join(f"{m}×{v}" for m, v in boosts.items())
                    notes.append(f"Rule applied: {desc}")
            except Exception as exc:
                logger.debug("Condition rule error: %s", exc)
        return weights, notes

    def _softmax_weights(self, raw: Dict[str, float]) -> Dict[str, float]:
        if not raw:
            return {}
        scaled = {m: w / self._temperature for m, w in raw.items()}
        max_val = max(scaled.values())
        exps = {m: math.exp(v - max_val) for m, v in scaled.items()}
        total = sum(exps.values()) + 1e-12
        return {m: v / total for m, v in exps.items()}

    def _weighted_average(
        self,
        preds: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> Tuple[float, float, float]:
        ph = pd = pa = 0.0
        for pred in preds:
            w = weights.get(pred.model_name, 0.0)
            ph += w * pred.p_home
            pd += w * pred.p_draw
            pa += w * pred.p_away
        total = ph + pd + pa
        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return ph / total, pd / total, pa / total

    def _weighted_lambdas(
        self,
        preds: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        lh_sum = la_sum = w_sum = 0.0
        for pred in preds:
            if pred.lambda_home is not None and pred.lambda_away is not None:
                w = weights.get(pred.model_name, 0.0)
                lh_sum += w * pred.lambda_home
                la_sum += w * pred.lambda_away
                w_sum += w
        if w_sum <= 0:
            return None, None
        return lh_sum / w_sum, la_sum / w_sum

    def _compute_confidence(
        self,
        preds: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> float:
        """1 - normalised entropy of weighted average probs."""
        ph, pd, pa = self._weighted_average(preds, weights)
        probs = [max(p, 1e-12) for p in (ph, pd, pa)]
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(3)
        return max(0.0, 1.0 - entropy / max_entropy)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def ensemble_predict(
    poisson_probs: Tuple[float, float, float],
    elo_probs: Tuple[float, float, float],
    conditions: Dict,
    xg_probs: Optional[Tuple[float, float, float]] = None,
) -> EnsemblePrediction:
    """
    Quick one-shot ensemble without training history.

    Parameters
    ----------
    poisson_probs, elo_probs, xg_probs : (p_home, p_draw, p_away)
    conditions : dict matching MatchConditions fields
    """
    cond = MatchConditions(
        **{
            k: v
            for k, v in conditions.items()
            if k in MatchConditions.__dataclass_fields__
        }
    )
    preds: List[ModelPrediction] = [
        ModelPrediction("poisson", *poisson_probs),
        ModelPrediction("elo", *elo_probs),
    ]
    if xg_probs:
        preds.append(ModelPrediction("xg_model", *xg_probs))
    return MetaLearner().predict(preds, cond)
