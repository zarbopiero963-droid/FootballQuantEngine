"""
Multi-Market Engine
===================
Single entry-point that generates predictions for ALL markets for a fixture:

  1X2         — Home / Draw / Away
  OU 1.5      — Over/Under 1.5 goals
  OU 2.5      — Over/Under 2.5 goals  (main market)
  OU 3.5      — Over/Under 3.5 goals
  GG / NG     — Both Teams To Score (Yes / No)
  HT OU 0.5   — Over/Under 0.5 goals first half
  HT OU 1.5   — Over/Under 1.5 goals first half
  SH OU 0.5   — Over/Under 0.5 goals second half
  SH OU 1.5   — Over/Under 1.5 goals second half
  Corners 8.5 — Over/Under corners
  Corners 9.5
  Corners 10.5
  Correct Score — Top 12 most likely exact scores

Usage
-----
    from quant.markets.multi_market_engine import MultiMarketEngine

    engine = MultiMarketEngine()
    engine.fit(completed_matches)          # list[dict] from DB / API
    results = engine.predict_all(upcoming) # list[dict] upcoming fixtures + odds
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from quant.models.corners_model import CornersModel
from quant.models.correct_score_model import CorrectScoreModel
from quant.models.halftime_model import HalftimeModel
from quant.models.poisson_engine import PoissonEngine

logger = logging.getLogger(__name__)

# Minimum model edge to include a selection in the output
_MIN_EDGE = 0.0


class MultiMarketEngine:

    def __init__(self):
        self.poisson = PoissonEngine(max_goals=8)
        self.corners = CornersModel(max_corners=20)
        self.halftime = HalftimeModel(max_goals=6)
        self.correct_score = CorrectScoreModel(max_goals=8, top_n=12)
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, completed_matches: list[dict]) -> None:
        """
        completed_matches: list of dicts with at minimum:
            home_team, away_team, home_goals, away_goals
        Optional (enriches halftime / corners models):
            ht_home, ht_away, home_corners, away_corners
        """
        if not completed_matches:
            logger.warning("MultiMarketEngine.fit: no completed matches provided")
            return

        self.poisson.fit(completed_matches)
        self.corners.fit(completed_matches)
        self.halftime.fit(completed_matches)
        self._fitted = True
        logger.info("MultiMarketEngine fitted on %d matches", len(completed_matches))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_all(self, upcoming_matches: list[dict], odds_map: dict | None = None) -> list[dict]:
        """
        Generate predictions for all markets for every upcoming match.

        Args:
            upcoming_matches: list of fixture dicts (fixture_id, home_team, away_team)
            odds_map: {fixture_id: {'1x2': {...}, 'ou25': {...}, ...}} from APIFootballClient

        Returns: list of market-row dicts ready for DB / UI display.
        """
        if not self._fitted:
            logger.warning("MultiMarketEngine not fitted — call fit() first")
            return []

        odds_map = odds_map or {}
        all_rows: list[dict] = []

        for fixture in upcoming_matches:
            fid = str(fixture.get("fixture_id", ""))
            home = fixture.get("home_team", fixture.get("home", ""))
            away = fixture.get("away_team", fixture.get("away", ""))

            if not home or not away:
                continue

            fixture_odds = odds_map.get(fid, {})

            try:
                rows = self._predict_fixture(fid, home, away, fixture_odds, fixture)
                all_rows.extend(rows)
            except Exception as exc:
                logger.warning("MultiMarketEngine: prediction failed for %s vs %s: %s", home, away, exc)

        return all_rows

    # ------------------------------------------------------------------
    # Per-fixture prediction
    # ------------------------------------------------------------------

    def _predict_fixture(
        self,
        fixture_id: str,
        home: str,
        away: str,
        odds: dict,
        fixture_meta: dict,
    ) -> list[dict]:
        rows: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()

        lam_h, lam_a = self.poisson.expected_goals(home, away)
        score_matrix = self.poisson.score_matrix(home, away)

        # ---- 1X2 ----
        probs_1x2 = self.poisson.probabilities_1x2(home, away)
        odds_1x2  = odds.get("1x2", {})
        for sel, prob_key in [("home", "home_win"), ("draw", "draw"), ("away", "away_win")]:
            p = probs_1x2.get(prob_key, probs_1x2.get(sel, 0.0))
            rows.append(self._row(fixture_id, home, away, "1x2", sel, p,
                                  odds_1x2.get(sel), now, fixture_meta))

        # ---- Over/Under goals ----
        ou_model = self.poisson.probabilities_ou_btts(home, away)

        # OU 2.5 (from existing engine)
        odds_ou25 = odds.get("ou25", {})
        rows.append(self._row(fixture_id, home, away, "ou25", "over",  ou_model["over_25"],  odds_ou25.get("over"),  now, fixture_meta))
        rows.append(self._row(fixture_id, home, away, "ou25", "under", ou_model["under_25"], odds_ou25.get("under"), now, fixture_meta))

        # OU 1.5 and 3.5 via OverUnderModel
        for line_key, line_val in [("ou15", 1.5), ("ou35", 3.5)]:
            p_over, p_under = self._ou_from_matrix(score_matrix, line_val)
            odds_ou = odds.get(line_key, {})
            rows.append(self._row(fixture_id, home, away, line_key, "over",  p_over,  odds_ou.get("over"),  now, fixture_meta))
            rows.append(self._row(fixture_id, home, away, line_key, "under", p_under, odds_ou.get("under"), now, fixture_meta))

        # ---- BTTS (GG / NG) ----
        odds_btts = odds.get("btts", {})
        rows.append(self._row(fixture_id, home, away, "btts", "yes", ou_model["btts_yes"], odds_btts.get("yes"), now, fixture_meta))
        rows.append(self._row(fixture_id, home, away, "btts", "no",  ou_model["btts_no"],  odds_btts.get("no"),  now, fixture_meta))

        # ---- Halftime O/U ----
        ht_probs = self.halftime.all_lines(lam_h, lam_a)
        for market_key, probs in ht_probs.items():
            mkt_odds = odds.get(market_key, {})
            rows.append(self._row(fixture_id, home, away, market_key, "over",  probs["over"],  mkt_odds.get("over"),  now, fixture_meta))
            rows.append(self._row(fixture_id, home, away, market_key, "under", probs["under"], mkt_odds.get("under"), now, fixture_meta))

        # ---- Corners ----
        corner_probs = self.corners.all_lines(home, away)
        for market_key, probs in corner_probs.items():
            line_str = market_key.split("_")[-1]
            mkt_odds = odds.get(f"corners_{line_str}", {})
            rows.append(self._row(fixture_id, home, away, market_key, "over",  probs["over"],  mkt_odds.get("over"),  now, fixture_meta))
            rows.append(self._row(fixture_id, home, away, market_key, "under", probs["under"], mkt_odds.get("under"), now, fixture_meta))

        # ---- Correct Score (top 12) ----
        cs_list = self.correct_score.probabilities(score_matrix)
        cs_odds = odds.get("cs", {})
        for entry in cs_list:
            score_str = entry["score"]
            rows.append(self._row(
                fixture_id, home, away, "cs", score_str,
                entry["probability"],
                cs_odds.get(score_str),
                now, fixture_meta,
            ))

        return rows

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    @staticmethod
    def _row(
        fixture_id: str,
        home: str,
        away: str,
        market: str,
        selection: str,
        probability: float,
        market_odds: float | None,
        created_at: str,
        meta: dict,
    ) -> dict:
        fair_odds = round(1.0 / probability, 3) if probability > 0 else 999.0
        edge = round((probability - (1.0 / market_odds)) if market_odds and market_odds > 1.0 else 0.0, 4)
        ev   = round(probability * market_odds - 1.0 if market_odds and market_odds > 1.0 else 0.0, 4)
        b    = (market_odds - 1.0) if market_odds and market_odds > 1.0 else 0.0
        q    = 1.0 - probability
        kelly = round(((b * probability - q) / b) if b > 0 else 0.0, 4)
        kelly = max(0.0, kelly)

        return {
            "fixture_id":   fixture_id,
            "home":         home,
            "away":         away,
            "match_date":   meta.get("match_date", ""),
            "league":       meta.get("league", ""),
            "market":       market,
            "selection":    selection,
            "probability":  round(probability, 4),
            "fair_odds":    fair_odds,
            "odds_market":  market_odds,
            "edge":         edge,
            "ev":           ev,
            "kelly":        kelly,
            "created_at":   created_at,
        }

    # ------------------------------------------------------------------
    # Helper: O/U from pre-computed score matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _ou_from_matrix(matrix: list[list[float]], line: float) -> tuple[float, float]:
        over = under = 0.0
        for hg, row in enumerate(matrix):
            for ag, p in enumerate(row):
                if hg + ag > line:
                    over += p
                else:
                    under += p
        total = over + under or 1.0
        return over / total, under / total

    # ------------------------------------------------------------------
    # DB persistence helper
    # ------------------------------------------------------------------

    def save_predictions(self, rows: list[dict], db_name: str) -> int:
        """
        Upsert all prediction rows into multi_market_predictions.
        Returns number of rows written.
        """
        import sqlite3
        if not rows:
            return 0

        conn = sqlite3.connect(db_name)
        conn.execute("PRAGMA journal_mode=WAL")

        conn.executemany(
            """
            INSERT INTO multi_market_predictions(
                fixture_id, market, selection, probability,
                odds_model, odds_market, edge, ev, kelly, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    r["fixture_id"], r["market"], r["selection"],
                    r["probability"], r["fair_odds"], r.get("odds_market"),
                    r["edge"], r["ev"], r["kelly"], r["created_at"],
                )
                for r in rows
            ],
        )
        conn.commit()
        conn.close()
        return len(rows)
