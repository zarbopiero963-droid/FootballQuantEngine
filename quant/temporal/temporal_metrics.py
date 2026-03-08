from __future__ import annotations

import pandas as pd


class TemporalMetrics:

    def _safe_float(self, value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def build(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {
                "rows": 0,
                "bet_count": 0,
                "total_stake": 0.0,
                "total_pnl": 0.0,
                "roi": 0.0,
                "yield": 0.0,
                "hit_rate": 0.0,
                "avg_ev": 0.0,
                "avg_clv_abs": 0.0,
                "avg_clv_pct": 0.0,
            }

        work = df.copy()

        if "decision" in work.columns:
            bet_mask = work["decision"].astype(str).eq("BET")
            bets = work[bet_mask].copy()
        else:
            bets = work.copy()

        if bets.empty:
            return {
                "rows": len(work.index),
                "bet_count": 0,
                "total_stake": 0.0,
                "total_pnl": 0.0,
                "roi": 0.0,
                "yield": 0.0,
                "hit_rate": 0.0,
                "avg_ev": 0.0,
                "avg_clv_abs": 0.0,
                "avg_clv_pct": 0.0,
            }

        total_stake = sum(
            self._safe_float(v)
            for v in bets.get("stake", pd.Series([0.0] * len(bets.index)))
        )
        total_pnl = sum(
            self._safe_float(v)
            for v in bets.get("pnl", pd.Series([0.0] * len(bets.index)))
        )

        wins = 0
        if "result" in bets.columns:
            wins = int(bets["result"].astype(str).str.upper().eq("WIN").sum())
        elif "pnl" in bets.columns:
            wins = int((bets["pnl"].astype(float) > 0).sum())

        bet_count = len(bets.index)

        roi = (total_pnl / total_stake) if total_stake > 0 else 0.0
        yield_value = roi
        hit_rate = (wins / bet_count) if bet_count > 0 else 0.0

        avg_ev = (
            float(bets["ev"].astype(float).mean())
            if "ev" in bets.columns and bet_count > 0
            else 0.0
        )
        avg_clv_abs = (
            float(bets["clv_abs"].astype(float).mean())
            if "clv_abs" in bets.columns and bet_count > 0
            else 0.0
        )
        avg_clv_pct = (
            float(bets["clv_pct"].astype(float).mean())
            if "clv_pct" in bets.columns and bet_count > 0
            else 0.0
        )

        return {
            "rows": int(len(work.index)),
            "bet_count": int(bet_count),
            "total_stake": round(total_stake, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 6),
            "yield": round(yield_value, 6),
            "hit_rate": round(hit_rate, 6),
            "avg_ev": round(avg_ev, 6),
            "avg_clv_abs": round(avg_clv_abs, 6),
            "avg_clv_pct": round(avg_clv_pct, 6),
        }
