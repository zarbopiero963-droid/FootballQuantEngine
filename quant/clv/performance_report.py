from __future__ import annotations


class PerformanceReport:

    def build(self, rows: list[dict]) -> dict:
        if not rows:
            return {
                "total_bets": 0,
                "open_bets": 0,
                "settled_bets": 0,
                "wins": 0,
                "losses": 0,
                "voids": 0,
                "total_stake": 0.0,
                "total_pnl": 0.0,
                "roi": 0.0,
                "yield": 0.0,
                "avg_ev": 0.0,
                "avg_clv_abs": 0.0,
                "avg_clv_pct": 0.0,
            }

        total_bets = len(rows)
        open_bets = sum(
            1 for row in rows if str(row.get("status", "")).upper() == "OPEN"
        )
        settled_rows = [
            row for row in rows if str(row.get("status", "")).upper() == "SETTLED"
        ]

        wins = sum(
            1 for row in settled_rows if str(row.get("result", "")).upper() == "WIN"
        )
        losses = sum(
            1 for row in settled_rows if str(row.get("result", "")).upper() == "LOSE"
        )
        voids = sum(
            1 for row in settled_rows if str(row.get("result", "")).upper() == "VOID"
        )

        total_stake = round(sum(float(row.get("stake", 0.0) or 0.0) for row in rows), 2)
        total_pnl = round(
            sum(float(row.get("pnl", 0.0) or 0.0) for row in settled_rows), 2
        )

        settled_bets = len(settled_rows)
        roi = (total_pnl / total_stake) if total_stake > 0 else 0.0
        yield_value = (total_pnl / total_stake) if total_stake > 0 else 0.0

        ev_values = [float(row.get("ev", 0.0) or 0.0) for row in rows]
        clv_abs_values = [
            float(row.get("clv_abs", 0.0) or 0.0)
            for row in settled_rows
            if row.get("clv_abs") is not None
        ]
        clv_pct_values = [
            float(row.get("clv_pct", 0.0) or 0.0)
            for row in settled_rows
            if row.get("clv_pct") is not None
        ]

        return {
            "total_bets": total_bets,
            "open_bets": open_bets,
            "settled_bets": settled_bets,
            "wins": wins,
            "losses": losses,
            "voids": voids,
            "total_stake": total_stake,
            "total_pnl": total_pnl,
            "roi": round(roi, 6),
            "yield": round(yield_value, 6),
            "avg_ev": round(sum(ev_values) / max(len(ev_values), 1), 6),
            "avg_clv_abs": round(sum(clv_abs_values) / max(len(clv_abs_values), 1), 6),
            "avg_clv_pct": round(sum(clv_pct_values) / max(len(clv_pct_values), 1), 6),
        }
