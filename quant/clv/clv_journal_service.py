from __future__ import annotations

from quant.clv.bet_journal import BetJournal
from quant.clv.clv_tracker import CLVTracker
from quant.clv.performance_report import PerformanceReport
from quant.clv.pnl_calculator import PnLCalculator


class CLVJournalService:

    def __init__(self):
        self.journal = BetJournal()
        self.tracker = CLVTracker()
        self.pnl = PnLCalculator()
        self.report_builder = PerformanceReport()

    def record_bets(self, rows: list[dict]) -> list[dict]:
        saved = []

        for row in rows:
            payload = {
                "fixture_id": row.get("fixture_id"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "market": row.get("market"),
                "probability": row.get("probability"),
                "fair_odds": row.get("fair_odds"),
                "bookmaker_odds": row.get("bookmaker_odds"),
                "opening_odds": row.get("opening_odds", row.get("bookmaker_odds")),
                "ev": row.get("ev"),
                "confidence": row.get("confidence"),
                "agreement": row.get("agreement"),
                "decision": row.get("decision"),
                "stake": row.get("suggested_stake", 0.0),
            }
            saved.append(self.journal.add_bet(payload))

        return saved

    def settle_bet(
        self,
        fixture_id: str,
        market: str,
        closing_odds: float,
        result: str,
    ) -> dict | None:
        rows = self.journal.list_bets()

        target = None
        for row in rows:
            if str(row.get("fixture_id")) == str(fixture_id) and str(
                row.get("market")
            ) == str(market):
                target = row
                break

        if not target:
            return None

        bookmaker_odds = float(target.get("bookmaker_odds", 0.0) or 0.0)
        stake = float(target.get("stake", 0.0) or 0.0)

        clv_data = self.tracker.evaluate(
            placed_odds=bookmaker_odds,
            closing_odds=float(closing_odds),
        )

        pnl = self.pnl.settle(
            stake=stake,
            odds=bookmaker_odds,
            result=result,
        )

        updated = self.journal.update_bet(
            fixture_id=fixture_id,
            market=market,
            fields={
                "closing_odds": float(closing_odds),
                "result": str(result).upper(),
                "status": "SETTLED",
                "pnl": pnl,
                "clv_abs": clv_data["clv_abs"],
                "clv_pct": clv_data["clv_pct"],
            },
        )

        return updated

    def performance_summary(self) -> dict:
        return self.report_builder.build(self.journal.list_bets())

    def export_csv(self) -> str:
        return self.journal.export_csv()
