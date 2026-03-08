from __future__ import annotations

from quant.clv.clv_journal_service import CLVJournalService
from quant.value.value_bet_runner import ValueBetRunner


class CLVRunner:

    def __init__(self):
        self.value_runner = ValueBetRunner()
        self.service = CLVJournalService()

    def run(self, bankroll: float = 1000.0) -> dict:
        bets = self.value_runner.run(bankroll=bankroll)
        saved = self.service.record_bets(bets)
        summary = self.service.performance_summary()
        csv_path = self.service.export_csv()

        return {
            "bets": saved,
            "summary": summary,
            "csv_path": csv_path,
        }
