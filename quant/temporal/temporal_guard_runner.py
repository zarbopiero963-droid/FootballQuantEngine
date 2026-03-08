from __future__ import annotations

import pandas as pd

from quant.clv.bet_journal import BetJournal
from quant.temporal.temporal_backtest_guard import TemporalBacktestGuard


class TemporalGuardRunner:

    def __init__(self):
        self.journal = BetJournal()
        self.guard = TemporalBacktestGuard()

    def _load_dataframe(self) -> pd.DataFrame:
        rows = self.journal.list_bets()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if "created_at" in df.columns and "match_date" not in df.columns:
            df["match_date"] = df["created_at"]

        return df

    def run(self) -> dict:
        df = self._load_dataframe()
        return self.guard.run(df)
