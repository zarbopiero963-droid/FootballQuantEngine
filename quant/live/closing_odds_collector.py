from __future__ import annotations

import json
import os
from datetime import datetime


class ClosingOddsCollector:

    def __init__(self, path: str = "outputs/ops/closing_odds.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump([], handle, indent=2)

    def _load(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, rows: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

    def collect(self, fixture_id: str, market: str, closing_odds: float) -> dict:
        rows = self._load()

        row = {
            "fixture_id": str(fixture_id),
            "market": str(market),
            "closing_odds": float(closing_odds),
            "collected_at": datetime.utcnow().isoformat() + "Z",
        }

        rows.append(row)
        self._save(rows)

        return row

    def list_rows(self) -> list[dict]:
        return self._load()
