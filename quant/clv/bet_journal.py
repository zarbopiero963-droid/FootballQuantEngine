from __future__ import annotations

import csv
import json
import os
from datetime import datetime


class BetJournal:

    def __init__(self, path: str = "outputs/clv/bet_journal.json"):
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

    def add_bet(self, payload: dict) -> dict:
        rows = self._load()

        row = dict(payload)
        row.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
        row.setdefault("status", "OPEN")
        row.setdefault("closing_odds", None)
        row.setdefault("result", None)
        row.setdefault("pnl", None)
        row.setdefault("clv_abs", None)
        row.setdefault("clv_pct", None)

        rows.append(row)
        self._save(rows)
        return row

    def list_bets(self) -> list[dict]:
        return self._load()

    def update_bet(self, fixture_id: str, market: str, fields: dict) -> dict | None:
        rows = self._load()
        updated = None

        for row in rows:
            if str(row.get("fixture_id")) == str(fixture_id) and str(
                row.get("market")
            ) == str(market):
                row.update(fields)
                updated = row
                break

        self._save(rows)
        return updated

    def export_csv(self, path: str = "outputs/clv/bet_journal.csv") -> str:
        rows = self._load()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["empty"])
            return path

        fieldnames = sorted({key for row in rows for key in row.keys()})

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return path
