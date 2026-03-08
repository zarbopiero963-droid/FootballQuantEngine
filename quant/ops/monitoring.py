from __future__ import annotations

import json
import os
from datetime import datetime


class Monitoring:

    def __init__(self, path: str = "outputs/ops/monitoring_log.json"):
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
            json.dump(rows[-500:], handle, indent=2)

    def log(self, event: str, payload: dict | None = None) -> dict:
        rows = self._load()

        row = {
            "event": str(event),
            "payload": payload or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        rows.append(row)
        self._save(rows)

        return row
