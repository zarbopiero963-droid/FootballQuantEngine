from __future__ import annotations

import json
import os


class QuantSnapshotManager:

    def __init__(self, path: str = "outputs/ci/quant_snapshot.json"):
        self.path = path

    def save(self, summary: dict) -> str:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        return self.path

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None

        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)
