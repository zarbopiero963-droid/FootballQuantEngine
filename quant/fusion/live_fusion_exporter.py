from __future__ import annotations

import json
import os


class LiveFusionExporter:

    def __init__(self, output_dir: str = "outputs/fusion"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_json(
        self, fused_rows: list[dict], filename: str = "live_fusion_snapshot.json"
    ) -> str:
        path = os.path.join(self.output_dir, filename)

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fused_rows, handle, indent=2)

        return path
