"""
LiveFusionRunner — aggregates quant engine output into a single
exportable snapshot.

Replaces the former Understat-dependent implementation. Now uses only
the API-Football provider (or sample data fallback) and the quant engine.
"""

from __future__ import annotations

from quant.fusion.live_fusion_exporter import LiveFusionExporter


class LiveFusionRunner:
    """
    Runs the quant pipeline and exports a fused JSON snapshot of all
    upcoming-match predictions with their signal details.
    """

    def __init__(self, quant_engine=None, exporter: LiveFusionExporter | None = None):
        self.quant_engine = quant_engine
        self.exporter = exporter or LiveFusionExporter()

    def run(self) -> dict:
        """
        Execute the fusion pipeline.

        Returns
        -------
        dict with keys: rows (list[dict]), count (int), export_path (str)
        """
        rows: list[dict] = []

        if self.quant_engine is not None:
            try:
                predictions = self.quant_engine.predict_all()
                rows = [
                    p.to_dict() if hasattr(p, "to_dict") else p for p in predictions
                ]
            except Exception:
                rows = []

        export_path = self.exporter.save_json(rows)

        return {
            "rows": rows,
            "count": len(rows),
            "export_path": export_path,
        }
