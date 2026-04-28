from __future__ import annotations

import os


class ChartGenerator:

    def __init__(self):

        os.makedirs("outputs", exist_ok=True)

    def _line_svg(self, values, title, color="#2c7be5", width=900, height=260):

        if not values:
            return f"<h3>{title}</h3><p>No data</p>"

        padding = 30
        min_v = min(values)
        max_v = max(values)

        if max_v == min_v:
            max_v += 1

        points = []

        for index, value in enumerate(values):
            x = padding + index * (width - 2 * padding) / max(len(values) - 1, 1)
            y = (
                height
                - padding
                - ((value - min_v) / (max_v - min_v)) * (height - 2 * padding)
            )
            points.append(f"{x},{y}")

        polyline = " ".join(points)

        return (
            f"<h3>{title}</h3>"
            f"<svg width='{width}' height='{height}' "
            f"style='background:#fff;border:1px solid #ccc;border-radius:8px'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{polyline}' />"
            f"</svg>"
        )

    def generate_all(self, metrics):

        html = []

        html.append(
            self._line_svg(
                metrics.get("bankroll_history", []),
                "Bankroll / Equity Curve",
                "#2c7be5",
            )
        )

        html.append(
            self._line_svg(
                metrics.get("accuracy_history", []),
                "Accuracy Curve",
                "#28a745",
            )
        )

        drawdown_series = metrics.get("drawdown_history", [])
        html.append(
            self._line_svg(
                drawdown_series,
                "Drawdown Curve",
                "#dc3545",
            )
        )

        path = "outputs/charts_report.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        return path
