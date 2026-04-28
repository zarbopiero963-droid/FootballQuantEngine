from __future__ import annotations


class SvgChartBuilder:

    def line_chart(self, values, title, color="#2c7be5", width=700, height=220):

        if not values:
            return f"<h3>{title}</h3><p>No data</p>"

        padding = 25
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
            f"viewBox='0 0 {width} {height}' "
            f"style='background:#ffffff;border:1px solid #cccccc;border-radius:8px'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{polyline}' />"
            f"</svg>"
        )
