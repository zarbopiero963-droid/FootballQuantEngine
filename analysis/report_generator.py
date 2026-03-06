import os


class ReportGenerator:

    def __init__(self):

        os.makedirs("outputs", exist_ok=True)

    def _line_svg(self, values, title):

        if not values:
            return f"<h3>{title}</h3><p>No data</p>"

        width = 700
        height = 220
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
            f"style='background:#fff;border:1px solid #ccc'>"
            f"<polyline fill='none' stroke='#2c7be5' stroke-width='2' "
            f"points='{polyline}' />"
            f"</svg>"
        )

    def generate(self, metrics, ranked_df=None):

        html = []
        html.append("<html><head><title>Football Quant Report</title></head><body>")
        html.append("<h1>Football Quant Engine Report</h1>")

        if metrics:
            html.append("<h2>Metrics</h2>")
            html.append("<ul>")
            html.append(f"<li>ROI: {metrics.get('roi', 0):.4f}</li>")
            html.append(f"<li>Yield: {metrics.get('yield', 0):.4f}</li>")
            html.append(f"<li>Hit Rate: {metrics.get('hit_rate', 0):.4f}</li>")
            html.append(f"<li>Brier Score: {metrics.get('brier_score', 0):.4f}</li>")
            html.append(f"<li>Log Loss: {metrics.get('log_loss', 0):.4f}</li>")
            html.append("</ul>")

            html.append(
                self._line_svg(
                    metrics.get("bankroll_history", []),
                    "Bankroll Chart",
                )
            )
            html.append(
                self._line_svg(
                    metrics.get("accuracy_history", []),
                    "Accuracy Chart",
                )
            )

        if ranked_df is not None and not ranked_df.empty:
            html.append("<h2>Ranked Value Bets</h2>")
            html.append("<table border='1' cellpadding='6' cellspacing='0'>")
            html.append("<tr>")
            for col in ranked_df.columns:
                html.append(f"<th>{col}</th>")
            html.append("</tr>")

            for _, row in ranked_df.iterrows():
                html.append("<tr>")
                for col in ranked_df.columns:
                    html.append(f"<td>{row[col]}</td>")
                html.append("</tr>")

            html.append("</table>")

        html.append("</body></html>")

        with open("outputs/advanced_report.html", "w", encoding="utf-8") as f:
            f.write("\n".join(html))
