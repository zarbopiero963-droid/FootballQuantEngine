from __future__ import annotations

import csv
import json
import os


class TemporalReporter:

    def __init__(self, output_dir: str = "outputs/temporal"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_json(
        self, payload: dict, filename: str = "temporal_guard_report.json"
    ) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def save_csv(
        self, windows: list[dict], filename: str = "temporal_windows.csv"
    ) -> str:
        path = os.path.join(self.output_dir, filename)

        if not windows:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["empty"])
            return path

        fieldnames = sorted({key for row in windows for key in row.keys()})

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(windows)

        return path

    def save_html(
        self, payload: dict, filename: str = "temporal_guard_report.html"
    ) -> str:
        path = os.path.join(self.output_dir, filename)

        windows = payload.get("windows", [])
        guard = payload.get("guard", {})
        checks = guard.get("checks", [])
        metrics = guard.get("metrics", {})

        rows = []
        for row in windows:
            rows.append(
                "<tr>"
                f"<td>{row.get('window_id','')}</td>"
                f"<td>{row.get('train_start','')}</td>"
                f"<td>{row.get('train_end','')}</td>"
                f"<td>{row.get('test_start','')}</td>"
                f"<td>{row.get('test_end','')}</td>"
                f"<td>{row.get('bet_count',0)}</td>"
                f"<td>{row.get('roi',0)}</td>"
                f"<td>{row.get('yield',0)}</td>"
                f"<td>{row.get('hit_rate',0)}</td>"
                f"<td>{row.get('avg_ev',0)}</td>"
                f"<td>{row.get('avg_clv_abs',0)}</td>"
                "</tr>"
            )

        check_rows = []
        for item in checks:
            check_rows.append(
                "<tr>"
                f"<td>{item.get('name','')}</td>"
                f"<td>{item.get('passed',False)}</td>"
                f"<td>{item.get('value','')}</td>"
                "</tr>"
            )

        metric_rows = []
        for key, value in metrics.items():
            metric_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Temporal Backtest Guard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Temporal Backtest Guard</h1>

<h2>Aggregate Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{''.join(metric_rows)}
</table>

<h2>Guard Checks</h2>
<table>
<tr><th>Check</th><th>Passed</th><th>Value</th></tr>
{''.join(check_rows)}
</table>

<h2>Rolling Windows</h2>
<table>
<tr>
<th>Window</th>
<th>Train Start</th>
<th>Train End</th>
<th>Test Start</th>
<th>Test End</th>
<th>Bet Count</th>
<th>ROI</th>
<th>Yield</th>
<th>Hit Rate</th>
<th>Avg EV</th>
<th>Avg CLV Abs</th>
</tr>
{''.join(rows)}
</table>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)

        return path
