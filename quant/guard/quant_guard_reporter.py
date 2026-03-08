from __future__ import annotations

import html
import json
import os
from datetime import datetime


class QuantGuardReporter:

    def __init__(self, output_dir: str = "outputs/ci"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def build_markdown(self, payload: dict) -> str:
        checks = payload.get("checks", [])
        summary = payload.get("summary", {})
        passed = bool(payload.get("passed", False))

        lines = []
        lines.append("# Quant Engine Guard Report")
        lines.append("")
        lines.append(f"- Generated: {self._timestamp()}")
        lines.append(f"- Passed: **{passed}**")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        for key, value in summary.items():
            lines.append(f"- {key}: `{value}`")

        lines.append("")
        lines.append("## Checks")
        lines.append("")
        lines.append("| Check | Passed | Value |")
        lines.append("|---|---:|---|")
        for check in checks:
            lines.append(
                f"| {check.get('name','')} | {check.get('passed', False)} | {check.get('value','')} |"
            )

        return "\n".join(lines)

    def build_html(self, payload: dict) -> str:
        checks = payload.get("checks", [])
        summary = payload.get("summary", {})
        passed = bool(payload.get("passed", False))

        rows = []
        for check in checks:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(check.get('name', '')))}</td>"
                f"<td>{html.escape(str(check.get('passed', False)))}</td>"
                f"<td>{html.escape(str(check.get('value', '')))}</td>"
                "</tr>"
            )

        summary_rows = []
        for key, value in summary.items():
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(str(key))}</td>"
                f"<td>{html.escape(str(value))}</td>"
                "</tr>"
            )

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Quant Engine Guard Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
th {{ background: #f4f4f4; }}
.good {{ color: green; font-weight: bold; }}
.bad {{ color: red; font-weight: bold; }}
</style>
</head>
<body>
<h1>Quant Engine Guard Report</h1>
<p><strong>Generated:</strong> {html.escape(self._timestamp())}</p>
<p><strong>Passed:</strong> <span class="{'good' if passed else 'bad'}">{html.escape(str(passed))}</span></p>

<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{''.join(summary_rows)}
</table>

<h2>Checks</h2>
<table>
<tr><th>Check</th><th>Passed</th><th>Value</th></tr>
{''.join(rows)}
</table>
</body>
</html>"""

    def save_bundle(self, payload: dict) -> dict:
        json_path = os.path.join(self.output_dir, "quant_guard_report.json")
        md_path = os.path.join(self.output_dir, "quant_guard_report.md")
        html_path = os.path.join(self.output_dir, "quant_guard_report.html")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(self.build_markdown(payload))

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(self.build_html(payload))

        return {
            "json": json_path,
            "markdown": md_path,
            "html": html_path,
        }
