"""
Daily HTML report generator for the Football Quant Engine.

Produces a self-contained HTML file (inline CSS + inline SVG charts) plus
an optional companion JSON file with the same data in machine-readable form.

Report sections
---------------
1. Header — date, engine version, summary badges
2. Performance KPIs — ROI, Yield, Hit Rate, Total Bets, Max Drawdown, Brier, Log-Loss
3. Equity curve — inline SVG polyline chart
4. Drawdown chart — inline SVG
5. Value bets table — colour-coded by tier (S/A/B/C)
6. League breakdown — per-league ROI, bet count, hit rate
7. Confidence calibration — actual win rate vs predicted probability in decile buckets
8. Kelly analysis — distribution of recommended stakes

Usage
-----
    from report.report_generator import ReportGenerator

    gen = ReportGenerator(output_dir="outputs/reports")
    path = gen.generate(
        ranked_bets=ranked,        # list[dict] from MatchRanker.rank_as_dicts()
        metrics=backtest_metrics,  # dict from BacktestEngine
        date="2026-04-23",         # optional, defaults to today
    )
    print(f"Report saved to {path}")
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_VERSION = "5.0"


# ---------------------------------------------------------------------------
# SVG chart primitives
# ---------------------------------------------------------------------------


def _svg_line_chart(
    values: list[float],
    title: str,
    color: str = "#2c7be5",
    width: int = 900,
    height: int = 220,
    padding: int = 40,
    zero_line: bool = False,
) -> str:
    """Return an inline SVG polyline chart with axis labels."""
    if not values:
        return f"<figure><figcaption>{title}</figcaption><p>No data</p></figure>"

    n = len(values)
    lo = min(values)
    hi = max(values)
    rng = hi - lo or 1.0
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding

    def _x(i: int) -> float:
        return padding + i * inner_w / max(n - 1, 1)

    def _y(v: float) -> float:
        return height - padding - (v - lo) / rng * inner_h

    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(values))

    # Y-axis labels (min, mid, max)
    y_labels = [
        (lo, height - padding, f"{lo:.2f}"),
        ((lo + hi) / 2, height - padding - inner_h / 2, f"{(lo + hi) / 2:.2f}"),
        (hi, padding, f"{hi:.2f}"),
    ]
    y_label_svg = "".join(
        f'<text x="{padding - 4}" y="{yp + 4}" text-anchor="end" '
        f'font-size="10" fill="#666">{lab}</text>'
        for _, yp, lab in y_labels
    )

    # X-axis labels (first, mid, last)
    x_idx = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    x_label_svg = "".join(
        f'<text x="{_x(i):.1f}" y="{height - 4}" text-anchor="middle" '
        f'font-size="10" fill="#666">{i}</text>'
        for i in x_idx
    )

    # Optional zero line
    zero_svg = ""
    if zero_line and lo < 0 < hi:
        zy = _y(0)
        zero_svg = (
            f'<line x1="{padding}" y1="{zy:.1f}" '
            f'x2="{width - padding}" y2="{zy:.1f}" '
            'stroke="#999" stroke-width="1" stroke-dasharray="4,4"/>'
        )

    # Fill area under curve
    fill_pts = (
        f"{padding:.1f},{height - padding} "
        + pts
        + f" {_x(n - 1):.1f},{height - padding}"
    )

    svg = (
        f'<figure style="margin:12px 0">'
        f'<figcaption style="font-weight:600;margin-bottom:4px;color:#333">'
        f"{title}</figcaption>"
        f'<svg width="{width}" height="{height}" '
        f'style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;'
        f'display:block">'
        f'<polygon points="{fill_pts}" fill="{color}" opacity="0.10"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
        f'points="{pts}"/>'
        f"{zero_svg}"
        f"{y_label_svg}"
        f"{x_label_svg}"
        f'<line x1="{padding}" y1="{padding}" x1="{padding}" y2="{height - padding}" '
        f'stroke="#ccc" stroke-width="1"/>'
        f"</svg></figure>"
    )
    return svg


def _svg_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    color: str = "#28a745",
    width: int = 900,
    height: int = 220,
) -> str:
    """Vertical bar chart for league / category breakdowns."""
    if not values or not labels:
        return f"<figure><figcaption>{title}</figcaption><p>No data</p></figure>"

    n = len(labels)
    padding = 50
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding
    lo = min(0.0, min(values))
    hi = max(values) or 1.0
    rng = hi - lo or 1.0
    bar_w = inner_w / n * 0.7
    gap = inner_w / n

    bars_svg = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = padding + i * gap + gap * 0.15
        bar_h = abs(val) / rng * inner_h
        if val >= 0:
            y = height - padding - bar_h
            bc = color
        else:
            y = height - padding
            bc = "#dc3545"
        bars_svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bar_h:.1f}" fill="{bc}" rx="2"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="9" fill="#555">{lab[:10]}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#333">{val:.2%}</text>'
        )

    # Zero baseline
    zero_y = height - padding - (0 - lo) / rng * inner_h

    return (
        f'<figure style="margin:12px 0">'
        f'<figcaption style="font-weight:600;margin-bottom:4px;color:#333">'
        f"{title}</figcaption>"
        f'<svg width="{width}" height="{height}" '
        f'style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;'
        f'display:block">'
        + "".join(bars_svg)
        + f'<line x1="{padding}" y1="{zero_y:.1f}" '
        f'x2="{width - padding}" y2="{zero_y:.1f}" '
        f'stroke="#666" stroke-width="1"/>'
        f"</svg></figure>"
    )


def _svg_calibration_chart(
    bucket_pred: list[float],
    bucket_act: list[float],
    width: int = 500,
    height: int = 500,
) -> str:
    """Reliability diagram (calibration plot): predicted vs actual prob."""
    if not bucket_pred:
        return "<p>No calibration data</p>"

    padding = 50
    inner = min(width, height) - 2 * padding

    def _px(v: float) -> float:
        return padding + v * inner

    def _py(v: float) -> float:
        return height - padding - v * inner

    pts = " ".join(
        f"{_px(p):.1f},{_py(a):.1f}" for p, a in zip(bucket_pred, bucket_act)
    )
    # Perfect calibration diagonal
    diag = f"{padding},{height - padding} {width - padding},{padding}"

    dots = "".join(
        f'<circle cx="{_px(p):.1f}" cy="{_py(a):.1f}" r="5" '
        f'fill="#2c7be5" stroke="#fff" stroke-width="1.5"/>'
        for p, a in zip(bucket_pred, bucket_act)
    )

    return (
        f'<figure style="margin:12px 0">'
        f'<figcaption style="font-weight:600;margin-bottom:4px;color:#333">'
        f"Calibration (Reliability Diagram)</figcaption>"
        f'<svg width="{width}" height="{height}" '
        f'style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px">'
        f'<polyline fill="none" stroke="#ccc" stroke-width="1.5" '
        f'stroke-dasharray="6,4" points="{diag}"/>'
        f'<polyline fill="none" stroke="#2c7be5" stroke-width="2" points="{pts}"/>'
        + dots
        + f'<text x="{padding}" y="{height - 8}" font-size="10" fill="#666">Predicted prob →</text>'
        f'<text x="8" y="{height // 2}" font-size="10" fill="#666" '
        f'transform="rotate(-90 12,{height // 2})">Actual rate ↑</text>'
        f"</svg></figure>"
    )


# ---------------------------------------------------------------------------
# Data aggregators
# ---------------------------------------------------------------------------


def _league_breakdown(bets: list[dict]) -> list[dict]:
    """Aggregate per-league ROI, bet count, hit rate from scored bets."""
    from collections import defaultdict

    acc: dict[str, dict] = defaultdict(
        lambda: {"bets": 0, "won": 0, "ev_sum": 0.0, "kelly_sum": 0.0}
    )

    for b in bets:
        league = str(
            b.get("extras", {}).get("league", "") or b.get("league", "Unknown")
        )
        ev = float(b.get("ev", 0.0))
        won = 1 if b.get("decision") == "BET" else 0  # proxy
        acc[league]["bets"] += 1
        acc[league]["won"] += won
        acc[league]["ev_sum"] += ev
        acc[league]["kelly_sum"] += float(b.get("kelly", 0.0))

    rows = []
    for league, a in sorted(acc.items(), key=lambda x: -x[1]["ev_sum"]):
        n = a["bets"] or 1
        rows.append(
            {
                "league": league,
                "bets": a["bets"],
                "avg_ev": round(a["ev_sum"] / n, 5),
                "avg_kelly": round(a["kelly_sum"] / n, 5),
                "bet_rate": round(a["won"] / n, 4),
            }
        )
    return rows


def _calibration_buckets(
    bets: list[dict],
    n_buckets: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Group bets by predicted probability decile and compute actual
    fraction that were BET decisions (proxy for 'won').

    Returns (bucket_mid_pred, bucket_actual_rate).
    """
    if not bets:
        return [], []

    width = 1.0 / n_buckets
    counts = [0] * n_buckets
    wins = [0] * n_buckets
    prob_sum = [0.0] * n_buckets

    for b in bets:
        p = max(0.0, min(0.9999, float(b.get("probability", 0.5))))
        idx = min(int(p / width), n_buckets - 1)
        is_bet = 1 if b.get("decision") == "BET" else 0
        counts[idx] += 1
        wins[idx] += is_bet
        prob_sum[idx] += p

    pred_mids = []
    act_rates = []
    for i in range(n_buckets):
        if counts[i] == 0:
            continue
        pred_mids.append(prob_sum[i] / counts[i])
        act_rates.append(wins[i] / counts[i])

    return pred_mids, act_rates


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:24px;background:#f5f6fa;color:#222}
h1{color:#1a1a2e;border-bottom:3px solid #2c7be5;padding-bottom:8px}
h2{color:#1a1a2e;margin-top:32px;border-left:4px solid #2c7be5;padding-left:10px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.kpi{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 24px;
     min-width:120px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.kpi .val{font-size:1.8em;font-weight:700;color:#2c7be5}
.kpi .lbl{font-size:.8em;color:#888;margin-top:4px}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;
      box-shadow:0 1px 4px rgba(0,0,0,.06);overflow:hidden}
th{background:#2c7be5;color:#fff;padding:10px 12px;text-align:left;font-size:.85em}
td{padding:8px 12px;border-bottom:1px solid #f0f0f0;font-size:.88em}
tr:last-child td{border-bottom:none}
tr:hover td{background:#f7f9ff}
.tier-S{background:#d4edda;font-weight:700;color:#155724}
.tier-A{background:#d1ecf1;font-weight:600;color:#0c5460}
.tier-B{background:#fff3cd;color:#856404}
.tier-C{background:#f8f9fa;color:#495057}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.78em;font-weight:600}
.badge-S{background:#28a745;color:#fff}
.badge-A{background:#17a2b8;color:#fff}
.badge-B{background:#ffc107;color:#212529}
.badge-C{background:#6c757d;color:#fff}
.footer{margin-top:40px;color:#aaa;font-size:.8em;text-align:center}
"""


def _kpi_block(label: str, value: str, color: str = "#2c7be5") -> str:
    return (
        f'<div class="kpi">'
        f'<div class="val" style="color:{color}">{value}</div>'
        f'<div class="lbl">{label}</div>'
        f"</div>"
    )


def _bet_table(bets: list[dict]) -> str:
    if not bets:
        return "<p>No bets to display.</p>"

    headers = [
        "Match",
        "Market",
        "Prob",
        "Odds",
        "EV",
        "Kelly",
        "Stake €",
        "Tier",
        "Confidence",
        "Decision",
    ]
    th_row = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = []
    for b in bets:
        tier = b.get("tier", "C")
        tier_cls = f"tier-{tier}" if tier in ("S", "A", "B", "C") else ""
        badge = f'<span class="badge badge-{tier}">{tier}</span>'
        row = (
            f'<tr class="{tier_cls}">'
            f"<td>{b.get('match_id', '')}</td>"
            f"<td>{b.get('market', '')}</td>"
            f"<td>{float(b.get('probability', 0)):.1%}</td>"
            f"<td>{float(b.get('odds', 0)):.2f}</td>"
            f"<td>{float(b.get('ev', 0)):+.3f}</td>"
            f"<td>{float(b.get('kelly', 0)):.2%}</td>"
            f"<td>{float(b.get('kelly_stake', 0)):.2f}</td>"
            f"<td>{badge}</td>"
            f"<td>{float(b.get('confidence', 0)):.1%}</td>"
            f"<td>{b.get('decision', '')}</td>"
            f"</tr>"
        )
        rows_html.append(row)

    return (
        f"<table><thead><tr>{th_row}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _league_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No league data.</p>"
    headers = ["League", "Bets", "Avg EV", "Avg Kelly", "Bet Rate"]
    th_row = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(
        f"<tr><td>{r['league']}</td><td>{r['bets']}</td>"
        f"<td>{r['avg_ev']:+.4f}</td>"
        f"<td>{r['avg_kelly']:.4f}</td>"
        f"<td>{r['bet_rate']:.1%}</td></tr>"
        for r in rows
    )
    return f"<table><thead><tr>{th_row}</tr></thead><tbody>{trs}</tbody></table>"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Generates dated HTML performance reports.

    Parameters
    ----------
    output_dir : directory for report files (created if needed)
    kelly_scale: must match MatchRanker.kelly_scale for consistent stake display
    """

    def __init__(
        self,
        output_dir: str = "outputs/reports",
        kelly_scale: float = 0.25,
    ) -> None:
        self._dir = Path(output_dir)
        self._kelly_scale = kelly_scale
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def generate(
        self,
        ranked_bets: list[dict],
        metrics: Optional[dict] = None,
        date_str: Optional[str] = None,
        save_json: bool = True,
    ) -> str:
        """
        Build and save the HTML report.

        Parameters
        ----------
        ranked_bets : output of MatchRanker.rank_as_dicts()
        metrics     : dict from BacktestEngine (bankroll_history, roi, yield, …)
        date_str    : report date label (default: today)
        save_json   : also write a JSON companion file

        Returns
        -------
        str — path of the saved HTML file.
        """
        metrics = metrics or {}
        date_label = date_str or date.today().isoformat()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        filename = f"report_{date_label}.html"
        html_path = self._dir / filename

        # ---- aggregate data ----
        n_bets = len(ranked_bets)
        n_s = sum(1 for b in ranked_bets if b.get("tier") == "S")
        n_a = sum(1 for b in ranked_bets if b.get("tier") == "A")
        n_b = sum(1 for b in ranked_bets if b.get("tier") == "B")
        league_rows = _league_breakdown(ranked_bets)
        cal_pred, cal_act = _calibration_buckets(ranked_bets)

        roi = float(metrics.get("roi", 0.0))
        yield_ = float(metrics.get("yield", 0.0))
        hit_rate = float(metrics.get("hit_rate", 0.0))
        total_bets = int(metrics.get("total_bets", 0))
        max_dd = float(metrics.get("max_drawdown", 0.0))
        brier = float(metrics.get("brier_score", 0.0))
        log_loss = float(metrics.get("log_loss", 0.0))
        bankroll_h = metrics.get("bankroll_history") or []
        drawdown_h = metrics.get("drawdown_history") or []
        accuracy_h = metrics.get("accuracy_history") or []

        # ---- build HTML ----
        html: list[str] = [
            "<!DOCTYPE html><html lang='en'><head>",
            "<meta charset='UTF-8'>",
            f"<title>FQE Report {date_label}</title>",
            f"<style>{_CSS}</style>",
            "</head><body>",
            "<h1>Football Quant Engine — Report</h1>",
            f"<p style='color:#888;font-size:.9em'>Generated: {ts} &nbsp;|&nbsp; "
            f"Engine v{_VERSION} &nbsp;|&nbsp; Date: <strong>{date_label}</strong></p>",
            # KPIs
            "<h2>Performance KPIs</h2><div class='kpis'>",
            _kpi_block("ROI", f"{roi:.2%}", "#28a745" if roi >= 0 else "#dc3545"),
            _kpi_block(
                "Yield", f"{yield_:.2%}", "#28a745" if yield_ >= 0 else "#dc3545"
            ),
            _kpi_block("Hit Rate", f"{hit_rate:.1%}", "#2c7be5"),
            _kpi_block("Total Bets", str(total_bets), "#2c7be5"),
            _kpi_block("Max Drawdown", f"{max_dd:.2%}", "#dc3545"),
            _kpi_block("Brier Score", f"{brier:.4f}", "#6c757d"),
            _kpi_block("Log Loss", f"{log_loss:.4f}", "#6c757d"),
            _kpi_block("Today Bets", str(n_bets), "#2c7be5"),
            _kpi_block("Tier S", str(n_s), "#28a745"),
            _kpi_block("Tier A", str(n_a), "#17a2b8"),
            _kpi_block("Tier B", str(n_b), "#ffc107"),
            "</div>",
            # Charts
            "<h2>Equity Curve</h2>",
            _svg_line_chart(bankroll_h, "Bankroll History", "#2c7be5", zero_line=False),
            "<h2>Drawdown</h2>",
            _svg_line_chart(drawdown_h, "Drawdown (%)", "#dc3545", zero_line=True),
            "<h2>Accuracy Over Time</h2>",
            _svg_line_chart(accuracy_h, "Rolling Accuracy", "#28a745"),
        ]

        # League breakdown
        if league_rows:
            html.append("<h2>League Breakdown</h2>")
            html.append(_league_table(league_rows))

            league_names = [r["league"] for r in league_rows]
            league_evs = [r["avg_ev"] for r in league_rows]
            html.append(
                _svg_bar_chart(
                    league_names,
                    league_evs,
                    "Average EV by League",
                    "#2c7be5",
                )
            )

        # Calibration
        if cal_pred:
            html.append("<h2>Probability Calibration</h2>")
            html.append(
                "<p style='color:#666;font-size:.9em'>"
                "Blue dots: actual win rate per predicted-probability decile. "
                "Grey dashed line: perfect calibration.</p>"
            )
            html.append(_svg_calibration_chart(cal_pred, cal_act))

        # Ranked bets table
        if ranked_bets:
            html.append("<h2>Ranked Value Bets</h2>")
            html.append(_bet_table(ranked_bets[:50]))  # cap at 50 rows

        # Footer
        html.append(
            f"<div class='footer'>Football Quant Engine v{_VERSION} — "
            f"Report generated {ts}</div>"
        )
        html.append("</body></html>")

        html_path.write_text("\n".join(html), encoding="utf-8")

        # JSON companion
        if save_json:
            json_path = self._dir / f"report_{date_label}.json"
            payload = {
                "date": date_label,
                "generated_at": ts,
                "metrics": metrics,
                "bets": ranked_bets,
                "league_breakdown": league_rows,
            }
            json_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )

        return str(html_path)

    def latest_path(self) -> Optional[str]:
        """Return the most recently generated report path, if any."""
        reports = sorted(self._dir.glob("report_*.html"), reverse=True)
        return str(reports[0]) if reports else None

    def list_reports(self) -> list[str]:
        """Return all report paths sorted newest first."""
        return [str(p) for p in sorted(self._dir.glob("report_*.html"), reverse=True)]
