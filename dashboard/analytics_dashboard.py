"""
Analytics Dashboard — PySide6 multi-tab widget.

Tabs
----
1. CLV Analysis      — closing-line value distribution + per-bet table
2. Market Inefficiency — inefficiency type breakdown + league ranking
3. League Predictability — RPS / Brier / CBI per league table + calibration chart
4. Feature Importance  — MI-ranked bar chart of top features

Design
------
- Each tab is a self-contained QWidget with its own refresh slot.
- All charts are rendered as inline SVG embedded in QTextBrowser widgets
  (no matplotlib required — pure Qt + hand-drawn SVG).
- Data flows in via load_data(bets, predictions, history_records).
- Thread-safe: data loading always runs in the main thread.

Usage
-----
    from dashboard.analytics_dashboard import AnalyticsDashboard

    win = AnalyticsDashboard(controller=app_controller)
    win.load_data(bets=ranked_bets, predictions=predictions, history=history)
    win.show()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SVG chart helpers (no external dependencies)
# ---------------------------------------------------------------------------

_SVG_W = 560
_SVG_H = 280
_PAD = 50


def _svg_open(w: int = _SVG_W, h: int = _SVG_H) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'style="background:#FAFAFA;border-radius:6px;">'
    )


def _svg_close() -> str:
    return "</svg>"


def _svg_axes(
    w: int,
    h: int,
    pad: int,
    x_labels: List[str],
    y_min: float,
    y_max: float,
    n_y: int = 5,
) -> str:
    parts = []
    # Frame
    parts.append(
        f'<rect x="{pad}" y="{pad // 2}" width="{w - 2 * pad}" height="{h - pad - pad // 2}" '
        f'fill="none" stroke="#CCCCCC" stroke-width="1"/>'
    )
    # Y-axis grid lines + labels
    plot_h = h - pad - pad // 2
    for i in range(n_y + 1):
        y = pad // 2 + plot_h * (1 - i / n_y)
        val = y_min + (y_max - y_min) * i / n_y
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" y2="{y:.1f}" '
            f'stroke="#EEEEEE" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        lbl = f"{val:.1f}" if abs(val) >= 0.1 else f"{val:.3f}"
        parts.append(
            f'<text x="{pad - 4}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="9" fill="#888">{lbl}</text>'
        )
    # X-axis labels
    n_x = len(x_labels)
    plot_w = w - 2 * pad
    for i, lbl in enumerate(x_labels):
        x = pad + plot_w * (i + 0.5) / max(n_x, 1)
        parts.append(
            f'<text x="{x:.1f}" y="{h - 4}" text-anchor="middle" '
            f'font-size="9" fill="#888">{lbl[:12]}</text>'
        )
    return "".join(parts)


def _data_to_xy(
    values: List[float],
    y_min: float,
    y_max: float,
    w: int = _SVG_W,
    h: int = _SVG_H,
    pad: int = _PAD,
) -> List[tuple]:
    n = len(values)
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2
    y_range = y_max - y_min or 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + plot_w * (i + 0.5) / n
        y_frac = (v - y_min) / y_range
        y = pad // 2 + plot_h * (1 - y_frac)
        pts.append((x, y))
    return pts


def _svg_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    color: str = "#4A90D9",
    neg_color: str = "#E74C3C",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    if not values:
        return f"{_svg_open(w, h)}<text x='50%' y='50%' text-anchor='middle'>No data</text>{_svg_close()}"

    y_min = min(0.0, min(values)) * 1.1
    y_max = max(0.0, max(values)) * 1.1 or 0.1
    pad = _PAD
    plot_h = h - pad - pad // 2
    plot_w = w - 2 * pad
    y_range = y_max - y_min or 1.0
    zero_y = pad // 2 + plot_h * (1 - (0 - y_min) / y_range)
    bar_w = max(4, plot_w // max(len(values), 1) - 2)

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )
    parts.append(_svg_axes(w, h, pad, labels, y_min, y_max))

    # Zero line
    parts.append(
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{w - pad}" y2="{zero_y:.1f}" '
        f'stroke="#AAAAAA" stroke-width="1"/>'
    )

    for i, (lbl, val) in enumerate(zip(labels, values)):
        cx = pad + plot_w * (i + 0.5) / len(values)
        bar_x = cx - bar_w / 2
        y_val = pad // 2 + plot_h * (1 - (val - y_min) / y_range)
        bar_top = min(y_val, zero_y)
        bar_ht = abs(y_val - zero_y)
        fill = color if val >= 0 else neg_color
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{bar_top:.1f}" width="{bar_w}" height="{max(1.0, bar_ht):.1f}" '
            f'fill="{fill}" opacity="0.85" rx="2"/>'
        )

    parts.append(_svg_close())
    return "".join(parts)


def _svg_calibration(
    p_model: List[float],
    p_actual: List[float],
    ns: List[int],
    title: str = "Calibration",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    pad = _PAD
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )
    parts.append(_svg_axes(w, h, pad, [f"{i / 10:.1f}" for i in range(11)], 0.0, 1.0))

    # Perfect calibration diagonal
    x0 = pad
    y0 = pad // 2 + plot_h
    x1 = w - pad
    y1 = pad // 2
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        f'stroke="#AAAAAA" stroke-width="1" stroke-dasharray="6,3"/>'
    )

    # Data points
    max_n = max(ns) if ns else 1
    for pm, pa, n in zip(p_model, p_actual, ns):
        if pa is None:
            continue
        cx = pad + plot_w * pm
        cy = pad // 2 + plot_h * (1 - pa)
        radius = max(3, 4 + 8 * n / max_n)
        color = (
            "#2ECC71"
            if abs(pm - pa) < 0.05
            else ("#E74C3C" if abs(pm - pa) > 0.15 else "#F39C12")
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{color}" opacity="0.75" stroke="white" stroke-width="1"/>'
        )

    parts.append(_svg_close())
    return "".join(parts)


def _svg_scatter(
    xs: List[float],
    ys: List[float],
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    w: int = _SVG_W,
    h: int = _SVG_H,
) -> str:
    if not xs:
        return f"{_svg_open(w, h)}<text x='50%' y='50%' text-anchor='middle'>No data</text>{_svg_close()}"
    pad = _PAD
    plot_w = w - 2 * pad
    plot_h = h - pad - pad // 2
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    parts = [_svg_open(w, h)]
    if title:
        parts.append(
            f'<text x="{w // 2}" y="14" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">{title}</text>'
        )

    x_labels = [f"{x_min + i * x_range / 5:.2f}" for i in range(6)]
    parts.append(_svg_axes(w, h, pad, x_labels, y_min, y_max))

    for xv, yv in zip(xs, ys):
        cx = pad + plot_w * (xv - x_min) / x_range
        cy = pad // 2 + plot_h * (1 - (yv - y_min) / y_range)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#4A90D9" opacity="0.6"/>'
        )

    parts.append(
        f'<text x="{w // 2}" y="{h - 2}" text-anchor="middle" font-size="9" fill="#666">{x_label}</text>'
    )
    parts.append(_svg_close())
    return "".join(parts)


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _make_table(headers: List[str], rows: List[List[str]]) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    t.horizontalHeader().setStretchLastSection(True)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setItem(r_idx, c_idx, item)
    t.resizeColumnsToContents()
    return t


# ---------------------------------------------------------------------------
# Tab: CLV Analysis
# ---------------------------------------------------------------------------


class _CLVTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._chart = QTextBrowser()
        self._chart.setFixedHeight(300)
        self._table = QTableWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Closing Line Value Distribution"))
        layout.addWidget(self._chart)
        layout.addWidget(QLabel("Per-Bet CLV Details"))
        layout.addWidget(self._table)

    def refresh(self, bets: List[Dict]) -> None:
        clv_vals = [float(b.get("clv_pct", 0.0)) for b in bets if "clv_pct" in b]
        if clv_vals:
            labels = [
                str(b.get("fixture_id", b.get("match_id", f"#{i}")))[:10]
                for i, b in enumerate(bets)
                if "clv_pct" in b
            ]
            svg = _svg_bar_chart(
                labels,
                clv_vals,
                title="CLV % per Bet",
                color="#2ECC71",
                neg_color="#E74C3C",
            )
            self._chart.setHtml(
                f"<html><body style='margin:0;padding:4px'>{svg}</body></html>"
            )
        else:
            self._chart.setHtml("<p style='color:grey'>No CLV data available</p>")

        headers = [
            "Fixture",
            "Market",
            "Bet Odds",
            "Pred. Closing",
            "CLV %",
            "Grade",
            "Confidence",
        ]
        rows = []
        for b in bets:
            if "clv_pct" not in b:
                continue
            rows.append(
                [
                    str(b.get("fixture_id", b.get("match_id", ""))),
                    str(b.get("market", "")),
                    f"{float(b.get('bet_odds', b.get('odds', 0))):.3f}",
                    f"{float(b.get('predicted_closing', 0)):.3f}",
                    f"{float(b.get('clv_pct', 0)):.2f}%",
                    str(b.get("clv_grade", "")),
                    f"{float(b.get('clv_confidence', 0)):.0%}",
                ]
            )

        self._table = _make_table(headers, rows)
        layout = self.layout()
        old = layout.takeAt(layout.count() - 1)
        if old and old.widget():
            old.widget().deleteLater()
        layout.addWidget(self._table)


# ---------------------------------------------------------------------------
# Tab: Market Inefficiency
# ---------------------------------------------------------------------------


class _InefficiencyTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._chart = QTextBrowser()
        self._chart.setFixedHeight(300)
        self._table = QTableWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mean Edge % by League"))
        layout.addWidget(self._chart)
        layout.addWidget(QLabel("Inefficiency Details"))
        layout.addWidget(self._table)

    def refresh(self, scan_results: List[Dict]) -> None:
        if not scan_results:
            self._chart.setHtml("<p style='color:grey'>No scan results</p>")
            return

        # Aggregate by league
        league_edge: Dict[str, List[float]] = {}
        for r in scan_results:
            lg = str(r.get("league", "Unknown"))
            edge = float(r.get("edge_pct", 0.0))
            league_edge.setdefault(lg, []).append(edge)

        sorted_lg = sorted(league_edge.items(), key=lambda x: -sum(x[1]) / len(x[1]))[
            :15
        ]
        labels = [lg[:12] for lg, _ in sorted_lg]
        values = [sum(v) / len(v) for _, v in sorted_lg]

        svg = _svg_bar_chart(labels, values, title="Mean Edge % by League")
        self._chart.setHtml(
            f"<html><body style='margin:0;padding:4px'>{svg}</body></html>"
        )

        headers = [
            "Fixture",
            "League",
            "Market",
            "Type",
            "p_model",
            "p_implied",
            "Gap",
            "Z",
            "Sig",
            "Edge%",
        ]
        rows = []
        for r in scan_results:
            if "gap" not in r:
                continue
            sig = "✓" if r.get("significant") else ""
            rows.append(
                [
                    str(r.get("fixture_id", ""))[:16],
                    str(r.get("league", ""))[:14],
                    str(r.get("market", "")),
                    str(r.get("inefficiency_type", "")),
                    f"{float(r.get('p_model', 0)):.3f}",
                    f"{float(r.get('p_implied', 0)):.3f}",
                    f"{float(r.get('gap', 0)):+.4f}",
                    f"{float(r.get('z_score', 0)):.2f}",
                    sig,
                    f"{float(r.get('edge_pct', 0)):+.2f}%",
                ]
            )

        tbl = _make_table(headers, rows)
        layout = self.layout()
        old = layout.takeAt(layout.count() - 1)
        if old and old.widget():
            old.widget().deleteLater()
        layout.addWidget(tbl)
        self._table = tbl


# ---------------------------------------------------------------------------
# Tab: League Predictability
# ---------------------------------------------------------------------------


class _PredictabilityTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._calib_chart = QTextBrowser()
        self._calib_chart.setFixedHeight(300)
        self._table = QTableWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Home Win Probability Calibration"))
        layout.addWidget(self._calib_chart)
        layout.addWidget(QLabel("League Predictability Scores"))
        layout.addWidget(self._table)

    def refresh(
        self, league_reports: List[Dict], calib_data: Optional[List[Dict]] = None
    ) -> None:
        if calib_data:
            pm = [d.get("p_model_mean", d.get("p_mid", 0.5)) for d in calib_data]
            pa = [d.get("p_actual") for d in calib_data]
            ns = [d.get("n", 1) for d in calib_data]
            pm_f = [x for x, y in zip(pm, pa) if y is not None]
            pa_f = [y for y in pa if y is not None]
            ns_f = [n for n, y in zip(ns, pa) if y is not None]
            svg = _svg_calibration(pm_f, pa_f, ns_f, title="Calibration — Home Win")
            self._calib_chart.setHtml(
                f"<html><body style='margin:0;padding:4px'>{svg}</body></html>"
            )
        else:
            self._calib_chart.setHtml("<p style='color:grey'>No calibration data</p>")

        headers = [
            "League",
            "N",
            "RPS",
            "RPS Skill",
            "Brier",
            "BS Skill",
            "CBI",
            "Upset%",
            "HAF",
            "Score",
            "Grade",
        ]
        rows = []
        for r in league_reports:
            rows.append(
                [
                    str(r.get("league", ""))[:18],
                    str(r.get("n_matches", "")),
                    f"{float(r.get('rps', 0)):.4f}",
                    f"{float(r.get('rps_skill', 0)):+.3f}",
                    f"{float(r.get('brier_brier_score', 0)):.4f}",
                    f"{float(r.get('brier_skill_score', 0)):+.3f}",
                    f"{float(r.get('competitive_balance_index', 0)):.3f}",
                    f"{float(r.get('upset_rate', 0)):.1%}",
                    f"{float(r.get('home_advantage_factor', 0)):.3f}",
                    f"{float(r.get('predictability_score', 0)):.3f}",
                    str(r.get("grade", "")),
                ]
            )

        tbl = _make_table(headers, rows)
        layout = self.layout()
        old = layout.takeAt(layout.count() - 1)
        if old and old.widget():
            old.widget().deleteLater()
        layout.addWidget(tbl)
        self._table = tbl


# ---------------------------------------------------------------------------
# Tab: Feature Importance
# ---------------------------------------------------------------------------


class _FeatureImportanceTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._chart = QTextBrowser()
        self._chart.setFixedHeight(320)
        self._table = QTableWidget()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Top-20 Feature Importance (Mutual Information)"))
        layout.addWidget(self._chart)
        layout.addWidget(self._table)

    def refresh(self, importance: Dict[str, float]) -> None:
        if not importance:
            self._chart.setHtml("<p style='color:grey'>No feature importance data</p>")
            return

        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = [k[:18] for k, _ in ranked]
        values = [v for _, v in ranked]

        svg = _svg_bar_chart(
            labels, values, title="Feature MI Scores (top-20)", color="#8E44AD"
        )
        self._chart.setHtml(
            f"<html><body style='margin:0;padding:4px'>{svg}</body></html>"
        )

        headers = ["Rank", "Feature", "MI Score"]
        rows = [[str(i + 1), k, f"{v:.6f}"] for i, (k, v) in enumerate(ranked)]
        tbl = _make_table(headers, rows)
        layout = self.layout()
        old = layout.takeAt(layout.count() - 1)
        if old and old.widget():
            old.widget().deleteLater()
        layout.addWidget(tbl)
        self._table = tbl


# ---------------------------------------------------------------------------
# Main dashboard window
# ---------------------------------------------------------------------------


class AnalyticsDashboard(QMainWindow):
    """
    Multi-tab analytics window.

    Parameters
    ----------
    controller : AppController (optional — used for the refresh button)
    parent     : parent QWidget
    """

    def __init__(
        self,
        controller=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("Football Quant Engine — Analytics")
        self.resize(1100, 700)

        # Toolbar
        from PySide6.QtWidgets import QToolBar

        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh)
        toolbar.addWidget(refresh_btn)

        self._status_lbl = QLabel("No data loaded")
        toolbar.addWidget(self._status_lbl)

        # Tab widget
        self._tabs = QTabWidget()
        self._clv_tab = _CLVTab()
        self._ineff_tab = _InefficiencyTab()
        self._pred_tab = _PredictabilityTab()
        self._feat_tab = _FeatureImportanceTab()

        self._tabs.addTab(self._clv_tab, "CLV Analysis")
        self._tabs.addTab(self._ineff_tab, "Market Inefficiency")
        self._tabs.addTab(self._pred_tab, "League Predictability")
        self._tabs.addTab(self._feat_tab, "Feature Importance")

        scroll = QScrollArea()
        scroll.setWidget(self._tabs)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        # Data store
        self._bets: List[Dict] = []
        self._predictions: List[Dict] = []
        self._history: List[Dict] = []
        self._league_reports: List[Dict] = []
        self._calib_data: Optional[List[Dict]] = None
        self._feature_importance: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(
        self,
        bets: Optional[List[Dict]] = None,
        predictions: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        league_reports: Optional[List[Dict]] = None,
        calib_data: Optional[List[Dict]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Load all analytics data and refresh all tabs.

        Parameters
        ----------
        bets              : ranked bet dicts (enriched with CLV fields)
        predictions       : raw model prediction dicts (with p_model, p_implied, odds)
        history           : historical match records (for league predictability)
        league_reports    : pre-computed league report dicts (from LeaguePredictabilityAnalyser)
        calib_data        : calibration curve dicts (from LeaguePredictabilityAnalyser.calibration_curve)
        feature_importance: {feature_name: mi_score} from FeatureGenerator.importance
        """
        if bets is not None:
            self._bets = bets
        if predictions is not None:
            self._predictions = predictions
        if history is not None:
            self._history = history
        if league_reports is not None:
            self._league_reports = league_reports
        if calib_data is not None:
            self._calib_data = calib_data
        if feature_importance is not None:
            self._feature_importance = feature_importance

        self._refresh_all()

    def load_from_controller(self) -> None:
        """
        Pull data from the connected AppController and refresh.
        Uses lazy analytics computation if modules are available.
        """
        if not self._controller:
            self._status_lbl.setText("No controller connected")
            return

        try:
            status = self._controller.status()
            bets = getattr(self._controller, "_last_ranked", []) or []

            # Scan for market inefficiencies if possible
            predictions_enriched = []
            try:
                from analytics.market_inefficiency_scanner import (
                    MarketInefficiencyScanner,
                )

                scanner = MarketInefficiencyScanner()
                predictions_enriched = scanner.scan(bets)
            except Exception as exc:
                logger.debug("AnalyticsDashboard: inefficiency scan failed: %s", exc)

            # CLV scoring
            bets_enriched = bets
            try:
                from analytics.closing_line_model import ClosingLineModel

                clv_model = ClosingLineModel()
                bets_enriched = clv_model.score_bets(bets)
            except Exception as exc:
                logger.debug("AnalyticsDashboard: CLV scoring failed: %s", exc)

            self.load_data(
                bets=bets_enriched,
                predictions=predictions_enriched,
                league_reports=self._league_reports,
                calib_data=self._calib_data,
                feature_importance=self._feature_importance,
            )
            n_bets = status.get("last_bet_count", 0)
            self._status_lbl.setText(f"Loaded {len(bets)} predictions, {n_bets} bets")

        except Exception as exc:
            logger.warning("AnalyticsDashboard.load_from_controller: %s", exc)
            self._status_lbl.setText(f"Load error: {exc}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        self._clv_tab.refresh(self._bets)
        self._ineff_tab.refresh(self._predictions)
        self._pred_tab.refresh(self._league_reports, self._calib_data)
        self._feat_tab.refresh(self._feature_importance)

        n = len(self._bets)
        self._status_lbl.setText(
            f"{n} bet(s) | "
            f"{len(self._predictions)} prediction(s) | "
            f"{len(self._league_reports)} league(s)"
        )

    def _on_refresh(self) -> None:
        if self._controller:
            self.load_from_controller()
        else:
            self._refresh_all()
