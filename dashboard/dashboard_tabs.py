"""
Tab widgets for the Analytics Dashboard: CLV, Market Inefficiency,
League Predictability, and Feature Importance.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from dashboard.dashboard_charts import (
    _make_table,
    _svg_bar_chart,
    _svg_calibration,
)


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

        league_edge: Dict[str, List[float]] = {}
        for r in scan_results:
            lg = str(r.get("league", "Unknown"))
            edge = float(r.get("edge_pct", 0.0))
            league_edge.setdefault(lg, []).append(edge)

        sorted_lg = sorted(league_edge.items(), key=lambda x: -sum(x[1]) / len(x[1]))[:15]
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
