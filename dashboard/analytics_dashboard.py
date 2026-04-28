"""
Analytics Dashboard — PySide6 multi-tab widget.

Tabs
----
1. CLV Analysis      — closing-line value distribution + per-bet table
2. Market Inefficiency — inefficiency type breakdown + league ranking
3. League Predictability — RPS / Brier / CBI per league table + calibration chart
4. Feature Importance  — MI-ranked bar chart of top features

Sub-modules
-----------
dashboard.dashboard_charts — SVG rendering helpers and Qt table factory
dashboard.dashboard_tabs   — _CLVTab, _InefficiencyTab, _PredictabilityTab, _FeatureImportanceTab
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from dashboard.dashboard_tabs import (
    _CLVTab,
    _FeatureImportanceTab,
    _InefficiencyTab,
    _PredictabilityTab,
)

logger = logging.getLogger(__name__)


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

        from PySide6.QtWidgets import QToolBar

        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh)
        toolbar.addWidget(refresh_btn)

        self._status_lbl = QLabel("No data loaded")
        toolbar.addWidget(self._status_lbl)

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

        self._bets: List[Dict] = []
        self._predictions: List[Dict] = []
        self._history: List[Dict] = []
        self._league_reports: List[Dict] = []
        self._calib_data: Optional[List[Dict]] = None
        self._feature_importance: Dict[str, float] = {}

    def load_data(
        self,
        bets: Optional[List[Dict]] = None,
        predictions: Optional[List[Dict]] = None,
        history: Optional[List[Dict]] = None,
        league_reports: Optional[List[Dict]] = None,
        calib_data: Optional[List[Dict]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> None:
        """Load all analytics data and refresh all tabs."""
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
        """Pull data from the connected AppController and refresh."""
        if not self._controller:
            self._status_lbl.setText("No controller connected")
            return

        try:
            status = self._controller.status()
            bets = getattr(self._controller, "_last_ranked", []) or []

            predictions_enriched = []
            try:
                from analytics.market_inefficiency_scanner import MarketInefficiencyScanner

                scanner = MarketInefficiencyScanner()
                predictions_enriched = scanner.scan(bets)
            except Exception as exc:
                logger.debug("AnalyticsDashboard: inefficiency scan failed: %s", exc)

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
