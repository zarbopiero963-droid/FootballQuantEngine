"""
CLV Tracker — Closing Line Value tracking window.

Closing Line Value is the gold standard metric for professional bettors:
if you consistently bet at higher odds than where the market closes, you
are mathematically finding value before the sharps do.

  CLV% = (our_odds / closing_odds − 1) × 100

Example: bet at 2.10, line closed at 1.85 → CLV = +13.5% (excellent)
Example: bet at 1.90, line closed at 2.10 → CLV = −9.5% (chasing bad prices)

A long-run avg CLV > 0 is the only rigorous proof that a model has edge,
independent of short-run variance or luck.

Usage
-----
    from ui.clv_tracker_window import CLVTrackerWindow
    window = CLVTrackerWindow()
    window.show()
"""

from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_DB_PATH = os.path.join("outputs", "clv_tracker.db")

_COL_ID = 0
_COL_ADDED = 1
_COL_FIXTURE = 2
_COL_MARKET = 3
_COL_OUR_ODDS = 4
_COL_CLOSING = 5
_COL_CLV = 6
_COL_RESULT = 7
_COL_PROFIT = 8

_HEADERS = [
    "ID",
    "Added",
    "Fixture",
    "Market",
    "Our Odds",
    "Closing",
    "CLV %",
    "Result",
    "Profit",
]

_COLOR_GREEN = QColor(200, 240, 200)
_COLOR_RED = QColor(255, 210, 210)
_COLOR_GRAY = QColor(230, 230, 230)
_COLOR_WHITE = QColor(255, 255, 255)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class CLVBet:
    id: Optional[int]
    added_at: str
    fixture_desc: str
    market: str
    our_odds: float
    closing_odds: Optional[float] = None
    clv_pct: Optional[float] = None
    result: Optional[str] = None
    profit: Optional[float] = None
    notes: str = ""

    @staticmethod
    def clv_from_odds(our_odds: float, closing_odds: float) -> float:
        """CLV % = (our_odds / closing_odds − 1) × 100."""
        if closing_odds <= 0:
            return 0.0
        return (our_odds / closing_odds - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class CLVDatabase:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clv_bets (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    added_at         TEXT    NOT NULL,
                    fixture_desc     TEXT    NOT NULL,
                    market           TEXT    NOT NULL,
                    our_odds         REAL    NOT NULL,
                    closing_odds     REAL,
                    clv_pct          REAL,
                    result           TEXT,
                    profit           REAL,
                    notes            TEXT    DEFAULT ''
                )
                """
            )

    def add_bet(
        self,
        fixture_desc: str,
        market: str,
        our_odds: float,
        notes: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO clv_bets (added_at, fixture_desc, market, our_odds, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, fixture_desc, market, our_odds, notes),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_closing(
        self,
        bet_id: int,
        closing_odds: float,
        result: Optional[str],
    ) -> None:
        clv = CLVBet.clv_from_odds(0.0, closing_odds)  # placeholder
        # fetch our_odds first
        with self._conn() as conn:
            row = conn.execute(
                "SELECT our_odds FROM clv_bets WHERE id = ?", (bet_id,)
            ).fetchone()
            if row is None:
                return
            our_odds = float(row["our_odds"])
            clv = CLVBet.clv_from_odds(our_odds, closing_odds)
            profit: Optional[float] = None
            if result == "WIN":
                profit = our_odds - 1.0
            elif result == "LOSS":
                profit = -1.0
            conn.execute(
                """
                UPDATE clv_bets
                SET closing_odds = ?, clv_pct = ?, result = ?, profit = ?
                WHERE id = ?
                """,
                (
                    closing_odds,
                    clv,
                    result if result != "— Pending —" else None,
                    profit,
                    bet_id,
                ),
            )

    def get_all(self) -> List[CLVBet]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM clv_bets ORDER BY id DESC").fetchall()
        bets = []
        for r in rows:
            bets.append(
                CLVBet(
                    id=r["id"],
                    added_at=r["added_at"],
                    fixture_desc=r["fixture_desc"],
                    market=r["market"],
                    our_odds=float(r["our_odds"]),
                    closing_odds=float(r["closing_odds"])
                    if r["closing_odds"] is not None
                    else None,
                    clv_pct=float(r["clv_pct"]) if r["clv_pct"] is not None else None,
                    result=r["result"],
                    profit=float(r["profit"]) if r["profit"] is not None else None,
                    notes=r["notes"] or "",
                )
            )
        return bets

    def export_csv(self, path: str) -> None:
        bets = self.get_all()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ID",
                    "Added",
                    "Fixture",
                    "Market",
                    "OurOdds",
                    "ClosingOdds",
                    "CLV%",
                    "Result",
                    "Profit",
                    "Notes",
                ]
            )
            for b in bets:
                writer.writerow(
                    [
                        b.id,
                        b.added_at,
                        b.fixture_desc,
                        b.market,
                        b.our_odds,
                        b.closing_odds,
                        f"{b.clv_pct:.2f}" if b.clv_pct is not None else "",
                        b.result or "",
                        f"{b.profit:.2f}" if b.profit is not None else "",
                        b.notes,
                    ]
                )


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


class _AddBetDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add New Bet")
        self.setFixedWidth(400)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._fixture = QLineEdit()
        self._fixture.setPlaceholderText("e.g. Arsenal vs Chelsea")
        form.addRow("Fixture:", self._fixture)

        self._market = QLineEdit()
        self._market.setPlaceholderText("e.g. home / draw / over2.5")
        form.addRow("Market:", self._market)

        self._odds = QDoubleSpinBox()
        self._odds.setRange(1.01, 1000.0)
        self._odds.setSingleStep(0.01)
        self._odds.setDecimals(2)
        self._odds.setValue(2.00)
        form.addRow("Our Odds:", self._odds)

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("Optional notes")
        form.addRow("Notes:", self._notes)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if not self._fixture.text().strip():
            QMessageBox.warning(self, "Validation", "Fixture description is required.")
            return
        if not self._market.text().strip():
            QMessageBox.warning(self, "Validation", "Market is required.")
            return
        self.accept()

    @property
    def fixture(self) -> str:
        return self._fixture.text().strip()

    @property
    def market(self) -> str:
        return self._market.text().strip()

    @property
    def odds(self) -> float:
        return self._odds.value()

    @property
    def notes(self) -> str:
        return self._notes.text().strip()


class _UpdateClosingDialog(QDialog):
    def __init__(self, bet: CLVBet, parent=None) -> None:
        super().__init__(parent)
        self._bet = bet
        self.setWindowTitle("Update Closing Odds")
        self.setFixedWidth(420)

        layout = QVBoxLayout(self)

        info = QLabel(
            f"<b>{bet.fixture_desc}</b> — {bet.market}<br>"
            f"Our odds: <b>{bet.our_odds:.2f}</b>"
        )
        layout.addWidget(info)

        form = QFormLayout()

        self._closing = QDoubleSpinBox()
        self._closing.setRange(1.01, 1000.0)
        self._closing.setSingleStep(0.01)
        self._closing.setDecimals(2)
        self._closing.setValue(bet.closing_odds or bet.our_odds)
        form.addRow("Closing Odds:", self._closing)

        self._clv_label = QLabel()
        self._clv_label.setFont(QFont("monospace"))
        form.addRow("Live CLV:", self._clv_label)

        self._result = QComboBox()
        self._result.addItems(["— Pending —", "WIN", "LOSS"])
        if bet.result == "WIN":
            self._result.setCurrentIndex(1)
        elif bet.result == "LOSS":
            self._result.setCurrentIndex(2)
        form.addRow("Result:", self._result)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._closing.valueChanged.connect(self._update_clv_preview)
        self._update_clv_preview(self._closing.value())

    def _update_clv_preview(self, closing: float) -> None:
        clv = CLVBet.clv_from_odds(self._bet.our_odds, closing)
        color = "green" if clv > 0 else "red"
        self._clv_label.setText(
            f"<span style='color:{color}'><b>{clv:+.2f}%</b></span>"
        )

    @property
    def closing_odds(self) -> float:
        return self._closing.value()

    @property
    def result(self) -> str:
        return self._result.currentText()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class CLVTrackerWindow(QWidget):
    """
    PySide6 window for tracking Closing Line Value across all bets.

    Persists bets to SQLite at outputs/clv_tracker.db.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CLV Tracker — Closing Line Value")
        self.resize(940, 600)
        self._db = CLVDatabase(_DB_PATH)
        self._bets: List[CLVBet] = []
        self._setup_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_add = QPushButton("+ Add Bet")
        self._btn_update = QPushButton("✏ Update Closing Odds")
        self._btn_export = QPushButton("Export CSV")
        self._btn_add.clicked.connect(self.add_bet)
        self._btn_update.clicked.connect(self.update_closing_odds)
        self._btn_export.clicked.connect(self.export_csv)
        toolbar.addWidget(self._btn_add)
        toolbar.addWidget(self._btn_update)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_export)
        root.addLayout(toolbar)

        # Table
        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setSectionResizeMode(_COL_FIXTURE, QHeaderView.Stretch)
        hdr.resizeSection(_COL_ID, 40)
        hdr.resizeSection(_COL_ADDED, 130)
        hdr.resizeSection(_COL_MARKET, 90)
        hdr.resizeSection(_COL_OUR_ODDS, 75)
        hdr.resizeSection(_COL_CLOSING, 75)
        hdr.resizeSection(_COL_CLV, 70)
        hdr.resizeSection(_COL_RESULT, 65)
        hdr.resizeSection(_COL_PROFIT, 65)
        root.addWidget(self._table, 1)

        # Summary group
        summary_box = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_box)

        self._lbl_bets = QLabel("Bets: —")
        self._lbl_avg_clv = QLabel("Avg CLV: —")
        self._lbl_clv_hit = QLabel("CLV Hit Rate: —")
        self._lbl_roi = QLabel("ROI: —")
        self._lbl_pl = QLabel("Total P/L: —")

        for lbl in (
            self._lbl_bets,
            self._lbl_avg_clv,
            self._lbl_clv_hit,
            self._lbl_roi,
            self._lbl_pl,
        ):
            lbl.setFont(QFont("monospace", 10))
            summary_layout.addWidget(lbl)

        root.addWidget(summary_box)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def add_bet(self) -> None:
        dlg = _AddBetDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._db.add_bet(dlg.fixture, dlg.market, dlg.odds, dlg.notes)
            self._refresh()

    def update_closing_odds(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            QMessageBox.information(
                self, "Select Row", "Please select a bet row first."
            )
            return
        row_idx = self._table.currentRow()
        if row_idx < 0 or row_idx >= len(self._bets):
            return
        bet = self._bets[row_idx]
        dlg = _UpdateClosingDialog(bet, self)
        if dlg.exec() == QDialog.Accepted:
            self._db.update_closing(bet.id, dlg.closing_odds, dlg.result)
            self._refresh()

    def export_csv(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("outputs", f"clv_export_{ts}.csv")
        os.makedirs("outputs", exist_ok=True)
        self._db.export_csv(path)
        QMessageBox.information(self, "Export", f"Saved to:\n{os.path.abspath(path)}")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._bets = self._db.get_all()
        self._populate_table(self._bets)
        self._update_summary(self._bets)

    def _populate_table(self, bets: List[CLVBet]) -> None:
        self._table.setRowCount(0)
        for bet in bets:
            row = self._table.rowCount()
            self._table.insertRow(row)

            clv_color = _COLOR_WHITE
            if bet.clv_pct is not None:
                clv_color = _COLOR_GREEN if bet.clv_pct > 0 else _COLOR_RED

            result_color = _COLOR_WHITE
            if bet.result == "WIN":
                result_color = _COLOR_GREEN
            elif bet.result == "LOSS":
                result_color = _COLOR_RED

            cells = [
                (str(bet.id), _COLOR_WHITE),
                (bet.added_at, _COLOR_WHITE),
                (bet.fixture_desc, _COLOR_WHITE),
                (bet.market, _COLOR_WHITE),
                (f"{bet.our_odds:.2f}", _COLOR_WHITE),
                (
                    f"{bet.closing_odds:.2f}" if bet.closing_odds is not None else "—",
                    _COLOR_WHITE,
                ),
                (f"{bet.clv_pct:+.2f}%" if bet.clv_pct is not None else "—", clv_color),
                (bet.result or "—", result_color),
                (f"{bet.profit:+.2f}" if bet.profit is not None else "—", _COLOR_WHITE),
            ]

            for col, (text, color) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if color != _COLOR_WHITE:
                    item.setBackground(color)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._table.setItem(row, col, item)

    def _update_summary(self, bets: List[CLVBet]) -> None:
        n_total = len(bets)
        with_clv = [b for b in bets if b.clv_pct is not None]
        settled = [b for b in bets if b.profit is not None]

        avg_clv = (
            (sum(b.clv_pct for b in with_clv) / len(with_clv)) if with_clv else None
        )
        clv_positive_rate = (
            (sum(1 for b in with_clv if b.clv_pct > 0) / len(with_clv))
            if with_clv
            else None
        )
        total_pl = sum(b.profit for b in settled) if settled else None
        roi = (total_pl / len(settled)) if settled and total_pl is not None else None

        self._lbl_bets.setText(
            f"Bets: {n_total}  (settled: {len(settled)}, with CLV: {len(with_clv)})"
        )
        self._lbl_avg_clv.setText(
            f"Avg CLV: {avg_clv:+.2f}%" if avg_clv is not None else "Avg CLV: —"
        )
        self._lbl_clv_hit.setText(
            f"CLV Hit Rate: {clv_positive_rate:.1%}"
            if clv_positive_rate is not None
            else "CLV Hit Rate: —"
        )
        self._lbl_roi.setText(f"ROI: {roi:+.2%}" if roi is not None else "ROI: —")
        self._lbl_pl.setText(
            f"Total P/L: {total_pl:+.2f}u" if total_pl is not None else "Total P/L: —"
        )
