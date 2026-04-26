"""
AppController
=============
Main orchestrator for the desktop application.

Responsibilities:
  - Initialise DB and run historical bootstrap (via BootstrapController)
  - Run a quant prediction cycle (via JobRunner)
  - Rank results with Kelly criterion and composite scoring (MatchRanker)
  - Export to CSV and Excel (CsvExporter, ExcelExporter)
  - Generate a dated HTML performance report (ReportGenerator)
  - Send Telegram alerts (TelegramNotifier)
  - Start / stop the live odds streaming engine (OddsStream)
  - Start / stop the live fixture updater (LiveUpdater via BootstrapController)
  - Track bankroll across cycles
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Optional

from app.bootstrap_controller import BootstrapController
from config.settings_manager import load_settings
from engine.job_runner import JobRunner
from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter
from notifications.telegram_notifier import TelegramNotifier
from ranking.match_ranker import MatchRanker
from report.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class AppController:
    """Central application controller."""

    def __init__(
        self,
        bankroll: float = 1000.0,
        kelly_scale: float = 0.25,
        output_dir: str = "outputs",
        dry_run: bool = False,
    ) -> None:
        self._settings = load_settings()
        self._output = Path(output_dir)
        self._output.mkdir(parents=True, exist_ok=True)

        # Bootstrap / data engine
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        self._bootstrap_ctrl = BootstrapController(api_key=api_key)
        self._bootstrap_ctrl.initialise()

        # Core prediction pipeline
        self.runner = JobRunner()

        # Post-processing
        self.ranker = MatchRanker(bankroll=bankroll, kelly_scale=kelly_scale)
        self.csv_exp = CsvExporter(output_dir=str(self._output))
        self.xlsx_exp = ExcelExporter(output_dir=str(self._output))
        self.reporter = ReportGenerator(output_dir=str(self._output / "reports"))
        self.notifier = TelegramNotifier(
            token=self._settings.telegram_token,
            chat_id=self._settings.telegram_chat_id,
            dry_run=dry_run,
        )

        # Live streaming (lazy-started)
        self._stream = None

        # State
        self._last_run_ts: Optional[float] = None
        self._last_result_count: int = 0
        self._last_bet_count: int = 0
        self._cumulative_metrics: dict = {}

    # ------------------------------------------------------------------
    # Bootstrap (first run / refresh historical data)
    # ------------------------------------------------------------------

    def run_bootstrap(
        self,
        league_id: int,
        progress_cb=None,
        fetch_stats: bool = True,
    ) -> dict:
        """Download all historical seasons for league_id into the local DB."""
        return self._bootstrap_ctrl.run_bootstrap(
            league_id=league_id,
            progress_cb=progress_cb,
            fetch_stats=fetch_stats,
        )

    def update_current_season(
        self, league_id: int, season: int, progress_cb=None
    ) -> int:
        """Re-fetch only the current season to pick up new results."""
        return self._bootstrap_ctrl.update_current_season(
            league_id=league_id,
            season=season,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def run_predictions(self) -> list[dict]:
        """Full cycle: predict → rank → export → notify."""
        t0 = time.monotonic()

        try:
            raw = self.runner.run_cycle()
        except Exception as exc:
            logger.error("AppController.run_predictions: runner failed: %s", exc)
            return []

        if not raw:
            logger.info("AppController: no prediction results.")
            return []

        records = self._to_list(raw)
        ranked = self.ranker.rank_as_dicts(records)

        today = date.today().isoformat()
        csv_path = self._export_csv(ranked, today)
        self._export_excel(ranked, today)

        bets_only = [r for r in ranked if r.get("decision") == "BET"]
        self._notify_bets(bets_only, csv_path)

        elapsed = time.monotonic() - t0
        self._last_run_ts = time.time()
        self._last_result_count = len(ranked)
        self._last_bet_count = len(bets_only)

        logger.info(
            "AppController: cycle done in %.1fs — %d predictions, %d bets.",
            elapsed,
            len(ranked),
            len(bets_only),
        )

        top_match = ranked[0].get("match_id") if ranked else None
        self.notifier.send_cycle_summary(
            n_processed=len(records),
            n_bets=len(bets_only),
            n_watchlist=sum(1 for r in ranked if r.get("decision") == "WATCHLIST"),
            elapsed_sec=elapsed,
            top_match=top_match,
        )

        return ranked

    def run_cycle(
        self, league_id: int | None = None, season: int | None = None
    ) -> list:
        """Update current season data then generate all-market predictions."""
        settings = load_settings()
        lid = league_id or getattr(settings, "league_id", None) or 135
        s   = season   or getattr(settings, "season",    None) or 2024
        self.update_current_season(lid, s)
        return self._bootstrap_ctrl.run_predictions(lid, s)

    # Aliases for backwards compatibility with MainWindow / tests
    def run_once(self) -> list[dict]:
        return self.run_predictions()

    def run_quant_cycle(self) -> list[dict]:
        return self.run_predictions()

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def generate_daily_report(
        self,
        ranked_bets: Optional[list[dict]] = None,
        metrics: Optional[dict] = None,
    ) -> str:
        if ranked_bets is None:
            ranked_bets = self.run_predictions()
        metrics = metrics or self._cumulative_metrics
        report_path = self.reporter.generate(
            ranked_bets=ranked_bets,
            metrics=metrics,
        )
        logger.info("AppController: daily report saved to %s", report_path)
        self.notifier.send_daily_report_alert(
            metrics=metrics,
            report_path=report_path,
        )
        return report_path

    # ------------------------------------------------------------------
    # Live fixture updates (data engine)
    # ------------------------------------------------------------------

    def start_live_updates(self) -> None:
        self._bootstrap_ctrl.start_live_updates()

    def stop_live_updates(self) -> None:
        self._bootstrap_ctrl.stop_live_updates()

    def on_live_update(self, callback) -> None:
        self._bootstrap_ctrl.on_live_update(callback)

    def live_health(self) -> dict:
        return self._bootstrap_ctrl.live_health()

    # ------------------------------------------------------------------
    # Live odds streaming
    # ------------------------------------------------------------------

    def start_stream(
        self, fetch_fn, source: str = "default", interval: float = 300.0
    ) -> None:
        from live.odds_stream import OddsStream

        if self._stream and self._stream.is_running():
            logger.warning("AppController: stream already running.")
            return

        def _on_steam(snap, market, magnitude):
            self.notifier.send_steam_alert(
                fixture_id=snap.fixture_id,
                market=market,
                magnitude=magnitude,
                source=snap.source,
            )

        self._stream = OddsStream(
            sources={source: fetch_fn},
            interval=interval,
            on_steam=_on_steam,
        )
        self._stream.start()
        logger.info(
            "AppController: live stream started (source=%s, interval=%.0fs).",
            source,
            interval,
        )

    def stop_stream(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream = None

    def stream_health(self) -> dict:
        if not self._stream:
            return {}
        return {
            name: {
                "healthy": h.healthy,
                "polls": h.total_polls,
                "errors": h.total_errors,
                "uptime_pct": h.uptime_pct,
            }
            for name, h in self._stream.health().items()
        }

    # ------------------------------------------------------------------
    # Bankroll management
    # ------------------------------------------------------------------

    def update_bankroll(self, bankroll: float) -> None:
        self.ranker.update_bankroll(bankroll)
        logger.info("AppController: bankroll updated to %.2f.", bankroll)

    def update_metrics(self, metrics: dict) -> None:
        self._cumulative_metrics = dict(metrics)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "last_run_ts": self._last_run_ts,
            "last_result_count": self._last_result_count,
            "last_bet_count": self._last_bet_count,
            "bankroll": self.ranker.bankroll,
            "stream_running": bool(self._stream and self._stream.is_running()),
            "telegram_ok": self.notifier.configured,
            "notify_stats": self.notifier.stats(),
            "live_updater": self.live_health(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(raw) -> list[dict]:
        if isinstance(raw, list):
            return raw
        try:
            return raw.to_dict("records")
        except Exception:
            return []

    def _export_csv(self, ranked: list[dict], today: str) -> Optional[str]:
        try:
            return self.csv_exp.export(ranked, mode="bets", date_str=today)
        except Exception as exc:
            logger.warning("AppController: CSV export failed: %s", exc)
            return None

    def _export_excel(self, ranked: list[dict], today: str) -> Optional[str]:
        try:
            return self.xlsx_exp.export(
                ranked,
                metrics=self._cumulative_metrics,
                date_str=today,
            )
        except Exception as exc:
            logger.warning("AppController: Excel export failed: %s", exc)
            return None

    def _notify_bets(
        self,
        bets: list[dict],
        csv_path: Optional[str],
    ) -> None:
        if not self.notifier.configured:
            return
        try:
            self.notifier.send_value_bets_alert(bets)
            if csv_path:
                self.notifier.send_document(
                    csv_path,
                    caption=f"Value bets {date.today().isoformat()} ({len(bets)} bets)",
                )
        except Exception as exc:
            logger.warning("AppController: Telegram notification failed: %s", exc)
