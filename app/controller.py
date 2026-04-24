"""
Application controller — orchestrates the full prediction-to-notification pipeline.

Responsibilities
----------------
1. Run a quant prediction cycle (via JobRunner)
2. Rank results with Kelly criterion and composite scoring (MatchRanker)
3. Export to CSV and Excel with full formatting (CsvExporter, ExcelExporter)
4. Generate a dated HTML performance report (ReportGenerator)
5. Send Telegram alerts: value-bet list + attach CSV + daily summary
6. Start / stop the live odds streaming engine (OddsStream)
7. Track bankroll across cycles and pass it to the ranker for stake sizing
8. Expose run stats for the UI (last run time, last result count, etc.)

Design decisions
----------------
- JobRunner / QuantJobRunner are the source of truth for predictions; this
  controller adds the post-processing layer on top without duplicating logic.
- Telegram notifications are best-effort: failures are logged but never
  raise exceptions so the UI cycle always completes.
- The OddsStream is started in a daemon thread and injected with a callback
  that triggers a Telegram steam alert.
- Bankroll defaults to 1000 and can be updated via update_bankroll().
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

from config.settings_manager import load_settings
from engine.job_runner import JobRunner
from export.csv_exporter import CsvExporter
from export.excel_exporter import ExcelExporter
from notifications.telegram_notifier import TelegramNotifier
from ranking.match_ranker import MatchRanker
from report.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class AppController:
    """
    Central application controller.

    Parameters
    ----------
    bankroll     : starting bankroll for Kelly stake sizing (default 1000)
    kelly_scale  : fraction of full Kelly to recommend (default 0.25)
    output_dir   : base directory for CSV / Excel / report files
    dry_run      : if True, Telegram sends are logged but not actually sent
    """

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

        # Core pipeline
        self.runner = JobRunner()

        # Post-processing components
        self.ranker = MatchRanker(bankroll=bankroll, kelly_scale=kelly_scale)
        self.csv_exp = CsvExporter(output_dir=str(self._output))
        self.xlsx_exp = ExcelExporter(output_dir=str(self._output))
        self.reporter = ReportGenerator(
            output_dir=str(self._output / "reports"),
        )
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
    # Primary entry points (called from MainWindow / QuantJobRunner)
    # ------------------------------------------------------------------

    def run_predictions(self) -> list[dict]:
        """
        Full cycle: predict → rank → export → notify.

        Returns the raw ranked list (list[dict]) so the UI can display it.
        """
        t0 = time.monotonic()

        # 1. Raw predictions from the quant engine
        try:
            raw = self.runner.run_cycle()
        except Exception as exc:
            logger.error("AppController.run_predictions: runner failed: %s", exc)
            return []

        if not raw:
            logger.info("AppController: no prediction results.")
            return []

        # Normalise: JobRunner may return list[dict] or DataFrame
        records = self._to_list(raw)

        # 2. Rank with Kelly + composite score
        ranked = self.ranker.rank_as_dicts(records)

        # 3. Export
        today = date.today().isoformat()
        csv_path = self._export_csv(ranked, today)
        self._export_excel(ranked, today)

        # 4. Telegram: value bets alert + CSV attachment
        bets_only = [r for r in ranked if r.get("decision") == "BET"]
        self._notify_bets(bets_only, csv_path)

        # 5. Stats
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

        # 6. Cycle summary notification
        top_match = ranked[0].get("match_id") if ranked else None
        self.notifier.send_cycle_summary(
            n_processed=len(records),
            n_bets=len(bets_only),
            n_watchlist=sum(1 for r in ranked if r.get("decision") == "WATCHLIST"),
            elapsed_sec=elapsed,
            top_match=top_match,
        )

        return ranked

    # Aliases kept for backwards compatibility with MainWindow / tests
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
        """
        Generate a dated HTML report and send a Telegram summary.

        If ranked_bets is None, runs a fresh prediction cycle first.

        Returns
        -------
        str — path of the generated HTML report.
        """
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
    # Live odds streaming
    # ------------------------------------------------------------------

    def start_stream(
        self, fetch_fn, source: str = "default", interval: float = 300.0
    ) -> None:
        """
        Start live odds streaming.

        Parameters
        ----------
        fetch_fn : callable() → list[dict]  (fixture_id, home_odds, …)
        source   : display name for this source
        interval : polling interval in seconds
        """
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
        """Store backtest metrics for inclusion in daily reports."""
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
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(raw) -> list[dict]:
        """Convert pipeline output (list or DataFrame) to list[dict]."""
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
        """Send Telegram value-bets alert + attach CSV."""
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
