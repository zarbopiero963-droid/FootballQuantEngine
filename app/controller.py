"""
AppController
=============
Main orchestrator for the desktop application.

Responsibilities:
  - Initialise DB (all tables)
  - Run historical bootstrap on first use
  - Run multi-market predictions on demand
  - Start / stop live updater
  - Expose legacy run_cycle() for backwards compatibility with existing UI
"""
from __future__ import annotations

import logging
import os

from app.bootstrap_controller import BootstrapController

logger = logging.getLogger(__name__)


class AppController:

    def __init__(self):
        api_key = os.getenv("API_FOOTBALL_KEY", "")
        self._bootstrap_ctrl = BootstrapController(api_key=api_key)
        self._bootstrap_ctrl.initialise()

    # ------------------------------------------------------------------
    # Bootstrap (first run / refresh)
    # ------------------------------------------------------------------

    def run_bootstrap(
        self,
        league_id: int,
        progress_cb=None,
        fetch_stats: bool = True,
    ) -> dict:
        """
        Download all historical seasons for league_id into the local DB.
        Safe to call repeatedly — skips already-downloaded seasons.
        """
        return self._bootstrap_ctrl.run_bootstrap(
            league_id=league_id,
            progress_cb=progress_cb,
            fetch_stats=fetch_stats,
        )

    def update_current_season(self, league_id: int, season: int, progress_cb=None) -> int:
        """Re-fetch only the current season to pick up new results."""
        return self._bootstrap_ctrl.update_current_season(
            league_id=league_id,
            season=season,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Predictions (all markets)
    # ------------------------------------------------------------------

    def run_predictions(self, league_id: int, season: int, progress_cb=None) -> list:
        """
        Generate 1X2 + OU + BTTS + HT + Corners + Correct Score predictions
        for all upcoming fixtures, save to DB, return rows for the UI.
        """
        return self._bootstrap_ctrl.run_predictions(
            league_id=league_id,
            season=season,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Legacy compatibility (existing UI calls run_cycle / run_once)
    # ------------------------------------------------------------------

    def run_once(self) -> list:
        return self.run_cycle()

    def run_quant_cycle(self) -> list:
        return self.run_cycle()

    def run_cycle(self, league_id: int | None = None, season: int | None = None) -> list:
        """
        Convenience wrapper: refresh current season + predict all markets.
        league_id / season default to env vars or settings.
        """
        from config.settings_manager import load_settings
        settings = load_settings()
        lid = league_id or settings.league_id or 135
        s   = season   or settings.season   or 2024

        self.update_current_season(lid, s)
        return self.run_predictions(lid, s)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------

    def start_live_updates(self) -> None:
        self._bootstrap_ctrl.start_live_updates()

    def stop_live_updates(self) -> None:
        self._bootstrap_ctrl.stop_live_updates()

    def on_live_update(self, callback) -> None:
        self._bootstrap_ctrl.on_live_update(callback)

    def live_health(self) -> dict:
        return self._bootstrap_ctrl.live_health()
