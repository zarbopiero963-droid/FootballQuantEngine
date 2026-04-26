"""
Historical data bootstrapper — downloads all fixtures for 18 Betfair Exchange
football leagues from 2008 to the current season.

Resume support
--------------
The ``downloaded_seasons`` table acts as a registry.  If a league/season row
already exists, it is skipped entirely, so an interrupted bootstrap can be
re-started without re-downloading already-complete data.

Rate limiting
-------------
The ``ApiFootballCollector`` already applies a token-bucket limiter (default
10 req/min).  The downloader adds an inter-season pause of 1 second to avoid
bursting and to give the API time to breathe between large responses.

Usage
-----
    from engine.historical_downloader import HistoricalDownloader

    dl = HistoricalDownloader()
    dl.run()           # full bootstrap — skips already-downloaded seasons
    dl.run(resume=False)   # force re-download everything
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from data.api_football_collector import ApiFootballCollector
from database.fixtures_repository import FixturesRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Betfair Exchange football leagues
# API-Football league IDs for the 18 markets regularly traded on Betfair
# ---------------------------------------------------------------------------

BETFAIR_LEAGUES: Dict[int, str] = {
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    39: "England - Premier League",
    40: "England - Championship",
    61: "France - Ligue 1",
    65: "France - Ligue 2",
    78: "Germany - Bundesliga",
    79: "Germany - 2. Bundesliga",
    88: "Netherlands - Eredivisie",
    94: "Portugal - Primeira Liga",
    119: "Denmark - Superliga",
    135: "Italy - Serie A",
    136: "Italy - Serie B",
    140: "Spain - La Liga",
    143: "Spain - Segunda División",
    144: "Belgium - Pro League",
    179: "Scotland - Premiership",
    203: "Turkey - Süper Lig",
}

# First season we want.  API-Football typically has data from ~2010 depending
# on league; we request from BOOTSTRAP_FROM and silently skip empty responses.
BOOTSTRAP_FROM = 2008

# Inter-season pause (seconds) — gives the API time between back-to-back calls
_INTER_SEASON_PAUSE = 1.0


# ---------------------------------------------------------------------------
# Progress dataclass
# ---------------------------------------------------------------------------


@dataclass
class DownloadProgress:
    total_leagues: int = 0
    total_seasons: int = 0
    completed_seasons: int = 0
    skipped_seasons: int = 0
    failed_seasons: List[str] = field(default_factory=list)
    total_fixtures: int = 0

    def pct_done(self) -> float:
        if self.total_seasons == 0:
            return 0.0
        return (
            100.0 * (self.completed_seasons + self.skipped_seasons) / self.total_seasons
        )

    def log_summary(self) -> None:
        logger.info(
            "HistoricalDownloader done — leagues=%d seasons=%d "
            "(completed=%d skipped=%d failed=%d) fixtures=%d",
            self.total_leagues,
            self.total_seasons,
            self.completed_seasons,
            self.skipped_seasons,
            len(self.failed_seasons),
            self.total_fixtures,
        )
        for fail in self.failed_seasons:
            logger.warning("  failed: %s", fail)


# ---------------------------------------------------------------------------
# HistoricalDownloader
# ---------------------------------------------------------------------------


class HistoricalDownloader:
    """
    Downloads all historical fixtures for BETFAIR_LEAGUES season by season.

    Parameters
    ----------
    collector        : ApiFootballCollector instance (default: new with 10 req/min)
    repo             : FixturesRepository instance (default: new)
    bootstrap_from   : first season year to download (default: 2008)
    current_season   : last season year to include (default: current calendar year)
    leagues          : dict of {league_id: name} (default: BETFAIR_LEAGUES)
    """

    def __init__(
        self,
        collector: Optional[ApiFootballCollector] = None,
        repo: Optional[FixturesRepository] = None,
        bootstrap_from: int = BOOTSTRAP_FROM,
        current_season: Optional[int] = None,
        leagues: Optional[Dict[int, str]] = None,
    ) -> None:
        self._collector = collector or ApiFootballCollector(requests_per_minute=10.0)
        self._repo = repo or FixturesRepository()
        self._bootstrap_from = bootstrap_from
        self._current_season = current_season or datetime.now(timezone.utc).year
        self._leagues = leagues if leagues is not None else BETFAIR_LEAGUES

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, resume: bool = True) -> DownloadProgress:
        """
        Download all historical fixtures.

        Parameters
        ----------
        resume : if True (default), already-downloaded league/season pairs
                 are skipped; if False, every season is re-downloaded.
        """
        target_seasons = list(range(self._bootstrap_from, self._current_season + 1))
        progress = DownloadProgress(
            total_leagues=len(self._leagues),
            total_seasons=len(self._leagues) * len(target_seasons),
        )

        for league_id, league_name in self._leagues.items():
            self._download_league(
                league_id=league_id,
                league_name=league_name,
                target_seasons=target_seasons,
                resume=resume,
                progress=progress,
            )

        progress.log_summary()
        return progress

    def run_league(
        self,
        league_id: int,
        resume: bool = True,
    ) -> DownloadProgress:
        """Download a single league (all seasons). Useful for targeted re-fetches."""
        league_name = self._leagues.get(league_id, f"league_{league_id}")
        target_seasons = list(range(self._bootstrap_from, self._current_season + 1))
        progress = DownloadProgress(
            total_leagues=1,
            total_seasons=len(target_seasons),
        )
        self._download_league(
            league_id=league_id,
            league_name=league_name,
            target_seasons=target_seasons,
            resume=resume,
            progress=progress,
        )
        progress.log_summary()
        return progress

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_league(
        self,
        league_id: int,
        league_name: str,
        target_seasons: List[int],
        resume: bool,
        progress: DownloadProgress,
    ) -> None:
        logger.info(
            "HistoricalDownloader: starting league %d (%s) | seasons %d–%d",
            league_id,
            league_name,
            target_seasons[0],
            target_seasons[-1],
        )

        for season in target_seasons:
            if resume and self._repo.is_season_downloaded(league_id, season):
                logger.debug(
                    "  skip league=%d season=%d (already downloaded)", league_id, season
                )
                progress.skipped_seasons += 1
                continue

            self._download_season(
                league_id=league_id,
                league_name=league_name,
                season=season,
                progress=progress,
            )

            # Pause between seasons to avoid bursting the rate limiter
            time.sleep(_INTER_SEASON_PAUSE)

    def _download_season(
        self,
        league_id: int,
        league_name: str,
        season: int,
        progress: DownloadProgress,
    ) -> None:
        logger.info(
            "  downloading league=%d (%s) season=%d …",
            league_id,
            league_name,
            season,
        )
        try:
            fixtures = self._collector.get_league_season(league_id, season)
        except Exception as exc:  # noqa: BLE001
            tag = f"league={league_id} season={season}: {exc}"
            progress.failed_seasons.append(tag)
            logger.exception("  FAILED %s", tag)
            return

        if not fixtures:
            # API returned zero results — league probably didn't exist yet this season.
            # Still register it to avoid re-fetching empty seasons on resume.
            logger.info(
                "  league=%d season=%d returned 0 fixtures (skipping registry)",
                league_id,
                season,
            )
            progress.completed_seasons += 1
            return

        self._repo.upsert_fixtures_bulk(fixtures)
        self._repo.mark_season_downloaded(league_id, season, n_fixtures=len(fixtures))

        progress.completed_seasons += 1
        progress.total_fixtures += len(fixtures)

        logger.info(
            "  league=%d (%s) season=%d — %d fixtures saved (%.1f%% done)",
            league_id,
            league_name,
            season,
            len(fixtures),
            progress.pct_done(),
        )

    def available_seasons_for_league(self, league_id: int) -> List[int]:
        """
        Query the API for the list of seasons where a league has coverage.
        Useful to narrow the download window before starting the bootstrap.
        """
        return self._collector.get_available_seasons(league_id)

    def print_status(self) -> None:
        """Log how many seasons have been downloaded per league."""
        for league_id, league_name in sorted(self._leagues.items()):
            done = self._repo.downloaded_seasons_for_league(league_id)
            total = self._current_season - self._bootstrap_from + 1
            logger.info(
                "  [%d] %-38s  %d/%d seasons downloaded",
                league_id,
                league_name,
                len(done),
                total,
            )
