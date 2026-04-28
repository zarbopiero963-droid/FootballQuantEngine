from __future__ import annotations

from abc import ABC, abstractmethod


class BaseFixturesProvider(ABC):

    @abstractmethod
    def get_completed_matches(self, league=None, season=None):
        raise NotImplementedError

    @abstractmethod
    def get_upcoming_matches(self, league=None, season=None):
        raise NotImplementedError


class BaseOddsProvider(ABC):

    @abstractmethod
    def get_prematch_odds(self, fixture_ids):
        raise NotImplementedError


class BaseAdvancedStatsProvider(ABC):

    @abstractmethod
    def get_team_advanced_stats(self, league=None, season=None):
        raise NotImplementedError
