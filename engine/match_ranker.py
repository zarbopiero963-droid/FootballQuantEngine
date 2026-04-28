from __future__ import annotations

from analytics.match_ranking import MatchRanking


class MatchRanker:

    def __init__(self):

        self.ranking = MatchRanking()

    def rank(self, value_bets):

        return self.ranking.rank(value_bets)
