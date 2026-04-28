from __future__ import annotations

from quant.live.edge_stability_detector import EdgeStabilityDetector
from quant.live.league_performance_analyzer import LeaguePerformanceAnalyzer
from quant.live.market_optimizer import LeagueMarketPerformanceOptimizer


class IntegratedAnalytics:

    def __init__(self):
        self.league_analyzer = LeaguePerformanceAnalyzer()
        self.edge_detector = EdgeStabilityDetector()
        self.market_optimizer = LeagueMarketPerformanceOptimizer()

    def build(self, rows: list[dict]) -> dict:
        rows = rows or []

        return {
            "league_summary": self.league_analyzer.analyze(rows),
            "edge_summary": self.edge_detector.analyze(rows),
            "market_summary": self.market_optimizer.optimize(rows),
        }
