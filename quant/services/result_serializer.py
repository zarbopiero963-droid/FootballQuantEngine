from __future__ import annotations

from typing import Any


def to_records(results: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for item in results:
        records.append(
            {
                "fixture_id": item.fixture_id,
                "league_id": item.league_id,
                "league_name": item.league_name,
                "league": item.league_name,
                "home_team": item.home_team,
                "away_team": item.away_team,
                "market": item.market,
                "probability": item.probability,
                "fair_odds": item.fair_odds,
                "bookmaker_odds": item.bookmaker_odds,
                "market_edge": item.market_edge,
                "model_edge": item.model_edge,
                "confidence": item.confidence,
                "agreement": item.agreement,
                "decision": item.decision,
            }
        )

    return records
