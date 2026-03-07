def to_records(results):
    records = []

    for item in results:
        records.append(
            {
                "fixture_id": item.fixture_id,
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
