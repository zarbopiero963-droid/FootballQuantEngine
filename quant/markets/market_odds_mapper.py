from __future__ import annotations


class MarketOddsMapper:

    def odds_for_market(self, odds_payload: dict, market: str) -> float:
        odds_payload = odds_payload or {}
        market = str(market)

        mapping = {
            "home": odds_payload.get("home"),
            "draw": odds_payload.get("draw"),
            "away": odds_payload.get("away"),
            "over_25": odds_payload.get("over_25"),
            "under_25": odds_payload.get("under_25"),
            "btts_yes": odds_payload.get("btts_yes"),
            "btts_no": odds_payload.get("btts_no"),
        }

        value = mapping.get(market)
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0
