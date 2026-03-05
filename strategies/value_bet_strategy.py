class ValueBetStrategy:

    def find_value_bets(self, predictions, odds_data, edge_detector):

        results = []

        for match_id in predictions:

            prediction = predictions[match_id]

            if match_id not in odds_data:
                continue

            odds = odds_data[match_id]

            home_prob = prediction.get("home_win")
            draw_prob = prediction.get("draw")
            away_prob = prediction.get("away_win")

            home_odds = odds.get("home")
            draw_odds = odds.get("draw")
            away_odds = odds.get("away")

            if edge_detector.is_value_bet(home_prob, home_odds):
                results.append(
                    {
                        "match_id": match_id,
                        "market": "home",
                        "probability": home_prob,
                        "odds": home_odds,
                    }
                )

            if edge_detector.is_value_bet(draw_prob, draw_odds):
                results.append(
                    {
                        "match_id": match_id,
                        "market": "draw",
                        "probability": draw_prob,
                        "odds": draw_odds,
                    }
                )

            if edge_detector.is_value_bet(away_prob, away_odds):
                results.append(
                    {
                        "match_id": match_id,
                        "market": "away",
                        "probability": away_prob,
                        "odds": away_odds,
                    }
                )

        return results
