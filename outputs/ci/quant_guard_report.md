# Quant Engine Guard Report

- Generated: 2026-04-28 05:33:24 UTC
- Passed: **True**

## Summary

- record_count: `9`
- bet_count: `1`
- watchlist_count: `1`
- no_bet_count: `7`
- avg_probability: `0.333333`
- avg_confidence: `0.469505`
- avg_agreement: `0.799973`
- max_market_edge: `0.126967`

## Checks

| Check | Passed | Value |
|---|---:|---|
| records_is_list | True | list |
| records_not_empty | True | 9 |
| required_fields_present | True | ['agreement', 'away_team', 'bookmaker_odds', 'confidence', 'decision', 'fair_odds', 'fixture_id', 'home_team', 'league', 'league_id', 'league_name', 'market', 'market_edge', 'model_edge', 'probability'] |
| all_required_fields_present_in_all_records | True | True |
| valid_decisions | True | 0 |
| valid_markets | True | 0 |
| probabilities_in_range | True | 0 |
| confidences_in_range | True | 0 |
| agreements_in_range | True | 0 |
| odds_positive | True | 0 |
| has_any_decision_bucket | True | {'BET': 1, 'WATCHLIST': 1, 'NO_BET': 7} |