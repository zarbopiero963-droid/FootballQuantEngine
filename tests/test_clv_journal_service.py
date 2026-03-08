from quant.clv.clv_journal_service import CLVJournalService


def test_clv_journal_service_record_and_settle(tmp_path):
    service = CLVJournalService()
    service.journal.path = str(tmp_path / "bet_journal.json")
    service.journal.__init__(service.journal.path)

    service.record_bets(
        [
            {
                "fixture_id": "1",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "market": "home",
                "probability": 0.58,
                "fair_odds": 1.72,
                "bookmaker_odds": 2.05,
                "ev": 0.18,
                "confidence": 0.72,
                "agreement": 0.68,
                "decision": "BET",
                "suggested_stake": 10.0,
            }
        ]
    )

    updated = service.settle_bet("1", "home", 1.95, "WIN")

    assert updated is not None
    assert updated["status"] == "SETTLED"
    assert "clv_abs" in updated
    assert "pnl" in updated
