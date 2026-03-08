from quant.clv.bet_journal import BetJournal


def test_bet_journal_add_and_list(tmp_path):
    journal = BetJournal(str(tmp_path / "bet_journal.json"))

    journal.add_bet(
        {
            "fixture_id": "1",
            "market": "home",
            "stake": 10,
            "bookmaker_odds": 2.00,
        }
    )

    rows = journal.list_bets()

    assert len(rows) == 1
    assert rows[0]["fixture_id"] == "1"
