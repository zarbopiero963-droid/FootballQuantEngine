from __future__ import annotations

BETFAIR_EXCHANGE_ITALIA_LEAGUES = {
    "Serie A",
    "Serie B",
    "Premier League",
    "Championship",
    "League One",
    "League Two",
    "La Liga",
    "Segunda Division",
    "Bundesliga",
    "2. Bundesliga",
    "Ligue 1",
    "Ligue 2",
    "Eredivisie",
    "Primeira Liga",
    "Super Lig",
    "Belgian Pro League",
    "Serie A Brazil",
    "Primera Division",
}


def is_betfair_league(league_name):

    if not league_name:
        return False

    return league_name in BETFAIR_EXCHANGE_ITALIA_LEAGUES
