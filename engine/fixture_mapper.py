def map_api_football_response_to_fixtures(fixtures_response):

    fixtures = []

    for match in fixtures_response.get("response", []):

        fixture = match.get("fixture", {})
        league = match.get("league", {})
        teams = match.get("teams", {})
        goals = match.get("goals", {})

        fixture_id = fixture.get("id")
        match_date = fixture.get("date")
        status = fixture.get("status", {}).get("short")

        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")

        if fixture_id is None or not home or not away:
            continue

        fixtures.append(
            {
                "fixture_id": fixture_id,
                "league": league.get("name"),
                "season": league.get("season"),
                "home": home,
                "away": away,
                "match_date": match_date,
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": status,
            }
        )

    return fixtures
