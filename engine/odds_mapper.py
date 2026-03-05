def map_odds_events_to_match_id(odds_events):

    odds_data = {}

    for event in odds_events:

        home = event.get("home_team")
        away = event.get("away_team")

        if not home or not away:
            continue

        match_id = f"{home}_vs_{away}"

        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        markets = bookmakers[0].get("markets", [])
        if not markets:
            continue

        outcomes = markets[0].get("outcomes", [])
        if not outcomes:
            continue

        odds_map = {}

        for outcome in outcomes:

            name = outcome.get("name")
            price = outcome.get("price")

            if name == home:
                odds_map["home"] = price
            elif name == away:
                odds_map["away"] = price
            elif isinstance(name, str) and name.lower() == "draw":
                odds_map["draw"] = price

        if "home" in odds_map and "draw" in odds_map and "away" in odds_map:
            odds_data[match_id] = odds_map

    return odds_data
