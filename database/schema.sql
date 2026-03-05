CREATE TABLE IF NOT EXISTS fixtures(

    fixture_id INTEGER PRIMARY KEY,
    league TEXT,
    season INTEGER,
    home TEXT,
    away TEXT,
    match_date TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    status TEXT
);

CREATE TABLE IF NOT EXISTS odds_history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    timestamp TEXT,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL
);

CREATE TABLE IF NOT EXISTS predictions(

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER,
    model TEXT,
    home_prob REAL,
    draw_prob REAL,
    away_prob REAL,
    edge REAL,
    created_at TEXT
);
