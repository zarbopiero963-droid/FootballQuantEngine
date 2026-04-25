-- ---------------------------------------------------------------------------
-- Core fixtures table (extended with halftime, venue, referee, league_id)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fixtures(
    fixture_id   INTEGER PRIMARY KEY,
    league_id    INTEGER,
    league       TEXT,
    season       INTEGER,
    home         TEXT,
    away         TEXT,
    match_date   TEXT,
    home_goals   INTEGER,
    away_goals   INTEGER,
    ht_home      INTEGER,
    ht_away      INTEGER,
    status       TEXT,
    venue        TEXT,
    referee      TEXT,
    elapsed      INTEGER,
    updated_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_fixtures_status       ON fixtures(status);
CREATE INDEX IF NOT EXISTS idx_fixtures_match_date   ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_league_season ON fixtures(league_id, season);

-- ---------------------------------------------------------------------------
-- Per-fixture statistics: corners, shots, cards, possession
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fixture_stats(
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id      INTEGER NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    home_corners    INTEGER,
    away_corners    INTEGER,
    home_shots      INTEGER,
    away_shots      INTEGER,
    home_shots_on   INTEGER,
    away_shots_on   INTEGER,
    home_possession REAL,
    away_possession REAL,
    home_yellow     INTEGER,
    away_yellow     INTEGER,
    home_red        INTEGER,
    away_red        INTEGER,
    home_xg         REAL,
    away_xg         REAL,
    updated_at      TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fixture_stats_unique ON fixture_stats(fixture_id);

-- ---------------------------------------------------------------------------
-- Tracks which (league, season) combos have been fully bootstrapped
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS downloaded_seasons(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id     INTEGER NOT NULL,
    season        INTEGER NOT NULL,
    n_fixtures    INTEGER NOT NULL DEFAULT 0,
    downloaded_at TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dl_seasons_unique ON downloaded_seasons(league_id, season);

-- ---------------------------------------------------------------------------
-- Multi-market predictions: one row per fixture × market
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS multi_market_predictions(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id    INTEGER NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    market        TEXT    NOT NULL,   -- '1x2','ou25','ou15','ou35','btts','ou_corners','ht_ou15','cs'
    selection     TEXT    NOT NULL,   -- 'home','draw','away','over','under','yes','no','1-0', ...
    probability   REAL    NOT NULL,
    odds_model    REAL    NOT NULL,   -- fair odds = 1 / probability
    odds_market   REAL,               -- bookmaker odds (nullable until fetched)
    edge          REAL,
    ev            REAL,
    kelly         REAL,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mmp_fixture  ON multi_market_predictions(fixture_id);
CREATE INDEX IF NOT EXISTS idx_mmp_market   ON multi_market_predictions(market);
CREATE INDEX IF NOT EXISTS idx_mmp_created  ON multi_market_predictions(created_at);

-- ---------------------------------------------------------------------------
-- Odds history (kept for backtest temporal guard)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_history(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(fixture_id),
    market     TEXT    DEFAULT '1x2',
    timestamp  TEXT,
    home_odds  REAL,
    draw_odds  REAL,
    away_odds  REAL
);

CREATE INDEX IF NOT EXISTS idx_odds_history_fixture ON odds_history(fixture_id);

-- ---------------------------------------------------------------------------
-- Legacy 1X2 predictions (kept for backtest compatibility)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(fixture_id),
    model      TEXT,
    home_prob  REAL,
    draw_prob  REAL,
    away_prob  REAL,
    edge       REAL,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_fixture ON predictions(fixture_id);
