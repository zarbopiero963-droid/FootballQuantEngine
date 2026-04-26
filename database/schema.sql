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
    ht_home      INTEGER,          -- added in schema v2
    ht_away      INTEGER,          -- added in schema v2
    status       TEXT,
    venue        TEXT,             -- added in schema v2
    referee      TEXT,             -- added in schema v2
    elapsed      INTEGER,          -- added in schema v2
    updated_at   TEXT              -- added in schema v2
);

CREATE INDEX IF NOT EXISTS idx_fixtures_status        ON fixtures(status);
CREATE INDEX IF NOT EXISTS idx_fixtures_match_date    ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_league_season ON fixtures(league_id, season);

-- ---------------------------------------------------------------------------
-- Snapshot table: one row per fixture per calendar day; tracks status history
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS snapshots(

    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT    NOT NULL,
    fixture_id    INTEGER NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    status        TEXT    NOT NULL,
    home_goals    INTEGER,
    away_goals    INTEGER,
    home_odds     REAL,
    draw_odds     REAL,
    away_odds     REAL,
    created_at    TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_unique
    ON snapshots(snapshot_date, fixture_id);

CREATE INDEX IF NOT EXISTS idx_snapshots_date    ON snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_fixture ON snapshots(fixture_id);

-- ---------------------------------------------------------------------------
-- Per-fixture statistics: corners, shots, cards, possession, xG
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_dl_seasons_unique
    ON downloaded_seasons(league_id, season);

-- ---------------------------------------------------------------------------
-- Multi-market predictions: one row per fixture × market × selection
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS multi_market_predictions(

    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id    INTEGER NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
    market        TEXT    NOT NULL,
    selection     TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    odds_model    REAL    NOT NULL,
    odds_market   REAL,
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
    market     TEXT    DEFAULT '1x2',   -- added in schema v2
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

-- ---------------------------------------------------------------------------
-- Schema migration registry — tracks which migrations have been applied.
-- Managed exclusively by database/db_manager.py; do not write directly.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations(

    version     INTEGER PRIMARY KEY,
    description TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
