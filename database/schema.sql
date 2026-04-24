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
    status       TEXT,
    venue        TEXT,
    referee      TEXT,
    elapsed      INTEGER,
    updated_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_fixtures_status     ON fixtures(status);
CREATE INDEX IF NOT EXISTS idx_fixtures_match_date ON fixtures(match_date);
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

CREATE INDEX IF NOT EXISTS idx_snapshots_date      ON snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_fixture   ON snapshots(fixture_id);

-- ---------------------------------------------------------------------------
-- Downloaded-seasons registry: enables resume on interrupted bootstraps
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
-- Existing tables (kept for backwards compatibility)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS odds_history(

    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id INTEGER NOT NULL REFERENCES fixtures(fixture_id),
    timestamp  TEXT,
    home_odds  REAL,
    draw_odds  REAL,
    away_odds  REAL
);

CREATE INDEX IF NOT EXISTS idx_odds_history_fixture_id ON odds_history(fixture_id);

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

CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id ON predictions(fixture_id);
