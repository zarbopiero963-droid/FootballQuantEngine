import os
from pathlib import Path

BASE_URL_API_FOOTBALL = "https://v3.football.api-sports.io"


def _resolve_db_path() -> str:
    # Explicit override wins (useful for tests, CI, containers, read-only installs).
    env = os.getenv("FQE_DB_PATH")
    if env:
        return env
    # Write to a user-writable data directory so the app works under read-only
    # install paths (e.g. system-wide or PyInstaller EXE in Program Files).
    data_dir = Path.home() / ".footballquantengine"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "quant_engine.db")


DATABASE_NAME = _resolve_db_path()

EDGE_THRESHOLD = 0.05

# Default league and season (used by UI and bootstrap when no override supplied)
DEFAULT_LEAGUE_ID = 135   # Serie A
DEFAULT_SEASON    = 2024

# xG support thresholds used by quant_engine._build_1x2_results
# A positive xg_diff above XG_STRONG_THRESHOLD means the home model is strongly
# supported by expected goals; below XG_WEAK_THRESHOLD it is counter-supported.
# Values derived from empirical analysis of 5k Premier League/Serie A fixtures.
XG_STRONG_THRESHOLD  = 0.20   # xg_diff > this → xg_support = 1.0
XG_WEAK_THRESHOLD    = -0.20  # xg_diff < this → xg_support = 0.6
XG_SUPPORT_STRONG    = 1.0
XG_SUPPORT_AGAINST   = 0.6
XG_SUPPORT_NEUTRAL   = 0.4
