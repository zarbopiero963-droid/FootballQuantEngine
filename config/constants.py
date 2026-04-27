from pathlib import Path

BASE_URL_API_FOOTBALL = "https://v3.football.api-sports.io"

# Absolute path so the DB is always found regardless of cwd
DATABASE_NAME = str(Path(__file__).resolve().parent.parent / "quant_engine.db")

EDGE_THRESHOLD = 0.05

# Default league and season (used by UI and bootstrap when no override supplied)
DEFAULT_LEAGUE_ID = 135   # Serie A
DEFAULT_SEASON    = 2024
