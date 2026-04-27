import json
import os

SETTINGS_FILE = "settings.json"


class Settings:
    def __init__(self):

        self.api_football_key = ""
        self.telegram_token = ""
        self.telegram_chat_id = ""
        self.league_id = 135  # API-Football numeric ID (default: Serie A)
        self.season = 2024
        self.openweather_key = ""  # OpenWeatherMap API key (free tier)


def load_settings():

    data: dict = {}

    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            data = json.load(f)

    s = Settings()

    # Env vars take precedence over the settings file so secrets don't have
    # to be stored on disk (set API_FOOTBALL_KEY, ODDS_API_KEY, etc.).
    s.api_football_key = os.getenv("API_FOOTBALL_KEY") or data.get(
        "api_football_key", ""
    )
    s.telegram_token = os.getenv("TELEGRAM_TOKEN") or data.get("telegram_token", "")
    s.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID") or data.get(
        "telegram_chat_id", ""
    )
    s.league_id = int(data.get("league_id", 135))
    s.season = int(data.get("season", 2024))
    s.openweather_key = os.getenv("OPENWEATHER_KEY") or data.get("openweather_key", "")

    return s


def save_settings(settings):
    # Read whatever is already on disk so we can preserve existing credentials.
    existing: dict = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Non-sensitive UI preferences are always written.
    data = {
        "league_id": settings.league_id,
        "season": settings.season,
    }

    # Preserve credentials that are already stored in the file but are NOT
    # currently supplied via environment variables (env vars make the file
    # copy redundant and should not be duplicated on disk).
    _SECRET_ENV: dict = {
        "api_football_key": "API_FOOTBALL_KEY",
        "telegram_token": "TELEGRAM_TOKEN",
        "telegram_chat_id": "TELEGRAM_CHAT_ID",
        "openweather_key": "OPENWEATHER_KEY",
    }
    for field, env_var in _SECRET_ENV.items():
        if not os.getenv(env_var) and existing.get(field):
            data[field] = existing[field]

    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
