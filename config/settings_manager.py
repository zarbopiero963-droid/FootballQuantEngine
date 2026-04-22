import json
import os

SETTINGS_FILE = "settings.json"


class Settings:

    def __init__(self):

        self.api_football_key = ""
        self.odds_api_key = ""
        self.telegram_token = ""
        self.telegram_chat_id = ""


def load_settings():

    data: dict = {}

    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            data = json.load(f)

    s = Settings()

    # Env vars take precedence over the settings file so secrets don't have
    # to be stored on disk (set API_FOOTBALL_KEY, ODDS_API_KEY, etc.).
    s.api_football_key = (
        os.getenv("API_FOOTBALL_KEY") or data.get("api_football_key", "")
    )
    s.odds_api_key = (
        os.getenv("ODDS_API_KEY") or data.get("odds_api_key", "")
    )
    s.telegram_token = (
        os.getenv("TELEGRAM_TOKEN") or data.get("telegram_token", "")
    )
    s.telegram_chat_id = (
        os.getenv("TELEGRAM_CHAT_ID") or data.get("telegram_chat_id", "")
    )

    return s


def save_settings(settings):

    data = {
        "api_football_key": settings.api_football_key,
        "odds_api_key": settings.odds_api_key,
        "telegram_token": settings.telegram_token,
        "telegram_chat_id": settings.telegram_chat_id,
    }

    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
