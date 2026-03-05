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

    if not os.path.exists(SETTINGS_FILE):
        return Settings()

    with open(SETTINGS_FILE) as f:
        data = json.load(f)

    s = Settings()

    s.api_football_key = data.get("api_football_key", "")
    s.odds_api_key = data.get("odds_api_key", "")
    s.telegram_token = data.get("telegram_token", "")
    s.telegram_chat_id = data.get("telegram_chat_id", "")

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
