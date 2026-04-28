from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"

# Mapping: (settings attribute, env var name, file key)
_SECRET_FIELDS: list[tuple[str, str, str]] = [
    ("api_football_key", "API_FOOTBALL_KEY", "api_football_key"),
    ("telegram_token", "TELEGRAM_TOKEN", "telegram_token"),
    ("telegram_chat_id", "TELEGRAM_CHAT_ID", "telegram_chat_id"),
    ("openweather_key", "OPENWEATHER_KEY", "openweather_key"),
]


class Settings:
    def __init__(self) -> None:
        self.api_football_key: str = ""
        self.telegram_token: str = ""
        self.telegram_chat_id: str = ""
        self.league_id: int = 135  # API-Football numeric ID (default: Serie A)
        self.season: int = 2024
        self.openweather_key: str = ""


def load_settings() -> Settings:
    data: dict = {}

    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            data = json.load(f)

    s = Settings()
    file_sourced: list[str] = []

    for attr, env_var, file_key in _SECRET_FIELDS:
        env_val = os.getenv(env_var)
        if env_val:
            setattr(s, attr, env_val)
        else:
            file_val = data.get(file_key, "")
            if file_val:
                file_sourced.append(env_var)
            setattr(s, attr, file_val)

    if file_sourced:
        logger.warning(
            "Credentials sourced from settings.json (not env vars): %s. "
            "Set the corresponding environment variables to avoid storing "
            "secrets on disk: %s",
            ", ".join(file_sourced),
            ", ".join(file_sourced),
        )

    s.league_id = int(data.get("league_id", 135))
    s.season = int(data.get("season", 2024))

    return s


def save_settings(settings: Settings) -> None:
    # Only non-sensitive preferences are ever written to disk.
    # Credentials (API keys, tokens) must be supplied via environment variables
    # and are NEVER persisted to the settings file.
    data = {
        "league_id": settings.league_id,
        "season": settings.season,
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
