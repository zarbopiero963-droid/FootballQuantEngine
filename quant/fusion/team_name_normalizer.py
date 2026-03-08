from __future__ import annotations

import re
import unicodedata


class TeamNameNormalizer:

    DEFAULT_ALIASES = {
        "inter milan": "inter",
        "internazionale": "inter",
        "ac milan": "milan",
        "juve": "juventus",
        "man city": "manchester city",
        "man utd": "manchester united",
        "psg": "paris saint germain",
        "atletico madrid": "atletico",
        "ath madrid": "atletico",
        "bayern munich": "bayern",
        "bayern munchen": "bayern",
    }

    def __init__(self, aliases: dict | None = None):
        self.aliases = dict(self.DEFAULT_ALIASES)
        if aliases:
            for key, value in aliases.items():
                self.aliases[self.normalize_raw(key)] = self.normalize_raw(value)

    def normalize_raw(self, value: str) -> str:
        value = str(value or "").strip().lower()
        value = unicodedata.normalize("NFKD", value)
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return self.aliases.get(value, value)

    def normalize(self, value: str) -> str:
        return self.normalize_raw(value)
