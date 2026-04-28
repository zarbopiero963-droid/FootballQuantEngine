from __future__ import annotations

import json
import os


class ManualContextManager:

    def __init__(self, filepath="manual_context.json"):

        self.filepath = filepath

    def load(self):

        if not os.path.exists(self.filepath):
            return {
                "lineups": {},
                "injuries": {},
                "weather": {},
            }

        with open(self.filepath, encoding="utf-8") as f:
            return json.load(f)

    def save(self, data):

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_team_context(self, team_name):

        data = self.load()

        return {
            "lineup": data.get("lineups", {}).get(team_name, []),
            "injuries": data.get("injuries", {}).get(team_name, []),
            "weather": data.get("weather", {}).get(team_name, {}),
        }
