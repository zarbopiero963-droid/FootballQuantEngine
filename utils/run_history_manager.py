import json
import os
from datetime import datetime


class RunHistoryManager:

    def __init__(self, path="data/run_history.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def add_run(self, metrics):
        with open(self.path, "r") as f:
            data = json.load(f)

        metrics["timestamp"] = datetime.utcnow().isoformat()

        data.append(metrics)

        with open(self.path, "w") as f:
            json.dump(data[-100:], f, indent=2)

    def get_runs(self):
        with open(self.path, "r") as f:
            return json.load(f)
