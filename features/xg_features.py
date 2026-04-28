from __future__ import annotations

from typing import Any

import pandas as pd


def extract_xg(team_data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for match in team_data:
        rows.append(
            {
                "date": match.get("datetime"),
                "xg": float(match.get("xG", 0)),
                "xga": float(match.get("xGA", 0)),
                "result": match.get("result"),
            }
        )

    return pd.DataFrame(rows)
