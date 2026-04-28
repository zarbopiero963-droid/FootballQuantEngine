from __future__ import annotations

from typing import Any


class AgreementEngine:

    def three_way_agreement(self, probs_list: list[dict[str, Any]]) -> float:
        if not probs_list:
            return 0.0

        def dispersion(values: list[float]) -> float:
            if not values:
                return 1.0
            mean_value = sum(values) / len(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            return variance**0.5

        home_values = [float(p.get("home_win", 0.0)) for p in probs_list]
        draw_values = [float(p.get("draw", 0.0)) for p in probs_list]
        away_values = [float(p.get("away_win", 0.0)) for p in probs_list]

        mean_disp = (
            dispersion(home_values) + dispersion(draw_values) + dispersion(away_values)
        ) / 3.0

        agreement = 1.0 - mean_disp * 3.0
        agreement = max(0.0, min(1.0, agreement))
        return agreement
