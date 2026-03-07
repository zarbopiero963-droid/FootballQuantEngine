from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class GuardResult:
    passed: bool
    checks: list[dict]
    summary: dict


class QuantEngineGuard:

    REQUIRED_FIELDS = {
        "fixture_id",
        "home_team",
        "away_team",
        "market",
        "probability",
        "fair_odds",
        "bookmaker_odds",
        "market_edge",
        "model_edge",
        "confidence",
        "agreement",
        "decision",
    }

    VALID_DECISIONS = {"BET", "WATCHLIST", "NO_BET"}
    VALID_MARKETS = {"home", "draw", "away"}

    def validate_records(self, records: list[dict]) -> GuardResult:
        checks = []

        checks.append(
            {
                "name": "records_is_list",
                "passed": isinstance(records, list),
                "value": type(records).__name__,
            }
        )

        record_count = len(records) if isinstance(records, list) else 0
        checks.append(
            {
                "name": "records_not_empty",
                "passed": record_count > 0,
                "value": record_count,
            }
        )

        if not isinstance(records, list) or not records:
            return GuardResult(
                passed=all(check["passed"] for check in checks),
                checks=checks,
                summary={
                    "record_count": record_count,
                    "bet_count": 0,
                    "watchlist_count": 0,
                    "no_bet_count": 0,
                    "avg_probability": 0.0,
                    "avg_confidence": 0.0,
                    "avg_agreement": 0.0,
                    "max_market_edge": 0.0,
                },
            )

        first = records[0]
        checks.append(
            {
                "name": "required_fields_present",
                "passed": self.REQUIRED_FIELDS.issubset(set(first.keys())),
                "value": sorted(list(set(first.keys()))),
            }
        )

        missing_any = False
        invalid_decision_count = 0
        invalid_market_count = 0
        invalid_probability_count = 0
        invalid_confidence_count = 0
        invalid_agreement_count = 0
        invalid_odds_count = 0

        probabilities = []
        confidences = []
        agreements = []
        market_edges = []

        decision_counts = {"BET": 0, "WATCHLIST": 0, "NO_BET": 0}

        for record in records:
            if not self.REQUIRED_FIELDS.issubset(set(record.keys())):
                missing_any = True

            decision = str(record.get("decision", ""))
            market = str(record.get("market", ""))

            if decision not in self.VALID_DECISIONS:
                invalid_decision_count += 1
            else:
                decision_counts[decision] += 1

            if market not in self.VALID_MARKETS:
                invalid_market_count += 1

            try:
                probability = float(record.get("probability", 0.0))
                probabilities.append(probability)
                if probability < 0.0 or probability > 1.0:
                    invalid_probability_count += 1
            except Exception:
                invalid_probability_count += 1

            try:
                confidence = float(record.get("confidence", 0.0))
                confidences.append(confidence)
                if confidence < 0.0 or confidence > 1.0:
                    invalid_confidence_count += 1
            except Exception:
                invalid_confidence_count += 1

            try:
                agreement = float(record.get("agreement", 0.0))
                agreements.append(agreement)
                if agreement < 0.0 or agreement > 1.0:
                    invalid_agreement_count += 1
            except Exception:
                invalid_agreement_count += 1

            try:
                fair_odds = float(record.get("fair_odds", 0.0))
                bookmaker_odds = float(record.get("bookmaker_odds", 0.0))
                if fair_odds <= 0.0 or bookmaker_odds <= 0.0:
                    invalid_odds_count += 1
            except Exception:
                invalid_odds_count += 1

            try:
                market_edges.append(float(record.get("market_edge", 0.0)))
            except Exception:
                market_edges.append(0.0)

        checks.append(
            {
                "name": "all_required_fields_present_in_all_records",
                "passed": not missing_any,
                "value": not missing_any,
            }
        )

        checks.append(
            {
                "name": "valid_decisions",
                "passed": invalid_decision_count == 0,
                "value": invalid_decision_count,
            }
        )

        checks.append(
            {
                "name": "valid_markets",
                "passed": invalid_market_count == 0,
                "value": invalid_market_count,
            }
        )

        checks.append(
            {
                "name": "probabilities_in_range",
                "passed": invalid_probability_count == 0,
                "value": invalid_probability_count,
            }
        )

        checks.append(
            {
                "name": "confidences_in_range",
                "passed": invalid_confidence_count == 0,
                "value": invalid_confidence_count,
            }
        )

        checks.append(
            {
                "name": "agreements_in_range",
                "passed": invalid_agreement_count == 0,
                "value": invalid_agreement_count,
            }
        )

        checks.append(
            {
                "name": "odds_positive",
                "passed": invalid_odds_count == 0,
                "value": invalid_odds_count,
            }
        )

        checks.append(
            {
                "name": "has_any_decision_bucket",
                "passed": sum(decision_counts.values()) == len(records),
                "value": dict(decision_counts),
            }
        )

        summary = {
            "record_count": len(records),
            "bet_count": decision_counts["BET"],
            "watchlist_count": decision_counts["WATCHLIST"],
            "no_bet_count": decision_counts["NO_BET"],
            "avg_probability": round(
                sum(probabilities) / max(len(probabilities), 1), 6
            ),
            "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 6),
            "avg_agreement": round(sum(agreements) / max(len(agreements), 1), 6),
            "max_market_edge": round(max(market_edges) if market_edges else 0.0, 6),
        }

        return GuardResult(
            passed=all(check["passed"] for check in checks),
            checks=checks,
            summary=summary,
        )

    def save_report(
        self, result: GuardResult, path: str = "outputs/ci/quant_guard_report.json"
    ) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "passed": result.passed,
            "checks": result.checks,
            "summary": result.summary,
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return path
