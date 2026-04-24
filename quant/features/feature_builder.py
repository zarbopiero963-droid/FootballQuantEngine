from __future__ import annotations

from datetime import datetime, timezone

from quant.features.advanced_stats_features import AdvancedStatsFeatureBuilder


class FeatureBuilder:
    def __init__(
        self,
        elo_engine,
        dc_engine,
        form_engine,
        h2h_engine=None,
        momentum_engine=None,
        rest_engine=None,
        standings_engine=None,
        injury_engine=None,
        referee_engine=None,
        weather_engine=None,
        advanced_stats=None,
    ):
        self.elo_engine = elo_engine
        self.dc_engine = dc_engine
        self.form_engine = form_engine
        self.h2h_engine = h2h_engine
        self.momentum_engine = momentum_engine
        self.rest_engine = rest_engine
        self.standings_engine = standings_engine
        self.injury_engine = injury_engine
        self.referee_engine = referee_engine
        self.weather_engine = weather_engine
        self.advanced_builder = AdvancedStatsFeatureBuilder(advanced_stats or {})

    def _parse_fixture_dt(self, fixture: dict) -> datetime | None:
        date_str = fixture.get("match_date") or fixture.get("date")
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def build_fixture_features(self, fixture: dict) -> dict:
        home = fixture["home_team"]
        away = fixture["away_team"]
        referee = fixture.get("referee")
        fixture_dt = self._parse_fixture_dt(fixture)

        # --- Lambda modifiers (rest + injuries + weather + referee) ---
        home_mod = away_mod = 1.0

        if self.rest_engine:
            rm_h, rm_a = self.rest_engine.get_lambda_modifiers(home, away, fixture_dt)
            home_mod *= rm_h
            away_mod *= rm_a

        if self.injury_engine:
            im_h, im_a = self.injury_engine.get_lambda_modifiers(home, away)
            home_mod *= im_h
            away_mod *= im_a

        if self.referee_engine:
            ref_h, ref_a = self.referee_engine.get_lambda_modifiers(referee)
            home_mod *= ref_h
            away_mod *= ref_a

        # Apply modifiers to DC expected goals
        lh_base, la_base = self.dc_engine.expected_goals(home, away)
        lh = max(0.10, lh_base * home_mod)
        la = max(0.10, la_base * away_mod)

        weather_factor = 1.0
        if self.weather_engine:
            weather = fixture.get("weather")
            weather_factor = self.weather_engine.get_lambda_modifier(weather)
        lh *= weather_factor
        la *= weather_factor

        # Temporarily override DC lambdas via a lightweight wrapper
        dc_probs = self.dc_engine.probabilities_1x2_from_lambdas(lh, la)

        # --- Diff signals ---
        elo_diff = self.elo_engine.get_elo_diff(home, away)
        form_diff = self.form_engine.get_form_diff(home, away)

        h2h_diff = self.h2h_engine.get_h2h_diff(home, away) if self.h2h_engine else 0.0
        momentum_diff = (
            self.momentum_engine.get_momentum_diff(home, away)
            if self.momentum_engine
            else 0.0
        )
        motivation_diff = (
            self.standings_engine.get_motivation_diff(home, away)
            if self.standings_engine
            else 0.0
        )
        rest_diff = (
            self.rest_engine.get_rest_diff(home, away, fixture_dt)
            if self.rest_engine
            else 0.0
        )
        injury_diff = (
            self.injury_engine.get_injury_diff(home, away)
            if self.injury_engine
            else 0.0
        )

        advanced = self.advanced_builder.build(home, away)

        features = {
            "fixture_id": fixture["fixture_id"],
            "home_team": home,
            "away_team": away,
            "referee": referee,
            "match_date": fixture.get("match_date", ""),
            # Base lambdas (after modifiers)
            "expected_goals_home": round(lh, 4),
            "expected_goals_away": round(la, 4),
            "goal_expectancy_diff": round(lh - la, 4),
            # Pre-computed DC probs (passed to blend model)
            "dc_probs": dc_probs,
            # Diff signals
            "elo_diff": elo_diff,
            "form_diff": form_diff,
            "h2h_diff": h2h_diff,
            "momentum_diff": momentum_diff,
            "motivation_diff": motivation_diff,
            "rest_diff": rest_diff,
            "injury_diff": injury_diff,
        }
        features.update(advanced)
        return features

    def build_many(self, fixtures: list[dict]) -> list[dict]:
        return [self.build_fixture_features(f) for f in fixtures]
