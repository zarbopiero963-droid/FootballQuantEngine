"""
Weather Engine for FootballQuantEngine
======================================

API
---
Uses the OpenWeatherMap *Current Weather* endpoint:
    https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric

All temperature values are in **Celsius** (``units=metric``).
Wind speed is returned in m/s and converted internally to km/h (× 3.6).
Precipitation is taken from the ``rain["1h"]`` field (mm/hour); defaults to 0.0
when the key is absent (no precipitation reported).

Impact Model
------------
Goal-rate multipliers are computed by accumulating additive adjustments to a
baseline of 1.0, then clamping to a minimum of 0.50:

    Wind adjustments (applied for the highest matching bracket only — additive
    with rain/temp, but only the single worst wind bracket is applied):
        wind_kph > 50  →  −0.20  ("gale-force wind")
        wind_kph > 35  →  −0.12  ("strong wind")
        wind_kph > 20  →  −0.06  ("moderate wind")

    Rain adjustments (highest matching bracket applied):
        rain_mm_h > 8   →  −0.18  ("heavy rain")
        rain_mm_h > 3   →  −0.10  ("moderate rain")
        rain_mm_h > 0.5 →  −0.05  ("light rain")

    Temperature adjustments (all matching brackets are additive):
        temp_c < 0  →  −0.05  ("freezing")
        temp_c < 2  →  −0.03  ("near-freezing")

    multiplier        = max(0.50, 1.0 + sum_of_adjustments)
    goals_adjustment  = (multiplier − 1.0) × 2.5

The resulting ``lambda_multiplier`` is applied symmetrically to both the home
and away Poisson lambdas, with a per-lambda minimum of 0.10.

Units
-----
* Temperature : Celsius
* Wind speed  : km/h  (converted from OWM m/s)
* Precipitation: mm/hour

Usage Example
-------------
>>> from engine.weather_engine import build_engine
>>> engine = build_engine("YOUR_OWM_API_KEY")
>>> result = engine.full_analysis("arsenal", lambda_home=1.45, lambda_away=1.10)
>>> if result:
...     print(result["adj_lambda_home"], result["adj_lambda_away"])
...     print(result["impact"].notes)
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stadium registry
# ---------------------------------------------------------------------------

#: Mapping of lowercase team slug → (latitude, longitude) of the stadium.
#: Covers Premier League, Serie A, Bundesliga, La Liga, Ligue 1, Eredivisie,
#: Primeira Liga, Belgian Pro League, Scottish Premiership, Turkish Süper Lig.
STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    # ── Premier League ───────────────────────────────────────────────────────
    "manchester_city": (53.4831, -2.2004),
    "arsenal": (51.5549, -0.1084),
    "liverpool": (53.4308, -2.9608),
    "chelsea": (51.4816, -0.1910),
    "tottenham": (51.6043, -0.0665),
    "manchester_united": (53.4631, -2.2913),
    "newcastle": (54.9756, -1.6217),
    "west_ham": (51.5386, -0.0164),
    "aston_villa": (52.5090, -1.8847),
    "brighton": (50.8618, -0.0837),
    "brentford": (51.4882, -0.2886),
    "fulham": (51.4749, -0.2219),
    "crystal_palace": (51.3983, -0.0855),
    "wolves": (52.5901, -2.1303),
    "everton": (53.4388, -2.9662),
    "nottingham_forest": (52.9399, -1.1326),
    "leicester": (52.6204, -1.1423),
    "southampton": (50.9058, -1.3914),
    "ipswich": (52.0551, 1.1446),
    "bournemouth": (50.7352, -1.8382),
    # ── Serie A ──────────────────────────────────────────────────────────────
    "juventus": (45.1096, 7.6413),
    "inter_milan": (45.4781, 9.1240),
    "ac_milan": (45.4781, 9.1240),
    "napoli": (40.8279, 14.1931),
    "roma": (41.9342, 12.4547),
    "lazio": (41.9342, 12.4547),
    "atalanta": (45.7089, 9.6797),
    "fiorentina": (43.7800, 11.2825),
    "torino": (45.0416, 7.6505),
    "bologna": (44.5021, 11.3091),
    # ── Bundesliga ───────────────────────────────────────────────────────────
    "bayern_munich": (48.2188, 11.6247),
    "borussia_dortmund": (51.4926, 7.4519),
    "rb_leipzig": (51.3457, 12.3483),
    "bayer_leverkusen": (51.0384, 7.0021),
    "eintracht_frankfurt": (50.0686, 8.6453),
    "wolfsburg": (52.4326, 10.8038),
    "borussia_monchengladbach": (51.1744, 6.3854),
    "hoffenheim": (49.2388, 8.8897),
    "freiburg": (47.9943, 7.8974),
    "union_berlin": (52.4575, 13.5688),
    # ── La Liga ──────────────────────────────────────────────────────────────
    "real_madrid": (40.4531, -3.6883),
    "barcelona": (41.3809, 2.1228),
    "atletico_madrid": (40.4361, -3.5995),
    "sevilla": (37.3841, -5.9705),
    "real_sociedad": (43.3015, -1.9737),
    "villarreal": (39.9447, -0.1030),
    "athletic_bilbao": (43.2640, -2.9494),
    "real_betis": (37.3567, -5.9814),
    "valencia": (39.4750, -0.3585),
    "getafe": (40.3263, -3.7134),
    # ── Ligue 1 ──────────────────────────────────────────────────────────────
    "paris_saint_germain": (48.8414, 2.2530),
    "olympique_marseille": (43.2697, 5.3959),
    "olympique_lyonnais": (45.7653, 4.9822),
    "monaco": (43.7275, 7.4150),
    "lille": (50.6120, 3.1305),
    "rennes": (48.1075, -1.7126),
    "lens": (50.4328, 2.8221),
    "nice": (43.7051, 7.1926),
    # ── Eredivisie ───────────────────────────────────────────────────────────
    "ajax": (52.3143, 4.9419),
    "psv_eindhoven": (51.4416, 5.4675),
    "feyenoord": (51.8938, 4.5232),
    "az_alkmaar": (52.6140, 4.7498),
    # ── Primeira Liga ────────────────────────────────────────────────────────
    "benfica": (38.7523, -9.1845),
    "sporting_cp": (38.7613, -9.1602),
    "fc_porto": (41.1619, -8.5836),
    "sc_braga": (41.5680, -8.4064),
    # ── Belgian Pro League ───────────────────────────────────────────────────
    "club_brugge": (51.2093, 3.2247),
    "anderlecht": (50.8356, 4.2978),
    "gent": (51.0593, 3.7101),
    "standard_liege": (50.6097, 5.5426),
    # ── Scottish Premiership ─────────────────────────────────────────────────
    "celtic": (55.8497, -4.2057),
    "rangers": (55.8553, -4.3090),
    "hearts": (55.9383, -3.2340),
    "hibernian": (55.9613, -3.1653),
    # ── Turkish Süper Lig ────────────────────────────────────────────────────
    "galatasaray": (41.0664, 28.9919),
    "fenerbahce": (40.9828, 29.0425),
    "besiktas": (41.0440, 29.0084),
    "trabzonspor": (40.9980, 39.7460),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WeatherCondition:
    """Snapshot of weather at a stadium location."""

    stadium_key: str
    lat: float
    lon: float
    temp_c: float  # temperature in Celsius
    wind_kph: float  # wind speed in km/h
    rain_mm_h: float  # precipitation in mm/hour (0 if none)
    description: str  # e.g. "light rain"
    fetched_at: float  # unix timestamp (seconds since epoch)

    @property
    def is_rainy(self) -> bool:
        """True when precipitation exceeds 1 mm/hour."""
        return self.rain_mm_h > 1.0

    @property
    def is_windy(self) -> bool:
        """True when wind speed exceeds 30 km/h."""
        return self.wind_kph > 30.0

    @property
    def is_freezing(self) -> bool:
        """True when temperature is below 2 °C."""
        return self.temp_c < 2.0

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"WeatherCondition(stadium={self.stadium_key!r}, "
            f"temp={self.temp_c:.1f}°C, wind={self.wind_kph:.1f} km/h, "
            f"rain={self.rain_mm_h:.2f} mm/h, desc={self.description!r})"
        )


@dataclass
class WeatherImpact:
    """Goal-rate impact derived from a WeatherCondition."""

    condition: WeatherCondition
    lambda_multiplier: float  # applied symmetrically to both team lambdas
    goals_adjustment: float  # absolute expected-goals change (negative = fewer goals)
    notes: List[str] = field(default_factory=list)  # human-readable reasons

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"WeatherImpact(multiplier={self.lambda_multiplier:.3f}, "
            f"goals_adj={self.goals_adjustment:+.3f}, notes={self.notes})"
        )


# ---------------------------------------------------------------------------
# WeatherEngine
# ---------------------------------------------------------------------------


class WeatherEngine:
    """Fetches current weather from OpenWeatherMap and converts it to
    goal-rate multipliers suitable for Poisson-model adjustments."""

    _BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str, timeout: int = 8) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string.")
        self._api_key = api_key
        self._timeout = timeout
        logger.debug("WeatherEngine initialised (timeout=%ds).", timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self, lat: float, lon: float, stadium_key: str = "unknown"
    ) -> WeatherCondition:
        """Call the OWM Current Weather endpoint and return a WeatherCondition.

        Parameters
        ----------
        lat:
            Latitude of the stadium.
        lon:
            Longitude of the stadium.
        stadium_key:
            Informational label stored on the returned object.

        Raises
        ------
        RuntimeError
            If the HTTP request fails, OWM returns a non-200 status code, or
            the response JSON cannot be parsed.
        """
        url = f"{self._BASE_URL}?lat={lat}&lon={lon}&appid={self._api_key}&units=metric"
        logger.info(
            "Fetching weather for '%s' (lat=%.4f, lon=%.4f).", stadium_key, lat, lon
        )

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "FootballQuantEngine/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"OWM HTTP error {exc.code} for stadium '{stadium_key}': {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OWM URL error for stadium '{stadium_key}': {exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error fetching weather for '{stadium_key}': {exc}"
            ) from exc

        if status != 200:
            raise RuntimeError(
                f"OWM returned HTTP {status} for stadium '{stadium_key}'."
            )

        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse OWM JSON for stadium '{stadium_key}': {exc}"
            ) from exc

        return self._parse_owm_response(data, lat=lat, lon=lon, stadium_key=stadium_key)

    def fetch_by_team(self, team_slug: str) -> Optional[WeatherCondition]:
        """Fetch weather for a registered team slug.

        Returns ``None`` if the slug is not present in :data:`STADIUM_COORDS`,
        or if the API call raises a :class:`RuntimeError` (error is logged).
        """
        key = slugify(team_slug)
        coords = STADIUM_COORDS.get(key)
        if coords is None:
            logger.warning("Team slug '%s' not found in STADIUM_COORDS.", key)
            return None

        lat, lon = coords
        try:
            return self.fetch(lat=lat, lon=lon, stadium_key=key)
        except RuntimeError as exc:
            logger.error("Failed to fetch weather for team '%s': %s", key, exc)
            return None

    def compute_impact(self, condition: WeatherCondition) -> WeatherImpact:
        """Convert a WeatherCondition into goal-rate adjustments.

        The multiplier starts at 1.0 and receives additive adjustments based
        on wind, rain and temperature brackets.  Only the *worst* wind bracket
        and the *worst* rain bracket apply (they do not stack within their
        category); temperature brackets are fully additive with each other.

        The final multiplier is clamped to a minimum of 0.50.
        ``goals_adjustment = (multiplier − 1.0) × 2.5``.
        """
        adjustment = 0.0
        notes: List[str] = []

        # ── Wind (highest bracket wins) ────────────────────────────────────
        wind = condition.wind_kph
        if wind > 50.0:
            adjustment -= 0.20
            notes.append(f"gale-force wind ({wind:.1f} km/h): −0.20")
        elif wind > 35.0:
            adjustment -= 0.12
            notes.append(f"strong wind ({wind:.1f} km/h): −0.12")
        elif wind > 20.0:
            adjustment -= 0.06
            notes.append(f"moderate wind ({wind:.1f} km/h): −0.06")

        # ── Rain (highest bracket wins) ────────────────────────────────────
        rain = condition.rain_mm_h
        if rain > 8.0:
            adjustment -= 0.18
            notes.append(f"heavy rain ({rain:.2f} mm/h): −0.18")
        elif rain > 3.0:
            adjustment -= 0.10
            notes.append(f"moderate rain ({rain:.2f} mm/h): −0.10")
        elif rain > 0.5:
            adjustment -= 0.05
            notes.append(f"light rain ({rain:.2f} mm/h): −0.05")

        # ── Temperature (additive across both brackets) ────────────────────
        temp = condition.temp_c
        if temp < 0.0:
            adjustment -= 0.05
            notes.append(f"freezing ({temp:.1f}°C): −0.05")
        if temp < 2.0:
            adjustment -= 0.03
            notes.append(f"near-freezing ({temp:.1f}°C): −0.03")

        multiplier = max(0.50, 1.0 + adjustment)
        goals_adjustment = (multiplier - 1.0) * 2.5

        if not notes:
            notes.append("no significant weather impact")

        logger.debug(
            "Impact for '%s': multiplier=%.3f, goals_adj=%.3f, notes=%s",
            condition.stadium_key,
            multiplier,
            goals_adjustment,
            notes,
        )

        return WeatherImpact(
            condition=condition,
            lambda_multiplier=multiplier,
            goals_adjustment=goals_adjustment,
            notes=notes,
        )

    def adjust_lambdas(
        self,
        lambda_home: float,
        lambda_away: float,
        condition: WeatherCondition,
    ) -> Tuple[float, float]:
        """Apply weather impact multiplier to both Poisson lambdas.

        Parameters
        ----------
        lambda_home:
            Home team expected-goals rate (pre-adjustment).
        lambda_away:
            Away team expected-goals rate (pre-adjustment).
        condition:
            Current weather at the stadium.

        Returns
        -------
        Tuple[float, float]
            Adjusted ``(lambda_home, lambda_away)`` each clamped to ≥ 0.10.
        """
        impact = self.compute_impact(condition)
        m = impact.lambda_multiplier
        adj_home = max(0.10, lambda_home * m)
        adj_away = max(0.10, lambda_away * m)
        logger.info(
            "Lambda adjustment for '%s': home %.3f→%.3f, away %.3f→%.3f (×%.3f).",
            condition.stadium_key,
            lambda_home,
            adj_home,
            lambda_away,
            adj_away,
            m,
        )
        return adj_home, adj_away

    def full_analysis(
        self,
        team_home_slug: str,
        lambda_home: float,
        lambda_away: float,
    ) -> Optional[Dict]:
        """End-to-end pipeline: fetch weather → compute impact → adjust lambdas.

        Parameters
        ----------
        team_home_slug:
            Home team identifier (looked up in :data:`STADIUM_COORDS`).
        lambda_home:
            Pre-weather home expected-goals rate.
        lambda_away:
            Pre-weather away expected-goals rate.

        Returns
        -------
        dict or None
            Keys: ``condition``, ``impact``, ``adj_lambda_home``,
            ``adj_lambda_away``.  Returns ``None`` if the slug is unknown or
            the API call fails.
        """
        condition = self.fetch_by_team(team_home_slug)
        if condition is None:
            logger.warning(
                "full_analysis returning None: could not get weather for '%s'.",
                team_home_slug,
            )
            return None

        impact = self.compute_impact(condition)
        adj_home, adj_away = self.adjust_lambdas(lambda_home, lambda_away, condition)

        return {
            "condition": condition,
            "impact": impact,
            "adj_lambda_home": adj_home,
            "adj_lambda_away": adj_away,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_owm_response(
        self,
        data: dict,
        *,
        lat: float,
        lon: float,
        stadium_key: str,
    ) -> WeatherCondition:
        """Extract relevant fields from a raw OWM JSON response dict."""
        try:
            temp_c: float = float(data["main"]["temp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Missing 'main.temp' in OWM response for '{stadium_key}': {exc}"
            ) from exc

        try:
            wind_ms: float = float(data["wind"]["speed"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Missing wind speed in OWM response for '%s'; defaulting to 0.",
                stadium_key,
            )
            wind_ms = 0.0
        wind_kph = wind_ms * 3.6

        # rain["1h"] is absent when there is no precipitation — default to 0.0
        rain_mm_h: float = 0.0
        rain_block = data.get("rain")
        if isinstance(rain_block, dict):
            rain_mm_h = float(rain_block.get("1h", 0.0))

        try:
            description: str = data["weather"][0]["description"]
        except (KeyError, IndexError, TypeError):
            description = "unknown"

        return WeatherCondition(
            stadium_key=stadium_key,
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            wind_kph=wind_kph,
            rain_mm_h=rain_mm_h,
            description=description,
            fetched_at=time.time(),
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def build_engine(api_key: str) -> WeatherEngine:
    """Construct and return a :class:`WeatherEngine` with default settings.

    Parameters
    ----------
    api_key:
        OpenWeatherMap API key.
    """
    return WeatherEngine(api_key=api_key)


def slugify(name: str) -> str:
    """Normalise a team name into a registry key.

    Converts to lowercase, replaces spaces and hyphens with underscores, and
    strips any characters that are not alphanumeric or underscores.

    Examples
    --------
    >>> slugify("Manchester City")
    'manchester_city'
    >>> slugify("Borussia Mönchengladbach")
    'borussia_mnchengladbach'
    """
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^\w]", "", name)  # \w matches [a-zA-Z0-9_]
    return name
