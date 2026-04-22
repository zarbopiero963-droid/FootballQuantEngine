from __future__ import annotations

import requests


_DEFAULT_WEATHER = {
    "temperature_c": 15.0,
    "wind_speed_ms": 3.0,
    "precipitation_mm": 0.0,
    "condition": "clear",
}


class WeatherClient:
    """
    Fetches weather forecast for a match venue via OpenWeatherMap API.
    Requires a separate OPENWEATHER_API_KEY (free tier sufficient).
    Falls back to neutral defaults when key is absent or request fails.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def get_match_weather(
        self, lat: float, lon: float, match_timestamp: int | None = None
    ) -> dict:
        if not self.api_key:
            return dict(_DEFAULT_WEATHER)
        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": 8,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            forecast = data["list"][0]
            return {
                "temperature_c": float(forecast["main"]["temp"]),
                "wind_speed_ms": float(forecast["wind"]["speed"]),
                "precipitation_mm": float(
                    forecast.get("rain", {}).get("3h", 0.0)
                ),
                "condition": self._classify(forecast),
            }
        except Exception:
            return dict(_DEFAULT_WEATHER)

    def _classify(self, forecast: dict) -> str:
        main = forecast.get("weather", [{}])[0].get("main", "").lower()
        wind = forecast.get("wind", {}).get("speed", 0.0)
        rain = forecast.get("rain", {}).get("3h", 0.0)
        if "snow" in main:
            return "snow"
        if "rain" in main or "drizzle" in main or rain > 2.0:
            return "rain"
        if wind > 10.0:
            return "wind"
        return "clear"


class WeatherEngine:
    """
    Converts weather conditions into a goal-expectancy multiplier.
    Rain / snow / strong wind slightly reduce total expected goals.
    """

    _MODIFIERS: dict[str, float] = {
        "snow": 0.88,
        "rain": 0.93,
        "wind": 0.96,
        "clear": 1.00,
    }

    def get_lambda_modifier(self, weather: dict | None) -> float:
        if not weather:
            return 1.0
        condition = weather.get("condition", "clear")
        base = self._MODIFIERS.get(condition, 1.0)
        # Extra penalty for very heavy rain
        if weather.get("precipitation_mm", 0.0) > 5.0:
            base = max(0.85, base - 0.03)
        return base
