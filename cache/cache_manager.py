from __future__ import annotations

import json
import os
import time

CACHE_DIR = "cache_data"

# Default TTLs in seconds
TTL = {
    "fixture_statistics": 0,        # permanent — completed matches never change
    "completed_matches": 6 * 3600,  # 6 hours
    "standings": 24 * 3600,         # 24 hours
    "injuries": 6 * 3600,           # 6 hours
    "referee_stats": 24 * 3600,     # 24 hours
    "xg_averages": 6 * 3600,        # 6 hours
    "upcoming_matches": 3600,        # 1 hour
    "odds": 10 * 60,                # 10 minutes
}


def _path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.json")


def save_cache(name: str, data) -> None:
    entry = {"ts": time.time(), "data": data}
    with open(_path(name), "w", encoding="utf-8") as fh:
        json.dump(entry, fh)


def load_cache(name: str, ttl: int | None = None) -> object | None:
    """
    Return cached data if it exists and has not expired.
    ttl=0  → permanent (never expires).
    ttl=None → use the default TTL for this name if defined, else 1 hour.
    Returns None on miss or expiry.
    """
    path = _path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            entry = json.load(fh)
        if ttl is None:
            # Pick TTL by matching prefix of the cache key
            ttl = next(
                (v for k, v in TTL.items() if name.startswith(k)),
                3600,
            )
        if ttl != 0 and (time.time() - entry["ts"]) > ttl:
            return None
        return entry["data"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def invalidate(name: str) -> None:
    path = _path(name)
    if os.path.exists(path):
        os.remove(path)


def invalidate_all() -> None:
    if not os.path.isdir(CACHE_DIR):
        return
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json"):
            os.remove(os.path.join(CACHE_DIR, fname))
