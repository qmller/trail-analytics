"""Optional online data (weather: Open-Meteo, trail tags: OpenStreetMap Overpass), cached on disk."""

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
TIMEOUT_S = 30
OVERPASS_TIMEOUT_S = 90
MAX_RETRIES = 3  # on HTTP 429 / 504 (Overpass rate limiting)
USER_AGENT = "trailsport/0.2 (personal race-time prediction)"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
FORECAST_DAYS = 15
ARCHIVE_DELAY_DAYS = 5
MATCH_RADIUS_M = 30.0

SAC_SCALE = {
    "hiking": 1.0,
    "mountain_hiking": 1.05,
    "demanding_mountain_hiking": 1.15,
    "alpine_hiking": 1.3,
    "demanding_alpine_hiking": 1.5,
    "difficult_alpine_hiking": 1.7,
}
SURFACE = {
    "asphalt": 0.97, "paved": 0.97, "concrete": 0.97, "paving_stones": 0.98,
    "compacted": 1.0, "fine_gravel": 1.0, "gravel": 1.01, "unpaved": 1.02,
    "dirt": 1.02, "earth": 1.02, "ground": 1.02, "grass": 1.03, "pebblestone": 1.05,
    "mud": 1.08, "rock": 1.08, "stone": 1.08, "sand": 1.1, "scree": 1.15,
}


class OnlineError(RuntimeError):
    pass


_failed: set[str] = set()  # services that failed during this run


def _check_coords(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()
            and (np.abs(lat) <= 90).all() and (np.abs(lon) <= 180).all()):
        raise OnlineError("invalid coordinates")
    return lat, lon


def _request(method, url, cache_dir=None, timeout=TIMEOUT_S, **kwargs) -> dict:
    cache_dir = Path(cache_dir or CACHE_DIR)
    key = hashlib.sha256(json.dumps([method, url, kwargs], sort_keys=True, default=str).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if url in _failed:
        raise OnlineError(f"{url}: skipped after a previous failure")
    try:
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.request(method, url, timeout=timeout, headers={"User-Agent": USER_AGENT}, **kwargs)
            if resp.status_code not in (429, 504) or attempt == MAX_RETRIES:
                break
            retry_after = resp.headers.get("Retry-After", "")
            time.sleep(min(float(retry_after), 60.0) if retry_after.isdigit() else 15.0 * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        _failed.add(url)
        raise OnlineError(f"{url}: {exc}") from exc
    if not isinstance(data, dict):
        raise OnlineError(f"{url}: unexpected response")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def weather(lat, lon, start: pd.Timestamp, hours: float, cache_dir=None) -> pd.DataFrame:
    """Hourly air temperature (degC) and precipitation (mm), UTC index."""
    lat, lon = (float(v) for v in _check_coords(lat, lon))
    start = pd.Timestamp(start).tz_convert("UTC")
    end = start + pd.Timedelta(hours=max(hours, 1.0))
    today = pd.Timestamp.now(tz="UTC").normalize()
    if end < today - pd.Timedelta(days=ARCHIVE_DELAY_DAYS):
        url = ARCHIVE_URL
    elif start < today + pd.Timedelta(days=FORECAST_DAYS):
        url = FORECAST_URL
    else:
        raise OnlineError(f"no forecast available yet for {start.date()}")
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "temperature_2m,precipitation",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    data = _request("GET", url, cache_dir, params=params)
    hourly = data.get("hourly")
    try:
        df = pd.DataFrame({
            "temperature": np.asarray(hourly["temperature_2m"], dtype=float),
            "precipitation": np.asarray(hourly["precipitation"], dtype=float),
        }, index=pd.to_datetime(hourly["time"], utc=True))
    except (TypeError, KeyError, ValueError) as exc:
        raise OnlineError(f"{url}: malformed weather data") from exc
    return df


def _ways(lat, lon, cache_dir=None) -> list[dict]:
    m = 0.001  # ~100 m margin
    bbox = f"{lat.min() - m:.5f},{lon.min() - m:.5f},{lat.max() + m:.5f},{lon.max() + m:.5f}"
    query = f'[out:json][timeout:{OVERPASS_TIMEOUT_S}];way["highway"]({bbox});out tags geom;'
    data = _request("POST", OVERPASS_URL, cache_dir, timeout=OVERPASS_TIMEOUT_S + 10, data={"data": query})
    elements = data.get("elements")
    if not isinstance(elements, list):
        raise OnlineError("overpass: malformed response")
    return [e for e in elements if isinstance(e, dict) and isinstance(e.get("geometry"), list)]


def way_multiplier(tags: dict) -> float:
    return SAC_SCALE.get(tags.get("sac_scale"), 1.0) * SURFACE.get(tags.get("surface"), 1.0)


def technicality(blocks: pd.DataFrame, cache_dir=None) -> np.ndarray:
    """Per-block multiplier from the OSM way nearest to each block end (1.0 if none within 30 m)."""
    lat, lon = _check_coords(blocks["lat"], blocks["lon"])
    ways = _ways(lat, lon, cache_dir)
    pts, mult = [], []
    for way in ways:
        m = way_multiplier(way.get("tags") or {})
        for node in way["geometry"]:
            if isinstance(node, dict) and "lat" in node and "lon" in node:
                pts.append((float(node["lat"]), float(node["lon"])))
                mult.append(m)
    out = np.ones(len(blocks))
    if not pts:
        return out
    pts, mult = np.asarray(pts), np.asarray(mult)
    k = 111_320.0
    coslat = math.cos(math.radians(float(lat.mean())))
    for i, (a, o) in enumerate(zip(lat, lon)):
        d2 = ((pts[:, 0] - a) * k) ** 2 + ((pts[:, 1] - o) * k * coslat) ** 2
        j = int(np.argmin(d2))
        if d2[j] <= MATCH_RADIUS_M**2:
            out[i] = mult[j]
    return out
