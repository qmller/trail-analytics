"""GPX reading (tracks, or routes as fallback) including Garmin TrackPointExtension fields."""

from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd

COLUMNS = ["lat", "lon", "ele", "time", "hr", "cad", "atemp"]
_EXT_FIELDS = {"hr", "cad", "atemp"}


def _extensions(point) -> dict:
    values = {}
    for ext in point.extensions or ():
        for el in ext.iter():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in _EXT_FIELDS and el.text:
                try:
                    values[tag] = float(el.text)
                except ValueError:
                    pass
    return values


def _row(point) -> dict:
    return {
        "lat": point.latitude,
        "lon": point.longitude,
        "ele": point.elevation,
        "time": point.time.timestamp() if point.time else np.nan,
        **_extensions(point),
    }


def read_gpx(path) -> pd.DataFrame:
    """Return one row per point with columns COLUMNS; `time` is UTC (NaT when absent)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    rows = [_row(p) for trk in gpx.tracks for seg in trk.segments for p in seg.points]
    if not rows:
        rows = [_row(p) for rte in gpx.routes for p in rte.points]
    if not rows:
        raise ValueError(f"{path}: no track or route points")

    df = pd.DataFrame(rows, columns=COLUMNS).astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    name = next((t.name for t in gpx.tracks if t.name), None) or gpx.name or path.stem
    df.attrs["name"] = name
    df.attrs["waypoints"] = [
        {"name": w.name or "", "type": w.type or "", "lat": w.latitude, "lon": w.longitude}
        for w in gpx.waypoints
    ]
    return df
