"""Turn raw GPX points into 20 m segments and 100 m blocks, per activity."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gpxpy.gpx import GPXException
from scipy.signal import savgol_filter

from .gpx import read_gpx
from .hr import clean_hr

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6371000.0
STEP_M = 20.0
BLOCK_M = 100.0
PAUSE_SPEED = 0.3  # m/s, below this the athlete is considered stopped
PAUSE_WINDOW = 10  # points
GRADE_CLIP = 60.0
SMOOTH_POINTS = 9  # Savitzky-Golay window on the 20 m grid (~180 m)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dlambda) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlambda)
    return np.degrees(np.arctan2(y, x)) % 360.0


def angle_diff(a):
    """Absolute heading change (deg, 0..180) between consecutive headings."""
    d = np.abs(np.diff(a, prepend=a[:1])) % 360.0
    return np.minimum(d, 360.0 - d)


def hysteresis_gain(ele, threshold=3.0):
    """Cumulated D+ / D- ignoring oscillations smaller than `threshold` metres."""
    up = down = 0.0
    ref = ele[0]
    for e in ele[1:]:
        if e - ref >= threshold:
            up += e - ref
            ref = e
        elif ref - e >= threshold:
            down += ref - e
            ref = e
    return up, down


def detect_pauses(dd, dt, min_speed=PAUSE_SPEED, window=PAUSE_WINDOW):
    dist = pd.Series(dd).rolling(window, center=True, min_periods=1).sum()
    dur = pd.Series(dt).rolling(window, center=True, min_periods=1).sum()
    speed = (dist / dur.where(dur > 0)).to_numpy()
    return (np.asarray(dt) > 0) & ~(speed >= min_speed)


def _interp_optional(grid, xp, values):
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, xp[ok], values[ok])


def resample(raw: pd.DataFrame, step=STEP_M) -> pd.DataFrame:
    """Points every `step` metres of horizontal distance, with smoothed elevation."""
    raw = raw.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(raw) < 2:
        raise ValueError("not enough points")
    lat, lon = raw["lat"].to_numpy(), raw["lon"].to_numpy()
    dd = np.r_[0.0, haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])]
    dist = np.cumsum(dd)
    ele = raw["ele"].interpolate(limit_direction="both").to_numpy()
    if np.isnan(ele).all():
        log.warning("%s: no elevation data", raw.attrs.get("name"))
        ele = np.zeros(len(raw))

    has_time = bool(raw["time"].notna().all())
    if has_time:
        t = (raw["time"] - raw["time"].iloc[0]).dt.total_seconds().to_numpy()
        dt = np.diff(t, prepend=0.0)
        t_moving = np.cumsum(np.where(detect_pauses(dd, dt), 0.0, dt))
    else:
        t = t_moving = np.full(len(raw), np.nan)

    keep = np.r_[True, np.diff(dist) > 0]
    dist, lat, lon, ele, t, t_moving = (a[keep] for a in (dist, lat, lon, ele, t, t_moving))
    total = dist[-1]
    if total <= 0:
        raise ValueError("track has no length")
    grid = np.arange(0.0, total, step)
    if len(grid) > 1 and total - grid[-1] < step / 2:
        grid = grid[:-1]
    grid = np.r_[grid, total]

    pts = pd.DataFrame({
        "dist": grid,
        "lat": np.interp(grid, dist, lat),
        "lon": np.interp(grid, dist, lon),
        "ele": np.interp(grid, dist, ele),
        "t": np.interp(grid, dist, t_moving) if has_time else np.nan,
        "t_elapsed": np.interp(grid, dist, t) if has_time else np.nan,
    })
    for col in ("hr", "cad", "atemp"):
        pts[col] = _interp_optional(grid, dist, raw[col].to_numpy(float)[keep])
    if len(pts) >= SMOOTH_POINTS:
        pts["ele"] = savgol_filter(pts["ele"].to_numpy(), SMOOTH_POINTS, 2)
    return pts


def segments(pts: pd.DataFrame) -> pd.DataFrame:
    """One row per resampled step; values refer to the segment end unless averaged."""
    def mid(col):
        a = pts[col].to_numpy()
        return (a[1:] + a[:-1]) / 2

    d, e = pts["dist"].to_numpy(), pts["ele"].to_numpy()
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    dx, dz = np.diff(d), np.diff(e)
    seg = pd.DataFrame({
        "dist": d[1:],
        "dx": dx,
        "dx3d": np.hypot(dx, dz),
        "dz": dz,
        "ele": e[1:],
        "grade": np.clip(dz / dx * 100.0, -GRADE_CLIP, GRADE_CLIP),
        "lat": lat[1:],
        "lon": lon[1:],
        "dt": np.diff(pts["t"].to_numpy()),
        "t": pts["t"].to_numpy()[1:],
        "elapsed": np.diff(pts["t_elapsed"].to_numpy()),
        "t_elapsed": pts["t_elapsed"].to_numpy()[1:],
        "hr": mid("hr"),
        "cad": mid("cad"),
        "atemp": mid("atemp"),
    })
    seg["stop"] = seg["elapsed"] - seg["dt"]
    seg.attrs["ele0"] = float(e[0])
    return seg


def make_blocks(segs: pd.DataFrame, length=BLOCK_M) -> pd.DataFrame:
    """Aggregate segments into blocks of ~`length` metres (model and prediction unit)."""
    start = (segs["dist"] - segs["dx"]).to_numpy()
    key = np.floor(start / length + 1e-9).astype(int)
    last = key.max()
    if last > 0 and segs["dx"][key == last].sum() < length / 2:
        key[key == last] = last - 1

    s = segs.assign(
        key=key,
        dplus=segs["dz"].clip(lower=0),
        dminus=segs["dz"].clip(upper=0),
        hr_x=segs["hr"] * segs["dx"],
    )
    g = s.groupby("key")
    b = pd.DataFrame({
        "dist": g["dist"].last(),
        "dx": g["dx"].sum(),
        "dz": g["dz"].sum(),
        "dplus": g["dplus"].sum(),
        "dminus": g["dminus"].sum(),
        "ele": g["ele"].mean(),
        "lat0": g["lat"].first(),
        "lon0": g["lon"].first(),
        "lat": g["lat"].last(),
        "lon": g["lon"].last(),
        "dt": g["dt"].sum(min_count=1),
        "elapsed": g["elapsed"].sum(min_count=1),
        "t": g["t"].last(),
        "t_elapsed": g["t_elapsed"].last(),
        "hr": g["hr"].mean(),
        "cad": g["cad"].mean(),
        "atemp": g["atemp"].mean(),
    }).reset_index(drop=True)
    b["stop"] = b["elapsed"] - b["dt"]
    b["grade"] = np.clip(b["dz"] / b["dx"] * 100.0, -GRADE_CLIP, GRADE_CLIP)
    heading = bearing(b["lat0"], b["lon0"], b["lat"], b["lon"])
    b["turn"] = angle_diff(heading.to_numpy()) / (b["dx"] / 1000.0)
    b["effort"] = (b["dist"] - b["dx"]) / 1000.0 + (b["dplus"].cumsum() - b["dplus"]) / 100.0
    b["t_elapsed_mid"] = b["t_elapsed"] - b["elapsed"] / 2
    return b


@dataclass
class Activity:
    name: str
    path: Path
    raw: pd.DataFrame
    segs: pd.DataFrame
    blocks: pd.DataFrame

    @property
    def has_time(self) -> bool:
        return bool(self.blocks["dt"].notna().all() and self.blocks["dt"].sum() > 0)

    @property
    def has_hr(self) -> bool:
        return bool(self.raw["hr"].notna().any())

    @property
    def start(self):
        t = self.raw["time"].iloc[0]
        return None if pd.isna(t) else t


def load_activity(path, step=STEP_M, block=BLOCK_M) -> Activity:
    path = Path(path)
    raw = read_gpx(path)
    raw["hr"] = clean_hr(raw["hr"])
    segs = segments(resample(raw, step))
    return Activity(raw.attrs.get("name") or path.stem, path, raw, segs, make_blocks(segs, block))


def load_dir(directory, step=STEP_M, block=BLOCK_M) -> list[Activity]:
    acts = []
    for path in sorted(Path(directory).glob("*.gpx")):
        try:
            acts.append(load_activity(path, step, block))
        except (ValueError, OSError, GPXException) as exc:
            log.warning("skipping %s: %s", path, exc)
    return acts
