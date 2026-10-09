"""Running metrics, summaries and formatting helpers."""

import math

import numpy as np
import pandas as pd

from .preprocess import Activity, hysteresis_gain


def gap_factor(grade):
    """Strava-like cost factor; GAP pace = pace / factor."""
    g = np.asarray(grade, dtype=float) / 100.0
    return np.where(g >= 0, 1 + 4.0 * g + 10.0 * g**2, 1 + 2.0 * g + 5.0 * g**2)


def minetti_cost(grade):
    """Metabolic cost of running (J/kg/m), Minetti et al. 2002, grade clipped to its +/-45 % validity."""
    i = np.clip(np.asarray(grade, dtype=float) / 100.0, -0.45, 0.45)
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


def vo2(speed_m_s, grade):
    """ACSM running equation (ml/kg/min); speed converted to m/min, uphill grade only."""
    s = np.asarray(speed_m_s, dtype=float) * 60.0
    return 3.5 + 0.2 * s + 0.9 * s * np.clip(np.asarray(grade, dtype=float) / 100.0, 0.0, None)


def pace_min_km(spm):
    """Seconds per metre -> minutes per km."""
    return np.asarray(spm, dtype=float) * 1000.0 / 60.0


def format_duration(seconds) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def format_pace(spm) -> str:
    """Seconds per metre -> 'M:SS' min/km."""
    if spm is None or not math.isfinite(spm) or spm <= 0:
        return "--"
    s = int(round(spm * 1000.0))
    return f"{s // 60}:{s % 60:02d}"


def elevation_totals(segs: pd.DataFrame, threshold=3.0):
    ele = np.r_[segs.attrs.get("ele0", segs["ele"].iloc[0]), segs["ele"].to_numpy()]
    return hysteresis_gain(ele, threshold)


def summarize(act: Activity) -> dict:
    b = act.blocks
    dist = float(b["dx"].sum())
    d_plus, d_minus = elevation_totals(act.segs)
    moving = float(b["dt"].sum()) if act.has_time else math.nan
    elapsed = float(b["elapsed"].sum()) if act.has_time else math.nan
    hr = act.raw["hr"]
    return {
        "name": act.name,
        "file": act.path.name,
        "start": act.start,
        "distance_km": dist / 1000.0,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "moving": moving,
        "elapsed": elapsed,
        "stopped": elapsed - moving,
        "pace": moving / dist if dist else math.nan,
        "hr_mean": float(hr.mean()) if hr.notna().any() else math.nan,
    }


def pace_by_grade(blocks: pd.DataFrame, bin_size=2.0) -> pd.DataFrame:
    """Time-weighted pace (sum dt / sum dx) per grade bin."""
    b = blocks[(blocks["dt"] > 0) & blocks["dt"].notna()]
    edges = np.arange(-60.0, 60.0 + bin_size, bin_size)
    cat = pd.cut(b["grade"], edges)
    g = b.groupby(cat, observed=True)[["dt", "dx"]].sum()
    g = g[g["dx"] >= 300.0]
    return pd.DataFrame({
        "grade": [i.mid for i in g.index],
        "spm": (g["dt"] / g["dx"]).to_numpy(),
        "dx": g["dx"].to_numpy(),
    })
