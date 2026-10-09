"""Course complexity: climbs/descents, grade distribution, effort distance, technicality proxies."""

import math

import numpy as np
import pandas as pd

from .metrics import elevation_totals
from .preprocess import Activity

MIN_GAIN = 30.0  # m, smallest climb/descent kept as a section
STEEP = 15.0  # %
HIKE_CADENCE = 70.0  # steps/min on one foot (GPX `cad` convention)


def sections(segs: pd.DataFrame, min_gain=MIN_GAIN) -> pd.DataFrame:
    """Zig-zag segmentation of the smoothed profile into climb / descent / rolling sections."""
    ele = np.r_[segs.attrs.get("ele0", segs["ele"].iloc[0]), segs["ele"].to_numpy()]
    dist = np.r_[0.0, segs["dist"].to_numpy()]
    n = len(ele)
    pivots, direction, ext, lo, hi = [0], 0, 0, 0, 0
    for i in range(1, n):
        if direction == 0:
            lo = i if ele[i] < ele[lo] else lo
            hi = i if ele[i] > ele[hi] else hi
            if ele[i] - ele[lo] >= min_gain:
                direction, ext = 1, i
                if lo:
                    pivots.append(lo)
            elif ele[hi] - ele[i] >= min_gain:
                direction, ext = -1, i
                if hi:
                    pivots.append(hi)
        elif direction == 1:
            if ele[i] >= ele[ext]:
                ext = i
            elif ele[ext] - ele[i] >= min_gain:
                pivots.append(ext)
                direction, ext = -1, i
        else:
            if ele[i] <= ele[ext]:
                ext = i
            elif ele[i] - ele[ext] >= min_gain:
                pivots.append(ext)
                direction, ext = 1, i
    if direction and ext not in pivots:
        pivots.append(ext)
    if pivots[-1] != n - 1:
        pivots.append(n - 1)

    grade = segs["grade"].rolling(5, center=True, min_periods=1).mean().to_numpy()
    rows = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        gain = ele[b] - ele[a]
        length = dist[b] - dist[a]
        kind = "climb" if gain >= min_gain else "descent" if gain <= -min_gain else "rolling"
        dz = np.diff(ele[a:b + 1])
        g = grade[a:b] if b > a else np.array([0.0])
        rows.append({
            "kind": kind,
            "start_km": dist[a] / 1000.0,
            "end_km": dist[b] / 1000.0,
            "length_km": length / 1000.0,
            "ele_start": ele[a],
            "ele_end": ele[b],
            "d_plus": dz[dz > 0].sum(),
            "d_minus": dz[dz < 0].sum(),
            "avg_grade": gain / length * 100.0 if length else 0.0,
            "max_grade": g.max() if kind != "descent" else g.min(),
        })
    return pd.DataFrame(rows)


def features(act: Activity) -> dict:
    segs, b = act.segs, act.blocks
    km = segs["dx"].sum() / 1000.0
    d_plus, d_minus = elevation_totals(segs)
    grade = segs["grade"].to_numpy()
    w = segs["dx"].to_numpy()
    steep_up, steep_down = grade > STEEP, grade < -STEEP
    steep = steep_up | steep_down
    hard_position = (
        float(np.average(segs["dist"][steep], weights=w[steep]) / (km * 1000.0)) if steep.any() else math.nan
    )
    sec = sections(segs)
    return {
        "distance_km": km,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "dplus_per_km": d_plus / km,
        "dminus_per_km": d_minus / km,
        "effort_km": km + d_plus / 100.0 + d_minus / 200.0,
        "grade_p90": float(np.percentile(grade, 90)),
        "grade_p10": float(np.percentile(grade, 10)),
        "grade_abs_p99": float(np.percentile(np.abs(grade), 99)),
        "km_steep_up": float(w[steep_up].sum() / 1000.0),
        "km_steep_down": float(w[steep_down].sum() / 1000.0),
        "n_climbs": int((sec["kind"] == "climb").sum()) if len(sec) else 0,
        "longest_climb_m": float(sec.loc[sec["kind"] == "climb", "d_plus"].max()) if (sec["kind"] == "climb").any() else 0.0,
        "ele_mean": float(np.average(segs["ele"], weights=w)),
        "ele_max": float(segs["ele"].max()),
        "turn_deg_per_km": float((b["turn"] * b["dx"] / 1000.0).sum() / km),
        "hard_position": hard_position,
    }


def similarity(a: dict, b: dict) -> float:
    """1 for identical courses, decreasing with D+/km and distance ratios."""
    dpk = abs(math.log(max(a["dplus_per_km"], 5.0) / max(b["dplus_per_km"], 5.0)))
    dist = abs(math.log(a["distance_km"] / b["distance_km"]))
    return math.exp(-dpk - 0.5 * dist)


def hike_threshold(acts: list[Activity], bin_size=2.0) -> float | None:
    """Lowest uphill grade where most of the time is spent hiking (cadence below HIKE_CADENCE)."""
    frames = [a.blocks[["grade", "cad", "dt"]] for a in acts if a.has_time]
    if not frames:
        return None
    b = pd.concat(frames).dropna()
    b = b[(b["dt"] > 0) & (b["grade"] > 0)]
    if b.empty:
        return None
    edges = np.arange(0.0, 60.0 + bin_size, bin_size)
    hiking = (b["cad"] < HIKE_CADENCE) * b["dt"]
    g = pd.DataFrame({"bin": pd.cut(b["grade"], edges), "hike": hiking, "dt": b["dt"]})
    share = g.groupby("bin", observed=True).sum()
    share = share[share["dt"] >= 120.0]
    frac = share["hike"] / share["dt"]
    above = frac[frac > 0.5]
    return float(above.index[0].left) if len(above) else None
