"""Pace multipliers for external factors (>1 = slower). Applied to references (normalisation) and to the course."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FACTORS = ["altitude", "heat", "night", "wet", "poles", "form", "technicality", "fitness", "age"]


@dataclass
class FactorParams:
    altitude_start: float = 1500.0  # m
    altitude_coef: float = 1e-4  # +1 % per 100 m above altitude_start
    heat_ref: float = 15.0  # degC
    heat_coef: float = 0.004  # +2 % per 5 degC above heat_ref
    watch_temp_offset: float = 5.0  # wrist sensor reads warmer than air
    night: float = 1.10
    night_sun_elevation: float = -6.0  # deg, civil twilight
    wet: float = 1.15
    wet_grade: float = -10.0  # %, descents steeper than this are slowed when wet
    wet_precip_mm: float = 1.0
    poles: float = 0.97
    poles_grade: float = 15.0
    form_coef: float = 0.001  # per TSB point
    form_cap: float = 0.03


def solar_elevation(times_utc, lat, lon) -> np.ndarray:
    """Approximate solar elevation (deg), NOAA low-precision formulas."""
    ts = pd.DatetimeIndex(times_utc)
    ts = ts.tz_convert("UTC") if ts.tz is not None else ts
    n = ts.to_julian_date().to_numpy() - 2451545.0
    mean_long = (280.460 + 0.9856474 * n) % 360.0
    anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_long = np.radians(mean_long + 1.915 * np.sin(anomaly) + 0.020 * np.sin(2 * anomaly))
    obliquity = np.radians(23.439 - 4e-7 * n)
    ra = np.arctan2(np.cos(obliquity) * np.sin(ecl_long), np.cos(ecl_long))
    dec = np.arcsin(np.sin(obliquity) * np.sin(ecl_long))
    gmst_h = (18.697374558 + 24.06570982441908 * n) % 24.0
    hour_angle = np.radians(gmst_h * 15.0 + np.asarray(lon, dtype=float)) - ra
    phi = np.radians(np.asarray(lat, dtype=float))
    return np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle)))


def multipliers(
    blocks: pd.DataFrame,
    params: FactorParams,
    clock=None,
    temp=None,
    wet=False,
    poles=False,
    tsb=None,
    tech=None,
    fitness=None,
    age=None,
) -> pd.DataFrame:
    """One column per factor in FACTORS; `clock` (UTC) and `temp` are per block or scalar."""
    n = len(blocks)
    grade = blocks["grade"].to_numpy()
    out = pd.DataFrame(1.0, index=blocks.index, columns=FACTORS)
    out["altitude"] = 1.0 + params.altitude_coef * np.clip(blocks["ele"].to_numpy() - params.altitude_start, 0.0, None)
    if temp is not None:
        t = np.broadcast_to(np.asarray(temp, dtype=float), (n,))
        out["heat"] = np.where(np.isfinite(t), 1.0 + params.heat_coef * np.clip(t - params.heat_ref, 0.0, None), 1.0)
    if clock is not None:
        elev = solar_elevation(clock, blocks["lat"], blocks["lon"])
        out["night"] = np.where(elev < params.night_sun_elevation, params.night, 1.0)
    if wet:
        out["wet"] = np.where(grade < params.wet_grade, params.wet, 1.0)
    if poles:
        out["poles"] = np.where(grade > params.poles_grade, params.poles, 1.0)
    if tsb is not None:
        out["form"] = 1.0 - float(np.clip(tsb * params.form_coef, -params.form_cap, params.form_cap))
    if tech is not None:
        out["technicality"] = np.asarray(tech, dtype=float)
    if fitness is not None:
        out["fitness"] = float(fitness)
    if age is not None:
        out["age"] = float(age)
    return out
