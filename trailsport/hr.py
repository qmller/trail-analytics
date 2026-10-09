"""Heart-rate cleaning, athlete estimates, zones, decoupling, training load."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .athlete import Profile, tanaka_hrmax

if TYPE_CHECKING:
    from .preprocess import Activity

HR_VALID = (35.0, 230.0)
HR_MAX_JUMP = 25.0  # bpm between consecutive samples
ZONE_BOUNDS = (0.85, 0.90, 0.95, 1.00)  # Friel running zones, fraction of LTHR
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
CTL_DAYS, ATL_DAYS = 42.0, 7.0


def clean_hr(hr: pd.Series) -> pd.Series:
    hr = hr.where((hr >= HR_VALID[0]) & (hr <= HR_VALID[1]))
    hr = hr.where(~(hr.diff().abs() > HR_MAX_JUMP))
    return hr.interpolate(limit=10, limit_area="inside")


@dataclass
class Athlete:
    hrmax: float = math.nan
    hrrest: float = math.nan  # rough: lowest sustained HR during exercise
    lthr: float = math.nan
    sources: dict = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return math.isfinite(self.lthr)


def _hr_series(act: Activity) -> pd.Series:
    raw = act.raw.dropna(subset=["hr", "time"])
    return pd.Series(raw["hr"].to_numpy(), index=pd.DatetimeIndex(raw["time"])).sort_index()


def estimate_athlete(acts: list[Activity], profile: Profile | None = None) -> Athlete:
    """HR values from the profile when given, else estimated from the GPX files (HRmax: Tanaka from age as last resort)."""
    maxes, rests, lthrs = [], [], []
    for act in acts:
        s = _hr_series(act)
        if len(s) < 60:
            continue
        maxes.append(s.rolling("30s").median().max())
        rests.append(s.rolling("60s").median().min())
        if s.index[-1] - s.index[0] >= pd.Timedelta("30min"):
            roll = s.rolling("30min").mean()
            lthrs.append(roll[roll.index >= s.index[0] + pd.Timedelta("30min")].max())
    profile = profile or Profile()
    sources = {}

    def pick(name, given, observed, fallback=None, fallback_src=""):
        if given:
            sources[name] = "profile"
            return float(given)
        if observed is not None:
            sources[name] = "gpx"
            return float(observed)
        if fallback is not None:
            sources[name] = fallback_src
            return float(fallback)
        return math.nan

    hrmax = pick("hrmax", profile.hrmax, max(maxes) if maxes else None,
                 tanaka_hrmax(profile.age) if profile.age else None, "age (Tanaka)")
    lthr = pick("lthr", profile.lthr, max(lthrs) if lthrs else None,
                0.9 * hrmax if math.isfinite(hrmax) else None, "0.9 x HRmax")
    hrrest = pick("hrrest", profile.hrrest, min(rests) if rests else None)
    return Athlete(hrmax=hrmax, hrrest=hrrest, lthr=lthr, sources=sources)


def zone_of(hr, lthr):
    return np.digitize(np.asarray(hr, dtype=float) / lthr, ZONE_BOUNDS)


def _raw_dt(act: Activity) -> np.ndarray:
    t = (act.raw["time"] - act.raw["time"].iloc[0]).dt.total_seconds().to_numpy()
    return np.clip(np.diff(t, prepend=t[0]), 0.0, 10.0)  # ignore recording gaps


def efficiency(blocks: pd.DataFrame) -> tuple[float, float]:
    """(efficiency factor, decoupling %) on grade-adjusted speed vs HR, first vs second half."""
    from .metrics import gap_factor

    b = blocks[(blocks["dt"] > 0) & blocks["hr"].notna()]
    if len(b) < 10:
        return math.nan, math.nan
    half = b["t"] <= b["t"].iloc[-1] / 2

    def ef(part):
        speed = (part["dx"] * gap_factor(part["grade"])).sum() / part["dt"].sum() * 60.0
        return speed / np.average(part["hr"], weights=part["dt"])

    ef1, ef2 = ef(b[half]), ef(b[~half])
    return ef(b), (ef1 - ef2) / ef1 * 100.0


def activity_stats(act: Activity, ath: Athlete) -> dict:
    hr = act.raw["hr"].to_numpy(float)
    ok = np.isfinite(hr)
    out = {"name": act.name, "start": act.start}
    if not ok.any() or not ath.known or act.start is None:
        return out
    dt = _raw_dt(act)[ok]
    hr = hr[ok]
    zones = zone_of(hr, ath.lthr)
    for i, z in enumerate(ZONES):
        out[z] = float(dt[zones == i].sum())
    out["hr_mean"] = float(np.average(hr, weights=dt)) if dt.sum() else math.nan
    out["hr_max"] = float(hr.max())
    out["hrTSS"] = float((dt / 3600.0 * (hr / ath.lthr) ** 2).sum() * 100.0)
    x = np.clip((hr - ath.hrrest) / (ath.hrmax - ath.hrrest), 0.0, 1.0)
    out["TRIMP"] = float((dt / 60.0 * x * 0.64 * np.exp(1.92 * x)).sum())
    out["EF"], out["decoupling"] = efficiency(act.blocks)
    return out


def effort_weight(act: Activity, ath: Athlete) -> float:
    """1 for race-like effort (mean HR >= 90% LTHR), down to 0.2 for easy runs."""
    if not ath.known or not act.has_hr:
        return 1.0
    ratio = act.raw["hr"].mean() / ath.lthr
    return float(np.clip((ratio - 0.75) / 0.15, 0.2, 1.0))


def load_series(stats: list[dict], until=None) -> pd.DataFrame | None:
    """Daily hrTSS with CTL / ATL / TSB (Banister impulse-response, EWMA)."""
    rows = [(s["start"], s["hrTSS"]) for s in stats if "hrTSS" in s]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "load"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.normalize()
    daily = df.groupby("date")["load"].sum()
    end = pd.Timestamp(until).tz_localize(None).normalize() if until is not None else daily.index.max()
    idx = pd.date_range(daily.index.min() - pd.Timedelta(days=1), max(end, daily.index.max()), freq="D")
    load = daily.reindex(idx, fill_value=0.0)
    ctl = load.ewm(alpha=1.0 / CTL_DAYS, adjust=False).mean()
    atl = load.ewm(alpha=1.0 / ATL_DAYS, adjust=False).mean()
    return pd.DataFrame({"load": load, "ctl": ctl, "atl": atl, "tsb": ctl - atl})


def tsb_at(loads: pd.DataFrame | None, when) -> float | None:
    """Form entering the day `when` (TSB at the end of the previous day)."""
    if loads is None or when is None:
        return None
    day = pd.Timestamp(when)
    day = (day.tz_convert(None) if day.tzinfo else day).normalize() - pd.Timedelta(days=1)
    prev = loads.loc[:day]
    return float(prev["tsb"].iloc[-1]) if len(prev) else None


def zone_by_grade(acts: list[Activity], ath: Athlete, bin_size=5.0) -> pd.DataFrame:
    """Share of moving time (%) in each HR zone per grade bin."""
    frames = [a.blocks[["grade", "hr", "dt"]] for a in acts if a.has_hr and a.has_time]
    if not frames or not ath.known:
        return pd.DataFrame()
    b = pd.concat(frames).dropna()
    b = b[b["dt"] > 0]
    edges = np.arange(-40.0, 40.0 + bin_size, bin_size)
    b = b.assign(grade_bin=pd.cut(b["grade"], edges), zone=[ZONES[z] for z in zone_of(b["hr"], ath.lthr)])
    table = b.pivot_table(index="grade_bin", columns="zone", values="dt", aggfunc="sum", observed=True).fillna(0.0)
    table = table.reindex(columns=ZONES, fill_value=0.0)
    return table.div(table.sum(axis=1), axis=0).mul(100.0).round(1)


def course_estimate_blocks(course: Activity, refs: list[Activity], ath: Athlete, bin_size=5.0) -> pd.DataFrame | None:
    """Per course block: time, mean HR and time per HR zone, if run like the reference activities.

    Each block takes the time-weighted pace, zone shares and mean HR of its grade bin in the references.
    """
    from .model import BinLookupModel, features
    from .predict import SPM_RANGE

    frames = [a.blocks for a in refs if a.has_hr and a.has_time]
    if not frames or not ath.known:
        return None
    b = pd.concat(frames, ignore_index=True)
    b = b[(b["dt"] > 0) & b["hr"].notna()]
    spm = (b["dt"] / b["dx"]).to_numpy()
    ok = (spm >= SPM_RANGE[0]) & (spm <= SPM_RANGE[1])
    b, spm = b[ok], spm[ok]
    pace = BinLookupModel().fit(features(b), spm, b["dx"].to_numpy())

    edges = np.arange(-60.0, 60.0 + bin_size, bin_size)
    zones = pd.get_dummies(pd.Categorical([ZONES[z] for z in zone_of(b["hr"], ath.lthr)], categories=ZONES))
    ref = zones.mul(b["dt"].to_numpy(), axis=0).assign(hr=(b["hr"] * b["dt"]).to_numpy(), dt=b["dt"].to_numpy())
    by_bin = ref.groupby(pd.cut(b["grade"], edges).to_numpy(), observed=True).sum()
    overall = ref.sum()
    share = by_bin.div(by_bin["dt"], axis=0)

    cb = course.blocks
    t = pace.predict(features(cb)) * cb["dx"].to_numpy()
    rows = share.reindex(pd.cut(cb["grade"], edges)).fillna(overall / overall["dt"]).reset_index(drop=True)
    out = rows[ZONES].mul(t, axis=0)
    out["hr"] = rows["hr"].to_numpy()
    out["time"] = t
    return out


def course_estimate(course: Activity, refs: list[Activity], ath: Athlete, bin_size=5.0) -> dict:
    """Expected moving time, mean HR and time per HR zone on a course (see `course_estimate_blocks`)."""
    est = course_estimate_blocks(course, refs, ath, bin_size)
    if est is None:
        return {"name": course.name}
    total = float(est["time"].sum())
    out = {"name": course.name, "time": total, "hr_mean": float((est["hr"] * est["time"]).sum() / total)}
    out.update({z: float(est[z].sum()) for z in ZONES})
    return out
