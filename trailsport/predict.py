"""Reference context, model fitting and course prediction."""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import course as course_mod
from . import factors, hr, online
from .athlete import Profile
from .metrics import minetti_cost
from .model import PaceModel, features, make_model
from .preprocess import Activity

log = logging.getLogger(__name__)

SPM_RANGE = (2.5 * 60 / 1000, 40.0 * 60 / 1000)  # 2:30 .. 40:00 min/km, outside = GPS glitch or stop
HALF_LIFE_DAYS = 365.0


@dataclass
class Options:
    model: str = "curve"
    fatigue: bool = True
    start: pd.Timestamp | None = None
    temp: float | None = None
    wet: bool = False
    poles: bool = False
    form: bool = False
    aid_stations: list[tuple[float, float]] = field(default_factory=list)  # (km, minutes)
    target_hr: float | None = None
    online: bool = False
    profile: Profile = field(default_factory=Profile)
    params: factors.FactorParams = field(default_factory=factors.FactorParams)


@dataclass
class Context:
    """Everything derived from the reference activities."""

    refs: list[Activity]
    athlete: hr.Athlete
    feats: list[dict]
    ref_mult: list[pd.DataFrame]
    stats: list[dict]
    loads: pd.DataFrame | None
    hike_grade: float | None
    newest: pd.Timestamp | None
    ref_temp: float | None

    @property
    def use_hr(self) -> bool:
        return all(a.has_hr for a in self.refs) and self.athlete.known


def _online(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except online.OnlineError as exc:
        log.warning("online data unavailable, using defaults: %s", exc)
        return None


def reference_multipliers(act: Activity, opts: Options, loads) -> pd.DataFrame:
    """Multipliers for the conditions recorded during a reference activity."""
    b = act.blocks
    p = opts.params
    clock = None
    if act.start is not None and act.has_time:
        clock = act.start + pd.to_timedelta(b["t_elapsed_mid"], unit="s")
    temp = (b["atemp"] - p.watch_temp_offset).to_numpy() if b["atemp"].notna().any() else None
    tsb = hr.tsb_at(loads, act.start) if opts.form else None
    tech = _online(online.technicality, b) if opts.online else None
    age = None
    if act.start is not None:
        race_day = opts.start if opts.start is not None else pd.Timestamp.now(tz="UTC")
        age = opts.profile.age_multiplier((race_day - act.start).days / 365.25)
    return factors.multipliers(b, p, clock=clock, temp=temp, tsb=tsb, tech=tech, age=age)


def build_context(refs: list[Activity], opts: Options) -> Context:
    refs = [a for a in refs if a.has_time]
    if not refs:
        raise ValueError("no reference activity with timestamps")
    athlete = hr.estimate_athlete(refs, opts.profile)
    stats = [hr.activity_stats(a, athlete) for a in refs]
    loads = hr.load_series(stats, opts.start)
    starts = [a.start for a in refs if a.start is not None]
    temps = [a.blocks["atemp"].median() for a in refs if a.blocks["atemp"].notna().any()]
    return Context(
        refs=refs,
        athlete=athlete,
        feats=[course_mod.features(a) for a in refs],
        ref_mult=[reference_multipliers(a, opts, loads) for a in refs],
        stats=stats,
        loads=loads,
        hike_grade=course_mod.hike_threshold(refs),
        newest=max(starts) if starts else None,
        ref_temp=float(np.median(temps)) - opts.params.watch_temp_offset if temps else None,
    )


def activity_weights(ctx: Context, target: dict, exclude: int | None = None) -> np.ndarray:
    w = []
    for i, (act, feat) in enumerate(zip(ctx.refs, ctx.feats)):
        if i == exclude:
            w.append(0.0)
            continue
        age = (ctx.newest - act.start).days if ctx.newest is not None and act.start is not None else 0
        recency = 0.5 ** (age / HALF_LIFE_DAYS)
        w.append(recency * course_mod.similarity(feat, target) * hr.effort_weight(act, ctx.athlete))
    w = np.asarray(w)
    return w / w.max()


def fit_model(ctx: Context, opts: Options, target: dict, exclude: int | None = None):
    weights = activity_weights(ctx, target, exclude)
    use_hr = opts.target_hr is not None and ctx.use_hr
    Xs, ys, ws = [], [], []
    for act, mult, wa in zip(ctx.refs, ctx.ref_mult, weights):
        if wa <= 0:
            continue
        b = act.blocks
        spm = (b["dt"] / b["dx"] / mult.prod(axis=1)).to_numpy()
        ok = np.isfinite(spm) & (spm >= SPM_RANGE[0]) & (spm <= SPM_RANGE[1])
        hr_rel = b["hr"].to_numpy() / ctx.athlete.lthr if use_hr else None
        if use_hr:
            ok &= np.isfinite(hr_rel)
        Xs.append(features(b, hr_rel)[ok])
        ys.append(spm[ok])
        ws.append(wa * b["dx"].to_numpy()[ok])
    model = make_model(opts.model, fatigue=opts.fatigue)
    model.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws))
    return model, weights


def _hr_rel_for(ctx: Context, opts: Options, blocks: pd.DataFrame | None = None):
    if opts.target_hr is None or not ctx.use_hr:
        return None
    if blocks is not None:
        return (blocks["hr"] / ctx.athlete.lthr).fillna(opts.target_hr / ctx.athlete.lthr).to_numpy()
    return opts.target_hr / ctx.athlete.lthr


def aid_stops(blocks: pd.DataFrame, stations) -> np.ndarray:
    """Stop seconds added at the end of the block containing each aid station."""
    stops = np.zeros(len(blocks))
    dist = blocks["dist"].to_numpy()
    for km, minutes in stations:
        i = min(int(np.searchsorted(dist, km * 1000.0)), len(blocks) - 1)
        stops[i] += minutes * 60.0
    return stops


def loo(ctx: Context, opts: Options) -> pd.DataFrame:
    """Leave-one-activity-out: predict each reference from the others with its recorded conditions."""
    rows = []
    for i, act in enumerate(ctx.refs):
        if len(ctx.refs) < 2:
            break
        model, _ = fit_model(ctx, opts, ctx.feats[i], exclude=i)
        b = act.blocks
        pred = model.predict(features(b, _hr_rel_for(ctx, opts, b))) * b["dx"].to_numpy() * ctx.ref_mult[i].prod(axis=1).to_numpy()
        actual = b["dt"].to_numpy()
        km = np.ceil(b["dist"].to_numpy() / 1000.0 - 1e-9)
        per_km = pd.DataFrame({"km": km, "pred": pred, "actual": actual}).groupby("km").sum()
        rows.append({
            "name": act.name,
            "actual": actual.sum(),
            "predicted": pred.sum(),
            "error_pct": (pred.sum() - actual.sum()) / actual.sum() * 100.0,
            "split_mae_s": float((per_km["pred"] - per_km["actual"]).abs().mean()),
        })
    return pd.DataFrame(rows)


@dataclass
class Prediction:
    course: Activity
    model: PaceModel
    blocks: pd.DataFrame
    splits: pd.DataFrame
    sections: pd.DataFrame
    summary: dict
    breakdown: dict
    references: pd.DataFrame
    features: dict


def _race_conditions(course: Activity, opts: Options, hours: float):
    """Weather and technicality for the race (online), or None."""
    weather = tech = None
    if opts.online:
        tech = _online(online.technicality, course.blocks)
        if opts.start is not None:
            b = course.blocks
            weather = _online(online.weather, b["lat"].mean(), b["lon"].mean(), opts.start, hours + 1)
    return weather, tech


def predict_course(course: Activity, ctx: Context, opts: Options) -> Prediction:
    target = course_mod.features(course)
    model, weights = fit_model(ctx, opts, target)
    b = course.blocks
    dx = b["dx"].to_numpy()
    X = features(b, _hr_rel_for(ctx, opts))
    base = model.predict(X)
    X0 = X.assign(effort=0.0)
    base_fresh = model.predict(X0)
    aid = aid_stops(b, opts.aid_stations)
    tsb = hr.tsb_at(ctx.loads, opts.start or pd.Timestamp.now(tz="UTC")) if opts.form else None
    weather, tech = _race_conditions(course, opts, float((base * dx).sum()) / 3600.0)

    mult = np.ones(len(b))
    for _ in range(3):  # night / weather depend on predicted clock times
        dt = base * dx * mult
        elapsed_end = np.cumsum(dt + aid)
        elapsed_mid = elapsed_end - aid - dt / 2
        clock = opts.start + pd.to_timedelta(elapsed_mid, unit="s") if opts.start is not None else None
        temp, wet = opts.temp, opts.wet
        if weather is not None and clock is not None:
            hours = (weather.index - weather.index[0]).total_seconds().to_numpy()
            at = (clock - weather.index[0]).total_seconds().to_numpy()
            if temp is None:
                temp = np.interp(at, hours, weather["temperature"].to_numpy())
            in_race = (weather.index >= clock.min().floor("h")) & (weather.index <= clock.max().ceil("h"))
            wet = wet or bool(weather.loc[in_race, "precipitation"].sum() > opts.params.wet_precip_mm)
        if temp is None:
            temp = ctx.ref_temp
        M = factors.multipliers(b, opts.params, clock=clock, temp=temp, wet=wet, poles=opts.poles, tsb=tsb, tech=tech,
                                fitness=opts.profile.fitness_multiplier())
        mult = M.prod(axis=1).to_numpy()

    dt = base * dx * mult
    out = b.assign(
        pace=dt / dx,
        base=base * dx,
        time=dt,
        aid=aid,
        t_pred=np.cumsum(dt + aid),
    )
    out = pd.concat([out, M.add_prefix("m_")], axis=1)
    if clock is not None:
        out["clock"] = opts.start + pd.to_timedelta(out["t_pred"], unit="s")

    moving = float(dt.sum())
    total = moving + float(aid.sum())
    breakdown = {"base (fresh)": float((base_fresh * dx).sum()), "fatigue": float(((base - base_fresh) * dx).sum())}
    for f in factors.FACTORS:
        breakdown[f] = float((base * dx * (M[f].to_numpy() - 1.0)).sum())
    breakdown["aid stations"] = float(aid.sum())
    breakdown["total"] = total

    summary = {
        "course": course.name,
        "model": model.name,
        "distance_km": target["distance_km"],
        "d_plus": target["d_plus"],
        "d_minus": target["d_minus"],
        "effort_km": target["effort_km"],
        "moving": moving,
        "total": total,
        "pace": moving / dx.sum(),
        "start": opts.start,
        "finish": opts.start + pd.Timedelta(seconds=total) if opts.start is not None else None,
        "hike_grade": ctx.hike_grade,
        "fatigue_per_effort_km_pct": getattr(model, "fatigue_per_effort_km", math.nan) * 100.0,
        "energy_kcal": (opts.profile.weight * float((minetti_cost(b["grade"]) * dx).sum()) / 4184.0
                        if opts.profile.weight else None),
        "athlete_hr": " | ".join(
            f"{k} {getattr(ctx.athlete, k):.0f} ({src})" for k, src in ctx.athlete.sources.items()
        ),
    }
    ratios = loo(ctx, opts)
    if len(ratios) >= 3:
        r = np.log(ratios["actual"] / ratios["predicted"])
        summary["p10"] = total * float(np.exp(np.quantile(r, 0.1)))
        summary["p90"] = total * float(np.exp(np.quantile(r, 0.9)))
        summary["loo_mae_pct"] = float(ratios["error_pct"].abs().mean())

    refs = pd.DataFrame({
        "name": [a.name for a in ctx.refs],
        "weight": weights,
        "similarity": [course_mod.similarity(f, target) for f in ctx.feats],
        "effort": [hr.effort_weight(a, ctx.athlete) for a in ctx.refs],
    })
    return Prediction(course, model, out, _splits(out, ctx.hike_grade), _sections(course, out), summary, breakdown, refs, target)


def _splits(out: pd.DataFrame, hike_grade) -> pd.DataFrame:
    key = np.ceil(out["dist"] / 1000.0 - 1e-9)
    g = out.groupby(key)
    s = pd.DataFrame({
        "km": g["dist"].last() / 1000.0,
        "d_plus": g["dplus"].sum(),
        "d_minus": g["dminus"].sum(),
        "grade": g["dz"].sum() / g["dx"].sum() * 100.0,
        "pace": g["time"].sum() / g["dx"].sum(),
        "time": g["time"].sum(),
        "aid": g["aid"].sum(),
        "cum_time": g["t_pred"].last(),
    }).reset_index(drop=True)
    if "clock" in out:
        s["clock"] = g["clock"].last().to_numpy()
    if hike_grade is not None:
        s["mode"] = np.where(s["grade"] >= hike_grade, "hike", "run")
    return s


def _sections(course: Activity, out: pd.DataFrame) -> pd.DataFrame:
    sec = course_mod.sections(course.segs)
    if sec.empty:
        return sec
    mid = (out["dist"] - out["dx"] / 2).to_numpy() / 1000.0
    times = []
    for start, end in zip(sec["start_km"], sec["end_km"]):
        times.append(float(out["time"][(mid >= start) & (mid < end)].sum()))
    sec["time"] = times
    sec["pace"] = sec["time"] / (sec["length_km"] * 1000.0)
    return sec
