import math

import numpy as np
import pandas as pd
import pytest

from trailsport import course, factors, hr, metrics
from trailsport.gpx import read_gpx
from trailsport.model import BinLookupModel, CurveModel, features
from trailsport.preprocess import haversine, hysteresis_gain, load_activity

from conftest import straight_track


def test_read_gpx_extensions(gpx_factory):
    df = read_gpx(gpx_factory(straight_track(100, 0, 2.0, hr=142)))
    assert list(df.columns) == ["lat", "lon", "ele", "time", "hr", "cad", "atemp"]
    assert (df["hr"] == 142).all() and (df["cad"] == 85).all() and (df["atemp"] == 20).all()
    assert df["time"].dt.tz is not None


def test_read_gpx_route_without_time(gpx_factory):
    pts = [{k: v for k, v in p.items() if k in ("lat", "lon", "ele")} for p in straight_track(200, 5, 2.0)]
    df = read_gpx(gpx_factory(pts, route=True))
    assert len(df) == len(pts)
    assert df["time"].isna().all() and df["hr"].isna().all()


def test_haversine_one_degree_latitude():
    assert haversine(45.0, 6.0, 46.0, 6.0) == pytest.approx(111_195, rel=1e-4)


def test_hysteresis_ignores_noise():
    ele = np.r_[np.zeros(50) + np.tile([0, 1], 25), np.linspace(0, 100, 50)]
    up, down = hysteresis_gain(ele, threshold=3.0)
    assert up == pytest.approx(100, abs=3) and down < 3


def test_constant_grade_track(gpx_factory):
    act = load_activity(gpx_factory(straight_track(2000, 10.0, 1.0)))
    inner = act.blocks.iloc[2:-2]
    assert inner["grade"].to_numpy() == pytest.approx(10.0, abs=0.5)
    assert act.blocks["dx"].sum() == pytest.approx(2000, abs=5)
    d_plus, _ = metrics.elevation_totals(act.segs)
    assert d_plus == pytest.approx(200, abs=10)
    assert (act.blocks["dt"] / act.blocks["dx"]).iloc[2:-2].to_numpy() == pytest.approx(1.0, rel=0.05)


def test_pause_detection(gpx_factory):
    act = load_activity(gpx_factory(straight_track(1000, 0.0, 2.0, pause_at=500, pause_s=120)))
    b = act.blocks
    assert b["elapsed"].sum() - b["dt"].sum() == pytest.approx(120, abs=10)
    assert b["dt"].sum() == pytest.approx(500, abs=10)


def test_course_without_time(gpx_factory):
    pts = [{k: v for k, v in p.items() if k in ("lat", "lon", "ele")} for p in straight_track(1500, 5, 2.0)]
    act = load_activity(gpx_factory(pts))
    assert not act.has_time and act.start is None
    assert act.blocks["dt"].isna().all()


def test_format_duration_beyond_24h():
    assert metrics.format_duration(25 * 3600 + 61) == "25:01:01"
    assert metrics.format_duration(math.nan) == "--"
    assert metrics.format_pace(0.36) == "6:00"


def test_vo2_units():
    # 10 km/h on the flat ~ 3.5 + 0.2 * 166.7 m/min
    assert float(metrics.vo2(10 / 3.6, 0.0)) == pytest.approx(36.8, abs=0.1)


def test_sections_sawtooth():
    dist = np.arange(1, 601) * 20.0
    ele = 500 + 100 * np.abs(((dist / 1000.0) % 2) - 1)  # 100 m up/down every km
    segs = pd.DataFrame({"dist": dist, "ele": ele, "grade": np.gradient(ele, dist) * 100, "dx": 20.0})
    segs.attrs["ele0"] = 600.0
    sec = course.sections(segs)
    kinds = sec["kind"].tolist()
    assert kinds.count("climb") == 6 and kinds.count("descent") == 6
    assert sec.loc[sec["kind"] == "climb", "d_plus"].to_numpy() == pytest.approx(100, abs=1)


def test_clean_hr_removes_spikes():
    s = pd.Series([120.0, 121, 250, 122, 123, 20, 124, 190, 125])
    out = hr.clean_hr(s)
    assert out.max() < 130 and out.min() > 115


def test_zones_and_load():
    assert hr.zone_of([120, 140, 145, 152, 160, 170], 160).tolist() == [0, 1, 2, 3, 4, 4]
    stats = [{"start": pd.Timestamp("2026-06-01", tz="UTC"), "hrTSS": 100.0}]
    loads = hr.load_series(stats, pd.Timestamp("2026-06-10", tz="UTC"))
    assert loads.loc["2026-06-01", "ctl"] == pytest.approx(100 / 42)
    assert hr.tsb_at(loads, pd.Timestamp("2026-06-02", tz="UTC")) == pytest.approx(100 / 42 - 100 / 7)


def test_solar_elevation_and_night_factor():
    noon = pd.Timestamp("2026-03-20 12:00", tz="UTC")
    midnight = pd.Timestamp("2026-03-20 00:00", tz="UTC")
    elev = factors.solar_elevation([noon, midnight], [0.0, 45.0], [0.0, 6.0])
    assert elev[0] > 85 and elev[1] < -30
    blocks = pd.DataFrame({"grade": [0.0, -20.0], "ele": [500.0, 2500.0], "lat": [45.0, 45.0], "lon": [6.0, 6.0]})
    m = factors.multipliers(blocks, factors.FactorParams(), clock=[noon, midnight], wet=True, temp=25.0)
    assert m["night"].tolist() == [1.0, 1.1]
    assert m["wet"].tolist() == [1.0, 1.15]
    assert m["altitude"].tolist() == pytest.approx([1.0, 1.1])
    assert m["heat"].tolist() == pytest.approx([1.04, 1.04])


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    grade = rng.uniform(-30, 30, n)
    effort = rng.uniform(0, 60, n)
    blocks = pd.DataFrame({"grade": grade, "ele": 1000.0, "effort": effort})
    true = 0.36 * np.exp(1.2 * (grade / 100) ** 2 * 10 + 0.8 * grade / 100) * np.exp(0.004 * effort)
    return features(blocks), true * np.exp(rng.normal(0, 0.05, n))


def test_curve_model_recovers_fatigue():
    X, spm = _synthetic()
    m = CurveModel().fit(X, spm, np.ones(len(spm)))
    assert m.fatigue_per_effort_km == pytest.approx(0.004, abs=0.001)
    assert m.predict(X.iloc[:1].assign(grade=0.0, effort=0.0))[0] == pytest.approx(0.36, rel=0.05)


def test_bins_model_interpolates():
    X = features(pd.DataFrame({"grade": [-10.2, -10.4, -10.6, 10.2, 10.4, 10.6], "ele": 0.0, "effort": 0.0}))
    m = BinLookupModel().fit(X, np.array([0.3, 0.3, 0.3, 0.9, 0.9, 0.9]), np.ones(6))
    assert m.predict(X.iloc[:1].assign(grade=0.0))[0] == pytest.approx(0.6, abs=0.01)


def test_course_hr_estimate(gpx_factory):
    ref = load_activity(gpx_factory(straight_track(3000, 5.0, 2.0, hr=150), "ref.gpx"))
    pts = [{k: v for k, v in p.items() if k in ("lat", "lon", "ele")} for p in straight_track(2000, 5.0, 2.0)]
    course = load_activity(gpx_factory(pts, "course.gpx"))
    est = hr.course_estimate(course, [ref], hr.Athlete(hrmax=180, hrrest=50, lthr=160))
    assert est["time"] == pytest.approx(1000, rel=0.05)
    assert est["hr_mean"] == pytest.approx(150)
    assert est["Z3"] == pytest.approx(est["time"])  # 150 / 160 = 94 % of LTHR
