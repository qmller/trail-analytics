from datetime import datetime, timedelta, timezone

import pytest

M_PER_DEG_LAT = 111_194.93


def write_gpx(path, points, route=False):
    """points: dicts with lat, lon, ele and optional time (datetime), hr, cad, atemp."""
    pts = []
    tag = "rtept" if route else "trkpt"
    for p in points:
        body = f"<ele>{p['ele']:.2f}</ele>"
        if p.get("time"):
            body += f"<time>{p['time'].strftime('%Y-%m-%dT%H:%M:%SZ')}</time>"
        ext = "".join(f"<gpxtpx:{k}>{p[k]}</gpxtpx:{k}>" for k in ("atemp", "hr", "cad") if p.get(k) is not None)
        if ext:
            body += f"<extensions><gpxtpx:TrackPointExtension>{ext}</gpxtpx:TrackPointExtension></extensions>"
        pts.append(f'<{tag} lat="{p["lat"]:.7f}" lon="{p["lon"]:.7f}">{body}</{tag}>')
    inner = f"<rte><name>test</name>{''.join(pts)}</rte>" if route else f"<trk><name>test</name><trkseg>{''.join(pts)}</trkseg></trk>"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
        f"{inner}</gpx>",
        encoding="utf-8",
    )
    return path


def straight_track(length_m, grade_pct, speed, start=None, hr=150, pause_at=None, pause_s=0, step_s=1.0):
    """Points heading north every `step_s` seconds at constant speed and grade."""
    start = start or datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc)
    pts, t, d = [], 0.0, 0.0
    paused = False
    while d <= length_m:
        pts.append({
            "lat": 45.0 + d / M_PER_DEG_LAT, "lon": 6.0, "ele": 500.0 + d * grade_pct / 100.0,
            "time": start + timedelta(seconds=t), "hr": hr, "cad": 85, "atemp": 20,
        })
        if pause_at is not None and not paused and d >= pause_at:
            paused = True
            t += pause_s
            pts.append({**pts[-1], "time": start + timedelta(seconds=t)})
        t += step_s
        d += speed * step_s
    return pts


@pytest.fixture
def gpx_factory(tmp_path):
    def make(points, name="track.gpx", route=False):
        return write_gpx(tmp_path / name, points, route)
    return make
