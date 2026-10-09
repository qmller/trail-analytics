import json
import re

from trailsport.maps import course_map, recorded_blocks
from trailsport.preprocess import load_activity

from conftest import straight_track


def _payload(page: str) -> dict:
    return json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))


def test_recorded_map_layers_and_escaping(gpx_factory):
    act = load_activity(gpx_factory(straight_track(2500, 5.0, 2.0)))
    act.raw.attrs["waypoints"] = [{"name": "<script>alert(1)</script>", "type": "aid", "lat": 45.01, "lon": 6.0}]
    page = course_map(act, recorded_blocks(act), "Title </script>", "sub")
    data = _payload(page)
    assert set(data["layers"]) == {"grade", "pace", "time", "hr"}
    assert len(data["lines"]) == len(act.blocks)
    assert [m["label"] for m in data["markers"] if m["group"] == "km"] == ["1", "2"]
    assert "<script>alert" not in page and "Title </script>" not in page
    wp = next(m for m in data["markers"] if m["group"] == "waypoints")
    assert "&lt;script&gt;" in wp["text"] and "km 1.1" in wp["text"]


def test_synthetic_timestamps_are_ignored(gpx_factory):
    act = load_activity(gpx_factory(straight_track(2000, 0.0, 20.0)))  # 20 m/s is not running
    assert set(_payload(course_map(act, recorded_blocks(act), "t"))["layers"]) == {"grade", "hr"}


def test_hr_command_formats(gpx_factory, tmp_path):
    from trailsport.cli import main

    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    for i, hr in enumerate((140, 150, 160)):
        gpx_factory(straight_track(3000, 5.0 * i, 2.0, hr=hr), f"refs/r{i}.gpx")
    pts = [{k: v for k, v in p.items() if k in ("lat", "lon", "ele")} for p in straight_track(2000, 5.0, 2.0)]
    course = gpx_factory(pts, "course.gpx")
    out = tmp_path / "out"
    main(["hr", str(ref_dir / "r1.gpx"), str(course), "--ref-dir", str(ref_dir), "--out", str(out),
          "--format", "table", "csv", "json", "html", "map", "--plot", "none"])
    names = {p.name for p in out.iterdir()}
    assert {"activities.csv", "courses.csv", "zones_by_grade.csv", "hr.json", "hr.html",
            "r1_map.html", "course_map.html"} <= names
    for name in ("r1_map.html", "course_map.html"):
        assert "hr" in _payload((out / name).read_text(encoding="utf-8"))["layers"]
