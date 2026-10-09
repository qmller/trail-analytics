import argparse
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from trailsport import online
from trailsport.cli import _aid_stations, _start_time


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.headers = data, status, {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def _reset_failures():
    online._failed.clear()
    yield
    online._failed.clear()


def test_weather_is_cached(tmp_path):
    data = {"hourly": {"time": ["2026-10-02T07:00", "2026-10-02T08:00"], "temperature_2m": [10, 12], "precipitation": [0, 0.5]}}
    start = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    with mock.patch.object(online.requests, "request", return_value=_Resp(data)) as req:
        w1 = online.weather(45.2, 5.8, start, 2, cache_dir=tmp_path)
        w2 = online.weather(45.2, 5.8, start, 2, cache_dir=tmp_path)
    assert req.call_count == 1
    assert w1.equals(w2) and w1["temperature"].tolist() == [10, 12]


def test_online_failure_raises_and_is_not_retried(tmp_path):
    with mock.patch.object(online.requests, "request", side_effect=requests.ConnectionError("down")) as req:
        with pytest.raises(online.OnlineError):
            online._request("GET", "https://example.invalid", tmp_path)
        with pytest.raises(online.OnlineError):
            online._request("GET", "https://example.invalid", tmp_path, params={"x": 1})
    assert req.call_count == 1


def test_invalid_coordinates_rejected():
    with pytest.raises(online.OnlineError):
        online.weather(95.0, 5.0, pd.Timestamp.now(tz="UTC"), 1)


def test_technicality_matches_nearest_way(tmp_path):
    ways = {"elements": [
        {"tags": {"highway": "path", "sac_scale": "demanding_mountain_hiking", "surface": "rock"},
         "geometry": [{"lat": 45.0, "lon": 6.0}]},
        {"tags": {"highway": "residential", "surface": "asphalt"}, "geometry": [{"lat": 45.01, "lon": 6.0}]},
    ]}
    blocks = pd.DataFrame({"lat": [45.0001, 45.01, 45.02], "lon": [6.0, 6.0, 6.0]})
    with mock.patch.object(online.requests, "request", return_value=_Resp(ways)):
        m = online.technicality(blocks, cache_dir=tmp_path)
    assert m == pytest.approx([1.15 * 1.08, 0.97, 1.0])


def test_aid_station_parsing(tmp_path):
    assert _aid_stations("12.5:3, 24:5") == [(12.5, 3.0), (24.0, 5.0)]
    csv = tmp_path / "aid.csv"
    csv.write_text("km,minutes\n10,2\n", encoding="utf-8")
    assert _aid_stations(str(csv)) == [(10.0, 2.0)]
    for bad in ("12", "a:b", "-1:2"):
        with pytest.raises(argparse.ArgumentTypeError):
            _aid_stations(bad)


def test_start_time_gets_timezone():
    assert _start_time("2026-10-11T07:30").tzinfo is not None
    assert _start_time("2026-10-11T07:30+02:00").utcoffset().total_seconds() == 7200


def test_load_files_accepts_directories_and_dedupes(gpx_factory, tmp_path):
    from conftest import straight_track

    from trailsport.cli import _load_files

    a = gpx_factory(straight_track(500, 0, 2.0), "a.gpx")
    gpx_factory(straight_track(500, 0, 2.0), "b.gpx")
    assert [x.path.name for x in _load_files([a, tmp_path, a])] == ["a.gpx", "b.gpx"]
    with pytest.raises(SystemExit, match="file not found"):
        _load_files([tmp_path / "missing.gpx"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no .gpx file"):
        _load_files([tmp_path / "empty"])
