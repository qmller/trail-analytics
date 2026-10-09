"""End-to-end checks on the real reference data (skipped when absent)."""

from pathlib import Path

import pytest

from trailsport.predict import Options, build_context, loo, predict_course
from trailsport.preprocess import load_activity, load_dir

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "gpx"
COURSE = ROOT / "predict" / "course_moyen_duc_2026_39km_2350m.gpx"
BASELINE_S = 5 * 3600 + 38 * 60 + 56  # legacy_main.py, see tests/fixtures/baseline_course_moyen_duc.txt
BASELINE_REFS = [
    "col_de_l_alpe_30k_1800m.gpx",
    "course_gap_en_cimes_les_3_cols_47k_2800m.gpx",
    "course_grenoble_vizille_2026_30k_1000m.gpx",
    "le_dragon_débrouillard_40k_3400m.gpx",
    "le_martignon_24k_1100m.gpx",
    "letouvet_montbonnot_27km_1000m.gpx",
]

pytestmark = pytest.mark.skipif(not (REF_DIR.is_dir() and COURSE.is_file()), reason="reference data not available")


@pytest.fixture(scope="module")
def refs():
    return load_dir(REF_DIR)


@pytest.fixture(scope="module")
def course():
    return load_activity(COURSE)


def test_bins_model_close_to_legacy_baseline(refs, course):
    # Differences come from the bug fixes: moving time only, per-activity smoothing, time-weighted bins.
    refs = [a for a in refs if a.path.name in BASELINE_REFS]
    assert len(refs) == len(BASELINE_REFS)
    opts = Options(model="bins")
    pred = predict_course(course, build_context(refs, opts), opts)
    assert pred.summary["moving"] == pytest.approx(BASELINE_S, rel=0.15)


@pytest.mark.parametrize("model", ["curve", "gbm"])
def test_models_beat_bins_in_leave_one_out(refs, model):
    ctx = build_context(refs, Options())
    mae = {m: loo(ctx, Options(model=m))["error_pct"].abs().mean() for m in ("bins", model)}
    assert mae[model] < mae["bins"]


def test_prediction_outputs(refs, course, tmp_path):
    import pandas as pd

    from trailsport import report

    opts = Options(start=pd.Timestamp("2026-10-11 07:30", tz="Europe/Paris"), aid_stations=[(12.5, 3), (25, 5)], poles=True)
    pred = predict_course(course, build_context(refs, opts), opts)
    assert pred.breakdown["aid stations"] == 480
    assert pred.summary["total"] == pytest.approx(pred.summary["moving"] + 480)
    assert pred.splits["km"].iloc[-1] == pytest.approx(pred.summary["distance_km"], abs=0.01)
    assert pred.summary["p10"] < pred.summary["total"] < pred.summary["p90"]
    files = report.write_prediction(pred, tmp_path, ["csv", "json", "html"])
    assert {f.name for f in files} == {"splits.csv", "sections.csv", "blocks.csv", "prediction.json", "report.html"}
