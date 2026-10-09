# trail-sport

Trail-running race time predictor based on your own GPX history.

The `trailsport` package reads past activities (GPX 1.1, e.g. Strava exports, including
heart rate, cadence and temperature), learns a personal **pace vs. grade** model with fatigue,
and applies it to a course GPX with external factors (altitude, heat, night, wet trails,
technicality, poles, form, aid stations) to predict splits and finish time.

## How it works

1. **Parse** (`gpx.py`): track (or route) points with `hr`, `cad`, `atemp` extensions.
2. **Preprocess** (`preprocess.py`), per activity: resampling every 20 m, Savitzky–Golay
   elevation smoothing, pause detection (speed < 0.3 m/s), 100 m blocks with grade,
   D+/D-, moving/elapsed time, HR, cadence, temperature, turn density, cumulated effort-km
   (`km + D+/100`).
3. **Reference context** (`predict.py`): athlete HR estimates, recorded conditions turned into
   pace multipliers (`factors.py`) so paces are normalised, activity weights
   (recency × similarity to the course × effort from HR).
4. **Pace model** (`model.py`), fitted on normalised paces of reference blocks:
   - `bins`: time-weighted pace per 1 % grade bin (fixed version of the original script);
   - `curve` (default): `log(pace) = poly4(grade) + k·effort-km [+ b·(HR/LTHR − 1)]`, linear
     extension beyond the observed grades;
   - `gbm`: LightGBM on grade, neighbour grades, altitude, effort-km [, HR].
5. **Predict**: base pace per 100 m × factor multipliers (iterated, since night and weather depend
   on the predicted clock time) + aid-station stops. P10–P90 range from leave-one-out errors.
6. **Report** (`report.py`): per-km splits (with run/hike mode from cadence), climb/descent
   sections, time breakdown per factor, CSV/JSON/HTML exports and plots.

## Install

Requires Python ≥ 3.11.

```bash
cd trail-sport
python -m venv pyenv
source pyenv/bin/activate
uv pip install -e ".[dev]"
```

## Procedure

1. Export your past trail activities as GPX (Strava: *Activity › … › Export GPX*) into
   `data/gpx/`. Prefer activities with similar terrain and race-like effort.
2. Put the course GPX in `predict/` (timestamps not needed).
3. Check the reference data: `trailsport analyze` and `trailsport hr`.
4. Choose a model with `trailsport validate`, then run `trailsport predict`.

## Usage

```bash
source pyenv/bin/activate

# prediction (python main.py <course.gpx> is a shortcut for `predict`)
trailsport predict predict/course_moyen_duc_2026_39km_2350m.gpx
trailsport predict predict/course.gpx --model gbm \
    --start-time 2026-10-11T07:30 --aid-stations 12.5:3,25:5 --poles \
    --temp 22 --wet --target-hr 150 --form --online \
    --format table csv json html map --plot save --out output/my_race

trailsport analyze                  # summary + complexity of data/gpx/*.gpx
trailsport analyze predict/*.gpx    # same for courses, with climb/descent sections
trailsport validate                 # leave-one-out comparison of bins / curve / gbm
trailsport hr --date 2026-10-11     # HR zones, decoupling, efficiency, load / form
trailsport hr data/gpx/known_course*.gpx predict/course.gpx  # one activity, or HR zones expected on a course
trailsport hr data/gpx/run.gpx predict --format table csv json html map  # files/folders; exports + HR maps
trailsport map data/gpx/*.gpx       # maps of recorded activities (actual pace / time / HR)
```

| Option | Effect |
|---|---|
| `--model {bins,curve,gbm}` | pace model (default `curve`) |
| `--no-fatigue` | disable the effort-km fatigue term |
| `--start-time` | ISO date/time, local timezone if no offset; enables night and weather |
| `--temp` | race air temperature; default: median of the references |
| `--wet` | slower descents steeper than −10 % |
| `--poles` | faster climbs steeper than 15 % |
| `--aid-stations` | `km:minutes,...` or CSV with `km,minutes` columns |
| `--target-hr` | predict at a planned heart rate (fits pace = f(grade, HR)) |
| `--form` | ±3 % max from training stress balance (only the reference activities are counted) |
| `--online` | Open-Meteo forecast/archive and OpenStreetMap trail tags (`sac_scale`, `surface`), cached in `.cache/` |
| `--plot {show,save,none}` | interactive windows, PNG files in `--out`, or nothing |
| `--format` | any of `table csv json html map` |
| `--exclude PATTERN...` | ignore reference files matching a glob (e.g. `'Grand_duc*'`); all commands |
| `--athlete FILE` | athlete profile TOML (default `./athlete.toml`, see `athlete.example.toml`) |
| `--age`, `--weight`, `--vo2max` | current values: HRmax fallback, age decline of old runs, energy estimate |
| `--ref-vo2max`, `--ref-weight` | values when the reference runs were recorded: race pace × VO2max ratio (±15 % max) |
| `--hrmax`, `--lthr`, `--hrrest` | known heart-rate values, override the GPX estimates |

Factor defaults (heat +2 % per 5 °C above 15 °C, night ×1.10, wet ×1.15, altitude +1 % per 100 m
above 1500 m, etc.) are in `FactorParams` in `trailsport/factors.py`.

## Maps

`--format map` (predict) and `trailsport map` write a standalone `map.html` (Leaflet, OpenTopoMap or
OpenStreetMap tiles, internet needed to display it). Open it in a browser:

- switchable coloured layers: **pace** (green fast, red slow), **grade** (blue descents, red climbs,
  saturates at ±30 %), **elapsed time**; hover a section for km, grade, pace, time and clock time;
- km markers with the predicted (or recorded) time and clock time;
- **aid stations** (`A`, from `--aid-stations`) with arrival time and stop duration;
- **GPX waypoints** (`W`, e.g. summits, water points, cut-off times like "Col du Coq ; BH 13h30")
  with the predicted arrival time at the nearest point of the course ("off course" beyond 200 m);
- start (`S`) and finish (`F`).

For recorded activities, the map shows the actual pace, time and heart rate. Timestamps generated by route
planners (implausible pace) are ignored, so only the grade layer is shown.

`trailsport hr ... --format map` writes one map per GPX with a **heart-rate** layer: recorded HR for
activities, or HR and pace estimated from the reference activities (same grade) for courses.
With `csv`/`json`/`html`, `hr` also writes `activities.csv`, `courses.csv`, `zones_by_grade.csv`,
`hr.json` and `hr.html` to `--out` (default `output/<file>` for one GPX, else `output/hr`).

## Validation: choosing and trusting a model

`trailsport validate` answers "how wrong would the prediction have been on races I already ran?".
For each reference activity, it fits the model on **all the other** activities (leave-one-out),
predicts the held-out course with the conditions recorded that day (temperature, night, altitude)
and compares with the actual **moving time** (stops excluded).

```bash
trailsport validate                                 # the 3 models, default options
trailsport validate --models curve gbm --no-fatigue # effect of an option
trailsport validate --target-hr 155 --age 42        # HR-driven model, athlete profile
trailsport validate --exclude 'Grand_duc*' 'Trail_des_petites*'
```

Per model, one line per activity:

| Column | Meaning |
|---|---|
| `actual` / `predicted` | moving time recorded / predicted without that activity |
| `error_pct` | (predicted − actual) / actual; negative = the model is too optimistic |
| `split_mae_s` | mean absolute error of the per-km splits, in seconds |

Then a summary (also written to `output/validate/summary.csv`, details in `leave_one_out.csv`):

```
model  finish_mae_pct  finish_bias_pct  worst_pct  worst                          split_mae
 bins            10.6             -0.9       22.8  Grenoble vizille 2026 30k        0:01:31
curve             8.8             -3.8      -23.1  Soupe de cailloux au Moretan     0:01:25
  gbm             6.2             -2.9      -13.3  Sentier des gardes et col de M   0:01:03
```

How to use it:

1. **Pick the model** with the lowest `finish_mae_pct` and `split_mae`, then use it in `predict --model`.
   Re-run after adding activities: the ranking can change as data grows.
2. **Correct the bias**: `finish_bias_pct` = −3 % means predictions are on average 3 % too fast;
   add that margin to the plan (the P10–P90 range of `predict` is built from these same errors).
3. **Test options** the same way: keep an option (`--no-fatigue`, `--target-hr`, athlete values) only if
   it lowers the error.
4. **Inspect the worst activities**: a large error usually means an unusual day (injury, heat, snow,
   long stops counted as moving, very technical terrain). Exclude them with `--exclude` if they
   don't represent the race you plan.
5. **Back-test a race you already ran**: exclude its recording and predict its course, e.g.
   `trailsport predict predict/course_moyen_duc_2026_39km_2350m.gpx --exclude 'Grand_duc*' --model gbm`
   (predicted 6:37 moving vs 7:01 actual).

Caveats: leave-one-out is optimistic when several activities cover the same trails or the same event
(e.g. two parts of the same race), and the moving time excludes stops, so compare with the
race's elapsed time only after adding aid-station stops.

`tests/` contains unit tests on synthetic GPX files and end-to-end checks on `data/gpx`
(`pytest`); its output for the example course is in `tests/fixtures/baseline_course_moyen_duc.txt`.

## Limitations

- HRmax, LTHR and resting HR are estimated from the GPX files only; resting HR is a rough proxy
  (lowest sustained HR during exercise), used only by TRIMP.
- The watch temperature reads warmer than the air (`watch_temp_offset` = 5 °C).
- Form (TSB) only counts the activities in `data/gpx/`, so it's off by default (`--form`).
- `gbm` can't extrapolate: pace is flat beyond the steepest grades and longest efforts in the data.
- The Overpass API is rate limited; the first `--online` run can be slow or fall back to defaults.
- Leave-one-out validation is optimistic when reference courses share the same trails.

## Improvement tracks

> Status: items 1–4 below are implemented, with two exceptions: the per-segment quantile GBM is
> replaced by a leave-one-out P10–P90 range on the finish time, and turn density is reported by
> `analyze` but not used by the models.

### 1. Time prediction on a specific GPX
- **Accept the course as a CLI argument** (`argparse`: `course`, `--ref-dir`, `--bin-km`, `--bin-grade`, `--no-plot`).
- **Clean the input signal**: resample the track at a fixed distance step (e.g. 10–25 m),
  smooth elevation (Savitzky–Golay or a DEM correction) before computing grade/D+, and use
  3D distance.
- **Detect pauses**: drop segments where speed < ~0.3 m/s or `dt` > N s. Report moving time
  and stop time separately.
- **Fit a continuous pace(grade) model** instead of looking up the nearest bin: polynomial or
  Minetti-cost based fit, isotonic/LOESS regression, weighted by time spent in each bin.
  This handles steep grades that have little data and removes `NaN` bins.
- **Use finer segments** for the target course: split on grade changes (climbs/descents), not
  fixed km bins, and integrate per 50–100 m.
- **Model fatigue**: pace multiplier as a function of elapsed time or cumulated effort
  (e.g. Riegel exponent `T2 = T1·(D2/D1)^1.06`, or a decay fitted on the late vs. early splits of
  long reference runs).
- **Weight reference activities** by recency, similarity (distance, D+/km, terrain) and race
  vs. training effort.
- **Validate**: leave-one-out on the reference activities (predict each one from the others), and
  report MAE / % error on finish time and splits.
- **Machine learning** once enough data is available: gradient boosting (LightGBM/XGBoost) on
  per-segment features → pace, with quantile regression for an optimistic/realistic/pessimistic range.

### 2. Heart-rate analysis
- Parse `gpxtpx:hr`, `cad`, `atemp` in `parse_gpx()` (namespace
  `http://www.garmin.com/xmlschemas/TrackPointExtension/v1`).
- HR zones (from HRmax/HRrest or LTHR) → time in zone per activity and per grade bin.
- **Cardiac drift / decoupling**: ratio of GAP speed to HR, first half vs. second half
  (Pa:HR). > 5 % suggests insufficient aerobic endurance for that duration.
- **Efficiency factor** = GAP speed / HR, tracked over time as a fitness trend.
- **Training load**: TRIMP or hrTSS per activity; CTL/ATL/TSB for current form on race day.
- **Effort-normalised reference**: build pace(grade) only from segments in a target HR zone
  (race intensity), or model `pace = f(grade, HR)` and predict at the planned race HR.

### 3. Course complexity
Features to compute on the target GPX (and on reference runs so they are comparable):
- D+/km and D-/km, max/percentile grades, length of climbs > 15 % / descents < -15 %.
- Number and length of climbs/descents (segmentation on smoothed elevation).
- **Effort distance** / "km-effort" (ITRA-like): `km + D+/100` (and a descent term).
- Altitude (mean/max): VO2max drops ~1 % per 100 m above ~1500 m.
- **Technicality**: curvature/turn density (heading changes per km), speed variance on flat
  sections of reference runs, OSM `sac_scale` / `trail_visibility` / `surface` tags along the track.
- Position of the hard sections (a steep climb at km 35 costs more than at km 5).

### 4. Other factors to take into account
| Factor | Source | Use in the model |
|---|---|---|
| Temperature / humidity | `gpxtpx:atemp`, weather API (Open-Meteo) | Heat penalty on pace (~1–3 % per 5 °C above ~15 °C) |
| Race-day weather (rain, mud, snow) | Forecast | Terrain/technicality multiplier, mainly on descents |
| Night sections | Start time + sunrise/sunset | Slower pace in the dark |
| Aid stations | Race roadbook | Fixed stop time per aid station |
| Fitness / form | HR-based load (CTL/TSB) | Adjust the reference profile to current form |
| Race vs. training effort | Activity type/tag, avg HR | Weight or scale the reference runs |
| Poles, cadence | `gpxtpx:cad`, user input | Separate hiking vs. running regimes on steep climbs |
| Pacing strategy | User target HR/zone | Predict at the planned intensity |
