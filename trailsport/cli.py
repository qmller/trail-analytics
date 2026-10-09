"""Command line: `trailsport {predict,analyze,validate,hr}`."""

import argparse
import dataclasses
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .model import MODELS

log = logging.getLogger("trailsport")


def _start_time(value: str) -> pd.Timestamp:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date/time: {value!r}") from exc
    # naive values follow the system timezone rules (DST included) for that date
    return pd.Timestamp(dt if dt.tzinfo else dt.astimezone())


def _aid_stations(value: str) -> list[tuple[float, float]]:
    path = Path(value)
    try:
        if path.suffix.lower() == ".csv" and path.is_file():
            df = pd.read_csv(path)
            pairs = list(zip(df["km"], df["minutes"]))
        else:
            pairs = [item.split(":") for item in value.split(",") if item.strip()]
        out = [(float(km), float(minutes)) for km, minutes in pairs]
    except (ValueError, KeyError, OSError) as exc:
        raise argparse.ArgumentTypeError(f"aid stations: expected 'km:minutes,...' or a CSV with km,minutes ({exc})") from exc
    if any(not (math.isfinite(k) and math.isfinite(m) and k >= 0 and m >= 0) for k, m in out):
        raise argparse.ArgumentTypeError("aid stations: km and minutes must be positive numbers")
    return out


def _positive_float(value: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trailsport", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ref-dir", type=Path, default=Path("data/gpx"), help="reference activities (default: data/gpx)")
    common.add_argument("--exclude", nargs="+", default=[], metavar="PATTERN",
                        help="reference file names to ignore (glob, e.g. 'Grand_duc*')")
    common.add_argument("--plot", choices=["show", "save", "none"], default="save", help="plots: interactive, PNG in --out, or none")
    common.add_argument("--out", type=Path, help="output directory (default: output/<name>)")

    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument("--no-fatigue", dest="fatigue", action="store_false", help="disable the fatigue term")
    model_opts.add_argument("--target-hr", type=_positive_float, help="planned race heart rate (bpm), enables pace = f(grade, HR)")
    model_opts.add_argument("--form", action="store_true", help="adjust pace with training form (TSB from HR load)")
    model_opts.add_argument("--online", action="store_true", help="fetch weather (Open-Meteo) and trail tags (OSM), cached in .cache/")

    athlete = argparse.ArgumentParser(add_help=False)
    athlete.add_argument("--athlete", type=Path, help="athlete profile TOML (default: ./athlete.toml if present)")
    athlete.add_argument("--age", type=_positive_float, help="current age (years)")
    athlete.add_argument("--weight", type=_positive_float, help="current weight (kg)")
    athlete.add_argument("--vo2max", type=_positive_float, help="current VO2max (ml/kg/min)")
    athlete.add_argument("--ref-weight", type=_positive_float, help="weight when the reference activities were recorded")
    athlete.add_argument("--ref-vo2max", type=_positive_float, help="VO2max when the reference activities were recorded")
    athlete.add_argument("--hrmax", type=_positive_float, help="known max heart rate (overrides the GPX estimate)")
    athlete.add_argument("--lthr", type=_positive_float, help="known lactate threshold heart rate")
    athlete.add_argument("--hrrest", type=_positive_float, help="resting heart rate")

    p = sub.add_parser("predict", parents=[common, model_opts, athlete], help="predict the time on a course GPX")
    p.add_argument("course", type=Path)
    p.add_argument("--model", choices=sorted(MODELS), default="gbm")
    p.add_argument("--format", nargs="+", choices=["table", "csv", "json", "html", "map"], default=["table"])
    p.add_argument("--start-time", type=_start_time, help="race start, ISO format (e.g. 2026-10-12T07:00), local time if no offset")
    p.add_argument("--temp", type=float, help="race air temperature (degC)")
    p.add_argument("--wet", action="store_true", help="wet / muddy trails (slower steep descents)")
    p.add_argument("--poles", action="store_true", help="use poles on steep climbs")
    p.add_argument("--aid-stations", type=_aid_stations, default=[], help="'km:minutes,...' or CSV file with km,minutes")

    a = sub.add_parser("analyze", parents=[common], help="summary and complexity of GPX files")
    a.add_argument("gpx", nargs="*", type=Path, help="GPX files or directories (default: all in --ref-dir)")

    v = sub.add_parser("validate", parents=[common, model_opts, athlete], help="leave-one-out validation of the models")
    v.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))

    h = sub.add_parser("hr", parents=[common, athlete], help="heart-rate analysis of the reference activities or given GPX files")
    h.add_argument("gpx", nargs="*", type=Path,
                   help="GPX files or directories: activities to analyse, courses to estimate (default: --ref-dir)")
    h.add_argument("--date", type=_start_time, help="date for the form (TSB) estimate (default: today)")
    h.add_argument("--format", nargs="+", choices=["table", "csv", "json", "html", "map"], default=["table"],
                   help="console tables, files in --out, and/or one map per GPX with a heart-rate layer")

    m = sub.add_parser("map", help="interactive map (grade, and pace / time for recorded activities)")
    m.add_argument("gpx", nargs="+", type=Path, help="GPX files or directories")
    m.add_argument("--out", type=Path, help="output directory (default: output/<name>)")
    return parser


def _load_refs(args):
    from fnmatch import fnmatch

    from .preprocess import load_dir

    refs = [a for a in load_dir(args.ref_dir) if not any(fnmatch(a.path.name, p) for p in args.exclude)]
    if not refs:
        sys.exit(f"no GPX activity found in {args.ref_dir}")
    return refs


def _load_files(paths):
    """Activities from GPX files and directories (all *.gpx inside), without duplicates."""
    from gpxpy.gpx import GPXException

    from .preprocess import load_activity

    files = []
    for p in paths:
        if p.is_dir():
            found = sorted(p.glob("*.gpx"))
            if not found:
                sys.exit(f"no .gpx file in directory {p}")
            files += found
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"file not found: {p}")
    acts = []
    for f in dict.fromkeys(f.resolve() for f in files):
        try:
            acts.append(load_activity(f))
        except (ValueError, OSError, GPXException) as exc:
            sys.exit(f"cannot read {f}: {exc}")
    return acts


def _profile(args):
    from .athlete import load_profile

    keys = ("age", "weight", "vo2max", "ref_weight", "ref_vo2max", "hrmax", "lthr", "hrrest")
    try:
        return load_profile(args.athlete, **{k: getattr(args, k) for k in keys})
    except (ValueError, OSError) as exc:
        sys.exit(f"athlete profile: {exc}")


def _options(args, **extra):
    from .predict import Options

    return Options(
        fatigue=args.fatigue,
        target_hr=args.target_hr,
        form=args.form,
        online=args.online,
        profile=_profile(args),
        **extra,
    )


def cmd_predict(args):
    from . import report
    from .predict import build_context, predict_course

    if not args.course.is_file():
        sys.exit(f"course GPX file not found: {args.course}")
    course = _load_files([args.course])[0]
    opts = _options(
        args,
        model=args.model,
        start=args.start_time,
        temp=args.temp,
        wet=args.wet,
        poles=args.poles,
        aid_stations=args.aid_stations,
    )
    ctx = build_context(_load_refs(args), opts)
    pred = predict_course(course, ctx, opts)
    out_dir = args.out or Path("output") / args.course.stem

    if "table" in args.format:
        print(report.prediction_text(pred))
    figs = {}
    if args.plot != "none" or "html" in args.format:
        figs = {
            "prediction": report.plot_prediction(pred),
            "pace_grade": report.plot_pace_grade(ctx.refs, pred.model, ctx.hike_grade),
        }
    for path in report.write_prediction(pred, out_dir, args.format, figs):
        print(f"written {path}")
    report.output_figures(figs if args.plot != "none" else {}, args.plot, out_dir)


def cmd_analyze(args):
    from . import report
    from .course import features, sections
    from .metrics import summarize

    acts = _load_files(args.gpx) if args.gpx else _load_refs(args)
    summary = pd.DataFrame([summarize(a) for a in acts]).drop(columns=["name", "start"])
    print(report.table(summary))
    print()
    feats = pd.DataFrame([features(a) for a in acts], index=[a.path.stem[:20] for a in acts])
    print(feats.round(2).T.to_string())
    if args.gpx:
        for act in acts:
            print(f"\nSections - {act.name}")
            print(report.table(sections(act.segs)))
    if args.plot != "none":
        figs = {"profiles": report.plot_profiles(acts)}
        timed = [a for a in acts if a.has_time]
        if timed:
            figs["pace_grade"] = _plot_recorded_pace_grade(report, timed)
        report.output_figures(figs, args.plot, args.out or Path("output") / "analyze")


def _plot_recorded_pace_grade(report, acts):
    from .course import hike_threshold
    from .model import BinLookupModel, features
    from .predict import SPM_RANGE

    b = pd.concat([a.blocks for a in acts], ignore_index=True)
    spm = (b["dt"] / b["dx"]).to_numpy()
    ok = (spm >= SPM_RANGE[0]) & (spm <= SPM_RANGE[1])
    model = BinLookupModel().fit(features(b)[ok], spm[ok], b["dx"].to_numpy()[ok])
    return report.plot_pace_grade(acts, model, hike_threshold(acts))


def cmd_validate(args):
    from . import report
    from .metrics import format_duration
    from .predict import build_context, loo

    opts = _options(args)
    ctx = build_context(_load_refs(args), opts)
    rows, details = [], []
    for name in args.models:
        res = loo(ctx, dataclasses.replace(opts, model=name))
        print(f"\nModel '{name}' - leave-one-out")
        print(report.table(res))
        worst = res.loc[res["error_pct"].abs().idxmax()]
        rows.append({
            "model": name,
            "finish_mae_pct": res["error_pct"].abs().mean(),
            "finish_bias_pct": res["error_pct"].mean(),
            "worst_pct": worst["error_pct"],
            "worst": worst["name"][:30],
            "split_mae": format_duration(res["split_mae_s"].mean()),
        })
        details.append(res.assign(model=name))
    summary = pd.DataFrame(rows).round(1)
    print("\nSummary")
    print(summary.to_string(index=False))
    out_dir = args.out or Path("output") / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(details).to_csv(out_dir / "leave_one_out.csv", index=False)
    print(f"written {out_dir / 'summary.csv'}, {out_dir / 'leave_one_out.csv'}")


def cmd_hr(args):
    from . import report
    from .athlete import tanaka_hrmax
    from .hr import activity_stats, course_estimate, estimate_athlete, load_series, tsb_at, zone_by_grade

    refs = _load_refs(args)
    profile = _profile(args)
    ath = estimate_athlete(refs, profile)
    if not ath.known:
        sys.exit("no heart-rate data in the reference activities and no --lthr/--hrmax/--age given")
    header = ["Heart rate: " + " | ".join(f"{k} {getattr(ath, k):.0f} bpm ({src})" for k, src in ath.sources.items())]
    if profile.age:
        header.append(f"Age-predicted HRmax (Tanaka, 208 - 0.7 x age): {tanaka_hrmax(profile.age):.0f} bpm")

    targets = _load_files(args.gpx) if args.gpx else refs
    acts = [a for a in targets if a.has_hr and a.has_time]
    courses = [a for a in targets if not (a.has_hr and a.has_time)]
    est = pd.DataFrame([course_estimate(c, refs, ath) for c in courses]).rename(columns={"time": "moving"})
    ref_stats = pd.DataFrame([activity_stats(a, ath) for a in refs])
    stats = ref_stats if targets is refs else pd.DataFrame([activity_stats(a, ath) for a in acts])
    zg = zone_by_grade(acts, ath) if acts else pd.DataFrame()

    when = args.date or pd.Timestamp.now(tz="UTC")
    loads = load_series(ref_stats.to_dict(orient="records"), when)
    tsb = tsb_at(loads, when)
    form = {}
    if tsb is not None:
        last = loads.loc[: when.tz_convert(None).normalize() - pd.Timedelta(days=1)].iloc[-1]
        form = {"date": str(when.date()), "ctl": last["ctl"], "atl": last["atl"], "tsb": tsb}
        header.append(f"Form on {when.date()}: CTL {last['ctl']:.1f} | ATL {last['atl']:.1f} | TSB {tsb:.1f} "
                      "(only activities in the reference folder are counted)")
    text = "\n".join(header)

    if "table" in args.format:
        print(text)
        if len(est):
            print("\nCourses without heart rate - estimate if run like the reference activities")
            print(report.table(est))
        if len(stats):
            print()
            print(report.table(stats.assign(start=pd.to_datetime(stats["start"]).dt.strftime("%Y-%m-%d"))))
        if len(zg):
            print("\nHR zone share (%) by grade")
            print(zg.to_string())

    out_dir = args.out or Path("output") / (targets[0].path.stem if len(targets) == 1 else "hr")
    figs = {}
    if acts and (args.plot != "none" or "html" in args.format):
        figs = {"hr": report.plot_hr(stats, zg)}
    tables = {"activities": stats, "courses": est, "zones_by_grade": zg}
    extra = {"athlete": {k: getattr(ath, k) for k in ("hrmax", "lthr", "hrrest")} | {"sources": ath.sources}, "form": form}
    for path in report.write_tables(out_dir, "hr", args.format, text, tables, figs, extra):
        print(f"written {path}")
    if "map" in args.format:
        for act in targets:
            path = _hr_map(act, refs, ath, out_dir, single=len(targets) == 1)
            print(f"written {path}")
    report.output_figures(figs if args.plot != "none" else {}, args.plot, out_dir)


def _hr_map(act, refs, ath, out_dir: Path, single: bool) -> Path:
    """Map with recorded HR, or HR / pace estimated from the references for a course."""
    from .hr import course_estimate_blocks
    from .maps import course_map, recorded_blocks
    from .metrics import format_duration, summarize

    s = summarize(act)
    subtitle = f"{s['distance_km']:.1f} km, D+ {s['d_plus']:.0f} m"
    blocks = recorded_blocks(act)
    if act.has_hr and "cum" in blocks:
        subtitle += f" - recorded {format_duration(s['elapsed'])}, HR {s['hr_mean']:.0f} bpm"
    else:
        est = course_estimate_blocks(act, refs, ath)
        if est is not None:
            blocks = act.blocks.assign(pace=est["time"].to_numpy() / act.blocks["dx"].to_numpy(),
                                       hr=est["hr"].to_numpy(), cum=est["time"].cumsum().to_numpy())
            subtitle += f" - estimate if run like the references: {format_duration(est['time'].sum())}"
    out = out_dir / ("map.html" if single else f"{act.path.stem}_map.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(course_map(act, blocks, act.name, subtitle), encoding="utf-8")
    return out


def cmd_map(args):
    from .maps import course_map, recorded_blocks
    from .metrics import format_duration, summarize

    for act in _load_files(args.gpx):
        path = act.path
        blocks = recorded_blocks(act)
        s = summarize(act)
        subtitle = f"{s['distance_km']:.1f} km, D+ {s['d_plus']:.0f} m"
        if "cum" in blocks:
            subtitle += f" - recorded {format_duration(s['elapsed'])}"
        out = args.out / f"{path.stem}_map.html" if args.out else Path("output") / path.stem / "map.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(course_map(act, blocks, act.name, subtitle), encoding="utf-8")
        print(f"written {out}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].lower().endswith(".gpx"):
        argv.insert(0, "predict")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if getattr(args, "plot", "none") != "show":
        import matplotlib

        matplotlib.use("Agg")
    commands = {"predict": cmd_predict, "analyze": cmd_analyze, "validate": cmd_validate, "hr": cmd_hr, "map": cmd_map}
    commands[args.command](args)


if __name__ == "__main__":
    main()
