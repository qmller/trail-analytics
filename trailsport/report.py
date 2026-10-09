"""Console tables, file exports (CSV / JSON / HTML) and plots."""

import base64
import html
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import format_duration, format_pace, pace_by_grade, pace_min_km
from .model import features


def _fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in ("time", "cum_time", "aid", "moving", "elapsed", "stopped", "actual", "predicted") or col in ("Z1", "Z2", "Z3", "Z4", "Z5"):
            df[col] = df[col].map(format_duration)
        elif col in ("pace", "spm"):
            df[col] = df[col].map(format_pace)
        elif col == "clock":
            df[col] = pd.to_datetime(df[col]).dt.strftime("%H:%M")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(1)
    return df


def table(df: pd.DataFrame) -> str:
    return _fmt_table(df).to_string(index=False) if len(df) else "(empty)"


def local_clock(ts):
    if ts is None or pd.isna(ts):
        return "--"
    return ts.strftime("%Y-%m-%d %H:%M %Z")


def prediction_text(pred) -> str:
    s, bd = pred.summary, pred.breakdown
    lines = [
        f"Course : {s['course']}  ({s['distance_km']:.1f} km, D+ {s['d_plus']:.0f} m, D- {s['d_minus']:.0f} m, "
        f"effort {s['effort_km']:.1f} km-effort)",
        f"Model  : {s['model']}",
        "",
        "Splits per km",
        table(pred.splits),
        "",
        "Sections",
        table(pred.sections[["kind", "start_km", "end_km", "length_km", "d_plus", "d_minus", "avg_grade", "time", "pace"]])
        if len(pred.sections) else "(none)",
        "",
        "Breakdown (approximate, factors are independent)",
    ]
    for k, v in bd.items():
        if k in ("base (fresh)", "total"):
            lines.append(f"  {k:<16s} {format_duration(v):>10s}")
        elif abs(v) >= 1:
            lines.append(f"  {k:<16s} {'+' if v >= 0 else '-'}{format_duration(abs(v)):>9s}")
    lines += [
        "",
        "Reference weights",
        pred.references.round(2).to_string(index=False),
        "",
        "Summary",
        f"  moving time      {format_duration(s['moving'])}",
        f"  total time       {format_duration(s['total'])}",
        f"  average pace     {format_pace(s['pace'])} min/km",
    ]
    if "p10" in s:
        lines.append(f"  range P10-P90    {format_duration(s['p10'])} - {format_duration(s['p90'])} "
                     f"(leave-one-out error {s['loo_mae_pct']:.1f} %)")
    if s.get("finish") is not None:
        lines.append(f"  start / finish   {local_clock(s['start'])} / {local_clock(s['finish'])}")
    if s.get("hike_grade") is not None:
        lines.append(f"  hiking above     {s['hike_grade']:.0f} % grade (from cadence)")
    if math.isfinite(s.get("fatigue_per_effort_km_pct", math.nan)):
        lines.append(f"  fatigue          +{s['fatigue_per_effort_km_pct']:.2f} % pace per effort-km")
    if s.get("energy_kcal"):
        lines.append(f"  energy           ~{s['energy_kcal']:.0f} kcal ({s['energy_kcal'] / s['moving'] * 3600:.0f} kcal/h moving)")
    if s.get("athlete_hr"):
        lines.append(f"  heart rate       {s['athlete_hr']}")
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return value


def write_prediction(pred, out_dir: Path, formats, figures=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "csv" in formats:
        for name, df in (("splits", pred.splits), ("sections", pred.sections), ("blocks", pred.blocks)):
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            written.append(path)
    if "json" in formats:
        data = {
            "summary": pred.summary,
            "breakdown": pred.breakdown,
            "features": pred.features,
            "references": pred.references.to_dict(orient="records"),
            "splits": pred.splits.to_dict(orient="records"),
            "sections": pred.sections.to_dict(orient="records"),
        }
        path = out_dir / "prediction.json"
        path.write_text(json.dumps(_clean_json(json.loads(json.dumps(data, default=_json_default))), indent=2), encoding="utf-8")
        written.append(path)
    if "html" in formats:
        path = out_dir / "report.html"
        path.write_text(html_report(pred, figures or {}), encoding="utf-8")
        written.append(path)
    if "map" in formats:
        from .maps import course_map

        s = pred.summary
        subtitle = (f"{s['distance_km']:.1f} km, D+ {s['d_plus']:.0f} m - model {s['model']} - "
                    f"predicted {format_duration(s['total'])}")
        path = out_dir / "map.html"
        path.write_text(course_map(pred.course, pred.blocks.rename(columns={"t_pred": "cum"}), s["course"], subtitle),
                        encoding="utf-8")
        written.append(path)
    return written


def _fig_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def html_report(pred, figures: dict) -> str:
    title = html.escape(f"Prediction - {pred.course.name}")
    imgs = "".join(f'<h2>{html.escape(k)}</h2><img src="data:image/png;base64,{_fig_base64(f)}">' for k, f in figures.items())
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}} img{{max-width:100%}}</style></head>
<body><h1>{title}</h1>
<pre>{html.escape(prediction_text(pred))}</pre>
{imgs}
<h2>Splits</h2>{_fmt_table(pred.splits).to_html(index=False, escape=True)}
</body></html>
"""


def write_tables(out_dir: Path, name: str, formats, text: str, tables: dict, figures=None, extra=None) -> list[Path]:
    """CSV per table, `<name>.json`, `<name>.html`. Tables with a RangeIndex are formatted (durations, paces) in HTML."""
    formats = set(formats)
    if not formats & {"csv", "json", "html"}:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {k: df for k, df in tables.items() if df is not None and len(df)}
    written = []
    if "csv" in formats:
        for key, df in tables.items():
            path = out_dir / f"{key}.csv"
            df.to_csv(path, index=not isinstance(df.index, pd.RangeIndex))
            written.append(path)
    if "json" in formats:
        data = {**(extra or {}), **{k: df.reset_index(drop=isinstance(df.index, pd.RangeIndex)).to_dict(orient="records")
                                    for k, df in tables.items()}}
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(_clean_json(json.loads(json.dumps(data, default=_json_default))), indent=2), encoding="utf-8")
        written.append(path)
    if "html" in formats:
        title = html.escape(name)
        imgs = "".join(f'<h2>{html.escape(k)}</h2><img src="data:image/png;base64,{_fig_base64(f)}">'
                       for k, f in (figures or {}).items())
        sections = "".join(
            f"<h2>{html.escape(k)}</h2>"
            + (_fmt_table(df).to_html(index=False, escape=True) if isinstance(df.index, pd.RangeIndex) else df.to_html(escape=True))
            for k, df in tables.items()
        )
        path = out_dir / f"{name}.html"
        path.write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}} img{{max-width:100%}}</style></head>
<body><h1>{title}</h1>
<pre>{html.escape(text)}</pre>
{sections}
{imgs}
</body></html>
""", encoding="utf-8")
        written.append(path)
    return written


# --- plots -----------------------------------------------------------------

def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_prediction(pred):
    plt = _plt()
    b, s = pred.blocks, pred.splits
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    km = b["dist"] / 1000.0
    ax[0].fill_between(km, b["ele"], b["ele"].min() - 50, color="tan", alpha=0.7)
    colors = {"climb": "red", "descent": "blue", "rolling": "grey"}
    for _, r in pred.sections.iterrows():
        ax[0].axvspan(r["start_km"], r["end_km"], color=colors[r["kind"]], alpha=0.08)
    ax[0].set_ylabel("Altitude (m)")
    ax[0].set_title(f"{pred.course.name} - profile and sections")
    ax[0].grid(linewidth=0.5)
    edges = np.r_[0.0, s["km"].to_numpy()]
    ax[1].stairs(pace_min_km(s["pace"]), edges, fill=True, color="lightblue")
    ax[1].stairs(pace_min_km(s["pace"]), edges, color="blue", linewidth=2, label="predicted pace")
    ax2 = ax[1].twinx()
    ax2.plot(edges[1:], s["cum_time"] / 3600.0, color="black", label="cumulated time (h)")
    ax2.set_ylabel("Time (h)")
    ax[1].set_xlabel("Distance (km)")
    ax[1].set_ylabel("Pace (min/km)")
    ax[1].grid(linewidth=0.5)
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_pace_grade(refs, model, hike_grade=None, effort=0.0):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 7))
    for act in refs:
        pg = pace_by_grade(act.blocks)
        ax.plot(pg["grade"], pace_min_km(pg["spm"]), marker=".", linewidth=1, alpha=0.6, label=act.name[:25])
    grades = np.linspace(-40, 40, 161)
    ele = float(np.median(np.concatenate([a.blocks["ele"].to_numpy() for a in refs])))
    X = features(pd.DataFrame({"grade": grades, "ele": ele, "effort": effort}))
    if model.use_hr:
        X["hr_rel"] = 1.0
    ax.plot(grades, pace_min_km(model.predict(X)), color="black", linewidth=3,
            label=f"model '{model.name}' (fresh, {ele:.0f} m{', HR = LTHR' if model.use_hr else ''})")
    if hike_grade is not None:
        ax.axvline(hike_grade, color="green", linestyle="--", label=f"hiking above {hike_grade:.0f} %")
    ax.set_xlabel("Grade (%)")
    ax.set_ylabel("Pace (min/km)")
    ax.set_ylim(0, 30)
    ax.set_title("Pace vs grade (recorded, time-weighted)")
    ax.grid(linewidth=0.5)
    ax.legend(fontsize=8)
    return fig


def plot_profiles(acts):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(14, 6))
    for act in acts:
        ax.plot(act.segs["dist"] / 1000.0, act.segs["ele"], label=act.name[:30])
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(linewidth=0.5)
    ax.legend(fontsize=8)
    return fig


def plot_hr(stats: pd.DataFrame, zone_grade: pd.DataFrame):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    zones = [z for z in ("Z1", "Z2", "Z3", "Z4", "Z5") if z in stats]
    if zones:
        (stats.set_index("name")[zones] / 60.0).plot.barh(stacked=True, ax=ax[0], colormap="RdYlGn_r")
        ax[0].set_xlabel("Minutes")
        ax[0].set_title("Time in zone")
    if "EF" in stats:
        s = stats.dropna(subset=["EF"]).sort_values("start")
        ax[1].plot(pd.to_datetime(s["start"]), s["EF"], marker="o")
        ax[1].set_title("Efficiency factor (GAP m/min per bpm)")
        ax[1].tick_params(axis="x", rotation=30)
        ax2 = ax[1].twinx()
        ax2.bar(pd.to_datetime(s["start"]), s["decoupling"], width=3, alpha=0.3, color="red")
        ax2.set_ylabel("Decoupling (%)")
    if len(zone_grade):
        zg = zone_grade.copy()
        zg.index = [f"{i.left:.0f}..{i.right:.0f}" for i in zg.index]
        zg.plot.bar(stacked=True, ax=ax[2], colormap="RdYlGn_r")
        ax[2].set_xlabel("Grade (%)")
        ax[2].set_ylabel("% of time")
        ax[2].set_title("HR zone by grade")
    fig.tight_layout()
    return fig


def output_figures(figs: dict, mode: str, out_dir: Path | None = None):
    plt = _plt()
    if mode == "save" and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out_dir / f"{name}.png", dpi=100, bbox_inches="tight")
    elif mode == "show":
        plt.show()
    plt.close("all")
