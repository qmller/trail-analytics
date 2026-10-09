"""Interactive course map: Leaflet with OpenTopoMap / OpenStreetMap tiles, as a standalone HTML file."""

import html
import json

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

from .metrics import format_duration, format_pace
from .preprocess import Activity, haversine

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
LEAFLET_CSS_SRI = "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
LEAFLET_JS_SRI = "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
WAYPOINT_MAX_M = 200.0
GRADE_SCALE = 30.0  # %, colour saturates beyond +/- this grade


def _colors(values, cmap, vmin, vmax):
    norm, cm = Normalize(vmin, vmax, clip=True), colormaps[cmap]
    return [to_hex(cm(norm(v))) if np.isfinite(v) else "#888888" for v in values]


def _legend(label, cmap, vmin, vmax, fmt):
    stops = ", ".join(to_hex(colormaps[cmap](x)) for x in np.linspace(0, 1, 7))
    return {"label": label, "gradient": f"linear-gradient(to right, {stops})", "min": fmt(vmin), "max": fmt(vmax)}


def _clock(ts) -> str:
    return "" if ts is None or pd.isna(ts) else f" ({pd.Timestamp(ts).strftime('%H:%M')})"


def _at(blocks: pd.DataFrame, i: int) -> str:
    """'km x.x - elapsed (clock)' at the end of block i."""
    text = f"km {blocks['dist'].iloc[i] / 1000:.1f}"
    if "cum" in blocks and np.isfinite(blocks["cum"].iloc[i]):
        clock = blocks["clock"].iloc[i] if "clock" in blocks else None
        text += f" - {format_duration(blocks['cum'].iloc[i])}{_clock(clock)}"
    return text


def course_map(act: Activity, blocks: pd.DataFrame, title: str, subtitle: str = "") -> str:
    """HTML map of `blocks` (course blocks with optional `pace` s/m, `cum` elapsed s at block end, `clock`, `aid` s, `hr`)."""
    esc = html.escape
    dist = blocks["dist"].to_numpy()
    seg_block = np.minimum(np.searchsorted(dist, act.segs["dist"].to_numpy() - 1e-6), len(blocks) - 1)
    first = act.raw.dropna(subset=["lat", "lon"]).iloc[0]
    prev = [float(first["lat"]), float(first["lon"])]
    lines = []
    for i in range(len(blocks)):
        segs = act.segs[seg_block == i]
        pts = [prev] + [[round(a, 6), round(o, 6)] for a, o in zip(segs["lat"], segs["lon"])]
        lines.append(pts)
        prev = pts[-1]

    tips = []
    for i in range(len(blocks)):
        tip = f"{_at(blocks, i)}<br>grade {blocks['grade'].iloc[i]:+.0f} %"
        if "pace" in blocks and np.isfinite(blocks["pace"].iloc[i]):
            tip += f"<br>pace {format_pace(blocks['pace'].iloc[i])} min/km"
        if "hr" in blocks and np.isfinite(blocks["hr"].iloc[i]):
            tip += f"<br>HR {blocks['hr'].iloc[i]:.0f} bpm"
        tips.append(tip)

    layers = {"grade": {**_legend("Grade (%)", "RdBu_r", -GRADE_SCALE, GRADE_SCALE, lambda v: f"{v:+.0f}"),
                        "colors": _colors(blocks["grade"], "RdBu_r", -GRADE_SCALE, GRADE_SCALE)}}
    if "pace" in blocks and blocks["pace"].notna().any():
        pace = blocks["pace"].to_numpy() * 1000 / 60
        lo, hi = np.nanpercentile(pace, [5, 95])
        layers["pace"] = {**_legend("Pace (min/km)", "RdYlGn_r", lo, hi, lambda v: format_pace(v * 60 / 1000)),
                          "colors": _colors(pace, "RdYlGn_r", lo, hi)}
    if "cum" in blocks and blocks["cum"].notna().any():
        total = float(blocks["cum"].max())
        layers["time"] = {**_legend("Elapsed time", "viridis", 0, total, format_duration),
                          "colors": _colors(blocks["cum"], "viridis", 0, total)}
    if "hr" in blocks and blocks["hr"].notna().any():
        hr = blocks["hr"].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(hr, [5, 95])
        layers["hr"] = {**_legend("Heart rate (bpm)", "plasma", lo, hi, lambda v: f"{v:.0f}"),
                        "colors": _colors(hr, "plasma", lo, hi)}
    default = "pace" if "pace" in layers else "grade"

    markers = []
    for km in range(1, int(dist[-1] // 1000) + 1):
        i = int(np.searchsorted(dist, km * 1000.0 - 1e-6))
        markers.append({"group": "km", "lat": blocks["lat"].iloc[i], "lon": blocks["lon"].iloc[i],
                        "label": str(km), "text": _at(blocks, i)})
    if "aid" in blocks:
        for i in np.flatnonzero(blocks["aid"].to_numpy() > 0):
            markers.append({"group": "aid", "lat": blocks["lat"].iloc[i], "lon": blocks["lon"].iloc[i], "label": "A",
                            "text": f"Aid station<br>{_at(blocks, i)}<br>stop {blocks['aid'].iloc[i] / 60:.0f} min"})
    for w in act.raw.attrs.get("waypoints", []):
        d = haversine(w["lat"], w["lon"], blocks["lat"].to_numpy(), blocks["lon"].to_numpy())
        i = int(np.argmin(d))
        where = _at(blocks, i) if d[i] <= WAYPOINT_MAX_M else "off course"
        name = esc(w["name"]) + (f" <i>({esc(w['type'])})</i>" if w["type"] else "")
        markers.append({"group": "waypoints", "lat": w["lat"], "lon": w["lon"], "label": "W", "text": f"<b>{name}</b><br>{where}"})
    start = lines[0][0]
    markers.append({"group": "start", "lat": start[0], "lon": start[1], "label": "S", "text": "Start"})
    markers.append({"group": "start", "lat": prev[0], "lon": prev[1], "label": "F",
                    "text": "Finish<br>" + _at(blocks, len(blocks) - 1)})

    data = {"title": f"<b>{esc(title)}</b><br>{esc(subtitle)}", "lines": lines, "tips": tips,
            "layers": layers, "default": default, "markers": markers}
    payload = json.dumps(data).replace("</", "<\\/")
    return _TEMPLATE.format(
        title=esc(title), payload=payload,
        css=LEAFLET_CSS, css_sri=LEAFLET_CSS_SRI, js=LEAFLET_JS, js_sri=LEAFLET_JS_SRI,
    )


def recorded_blocks(act: Activity) -> pd.DataFrame:
    """Blocks of a GPX with its own timing for `course_map`; planner-generated timestamps are ignored."""
    from .predict import SPM_RANGE

    b = act.blocks.copy()
    if act.has_time and SPM_RANGE[0] <= float((b["dt"] / b["dx"]).median()) <= SPM_RANGE[1]:
        b["pace"] = b["dt"] / b["dx"]
        b["cum"] = b["t_elapsed"]
        if act.start is not None:
            b["clock"] = act.start + pd.to_timedelta(b["cum"], unit="s")
    return b


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="{css}" integrity="{css_sri}" crossorigin="">
<script src="{js}" integrity="{js_sri}" crossorigin=""></script>
<style>
html, body, #map {{ height: 100%; margin: 0; }}
.box {{ background: white; padding: 6px 10px; border-radius: 4px; box-shadow: 0 0 4px #888; font: 13px sans-serif; }}
.bar {{ width: 180px; height: 10px; }}
.lbl {{ background: white; border: 2px solid #333; border-radius: 50%; width: 18px; height: 18px;
        line-height: 18px; text-align: center; font: bold 11px sans-serif; }}
.lbl.aid {{ background: #2ca02c; color: white; }} .lbl.waypoints {{ background: #1f77b4; color: white; }}
.lbl.start {{ background: black; color: white; }}
</style></head>
<body><div id="map"></div>
<script>
const D = {payload};
const map = L.map("map");
const topo = L.tileLayer("https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png", {{maxZoom: 17,
  attribution: "Map data &copy; OpenStreetMap contributors, SRTM | Style &copy; OpenTopoMap (CC-BY-SA)"}}).addTo(map);
const osm = L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"}});
const overlays = {{}}, byLabel = {{}};
for (const [key, spec] of Object.entries(D.layers)) {{
  const g = L.layerGroup();
  D.lines.forEach((pts, i) => L.polyline(pts, {{color: spec.colors[i], weight: 6, opacity: 0.9}})
    .bindTooltip(D.tips[i], {{sticky: true}}).addTo(g));
  overlays[spec.label] = g; byLabel[spec.label] = spec;
  if (key === D.default) g.addTo(map);
}}
const groups = {{km: "Km markers", aid: "Aid stations", waypoints: "Waypoints", start: "Start / finish"}};
for (const [key, label] of Object.entries(groups)) {{
  const items = D.markers.filter(m => m.group === key);
  if (!items.length) continue;
  const g = L.layerGroup(items.map(m => L.marker([m.lat, m.lon], {{icon: L.divIcon({{
    className: "", html: `<div class="lbl ${{key}}">${{m.label}}</div>`, iconSize: [22, 22]}})}}).bindTooltip(m.text)));
  g.addTo(map); overlays[label] = g;
}}
L.control.layers({{"OpenTopoMap": topo, "OpenStreetMap": osm}}, overlays, {{collapsed: false}}).addTo(map);
const title = L.control({{position: "topleft"}});
title.onAdd = () => {{ const d = L.DomUtil.create("div", "box"); d.innerHTML = D.title; return d; }};
title.addTo(map);
const legend = L.control({{position: "bottomright"}});
legend.onAdd = () => L.DomUtil.create("div", "box");
legend.addTo(map);
function updateLegend() {{
  legend.getContainer().innerHTML = Object.entries(overlays).filter(([l, g]) => byLabel[l] && map.hasLayer(g))
    .map(([l]) => `${{l}}<div class="bar" style="background:${{byLabel[l].gradient}}"></div>` +
      `<span>${{byLabel[l].min}}</span><span style="float:right">${{byLabel[l].max}}</span>`).join("<br>");
}}
map.on("overlayadd overlayremove", updateLegend);
updateLegend();
map.fitBounds(L.latLngBounds(D.lines.flat()));
</script></body></html>
"""
