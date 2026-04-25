"""Split designer.js into ordered chunks (run from repo root or this folder)."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "designer.js"
lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
n = len(lines)
# Cumulative cut indices (0-based, exclusive ends) from main Layout_Design.py script line map
cuts = [
    0,
    330,
    620,
    1450,
    1770,
    2502,  # +3 after sim payload timeline pathType
    3177,
    3627,
    4986,
    5846,
    6396,
    6766,
    7151,
    8270,  # simulation: getFlightPose + EObT/apron dep pushback nose helper; designer-mid = ForDraw
    11009,
    12169,
    13838,  # path-dir edge filter for arrival RET
    n,
]
if cuts[-1] != n:
    raise SystemExit(f"cut mismatch: last {cuts[-1]} vs n={n}")
names = [
    "config.js",
    "state.js",
    "layout-objects.js",
    "geometry.js",
    "path-engine.js",
    "panel-sync.js",
    "flight-bridge.js",
    "flight-timeline.js",
    "flight-ui.js",
    "gantt.js",
    "kpi.js",
    "runway-sep.js",
    "simulation.js",
    "designer-mid.js",
    "panel-dom.js",
    "canvas2d.js",
    "canvas3d.js",
]
if len(names) != len(cuts) - 1:
    raise SystemExit("names/cuts mismatch")
for i, name in enumerate(names):
    chunk = "".join(lines[cuts[i] : cuts[i + 1]])
    (HERE / name).write_text(chunk, encoding="utf-8")
print("wrote", len(names), "files")
