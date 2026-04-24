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
    2490,
    3165,
    3615,
    4945,
    5805,
    6355,
    6725,
    7110,
    8207,
    10953,
    12113,
    13782,  # +35: path-dir edge filter for arrival RET
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
