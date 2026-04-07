"""
Post-run KPI checks for default_layout-style sim I/O (positions + schedule).

Usage:
  python -m harness.kpi_audit \\
    --input data/Result_storage/default_layout_sim_input.json \\
    --result data/Result_storage/default_layout_sim_result.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected object: {path}")
    return obj


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--deadlock-max", type=int, default=5)
    p.add_argument("--warp-px", type=float, default=380.0)
    p.add_argument(
        "--runway-strip-m",
        type=float,
        default=12.0,
        help="Half-width (m) of runway centerline strip for 2-ship occupancy checks",
    )
    p.add_argument("--dep-v-threshold", type=float, default=10.0)
    p.add_argument("--others-min-mean", type=float, default=5.5)
    p.add_argument("--others-min-count", type=int, default=4)
    args = p.parse_args(argv)

    sys.path.insert(0, str(_ROOT))
    from utils import airside_sim as sim  # noqa: E402

    inp = _load(Path(args.input))
    res = _load(Path(args.result))

    cell_size = float(inp.get("grid", {}).get("cellSize", 20.0))
    rw_coords: List[List[Tuple[float, float]]] = []
    seen: set[str] = set()
    for f in inp.get("flights") or []:
        if not isinstance(f, dict):
            continue
        tok = f.get("token") if isinstance(f.get("token"), dict) else {}
        for _rw in (
            str(f.get("arrRunwayId") or tok.get("arrRunwayId") or "").strip(),
            str(f.get("depRunwayId") or tok.get("depRunwayId") or "").strip(),
        ):
            if not _rw or _rw in seen:
                continue
            seen.add(_rw)
            c = sim._runway_polyline_coords_px(inp, cell_size, _rw)
            if c and len(c) >= 2:
                rw_coords.append(c)
    ppm = max(float(sim._layout_pixels_per_meter(inp)), 1e-9)
    rw_strip_m = max(1.0, float(args.runway_strip_m))

    def on_rw(x: float, y: float) -> bool:
        for verts in rw_coords:
            d_px = sim._min_distance_point_to_polyline(float(x), float(y), verts)
            if (d_px / ppm) <= rw_strip_m + 1e-9:
                return True
        return False

    positions = res.get("positions")
    if not isinstance(positions, dict):
        print("FAIL positions missing or not dict")
        return 2

    eldt_by_fid: Dict[str, float] = {}
    exit_runway_by_fid: Dict[str, float] = {}
    for sr in res.get("schedule") or []:
        if not isinstance(sr, dict):
            continue
        fid_s = str(sr.get("flight_id") or "")
        if not fid_s:
            continue
        ev: Optional[Any] = sr.get("ELDT")
        if ev is not None:
            try:
                eldt_by_fid[fid_s] = float(ev)
            except (TypeError, ValueError):
                pass
        exv: Optional[Any] = sr.get("EXIT_RUNWAY")
        if exv is not None:
            try:
                exit_runway_by_fid[fid_s] = float(exv)
            except (TypeError, ValueError):
                pass

    by_t: Dict[int, List[Tuple[str, float, float, float, bool]]] = {}
    for fid, plist in positions.items():
        if not isinstance(plist, list):
            continue
        for pt in plist:
            if not isinstance(pt, dict):
                continue
            t = int(pt.get("t", 0))
            by_t.setdefault(t, []).append(
                (
                    str(fid),
                    float(pt.get("x", 0.0)),
                    float(pt.get("y", 0.0)),
                    float(pt.get("v", 0.0)),
                    bool(pt.get("deadlockGhost", False)),
                )
            )

    fails: List[str] = []

    dc = res.get("deadlock_resolve_event_count", 0)
    try:
        dc_int = int(dc)
    except (TypeError, ValueError):
        dc_int = 999
    if dc_int > int(args.deadlock_max):
        fails.append(f"deadlock_resolve_event_count={dc_int} > {args.deadlock_max}")

    warps = 0
    for _fid, plist in positions.items():
        if not isinstance(plist, list):
            continue
        for i in range(1, len(plist)):
            if not isinstance(plist[i], dict) or not isinstance(plist[i - 1], dict):
                continue
            dt = float(plist[i]["t"]) - float(plist[i - 1]["t"])
            if dt <= 0:
                continue
            d = math.hypot(
                float(plist[i]["x"]) - float(plist[i - 1]["x"]),
                float(plist[i]["y"]) - float(plist[i - 1]["y"]),
            )
            if d > float(args.warp_px):
                warps += 1
    if warps:
        fails.append(f"warp_steps>{args.warp_px}px count={warps}")

    rw_bad_ticks = 0
    for t in sorted(by_t.keys()):
        onr: List[Tuple[str, float]] = []
        for fid, x, y, v, gh in by_t[t]:
            if gh:
                continue
            if not on_rw(x, y):
                continue
            ex_gate = exit_runway_by_fid.get(fid)
            if ex_gate is not None and float(t) + 1e-9 > float(ex_gate):
                continue
            eldt_gate = eldt_by_fid.get(fid)
            if (
                eldt_gate is not None
                and float(t) + 1e-9 < float(eldt_gate)
                and abs(float(v)) < 0.5
            ):
                continue
            onr.append((fid, v))
        tick_bad = len(onr) >= 2
        if tick_bad:
            rw_bad_ticks += 1
    if rw_bad_ticks:
        fails.append(f"runway_multi_or_stopped_ticks={rw_bad_ticks}")

    # High-speed on runway (departure roll / fast phase) must not coincide with crawling taxis elsewhere.
    dep_corr = 0
    worst: Tuple[float, int, float] = (0.0, 0, 0.0)
    for t in sorted(by_t.keys()):
        fast_on_rw = False
        for fid, x, y, v, gh in by_t[t]:
            if gh:
                continue
            if on_rw(x, y) and v >= float(args.dep_v_threshold):
                fast_on_rw = True
                break
        if not fast_on_rw:
            continue
        ovs: List[float] = []
        for _fid, x, y, v, gh in by_t[t]:
            if gh or v < 1.0:
                continue
            if on_rw(x, y):
                continue
            ovs.append(v)
        if len(ovs) < int(args.others_min_count):
            continue
        m = sum(ovs) / len(ovs)
        if m < float(args.others_min_mean):
            dep_corr += 1
            if m < worst[0] or worst[0] == 0.0:
                worst = (m, t, float(len(ovs)))

    if dep_corr:
        fails.append(
            f"fast_runway_dep_corr_slow_others ticks={dep_corr} "
            f"(others_mean<{args.others_min_mean}, n>={args.others_min_count}; "
            f"worst_mean={worst[0]:.2f} at t={worst[1]} n={int(worst[2])})"
        )

    if fails:
        print("FAIL kpi_audit")
        for f in fails:
            print(" ", f)
        return 1
    print(
        "PASS kpi_audit "
        f"(deadlock={dc_int} runway_bad_ticks=0 dep_corr_ticks=0 "
        f"times={len(by_t)} flights={len(positions)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
