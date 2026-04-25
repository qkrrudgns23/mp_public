"""
Build a self-contained HTML debug view: ground track vs Layout pose (designer.js getFlightPoseAtTime logic).
Bicycle + outgoing time-vertex; after schedule EObT, on Dep_taxi + apron_link only, flip nose
180 deg when (nose . track) &gt; 0 so R3 pushback shows retro without discarding the bicycle.
Usage (from repo root):
  PYTHONPATH=. python harness/pose_debug_from_result.py \\
    --result data/Result_storage/Test_sim_result.json \\
    --input data/Result_storage/Test_sim_input.json \\
    --flight-id id_xocmnln85 \\
    --t0 4470 --t1 4565 \\
    --out harness/r3_pose_debug_01_14_30.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]


def _hypot(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def walk_timeline_polyline_from_point(
    tl: List[Dict[str, Any]],
    seg_index: int,
    fx: float,
    fy: float,
    dist_m: float,
    forward: bool,
) -> Optional[Dict[str, float]]:
    eps = 1e-6
    if not tl or len(tl) < 2 or not (dist_m > eps) or not all(
        map(math.isfinite, (fx, fy, dist_m))
    ):
        return None
    if seg_index < 0 or seg_index > len(tl) - 2:
        return None
    rem = dist_m
    x, y = float(fx), float(fy)
    s = seg_index
    while rem > eps:
        if forward:
            if s > len(tl) - 2:
                if len(tl) < 2:
                    return {"x": x, "y": y}
                n = len(tl)
                pa, pb = tl[n - 2], tl[n - 1]
                bx, by = float(pb["x"] - pa["x"]), float(pb["y"] - pa["y"])
                bl = _hypot(bx, by)
                if bl < eps:
                    return {"x": x, "y": y}
                inv = 1.0 / bl
                return {
                    "x": x + bx * inv * rem,
                    "y": y + by * inv * rem,
                }
            b = tl[s + 1]
            ddx, ddy = float(b["x"]) - x, float(b["y"]) - y
            dlen = _hypot(ddx, ddy)
            if dlen < eps:
                x, y = float(b["x"]), float(b["y"])
                s += 1
                continue
            step = min(rem, dlen)
            inv = 1.0 / dlen
            x += ddx * inv * step
            y += ddy * inv * step
            rem -= step
            if rem < eps:
                return {"x": x, "y": y}
            if dlen - step < eps:
                x, y = float(b["x"]), float(b["y"])
                s += 1
        else:
            if s < 0:
                if len(tl) < 2:
                    return {"x": x, "y": y}
                p0, p1 = tl[0], tl[1]
                bx, by = float(p0["x"] - p1["x"]), float(p0["y"] - p1["y"])
                bl = _hypot(bx, by)
                if bl < eps:
                    return {"x": x, "y": y}
                inv = 1.0 / bl
                return {
                    "x": x + bx * inv * rem,
                    "y": y + by * inv * rem,
                }
            tx, ty = float(tl[s]["x"]), float(tl[s]["y"])
            ddx, ddy = tx - x, ty - y
            dlen = _hypot(ddx, ddy)
            if dlen < eps:
                x, y = tx, ty
                s -= 1
                continue
            step = min(rem, dlen)
            inv = 1.0 / dlen
            x += ddx * inv * step
            y += ddy * inv * step
            rem -= step
            if rem < eps:
                return {"x": x, "y": y}
            if dlen - step < eps:
                x, y = tx, ty
                s -= 1
    return {"x": x, "y": y}


def get_flight_pose_at_time(
    timeline: List[Dict[str, Any]],
    t_sec: float,
    len_m: float,
) -> Optional[Dict[str, float]]:
    tl = timeline
    if not tl:
        return None
    motion_chord_eps = 0.08
    motion_chord_eps2 = motion_chord_eps * motion_chord_eps

    def seg_unit_dir(seg_idx: int) -> Optional[Dict[str, float]]:
        if seg_idx < 0 or seg_idx > len(tl) - 2:
            return None
        p, q = tl[seg_idx], tl[seg_idx + 1]
        ddx = float(q["x"] - p["x"])
        ddy = float(q["y"] - p["y"])
        l2 = ddx * ddx + ddy * ddy
        if l2 < motion_chord_eps2:
            return None
        inv = 1.0 / math.sqrt(l2)
        return {"dx": ddx * inv, "dy": ddy * inv}

    def last_unit_before(i: int) -> Optional[Dict[str, float]]:
        for j in range(i - 1, -1, -1):
            u = seg_unit_dir(j)
            if u:
                return u
        return None

    def first_unit_from(start_seg: int) -> Optional[Dict[str, float]]:
        for j in range(start_seg, len(tl) - 1):
            u = seg_unit_dir(j)
            if u:
                return u
        return None

    def heading_for_interval(i: int) -> Dict[str, float]:
        a, b = tl[i], tl[i + 1]
        dx = float(b["x"] - a["x"])
        dy = float(b["y"] - a["y"])
        l2 = dx * dx + dy * dy
        if l2 >= motion_chord_eps2:
            return {"dx": dx, "dy": dy}
        pr = last_unit_before(i)
        if pr:
            return {"dx": pr["dx"], "dy": pr["dy"]}
        nxt = first_unit_from(i + 1)
        if nxt:
            return {"dx": nxt["dx"], "dy": nxt["dy"]}
        return {"dx": 1.0, "dy": 0.0}

    def fr_bicycle(
        r: Optional[Dict[str, float]], x: float, y: float, bmin: float
    ) -> Optional[Dict[str, float]]:
        if not r or len_m <= 1e-6:
            return None
        vdx = x - r["x"]
        vdy = y - r["y"]
        vl = _hypot(vdx, vdy)
        if vl < bmin:
            return None
        return {"x": x, "y": y, "dx": vdx / vl, "dy": vdy / vl}

    if len(tl) == 1:
        a = tl[0]
        if t_sec + 1e-6 < a["t"] or t_sec - 1e-6 > a["t"]:
            return None
        return {"x": a["x"], "y": a["y"], "dx": 1.0, "dy": 0.0}

    if t_sec < tl[0]["t"] or t_sec > tl[-1]["t"]:
        return None

    wheel_base_m = 0.55 * len_m
    bicycle_min = max(0.15 * motion_chord_eps, 0.005 * len_m, 0.04)

    te = 1e-5
    for i in range(len(tl) - 1):
        a, b = tl[i], tl[i + 1]
        if t_sec >= a["t"] and t_sec <= b["t"]:
            use_i = i
            if i + 1 < len(tl) - 1:
                a2, b2 = tl[i + 1], tl[i + 2]
                if b2["t"] > a2["t"] and abs(t_sec - b["t"]) < te and abs(b["t"] - a2["t"]) < te:
                    use_i = i + 1
                    a, b = a2, b2
            span = (b["t"] - a["t"]) or 1.0
            uu = (t_sec - a["t"]) / span
            x = a["x"] + (b["x"] - a["x"]) * uu
            y = a["y"] + (b["y"] - a["y"]) * uu
            h = heading_for_interval(use_i)
            wline = walk_timeline_polyline_from_point(
                tl, use_i, x, y, wheel_base_m, False
            )
            out = fr_bicycle(wline, x, y, bicycle_min)
            if not out:
                out = {"x": x, "y": y, "dx": h["dx"], "dy": h["dy"]}
            return out
    return None


def _maybe_eobt_apron_dep_taxi_pushback_flip_pose(
    p: Optional[Dict[str, Any]],
    pose: Optional[Dict[str, float]],
    eobt_sec: Optional[int],
    by_t: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """
    Match designer.js applyEobtApronDepTaxiPushbackNoseIfNeeded: EOBT <= t, Dep_taxi, apron_link,
    one-second track exists; if unit nose . track > 0.05, flip dx/dy. x,y from bicycle unchanged.
    """
    if eobt_sec is None or pose is None or p is None:
        return pose
    t = int(p["t"])
    if t < int(eobt_sec):
        return pose
    if p.get("pathType") != "apron_link":
        return pose
    ph = p.get("phase")
    if ph and str(ph) != "Dep_taxi":
        return pose
    prev = by_t.get(t - 1)
    if not prev:
        return pose
    ddx = float(p["x"]) - float(prev["x"])
    ddy = float(p["y"]) - float(prev["y"])
    dlen = _hypot(ddx, ddy)
    if dlen < 1e-9:
        return pose
    ux, uy = ddx / dlen, ddy / dlen
    plen = _hypot(pose["dx"], pose["dy"])
    if plen < 1e-9:
        return pose
    px, py = pose["dx"] / plen, pose["dy"] / plen
    if px * ux + py * uy <= 0.05:
        return pose
    return {
        "x": pose["x"],
        "y": pose["y"],
        "dx": -pose["dx"],
        "dy": -pose["dy"],
    }


def _sec_to_hhmmss(t_sec: int) -> str:
    """Wall-clock time of day from seconds since midnight (same convention as positions[].t)."""
    t = int(t_sec) % 86400
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _len_m_for_flight(fobj: Optional[Dict[str, Any]], default: float = 50.0) -> float:
    if not fobj:
        return default
    t = fobj.get("aircraftType")
    # Without aircraft DB: use layout default length in Information if present; else default
    return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--flight-id", default="id_xocmnln85")
    ap.add_argument("--t0", type=int, default=4470)
    ap.add_argument("--t1", type=int, default=4485)
    ap.add_argument("--out", type=Path, default=Path("harness/r3_pose_debug.html"))
    args = ap.parse_args()

    res = json.loads(args.result.read_text(encoding="utf-8"))
    fid = args.flight_id
    pos = (res.get("positions") or {}).get(fid)
    if not pos:
        print("no positions for", fid)
        return 2

    eobt_sec: Optional[int] = None
    for row in res.get("schedule") or []:
        if str(row.get("flight_id")) == fid and row.get("EOBT") is not None:
            try:
                eobt_sec = int(float(row["EOBT"]))
            except (TypeError, ValueError):
                eobt_sec = None
            break

    fobj: Optional[Dict[str, Any]] = None
    if args.input and args.input.is_file():
        inp = json.loads(args.input.read_text(encoding="utf-8"))
        for f in inp.get("flights") or []:
            if str(f.get("id")) == fid:
                fobj = f
                break

    # Prefer A320 length from typical data if no DB
    len_m = 37.57
    if fobj and str(fobj.get("aircraftType", "")).strip().upper() in ("A320",):
        len_m = 37.57

    tl = []
    for p in sorted(pos, key=lambda r: int(r["t"])):
        tl.append(
            {
                "t": float(p["t"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
            }
        )

    rows: List[Dict[str, Any]] = []
    by_t = {int(p["t"]): p for p in pos}
    for t in range(args.t0, args.t1 + 1):
        p = by_t.get(t)
        pose = get_flight_pose_at_time(tl, float(t), len_m)
        if p is not None:
            pose = _maybe_eobt_apron_dep_taxi_pushback_flip_pose(
                p, pose, eobt_sec, by_t
            )
        track_ux, track_uy = None, None
        retro = None
        dot = None
        if p:
            t_prev = t - 1
            p_prev = by_t.get(t_prev)
            if p_prev:
                ddx = float(p["x"]) - float(p_prev["x"])
                ddy = float(p["y"]) - float(p_prev["y"])
                dlen = _hypot(ddx, ddy) or 1.0
                track_ux, track_uy = ddx / dlen, ddy / dlen
        if pose and track_ux is not None and (_hypot(pose["dx"], pose["dy"]) > 1e-9):
            pdx, pdy = pose["dx"], pose["dy"]
            plen = _hypot(pdx, pdy)
            pdx, pdy = pdx / plen, pdy / plen
            dot = pdx * track_ux + pdy * track_uy
            retro = bool(dot < -0.05)
        rows.append(
            {
                "t": t,
                "hhmmss": _sec_to_hhmmss(t),
                "x": float(p["x"]) if p else None,
                "y": float(p["y"]) if p else None,
                "v_json": float(p.get("v", 0) or 0) if p else None,
                "phase": p.get("phase") if p else None,
                "pathType": p.get("pathType") if p else None,
                "pose_dx": round(pose["dx"], 6) if pose else None,
                "pose_dy": round(pose["dy"], 6) if pose else None,
                "track_ux": round(track_ux, 6) if track_ux is not None else None,
                "track_uy": round(track_uy, 6) if track_uy is not None else None,
                "pose_dot_track": round(dot, 4) if dot is not None else None,
                "retro_nose_vs_track": retro,
            }
        )

    payload = {
        "flightId": fid,
        "reg": (fobj.get("reg") if fobj else None),
        "lenM": len_m,
        "t0": args.t0,
        "t1": args.t1,
        "t0_hhmmss": _sec_to_hhmmss(args.t0),
        "t1_hhmmss": _sec_to_hhmmss(args.t1),
        "baseDate": res.get("baseDate"),
        "eobtSec": eobt_sec,
        "eobtHhmmss": _sec_to_hhmmss(eobt_sec) if eobt_sec is not None else None,
        "rows": rows,
    }

    json_embed = json.dumps(payload, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>R3 pose vs track (Test_sim_result window)</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; background: #18181b; color: #e4e4e7; }}
    h1 {{ font-size: 1.1rem; }}
    p.note {{ color: #a1a1aa; font-size: 0.9rem; max-width: 72ch; }}
    table {{ border-collapse: collapse; font-size: 12px; margin-top: 12px; }}
    th, td {{ border: 1px solid #3f3f46; padding: 4px 8px; text-align: right; }}
    th {{ background: #27272a; }}
    tr.retro {{ background: #3f1a1a; }}
    td.str {{ text-align: left; }}
    #log {{ font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap; background: #09090b; padding: 10px; border: 1px solid #3f3f46; max-height: 200px; overflow: auto; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>R3 — Layout pose vs ground track (designer <code>getFlightPoseAtTime</code> port)</h1>
  <p class="note">
    <code>pose_*</code>: bicycle + outgoing time-vertex; after schedule <code>EOBT</code>, on <code>Dep_taxi</code>
    + <code>apron_link</code> only, flip nose 180&deg; when (nose &middot; track) &gt; 0 (towed look). <code>track_*</code>: t-1&rarr;t.
    <strong>retro</strong> = dot &lt; -0.05. Time: <code>positions[].t</code> as HH:MM:SS.
  </p>
  <div id="log"></div>
  <table>
    <thead>
      <tr>
        <th>time (HH:MM:SS)</th><th>x</th><th>y</th><th>v</th>
        <th>pose dx</th><th>pose dy</th><th>track ux</th><th>track uy</th>
        <th>dot</th><th>retro</th>
        <th class="str">phase</th><th class="str">path</th>
      </tr>
    </thead>
    <tbody id="tb"></tbody>
  </table>
  <script>
    const P = {json_embed};
    const log = document.getElementById("log");
    const tb = document.getElementById("tb");
    let retroCount = 0;
    for (const r of P.rows) {{
      if (r.retro_nose_vs_track) retroCount++;
      const tr = document.createElement("tr");
      if (r.retro_nose_vs_track) tr.className = "retro";
      tr.innerHTML = `
        <td class="str">${{r.hhmmss}}</td>
        <td>${{r.x != null ? r.x.toFixed(3) : "—"}}</td>
        <td>${{r.y != null ? r.y.toFixed(3) : "—"}}</td>
        <td>${{r.v_json != null ? r.v_json.toFixed(2) : "—"}}</td>
        <td>${{r.pose_dx ?? "—"}}</td>
        <td>${{r.pose_dy ?? "—"}}</td>
        <td>${{r.track_ux ?? "—"}}</td>
        <td>${{r.track_uy ?? "—"}}</td>
        <td>${{r.pose_dot_track ?? "—"}}</td>
        <td>${{r.retro_nose_vs_track == null ? "—" : r.retro_nose_vs_track}}</td>
        <td class="str">${{(r.phase || "—") + ""}}</td>
        <td class="str">${{(r.pathType || "—") + ""}}</td>
      `;
      tb.appendChild(tr);
    }}
    log.textContent =
      "baseDate=" + (P.baseDate || "—")
      + " flight=" + P.flightId + " reg=" + (P.reg || "—")
      + " lenM=" + P.lenM
      + "\\\\nEOBT t=" + (P.eobtSec != null ? P.eobtSec : "—") + " " + (P.eobtHhmmss || "")
      + "\\\\nwindow: " + (P.t0_hhmmss || "")
      + " – " + (P.t1_hhmmss || "")
      + "  (raw t_sec " + P.t0 + ".." + P.t1 + ")"
      + "\\\\nretro rows: " + retroCount + " / " + P.rows.length;
  </script>
</body>
</html>
"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print("Wrote", args.out, "- open in browser: file://%s" % (args.out.resolve().as_posix(),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
