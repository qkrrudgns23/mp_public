"""
Airside simulation: Dijkstra paths on the Layout_Design path graph, then a time-step loop (no DES
events) moving each flight along edge polylines with per-segment ``avgMoveVelocity``, landing
deceleration, and runway-exit decel (see ``layoutPixelsPerMeter`` in Information.json for px/m scale).

Schedule inputs: S series (``*_Min_orig``) and Sd series (``*_Min_d`` minutes) are read from each
flight; routing and time axis use Sd only (``eldtMin_d`` or ``sldtMin_d`` → ELDT anchor in sim
seconds). Outputs ``schedule`` with S, Sd echo, and E times (ELDT/EIBT/EOBT/ETOT) derived from path
lengths and ``dwellMin``.

``positions`` timelines use ``t`` = sim seconds from day base (ELDT anchor + local sim time);
``v`` is m/s.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.designer_path_graph import (
    PathGraph,
    _stand_end_node_index,
    _vertex_to_px,
    build_path_graph,
    find_stand_by_id,
    get_ordered_points,
    get_runway_path_px,
    get_stand_connection_px,
    nearest_path_node_on_runway_polyline,
    normalize_allowed_runway_directions,
    normalize_rw_direction_value,
    path_dijkstra,
    path_graph_from_layout_sim_export,
    path_total_dist,
    project_on_segment,
    segment_segment_intersection,
)

_ROOT = Path(__file__).resolve().parents[1]
_INFORMATION_PATH = (_ROOT / "data" / "Info_storage" / "Information.json").resolve()
_DEFAULT_RW_DIR = "clockwise"
TAXI_SPEED_MPS = 15.0
MIN_LANDING_VELOCITY_MS = 15.0
ARR_RET_DECEL_MS2 = 0.5
Point = Tuple[float, float]
PHASE_LANDING = "Landing"
PHASE_ARR_TAXI = "Arr_taxi"
PHASE_DEP_TAXI = "Dep_taxi"
_EXTRACT_LEG_PHASES: Tuple[str, str, str] = (PHASE_LANDING, PHASE_ARR_TAXI, PHASE_DEP_TAXI)


def _flight_rw_dir_for_leg(flight: Dict[str, Any], leg_index: int) -> str:
    """
    Operations direction matching Layout_Design path graph export:
    ``simPathGraph.clockwise`` vs ``simPathGraph.counter_clockwise``.

    Legs 0–1 use arrival runway direction; leg 2 uses departure when present, else arrival.
    Reads ``arrRunwayDirUsed`` / ``arrRunwayDir``, ``depRunwayDirUsed`` / ``depRunwayDir``, and token.
    """
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    if leg_index >= 2:
        for k in ("depRunwayDirUsed", "depRunwayDir"):
            v = flight.get(k)
            if v is None or str(v).strip() == "":
                continue
            nd = normalize_rw_direction_value(str(v))
            if nd == "counter_clockwise":
                return "counter_clockwise"
            if nd == "clockwise":
                return "clockwise"
        v = token.get("depRunwayDir")
        if v is not None and str(v).strip():
            nd = normalize_rw_direction_value(str(v))
            if nd == "counter_clockwise":
                return "counter_clockwise"
            if nd == "clockwise":
                return "clockwise"
    for k in ("arrRunwayDirUsed", "arrRunwayDir"):
        v = flight.get(k)
        if v is None or str(v).strip() == "":
            continue
        nd = normalize_rw_direction_value(str(v))
        if nd == "counter_clockwise":
            return "counter_clockwise"
        if nd == "clockwise":
            return "clockwise"
    v = token.get("arrRunwayDir") or token.get("runwayDir")
    if v is not None and str(v).strip():
        nd = normalize_rw_direction_value(str(v))
        if nd == "counter_clockwise":
            return "counter_clockwise"
        if nd == "clockwise":
            return "clockwise"
    return _DEFAULT_RW_DIR


def _deep_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_information_json() -> Dict[str, Any]:
    try:
        if _INFORMATION_PATH.is_file():
            return json.loads(_INFORMATION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _minutes_to_sec(m: Any) -> Optional[float]:
    v = _safe_float(m, float("nan"))
    return v * 60.0 if math.isfinite(v) else None


def _sim_sec_optional(sec: Optional[float]) -> Optional[int]:
    """Snap schedule times to integer seconds (same convention as airside_sim_orig)."""
    if sec is None:
        return None
    try:
        v = float(sec)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(round(v))


def _sec_to_datetime_str(sec: Optional[float], base_date: str) -> Optional[str]:
    if sec is None:
        return None
    try:
        sec_v = float(sec)
        if not math.isfinite(sec_v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        parts = base_date.split("-")
        base = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        base = datetime(2026, 3, 31)
    result = base + timedelta(seconds=sec_v)
    return result.strftime("%m/%d %H:%M:%S")


def _schedule_sd_sec(flight: Dict[str, Any], key_d: str) -> Optional[int]:
    """Simulation schedule axis: Sd series only (minutes → sim seconds)."""
    return _sim_sec_optional(_minutes_to_sec(flight.get(key_d)))


def _schedule_s_sec(flight: Dict[str, Any], key_orig: str) -> Optional[int]:
    """S series for result display: ``*_orig`` minutes → seconds (airside_sim_orig input shape)."""
    return _sim_sec_optional(_minutes_to_sec(flight.get(key_orig)))


def _sd_eldt_sec(flight: Dict[str, Any]) -> Optional[int]:
    """ELDT anchor from Sd: ``eldtMin_d`` if set, else ``sldtMin_d`` (scheduled landing)."""
    eldt = _schedule_sd_sec(flight, "eldtMin_d")
    if eldt is not None:
        return eldt
    return _schedule_sd_sec(flight, "sldtMin_d")


def _path_search_params(information: Dict[str, Any]) -> Tuple[float, float, float]:
    algo = _deep_get(information, "tiers", "algorithm", default={}) or {}
    path_cfg = algo.get("pathSearch") if isinstance(algo.get("pathSearch"), dict) else {}
    reverse_cost = float(path_cfg.get("reverseCost", 1_000_000) or 1_000_000)
    merge_r = float(path_cfg.get("junctionMergeRadiusPx", 7.0) or 7.0)
    th = path_cfg.get("taxiwayHeuristicCost")
    if th is not None and math.isfinite(float(th)) and float(th) == 0.0:
        taxiway_h = 0.0
    elif th is not None and float(th) > 0:
        taxiway_h = float(th)
    else:
        taxiway_h = 200.0
    return max(reverse_cost, 1.0), max(merge_r, 1e-6), max(0.0, taxiway_h)


def _graph_for_direction(
    layout: Dict[str, Any],
    cell_size: float,
    rw_dir: str,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    *,
    pure_ground_exclude_runway: bool,
) -> Optional[PathGraph]:
    g = path_graph_from_layout_sim_export(
        layout,
        rw_dir,
        pure_ground_exclude_runway=pure_ground_exclude_runway,
        reverse_cost=reverse_cost,
        merge_radius_px=merge_r,
        taxiway_heuristic_bonus=taxiway_h,
        apply_taxiway_ret_heuristic=False,
    )
    if g is not None:
        return g
    flight_sched = _deep_get(information, "tiers", "flight_schedule", default={}) or {}
    rw_exit_default = normalize_allowed_runway_directions(flight_sched.get("rwExitAllowedDefaultRaw"))
    direction_modes = layout.get("directionModes") or []
    if not isinstance(direction_modes, list):
        direction_modes = []
    return build_path_graph(
        layout,
        cell_size,
        reverse_cost,
        taxiway_h,
        merge_r,
        rw_exit_default,
        direction_modes,
        None,
        rw_dir,
        None,
    )


def _pair_index_from_layout_edge(layout: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    raw = layout.get("Edge") or layout.get("edges")
    out: Dict[Tuple[int, int], str] = {}
    if not isinstance(raw, list):
        return out
    for ed in raw:
        if not isinstance(ed, dict):
            continue
        try:
            a = int(ed["fromIdx"])
            b = int(ed["toIdx"])
        except (KeyError, TypeError, ValueError):
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        eid = str(ed.get("id") or "").strip()
        if eid:
            out[(lo, hi)] = eid
    return out


def _pair_index_from_path_graph(g: PathGraph) -> Dict[Tuple[int, int], str]:
    """Match designer.js rebuildDerivedGraphEdges: undirected unique pairs, sort, label 001…"""
    seen: set = set()
    raw: List[Tuple[int, int]] = []
    rc = g.reverse_cost
    for rec in g.edge_map.values():
        if rec.cost >= rc * 0.999 or rec.cost < 1e-6:
            continue
        a, b = rec.from_idx, rec.to_idx
        lo, hi = (a, b) if a < b else (b, a)
        k = f"{lo}:{hi}"
        if k in seen:
            continue
        seen.add(k)
        raw.append((lo, hi))
    raw.sort(key=lambda t: (t[0], t[1]))
    out: Dict[Tuple[int, int], str] = {}
    for i, (lo, hi) in enumerate(raw[:999]):
        label = str(i + 1).zfill(3)
        out[(lo, hi)] = f"layout-edge-{label}"
    return out


def _path_to_edge_ids(path: List[int], pair_index: Dict[Tuple[int, int], str]) -> List[str]:
    out: List[str] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        lo, hi = (u, v) if u <= v else (v, u)
        eid = pair_index.get((lo, hi))
        if eid:
            out.append(eid)
    return out


@dataclass(frozen=True)
class RouteEndpoint:
    """Path-graph endpoint: stand id, runway polyline snap, or layout ``(x,y)`` → nearest graph node."""

    apron_stand_id: Optional[str] = None
    runway_id: Optional[str] = None
    runway_pixel_xy: Optional[Tuple[float, float]] = None
    token_pixel_xy: Optional[Tuple[float, float]] = None


@dataclass
class PreparedFlightPath:
    """Result of path search before the time-step loop (one flight)."""

    edge_ids: List[str] = field(default_factory=list)
    segment_phases: List[str] = field(default_factory=list)
    logical_edge_list: List[Dict[str, str]] = field(default_factory=list)
    segment_endpoints: List[Tuple[Point, Point]] = field(default_factory=list)
    leg_lengths_px: List[float] = field(default_factory=list)
    leg_micro_counts: List[int] = field(default_factory=list)
    segment_link_ids: List[str] = field(default_factory=list)
    segment_path_types: List[str] = field(default_factory=list)
    segment_start_velocity_ms: List[float] = field(default_factory=list)
    segment_accel_ms2: List[float] = field(default_factory=list)
    segment_duration_sec: List[float] = field(default_factory=list)
    spawn_skip_landing_px: float = 0.0
    spawn_along_first_segment_px: float = 0.0
    playback_first_segment_index: int = 0
    ok: bool = False
    direction_violation: bool = False


@dataclass
class Flight:
    """
    Playback agent: expanded path queue ``edge_ids`` + ``edge_phases`` + ``segment_endpoints``.

    Invariant: each finished segment is popped from the heads of those queues into
    ``edge_ids_finished``. When the route is fully traversed, ``edge_ids`` (and ``edge_phases``)
    must be empty and every segment lives in ``edge_ids_finished``.

    ``planned_edge_list`` is the coarser Dijkstra plan (unchanged during playback).
    """

    id: str
    edge_ids: List[str] = field(default_factory=list)
    edge_phases: List[str] = field(default_factory=list)
    edge_ids_finished: List[Dict[str, str]] = field(default_factory=list)
    segment_endpoints: List[Tuple[Point, Point]] = field(default_factory=list)
    planned_edge_list: List[Dict[str, str]] = field(default_factory=list)
    edge_s_along_px: float = 0.0
    col: float = 0.0
    row: float = 0.0
    velocity_ms: float = 0.0
    segment_v0_ms: List[float] = field(default_factory=list)
    segment_accel_ms2: List[float] = field(default_factory=list)
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    eldt_anchor_sec: Optional[float] = None


def resolve_route_endpoint_index(
    g: PathGraph,
    layout: Dict[str, Any],
    cell_size: float,
    endpoint: RouteEndpoint,
) -> Optional[int]:
    sid = endpoint.apron_stand_id
    if sid is not None and str(sid).strip() != "":
        return _stand_end_node_index(g, layout, str(sid), cell_size)
    rid = endpoint.runway_id
    px = endpoint.runway_pixel_xy
    if rid and str(rid).strip() and px is not None:
        return nearest_path_node_on_runway_polyline(g, str(rid), px)
    txy = endpoint.token_pixel_xy
    if txy is not None and len(txy) >= 2:
        return g.nearest_path_node((float(txy[0]), float(txy[1])))
    return None


def flight_route(
    g: PathGraph,
    layout: Dict[str, Any],
    cell_size: float,
    pair_index: Dict[Tuple[int, int], str],
    start_point: RouteEndpoint,
    end_point: RouteEndpoint,
) -> Tuple[List[str], float, Optional[List[int]]]:
    """
    Shortest path on ``g`` between two endpoints.

    Returns ``(edge_ids, path_length, node_path)``. ``node_path`` is ``None`` if unreachable.

    Per leg from ``extract_point_to_paths`` (e.g. leg 0 = threshold → RET on runway). Touchdown spawn is applied later in ``run_simulation`` (``_split_flight_path_at_touchdown``).
    Departure (apron → runway): ``start_point`` = apron, ``end_point`` = runway lineup pixel.
    """
    start_idx = resolve_route_endpoint_index(g, layout, cell_size, start_point)
    end_idx = resolve_route_endpoint_index(g, layout, cell_size, end_point)
    if start_idx is None or end_idx is None:
        return [], float("inf"), None
    path = path_dijkstra(g, start_idx, end_idx)
    if not path or len(path) < 2:
        return [], float("inf"), None
    edges = _path_to_edge_ids(path, pair_index)
    dist = path_total_dist(g, path)
    return edges, dist, path


def _path_uses_reverse_penalty_edges(g: PathGraph, path: List[int]) -> bool:
    rc = max(float(g.reverse_cost), 1.0)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rec = g.edge_map.get(f"{u}:{v}")
        if rec is None:
            return True
        if rec.cost >= rc * 0.999:
            return True
    return False


def _flight_route_impl(
    layout: Dict[str, Any],
    cell_size: float,
    pair_index: Dict[Tuple[int, int], str],
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    runway_ops_dir: str,
    start: RouteEndpoint,
    end: RouteEndpoint,
) -> Tuple[List[str], bool, Optional[List[int]], Optional[PathGraph]]:
    """Same graph build and routing as airside_sim_rev3 ``_flight_route``; returns path for geometry."""
    nd = normalize_rw_direction_value(str(runway_ops_dir).strip() if runway_ops_dir else "")
    if nd not in ("clockwise", "counter_clockwise"):
        nd = _DEFAULT_RW_DIR
    g = _graph_for_direction(
        layout,
        cell_size,
        nd,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        pure_ground_exclude_runway=False,
    )
    if g is None or not g.nodes:
        return [], False, None, None
    edges, _dist, path = flight_route(g, layout, cell_size, pair_index, start, end)
    if path is None or len(path) < 2:
        return [], False, None, g
    if _path_uses_reverse_penalty_edges(g, path):
        return [], True, None, g
    return edges, False, path, g


def _dep_lineup_token_xy(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    runway_ops_dir: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """Layout px: ``runwayPaths[].lineup_point`` from sim export (Pro Sim serialize)."""
    if not runway_id or not str(runway_id).strip():
        return None
    rid = str(runway_id)
    for rw in layout.get("runwayPaths") or []:
        if not isinstance(rw, dict) or str(rw.get("id", "")) != rid:
            continue
        lp = rw.get("lineup_point")
        if not isinstance(lp, dict):
            break
        lx, ly = lp.get("x"), lp.get("y")
        if lx is None or ly is None:
            break
        try:
            fx, fy = float(lx), float(ly)
        except (TypeError, ValueError):
            break
        if math.isfinite(fx) and math.isfinite(fy):
            return (fx, fy)
        break
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    rd = normalize_rw_direction_value(str(runway_ops_dir).strip() if runway_ops_dir else _DEFAULT_RW_DIR)
    if rd not in ("clockwise", "counter_clockwise"):
        rd = _DEFAULT_RW_DIR
    return r["endPx"] if rd == "clockwise" else r["startPx"]


def _apron_token_xy(layout: Dict[str, Any], cell_size: float, stand_id: str) -> Optional[Tuple[float, float]]:
    """Layout px: ``apronSiteX``/``apronSiteY`` on PBB in sim export; else ``get_stand_connection_px`` (remote x/y, …)."""
    st = find_stand_by_id(layout, str(stand_id))
    if not st or not isinstance(st, dict):
        return None
    ax, ay = st.get("apronSiteX"), st.get("apronSiteY")
    if ax is not None and ay is not None:
        try:
            fx, fy = float(ax), float(ay)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(fx) and math.isfinite(fy):
                return (fx, fy)
    return get_stand_connection_px(st, cell_size)


def _as_xy_pairs(pts: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if not isinstance(pts, list):
        return out
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                out.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                return []
        else:
            return []
    return out


def _closest_on_polyline_with_cum_dist(
    pts: List[Tuple[float, float]], q: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Closest point on polyline to ``q``, and cumulative distance from ``pts[0]`` to that point."""
    best_d2 = float("inf")
    best_xy = (float(q[0]), float(q[1]))
    best_cum = 0.0
    acc = 0.0
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        _t, proj = project_on_segment(p1, p2, q)
        d2 = (q[0] - proj[0]) ** 2 + (q[1] - proj[1]) ** 2
        seg0 = math.hypot(proj[0] - p1[0], proj[1] - p1[1])
        cum = acc + seg0
        if d2 < best_d2:
            best_d2 = d2
            best_cum = cum
            best_xy = (proj[0], proj[1])
        acc += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return (best_xy[0], best_xy[1], best_cum)


def _arr_ret_runway_junction_xy(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    ret_tw_id: Optional[str],
) -> Optional[Tuple[float, float]]:
    """
    Layout px: where the selected arrival runway-exit (RET) meets the arrival runway polyline —
    segment intersection, else snap of RET endpoint onto the runway within tolerance.
    """
    if not runway_id or not str(runway_id).strip() or not ret_tw_id or not str(ret_tw_id).strip():
        return None
    r = get_runway_path_px(layout, cell_size, str(runway_id))
    if not r:
        return None
    rw_pts = _as_xy_pairs(r.get("pts"))
    if len(rw_pts) < 2:
        return None
    ret_obj: Optional[Dict[str, Any]] = None
    for tw in layout.get("runwayTaxiways") or []:
        if isinstance(tw, dict) and str(tw.get("id", "")) == str(ret_tw_id):
            ret_obj = tw
            break
    if ret_obj is None:
        for tw in layout.get("taxiways") or []:
            if (
                isinstance(tw, dict)
                and str(tw.get("id", "")) == str(ret_tw_id)
                and tw.get("pathType") == "runway_exit"
            ):
                ret_obj = tw
                break
    if not ret_obj:
        return None
    ex_pts_raw = get_ordered_points(ret_obj, layout, cell_size)
    ex_pts = _as_xy_pairs(ex_pts_raw or [])
    if len(ex_pts) < 2:
        return None
    cand: List[Tuple[float, float, float]] = []
    for i in range(len(rw_pts) - 1):
        for j in range(len(ex_pts) - 1):
            ip = segment_segment_intersection(rw_pts[i], rw_pts[i + 1], ex_pts[j], ex_pts[j + 1])
            if ip is None:
                continue
            _x, _y, cum = _closest_on_polyline_with_cum_dist(rw_pts, ip)
            cand.append((ip[0], ip[1], cum))
    if cand:
        cand.sort(key=lambda t: t[2])
        return (cand[0][0], cand[0][1])
    snap_d2 = 70.0**2
    for vtx in (ex_pts[0], ex_pts[-1]):
        q = (float(vtx[0]), float(vtx[1]))
        sx, sy, _c = _closest_on_polyline_with_cum_dist(rw_pts, q)
        if (sx - q[0]) ** 2 + (sy - q[1]) ** 2 <= snap_d2:
            return (sx, sy)
    return None


def _polyline_total_length_px(pts: List[Tuple[float, float]]) -> float:
    if not pts or len(pts) < 2:
        return 0.0
    s = 0.0
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        s += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return s


def _polyline_point_at_dist_px(
    pts: List[Tuple[float, float]], dist_px: float
) -> Optional[Tuple[float, float]]:
    """Point along polyline at cumulative Euclidean distance from ``pts[0]`` (layout px)."""
    if not pts or len(pts) < 2:
        return None
    target = max(0.0, float(dist_px))
    acc = 0.0
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-9:
            continue
        if acc + seg_len >= target - 1e-9:
            t = max(0.0, min(1.0, (target - acc) / seg_len))
            return (p1[0] + dx * t, p1[1] + dy * t)
        acc += seg_len
    last = pts[-1]
    return (float(last[0]), float(last[1]))


def _flight_arr_td_dist_px(flight: Dict[str, Any]) -> Optional[float]:
    """``arrTdDistM`` on flight or token: distance along runway polyline in layout px (legacy name)."""
    v = flight.get("arrTdDistM")
    token = flight.get("token") if isinstance(flight.get("token"), dict) else None
    if v is None and token is not None:
        v = token.get("arrTdDistM")
    if v is None:
        return None
    try:
        d = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d < 0:
        return None
    return d


def _runway_polyline_coords_px(
    layout: Dict[str, Any], cell_size: float, runway_id: str
) -> Optional[List[Tuple[float, float]]]:
    """Oriented runway centerline in layout px (same order as ``get_runway_path_px``)."""
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    pts_raw = r.get("pts")
    if not isinstance(pts_raw, list) or len(pts_raw) < 2:
        return None
    coords: List[Tuple[float, float]] = []
    for p in pts_raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                coords.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                return None
        else:
            return None
    return coords


def _arr_runway_threshold_point_xy(
    layout: Dict[str, Any], cell_size: float, runway_id: str
) -> Optional[Tuple[float, float]]:
    """Runway threshold / start Point: first vertex of oriented arrival runway polyline."""
    coords = _runway_polyline_coords_px(layout, cell_size, runway_id)
    if not coords:
        return None
    return (float(coords[0][0]), float(coords[0][1]))


def _arr_touchdown_point_xy(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
) -> Optional[Tuple[float, float]]:
    """Touchdown in layout px: oriented runway polyline, ``arrTdDistM`` from threshold; else threshold vertex."""
    coords = _runway_polyline_coords_px(layout, cell_size, runway_id)
    if not coords:
        return None
    td = _flight_arr_td_dist_px(flight)
    if td is None:
        return (float(coords[0][0]), float(coords[0][1]))
    total = _polyline_total_length_px(coords)
    d_along = min(td, total) if total > 1e-9 else 0.0
    out = _polyline_point_at_dist_px(coords, d_along)
    return out if out is not None else (float(coords[0][0]), float(coords[0][1]))


def extract_point_to_paths(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
) -> List[List[float]]:
    """
    Token pixels as path legs for combined arrival + departure:
    runway threshold → Arr RET on runway → apron → lineup.

    Leg semantics (for phase tagging downstream): leg 0 = ``Landing``, leg 1 = ``Arr_taxi``,
    leg 2 = ``Dep_taxi``.

    Leg 0 uses the runway **start Point** (threshold) to the RET junction so graph routing keeps all
    runway edges. Playback starts at the touchdown pixel (``arrTdDistM``); see
    ``_split_flight_path_at_touchdown`` in ``run_simulation``.
    Each leg is ``[start_x, start_y, end_x, end_y]``. Returns ``[]`` if required ids or anchors are missing.
    """
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    arr_rwy = flight.get("arrRunwayId") or token.get("arrRunwayId")
    arr_ret_tw = flight.get("ExitTaxiwayId") or token.get("ExitTaxiwayId")
    dep_rwy = flight.get("depRunwayId") or token.get("depRunwayId")
    stand_id = flight.get("standId")
    if stand_id is None:
        stand_id = token.get("apronId")
    if stand_id is None or str(stand_id).strip() == "":
        return []

    thr_point = (
        _arr_runway_threshold_point_xy(layout, cell_size, str(arr_rwy)) if arr_rwy else None
    )
    td_point = (
        _arr_touchdown_point_xy(flight, layout, cell_size, str(arr_rwy)) if arr_rwy else None
    )
    ret_on_rw = (
        _arr_ret_runway_junction_xy(layout, cell_size, str(arr_rwy), arr_ret_tw) if arr_rwy else None
    )
    apron_point = _apron_token_xy(layout, cell_size, str(stand_id))
    dep_rw_lineup_point = (
        _dep_lineup_token_xy(
            layout,
            cell_size,
            str(dep_rwy),
            _flight_rw_dir_for_leg(flight, 2),
        )
        if dep_rwy
        else None
    )

    if not arr_rwy or not dep_rwy:
        return []
    if (
        thr_point is None
        or td_point is None
        or ret_on_rw is None
        or apron_point is None
        or dep_rw_lineup_point is None
    ):
        return []
    wx, wy = thr_point
    ret_x, ret_y = ret_on_rw
    px, py = apron_point
    lx, ly = dep_rw_lineup_point
    return [
        [float(wx), float(wy), float(ret_x), float(ret_y)],
        [float(ret_x), float(ret_y), float(px), float(py)],
        [float(px), float(py), float(lx), float(ly)],
    ]


def _avg_move_velocity_ms_for_link(layout: Dict[str, Any], link_id: str, flight_id: str) -> float:
    lid = str(link_id).strip()
    for bucket in (
        layout.get("taxiways"),
        layout.get("runwayTaxiways"),
        layout.get("runwayPaths"),
    ):
        if not isinstance(bucket, list):
            continue
        for obj in bucket:
            if not isinstance(obj, dict) or str(obj.get("id", "")).strip() != lid:
                continue
            v = _safe_float(obj.get("avgMoveVelocity"), float("nan"))
            if math.isfinite(v) and v > 0:
                return float(v)
            raise ValueError(
                f"avgMoveVelocity missing or invalid for link_id={lid!r} (flight_id={flight_id!r})"
            )
    raise ValueError(f"link_id={lid!r} not found in layout taxiways/runwayTaxiways/runwayPaths (flight_id={flight_id!r})")


def _velocity_ms_at_distance_on_segment(
    v0_ms: float, accel_ms2: float, s_m: float, apply_landing_velocity_floor: bool
) -> float:
    if abs(accel_ms2) < 1e-12:
        v = float(v0_ms)
    else:
        inner = float(v0_ms) * float(v0_ms) + 2.0 * float(accel_ms2) * float(s_m)
        v = math.sqrt(max(0.0, inner))
    if apply_landing_velocity_floor and float(accel_ms2) < -1e-12:
        v = max(v, MIN_LANDING_VELOCITY_MS)
    return float(v)


def _duration_slice_sec(
    v0_ms: float,
    accel_ms2: float,
    s0_m: float,
    s1_m: float,
    apply_landing_velocity_floor: bool,
) -> float:
    if s1_m <= s0_m + 1e-12:
        return 0.0
    if abs(accel_ms2) < 1e-12:
        v = _velocity_ms_at_distance_on_segment(v0_ms, accel_ms2, s0_m, apply_landing_velocity_floor)
        return (s1_m - s0_m) / max(v, 1e-9)
    n = max(8, min(128, int((s1_m - s0_m) / 3.0) + 1))
    ds = (s1_m - s0_m) / float(n)
    t = 0.0
    for i in range(n):
        sm = s0_m + (i + 0.5) * ds
        vm = _velocity_ms_at_distance_on_segment(v0_ms, accel_ms2, sm, apply_landing_velocity_floor)
        t += ds / max(vm, 1e-6)
    return float(t)


def _annotate_segment_kinematics(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    segment_phases: List[str],
    segment_endpoints: List[Tuple[Point, Point]],
    segment_link_ids: List[str],
    segment_path_types: List[str],
    pixels_per_meter: float,
    flight_id: str,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Parallel to micro-segments: start velocity (m/s) and constant acceleration (m/s^2) on each segment,
    plus full-segment travel time (s) for schedule export.
    """
    n = len(segment_endpoints)
    if n != len(segment_phases) or n != len(segment_link_ids) or n != len(segment_path_types):
        raise ValueError(f"segment meta length mismatch (flight_id={flight_id!r})")
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    exit_tw = flight.get("ExitTaxiwayId") or token.get("ExitTaxiwayId")
    if exit_tw is None or str(exit_tw).strip() == "":
        raise ValueError(f"ExitTaxiwayId missing (flight_id={flight_id!r})")
    exit_tw_s = str(exit_tw).strip()

    v0_out: List[float] = [0.0] * n
    a_out: List[float] = [0.0] * n
    dur_out: List[float] = [0.0] * n
    v_cur = 0.0
    landing_started = False
    arr_dec = flight.get("arrDecelMs2")
    if arr_dec is None:
        arr_dec = token.get("arrDecelMs2")
    arr_dec_f = _safe_float(arr_dec, float("nan"))
    arr_vtd = flight.get("arrVTdMs")
    if arr_vtd is None:
        arr_vtd = token.get("arrVTdMs")
    arr_vtd_f = _safe_float(arr_vtd, float("nan"))

    ppm = max(float(pixels_per_meter), 1e-9)

    for i in range(n):
        phase = segment_phases[i]
        link_id = str(segment_link_ids[i])
        pt = str(segment_path_types[i] or "")
        p0, p1 = segment_endpoints[i]
        seg_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        seg_m = seg_px / ppm
        if seg_m < 1e-12:
            v0_out[i] = float(v_cur)
            a_out[i] = 0.0
            dur_out[i] = 0.0
            continue

        if phase == PHASE_LANDING:
            if not landing_started:
                if not math.isfinite(arr_vtd_f) or arr_vtd_f <= 0:
                    raise ValueError(f"arrVTdMs missing or invalid (flight_id={flight_id!r})")
                if not math.isfinite(arr_dec_f) or arr_dec_f <= 0:
                    raise ValueError(f"arrDecelMs2 missing or invalid (flight_id={flight_id!r})")
                v0_out[i] = float(arr_vtd_f)
                a_out[i] = -abs(float(arr_dec_f))
                landing_started = True
                v_cur = float(arr_vtd_f)
            else:
                v0_out[i] = float(v_cur)
                a_out[i] = -abs(float(arr_dec_f))
            apply_floor = float(a_out[i]) < -1e-12
            dur_out[i] = _duration_slice_sec(v0_out[i], a_out[i], 0.0, seg_m, apply_floor)
            v_end = _velocity_ms_at_distance_on_segment(v0_out[i], a_out[i], seg_m, apply_floor)
            v_cur = float(v_end)
            continue

        if phase == PHASE_ARR_TAXI or phase == PHASE_DEP_TAXI:
            is_ret = pt == "runway_exit" and str(link_id) == exit_tw_s
            if pt == "runway_taxiway":
                if i == 0:
                    raise ValueError(
                        f"runway_taxiway at path start has no previous link (flight_id={flight_id!r})"
                    )
                prev_lid = str(segment_link_ids[i - 1])
                v_t = _avg_move_velocity_ms_for_link(layout, prev_lid, flight_id)
                v0_out[i] = float(v_t)
                a_out[i] = 0.0
                v_cur = float(v_t)
            elif is_ret:
                v0_out[i] = float(v_cur)
                a_out[i] = -float(ARR_RET_DECEL_MS2)
                dur_out[i] = _duration_slice_sec(v0_out[i], a_out[i], 0.0, seg_m, False)
                v_end = _velocity_ms_at_distance_on_segment(v0_out[i], a_out[i], seg_m, False)
                v_cur = float(v_end)
                continue
            else:
                v_t = _avg_move_velocity_ms_for_link(layout, link_id, flight_id)
                v0_out[i] = float(v_t)
                a_out[i] = 0.0
                v_cur = float(v_t)
            apply_floor = False
            dur_out[i] = _duration_slice_sec(v0_out[i], a_out[i], 0.0, seg_m, apply_floor)
            continue

        raise ValueError(f"unknown phase {phase!r} for kinematics (flight_id={flight_id!r})")

    return v0_out, a_out, dur_out


def _expand_geometry_from_graph_path(
    g: PathGraph,
    merged_nodes: List[int],
    pair_index: Dict[Tuple[int, int], str],
    leg_phase: str,
) -> Tuple[List[str], List[Tuple[Point, Point]], List[str], List[str], List[str]]:
    """
    One layout edge id per graph hop; duplicate ids when splitting ``DirectedEdgeRecord.pts`` polylines.
    ``leg_phase`` is repeated for every expanded sub-segment on this leg.
    """
    expanded_ids: List[str] = []
    segments: List[Tuple[Point, Point]] = []
    phases: List[str] = []
    link_ids: List[str] = []
    path_types: List[str] = []
    n_nodes = len(g.nodes)
    for i in range(len(merged_nodes) - 1):
        u, v = merged_nodes[i], merged_nodes[i + 1]
        lo, hi = (u, v) if u <= v else (v, u)
        eid = pair_index.get((lo, hi))
        if not eid:
            return [], [], [], [], []
        rec = g.edge_map.get(f"{u}:{v}")
        if rec is None:
            return [], [], [], [], []
        lid = str(rec.link_id)
        pt = str(rec.path_type or "")
        if len(rec.pts) >= 2:
            pts = rec.pts
            for j in range(len(pts) - 1):
                p0 = pts[j]
                p1 = pts[j + 1]
                expanded_ids.append(str(eid))
                segments.append(
                    ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
                )
                phases.append(leg_phase)
                link_ids.append(lid)
                path_types.append(pt)
        else:
            if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
                return [], [], [], [], []
            p0 = g.nodes[u]
            p1 = g.nodes[v]
            expanded_ids.append(str(eid))
            segments.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))))
            phases.append(leg_phase)
            link_ids.append(lid)
            path_types.append(pt)
    return expanded_ids, segments, phases, link_ids, path_types


def _finished_entry(
    eid: str,
    ph: str,
    j: int,
    v0_full: Optional[List[float]],
    acc_full: Optional[List[float]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"edge_id": str(eid), "phase": str(ph)}
    if v0_full is not None and acc_full is not None and j < len(v0_full) and j < len(acc_full):
        row["start_velocity_ms"] = float(v0_full[j])
        row["acceleration_ms2"] = float(acc_full[j])
    return row


def _split_flight_path_at_touchdown(
    edge_ids: List[str],
    edge_phases: List[str],
    segment_endpoints: List[Tuple[Point, Point]],
    touchdown_xy: Optional[Tuple[float, float]],
    segment_start_velocity_ms: Optional[List[float]] = None,
    segment_accel_ms2: Optional[List[float]] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[str],
    List[str],
    List[Tuple[Point, Point]],
    float,
    float,
    float,
    float,
    List[float],
    List[float],
    int,
]:
    """
    Landing segments strictly before ``touchdown_xy`` are returned as finished (for ``edge_list_finished``).
    Remaining queues retain the full edge id list from the touchdown offset onward. ``skipped_landing_px``
    is path length from threshold to the spawn point along landing segments (for taxi-in schedule scaling).
    """
    v_full = segment_start_velocity_ms
    a_full = segment_accel_ms2
    if v_full is not None and (len(v_full) != len(edge_ids) or a_full is None or len(a_full) != len(edge_ids)):
        v_full, a_full = None, None

    if (
        not touchdown_xy
        or not edge_ids
        or not segment_endpoints
        or len(edge_ids) != len(segment_endpoints)
        or len(edge_phases) != len(edge_ids)
    ):
        p0 = segment_endpoints[0][0] if segment_endpoints else (0.0, 0.0)
        v_rem = list(v_full) if v_full is not None else []
        a_rem = list(a_full) if a_full is not None else []
        return (
            [],
            list(edge_ids),
            list(edge_phases),
            list(segment_endpoints),
            0.0,
            float(p0[0]),
            float(p0[1]),
            0.0,
            v_rem,
            a_rem,
            0,
        )

    tx, ty = float(touchdown_xy[0]), float(touchdown_xy[1])
    landing_idxs = [i for i, ph in enumerate(edge_phases) if ph == PHASE_LANDING]
    if not landing_idxs:
        p0 = segment_endpoints[0][0]
        v_rem = list(v_full) if v_full is not None else []
        a_rem = list(a_full) if a_full is not None else []
        return (
            [],
            list(edge_ids),
            list(edge_phases),
            list(segment_endpoints),
            0.0,
            float(p0[0]),
            float(p0[1]),
            0.0,
            v_rem,
            a_rem,
            0,
        )

    best_i: Optional[int] = None
    best_d2 = float("inf")
    for i in landing_idxs:
        p0, p1 = segment_endpoints[i]
        _t, proj = project_on_segment(p0, p1, (tx, ty))
        d2 = (tx - proj[0]) ** 2 + (ty - proj[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    if best_i is None:
        p0 = segment_endpoints[0][0]
        v_rem = list(v_full) if v_full is not None else []
        a_rem = list(a_full) if a_full is not None else []
        return (
            [],
            list(edge_ids),
            list(edge_phases),
            list(segment_endpoints),
            0.0,
            float(p0[0]),
            float(p0[1]),
            0.0,
            v_rem,
            a_rem,
            0,
        )

    seg_i = int(best_i)
    finished: List[Dict[str, Any]] = []
    skipped = 0.0

    for j in range(seg_i):
        p0, p1 = segment_endpoints[j]
        skipped += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        finished.append(_finished_entry(str(edge_ids[j]), str(edge_phases[j]), j, v_full, a_full))

    while seg_i < len(edge_ids) and edge_phases[seg_i] == PHASE_LANDING:
        p0, p1 = segment_endpoints[seg_i]
        slen = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if slen < 1e-9:
            finished.append(_finished_entry(str(edge_ids[seg_i]), str(edge_phases[seg_i]), seg_i, v_full, a_full))
            seg_i += 1
            continue
        t, proj = project_on_segment(p0, p1, (tx, ty))
        if t < 1.0 - 1e-9:
            along = t * slen
            skipped += along
            v_rem = list(v_full[seg_i:]) if v_full is not None else []
            a_rem = list(a_full[seg_i:]) if a_full is not None else []
            return (
                finished,
                list(edge_ids[seg_i:]),
                list(edge_phases[seg_i:]),
                list(segment_endpoints[seg_i:]),
                along,
                float(proj[0]),
                float(proj[1]),
                skipped,
                v_rem,
                a_rem,
                int(seg_i),
            )
        finished.append(_finished_entry(str(edge_ids[seg_i]), str(edge_phases[seg_i]), seg_i, v_full, a_full))
        skipped += slen
        seg_i += 1

    if seg_i >= len(edge_ids):
        last = segment_endpoints[-1][1]
        return (finished, [], [], [], 0.0, float(last[0]), float(last[1]), skipped, [], [], int(seg_i))

    p0 = segment_endpoints[seg_i][0]
    v_rem = list(v_full[seg_i:]) if v_full is not None else []
    a_rem = list(a_full[seg_i:]) if a_full is not None else []
    return (
        finished,
        list(edge_ids[seg_i:]),
        list(edge_phases[seg_i:]),
        list(segment_endpoints[seg_i:]),
        0.0,
        float(p0[0]),
        float(p0[1]),
        skipped,
        v_rem,
        a_rem,
        int(seg_i),
    )


def prepare_flight_path(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
) -> PreparedFlightPath:
    """
    ``extract_point_to_paths`` 레그마다 airside_sim_rev3와 동일하게: layout ``Edge`` 기반
    ``pair_index``(없으면 그래프에서 ``layout-edge-*``), 레그마다 그래프 재구성 후
    ``token_pixel_xy`` 끝점만으로 ``flight_route``, 역주행 패널티 구간이면 전체 ``edge_list`` 비움.
    재생용 세그먼트는 각 레그의 노드 경로를 ``_expand_geometry_from_graph_path``로 확장한다.
    """
    paths = extract_point_to_paths(flight, layout, cell_size)
    if not paths:
        return PreparedFlightPath()
    pair_index = _pair_index_from_layout_edge(layout)
    if not pair_index:
        g0 = _graph_for_direction(
            layout,
            cell_size,
            _flight_rw_dir_for_leg(flight, 0),
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            pure_ground_exclude_runway=False,
        )
        pair_index = _pair_index_from_path_graph(g0) if g0 else {}

    logical_edge_list: List[Dict[str, str]] = []
    leg_route_rows: List[Tuple[List[str], Optional[List[int]], Optional[PathGraph]]] = []
    direction_violation = False

    for leg_i, leg in enumerate(paths):
        if len(leg) < 4:
            return PreparedFlightPath()
        phase = (
            _EXTRACT_LEG_PHASES[leg_i] if leg_i < len(_EXTRACT_LEG_PHASES) else PHASE_DEP_TAXI
        )
        sx, sy, ex, ey = float(leg[0]), float(leg[1]), float(leg[2]), float(leg[3])
        rw_leg = _flight_rw_dir_for_leg(flight, leg_i)
        edges, dv, path, g = _flight_route_impl(
            layout,
            cell_size,
            pair_index,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            rw_leg,
            RouteEndpoint(token_pixel_xy=(sx, sy)),
            RouteEndpoint(token_pixel_xy=(ex, ey)),
        )
        if dv:
            logical_edge_list = []
            direction_violation = True
            break
        for eid in edges:
            logical_edge_list.append({"edge_id": str(eid), "phase": phase})
        leg_route_rows.append((edges, path, g))

    expanded_ids: List[str] = []
    segment_phases: List[str] = []
    segments: List[Tuple[Point, Point]] = []
    segment_link_ids: List[str] = []
    segment_path_types: List[str] = []
    leg_lengths_px: List[float] = []
    leg_micro_counts: List[int] = []
    if not direction_violation:
        for leg_i, (_edges, path, g) in enumerate(leg_route_rows):
            phase = (
                _EXTRACT_LEG_PHASES[leg_i] if leg_i < len(_EXTRACT_LEG_PHASES) else PHASE_DEP_TAXI
            )
            if path is None or g is None:
                leg_lengths_px.append(0.0)
                leg_micro_counts.append(0)
                continue
            ex_ids, segs, phs, lnks, ptyps = _expand_geometry_from_graph_path(
                g, path, pair_index, phase
            )
            if (
                not ex_ids
                or not segs
                or not phs
                or len(ex_ids) != len(segs)
                or len(ex_ids) != len(phs)
                or len(lnks) != len(ex_ids)
                or len(ptyps) != len(ex_ids)
            ):
                return PreparedFlightPath(
                    logical_edge_list=logical_edge_list,
                    direction_violation=False,
                    ok=False,
                )
            expanded_ids.extend(ex_ids)
            segments.extend(segs)
            segment_phases.extend(phs)
            segment_link_ids.extend(lnks)
            segment_path_types.extend(ptyps)
            leg_lengths_px.append(_path_length_px(segs))
            leg_micro_counts.append(len(ex_ids))

    playback_ok = (
        bool(expanded_ids)
        and bool(segments)
        and len(expanded_ids) == len(segments) == len(segment_phases)
        and len(segment_link_ids) == len(expanded_ids)
        and len(segment_path_types) == len(expanded_ids)
    )
    v0s: List[float] = []
    accs: List[float] = []
    durs: List[float] = []
    if playback_ok and not direction_violation:
        ppm = _layout_pixels_per_meter(information)
        fid = str(flight.get("id", ""))
        try:
            v0s, accs, durs = _annotate_segment_kinematics(
                flight,
                layout,
                segment_phases,
                segments,
                segment_link_ids,
                segment_path_types,
                ppm,
                fid,
            )
        except ValueError:
            return PreparedFlightPath(
                logical_edge_list=list(logical_edge_list),
                edge_ids=list(expanded_ids),
                segment_phases=list(segment_phases),
                segment_endpoints=list(segments),
                segment_link_ids=list(segment_link_ids),
                segment_path_types=list(segment_path_types),
                leg_lengths_px=list(leg_lengths_px),
                leg_micro_counts=list(leg_micro_counts),
                direction_violation=False,
                ok=False,
            )
        if len(v0s) != len(expanded_ids) or len(accs) != len(expanded_ids) or len(durs) != len(
            expanded_ids
        ):
            return PreparedFlightPath(
                logical_edge_list=list(logical_edge_list),
                direction_violation=False,
                ok=False,
            )

    return PreparedFlightPath(
        edge_ids=list(expanded_ids),
        segment_phases=list(segment_phases),
        logical_edge_list=list(logical_edge_list),
        segment_endpoints=segments,
        leg_lengths_px=list(leg_lengths_px),
        leg_micro_counts=list(leg_micro_counts),
        segment_link_ids=list(segment_link_ids),
        segment_path_types=list(segment_path_types),
        segment_start_velocity_ms=list(v0s),
        segment_accel_ms2=list(accs),
        segment_duration_sec=list(durs),
        ok=playback_ok and not direction_violation and bool(v0s),
        direction_violation=direction_violation,
    )


def _finish_edge_segment(agent: Flight) -> None:
    eid = agent.edge_ids.pop(0)
    ph = agent.edge_phases.pop(0)
    v0 = float(agent.segment_v0_ms.pop(0))
    acc = float(agent.segment_accel_ms2.pop(0))
    agent.edge_ids_finished.append(
        {
            "edge_id": str(eid),
            "phase": str(ph),
            "start_velocity_ms": v0,
            "acceleration_ms2": acc,
        }
    )
    agent.segment_endpoints.pop(0)


def move_agent(agent: Flight, dt: float, pixels_per_meter: float) -> None:
    """Advance along ``segment_endpoints`` using per-segment :math:`v_0` + constant ``acceleration_ms2``."""
    col0, row0 = agent.col, agent.row
    rem_t = float(dt)
    ppm = max(float(pixels_per_meter), 1e-9)
    if rem_t <= 1e-12 or not agent.edge_ids or not agent.segment_endpoints:
        agent.velocity_ms = 0.0
        return
    if (
        len(agent.edge_ids) != len(agent.segment_endpoints)
        or len(agent.edge_phases) != len(agent.edge_ids)
        or len(agent.segment_v0_ms) != len(agent.edge_ids)
        or len(agent.segment_accel_ms2) != len(agent.edge_ids)
    ):
        agent.velocity_ms = 0.0
        return

    while rem_t > 1e-12 and agent.edge_ids and agent.segment_endpoints:
        p0, p1 = agent.segment_endpoints[0]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_px = math.hypot(dx, dy)
        if seg_len_px < 1e-9:
            _finish_edge_segment(agent)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue

        v0s = float(agent.segment_v0_ms[0])
        ac = float(agent.segment_accel_ms2[0])
        ph = agent.edge_phases[0]
        apply_floor = ph == PHASE_LANDING and ac < -1e-12
        seg_len_m = seg_len_px / ppm

        s_m = agent.edge_s_along_px / ppm
        room_m = seg_len_m - s_m
        if room_m <= 1e-9:
            _finish_edge_segment(agent)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue

        v_now = _velocity_ms_at_distance_on_segment(v0s, ac, s_m, apply_floor)
        dt_step = min(0.05, rem_t)
        ds = v_now * dt_step
        dt_used = dt_step
        if ds >= room_m:
            ds = room_m
            dt_used = min(rem_t, ds / max(v_now, 1e-6))
        s_new_m = s_m + ds
        agent.edge_s_along_px = min(s_new_m * ppm, seg_len_px)
        t_along = agent.edge_s_along_px / seg_len_px if seg_len_px > 1e-9 else 1.0
        agent.col = p0[0] + t_along * dx
        agent.row = p0[1] + t_along * dy
        rem_t -= dt_used
        if s_new_m >= seg_len_m - 1e-9:
            _finish_edge_segment(agent)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]

    dist_px = math.hypot(agent.col - col0, agent.row - row0)
    agent.velocity_ms = (dist_px / max(float(dt), 1e-9)) / ppm


def _sim_time_step_sec(information: Dict[str, Any], dt: float) -> float:
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if isinstance(sim, dict) and sim.get("timeStepSec") is not None:
        try:
            return max(1.0, float(sim["timeStepSec"]))
        except (TypeError, ValueError):
            pass
    return max(1.0, float(dt))


def _layout_pixels_per_meter(information: Dict[str, Any]) -> float:
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if isinstance(sim, dict) and sim.get("layoutPixelsPerMeter") is not None:
        try:
            v = float(sim["layoutPixelsPerMeter"])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return 1.0


def _path_length_px(segments: List[Tuple[Point, Point]]) -> float:
    s = 0.0
    for p0, p1 in segments:
        s += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return s


def _flight_opt_str(fobj: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        v = fobj.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _taxi_in_out_sec_from_prep(
    prep: PreparedFlightPath,
    pixels_per_meter: float,
) -> Tuple[Optional[float], Optional[float]]:
    if not prep.ok:
        return None, None
    durs = prep.segment_duration_sec
    segs = prep.segment_endpoints
    phs = prep.segment_phases
    v0s = prep.segment_start_velocity_ms
    accs = prep.segment_accel_ms2
    if (
        not durs
        or len(durs) != len(segs)
        or len(prep.leg_micro_counts) < 3
        or len(v0s) != len(segs)
        or len(accs) != len(segs)
        or len(phs) != len(segs)
    ):
        return None, None
    c0, c1, c2 = (
        int(prep.leg_micro_counts[0]),
        int(prep.leg_micro_counts[1]),
        int(prep.leg_micro_counts[2]),
    )
    c01 = c0 + c1
    c012 = c01 + c2
    if c012 != len(segs):
        return None, None
    gi = max(0, int(prep.playback_first_segment_index))
    along0_px = float(prep.spawn_along_first_segment_px or 0.0)
    ppm = max(float(pixels_per_meter), 1e-9)

    def dur_full(g: int) -> float:
        return float(durs[g])

    def dur_from_playback_start(g: int) -> float:
        if g != gi or along0_px <= 1e-9:
            return dur_full(g)
        p0, p1 = segs[g]
        seg_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        seg_m = seg_px / ppm
        s0_m = along0_px / ppm
        if s0_m >= seg_m - 1e-9:
            return 0.0
        return _duration_slice_sec(
            float(v0s[g]),
            float(accs[g]),
            s0_m,
            seg_m,
            phs[g] == PHASE_LANDING and float(accs[g]) < -1e-12,
        )

    taxi_in = 0.0
    for g in range(gi, c01):
        if g < 0 or g >= len(segs):
            continue
        taxi_in += dur_from_playback_start(g)

    taxi_out = 0.0
    for g in range(c01, c012):
        if g < 0 or g >= len(segs):
            continue
        taxi_out += dur_full(g)
    return taxi_in, taxi_out


def _build_schedule_row(
    fobj: Dict[str, Any],
    fid: str,
    prep: PreparedFlightPath,
    pixels_per_meter: float,
    base_date: str,
) -> Dict[str, Any]:
    """
    S series from ``*_Min_orig``; Sd echo from ``*_Min_d``; E series from path timing + dwell.
    ELDT anchor and taxi splits use Sd-only fields (see ``_sd_eldt_sec``).
    """
    dwell_sec = _safe_float(fobj.get("dwellMin"), float("nan")) * 60.0
    if not math.isfinite(dwell_sec) or dwell_sec < 0:
        dwell_sec = 0.0

    eldt_sec = _sd_eldt_sec(fobj)

    sldt_s = _schedule_s_sec(fobj, "sldtMin_orig")
    sibt_s = _schedule_s_sec(fobj, "sibtMin_orig")
    sobt_s = _schedule_s_sec(fobj, "sobtMin_orig")
    stot_s = _schedule_s_sec(fobj, "stotMin_orig")

    sldt_d = _schedule_sd_sec(fobj, "sldtMin_d")
    sibt_d = _schedule_sd_sec(fobj, "sibtMin_d")
    sobt_d = _schedule_sd_sec(fobj, "sobtMin_d")
    stot_d = _schedule_sd_sec(fobj, "stotMin_d")

    taxi_in_sec: Optional[float] = None
    taxi_out_sec: Optional[float] = None
    ti, to_out = _taxi_in_out_sec_from_prep(prep, pixels_per_meter)
    if ti is not None and to_out is not None:
        taxi_in_sec, taxi_out_sec = ti, to_out

    eibt_sec: Optional[float] = None
    eobt_sec: Optional[float] = None
    etot_sec: Optional[float] = None
    if eldt_sec is not None and taxi_in_sec is not None:
        eibt_sec = float(eldt_sec) + float(taxi_in_sec)
    if eibt_sec is not None:
        eobt_sec = float(eibt_sec) + float(dwell_sec)
    if eobt_sec is not None and taxi_out_sec is not None:
        etot_sec = float(eobt_sec) + float(taxi_out_sec)

    def _sf(x: Optional[int]) -> Optional[float]:
        return float(x) if x is not None else None

    return {
        "flight_id": fid,
        "reg": _flight_opt_str(fobj, "reg"),
        "flight_number": _flight_opt_str(fobj, "flightNumber", "flight_number"),
        "aircraft_type": _flight_opt_str(fobj, "aircraftType", "aircraft_type"),
        "SLDT": sldt_s,
        "SLDT_dt": _sec_to_datetime_str(_sf(sldt_s), base_date),
        "SIBT": sibt_s,
        "SIBT_dt": _sec_to_datetime_str(_sf(sibt_s), base_date),
        "SOBT": sobt_s,
        "SOBT_dt": _sec_to_datetime_str(_sf(sobt_s), base_date),
        "STOT": stot_s,
        "STOT_dt": _sec_to_datetime_str(_sf(stot_s), base_date),
        "SLDT_sd": sldt_d,
        "SLDT_sd_dt": _sec_to_datetime_str(_sf(sldt_d), base_date),
        "SIBT_sd": sibt_d,
        "SIBT_sd_dt": _sec_to_datetime_str(_sf(sibt_d), base_date),
        "SOBT_sd": sobt_d,
        "SOBT_sd_dt": _sec_to_datetime_str(_sf(sobt_d), base_date),
        "STOT_sd": stot_d,
        "STOT_sd_dt": _sec_to_datetime_str(_sf(stot_d), base_date),
        "ELDT": eldt_sec,
        "ELDT_dt": _sec_to_datetime_str(_sf(eldt_sec), base_date),
        "EXIT_RUNWAY": None,
        "EXIT_RUNWAY_dt": None,
        "ARR_ROT_SEC": None,
        "EIBT": _sim_sec_optional(eibt_sec) if eibt_sec is not None else None,
        "EIBT_dt": _sec_to_datetime_str(eibt_sec, base_date),
        "EOBT": _sim_sec_optional(eobt_sec) if eobt_sec is not None else None,
        "EOBT_dt": _sec_to_datetime_str(eobt_sec, base_date),
        "ETOT": _sim_sec_optional(etot_sec) if etot_sec is not None else None,
        "ETOT_dt": _sec_to_datetime_str(etot_sec, base_date),
    }


def run_simulation(
    layout: Dict[str, Any],
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> Dict[str, Any]:
    information = _load_information_json()
    reverse_cost, merge_r, taxiway_h = _path_search_params(information)
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    dt_sec = _sim_time_step_sec(information, dt)
    pixels_per_meter = _layout_pixels_per_meter(information)

    flights_raw = layout.get("flights") if isinstance(layout.get("flights"), list) else []
    total = max(1, len(flights_raw))
    prep_list: List[PreparedFlightPath] = []
    agents_by_id: Dict[str, Flight] = {}

    for i, fobj in enumerate(flights_raw):
        prep = prepare_flight_path(
            fobj,
            layout,
            cell_size,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
        )
        prep_list.append(prep)
        fid = str(fobj.get("id", ""))
        if prep.ok and prep.edge_ids and prep.segment_endpoints:
            token_o = fobj.get("token") if isinstance(fobj.get("token"), dict) else {}
            arr_rwy_o = fobj.get("arrRunwayId") or token_o.get("arrRunwayId")
            td_xy_o = (
                _arr_touchdown_point_xy(fobj, layout, cell_size, str(arr_rwy_o))
                if arr_rwy_o
                else None
            )
            fin_pre, eids, eph, segs, along0, cx0, cy0, skip_ldg, v0_rem, acc_rem, g_start = (
                _split_flight_path_at_touchdown(
                    prep.edge_ids,
                    prep.segment_phases,
                    prep.segment_endpoints,
                    td_xy_o,
                    prep.segment_start_velocity_ms,
                    prep.segment_accel_ms2,
                )
            )
            prep.spawn_skip_landing_px = float(skip_ldg)
            prep.spawn_along_first_segment_px = float(along0)
            prep.playback_first_segment_index = int(g_start)
            anchor = _sd_eldt_sec(fobj)
            ppm = max(float(pixels_per_meter), 1e-9)
            ag_new = Flight(
                id=fid,
                edge_ids=list(eids),
                edge_phases=list(eph),
                edge_ids_finished=list(fin_pre),
                segment_endpoints=[(tuple(a), tuple(b)) for a, b in segs],
                planned_edge_list=list(prep.logical_edge_list),
                edge_s_along_px=float(along0),
                col=float(cx0),
                row=float(cy0),
                eldt_anchor_sec=float(anchor) if anchor is not None else None,
                segment_v0_ms=list(v0_rem),
                segment_accel_ms2=list(acc_rem),
            )
            if v0_rem and acc_rem and eph:
                ag_new.velocity_ms = _velocity_ms_at_distance_on_segment(
                    float(v0_rem[0]),
                    float(acc_rem[0]),
                    float(along0) / ppm,
                    eph[0] == PHASE_LANDING and float(acc_rem[0]) < -1e-12,
                )
            agents_by_id[fid] = ag_new
        if progress_cb:
            progress_cb(float(i + 1), float(total))

    max_dur = 0.0
    for prep in prep_list:
        if prep.ok and prep.segment_duration_sec:
            max_dur = max(max_dur, float(sum(prep.segment_duration_sec)))
    total_end = max(dt_sec, max_dur + 3.0 * dt_sec)

    agents = list(agents_by_id.values())
    for ag in agents:
        ag.history.append((0.0, ag.col, ag.row, float(ag.velocity_ms)))

    current_time = 0.0
    while True:
        has_remaining = any(ag.edge_ids for ag in agents)
        if not has_remaining:
            break
        if current_time > total_end + 1e-6:
            break
        for ag in agents:
            move_agent(ag, dt_sec, pixels_per_meter)
            ag.history.append((current_time + dt_sec, ag.col, ag.row, ag.velocity_ms))
        current_time += dt_sec
        if progress_cb:
            progress_cb(current_time, max(total_end, current_time))

    # Pop every remaining expanded segment into edge_ids_finished so edge_ids ends empty.
    drain_dt = max(3600.0, max_dur * 4.0, total_end * 2.0, dt_sec)
    for _ in range(10_000):
        if not any(ag.edge_ids for ag in agents):
            break
        for ag in agents:
            if ag.edge_ids:
                move_agent(ag, drain_dt, pixels_per_meter)
    for ag in agents:
        if not ag.edge_ids:
            ag.edge_phases.clear()
            if ag.segment_endpoints:
                ag.segment_endpoints.clear()
            ag.segment_v0_ms.clear()
            ag.segment_accel_ms2.clear()

    base_date = str(
        _deep_get(information, "tiers", "algorithm", "simulation", "baseDate", default="2026-03-31")
    )

    positions: Dict[str, List[Dict[str, Any]]] = {}
    for ag in agents:
        anchor = ag.eldt_anchor_sec
        positions[ag.id] = [
            {
                "t": int(round((anchor or 0.0) + float(t))),
                "x": round(c, 3),
                "y": round(r, 3),
                "v": round(v, 3),
            }
            for t, c, r, v in ag.history
        ]

    schedule_list: List[Dict[str, Any]] = []
    flights_detail: List[Dict[str, Any]] = []
    for i, fobj in enumerate(flights_raw):
        fid = str(fobj.get("id", ""))
        prep = prep_list[i] if i < len(prep_list) else PreparedFlightPath()
        ag = agents_by_id.get(fid)
        schedule_list.append(
            _build_schedule_row(
                fobj if isinstance(fobj, dict) else {},
                fid,
                prep,
                pixels_per_meter,
                base_date,
            )
        )
        flights_detail.append(
            {
                "flight_id": fid,
                "edge_list": list(ag.edge_ids) if ag else [],
                "edge_list_finished": list(ag.edge_ids_finished) if ag else [],
                "ok": prep.ok and ag is not None,
            }
        )

    return {
        "baseDate": base_date,
        "positions": positions,
        "schedule": schedule_list,
        "layout": None,
        "kpi": None,
        "flights_detail": flights_detail,
    }
