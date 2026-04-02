"""
Airside simulation: Dijkstra paths on the Layout_Design path graph, then a simple time-step loop
(no DES events) moving each flight along edge polylines at fixed 15 m/s (see ``layoutPixelsPerMeter``
in Information.json for px/m scale).

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
Point = Tuple[float, float]


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
    logical_edge_ids: List[str] = field(default_factory=list)
    segment_endpoints: List[Tuple[Point, Point]] = field(default_factory=list)
    leg_lengths_px: List[float] = field(default_factory=list)
    ok: bool = False
    direction_violation: bool = False


@dataclass
class Flight:
    """Playback agent: remaining ``edge_ids`` (head = current edge), completed ``edge_ids_finished``."""

    id: str
    edge_ids: List[str] = field(default_factory=list)
    edge_ids_finished: List[str] = field(default_factory=list)
    segment_endpoints: List[Tuple[Point, Point]] = field(default_factory=list)
    planned_edge_ids: List[str] = field(default_factory=list)
    edge_s_along_px: float = 0.0
    col: float = 0.0
    row: float = 0.0
    velocity_ms: float = 0.0
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

    Arrival (touchdown → apron): ``start_point`` = runway pixel on touchdown, ``end_point`` = apron.
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
    start: RouteEndpoint,
    end: RouteEndpoint,
) -> Tuple[List[str], bool, Optional[List[int]], Optional[PathGraph]]:
    """Same graph build and routing as airside_sim_rev3 ``_flight_route``; returns path for geometry."""
    g = _graph_for_direction(
        layout,
        cell_size,
        _DEFAULT_RW_DIR,
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


def _dep_lineup_token_xy(layout: Dict[str, Any], cell_size: float, runway_id: str) -> Optional[Tuple[float, float]]:
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
    rd = _DEFAULT_RW_DIR
    return r["endPx"] if normalize_rw_direction_value(rd) == "clockwise" else r["startPx"]


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


def _arr_runway_start_point_xy(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
) -> Optional[Tuple[float, float]]:
    """Layout px: ``runwayPaths[].start_point`` (sim export / Pro Sim layout)."""
    if not runway_id or not str(runway_id).strip():
        return None
    rid = str(runway_id)
    for rw in layout.get("runwayPaths") or []:
        if not isinstance(rw, dict) or str(rw.get("id", "")) != rid:
            continue
        sp = rw.get("start_point")
        if not isinstance(sp, dict):
            return None
        return _vertex_to_px(sp, cell_size)
    return None


def extract_point_to_paths(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
) -> List[List[float]]:
    """
    Token pixels as path legs for combined arrival + departure:
    arr runway start → Arr RET on runway → apron → lineup.

    First leg start uses only ``runwayPaths[].start_point`` from the layout (sim export).
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

    arr_rw_start_point = (
        _arr_runway_start_point_xy(layout, cell_size, str(arr_rwy)) if arr_rwy else None
    )
    ret_on_rw = (
        _arr_ret_runway_junction_xy(layout, cell_size, str(arr_rwy), arr_ret_tw) if arr_rwy else None
    )
    apron_point = _apron_token_xy(layout, cell_size, str(stand_id))
    dep_rw_lineup_point = _dep_lineup_token_xy(layout, cell_size, str(dep_rwy)) if dep_rwy else None

    if not arr_rwy or not dep_rwy:
        return []
    if (
        arr_rw_start_point is None
        or ret_on_rw is None
        or apron_point is None
        or dep_rw_lineup_point is None
    ):
        return []
    ax, ay = arr_rw_start_point
    ret_x, ret_y = ret_on_rw
    px, py = apron_point
    lx, ly = dep_rw_lineup_point
    return [
        [float(ax), float(ay), float(ret_x), float(ret_y)],
        [float(ret_x), float(ret_y), float(px), float(py)],
        [float(px), float(py), float(lx), float(ly)],
    ]


def _expand_geometry_from_graph_path(
    g: PathGraph,
    merged_nodes: List[int],
    pair_index: Dict[Tuple[int, int], str],
) -> Tuple[List[str], List[Tuple[Point, Point]]]:
    """
    One layout edge id per graph hop; duplicate ids when splitting ``DirectedEdgeRecord.pts`` polylines.
    """
    expanded_ids: List[str] = []
    segments: List[Tuple[Point, Point]] = []
    n_nodes = len(g.nodes)
    for i in range(len(merged_nodes) - 1):
        u, v = merged_nodes[i], merged_nodes[i + 1]
        lo, hi = (u, v) if u <= v else (v, u)
        eid = pair_index.get((lo, hi))
        if not eid:
            return [], []
        rec = g.edge_map.get(f"{u}:{v}")
        if rec is not None and len(rec.pts) >= 2:
            pts = rec.pts
            for j in range(len(pts) - 1):
                p0 = pts[j]
                p1 = pts[j + 1]
                expanded_ids.append(str(eid))
                segments.append(
                    ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
                )
        else:
            if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
                return [], []
            p0 = g.nodes[u]
            p1 = g.nodes[v]
            expanded_ids.append(str(eid))
            segments.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))))
    return expanded_ids, segments


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
            "clockwise",
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            pure_ground_exclude_runway=False,
        )
        pair_index = _pair_index_from_path_graph(g0) if g0 else {}

    logical_edge_ids: List[str] = []
    leg_route_rows: List[Tuple[List[str], Optional[List[int]], Optional[PathGraph]]] = []
    direction_violation = False

    for leg in paths:
        if len(leg) < 4:
            return PreparedFlightPath()
        sx, sy, ex, ey = float(leg[0]), float(leg[1]), float(leg[2]), float(leg[3])
        edges, dv, path, g = _flight_route_impl(
            layout,
            cell_size,
            pair_index,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            RouteEndpoint(token_pixel_xy=(sx, sy)),
            RouteEndpoint(token_pixel_xy=(ex, ey)),
        )
        if dv:
            logical_edge_ids = []
            direction_violation = True
            break
        logical_edge_ids.extend(edges)
        leg_route_rows.append((edges, path, g))

    expanded_ids: List[str] = []
    segments: List[Tuple[Point, Point]] = []
    leg_lengths_px: List[float] = []
    if not direction_violation:
        for _edges, path, g in leg_route_rows:
            if path is None or g is None:
                leg_lengths_px.append(0.0)
                continue
            ex_ids, segs = _expand_geometry_from_graph_path(g, path, pair_index)
            if not ex_ids or not segs or len(ex_ids) != len(segs):
                return PreparedFlightPath(
                    logical_edge_ids=logical_edge_ids,
                    direction_violation=False,
                    ok=False,
                )
            expanded_ids.extend(ex_ids)
            segments.extend(segs)
            leg_lengths_px.append(_path_length_px(segs))

    playback_ok = bool(expanded_ids) and bool(segments) and len(expanded_ids) == len(segments)
    return PreparedFlightPath(
        edge_ids=list(expanded_ids),
        logical_edge_ids=list(logical_edge_ids),
        segment_endpoints=segments,
        leg_lengths_px=list(leg_lengths_px),
        ok=playback_ok and not direction_violation,
        direction_violation=direction_violation,
    )


def move_agent(agent: Flight, dt: float, speed_px_per_sec: float, pixels_per_meter: float) -> None:
    """Advance along ``segment_endpoints``; completed edges move from ``edge_ids`` to ``edge_ids_finished``."""
    col0, row0 = agent.col, agent.row
    rem_t = float(dt)
    ppm = max(float(pixels_per_meter), 1e-9)
    spd = max(float(speed_px_per_sec), 1e-9)
    if rem_t <= 1e-12 or not agent.edge_ids or not agent.segment_endpoints:
        agent.velocity_ms = 0.0
        return
    if len(agent.edge_ids) != len(agent.segment_endpoints):
        agent.velocity_ms = 0.0
        return
    while rem_t > 1e-12 and agent.edge_ids and agent.segment_endpoints:
        p0, p1 = agent.segment_endpoints[0]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            agent.edge_ids_finished.append(agent.edge_ids.pop(0))
            agent.segment_endpoints.pop(0)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue
        room = seg_len - agent.edge_s_along_px
        if room <= 1e-9:
            agent.edge_ids_finished.append(agent.edge_ids.pop(0))
            agent.segment_endpoints.pop(0)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue
        max_step = spd * rem_t
        step = min(max_step, room)
        time_used = step / spd
        rem_t -= time_used
        agent.edge_s_along_px += step
        t_along = agent.edge_s_along_px / seg_len
        agent.col = p0[0] + t_along * dx
        agent.row = p0[1] + t_along * dy
        if agent.edge_s_along_px >= seg_len - 1e-9:
            agent.edge_ids_finished.append(agent.edge_ids.pop(0))
            agent.segment_endpoints.pop(0)
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


def _build_schedule_row(
    fobj: Dict[str, Any],
    fid: str,
    prep: PreparedFlightPath,
    speed_px_per_sec: float,
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
    spd = max(float(speed_px_per_sec), 1e-9)
    if prep.ok and len(prep.leg_lengths_px) >= 3:
        taxi_in_sec = (prep.leg_lengths_px[0] + prep.leg_lengths_px[1]) / spd
        taxi_out_sec = prep.leg_lengths_px[2] / spd

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
        "edge_list": [] if prep.direction_violation else list(prep.logical_edge_ids),
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
    speed_px_per_sec = TAXI_SPEED_MPS * pixels_per_meter

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
            p0 = prep.segment_endpoints[0][0]
            anchor = _sd_eldt_sec(fobj)
            agents_by_id[fid] = Flight(
                id=fid,
                edge_ids=list(prep.edge_ids),
                edge_ids_finished=[],
                segment_endpoints=[(tuple(a), tuple(b)) for a, b in prep.segment_endpoints],
                planned_edge_ids=list(prep.logical_edge_ids),
                edge_s_along_px=0.0,
                col=float(p0[0]),
                row=float(p0[1]),
                eldt_anchor_sec=float(anchor) if anchor is not None else None,
            )
        if progress_cb:
            progress_cb(float(i + 1), float(total))

    max_dur = 0.0
    for prep in prep_list:
        if prep.ok and prep.segment_endpoints:
            max_dur = max(max_dur, _path_length_px(prep.segment_endpoints) / max(speed_px_per_sec, 1e-9))
    total_end = max(dt_sec, max_dur + 3.0 * dt_sec)

    agents = list(agents_by_id.values())
    for ag in agents:
        ag.history.append((0.0, ag.col, ag.row, 0.0))

    current_time = 0.0
    while True:
        has_remaining = any(ag.edge_ids for ag in agents)
        if not has_remaining:
            break
        if current_time > total_end + 1e-6:
            break
        for ag in agents:
            move_agent(ag, dt_sec, speed_px_per_sec, pixels_per_meter)
            ag.history.append((current_time + dt_sec, ag.col, ag.row, ag.velocity_ms))
        current_time += dt_sec
        if progress_cb:
            progress_cb(current_time, max(total_end, current_time))

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
                speed_px_per_sec,
                base_date,
            )
        )
        flights_detail.append(
            {
                "flight_id": fid,
                "edge_list": [] if prep.direction_violation else list(prep.logical_edge_ids),
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
