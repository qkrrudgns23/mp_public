"""
Airside simulation: Dijkstra paths on the Layout_Design path graph, then a time-step loop (no DES
events) moving each flight along edge polylines with per-segment ``avgMoveVelocity``, landing
deceleration, and runway-exit decel (see ``layoutPixelsPerMeter`` in Information.json for px/m scale).

Schedule inputs: S series (``*_Min_orig``) and Sd series (``*_Min_d`` minutes) are read from each
flight; routing and time axis use Sd only (``eldtMin_d`` or ``sldtMin_d`` → ELDT anchor in sim
seconds). Same arrival runway: ``ELDT`` is pushed forward so the next landing is not earlier than
the previous touchdown plus landing-leg duration (touchdown through end of Landing micro-legs)
plus ``RWY_ARRIVAL_SPACING_BUFFER_SEC`` (taxing / hold-wait margin). Outputs ``schedule`` with S, Sd echo,
E times (ELDT/EIBT/EOBT/ETOT) from path lengths and ``dwellMin``.

``positions`` timelines use ``t`` = absolute schedule seconds (day base, same as ELDT scale);
``Dep_taxi`` motion starts after gate dwell so playback time aligns with EOBT. ``v`` is m/s.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from utils.designer_path_graph import (
    DirectedEdgeRecord,
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
    path_dist,
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
MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS = 15.0
ARR_RET_DECEL_MS2 = 0.5
Point = Tuple[float, float]
PHASE_LANDING = "Landing"
PHASE_ARR_TAXI = "Arr_taxi"
PHASE_DEP_TAXI = "Dep_taxi"
_EXTRACT_LEG_PHASES: Tuple[str, str, str] = (PHASE_LANDING, PHASE_ARR_TAXI, PHASE_DEP_TAXI)
SIM_MAX_TIME_SEC = 200_000.0

_LOG = logging.getLogger(__name__)

DEFAULT_EDGE_CAPACITY = 1
DEFAULT_INTERSECTION_CAPACITY = 1
DEFAULT_RUNWAY_CAPACITY = 1
DEFAULT_MIN_SEPARATION_M = 60.0
LOOKAHEAD_EDGE_COUNT = 6
EDGE_RESERVATION_DEPTH = 2
RWY_ARRIVAL_SPACING_BUFFER_SEC = 60.0
HEAVY_DECISION_INTERVAL_SEC = 5.0
DEADLOCK_THRESHOLD_SEC = 900.0
DEADLOCK_FORCE_MOVE_DURATION_SEC = 120.0
DEADLOCK_FORCED_OPEN_EXTRA_SEC = 60.0
STAGNATION_PROGRESS_EPS_M = 2.0
REROUTE_WAIT_THRESHOLD_SEC = 30.0
REROUTE_IMPROVEMENT_RATIO = 0.85
REROUTE_MAX_ATTEMPTS = 12
REVERSE_PENALTY_COST = 1_000_000.0
REROUTE_YIELD_EDGE_PENALTY = REVERSE_PENALTY_COST
REROUTE_MIN_OLD_PATH_M = 50.0
NODE_OCCUPANCY_RADIUS_M = 12.0
FOLLOW_REACTION_SEC = 2.0
FOLLOW_GAP_BUFFER_M = 5.0
_HEAD_ON_COS_THRESHOLD = -0.05
_SAME_DIR_COS_THRESHOLD = 0.5


def _arr_ret_decel_floor_ms(phase: str, path_type: str, accel_ms2: float) -> float:
    """지정 RET(Arr_taxi·runway_exit·감속): 착륙/고속탈출과 동일 최소 속도 바닥."""
    if (
        phase == PHASE_ARR_TAXI
        and str(path_type or "") == "runway_exit"
        and float(accel_ms2) < -1e-12
    ):
        return float(MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
    return 0.0


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
    segment_path_types: List[str] = field(default_factory=list)
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    eldt_anchor_sec: Optional[float] = None
    eldt_raw_sec: Optional[float] = None
    dwell_sec: float = 0.0
    actual_eibt_abs_sec: Optional[float] = None
    dep_taxi_start_sim_time: Optional[float] = None
    dep_taxi_start_abs_sec: Optional[float] = None
    arr_runway_id: Optional[str] = None
    arr_runway_dir: Optional[str] = None
    exit_runway_abs_sec: Optional[float] = None
    # 터치다운(eldt)부터 Landing 마이크로 구간 종료까지 시초(내부 간격·활주로 시간 점유).
    runway_rot_sec: float = 0.0
    motion_integrated_until_abs_sec: Optional[float] = None
    fsm_state: str = "TAXI"
    current_edge_id: Optional[str] = None
    next_edge_id: Optional[str] = None
    heading_rad: Optional[float] = None
    control_halt: bool = False
    control_speed_cap_ms: Optional[float] = None


@dataclass
class EdgeResource:
    edge_id: str
    capacity: int = DEFAULT_EDGE_CAPACITY
    min_separation_m: float = DEFAULT_MIN_SEPARATION_M
    direction_mode: str = "bidirectional"
    length_m: float = 0.0
    travel_time_sec: float = 0.0
    runway_id: Optional[str] = None
    intersection_in: Optional[str] = None
    intersection_out: Optional[str] = None
    occupied_by: List[str] = field(default_factory=list)
    reserved_by: List[str] = field(default_factory=list)
    forced_open: bool = False
    forced_open_until_sec: Optional[float] = None


@dataclass
class IntersectionResource:
    intersection_id: str
    capacity: int = DEFAULT_INTERSECTION_CAPACITY
    occupied_by: List[str] = field(default_factory=list)
    reserved_by: List[str] = field(default_factory=list)
    forced_open: bool = False
    forced_open_until_sec: Optional[float] = None


@dataclass
class RunwayResource:
    runway_id: str
    capacity: int = DEFAULT_RUNWAY_CAPACITY
    edge_ids: set[str] = field(default_factory=set)
    occupied_by: List[str] = field(default_factory=list)
    reserved_by: List[str] = field(default_factory=list)
    forced_open: bool = False
    forced_open_until_sec: Optional[float] = None


@dataclass
class AgentControlState:
    flight_id: str
    priority_rank: int = 99
    clearance: str = "PROCEED"
    wait_reason: Optional[str] = None
    reserved_edges: List[str] = field(default_factory=list)
    reserved_intersections: List[str] = field(default_factory=list)
    wait_start_sec: Optional[float] = None
    total_wait_sec: float = 0.0
    blocked_since_sec: Optional[float] = None
    reroute_attempts: int = 0
    deadlock_flag: bool = False
    forced_move_until_sec: Optional[float] = None
    stagnation_anchor_sec: Optional[float] = None
    progress_snapshot_along_m: float = 0.0
    progress_snapshot_edge_id: Optional[str] = None


@dataclass
class SimulationControlState:
    edge_resources: Dict[str, EdgeResource] = field(default_factory=dict)
    intersection_resources: Dict[str, IntersectionResource] = field(default_factory=dict)
    runway_resources: Dict[str, RunwayResource] = field(default_factory=dict)
    agent_states: Dict[str, AgentControlState] = field(default_factory=dict)
    decision_interval_sec: float = HEAVY_DECISION_INTERVAL_SEC
    lookahead_edges: int = LOOKAHEAD_EDGE_COUNT
    deadlock_threshold_sec: float = DEADLOCK_THRESHOLD_SEC
    path_graph: Optional[PathGraph] = None
    pixels_per_meter: float = 1.0
    last_decision_sim_time: float = -1e30
    taxi_speed_assumed_mps: float = TAXI_SPEED_MPS


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


def _penalize_layout_edges_on_graph(
    g: PathGraph,
    layout: Dict[str, Any],
    penalized_eids: set[str],
    penalty_add: float,
) -> None:
    if not penalized_eids or penalty_add <= 0:
        return
    pairs: List[Tuple[int, int]] = []
    raw = layout.get("Edge") or layout.get("edges") or []
    for ed in raw:
        if not isinstance(ed, dict):
            continue
        eid = str(ed.get("id") or "").strip()
        if eid not in penalized_eids:
            continue
        try:
            pairs.append((int(ed["fromIdx"]), int(ed["toIdx"])))
        except (KeyError, TypeError, ValueError):
            continue
    p = float(penalty_add)
    for a, b in pairs:
        for u, v in ((a, b), (b, a)):
            rec = g.edge_map.get(f"{u}:{v}")
            if rec is None:
                continue
            rec.cost = float(rec.cost) + p
    for ui in range(len(g.adj)):
        new_row: List[Tuple[int, float]] = []
        for v, _ in g.adj[ui]:
            rec = g.edge_map.get(f"{ui}:{v}")
            w = float(rec.cost) if rec else float(_)
            new_row.append((v, w))
        g.adj[ui] = new_row


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
    *,
    penalized_layout_edges: Optional[set[str]] = None,
    penalty_add: float = 0.0,
    accept_reverse_penalty_path: bool = False,
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
    if penalized_layout_edges and penalty_add > 0:
        _penalize_layout_edges_on_graph(g, layout, penalized_layout_edges, penalty_add)
    edges, _dist, path = flight_route(g, layout, cell_size, pair_index, start, end)
    if path is None or len(path) < 2:
        return [], False, None, g
    if not accept_reverse_penalty_path and _path_uses_reverse_penalty_edges(g, path):
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


def _point_segment_distance_sq(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx * dx + dy * dy
    if d2 < 1e-18:
        dx0 = px - x1
        dy0 = py - y1
        return dx0 * dx0 + dy0 * dy0
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / d2))
    qx = x1 + t * dx
    qy = y1 + t * dy
    dxq = px - qx
    dyq = py - qy
    return dxq * dxq + dyq * dyq


def _min_distance_point_to_polyline(
    px: float, py: float, verts: List[Tuple[float, float]]
) -> float:
    if len(verts) < 2:
        return float("inf")
    best_sq = float("inf")
    for i in range(len(verts) - 1):
        dsq = _point_segment_distance_sq(
            px, py, verts[i][0], verts[i][1], verts[i + 1][0], verts[i + 1][1]
        )
        if dsq < best_sq:
            best_sq = dsq
    return math.sqrt(best_sq)


def _oriented_arr_runway_centerline_px(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    ops_dir: Optional[str],
) -> Optional[List[Tuple[float, float]]]:
    coords = _runway_polyline_coords_px(layout, cell_size, runway_id)
    if not coords or len(coords) < 2:
        return None
    if ops_dir == "counter_clockwise":
        return list(reversed(coords))
    return coords


def _try_record_exit_runway_abs_sec(
    agent: Flight,
    layout: Dict[str, Any],
    cell_size: float,
    pixels_per_meter: float,
    threshold_m: float,
    rel_t_after_step: float,
) -> None:
    """
    First sim time (absolute schedule sec) when, after Arr RET / post-Landing segment,
    perpendicular distance from aircraft center to oriented arr runway centerline ≥ threshold_m.
    """
    if agent.exit_runway_abs_sec is not None:
        return
    if agent.eldt_anchor_sec is None:
        return
    rid = agent.arr_runway_id
    if not rid or not str(rid).strip():
        return
    if not agent.edge_phases or agent.edge_phases[0] == PHASE_LANDING:
        return
    verts = _oriented_arr_runway_centerline_px(
        layout, cell_size, str(rid), agent.arr_runway_dir
    )
    if not verts:
        return
    ppm = max(float(pixels_per_meter), 1e-9)
    d_px = _min_distance_point_to_polyline(float(agent.col), float(agent.row), verts)
    d_m = d_px / ppm
    if d_m + 1e-9 >= float(threshold_m):
        agent.exit_runway_abs_sec = float(agent.eldt_anchor_sec) + float(
            rel_t_after_step
        )


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


def _avg_move_velocity_ms_for_taxiway_id(
    layout: Dict[str, Any], taxiway_id: str, flight_id: str
) -> Optional[float]:
    """Return avg speed (m/s) if ``taxiway_id`` matches a taxiway-like record; ``None`` if no match."""
    tid = str(taxiway_id).strip()
    for bucket in (
        layout.get("taxiways"),
        layout.get("runwayTaxiways"),
        layout.get("runwayPaths"),
    ):
        if not isinstance(bucket, list):
            continue
        for obj in bucket:
            if not isinstance(obj, dict) or str(obj.get("id", "")).strip() != tid:
                continue
            v = _safe_float(obj.get("avgMoveVelocity"), float("nan"))
            if math.isfinite(v) and v > 0:
                return float(v)
            raise ValueError(
                f"avgMoveVelocity missing or invalid for link_id={tid!r} (flight_id={flight_id!r})"
            )
    return None


def _avg_move_velocity_ms_for_link(layout: Dict[str, Any], link_id: str, flight_id: str) -> float:
    lid = str(link_id).strip()
    direct = _avg_move_velocity_ms_for_taxiway_id(layout, lid, flight_id)
    if direct is not None:
        return direct
    for al in layout.get("apronLinks") or []:
        if not isinstance(al, dict) or str(al.get("id", "")).strip() != lid:
            continue
        tw = al.get("taxiwayId")
        if tw is not None and str(tw).strip() != "":
            from_tw = _avg_move_velocity_ms_for_taxiway_id(layout, str(tw).strip(), flight_id)
            if from_tw is not None:
                return from_tw
        v = _safe_float(al.get("avgMoveVelocity"), float("nan"))
        if math.isfinite(v) and v > 0:
            return float(v)
        return float(TAXI_SPEED_MPS)
    raise ValueError(
        f"link_id={lid!r} not found in layout taxiways/runwayTaxiways/runwayPaths/apronLinks "
        f"(flight_id={flight_id!r})"
    )


def _velocity_ms_at_distance_on_segment(
    v0_ms: float,
    accel_ms2: float,
    s_m: float,
    apply_landing_velocity_floor: bool,
    *,
    decel_floor_ms: float = 0.0,
) -> float:
    if abs(accel_ms2) < 1e-12:
        v = float(v0_ms)
    else:
        inner = float(v0_ms) * float(v0_ms) + 2.0 * float(accel_ms2) * float(s_m)
        v = math.sqrt(max(0.0, inner))
    if apply_landing_velocity_floor and float(accel_ms2) < -1e-12:
        v = max(v, MIN_LANDING_VELOCITY_MS)
    elif float(decel_floor_ms) > 1e-12 and float(accel_ms2) < -1e-12:
        v = max(v, float(decel_floor_ms))
    elif (
        not apply_landing_velocity_floor
        and float(accel_ms2) < -1e-12
        and v < 1e-6
    ):
        # 기타 감속: 정지 후 폴리라인 잔여 — RET는 decel_floor_ms로 처리
        v = max(v, 1.0)
    return float(v)


def _duration_slice_sec(
    v0_ms: float,
    accel_ms2: float,
    s0_m: float,
    s1_m: float,
    apply_landing_velocity_floor: bool,
    *,
    decel_floor_ms: float = 0.0,
) -> float:
    if s1_m <= s0_m + 1e-12:
        return 0.0
    if abs(accel_ms2) < 1e-12:
        v = _velocity_ms_at_distance_on_segment(
            v0_ms, accel_ms2, s0_m, apply_landing_velocity_floor, decel_floor_ms=decel_floor_ms
        )
        return (s1_m - s0_m) / max(v, 1e-9)
    n = max(8, min(128, int((s1_m - s0_m) / 3.0) + 1))
    ds = (s1_m - s0_m) / float(n)
    t = 0.0
    for i in range(n):
        sm = s0_m + (i + 0.5) * ds
        vm = _velocity_ms_at_distance_on_segment(
            v0_ms, accel_ms2, sm, apply_landing_velocity_floor, decel_floor_ms=decel_floor_ms
        )
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
            if pt == "runway_taxiway":
                v_cur = max(v_cur, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
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
                if phase == PHASE_ARR_TAXI:
                    v_t = max(float(v_t), MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                v0_out[i] = float(v_t)
                a_out[i] = 0.0
                v_cur = float(v_t)
            elif is_ret:
                _ret_floor = float(MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                # RET micro-segments: after speed drops ~0, do not integrate decel from v0≈0 (durations explode).
                if float(v_cur) < 1e-3:
                    v_t = _avg_move_velocity_ms_for_link(layout, link_id, flight_id)
                    v_t = max(float(v_t), _ret_floor)
                    v0_out[i] = float(v_t)
                    a_out[i] = 0.0
                    v_cur = float(v_t)
                    dur_out[i] = _duration_slice_sec(
                        v0_out[i], a_out[i], 0.0, seg_m, False
                    )
                    continue
                v0_out[i] = float(v_cur)
                a_out[i] = -float(ARR_RET_DECEL_MS2)
                dur_out[i] = _duration_slice_sec(
                    v0_out[i], a_out[i], 0.0, seg_m, False, decel_floor_ms=_ret_floor
                )
                v_end = _velocity_ms_at_distance_on_segment(
                    v0_out[i], a_out[i], seg_m, False, decel_floor_ms=_ret_floor
                )
                v_cur = max(float(v_end), _ret_floor)
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


def _collapse_finished_edges_for_export(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """폴리라인 마이크로 조각은 시뮬에만 쓰고, 결과 JSON은 동일 (edge_id, phase) 연속 구간을 한 줄로 합친다."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not row or not isinstance(row, dict):
            continue
        eid = str(row.get("edge_id", "")).strip()
        ph = str(row.get("phase", "")).strip()
        if not eid:
            continue
        if out:
            p = out[-1]
            if (
                str(p.get("edge_id", "")).strip() == eid
                and str(p.get("phase", "")).strip() == ph
            ):
                continue
        out.append(dict(row))
    return out


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


def _snap_agent_to_first_segment(agent: Flight) -> None:
    """현재 마이크로 세그먼트 직선 위로 (col,row)와 edge_s_along_px를 맞춘다."""
    if not agent.segment_endpoints:
        return
    p0, p1 = agent.segment_endpoints[0]
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    seg_len_px = math.hypot(dx, dy)
    if seg_len_px < 1e-9:
        return
    t, proj = project_on_segment(
        (float(p0[0]), float(p0[1])),
        (float(p1[0]), float(p1[1])),
        (float(agent.col), float(agent.row)),
    )
    agent.col = float(proj[0])
    agent.row = float(proj[1])
    agent.edge_s_along_px = float(t) * seg_len_px


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
    if agent.segment_path_types:
        agent.segment_path_types.pop(0)


def move_agent(
    agent: Flight,
    dt: float,
    pixels_per_meter: float,
    sim_time: Optional[float] = None,
    sim_time_abs: Optional[float] = None,
) -> None:
    """Advance along ``segment_endpoints`` using per-segment :math:`v_0` + constant ``acceleration_ms2``.

    If ``sim_time`` is set, the first ``Dep_taxi`` segment does not move until
    ``dep_taxi_start_sim_time`` (EOBT minus ELDT in local sim time: taxi-in + dwell).
    If ``sim_time_abs`` and ``dep_taxi_start_abs_sec`` are available, the absolute
    schedule gate takes precedence so exported ``EOBT`` can remain on-stand.
    Use ``sim_time=None`` to skip this (e.g. drain pass).
    """
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
    if (
        agent.segment_path_types
        and len(agent.segment_path_types) != len(agent.edge_ids)
    ):
        agent.velocity_ms = 0.0
        return
    _snap_agent_to_first_segment(agent)
    if (
        sim_time_abs is not None
        and agent.dep_taxi_start_abs_sec is not None
        and agent.edge_phases
        and agent.edge_phases[0] == PHASE_DEP_TAXI
        and float(sim_time_abs) <= float(agent.dep_taxi_start_abs_sec) + 1e-9
    ):
        agent.velocity_ms = 0.0
        return
    if (
        sim_time is not None
        and agent.dep_taxi_start_sim_time is not None
        and agent.edge_phases
        and agent.edge_phases[0] == PHASE_DEP_TAXI
        and float(sim_time) + 1e-9 < float(agent.dep_taxi_start_sim_time)
    ):
        agent.velocity_ms = 0.0
        return
    if agent.control_halt:
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
        _pt0 = (
            str(agent.segment_path_types[0] or "")
            if agent.segment_path_types
            else ""
        )
        _ret_floor_ma = _arr_ret_decel_floor_ms(ph, _pt0, ac)

        s_m = agent.edge_s_along_px / ppm
        room_m = seg_len_m - s_m
        if room_m <= 1e-9:
            _finish_edge_segment(agent)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue

        v_now = _velocity_ms_at_distance_on_segment(
            v0s, ac, s_m, apply_floor, decel_floor_ms=_ret_floor_ma
        )
        if agent.control_speed_cap_ms is not None and math.isfinite(float(agent.control_speed_cap_ms)):
            v_now = min(v_now, float(agent.control_speed_cap_ms))
        if agent.segment_path_types and str(agent.segment_path_types[0] or "") == "runway_taxiway":
            if ph == PHASE_LANDING or ph == PHASE_ARR_TAXI:
                v_now = max(v_now, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
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
    if (
        agent.edge_ids
        and agent.segment_path_types
        and len(agent.segment_path_types) == len(agent.edge_ids)
        and agent.edge_phases
        and agent.segment_accel_ms2
        and len(agent.segment_accel_ms2) == len(agent.edge_ids)
    ):
        _ptn = str(agent.segment_path_types[0] or "")
        _phn = agent.edge_phases[0]
        _acn = float(agent.segment_accel_ms2[0])
        if (
            _ptn == "runway_taxiway"
            and (_phn == PHASE_LANDING or _phn == PHASE_ARR_TAXI)
        ) or _arr_ret_decel_floor_ms(_phn, _ptn, _acn) > 1e-12:
            agent.velocity_ms = max(
                agent.velocity_ms, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS
            )


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


def _exit_runway_min_perpendicular_distance_m(information: Dict[str, Any]) -> float:
    v = _deep_get(
        information,
        "tiers",
        "algorithm",
        "simulation",
        "exitRunwayMinPerpendicularDistanceFromCenterlineM",
    )
    if v is not None:
        try:
            x = float(v)
            if math.isfinite(x) and x > 0:
                return x
        except (TypeError, ValueError):
            pass
    return 120.0


def _runway_release_lag_sec(information: Dict[str, Any]) -> float:
    """선행기 ``EXIT_RUNWAY`` 직후 후속 착륙 롤을 허용할 최소 여유(초). 0이면 이탈 시각 다음 스텝부터 진입."""
    v = _deep_get(
        information,
        "tiers",
        "algorithm",
        "simulation",
        "runwayReleaseLagSec",
    )
    if v is not None:
        try:
            x = float(v)
            if math.isfinite(x) and x >= 0.0:
                return x
        except (TypeError, ValueError):
            pass
    return 0.0


def _reroute_wait_threshold_sec(information: Dict[str, Any]) -> float:
    v = _deep_get(
        information,
        "tiers",
        "algorithm",
        "simulation",
        "rerouteWaitThresholdSec",
    )
    if v is not None:
        try:
            x = float(v)
            if math.isfinite(x) and x >= 1.0:
                return x
        except (TypeError, ValueError):
            pass
    return float(REROUTE_WAIT_THRESHOLD_SEC)


def _layout_edges_touching_intersection(
    control_state: SimulationControlState,
    intersection_id: str,
) -> set[str]:
    iid = str(intersection_id).strip()
    out: set[str] = set()
    if not iid:
        return out
    for eid, er in control_state.edge_resources.items():
        if er.intersection_in == iid or er.intersection_out == iid:
            out.add(str(eid))
    return out


def _reroute_penalized_edges_from_wait(
    agent: Flight,
    wait_reason: Optional[str],
    control_state: SimulationControlState,
    lookahead_edges: List[str],
) -> set[str]:
    """예약 실패 원인 엣지·교차로 인접 엣지를 우회 유도용 페널티 집합에 넣는다."""
    out: set[str] = set()
    wr = str(wait_reason or "").strip()
    if wr.startswith("edge_capacity:"):
        out.add(wr.split(":", 1)[1].strip())
    elif wr.startswith("separation:"):
        out.add(wr.split(":", 1)[1].strip())
    elif wr.startswith("intersection:"):
        iid = wr.split(":", 1)[1].strip()
        out |= _layout_edges_touching_intersection(control_state, iid)
    elif wr.startswith("runway_rot_busy:") and lookahead_edges:
        for e in lookahead_edges[:4]:
            s = str(e).strip()
            if s:
                out.add(s)
    for e in lookahead_edges[:2]:
        s = str(e).strip()
        if s:
            out.add(s)
    if agent.edge_ids:
        out.add(str(agent.edge_ids[0]))
    return {e for e in out if e}


def _arr_touchdown_motion_abs_sec(
    agent: Flight,
    agents: List[Flight],
    runway_release_lag_sec: float,
) -> Optional[float]:
    """
    실제 착륙 롤(위치·점유·승인)이 시작될 최소 절대 시각.

    동일 ``arr_runway_id`` 선행편(간격 조정 후 ``eldt_anchor``가 앞선 기체)이 모두
    ``exit_runway_abs_sec``를 가지면 ``max(입력 ELDT 초, max(선행 EXIT)+lag)``.
    선행이 아직 이탈 기록 전이면 간격 조정 ELDT까지 롤을 미룬다.
    """
    if agent.eldt_anchor_sec is None:
        return None
    raw_opt = agent.eldt_raw_sec
    raw = float(raw_opt) if raw_opt is not None else float(agent.eldt_anchor_sec)
    anch_f = float(agent.eldt_anchor_sec)
    rw = str(agent.arr_runway_id or "").strip()
    if not rw:
        return anch_f
    my = anch_f
    lag = max(0.0, float(runway_release_lag_sec))
    need_exit = 0.0
    any_pred = False
    for o in agents:
        if o.id == agent.id:
            continue
        if str(o.arr_runway_id or "").strip() != rw:
            continue
        if o.eldt_anchor_sec is None:
            continue
        if float(o.eldt_anchor_sec) + 1e-9 >= my:
            continue
        any_pred = True
        ex = o.exit_runway_abs_sec
        if ex is None:
            return anch_f
        need_exit = max(need_exit, float(ex))
    if not any_pred:
        return max(raw, anch_f)
    return max(raw, need_exit + lag)


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
        _pty = (
            str(prep.segment_path_types[g] or "")
            if len(prep.segment_path_types) == len(segs)
            else ""
        )
        _df = _arr_ret_decel_floor_ms(str(phs[g]), _pty, float(accs[g]))
        return _duration_slice_sec(
            float(v0s[g]),
            float(accs[g]),
            s0_m,
            seg_m,
            phs[g] == PHASE_LANDING and float(accs[g]) < -1e-12,
            decel_floor_ms=_df,
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


def _arr_rot_sec_from_prep(
    prep: PreparedFlightPath,
    pixels_per_meter: float,
) -> Optional[float]:
    """Runway occupancy from touchdown through end of landing-leg micro-segments (sim seconds)."""
    if not prep.ok:
        return None
    durs = prep.segment_duration_sec
    segs = prep.segment_endpoints
    phs = prep.segment_phases
    v0s = prep.segment_start_velocity_ms
    accs = prep.segment_accel_ms2
    if (
        not durs
        or len(durs) != len(segs)
        or not prep.leg_micro_counts
        or len(v0s) != len(segs)
        or len(accs) != len(segs)
        or len(phs) != len(segs)
    ):
        return None
    c0 = int(prep.leg_micro_counts[0])
    if c0 <= 0 or c0 > len(segs):
        return None
    gi = max(0, int(prep.playback_first_segment_index))
    if gi >= c0:
        return 0.0
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
        _pty = (
            str(prep.segment_path_types[g] or "")
            if len(prep.segment_path_types) == len(segs)
            else ""
        )
        _df = _arr_ret_decel_floor_ms(str(phs[g]), _pty, float(accs[g]))
        return _duration_slice_sec(
            float(v0s[g]),
            float(accs[g]),
            s0_m,
            seg_m,
            phs[g] == PHASE_LANDING and float(accs[g]) < -1e-12,
            decel_floor_ms=_df,
        )

    rot = 0.0
    for g in range(gi, c0):
        rot += dur_from_playback_start(g)
    return rot


def _adjust_eldt_for_runway_arrival_spacing(
    flights_raw: List[Any],
    prep_list: List[PreparedFlightPath],
    pixels_per_meter: float,
) -> Dict[str, int]:
    """
    동일 ``arrRunwayId`` 도착편: 다음 편 최소 ELDT는
    ``max(입력_ELDT_k, ELDT_{k-1}+착륙구간초+RWY_ARRIVAL_SPACING_BUFFER_SEC)`` 를 따른다.
    착륙구간초는 Landing 마이크로 합(``_arr_rot_sec_from_prep``)이다.
    버퍼는 선행 택시·이탈·hold-wait 지연을 흡수한다.
    """
    rows: List[Tuple[str, Dict[str, Any], PreparedFlightPath, str, float]] = []
    ppm = max(float(pixels_per_meter), 1e-9)
    for i, fobj in enumerate(flights_raw):
        if not isinstance(fobj, dict):
            continue
        fid = str(fobj.get("id", "")).strip()
        if not fid:
            continue
        prep = prep_list[i] if i < len(prep_list) else PreparedFlightPath()
        if not prep.ok:
            continue
        token_o = fobj.get("token") if isinstance(fobj.get("token"), dict) else {}
        arr_rwy = fobj.get("arrRunwayId") or token_o.get("arrRunwayId")
        rw = str(arr_rwy).strip() if arr_rwy else ""
        if not rw:
            continue
        raw_eldt = _sd_eldt_sec(fobj)
        if raw_eldt is None:
            continue
        rot_opt = _arr_rot_sec_from_prep(prep, ppm)
        rot = float(rot_opt) if rot_opt is not None else 0.0
        rows.append((fid, fobj, prep, rw, rot))
    by_rw: Dict[str, List[Tuple[str, Dict[str, Any], PreparedFlightPath, str, float]]] = {}
    for row in rows:
        by_rw.setdefault(row[3], []).append(row)
    out: Dict[str, int] = {}
    for _rwid, lst in by_rw.items():
        lst.sort(key=lambda r: (int(_sd_eldt_sec(r[1]) or 0), r[0]))
        next_floor: Optional[float] = None
        for fid, fobj, _prep, __rw, rot in lst:
            raw = int(_sd_eldt_sec(fobj) or 0)
            if next_floor is None:
                adj = raw
            else:
                adj = max(raw, int(math.ceil(float(next_floor) - 1e-9)))
            out[fid] = adj
            next_floor = (
                float(adj)
                + float(rot)
                + float(RWY_ARRIVAL_SPACING_BUFFER_SEC)
            )
    return out


def _dwell_sec_from_flight(fobj: Dict[str, Any]) -> float:
    dwell_sec = _safe_float(fobj.get("dwellMin"), float("nan")) * 60.0
    if not math.isfinite(dwell_sec) or dwell_sec < 0:
        return 0.0
    return dwell_sec


def _compress_agent_history_for_dwell_export(
    history: List[Tuple[float, float, float, float]],
    anchor_sec: Optional[float],
    eibt_sec: Optional[float],
    eobt_sec: Optional[float],
) -> List[Tuple[float, float, float, float]]:
    """
    Drop interior samples while gate dwell (EIBT..EOBT inclusive), keeping endpoints only.
    Absolute time per row matches export: ``anchor_sec + t_rel``.
    """
    if anchor_sec is None or eibt_sec is None or eobt_sec is None:
        return history
    eibt = float(eibt_sec)
    eobt = float(eobt_sec)
    if eobt <= eibt + 1e-9:
        return history
    anch = float(anchor_sec)
    n = len(history)
    if n <= 2:
        return history
    in_band: List[int] = []
    for i in range(n):
        t_abs = float(history[i][0])
        if eibt - 1e-9 <= t_abs <= eobt + 1e-9:
            in_band.append(i)
    if len(in_band) <= 2:
        return history
    first_i = int(in_band[0])
    next_i = int(in_band[1])
    first_t, first_x, first_y, first_v = history[first_i]
    next_t, next_x, next_y, next_v = history[next_i]
    if (
        abs(float(first_t) - eibt) <= 1e-9
        and (
            abs(float(first_x) - float(next_x)) > 1e-9
            or abs(float(first_y) - float(next_y)) > 1e-9
            or abs(float(first_v)) > 1e-9
        )
    ):
        hist_mut = list(history)
        hist_mut[first_i] = (float(first_t), float(next_x), float(next_y), 0.0)
        history = hist_mut
    last_i = int(in_band[-1])
    prev_i = int(in_band[-2])
    last_t, last_x, last_y, last_v = history[last_i]
    prev_t, prev_x, prev_y, prev_v = history[prev_i]
    if (
        abs(float(last_t) - eobt) <= 1e-9
        and (
            abs(float(last_x) - float(prev_x)) > 1e-9
            or abs(float(last_y) - float(prev_y)) > 1e-9
            or abs(float(last_v)) > 1e-9
        )
    ):
        hist_mut = list(history)
        hist_mut[last_i] = (float(last_t), float(prev_x), float(prev_y), 0.0)
        history = hist_mut
    drop = set(in_band[1:-1])
    return [history[i] for i in range(n) if i not in drop]


def _build_schedule_row(
    fobj: Dict[str, Any],
    fid: str,
    prep: PreparedFlightPath,
    pixels_per_meter: float,
    base_date: str,
    exit_runway_abs_sec: Optional[float] = None,
    eldt_schedule_sec: Optional[int] = None,
    actual_eibt_abs_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    S series from ``*_Min_orig``; Sd echo from ``*_Min_d``; E series from path timing + dwell.
    ``eldt_schedule_sec``가 있으면 동일 활주로 간격 조정된 ELDT를 쓰고, 없으면 ``_sd_eldt_sec``만 쓴다.
    """
    dwell_sec = _dwell_sec_from_flight(fobj)

    eldt_sec = (
        int(eldt_schedule_sec)
        if eldt_schedule_sec is not None
        else _sd_eldt_sec(fobj)
    )

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
    if actual_eibt_abs_sec is not None:
        eibt_sec = float(actual_eibt_abs_sec)
    elif eldt_sec is not None and taxi_in_sec is not None:
        eibt_sec = float(eldt_sec) + float(taxi_in_sec)
    if eibt_sec is not None:
        eobt_sec = float(eibt_sec) + float(dwell_sec)
    if eobt_sec is not None and taxi_out_sec is not None:
        etot_sec = float(eobt_sec) + float(taxi_out_sec)

    def _sf(x: Optional[int]) -> Optional[float]:
        return float(x) if x is not None else None

    exit_rw_s = (
        _sim_sec_optional(exit_runway_abs_sec)
        if exit_runway_abs_sec is not None
        else None
    )

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
        "EXIT_RUNWAY": exit_rw_s,
        "EXIT_RUNWAY_dt": _sec_to_datetime_str(exit_runway_abs_sec, base_date),
        "EIBT": _sim_sec_optional(eibt_sec) if eibt_sec is not None else None,
        "EIBT_dt": _sec_to_datetime_str(eibt_sec, base_date),
        "EOBT": _sim_sec_optional(eobt_sec) if eobt_sec is not None else None,
        "EOBT_dt": _sec_to_datetime_str(eobt_sec, base_date),
        "ETOT": _sim_sec_optional(etot_sec) if etot_sec is not None else None,
        "ETOT_dt": _sec_to_datetime_str(etot_sec, base_date),
    }


def _layout_edge_capacity(ed: Dict[str, Any]) -> int:
    raw = ed.get("capacity")
    try:
        c = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_EDGE_CAPACITY
    return max(1, min(999, c))


def _intersection_node_id(node_idx: int) -> str:
    return f"N{int(node_idx)}"


def _directed_rec_for_pair(
    g: PathGraph, a: int, b: int
) -> Optional[DirectedEdgeRecord]:
    return g.edge_map.get(f"{a}:{b}") or g.edge_map.get(f"{b}:{a}")


def _edge_direction_mode_from_graph_rec(rec: DirectedEdgeRecord) -> str:
    d = str(rec.direction or "").strip()
    if d == "both":
        return "bidirectional"
    return "one_way"


def build_resource_model(
    layout: Dict[str, Any],
    information: Dict[str, Any],
    *,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    pixels_per_meter: float,
) -> SimulationControlState:
    sep_cfg = _deep_get(
        information,
        "tiers",
        "algorithm",
        "simulation",
        "minEdgeSeparationM",
    )
    default_sep = DEFAULT_MIN_SEPARATION_M
    if sep_cfg is not None:
        try:
            s = float(sep_cfg)
            if math.isfinite(s) and s > 0:
                default_sep = s
        except (TypeError, ValueError):
            pass
    ppm = max(float(pixels_per_meter), 1e-9)
    g = _graph_for_direction(
        layout,
        float(cell_size),
        _DEFAULT_RW_DIR,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        pure_ground_exclude_runway=False,
    )
    if g is None:
        _LOG.warning(
            "path graph build returned None; resource lengths/runway metadata may be incomplete"
        )
    raw_edges = layout.get("Edge") or layout.get("edges") or []
    edge_rows: List[Dict[str, Any]] = [e for e in raw_edges if isinstance(e, dict)]
    edge_resources: Dict[str, EdgeResource] = {}
    runway_to_edges: Dict[str, set[str]] = {}
    if not edge_rows:
        raise ValueError("layout has no Edge[] entries; cannot build resource model")
    for ed in edge_rows:
        eid = str(ed.get("id") or "").strip()
        if not eid:
            continue
        try:
            fi = int(ed["fromIdx"])
            ti = int(ed["toIdx"])
        except (KeyError, TypeError, ValueError):
            continue
        cap = _layout_edge_capacity(ed)
        n_in = _intersection_node_id(fi)
        n_out = _intersection_node_id(ti)
        length_m = 0.0
        direction_mode = "bidirectional"
        runway_id: Optional[str] = None
        path_type = "taxiway"
        rec = _directed_rec_for_pair(g, fi, ti) if g is not None else None
        if rec is not None:
            length_m = float(rec.raw_dist) / ppm
            direction_mode = _edge_direction_mode_from_graph_rec(rec)
            path_type = str(rec.path_type or "")
            if path_type == "runway" and rec.link_id:
                runway_id = str(rec.link_id)
        travel_time = (
            length_m / max(float(TAXI_SPEED_MPS), 1e-6) if length_m > 1e-9 else 0.0
        )
        er = EdgeResource(
            edge_id=eid,
            capacity=cap,
            min_separation_m=float(default_sep),
            direction_mode=direction_mode,
            length_m=length_m,
            travel_time_sec=travel_time,
            runway_id=runway_id,
            intersection_in=n_in,
            intersection_out=n_out,
        )
        edge_resources[eid] = er
        if runway_id is not None:
            runway_to_edges.setdefault(str(runway_id), set()).add(eid)
    intersection_resources: Dict[str, IntersectionResource] = {}
    if g is not None:
        for ni in range(len(g.nodes)):
            iid = _intersection_node_id(ni)
            intersection_resources[iid] = IntersectionResource(intersection_id=iid)
    runway_resources: Dict[str, RunwayResource] = {}
    for rwid, eids in runway_to_edges.items():
        rr = RunwayResource(runway_id=str(rwid), capacity=DEFAULT_RUNWAY_CAPACITY)
        rr.edge_ids = set(eids)
        runway_resources[str(rwid)] = rr
    return SimulationControlState(
        edge_resources=edge_resources,
        intersection_resources=intersection_resources,
        runway_resources=runway_resources,
        agent_states={},
        path_graph=g,
        pixels_per_meter=ppm,
    )


def ensure_agent_control_states(control_state: SimulationControlState, agents: Iterable[Flight]) -> None:
    for ag in agents:
        fid = str(ag.id)
        if fid not in control_state.agent_states:
            control_state.agent_states[fid] = AgentControlState(flight_id=fid)


def refresh_agent_edge_fsm(agents: Iterable[Flight]) -> None:
    for ag in agents:
        if ag.edge_ids:
            ag.current_edge_id = str(ag.edge_ids[0])
            ag.next_edge_id = str(ag.edge_ids[1]) if len(ag.edge_ids) > 1 else None
            if ag.segment_endpoints:
                p0, p1 = ag.segment_endpoints[0]
                ag.heading_rad = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            ag.fsm_state = str(ag.edge_phases[0]) if ag.edge_phases else "TAXI"
        else:
            ag.current_edge_id = None
            ag.next_edge_id = None
            ag.heading_rad = None


def refresh_resource_occupancy(
    control_state: SimulationControlState,
    agents: List[Flight],
    pixels_per_meter: float,
    sim_time_abs: float,
    runway_release_lag_sec: float = 0.0,
) -> None:
    for e in control_state.edge_resources.values():
        e.occupied_by.clear()
    for ir in control_state.intersection_resources.values():
        ir.occupied_by.clear()
    for rr in control_state.runway_resources.values():
        rr.occupied_by.clear()
    g = control_state.path_graph
    ppm = max(float(pixels_per_meter), 1e-9)
    rad_px = NODE_OCCUPANCY_RADIUS_M * ppm
    t_abs = float(sim_time_abs)
    rw_lag = float(runway_release_lag_sec)
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(ag, agents, rw_lag)
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        if not ag.edge_ids:
            continue
        eid0 = str(ag.edge_ids[0])
        er = control_state.edge_resources.get(eid0)
        if er and ag.id not in er.occupied_by:
            er.occupied_by.append(ag.id)
        if er and er.runway_id:
            rr = control_state.runway_resources.get(str(er.runway_id))
            if rr and ag.id not in rr.occupied_by:
                rr.occupied_by.append(ag.id)
        if not g or not er or not ag.segment_endpoints:
            continue
        pxy = (float(ag.col), float(ag.row))
        for nid in (er.intersection_in, er.intersection_out):
            if not nid:
                continue
            idx_s = nid[1:] if nid.startswith("N") else ""
            if not idx_s.isdigit():
                continue
            ni = int(idx_s)
            if ni < 0 or ni >= len(g.nodes):
                continue
            if path_dist(pxy, g.nodes[ni]) <= rad_px:
                ir = control_state.intersection_resources.get(nid)
                if ir and ag.id not in ir.occupied_by:
                    ir.occupied_by.append(ag.id)


def get_agent_priority_rank(agent: Flight) -> int:
    if not agent.edge_phases:
        return 99
    ph = str(agent.edge_phases[0])
    pt = (
        str(agent.segment_path_types[0] or "")
        if agent.segment_path_types and len(agent.segment_path_types) == len(agent.edge_ids)
        else ""
    )
    if ph == PHASE_LANDING:
        return 1
    if ph == PHASE_ARR_TAXI and pt in ("runway", "runway_taxiway"):
        return 1
    if ph == PHASE_DEP_TAXI and pt in ("runway", "runway_taxiway"):
        return 2
    if ph == PHASE_ARR_TAXI:
        return 3
    if ph == PHASE_DEP_TAXI:
        return 4
    return 5


def _edge_progress_ratio(agent: Flight) -> float:
    if not agent.segment_endpoints or not agent.edge_ids:
        return 0.0
    p0, p1 = agent.segment_endpoints[0]
    sl = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if sl < 1e-9:
        return 0.0
    return float(agent.edge_s_along_px) / sl


def compare_agents(agent_a: Flight, agent_b: Flight, control_state: SimulationControlState) -> int:
    pa = get_agent_priority_rank(agent_a)
    pb = get_agent_priority_rank(agent_b)
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    sa = control_state.agent_states.get(agent_a.id)
    sb = control_state.agent_states.get(agent_b.id)
    wa = float(sa.total_wait_sec if sa else 0.0)
    wb = float(sb.total_wait_sec if sb else 0.0)
    if wa > wb:
        return -1
    if wa < wb:
        return 1
    pra = _edge_progress_ratio(agent_a)
    prb = _edge_progress_ratio(agent_b)
    if pra > prb:
        return -1
    if pra < prb:
        return 1
    if str(agent_a.id) < str(agent_b.id):
        return -1
    if str(agent_a.id) > str(agent_b.id):
        return 1
    return 0


def get_lookahead_edges(agent: Flight, n: int) -> List[str]:
    if not agent.edge_ids:
        return []
    nn = max(0, int(n))
    return [str(x) for x in agent.edge_ids[:nn]]


def _agents_on_edge(eid: str, agents: List[Flight]) -> List[Flight]:
    return [ag for ag in agents if ag.edge_ids and str(ag.edge_ids[0]) == eid]


def _agent_current_runway_id(
    ag: Flight, control_state: SimulationControlState
) -> Optional[str]:
    if not ag.edge_ids:
        return None
    er0 = control_state.edge_resources.get(str(ag.edge_ids[0]))
    if er0 is None or not er0.runway_id:
        return None
    return str(er0.runway_id)


def _runway_rot_reservation_blocked(
    t_abs: float, rwid: str, agent_id: str, agents: List[Flight]
) -> bool:
    """동일 도착 활주로: 다른 기체가 아직 이탈 전이면 점유로 본다.

    구간은 ``[ELDT, release)`` 이고, ``release``는 ``exit_runway_abs_sec``가
    있으면 그 절대 시각이다. 이탈 시각이 아직 없으면 선행기가 경로상으로 활주로를
    빠져나올 때까지 점유가 끝나지 않은 것으로 본다(``release = +inf``).
    """
    w = str(rwid).strip()
    if not w:
        return False
    tt = float(t_abs)
    for o in agents:
        if o.id == agent_id:
            continue
        if str(o.arr_runway_id or "").strip() != w:
            continue
        if o.eldt_anchor_sec is None:
            continue
        t0 = float(o.eldt_anchor_sec)
        ex = o.exit_runway_abs_sec
        if ex is not None:
            t1 = float(ex)
        else:
            t1 = float("inf")
        if tt + 1e-9 <= t0:
            continue
        if tt + 1e-9 < t1:
            return True
    return False


def _resource_use_count(
    occupied: List[str], reserved: List[str], agent_id: str
) -> int:
    return len({x for x in (occupied + reserved) if x != agent_id})


def _current_edge_separation_ok(
    er: EdgeResource,
    agent: Flight,
    agents: List[Flight],
    ppm: float,
    control_state: SimulationControlState,
    sim_time: float,
    runway_release_lag_sec: float,
) -> bool:
    min_sep = float(er.min_separation_m)
    if not agent.segment_endpoints:
        return True
    p0, p1 = agent.segment_endpoints[0]
    h0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    my_along = float(agent.edge_s_along_px) / ppm
    t_eff = float(sim_time)
    rw_lag = float(runway_release_lag_sec)
    for o in _agents_on_edge(er.edge_id, agents):
        if o.id == agent.id or not o.segment_endpoints:
            continue
        o_td = _arr_touchdown_motion_abs_sec(o, agents, rw_lag)
        if o_td is not None and t_eff + 1e-9 < float(o_td):
            continue
        st_o = control_state.agent_states.get(o.id)
        if (
            st_o
            and st_o.clearance == "YIELD"
            and st_o.wait_reason == "head_on"
        ):
            continue
        q0, q1 = o.segment_endpoints[0]
        h1 = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
        if math.cos(h0 - h1) <= 0.0:
            return False
        o_along = float(o.edge_s_along_px) / ppm
        if abs(my_along - o_along) < min_sep - 1e-6:
            return False
    return True


def can_reserve_path(
    agent: Flight,
    lookahead: List[str],
    control_state: SimulationControlState,
    agents: List[Flight],
    ppm: float,
    sim_time: float,
    runway_release_lag_sec: float = 0.0,
) -> Tuple[bool, str]:
    if not lookahead:
        return False, "empty_lookahead"
    aid = agent.id
    t_abs = float(sim_time)

    for idx, eid in enumerate(lookahead):
        er = control_state.edge_resources.get(eid)
        if er is None:
            return False, f"unknown_edge:{eid}"
        if er.forced_open:
            continue
        if er.runway_id:
            rwid = str(er.runway_id)
            rr = control_state.runway_resources.get(rwid)
            if rr is not None and not rr.forced_open:
                if _runway_rot_reservation_blocked(t_abs, rwid, aid, agents):
                    return False, f"runway_rot_busy:{rwid}"
        else:
            depth_cap = max(1, int(EDGE_RESERVATION_DEPTH))
            if idx >= depth_cap:
                cap_use = _resource_use_count(er.occupied_by, [], aid)
            else:
                cap_use = _resource_use_count(er.occupied_by, er.reserved_by, aid)
            if cap_use >= max(1, int(er.capacity)):
                return False, f"edge_capacity:{eid}"
        if idx == 0 and not _current_edge_separation_ok(
            er,
            agent,
            agents,
            ppm,
            control_state,
            sim_time,
            runway_release_lag_sec,
        ):
            return False, f"separation:{eid}"
        if idx < len(lookahead) - 1:
            ir_id = er.intersection_out
            if ir_id:
                ir = control_state.intersection_resources.get(ir_id)
                if ir is not None and not ir.forced_open:
                    depth_ir = max(1, int(EDGE_RESERVATION_DEPTH))
                    if idx >= depth_ir:
                        iu = _resource_use_count(ir.occupied_by, [], aid)
                    else:
                        iu = _resource_use_count(ir.occupied_by, ir.reserved_by, aid)
                    if iu >= max(1, int(ir.capacity)):
                        return False, f"intersection:{ir_id}"
    return True, ""


def _clear_all_reservations(control_state: SimulationControlState) -> None:
    for e in control_state.edge_resources.values():
        e.reserved_by.clear()
    for ir in control_state.intersection_resources.values():
        ir.reserved_by.clear()
    for rr in control_state.runway_resources.values():
        rr.reserved_by.clear()


def _expire_forced_open_resources(control_state: SimulationControlState, sim_time: float) -> None:
    t = float(sim_time)
    for e in control_state.edge_resources.values():
        u = e.forced_open_until_sec
        if e.forced_open and u is not None and t > float(u):
            e.forced_open = False
            e.forced_open_until_sec = None
    for ir in control_state.intersection_resources.values():
        u = ir.forced_open_until_sec
        if ir.forced_open and u is not None and t > float(u):
            ir.forced_open = False
            ir.forced_open_until_sec = None
    for rr in control_state.runway_resources.values():
        u = rr.forced_open_until_sec
        if rr.forced_open and u is not None and t > float(u):
            rr.forced_open = False
            rr.forced_open_until_sec = None


def _update_deadlock_stagnation_probe(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
    pixels_per_meter: float,
    runway_release_lag_sec: float = 0.0,
) -> None:
    ppm = max(float(pixels_per_meter), 1e-9)
    t = float(sim_time)
    for ag in agents:
        st = control_state.agent_states.get(ag.id)
        if st is None:
            continue
        if st.clearance == "FORCE_MOVE":
            continue
        td_ag = _arr_touchdown_motion_abs_sec(ag, agents, float(runway_release_lag_sec))
        if td_ag is not None and t + 1e-9 < float(td_ag):
            st.stagnation_anchor_sec = None
            st.progress_snapshot_edge_id = None
            continue
        if st.wait_reason == "pre_eldt":
            st.stagnation_anchor_sec = None
            st.progress_snapshot_edge_id = None
            continue
        if st.clearance not in ("WAIT", "YIELD"):
            st.stagnation_anchor_sec = None
            st.progress_snapshot_edge_id = None
            continue
        eid = str(ag.edge_ids[0]) if ag.edge_ids else ""
        along_m = float(ag.edge_s_along_px) / ppm
        if st.stagnation_anchor_sec is None:
            st.stagnation_anchor_sec = t
            st.progress_snapshot_edge_id = eid or None
            st.progress_snapshot_along_m = along_m
            continue
        if not eid or st.progress_snapshot_edge_id != eid:
            st.stagnation_anchor_sec = t
            st.progress_snapshot_edge_id = eid
            st.progress_snapshot_along_m = along_m
            continue
        if abs(along_m - float(st.progress_snapshot_along_m)) >= STAGNATION_PROGRESS_EPS_M:
            st.stagnation_anchor_sec = t
            st.progress_snapshot_along_m = along_m


def _collect_fail_safe_resources_for_agents(
    control_state: SimulationControlState,
    agents_subset: List[Flight],
) -> Tuple[List[str], List[str], List[str]]:
    edge_out: List[str] = []
    int_out: List[str] = []
    rw_out: List[str] = []
    seen_e: set[str] = set()
    seen_i: set[str] = set()
    seen_r: set[str] = set()
    n_la = max(int(control_state.lookahead_edges), 1)
    for ag in agents_subset:
        if not ag.edge_ids:
            continue
        la = get_lookahead_edges(ag, n_la)
        for eid in la:
            if eid in seen_e:
                continue
            seen_e.add(eid)
            edge_out.append(eid)
            er = control_state.edge_resources.get(eid)
            if er is None:
                continue
            if er.intersection_in and er.intersection_in not in seen_i:
                seen_i.add(er.intersection_in)
                int_out.append(er.intersection_in)
            if er.intersection_out and er.intersection_out not in seen_i:
                seen_i.add(er.intersection_out)
                int_out.append(er.intersection_out)
            if er.runway_id and str(er.runway_id) not in seen_r:
                seen_r.add(str(er.runway_id))
                rw_out.append(str(er.runway_id))
    return edge_out, int_out, rw_out


def reserve_path(
    agent: Flight,
    lookahead: List[str],
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    st = control_state.agent_states.get(agent.id)
    if st is None:
        return
    st.reserved_edges = list(lookahead)
    st.reserved_intersections.clear()
    aid = agent.id
    depth = min(len(lookahead), max(1, int(EDGE_RESERVATION_DEPTH)))
    for eid in lookahead[:depth]:
        er = control_state.edge_resources.get(eid)
        if er is None:
            continue
        if aid not in er.reserved_by:
            er.reserved_by.append(aid)
        if er.runway_id:
            rr = control_state.runway_resources.get(str(er.runway_id))
            if rr is not None and aid not in rr.reserved_by:
                rr.reserved_by.append(aid)
    for k in range(max(0, depth - 1)):
        er_k = control_state.edge_resources.get(lookahead[k])
        if er_k is None or not er_k.intersection_out:
            continue
        iid = er_k.intersection_out
        ir = control_state.intersection_resources.get(iid)
        if ir is not None and aid not in ir.reserved_by:
            ir.reserved_by.append(aid)
        st.reserved_intersections.append(iid)


def detect_head_on_conflict(
    agent_a: Flight,
    agent_b: Flight,
    control_state: SimulationControlState,
) -> bool:
    if (
        not agent_a.edge_ids
        or not agent_b.edge_ids
        or str(agent_a.edge_ids[0]) != str(agent_b.edge_ids[0])
    ):
        return False
    eid = str(agent_a.edge_ids[0])
    er = control_state.edge_resources.get(eid)
    if er is None or er.direction_mode != "bidirectional":
        return False
    if not agent_a.segment_endpoints or not agent_b.segment_endpoints:
        return False
    p0a, p1a = agent_a.segment_endpoints[0]
    p0b, p1b = agent_b.segment_endpoints[0]
    ha = math.atan2(p1a[1] - p0a[1], p1a[0] - p0a[0])
    hb = math.atan2(p1b[1] - p0b[1], p1b[0] - p0b[0])
    return bool(math.cos(ha - hb) < _HEAD_ON_COS_THRESHOLD)


def resolve_head_on_conflict(
    agent_a: Flight,
    agent_b: Flight,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    if not detect_head_on_conflict(agent_a, agent_b, control_state):
        return
    sta = control_state.agent_states.get(agent_a.id)
    stb = control_state.agent_states.get(agent_b.id)
    if sta and sta.clearance == "FORCE_MOVE":
        return
    if stb and stb.clearance == "FORCE_MOVE":
        return
    c = compare_agents(agent_a, agent_b, control_state)
    if c == 0:
        loser = agent_b if str(agent_a.id) > str(agent_b.id) else agent_a
    elif c < 0:
        loser = agent_b
    else:
        loser = agent_a
    st = control_state.agent_states.get(loser.id)
    if st:
        st.clearance = "YIELD"
        st.wait_reason = "head_on"
        if st.wait_start_sec is None:
            st.wait_start_sec = float(sim_time)


def _resolve_all_head_on(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
) -> None:
    by_eid: Dict[str, List[Flight]] = {}
    t_eff = float(sim_time)
    for ag in agents:
        if not ag.edge_ids:
            continue
        if ag.eldt_anchor_sec is not None and t_eff + 1e-9 < float(ag.eldt_anchor_sec):
            continue
        eid = str(ag.edge_ids[0])
        by_eid.setdefault(eid, []).append(ag)
    for grp in by_eid.values():
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                resolve_head_on_conflict(grp[i], grp[j], control_state, sim_time)


def detect_same_direction_conflict(
    agent_a: Flight,
    agent_b: Flight,
    control_state: SimulationControlState,
) -> bool:
    del control_state
    if (
        not agent_a.edge_ids
        or not agent_b.edge_ids
        or str(agent_a.edge_ids[0]) != str(agent_b.edge_ids[0])
    ):
        return False
    if not agent_a.segment_endpoints or not agent_b.segment_endpoints:
        return False
    p0a, p1a = agent_a.segment_endpoints[0]
    p0b, p1b = agent_b.segment_endpoints[0]
    ha = math.atan2(p1a[1] - p0a[1], p1a[0] - p0a[0])
    hb = math.atan2(p1b[1] - p0b[1], p1b[0] - p0b[0])
    return bool(math.cos(ha - hb) >= _SAME_DIR_COS_THRESHOLD)


def compute_following_speed(
    follower: Flight,
    leader: Flight,
    control_state: SimulationControlState,
    ppm: float,
) -> float:
    er_id = str(follower.edge_ids[0]) if follower.edge_ids else ""
    er = control_state.edge_resources.get(er_id)
    min_sep = float(er.min_separation_m) if er else DEFAULT_MIN_SEPARATION_M
    if not follower.segment_endpoints or not leader.segment_endpoints:
        return float(TAXI_SPEED_MPS)
    p0f, p1f = follower.segment_endpoints[0]
    p0l, p1l = leader.segment_endpoints[0]
    hf = math.atan2(p1f[1] - p0f[1], p1f[0] - p0f[0])
    hl = math.atan2(p1l[1] - p0l[1], p1l[0] - p0l[0])
    if math.cos(hf - hl) < _SAME_DIR_COS_THRESHOLD:
        return float(TAXI_SPEED_MPS)
    f_along = float(follower.edge_s_along_px) / ppm
    l_along = float(leader.edge_s_along_px) / ppm
    if l_along <= f_along + 1e-6:
        return float(TAXI_SPEED_MPS)
    gap_m = l_along - f_along
    leader_v = max(float(leader.velocity_ms), 0.0)
    raw_cap = (gap_m - FOLLOW_GAP_BUFFER_M) / max(FOLLOW_REACTION_SEC, 0.5)
    safe_spd = max(0.0, min(raw_cap, leader_v))
    v_seg = float(follower.segment_v0_ms[0]) if follower.segment_v0_ms else TAXI_SPEED_MPS
    return max(0.0, min(v_seg, safe_spd, leader_v))


def can_enter_intersection(
    agent: Flight, intersection_id: str, control_state: SimulationControlState
) -> bool:
    ir = control_state.intersection_resources.get(intersection_id)
    if ir is None or ir.forced_open:
        return True
    return (
        _resource_use_count(ir.occupied_by, ir.reserved_by, agent.id)
        < max(1, int(ir.capacity))
    )


def reserve_intersection(
    agent: Flight,
    intersection_id: str,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    del sim_time
    ir = control_state.intersection_resources.get(intersection_id)
    if ir is None:
        return
    if agent.id not in ir.reserved_by:
        ir.reserved_by.append(agent.id)


def should_run_heavy_decision(control_state: SimulationControlState, sim_time: float) -> bool:
    return float(sim_time) - float(control_state.last_decision_sim_time) + 1e-9 >= float(
        control_state.decision_interval_sec
    )


def _leg_index_for_phase(phase: str) -> int:
    if phase == PHASE_LANDING:
        return 0
    if phase == PHASE_ARR_TAXI:
        return 1
    return 2


def _reroute_leg_destination_xy(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    phase: str,
) -> Optional[Tuple[float, float]]:
    paths = extract_point_to_paths(flight, layout, cell_size)
    if not paths:
        return None
    if phase == PHASE_LANDING and paths[0] and len(paths[0]) >= 4:
        return (float(paths[0][2]), float(paths[0][3]))
    if phase == PHASE_ARR_TAXI and len(paths) > 1 and len(paths[1]) >= 4:
        return (float(paths[1][2]), float(paths[1][3]))
    if phase == PHASE_DEP_TAXI and len(paths) > 2 and len(paths[2]) >= 4:
        return (float(paths[2][2]), float(paths[2][3]))
    return None


def _yield_penalized_layout_edges_for_reroute(
    control_state: SimulationControlState,
    exclude_flight_id: str,
) -> set[str]:
    out: set[str] = set()
    for eid, er in control_state.edge_resources.items():
        for oid in er.occupied_by:
            if oid == exclude_flight_id:
                continue
            st_o = control_state.agent_states.get(oid)
            if (
                st_o
                and st_o.clearance == "YIELD"
                and st_o.wait_reason == "head_on"
            ):
                out.add(str(eid))
                break
    return out


def _estimate_remaining_route_length_m(
    agent: Flight,
    control_state: SimulationControlState,
) -> float:
    s = 0.0
    for eid in agent.edge_ids:
        er = control_state.edge_resources.get(str(eid))
        if er is not None:
            s += float(er.length_m)
    return s


def _slice_expanded_reroute_at_current_edge(
    agent: Flight,
    ex_ids: List[str],
    segs: List[Tuple[Point, Point]],
    phs: List[str],
    lnks: List[str],
    ptyps: List[str],
) -> Tuple[List[str], List[Tuple[Point, Point]], List[str], List[str], List[str]]:
    if not ex_ids or not segs or not agent.edge_ids:
        return ex_ids, segs, phs, lnks, ptyps
    if len(segs) != len(ex_ids) or len(phs) != len(ex_ids):
        return ex_ids, segs, phs, lnks, ptyps
    cur = str(agent.edge_ids[0])
    idx = next((i for i, e in enumerate(ex_ids) if str(e) == cur), None)
    if idx is None:
        return ex_ids, segs, phs, lnks, ptyps
    ex_ids = ex_ids[idx:]
    segs = list(segs[idx:])
    phs = phs[idx:]
    lnks = lnks[idx:]
    ptyps = ptyps[idx:]
    if segs:
        _p0, p1 = segs[0]
        _t, proj = project_on_segment(
            (float(_p0[0]), float(_p0[1])),
            (float(p1[0]), float(p1[1])),
            (float(agent.col), float(agent.row)),
        )
        segs[0] = (proj, (float(p1[0]), float(p1[1])))
    return ex_ids, segs, phs, lnks, ptyps


def build_reroute_path_from_xy(
    start_xy: Tuple[float, float],
    destination_xy: Tuple[float, float],
    flight: Dict[str, Any],
    agent: Flight,
    layout: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    control_state: SimulationControlState,
    *,
    extra_penalized_layout_edges: Optional[set[str]] = None,
    accept_reverse_penalty_path: bool = False,
    skip_length_improvement_check: bool = False,
) -> Optional[PreparedFlightPath]:
    """
    현재 (x,y)와 목적지 픽셀 좌표로 그래프 상 Dijkstra 경로를 구하고,
    ``prepare_flight_path``와 동일한 방식으로 마이크로 세그먼트·운동학을 채운 ``PreparedFlightPath``를 만든다.
    양보(YIELD/head_on)·예약 막힌 엣지에는 큰 페널티를 더해 우회·역주행(허용 시)을 유도한다.
    """
    if not agent.edge_phases:
        return None
    phase = str(agent.edge_phases[0])
    pair_index = _pair_index_from_layout_edge(layout)
    if not pair_index:
        g0 = _graph_for_direction(
            layout,
            cell_size,
            _flight_rw_dir_for_leg(flight, _leg_index_for_phase(phase)),
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            pure_ground_exclude_runway=False,
        )
        pair_index = _pair_index_from_path_graph(g0) if g0 else {}
    if not pair_index:
        return None
    leg_i = _leg_index_for_phase(phase)
    rw_dir = _flight_rw_dir_for_leg(flight, leg_i)
    penalized: set[str] = set(_yield_penalized_layout_edges_for_reroute(control_state, agent.id))
    if extra_penalized_layout_edges:
        penalized |= extra_penalized_layout_edges
    penalty_use = float(REROUTE_YIELD_EDGE_PENALTY) if penalized else 0.0
    _edges_route, dv, path, g = _flight_route_impl(
        layout,
        cell_size,
        pair_index,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        rw_dir,
        RouteEndpoint(token_pixel_xy=(float(start_xy[0]), float(start_xy[1]))),
        RouteEndpoint(token_pixel_xy=(float(destination_xy[0]), float(destination_xy[1]))),
        penalized_layout_edges=penalized if penalized else None,
        penalty_add=penalty_use,
        accept_reverse_penalty_path=accept_reverse_penalty_path,
    )
    if dv or path is None or g is None or len(path) < 2:
        return None
    ppm = max(float(_layout_pixels_per_meter(information)), 1e-9)
    old_remaining = _estimate_remaining_route_length_m(agent, control_state)
    new_len_m = path_total_dist(g, path) / ppm
    if not skip_length_improvement_check:
        if (
            old_remaining > float(REROUTE_MIN_OLD_PATH_M)
            and new_len_m > old_remaining * float(REROUTE_IMPROVEMENT_RATIO)
        ):
            _LOG.info(
                "reroute rejected (length) flight=%s new_m=%.1f est_old_m=%.1f",
                agent.id,
                new_len_m,
                old_remaining,
            )
            return None
    ex_ids, segs, phs, lnks, ptyps = _expand_geometry_from_graph_path(g, path, pair_index, phase)
    if (
        not ex_ids
        or not segs
        or len(ex_ids) != len(segs)
        or len(phs) != len(ex_ids)
        or len(lnks) != len(ex_ids)
        or len(ptyps) != len(ex_ids)
    ):
        return None
    ex_ids, segs, phs, lnks, ptyps = _slice_expanded_reroute_at_current_edge(
        agent, ex_ids, segs, phs, lnks, ptyps
    )
    if not ex_ids:
        return None
    fid = str(flight.get("id", agent.id))
    try:
        v0s, accs, durs = _annotate_segment_kinematics(
            flight,
            layout,
            phs,
            segs,
            lnks,
            ptyps,
            ppm,
            fid,
        )
    except ValueError:
        return None
    if len(v0s) != len(ex_ids) or len(accs) != len(ex_ids) or len(durs) != len(ex_ids):
        return None
    return PreparedFlightPath(
        edge_ids=list(ex_ids),
        segment_phases=list(phs),
        logical_edge_list=[{"edge_id": str(e), "phase": str(ph)} for e, ph in zip(ex_ids, phs)],
        segment_endpoints=segs,
        leg_lengths_px=[_path_length_px(segs)],
        leg_micro_counts=[len(ex_ids)],
        segment_link_ids=list(lnks),
        segment_path_types=list(ptyps),
        segment_start_velocity_ms=list(v0s),
        segment_accel_ms2=list(accs),
        segment_duration_sec=list(durs),
        ok=True,
        direction_violation=False,
    )


def _apply_reroute_prepared_flight_state(
    agent: Flight,
    prep: PreparedFlightPath,
    control_state: SimulationControlState,
    sim_time: float,
) -> None:
    st = control_state.agent_states.get(agent.id)
    if st is None:
        return
    agent.edge_ids = list(prep.edge_ids)
    agent.edge_phases = list(prep.segment_phases)
    agent.segment_endpoints = list(prep.segment_endpoints)
    agent.segment_v0_ms = list(prep.segment_start_velocity_ms)
    agent.segment_accel_ms2 = list(prep.segment_accel_ms2)
    agent.segment_path_types = list(prep.segment_path_types)
    if agent.segment_endpoints:
        p0, p1 = agent.segment_endpoints[0]
        seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if seg_len > 1e-9:
            t, proj = project_on_segment(p0, p1, (float(agent.col), float(agent.row)))
            agent.edge_s_along_px = float(t) * seg_len
            agent.col = float(proj[0])
            agent.row = float(proj[1])
        else:
            agent.edge_s_along_px = 0.0
    else:
        agent.edge_s_along_px = 0.0
    agent.motion_integrated_until_abs_sec = float(sim_time)
    st.reroute_attempts = int(st.reroute_attempts) + 1
    st.clearance = "PROCEED"
    st.wait_reason = None
    st.wait_start_sec = None
    st.stagnation_anchor_sec = None
    st.progress_snapshot_edge_id = None
    st.progress_snapshot_along_m = 0.0


def _try_reroute_agent_off_path_block(
    agent: Flight,
    flight: Dict[str, Any],
    control_state: SimulationControlState,
    layout: Dict[str, Any],
    information: Dict[str, Any],
    sim_time: float,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    *,
    aggressive: bool,
) -> bool:
    if not agent.edge_phases or not agent.edge_ids:
        return False
    if str(agent.edge_phases[0]) == PHASE_LANDING:
        return False
    st = control_state.agent_states.get(agent.id)
    if st is None or st.clearance == "FORCE_MOVE":
        return False
    if not aggressive and int(st.reroute_attempts) >= int(REROUTE_MAX_ATTEMPTS):
        return False
    phase = str(agent.edge_phases[0])
    dest = _reroute_leg_destination_xy(flight, layout, cell_size, phase)
    if dest is None:
        return False
    la = get_lookahead_edges(agent, control_state.lookahead_edges)
    blocked = _reroute_penalized_edges_from_wait(agent, st.wait_reason, control_state, la)
    start_xy = (float(agent.col), float(agent.row))
    prep = build_reroute_path_from_xy(
        start_xy,
        dest,
        flight,
        agent,
        layout,
        float(cell_size),
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        control_state,
        extra_penalized_layout_edges=blocked if blocked else None,
        accept_reverse_penalty_path=True,
        skip_length_improvement_check=True,
    )
    if prep is None or not prep.ok or not prep.edge_ids:
        return False
    _apply_reroute_prepared_flight_state(agent, prep, control_state, sim_time)
    _LOG.info(
        "REROUTE_OK flight=%s t=%.1f attempts=%s edges=%s aggressive=%s",
        agent.id,
        float(sim_time),
        st.reroute_attempts,
        len(agent.edge_ids),
        aggressive,
    )
    return True


def should_reroute_agent(
    agent: Flight,
    control_state: SimulationControlState,
    sim_time: float,
    information: Dict[str, Any],
) -> bool:
    del sim_time
    st = control_state.agent_states.get(agent.id)
    if st is None or not agent.edge_ids or not agent.edge_phases:
        return False
    if st.clearance == "FORCE_MOVE":
        return False
    if str(agent.edge_phases[0]) == PHASE_LANDING:
        return False
    if st.clearance not in ("WAIT", "YIELD"):
        return False
    if int(st.reroute_attempts) >= int(REROUTE_MAX_ATTEMPTS):
        return False
    thr = _reroute_wait_threshold_sec(information)
    if float(st.total_wait_sec) < float(thr):
        return False
    return True


def reroute_agent_if_needed(
    agent: Flight,
    flight: Dict[str, Any],
    control_state: SimulationControlState,
    layout: Dict[str, Any],
    information: Dict[str, Any],
    sim_time: float,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
) -> bool:
    if not should_reroute_agent(agent, control_state, sim_time, information):
        return False
    return _try_reroute_agent_off_path_block(
        agent,
        flight,
        control_state,
        layout,
        information,
        sim_time,
        cell_size,
        reverse_cost,
        merge_r,
        taxiway_h,
        aggressive=False,
    )


def _try_one_aggressive_deadlock_reroute(
    deadlocked_ids: List[str],
    agents: List[Flight],
    flights_by_id: Dict[str, Dict[str, Any]],
    control_state: SimulationControlState,
    layout: Dict[str, Any],
    information: Dict[str, Any],
    sim_time: float,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
) -> bool:
    agents_by_id: Dict[str, Flight] = {str(a.id): a for a in agents}

    def _wait_rank(fid: str) -> float:
        st = control_state.agent_states.get(fid)
        return float(st.total_wait_sec) if st else 0.0

    for fid in sorted(deadlocked_ids, key=_wait_rank, reverse=True):
        ag = agents_by_id.get(str(fid))
        fo = flights_by_id.get(str(fid))
        if ag is None or not isinstance(fo, dict):
            continue
        if _try_reroute_agent_off_path_block(
            ag,
            fo,
            control_state,
            layout,
            information,
            sim_time,
            cell_size,
            reverse_cost,
            merge_r,
            taxiway_h,
            aggressive=True,
        ):
            return True
    return False


def detect_deadlock(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
) -> List[str]:
    thr = float(control_state.deadlock_threshold_sec)
    t = float(sim_time)
    out: List[str] = []
    for ag in agents:
        st = control_state.agent_states.get(ag.id)
        if st is None:
            continue
        if st.clearance == "FORCE_MOVE":
            continue
        if st.clearance not in ("WAIT", "YIELD"):
            continue
        anchor = st.stagnation_anchor_sec
        if anchor is None:
            continue
        if t - float(anchor) + 1e-9 < thr:
            continue
        out.append(ag.id)
    return out


def resolve_deadlock(
    control_state: SimulationControlState,
    agents: List[Flight],
    deadlocked_ids: List[str],
    sim_time: float,
) -> None:
    if not deadlocked_ids:
        return
    id_set = {str(x) for x in deadlocked_ids}
    ag_list = [ag for ag in agents if str(ag.id) in id_set]
    force_until = float(sim_time) + float(DEADLOCK_FORCE_MOVE_DURATION_SEC)
    open_until = force_until + float(DEADLOCK_FORCED_OPEN_EXTRA_SEC)
    edge_ids, int_ids, rw_ids = _collect_fail_safe_resources_for_agents(
        control_state, ag_list
    )
    for eid in edge_ids:
        er = control_state.edge_resources.get(eid)
        if er is None:
            continue
        er.forced_open = True
        er.forced_open_until_sec = open_until
    for iid in int_ids:
        ir = control_state.intersection_resources.get(iid)
        if ir is None:
            continue
        ir.forced_open = True
        ir.forced_open_until_sec = open_until
    for rid in rw_ids:
        rr = control_state.runway_resources.get(rid)
        if rr is None:
            continue
        rr.forced_open = True
        rr.forced_open_until_sec = open_until
    wait_snap: Dict[str, float] = {}
    stall_snap: Dict[str, Optional[float]] = {}
    for fid in id_set:
        st = control_state.agent_states.get(fid)
        if st is None:
            continue
        st.deadlock_flag = True
        st.clearance = "FORCE_MOVE"
        st.wait_reason = "deadlock_failsafe"
        st.forced_move_until_sec = force_until
        wait_snap[fid] = float(st.total_wait_sec)
        stall_snap[fid] = st.stagnation_anchor_sec
        st.stagnation_anchor_sec = None
        st.progress_snapshot_edge_id = None
        st.wait_start_sec = None
    _LOG.warning(
        "DEADLOCK_FAILSAFE t=%.1f flights=%s total_wait_sec=%s stagnation_anchor_sec=%s "
        "force_move_until=%.1f forced_open_until=%.1f edges=%s intersections=%s runways=%s",
        float(sim_time),
        sorted(id_set),
        wait_snap,
        stall_snap,
        force_until,
        open_until,
        edge_ids,
        int_ids,
        rw_ids,
    )


def update_decisions_every_10s(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
    layout: Dict[str, Any],
    information: Dict[str, Any],
    pixels_per_meter: float,
    cell_size: float,
    flights_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    control_state.last_decision_sim_time = float(sim_time)
    ppm = max(float(pixels_per_meter), 1e-9)
    rw_lag = _runway_release_lag_sec(information)
    _expire_forced_open_resources(control_state, float(sim_time))
    _clear_all_reservations(control_state)
    t_dec = float(sim_time)
    for st in control_state.agent_states.values():
        fm = st.forced_move_until_sec
        if fm is not None and t_dec < float(fm):
            st.clearance = "FORCE_MOVE"
            st.wait_reason = None
            continue
        if fm is not None and t_dec >= float(fm):
            st.forced_move_until_sec = None
            st.deadlock_flag = False
        st.clearance = "PROCEED"
        st.wait_reason = None
    _resolve_all_head_on(control_state, agents, sim_time)

    def _decision_sort_key(ag: Flight) -> Tuple[int, int, int, float, float, str]:
        st0 = control_state.agent_states.get(ag.id)
        tw = float(st0.total_wait_sec) if st0 else 0.0
        pr = get_agent_priority_rank(ag)
        eldt_i = (
            int(round(float(ag.eldt_anchor_sec)))
            if ag.eldt_anchor_sec is not None
            else 0
        )
        eldt_tie = int(hash((eldt_i, str(ag.id), int(sim_time) // 10))) & 0x7FFFFFFF
        return (
            pr,
            eldt_i,
            eldt_tie,
            -tw,
            -_edge_progress_ratio(ag),
            str(ag.id),
        )

    ordered = sorted([ag for ag in agents if ag.edge_ids], key=_decision_sort_key)

    def _rebook_ordered_reservations() -> None:
        _clear_all_reservations(control_state)
        for ag2 in ordered:
            st2 = control_state.agent_states.get(ag2.id)
            if st2 is None or st2.clearance == "FORCE_MOVE":
                continue
            motion_td2 = _arr_touchdown_motion_abs_sec(ag2, agents, rw_lag)
            if motion_td2 is not None and t_dec + 1e-9 < float(motion_td2):
                st2.clearance = "WAIT"
                st2.wait_reason = "pre_eldt"
                continue
            la2 = get_lookahead_edges(ag2, control_state.lookahead_edges)
            ok2, reason2 = can_reserve_path(
                ag2, la2, control_state, agents, ppm, sim_time, rw_lag
            )
            if ok2:
                reserve_path(ag2, la2, control_state, sim_time)
                st2.clearance = "PROCEED"
                st2.wait_reason = None
                st2.wait_start_sec = None
            else:
                st2.clearance = "WAIT"
                st2.wait_reason = reason2 or "reservation"
                if st2.wait_start_sec is None:
                    st2.wait_start_sec = float(sim_time)

    for ag in ordered:
        st = control_state.agent_states.get(ag.id)
        if st is None:
            continue
        st.priority_rank = get_agent_priority_rank(ag)
        if st.clearance == "FORCE_MOVE":
            continue
        motion_td = _arr_touchdown_motion_abs_sec(ag, agents, rw_lag)
        if motion_td is not None and t_dec + 1e-9 < float(motion_td):
            st.clearance = "WAIT"
            st.wait_reason = "pre_eldt"
            continue
        la = get_lookahead_edges(ag, control_state.lookahead_edges)
        ok, reason = can_reserve_path(
            ag, la, control_state, agents, ppm, sim_time, rw_lag
        )
        if ok:
            reserve_path(ag, la, control_state, sim_time)
            st.clearance = "PROCEED"
            st.wait_reason = None
            st.wait_start_sec = None
        else:
            st.clearance = "WAIT"
            st.wait_reason = reason or "reservation"
            if st.wait_start_sec is None:
                st.wait_start_sec = float(sim_time)
    reverse_cost, merge_r, taxiway_h = _path_search_params(information)
    if flights_by_id:
        for ag in ordered:
            fo = flights_by_id.get(str(ag.id))
            if not isinstance(fo, dict):
                continue
            if reroute_agent_if_needed(
                ag,
                fo,
                control_state,
                layout,
                information,
                sim_time,
                cell_size,
                reverse_cost,
                merge_r,
                taxiway_h,
            ):
                _rebook_ordered_reservations()
                break
    cand = detect_deadlock(control_state, agents, sim_time)
    if cand and flights_by_id:
        _dl_limit = max(len(cand) * 4, 16)
        _dl_i = 0
        while cand and _dl_i < _dl_limit:
            _dl_i += 1
            if not _try_one_aggressive_deadlock_reroute(
                cand,
                agents,
                flights_by_id,
                control_state,
                layout,
                information,
                sim_time,
                cell_size,
                reverse_cost,
                merge_r,
                taxiway_h,
            ):
                break
            _rebook_ordered_reservations()
            cand = detect_deadlock(control_state, agents, sim_time)
    if cand:
        resolve_deadlock(control_state, agents, cand, sim_time)
    _LOG.info("airside control tick t=%.3f processed=%s deadlocked=%s", float(sim_time), len(ordered), len(cand))


def _apply_same_direction_following_caps(
    control_state: SimulationControlState,
    agents: List[Flight],
    ppm: float,
) -> None:
    by_edge: Dict[str, List[Flight]] = {}
    for ag in agents:
        if not ag.edge_ids or ag.control_halt:
            continue
        st = control_state.agent_states.get(ag.id)
        if st and st.clearance == "FORCE_MOVE":
            continue
        if st and st.clearance in ("WAIT", "YIELD"):
            continue
        eid = str(ag.edge_ids[0])
        by_edge.setdefault(eid, []).append(ag)
    for eid, group in by_edge.items():
        if len(group) < 2:
            continue
        scored: List[Tuple[float, Flight]] = []
        for ag in group:
            if not ag.segment_endpoints:
                continue
            along_m = float(ag.edge_s_along_px) / ppm
            scored.append((along_m, ag))
        scored.sort(key=lambda x: x[0])
        for i in range(len(scored) - 1):
            along_f, ag_f = scored[i]
            along_l, ag_l = scored[i + 1]
            if not detect_same_direction_conflict(ag_f, ag_l, control_state):
                continue
            gap_m = along_l - along_f
            er = control_state.edge_resources.get(eid)
            min_sep = float(er.min_separation_m) if er else DEFAULT_MIN_SEPARATION_M
            if gap_m >= min_sep + FOLLOW_GAP_BUFFER_M:
                continue
            cap = compute_following_speed(ag_f, ag_l, control_state, ppm)
            prev = ag_f.control_speed_cap_ms
            if prev is None:
                ag_f.control_speed_cap_ms = cap
            else:
                ag_f.control_speed_cap_ms = min(float(prev), float(cap))


def apply_movement_controls(
    control_state: SimulationControlState,
    agents: List[Flight],
    dt: float,
    pixels_per_meter: float,
    sim_time_abs: float,
    runway_release_lag_sec: float = 0.0,
) -> None:
    ppm = max(float(pixels_per_meter), 1e-9)
    t_end = float(sim_time_abs)
    rw_lag = float(runway_release_lag_sec)
    for ag in agents:
        ag.control_halt = False
        ag.control_speed_cap_ms = None
        st = control_state.agent_states.get(ag.id)
        if st is None:
            continue
        td0 = _arr_touchdown_motion_abs_sec(ag, agents, rw_lag)
        if td0 is not None and t_end + 1e-12 < float(td0):
            ag.control_halt = True
        elif st.clearance in ("WAIT", "YIELD"):
            ag.control_halt = True
        elif st.clearance == "FORCE_MOVE":
            ag.control_halt = False
    _apply_same_direction_following_caps(control_state, agents, ppm)
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(ag, agents, rw_lag)
        if td is not None and t_end + 1e-12 < float(td):
            continue
        eldt_eff = float(td) if td is not None else t_end
        if ag.control_halt:
            prev_h = ag.motion_integrated_until_abs_sec
            t_from_h = float(prev_h) if prev_h is not None else eldt_eff
            if t_from_h < eldt_eff:
                t_from_h = eldt_eff
            if t_end > t_from_h + 1e-12:
                ag.motion_integrated_until_abs_sec = t_end
            continue
        prev = ag.motion_integrated_until_abs_sec
        t_from = float(prev) if prev is not None else eldt_eff
        if t_from < eldt_eff:
            t_from = eldt_eff
        dt_move = t_end - t_from
        if dt_move <= 1e-12:
            continue
        move_agent(ag, dt_move, ppm, sim_time=t_end - eldt_eff, sim_time_abs=t_end)
        ag.motion_integrated_until_abs_sec = t_end


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
    exit_rw_thr_m = _exit_runway_min_perpendicular_distance_m(information)
    rw_release_lag = _runway_release_lag_sec(information)

    flights_raw = layout.get("flights") if isinstance(layout.get("flights"), list) else []
    total = max(1, len(flights_raw))
    prep_list: List[PreparedFlightPath] = []
    agents_by_id: Dict[str, Flight] = {}

    for i, fobj in enumerate(flights_raw):
        prep_list.append(
            prepare_flight_path(
                fobj,
                layout,
                cell_size,
                reverse_cost,
                merge_r,
                taxiway_h,
                information,
            )
        )

    eldt_adjust_map = _adjust_eldt_for_runway_arrival_spacing(
        flights_raw, prep_list, pixels_per_meter
    )

    for i, fobj in enumerate(flights_raw):
        prep = prep_list[i] if i < len(prep_list) else PreparedFlightPath()
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
            _prep_n = len(prep.edge_ids)
            _pt_src = prep.segment_path_types
            if (
                len(_pt_src) == _prep_n
                and 0 <= int(g_start) <= _prep_n
                and len(_pt_src[int(g_start) :]) == len(eids)
            ):
                path_rem: List[str] = list(_pt_src[int(g_start) :])
            else:
                path_rem = []
            prep.spawn_skip_landing_px = float(skip_ldg)
            prep.spawn_along_first_segment_px = float(along0)
            prep.playback_first_segment_index = int(g_start)
            ppm = max(float(pixels_per_meter), 1e-9)
            anchor_raw = _sd_eldt_sec(fobj)
            anchor_adj = eldt_adjust_map.get(fid)
            if anchor_adj is not None:
                anchor_use: Optional[float] = float(anchor_adj)
            else:
                anchor_use = float(anchor_raw) if anchor_raw is not None else None
            rot_opt = _arr_rot_sec_from_prep(prep, ppm)
            rot_sec = float(rot_opt) if rot_opt is not None else 0.0
            ti_hold, _to_unused = _taxi_in_out_sec_from_prep(prep, ppm)
            dwell_s = _dwell_sec_from_flight(fobj if isinstance(fobj, dict) else {})
            dep_start_sim = (
                float(ti_hold) + float(dwell_s) if ti_hold is not None else None
            )
            _arr_rid = str(arr_rwy_o).strip() if arr_rwy_o else ""
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
                eldt_anchor_sec=anchor_use,
                eldt_raw_sec=float(anchor_raw) if anchor_raw is not None else None,
                dwell_sec=float(dwell_s),
                segment_v0_ms=list(v0_rem),
                segment_accel_ms2=list(acc_rem),
                segment_path_types=path_rem,
                dep_taxi_start_sim_time=dep_start_sim,
                dep_taxi_start_abs_sec=(
                    float(anchor_use) + float(dep_start_sim)
                    if anchor_use is not None and dep_start_sim is not None
                    else None
                ),
                arr_runway_id=_arr_rid if _arr_rid else None,
                arr_runway_dir=_flight_rw_dir_for_leg(fobj if isinstance(fobj, dict) else {}, 0)
                if _arr_rid
                else None,
                runway_rot_sec=rot_sec,
            )
            if v0_rem and acc_rem and eph:
                _v_apply = eph[0] == PHASE_LANDING and float(acc_rem[0]) < -1e-12
                _pt0s = str(path_rem[0] or "") if path_rem else ""
                _rf = _arr_ret_decel_floor_ms(str(eph[0]), _pt0s, float(acc_rem[0]))
                v_init = _velocity_ms_at_distance_on_segment(
                    float(v0_rem[0]),
                    float(acc_rem[0]),
                    float(along0) / ppm,
                    _v_apply,
                    decel_floor_ms=_rf,
                )
                if (
                    path_rem
                    and str(path_rem[0] or "") == "runway_taxiway"
                    and eph[0] in (PHASE_LANDING, PHASE_ARR_TAXI)
                ):
                    v_init = max(v_init, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                if _rf > 1e-12:
                    v_init = max(v_init, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                ag_new.velocity_ms = v_init
            agents_by_id[fid] = ag_new
        if progress_cb:
            progress_cb(float(i + 1), float(total))

    agents = list(agents_by_id.values())

    control_state = build_resource_model(
        layout,
        information,
        cell_size=cell_size,
        reverse_cost=reverse_cost,
        merge_r=merge_r,
        taxiway_h=taxiway_h,
        pixels_per_meter=pixels_per_meter,
    )
    ensure_agent_control_states(control_state, agents)

    flights_by_id: Dict[str, Dict[str, Any]] = {}
    for fobj in flights_raw:
        if isinstance(fobj, dict) and fobj.get("id") is not None:
            flights_by_id[str(fobj["id"])] = fobj

    eldt_vals = [
        float(ag.eldt_anchor_sec)
        for ag in agents
        if ag.eldt_anchor_sec is not None
    ]
    ref_t0 = min(eldt_vals) if eldt_vals else 0.0
    current_time_abs = float(ref_t0)

    while True:
        if not any(ag.edge_ids for ag in agents):
            break
        if current_time_abs - float(ref_t0) + dt_sec > SIM_MAX_TIME_SEC + 1e-6:
            break
        current_time_abs += dt_sec
        refresh_resource_occupancy(
            control_state,
            agents,
            pixels_per_meter,
            current_time_abs,
            rw_release_lag,
        )
        refresh_agent_edge_fsm(agents)
        _update_deadlock_stagnation_probe(
            control_state,
            agents,
            current_time_abs,
            pixels_per_meter,
            rw_release_lag,
        )
        if should_run_heavy_decision(control_state, current_time_abs):
            update_decisions_every_10s(
                control_state,
                agents,
                current_time_abs,
                layout,
                information,
                pixels_per_meter,
                cell_size,
                flights_by_id,
            )
        apply_movement_controls(
            control_state,
            agents,
            dt_sec,
            pixels_per_meter,
            current_time_abs,
            rw_release_lag,
        )
        for ag in agents:
            td_h = _arr_touchdown_motion_abs_sec(ag, agents, rw_release_lag)
            if td_h is not None and current_time_abs <= float(td_h) + 1e-9:
                continue
            if (
                ag.actual_eibt_abs_sec is None
                and ag.edge_phases
                and str(ag.edge_phases[0]) == PHASE_DEP_TAXI
                and abs(float(ag.velocity_ms)) <= 1e-9
            ):
                ag.actual_eibt_abs_sec = float(current_time_abs)
                ag.dep_taxi_start_abs_sec = float(current_time_abs) + float(ag.dwell_sec)
            ag.history.append((current_time_abs, ag.col, ag.row, ag.velocity_ms))
            rel_after_td = (
                current_time_abs - float(td_h) if td_h is not None else 0.0
            )
            _try_record_exit_runway_abs_sec(
                ag,
                layout,
                cell_size,
                pixels_per_meter,
                exit_rw_thr_m,
                rel_after_td,
            )
            st_w = control_state.agent_states.get(ag.id)
            if st_w and st_w.clearance in ("WAIT", "YIELD"):
                st_w.total_wait_sec += float(dt_sec)
        if progress_cb:
            progress_cb(current_time_abs - float(ref_t0), SIM_MAX_TIME_SEC)



    # 불필요한 ag 안의 것들을 정리하는것
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

    for i, fobj in enumerate(flights_raw):
        fid = str(fobj.get("id", "")) if isinstance(fobj, dict) else ""
        if not fid:
            continue
        ag = agents_by_id.get(fid)
        if ag is None:
            continue
        prep_i = prep_list[i] if i < len(prep_list) else PreparedFlightPath()
        sched_row = _build_schedule_row(
            fobj if isinstance(fobj, dict) else {},
            fid,
            prep_i,
            pixels_per_meter,
            base_date,
            eldt_schedule_sec=eldt_adjust_map.get(fid),
            actual_eibt_abs_sec=(ag.actual_eibt_abs_sec if ag else None),
        )
        eibt_raw = sched_row.get("EIBT")
        eobt_raw = sched_row.get("EOBT")
        if eibt_raw is None or eobt_raw is None:
            continue
        ag.history = _compress_agent_history_for_dwell_export(
            ag.history,
            ag.eldt_anchor_sec,
            float(eibt_raw),
            float(eobt_raw),
        )

    positions: Dict[str, List[Dict[str, Any]]] = {}
    for ag in agents:
        positions[ag.id] = [
            {
                "t": int(round(float(t))),
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
                exit_runway_abs_sec=(ag.exit_runway_abs_sec if ag else None),
                eldt_schedule_sec=eldt_adjust_map.get(str(fid)),
                actual_eibt_abs_sec=(ag.actual_eibt_abs_sec if ag else None),
            )
        )
        _fin_raw = list(ag.edge_ids_finished) if ag else []
        flights_detail.append(
            {
                "flight_id": fid,
                "edge_list": list(ag.edge_ids) if ag else [],
                "edge_list_finished": _collapse_finished_edges_for_export(_fin_raw),
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
