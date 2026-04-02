"""
Airside simulation: agent-based DES with Layout_Design-aligned path graph (designer_path_graph).
"""
from __future__ import annotations

import heapq
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.designer_path_graph import (
    DirectedEdgeRecord,
    PathGraph,
    build_path_graph,
    motion_span_for_record,
    normalize_allowed_runway_directions,
    path_graph_from_layout_sim_export,
    path_indices_to_edge_segments,
    plan_taxi_route,
    polyline_apron_junctions_xy_for_sim_result,
    solve_arrival_path_indices,
)

_logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]
_INFO_FILE = _ROOT / "data" / "Info_storage" / "Information.json"

APPROACH_OFFSET_PX: float = 10_000.0
DEP_LINEUP_HOLD_SEC: float = 20.0
DEP_TAKEOFF_ACCEL_MS2: float = 2.5
DEFAULT_TAXI_SPEED_MS: float = 10.0
MIN_ARR_RUNWAY_VELOCITY_MS: float = 15.0
EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M: float = 150.0
_SIM_STEP_SEC: int = 1
_SIM_CELL_SIZE: float = 20.0


def _load_information_json() -> dict:
    try:
        if _INFO_FILE.is_file():
            return json.loads(_INFO_FILE.read_text(encoding="utf-8"))
    except Exception:
        _logger.warning("Failed to load Information.json for simulation", exc_info=True)
    return {}


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        val = float(v)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _vertex_xy(v: dict, layout_cell_size: float) -> Tuple[float, float]:
    if not isinstance(v, dict):
        return (0.0, 0.0)
    cs = max(float(layout_cell_size), 1e-9)
    vx, vy = v.get("x"), v.get("y")
    if vx is not None and vy is not None:
        return (_safe_float(vx), _safe_float(vy))
    return (
        _safe_float(v.get("col"), 0.0) * cs,
        _safe_float(v.get("row"), 0.0) * cs,
    )


def _minutes_to_sec(m) -> Optional[float]:
    v = _safe_float(m, float("nan"))
    return v * 60.0 if math.isfinite(v) else None


def _sim_sec_optional(sec: Optional[float]) -> Optional[int]:
    if sec is None:
        return None
    try:
        v = float(sec)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(round(v))


def _sim_sec(sec) -> int:
    try:
        v = float(sec)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(v):
        return 0
    return int(round(v))


def _polyline_length(pts: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(pts) - 1):
        total += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return total


def _polyline_point_at_distance(pts: List[Tuple[float, float]], dist: float) -> Tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    if dist <= 0:
        return pts[0]
    acc = 0.0
    for i in range(len(pts) - 1):
        seg = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if seg < 1e-9:
            continue
        if acc + seg >= dist - 1e-9:
            t = max(0, min(1, (dist - acc) / seg))
            return (
                pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                pts[i][1] + t * (pts[i + 1][1] - pts[i][1]),
            )
        acc += seg
    return pts[-1]


def _subdivide_polyline(
    pts: List[Tuple[float, float]], d_start: float, d_end: float, num_sub: int = 10
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for i in range(num_sub + 1):
        frac = i / num_sub
        d = d_start + frac * (d_end - d_start)
        result.append(_polyline_point_at_distance(pts, d))
    return result


def _runway_centerline_waypoints_td_to_ret(
    verts: List[Tuple[float, float]],
    td_d: float,
    ret_d: float,
    origin_xy: Tuple[float, float],
) -> List[Tuple[float, float]]:
    span = max(0.0, float(ret_d) - float(td_d))
    if span < 1e-6:
        return []
    num_sub = max(4, min(80, int(span / 200.0)))
    sampled = _subdivide_polyline(verts, float(td_d), float(ret_d), num_sub=num_sub)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    out: List[Tuple[float, float]] = []
    for px, py in sampled:
        if math.hypot(px - ox, py - oy) < 1.0:
            continue
        out.append((px, py))
    end_pt = _polyline_point_at_distance(verts, float(ret_d))
    if not out:
        return [end_pt]
    if math.hypot(out[-1][0] - end_pt[0], out[-1][1] - end_pt[1]) > 1.0:
        out.append(end_pt)
    return out


def _merge_runway_and_ret_polyline(
    runway_wp: List[Tuple[float, float]],
    ret_verts: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = list(runway_wp)
    if len(ret_verts) < 2:
        return out
    rv = [(float(p[0]), float(p[1])) for p in ret_verts]
    if out:
        lx, ly = out[-1][0], out[-1][1]
        fx, fy = rv[0][0], rv[0][1]
        if (fx - lx) ** 2 + (fy - ly) ** 2 < 4.0:
            rv = rv[1:]
    out.extend(rv)
    return out


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


def _min_distance_point_to_polyline(px: float, py: float, verts: List[Tuple[float, float]]) -> float:
    if len(verts) < 2:
        return math.inf
    best_sq = math.inf
    for i in range(len(verts) - 1):
        dsq = _point_segment_distance_sq(
            px, py, verts[i][0], verts[i][1], verts[i + 1][0], verts[i + 1][1]
        )
        if dsq < best_sq:
            best_sq = dsq
    return math.sqrt(best_sq)


@dataclass
class ScheduleTimes:
    sldt_sec: Optional[float] = None
    sibt_sec: Optional[float] = None
    sobt_sec: Optional[float] = None
    stot_sec: Optional[float] = None
    aldt_sec: Optional[float] = None
    e_rw_exit_sec: Optional[float] = None
    aibt_sec: Optional[float] = None
    aobt_sec: Optional[float] = None
    atot_sec: Optional[float] = None


@dataclass
class Edge:
    from_id: str
    to_id: str
    cost: float
    link_id: str
    direction: str = "both"
    path_type: str = "taxiway"


def _arclength_projection_on_polyline(
    px: float, py: float, verts: List[Tuple[float, float]]
) -> Tuple[float, Tuple[float, float]]:
    """Cumulative arclength from verts[0] to closest point on polyline, and that point."""
    if len(verts) < 2:
        return (0.0, (float(verts[0][0]), float(verts[0][1])) if verts else (0.0, (px, py)))
    best_s = 0.0
    best_pt = (float(verts[0][0]), float(verts[0][1]))
    best_d2 = float("inf")
    acc = 0.0
    for i in range(len(verts) - 1):
        ax, ay = float(verts[i][0]), float(verts[i][1])
        bx, by = float(verts[i + 1][0]), float(verts[i + 1][1])
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        if den < 1e-18:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
        qx, qy = ax + t * dx, ay + t * dy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        s_here = acc + t * math.sqrt(den)
        if d2 < best_d2:
            best_d2 = d2
            best_s = s_here
            best_pt = (qx, qy)
        acc += math.sqrt(den)
    return (best_s, best_pt)


def _forward_runway_graph_node_index(
    g: PathGraph,
    verts: List[Tuple[float, float]],
    td_xy: Tuple[float, float],
    runway_id: str,
) -> Optional[int]:
    rw_set = g.runway_node_indices_by_id.get(str(runway_id)) or set()
    if not rw_set:
        return None
    s_td, _ = _arclength_projection_on_polyline(td_xy[0], td_xy[1], verts)
    candidates: List[Tuple[float, int]] = []
    back_candidates: List[Tuple[float, int]] = []
    for idx in rw_set:
        if idx < 0 or idx >= len(g.nodes):
            continue
        nx, ny = float(g.nodes[idx][0]), float(g.nodes[idx][1])
        s_n, _ = _arclength_projection_on_polyline(nx, ny, verts)
        d = math.hypot(nx - td_xy[0], ny - td_xy[1])
        if s_n + 0.05 >= s_td:
            candidates.append((d, idx))
        else:
            back_candidates.append((d, idx))
    if candidates:
        return min(candidates)[1]
    if back_candidates:
        return min(back_candidates)[1]
    return g.nearest_path_node_from_set(rw_set, td_xy)


def _path_dijkstra_edge_filter(
    g: PathGraph,
    start_idx: int,
    end_idx: int,
    allow_uv: Callable[[int, int], bool],
) -> Optional[List[int]]:
    """Dijkstra using only arcs (u,v) for which allow_uv(u,v) is True."""
    n = len(g.nodes)
    if (
        start_idx is None
        or end_idx is None
        or n == 0
        or start_idx < 0
        or end_idx < 0
        or start_idx >= n
        or end_idx >= n
    ):
        return None
    dist = [math.inf] * n
    prev: List[Optional[int]] = [None] * n
    dist[start_idx] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end_idx:
            break
        for v, w in g.adj[u]:
            if not allow_uv(u, v):
                continue
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dist[end_idx] == math.inf or dist[end_idx] >= g.reverse_cost:
        return None
    path: List[int] = []
    cur: Optional[int] = end_idx
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _rollout_edge_allowed(
    g: PathGraph, u: int, v: int, runway_id: str, ret_id: Optional[str]
) -> bool:
    rec = g.edge_map.get(f"{u}:{v}")
    if rec is None:
        return False
    pt = str(rec.path_type or "")
    lid = str(rec.link_id or "")
    if pt == "runway" and lid == str(runway_id):
        return True
    if ret_id and pt == "runway_exit" and lid == str(ret_id):
        return True
    return False


def _merge_arrival_index_paths(pref: List[int], main: List[int]) -> List[int]:
    if not main:
        return list(pref)
    if not pref:
        return list(main)
    if pref[-1] == main[0]:
        return pref[:-1] + main
    return pref + main


def _arrival_phase1_edge_end_exclusive(edges: List[Edge]) -> int:
    k = 0
    for e in edges:
        if str(e.path_type) in ("runway", "runway_exit"):
            k += 1
        else:
            break
    return k


def _flatten_spans_to_waypoints(
    spans: List[Tuple[Tuple[float, float], Tuple[float, float]]], end_exclusive: int
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    n = min(end_exclusive, len(spans))
    for i in range(n):
        p0, p1 = spans[i]
        if not out or (out[-1][0] - p0[0]) ** 2 + (out[-1][1] - p0[1]) ** 2 > 1e-8:
            out.append((float(p0[0]), float(p0[1])))
        out.append((float(p1[0]), float(p1[1])))
    return out


@dataclass
class Flight:
    id: str
    reg: Optional[str] = None
    flight_number: Optional[str] = None
    aircraft_type: Optional[str] = None
    icao_category: Optional[str] = None
    col: float = 0.0
    row: float = 0.0
    speed_ms: float = 0.0
    velocity_ms: float = 0.0
    state: str = "SCHEDULED"
    history: list = field(default_factory=list)
    schedule: ScheduleTimes = field(default_factory=ScheduleTimes)
    assigned_stand_id: Optional[str] = None
    arr_runway_id: Optional[str] = None
    dep_runway_id: Optional[str] = None
    path_queue: list = field(default_factory=list)
    dwell_sec: Optional[float] = None
    arr_rot_sec: Optional[float] = None
    sampled_arr_ret: Optional[str] = None
    arr_runway_dir: Optional[str] = None
    dep_runway_dir: Optional[str] = None
    arr_v_td_ms: Optional[float] = None
    arr_v_ret_in_ms: Optional[float] = None
    arr_v_ret_out_ms: Optional[float] = None
    arr_td_dist_m: Optional[float] = None
    arr_ret_dist_m: Optional[float] = None
    arr_decel_ms2: Optional[float] = None
    touchdown_px: Optional[Tuple[float, float]] = None
    target_speed_ms: float = 0.0
    decel_rate: float = 0.0
    accel_rate: float = 0.0
    _departed_recorded: bool = field(default=False, repr=False)
    pending_ret_enter: bool = field(default=False, repr=False)
    arr_runway_ret_waypoints: Optional[List[Tuple[float, float]]] = None
    edge_list: List[Edge] = field(default_factory=list)
    edge_segment_endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list
    )
    edge_cursor: int = 0
    edge_s_along_px: float = 0.0
    sim_export_arrival_edge_list: List[Edge] = field(default_factory=list)
    schedule_arr_ret_id: Optional[str] = None
    _path_graph_ref: Optional[PathGraph] = field(default=None, repr=False)
    """Exclusive index in edge_list: rollout (runway + runway_exit) ends; taxi-in starts here."""
    arrival_phase1_edge_end: int = 0

    def record_position(self, t: float) -> None:
        if self.state == "SCHEDULED":
            return
        if self.history:
            t0, x0, y0, _v0 = self.history[-1]
            dt_tr = float(t) - float(t0)
            if dt_tr > 1e-9:
                v_tr = math.hypot(self.col - x0, self.row - y0) / dt_tr
            else:
                v_tr = 0.0
        else:
            v_tr = self.velocity_ms
        self.velocity_ms = v_tr
        if self.state == "DEPARTED":
            if not self._departed_recorded:
                self.history.append((t, self.col, self.row, self.velocity_ms))
                self._departed_recorded = True
            return
        self.history.append((t, self.col, self.row, self.velocity_ms))

    def is_stationary(self) -> bool:
        return self.state in ("PARKED", "SCHEDULED", "DEPARTED", "LINEUP_HOLD")

    def update_state(self, event: "Event") -> None:
        _STATE_MAP = {
            "ARRIVAL_REQUEST": "APPROACH",
            "TOUCHDOWN": "ARR_RUNWAY",
            "RET_ENTER": "RET",
            "TAXI_START": "TAXI_IN",
            "STAND_ENTER": "PARKED",
            "PUSHBACK": "TAXI_OUT",
            "LINEUP": "LINEUP_HOLD",
            "TAKEOFF_REQUEST": "TAKEOFF",
            "DEPARTED": "DEPARTED",
        }
        new = _STATE_MAP.get(event.event_type)
        if new:
            self.state = new

    def on_accepted(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "ACCEPT")

    def on_rejected(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "REJECT")

    def on_wait(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "WAIT")

    def on_reroute(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "REROUTE")

    def on_hold(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "HOLD")

    def on_speed_adjust(self, resource: "Resource", event: "Event") -> List["Event"]:
        return resource.generate_next_events(self, event, "SPEED_ADJUST")

    def on_default(self, resource: "Resource", event: "Event") -> List["Event"]:
        return []

    def create_initial_event(self, resources: Dict[str, "Resource"]) -> Optional["Event"]:
        t = self.schedule.sldt_sec
        if t is None:
            t = self.schedule.sibt_sec
        if t is None:
            return None
        rwy_res = resources.get(self.arr_runway_id)
        if rwy_res is None:
            for r in resources.values():
                if isinstance(r, RunwayResource):
                    rwy_res = r
                    break
        if rwy_res is None:
            return None
        if self.schedule.sldt_sec is not None and isinstance(rwy_res, RunwayResource):
            plan = _plan_arrival_approach_leg(rwy_res, self)
            if plan is not None:
                approach_dur, _, _, _ = plan
                t = float(self.schedule.sldt_sec) - approach_dur
        return Event(time=t, event_type="ARRIVAL_REQUEST", agent=self, resource=rwy_res)


_event_counter = 0


@dataclass(order=False)
class Event:
    time: int
    event_type: str
    agent: Flight
    resource: "Resource"
    priority: int = 0
    _seq: int = field(default=0, repr=False)

    def __post_init__(self):
        global _event_counter
        _event_counter += 1
        self._seq = _event_counter
        self.time = _sim_sec(self.time)

    def __lt__(self, other: "Event") -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._seq < other._seq


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def push(self, events) -> None:
        if events is None:
            return
        if isinstance(events, Event):
            heapq.heappush(self._heap, events)
            return
        for ev in events:
            if ev is not None:
                heapq.heappush(self._heap, ev)

    def pop(self, current_time: Union[int, float]) -> List[Event]:
        out: list[Event] = []
        while self._heap and self._heap[0].time <= current_time:
            out.append(heapq.heappop(self._heap))
        return out

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0


class Resource:
    def __init__(self, id: str, name: str = "", **kw):
        self.id = id
        self.name = name

    def request(self, agent: Flight, event: Event) -> str:
        return "ACCEPT"

    def generate_next_events(self, agent: Flight, event: Event, result: str) -> List[Event]:
        return []


def _plan_arrival_approach_leg(
    rwy: "RunwayResource", agent: Flight
) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float], List[Tuple[float, float]]]]:
    verts = rwy._vertices_pixels()
    if not verts or len(verts) < 2:
        return None
    if agent.arr_runway_dir == "counter_clockwise":
        verts = list(reversed(verts))
    rwy_start = verts[0]
    rwy_second = verts[1]
    dx = rwy_second[0] - rwy_start[0]
    dy = rwy_second[1] - rwy_start[1]
    seg_len = math.hypot(dx, dy)
    if seg_len > 1e-6:
        ux, uy = dx / seg_len, dy / seg_len
    else:
        ux, uy = 1.0, 0.0
    approach_dist_px = APPROACH_OFFSET_PX
    approach_start = (
        rwy_start[0] - ux * approach_dist_px,
        rwy_start[1] - uy * approach_dist_px,
    )
    td_dist_px = float(agent.arr_td_dist_m or 0.0)
    touchdown_pt = _polyline_point_at_distance(verts, td_dist_px)
    approach_pts = [approach_start, rwy_start, touchdown_pt]
    approach_len_px = _polyline_length(approach_pts)
    approach_speed_ms = float(agent.arr_v_td_ms or 0.0)
    approach_dur = approach_len_px / max(approach_speed_ms, 1.0)
    td_xy = (float(touchdown_pt[0]), float(touchdown_pt[1]))
    return (approach_dur, approach_start, td_xy, verts)


class RunwayResource(Resource):
    def __init__(self, id: str, name: str = "", separation_matrix: Optional[dict] = None, **kw):
        super().__init__(id, name)
        self.separation_matrix = separation_matrix or {}
        self.last_operation: Optional[Tuple[str, float, str]] = None
        self.vertices: list = kw.get("vertices", [])
        self._layout_cell_size: float = _safe_float(kw.get("layout_cell_size"), 20.0)

    def _vertices_pixels(self) -> List[Tuple[float, float]]:
        cs = max(self._layout_cell_size, 1e-9)
        return [_vertex_xy(v, cs) for v in self.vertices if isinstance(v, dict)]

    def _get_sep_sec(self, prev_op: str, cur_op: str, prev_cat: str, cur_cat: str) -> float:
        key = f"{prev_op}\u2192{cur_op}"
        matrix = self.separation_matrix.get(key, {})
        if isinstance(matrix, dict):
            row = matrix.get(prev_cat, {})
            if isinstance(row, dict):
                return _safe_float(row.get(cur_cat, 90), 90)
            return _safe_float(row, 90)
        return 90.0

    def request(self, agent: Flight, event: Event) -> str:
        if event.event_type == "ARRIVAL_REQUEST":
            if self.last_operation is None:
                return "ACCEPT"
            prev_op, prev_time, prev_cat = self.last_operation
            sep = self._get_sep_sec(prev_op, "ARR", prev_cat, str(agent.icao_category or "M"))
            if event.time - prev_time >= sep:
                return "ACCEPT"
            return "WAIT"
        if event.event_type == "TAKEOFF_REQUEST":
            if self.last_operation is None:
                return "ACCEPT"
            prev_op, prev_time, prev_cat = self.last_operation
            sep = self._get_sep_sec(prev_op, "DEP", prev_cat, str(agent.icao_category or "M"))
            if event.time - prev_time >= sep:
                return "ACCEPT"
            return "WAIT"
        return "ACCEPT"

    def generate_next_events(self, agent: Flight, event: Event, result: str) -> List[Event]:
        events: list[Event] = []
        if result == "ACCEPT":
            if event.event_type == "ARRIVAL_REQUEST":
                self.last_operation = ("ARR", event.time, str(agent.icao_category or "M"))
                plan = _plan_arrival_approach_leg(self, agent)
                if plan is not None:
                    approach_dur, approach_start, td_xy, verts = plan
                    rwy_start = verts[0]
                    touchdown_pt = (td_xy[0], td_xy[1])
                    agent.touchdown_px = td_xy
                    agent.col, agent.row = approach_start
                    agent.path_queue = [rwy_start, touchdown_pt]
                    approach_speed_ms = float(agent.arr_v_td_ms or 0.0)
                    agent.speed_ms = approach_speed_ms
                    agent.decel_rate = 0.0
                    agent.accel_rate = 0.0
                    agent.target_speed_ms = agent.speed_ms
                    events.append(
                        Event(
                            time=event.time + approach_dur,
                            event_type="TOUCHDOWN",
                            agent=agent,
                            resource=self,
                        )
                    )
                else:
                    verts_fb = self._vertices_pixels()
                    if verts_fb:
                        agent.col, agent.row = verts_fb[0]
                        agent.touchdown_px = (float(verts_fb[0][0]), float(verts_fb[0][1]))
                    else:
                        agent.touchdown_px = None
                    agent.schedule.aldt_sec = event.time
                    events.append(
                        Event(
                            time=_sim_sec(event.time + _SIM_STEP_SEC),
                            event_type="TAXI_START",
                            agent=agent,
                            resource=self,
                        )
                    )
            elif event.event_type == "TAKEOFF_REQUEST":
                self.last_operation = ("DEP", event.time, str(agent.icao_category or "M"))
                agent.schedule.atot_sec = event.time
                verts = self._vertices_pixels()
                if verts and len(verts) >= 2:
                    dep_dir = agent.dep_runway_dir
                    if dep_dir == "counter_clockwise":
                        verts = list(reversed(verts))
                    agent.path_queue = list(verts[1:])
                    agent.speed_ms = 0.1
                    agent.accel_rate = DEP_TAKEOFF_ACCEL_MS2
                    agent.decel_rate = 0.0
                    agent.target_speed_ms = 0.0
                    roll_len_m = _polyline_length(verts)
                    if roll_len_m > 0 and DEP_TAKEOFF_ACCEL_MS2 > 0:
                        roll_dur = math.sqrt(2 * roll_len_m / DEP_TAKEOFF_ACCEL_MS2)
                    else:
                        roll_dur = 30.0
                    events.append(
                        Event(time=event.time + roll_dur, event_type="DEPARTED", agent=agent, resource=self)
                    )
                else:
                    events.append(Event(time=event.time + 30.0, event_type="DEPARTED", agent=agent, resource=self))
        elif result in ("WAIT", "HOLD"):
            events.append(
                Event(
                    time=event.time + 5.0,
                    event_type=event.event_type,
                    agent=agent,
                    resource=event.resource,
                    priority=event.priority + 1,
                )
            )
        return events


class TaxiwayResource(Resource):
    def __init__(self, id: str, name: str = "", vertices: Optional[list] = None, **kw):
        super().__init__(id, name)
        self.vertices = vertices or []
        self.avg_velocity_ms: float = kw.get("avg_velocity_ms", 10.0)
        self._layout_cell_size: float = _safe_float(kw.get("layout_cell_size"), 20.0)

    def vertices_pixels(self) -> List[Tuple[float, float]]:
        cs = max(self._layout_cell_size, 1e-9)
        return [_vertex_xy(v, cs) for v in self.vertices if isinstance(v, dict)]

    def request(self, agent: Flight, event: Event) -> str:
        return "ACCEPT"

    def generate_next_events(self, agent: Flight, event: Event, result: str) -> List[Event]:
        if result != "ACCEPT":
            return [
                Event(
                    time=event.time + 3.0,
                    event_type=event.event_type,
                    agent=agent,
                    resource=self,
                    priority=event.priority + 1,
                )
            ]
        return []


class StandResource(Resource):
    def __init__(self, id: str, name: str = "", col: float = 0, row: float = 0, **kw):
        super().__init__(id, name)
        self.col = col
        self.row = row
        self.occupant: Optional[str] = None

    def request(self, agent: Flight, event: Event) -> str:
        if event.event_type == "STAND_ENTER":
            if self.occupant is None or self.occupant == agent.id:
                return "ACCEPT"
            return "WAIT"
        if event.event_type == "PUSHBACK":
            if self.occupant == agent.id:
                return "ACCEPT"
            return "REJECT"
        return "ACCEPT"

    def generate_next_events(self, agent: Flight, event: Event, result: str) -> List[Event]:
        events: list[Event] = []
        if result == "ACCEPT":
            if event.event_type == "STAND_ENTER":
                self.occupant = agent.id
                agent.col, agent.row = self.col, self.row
                agent.speed_ms = 0.0
                agent.decel_rate = 0.0
                agent.accel_rate = 0.0
                agent.path_queue = []
                agent.edge_list = []
                agent.edge_segment_endpoints = []
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                agent.schedule.aibt_sec = event.time
                pushback_time = _sim_sec(event.time + float(agent.dwell_sec or 0.0))
                agent.schedule.aobt_sec = pushback_time
                events.append(Event(time=pushback_time, event_type="PUSHBACK", agent=agent, resource=self))
            elif event.event_type == "PUSHBACK":
                self.occupant = None
        elif result in ("WAIT", "HOLD"):
            events.append(
                Event(
                    time=event.time + 5.0,
                    event_type=event.event_type,
                    agent=agent,
                    resource=self,
                    priority=event.priority + 1,
                )
            )
        return events


def _try_record_runway_perpendicular_exit(
    agent: Flight,
    current_time: int,
    resources: Dict[str, Resource],
) -> None:
    if agent.schedule.aldt_sec is None or agent.schedule.e_rw_exit_sec is not None:
        return
    rw = resources.get(agent.arr_runway_id)
    if rw is None or not isinstance(rw, RunwayResource):
        return
    if not agent.arr_runway_dir:
        return
    verts = rw._vertices_pixels()
    if len(verts) < 2:
        return
    if agent.arr_runway_dir == "counter_clockwise":
        verts = list(reversed(verts))
    d = _min_distance_point_to_polyline(agent.col, agent.row, verts)
    if d >= EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M:
        agent.schedule.e_rw_exit_sec = float(current_time)
        agent.arr_rot_sec = float(current_time) - float(agent.schedule.aldt_sec)


def _record_to_edge(g: PathGraph, rec: DirectedEdgeRecord) -> Edge:
    return Edge(
        from_id=g.node_id_str(rec.from_idx),
        to_id=g.node_id_str(rec.to_idx),
        cost=float(rec.raw_dist),
        link_id=str(rec.link_id or ""),
        direction=str(rec.direction or "both"),
        path_type=str(rec.path_type or "taxiway"),
    )


def _speed_ms_for_edge(resources: Dict[str, Resource], edge: Optional[Edge]) -> float:
    if edge is None:
        return DEFAULT_TAXI_SPEED_MS
    res = resources.get(edge.link_id)
    if isinstance(res, TaxiwayResource):
        return float(res.avg_velocity_ms)
    return DEFAULT_TAXI_SPEED_MS


def _snap_agent_onto_current_edge(agent: Flight) -> None:
    if not agent.edge_list or not agent.edge_segment_endpoints:
        return
    if agent.edge_cursor >= len(agent.edge_segment_endpoints):
        return
    p0, p1 = agent.edge_segment_endpoints[agent.edge_cursor]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    sl = math.hypot(dx, dy)
    if sl < 1e-9:
        agent.edge_s_along_px = 0.0
        agent.col, agent.row = p0[0], p0[1]
        return
    t = ((agent.col - p0[0]) * dx + (agent.row - p0[1]) * dy) / (sl * sl)
    t = max(0.0, min(1.0, t))
    agent.edge_s_along_px = t * sl
    agent.col = p0[0] + t * dx
    agent.row = p0[1] + t * dy


def _taxi_remaining_sec_from_edge_cursor(agent: Flight, resources: Dict[str, Resource]) -> float:
    edges = agent.edge_list
    spans = agent.edge_segment_endpoints
    if not edges or len(edges) != len(spans) or agent.edge_cursor >= len(edges):
        return 0.0
    total = 0.0
    for i in range(agent.edge_cursor, len(edges)):
        p0, p1 = spans[i]
        seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        spd = max(_speed_ms_for_edge(resources, edges[i]), 0.1)
        if i == agent.edge_cursor:
            left = max(0.0, seg - agent.edge_s_along_px)
            total += left / spd
        else:
            total += seg / spd
    return total


def _advance_agent_on_edge_list(agent: Flight, dt: float, resources: Dict[str, Resource]) -> None:
    col0, row0 = agent.col, agent.row
    rem_t = float(dt)
    if rem_t <= 1e-12:
        agent.velocity_ms = 0.0
        return
    edges = agent.edge_list
    spans = agent.edge_segment_endpoints
    n = len(edges)
    if n == 0 or n != len(spans) or agent.edge_cursor >= n:
        agent.velocity_ms = 0.0
        return
    while rem_t > 1e-12 and agent.edge_cursor < n:
        ei = agent.edge_cursor
        p0, p1 = spans[ei]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            agent.edge_cursor += 1
            agent.edge_s_along_px = 0.0
            continue
        spd = max(_speed_ms_for_edge(resources, edges[ei]), 0.1)
        room = seg_len - agent.edge_s_along_px
        if room <= 1e-9:
            agent.edge_cursor += 1
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
            agent.edge_cursor += 1
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
    agent.velocity_ms = math.hypot(agent.col - col0, agent.row - row0) / max(float(dt), 1e-9)


def _move_agent_along_path(
    agent: Flight, dt: float, cell_size: float, resources: Dict[str, Resource]
) -> None:
    if agent.is_stationary():
        agent.velocity_ms = 0.0
        return
    col0, row0 = agent.col, agent.row
    if agent.state == "ARR_RUNWAY":
        if not agent.path_queue:
            agent.velocity_ms = 0.0
            return
        dec = float(agent.arr_decel_ms2 or 0.0)
        if math.isfinite(dec) and dec > 0:
            agent.speed_ms = max(MIN_ARR_RUNWAY_VELOCITY_MS, agent.speed_ms - dec * dt)
        remaining = agent.speed_ms * dt
        while remaining > 0 and agent.path_queue:
            target_col, target_row = agent.path_queue[0]
            dx = target_col - agent.col
            dy = target_row - agent.row
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                agent.path_queue.pop(0)
                continue
            if remaining >= dist:
                agent.col = target_col
                agent.row = target_row
                remaining -= dist
                agent.path_queue.pop(0)
            else:
                ratio = remaining / dist
                agent.col += dx * ratio
                agent.row += dy * ratio
                remaining = 0
        agent.velocity_ms = math.hypot(agent.col - col0, agent.row - row0) / max(dt, 1e-9)
        return
    if agent.state in ("TAXI_IN", "TAXI_OUT") and agent.edge_list and agent.edge_segment_endpoints:
        if len(agent.edge_list) == len(agent.edge_segment_endpoints) and agent.edge_cursor < len(agent.edge_list):
            _advance_agent_on_edge_list(agent, dt, resources)
            return
    if not agent.path_queue:
        agent.velocity_ms = 0.0
        return
    if agent.decel_rate > 0:
        target = agent.target_speed_ms
        agent.speed_ms = max(target, agent.speed_ms - agent.decel_rate * dt)
    elif agent.accel_rate > 0:
        agent.speed_ms = agent.speed_ms + agent.accel_rate * dt
    remaining = agent.speed_ms * dt
    while remaining > 0 and agent.path_queue:
        target_col, target_row = agent.path_queue[0]
        dx = target_col - agent.col
        dy = target_row - agent.row
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            agent.path_queue.pop(0)
            continue
        if remaining >= dist:
            agent.col = target_col
            agent.row = target_row
            remaining -= dist
            agent.path_queue.pop(0)
        else:
            ratio = remaining / dist
            agent.col += dx * ratio
            agent.row += dy * ratio
            remaining = 0
    agent.velocity_ms = math.hypot(agent.col - col0, agent.row - row0) / max(dt, 1e-9)


def _plan_taxi_path_fallback(
    layout: dict,
    information: dict,
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    dep_dir: str,
) -> List[Tuple[float, float]]:
    recs, g = plan_taxi_route(layout, information, origin, dest, dep_dir)
    if recs and g:
        out: List[Tuple[float, float]] = [origin]
        for rec in recs:
            p0, p1 = motion_span_for_record(g, rec)
            if not out or math.hypot(out[-1][0] - p0[0], out[-1][1] - p0[1]) > 1.0:
                out.append(p0)
            out.append(p1)
        if math.hypot(out[-1][0] - dest[0], out[-1][1] - dest[1]) > 1.0:
            out.append(dest)
        return out if len(out) >= 2 else [origin, dest]
    return [origin, dest]


def build_resources(layout: dict, information: dict) -> Dict[str, Resource]:
    resources: Dict[str, Resource] = {}
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    cs = max(cell_size, 1e-9)
    icao_std = _deep_get(information, "tiers", "runway", "standards", "ICAO", default={})
    sep_defaults = icao_std.get("separationDefaults", {})
    rot_defaults = icao_std.get("ROT", {})
    for rw in layout.get("runwayPaths", []):
        rw_id = rw.get("id", "")
        rwy_sep_cfg = rw.get("rwySepConfig", {})
        if isinstance(rwy_sep_cfg, dict) and rwy_sep_cfg.get("seqData"):
            sep_matrix = rwy_sep_cfg["seqData"]
            rot_matrix = rwy_sep_cfg.get("rot", rot_defaults)
        else:
            sep_matrix = sep_defaults
            rot_matrix = rot_defaults
        combined_sep = dict(sep_matrix)
        if "ARR\u2192DEP" not in combined_sep and rot_matrix:
            combined_sep["ARR\u2192DEP"] = rot_matrix
        resources[rw_id] = RunwayResource(
            id=rw_id,
            name=rw.get("name", ""),
            separation_matrix=combined_sep,
            vertices=rw.get("vertices", []),
            layout_cell_size=cell_size,
        )
    for tw in layout.get("runwayTaxiways", []):
        tw_id = tw.get("id", "")
        resources[tw_id] = TaxiwayResource(
            id=tw_id,
            name=tw.get("name", ""),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 15)),
            layout_cell_size=cell_size,
        )
    for tw in layout.get("taxiways", []):
        tw_id = tw.get("id", "")
        resources[tw_id] = TaxiwayResource(
            id=tw_id,
            name=tw.get("name", ""),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 10)),
            layout_cell_size=cell_size,
        )
    for st in layout.get("pbbStands", []):
        st_id = st.get("id", "")
        col_cell = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row_cell = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""), col=col_cell * cs, row=row_cell * cs
        )
    for st in layout.get("remoteStands", []):
        st_id = st.get("id", "")
        col_cell = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row_cell = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""), col=col_cell * cs, row=row_cell * cs
        )
    return resources


def _normalize_flight_dict_for_agent(f: dict, information: dict) -> dict:
    out = dict(f)
    token = out.get("token") or {}
    if not isinstance(token, dict):
        token = {}
    rw_used = out.get("arrRunwayIdUsed")
    if rw_used is None or (isinstance(rw_used, str) and not str(rw_used).strip()):
        alt = token.get("arrRunwayId") or token.get("runwayId")
        if alt:
            out["arrRunwayIdUsed"] = alt
    ret_top = out.get("sampledArrRet")
    if ret_top is None or (isinstance(ret_top, str) and not str(ret_top).strip()):
        ex = token.get("ExitTaxiwayId")
        if ex is not None and str(ex).strip():
            out["sampledArrRet"] = ex
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    if not out.get("arrRunwayDirUsed"):
        d = sim_cfg.get("defaultArrRunwayDir")
        if d is not None and str(d).strip():
            out["arrRunwayDirUsed"] = str(d).strip()
    if not out.get("depRunwayDirUsed"):
        d = sim_cfg.get("defaultDepRunwayDir")
        if d is not None and str(d).strip():
            out["depRunwayDirUsed"] = str(d).strip()
    if out.get("dwellMin") is None and sim_cfg.get("defaultDwellMin") is not None:
        out["dwellMin"] = sim_cfg.get("defaultDwellMin")
    if not out.get("code"):
        c = sim_cfg.get("defaultIcaoCategory")
        if c is not None and str(c).strip():
            out["code"] = str(c).strip()
    if out.get("arrVRetInMs") is None:
        out["arrVRetInMs"] = _safe_float(sim_cfg.get("arrVRetInMs"), 30.0)
    if out.get("arrVRetOutMs") is None:
        out["arrVRetOutMs"] = _safe_float(sim_cfg.get("arrVRetOutMs"), 15.0)
    if out.get("arrRetDistM") is None:
        out["arrRetDistM"] = _safe_float(sim_cfg.get("arrRetDistM"), 1500.0)
    return out


def _validate_required_sim_flight_fields(f: dict, label: str) -> None:
    tok = f.get("token") if isinstance(f.get("token"), dict) else {}
    ret = f.get("sampledArrRet") or tok.get("ExitTaxiwayId")
    sret = str(ret).strip() if ret is not None else ""
    if not sret:
        raise ValueError(f"{label}: missing required ExitTaxiwayId (token) or sampledArrRet")
    rwy = f.get("arrRunwayIdUsed") or tok.get("arrRunwayId") or tok.get("runwayId")
    srwy = str(rwy).strip() if rwy is not None else ""
    if not srwy:
        raise ValueError(f"{label}: missing required token.arrRunwayId (or legacy arrRunwayIdUsed)")
    td = _safe_float(f.get("arrTdDistM"), float("nan"))
    vt = _safe_float(f.get("arrVTdMs"), float("nan"))
    if not math.isfinite(td) or td <= 0:
        raise ValueError(f"{label}: arrTdDistM must be a finite number > 0")
    if not math.isfinite(vt) or vt <= 0:
        raise ValueError(f"{label}: arrVTdMs must be a finite number > 0")
    ad = _safe_float(f.get("arrDecelMs2"), float("nan"))
    if not math.isfinite(ad) or ad <= 0:
        raise ValueError(f"{label}: arrDecelMs2 must be a finite number > 0")


def _validate_flight_sim_metadata(f: dict, label: str) -> None:
    if not str(f.get("arrRunwayDirUsed") or "").strip():
        raise ValueError(
            f"{label}: missing arrRunwayDirUsed (set on flight or tiers.algorithm.simulation.defaultArrRunwayDir)"
        )
    if not str(f.get("depRunwayDirUsed") or "").strip():
        raise ValueError(
            f"{label}: missing depRunwayDirUsed (set on flight or tiers.algorithm.simulation.defaultDepRunwayDir)"
        )
    if f.get("dwellMin") is None:
        raise ValueError(
            f"{label}: missing dwellMin (set on flight or tiers.algorithm.simulation.defaultDwellMin)"
        )
    if not str(f.get("code") or "").strip():
        raise ValueError(
            f"{label}: missing code (ICAO category; set on flight or defaultIcaoCategory in Information.json)"
        )


def _compute_arrival_phase1_and_edges(
    agent: Flight,
    layout: dict,
    information: dict,
    resources: Dict[str, Resource],
    cell_size: float,
) -> None:
    agent.arr_runway_ret_waypoints = None
    agent.edge_list = []
    agent.edge_segment_endpoints = []
    agent.edge_cursor = 0
    agent.edge_s_along_px = 0.0
    agent.sim_export_arrival_edge_list = []
    agent._path_graph_ref = None

    arr_rwy_res = resources.get(agent.arr_runway_id)
    if not arr_rwy_res or not isinstance(arr_rwy_res, RunwayResource):
        return
    verts = arr_rwy_res._vertices_pixels()
    if agent.arr_runway_dir == "counter_clockwise":
        verts = list(reversed(verts))
    if len(verts) < 2:
        return
    rwy_total_len = _polyline_length(verts)
    td_d = min(float(agent.arr_td_dist_m or 0.0), rwy_total_len)
    ret_d = min(float(agent.arr_ret_dist_m or 0.0), rwy_total_len)
    if ret_d < td_d:
        ret_d = td_d
    origin = (
        float(agent.touchdown_px[0]),
        float(agent.touchdown_px[1]),
    ) if agent.touchdown_px else (agent.col, agent.row)
    runway_wp = _runway_centerline_waypoints_td_to_ret(verts, td_d, ret_d, origin)
    ret_res = resources.get(agent.sampled_arr_ret)
    ret_px: List[Tuple[float, float]] = []
    if ret_res and isinstance(ret_res, TaxiwayResource):
        ret_px = ret_res.vertices_pixels()
    combined = _merge_runway_and_ret_polyline(runway_wp, ret_px)
    agent.arr_runway_ret_waypoints = combined if combined else None
    if not combined:
        return
    phase1_end = combined[-1]

    g_path, idxs = solve_arrival_path_indices(
        layout,
        information,
        str(agent.arr_runway_id or ""),
        str(agent.assigned_stand_id or ""),
        agent.schedule_arr_ret_id,
        agent.sampled_arr_ret,
        str(agent.arr_runway_dir or ""),
    )
    if not g_path or not idxs or len(idxs) < 2:
        return
    agent._path_graph_ref = g_path
    recs = path_indices_to_edge_segments(g_path, idxs)
    edges: List[Edge] = [_record_to_edge(g_path, r) for r in recs]
    spans: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
        motion_span_for_record(g_path, r) for r in recs
    ]
    if spans:
        fx, fy = spans[0][0]
        if math.hypot(phase1_end[0] - fx, phase1_end[1] - fy) > 2.0:
            d0 = math.hypot(phase1_end[0] - fx, phase1_end[1] - fy)
            lid = str(agent.sampled_arr_ret or "").strip() or "_ret_"
            syn = Edge(
                from_id="_phase1_end",
                to_id=edges[0].from_id,
                cost=d0,
                link_id=lid,
                direction="both",
                path_type="runway_exit",
            )
            edges.insert(0, syn)
            spans.insert(0, (phase1_end, (fx, fy)))
    agent.edge_list = edges
    agent.edge_segment_endpoints = spans
    agent.sim_export_arrival_edge_list = list(edges)


def build_agents(
    flights: list,
    layout: dict,
    resources: Dict[str, Resource],
    information: dict,
) -> List[Flight]:
    agents: list[Flight] = []
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    wingspan = _safe_float(sim_cfg.get("aircraftWingspanM"), float("nan"))
    fuselage = _safe_float(sim_cfg.get("aircraftFuselageLengthM"), float("nan"))
    if not math.isfinite(wingspan) or wingspan <= 0:
        raise ValueError("Information.json tiers.algorithm.simulation.aircraftWingspanM must be > 0")
    if not math.isfinite(fuselage) or fuselage <= 0:
        raise ValueError("Information.json tiers.algorithm.simulation.aircraftFuselageLengthM must be > 0")
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    for idx, raw in enumerate(flights):
        if not isinstance(raw, dict):
            continue
        f = _normalize_flight_dict_for_agent(raw, information)
        if f.get("noWayArr") or f.get("noWayDep") or f.get("arrRetFailed"):
            continue
        fid = f.get("id", "")
        _validate_required_sim_flight_fields(f, f"Flight[{idx}] id={fid!r}")
        _validate_flight_sim_metadata(f, f"Flight[{idx}] id={fid!r}")
        dwell_sec = _safe_float(f.get("dwellMin"), float("nan")) * 60.0
        if not math.isfinite(dwell_sec) or dwell_sec <= 0:
            raise ValueError(f"Flight[{idx}] id={fid!r}: dwellMin must be a finite number > 0")
        token = f.get("token", {}) or {}
        arr_rwy = f.get("arrRunwayIdUsed") or token.get("arrRunwayId") or token.get("runwayId")
        dep_rwy = token.get("depRunwayId") or arr_rwy
        stand_id = f.get("standId") or token.get("apronId")
        sched_ret = f.get("scheduleArrRetId")
        if isinstance(sched_ret, str):
            sched_ret = sched_ret.strip() or None
        arr_ret_id = sched_ret or f.get("sampledArrRet") or token.get("ExitTaxiwayId")
        arr_decel_val = _safe_float(f.get("arrDecelMs2"), float("nan"))
        vret_in = _safe_float(f.get("arrVRetInMs"), float("nan"))
        vret_out = _safe_float(f.get("arrVRetOutMs"), float("nan"))
        if not math.isfinite(vret_in) or vret_in <= 0:
            raise ValueError(f"Flight[{idx}] id={fid!r}: arrVRetInMs invalid")
        if not math.isfinite(vret_out) or vret_out <= 0:
            raise ValueError(f"Flight[{idx}] id={fid!r}: arrVRetOutMs invalid")

        def _opt_flight_str(key: str) -> Optional[str]:
            v = f.get(key)
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        reg = f.get("reg")
        reg_s = str(reg).strip() if reg is not None and str(reg).strip() else None
        agent = Flight(
            id=fid,
            reg=reg_s,
            flight_number=_opt_flight_str("flightNumber"),
            aircraft_type=_opt_flight_str("aircraftType"),
            icao_category=str(f.get("code", "")).strip(),
            dwell_sec=dwell_sec,
            arr_runway_id=arr_rwy,
            dep_runway_id=dep_rwy,
            assigned_stand_id=stand_id,
            sampled_arr_ret=arr_ret_id,
            schedule_arr_ret_id=sched_ret,
            arr_runway_dir=str(f.get("arrRunwayDirUsed", "")).strip(),
            dep_runway_dir=str(f.get("depRunwayDirUsed", "")).strip(),
            arr_v_td_ms=_safe_float(f.get("arrVTdMs"), float("nan")),
            arr_v_ret_in_ms=vret_in,
            arr_v_ret_out_ms=vret_out,
            arr_td_dist_m=_safe_float(f.get("arrTdDistM"), float("nan")),
            arr_ret_dist_m=_safe_float(f.get("arrRetDistM"), float("nan")),
            arr_decel_ms2=float(arr_decel_val),
        )
        _compute_arrival_phase1_and_edges(agent, layout, information, resources, cell_size)
        agent.schedule = ScheduleTimes(
            sldt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sldtMin_d"))),
            sibt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sibtMin_d"))),
            sobt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sobtMin_d"))),
            stot_sec=_sim_sec_optional(_minutes_to_sec(f.get("stotMin_d"))),
        )
        agents.append(agent)
    return agents


def _handle_post_event(
    agent: Flight,
    event: Event,
    result: str,
    layout: dict,
    information: dict,
    resources: Dict[str, Resource],
    event_queue: EventQueue,
    cell_size: float,
    sim_dt: int = 1,
) -> None:
    if result != "ACCEPT":
        return
    if event.event_type == "TOUCHDOWN" and agent.state == "ARR_RUNWAY":
        agent.schedule.aldt_sec = event.time
        if agent.touchdown_px is not None:
            agent.col = float(agent.touchdown_px[0])
            agent.row = float(agent.touchdown_px[1])
        rwy_ret_ready = False
        pre = agent.arr_runway_ret_waypoints
        if pre and len(pre) >= 1:
            agent.path_queue = list(pre)
            agent.speed_ms = float(agent.arr_v_td_ms or 0.0)
            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = 0.0
            agent.pending_ret_enter = True
            rwy_ret_ready = True
        arr_rwy_res = resources.get(agent.arr_runway_id)
        if not rwy_ret_ready and arr_rwy_res and isinstance(arr_rwy_res, RunwayResource):
            verts = arr_rwy_res._vertices_pixels()
            if agent.arr_runway_dir == "counter_clockwise":
                verts = list(reversed(verts))
            if verts and len(verts) >= 2:
                rwy_total_len = _polyline_length(verts)
                td_d = min(float(agent.arr_td_dist_m or 0.0), rwy_total_len)
                ret_d = min(float(agent.arr_ret_dist_m or 0.0), rwy_total_len)
                if ret_d < td_d:
                    ret_d = td_d
                origin = (
                    float(agent.touchdown_px[0]),
                    float(agent.touchdown_px[1]),
                ) if agent.touchdown_px is not None else (agent.col, agent.row)
                runway_wp = _runway_centerline_waypoints_td_to_ret(verts, td_d, ret_d, origin)
                ret_res = resources.get(agent.sampled_arr_ret)
                ret_px: list = []
                if ret_res and isinstance(ret_res, TaxiwayResource):
                    ret_px = ret_res.vertices_pixels()
                combined_path = _merge_runway_and_ret_polyline(runway_wp, ret_px)
                agent.path_queue = combined_path
                agent.speed_ms = float(agent.arr_v_td_ms or 0.0)
                agent.decel_rate = 0.0
                agent.accel_rate = 0.0
                agent.target_speed_ms = 0.0
                agent.pending_ret_enter = True
                rwy_ret_ready = True
        if not rwy_ret_ready:
            agent.pending_ret_enter = False
            ret_res_fb = resources.get(agent.sampled_arr_ret) or event.resource
            event_queue.push(
                Event(
                    time=_sim_sec(event.time + sim_dt),
                    event_type="RET_ENTER",
                    agent=agent,
                    resource=ret_res_fb,
                )
            )
    elif event.event_type == "RET_ENTER" and agent.state == "RET":
        ret_res = resources.get(agent.sampled_arr_ret) or event.resource
        agent.path_queue = []
        agent.decel_rate = 0.0
        agent.accel_rate = 0.0
        event_queue.push(
            Event(
                time=_sim_sec(event.time + sim_dt),
                event_type="TAXI_START",
                agent=agent,
                resource=ret_res,
            )
        )
    elif event.event_type == "TAXI_START" and agent.state == "TAXI_IN":
        stand_res = resources.get(agent.assigned_stand_id)
        if stand_res and isinstance(stand_res, StandResource):
            el = agent.edge_list
            esp = agent.edge_segment_endpoints
            if el and esp and len(el) == len(esp) and len(el) > 0:
                agent.path_queue = []
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                _snap_agent_onto_current_edge(agent)
                agent.speed_ms = _speed_ms_for_edge(resources, el[agent.edge_cursor])
                taxi_time = _taxi_remaining_sec_from_edge_cursor(agent, resources)
                if taxi_time <= 0:
                    taxi_time = 60.0
                agent.sim_export_arrival_edge_list = list(agent.edge_list)
            else:
                agent.edge_list = []
                agent.edge_segment_endpoints = []
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                origin = (agent.col, agent.row)
                dest = (stand_res.col, stand_res.row)
                path = _plan_taxi_path_fallback(
                    layout, information, origin, dest, str(agent.arr_runway_dir or "clockwise")
                )
                agent.path_queue = path
                taxi_speed = DEFAULT_TAXI_SPEED_MS
                agent.speed_ms = taxi_speed
                dist_m = (
                    sum(
                        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                        for i in range(len(path) - 1)
                    )
                    if len(path) > 1
                    else 0
                )
                taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0
            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms
            event_queue.push(
                Event(
                    time=event.time + taxi_time,
                    event_type="STAND_ENTER",
                    agent=agent,
                    resource=stand_res,
                )
            )
        else:
            agent.state = "PARKED"
            agent.speed_ms = 0.0
            agent.schedule.aibt_sec = event.time
            agent.schedule.aobt_sec = _sim_sec(event.time + float(agent.dwell_sec or 0.0))
    elif event.event_type == "PUSHBACK" and agent.state == "TAXI_OUT":
        dep_rwy_res = resources.get(agent.dep_runway_id)
        if dep_rwy_res and isinstance(dep_rwy_res, RunwayResource):
            verts = dep_rwy_res._vertices_pixels()
            if agent.dep_runway_dir == "counter_clockwise":
                verts = list(reversed(verts))
            origin = (agent.col, agent.row)
            dest = verts[0] if verts else origin
            dep_dir = str(agent.dep_runway_dir or "clockwise")
            recs, g = plan_taxi_route(layout, information, origin, dest, dep_dir)
            if recs and g and len(recs) > 0:
                agent.path_queue = []
                agent.edge_list = [_record_to_edge(g, r) for r in recs]
                agent.edge_segment_endpoints = [motion_span_for_record(g, r) for r in recs]
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                _snap_agent_onto_current_edge(agent)
                agent.speed_ms = _speed_ms_for_edge(resources, agent.edge_list[agent.edge_cursor])
                taxi_time = _taxi_remaining_sec_from_edge_cursor(agent, resources)
                if taxi_time <= 0:
                    taxi_time = 60.0
            else:
                agent.edge_list = []
                agent.edge_segment_endpoints = []
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                path = _plan_taxi_path_fallback(layout, information, origin, dest, dep_dir)
                agent.path_queue = path
                dist_m = (
                    sum(
                        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                        for i in range(len(path) - 1)
                    )
                    if len(path) > 1
                    else 0
                )
                taxi_speed = DEFAULT_TAXI_SPEED_MS
                agent.speed_ms = taxi_speed
                taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0
            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms
            event_queue.push(
                Event(time=event.time + taxi_time, event_type="LINEUP", agent=agent, resource=dep_rwy_res)
            )
        else:
            agent.state = "DEPARTED"
            agent.speed_ms = 0.0
            agent.schedule.atot_sec = event.time
    elif event.event_type == "LINEUP" and agent.state == "LINEUP_HOLD":
        agent.speed_ms = 0.0
        agent.path_queue = []
        agent.edge_list = []
        agent.edge_segment_endpoints = []
        agent.edge_cursor = 0
        agent.edge_s_along_px = 0.0
        agent.decel_rate = 0.0
        agent.accel_rate = 0.0
        event_queue.push(
            Event(
                time=event.time + DEP_LINEUP_HOLD_SEC,
                event_type="TAKEOFF_REQUEST",
                agent=agent,
                resource=event.resource,
            )
        )
    elif event.event_type == "DEPARTED":
        agent.state = "DEPARTED"
        agent.speed_ms = 0.0
        agent.decel_rate = 0.0
        agent.accel_rate = 0.0
        agent.path_queue = []
        agent.edge_list = []
        agent.edge_segment_endpoints = []
        agent.edge_cursor = 0
        agent.edge_s_along_px = 0.0


@dataclass
class TimeRange:
    start: int
    end: int


def compute_time_range(agents: List[Flight]) -> TimeRange:
    times: list[float] = []
    for a in agents:
        s = a.schedule
        for t in (s.sldt_sec, s.sibt_sec, s.sobt_sec, s.stot_sec):
            if t is not None:
                times.append(float(t))
    if not times:
        return TimeRange(0, 0)
    lo = min(times) - 300.0
    hi = max(times) + 7200.0
    return TimeRange(int(math.floor(lo)), int(math.ceil(hi)))


def _build_export_graph_for_junctions(layout: dict, information: dict) -> PathGraph:
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    algo = information.get("tiers", {}).get("algorithm", {}) if isinstance(information, dict) else {}
    path_cfg = algo.get("pathSearch", {}) if isinstance(algo, dict) else {}
    reverse_cost = float(path_cfg.get("reverseCost", 1_000_000) or 1_000_000)
    th = path_cfg.get("taxiwayHeuristicCost")
    if th is not None and math.isfinite(float(th)) and float(th) == 0.0:
        taxiway_h = 0.0
    elif th is not None and float(th) > 0:
        taxiway_h = float(th)
    else:
        taxiway_h = 200.0
    merge_r = float(path_cfg.get("junctionMergeRadiusPx", 7.0) or 7.0)
    flight_sched = information.get("tiers", {}).get("flight_schedule", {}) if isinstance(information, dict) else {}
    rw_exit_default = normalize_allowed_runway_directions(flight_sched.get("rwExitAllowedDefaultRaw"))
    direction_modes = layout.get("directionModes") or []
    if not isinstance(direction_modes, list):
        direction_modes = []
    g_pre = path_graph_from_layout_sim_export(
        layout,
        "clockwise",
        pure_ground_exclude_runway=False,
        reverse_cost=reverse_cost,
        merge_radius_px=merge_r,
        taxiway_heuristic_bonus=0.0,
        apply_taxiway_ret_heuristic=False,
    )
    if g_pre is not None:
        return g_pre
    return build_path_graph(
        layout,
        cell_size,
        reverse_cost,
        taxiway_h,
        merge_r,
        rw_exit_default,
        direction_modes,
        None,
        "clockwise",
        None,
    )


def _sec_to_datetime_str(sec, base_date: str) -> Optional[str]:
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


def _edge_to_sim_result_dict(e: Edge) -> dict:
    return {
        "from_id": e.from_id,
        "to_id": e.to_id,
        "cost": round(float(e.cost), 4),
        "link_id": e.link_id,
        "direction": e.direction,
        "path_type": e.path_type,
    }


def _build_output(
    agents: List[Flight],
    base_date: str,
    network_junctions_xy: Optional[List[dict]] = None,
) -> dict:
    positions: Dict[str, list] = {}
    schedule_list: list = []
    taxi_in_times: list[float] = []
    taxi_out_times: list[float] = []
    total_delay: float = 0.0
    for a in agents:
        positions[a.id] = [
            {"t": int(t), "x": round(c, 3), "y": round(r, 3), "v": round(v, 3)}
            for t, c, r, v in a.history
        ]
        sched_entry = {
            "flight_id": a.id,
            "reg": a.reg,
            "flight_number": a.flight_number,
            "aircraft_type": a.aircraft_type,
            "SLDT": a.schedule.sldt_sec,
            "SLDT_dt": _sec_to_datetime_str(a.schedule.sldt_sec, base_date),
            "SIBT": a.schedule.sibt_sec,
            "SIBT_dt": _sec_to_datetime_str(a.schedule.sibt_sec, base_date),
            "SOBT": a.schedule.sobt_sec,
            "SOBT_dt": _sec_to_datetime_str(a.schedule.sobt_sec, base_date),
            "STOT": a.schedule.stot_sec,
            "STOT_dt": _sec_to_datetime_str(a.schedule.stot_sec, base_date),
            "ELDT": a.schedule.aldt_sec,
            "ELDT_dt": _sec_to_datetime_str(a.schedule.aldt_sec, base_date),
            "EXIT_RUNWAY": a.schedule.e_rw_exit_sec,
            "EXIT_RUNWAY_dt": _sec_to_datetime_str(a.schedule.e_rw_exit_sec, base_date),
            "ARR_ROT_SEC": a.arr_rot_sec,
            "EIBT": a.schedule.aibt_sec,
            "EIBT_dt": _sec_to_datetime_str(a.schedule.aibt_sec, base_date),
            "EOBT": a.schedule.aobt_sec,
            "EOBT_dt": _sec_to_datetime_str(a.schedule.aobt_sec, base_date),
            "ETOT": a.schedule.atot_sec,
            "ETOT_dt": _sec_to_datetime_str(a.schedule.atot_sec, base_date),
            "edge_list": [_edge_to_sim_result_dict(e) for e in (a.sim_export_arrival_edge_list or [])],
        }
        schedule_list.append(sched_entry)
        if a.schedule.aldt_sec is not None and a.schedule.aibt_sec is not None:
            taxi_in_times.append(a.schedule.aibt_sec - a.schedule.aldt_sec)
        if a.schedule.aobt_sec is not None and a.schedule.atot_sec is not None:
            taxi_out_times.append(a.schedule.atot_sec - a.schedule.aobt_sec)
        if a.schedule.aldt_sec is not None and a.schedule.sldt_sec is not None:
            delay = float(a.schedule.aldt_sec) - float(a.schedule.sldt_sec)
            if delay > 0:
                total_delay += delay
    kpi_data = {
        "total_flights": len(agents),
        "avg_taxi_in_sec": round(float(np.mean(taxi_in_times)), 1) if taxi_in_times else None,
        "avg_taxi_out_sec": round(float(np.mean(taxi_out_times)), 1) if taxi_out_times else None,
        "total_delay_sec": round(total_delay, 1),
        "flights_completed": sum(1 for a in agents if a.state == "DEPARTED"),
        "flights_parked": sum(1 for a in agents if a.state == "PARKED"),
    }
    return {
        "baseDate": base_date,
        "networkJunctions": list(network_junctions_xy or []),
        "positions": positions,
        "schedule": schedule_list,
        "kpi": kpi_data,
    }


def run_simulation(
    layout: dict,
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> dict:
    global _SIM_CELL_SIZE, APPROACH_OFFSET_PX, DEP_LINEUP_HOLD_SEC, DEP_TAKEOFF_ACCEL_MS2, _SIM_STEP_SEC

    information = _load_information_json()
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    dt = max(1, int(round(float(sim_cfg.get("timeStepSec", dt)))))
    cell_size = layout.get("grid", {}).get("cellSize", 20.0)
    _SIM_CELL_SIZE = cell_size
    _SIM_STEP_SEC = int(dt)
    APPROACH_OFFSET_PX = _safe_float(sim_cfg.get("approachOffsetM"), 10_000.0)
    flight_cfg = _deep_get(information, "tiers", "flight_schedule", default={})
    DEP_LINEUP_HOLD_SEC = _safe_float(flight_cfg.get("depLineupHoldSec"), 20.0)
    DEP_TAKEOFF_ACCEL_MS2 = _safe_float(flight_cfg.get("depTakeoffAccelSmallMs2"), 2.5)
    base_date = str(
        _deep_get(information, "tiers", "algorithm", "simulation", "baseDate", default="2026-03-31")
    )

    g_export = _build_export_graph_for_junctions(layout, information)
    poly_net_j = polyline_apron_junctions_xy_for_sim_result(g_export)
    resources = build_resources(layout, information)
    flights_raw = layout.get("flights", [])

    if not flights_raw:
        return _build_output([], base_date, poly_net_j)

    agents = build_agents(flights_raw, layout, resources, information)
    if not agents:
        return _build_output([], base_date, poly_net_j)

    event_queue = EventQueue()
    for agent in agents:
        ev = agent.create_initial_event(resources)
        if ev:
            event_queue.push(ev)

    time_range = compute_time_range(agents)
    current_time = time_range.start
    total_end = time_range.end
    record_interval = dt
    last_record_time = current_time - record_interval

    while current_time <= total_end:
        events = event_queue.pop(current_time)
        if events:
            for event in events:
                agent = event.agent
                resource = event.resource
                agent.update_state(event)
                result = resource.request(agent, event)
                if result == "ACCEPT":
                    new_events = agent.on_accepted(resource, event)
                elif result == "REJECT":
                    new_events = agent.on_rejected(resource, event)
                elif result == "WAIT":
                    new_events = agent.on_wait(resource, event)
                elif result == "REROUTE":
                    new_events = agent.on_reroute(resource, event)
                elif result == "HOLD":
                    new_events = agent.on_hold(resource, event)
                elif result == "SPEED_ADJUST":
                    new_events = agent.on_speed_adjust(resource, event)
                else:
                    new_events = agent.on_default(resource, event)
                event_queue.push(new_events)
                _handle_post_event(
                    agent, event, result, layout, information, resources, event_queue, cell_size, dt
                )
        for agent in agents:
            if not agent.is_stationary():
                _move_agent_along_path(agent, float(dt), cell_size, resources)
        for agent in agents:
            if agent.state == "ARR_RUNWAY" and agent.pending_ret_enter and not agent.path_queue:
                agent.pending_ret_enter = False
                ret_r = resources.get(agent.sampled_arr_ret) or resources.get(agent.arr_runway_id)
                if ret_r is None:
                    continue
                event_queue.push(
                    Event(
                        time=_sim_sec(current_time + dt),
                        event_type="RET_ENTER",
                        agent=agent,
                        resource=ret_r,
                    )
                )
        for agent in agents:
            _try_record_runway_perpendicular_exit(agent, current_time, resources)
        if current_time - last_record_time >= record_interval:
            for agent in agents:
                agent.record_position(current_time)
            last_record_time = current_time
        if progress_cb:
            elapsed = current_time - time_range.start
            total_span = total_end - time_range.start
            progress_cb(elapsed, total_span)
        current_time += dt
        if event_queue.empty and all(a.is_stationary() for a in agents):
            for agent in agents:
                agent.record_position(current_time)
            break

    return _build_output(agents, base_date, poly_net_j)
