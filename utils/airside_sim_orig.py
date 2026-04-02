"""
Airside Simulation Engine — Agent-Based + Discrete Event Simulation.

Reads a serialized layout (from Layout_Design) and Information.json,
runs a time-stepped DES loop where each Flight is an Agent interacting
with shared Resources (Runway, Taxiway, Stand, etc.), and returns (flight-only;
layout/terminals/aprons stay client-side):
- per-flight position timeline  (t, x, y layout pixels; v = |Δposition|/Δt over each sim step, same units as layout px/s)
- E-schedule (JSON)             (ELDT, EIBT, EOBT, ETOT per flight; internal attrs remain aldt_sec, …)
- SLDT in layout input          (scheduled touchdown; ARRIVAL_REQUEST is issued approach_dur earlier)
- KPI aggregation               (utilization, delay, throughput …)

Layout geometry uses pixel coordinates: vertex objects prefer {"x","y"}; legacy
{"col","row"} are grid cells and are converted with grid.cellSize. Graph Junction ids
are sequential J0001… (not coordinate strings). Nodes merge within junctionMergeRadiusPx;
edge costs use Euclidean distance in pixels.

Token-based path resolution:
Each flight carries a token {arrRunwayId, apronId, depRunwayId} plus
token.ExitTaxiwayId (RET) / sampledArrRet (legacy) and physics params (arrVTdMs, arrDecelMs2, arrRunwayDirUsed, …).
Phase 1 (runway + RET polyline) and Phase 2 (reroute on taxi-type Edges only) are precomputed in build_agents when possible.
TOUCHDOWN consumes the Phase 1 waypoint list (arr_runway_ret_waypoints). After reroute, each flight holds taxi edge_list (graph
Edge objects in order) plus parallel segment endpoints in layout px; TAXI_IN / TAXI_OUT advance along those segments only, using
per-link avg_velocity_ms from resources. Layout may include networkJunctions (same points as the designer green Junctions) so graph
Junction ids align with the UI.
The aircraft stays in ARR_RUNWAY and follows Phase 1 with arrDecelMs2 (MIN_ARR_RUNWAY_VELOCITY_MS floor). RET_ENTER is only a
discrete transition to TAXI_START after the path is exhausted.
ARR_ROT_SEC / EXIT_RUNWAY (sim sec) are outputs: time from touchdown until perpendicular distance from aircraft center to arr runway centerline ≥ EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M.
The DES uses these to plan multi-phase movement:
APPROACH → TOUCHDOWN (ARR_RUNWAY: prebuilt runway+RET path, decel) → RET_ENTER → TAXI_START → STAND_ENTER →
PUSHBACK → LINEUP → TAKEOFF_REQUEST → DEPARTED
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

_logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


def _arrival_path_debug_print(agent_id: str, msg: str) -> None:
    """Stdout debug for empty sim_result edge_list. Default: on. Silence: AIRSIDE_DEBUG_ARRIVAL_PATH=0."""
    v = os.environ.get("AIRSIDE_DEBUG_ARRIVAL_PATH", "1").strip().lower()
    if v in ("0", "false", "no", "n", "off"):
        return
    print(f"[airside_sim._compute_arrival_paths_for_agent][{agent_id}] {msg}", flush=True)
_INFO_FILE = _ROOT / "data" / "Info_storage" / "Information.json"


APPROACH_OFFSET_PX: float = 10_000.0
DEP_LINEUP_HOLD_SEC: float = 20.0
DEP_TAKEOFF_ACCEL_MS2: float = 2.5
DEFAULT_TAXI_SPEED_MS: float = 10.0
MIN_ARR_RUNWAY_VELOCITY_MS: float = 15.0
EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M: float = 150.0
_SIM_STEP_SEC: int = 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    """Map a layout vertex to pixel coordinates. Prefer x,y; legacy col,row are grid cells × cellSize."""
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
    """Snap schedule times to integer seconds (simulation time axis is discrete)."""
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
    """Finite time → integer simulation second."""
    try:
        v = float(sec)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(v):
        return 0
    return int(round(v))


# ---------------------------------------------------------------------------
# polyline helpers
# ---------------------------------------------------------------------------

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


def _subdivide_polyline(pts: List[Tuple[float, float]], d_start: float, d_end: float,
                        num_sub: int = 10) -> List[Tuple[float, float]]:
    """Extract sub-polyline from distance d_start to d_end along pts."""
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
    """Waypoints along runway centerline from touchdown (td_d) to RET junction (ret_d), skipping near-duplicate of origin."""
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


def _format_sequential_junction_id(seq: int) -> str:
    """J0001…J9999 then J10000… (layout graph nodes; not coordinate-derived)."""
    if seq <= 0:
        raise ValueError("junction sequence must be positive")
    if seq <= 9999:
        return f"J{seq:04d}"
    return f"J{seq}"


def _ensure_junction_with_merge(
    graph: AirsideGraph,
    col: float,
    row: float,
    merge_radius_px: float,
) -> str:
    """Reuse an existing Junction within merge_radius_px (layout px); else allocate J0001…"""
    r = max(0.0, float(merge_radius_px))
    if r > 0 and graph.junctions:
        best_id: Optional[str] = None
        best_d = r + 1.0
        for nid, n in graph.junctions.items():
            d = math.hypot(float(n.col) - float(col), float(n.row) - float(row))
            if d <= r + 1e-9 and d < best_d:
                best_d = d
                best_id = nid
        if best_id is not None:
            return best_id
    seq = graph._next_junction_seq
    graph._next_junction_seq = seq + 1
    jid = _format_sequential_junction_id(seq)
    graph.add_junction(Junction(id=jid, col=float(col), row=float(row)))
    return jid


def _merge_runway_and_ret_polyline(
    runway_wp: List[Tuple[float, float]],
    ret_verts: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Concatenate runway waypoints and RET polyline; skip duplicate junction if first RET ≈ last runway."""
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


# ---------------------------------------------------------------------------
# data models
# ---------------------------------------------------------------------------

@dataclass
class ScheduleTimes:
    sldt_sec: Optional[float] = None
    sibt_sec: Optional[float] = None
    sobt_sec: Optional[float] = None
    stot_sec: Optional[float] = None
    aldt_sec: Optional[float] = None
    # Sim time when perpendicular distance from aircraft center to arr runway centerline ≥ EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M.
    e_rw_exit_sec: Optional[float] = None
    aibt_sec: Optional[float] = None
    aobt_sec: Optional[float] = None
    atot_sec: Optional[float] = None


@dataclass
class Flight:
    id: str
    reg: Optional[str] = None
    flight_number: Optional[str] = None
    aircraft_type: Optional[str] = None
    icao_category: Optional[str] = None
    col: float = 0.0  # layout X in pixels (same unit as serialized path vertices)
    row: float = 0.0  # layout Y in pixels
    # Physics integration (decel/accel targets); not used as the exported trail speed when velocity_ms is set from motion.
    speed_ms: float = 0.0
    # Trail speed (layout px/s): last value written in record_position = |Δpos|/Δt vs previous history sample (matches output x,y).
    velocity_ms: float = 0.0
    state: str = "SCHEDULED"
    history: list = field(default_factory=list)
    schedule: ScheduleTimes = field(default_factory=ScheduleTimes)
    assigned_stand_id: Optional[str] = None
    arr_runway_id: Optional[str] = None
    dep_runway_id: Optional[str] = None
    path_queue: list = field(default_factory=list)
    dwell_sec: Optional[float] = None
    # Output only: seconds from touchdown (ELDT) until perpendicular distance from aircraft center to arr runway centerline ≥ EXIT_RUNWAY_MIN_PERPENDICULAR_DISTANCE_FROM_CENTERLINE_M.
    arr_rot_sec: Optional[float] = None

    # --- token-derived waypoint IDs ---
    sampled_arr_ret: Optional[str] = None
    arr_runway_dir: Optional[str] = None
    dep_runway_dir: Optional[str] = None

    # --- physics params from serialized data (set in build_agents from sim_input / Information.json) ---
    arr_v_td_ms: Optional[float] = None
    arr_v_ret_in_ms: Optional[float] = None
    arr_v_ret_out_ms: Optional[float] = None
    # Distance along runway centerline polyline to touchdown point (layout pixels; naming legacy).
    arr_td_dist_m: Optional[float] = None
    # Distance along runway centerline to RET junction (layout pixels; naming legacy).
    arr_ret_dist_m: Optional[float] = None
    # Deceleration on arrival runway segment (m/s²); required on sim flights (validated in build_agents).
    arr_decel_ms2: Optional[float] = None

    # Runway touchdown point in layout pixels; set on ARRIVAL accept, applied on TOUCHDOWN event.
    touchdown_px: Optional[Tuple[float, float]] = None

    # --- dynamic speed control ---
    target_speed_ms: float = 0.0
    decel_rate: float = 0.0
    accel_rate: float = 0.0

    _departed_recorded: bool = field(default=False, repr=False)
    # After TOUCHDOWN with a built path: fire RET_ENTER once path_queue is empty.
    pending_ret_enter: bool = field(default=False, repr=False)
    # Phase 1 (init): runway centerline td→RET + full RET polyline (layout px).
    arr_runway_ret_waypoints: Optional[List[Tuple[float, float]]] = None
    # Phase 2 (reroute): ordered graph Edges + matching segment (p0→p1) in layout px; sole authority for taxi-in motion.
    edge_list: List["Edge"] = field(default_factory=list)
    edge_segment_endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    edge_cursor: int = 0
    edge_s_along_px: float = 0.0
    # Copy of taxi-in edge_list for sim_result.json (survives STAND_ENTER clearing edge_list).
    sim_export_arrival_edge_list: List["Edge"] = field(default_factory=list)

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

    # --- agent reaction methods -------------------------------------------

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
        # SLDT = scheduled touchdown; ARRIVAL_REQUEST fires (approach_dur) earlier so on-time ELDT ≈ SLDT.
        if self.schedule.sldt_sec is not None and isinstance(rwy_res, RunwayResource):
            plan = _plan_arrival_approach_leg(rwy_res, self)
            if plan is not None:
                approach_dur, _, _, _ = plan
                t = float(self.schedule.sldt_sec) - approach_dur
        return Event(time=t, event_type="ARRIVAL_REQUEST", agent=self, resource=rwy_res)


# ---------------------------------------------------------------------------
# Event & EventQueue
# ---------------------------------------------------------------------------

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

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0


# ---------------------------------------------------------------------------
# Resource base & subclasses
# ---------------------------------------------------------------------------

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
    """Full modeled approach: duration (s), approach_start px, touchdown px, ordered runway verts. None if degenerate."""
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

                    events.append(Event(
                        time=event.time + approach_dur,
                        event_type="TOUCHDOWN",
                        agent=agent,
                        resource=self,
                    ))
                else:
                    verts_fb = self._vertices_pixels()
                    if verts_fb:
                        agent.col, agent.row = verts_fb[0]
                        agent.touchdown_px = (float(verts_fb[0][0]), float(verts_fb[0][1]))
                    else:
                        agent.touchdown_px = None
                    # No modeled approach: touchdown = runway sequence start (= SLDT when request at SLDT).
                    agent.schedule.aldt_sec = event.time
                    events.append(Event(
                        time=_sim_sec(event.time + _SIM_STEP_SEC),
                        event_type="TAXI_START",
                        agent=agent,
                        resource=self,
                    ))

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

                    events.append(Event(
                        time=event.time + roll_dur,
                        event_type="DEPARTED",
                        agent=agent,
                        resource=self,
                    ))
                else:
                    events.append(Event(
                        time=event.time + 30.0,
                        event_type="DEPARTED",
                        agent=agent,
                        resource=self,
                    ))

        elif result in ("WAIT", "HOLD"):
            retry_delay = 5.0
            events.append(Event(
                time=event.time + retry_delay,
                event_type=event.event_type,
                agent=agent,
                resource=event.resource,
                priority=event.priority + 1,
            ))
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
            return [Event(
                time=event.time + 3.0,
                event_type=event.event_type,
                agent=agent,
                resource=self,
                priority=event.priority + 1,
            )]
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
                events.append(Event(
                    time=pushback_time,
                    event_type="PUSHBACK",
                    agent=agent,
                    resource=self,
                ))
            elif event.event_type == "PUSHBACK":
                self.occupant = None
        elif result in ("WAIT", "HOLD"):
            events.append(Event(
                time=event.time + 5.0,
                event_type=event.event_type,
                agent=agent,
                resource=self,
                priority=event.priority + 1,
            ))
        return events


def _try_record_runway_perpendicular_exit(
    agent: Flight,
    current_time: int,
    resources: Dict[str, Resource],
) -> None:
    """Set schedule.e_rw_exit_sec and arr_rot_sec when aircraft is far enough from arr runway centerline."""
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


# ---------------------------------------------------------------------------
# Junction / Edge graph (layout polylines → directed edges; sim_input networkJunctions)
# ---------------------------------------------------------------------------

REROUTE_ALLOWED_PATH_TYPES: frozenset = frozenset({
    "taxiway", "runway_taxiway", "apron_taxiway", "apron_link", "runway_exit",
})


@dataclass
class Junction:
    """Graph node: a connection point in layout pixel space."""
    id: str
    col: float
    row: float


@dataclass
class Edge:
    """Directed link between Junctions (one polyline segment of a layout object)."""
    from_id: str
    to_id: str
    cost: float
    link_id: str
    direction: str = "both"
    path_type: str = "taxiway"


class AirsideGraph:
    def __init__(self) -> None:
        self.junctions: Dict[str, Junction] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self._path_cache: Dict[Tuple[str, str, str], Optional[List[str]]] = {}
        # Junction ids created from runway/runwayTaxiway/taxiway polylines + apronLinks only (not layout.networkJunctions).
        self.polyline_apron_junction_ids: set[str] = set()
        self._next_junction_seq: int = 1
        # Set in build_graph; used by _resolve_junction_at_xy for the same merge rule as ensure_junction.
        self.junction_merge_radius_px: float = 0.0

    def add_junction(self, j: Junction) -> None:
        self.junctions[j.id] = j
        if j.id not in self.edges:
            self.edges[j.id] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.from_id not in self.edges:
            self.edges[edge.from_id] = []
        self.edges[edge.from_id].append(edge)

    def _cache_key(self, start_id: str, end_id: str, filt: str) -> Tuple[str, str, str]:
        return (start_id, end_id, filt)

    def dijkstra(
        self,
        start_id: str,
        end_id: str,
        edge_allowed: Optional[Callable[[Edge], bool]] = None,
    ) -> Optional[List[str]]:
        filt = "all" if edge_allowed is None else "f"
        cache_key = self._cache_key(start_id, end_id, filt)
        if edge_allowed is None and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if start_id not in self.junctions or end_id not in self.junctions:
            if edge_allowed is None:
                self._path_cache[cache_key] = None
            return None

        dist: Dict[str, float] = {start_id: 0.0}
        prev: Dict[str, Optional[str]] = {start_id: None}
        heap: list[Tuple[float, str]] = [(0.0, start_id)]
        visited: set = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == end_id:
                break
            for edge in self.edges.get(u, []):
                if edge_allowed is not None and not edge_allowed(edge):
                    continue
                v = edge.to_id
                if v in visited:
                    continue
                nd = d + edge.cost
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if end_id not in prev and end_id != start_id:
            if edge_allowed is None:
                self._path_cache[cache_key] = None
            return None

        path: list[str] = []
        cur: Optional[str] = end_id
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        if edge_allowed is None:
            self._path_cache[cache_key] = path
        return path

    def junction_path_xy(self, junction_ids: List[str]) -> List[Tuple[float, float]]:
        coords: List[Tuple[float, float]] = []
        for nid in junction_ids:
            n = self.junctions.get(nid)
            if n:
                coords.append((n.col, n.row))
        return coords


def reroute(
    graph: AirsideGraph,
    start_junction_id: str,
    end_junction_id: str,
    edge_allowed: Optional[Callable[[Edge], bool]] = None,
) -> Optional[List[str]]:
    """Shortest path (Dijkstra) between Junction ids; optional per-Edge filter."""
    return graph.dijkstra(start_junction_id, end_junction_id, edge_allowed=edge_allowed)


def _reroute_arrival_taxi_edge_ok(e: Edge) -> bool:
    return e.path_type in REROUTE_ALLOWED_PATH_TYPES


def build_graph(layout: dict, information: dict) -> AirsideGraph:
    """Build one graph from sim_input: runway / runway taxiway / taxiway / apron links + optional networkJunctions."""
    graph = AirsideGraph()
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    reverse_cost = float(_deep_get(information, "tiers", "algorithm", "pathSearch", "reverseCost", default=1_000_000))
    merge_r_px = float(
        _deep_get(information, "tiers", "algorithm", "pathSearch", "junctionMergeRadiusPx", default=7.0)
    )
    graph.junction_merge_radius_px = merge_r_px

    def ensure_junction(col: float, row: float) -> str:
        return _ensure_junction_with_merge(graph, col, row, merge_r_px)

    def ensure_polyline_apron_junction(col: float, row: float) -> str:
        jid = ensure_junction(col, row)
        graph.polyline_apron_junction_ids.add(jid)
        return jid

    for jspec in layout.get("networkJunctions") or []:
        if not isinstance(jspec, dict):
            continue
        jx = jspec.get("x")
        jy = jspec.get("y")
        if jx is not None and jy is not None:
            ensure_junction(_safe_float(jx), _safe_float(jy))
            continue
        jc = jspec.get("col")
        jr = jspec.get("row")
        if jc is not None and jr is not None:
            cs = max(cell_size, 1e-9)
            ensure_junction(_safe_float(jc) * cs, _safe_float(jr) * cs)

    all_links: list[dict] = []
    for tw in layout.get("runwayPaths", []):
        all_links.append(dict(tw, pathType="runway"))
    for tw in layout.get("runwayTaxiways", []):
        all_links.append(dict(tw, pathType="runway_taxiway"))
    for tw in layout.get("taxiways", []):
        pt = tw.get("pathType", "taxiway")
        all_links.append(dict(tw, pathType=pt))

    for tw in all_links:
        verts = [v for v in tw.get("vertices", []) if isinstance(v, dict)]
        if len(verts) < 2:
            continue
        tw_id = str(tw.get("id", "") or "")
        direction = str(tw.get("direction", "both") or "both")
        path_type = str(tw.get("pathType", "taxiway") or "taxiway")

        node_ids = []
        for v in verts:
            if not isinstance(v, dict):
                continue
            px, py = _vertex_xy(v, cell_size)
            node_ids.append(ensure_polyline_apron_junction(px, py))

        for i in range(len(node_ids) - 1):
            a_id = node_ids[i]
            b_id = node_ids[i + 1]
            a_n = graph.junctions[a_id]
            b_n = graph.junctions[b_id]
            seg_len = math.hypot(b_n.col - a_n.col, b_n.row - a_n.row)
            cost = max(seg_len, 0.01)

            if direction == "clockwise":
                graph.add_edge(Edge(a_id, b_id, cost, tw_id, direction, path_type))
                graph.add_edge(Edge(b_id, a_id, cost + reverse_cost, tw_id, direction, path_type))
            elif direction == "counter_clockwise":
                graph.add_edge(Edge(b_id, a_id, cost, tw_id, direction, path_type))
                graph.add_edge(Edge(a_id, b_id, cost + reverse_cost, tw_id, direction, path_type))
            else:
                graph.add_edge(Edge(a_id, b_id, cost, tw_id, direction, path_type))
                graph.add_edge(Edge(b_id, a_id, cost, tw_id, direction, path_type))

    stand_lookup: Dict[str, dict] = {}
    for st in layout.get("pbbStands", []):
        sid = st.get("id", "")
        if sid:
            stand_lookup[sid] = st
    for st in layout.get("remoteStands", []):
        sid = st.get("id", "")
        if sid:
            stand_lookup[sid] = st

    for lk in layout.get("apronLinks", []):
        pbb_id = lk.get("pbbId", "")
        stand = stand_lookup.get(pbb_id)
        if not stand:
            continue

        stand_col = _safe_float(stand.get("edgeCol"), _safe_float(stand.get("col"), 0))
        stand_row = _safe_float(stand.get("edgeRow"), _safe_float(stand.get("row"), 0))
        cs = max(cell_size, 1e-9)
        stand_px = (stand_col * cs, stand_row * cs)

        mid_verts = lk.get("midVertices", [])
        tx_px = lk.get("tx")
        ty_px = lk.get("ty")

        apron_pts: list[Tuple[float, float]] = [stand_px]

        if isinstance(mid_verts, list) and mid_verts:
            for mv in mid_verts:
                if isinstance(mv, dict):
                    apron_pts.append(_vertex_xy(mv, cell_size))
        elif tx_px is not None and ty_px is not None:
            apron_pts.append((_safe_float(tx_px), _safe_float(ty_px)))

        if len(apron_pts) < 2:
            continue

        lk_id = str(lk.get("id", "apron_link") or "apron_link")
        apron_node_ids = [ensure_polyline_apron_junction(p[0], p[1]) for p in apron_pts]
        for i in range(len(apron_node_ids) - 1):
            a_id = apron_node_ids[i]
            b_id = apron_node_ids[i + 1]
            a_n = graph.junctions[a_id]
            b_n = graph.junctions[b_id]
            seg_len = math.hypot(b_n.col - a_n.col, b_n.row - a_n.row)
            cost = max(seg_len, 0.01)
            graph.add_edge(Edge(a_id, b_id, cost, lk_id, "both", "apron_link"))
            graph.add_edge(Edge(b_id, a_id, cost, lk_id, "both", "apron_link"))

    return graph


def _polyline_apron_junctions_for_sim_result(graph: AirsideGraph) -> List[dict]:
    """Junction points from polyline/apron vertex chains only; layout px as x,y for sim_result."""
    out: list[dict] = []
    for jid in sorted(graph.polyline_apron_junction_ids):
        jn = graph.junctions.get(jid)
        if jn is None:
            continue
        out.append({"x": round(float(jn.col), 3), "y": round(float(jn.row), 3)})
    return out


def _find_closest_junction_id(graph: AirsideGraph, col: float, row: float) -> Optional[str]:
    best_id: Optional[str] = None
    best_dist = math.inf
    for nid, n in graph.junctions.items():
        d = math.hypot(n.col - col, n.row - row)
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


def _resolve_junction_at_xy(graph: AirsideGraph, col: float, row: float) -> Optional[str]:
    """Map a layout px point to an existing Junction: merge-radius match first, else global closest."""
    r = max(0.0, float(getattr(graph, "junction_merge_radius_px", 0.0)))
    if r > 0 and graph.junctions:
        best_id: Optional[str] = None
        best_d = r + 1.0
        for nid, n in graph.junctions.items():
            d = math.hypot(float(n.col) - float(col), float(n.row) - float(row))
            if d <= r + 1e-9 and d < best_d:
                best_d = d
                best_id = nid
        if best_id is not None:
            return best_id
    return _find_closest_junction_id(graph, col, row)


def _stand_junction_id_from_layout(
    layout: dict, stand_id: Optional[str], cell_size: float, graph: AirsideGraph
) -> Optional[str]:
    if not stand_id or not str(stand_id).strip():
        return None
    sid = str(stand_id).strip()
    stand_lookup: Dict[str, dict] = {}
    for st in layout.get("pbbStands", []):
        i = st.get("id", "")
        if i:
            stand_lookup[str(i)] = st
    for st in layout.get("remoteStands", []):
        i = st.get("id", "")
        if i:
            stand_lookup[str(i)] = st
    for lk in layout.get("apronLinks", []):
        if str(lk.get("pbbId", "")).strip() != sid:
            continue
        stand = stand_lookup.get(sid)
        if not stand:
            return None
        stand_col = _safe_float(stand.get("edgeCol"), _safe_float(stand.get("col"), 0))
        stand_row = _safe_float(stand.get("edgeRow"), _safe_float(stand.get("row"), 0))
        cs = max(float(cell_size), 1e-9)
        sx, sy = stand_col * cs, stand_row * cs
        return _resolve_junction_at_xy(graph, sx, sy)
    return None


def _directed_edge_between(graph: AirsideGraph, fr_id: str, to_id: str) -> Optional[Edge]:
    for e in graph.edges.get(fr_id, []):
        if e.to_id == to_id:
            return e
    return None


_SYNTH_EDGE_FROM = "_synthetic_a_"
_SYNTH_EDGE_TO = "_synthetic_b_"


def _synthetic_connector_edge(link_id: str, path_type: str, seg_len_px: float) -> Edge:
    """Geometry-free graph Edge for a straight snap segment (RET exit → first Junction, etc.)."""
    lid = str(link_id or "").strip() or "_snap_"
    return Edge(
        _SYNTH_EDGE_FROM,
        _SYNTH_EDGE_TO,
        max(float(seg_len_px), 0.01),
        lid,
        "both",
        str(path_type or "taxiway"),
    )


def _junction_path_to_edge_list(
    graph: AirsideGraph, junction_ids: List[str]
) -> Tuple[List[Edge], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    edges_out: List[Edge] = []
    spans_out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(junction_ids) - 1):
        a_id = junction_ids[i]
        b_id = junction_ids[i + 1]
        e = _directed_edge_between(graph, a_id, b_id)
        if e is None:
            return [], []
        ja = graph.junctions.get(a_id)
        jb = graph.junctions.get(b_id)
        if ja is None or jb is None:
            return [], []
        edges_out.append(e)
        spans_out.append(((float(ja.col), float(ja.row)), (float(jb.col), float(jb.row))))
    return edges_out, spans_out


def _taxi_duration_sec_from_edge_route(
    resources: Dict[str, Resource],
    edges: List[Edge],
    spans: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> float:
    total = 0.0
    for e, (p0, p1) in zip(edges, spans):
        seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        spd = _speed_ms_for_edge(resources, e)
        total += seg / max(spd, 0.1)
    return total


def _taxi_remaining_sec_from_edge_cursor(agent: Flight, resources: Dict[str, Resource]) -> float:
    """Travel time left on edge_list from current cursor / s_along (s)."""
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


def _snap_agent_onto_current_edge(agent: Flight) -> None:
    """Place agent on the current edge segment and set edge_s_along_px from projection (px)."""
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


def _advance_agent_on_edge_list(agent: Flight, dt: float, resources: Dict[str, Resource]) -> None:
    """Move only along edge_list / edge_segment_endpoints; pop edges by completing each segment (px geometry)."""
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
        e = edges[ei]
        p0, p1 = spans[ei]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            agent.edge_cursor += 1
            agent.edge_s_along_px = 0.0
            continue
        spd = max(_speed_ms_for_edge(resources, e), 0.1)
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


def _plan_taxi_edge_route_full_graph(
    graph: AirsideGraph,
    origin: Tuple[float, float],
    dest: Tuple[float, float],
) -> Tuple[List[Edge], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Dijkstra on full graph; return Edge chain + segment endpoints, with origin/dest snap connectors."""
    o_id = _resolve_junction_at_xy(graph, origin[0], origin[1])
    d_id = _resolve_junction_at_xy(graph, dest[0], dest[1])
    jpath: Optional[List[str]] = None
    if o_id and d_id and o_id in graph.junctions and d_id in graph.junctions:
        jpath = reroute(graph, o_id, d_id, None)
    if not jpath:
        cs = _find_closest_junction_id(graph, origin[0], origin[1])
        ce = _find_closest_junction_id(graph, dest[0], dest[1])
        if cs and ce and cs != ce:
            jpath = reroute(graph, cs, ce, None)
    if not jpath or len(jpath) < 2:
        return [], []
    el, sp = _junction_path_to_edge_list(graph, jpath)
    if not el or not sp:
        return [], []
    j0 = graph.junctions.get(jpath[0])
    if j0 is not None:
        ox, oy = float(j0.col), float(j0.row)
        if math.hypot(origin[0] - ox, origin[1] - oy) > 2.0:
            d0 = math.hypot(origin[0] - ox, origin[1] - oy)
            link0 = el[0].link_id
            pt0 = el[0].path_type
            el.insert(0, _synthetic_connector_edge(link0, pt0, d0))
            sp.insert(0, ((float(origin[0]), float(origin[1])), (ox, oy)))
    jn = graph.junctions.get(jpath[-1])
    if jn is not None:
        nx, ny = float(jn.col), float(jn.row)
        if math.hypot(dest[0] - nx, dest[1] - ny) > 2.0:
            dn = math.hypot(dest[0] - nx, dest[1] - ny)
            linkn = el[-1].link_id
            ptn = el[-1].path_type
            el.append(_synthetic_connector_edge(linkn, ptn, dn))
            sp.append(((nx, ny), (float(dest[0]), float(dest[1]))))
    return el, sp


def _speed_ms_for_edge(resources: Dict[str, Resource], edge: Optional[Edge]) -> float:
    if edge is None:
        return DEFAULT_TAXI_SPEED_MS
    res = resources.get(edge.link_id)
    if isinstance(res, TaxiwayResource):
        return float(res.avg_velocity_ms)
    return DEFAULT_TAXI_SPEED_MS


def _compute_arrival_paths_for_agent(
    agent: Flight, layout: dict, graph: AirsideGraph, resources: Dict[str, Resource], cell_size: float
) -> None:
    """Phase 1 runway+RET polyline and Phase 2 reroute (taxi network only); mutates agent."""
    _fid = str(agent.id or "?")
    _arrival_path_debug_print(
        _fid,
        "enter | arrRwy=%s ret=%s stand=%s | graph junctions=%d directed_edge_starts=%d"
        % (
            agent.arr_runway_id,
            agent.sampled_arr_ret,
            agent.assigned_stand_id,
            len(graph.junctions),
            len(graph.edges),
        ),
    )

    agent.arr_runway_ret_waypoints = None
    agent.edge_list = []
    agent.edge_segment_endpoints = []
    agent.edge_cursor = 0
    agent.edge_s_along_px = 0.0
    agent.sim_export_arrival_edge_list = []

    arr_rwy_res = resources.get(agent.arr_runway_id)
    if not arr_rwy_res or not isinstance(arr_rwy_res, RunwayResource):
        _arrival_path_debug_print(
            _fid,
            "EXIT: no RunwayResource for arr_runway_id=%r (got type=%s)"
            % (agent.arr_runway_id, type(arr_rwy_res).__name__),
        )
        return
    plan = _plan_arrival_approach_leg(arr_rwy_res, agent)
    if plan is None:
        _arrival_path_debug_print(_fid, "EXIT: _plan_arrival_approach_leg returned None (degenerate runway verts?)")
        return
    _approach_dur, _approach_start, td_xy, verts = plan
    verts_use = list(verts)
    if agent.arr_runway_dir == "counter_clockwise":
        verts_use = list(reversed(verts_use))
    rwy_total_len = _polyline_length(verts_use)
    td_d = min(float(agent.arr_td_dist_m or 0.0), rwy_total_len)
    ret_d = min(float(agent.arr_ret_dist_m or 0.0), rwy_total_len)
    if ret_d < td_d:
        ret_d = td_d
    origin = (float(td_xy[0]), float(td_xy[1]))
    runway_wp = _runway_centerline_waypoints_td_to_ret(verts_use, td_d, ret_d, origin)

    ret_res = resources.get(agent.sampled_arr_ret)
    ret_px: List[Tuple[float, float]] = []
    if ret_res and isinstance(ret_res, TaxiwayResource):
        ret_px = ret_res.vertices_pixels()

    mr = max(0.0, float(getattr(graph, "junction_merge_radius_px", 0.0)))
    for p in ret_px[:2]:
        jn_id = _resolve_junction_at_xy(graph, p[0], p[1])
        jn = graph.junctions.get(jn_id) if jn_id else None
        if jn is None or math.hypot(float(jn.col) - p[0], float(jn.row) - p[1]) > mr + 1.0:
            _logger.warning(
                "RET vertex (%.2f, %.2f) is not within junction merge radius of any graph Junction (flight %s)",
                p[0], p[1], agent.id,
            )

    combined = _merge_runway_and_ret_polyline(runway_wp, ret_px)
    agent.arr_runway_ret_waypoints = combined if combined else None
    if not combined:
        _arrival_path_debug_print(
            _fid,
            "EXIT: empty combined runway+RET polyline | runway_wp_pts=%d ret_px_pts=%d"
            % (len(runway_wp), len(ret_px)),
        )
        return

    phase1_end = combined[-1]
    _arrival_path_debug_print(
        _fid,
        "phase1 | td_d=%.2f ret_d=%.2f rwy_len=%.2f | combined_pts=%d phase1_end=(%.2f,%.2f)"
        % (td_d, ret_d, rwy_total_len, len(combined), phase1_end[0], phase1_end[1]),
    )
    start_j = _resolve_junction_at_xy(graph, phase1_end[0], phase1_end[1])
    end_j = _stand_junction_id_from_layout(layout, agent.assigned_stand_id, cell_size, graph)
    if not end_j:
        st_res = resources.get(agent.assigned_stand_id or "")
        if st_res and isinstance(st_res, StandResource):
            end_j = _resolve_junction_at_xy(graph, st_res.col, st_res.row)
    if not start_j or not end_j:
        _arrival_path_debug_print(
            _fid,
            "EXIT: missing junction | start_j=%r end_j=%r stand=%r phase1_end=(%.2f,%.2f)"
            % (start_j, end_j, agent.assigned_stand_id, phase1_end[0], phase1_end[1]),
        )
        _logger.warning(
            "Arrival taxi plan: missing start/end Junction (flight=%s start=%s end=%s stand=%s)",
            agent.id, start_j, end_j, agent.assigned_stand_id,
        )
        return

    jpath_filtered = reroute(graph, start_j, end_j, _reroute_arrival_taxi_edge_ok)
    _arrival_path_debug_print(
        _fid,
        "reroute(filtered taxi types) | %s -> %s | path=%s"
        % (start_j, end_j, "None" if not jpath_filtered else "len=%d %s…" % (len(jpath_filtered), jpath_filtered[:4])),
    )
    jpath = jpath_filtered
    if not jpath:
        jpath = reroute(graph, start_j, end_j, None)
        _arrival_path_debug_print(
            _fid,
            "reroute(FALLBACK no filter) | %s -> %s | path=%s"
            % (start_j, end_j, "None" if not jpath else "len=%d %s…" % (len(jpath), jpath[:4])),
        )
    if not jpath:
        _arrival_path_debug_print(
            _fid,
            "EXIT: Dijkstra found no path (even with full graph). Check disconnected network or wrong junction ids.",
        )
        _logger.warning("Arrival taxi plan: reroute failed (flight=%s %s→%s)", agent.id, start_j, end_j)
        return

    edges, spans = _junction_path_to_edge_list(graph, jpath)
    if not edges or not spans:
        miss: Optional[Tuple[str, str]] = None
        if len(jpath) >= 2:
            for _i in range(len(jpath) - 1):
                _a, _b = jpath[_i], jpath[_i + 1]
                if _directed_edge_between(graph, _a, _b) is None:
                    miss = (_a, _b)
                    break
        _arrival_path_debug_print(
            _fid,
            "EXIT: junction_path_to_edge_list empty | jpath_len=%d head=%s tail=%s first_missing_directed_edge=%s"
            % (len(jpath), jpath[:3], jpath[-3:] if len(jpath) >= 3 else jpath, miss),
        )
        _logger.warning("Arrival taxi plan: missing graph Edges along junction path (flight=%s)", agent.id)
        return

    first_j = graph.junctions.get(jpath[0])
    synth = False
    if first_j is not None:
        fx, fy = float(first_j.col), float(first_j.row)
        if math.hypot(phase1_end[0] - fx, phase1_end[1] - fy) > 2.0:
            d_conn = math.hypot(phase1_end[0] - fx, phase1_end[1] - fy)
            ret_lid = str(agent.sampled_arr_ret or "").strip() or "_ret_"
            edges.insert(0, _synthetic_connector_edge(ret_lid, "runway_exit", d_conn))
            spans.insert(0, ((float(phase1_end[0]), float(phase1_end[1])), (fx, fy)))
            synth = True

    agent.edge_list = edges
    agent.edge_segment_endpoints = spans
    agent.sim_export_arrival_edge_list = list(edges)
    _arrival_path_debug_print(
        _fid,
        "OK | edge_list_len=%d synthetic_first_segment=%s sim_export_len=%d"
        % (len(agent.edge_list), synth, len(agent.sim_export_arrival_edge_list)),
    )


# ---------------------------------------------------------------------------
# Resource builder
# ---------------------------------------------------------------------------

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
            id=tw_id, name=tw.get("name", ""),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 15)),
            layout_cell_size=cell_size,
        )

    for tw in layout.get("taxiways", []):
        tw_id = tw.get("id", "")
        resources[tw_id] = TaxiwayResource(
            id=tw_id, name=tw.get("name", ""),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 10)),
            layout_cell_size=cell_size,
        )

    for st in layout.get("pbbStands", []):
        st_id = st.get("id", "")
        col_cell = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row_cell = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            col=col_cell * cs, row=row_cell * cs,
        )

    for st in layout.get("remoteStands", []):
        st_id = st.get("id", "")
        col_cell = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row_cell = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            col=col_cell * cs, row=row_cell * cs,
        )

    return resources


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _normalize_flight_dict_for_agent(f: dict, information: dict) -> dict:
    """Shallow copy: fill arr runway / RET from token; fill omitted ROT/RET physics from Information.json."""
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
    """Pro Sim input must include RET, landing runway, touchdown distance, and TD speed."""
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


def build_agents(
    flights: list,
    graph: AirsideGraph,
    resources: Dict[str, Resource],
    information: dict,
    layout: dict,
) -> List[Flight]:
    agents: list[Flight] = []
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    wingspan = _safe_float(sim_cfg.get("aircraftWingspanM"), float("nan"))
    fuselage = _safe_float(sim_cfg.get("aircraftFuselageLengthM"), float("nan"))
    if not math.isfinite(wingspan) or wingspan <= 0:
        raise ValueError("Information.json tiers.algorithm.simulation.aircraftWingspanM must be > 0")
    if not math.isfinite(fuselage) or fuselage <= 0:
        raise ValueError("Information.json tiers.algorithm.simulation.aircraftFuselageLengthM must be > 0")

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
            raise ValueError(f"Flight[{idx}] id={fid!r}: arrVRetInMs invalid (flight or simulation defaults)")
        if not math.isfinite(vret_out) or vret_out <= 0:
            raise ValueError(f"Flight[{idx}] id={fid!r}: arrVRetOutMs invalid (flight or simulation defaults)")

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
            arr_runway_dir=str(f.get("arrRunwayDirUsed", "")).strip(),
            dep_runway_dir=str(f.get("depRunwayDirUsed", "")).strip(),
            arr_v_td_ms=_safe_float(f.get("arrVTdMs"), float("nan")),
            arr_v_ret_in_ms=vret_in,
            arr_v_ret_out_ms=vret_out,
            arr_td_dist_m=_safe_float(f.get("arrTdDistM"), float("nan")),
            arr_ret_dist_m=_safe_float(f.get("arrRetDistM"), float("nan")),
            arr_decel_ms2=float(arr_decel_val),
        )

        cs = float(layout.get("grid", {}).get("cellSize", 20.0))
        _compute_arrival_paths_for_agent(agent, layout, graph, resources, cs)

        agent.schedule = ScheduleTimes(
            sldt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sldtMin_d"))),
            sibt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sibtMin_d"))),
            sobt_sec=_sim_sec_optional(_minutes_to_sec(f.get("sobtMin_d"))),
            stot_sec=_sim_sec_optional(_minutes_to_sec(f.get("stotMin_d"))),
        )

        agents.append(agent)
    return agents


# ---------------------------------------------------------------------------
# Movement interpolation (with decel/accel support)
# ---------------------------------------------------------------------------

def _move_agent_along_path(
    agent: Flight, dt: float, cell_size: float, resources: Dict[str, Resource]
) -> None:
    """Advance along path_queue (approach / fallback) or edge_list (taxi); trail speed = |Δpos|/dt."""
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


# ---------------------------------------------------------------------------
# Taxi path planning
# ---------------------------------------------------------------------------

def _plan_taxi_path(
    graph: AirsideGraph, origin: Tuple[float, float], dest: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Departure / fallback taxi: full graph Dijkstra (may use runway edges)."""
    origin_id = _resolve_junction_at_xy(graph, origin[0], origin[1])
    dest_id = _resolve_junction_at_xy(graph, dest[0], dest[1])

    if origin_id and dest_id and origin_id in graph.junctions and dest_id in graph.junctions:
        node_path = reroute(graph, origin_id, dest_id, None)
        if node_path:
            return graph.junction_path_xy(node_path)

    closest_start = _find_closest_junction_id(graph, origin[0], origin[1])
    closest_end = _find_closest_junction_id(graph, dest[0], dest[1])
    if closest_start and closest_end and closest_start != closest_end:
        node_path = reroute(graph, closest_start, closest_end, None)
        if node_path:
            coords = graph.junction_path_xy(node_path)
            return [origin] + coords + [dest]

    return [origin, dest]


# ---------------------------------------------------------------------------
# Post-event handler (token-based path resolution)
# ---------------------------------------------------------------------------

_SIM_CELL_SIZE: float = 20.0


def _handle_post_event(agent: Flight, event: Event, result: str,
                       graph: AirsideGraph, resources: Dict[str, Resource],
                       event_queue: EventQueue, cell_size: float, sim_dt: int = 1) -> None:
    """Generate follow-up events for state transitions that need path planning.
    Uses token waypoints (arrRunwayId, ExitTaxiwayId, apronId, depRunwayId)
    to resolve multi-phase paths."""
    if result != "ACCEPT":
        return

    # --- TOUCHDOWN: Phase 1 path (runway td→RET + RET) from init or built here; decel on ARR_RUNWAY ---
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
            event_queue.push(Event(
                time=_sim_sec(event.time + sim_dt),
                event_type="RET_ENTER",
                agent=agent,
                resource=ret_res_fb,
            ))

    # --- RET_ENTER: path already completed under ARR_RUNWAY; hand off to taxi ---
    elif event.event_type == "RET_ENTER" and agent.state == "RET":
        ret_res = resources.get(agent.sampled_arr_ret) or event.resource
        agent.path_queue = []
        agent.decel_rate = 0.0
        agent.accel_rate = 0.0
        event_queue.push(Event(
            time=_sim_sec(event.time + sim_dt),
            event_type="TAXI_START",
            agent=agent,
            resource=ret_res,
        ))

    # --- TAXI_START: follow precomputed edge_list only, else waypoint fallback ---
    elif event.event_type == "TAXI_START" and agent.state == "TAXI_IN":
        stand_res = resources.get(agent.assigned_stand_id)
        if stand_res and isinstance(stand_res, StandResource):
            origin = (agent.col, agent.row)
            dest = (stand_res.col, stand_res.row)
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
                path = _plan_taxi_path(graph, origin, dest)
                agent.path_queue = path
                taxi_speed = DEFAULT_TAXI_SPEED_MS
                agent.speed_ms = taxi_speed
                dist_m = sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                ) if len(path) > 1 else 0
                taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0

            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms

            event_queue.push(Event(
                time=event.time + taxi_time,
                event_type="STAND_ENTER",
                agent=agent,
                resource=stand_res,
            ))
        else:
            agent.state = "PARKED"
            agent.speed_ms = 0.0
            agent.schedule.aibt_sec = event.time
            agent.schedule.aobt_sec = _sim_sec(event.time + float(agent.dwell_sec or 0.0))

    # --- PUSHBACK: taxi from stand to DEP runway start (edge_list if graph path exists) ---
    elif event.event_type == "PUSHBACK" and agent.state == "TAXI_OUT":
        dep_rwy_res = resources.get(agent.dep_runway_id)
        if dep_rwy_res and isinstance(dep_rwy_res, RunwayResource):
            verts = dep_rwy_res._vertices_pixels()
            if agent.dep_runway_dir == "counter_clockwise":
                verts = list(reversed(verts))

            origin = (agent.col, agent.row)
            dest = verts[0] if verts else origin
            de, dsp = _plan_taxi_edge_route_full_graph(graph, origin, dest)
            if de and dsp and len(de) == len(dsp):
                agent.path_queue = []
                agent.edge_list = de
                agent.edge_segment_endpoints = dsp
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                _snap_agent_onto_current_edge(agent)
                agent.speed_ms = _speed_ms_for_edge(resources, de[agent.edge_cursor])
                taxi_time = _taxi_remaining_sec_from_edge_cursor(agent, resources)
                if taxi_time <= 0:
                    taxi_time = 60.0
            else:
                agent.edge_list = []
                agent.edge_segment_endpoints = []
                agent.edge_cursor = 0
                agent.edge_s_along_px = 0.0
                path = _plan_taxi_path(graph, origin, dest)
                agent.path_queue = path
                dist_m = sum(
                    math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                ) if len(path) > 1 else 0
                taxi_speed = DEFAULT_TAXI_SPEED_MS
                agent.speed_ms = taxi_speed
                taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0

            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms

            event_queue.push(Event(
                time=event.time + taxi_time,
                event_type="LINEUP",
                agent=agent,
                resource=dep_rwy_res,
            ))
        else:
            agent.state = "DEPARTED"
            agent.speed_ms = 0.0
            agent.schedule.atot_sec = event.time

    # --- LINEUP: hold at runway start, then request takeoff ---
    elif event.event_type == "LINEUP" and agent.state == "LINEUP_HOLD":
        agent.speed_ms = 0.0
        agent.path_queue = []
        agent.edge_list = []
        agent.edge_segment_endpoints = []
        agent.edge_cursor = 0
        agent.edge_s_along_px = 0.0
        agent.decel_rate = 0.0
        agent.accel_rate = 0.0

        event_queue.push(Event(
            time=event.time + DEP_LINEUP_HOLD_SEC,
            event_type="TAKEOFF_REQUEST",
            agent=agent,
            resource=event.resource,
        ))

    # --- DEPARTED: stop movement ---
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


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

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

    # Config key approachOffsetM: numeric offset in layout pixels for inbound leg start.
    APPROACH_OFFSET_PX = _safe_float(sim_cfg.get("approachOffsetM"), 10_000.0)
    flight_cfg = _deep_get(information, "tiers", "flight_schedule", default={})
    DEP_LINEUP_HOLD_SEC = _safe_float(flight_cfg.get("depLineupHoldSec"), 20.0)
    DEP_TAKEOFF_ACCEL_MS2 = _safe_float(flight_cfg.get("depTakeoffAccelSmallMs2"), 2.5)

    base_date = str(_deep_get(information, "tiers", "algorithm", "simulation", "baseDate",
                              default="2026-03-31"))

    graph = build_graph(layout, information)
    poly_net_j = _polyline_apron_junctions_for_sim_result(graph)
    resources = build_resources(layout, information)
    flights_raw = layout.get("flights", [])

    if not flights_raw:
        return _build_output([], base_date, poly_net_j)

    agents = build_agents(flights_raw, graph, resources, information, layout)
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

                _handle_post_event(agent, event, result, graph, resources, event_queue, cell_size, dt)

        for agent in agents:
            if not agent.is_stationary():
                _move_agent_along_path(agent, float(dt), cell_size, resources)

        for agent in agents:
            if (
                agent.state == "ARR_RUNWAY"
                and agent.pending_ret_enter
                and not agent.path_queue
            ):
                agent.pending_ret_enter = False
                ret_r = resources.get(agent.sampled_arr_ret) or resources.get(agent.arr_runway_id)
                if ret_r is None:
                    continue
                event_queue.push(Event(
                    time=_sim_sec(current_time + dt),
                    event_type="RET_ENTER",
                    agent=agent,
                    resource=ret_r,
                ))

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


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------

def _sec_to_datetime_str(sec, base_date: str) -> Optional[str]:
    if sec is None:
        return None
    try:
        sec_v = float(sec)
        if not math.isfinite(sec_v):
            return None
    except (TypeError, ValueError):
        return None
    from datetime import datetime, timedelta
    try:
        parts = base_date.split("-")
        base = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        base = datetime(2026, 3, 31)
    result = base + timedelta(seconds=sec_v)
    return result.strftime("%m/%d %H:%M:%S")


def _edge_to_sim_result_dict(e: Edge) -> dict:
    """JSON-safe Edge for sim_result schedule rows."""
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
            "edge_list": [
                _edge_to_sim_result_dict(e) for e in (a.sim_export_arrival_edge_list or [])
            ],
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
