"""
Airside Simulation Engine — Agent-Based + Discrete Event Simulation.

Reads a serialized layout (from Layout_Design) and Information.json,
runs a time-stepped DES loop where each Flight is an Agent interacting
with shared Resources (Runway, Taxiway, Stand, etc.), and returns:
  - per-flight position timeline  (col, row at each time step)
  - A-schedule                    (ALDT, AIBT, AOBT, ATOT per flight)
  - KPI aggregation               (utilization, delay, throughput …)

Token-based path resolution:
  Each flight carries a token {arrRunwayId, apronId, depRunwayId} plus
  sampledArrRet (RET taxiway id) and physics params (arrVTdMs, arrRotSec,
  etc.). The DES uses these to plan multi-phase movement:
    APPROACH → TOUCHDOWN → RET_ENTER → TAXI_START → STAND_ENTER →
    PUSHBACK → LINEUP → TAKEOFF_REQUEST → DEPARTED
"""
from __future__ import annotations

import heapq
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_INFO_FILE = _ROOT / "data" / "Info_storage" / "Information.json"

APPROACH_OFFSET_M: float = 10_000.0
DEP_LINEUP_HOLD_SEC: float = 20.0
DEP_TAKEOFF_ACCEL_MS2: float = 2.5
DEFAULT_TAXI_SPEED_MS: float = 10.0

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


def _minutes_to_sec(m) -> Optional[float]:
    v = _safe_float(m, float("nan"))
    return v * 60.0 if math.isfinite(v) else None


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
    aibt_sec: Optional[float] = None
    aobt_sec: Optional[float] = None
    atot_sec: Optional[float] = None


@dataclass
class Flight:
    id: str
    reg: str = ""
    airline_code: str = ""
    flight_number: str = ""
    aircraft_type: str = ""
    icao_category: str = "M"
    wingspan_m: float = 40.0
    fuselage_length_m: float = 50.0
    col: float = 0.0
    row: float = 0.0
    heading_deg: float = 0.0
    speed_ms: float = 0.0
    state: str = "SCHEDULED"
    history: list = field(default_factory=list)
    schedule: ScheduleTimes = field(default_factory=ScheduleTimes)
    assigned_stand_id: Optional[str] = None
    arr_runway_id: Optional[str] = None
    dep_runway_id: Optional[str] = None
    path_queue: list = field(default_factory=list)
    dwell_sec: float = 2700.0
    arr_rot_sec: float = 55.0
    token: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict, repr=False)

    # --- token-derived waypoint IDs ---
    sampled_arr_ret: Optional[str] = None
    arr_runway_dir: str = "clockwise"
    dep_runway_dir: str = "clockwise"

    # --- physics params from serialized data ---
    arr_v_td_ms: float = 70.0
    arr_v_ret_in_ms: float = 30.0
    arr_v_ret_out_ms: float = 15.0
    arr_td_dist_m: float = 400.0
    arr_ret_dist_m: float = 1500.0

    # --- movement phase + dynamic speed control ---
    phase: str = "SCHEDULED"
    target_speed_ms: float = 0.0
    decel_rate: float = 0.0
    accel_rate: float = 0.0

    _departed_recorded: bool = field(default=False, repr=False)

    def record_position(self, t: float) -> None:
        if self.state == "SCHEDULED":
            return
        if self.state == "DEPARTED":
            if not self._departed_recorded:
                self.history.append((t, self.col, self.row))
                self._departed_recorded = True
            return
        self.history.append((t, self.col, self.row))

    def is_stationary(self) -> bool:
        return self.state in ("PARKED", "SCHEDULED", "DEPARTED", "LINEUP_HOLD")

    # --- agent reaction methods -------------------------------------------

    def update_state(self, event: "Event") -> None:
        _STATE_MAP = {
            "ARRIVAL_REQUEST": "APPROACH",
            "TOUCHDOWN": "RUNWAY_ROLL",
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
            self.phase = new

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
        return Event(time=t, event_type="ARRIVAL_REQUEST", agent=self, resource=rwy_res)


# ---------------------------------------------------------------------------
# Event & EventQueue
# ---------------------------------------------------------------------------

_event_counter = 0


@dataclass(order=False)
class Event:
    time: float
    event_type: str
    agent: Flight
    resource: "Resource"
    priority: int = 0
    _seq: int = field(default=0, repr=False)

    def __post_init__(self):
        global _event_counter
        _event_counter += 1
        self._seq = _event_counter

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

    def pop(self, current_time: float) -> List[Event]:
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


class RunwayResource(Resource):
    def __init__(self, id: str, name: str = "", separation_matrix: Optional[dict] = None,
                 rot_defaults: Optional[dict] = None, **kw):
        super().__init__(id, name)
        self.separation_matrix = separation_matrix or {}
        self.rot_defaults = rot_defaults or {}
        self.last_operation: Optional[Tuple[str, float, str]] = None
        self.vertices: list = kw.get("vertices", [])
        self.direction: str = kw.get("direction", "both")

    def _vertices_cells(self) -> List[Tuple[float, float]]:
        return [(v.get("col", 0.0), v.get("row", 0.0)) for v in self.vertices]

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
            sep = self._get_sep_sec(prev_op, "ARR", prev_cat, agent.icao_category)
            if event.time - prev_time >= sep:
                return "ACCEPT"
            return "WAIT"
        if event.event_type == "TAKEOFF_REQUEST":
            if self.last_operation is None:
                return "ACCEPT"
            prev_op, prev_time, prev_cat = self.last_operation
            sep = self._get_sep_sec(prev_op, "DEP", prev_cat, agent.icao_category)
            if event.time - prev_time >= sep:
                return "ACCEPT"
            return "WAIT"
        return "ACCEPT"

    def generate_next_events(self, agent: Flight, event: Event, result: str) -> List[Event]:
        events: list[Event] = []
        if result == "ACCEPT":
            if event.event_type == "ARRIVAL_REQUEST":
                self.last_operation = ("ARR", event.time, agent.icao_category)
                agent.schedule.aldt_sec = event.time

                verts = self._vertices_cells()
                cell_size = _SIM_CELL_SIZE

                if verts and len(verts) >= 2:
                    rwy_dir = agent.arr_runway_dir
                    if rwy_dir == "counter_clockwise":
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

                    approach_dist_cells = APPROACH_OFFSET_M / max(cell_size, 1.0)
                    approach_start = (
                        rwy_start[0] - ux * approach_dist_cells,
                        rwy_start[1] - uy * approach_dist_cells,
                    )

                    td_dist_cells = agent.arr_td_dist_m / max(cell_size, 1.0)
                    touchdown_pt = _polyline_point_at_distance(verts, td_dist_cells)

                    approach_pts = [approach_start, rwy_start, touchdown_pt]
                    approach_len_m = _polyline_length(approach_pts) * cell_size
                    approach_speed_ms = float(agent.arr_v_td_ms)
                    approach_dur = approach_len_m / max(approach_speed_ms, 1.0)

                    agent.col, agent.row = approach_start
                    agent.path_queue = [rwy_start, touchdown_pt]
                    agent.speed_ms = approach_speed_ms / max(cell_size, 1.0)
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
                    if verts:
                        agent.col, agent.row = verts[0]
                    events.append(Event(
                        time=event.time + agent.arr_rot_sec,
                        event_type="TAXI_START",
                        agent=agent,
                        resource=self,
                    ))

            elif event.event_type == "TAKEOFF_REQUEST":
                self.last_operation = ("DEP", event.time, agent.icao_category)
                agent.schedule.atot_sec = event.time

                verts = self._vertices_cells()
                cell_size = _SIM_CELL_SIZE

                if verts and len(verts) >= 2:
                    dep_dir = agent.dep_runway_dir
                    if dep_dir == "counter_clockwise":
                        verts = list(reversed(verts))

                    agent.path_queue = list(verts[1:])
                    agent.speed_ms = 0.1 / max(cell_size, 1.0)
                    agent.accel_rate = DEP_TAKEOFF_ACCEL_MS2 / max(cell_size, 1.0)
                    agent.decel_rate = 0.0
                    agent.target_speed_ms = 0.0

                    roll_len_m = _polyline_length(verts) * cell_size
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
    def __init__(self, id: str, name: str = "", direction: str = "both",
                 vertices: Optional[list] = None, reverse_cost: float = 1_000_000, **kw):
        super().__init__(id, name)
        self.direction = direction
        self.vertices = vertices or []
        self.reverse_cost = reverse_cost
        self.occupants: list[str] = []
        self.avg_velocity_ms: float = kw.get("avg_velocity_ms", 10.0)

    def vertices_cells(self) -> List[Tuple[float, float]]:
        return [(v.get("col", 0.0), v.get("row", 0.0)) for v in self.vertices]

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
    def __init__(self, id: str, name: str = "", stand_type: str = "pbb",
                 col: float = 0, row: float = 0, **kw):
        super().__init__(id, name)
        self.stand_type = stand_type
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
                agent.schedule.aibt_sec = event.time
                pushback_time = event.time + agent.dwell_sec
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


# ---------------------------------------------------------------------------
# Graph & Pathfinding
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    col: float
    row: float


@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    cost: float
    taxiway_id: str
    direction: str = "both"


class AirsideGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, List[GraphEdge]] = {}
        self._path_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.from_id not in self.edges:
            self.edges[edge.from_id] = []
        self.edges[edge.from_id].append(edge)

    def dijkstra(self, start_id: str, end_id: str) -> Optional[List[str]]:
        cache_key = (start_id, end_id)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if start_id not in self.nodes or end_id not in self.nodes:
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
                v = edge.to_id
                if v in visited:
                    continue
                nd = d + edge.cost
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if end_id not in prev:
            self._path_cache[cache_key] = None
            return None

        path: list[str] = []
        cur: Optional[str] = end_id
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        self._path_cache[cache_key] = path
        return path

    def path_coords(self, node_ids: List[str]) -> List[Tuple[float, float]]:
        coords = []
        for nid in node_ids:
            n = self.nodes.get(nid)
            if n:
                coords.append((n.col, n.row))
        return coords


def build_graph(layout: dict, information: dict) -> AirsideGraph:
    graph = AirsideGraph()
    cell_size = layout.get("grid", {}).get("cellSize", 20.0)
    reverse_cost = float(_deep_get(information, "tiers", "algorithm", "pathSearch", "reverseCost", default=1_000_000))

    def _vertex_id(col: float, row: float) -> str:
        key = f"v_{col:.2f}_{row:.2f}"
        if key not in graph.nodes:
            graph.add_node(GraphNode(id=key, col=col, row=row))
        return key

    all_taxiways: list[dict] = []
    for tw in layout.get("runwayPaths", []):
        all_taxiways.append(dict(tw, pathType="runway"))
    for tw in layout.get("runwayTaxiways", []):
        all_taxiways.append(dict(tw, pathType="runway_taxiway"))
    for tw in layout.get("taxiways", []):
        pt = tw.get("pathType", "taxiway")
        all_taxiways.append(dict(tw, pathType=pt))

    for tw in all_taxiways:
        verts = tw.get("vertices", [])
        if len(verts) < 2:
            continue
        tw_id = tw.get("id", "")
        direction = tw.get("direction", "both")

        node_ids = [_vertex_id(v["col"], v["row"]) for v in verts]

        for i in range(len(node_ids) - 1):
            a_id = node_ids[i]
            b_id = node_ids[i + 1]
            a_n = graph.nodes[a_id]
            b_n = graph.nodes[b_id]
            dist_m = cell_size * math.hypot(b_n.col - a_n.col, b_n.row - a_n.row)
            cost = max(dist_m, 0.01)

            if direction == "clockwise":
                graph.add_edge(GraphEdge(a_id, b_id, cost, tw_id, direction))
                graph.add_edge(GraphEdge(b_id, a_id, cost + reverse_cost, tw_id, direction))
            elif direction == "counter_clockwise":
                graph.add_edge(GraphEdge(b_id, a_id, cost, tw_id, direction))
                graph.add_edge(GraphEdge(a_id, b_id, cost + reverse_cost, tw_id, direction))
            else:
                graph.add_edge(GraphEdge(a_id, b_id, cost, tw_id, direction))
                graph.add_edge(GraphEdge(b_id, a_id, cost, tw_id, direction))

    # --- apronLinks: connect stands to the taxiway network ---
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

        mid_verts = lk.get("midVertices", [])
        tx_px = lk.get("tx")
        ty_px = lk.get("ty")

        apron_pts: list[Tuple[float, float]] = [(stand_col, stand_row)]

        if isinstance(mid_verts, list) and mid_verts:
            for mv in mid_verts:
                if isinstance(mv, dict) and "col" in mv and "row" in mv:
                    apron_pts.append((_safe_float(mv["col"]), _safe_float(mv["row"])))
        elif tx_px is not None and ty_px is not None:
            tc = _safe_float(tx_px) / max(cell_size, 1.0)
            tr = _safe_float(ty_px) / max(cell_size, 1.0)
            apron_pts.append((tc, tr))

        if len(apron_pts) < 2:
            continue

        lk_id = lk.get("id", "apron_link")
        apron_node_ids = [_vertex_id(p[0], p[1]) for p in apron_pts]
        for i in range(len(apron_node_ids) - 1):
            a_id = apron_node_ids[i]
            b_id = apron_node_ids[i + 1]
            a_n = graph.nodes[a_id]
            b_n = graph.nodes[b_id]
            dist_m = cell_size * math.hypot(b_n.col - a_n.col, b_n.row - a_n.row)
            cost = max(dist_m, 0.01)
            graph.add_edge(GraphEdge(a_id, b_id, cost, lk_id, "both"))
            graph.add_edge(GraphEdge(b_id, a_id, cost, lk_id, "both"))

    return graph


# ---------------------------------------------------------------------------
# Resource builder
# ---------------------------------------------------------------------------

def build_resources(layout: dict, information: dict) -> Dict[str, Resource]:
    resources: Dict[str, Resource] = {}

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
            rot_defaults=rot_matrix,
            vertices=rw.get("vertices", []),
            direction=rw.get("direction", "both"),
        )

    for tw in layout.get("runwayTaxiways", []):
        tw_id = tw.get("id", "")
        resources[tw_id] = TaxiwayResource(
            id=tw_id, name=tw.get("name", ""),
            direction=tw.get("direction", "both"),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 15)),
        )

    for tw in layout.get("taxiways", []):
        tw_id = tw.get("id", "")
        resources[tw_id] = TaxiwayResource(
            id=tw_id, name=tw.get("name", ""),
            direction=tw.get("direction", "both"),
            vertices=tw.get("vertices", []),
            avg_velocity_ms=float(tw.get("avgMoveVelocity", 10)),
        )

    for st in layout.get("pbbStands", []):
        st_id = st.get("id", "")
        col = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            stand_type="pbb", col=col, row=row,
        )

    for st in layout.get("remoteStands", []):
        st_id = st.get("id", "")
        col = _safe_float(st.get("edgeCol"), _safe_float(st.get("col"), 0))
        row = _safe_float(st.get("edgeRow"), _safe_float(st.get("row"), 0))
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            stand_type="remote", col=col, row=row,
        )

    return resources


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agents(flights: list, graph: AirsideGraph, resources: Dict[str, Resource],
                 information: dict) -> List[Flight]:
    agents: list[Flight] = []
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    default_wingspan = float(sim_cfg.get("aircraftWingspanM", 40))
    default_fuselage = float(sim_cfg.get("aircraftFuselageLengthM", 50))
    rot_defaults = _deep_get(information, "tiers", "runway", "standards", "ICAO", "ROT", default={})

    for f in flights:
        if not isinstance(f, dict):
            continue
        if f.get("noWayArr") or f.get("noWayDep") or f.get("arrRetFailed"):
            continue

        fid = f.get("id", "")
        cat = f.get("code", "M")
        rot_sec = _safe_float(f.get("arrRotSec"), _safe_float(rot_defaults.get(cat), 55))
        dwell_sec = _safe_float(f.get("dwellMin"), 45) * 60.0

        token = f.get("token", {}) or {}
        arr_rwy = f.get("arrRunwayIdUsed") or token.get("arrRunwayId") or token.get("runwayId")
        dep_rwy = token.get("depRunwayId") or arr_rwy
        stand_id = f.get("standId") or token.get("apronId")
        sched_ret = f.get("scheduleArrRetId")
        if isinstance(sched_ret, str):
            sched_ret = sched_ret.strip() or None
        arr_ret_id = sched_ret or f.get("sampledArrRet")

        agent = Flight(
            id=fid,
            reg=f.get("reg", ""),
            airline_code=f.get("airlineCode", ""),
            flight_number=f.get("flightNumber", ""),
            aircraft_type=f.get("aircraftType", ""),
            icao_category=cat,
            wingspan_m=default_wingspan,
            fuselage_length_m=default_fuselage,
            dwell_sec=dwell_sec,
            arr_rot_sec=rot_sec,
            arr_runway_id=arr_rwy,
            dep_runway_id=dep_rwy,
            assigned_stand_id=stand_id,
            token=token,
            _raw=f,
            # --- token-derived fields ---
            sampled_arr_ret=arr_ret_id,
            arr_runway_dir=f.get("arrRunwayDirUsed", "clockwise"),
            dep_runway_dir=f.get("depRunwayDirUsed", "clockwise"),
            # --- physics params ---
            arr_v_td_ms=_safe_float(f.get("arrVTdMs"), 70),
            arr_v_ret_in_ms=_safe_float(f.get("arrVRetInMs"), 30),
            arr_v_ret_out_ms=_safe_float(f.get("arrVRetOutMs"), 15),
            arr_td_dist_m=_safe_float(f.get("arrTdDistM"), 400),
            arr_ret_dist_m=_safe_float(f.get("arrRetDistM"), 1500),
        )

        agent.schedule = ScheduleTimes(
            sldt_sec=_minutes_to_sec(f.get("sldtMin_d")),
            sibt_sec=_minutes_to_sec(f.get("sibtMin_d")),
            sobt_sec=_minutes_to_sec(f.get("sobtMin_d")),
            stot_sec=_minutes_to_sec(f.get("stotMin_d")),
        )

        agents.append(agent)
    return agents


# ---------------------------------------------------------------------------
# Movement interpolation (with decel/accel support)
# ---------------------------------------------------------------------------

def _move_agent_along_path(agent: Flight, dt: float, cell_size: float) -> None:
    """Advance agent along its path_queue with dynamic speed (decel/accel)."""
    if agent.is_stationary() or not agent.path_queue:
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
        if dist > 0:
            agent.heading_deg = math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Taxi path planning
# ---------------------------------------------------------------------------

def _plan_taxi_path(graph: AirsideGraph,
                    origin: Tuple[float, float], dest: Tuple[float, float]) -> List[Tuple[float, float]]:
    origin_id = f"v_{origin[0]:.2f}_{origin[1]:.2f}"
    dest_id = f"v_{dest[0]:.2f}_{dest[1]:.2f}"

    if origin_id in graph.nodes and dest_id in graph.nodes:
        node_path = graph.dijkstra(origin_id, dest_id)
        if node_path:
            return graph.path_coords(node_path)

    closest_start = _find_closest_node(graph, origin[0], origin[1])
    closest_end = _find_closest_node(graph, dest[0], dest[1])
    if closest_start and closest_end and closest_start != closest_end:
        node_path = graph.dijkstra(closest_start, closest_end)
        if node_path:
            coords = graph.path_coords(node_path)
            return [origin] + coords + [dest]

    return [origin, dest]


def _find_closest_node(graph: AirsideGraph, col: float, row: float) -> Optional[str]:
    best_id: Optional[str] = None
    best_dist = math.inf
    for nid, n in graph.nodes.items():
        d = math.hypot(n.col - col, n.row - row)
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id


# ---------------------------------------------------------------------------
# Post-event handler (token-based path resolution)
# ---------------------------------------------------------------------------

_SIM_CELL_SIZE: float = 20.0


def _handle_post_event(agent: Flight, event: Event, result: str,
                       graph: AirsideGraph, resources: Dict[str, Resource],
                       event_queue: EventQueue, cell_size: float) -> None:
    """Generate follow-up events for state transitions that need path planning.
    Uses token waypoints (arrRunwayId, sampledArrRet, apronId, depRunwayId)
    to resolve multi-phase paths."""
    if result != "ACCEPT":
        return

    # --- TOUCHDOWN: runway roll with deceleration ---
    if event.event_type == "TOUCHDOWN" and agent.state == "RUNWAY_ROLL":
        arr_rwy_res = resources.get(agent.arr_runway_id)
        if arr_rwy_res and isinstance(arr_rwy_res, RunwayResource):
            verts = arr_rwy_res._vertices_cells()
            if agent.arr_runway_dir == "counter_clockwise":
                verts = list(reversed(verts))

            if verts and len(verts) >= 2:
                rwy_total_len = _polyline_length(verts)
                td_d = min(agent.arr_td_dist_m / max(cell_size, 1), rwy_total_len)
                ret_d = min(agent.arr_ret_dist_m / max(cell_size, 1), rwy_total_len)
                if ret_d <= td_d:
                    ret_d = min(td_d + 50 / max(cell_size, 1), rwy_total_len)

                roll_pts = _subdivide_polyline(verts, td_d, ret_d, 8)
                agent.path_queue = roll_pts[1:]
                agent.speed_ms = agent.arr_v_td_ms / max(cell_size, 1.0)
                target_spd = agent.arr_v_ret_in_ms / max(cell_size, 1.0)
                agent.target_speed_ms = target_spd
                if agent.arr_rot_sec > 0:
                    agent.decel_rate = (agent.speed_ms - target_spd) / agent.arr_rot_sec
                else:
                    agent.decel_rate = 0.0
                agent.accel_rate = 0.0

        ret_res = resources.get(agent.sampled_arr_ret)
        if ret_res and isinstance(ret_res, TaxiwayResource):
            events_time = event.time + agent.arr_rot_sec
        else:
            events_time = event.time + agent.arr_rot_sec
        event_queue.push(Event(
            time=event.time + agent.arr_rot_sec,
            event_type="RET_ENTER",
            agent=agent,
            resource=resources.get(agent.sampled_arr_ret) or event.resource,
        ))

    # --- RET_ENTER: travel along RET taxiway ---
    elif event.event_type == "RET_ENTER" and agent.state == "RET":
        ret_res = resources.get(agent.sampled_arr_ret)
        if ret_res and isinstance(ret_res, TaxiwayResource):
            ret_verts = ret_res.vertices_cells()
            if ret_verts and len(ret_verts) >= 2:
                agent.path_queue = list(ret_verts)
                agent.col, agent.row = ret_verts[0]
                ret_speed = agent.arr_v_ret_out_ms / max(cell_size, 1.0)
                agent.speed_ms = ret_speed
                agent.decel_rate = 0.0
                agent.accel_rate = 0.0
                agent.target_speed_ms = ret_speed

                ret_len_m = _polyline_length(ret_verts) * cell_size
                ret_dur = ret_len_m / max(agent.arr_v_ret_out_ms, 1.0)

                event_queue.push(Event(
                    time=event.time + ret_dur,
                    event_type="TAXI_START",
                    agent=agent,
                    resource=ret_res,
                ))
                return

        event_queue.push(Event(
            time=event.time + 5.0,
            event_type="TAXI_START",
            agent=agent,
            resource=event.resource,
        ))

    # --- TAXI_START: taxi from current position to stand ---
    elif event.event_type == "TAXI_START" and agent.state == "TAXI_IN":
        stand_res = resources.get(agent.assigned_stand_id)
        if stand_res and isinstance(stand_res, StandResource):
            origin = (agent.col, agent.row)
            dest = (stand_res.col, stand_res.row)
            path = _plan_taxi_path(graph, origin, dest)
            agent.path_queue = path

            dist_m = sum(
                cell_size * math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0

            taxi_speed = DEFAULT_TAXI_SPEED_MS
            agent.speed_ms = taxi_speed / max(cell_size, 1.0)
            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms
            taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0

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
            agent.schedule.aobt_sec = event.time + agent.dwell_sec

    # --- PUSHBACK: taxi from stand to DEP runway start ---
    elif event.event_type == "PUSHBACK" and agent.state == "TAXI_OUT":
        dep_rwy_res = resources.get(agent.dep_runway_id)
        if dep_rwy_res and isinstance(dep_rwy_res, RunwayResource):
            verts = dep_rwy_res._vertices_cells()
            if agent.dep_runway_dir == "counter_clockwise":
                verts = list(reversed(verts))

            origin = (agent.col, agent.row)
            dest = verts[0] if verts else origin
            path = _plan_taxi_path(graph, origin, dest)
            agent.path_queue = path

            dist_m = sum(
                cell_size * math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0

            taxi_speed = DEFAULT_TAXI_SPEED_MS
            agent.speed_ms = taxi_speed / max(cell_size, 1.0)
            agent.decel_rate = 0.0
            agent.accel_rate = 0.0
            agent.target_speed_ms = agent.speed_ms
            taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0

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


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

@dataclass
class TimeRange:
    start: float
    end: float


def compute_time_range(agents: List[Flight]) -> TimeRange:
    times: list[float] = []
    for a in agents:
        s = a.schedule
        for t in (s.sldt_sec, s.sibt_sec, s.sobt_sec, s.stot_sec):
            if t is not None:
                times.append(t)
    if not times:
        return TimeRange(0.0, 0.0)
    return TimeRange(min(times) - 300, max(times) + 7200)


def run_simulation(
    layout: dict,
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> dict:
    global _SIM_CELL_SIZE

    information = _load_information_json()
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    dt = float(sim_cfg.get("timeStepSec", dt))
    cell_size = layout.get("grid", {}).get("cellSize", 20.0)
    _SIM_CELL_SIZE = cell_size

    global APPROACH_OFFSET_M, DEP_LINEUP_HOLD_SEC, DEP_TAKEOFF_ACCEL_MS2
    APPROACH_OFFSET_M = _safe_float(sim_cfg.get("approachOffsetM"), 10_000.0)
    flight_cfg = _deep_get(information, "tiers", "flight_schedule", default={})
    DEP_LINEUP_HOLD_SEC = _safe_float(flight_cfg.get("depLineupHoldSec"), 20.0)
    DEP_TAKEOFF_ACCEL_MS2 = _safe_float(flight_cfg.get("depTakeoffAccelSmallMs2"), 2.5)

    base_date = str(_deep_get(information, "tiers", "algorithm", "simulation", "baseDate",
                              default="2026-03-31"))

    graph = build_graph(layout, information)
    resources = build_resources(layout, information)
    flights_raw = layout.get("flights", [])

    if not flights_raw:
        return _build_output(layout, [], information, base_date)

    agents = build_agents(flights_raw, graph, resources, information)
    if not agents:
        return _build_output(layout, [], information, base_date)

    event_queue = EventQueue()
    for agent in agents:
        ev = agent.create_initial_event(resources)
        if ev:
            event_queue.push(ev)

    time_range = compute_time_range(agents)
    current_time = time_range.start
    total_end = time_range.end

    record_interval = max(dt, 1.0)
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

                _handle_post_event(agent, event, result, graph, resources, event_queue, cell_size)

        for agent in agents:
            if not agent.is_stationary():
                _move_agent_along_path(agent, dt, cell_size)

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

    return _build_output(layout, agents, information, base_date)


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


def _build_output(layout: dict, agents: List[Flight], information: dict, base_date: str) -> dict:
    positions: Dict[str, list] = {}
    schedule_list: list = []

    taxi_in_times: list[float] = []
    taxi_out_times: list[float] = []
    total_delay: float = 0.0

    for a in agents:
        positions[a.id] = [
            {"t": round(t, 1), "col": round(c, 3), "row": round(r, 3)}
            for t, c, r in a.history
        ]

        sched_entry = {
            "flight_id": a.id,
            "reg": a.reg,
            "flight_number": a.flight_number,
            "aircraft_type": a.aircraft_type,
            "SLDT": a.schedule.sldt_sec,
            "SIBT": a.schedule.sibt_sec,
            "SOBT": a.schedule.sobt_sec,
            "STOT": a.schedule.stot_sec,
            "ALDT": a.schedule.aldt_sec,
            "AIBT": a.schedule.aibt_sec,
            "AOBT": a.schedule.aobt_sec,
            "ATOT": a.schedule.atot_sec,
        }
        for key in ("SLDT", "SIBT", "SOBT", "STOT", "ALDT", "AIBT", "AOBT", "ATOT"):
            sched_entry[f"{key}_dt"] = _sec_to_datetime_str(sched_entry.get(key), base_date)
        schedule_list.append(sched_entry)

        if a.schedule.aldt_sec is not None and a.schedule.aibt_sec is not None:
            taxi_in_times.append(a.schedule.aibt_sec - a.schedule.aldt_sec)
        if a.schedule.aobt_sec is not None and a.schedule.atot_sec is not None:
            taxi_out_times.append(a.schedule.atot_sec - a.schedule.aobt_sec)
        if a.schedule.aldt_sec is not None and a.schedule.sldt_sec is not None:
            delay = a.schedule.aldt_sec - a.schedule.sldt_sec
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
        "layout": layout,
        "baseDate": base_date,
        "positions": positions,
        "schedule": schedule_list,
        "kpi": kpi_data,
    }
