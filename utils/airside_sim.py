"""
Airside Simulation Engine — Agent-Based + Discrete Event Simulation.

Reads a serialized layout (from Layout_Design) and Information.json,
runs a time-stepped DES loop where each Flight is an Agent interacting
with shared Resources (Runway, Taxiway, Stand, etc.), and returns:
  - per-flight position timeline  (col, row at each time step)
  - A-schedule                    (ALDT, AIBT, AOBT, ATOT per flight)
  - KPI aggregation               (utilization, delay, throughput …)
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


def _minutes_to_sec(m: Optional[float]) -> Optional[float]:
    return m * 60.0 if m is not None else None


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
    collision_info: dict = field(default_factory=dict)
    assigned_stand_id: Optional[str] = None
    arr_runway_id: Optional[str] = None
    dep_runway_id: Optional[str] = None
    path_queue: list = field(default_factory=list)
    dwell_sec: float = 2700.0
    arr_rot_sec: float = 55.0
    token: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict, repr=False)

    def record_position(self, t: float) -> None:
        self.history.append((t, self.col, self.row))

    def is_stationary(self) -> bool:
        return self.state in ("PARKED", "SCHEDULED", "DEPARTED")

    # --- agent reaction methods -------------------------------------------

    def update_state(self, event: "Event") -> None:
        _STATE_MAP = {
            "ARRIVAL_REQUEST": "APPROACH",
            "LANDING": "LANDING",
            "TAXI_START": "TAXI_IN",
            "STAND_ENTER": "PARKED",
            "PUSHBACK": "TAXI_OUT",
            "TAXI_OUT_START": "TAXI_OUT",
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

    def _get_sep_sec(self, prev_op: str, cur_op: str, prev_cat: str, cur_cat: str) -> float:
        key = f"{prev_op}\u2192{cur_op}"
        matrix = self.separation_matrix.get(key, {})
        if isinstance(matrix, dict):
            row = matrix.get(prev_cat, {})
            if isinstance(row, dict):
                return float(row.get(cur_cat, 90))
            return float(row) if row else 90.0
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
                if self.vertices:
                    v = self.vertices[0]
                    agent.col, agent.row = v.get("col", 0), v.get("row", 0)
                landing_dur = agent.arr_rot_sec
                events.append(Event(
                    time=event.time + landing_dur,
                    event_type="TAXI_START",
                    agent=agent,
                    resource=self,
                ))
            elif event.event_type == "TAKEOFF_REQUEST":
                self.last_operation = ("DEP", event.time, agent.icao_category)
                agent.schedule.atot_sec = event.time
                if self.vertices:
                    v = self.vertices[-1]
                    agent.col, agent.row = v.get("col", 0), v.get("row", 0)
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

    node_counter = [0]

    def _vertex_id(col: float, row: float) -> str:
        key = f"v_{col:.2f}_{row:.2f}"
        if key not in graph.nodes:
            graph.add_node(GraphNode(id=key, col=col, row=row))
        return key

    all_taxiways = []
    for tw in layout.get("runwayPaths", []):
        tw = dict(tw, pathType="runway")
        all_taxiways.append(tw)
    for tw in layout.get("runwayTaxiways", []):
        tw = dict(tw, pathType="runway_taxiway")
        all_taxiways.append(tw)
    for tw in layout.get("taxiways", []):
        pt = tw.get("pathType", "taxiway")
        tw = dict(tw, pathType=pt)
        all_taxiways.append(tw)

    for tw in all_taxiways:
        verts = tw.get("vertices", [])
        if len(verts) < 2:
            continue
        tw_id = tw.get("id", "")
        direction = tw.get("direction", "both")
        tw_name = tw.get("name", "")

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
        resources[rw_id] = RunwayResource(
            id=rw_id,
            name=rw.get("name", ""),
            separation_matrix=sep_defaults,
            rot_defaults=rot_defaults,
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
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            stand_type="pbb",
            col=st.get("col", 0), row=st.get("row", 0),
        )

    for st in layout.get("remoteStands", []):
        st_id = st.get("id", "")
        resources[st_id] = StandResource(
            id=st_id, name=st.get("name", ""),
            stand_type="remote",
            col=st.get("col", 0), row=st.get("row", 0),
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
        fid = f.get("id", "")
        cat = f.get("code", "M")
        rot_sec = float(f.get("arrRotSec", rot_defaults.get(cat, 55)))
        dwell_sec = float(f.get("dwellMin", 45)) * 60.0

        token = f.get("token", {}) or {}
        arr_rwy = f.get("arrRunwayIdUsed") or token.get("arrRunwayId") or token.get("runwayId")
        dep_rwy = token.get("depRunwayId") or arr_rwy
        stand_id = f.get("standId") or token.get("apronId")

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
# Movement interpolation
# ---------------------------------------------------------------------------

def _move_agent_along_path(agent: Flight, dt: float, cell_size: float) -> None:
    """Advance agent along its path_queue by speed_ms * dt (in cell coords)."""
    if agent.is_stationary() or not agent.path_queue:
        return
    remaining = agent.speed_ms * dt / max(cell_size, 1.0)
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

def _plan_taxi_path(agent: Flight, graph: AirsideGraph, resources: Dict[str, Resource],
                    origin: Tuple[float, float], dest: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Find path from origin to dest using the graph. Falls back to straight line."""
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
            return [(origin[0], origin[1])] + coords + [(dest[0], dest[1])]

    return [(origin[0], origin[1]), (dest[0], dest[1])]


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
    return TimeRange(min(times) - 300, max(times) + 600)


def run_simulation(
    layout: dict,
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> dict:
    information = _load_information_json()
    sim_cfg = _deep_get(information, "tiers", "algorithm", "simulation", default={})
    dt = float(sim_cfg.get("timeStepSec", dt))
    cell_size = layout.get("grid", {}).get("cellSize", 20.0)

    graph = build_graph(layout, information)
    resources = build_resources(layout, information)
    flights_raw = layout.get("flights", [])

    if not flights_raw:
        return _build_output(layout, [], information)

    agents = build_agents(flights_raw, graph, resources, information)
    if not agents:
        return _build_output(layout, [], information)

    event_queue = EventQueue()
    for agent in agents:
        ev = agent.create_initial_event(resources)
        if ev:
            event_queue.push(ev)

    time_range = compute_time_range(agents)
    current_time = time_range.start
    total_end = time_range.end

    active_agents_arr: np.ndarray = np.zeros((len(agents), 2), dtype=np.float64)

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

    return _build_output(layout, agents, information)


def _handle_post_event(agent: Flight, event: Event, result: str,
                       graph: AirsideGraph, resources: Dict[str, Resource],
                       event_queue: EventQueue, cell_size: float) -> None:
    """Generate follow-up events for state transitions that need taxi planning."""
    if result != "ACCEPT":
        return

    if event.event_type == "TAXI_START" and agent.state == "TAXI_IN":
        stand_res = resources.get(agent.assigned_stand_id)
        if stand_res and isinstance(stand_res, StandResource):
            origin = (agent.col, agent.row)
            dest = (stand_res.col, stand_res.row)
            path = _plan_taxi_path(agent, graph, resources, origin, dest)
            agent.path_queue = path
            dist_m = sum(
                cell_size * math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0
            taxi_speed = 10.0
            taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0
            agent.speed_ms = taxi_speed
            event_queue.push(Event(
                time=event.time + taxi_time,
                event_type="STAND_ENTER",
                agent=agent,
                resource=stand_res,
            ))
        else:
            agent.state = "PARKED"
            agent.schedule.aibt_sec = event.time
            agent.schedule.aobt_sec = event.time + agent.dwell_sec

    elif event.event_type == "PUSHBACK" and agent.state == "TAXI_OUT":
        dep_rwy_res = resources.get(agent.dep_runway_id)
        if dep_rwy_res and isinstance(dep_rwy_res, RunwayResource):
            origin = (agent.col, agent.row)
            if dep_rwy_res.vertices:
                v0 = dep_rwy_res.vertices[0]
                dest = (v0.get("col", 0), v0.get("row", 0))
            else:
                dest = origin
            path = _plan_taxi_path(agent, graph, resources, origin, dest)
            agent.path_queue = path
            dist_m = sum(
                cell_size * math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                for i in range(len(path) - 1)
            ) if len(path) > 1 else 0
            taxi_speed = 10.0
            taxi_time = dist_m / max(taxi_speed, 0.1) if dist_m > 0 else 60.0
            agent.speed_ms = taxi_speed
            event_queue.push(Event(
                time=event.time + taxi_time,
                event_type="TAKEOFF_REQUEST",
                agent=agent,
                resource=dep_rwy_res,
            ))
        else:
            agent.state = "DEPARTED"
            agent.schedule.atot_sec = event.time

    elif event.event_type == "DEPARTED":
        agent.state = "DEPARTED"
        agent.speed_ms = 0


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------

def _build_output(layout: dict, agents: List[Flight], information: dict) -> dict:
    positions: Dict[str, list] = {}
    schedule_list: list = []
    kpi_data: dict = {}

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
        schedule_list.append(sched_entry)

        if a.schedule.aldt_sec is not None and a.schedule.aibt_sec is not None:
            taxi_in = a.schedule.aibt_sec - a.schedule.aldt_sec
            taxi_in_times.append(taxi_in)
        if a.schedule.aobt_sec is not None and a.schedule.atot_sec is not None:
            taxi_out = a.schedule.atot_sec - a.schedule.aobt_sec
            taxi_out_times.append(taxi_out)
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
        "positions": positions,
        "schedule": schedule_list,
        "kpi": kpi_data,
    }
