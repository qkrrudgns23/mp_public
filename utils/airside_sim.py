"""
Airside simulation: Dijkstra paths on the Layout_Design path graph, then a time-step loop (no DES
events) moving each flight along edge polylines with per-segment ``avgMoveVelocity``, landing
deceleration, and runway-exit decel (see ``layoutPixelsPerMeter`` in Information.json for px/m scale).

Schedule inputs: S series (``*_Min_orig``) and Sd series (``*_Min_d`` minutes) are read from each
flight; routing and time axis use Sd only (``eldtMin_d`` or ``sldtMin_d`` → ELDT anchor in sim
seconds). Same arrival runway: ``ELDT`` is pushed forward so the next landing is not earlier than
the previous touchdown plus landing-leg duration (touchdown through end of Landing micro-legs)
plus ``RWY_ARRIVAL_SPACING_BUFFER_SEC`` (taxing / hold-wait margin). Outputs ``schedule`` with S, Sd echo,
E times (ELDT/EIBT/EOBT/ETOT); ``EIBT`` is **only** from simulation: first time within ``standArrivalStopRadiusM`` of
the stand token on ``Arr_taxi`` (any path type except pure ``runway``) with speed below ``standStoppedVelocityMaxMs``, only when
``PROCEED`` (not ``WAIT``/``YIELD``) and the destination stand pipeline has capacity (no cooldown). Phase-change
fallback if stand px is missing. No schedule/path nominal fill-in for EIBT. ``EOBT`` is first time on Dep_taxi ``apron_link`` after
``dep_taxi_start_abs_sec`` (set at in-blocks + dwell) with speed above a small threshold, else path timing + ``dwellMin`` when not
recorded.
``ETOT`` (when the path includes a ``Lineup_departure`` / takeoff-roll leg) is **only** the simulated time when the flight
finishes the last path segment (``path_completed_abs_sec``); no nominal EOBT+leg projection. If the sim never completes that path,
``ETOT`` is omitted (null).

``positions`` timelines use ``t`` = absolute schedule seconds (day base, same as ELDT scale);
``Dep_taxi`` pushback/taxi-out is gated until **physical** in-blocks at stand + ``dwell_sec`` (not ELDT+nominal taxi-in).
Between heavy control ticks, a full reservation rebook runs every ``LIGHT_RESERVATION_RETRY_INTERVAL_SEC`` so agents
re-check stand pipeline and departure-runway resources and regain ``PROCEED`` when slots free (same rules as ``can_reserve_path``).
Per-flight lookahead / billed reservation depth by regime (runway, Dep_taxi, Arr_taxi, Arr_taxi stand-busy after ELDT);
``apron_taxiway`` edges use billed depth where those edges
add no slot; consecutive ``taxiway`` edges still share one slot. Physical edges remain reserved along the path.
Each apron stand has configurable capacity (layout ``pbbStands[].capacity`` or ``defaultApronStandCapacity``);
Arr_taxi waits when the reservation lookahead reaches an ``apron_link`` edge and the stand pipeline is full (Landing/rollout is not blocked by stand capacity).
``v`` is m/s.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from utils.designer_path_graph import (
    DirectedEdgeRecord,
    PathGraph,
    SPLIT_TOL_D2,
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

Point = Tuple[float, float]
DestinationStandHistorySnap = Tuple[str, int, int, int, bool, bool]
PHASE_LANDING = "Landing"
PHASE_ARR_TAXI = "Arr_taxi"
PHASE_ARR_TAXI_TEMP = "Arr_taxi_occupied"
PHASE_DEP_TAXI = "Dep_taxi"
PHASE_HOLDING_LINEUP = "Holding_lineup"
PHASE_LINEUP_DEPARTURE = "Lineup_departure"
_DEFAULT_RW_DIR = "clockwise"


_EXTRACT_LEG_PHASES: Tuple[str, ...] = (
    PHASE_LANDING,
    PHASE_ARR_TAXI,
    PHASE_DEP_TAXI,
    PHASE_HOLDING_LINEUP,
    PHASE_LINEUP_DEPARTURE,
)
SIM_MAX_TIME_SEC = 200_000.0
# After max scheduled STOT (Sd / ``stotMin_d``), advance sim time only this much (absolute seconds).
STOT_POST_BUFFER_SEC = 3_600.0

_LOG = logging.getLogger(__name__)

TAXI_SPEED_MPS = 15.0
MIN_LANDING_VELOCITY_MS = 15.0
MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS = 15.0
ARR_RET_DECEL_MS2 = 0.5
DEFAULT_EDGE_CAPACITY = 1
DEFAULT_INTERSECTION_CAPACITY = 1
DEFAULT_RUNWAY_CAPACITY = 1
DEFAULT_STAND_CAPACITY = 1
DEFAULT_MIN_SEPARATION_M = 60.0
# Lookahead edge count (T) and billed reservation depth per movement regime.
# Runway: Landing, or first segment ``runway`` / ``runway_taxiway`` on Arr_taxi / Dep_taxi, or
# Holding_lineup / Lineup_departure on those path types.
LOOKAHEAD_RUNWAY = 0
RESERV_DEPTH_RUNWAY = 0
# Dep_taxi (and holding/lineup when first segment is not runway above).
LOOKAHEAD_DEP_TAXI = 6
RESERV_DEPTH_DEP_TAXI = 4
# Arr_taxi (default).
LOOKAHEAD_ARR_TAXI = 6
RESERV_DEPTH_ARR_TAXI = 4
# Arr_taxi when sim ≥ ELDT and target stand occupied / pushback cooldown.
LOOKAHEAD_ARR_TAXI_BUSY = 12
RESERV_DEPTH_ARR_TAXI_BUSY = 6
# Upper bound for failsafe resource collection and control_state default.
LOOKAHEAD_EDGE_COUNT_MAX = max(
    LOOKAHEAD_RUNWAY,
    LOOKAHEAD_DEP_TAXI,
    LOOKAHEAD_ARR_TAXI,
    LOOKAHEAD_ARR_TAXI_BUSY,
)
RWY_ARRIVAL_SPACING_BUFFER_SEC = 25 #### 홪주로 이탈시간 여유분분
HEAVY_DECISION_INTERVAL_SEC = 15
# Between heavy ticks, re-run full reservation pass this often so Arr_taxi / Dep_taxi agents
# re-observe stand + departure-runway resources and regain PROCEED when slots free.
LIGHT_RESERVATION_RETRY_INTERVAL_SEC = 1.0
DEADLOCK_THRESHOLD_SEC = 300.0
DEADLOCK_FORCE_MOVE_DURATION_SEC = 60.0
DEADLOCK_RESOLVE_STOP_COUNT = 3
STAGNATION_PROGRESS_EPS_M = 2.0
# After pushback from a stand, block other arrivals to that stand for this many seconds.
STAND_POST_PUSHBACK_CLEARANCE_DELAY_SEC = 60.0
# After destination apron becomes free, temp-stand arrivals wait this long before taxi-in.
TEMP_TO_APRON_HOLD_SEC = 420.0  # 7 minutes
# Evaluate arrival temp-stand detour only near touchdown (ELDT - lead .. ELDT).
ARR_TEMP_DETOUR_DECISION_LEAD_SEC = 120.0
REROUTE_WAIT_THRESHOLD_SEC = 60.0
REROUTE_IMPROVEMENT_RATIO = 0.2
REROUTE_MAX_ATTEMPTS = 25
REVERSE_PENALTY_COST = 1_000_000.0
REROUTE_YIELD_EDGE_PENALTY = REVERSE_PENALTY_COST
REROUTE_MIN_OLD_PATH_M = 50.0
NODE_OCCUPANCY_RADIUS_M = 12.0
# Departure hold gate distance (m, along-path). On a ``runway_taxiway`` / ``runway_exit`` segment
# leading to a runway, if the departure runway is busy and the remaining along-path distance to
# the runway entry is within this buffer, the agent holds here. Keeps the stop point at a
# predictable distance from the runway regardless of polyline vertex density / graph node spacing
# along the approach lane. Tuned to ~100 m of path so the physical (perpendicular) distance from
# the runway polyline at the hold position is on the order of standard runway-holding offsets
# (roughly 50 m for typical rapid-exit taxiway angles).
DEP_RUNWAY_HOLD_BUFFER_M = 100.0

# Cleared at the start of each ``run_simulation``; keyed by layout object id + path-search params.
_PATH_GRAPH_BUILD_CACHE: Dict[Tuple[int, str, float, float, float, float, bool], PathGraph] = {}


def _stable_tie_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    d = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return int(d, 16) & 0x7FFFFFFF

# --- Global movement priority (lower = higher priority) ---
# Single source for decision ordering, head-on yield, and AgentControlState.priority_rank.
AGENT_PRIORITY_RUNWAY_ARR_DEP = 1  # 활주로 출발·착륙·활주로/연계 구간
AGENT_PRIORITY_LINEUP_HOLDING = 2  # 라인업·홀딩
AGENT_PRIORITY_DEP_TAXI = 3  # Dep_taxi (일반 택시)
AGENT_PRIORITY_ARR_TAXI = 4  # Arr_taxi (일반 택시)
AGENT_PRIORITY_APRON_TRANSIT = 5  # 주기장(apron_link / apron_taxiway)
AGENT_PRIORITY_UNKNOWN = 99


def _sim_default_stand_capacity(information: Dict[str, Any]) -> int:
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if not isinstance(sim, dict):
        return max(1, int(DEFAULT_STAND_CAPACITY))
    raw = sim.get("defaultApronStandCapacity", DEFAULT_STAND_CAPACITY)
    try:
        return max(1, min(99, int(raw)))
    except (TypeError, ValueError):
        return max(1, int(DEFAULT_STAND_CAPACITY))


def _sim_stand_arrival_stop_radius_m(information: Dict[str, Any]) -> float:
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if not isinstance(sim, dict):
        return 22.0
    raw = sim.get("standArrivalStopRadiusM", 22.0)
    try:
        r = float(raw)
    except (TypeError, ValueError):
        return 22.0
    return max(3.0, min(200.0, r))


def _sim_stand_stopped_velocity_max_ms(information: Dict[str, Any]) -> float:
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if not isinstance(sim, dict):
        return 0.12
    raw = sim.get("standStoppedVelocityMaxMs", 0.12)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.12
    return max(0.01, min(2.0, v))


def _layout_stand_capacity_for_id(
    layout: Dict[str, Any], stand_id: str, information: Dict[str, Any]
) -> int:
    default_c = _sim_default_stand_capacity(information)
    st = find_stand_by_id(layout, str(stand_id))
    if not st or not isinstance(st, dict):
        return default_c
    raw = st.get("capacity")
    if raw is None:
        return default_c
    try:
        return max(1, min(99, int(raw)))
    except (TypeError, ValueError):
        return default_c
FOLLOW_REACTION_SEC = 2.0
FOLLOW_GAP_BUFFER_M = 5.0
_HEAD_ON_COS_THRESHOLD = -0.05
_SAME_DIR_COS_THRESHOLD = 0.5


def _arr_ret_decel_floor_ms(phase: str, path_type: str, accel_ms2: float) -> float:
    """지정 RET(Arr_taxi·runway_exit·감속): 착륙/고속탈출과 동일 최소 속도 바닥."""
    if (
        phase in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP)
        and str(path_type or "") == "runway_exit"
        and float(accel_ms2) < -1e-12
    ):
        return float(MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
    return 0.0


def _runway_ops_dir_from_layout(layout: Dict[str, Any], runway_id: str) -> str:
    rid = str(runway_id).strip()
    if not rid:
        raise ValueError("runway_id is required to resolve runway direction")
    for rw in layout.get("runwayPaths") or []:
        if not isinstance(rw, dict) or str(rw.get("id", "")).strip() != rid:
            continue
        nd = normalize_rw_direction_value(str(rw.get("direction") or ""))
        if nd in ("clockwise", "counter_clockwise"):
            return nd
        raise ValueError(f"runway direction missing/invalid for runway_id={rid!r}")
    raise ValueError(f"runway not found for runway_id={rid!r}")


def _flight_rw_dir_for_leg(
    flight: Dict[str, Any], leg_index: int, layout: Dict[str, Any]
) -> str:
    """
    Operations direction matching Layout_Design path graph export:
    ``simPathGraph.clockwise`` vs ``simPathGraph.counter_clockwise``.

    Legs 0–1 use arrival runway direction; legs 2+ use departure when present, else arrival.
    Reads ``arrRunwayDirUsed`` / ``arrRunwayDir``, ``depRunwayDirUsed`` / ``depRunwayDir``, and token.
    """
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    if leg_index >= 2:
        dep_rwy = flight.get("depRunwayId") or token.get("depRunwayId")
        if dep_rwy is None or str(dep_rwy).strip() == "":
            raise ValueError(
                f"depRunwayId missing for flight_id={str(flight.get('id', ''))!r}"
            )
        return _runway_ops_dir_from_layout(layout, str(dep_rwy))
    arr_rwy = flight.get("arrRunwayId") or token.get("arrRunwayId")
    if arr_rwy is None or str(arr_rwy).strip() == "":
        raise ValueError(
            f"arrRunwayId missing for flight_id={str(flight.get('id', ''))!r}"
        )
    return _runway_ops_dir_from_layout(layout, str(arr_rwy))


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


def _max_stot_sd_sec(flights_raw: List[Any]) -> Optional[float]:
    """Largest ``stotMin_d`` (Sd → sim seconds) among flights, or ``None`` if none set."""
    max_stot: Optional[float] = None
    for fobj in flights_raw:
        if not isinstance(fobj, dict):
            continue
        st = _schedule_sd_sec(fobj, "stotMin_d")
        if st is None:
            continue
        v = float(st)
        max_stot = v if max_stot is None else max(max_stot, v)
    return max_stot


def _sim_progress_elapsed_total_sec(
    flights_raw: List[Any], ref_t0: float
) -> float:
    """Pro Sim progress denominator (same units as ``current_time_abs - ref_t0``).

    ``max(STOT_sd) + STOT_POST_BUFFER_SEC - ref_t0`` where ``STOT_sd`` is ``stotMin_d`` (Sd axis).
    Matches the main-loop time horizon when STOT is present. If no ``stotMin_d``, ``SIM_MAX_TIME_SEC``.
    """
    max_stot = _max_stot_sd_sec(flights_raw)
    if max_stot is None:
        return float(SIM_MAX_TIME_SEC)
    span = float(max_stot) + float(STOT_POST_BUFFER_SEC) - float(ref_t0)
    if not math.isfinite(span) or span <= 1e-6:
        return float(SIM_MAX_TIME_SEC)
    return float(span)


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
    tw_info = _deep_get(information, "tiers", "layout", "taxiway", default={}) or {}
    try:
        q_js = float(tw_info.get("queueJunctionSpacingM", 40.0))
    except (TypeError, ValueError):
        q_js = 40.0
    path_graph_opts = {"queueTaxiwayJunctionSpacingM": max(5.0, q_js)}
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
        path_graph_opts,
    )


def _cached_path_graph_for_direction(
    layout: Dict[str, Any],
    cell_size: float,
    runway_ops_dir: str,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    *,
    pure_ground_exclude_runway: bool,
) -> Optional[PathGraph]:
    nd = normalize_rw_direction_value(str(runway_ops_dir).strip() if runway_ops_dir else "")
    if nd not in ("clockwise", "counter_clockwise"):
        nd = _DEFAULT_RW_DIR
    key = (
        id(layout),
        nd,
        float(cell_size),
        float(reverse_cost),
        float(merge_r),
        float(taxiway_h),
        bool(pure_ground_exclude_runway),
    )
    hit = _PATH_GRAPH_BUILD_CACHE.get(key)
    if hit is not None:
        return hit
    g = _graph_for_direction(
        layout,
        cell_size,
        nd,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        pure_ground_exclude_runway=pure_ground_exclude_runway,
    )
    if g is not None:
        _PATH_GRAPH_BUILD_CACHE[key] = g
    return g


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
    segment_graph_uv: List[Tuple[int, int]] = field(default_factory=list)
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
    history: List[Tuple[float, float, float, float, bool, bool]] = field(
        default_factory=list
    )
    eldt_anchor_sec: Optional[float] = None
    eldt_raw_sec: Optional[float] = None
    dwell_sec: float = 0.0
    # Apron in-blocks: Arr_taxi → Dep_taxi segment transition (arrival complete at stand).
    actual_apron_inblocks_abs_sec: Optional[float] = None
    # Apron off-blocks: first Dep_taxi motion (pushback / taxi-out start from stand).
    actual_apron_offblocks_abs_sec: Optional[float] = None
    apron_stand_id: Optional[str] = None
    dep_taxi_start_sim_time: Optional[float] = None
    dep_taxi_start_abs_sec: Optional[float] = None
    arr_runway_id: Optional[str] = None
    arr_runway_dir: Optional[str] = None
    dep_runway_id: Optional[str] = None
    runway_entry_abs_sec: Optional[float] = None
    lineup_hold_release_abs_sec: Optional[float] = None
    path_completed_abs_sec: Optional[float] = None
    exit_runway_abs_sec: Optional[float] = None
    # 터치다운(eldt)부터 Landing 마이크로 구간 종료까지 시초(내부 간격·활주로 시간 점유).
    runway_rot_sec: float = 0.0
    motion_integrated_until_abs_sec: Optional[float] = None
    fsm_state: str = "TAXI"
    current_edge_id: Optional[str] = None
    next_edge_id: Optional[str] = None
    heading_rad: Optional[float] = None
    # True = nose along segment p0→p1; False = reverse (e.g. tow on apron_link).
    motion_is_forward: bool = True
    segment_graph_uv: List[Tuple[int, int]] = field(default_factory=list)
    completed_directed_hops: List[Tuple[str, int, int, str]] = field(default_factory=list)
    control_halt: bool = False
    control_speed_cap_ms: Optional[float] = None
    temp_stand_id: Optional[str] = None
    awaiting_apron_from_temp: bool = False
    temp_park_arrival_trigger_global_reroute: bool = False
    post_temp_route_tail_prep: Optional[PreparedFlightPath] = None
    arr_temp_detour_decided: bool = False
    # Absolute time when destination apron was first observed unoccupied (inject gated by TEMP_TO_APRON_HOLD_SEC).
    temp_dest_apron_cleared_abs_sec: Optional[float] = None


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
    path_type: str = "taxiway"


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
class StandResource:
    stand_id: str
    capacity: int = DEFAULT_STAND_CAPACITY
    occupied_by: List[str] = field(default_factory=list)


@dataclass
class AgentControlState:
    flight_id: str
    priority_rank: int = AGENT_PRIORITY_UNKNOWN
    clearance: str = "PROCEED"
    wait_reason: Optional[str] = None
    reserved_edges: List[str] = field(default_factory=list)
    reserved_intersections: List[str] = field(default_factory=list)
    wait_start_sec: Optional[float] = None
    total_wait_sec: float = 0.0
    blocked_since_sec: Optional[float] = None
    reroute_attempts: int = 0
    deadlock_flag: bool = False
    deadlock_ghost_until_abs_sec: Optional[float] = None
    stagnation_anchor_sec: Optional[float] = None
    progress_snapshot_along_m: float = 0.0
    progress_snapshot_edge_id: Optional[str] = None


@dataclass
class SimulationControlState:
    edge_resources: Dict[str, EdgeResource] = field(default_factory=dict)
    intersection_resources: Dict[str, IntersectionResource] = field(default_factory=dict)
    runway_resources: Dict[str, RunwayResource] = field(default_factory=dict)
    stand_resources: Dict[str, StandResource] = field(default_factory=dict)
    agent_states: Dict[str, AgentControlState] = field(default_factory=dict)
    decision_interval_sec: float = HEAVY_DECISION_INTERVAL_SEC
    lookahead_edges: int = LOOKAHEAD_EDGE_COUNT_MAX
    deadlock_threshold_sec: float = DEADLOCK_THRESHOLD_SEC
    path_graph: Optional[PathGraph] = None
    pixels_per_meter: float = 1.0
    last_decision_sim_time: float = -1e30
    taxi_speed_assumed_mps: float = TAXI_SPEED_MPS
    deadlock_resolve_event_count: int = 0
    stand_arrival_book_snapshot: Dict[str, int] = field(default_factory=dict)
    last_light_reservation_rebook_sim_time: float = -1e30
    temp_stand_incident_edges: Dict[str, set[str]] = field(default_factory=dict)
    # Per simulation tick: _compute_arr_touchdown_motion_abs_sec per flight (invalidated after movement).
    touchdown_motion_by_id: Optional[Dict[str, Optional[float]]] = field(
        default=None, repr=False
    )


CLEARANCE_DEADLOCK_GHOST = "DEADLOCK_GHOST"


def _agent_deadlock_ghost_at_time(
    st: Optional[AgentControlState], sim_time_abs: float
) -> bool:
    if st is None:
        return False
    u = st.deadlock_ghost_until_abs_sec
    return u is not None and float(sim_time_abs) + 1e-9 < float(u)


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


def _apron_link_ids_for_assigned_stand(layout: Dict[str, Any], flight: Dict[str, Any]) -> Set[str]:
    """Layout ``apronLinks[].id`` for the stand assigned to ``flight`` (contact/remote/temp)."""
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    sid = flight.get("standId") or token.get("apronId")
    if sid is None or str(sid).strip() == "":
        return set()
    out: Set[str] = set()
    for al in layout.get("apronLinks") or []:
        if not isinstance(al, dict):
            continue
        if str(al.get("pbbId") or "") != str(sid):
            continue
        lid = str(al.get("id") or "").strip()
        if lid:
            out.add(lid)
    return out


def flight_route(
    g: PathGraph,
    layout: Dict[str, Any],
    cell_size: float,
    pair_index: Dict[Tuple[int, int], str],
    start_point: RouteEndpoint,
    end_point: RouteEndpoint,
    *,
    penalized_arcs: Optional[set[Tuple[int, int]]] = None,
    penalty_add: float = 0.0,
    apron_transit_extra: float = 0.0,
    apron_allowed_link_ids: Optional[Set[str]] = None,
) -> Tuple[List[str], float, Optional[List[int]]]:
    """
    Shortest path on ``g`` between two endpoints.

    Returns ``(edge_ids, path_length, node_path)``. ``node_path`` is ``None`` if unreachable.

    Per leg from ``extract_point_to_paths`` (e.g. leg 0 = touchdown → first RET exit junction ``A``).
    Touchdown spawn is applied later in ``run_simulation`` (``_split_flight_path_at_touchdown``).
    Departure (apron → runway): ``start_point`` = apron, ``end_point`` = runway lineup pixel.
    """
    start_idx = resolve_route_endpoint_index(g, layout, cell_size, start_point)
    end_idx = resolve_route_endpoint_index(g, layout, cell_size, end_point)
    if start_idx is None or end_idx is None:
        return [], float("inf"), None
    path = path_dijkstra(
        g,
        start_idx,
        end_idx,
        penalized_arcs=penalized_arcs,
        penalty_add=penalty_add,
        apron_transit_extra=apron_transit_extra,
        apron_allowed_link_ids=apron_allowed_link_ids,
    )
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


def _penalized_directed_arc_keys(
    layout: Dict[str, Any], penalized_eids: Optional[set[str]]
) -> Optional[set[Tuple[int, int]]]:
    if not penalized_eids:
        return None
    out: set[Tuple[int, int]] = set()
    raw = layout.get("Edge") or layout.get("edges") or []
    for ed in raw:
        if not isinstance(ed, dict):
            continue
        eid = str(ed.get("id") or "").strip()
        if eid not in penalized_eids:
            continue
        try:
            a = int(ed["fromIdx"])
            b = int(ed["toIdx"])
        except (KeyError, TypeError, ValueError):
            continue
        out.add((a, b))
        out.add((b, a))
    return out if out else None


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
    apron_transit_extra: float = 0.0,
    apron_allowed_link_ids: Optional[Set[str]] = None,
) -> Tuple[List[str], bool, Optional[List[int]], Optional[PathGraph]]:
    """Same graph build and routing as airside_sim_rev3 ``_flight_route``; returns path for geometry."""
    g = _cached_path_graph_for_direction(
        layout,
        cell_size,
        str(runway_ops_dir).strip() if runway_ops_dir else "",
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        pure_ground_exclude_runway=False,
    )
    if g is None or not g.nodes:
        return [], False, None, None
    p_arcs: Optional[set[Tuple[int, int]]] = None
    p_add = 0.0
    if penalized_layout_edges and float(penalty_add) > 0:
        p_arcs = _penalized_directed_arc_keys(layout, penalized_layout_edges)
        if p_arcs:
            p_add = float(penalty_add)
    edges, _dist, path = flight_route(
        g,
        layout,
        cell_size,
        pair_index,
        start,
        end,
        penalized_arcs=p_arcs,
        penalty_add=p_add,
        apron_transit_extra=apron_transit_extra,
        apron_allowed_link_ids=apron_allowed_link_ids,
    )
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
    """Layout px lineup point from persisted runwayPaths[].lineup_point only."""
    _ = (cell_size, runway_ops_dir)
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
    return None


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


def _stand_nose_heading_deg(layout: Dict[str, Any], cell_size: float, stand_id: str) -> Optional[float]:
    def _norm_deg(ang: float) -> float:
        return ((float(ang) + 180.0) % 360.0) - 180.0

    st = find_stand_by_id(layout, str(stand_id))
    if not st or not isinstance(st, dict):
        return None
    raw_angle = st.get("angleDeg")
    try:
        if raw_angle is not None:
            ang = float(raw_angle)
            if math.isfinite(ang):
                return _norm_deg(ang + 180.0)
    except (TypeError, ValueError):
        pass
    apron_xy = _apron_token_xy(layout, cell_size, str(stand_id))
    x1 = st.get("x1")
    y1 = st.get("y1")
    try:
        if apron_xy is not None and x1 is not None and y1 is not None:
            dx = float(apron_xy[0]) - float(x1)
            dy = float(apron_xy[1]) - float(y1)
            if dx * dx + dy * dy > 1e-9:
                ang = math.degrees(math.atan2(dy, dx)) + 180.0
                return _norm_deg(ang)
    except (TypeError, ValueError):
        pass
    return None


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


def _arr_ret_first_edge_far_xy(
    layout: Dict[str, Any],
    cell_size: float,
    ret_tw_id: Optional[str],
    runway_junction_xy: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if not ret_tw_id or not str(ret_tw_id).strip() or runway_junction_xy is None:
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
    ex_pts = _as_xy_pairs(get_ordered_points(ret_obj, layout, cell_size) or [])
    if len(ex_pts) < 2:
        return None
    jx, jy = float(runway_junction_xy[0]), float(runway_junction_xy[1])
    d0 = (ex_pts[0][0] - jx) ** 2 + (ex_pts[0][1] - jy) ** 2
    d1 = (ex_pts[-1][0] - jx) ** 2 + (ex_pts[-1][1] - jy) ** 2
    far = ex_pts[1] if d0 <= d1 else ex_pts[-2]
    return (float(far[0]), float(far[1]))


def _layout_network_junctions_xy(layout: Dict[str, Any]) -> List[Tuple[float, float]]:
    raw = layout.get("networkJunctions")
    if not isinstance(raw, list):
        return []
    out: List[Tuple[float, float]] = []
    for j in raw:
        if not isinstance(j, dict):
            continue
        try:
            x = float(j.get("x"))
            y = float(j.get("y"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _arr_ret_exit_first_junction_a_xy(
    layout: Dict[str, Any],
    cell_size: float,
    ret_tw_id: Optional[str],
    runway_junction_xy: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Point ``A`` for Landing / Arr_taxi: first ``networkJunction`` on the ExitTaxiway (RET) polyline
    when walking from the runway–RET junction along the exit (away from the runway-adjacent RET
    end). The runway–RET touch itself is excluded. If no junction qualifies, uses
    ``_arr_ret_first_edge_far_xy`` (runway-far RET vertex).
    """
    if not ret_tw_id or not str(ret_tw_id).strip() or runway_junction_xy is None:
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
    ex_pts = _as_xy_pairs(get_ordered_points(ret_obj, layout, cell_size) or [])
    if len(ex_pts) < 2:
        return None
    jx, jy = float(runway_junction_xy[0]), float(runway_junction_xy[1])
    d0 = (ex_pts[0][0] - jx) ** 2 + (ex_pts[0][1] - jy) ** 2
    d1 = (ex_pts[-1][0] - jx) ** 2 + (ex_pts[-1][1] - jy) ** 2
    i_near = 0 if d0 <= d1 else len(ex_pts) - 1
    i_far = len(ex_pts) - 1 if i_near == 0 else 0
    _pjx, _pjy, s_jw = _closest_on_polyline_with_cum_dist(ex_pts, (jx, jy))
    s_total = _polyline_total_length_px(ex_pts)
    on_tol = max(25.0, math.sqrt(float(SPLIT_TOL_D2)))
    exclude_rw_d2 = max(49.0, (float(cell_size) * 0.35) ** 2)
    margin = max(3.0, float(cell_size) * 0.15)
    increasing = i_far > i_near
    best: Optional[Tuple[float, float]] = None
    best_delta = float("inf")
    for qx, qy in _layout_network_junctions_xy(layout):
        if (qx - jx) ** 2 + (qy - jy) ** 2 <= exclude_rw_d2:
            continue
        _px, _py, s_q = _closest_on_polyline_with_cum_dist(ex_pts, (qx, qy))
        if math.hypot(qx - _px, qy - _py) > on_tol:
            continue
        if increasing:
            if s_q < s_jw + margin or s_q > s_total + 1e-6:
                continue
            delta = s_q - s_jw
        else:
            if s_q > s_jw - margin or s_q < -1e-6:
                continue
            delta = s_jw - s_q
        if delta < best_delta:
            best_delta = delta
            best = (qx, qy)
    if best is not None:
        return best
    return _arr_ret_first_edge_far_xy(layout, cell_size, ret_tw_id, runway_junction_xy)


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


def _lineup_clearance_hold_sec(information: Dict[str, Any]) -> float:
    v = _deep_get(information, "tiers", "algorithm", "simulation", "lineupClearanceHoldSec")
    if v is not None:
        try:
            x = float(v)
            if math.isfinite(x) and x >= 0:
                return x
        except (TypeError, ValueError):
            pass
    return 20.0


def _dep_takeoff_accel_ms2(information: Dict[str, Any]) -> float:
    v = _deep_get(information, "tiers", "algorithm", "simulation", "depTakeoffAccelMs2")
    if v is not None:
        try:
            x = float(v)
            if math.isfinite(x) and x > 0:
                return x
        except (TypeError, ValueError):
            pass
    return 2.0


def _holding_point_kind_runway_holding(hp: Dict[str, Any]) -> bool:
    k = hp.get("hpKind")
    return str(k).strip() == "runway_holding"


def _holding_point_xy_px(hp: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not isinstance(hp, dict):
        return None
    try:
        x, y = float(hp.get("x")), float(hp.get("y"))
    except (TypeError, ValueError):
        return None
    if math.isfinite(x) and math.isfinite(y):
        return (x, y)
    return None


def _polyline_touches_polyline_for_graph(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> bool:
    touch_d2 = max(float(SPLIT_TOL_D2), 4.0)
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            ip = segment_segment_intersection(a[i], a[i + 1], b[j], b[j + 1])
            if ip is not None:
                return True
    for i in range(len(a) - 1):
        for vb in b:
            if (
                _point_segment_distance_sq(vb[0], vb[1], a[i][0], a[i][1], a[i + 1][0], a[i + 1][1])
                <= touch_d2
            ):
                return True
    for j in range(len(b) - 1):
        for va in a:
            if (
                _point_segment_distance_sq(va[0], va[1], b[j][0], b[j][1], b[j + 1][0], b[j + 1][1])
                <= touch_d2
            ):
                return True
    return False


def _point_near_polyline_sq(
    px: float, py: float, verts: List[Tuple[float, float]], tol_d2: float
) -> bool:
    if len(verts) < 2:
        return False
    for i in range(len(verts) - 1):
        if _point_segment_distance_sq(px, py, verts[i][0], verts[i][1], verts[i + 1][0], verts[i + 1][1]) <= tol_d2:
            return True
    return False


def _layout_runway_object_for_lineup_expansion(
    layout: Dict[str, Any], runway_id: str
) -> Optional[Dict[str, Any]]:
    """Runway centerline source: ``taxiways`` entry with ``pathType`` runway, else ``runwayPaths``."""
    rid = str(runway_id).strip()
    for tw in layout.get("taxiways") or []:
        if isinstance(tw, dict) and str(tw.get("id", "")) == rid and tw.get("pathType") == "runway":
            return tw
    for rw in layout.get("runwayPaths") or []:
        if isinstance(rw, dict) and str(rw.get("id", "")) == rid:
            return rw
    return None


def _iter_layout_rtx_objects(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runway taxiway polylines: ``taxiways`` with ``runway_exit`` / ``runway_taxiway``, then ``runwayTaxiways`` (dedup by id)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for tx in layout.get("taxiways") or []:
        if not isinstance(tx, dict):
            continue
        pt = str(tx.get("pathType") or "").strip()
        if pt not in ("runway_exit", "runway_taxiway"):
            continue
        tid = str(tx.get("id", "")).strip()
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tx)
    for tx in layout.get("runwayTaxiways") or []:
        if not isinstance(tx, dict):
            continue
        tid = str(tx.get("id", "")).strip()
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tx)
    return out


def _list_rtx_touching_lineup_on_runway(
    layout: Dict[str, Any],
    cell_size: float,
    runway_tw: Dict[str, Any],
    lineup_pt: Tuple[float, float],
    tol_px: Optional[float] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rw_pts_raw = get_ordered_points(runway_tw, layout, cell_size)
    rw_pts = _as_xy_pairs(rw_pts_raw or [])
    if len(rw_pts) < 2:
        return out
    _ = cell_size
    tol = float(tol_px) if isinstance(tol_px, (int, float)) and math.isfinite(float(tol_px)) and float(tol_px) > 0 else 7.0
    touch_d2 = max(float(SPLIT_TOL_D2), tol * tol)
    lx, ly = float(lineup_pt[0]), float(lineup_pt[1])
    for tx in _iter_layout_rtx_objects(layout):
        rtx_raw = get_ordered_points(tx, layout, cell_size)
        rtx = _as_xy_pairs(rtx_raw or [])
        if len(rtx) < 2:
            continue
        if not _polyline_touches_polyline_for_graph(rtx, rw_pts) and not _polyline_touches_polyline_for_graph(
            rw_pts, rtx
        ):
            continue
        if _point_near_polyline_sq(lx, ly, rtx, touch_d2):
            out.append(tx)
    return out


def _rtx_polylines_touch(
    layout: Dict[str, Any], cell_size: float, rtx_a: Dict[str, Any], rtx_b: Dict[str, Any]
) -> bool:
    pa = _as_xy_pairs(get_ordered_points(rtx_a, layout, cell_size) or [])
    pb = _as_xy_pairs(get_ordered_points(rtx_b, layout, cell_size) or [])
    if len(pa) < 2 or len(pb) < 2:
        return False
    return _polyline_touches_polyline_for_graph(pa, pb) or _polyline_touches_polyline_for_graph(pb, pa)


def _expand_rtx_candidate_ids_touching_lineup(
    layout: Dict[str, Any],
    cell_size: float,
    runway_tw: Optional[Dict[str, Any]],
    lineup_pt: Tuple[float, float],
    tol_px: Optional[float] = None,
) -> set[str]:
    ids: set[str] = set()
    if runway_tw is None:
        return ids
    hop1 = _list_rtx_touching_lineup_on_runway(layout, cell_size, runway_tw, lineup_pt, tol_px)
    for tx in hop1:
        if isinstance(tx, dict) and tx.get("id") is not None:
            ids.add(str(tx["id"]))
    rtx_list = _iter_layout_rtx_objects(layout)
    for a in hop1:
        for b in rtx_list:
            if b.get("id") == a.get("id"):
                continue
            bid = str(b.get("id", ""))
            if bid in ids:
                continue
            if _rtx_polylines_touch(layout, cell_size, a, b):
                ids.add(bid)
    return ids


def _runway_holding_near_rtx_candidate_set(
    layout: Dict[str, Any],
    cell_size: float,
    hp: Dict[str, Any],
    cand_ids: set[str],
    tol_px: Optional[float] = None,
) -> bool:
    p = _holding_point_xy_px(hp)
    if p is None or not _holding_point_kind_runway_holding(hp):
        return False
    _ = cell_size
    tol = float(tol_px) if isinstance(tol_px, (int, float)) and math.isfinite(float(tol_px)) and float(tol_px) > 0 else 7.0
    tol_d2 = max(float(SPLIT_TOL_D2), (tol * 1.15) ** 2)
    px, py = p[0], p[1]
    for tx in _iter_layout_rtx_objects(layout):
        tid = str(tx.get("id", ""))
        if tid not in cand_ids:
            continue
        rtx = _as_xy_pairs(get_ordered_points(tx, layout, cell_size) or [])
        if len(rtx) >= 2 and _point_near_polyline_sq(px, py, rtx, tol_d2):
            return True
    return False


def _cumulative_dist_along_polyline_to_point(
    pts: List[Tuple[float, float]], q: Tuple[float, float]
) -> Optional[Tuple[float, Tuple[float, float]]]:
    if len(pts) < 2:
        return None
    best_d2 = float("inf")
    best_cum = 0.0
    best_proj = (float(q[0]), float(q[1]))
    acc = 0.0
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        seg_len = path_dist(p1, p2)
        if seg_len < 1e-9:
            continue
        _t, proj = project_on_segment(p1, p2, q)
        t = max(0.0, min(1.0, float(_t)))
        d2 = (q[0] - proj[0]) ** 2 + (q[1] - proj[1]) ** 2
        cand_cum = acc + t * seg_len
        if d2 < best_d2:
            best_d2 = d2
            best_cum = cand_cum
            best_proj = (float(proj[0]), float(proj[1]))
        acc += seg_len
    return (best_cum, best_proj)


def _find_last_runway_holding_on_departure_path(
    layout: Dict[str, Any],
    cell_size: float,
    to_lineup_pts: List[Tuple[float, float]],
    cand_ids: set[str],
    tol_px: Optional[float] = None,
) -> Optional[Tuple[Dict[str, Any], float, Tuple[float, float]]]:
    if len(to_lineup_pts) < 2 or not cand_ids:
        return None
    hps = layout.get("holdingPoints") or []
    best: Optional[Tuple[Dict[str, Any], float, Tuple[float, float]]] = None
    _ = cell_size
    tol = float(tol_px) if isinstance(tol_px, (int, float)) and math.isfinite(float(tol_px)) and float(tol_px) > 0 else 7.0
    tol_line_d2 = max(float(SPLIT_TOL_D2), (tol * 1.3) ** 2)
    for hp in hps:
        if not isinstance(hp, dict):
            continue
        if not _holding_point_kind_runway_holding(hp):
            continue
        if not _runway_holding_near_rtx_candidate_set(layout, cell_size, hp, cand_ids, tol):
            continue
        p = _holding_point_xy_px(hp)
        if p is None:
            continue
        if not _point_near_polyline_sq(p[0], p[1], to_lineup_pts, tol_line_d2):
            continue
        cum = _cumulative_dist_along_polyline_to_point(to_lineup_pts, p)
        if cum is None:
            continue
        dist_along, proj = cum[0], cum[1]
        if dist_along <= 1e-3:
            continue
        if best is None or dist_along > best[1]:
            best = (hp, float(dist_along), proj)
    if best is None:
        return None
    return best


def _world_xy_chain_from_graph_path(
    g: PathGraph,
    path: List[int],
    pair_index: Dict[Tuple[int, int], str],
) -> List[Tuple[float, float]]:
    del pair_index
    if len(path) < 2:
        return []
    chain: List[Tuple[float, float]] = []
    n_nodes = len(g.nodes)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rec = g.edge_map.get(f"{u}:{v}")
        if rec is None:
            return []
        if len(rec.pts) >= 2:
            for j in range(len(rec.pts)):
                pj = rec.pts[j]
                p = (float(pj[0]), float(pj[1]))
                if not chain or (chain[-1][0] - p[0]) ** 2 + (chain[-1][1] - p[1]) ** 2 > 1e-12:
                    chain.append(p)
        else:
            if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
                return []
            pu = (float(g.nodes[u][0]), float(g.nodes[u][1]))
            pv = (float(g.nodes[v][0]), float(g.nodes[v][1]))
            if not chain:
                chain.append(pu)
            if (pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2 > 1e-12:
                chain.append(pv)
    return chain


def _dep_runway_far_end_xy(
    layout: Dict[str, Any],
    cell_size: float,
    dep_rwy_id: str,
    dep_ops_dir: str,
) -> Optional[Tuple[float, float]]:
    _ = dep_ops_dir
    coords = _runway_polyline_coords_px(layout, cell_size, str(dep_rwy_id))
    if not coords or len(coords) < 2:
        return None
    last = coords[-1]
    return (float(last[0]), float(last[1]))


def extract_point_to_paths(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    *,
    information: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """
    Token pixels as path legs: touchdown → RET exit junction ``A`` → apron → runway holding
    (RTX·lineup) → lineup → rwy end.

    ``A`` is the first ``networkJunction`` on ``ExitTaxiwayId`` along the runway-leave direction
    (excluding the runway–RET touch); if none, the runway-far RET vertex.

    Leg phases: ``Landing``, ``Arr_taxi``, ``Dep_taxi``, ``Holding_lineup``, ``Lineup_departure``.
    Requires a ``runway_holding`` on the apron→lineup route near lineup-connected RET; else ``[]``.
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

    td_point = (
        _arr_touchdown_point_xy(flight, layout, cell_size, str(arr_rwy)) if arr_rwy else None
    )
    ret_on_rw = (
        _arr_ret_runway_junction_xy(layout, cell_size, str(arr_rwy), arr_ret_tw) if arr_rwy else None
    )
    point_a = (
        _arr_ret_exit_first_junction_a_xy(layout, cell_size, arr_ret_tw, ret_on_rw)
        if arr_rwy
        else None
    )
    apron_point = _apron_token_xy(layout, cell_size, str(stand_id))
    dep_rw_lineup_point = (
        _dep_lineup_token_xy(
            layout,
            cell_size,
            str(dep_rwy),
            _flight_rw_dir_for_leg(flight, 2, layout),
        )
        if dep_rwy
        else None
    )

    if not arr_rwy or not dep_rwy:
        return []
    if (
        td_point is None
        or ret_on_rw is None
        or point_a is None
        or apron_point is None
        or dep_rw_lineup_point is None
    ):
        return []

    info = information if isinstance(information, dict) else _load_information_json()
    reverse_cost, merge_r, taxiway_h = _path_search_params(info)
    pair_index = _pair_index_from_layout_edge(layout)
    if not pair_index:
        g0 = _cached_path_graph_for_direction(
            layout,
            cell_size,
            _flight_rw_dir_for_leg(flight, 2, layout),
            reverse_cost,
            merge_r,
            taxiway_h,
            info,
            pure_ground_exclude_runway=False,
        )
        pair_index = _pair_index_from_path_graph(g0) if g0 else {}
    if not pair_index:
        return []

    px, py = apron_point
    lx, ly = dep_rw_lineup_point
    _edges_apron, dv_apron, path_apron, g_apron = _flight_route_impl(
        layout,
        cell_size,
        pair_index,
        reverse_cost,
        merge_r,
        taxiway_h,
        info,
        _flight_rw_dir_for_leg(flight, 2, layout),
        RouteEndpoint(token_pixel_xy=(float(px), float(py))),
        RouteEndpoint(token_pixel_xy=(float(lx), float(ly))),
    )
    if dv_apron or path_apron is None or g_apron is None or len(path_apron) < 2:
        return []
    to_lineup_pts = _world_xy_chain_from_graph_path(g_apron, path_apron, pair_index)
    if len(to_lineup_pts) < 2:
        return []

    runway_tw = _layout_runway_object_for_lineup_expansion(layout, str(dep_rwy))
    cand_ids = _expand_rtx_candidate_ids_touching_lineup(
        layout, cell_size, runway_tw, (float(lx), float(ly)), merge_r
    )
    hold_pick = _find_last_runway_holding_on_departure_path(
        layout, cell_size, to_lineup_pts, cand_ids, merge_r
    )
    if hold_pick is None:
        return []
    _hp_obj, _d_along, hp_proj = hold_pick
    hx, hy = float(hp_proj[0]), float(hp_proj[1])

    rw_end = _dep_runway_far_end_xy(
        layout, cell_size, str(dep_rwy), _flight_rw_dir_for_leg(flight, 2, layout)
    )
    if rw_end is None:
        return []

    wx, wy = td_point
    ax, ay = point_a
    ex, ey = rw_end
    return [
        [float(wx), float(wy), float(ax), float(ay)],
        [float(ax), float(ay), float(px), float(py)],
        [float(px), float(py), float(hx), float(hy)],
        [float(hx), float(hy), float(lx), float(ly)],
        [float(lx), float(ly), float(ex), float(ey)],
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
    information: Dict[str, Any],
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

        if phase in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP, PHASE_DEP_TAXI, PHASE_HOLDING_LINEUP):
            is_ret = pt == "runway_exit" and str(link_id) == exit_tw_s
            if pt == "runway_taxiway":
                if i == 0:
                    raise ValueError(
                        f"runway_taxiway at path start has no previous link (flight_id={flight_id!r})"
                    )
                prev_lid = str(segment_link_ids[i - 1])
                v_t = _avg_move_velocity_ms_for_link(layout, prev_lid, flight_id)
                if phase in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
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

        if phase == PHASE_LINEUP_DEPARTURE:
            takeoff_a = _dep_takeoff_accel_ms2(information)
            v0_out[i] = float(v_cur)
            a_out[i] = abs(float(takeoff_a))
            dur_out[i] = _duration_slice_sec(v0_out[i], a_out[i], 0.0, seg_m, False)
            v_cur = float(
                _velocity_ms_at_distance_on_segment(v0_out[i], a_out[i], seg_m, False)
            )
            continue

        raise ValueError(f"unknown phase {phase!r} for kinematics (flight_id={flight_id!r})")

    return v0_out, a_out, dur_out


def _expand_geometry_from_graph_path(
    g: PathGraph,
    merged_nodes: List[int],
    pair_index: Dict[Tuple[int, int], str],
    leg_phase: str,
) -> Tuple[
    List[str],
    List[Tuple[Point, Point]],
    List[str],
    List[str],
    List[str],
    List[Tuple[int, int]],
]:
    """
    One layout edge id per graph hop; duplicate ids when splitting ``DirectedEdgeRecord.pts`` polylines.
    ``leg_phase`` is repeated for every expanded sub-segment on this leg.
    ``segment_graph_uv`` repeats ``(u, v)`` for each micro-segment from the same graph hop.
    """
    expanded_ids: List[str] = []
    segments: List[Tuple[Point, Point]] = []
    phases: List[str] = []
    link_ids: List[str] = []
    path_types: List[str] = []
    graph_uv: List[Tuple[int, int]] = []
    n_nodes = len(g.nodes)
    for i in range(len(merged_nodes) - 1):
        u, v = merged_nodes[i], merged_nodes[i + 1]
        lo, hi = (u, v) if u <= v else (v, u)
        eid = pair_index.get((lo, hi))
        if not eid:
            return [], [], [], [], [], []
        rec = g.edge_map.get(f"{u}:{v}")
        if rec is None:
            return [], [], [], [], [], []
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
                graph_uv.append((int(u), int(v)))
        else:
            if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
                return [], [], [], [], [], []
            p0 = g.nodes[u]
            p1 = g.nodes[v]
            expanded_ids.append(str(eid))
            segments.append(((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))))
            phases.append(leg_phase)
            link_ids.append(lid)
            path_types.append(pt)
            graph_uv.append((int(u), int(v)))
    return expanded_ids, segments, phases, link_ids, path_types, graph_uv


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
    ``extract_point_to_paths`` 레그마다 layout ``Edge`` 기반
    ``pair_index``(없으면 그래프에서 ``layout-edge-*``), 레그마다 그래프 재구성 후
    ``token_pixel_xy`` 끝점만으로 ``flight_route``, 역주행 패널티 구간이면 전체 ``edge_list`` 비움.
    재생용 세그먼트는 각 레그의 노드 경로를 ``_expand_geometry_from_graph_path``로 확장한다.
    """
    paths = extract_point_to_paths(flight, layout, cell_size, information=information)
    if not paths:
        return PreparedFlightPath()
    pair_index = _pair_index_from_layout_edge(layout)
    if not pair_index:
        g0 = _cached_path_graph_for_direction(
            layout,
            cell_size,
            _flight_rw_dir_for_leg(flight, 0, layout),
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

    token_f = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    arr_rwy_f = flight.get("arrRunwayId") or token_f.get("arrRunwayId")

    for leg_i, leg in enumerate(paths):
        if len(leg) < 4:
            return PreparedFlightPath()
        phase = (
            _EXTRACT_LEG_PHASES[leg_i] if leg_i < len(_EXTRACT_LEG_PHASES) else PHASE_DEP_TAXI
        )
        sx, sy, ex, ey = float(leg[0]), float(leg[1]), float(leg[2]), float(leg[3])
        rw_leg = _flight_rw_dir_for_leg(flight, leg_i, layout)
        ap_ids = (
            _apron_link_ids_for_assigned_stand(layout, flight)
            if str(phase) == PHASE_ARR_TAXI
            else set()
        )
        ap_extra = 0.0
        if ap_ids and str(phase) == PHASE_ARR_TAXI:
            ap_extra = min(float(reverse_cost) * 0.04, 80_000.0)
        # Leg 0 (touchdown → RET A): start Dijkstra from the runway polyline only, not from the
        # full-graph nearest node (which can sit on a taxiway behind the touchpoint; see R2 case).
        if leg_i == 0 and str(phase) == PHASE_LANDING and arr_rwy_f and str(arr_rwy_f).strip():
            start_ep = RouteEndpoint(
                runway_id=str(arr_rwy_f).strip(),
                runway_pixel_xy=(float(sx), float(sy)),
            )
        else:
            start_ep = RouteEndpoint(token_pixel_xy=(sx, sy))
        edges, dv, path, g = _flight_route_impl(
            layout,
            cell_size,
            pair_index,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            rw_leg,
            start_ep,
            RouteEndpoint(token_pixel_xy=(ex, ey)),
            apron_transit_extra=ap_extra,
            apron_allowed_link_ids=ap_ids if ap_ids else None,
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
    expanded_graph_uv: List[Tuple[int, int]] = []
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
            ex_ids, segs, phs, lnks, ptyps, guvs = _expand_geometry_from_graph_path(
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
                or len(guvs) != len(ex_ids)
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
            expanded_graph_uv.extend(guvs)
            leg_lengths_px.append(_path_length_px(segs))
            leg_micro_counts.append(len(ex_ids))
        # Landing에서 runway 접점 이후 RET는 1개 edge만 포함하고,
        # 나머지 RET micro-edge는 Arr_taxi로 넘긴다.
        landing_ret_idxs = [
            i
            for i, (ph, pt) in enumerate(zip(segment_phases, segment_path_types))
            if str(ph) == PHASE_LANDING and str(pt) == "runway_exit"
        ]
        if len(landing_ret_idxs) > 1:
            for i in landing_ret_idxs[1:]:
                segment_phases[i] = PHASE_ARR_TAXI

    playback_ok = (
        bool(expanded_ids)
        and bool(segments)
        and len(expanded_ids) == len(segments) == len(segment_phases)
        and len(segment_link_ids) == len(expanded_ids)
        and len(segment_path_types) == len(expanded_ids)
        and len(expanded_graph_uv) == len(expanded_ids)
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
                information,
            )
        except ValueError:
            return PreparedFlightPath(
                logical_edge_list=list(logical_edge_list),
                edge_ids=list(expanded_ids),
                segment_phases=list(segment_phases),
                segment_endpoints=list(segments),
                segment_link_ids=list(segment_link_ids),
                segment_path_types=list(segment_path_types),
                segment_graph_uv=list(expanded_graph_uv),
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
        segment_graph_uv=list(expanded_graph_uv),
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


def _finish_edge_segment(agent: Flight, *, sim_time_abs: Optional[float] = None) -> None:
    old_ph = str(agent.edge_phases[0]) if agent.edge_phases else ""
    eid = agent.edge_ids.pop(0)
    ph = agent.edge_phases.pop(0)
    v0 = float(agent.segment_v0_ms.pop(0))
    acc = float(agent.segment_accel_ms2.pop(0))
    pt_done = ""
    if agent.segment_path_types:
        pt_done = str(agent.segment_path_types[0] or "")
    guv_done: Optional[Tuple[int, int]] = None
    if agent.segment_graph_uv:
        guv_done = agent.segment_graph_uv.pop(0)
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
    if guv_done is not None:
        agent.completed_directed_hops.append((str(eid), int(guv_done[0]), int(guv_done[1]), pt_done))
    if sim_time_abs is not None and not agent.edge_ids:
        if (
            str(old_ph) == PHASE_ARR_TAXI_TEMP
            and agent.post_temp_route_tail_prep is not None
            and str(agent.temp_stand_id or "").strip()
        ):
            agent.awaiting_apron_from_temp = True
            agent.temp_park_arrival_trigger_global_reroute = True
        elif not agent.awaiting_apron_from_temp:
            agent.path_completed_abs_sec = float(sim_time_abs)
    new_ph = str(agent.edge_phases[0]) if agent.edge_phases else ""
    if (
        sim_time_abs is not None
        and old_ph == PHASE_HOLDING_LINEUP
        and new_ph == PHASE_LINEUP_DEPARTURE
    ):
        if agent.runway_entry_abs_sec is None:
            agent.runway_entry_abs_sec = float(sim_time_abs)
        agent.lineup_hold_release_abs_sec = float(sim_time_abs) + _lineup_clearance_hold_sec(
            _load_information_json()
        )
    if (
        sim_time_abs is not None
        and old_ph == PHASE_ARR_TAXI
        and new_ph == PHASE_DEP_TAXI
        and agent.actual_apron_inblocks_abs_sec is None
    ):
        # Fallback EIBT if stand token / radius never matched during Arr_taxi.
        t_in = float(sim_time_abs)
        agent.actual_apron_inblocks_abs_sec = t_in
        agent.dep_taxi_start_abs_sec = t_in + float(agent.dwell_sec)
        if agent.eldt_anchor_sec is not None:
            agent.dep_taxi_start_sim_time = float(agent.dep_taxi_start_abs_sec) - float(
                agent.eldt_anchor_sec
            )


def move_agent(
    agent: Flight,
    dt: float,
    pixels_per_meter: float,
    sim_time: Optional[float] = None,
    sim_time_abs: Optional[float] = None,
) -> None:
    """Advance along ``segment_endpoints`` using per-segment :math:`v_0` + constant ``acceleration_ms2``.

    If ``sim_time`` / ``sim_time_abs`` are set, the first ``Dep_taxi`` segment does not move until
    ``dep_taxi_start_*`` is reached; those are set only when in-blocks is stamped (stand arrival + dwell),
    not from ELDT+nominal taxi-in. If both are ``None``, Dep_taxi is not time-gated here.
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
    if (
        sim_time_abs is not None
        and agent.lineup_hold_release_abs_sec is not None
        and agent.edge_phases
        and agent.edge_phases[0] == PHASE_LINEUP_DEPARTURE
        and float(sim_time_abs) + 1e-9 < float(agent.lineup_hold_release_abs_sec)
    ):
        agent.velocity_ms = 0.0
        return

    while rem_t > 1e-12 and agent.edge_ids and agent.segment_endpoints:
        p0, p1 = agent.segment_endpoints[0]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_px = math.hypot(dx, dy)
        if seg_len_px < 1e-9:
            _finish_edge_segment(agent, sim_time_abs=sim_time_abs)
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
            _finish_edge_segment(agent, sim_time_abs=sim_time_abs)
            agent.edge_s_along_px = 0.0
            agent.col, agent.row = p1[0], p1[1]
            continue

        v_now = _velocity_ms_at_distance_on_segment(
            v0s, ac, s_m, apply_floor, decel_floor_ms=_ret_floor_ma
        )
        if agent.control_speed_cap_ms is not None and math.isfinite(float(agent.control_speed_cap_ms)):
            v_now = min(v_now, float(agent.control_speed_cap_ms))
        if agent.segment_path_types and str(agent.segment_path_types[0] or "") == "runway_taxiway":
            if ph == PHASE_LANDING or ph in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
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
            _finish_edge_segment(agent, sim_time_abs=sim_time_abs)
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
            and (_phn == PHASE_LANDING or _phn in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP))
        ) or _arr_ret_decel_floor_ms(_phn, _ptn, _acn) > 1e-12:
            agent.velocity_ms = max(
                agent.velocity_ms, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS
            )
    if agent.edge_phases and agent.segment_path_types and agent.edge_ids:
        _ptm = str(agent.segment_path_types[0] or "")
        _phm = str(agent.edge_phases[0])
        if _ptm == "apron_link" and _phm == PHASE_DEP_TAXI:
            agent.motion_is_forward = False
        elif _ptm == "apron_link" and _phm in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
            agent.motion_is_forward = True
        elif _ptm != "apron_link":
            agent.motion_is_forward = True


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
    elif wr.startswith("temp_stand_busy:"):
        ts_id = wr.split(":", 1)[1].strip()
        out |= {
            str(e).strip()
            for e in control_state.temp_stand_incident_edges.get(ts_id, set())
            if str(e).strip()
        }
    elif wr.startswith("runway_rot_busy:") and lookahead_edges:
        for e in lookahead_edges[:4]:
            s = str(e).strip()
            if s:
                out.add(s)
    if wr:
        for e in lookahead_edges[:2]:
            s = str(e).strip()
            if s:
                out.add(s)
        if agent.edge_ids:
            out.add(str(agent.edge_ids[0]))
    return {e for e in out if e}


def _compute_arr_touchdown_motion_abs_sec(
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
    dep_release_buffer_sec = 20.0
    need_exit = 0.0
    any_pred = False
    pred_missing_exit = False
    dep_windows: List[Tuple[float, Optional[float]]] = []
    for o in agents:
        if o.id == agent.id:
            continue
        dep_rw = str(o.dep_runway_id or "").strip()
        if dep_rw == rw:
            dep_entry = o.runway_entry_abs_sec
            if dep_entry is not None:
                dep_end = (
                    float(o.path_completed_abs_sec) + float(dep_release_buffer_sec)
                    if o.path_completed_abs_sec is not None
                    else None
                )
                dep_windows.append((float(dep_entry), dep_end))
        if str(o.arr_runway_id or "").strip() != rw:
            continue
        if o.eldt_anchor_sec is None:
            continue
        if float(o.eldt_anchor_sec) + 1e-9 >= my:
            continue
        any_pred = True
        ex = o.exit_runway_abs_sec
        if ex is None:
            pred_missing_exit = True
            continue
        need_exit = max(need_exit, float(ex))
    if not any_pred:
        base = max(raw, anch_f)
    else:
        base = anch_f if pred_missing_exit else max(raw, need_exit + lag)
    # DEP runway occupancy windows: [runway_entry_abs_sec, ETOT + 20s).
    # If touchdown candidate falls inside any window, push it to that window end
    # and repeat to account for overlapping / chained windows.
    changed = True
    while changed:
        changed = False
        for dep_entry, dep_end in dep_windows:
            if dep_entry - 1e-9 > base:
                continue
            if dep_end is None:
                return float("inf")
            if base + 1e-9 < dep_end:
                base = float(dep_end)
                changed = True
    return base


def _refresh_touchdown_motion_cache(
    control_state: SimulationControlState,
    agents: List[Flight],
    runway_release_lag_sec: float,
) -> None:
    lag = float(runway_release_lag_sec)
    control_state.touchdown_motion_by_id = {
        str(ag.id): _compute_arr_touchdown_motion_abs_sec(ag, agents, lag)
        for ag in agents
    }


def _arr_touchdown_motion_abs_sec(
    agent: Flight,
    agents: List[Flight],
    runway_release_lag_sec: float,
    *,
    control_state: Optional[SimulationControlState] = None,
) -> Optional[float]:
    if control_state is not None and control_state.touchdown_motion_by_id is not None:
        return control_state.touchdown_motion_by_id.get(str(agent.id))
    return _compute_arr_touchdown_motion_abs_sec(agent, agents, runway_release_lag_sec)


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
    counts = prep.leg_micro_counts
    if (
        not durs
        or len(durs) != len(segs)
        or len(counts) < 5
        or len(v0s) != len(segs)
        or len(accs) != len(segs)
        or len(phs) != len(segs)
    ):
        return None, None
    c0, c1, c2, c3, c4 = (
        int(counts[0]),
        int(counts[1]),
        int(counts[2]),
        int(counts[3]),
        int(counts[4]),
    )
    c01 = c0 + c1
    c01234 = c01 + c2 + c3 + c4
    if c01234 != len(segs):
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
    for g in range(c01, c01234):
        if g < 0 or g >= len(segs):
            continue
        taxi_out += dur_full(g)
    return taxi_in, taxi_out


def _flight_path_has_lineup_departure(prep: PreparedFlightPath) -> bool:
    """True if prepared path includes at least one takeoff-roll micro-segment."""
    if not prep.segment_phases:
        return False
    return any(str(p) == PHASE_LINEUP_DEPARTURE for p in prep.segment_phases)


def _agent_path_includes_landing(agent: Flight) -> bool:
    """True if this agent's route includes a Landing phase (arrival roll / decel on runway)."""
    for row in agent.edge_ids_finished or []:
        if isinstance(row, dict) and str(row.get("phase", "")) == PHASE_LANDING:
            return True
    for ph in agent.edge_phases or []:
        if str(ph) == PHASE_LANDING:
            return True
    return False


def _departure_leg_durations_sec_for_schedule(
    prep: PreparedFlightPath,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Sums of segment_duration_sec for legs 2,3,4 (Dep_taxi, Holding_lineup, Lineup_departure)."""
    if not prep.ok:
        return None, None, None
    durs = prep.segment_duration_sec
    counts = prep.leg_micro_counts
    if not durs or len(counts) < 5 or sum(int(c) for c in counts) != len(durs):
        return None, None, None
    c0, c1, c2, c3, c4 = (int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]), int(counts[4]))
    i2 = c0 + c1
    i3 = i2 + c2
    i4 = i3 + c3
    end = i4 + c4
    if end != len(durs):
        return None, None, None
    t2 = sum(float(durs[g]) for g in range(i2, i3))
    t3 = sum(float(durs[g]) for g in range(i3, i4))
    t4 = sum(float(durs[g]) for g in range(i4, end))
    return t2, t3, t4


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


def _backfill_actual_apron_offblocks_from_history(ag: Flight) -> None:
    """If loop never stamped off-blocks, use first post-gate motion sample (same rule as main loop)."""
    if ag.actual_apron_offblocks_abs_sec is not None:
        return
    ds = ag.dep_taxi_start_abs_sec
    if ds is None or not ag.history:
        return
    ds_f = float(ds)
    for row in ag.history:
        if len(row) < 4:
            continue
        t_abs = float(row[0])
        v = float(row[3])
        if t_abs <= ds_f + 1e-9:
            continue
        if abs(v) > 0.01:
            ag.actual_apron_offblocks_abs_sec = t_abs
            return


def _compress_agent_history_for_dwell_export(
    history: List[Tuple[float, ...]],
    anchor_sec: Optional[float],
    eibt_sec: Optional[float],
    eobt_sec: Optional[float],
    dep_taxi_start_abs_sec: Optional[float] = None,
    parked_nose_heading_deg: Optional[float] = None,
) -> List[Tuple[float, ...]]:
    """
    Drop interior samples while on-stand dwell only, keeping endpoints only.

    Band is ``[EIBT, min(EOBT, dep_taxi_start_abs_sec)]`` when ``dep_taxi_start_abs_sec`` is set and
    begins before nominal EOBT (early pushback). Otherwise the full scheduled ``[EIBT, EOBT]``.
    Samples after the band end (taxi-out before nominal EOBT) must not be removed or playback
    interpolates a straight line across the grid.
    """
    if anchor_sec is None or eibt_sec is None or eobt_sec is None:
        return history
    eibt = float(eibt_sec)
    eobt = float(eobt_sec)
    if eobt <= eibt + 1e-9:
        return history
    band_hi = float(eobt)
    if dep_taxi_start_abs_sec is not None:
        ds = float(dep_taxi_start_abs_sec)
        if math.isfinite(ds) and ds < band_hi - 1e-9:
            band_hi = min(band_hi, ds)
    if band_hi <= eibt + 1e-9:
        return history
    n = len(history)
    if n <= 2:
        return history
    in_band: List[int] = []
    for i in range(n):
        t_abs = float(history[i][0])
        if eibt - 1e-9 <= t_abs <= band_hi + 1e-9:
            in_band.append(i)
    if len(in_band) <= 2:
        return history
    first_i = int(in_band[0])
    next_i = int(in_band[1])
    first_t, first_x, first_y, first_v, first_mf = history[first_i][:5]
    next_x = float(history[next_i][1])
    next_y = float(history[next_i][2])
    motion_eps2 = 0.08 * 0.08

    def _norm_deg(ang: float) -> float:
        return ((float(ang) + 180.0) % 360.0) - 180.0

    def _row_motion_angle(i0: int, i1: int, apply_motion_forward: bool) -> Optional[float]:
        if i0 < 0 or i1 <= i0 or i1 >= len(history):
            return None
        a = history[i0]
        b = history[i1]
        if len(a) < 5 or len(b) < 5:
            return None
        dx = float(b[1]) - float(a[1])
        dy = float(b[2]) - float(a[2])
        if dx * dx + dy * dy < motion_eps2:
            return None
        ang = math.degrees(math.atan2(dy, dx))
        if apply_motion_forward and not bool(b[4]):
            ang += 180.0
        return _norm_deg(ang)

    prev_raw_angle: Optional[float] = None
    for j in range(first_i - 1, -1, -1):
        ang = _row_motion_angle(j, j + 1, False)
        if ang is not None:
            prev_raw_angle = ang
            break

    dwell_mf = bool(first_mf)
    if prev_raw_angle is not None and parked_nose_heading_deg is not None:
        target_nose = _norm_deg(parked_nose_heading_deg)
        forward_err = abs(_norm_deg(prev_raw_angle - target_nose))
        reverse_err = abs(_norm_deg(prev_raw_angle + 180.0 - target_nose))
        dwell_mf = forward_err <= reverse_err

    def _rewrite_dwell_row(
        row: Tuple[float, ...],
        t_val: float,
        x_val: float,
        y_val: float,
        mf_val: bool,
        ghost_val: bool,
        dst_val: Any,
    ) -> Tuple[float, ...]:
        tail = tuple(row[7:]) if len(row) > 7 else ()
        return (
            float(t_val),
            float(x_val),
            float(y_val),
            0.0,
            bool(mf_val),
            bool(ghost_val),
            dst_val,
            *tail,
        )
    first_gh = bool(history[first_i][5]) if len(history[first_i]) > 5 else False
    next_gh = bool(history[next_i][5]) if len(history[next_i]) > 5 else False
    if (
        abs(float(first_t) - eibt) <= 1e-9
        and (
            abs(float(first_x) - float(next_x)) > 1e-9
            or abs(float(first_y) - float(next_y)) > 1e-9
            or abs(float(first_v)) > 1e-9
        )
    ):
        hist_mut = list(history)
        first_dst = history[first_i][6] if len(history[first_i]) > 6 else None
        hist_mut[first_i] = _rewrite_dwell_row(
            history[first_i],
            float(first_t),
            float(next_x),
            float(next_y),
            bool(dwell_mf),
            bool(next_gh or first_gh),
            first_dst,
        )
        history = hist_mut
    if abs(float(history[first_i][0]) - eibt) <= 1e-9 and bool(history[first_i][4]) != bool(dwell_mf):
        hist_mut = list(history)
        first_dst = history[first_i][6] if len(history[first_i]) > 6 else None
        first_gh = bool(history[first_i][5]) if len(history[first_i]) > 5 else False
        hist_mut[first_i] = _rewrite_dwell_row(
            history[first_i],
            float(history[first_i][0]),
            float(history[first_i][1]),
            float(history[first_i][2]),
            bool(dwell_mf),
            bool(first_gh),
            first_dst,
        )
        history = hist_mut
    last_i = int(in_band[-1])
    prev_i = int(in_band[-2])
    last_t, last_x, last_y, last_v = history[last_i][:4]
    prev_x = float(history[prev_i][1])
    prev_y = float(history[prev_i][2])
    last_gh = bool(history[last_i][5]) if len(history[last_i]) > 5 else False
    prev_gh = bool(history[prev_i][5]) if len(history[prev_i]) > 5 else False
    if (
        abs(float(last_t) - band_hi) <= 1e-9
        and (
            abs(float(last_x) - float(prev_x)) > 1e-9
            or abs(float(last_y) - float(prev_y)) > 1e-9
            or abs(float(last_v)) > 1e-9
        )
    ):
        hist_mut = list(history)
        last_dst = history[last_i][6] if len(history[last_i]) > 6 else None
        hist_mut[last_i] = _rewrite_dwell_row(
            history[last_i],
            float(last_t),
            float(prev_x),
            float(prev_y),
            bool(dwell_mf),
            bool(prev_gh or last_gh),
            last_dst,
        )
        history = hist_mut
    if abs(float(history[last_i][0]) - band_hi) <= 1e-9 and bool(history[last_i][4]) != bool(dwell_mf):
        hist_mut = list(history)
        last_dst = history[last_i][6] if len(history[last_i]) > 6 else None
        last_gh = bool(history[last_i][5]) if len(history[last_i]) > 5 else False
        hist_mut[last_i] = _rewrite_dwell_row(
            history[last_i],
            float(history[last_i][0]),
            float(history[last_i][1]),
            float(history[last_i][2]),
            bool(dwell_mf),
            bool(last_gh),
            last_dst,
        )
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
    runway_entry_abs_sec: Optional[float] = None,
    touchdown_motion_abs_sec: Optional[float] = None,
    arr_runway_id: Optional[str] = None,
    dep_runway_id: Optional[str] = None,
    has_landing_leg: bool = False,
    has_lineup_departure_leg: bool = False,
    eldt_schedule_sec: Optional[int] = None,
    actual_apron_inblocks_abs_sec: Optional[float] = None,
    actual_apron_offblocks_abs_sec: Optional[float] = None,
    path_completed_abs_sec: Optional[float] = None,
    information: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    S series from ``*_Min_orig``; Sd echo from ``*_Min_d``; E series from path timing + dwell.
    ``eldt_schedule_sec``가 있으면 동일 활주로 간격 조정된 ELDT를 쓰고, 없으면 ``_sd_eldt_sec``만 쓴다.
    ``EIBT`` is measured only (near stand on Arr_taxi apron segment, or phase-transition fallback); no ``ELDT``+taxi-in estimate.
    ``EOBT`` = measured off-blocks when set; else ``EIBT``+dwell when EIBT known; else path/nominal chain.
    ``ETOT`` with a takeoff leg in ``prep`` uses ``path_completed_abs_sec`` only; otherwise nominal ``E_LINEUP``+takeoff duration
    when that chain is available.
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

    info = information if isinstance(information, dict) else _load_information_json()
    lineup_hold = _lineup_clearance_hold_sec(info)
    t_dep2: Optional[float] = None
    t_dep3: Optional[float] = None
    t_dep4: Optional[float] = None
    d2, d3, d4 = _departure_leg_durations_sec_for_schedule(prep)
    if d2 is not None and d3 is not None and d4 is not None:
        t_dep2, t_dep3, t_dep4 = d2, d3, d4

    eibt_sec: Optional[float] = None
    eobt_sec: Optional[float] = None
    e_hold_sec: Optional[float] = None
    e_lineup_sec: Optional[float] = None
    etot_sec: Optional[float] = None
    if actual_apron_inblocks_abs_sec is not None:
        eibt_sec = float(actual_apron_inblocks_abs_sec)
    if actual_apron_offblocks_abs_sec is not None:
        eobt_sec = float(actual_apron_offblocks_abs_sec)
    elif eibt_sec is not None:
        eobt_sec = float(eibt_sec) + float(dwell_sec)
    if eobt_sec is not None and t_dep2 is not None:
        e_hold_sec = float(eobt_sec) + float(t_dep2)
    if e_hold_sec is not None and t_dep3 is not None:
        e_lineup_sec = float(e_hold_sec) + float(t_dep3) + float(lineup_hold)
    if _flight_path_has_lineup_departure(prep):
        etot_sec = (
            float(path_completed_abs_sec)
            if path_completed_abs_sec is not None
            else None
        )
    elif e_lineup_sec is not None and t_dep4 is not None:
        etot_sec = float(e_lineup_sec) + float(t_dep4)

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
        "ARR_RUNWAY_ID": str(arr_runway_id).strip() if arr_runway_id is not None else None,
        "DEP_RUNWAY_ID": str(dep_runway_id).strip() if dep_runway_id is not None else None,
        "HAS_LANDING": bool(has_landing_leg),
        "HAS_LINEUP_DEPARTURE": bool(has_lineup_departure_leg),
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
        "TOUCHDOWN_MOTION": _sim_sec_optional(touchdown_motion_abs_sec)
        if touchdown_motion_abs_sec is not None
        else None,
        "TOUCHDOWN_MOTION_dt": _sec_to_datetime_str(touchdown_motion_abs_sec, base_date),
        "RUNWAY_ENTRY": _sim_sec_optional(runway_entry_abs_sec)
        if runway_entry_abs_sec is not None
        else None,
        "RUNWAY_ENTRY_dt": _sec_to_datetime_str(runway_entry_abs_sec, base_date),
        "EXIT_RUNWAY": exit_rw_s,
        "EXIT_RUNWAY_dt": _sec_to_datetime_str(exit_runway_abs_sec, base_date),
        "EIBT": _sim_sec_optional(eibt_sec) if eibt_sec is not None else None,
        "EIBT_dt": _sec_to_datetime_str(eibt_sec, base_date),
        "EOBT": _sim_sec_optional(eobt_sec) if eobt_sec is not None else None,
        "EOBT_dt": _sec_to_datetime_str(eobt_sec, base_date),
        "E_HOLD": _sim_sec_optional(e_hold_sec) if e_hold_sec is not None else None,
        "E_HOLD_dt": _sec_to_datetime_str(e_hold_sec, base_date),
        "E_LINEUP": _sim_sec_optional(e_lineup_sec) if e_lineup_sec is not None else None,
        "E_LINEUP_dt": _sec_to_datetime_str(e_lineup_sec, base_date),
        "ETOT": _sim_sec_optional(etot_sec) if etot_sec is not None else None,
        "ETOT_dt": _sec_to_datetime_str(etot_sec, base_date),
    }


def _overlay_schedule_timing_from_playback_positions(
    out: Dict[str, Any],
    layout: Dict[str, Any],
    information: Dict[str, Any],
    flights_raw: List[Any],
    cell_size: float,
    pixels_per_meter: float,
) -> None:
    """
    Post-simulation only: overwrite schedule motion timing fields from exported
    ``positions`` samples (``t``, ``x``, ``y``, ``v``, optional ``phase`` / ``pathType`` /
    ``clearance``) plus the same layout geometry helpers used for playback (runway
    polyline, stand token). Runs after the time-step loop; values are not fed back
    into simulation state.
    """
    sched = out.get("schedule")
    posmap = out.get("positions")
    base_date = str(out.get("baseDate") or "2026-03-31")
    if not isinstance(sched, list) or not isinstance(posmap, dict):
        return
    ppm = max(float(pixels_per_meter), 1e-9)
    exit_thr_m = float(_exit_runway_min_perpendicular_distance_m(information))
    r_st_m = float(_sim_stand_arrival_stop_radius_m(information))
    v_stop = float(_sim_stand_stopped_velocity_max_ms(information))
    cs = float(cell_size)

    def _apply_timings(row: Dict[str, Any], fobj: Dict[str, Any], pts: List[Dict[str, Any]]) -> None:
        has_landing = bool(row.get("HAS_LANDING"))
        if has_landing and pts:
            t_td = float(pts[0].get("t", 0.0))
            row["ELDT"] = _sim_sec_optional(t_td)
            row["ELDT_dt"] = _sec_to_datetime_str(t_td, base_date)
            row["TOUCHDOWN_MOTION"] = row["ELDT"]
            row["TOUCHDOWN_MOTION_dt"] = row["ELDT_dt"]
        else:
            row["ELDT"] = None
            row["ELDT_dt"] = None
            row["TOUCHDOWN_MOTION"] = None
            row["TOUCHDOWN_MOTION_dt"] = None

        e_hold: Optional[float] = None
        e_lineup: Optional[float] = None
        rw_entry: Optional[float] = None
        for p in pts:
            ph = str(p.get("phase") or "")
            if e_hold is None and ph == PHASE_HOLDING_LINEUP:
                e_hold = float(p.get("t", 0.0))
            if rw_entry is None and ph == PHASE_LINEUP_DEPARTURE:
                rw_entry = float(p.get("t", 0.0))
            if e_lineup is None and ph == PHASE_LINEUP_DEPARTURE:
                e_lineup = float(p.get("t", 0.0))
        row["E_HOLD"] = _sim_sec_optional(e_hold) if e_hold is not None else None
        row["E_HOLD_dt"] = _sec_to_datetime_str(e_hold, base_date)
        row["E_LINEUP"] = _sim_sec_optional(e_lineup) if e_lineup is not None else None
        row["E_LINEUP_dt"] = _sec_to_datetime_str(e_lineup, base_date)
        row["RUNWAY_ENTRY"] = _sim_sec_optional(rw_entry) if rw_entry is not None else None
        row["RUNWAY_ENTRY_dt"] = _sec_to_datetime_str(rw_entry, base_date)

        exit_rw: Optional[float] = None
        arr_id = row.get("ARR_RUNWAY_ID")
        rid = str(arr_id).strip() if arr_id else ""
        if has_landing and rid and pts:
            verts = _oriented_arr_runway_centerline_px(
                layout, cs, rid, _flight_rw_dir_for_leg(fobj, 0, layout)
            )
            if verts:
                for p in pts:
                    if str(p.get("phase") or "") == PHASE_LANDING:
                        continue
                    x = float(p.get("x", 0.0))
                    y = float(p.get("y", 0.0))
                    d_m = float(_min_distance_point_to_polyline(x, y, verts)) / ppm
                    if d_m + 1e-9 >= exit_thr_m:
                        exit_rw = float(p.get("t", 0.0))
                        break
        row["EXIT_RUNWAY"] = _sim_sec_optional(exit_rw) if exit_rw is not None else None
        row["EXIT_RUNWAY_dt"] = _sec_to_datetime_str(exit_rw, base_date)

        eibt: Optional[float] = None
        sid = _flight_apron_stand_id_from_fobj(fobj)
        txy = _apron_token_xy(layout, cs, str(sid)) if sid else None
        if txy is not None and pts:
            tx, ty = float(txy[0]), float(txy[1])
            for p in pts:
                ph = str(p.get("phase") or "")
                if ph not in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
                    continue
                if str(p.get("pathType") or "").strip() == "runway":
                    continue
                clr = str(p.get("clearance") or "")
                if clr in ("WAIT", "YIELD"):
                    continue
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                v = float(p.get("v", 0.0))
                d_m = math.hypot(x - tx, y - ty) / ppm
                if d_m <= r_st_m + 1e-6 and abs(v) <= v_stop + 1e-9:
                    eibt = float(p.get("t", 0.0))
                    break
        row["EIBT"] = _sim_sec_optional(eibt) if eibt is not None else None
        row["EIBT_dt"] = _sec_to_datetime_str(eibt, base_date)

        eobt: Optional[float] = None
        if eibt is not None and pts:
            ds = float(eibt) + float(_dwell_sec_from_flight(fobj))
            for p in pts:
                if float(p.get("t", 0.0)) <= ds + 1e-9:
                    continue
                if str(p.get("phase") or "") != PHASE_DEP_TAXI:
                    continue
                if str(p.get("pathType") or "").strip() != "apron_link":
                    continue
                if abs(float(p.get("v", 0.0))) <= 0.01 + 1e-12:
                    continue
                eobt = float(p.get("t", 0.0))
                break
        row["EOBT"] = _sim_sec_optional(eobt) if eobt is not None else None
        row["EOBT_dt"] = _sec_to_datetime_str(eobt, base_date)

        etot: Optional[float] = None
        if pts:
            etot = float(pts[-1].get("t", 0.0))
        row["ETOT"] = _sim_sec_optional(etot) if etot is not None else None
        row["ETOT_dt"] = _sec_to_datetime_str(etot, base_date)

    for i, row in enumerate(sched):
        if not isinstance(row, dict):
            continue
        fid = str(row.get("flight_id", "")).strip()
        fobj = flights_raw[i] if i < len(flights_raw) and isinstance(flights_raw[i], dict) else {}
        raw_plist = posmap.get(fid)
        if not isinstance(raw_plist, list) or not raw_plist:
            for k, dk in (
                ("ELDT", "ELDT_dt"),
                ("TOUCHDOWN_MOTION", "TOUCHDOWN_MOTION_dt"),
                ("RUNWAY_ENTRY", "RUNWAY_ENTRY_dt"),
                ("EXIT_RUNWAY", "EXIT_RUNWAY_dt"),
                ("EIBT", "EIBT_dt"),
                ("EOBT", "EOBT_dt"),
                ("E_HOLD", "E_HOLD_dt"),
                ("E_LINEUP", "E_LINEUP_dt"),
                ("ETOT", "ETOT_dt"),
            ):
                row[k] = None
                row[dk] = None
            continue
        pts_sorted = sorted(
            (p for p in raw_plist if isinstance(p, dict)),
            key=lambda p: float(p.get("t", 0.0)),
        )
        _apply_timings(row, fobj, pts_sorted)


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


def _flight_apron_stand_id_from_fobj(fobj: Dict[str, Any]) -> Optional[str]:
    tok = fobj.get("token") if isinstance(fobj.get("token"), dict) else {}
    raw = (
        fobj.get("standId")
        or fobj.get("apronId")
        or tok.get("apronId")
        or tok.get("standId")
    )
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _history_destination_stand_id(
    history: List[Tuple[float, ...]],
    *,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
) -> Optional[str]:
    def _extract_sid(dst: Any) -> Optional[str]:
        if isinstance(dst, dict):
            raw = dst.get("standId")
        elif isinstance(dst, (list, tuple)) and dst:
            raw = dst[0]
        else:
            raw = None
        if raw is None:
            return None
        sid = str(raw).strip()
        return sid if sid else None

    for row in history:
        if len(row) <= 6:
            continue
        t_abs = float(row[0])
        if t_lo is not None and t_abs < float(t_lo) - 1e-9:
            continue
        if t_hi is not None and t_abs > float(t_hi) + 1e-9:
            continue
        sid = _extract_sid(row[6])
        if sid:
            return sid
    for row in history:
        if len(row) <= 6:
            continue
        sid = _extract_sid(row[6])
        if sid:
            return sid
    return None


def _resolve_parked_stand_id(
    planned_stand_id: Optional[str],
    history: List[Tuple[float, ...]],
    eibt_sec: float,
    eobt_sec: float,
) -> str:
    hist_sid = _history_destination_stand_id(
        history,
        t_lo=float(eibt_sec),
        t_hi=float(eobt_sec),
    )
    if hist_sid:
        return hist_sid
    planned = str(planned_stand_id or "").strip()
    return planned


def _collect_stand_ids_for_resource_model(layout: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for st in layout.get("pbbStands") or []:
        if not isinstance(st, dict) or st.get("id") is None:
            continue
        s = str(st["id"]).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for st in layout.get("tempStands") or []:
        if not isinstance(st, dict) or st.get("id") is None:
            continue
        s = str(st["id"]).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for fobj in layout.get("flights") or []:
        if not isinstance(fobj, dict):
            continue
        sid = _flight_apron_stand_id_from_fobj(fobj)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _build_temp_stand_incident_edges(
    layout: Dict[str, Any], g: Optional[PathGraph]
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    if g is None or not g.stand_id_to_node_index:
        return out
    raw_edges = layout.get("Edge") or layout.get("edges") or []
    for tst in layout.get("tempStands") or []:
        if not isinstance(tst, dict):
            continue
        tid = str(tst.get("id") or "").strip()
        if not tid:
            continue
        j = g.stand_id_to_node_index.get(tid)
        if j is None:
            # tempStands may not be registered in stand_id_to_node_index; map by nearest graph node.
            txy = _temp_stand_pixel_xy_from_sim_input(tst)
            if txy is not None and g.nodes:
                bx, by = float(txy[0]), float(txy[1])
                j = min(
                    range(len(g.nodes)),
                    key=lambda idx: (g.nodes[idx][0] - bx) ** 2 + (g.nodes[idx][1] - by) ** 2,
                )
        if j is None:
            continue
        se: set[str] = set()
        for ed in raw_edges:
            if not isinstance(ed, dict):
                continue
            eid = str(ed.get("id") or "").strip()
            if not eid:
                continue
            try:
                fi = int(ed["fromIdx"])
                ti = int(ed["toIdx"])
            except (KeyError, TypeError, ValueError):
                continue
            if fi == j or ti == j:
                se.add(eid)
        if se:
            out[tid] = se
    return out


def _yield_temp_occupied_incident_edges_for_pathfinding(
    control_state: SimulationControlState,
    exclude_flight_id: str,
) -> set[str]:
    """
    Layout edge ids touching a **temp** stand graph node while that stand is occupied by
    someone other than ``exclude_flight_id``. Used to penalize Dijkstra so other flights
    avoid temp-park links only for the duration of temp assignment (``stand_resources``).
    """
    out: set[str] = set()
    inc = control_state.temp_stand_incident_edges
    if not inc:
        return out
    aid = str(exclude_flight_id)
    for tid, e_set in inc.items():
        sr = control_state.stand_resources.get(str(tid))
        if sr is None:
            continue
        if not any(str(x) != aid for x in sr.occupied_by):
            continue
        for e in e_set:
            s = str(e).strip()
            if s:
                out.add(s)
    return out


def _flight_default_icao_category(
    fobj: Dict[str, Any], information: Dict[str, Any]
) -> str:
    raw = fobj.get("icaoCategory") or fobj.get("icao_category")
    if raw is not None and str(raw).strip():
        return str(raw).strip().upper()[:1]
    sim = _deep_get(information, "tiers", "algorithm", "simulation", default={}) or {}
    if isinstance(sim, dict) and sim.get("defaultIcaoCategory") is not None:
        try:
            s = str(sim.get("defaultIcaoCategory")).strip().upper()[:1]
            if s:
                return s
        except (TypeError, ValueError):
            pass
    return "C"


def _stand_accepts_flight_aircraft(
    stand: Dict[str, Any],
    fobj: Dict[str, Any],
    information: Dict[str, Any],
) -> bool:
    mode = str(stand.get("categoryMode") or "icao").strip().lower()
    raw_types = stand.get("allowedAircraftTypes") or stand.get("allowed_aircraft_types")
    if isinstance(raw_types, list) and len(raw_types) > 0:
        ac = str(fobj.get("aircraftType") or fobj.get("aircraft_type") or "").strip()
        if not ac:
            return False
        allow = {str(x).strip() for x in raw_types if str(x).strip()}
        return ac in allow
    if mode == "aircraft":
        return False
    scat = str(stand.get("category") or "C").strip().upper()[:1]
    fcat = _flight_default_icao_category(fobj, information)
    order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    fi = int(order.get(fcat, 3))
    si = int(order.get(scat, 3))
    return fi <= si


def _temp_stand_pixel_xy_from_sim_input(stand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Temp stand position in layout pixel space using **only** fields present on the stand
    object from sim_input layout JSON (no PBB apron site, ``x2``/``y2``, or ``col``/``row``).

    Preference: finite ``junctionX``/``junctionY``; else finite ``x``/``y``.
    """
    if not stand or not isinstance(stand, dict):
        return None
    jx, jy = stand.get("junctionX"), stand.get("junctionY")
    try:
        if jx is not None and jy is not None:
            fx, fy = float(jx), float(jy)
            if math.isfinite(fx) and math.isfinite(fy):
                return (fx, fy)
    except (TypeError, ValueError):
        pass
    x, y = stand.get("x"), stand.get("y")
    try:
        if x is not None and y is not None:
            fx, fy = float(x), float(y)
            if math.isfinite(fx) and math.isfinite(fy):
                return (fx, fy)
    except (TypeError, ValueError):
        pass
    return None


def _temp_stand_has_other_claimant_or_occupant(
    tid: str,
    aid: str,
    control_state: SimulationControlState,
    agents: List[Flight],
) -> bool:
    """
    True if temp stand ``tid`` is used by another flight: ``stand_resources`` occupancy
    lists someone other than ``aid``, or any other agent has ``temp_stand_id == tid``.

    Used so temp targets stay consistent with per-tick ``refresh_resource_occupancy`` and
    with same-tick splices before the next refresh.
    """
    sr = control_state.stand_resources.get(tid)
    if sr is not None:
        if any(str(x) != aid for x in sr.occupied_by):
            return True
    for ag2 in agents:
        if str(ag2.id) == aid:
            continue
        if str(ag2.temp_stand_id or "").strip() == tid:
            return True
    return False


def _pick_temp_stand_for_arrival_detour(
    layout: Dict[str, Any],
    fobj: Dict[str, Any],
    dest_stand_id: str,
    control_state: SimulationControlState,
    cell_size: float,
    information: Dict[str, Any],
    agents: List[Flight],
) -> Optional[str]:
    """
    Choose a free temp stand **closest to the destination apron** (pixel space, squared
    Euclidean on ``_temp_stand_pixel_xy_from_sim_input`` vs ``_apron_token_xy``), subject
    to category and occupancy. Ties break on stand id lexicographic order.
    """
    raw_ts = layout.get("tempStands") or []
    if not raw_ts:
        return None
    dest_xy = _apron_token_xy(layout, float(cell_size), str(dest_stand_id))
    if dest_xy is None:
        return None
    aid = str(fobj.get("id") or "")
    ranked: List[Tuple[float, str]] = []
    for tst in raw_ts:
        if not isinstance(tst, dict):
            continue
        tid = str(tst.get("id") or "").strip()
        if not tid:
            continue
        if not _stand_accepts_flight_aircraft(tst, fobj, information):
            continue
        sr = control_state.stand_resources.get(tid)
        if sr is None:
            continue
        if _temp_stand_has_other_claimant_or_occupant(tid, aid, control_state, agents):
            continue
        st_dict = find_stand_by_id(layout, tid)
        if not st_dict:
            continue
        txy = _temp_stand_pixel_xy_from_sim_input(st_dict)
        if txy is None:
            continue
        d2 = (txy[0] - dest_xy[0]) ** 2 + (txy[1] - dest_xy[1]) ** 2
        ranked.append((float(d2), tid))
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked[0][1]


def _first_contiguous_phase_block(agent: Flight, phase: str) -> Optional[Tuple[int, int]]:
    phases = agent.edge_phases
    if not phases:
        return None
    try:
        first = next(i for i, p in enumerate(phases) if str(p) == phase)
    except StopIteration:
        return None
    last = int(first)
    while last + 1 < len(phases) and str(phases[last + 1]) == phase:
        last += 1
    return (int(first), last)


def _prepared_flight_path_tail_from_index(agent: Flight, start_idx: int) -> Optional[PreparedFlightPath]:
    n = len(agent.edge_ids)
    if start_idx < 0 or start_idx >= n:
        return None
    pty = (
        list(agent.segment_path_types[start_idx:])
        if agent.segment_path_types and len(agent.segment_path_types) == n
        else []
    )
    guv = (
        list(agent.segment_graph_uv[start_idx:])
        if agent.segment_graph_uv and len(agent.segment_graph_uv) == n
        else []
    )
    return PreparedFlightPath(
        edge_ids=list(agent.edge_ids[start_idx:]),
        segment_phases=list(agent.edge_phases[start_idx:]),
        segment_endpoints=[(tuple(a), tuple(b)) for a, b in agent.segment_endpoints[start_idx:]],
        segment_start_velocity_ms=list(agent.segment_v0_ms[start_idx:]),
        segment_accel_ms2=list(agent.segment_accel_ms2[start_idx:]),
        segment_path_types=pty,
        segment_graph_uv=guv,
        ok=True,
    )


def _merge_prepared_flight_paths(
    a: PreparedFlightPath, b: PreparedFlightPath
) -> PreparedFlightPath:
    if not b.edge_ids:
        return PreparedFlightPath(
            edge_ids=list(a.edge_ids),
            segment_phases=list(a.segment_phases),
            segment_endpoints=list(a.segment_endpoints),
            segment_link_ids=list(a.segment_link_ids),
            segment_path_types=list(a.segment_path_types),
            segment_graph_uv=list(a.segment_graph_uv),
            segment_start_velocity_ms=list(a.segment_start_velocity_ms),
            segment_accel_ms2=list(a.segment_accel_ms2),
            segment_duration_sec=list(a.segment_duration_sec),
            ok=a.ok,
        )
    if not a.edge_ids:
        return PreparedFlightPath(
            edge_ids=list(b.edge_ids),
            segment_phases=list(b.segment_phases),
            segment_endpoints=list(b.segment_endpoints),
            segment_link_ids=list(b.segment_link_ids),
            segment_path_types=list(b.segment_path_types),
            segment_graph_uv=list(b.segment_graph_uv),
            segment_start_velocity_ms=list(b.segment_start_velocity_ms),
            segment_accel_ms2=list(b.segment_accel_ms2),
            segment_duration_sec=list(b.segment_duration_sec),
            ok=b.ok,
        )
    n_a = len(a.edge_ids)
    n_b = len(b.edge_ids)
    merged_ids = list(a.edge_ids) + list(b.edge_ids)
    merged_ph = list(a.segment_phases) + list(b.segment_phases)
    merged_se = list(a.segment_endpoints) + list(b.segment_endpoints)
    merged_v0 = list(a.segment_start_velocity_ms) + list(b.segment_start_velocity_ms)
    merged_acc = list(a.segment_accel_ms2) + list(b.segment_accel_ms2)
    merged_dur = list(a.segment_duration_sec) + list(b.segment_duration_sec)
    merged_lnk = []
    if a.segment_link_ids and len(a.segment_link_ids) == n_a:
        merged_lnk.extend(a.segment_link_ids)
    if b.segment_link_ids and len(b.segment_link_ids) == n_b:
        merged_lnk.extend(b.segment_link_ids)
    merged_pt: List[str] = []
    if a.segment_path_types and len(a.segment_path_types) == n_a:
        merged_pt.extend(a.segment_path_types)
    if b.segment_path_types and len(b.segment_path_types) == n_b:
        merged_pt.extend(b.segment_path_types)
    merged_guv: List[Tuple[int, int]] = []
    if a.segment_graph_uv and len(a.segment_graph_uv) == n_a:
        merged_guv.extend(a.segment_graph_uv)
    if b.segment_graph_uv and len(b.segment_graph_uv) == n_b:
        merged_guv.extend(b.segment_graph_uv)
    return PreparedFlightPath(
        edge_ids=merged_ids,
        segment_phases=merged_ph,
        segment_endpoints=merged_se,
        segment_link_ids=merged_lnk,
        segment_path_types=merged_pt,
        segment_graph_uv=merged_guv,
        segment_start_velocity_ms=merged_v0,
        segment_accel_ms2=merged_acc,
        segment_duration_sec=merged_dur,
        ok=bool(a.ok and b.ok),
    )


def _snap_segment_endpoints_to_stand_px(
    segs: List[Tuple[Point, Point]],
    *,
    start_xy: Optional[Tuple[float, float]] = None,
    end_xy: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Mutate expanded micro-segments so playback reaches exact layout stand pixels.

    With ``simPathGraph``, Dijkstra ends at the nearest exported node (often a taxiway
    junction), not necessarily the graph node. Temp detour and temp→apron legs snap to
    the sim_input temp anchor from ``_temp_stand_pixel_xy_from_sim_input``.
    """
    if not segs:
        return
    if end_xy is not None:
        ex, ey = float(end_xy[0]), float(end_xy[1])
        p0, _p1 = segs[-1]
        segs[-1] = ((float(p0[0]), float(p0[1])), (ex, ey))
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        _p0, p1 = segs[0]
        segs[0] = ((sx, sy), (float(p1[0]), float(p1[1])))


def _build_prep_xy_to_xy_phase(
    layout: Dict[str, Any],
    cell_size: float,
    fobj: Dict[str, Any],
    agent: Flight,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    phase_str: str,
    information: Dict[str, Any],
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    *,
    snap_exact_start_xy: bool = False,
    snap_exact_end_xy: bool = False,
    control_state: Optional[SimulationControlState] = None,
) -> Optional[PreparedFlightPath]:
    pair_index = _pair_index_from_layout_edge(layout)
    if not pair_index:
        g0 = _cached_path_graph_for_direction(
            layout,
            float(cell_size),
            _flight_rw_dir_for_leg(fobj, _leg_index_for_phase(phase_str), layout),
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            pure_ground_exclude_runway=False,
        )
        pair_index = _pair_index_from_path_graph(g0) if g0 else {}
    if not pair_index:
        return None
    leg_i = _leg_index_for_phase(phase_str)
    rw_dir = _flight_rw_dir_for_leg(fobj, leg_i, layout)
    temp_pen: Optional[set[str]] = None
    if control_state is not None:
        tb = _yield_temp_occupied_incident_edges_for_pathfinding(
            control_state, str(agent.id)
        )
        if tb:
            temp_pen = tb
    edges, dv, path, g = _flight_route_impl(
        layout,
        float(cell_size),
        pair_index,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        rw_dir,
        RouteEndpoint(token_pixel_xy=(float(start_xy[0]), float(start_xy[1]))),
        RouteEndpoint(token_pixel_xy=(float(end_xy[0]), float(end_xy[1]))),
        penalized_layout_edges=temp_pen,
        penalty_add=float(REVERSE_PENALTY_COST) if temp_pen else 0.0,
    )
    if dv or path is None or g is None or len(path) < 2:
        return None
    if _graph_path_has_disallowed_reverse_of_prior_hops(
        g, path, pair_index, agent.completed_directed_hops
    ):
        return None
    ex_ids, segs, phs, lnks, ptyps, guvs = _expand_geometry_from_graph_path(
        g, path, pair_index, phase_str
    )
    if (
        not ex_ids
        or not segs
        or len(ex_ids) != len(segs)
        or len(phs) != len(ex_ids)
        or len(lnks) != len(ex_ids)
        or len(ptyps) != len(ex_ids)
        or len(guvs) != len(ex_ids)
    ):
        return None
    if snap_exact_end_xy:
        _snap_segment_endpoints_to_stand_px(segs, end_xy=end_xy)
    if snap_exact_start_xy:
        _snap_segment_endpoints_to_stand_px(segs, start_xy=start_xy)
    fid = str(fobj.get("id", agent.id))
    ppm = max(float(_layout_pixels_per_meter(information)), 1e-9)
    try:
        v0s, accs, durs = _annotate_segment_kinematics(
            fobj,
            layout,
            phs,
            segs,
            lnks,
            ptyps,
            ppm,
            fid,
            information,
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
        segment_link_ids=list(lnks),
        segment_path_types=list(ptyps),
        segment_graph_uv=list(guvs),
        segment_start_velocity_ms=list(v0s),
        segment_accel_ms2=list(accs),
        segment_duration_sec=list(durs),
        ok=True,
    )


def _apply_prep_to_agent(
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
    agent.segment_graph_uv = list(prep.segment_graph_uv) if prep.segment_graph_uv else []
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
    st.reroute_attempts = int(st.reroute_attempts)
    st.clearance = "PROCEED"
    st.wait_reason = None
    st.wait_start_sec = None
    st.stagnation_anchor_sec = None
    st.progress_snapshot_edge_id = None
    st.progress_snapshot_along_m = 0.0


def _temp_stand_pipeline_sort_key(
    ag: Flight,
    flights_by_id: Dict[str, Dict[str, Any]],
    flight_input_order: Dict[str, int],
) -> Tuple[float, float, int, str]:
    """
    Order for temp inject / reroute / splice in one tick: **earlier flight picks the
    nearer temp first** (T001 before T002 when both need a slot the same tick).

    1. Earlier ``eldt_anchor_sec``
    2. Lower ``sldtMin_d`` from sim flight input
    3. Lower index in ``layout[\"flights\"]`` (먼저 입력·먼저 온 편)
    4. ``str(ag.id)`` (last resort)
    """
    eldt = (
        float(ag.eldt_anchor_sec) if ag.eldt_anchor_sec is not None else float("inf")
    )
    sldt_d = float("inf")
    fo = flights_by_id.get(str(ag.id))
    if isinstance(fo, dict):
        raw = fo.get("sldtMin_d")
        if raw is not None:
            try:
                sldt_d = float(raw)
            except (TypeError, ValueError):
                pass
    ord_i = int(flight_input_order.get(str(ag.id), 10**9))
    return (eldt, sldt_d, ord_i, str(ag.id))


def _tick_arr_temp_detour_eldt_flag(agent: Flight, current_time_abs: float) -> None:
    if agent.eldt_anchor_sec is None:
        return
    if float(current_time_abs) + 1e-9 >= float(agent.eldt_anchor_sec):
        agent.arr_temp_detour_decided = True


def _try_splice_temp_stand_arrival_detour(
    ag: Flight,
    fobj: Dict[str, Any],
    layout: Dict[str, Any],
    control_state: SimulationControlState,
    information: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    current_time_abs: float,
    agents: List[Flight],
) -> None:
    if ag.arr_temp_detour_decided and not ag.temp_stand_id:
        return
    if ag.temp_stand_id or ag.awaiting_apron_from_temp:
        return
    if not ag.edge_phases or not ag.edge_ids:
        return
    ph0 = str(ag.edge_phases[0])
    if ph0 == PHASE_ARR_TAXI:
        ag.arr_temp_detour_decided = True
        return
    if ag.eldt_anchor_sec is None:
        ag.arr_temp_detour_decided = True
        return
    if float(current_time_abs) + 1e-9 < float(ag.eldt_anchor_sec) - float(
        ARR_TEMP_DETOUR_DECISION_LEAD_SEC
    ):
        return
    if float(current_time_abs) + 1e-9 >= float(ag.eldt_anchor_sec):
        return
    if ph0 != PHASE_LANDING:
        ag.arr_temp_detour_decided = True
        return
    block = _first_contiguous_phase_block(ag, PHASE_ARR_TAXI)
    if block is None:
        return
    first_arr, last_arr = block
    dep_idx = last_arr + 1
    if dep_idx >= len(ag.edge_phases) or str(ag.edge_phases[dep_idx]) != PHASE_DEP_TAXI:
        ag.arr_temp_detour_decided = True
        return
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        ag.arr_temp_detour_decided = True
        return
    t_abs = float(current_time_abs)
    busy = _target_apron_stand_occupied_by_other(ag, control_state) or (
        _stand_pushback_clearance_cooldown_active(sid, str(ag.id), agents, t_abs)
    )
    if not busy:
        return
    temp_id = _pick_temp_stand_for_arrival_detour(
        layout, fobj, sid, control_state, float(cell_size), information, agents
    )
    if not temp_id:
        return
    p0_arr = ag.segment_endpoints[first_arr][0]
    start_xy = (float(p0_arr[0]), float(p0_arr[1]))
    tst = find_stand_by_id(layout, temp_id)
    if not tst:
        return
    end_xy = _temp_stand_pixel_xy_from_sim_input(tst)
    if end_xy is None:
        return
    temp_prep = _build_prep_xy_to_xy_phase(
        layout,
        float(cell_size),
        fobj,
        ag,
        start_xy,
        end_xy,
        PHASE_ARR_TAXI_TEMP,
        information,
        reverse_cost,
        merge_r,
        taxiway_h,
        snap_exact_start_xy=True,
        snap_exact_end_xy=True,
        control_state=control_state,
    )
    if not temp_prep or not temp_prep.edge_ids:
        return
    tail_prep = _prepared_flight_path_tail_from_index(ag, dep_idx)
    if tail_prep is None or not tail_prep.edge_ids:
        return
    prefix_slice = slice(0, first_arr)
    ag.edge_ids = list(ag.edge_ids[prefix_slice]) + list(temp_prep.edge_ids)
    ag.edge_phases = list(ag.edge_phases[prefix_slice]) + list(temp_prep.segment_phases)
    ag.segment_endpoints = list(ag.segment_endpoints[prefix_slice]) + list(
        temp_prep.segment_endpoints
    )
    ag.segment_v0_ms = list(ag.segment_v0_ms[prefix_slice]) + list(
        temp_prep.segment_start_velocity_ms
    )
    ag.segment_accel_ms2 = list(ag.segment_accel_ms2[prefix_slice]) + list(
        temp_prep.segment_accel_ms2
    )
    ptn = len(ag.edge_ids)
    ag.segment_path_types = (
        list(ag.segment_path_types[prefix_slice]) + list(temp_prep.segment_path_types)
        if temp_prep.segment_path_types and len(temp_prep.segment_path_types) == len(temp_prep.edge_ids)
        else []
    )
    if ag.segment_path_types and len(ag.segment_path_types) != ptn:
        ag.segment_path_types = []
    ag.segment_graph_uv = (
        list(ag.segment_graph_uv[prefix_slice]) + list(temp_prep.segment_graph_uv)
        if temp_prep.segment_graph_uv and len(temp_prep.segment_graph_uv) == len(temp_prep.edge_ids)
        else []
    )
    if ag.segment_graph_uv and len(ag.segment_graph_uv) != ptn:
        ag.segment_graph_uv = []
    ag.temp_stand_id = temp_id
    ag.post_temp_route_tail_prep = tail_prep
    ag.arr_temp_detour_decided = True
    refresh_agent_edge_fsm([ag])


def _try_reroute_temp_stand_if_contested(
    ag: Flight,
    fobj: Dict[str, Any],
    layout: Dict[str, Any],
    control_state: SimulationControlState,
    information: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    current_time_abs: float,
    agents: List[Flight],
) -> None:
    tid = str(ag.temp_stand_id or "").strip()
    if not tid or ag.awaiting_apron_from_temp:
        return
    if not ag.edge_phases or not ag.edge_ids:
        return
    aid = str(ag.id)
    if not _temp_stand_has_other_claimant_or_occupant(tid, aid, control_state, agents):
        return
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return
    t_abs = float(current_time_abs)
    busy = _target_apron_stand_occupied_by_other(ag, control_state) or (
        _stand_pushback_clearance_cooldown_active(sid, aid, agents, t_abs)
    )
    if not busy:
        return
    new_id = _pick_temp_stand_for_arrival_detour(
        layout, fobj, sid, control_state, float(cell_size), information, agents
    )
    if not new_id or new_id == tid:
        return
    block = _first_contiguous_phase_block(ag, PHASE_ARR_TAXI_TEMP)
    if block is None:
        return
    temp_first, temp_last = int(block[0]), int(block[1])
    ph0 = str(ag.edge_phases[0])
    if ph0 == PHASE_ARR_TAXI_TEMP:
        start_xy = (float(ag.col), float(ag.row))
        reset_along = True
    else:
        p0 = ag.segment_endpoints[temp_first][0]
        start_xy = (float(p0[0]), float(p0[1]))
        reset_along = False
    tst = find_stand_by_id(layout, new_id)
    if not tst:
        return
    end_xy = _temp_stand_pixel_xy_from_sim_input(tst)
    if end_xy is None:
        return
    temp_prep = _build_prep_xy_to_xy_phase(
        layout,
        float(cell_size),
        fobj,
        ag,
        start_xy,
        end_xy,
        PHASE_ARR_TAXI_TEMP,
        information,
        reverse_cost,
        merge_r,
        taxiway_h,
        snap_exact_end_xy=True,
        control_state=control_state,
    )
    if not temp_prep or not temp_prep.edge_ids:
        return
    tail_i = temp_last + 1
    ag.edge_ids = (
        list(ag.edge_ids[:temp_first])
        + list(temp_prep.edge_ids)
        + list(ag.edge_ids[tail_i:])
    )
    ag.edge_phases = (
        list(ag.edge_phases[:temp_first])
        + list(temp_prep.segment_phases)
        + list(ag.edge_phases[tail_i:])
    )
    ag.segment_endpoints = (
        list(ag.segment_endpoints[:temp_first])
        + list(temp_prep.segment_endpoints)
        + list(ag.segment_endpoints[tail_i:])
    )
    ag.segment_v0_ms = (
        list(ag.segment_v0_ms[:temp_first])
        + list(temp_prep.segment_start_velocity_ms)
        + list(ag.segment_v0_ms[tail_i:])
    )
    ag.segment_accel_ms2 = (
        list(ag.segment_accel_ms2[:temp_first])
        + list(temp_prep.segment_accel_ms2)
        + list(ag.segment_accel_ms2[tail_i:])
    )
    ptn = len(ag.edge_ids)
    ag.segment_path_types = (
        list(ag.segment_path_types[:temp_first]) + list(temp_prep.segment_path_types)
        if temp_prep.segment_path_types
        and len(temp_prep.segment_path_types) == len(temp_prep.edge_ids)
        else []
    )
    if ag.segment_path_types and len(ag.segment_path_types) != ptn:
        ag.segment_path_types = []
    ag.segment_graph_uv = (
        list(ag.segment_graph_uv[:temp_first]) + list(temp_prep.segment_graph_uv)
        if temp_prep.segment_graph_uv
        and len(temp_prep.segment_graph_uv) == len(temp_prep.edge_ids)
        else []
    )
    if ag.segment_graph_uv and len(ag.segment_graph_uv) != ptn:
        ag.segment_graph_uv = []
    ag.temp_stand_id = new_id
    if reset_along:
        ag.edge_s_along_px = 0.0
    refresh_agent_edge_fsm([ag])


def _try_inject_arr_taxi_from_temp_stand(
    ag: Flight,
    fobj: Dict[str, Any],
    layout: Dict[str, Any],
    control_state: SimulationControlState,
    information: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    current_time_abs: float,
    agents: List[Flight],
) -> None:
    if not ag.awaiting_apron_from_temp:
        return
    if ag.edge_ids:
        return
    sid = str(ag.apron_stand_id or "").strip()
    tid = str(ag.temp_stand_id or "").strip()
    tail = ag.post_temp_route_tail_prep
    if not sid or not tid or tail is None or not tail.edge_ids:
        ag.awaiting_apron_from_temp = False
        ag.post_temp_route_tail_prep = None
        ag.temp_stand_id = None
        ag.temp_dest_apron_cleared_abs_sec = None
        return
    t_abs = float(current_time_abs)
    if _target_apron_stand_occupied_by_other(ag, control_state):
        ag.temp_dest_apron_cleared_abs_sec = None
        return
    if ag.temp_dest_apron_cleared_abs_sec is None:
        ag.temp_dest_apron_cleared_abs_sec = t_abs
    if t_abs + 1e-9 < float(ag.temp_dest_apron_cleared_abs_sec) + float(
        TEMP_TO_APRON_HOLD_SEC
    ):
        return
    if _stand_pushback_clearance_cooldown_active(sid, str(ag.id), agents, t_abs):
        return
    tst = find_stand_by_id(layout, tid)
    if not tst:
        return
    start_xy = _temp_stand_pixel_xy_from_sim_input(tst)
    if start_xy is None:
        return
    end_xy = _apron_token_xy(layout, float(cell_size), sid)
    if end_xy is None:
        return
    arr_prep = _build_prep_xy_to_xy_phase(
        layout,
        float(cell_size),
        fobj,
        ag,
        start_xy,
        end_xy,
        PHASE_ARR_TAXI,
        information,
        reverse_cost,
        merge_r,
        taxiway_h,
        snap_exact_start_xy=True,
        control_state=control_state,
    )
    if not arr_prep or not arr_prep.edge_ids:
        return
    merged = _merge_prepared_flight_paths(arr_prep, tail)
    if (
        not merged.ok
        or not merged.edge_ids
        or len(merged.segment_start_velocity_ms) != len(merged.edge_ids)
        or len(merged.segment_accel_ms2) != len(merged.edge_ids)
    ):
        return
    ag.temp_stand_id = None
    ag.post_temp_route_tail_prep = None
    ag.awaiting_apron_from_temp = False
    ag.temp_dest_apron_cleared_abs_sec = None
    _apply_prep_to_agent(ag, merged, control_state, t_abs)
    refresh_agent_edge_fsm([ag])


def _agent_occupies_temp_stand_slot(
    ag: Flight, t_abs: float, control_state: SimulationControlState
) -> Optional[str]:
    tid = str(ag.temp_stand_id or "").strip()
    if not tid:
        return None
    st0 = control_state.agent_states.get(ag.id)
    if st0 is not None and _agent_deadlock_ghost_at_time(st0, float(t_abs)):
        return None
    # Physical occupancy only: agent must be PARKED at the temp stand
    # (``awaiting_apron_from_temp`` flips True at temp-leg finish). Pre-arrival
    # claim exclusion for _pick_temp_stand_for_arrival_detour is already enforced
    # by _temp_stand_has_other_claimant_or_occupant via ``ag2.temp_stand_id`` scan,
    # so occupied_by does not need to double-guard the claim. Reporting claim-time
    # occupancy here would also trigger ``temp_stand_busy`` in can_reserve_path for
    # pass-through traffic on taxiway edges incident to the temp stand's graph node,
    # blocking downstream flights from reaching other temp stands on the same
    # corridor (e.g., a later flight routed to T001 stuck behind T002 once T002's
    # claimant has set ``temp_stand_id`` but has not yet arrived).
    if not ag.awaiting_apron_from_temp:
        return None
    return tid


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
    g = _cached_path_graph_for_direction(
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
            path_type=str(path_type or "taxiway"),
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
    stand_resources: Dict[str, StandResource] = {}
    for sid in _collect_stand_ids_for_resource_model(layout):
        cap = _layout_stand_capacity_for_id(layout, sid, information)
        stand_resources[str(sid)] = StandResource(stand_id=str(sid), capacity=cap)
    temp_inc = _build_temp_stand_incident_edges(layout, g)
    return SimulationControlState(
        edge_resources=edge_resources,
        intersection_resources=intersection_resources,
        runway_resources=runway_resources,
        stand_resources=stand_resources,
        agent_states={},
        path_graph=g,
        pixels_per_meter=ppm,
        temp_stand_incident_edges=temp_inc,
    )


def ensure_agent_control_states(control_state: SimulationControlState, agents: Iterable[Flight]) -> None:
    for ag in agents:
        fid = str(ag.id)
        if fid not in control_state.agent_states:
            control_state.agent_states[fid] = AgentControlState(flight_id=fid)


def _agent_occupies_apron_stand_slot(
    ag: Flight, t_abs: float, control_state: SimulationControlState
) -> Optional[str]:
    """Stand capacity: Arr_taxi on ``apron_link`` (not holding outside for stand), or in-blocks…off-blocks."""
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return None
    tt = float(t_abs)
    st0 = control_state.agent_states.get(ag.id)
    if st0 is not None and _agent_deadlock_ghost_at_time(st0, tt):
        return None
    ob = ag.actual_apron_offblocks_abs_sec
    if ob is not None and tt + 1e-9 >= float(ob):
        return None
    if not ag.edge_phases or not ag.edge_ids:
        return None
    ph0 = str(ag.edge_phases[0])
    pt0 = (
        str(ag.segment_path_types[0] or "")
        if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
        else ""
    )
    if ph0 == PHASE_ARR_TAXI and pt0 == "apron_link":
        st = control_state.agent_states.get(ag.id)
        if (
            st
            and st.clearance in ("WAIT", "YIELD")
            and st.wait_reason == "stand_occupied"
        ):
            return None
        return sid
    if ag.actual_apron_inblocks_abs_sec is not None:
        return sid
    return None


def _try_stamp_actual_apron_inblocks_from_stand_position(
    ag: Flight,
    layout: Dict[str, Any],
    cell_size: float,
    pixels_per_meter: float,
    information: Dict[str, Any],
    t_abs: float,
    control_state: SimulationControlState,
    agents: List[Flight],
) -> None:
    """EIBT: first time near stand token on Arr_taxi with negligible speed (any link except pure runway)."""
    if ag.actual_apron_inblocks_abs_sec is not None:
        return
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return
    if not ag.edge_phases or str(ag.edge_phases[0]) != PHASE_ARR_TAXI:
        return
    if not ag.segment_path_types or len(ag.segment_path_types) != len(ag.edge_ids):
        return
    pt0s = str(ag.segment_path_types[0] or "").strip()
    if pt0s == "runway":
        return
    txy = _apron_token_xy(layout, float(cell_size), sid)
    if txy is None:
        return
    ppm = max(float(pixels_per_meter), 1e-9)
    r_m = _sim_stand_arrival_stop_radius_m(information)
    v_max = _sim_stand_stopped_velocity_max_ms(information)
    d_m = math.hypot(float(ag.col) - txy[0], float(ag.row) - txy[1]) / ppm
    if d_m > r_m + 1e-6:
        return
    if abs(float(ag.velocity_ms)) > v_max + 1e-9:
        return
    st_ag = control_state.agent_states.get(ag.id)
    if st_ag is not None and st_ag.clearance in ("WAIT", "YIELD"):
        return
    if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, float(t_abs)):
        return
    if not _stand_pipeline_allows_apron_inblocks_stamp(
        ag, control_state, agents, float(t_abs)
    ):
        return
    t_in = float(t_abs)
    ag.actual_apron_inblocks_abs_sec = t_in
    ag.dep_taxi_start_abs_sec = t_in + float(ag.dwell_sec)
    if ag.eldt_anchor_sec is not None:
        ag.dep_taxi_start_sim_time = float(ag.dep_taxi_start_abs_sec) - float(
            ag.eldt_anchor_sec
        )


def refresh_agent_edge_fsm(agents: Iterable[Flight]) -> None:
    for ag in agents:
        if ag.edge_ids:
            ag.current_edge_id = str(ag.edge_ids[0])
            ag.next_edge_id = str(ag.edge_ids[1]) if len(ag.edge_ids) > 1 else None
            if ag.segment_endpoints:
                p0, p1 = ag.segment_endpoints[0]
                h = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                if (
                    not ag.motion_is_forward
                    and ag.segment_path_types
                    and str(ag.segment_path_types[0] or "") == "apron_link"
                ):
                    h += math.pi
                ag.heading_rad = h
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
    for sr in control_state.stand_resources.values():
        sr.occupied_by.clear()
    g = control_state.path_graph
    ppm = max(float(pixels_per_meter), 1e-9)
    rad_px = NODE_OCCUPANCY_RADIUS_M * ppm
    t_abs = float(sim_time_abs)
    rw_lag = float(runway_release_lag_sec)
    for ag in agents:
        td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td is not None and t_abs + 1e-9 < float(td):
            continue
        st_ag = control_state.agent_states.get(ag.id)
        if st_ag is not None and _agent_deadlock_ghost_at_time(st_ag, t_abs):
            continue
        if not ag.edge_ids:
            temp_wait = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
            if temp_wait:
                sr_w = control_state.stand_resources.get(temp_wait)
                if sr_w is not None and ag.id not in sr_w.occupied_by:
                    sr_w.occupied_by.append(ag.id)
            continue
        eid0 = str(ag.edge_ids[0])
        er = control_state.edge_resources.get(eid0)
        if er and ag.id not in er.occupied_by:
            er.occupied_by.append(ag.id)
        if er and er.runway_id:
            rr = control_state.runway_resources.get(str(er.runway_id))
            if rr and ag.id not in rr.occupied_by:
                rr.occupied_by.append(ag.id)
        dep_rw = str(ag.dep_runway_id or "").strip()
        ph0 = str(ag.edge_phases[0]) if ag.edge_phases else ""
        pt0 = (
            str(ag.segment_path_types[0] or "")
            if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
            else ""
        )
        if (
            dep_rw
            and ph0 in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE)
            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_d = control_state.runway_resources.get(dep_rw)
            if rr_d and ag.id not in rr_d.occupied_by:
                rr_d.occupied_by.append(ag.id)
        sid_slot = _agent_occupies_apron_stand_slot(ag, t_abs, control_state)
        if sid_slot:
            sr = control_state.stand_resources.get(sid_slot)
            if sr is not None and ag.id not in sr.occupied_by:
                sr.occupied_by.append(ag.id)
        temp_slot = _agent_occupies_temp_stand_slot(ag, t_abs, control_state)
        if temp_slot:
            sr_t = control_state.stand_resources.get(temp_slot)
            if sr_t is not None and ag.id not in sr_t.occupied_by:
                sr_t.occupied_by.append(ag.id)
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
    """Uses :data:`AGENT_PRIORITY_*` constants (global movement priority)."""
    if not agent.edge_phases:
        return AGENT_PRIORITY_UNKNOWN
    ph = str(agent.edge_phases[0])
    pt = (
        str(agent.segment_path_types[0] or "")
        if agent.segment_path_types and len(agent.segment_path_types) == len(agent.edge_ids)
        else ""
    )
    pt_s = str(pt or "").strip()
    rw_pts = ("runway", "runway_taxiway")
    apron_pts = ("apron_link", "apron_taxiway")
    if ph == PHASE_LANDING:
        return AGENT_PRIORITY_RUNWAY_ARR_DEP
    if ph in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE):
        return AGENT_PRIORITY_LINEUP_HOLDING
    if ph in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP, PHASE_DEP_TAXI) and pt_s in rw_pts:
        return AGENT_PRIORITY_RUNWAY_ARR_DEP
    if pt_s in apron_pts:
        return AGENT_PRIORITY_APRON_TRANSIT
    if ph == PHASE_DEP_TAXI:
        return AGENT_PRIORITY_DEP_TAXI
    if ph in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
        return AGENT_PRIORITY_ARR_TAXI
    return AGENT_PRIORITY_UNKNOWN


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


def _prefix_has_apron_taxiway_edges(
    agent: Flight,
    up_to_idx_inclusive: int,
    control_state: SimulationControlState,
) -> bool:
    hi = int(up_to_idx_inclusive)
    if hi < 0 or not agent.edge_ids:
        return False
    for j in range(0, min(hi + 1, len(agent.edge_ids))):
        pt = _layout_edge_path_type(control_state, str(agent.edge_ids[j]))
        if str(pt or "").strip() == "apron_taxiway":
            return True
    return False


def _target_apron_stand_occupied_by_other(
    agent: Flight, control_state: SimulationControlState
) -> bool:
    sid = str(agent.apron_stand_id or "").strip()
    if not sid:
        return False
    sr = control_state.stand_resources.get(sid)
    if sr is None:
        return False
    aid = str(agent.id)
    return any(str(x) != aid for x in sr.occupied_by)


def _stand_pushback_clearance_cooldown_active(
    stand_id: str,
    excluding_agent_id: str,
    agents: List[Flight],
    t_abs: float,
) -> bool:
    """True if another flight that uses ``stand_id`` finished pushback recently (stand not yet open)."""
    sid = str(stand_id).strip()
    if not sid:
        return False
    t = float(t_abs)
    delay = float(STAND_POST_PUSHBACK_CLEARANCE_DELAY_SEC)
    ex = str(excluding_agent_id)
    for oth in agents:
        if str(oth.id) == ex:
            continue
        if str(oth.apron_stand_id or "").strip() != sid:
            continue
        ob = oth.actual_apron_offblocks_abs_sec
        if ob is None:
            continue
        if t + 1e-9 < float(ob) + delay:
            return True
    return False


def _stand_pipeline_allows_apron_inblocks_stamp(
    ag: Flight,
    control_state: SimulationControlState,
    agents: List[Flight],
    t_abs: float,
) -> bool:
    """Destination stand can accept this arrival for EIBT: capacity and post-pushback cooldown."""
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return False
    sr = control_state.stand_resources.get(sid)
    if sr is None:
        return True
    aid = str(ag.id)
    cap = max(1, int(sr.capacity))
    phys_others = len({str(x) for x in sr.occupied_by if str(x) != aid})
    if phys_others >= cap:
        return False
    if _stand_pushback_clearance_cooldown_active(sid, aid, agents, float(t_abs)):
        return False
    return True


def _destination_stand_history_snap(
    ag: Flight,
    control_state: SimulationControlState,
    agents: List[Flight],
    t_abs: float,
) -> Optional[DestinationStandHistorySnap]:
    """
    Per-step stand/apron diagnostics for export: matches ``can_reserve_path`` stand slot check when
    a stand resource exists (phys + pipeline book vs capacity, post-pushback cooldown). If the stand
    id is missing from ``stand_resources``, reservation does not apply a stand block — export
    ``capacity`` 0 and ``standPipelineOpen`` True.
    Pipeline book uses the last heavy-decision tick snapshot (``stand_arrival_book_snapshot``).
    """
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return None
    booked = int(control_state.stand_arrival_book_snapshot.get(sid, 0))
    cd = _stand_pushback_clearance_cooldown_active(sid, str(ag.id), agents, float(t_abs))
    sr = control_state.stand_resources.get(sid)
    aid = ag.id
    if sr is None:
        return (sid, 0, 0, booked, cd, True)
    cap = max(1, int(sr.capacity))
    phys_others = len({x for x in sr.occupied_by if x != aid})
    allow = phys_others + booked < cap and not cd
    return (sid, cap, phys_others, booked, cd, allow)


def _lookahead_and_reservation_depth_for_agent(
    agent: Flight,
    control_state: SimulationControlState,
    sim_time_abs: float,
    agents: List[Flight],
) -> Tuple[int, int]:
    """(lookahead_edge_count, billed_reservation_depth) by runway / Dep_taxi / Arr_taxi / Arr busy."""
    if not agent.edge_phases:
        return (LOOKAHEAD_ARR_TAXI, RESERV_DEPTH_ARR_TAXI)
    ph0 = str(agent.edge_phases[0])
    pt0 = (
        str(agent.segment_path_types[0] or "").strip()
        if agent.segment_path_types and len(agent.segment_path_types) == len(agent.edge_ids)
        else ""
    )
    rw_pts = ("runway", "runway_taxiway")
    on_runway_seg = pt0 in rw_pts
    if ph0 == PHASE_LANDING:
        return (LOOKAHEAD_RUNWAY, RESERV_DEPTH_RUNWAY)
    if ph0 in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP, PHASE_DEP_TAXI) and on_runway_seg:
        return (LOOKAHEAD_RUNWAY, RESERV_DEPTH_RUNWAY)
    if ph0 in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE):
        if on_runway_seg:
            return (LOOKAHEAD_RUNWAY, RESERV_DEPTH_RUNWAY)
        return (LOOKAHEAD_DEP_TAXI, RESERV_DEPTH_DEP_TAXI)
    if ph0 == PHASE_DEP_TAXI:
        return (LOOKAHEAD_DEP_TAXI, RESERV_DEPTH_DEP_TAXI)
    if ph0 in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
        eldt = agent.eldt_anchor_sec
        t_abs = float(sim_time_abs)
        eldt_reached = eldt is None or t_abs + 1e-9 >= float(eldt)
        sid = str(agent.apron_stand_id or "").strip()
        stand_busy = _target_apron_stand_occupied_by_other(agent, control_state) or (
            sid
            and _stand_pushback_clearance_cooldown_active(sid, str(agent.id), agents, t_abs)
        )
        if ph0 == PHASE_ARR_TAXI and eldt_reached and stand_busy:
            return (LOOKAHEAD_ARR_TAXI_BUSY, RESERV_DEPTH_ARR_TAXI_BUSY)
        return (LOOKAHEAD_ARR_TAXI, RESERV_DEPTH_ARR_TAXI)
    return (LOOKAHEAD_ARR_TAXI, RESERV_DEPTH_ARR_TAXI)


def get_lookahead_edges(
    agent: Flight,
    n: int,
    control_state: SimulationControlState,
) -> List[str]:
    if not agent.edge_ids:
        return []
    T = max(1, int(n))
    n_e = len(agent.edge_ids)
    k_max = min(n_e, max(T * 80, 400))
    for k in range(k_max):
        b = _lookahead_depth_billed_count(agent, k, control_state)
        if b >= T:
            return [str(agent.edge_ids[i]) for i in range(k + 1)]
        if (
            k + 1 == T
            and b < T
            and not _prefix_has_apron_taxiway_edges(agent, k, control_state)
        ):
            return [str(agent.edge_ids[i]) for i in range(T)]
    return [str(agent.edge_ids[i]) for i in range(min(T, n_e))]


def _temp_apron_hold_reservation_only_current_edge(ag: Flight) -> bool:
    """True while on temp taxi leg or waiting to inject apron after temp (``temp_stand_id`` set)."""
    if not str(ag.temp_stand_id or "").strip():
        return False
    if ag.awaiting_apron_from_temp:
        return True
    return bool(ag.edge_phases) and str(ag.edge_phases[0]) == PHASE_ARR_TAXI_TEMP


def _temp_apron_current_edge_lookahead(ag: Flight) -> Optional[List[str]]:
    """Single edge id: active segment head, else last finished segment while awaiting apron from temp."""
    if ag.edge_ids:
        return [str(ag.edge_ids[0])]
    if ag.awaiting_apron_from_temp and ag.edge_ids_finished:
        tail = ag.edge_ids_finished[-1]
        if isinstance(tail, dict):
            raw = tail.get("edge_id")
            if raw is not None and str(raw).strip():
                return [str(raw).strip()]
    return None


def _agent_eligible_for_reservation_pass(ag: Flight) -> bool:
    """Include temp-apron waiters with empty ``edge_ids`` so they keep a one-edge reservation."""
    if ag.edge_ids:
        return True
    return bool(
        ag.awaiting_apron_from_temp and str(ag.temp_stand_id or "").strip()
    )


def _layout_edge_path_type(control_state: SimulationControlState, eid: str) -> str:
    er = control_state.edge_resources.get(str(eid))
    if er is None:
        return "taxiway"
    return str(er.path_type or "taxiway")


def _lookahead_depth_billed_count(
    agent: Flight,
    up_to_idx_inclusive: int,
    control_state: SimulationControlState,
) -> int:
    """Billed slots: ``apron_taxiway`` free; ``taxiway`` merges; ``apron_link`` + stand resource slot."""
    c = 0
    hi = int(up_to_idx_inclusive)
    has_apron_link = False
    for j in range(0, hi + 1):
        if not agent.edge_ids or j >= len(agent.edge_ids):
            break
        pt = str(
            _layout_edge_path_type(control_state, str(agent.edge_ids[j])) or ""
        ).strip()
        if pt == "apron_taxiway":
            continue
        if pt == "apron_link":
            has_apron_link = True
        if pt == "taxiway":
            if j == 0:
                c += 1
            else:
                ptp = str(
                    _layout_edge_path_type(
                        control_state, str(agent.edge_ids[j - 1])
                    )
                    or ""
                ).strip()
                if ptp == "taxiway":
                    pass
                else:
                    c += 1
        else:
            c += 1
    if has_apron_link:
        c += 1
    return c


def _edge_uses_full_depth_reservation(
    agent: Flight,
    idx: int,
    control_state: SimulationControlState,
) -> bool:
    """Every physical edge participates in reservation; depth limits use billed count only."""
    return True


def _stand_arrival_book_if_pipeline_proceed(
    ag: Flight,
    control_state: SimulationControlState,
    stand_arrival_book: Dict[str, int],
) -> None:
    """Same-tick pipeline: only when reserved lookahead includes ``apron_link`` (depth sees stand)."""
    sid = str(ag.apron_stand_id or "").strip()
    if not sid:
        return
    if not ag.edge_phases or str(ag.edge_phases[0]) != PHASE_ARR_TAXI:
        return
    st = control_state.agent_states.get(ag.id)
    if st is None or not st.reserved_edges:
        return
    has_apron = False
    for eid in st.reserved_edges:
        er0 = control_state.edge_resources.get(str(eid))
        if er0 is not None and str(er0.path_type or "").strip() == "apron_link":
            has_apron = True
            break
    if not has_apron:
        return
    sr = control_state.stand_resources.get(sid)
    if sr is None or ag.id in sr.occupied_by:
        return
    stand_arrival_book[sid] = int(stand_arrival_book.get(sid, 0)) + 1


def _agents_on_edge(eid: str, agents: List[Flight]) -> List[Flight]:
    return [ag for ag in agents if ag.edge_ids and str(ag.edge_ids[0]) == eid]


_RUNWAY_APPROACH_PT = frozenset({"runway_taxiway", "runway_exit"})


def _dep_runway_entry_remaining_m(agent: Flight, ppm: float) -> Optional[float]:
    """Remaining distance (m) from the agent's current position to the first upcoming
    ``runway`` micro-segment on the queued path. Returns ``None`` if no ``runway``
    segment is upcoming among the remaining queued segments.

    Uses per-segment endpoints and progress along the current segment so the value is
    independent of how finely the approach polyline is subdivided (vertex / graph-node
    density on the ``runway_taxiway`` / ``runway_exit``).
    """
    if not agent.segment_endpoints or not agent.segment_path_types:
        return None
    n = len(agent.segment_endpoints)
    if n == 0 or len(agent.segment_path_types) != n:
        return None
    ppm_s = max(float(ppm), 1e-9)
    total_px = 0.0
    for i in range(n):
        pt = str(agent.segment_path_types[i] or "").strip()
        if pt == "runway":
            return float(total_px) / ppm_s
        p0, p1 = agent.segment_endpoints[i]
        seg_len = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        if i == 0:
            total_px += max(0.0, seg_len - float(agent.edge_s_along_px))
        else:
            total_px += seg_len
    return None


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
    t_abs: float,
    rwid: str,
    agent_id: str,
    agents: List[Flight],
    runway_release_lag_sec: float = 0.0,
    *,
    control_state: Optional[SimulationControlState] = None,
) -> bool:
    """동일 도착 활주로: 다른 기체가 아직 이탈 전이면 점유로 본다.

    구간은 ``[ELDT, release)`` 이고, ``release``는 ``exit_runway_abs_sec``가
    있으면 그 절대 시각이다. 이탈 시각이 아직 없으면 **착륙/활주로(또는 활주로
    연결 활주로) 상의 도착 레그**에 있을 때만 점유로 본다. 경로가 끝났거나 이미
    일반 택시로 이탈한 기체는 ``arr_runway_id``만 같아도 무한 점유로 막지 않는다.
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
        td_o = _arr_touchdown_motion_abs_sec(
            o,
            agents,
            float(runway_release_lag_sec),
            control_state=control_state,
        )
        if td_o is not None and tt + 1e-9 < float(td_o):
            continue
        t0 = float(td_o) if td_o is not None else float(o.eldt_anchor_sec)
        if tt + 1e-9 <= t0:
            continue
        ex = o.exit_runway_abs_sec
        if ex is not None:
            if tt + 1e-9 < float(ex):
                return True
            continue
        if not o.edge_ids or not o.edge_phases:
            continue
        ph0 = str(o.edge_phases[0])
        if ph0 == PHASE_LANDING:
            return True
        if ph0 in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
            pt0 = (
                str(o.segment_path_types[0] or "")
                if o.segment_path_types
                and len(o.segment_path_types) == len(o.edge_ids)
                else ""
            )
            if pt0 in ("runway", "runway_taxiway"):
                return True
    return False


def _resource_use_count(
    occupied: List[str], reserved: List[str], agent_id: str
) -> int:
    return len({x for x in (occupied + reserved) if x != agent_id})


def _blocking_temp_stand_for_edge(
    control_state: SimulationControlState,
    edge_id: str,
    agent_id: str,
) -> Optional[str]:
    eid = str(edge_id).strip()
    aid = str(agent_id)
    if not eid:
        return None
    for ts_id, e_set in control_state.temp_stand_incident_edges.items():
        if eid not in e_set:
            continue
        sr = control_state.stand_resources.get(str(ts_id))
        if sr is None:
            continue
        if any(str(x) != aid for x in sr.occupied_by):
            return str(ts_id)
    return None


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
        o_td = _arr_touchdown_motion_abs_sec(
            o, agents, rw_lag, control_state=control_state
        )
        if o_td is not None and t_eff + 1e-9 < float(o_td):
            continue
        st_o = control_state.agent_states.get(o.id)
        if _agent_deadlock_ghost_at_time(st_o, t_eff):
            continue
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
    stand_arrival_book: Optional[Dict[str, int]] = None,
    reservation_depth: int = RESERV_DEPTH_ARR_TAXI,
) -> Tuple[bool, str]:
    if not lookahead:
        return False, "empty_lookahead"
    aid = agent.id
    t_abs = float(sim_time)
    depth_cap = max(1, int(reservation_depth))

    for idx, eid in enumerate(lookahead):
        er = control_state.edge_resources.get(eid)
        if er is None:
            return False, f"unknown_edge:{eid}"
        for ts_id, e_set in control_state.temp_stand_incident_edges.items():
            if str(eid) not in e_set:
                continue
            sr_t = control_state.stand_resources.get(str(ts_id))
            if sr_t is None:
                continue
            if any(str(x) != str(aid) for x in sr_t.occupied_by):
                return False, f"temp_stand_busy:{ts_id}"
        # Never bypass runway safety via forced-open.
        if er.forced_open and not er.runway_id:
            continue
        ph0 = str(agent.edge_phases[0]) if agent.edge_phases else ""
        pt0 = (
            str(agent.segment_path_types[0] or "")
            if agent.segment_path_types and len(agent.segment_path_types) == len(agent.edge_ids)
            else ""
        )
        if (
            str(er.path_type or "").strip() == "apron_link"
            and ph0 == PHASE_ARR_TAXI
            and agent.actual_apron_inblocks_abs_sec is None
        ):
            sid = str(agent.apron_stand_id or "").strip()
            if sid:
                sr = control_state.stand_resources.get(sid)
                if sr is not None:
                    cap = max(1, int(sr.capacity))
                    phys_others = len({x for x in sr.occupied_by if x != aid})
                    booked = (
                        int(stand_arrival_book.get(sid, 0))
                        if stand_arrival_book is not None
                        else 0
                    )
                    if phys_others + booked >= cap:
                        return False, "stand_occupied"
                    if _stand_pushback_clearance_cooldown_active(sid, aid, agents, t_abs):
                        return False, "stand_occupied"
        dep_rwy = str(agent.dep_runway_id or "").strip()
        if (
            idx == 0
            and dep_rwy
            and ph0 in (PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE)
            and pt0 in ("runway", "runway_taxiway")
        ):
            rr_dep = control_state.runway_resources.get(dep_rwy)
            if rr_dep is not None and not rr_dep.forced_open:
                if _runway_rot_reservation_blocked(
                    t_abs,
                    dep_rwy,
                    aid,
                    agents,
                    float(runway_release_lag_sec),
                    control_state=control_state,
                ):
                    return False, f"runway_rot_busy:{dep_rwy}"
                ou_d = _resource_use_count(rr_dep.occupied_by, rr_dep.reserved_by, aid)
                if ou_d >= max(1, int(rr_dep.capacity)):
                    return False, f"runway_dep_busy:{dep_rwy}"
        # Distance-based runway hold gate on the approach path (``runway_taxiway`` or
        # ``runway_exit``): when the departure runway is busy and the upcoming runway
        # entry is within ``DEP_RUNWAY_HOLD_BUFFER_M``, hold here. Keeps the stop
        # distance predictable regardless of polyline vertex / graph node density
        # along the approach lane. Covers Dep_taxi too (existing phase-boundary check
        # only fires on Holding_lineup / Lineup_departure, whose first micro-segment
        # can snap very close to the runway when the approach lane has many vertices).
        if (
            idx == 0
            and dep_rwy
            and ph0 in (PHASE_DEP_TAXI, PHASE_HOLDING_LINEUP, PHASE_LINEUP_DEPARTURE)
            and pt0 in _RUNWAY_APPROACH_PT
        ):
            rem_m = _dep_runway_entry_remaining_m(agent, ppm)
            if rem_m is not None and rem_m <= float(DEP_RUNWAY_HOLD_BUFFER_M):
                rr_dep_b = control_state.runway_resources.get(dep_rwy)
                if rr_dep_b is not None and not rr_dep_b.forced_open:
                    if _runway_rot_reservation_blocked(
                        t_abs,
                        dep_rwy,
                        aid,
                        agents,
                        float(runway_release_lag_sec),
                        control_state=control_state,
                    ):
                        return False, f"runway_rot_busy:{dep_rwy}"
                    ou_d_b = _resource_use_count(
                        rr_dep_b.occupied_by, rr_dep_b.reserved_by, aid
                    )
                    if ou_d_b >= max(1, int(rr_dep_b.capacity)):
                        return False, f"runway_dep_busy:{dep_rwy}"
        if er.runway_id:
            rwid = str(er.runway_id)
            rr = control_state.runway_resources.get(rwid)
            if rr is not None:
                # Hard invariant: runway occupancy by another aircraft blocks reservation.
                if any(str(x) != str(aid) for x in rr.occupied_by):
                    return False, f"runway_occupied:{rwid}"
                if rr.forced_open:
                    continue
                if _runway_rot_reservation_blocked(
                    t_abs,
                    rwid,
                    aid,
                    agents,
                    float(runway_release_lag_sec),
                    control_state=control_state,
                ):
                    return False, f"runway_rot_busy:{rwid}"
                ou_rw = _resource_use_count(rr.occupied_by, rr.reserved_by, aid)
                if ou_rw >= max(1, int(rr.capacity)):
                    return False, f"runway_capacity:{rwid}"
        else:
            billed = _lookahead_depth_billed_count(agent, idx, control_state)
            interior = not _edge_uses_full_depth_reservation(agent, idx, control_state)
            if interior or billed > depth_cap:
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
                    depth_ir = depth_cap
                    billed_ir = _lookahead_depth_billed_count(agent, idx, control_state)
                    interior_ir = not _edge_uses_full_depth_reservation(
                        agent, idx, control_state
                    )
                    if interior_ir or billed_ir > depth_ir:
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
        if _agent_deadlock_ghost_at_time(st, t):
            continue
        td_ag = _arr_touchdown_motion_abs_sec(
            ag,
            agents,
            float(runway_release_lag_sec),
            control_state=control_state,
        )
        if td_ag is not None and t + 1e-9 < float(td_ag):
            st.stagnation_anchor_sec = None
            st.progress_snapshot_edge_id = None
            continue
        if st.wait_reason == "pre_eldt":
            st.stagnation_anchor_sec = None
            st.progress_snapshot_edge_id = None
            continue
        if st.wait_reason == "stand_occupied":
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


def reserve_path(
    agent: Flight,
    lookahead: List[str],
    control_state: SimulationControlState,
    sim_time: float,
    reservation_depth: int = RESERV_DEPTH_ARR_TAXI,
) -> None:
    st = control_state.agent_states.get(agent.id)
    if st is None:
        return
    st.reserved_edges = list(lookahead)
    st.reserved_intersections.clear()
    aid = agent.id
    depth_cap = max(1, int(reservation_depth))
    for idx, eid in enumerate(lookahead):
        er = control_state.edge_resources.get(eid)
        if er is None:
            continue
        if er.runway_id:
            if idx >= depth_cap:
                continue
            if aid not in er.reserved_by:
                er.reserved_by.append(aid)
            rr = control_state.runway_resources.get(str(er.runway_id))
            if rr is not None and aid not in rr.reserved_by:
                rr.reserved_by.append(aid)
            continue
        if not _edge_uses_full_depth_reservation(agent, idx, control_state):
            continue
        billed = _lookahead_depth_billed_count(agent, idx, control_state)
        if billed > depth_cap:
            continue
        if aid not in er.reserved_by:
            er.reserved_by.append(aid)
    for k in range(len(lookahead) - 1):
        er_k = control_state.edge_resources.get(lookahead[k])
        if er_k is None or not er_k.intersection_out:
            continue
        if er_k.runway_id:
            if k >= depth_cap:
                continue
        else:
            if not _edge_uses_full_depth_reservation(agent, k, control_state):
                continue
            bk = _lookahead_depth_billed_count(agent, k, control_state)
            if bk > depth_cap:
                continue
        iid = er_k.intersection_out
        ir = control_state.intersection_resources.get(iid)
        if ir is not None and aid not in ir.reserved_by:
            ir.reserved_by.append(aid)
        st.reserved_intersections.append(iid)


def _agent_reserved_lookahead_contains_edge(
    st: Optional[AgentControlState], edge_id: str
) -> bool:
    """True if this tick's reserved lookahead includes the edge (after ``reserve_path``)."""
    if st is None or not st.reserved_edges:
        return False
    e = str(edge_id)
    return any(str(x) == e for x in st.reserved_edges)


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
    eid0 = str(agent_a.edge_ids[0])
    sta = control_state.agent_states.get(agent_a.id)
    stb = control_state.agent_states.get(agent_b.id)
    if sta and _agent_deadlock_ghost_at_time(sta, sim_time):
        return
    if stb and _agent_deadlock_ghost_at_time(stb, sim_time):
        return
    if sta is None or stb is None:
        return
    if sta.clearance != "PROCEED" or stb.clearance != "PROCEED":
        return
    if not (
        _agent_reserved_lookahead_contains_edge(sta, eid0)
        and _agent_reserved_lookahead_contains_edge(stb, eid0)
    ):
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
    if phase in (PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP):
        return 1
    if phase == PHASE_DEP_TAXI:
        return 2
    if phase == PHASE_HOLDING_LINEUP:
        return 3
    if phase == PHASE_LINEUP_DEPARTURE:
        return 4
    return 2


def _reroute_leg_destination_xy(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    phase: str,
    information: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    info = information if isinstance(information, dict) else _load_information_json()
    paths = extract_point_to_paths(flight, layout, cell_size, information=info)
    if not paths or len(paths) < 5:
        return None
    if phase == PHASE_LANDING and paths[0] and len(paths[0]) >= 4:
        return (float(paths[0][2]), float(paths[0][3]))
    if phase == PHASE_ARR_TAXI and len(paths) > 1 and len(paths[1]) >= 4:
        return (float(paths[1][2]), float(paths[1][3]))
    if phase == PHASE_DEP_TAXI and len(paths) > 2 and len(paths[2]) >= 4:
        return (float(paths[2][2]), float(paths[2][3]))
    if phase == PHASE_HOLDING_LINEUP and len(paths) > 3 and len(paths[3]) >= 4:
        return (float(paths[3][2]), float(paths[3][3]))
    if phase == PHASE_LINEUP_DEPARTURE and len(paths) > 4 and len(paths[4]) >= 4:
        return (float(paths[4][2]), float(paths[4][3]))
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


def _graph_path_has_disallowed_reverse_of_prior_hops(
    g: PathGraph,
    path: List[int],
    pair_index: Dict[Tuple[int, int], str],
    completed: List[Tuple[str, int, int, str]],
) -> bool:
    """True if path uses a layout edge opposite to a prior traversal (aircraft may not reverse)."""
    if not completed:
        return False
    for i in range(len(path) - 1):
        u, v = int(path[i]), int(path[i + 1])
        lo, hi = (u, v) if u <= v else (v, u)
        eid = pair_index.get((lo, hi))
        if not eid:
            continue
        rec = g.edge_map.get(f"{u}:{v}")
        cand_pt = str(rec.path_type or "") if rec is not None else ""
        for ceid, u0, v0, _cpt in completed:
            if ceid != eid:
                continue
            if u == v0 and v == u0 and cand_pt != "apron_link":
                return True
    return False


def _slice_expanded_reroute_at_current_edge(
    agent: Flight,
    ex_ids: List[str],
    segs: List[Tuple[Point, Point]],
    phs: List[str],
    lnks: List[str],
    ptyps: List[str],
    guvs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[
    List[str],
    List[Tuple[Point, Point]],
    List[str],
    List[str],
    List[str],
    Optional[List[Tuple[int, int]]],
]:
    if not ex_ids or not segs or not agent.edge_ids:
        return ex_ids, segs, phs, lnks, ptyps, guvs
    if len(segs) != len(ex_ids) or len(phs) != len(ex_ids):
        return ex_ids, segs, phs, lnks, ptyps, guvs
    cur = str(agent.edge_ids[0])
    idx = next((i for i, e in enumerate(ex_ids) if str(e) == cur), None)
    if idx is None:
        return ex_ids, segs, phs, lnks, ptyps, guvs
    orig_n = len(ex_ids)
    ex_ids = ex_ids[idx:]
    segs = list(segs[idx:])
    phs = phs[idx:]
    lnks = lnks[idx:]
    ptyps = ptyps[idx:]
    guv_out: Optional[List[Tuple[int, int]]] = None
    if guvs is not None and len(guvs) == orig_n:
        guv_out = list(guvs[idx:])
    if segs:
        _p0, p1 = segs[0]
        _t, proj = project_on_segment(
            (float(_p0[0]), float(_p0[1])),
            (float(p1[0]), float(p1[1])),
            (float(agent.col), float(agent.row)),
        )
        segs[0] = (proj, (float(p1[0]), float(p1[1])))
    return ex_ids, segs, phs, lnks, ptyps, guv_out


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
        g0 = _cached_path_graph_for_direction(
            layout,
            cell_size,
            _flight_rw_dir_for_leg(flight, _leg_index_for_phase(phase), layout),
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
    rw_dir = _flight_rw_dir_for_leg(flight, leg_i, layout)
    penalized: set[str] = set(_yield_penalized_layout_edges_for_reroute(control_state, agent.id))
    penalized |= _yield_temp_occupied_incident_edges_for_pathfinding(
        control_state, str(agent.id)
    )
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
    if _graph_path_has_disallowed_reverse_of_prior_hops(
        g, path, pair_index, agent.completed_directed_hops
    ):
        _LOG.info("reroute rejected (no reverse) flight=%s", agent.id)
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
    ex_ids, segs, phs, lnks, ptyps, guvs = _expand_geometry_from_graph_path(
        g, path, pair_index, phase
    )
    if (
        not ex_ids
        or not segs
        or len(ex_ids) != len(segs)
        or len(phs) != len(ex_ids)
        or len(lnks) != len(ex_ids)
        or len(ptyps) != len(ex_ids)
        or len(guvs) != len(ex_ids)
    ):
        return None
    ex_ids, segs, phs, lnks, ptyps, guvs = _slice_expanded_reroute_at_current_edge(
        agent, ex_ids, segs, phs, lnks, ptyps, guvs
    )
    if not ex_ids:
        return None
    if guvs is None or len(guvs) != len(ex_ids):
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
            information,
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
        segment_graph_uv=list(guvs),
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
    agent.segment_graph_uv = list(prep.segment_graph_uv) if prep.segment_graph_uv else []
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


def _append_future_leg_suffix_to_reroute_prep(
    agent: Flight,
    prep: PreparedFlightPath,
) -> PreparedFlightPath:
    if not agent.edge_phases or not prep.edge_ids:
        return prep
    current_leg_i = _leg_index_for_phase(str(agent.edge_phases[0]))
    suffix_start = next(
        (
            i
            for i, ph in enumerate(agent.edge_phases)
            if _leg_index_for_phase(str(ph)) > current_leg_i
        ),
        None,
    )
    if suffix_start is None:
        return prep
    suffix_edge_ids = list(agent.edge_ids[suffix_start:])
    suffix_phases = list(agent.edge_phases[suffix_start:])
    suffix_segs = list(agent.segment_endpoints[suffix_start:])
    suffix_v0s = list(agent.segment_v0_ms[suffix_start:])
    suffix_accs = list(agent.segment_accel_ms2[suffix_start:])
    suffix_ptypes = list(agent.segment_path_types[suffix_start:])
    suffix_guvs = (
        list(agent.segment_graph_uv[suffix_start:])
        if agent.segment_graph_uv
        and len(agent.segment_graph_uv) >= suffix_start
        and len(agent.segment_graph_uv) == len(agent.edge_ids)
        else []
    )
    if (
        not suffix_edge_ids
        or len(suffix_segs) != len(suffix_edge_ids)
        or len(suffix_phases) != len(suffix_edge_ids)
        or len(suffix_v0s) != len(suffix_edge_ids)
        or len(suffix_accs) != len(suffix_edge_ids)
    ):
        return prep
    if suffix_ptypes and len(suffix_ptypes) != len(suffix_edge_ids):
        suffix_ptypes = []
    if suffix_guvs and len(suffix_guvs) != len(suffix_edge_ids):
        suffix_guvs = []
    merged_edge_ids = list(prep.edge_ids) + suffix_edge_ids
    merged_phases = list(prep.segment_phases) + suffix_phases
    merged_logical = list(prep.logical_edge_list) + [
        {"edge_id": str(e), "phase": str(ph)}
        for e, ph in zip(suffix_edge_ids, suffix_phases)
    ]
    merged_segs = list(prep.segment_endpoints) + suffix_segs
    merged_ptypes = (
        list(prep.segment_path_types) + suffix_ptypes
        if prep.segment_path_types or suffix_ptypes
        else []
    )
    _pg_uv = (
        list(prep.segment_graph_uv)
        if prep.segment_graph_uv and len(prep.segment_graph_uv) == len(prep.edge_ids)
        else []
    )
    _sg_uv = suffix_guvs if suffix_guvs and len(suffix_guvs) == len(suffix_edge_ids) else []
    merged_guvs = (
        _pg_uv + _sg_uv if len(_pg_uv) + len(_sg_uv) == len(merged_edge_ids) else []
    )
    merged_v0s = list(prep.segment_start_velocity_ms) + suffix_v0s
    merged_accs = list(prep.segment_accel_ms2) + suffix_accs
    merged_durs = list(prep.segment_duration_sec) + [0.0] * len(suffix_edge_ids)
    return PreparedFlightPath(
        edge_ids=merged_edge_ids,
        segment_phases=merged_phases,
        logical_edge_list=merged_logical,
        segment_endpoints=merged_segs,
        leg_lengths_px=list(prep.leg_lengths_px),
        leg_micro_counts=list(prep.leg_micro_counts),
        segment_link_ids=list(prep.segment_link_ids),
        segment_path_types=merged_ptypes,
        segment_graph_uv=merged_guvs,
        segment_start_velocity_ms=merged_v0s,
        segment_accel_ms2=merged_accs,
        segment_duration_sec=merged_durs,
        spawn_skip_landing_px=float(prep.spawn_skip_landing_px),
        spawn_along_first_segment_px=float(prep.spawn_along_first_segment_px),
        playback_first_segment_index=int(prep.playback_first_segment_index),
        ok=prep.ok,
        direction_violation=prep.direction_violation,
    )


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
    agents: List[Flight],
    *,
    aggressive: bool,
) -> bool:
    if not agent.edge_phases or not agent.edge_ids:
        return False
    # Once the aircraft is on the lineup path (RTX → lineup → runway takeoff),
    # the remaining route is fixed and it is only waiting for the runway to clear.
    # Rerouting from a mid-polyline RTX position can pick a `nearest_path_node`
    # that lies on the runway itself (not on the RTX graph edge), which then
    # teleports the agent onto the runway via segment snap. Skip reroute in
    # these phases so the aircraft stays on its RTX polyline to the lineup point.
    if str(agent.edge_phases[0]) in (
        PHASE_LANDING,
        PHASE_HOLDING_LINEUP,
        PHASE_LINEUP_DEPARTURE,
    ):
        return False
    st = control_state.agent_states.get(agent.id)
    if st is None or _agent_deadlock_ghost_at_time(st, float(sim_time)):
        return False
    if not aggressive and int(st.reroute_attempts) >= int(REROUTE_MAX_ATTEMPTS):
        return False
    phase = str(agent.edge_phases[0])
    dest = _reroute_leg_destination_xy(flight, layout, cell_size, phase, information=information)
    if dest is None:
        return False
    la_n, _ = _lookahead_and_reservation_depth_for_agent(
        agent, control_state, float(sim_time), agents
    )
    la = get_lookahead_edges(agent, la_n, control_state)
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
    prep = _append_future_leg_suffix_to_reroute_prep(agent, prep)
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


def _reroute_all_moving_flights_after_temp_park_arrival(
    agents: List[Flight],
    flights_by_id: Dict[str, Dict[str, Any]],
    control_state: SimulationControlState,
    layout: Dict[str, Any],
    information: Dict[str, Any],
    sim_time_abs: float,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
) -> int:
    """
    After a flight finishes temp taxi and parks (``awaiting_apron_from_temp``), other aircraft
    may still hold pre-temp paths that fail reservation; re-run Dijkstra once for every
    moving flight (aggressive reroute, same as deadlock path).
    """
    n_ok = 0
    t_abs = float(sim_time_abs)
    for ag in sorted(agents, key=lambda a: str(a.id)):
        if not ag.edge_ids or not ag.edge_phases:
            continue
        if str(ag.edge_phases[0]) == PHASE_LANDING:
            continue
        st = control_state.agent_states.get(ag.id)
        if st is None or _agent_deadlock_ghost_at_time(st, t_abs):
            continue
        fo = flights_by_id.get(str(ag.id))
        if not isinstance(fo, dict):
            continue
        la_n, _depth_n = _lookahead_and_reservation_depth_for_agent(
            ag, control_state, t_abs, agents
        )
        la = get_lookahead_edges(ag, la_n, control_state)
        temp_pen = _yield_temp_occupied_incident_edges_for_pathfinding(
            control_state, str(ag.id)
        )
        if not temp_pen or not any(str(eid) in temp_pen for eid in la):
            continue
        if _try_reroute_agent_off_path_block(
            ag,
            fo,
            control_state,
            layout,
            information,
            t_abs,
            float(cell_size),
            reverse_cost,
            merge_r,
            taxiway_h,
            agents,
            aggressive=True,
        ):
            n_ok += 1
    if n_ok:
        _LOG.info(
            "TEMP_PARK_GLOBAL_REROUTE t=%.1f rerouted_flights=%s",
            t_abs,
            n_ok,
        )
    return n_ok


def _run_temp_park_arrival_global_reroute_bundle(
    control_state: SimulationControlState,
    agents: List[Flight],
    flights_by_id: Dict[str, Dict[str, Any]],
    layout: Dict[str, Any],
    information: Dict[str, Any],
    sim_time_abs: float,
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    pixels_per_meter: float,
    runway_release_lag_sec: float,
) -> None:
    """Refresh occupancy → reroute all movers → rebook reservations (called immediately when temp park completes)."""
    rw_lag = float(runway_release_lag_sec)
    t_abs = float(sim_time_abs)
    ppm = max(float(pixels_per_meter), 1e-9)
    refresh_resource_occupancy(
        control_state, agents, ppm, t_abs, rw_lag
    )
    _reroute_all_moving_flights_after_temp_park_arrival(
        agents,
        flights_by_id,
        control_state,
        layout,
        information,
        t_abs,
        float(cell_size),
        reverse_cost,
        merge_r,
        taxiway_h,
    )
    for ag in agents:
        ag.temp_park_arrival_trigger_global_reroute = False
    refresh_resource_occupancy(
        control_state, agents, ppm, t_abs, rw_lag
    )
    _book_tp, _ = _single_full_reservation_pass(
        control_state,
        agents,
        t_abs,
        ppm,
        rw_lag,
    )
    control_state.stand_arrival_book_snapshot = dict(_book_tp)
    refresh_agent_edge_fsm(agents)


def should_reroute_agent(
    agent: Flight,
    control_state: SimulationControlState,
    sim_time: float,
    information: Dict[str, Any],
) -> bool:
    st = control_state.agent_states.get(agent.id)
    if st is None or not agent.edge_ids or not agent.edge_phases:
        return False
    if _agent_deadlock_ghost_at_time(st, float(sim_time)):
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
    agents: List[Flight],
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
        agents,
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
            agents,
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
        if _agent_deadlock_ghost_at_time(st, t):
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
    control_state.deadlock_resolve_event_count = int(control_state.deadlock_resolve_event_count) + 1
    ghost_until = float(sim_time) + float(DEADLOCK_FORCE_MOVE_DURATION_SEC)
    wait_snap: Dict[str, float] = {}
    stall_snap: Dict[str, Optional[float]] = {}
    for fid in id_set:
        st = control_state.agent_states.get(fid)
        if st is None:
            continue
        st.deadlock_flag = True
        st.clearance = CLEARANCE_DEADLOCK_GHOST
        st.wait_reason = "deadlock_ghost"
        st.deadlock_ghost_until_abs_sec = ghost_until
        wait_snap[fid] = float(st.total_wait_sec)
        stall_snap[fid] = st.stagnation_anchor_sec
        st.stagnation_anchor_sec = None
        st.progress_snapshot_edge_id = None
        st.wait_start_sec = None
    _LOG.warning(
        "DEADLOCK_GHOST t=%.1f flights=%s total_wait_sec=%s stagnation_anchor_sec=%s "
        "ghost_until=%.1f duration_sec=%.1f",
        float(sim_time),
        sorted(id_set),
        wait_snap,
        stall_snap,
        ghost_until,
        float(DEADLOCK_FORCE_MOVE_DURATION_SEC),
    )


def _single_full_reservation_pass(
    control_state: SimulationControlState,
    agents: List[Flight],
    sim_time: float,
    pixels_per_meter: float,
    runway_release_lag_sec: float,
) -> Tuple[Dict[str, int], List[Flight]]:
    """
    One global rebook: clear reservations, reset non-ghost clearances, assign PROCEED/WAIT from
    ``can_reserve_path`` (stands + departure runway + edges), then head-on resolution.
    Used on heavy ticks, after reroute/deadlock rebooks, and on light ticks every
    ``LIGHT_RESERVATION_RETRY_INTERVAL_SEC`` so blocked agents retry when destination resources free.
    """
    ppm = max(float(pixels_per_meter), 1e-9)
    rw_lag = float(runway_release_lag_sec)
    _expire_forced_open_resources(control_state, float(sim_time))
    _clear_all_reservations(control_state)
    t_dec = float(sim_time)
    for st in control_state.agent_states.values():
        gu = st.deadlock_ghost_until_abs_sec
        if gu is not None and t_dec + 1e-9 < float(gu):
            st.clearance = CLEARANCE_DEADLOCK_GHOST
            st.wait_reason = "deadlock_ghost"
            continue
        if gu is not None and t_dec >= float(gu):
            st.deadlock_ghost_until_abs_sec = None
            st.deadlock_flag = False
        st.clearance = "PROCEED"
        st.wait_reason = None

    def _decision_sort_key(ag: Flight) -> Tuple[int, int, int, float, float, str]:
        st0 = control_state.agent_states.get(ag.id)
        tw = float(st0.total_wait_sec) if st0 else 0.0
        pr = get_agent_priority_rank(ag)
        eldt_i = (
            int(round(float(ag.eldt_anchor_sec)))
            if ag.eldt_anchor_sec is not None
            else 0
        )
        eldt_tie = _stable_tie_seed(eldt_i, str(ag.id), int(sim_time) // 10)
        return (
            pr,
            eldt_i,
            eldt_tie,
            -tw,
            -_edge_progress_ratio(ag),
            str(ag.id),
        )

    ordered = sorted(
        [ag for ag in agents if _agent_eligible_for_reservation_pass(ag)],
        key=_decision_sort_key,
    )
    stand_arrival_book: Dict[str, int] = {}
    for ag in ordered:
        st = control_state.agent_states.get(ag.id)
        if st is None:
            continue
        st.priority_rank = get_agent_priority_rank(ag)
        if _agent_deadlock_ghost_at_time(st, t_dec):
            continue
        motion_td = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if motion_td is not None and t_dec + 1e-9 < float(motion_td):
            st.clearance = "WAIT"
            st.wait_reason = "pre_eldt"
            continue
        la_n, depth_n = _lookahead_and_reservation_depth_for_agent(
            ag, control_state, t_dec, agents
        )
        if _temp_apron_hold_reservation_only_current_edge(ag):
            la = _temp_apron_current_edge_lookahead(ag)
            depth_n = 1
        else:
            la = get_lookahead_edges(ag, la_n, control_state)
        ok, reason = can_reserve_path(
            ag,
            la,
            control_state,
            agents,
            ppm,
            sim_time,
            rw_lag,
            stand_arrival_book,
            reservation_depth=depth_n,
        )
        if ok:
            reserve_path(ag, la, control_state, sim_time, reservation_depth=depth_n)
            st.clearance = "PROCEED"
            st.wait_reason = None
            st.wait_start_sec = None
            _stand_arrival_book_if_pipeline_proceed(
                ag, control_state, stand_arrival_book
            )
        else:
            st.clearance = "WAIT"
            st.wait_reason = reason or "reservation"
            if st.wait_start_sec is None:
                st.wait_start_sec = float(sim_time)
    _resolve_all_head_on(control_state, agents, sim_time)
    return stand_arrival_book, ordered


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

    stand_arrival_book, ordered = _single_full_reservation_pass(
        control_state, agents, sim_time, ppm, rw_lag
    )

    def _rebook_ordered_reservations() -> None:
        nonlocal stand_arrival_book, ordered
        stand_arrival_book, ordered = _single_full_reservation_pass(
            control_state, agents, sim_time, ppm, rw_lag
        )

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
                agents,
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
    control_state.stand_arrival_book_snapshot = dict(stand_arrival_book)
    control_state.last_light_reservation_rebook_sim_time = float(sim_time)
    _LOG.info("airside control tick t=%.3f processed=%s deadlocked=%s", float(sim_time), len(ordered), len(cand))


def _apply_same_direction_following_caps(
    control_state: SimulationControlState,
    agents: List[Flight],
    ppm: float,
    sim_time_abs: float,
) -> None:
    by_edge: Dict[str, List[Flight]] = {}
    t_eff = float(sim_time_abs)
    for ag in agents:
        if not ag.edge_ids or ag.control_halt:
            continue
        st = control_state.agent_states.get(ag.id)
        if st and _agent_deadlock_ghost_at_time(st, t_eff):
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
    *,
    on_temp_park_arrival_immediate_reroute: Optional[Callable[[], None]] = None,
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
        td0 = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_lag, control_state=control_state
        )
        if td0 is not None and t_end + 1e-12 < float(td0):
            ag.control_halt = True
        elif _agent_deadlock_ghost_at_time(st, t_end):
            ag.control_halt = True
        elif ag.edge_ids:
            ts_block = _blocking_temp_stand_for_edge(
                control_state, str(ag.edge_ids[0]), str(ag.id)
            )
            if ts_block:
                ag.control_halt = True
                st.clearance = "WAIT"
                st.wait_reason = f"temp_stand_busy:{ts_block}"
        if st.clearance in ("WAIT", "YIELD"):
            ag.control_halt = True
        if ag.edge_ids:
            er0 = control_state.edge_resources.get(str(ag.edge_ids[0]))
            if er0 is not None and er0.runway_id:
                rr0 = control_state.runway_resources.get(str(er0.runway_id))
                if rr0 is not None and any(str(x) != str(ag.id) for x in rr0.occupied_by):
                    ag.control_halt = True
                    st.clearance = "WAIT"
                    st.wait_reason = f"runway_occupied:{er0.runway_id}"
    _apply_same_direction_following_caps(control_state, agents, ppm, t_end)
    for ag in agents:
        # Do not use touchdown cache: prior agents may have moved / exited runway this tick.
        td = _compute_arr_touchdown_motion_abs_sec(ag, agents, rw_lag)
        if td is not None and t_end + 1e-12 < float(td):
            continue
        eldt_eff = float(td) if td is not None else t_end
        if ag.control_halt:
            ag.velocity_ms = 0.0
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
        if (
            on_temp_park_arrival_immediate_reroute is not None
            and ag.temp_park_arrival_trigger_global_reroute
        ):
            on_temp_park_arrival_immediate_reroute()


def run_simulation(
    layout: Dict[str, Any],
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    _PATH_GRAPH_BUILD_CACHE.clear()
    information = _load_information_json()
    deadlock_resolve_stop_n = max(
        1,
        int(
            round(
                float(
                    _deep_get(
                        information,
                        "tiers",
                        "algorithm",
                        "simulation",
                        "deadlockResolveStopCount",
                        default=float(DEADLOCK_RESOLVE_STOP_COUNT),
                    )
                )
            )
        ),
    )
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
            _guv_src = prep.segment_graph_uv
            _cdh: List[Tuple[str, int, int, str]] = []
            _guv_rem: List[Tuple[int, int]] = []
            if (
                _guv_src
                and len(_guv_src) == _prep_n
                and _pt_src
                and len(_pt_src) == _prep_n
            ):
                _gs0 = int(g_start)
                for _j in range(_gs0):
                    _cdh.append(
                        (
                            str(prep.edge_ids[_j]),
                            int(_guv_src[_j][0]),
                            int(_guv_src[_j][1]),
                            str(_pt_src[_j] or ""),
                        )
                    )
                _guv_rem = list(_guv_src[_gs0:])
            if len(_guv_rem) != len(eids):
                _guv_rem = []
                _cdh = []
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
            dwell_s = _dwell_sec_from_flight(fobj if isinstance(fobj, dict) else {})
            _arr_rid = str(arr_rwy_o).strip() if arr_rwy_o else ""
            dep_rwy_o = fobj.get("depRunwayId") or token_o.get("depRunwayId")
            _dep_rid = str(dep_rwy_o).strip() if dep_rwy_o else ""
            _mf0 = True
            if path_rem and eph:
                if (
                    str(path_rem[0] or "") == "apron_link"
                    and str(eph[0]) == PHASE_DEP_TAXI
                ):
                    _mf0 = False
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
                apron_stand_id=_flight_apron_stand_id_from_fobj(
                    fobj if isinstance(fobj, dict) else {}
                ),
                segment_v0_ms=list(v0_rem),
                segment_accel_ms2=list(acc_rem),
                segment_path_types=path_rem,
                segment_graph_uv=list(_guv_rem),
                completed_directed_hops=list(_cdh),
                motion_is_forward=_mf0,
                dep_taxi_start_sim_time=None,
                dep_taxi_start_abs_sec=None,
                arr_runway_id=_arr_rid if _arr_rid else None,
                arr_runway_dir=_flight_rw_dir_for_leg(
                    fobj if isinstance(fobj, dict) else {}, 0, layout
                )
                if _arr_rid
                else None,
                dep_runway_id=_dep_rid if _dep_rid else None,
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
                    and eph[0] in (PHASE_LANDING, PHASE_ARR_TAXI, PHASE_ARR_TAXI_TEMP)
                ):
                    v_init = max(v_init, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                if _rf > 1e-12:
                    v_init = max(v_init, MIN_ARR_RUNWAY_TAXIWAY_VELOCITY_MS)
                ag_new.velocity_ms = v_init
            agents_by_id[fid] = ag_new
        if progress_cb:
            progress_cb(float(i + 1), float(total), None)

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
    flight_input_order: Dict[str, int] = {}
    for _fi, fobj in enumerate(flights_raw):
        if isinstance(fobj, dict) and fobj.get("id") is not None:
            _fid = str(fobj["id"])
            flights_by_id[_fid] = fobj
            flight_input_order.setdefault(_fid, int(_fi))

    eldt_vals = [
        float(ag.eldt_anchor_sec)
        for ag in agents
        if ag.eldt_anchor_sec is not None
    ]
    ref_t0 = min(eldt_vals) if eldt_vals else 0.0
    current_time_abs = float(ref_t0)
    progress_elapsed_total_sec = _sim_progress_elapsed_total_sec(
        flights_raw, float(ref_t0)
    )
    max_stot_abs = _max_stot_sd_sec(flights_raw)
    truncation_abs_sec: Optional[float] = None

    while True:
        if not any(bool(ag.edge_ids) or ag.awaiting_apron_from_temp for ag in agents):
            break
        if max_stot_abs is not None:
            if (
                float(current_time_abs) + float(dt_sec)
                > float(max_stot_abs) + float(STOT_POST_BUFFER_SEC) + 1e-6
            ):
                break
        elif current_time_abs - float(ref_t0) + dt_sec > SIM_MAX_TIME_SEC + 1e-6:
            break
        current_time_abs += dt_sec
        _refresh_touchdown_motion_cache(control_state, agents, rw_release_lag)
        refresh_resource_occupancy(
            control_state,
            agents,
            pixels_per_meter,
            current_time_abs,
            rw_release_lag,
        )
        for ag in agents:
            _tick_arr_temp_detour_eldt_flag(ag, float(current_time_abs))
        agents_temp_pipe = sorted(
            agents,
            key=lambda a: _temp_stand_pipeline_sort_key(
                a, flights_by_id, flight_input_order
            ),
        )
        for ag in agents_temp_pipe:
            fo = flights_by_id.get(str(ag.id))
            if isinstance(fo, dict):
                _try_inject_arr_taxi_from_temp_stand(
                    ag,
                    fo,
                    layout,
                    control_state,
                    information,
                    cell_size,
                    reverse_cost,
                    merge_r,
                    taxiway_h,
                    float(current_time_abs),
                    agents,
                )
                _try_reroute_temp_stand_if_contested(
                    ag,
                    fo,
                    layout,
                    control_state,
                    information,
                    cell_size,
                    reverse_cost,
                    merge_r,
                    taxiway_h,
                    float(current_time_abs),
                    agents,
                )
                _try_splice_temp_stand_arrival_detour(
                    ag,
                    fo,
                    layout,
                    control_state,
                    information,
                    cell_size,
                    reverse_cost,
                    merge_r,
                    taxiway_h,
                    float(current_time_abs),
                    agents,
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
        elif (
            float(current_time_abs)
            - float(control_state.last_light_reservation_rebook_sim_time)
            + 1e-9
            >= float(LIGHT_RESERVATION_RETRY_INTERVAL_SEC)
        ):
            _book_lt, _ = _single_full_reservation_pass(
                control_state,
                agents,
                current_time_abs,
                pixels_per_meter,
                rw_release_lag,
            )
            control_state.stand_arrival_book_snapshot = dict(_book_lt)
            control_state.last_light_reservation_rebook_sim_time = float(
                current_time_abs
            )
        apply_movement_controls(
            control_state,
            agents,
            dt_sec,
            pixels_per_meter,
            current_time_abs,
            rw_release_lag,
            on_temp_park_arrival_immediate_reroute=lambda: _run_temp_park_arrival_global_reroute_bundle(
                control_state,
                agents,
                flights_by_id,
                layout,
                information,
                float(current_time_abs),
                cell_size,
                reverse_cost,
                merge_r,
                taxiway_h,
                pixels_per_meter,
                rw_release_lag,
            ),
        )
        _refresh_touchdown_motion_cache(control_state, agents, rw_release_lag)
        for ag in agents:
            # Always record history (even before touchdown / during runway-separation hold).
            # Skipping rows here used to create multi‑second gaps in `positions` while
            # `control_halt` still froze the aircraft — the UI then linearly interpolated
            # across a straight chord (slow→fast, corner aliasing). Same x,y, v≈0 samples
            # over the hold yield stable playback; touchdown motion still only starts after
            # `td_h` in `apply_movement_controls` above.
            td_h = _arr_touchdown_motion_abs_sec(
                ag, agents, rw_release_lag, control_state=control_state
            )
            _try_stamp_actual_apron_inblocks_from_stand_position(
                ag,
                layout,
                cell_size,
                pixels_per_meter,
                information,
                current_time_abs,
                control_state,
                agents,
            )
            _pt_eobt = (
                str(ag.segment_path_types[0] or "")
                if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
                else ""
            )
            if (
                ag.actual_apron_offblocks_abs_sec is None
                and ag.actual_apron_inblocks_abs_sec is not None
                and ag.edge_phases
                and str(ag.edge_phases[0]) == PHASE_DEP_TAXI
                and _pt_eobt == "apron_link"
                and ag.dep_taxi_start_abs_sec is not None
                and float(current_time_abs) > float(ag.dep_taxi_start_abs_sec) + 1e-9
                and abs(float(ag.velocity_ms)) > 0.01
            ):
                ag.actual_apron_offblocks_abs_sec = float(current_time_abs)
            st_h = control_state.agent_states.get(ag.id)
            _gh = (
                _agent_deadlock_ghost_at_time(st_h, float(current_time_abs))
                if st_h is not None
                else False
            )
            _dst_snap = _destination_stand_history_snap(
                ag, control_state, agents, float(current_time_abs)
            )
            _st_dbg = control_state.agent_states.get(ag.id)
            _eid0 = str(ag.edge_ids[0]) if ag.edge_ids else ""
            _ph0 = str(ag.edge_phases[0]) if ag.edge_phases else ""
            _pt0 = (
                str(ag.segment_path_types[0] or "")
                if ag.segment_path_types and len(ag.segment_path_types) == len(ag.edge_ids)
                else ""
            )
            _rw0: Optional[str] = None
            if _eid0:
                _er0 = control_state.edge_resources.get(_eid0)
                if _er0 is not None and _er0.runway_id:
                    _rw0 = str(_er0.runway_id)
            ag.history.append(
                (
                    current_time_abs,
                    ag.col,
                    ag.row,
                    ag.velocity_ms,
                    bool(ag.motion_is_forward),
                    bool(_gh),
                    _dst_snap,
                    (_st_dbg.clearance if _st_dbg is not None else None),
                    (_st_dbg.wait_reason if _st_dbg is not None else None),
                    (_eid0 or None),
                    (_ph0 or None),
                    (_pt0 or None),
                    bool(ag.control_halt),
                    (float(ag.control_speed_cap_ms) if ag.control_speed_cap_ms is not None else None),
                    (_rw0 or None),
                )
            )
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
            progress_cb(
                current_time_abs - float(ref_t0),
                progress_elapsed_total_sec,
                float(current_time_abs),
            )
        if int(control_state.deadlock_resolve_event_count) >= int(deadlock_resolve_stop_n):
            truncation_abs_sec = float(current_time_abs)
            _LOG.warning(
                "SIM_STOP_DEADLOCK_CAP t=%.1f count=%s cap=%s",
                float(current_time_abs),
                int(control_state.deadlock_resolve_event_count),
                int(deadlock_resolve_stop_n),
            )
            break

    if truncation_abs_sec is not None:
        t_cut = float(truncation_abs_sec)
        for ag in agents:
            ag.history = [
                row for row in ag.history if float(row[0]) <= t_cut + 1e-9
            ]

    for ag in agents:
        _backfill_actual_apron_offblocks_from_history(ag)

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
            runway_entry_abs_sec=(ag.runway_entry_abs_sec if ag else None),
            touchdown_motion_abs_sec=(
                _arr_touchdown_motion_abs_sec(
                    ag, agents, rw_release_lag, control_state=control_state
                )
                if ag
                else None
            ),
            arr_runway_id=(ag.arr_runway_id if ag else None),
            dep_runway_id=(ag.dep_runway_id if ag else None),
            has_landing_leg=any(str(p) == PHASE_LANDING for p in prep_i.segment_phases),
            has_lineup_departure_leg=_flight_path_has_lineup_departure(prep_i),
            actual_apron_inblocks_abs_sec=(
                ag.actual_apron_inblocks_abs_sec if ag else None
            ),
            actual_apron_offblocks_abs_sec=(
                ag.actual_apron_offblocks_abs_sec if ag else None
            ),
            path_completed_abs_sec=(
                ag.path_completed_abs_sec if ag is not None else None
            ),
            information=information,
        )
        eibt_raw = sched_row.get("EIBT")
        eobt_raw = sched_row.get("EOBT")
        if eibt_raw is None or eobt_raw is None:
            continue
        parked_stand_id = _resolve_parked_stand_id(
            ag.apron_stand_id,
            ag.history,
            float(eibt_raw),
            float(eobt_raw),
        )
        parked_nose_heading_deg = _stand_nose_heading_deg(
            layout,
            float(cell_size),
            parked_stand_id,
        )
        ag.history = _compress_agent_history_for_dwell_export(
            ag.history,
            ag.eldt_anchor_sec,
            float(eibt_raw),
            float(eobt_raw),
            ag.dep_taxi_start_abs_sec,
            parked_nose_heading_deg,
        )

    positions: Dict[str, List[Dict[str, Any]]] = {}
    for ag in agents:
        _plist: List[Dict[str, Any]] = []
        _pc = ag.path_completed_abs_sec
        for row in ag.history:
            t, c, r, v = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if _pc is not None and t > float(_pc) + 1e-9:
                continue
            _mf = bool(row[4]) if len(row) > 4 else True
            _dghost = bool(row[5]) if len(row) > 5 else False
            _pos: Dict[str, Any] = {
                "t": int(round(t)),
                "x": round(c, 3),
                "y": round(r, 3),
                "v": round(v, 3),
                "motionForward": _mf,
                "deadlockGhost": _dghost,
            }
            _dsnap = row[6] if len(row) > 6 else None
            if _dsnap is not None:
                _sid, _cap, _phys_o, _booked, _cd, _ok = _dsnap
                _pos["destinationApron"] = {
                    "standId": str(_sid),
                    "capacity": int(_cap),
                    "physicalOthersOnStand": int(_phys_o),
                    "pipelineBooked": int(_booked),
                    "pushbackCooldownActive": bool(_cd),
                    "standPipelineOpen": bool(_ok),
                }
            if len(row) > 7 and row[7] is not None:
                _pos["clearance"] = str(row[7])
            if len(row) > 8 and row[8] is not None:
                _pos["waitReason"] = str(row[8])
            if len(row) > 9 and row[9] is not None:
                _pos["edgeId"] = str(row[9])
            if len(row) > 10 and row[10] is not None:
                _pos["phase"] = str(row[10])
            if len(row) > 11 and row[11] is not None:
                _pos["pathType"] = str(row[11])
            if len(row) > 12:
                _pos["controlHalt"] = bool(row[12])
            if len(row) > 13 and row[13] is not None:
                try:
                    _pos["controlSpeedCapMs"] = float(row[13])
                except (TypeError, ValueError):
                    pass
            if len(row) > 14 and row[14] is not None:
                _pos["runwayId"] = str(row[14])
            _plist.append(_pos)
        # Playback: do not export positions before actual touchdown motion (same clock as
        # schedule TOUCHDOWN_MOTION / _compute_arr_touchdown_motion_abs_sec). Sim internal
        # state and ag.history are unchanged; only the JSON `positions` list is trimmed.
        td_playback = _arr_touchdown_motion_abs_sec(
            ag, agents, rw_release_lag, control_state=control_state
        )
        if (
            _agent_path_includes_landing(ag)
            and td_playback is not None
            and math.isfinite(float(td_playback))
        ):
            td0 = float(td_playback)
            _plist = [
                p
                for p in _plist
                if float(p.get("t", 0.0)) + 1e-9 >= td0
            ]
        positions[ag.id] = _plist

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
                runway_entry_abs_sec=(ag.runway_entry_abs_sec if ag else None),
                touchdown_motion_abs_sec=(
                    _arr_touchdown_motion_abs_sec(
                        ag, agents, rw_release_lag, control_state=control_state
                    )
                    if ag
                    else None
                ),
                arr_runway_id=(ag.arr_runway_id if ag else None),
                dep_runway_id=(ag.dep_runway_id if ag else None),
                has_landing_leg=any(str(p) == PHASE_LANDING for p in prep.segment_phases),
                has_lineup_departure_leg=_flight_path_has_lineup_departure(prep),
                eldt_schedule_sec=eldt_adjust_map.get(str(fid)),
                actual_apron_inblocks_abs_sec=(
                    ag.actual_apron_inblocks_abs_sec if ag else None
                ),
                actual_apron_offblocks_abs_sec=(
                    ag.actual_apron_offblocks_abs_sec if ag else None
                ),
                path_completed_abs_sec=(
                    ag.path_completed_abs_sec if ag is not None else None
                ),
                information=information,
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

    out: Dict[str, Any] = {
        "baseDate": base_date,
        "positions": positions,
        "schedule": schedule_list,
        "layout": None,
        "kpi": None,
        "flights_detail": flights_detail,
        "deadlock_resolve_event_count": int(control_state.deadlock_resolve_event_count),
    }
    if truncation_abs_sec is not None:
        out["simulation_truncated_deadlock"] = True
        out["simulation_playback_end_abs_sec"] = float(truncation_abs_sec)
    _overlay_schedule_timing_from_playback_positions(
        out,
        layout,
        information,
        flights_raw,
        cell_size,
        pixels_per_meter,
    )
    return out
