"""
Airside path-only simulation: build the same junction graph as Layout_Design (via simPathGraph
or build_path_graph), run Dijkstra per flight, return ordered layout edge ids for UI highlight.

Each flight is a short list of path legs ``(start: RouteEndpoint, end: RouteEndpoint, …)`` —
e.g. arrival runway touch → apron, then apron → departure lineup. ``run_simulation`` loops legs
and calls ``_flight_route`` per leg to append edge ids.

DES/timeline positions are not computed here — payloads use empty positions/schedule so Apply
does not ingest playback data until that layer is reintroduced.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.designer_path_graph import (
    PathGraph,
    _stand_end_node_index,
    build_path_graph,
    get_runway_path_px,
    nearest_path_node_on_runway_polyline,
    normalize_allowed_runway_directions,
    normalize_rw_direction_value,
    path_dijkstra,
    path_graph_from_layout_sim_export,
    path_total_dist,
)

_ROOT = Path(__file__).resolve().parents[1]
_INFORMATION_PATH = (_ROOT / "data" / "Info_storage" / "Information.json").resolve()
_DEFAULT_RW_DIR = "clockwise"


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
    """Single path-graph endpoint: apron stand node, or nearest graph node on a runway polyline."""

    apron_stand_id: Optional[str] = None
    runway_id: Optional[str] = None
    runway_pixel_xy: Optional[Tuple[float, float]] = None


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


def flight_route_edges(
    g: PathGraph,
    layout: Dict[str, Any],
    cell_size: float,
    pair_index: Dict[Tuple[int, int], str],
    start_point: RouteEndpoint,
    end_point: RouteEndpoint,
) -> Tuple[List[str], Optional[str]]:
    """
    Edge id list for the shortest path on ``g``. Second value is ``direction_violation`` if
    the path uses reverse-penalty arcs; otherwise ``None`` (including no path → ``[]``).
    """
    edges, _dist, path = flight_route(g, layout, cell_size, pair_index, start_point, end_point)
    if path is None or len(path) < 2:
        return [], None
    if _path_uses_reverse_penalty_edges(g, path):
        return [], "direction_violation"
    return (edges, None) if edges else ([], None)


def _runway_direction_single(hint: Optional[str]) -> str:
    nd = normalize_rw_direction_value(hint or "")
    if nd == "clockwise":
        return "clockwise"
    if nd == "counter_clockwise":
        return "counter_clockwise"
    return _DEFAULT_RW_DIR


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


def _runway_touch_pixel(layout: Dict[str, Any], cell_size: float, runway_id: str, rw_dir: str) -> Optional[Tuple[float, float]]:
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    if normalize_rw_direction_value(rw_dir) == "counter_clockwise":
        return r["endPx"]
    return r["startPx"]


def _arrival_touch_to_apron(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    stand_id: str,
    rw_dir_hint: Optional[str],
) -> Optional[Tuple[RouteEndpoint, RouteEndpoint]]:
    rd = _runway_direction_single(rw_dir_hint)
    rw_px = _runway_touch_pixel(layout, cell_size, runway_id, rd)
    if rw_px is None:
        return None
    return (
        RouteEndpoint(runway_id=str(runway_id), runway_pixel_xy=rw_px),
        RouteEndpoint(apron_stand_id=str(stand_id)),
    )


def _apron_to_departure_lineup(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    stand_id: str,
    rw_dir_hint: Optional[str],
) -> Optional[Tuple[RouteEndpoint, RouteEndpoint]]:
    rd = _runway_direction_single(rw_dir_hint)
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    rw_px = r["endPx"] if normalize_rw_direction_value(rd) == "clockwise" else r["startPx"]
    return (
        RouteEndpoint(apron_stand_id=str(stand_id)),
        RouteEndpoint(runway_id=str(runway_id), runway_pixel_xy=rw_px),
    )


# (start, end, rw_dir_hint, error_if_empty_when_required, required_leg)
_FlightPathLeg = Tuple[RouteEndpoint, RouteEndpoint, Optional[str], str, bool]


def _flight_paths_for_fobject(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    stand_id: str,
) -> Tuple[List[_FlightPathLeg], Optional[str]]:
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    arr_dep = str(flight.get("arrDep") or "Arr")
    arr_rwy = flight.get("arrRunwayId") or token.get("arrRunwayId")
    dep_rwy = flight.get("depRunwayId") or token.get("depRunwayId")
    ah = flight.get("arrRunwayDirUsed")
    dh = flight.get("depRunwayDirUsed")
    arr_hint = str(ah) if ah else None
    dep_hint = str(dh) if dh else None

    paths: List[_FlightPathLeg] = []

    if arr_dep != "Dep":
        if not arr_rwy:
            return [], "missing arrRunwayId"
        pair = _arrival_touch_to_apron(layout, cell_size, str(arr_rwy), stand_id, arr_hint)
        if pair is None:
            return [], "no arrival path"
        s, e = pair
        paths.append((s, e, arr_hint, "no arrival path", True))
        if dep_rwy:
            pair2 = _apron_to_departure_lineup(layout, cell_size, str(dep_rwy), stand_id, dep_hint)
            if pair2 is not None:
                s2, e2 = pair2
                paths.append((s2, e2, dep_hint, "no departure path", False))
    else:
        if not dep_rwy:
            return [], "missing depRunwayId"
        pair = _apron_to_departure_lineup(layout, cell_size, str(dep_rwy), stand_id, dep_hint)
        if pair is None:
            return [], "no departure path"
        s, e = pair
        paths.append((s, e, dep_hint, "no departure path", True))
        if arr_rwy:
            pair2 = _arrival_touch_to_apron(layout, cell_size, str(arr_rwy), stand_id, arr_hint)
            if pair2 is not None:
                s2, e2 = pair2
                paths.append((s2, e2, arr_hint, "no arrival path", False))

    return paths, None


def _flight_route(
    layout: Dict[str, Any],
    cell_size: float,
    pair_index: Dict[Tuple[int, int], str],
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    start: RouteEndpoint,
    end: RouteEndpoint,
    rw_dir_hint: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    rd = _runway_direction_single(rw_dir_hint)
    g = _graph_for_direction(
        layout,
        cell_size,
        rd,
        reverse_cost,
        merge_r,
        taxiway_h,
        information,
        pure_ground_exclude_runway=False,
    )
    if g is None or not g.nodes:
        return None, None
    edges, fe_err = flight_route_edges(g, layout, cell_size, pair_index, start, end)
    if fe_err == "direction_violation":
        return [], "direction_violation"
    if not edges:
        return None, None
    return edges, None


def run_simulation(
    layout: Dict[str, Any],
    dt: float = 1.0,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> Dict[str, Any]:
    del dt  # time-step DES not used in path-only mode
    information = _load_information_json()
    reverse_cost, merge_r, taxiway_h = _path_search_params(information)
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))

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

    flights_raw = layout.get("flights") if isinstance(layout.get("flights"), list) else []

    total = max(1, len(flights_raw))
    flights_detail: List[Dict[str, Any]] = []

    for i, fobj in enumerate(flights_raw):
        if not isinstance(fobj, dict):
            continue
        fid = fobj.get("id")
        if fid is None:
            continue

        token = fobj.get("token") if isinstance(fobj.get("token"), dict) else {}
        stand_id = fobj.get("standId")
        if stand_id is None:
            stand_id = token.get("apronId")
        if stand_id is None or str(stand_id).strip() == "":
            flights_detail.append(
                {
                    "flight_id": str(fid),
                    "edge_list": [],
                    "ok": False,
                    "error": "missing apronId",
                }
            )
            if progress_cb:
                progress_cb(float(i + 1), float(total))
            continue

        paths, setup_err = _flight_paths_for_fobject(fobj, layout, cell_size, str(stand_id))
        edge_ids: List[str] = []
        err: Optional[str] = setup_err
        if err is None:
            for start, end, rw_hint, no_path_err, required in paths:
                seg, seg_err = _flight_route(
                    layout,
                    cell_size,
                    pair_index,
                    reverse_cost,
                    merge_r,
                    taxiway_h,
                    information,
                    start,
                    end,
                    rw_hint,
                )
                if seg_err == "direction_violation":
                    err = seg_err
                    edge_ids = []
                    break
                if not seg:
                    if required:
                        err = no_path_err
                        edge_ids = []
                        break
                    continue
                edge_ids.extend(seg)
            if not edge_ids and err is None:
                err = "no path"

        flights_detail.append(
            {
                "flight_id": str(fid),
                "edge_list": edge_ids,
                "ok": err is None,
                "error": err,
            }
        )
        if progress_cb:
            progress_cb(float(i + 1), float(total))

    base_date = str(
        _deep_get(information, "tiers", "algorithm", "simulation", "baseDate", default="2026-03-31")
    )

    return {
        "baseDate": base_date,
        "positions": {},
        "schedule": [],
        "layout": None,
        "kpi": None,
        "flights_detail": flights_detail,
    }
