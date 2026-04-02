"""
Airside path-only simulation: build the same junction graph as Layout_Design (via simPathGraph
or build_path_graph), run Dijkstra per flight, return ordered layout edge ids for UI highlight.

Token layout pixels are resolved in ``extract_point_to_paths`` as legs
``[sx, sy, ex, ey]``. ``run_simulation`` converts each leg to ``RouteEndpoint`` for routing
(single default runway direction for the graph).

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
    find_stand_by_id,
    get_runway_path_px,
    get_stand_connection_px,
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
    """Path-graph endpoint: stand id, runway polyline snap, or layout ``(x,y)`` → nearest graph node."""

    apron_stand_id: Optional[str] = None
    runway_id: Optional[str] = None
    runway_pixel_xy: Optional[Tuple[float, float]] = None
    token_pixel_xy: Optional[Tuple[float, float]] = None


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


def _runway_touch_pixel(layout: Dict[str, Any], cell_size: float, runway_id: str, rw_dir: str) -> Optional[Tuple[float, float]]:
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    if normalize_rw_direction_value(rw_dir) == "counter_clockwise":
        return r["endPx"]
    return r["startPx"]


def _arr_touchdown_token_xy(
    layout: Dict[str, Any], cell_size: float, runway_id: str
) -> Optional[Tuple[float, float]]:
    return _runway_touch_pixel(layout, cell_size, runway_id, _DEFAULT_RW_DIR)


def _dep_lineup_token_xy(layout: Dict[str, Any], cell_size: float, runway_id: str) -> Optional[Tuple[float, float]]:
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    rd = _DEFAULT_RW_DIR
    return r["endPx"] if normalize_rw_direction_value(rd) == "clockwise" else r["startPx"]


def _apron_token_xy(layout: Dict[str, Any], cell_size: float, stand_id: str) -> Optional[Tuple[float, float]]:
    st = find_stand_by_id(layout, str(stand_id))
    if not st:
        return None
    return get_stand_connection_px(st, cell_size)


def extract_point_to_paths(
    flight: Dict[str, Any],
    layout: Dict[str, Any],
    cell_size: float,
    stand_id: str,
) -> List[List[float]]:
    """
    Token pixels as path legs: each leg is ``[start_x, start_y, end_x, end_y]`` (layout px).
    Returns an empty list if legs cannot be built.
    """
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    arr_dep = str(flight.get("arrDep") or "Arr")
    arr_rwy = flight.get("arrRunwayId") or token.get("arrRunwayId")
    dep_rwy = flight.get("depRunwayId") or token.get("depRunwayId")


    arr_rw_touchdown_point = _arr_touchdown_token_xy(layout, cell_size, str(arr_rwy)) if arr_rwy else None
    apron_point = _apron_token_xy(layout, cell_size, str(stand_id))
    dep_rw_lineup_point = _dep_lineup_token_xy(layout, cell_size, str(dep_rwy)) if dep_rwy else None

    paths: List[List[float]] = []

    if arr_dep != "Dep":
        if not arr_rwy:
            return []
        if arr_rw_touchdown_point is None or apron_point is None:
            return []
        ax, ay = arr_rw_touchdown_point
        px, py = apron_point
        paths.append([float(ax), float(ay), float(px), float(py)])
        if dep_rwy and dep_rw_lineup_point is not None:
            lx, ly = dep_rw_lineup_point
            paths.append([float(px), float(py), float(lx), float(ly)])
    else:
        if not dep_rwy:
            return []
        if apron_point is None or dep_rw_lineup_point is None:
            return []
        px, py = apron_point
        lx, ly = dep_rw_lineup_point
        paths.append([float(px), float(py), float(lx), float(ly)])
        if arr_rwy and arr_rw_touchdown_point is not None:
            ax, ay = arr_rw_touchdown_point
            paths.append([float(ax), float(ay), float(px), float(py)])

    return paths


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
) -> Tuple[List[str], bool]:
    """``(edges, direction_violation)``. On violation, ``edges`` is empty and the flag is True."""
    rd = _DEFAULT_RW_DIR
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
        return [], False
    edges, _dist, path = flight_route(g, layout, cell_size, pair_index, start, end)
    if path is None or len(path) < 2:
        return [], False
    if _path_uses_reverse_penalty_edges(g, path):
        return [], True
    return edges, False


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
                }
            )
            if progress_cb:
                progress_cb(float(i + 1), float(total))
            continue

        paths = extract_point_to_paths(fobj, layout, cell_size, str(stand_id))
        edge_ids: List[str] = []
        for leg in paths:
            if len(leg) < 4:
                continue
            sx, sy, ex, ey = float(leg[0]), float(leg[1]), float(leg[2]), float(leg[3])
            seg, direction_violation = _flight_route(
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
            if direction_violation:
                edge_ids = []
                break
            edge_ids.extend(seg)

        flights_detail.append(
            {
                "flight_id": str(fid),
                "edge_list": edge_ids,
                "ok": len(edge_ids) > 0,
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
