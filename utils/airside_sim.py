"""
Airside path-only simulation: build the same junction graph as Layout_Design (via simPathGraph
or build_path_graph), run Dijkstra per flight (touch runway node → apron node or reverse for
departure), return ordered layout edge ids for UI highlight.

DES/timeline positions are not computed here — payloads use empty positions/schedule so Apply
does not ingest playback data until that layer is reintroduced.
"""
from __future__ import annotations

import json
import math
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


def _runway_touch_pixel(layout: Dict[str, Any], cell_size: float, runway_id: str, rw_dir: str) -> Optional[Tuple[float, float]]:
    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None
    if normalize_rw_direction_value(rw_dir) == "counter_clockwise":
        return r["endPx"]
    return r["startPx"]


def _best_path_arrival(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    apron_id: str,
    arr_dir_hint: Optional[str],
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
) -> Optional[List[int]]:
    nd = normalize_rw_direction_value(arr_dir_hint or "")
    candidates: List[str] = []
    if nd == "clockwise":
        candidates = ["clockwise"]
    elif nd == "counter_clockwise":
        candidates = ["counter_clockwise"]
    else:
        candidates = ["clockwise", "counter_clockwise"]

    best: Optional[List[int]] = None
    best_d = float("inf")
    for rd in candidates:
        for pure_g in (False, True):
            g = _graph_for_direction(
                layout,
                cell_size,
                rd,
                reverse_cost,
                merge_r,
                taxiway_h,
                information,
                pure_ground_exclude_runway=pure_g,
            )
            if g is None or not g.nodes:
                continue
            rw_px = _runway_touch_pixel(layout, cell_size, runway_id, rd)
            if rw_px is None:
                continue
            start = nearest_path_node_on_runway_polyline(g, runway_id, rw_px)
            end = _stand_end_node_index(g, layout, str(apron_id), cell_size)
            if end is None:
                continue
            path = path_dijkstra(g, start, end)
            if not path or len(path) < 2:
                continue
            d = path_total_dist(g, path)
            if d < best_d:
                best_d = d
                best = path
    return best


def _best_path_departure(
    layout: Dict[str, Any],
    cell_size: float,
    runway_id: str,
    apron_id: str,
    dep_dir_hint: Optional[str],
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
) -> Optional[List[int]]:
    nd = normalize_rw_direction_value(dep_dir_hint or "")
    candidates: List[str] = []
    if nd == "clockwise":
        candidates = ["clockwise"]
    elif nd == "counter_clockwise":
        candidates = ["counter_clockwise"]
    else:
        candidates = ["clockwise", "counter_clockwise"]

    best: Optional[List[int]] = None
    best_d = float("inf")
    for rd in candidates:
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
            continue
        start = _stand_end_node_index(g, layout, str(apron_id), cell_size)
        if start is None:
            continue
        r = get_runway_path_px(layout, cell_size, runway_id)
        if not r:
            continue
        rw_px = r["endPx"] if normalize_rw_direction_value(rd) == "clockwise" else r["startPx"]
        end = nearest_path_node_on_runway_polyline(g, runway_id, rw_px)
        path = path_dijkstra(g, start, end)
        if not path or len(path) < 2:
            continue
        d = path_total_dist(g, path)
        if d < best_d:
            best_d = d
            best = path
    return best


def _flight_route(
    layout: Dict[str, Any],
    flight: Dict[str, Any],
    cell_size: float,
    reverse_cost: float,
    merge_r: float,
    taxiway_h: float,
    information: Dict[str, Any],
    pair_index: Dict[Tuple[int, int], str],
) -> Tuple[List[str], Optional[str]]:
    fid = str(flight.get("id", ""))
    token = flight.get("token") if isinstance(flight.get("token"), dict) else {}
    apron_id = flight.get("standId")
    if apron_id is None:
        apron_id = token.get("apronId")
    arr_dep = str(flight.get("arrDep") or "Arr")

    if apron_id is None or str(apron_id).strip() == "":
        return [], "missing apronId"

    path: Optional[List[int]] = None
    err: Optional[str] = None

    if arr_dep == "Dep":
        dep_rwy = flight.get("depRunwayId") or token.get("depRunwayId")
        if not dep_rwy:
            return [], "missing depRunwayId"
        dir_hint = flight.get("depRunwayDirUsed")
        path = _best_path_departure(
            layout,
            cell_size,
            str(dep_rwy),
            str(apron_id),
            str(dir_hint) if dir_hint else None,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
        )
        if path is None:
            err = "no departure path"
    else:
        arr_rwy = flight.get("arrRunwayId") or token.get("arrRunwayId")
        if not arr_rwy:
            return [], "missing arrRunwayId"
        dir_hint = flight.get("arrRunwayDirUsed")
        path = _best_path_arrival(
            layout,
            cell_size,
            str(arr_rwy),
            str(apron_id),
            str(dir_hint) if dir_hint else None,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
        )
        if path is None:
            err = "no arrival path"

    if not path:
        return [], err or "no path"
    return _path_to_edge_ids(path, pair_index), None


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

        edge_ids, err = _flight_route(
            layout,
            fobj,
            cell_size,
            reverse_cost,
            merge_r,
            taxiway_h,
            information,
            pair_index,
        )
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
