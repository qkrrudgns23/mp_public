"""
Port of pages/Layout_Design/designer.js path graph: buildPathGraph, pathDijkstra,
and arrival path resolution (solve_arrival_path_indices). See designer.js ~9389–9862, ~10003–10095.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]

SPLIT_TOL_D2 = 0.25


def _dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def path_dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vertex_to_px(v: dict, cell_size: float) -> Point:
    if not isinstance(v, dict):
        return (0.0, 0.0)
    cs = max(float(cell_size), 1e-9)
    if v.get("x") is not None and v.get("y") is not None:
        return (float(v["x"]), float(v["y"]))
    return (float(v.get("col", 0) or 0) * cs, float(v.get("row", 0) or 0) * cs)


def project_on_segment(a: Point, b: Point, q: Point) -> Tuple[float, Point]:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    qx, qy = q[0], q[1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return (0.0, a)
    t = ((qx - ax) * dx + (qy - ay) * dy) / den
    t = max(0.0, min(1.0, t))
    return (t, (ax + t * dx, ay + t * dy))


def segment_segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Optional[Point]:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    dx_, dy_ = d[0], d[1]
    rx, ry = bx - ax, by - ay
    sx, sy = dx_ - cx, dy_ - cy
    cross = rx * sy - ry * sx
    if abs(cross) < 1e-12:
        return None
    t = ((cx - ax) * sy - (cy - ay) * sx) / cross
    s = ((cx - ax) * ry - (cy - ay) * rx) / cross
    if t < 0 or t > 1 or s < 0 or s > 1:
        return None
    return (ax + t * rx, ay + t * ry)


def collinear_segment_overlap_on_ab(a: Point, b: Point, c: Point, d: Point) -> Optional[Tuple[float, float]]:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    dx = bx - ax
    dy = by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return None
    length = math.sqrt(len2)

    def perp_dist_ab(p: Point) -> float:
        return abs((p[0] - ax) * dy - (p[1] - ay) * dx) / length

    line_tol = max(0.55, length * 1e-9)
    if perp_dist_ab(c) > line_tol or perp_dist_ab(d) > line_tol:
        return None

    def t_on_ab(p: Point) -> float:
        return ((p[0] - ax) * dx + (p[1] - ay) * dy) / len2

    tc, td = t_on_ab(c), t_on_ab(d)
    lo, hi = min(tc, td), max(tc, td)
    o0, o1 = max(0.0, lo), min(1.0, hi)
    if o1 < o0 - 1e-9:
        return None
    return (o0, o1)


def point_on_segment_strict(a: Point, b: Point, q: Point) -> bool:
    _, p = project_on_segment(a, b, q)
    return _dist2(p, q) <= SPLIT_TOL_D2


def dedupe_path_points(pts: List[Point]) -> List[Point]:
    out: List[Point] = []
    for p in pts or []:
        if len(p) < 2:
            continue
        if not out or _dist2(out[-1], p) > SPLIT_TOL_D2:
            out.append((float(p[0]), float(p[1])))
    return out


def polyline_distance_between_along(pts: List[Point], start_along: float, end_along: float) -> float:
    if not pts or len(pts) < 2:
        return 0.0
    a0 = max(0.0, float(start_along))
    a1 = max(a0, float(end_along))
    dist_acc = 0.0
    for seg in range(int(math.floor(a0)), min(len(pts) - 2, int(math.floor(a1))) + 1):
        seg_start = max(float(seg), a0)
        seg_end = min(float(seg + 1), a1)
        if seg_end <= seg_start:
            continue
        seg_len = path_dist(pts[seg], pts[seg + 1])
        if seg_len <= 1e-9:
            continue
        dist_acc += seg_len * (seg_end - seg_start)
    return dist_acc


def polyline_points_between_along(pts: List[Point], start_along: float, end_along: float) -> List[Point]:
    if not pts or len(pts) < 2:
        return []
    a0 = max(0.0, float(start_along))
    a1 = max(a0, float(end_along))
    start_seg = max(0, min(len(pts) - 2, int(math.floor(a0))))
    end_seg = max(0, min(len(pts) - 2, int(math.floor(a1))))
    start_t = a0 - start_seg
    end_t = a1 - end_seg
    p0, p1 = pts[start_seg], pts[start_seg + 1]
    start_pt = (p0[0] + (p1[0] - p0[0]) * start_t, p0[1] + (p1[1] - p0[1]) * start_t)
    p2, p3 = pts[end_seg], pts[end_seg + 1]
    end_pt = (p2[0] + (p3[0] - p2[0]) * end_t, p2[1] + (p3[1] - p2[1]) * end_t)
    out: List[Point] = [start_pt]
    for i in range(start_seg + 1, end_seg + 1):
        out.append((float(pts[i][0]), float(pts[i][1])))
    out.append(end_pt)
    return dedupe_path_points(out)


def _layout_path_objects(layout: dict) -> List[dict]:
    out: List[dict] = []
    for tw in layout.get("runwayPaths") or []:
        if isinstance(tw, dict):
            out.append(dict(tw, pathType="runway"))
    for tw in layout.get("runwayTaxiways") or []:
        if isinstance(tw, dict):
            out.append(dict(tw, pathType=tw.get("pathType", "runway_taxiway")))
    for tw in layout.get("taxiways") or []:
        if isinstance(tw, dict):
            out.append(dict(tw))
    return out


def get_runway_path_px(layout: dict, cell_size: float, runway_id: Optional[str]) -> Optional[Dict[str, Any]]:
    path_list = _layout_path_objects(layout)
    rw = None
    if runway_id:
        for t in path_list:
            if t.get("id") == runway_id and t.get("pathType") == "runway":
                rw = t
                break
    if rw is None:
        for t in path_list:
            if t.get("pathType") == "runway" and isinstance(t.get("vertices"), list) and len(t["vertices"]) >= 2:
                rw = t
                break
    if rw is None or not rw.get("vertices"):
        return None
    pts = [_vertex_to_px(v, cell_size) for v in rw["vertices"] if isinstance(v, dict)]
    if len(pts) < 2:
        return None
    sp = rw.get("start_point")
    ep = rw.get("end_point")
    if sp and ep and isinstance(sp, dict) and isinstance(ep, dict):
        start_px = _vertex_to_px(sp, cell_size)
        if _dist2(pts[-1], start_px) < _dist2(pts[0], start_px):
            pts = list(reversed(pts))
    return {"startPx": pts[0], "endPx": pts[-1], "pts": pts, "obj": rw}


def get_taxiway_ordered_points(obj: dict, cell_size: float) -> Optional[List[Point]]:
    verts = obj.get("vertices")
    if not isinstance(verts, list) or len(verts) < 2:
        return None
    pts = [_vertex_to_px(v, cell_size) for v in verts if isinstance(v, dict)]
    sp, ep = obj.get("start_point"), obj.get("end_point")
    if sp and ep and isinstance(sp, dict) and isinstance(ep, dict):
        start_px = _vertex_to_px(sp, cell_size)
        if _dist2(pts[-1], start_px) < _dist2(pts[0], start_px):
            pts = list(reversed(pts))
    return pts


def get_ordered_points(obj: dict, layout: dict, cell_size: float) -> Optional[List[Point]]:
    if not obj or not obj.get("vertices") or len(obj["vertices"]) < 2:
        return None
    if obj.get("pathType") == "runway":
        r = get_runway_path_px(layout, cell_size, obj.get("id"))
        return r["pts"] if r else None
    return get_taxiway_ordered_points(obj, cell_size)


def get_taxiway_direction(tw: dict, direction_modes: List[dict]) -> str:
    if not tw:
        return "both"
    d = tw.get("direction")
    if d is not None:
        if d == "topToBottom":
            return "clockwise"
        if d == "bottomToTop":
            return "counter_clockwise"
        return str(d) if d else "both"
    mode_id = tw.get("directionModeId")
    if mode_id:
        for m in direction_modes or []:
            if isinstance(m, dict) and m.get("id") == mode_id and m.get("direction"):
                return str(m["direction"])
    return "both"


def normalize_rw_direction_value(dir_v: Optional[str]) -> str:
    if dir_v in ("clockwise", "cw"):
        return "clockwise"
    if dir_v in ("counter_clockwise", "ccw"):
        return "counter_clockwise"
    return "both"


def normalize_allowed_runway_directions(raw: Any) -> List[str]:
    out: List[str] = []
    src = raw if isinstance(raw, list) else []
    for v in src:
        d = normalize_rw_direction_value(str(v) if v is not None else "")
        if d == "clockwise" and "clockwise" not in out:
            out.append("clockwise")
        if d == "counter_clockwise" and "counter_clockwise" not in out:
            out.append("counter_clockwise")
    return out


def get_taxiway_allowed_runway_directions(
    tw: dict, rw_exit_allowed_default: List[str]
) -> List[str]:
    if not tw or tw.get("pathType") != "runway_exit":
        base = rw_exit_allowed_default[:] if rw_exit_allowed_default else ["clockwise", "counter_clockwise"]
        return base
    arr = normalize_allowed_runway_directions(tw.get("allowedRwDirections"))
    if not arr:
        base = rw_exit_allowed_default[:] if rw_exit_allowed_default else ["clockwise", "counter_clockwise"]
        return base
    return arr


def is_runway_exit_direction_allowed(tw: dict, runway_dir: str, rw_exit_default: List[str]) -> bool:
    d = normalize_rw_direction_value(runway_dir)
    if d not in ("clockwise", "counter_clockwise"):
        return True
    allow = get_taxiway_allowed_runway_directions(tw, rw_exit_default)
    return d in allow


def find_stand_by_id(layout: dict, stand_id: str) -> Optional[dict]:
    for s in layout.get("pbbStands") or []:
        if isinstance(s, dict) and str(s.get("id", "")) == str(stand_id):
            return s
    for s in layout.get("remoteStands") or []:
        if isinstance(s, dict) and str(s.get("id", "")) == str(stand_id):
            return s
    return None


def get_stand_connection_px(stand: dict, cell_size: float) -> Point:
    if not stand:
        return (0.0, 0.0)
    if stand.get("apronSiteX") is not None and stand.get("apronSiteY") is not None:
        return (float(stand["apronSiteX"]), float(stand["apronSiteY"]))
    if stand.get("x2") is not None and stand.get("y2") is not None:
        return (float(stand["x2"]), float(stand["y2"]))
    if stand.get("x") is not None and stand.get("y") is not None:
        return (float(stand["x"]), float(stand["y"]))
    cs = max(float(cell_size), 1e-9)
    return (float(stand.get("col", 0) or 0) * cs, float(stand.get("row", 0) or 0) * cs)


def get_effective_runway_lineup_dist_m(tw: dict) -> float:
    if not tw or tw.get("pathType") != "runway":
        return 0.0
    v = tw.get("lineupDistM")
    if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
        return float(v)
    return 0.0


@dataclass
class DirectedEdgeRecord:
    """One directed arc in the path graph (designer.js edgeMap entry + metadata for sim export)."""

    from_idx: int
    to_idx: int
    cost: float
    raw_dist: float
    pts: List[Point]
    link_id: str
    path_type: str
    direction: str


@dataclass
class PathGraph:
    """designer.js buildPathGraph return shape (subset used by simulation)."""

    nodes: List[Point]
    adj: List[List[Tuple[int, float]]]
    edge_map: Dict[str, DirectedEdgeRecord] = field(default_factory=dict)
    runway_node_indices_by_id: Dict[str, set] = field(default_factory=dict)
    stand_id_to_node_index: Dict[str, int] = field(default_factory=dict)
    merge_radius_px: float = 7.0
    reverse_cost: float = 1_000_000.0

    def node_id_str(self, idx: int) -> str:
        return f"G{idx:05d}"

    def nearest_path_node(self, p: Point) -> int:
        best = 0
        best_d2 = _dist2(self.nodes[0], p)
        for i in range(1, len(self.nodes)):
            d2 = _dist2(self.nodes[i], p)
            if d2 < best_d2:
                best_d2 = d2
                best = i
        return best

    def nearest_path_node_from_set(self, node_set: set, p: Point) -> Optional[int]:
        if not node_set:
            return None
        best = None
        best_d2 = float("inf")
        for idx in node_set:
            if idx is None or idx < 0 or idx >= len(self.nodes):
                continue
            d2 = _dist2(self.nodes[idx], p)
            if d2 < best_d2:
                best_d2 = d2
                best = idx
        return best


def path_dijkstra(g: PathGraph, start_idx: Optional[int], end_idx: Optional[int]) -> Optional[List[int]]:
    """designer.js pathDijkstra (lazy heap; skip stale)."""
    n = len(g.nodes)
    if start_idx is None or end_idx is None or n == 0:
        return None
    if start_idx < 0 or end_idx < 0 or start_idx >= n or end_idx >= n:
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
            nd = d + w
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


def path_total_dist(g: PathGraph, path_indices: List[int]) -> float:
    d_acc = 0.0
    for i in range(len(path_indices) - 1):
        a, b = path_indices[i], path_indices[i + 1]
        key = f"{a}:{b}"
        e = g.edge_map.get(key)
        if e:
            d_acc += e.raw_dist
        else:
            d_acc += path_dist(g.nodes[a], g.nodes[b])
    return d_acc


def gather_ret_exit_pivot_indices_on_g_full(
    g_full: PathGraph, ret_end_px: Point, pivot_g1_px: Point, r_pts: Optional[List[Point]]
) -> List[int]:
    """designer.js gatherRetExitPivotIndicesOnGFull"""
    merge_rm = g_full.merge_radius_px
    px_pts: List[Point] = []
    if pivot_g1_px and len(pivot_g1_px) >= 2:
        px_pts.append(pivot_g1_px)
    if ret_end_px and len(ret_end_px) >= 2:
        px_pts.append(ret_end_px)
    if r_pts and len(r_pts) >= 2:
        px_pts.append(r_pts[-1])
        if len(r_pts) >= 3:
            px_pts.append(r_pts[-2])
    indices: List[int] = []
    seen: set = set()
    for p in px_pts:
        idx = g_full.nearest_path_node(p)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    r_near = merge_rm * 6
    r2 = r_near * r_near
    if ret_end_px and len(ret_end_px) >= 2 and g_full.nodes:
        scored = []
        for ni, node in enumerate(g_full.nodes):
            if _dist2(node, ret_end_px) <= r2:
                scored.append((ni, _dist2(node, ret_end_px)))
        scored.sort(key=lambda x: x[1])
        for k in range(min(36, len(scored))):
            ni = scored[k][0]
            if ni not in seen:
                seen.add(ni)
                indices.append(ni)
    return indices


def path_dijkstra_from_ret_exit_to_stand(
    g_full: PathGraph, end_node_full: int, candidate_start_indices: List[int]
) -> Tuple[Optional[List[int]], Optional[int]]:
    """designer.js pathDijkstraFromRetExitToStand"""
    if end_node_full is None or not candidate_start_indices:
        return None, None
    best_path = None
    best_d = float("inf")
    seen_start: set = set()
    for s in candidate_start_indices:
        if s is None or s in seen_start:
            continue
        seen_start.add(s)
        p = path_dijkstra(g_full, s, end_node_full)
        if not p or len(p) < 2:
            continue
        d = path_total_dist(g_full, p)
        if d >= g_full.reverse_cost:
            continue
        if d < best_d:
            best_d = d
            best_path = p
    if not best_path:
        return None, None
    return best_path, best_path[0]


def ret_split_path_indices_on_g_full(
    g1: PathGraph,
    g_full: PathGraph,
    p1: List[int],
    p2: List[int],
    pivot_idx: int,
    pivot_idx_full: Optional[int],
) -> Optional[List[int]]:
    """designer.js retSplitPathIndicesOnGFull"""
    if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
        return None
    p1_seg = p1 if pivot_idx == pivot_idx_full else p1[:-1]
    part1: List[int] = []
    for idx in p1_seg:
        if idx < 0 or idx >= len(g1.nodes):
            return None
        wp = g1.nodes[idx]
        ni = g_full.nearest_path_node(wp)
        if not part1 or part1[-1] != ni:
            part1.append(ni)
    p2_tail = p2[1:] if pivot_idx == pivot_idx_full else p2
    merged = part1 + p2_tail
    out: List[int] = []
    for idx in merged:
        if not out or out[-1] != idx:
            out.append(idx)
    return out if len(out) >= 2 else None


def nearest_path_node_on_runway_polyline(g: PathGraph, runway_id: str, runway_px: Point) -> int:
    rw_set = g.runway_node_indices_by_id.get(runway_id) or set()
    hit = g.nearest_path_node_from_set(rw_set, runway_px)
    if hit is not None:
        return hit
    return g.nearest_path_node(runway_px)


def build_path_graph(
    layout: dict,
    cell_size: float,
    reverse_cost: float,
    taxiway_heuristic_cost: float,
    merge_radius_px: float,
    rw_exit_allowed_default: List[str],
    direction_modes: List[dict],
    selected_arr_ret_id: Optional[str],
    runway_direction_for_exit: str,
    path_graph_opts: Optional[dict] = None,
) -> PathGraph:
    """
    designer.js buildPathGraph(selectedArrRetId, runwayDirectionForExit, pathGraphOpts).
    """
    opts = path_graph_opts if isinstance(path_graph_opts, dict) else {}
    pure_ground_exclude_runway = bool(opts.get("pureGroundExcludeRunway"))
    omit_other_runway_exits = bool(opts.get("omitOtherRunwayExits"))

    nodes: List[Point] = []
    adj: List[List[Tuple[int, float]]] = []
    edge_map: Dict[str, DirectedEdgeRecord] = {}
    node_bucket: Dict[str, List[int]] = {}
    merge_rm = max(float(merge_radius_px), 1e-6)
    runway_node_indices_by_id: Dict[str, set] = {}

    def node_bucket_key_for_point(p: Point) -> str:
        return f"{math.floor(p[0] / merge_rm)},{math.floor(p[1] / merge_rm)}"

    def find_node_index_within_merge_radius(p: Point) -> Optional[int]:
        bx = int(math.floor(p[0] / merge_rm))
        by = int(math.floor(p[1] / merge_rm))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                lst = node_bucket.get(f"{bx + dx},{by + dy}")
                if not lst:
                    continue
                for idx in lst:
                    if path_dist(nodes[idx], p) <= merge_rm:
                        return idx
        return None

    def get_or_add(p: Point) -> int:
        found = find_node_index_within_merge_radius(p)
        if found is not None:
            return found
        idx = len(nodes)
        nodes.append((float(p[0]), float(p[1])))
        adj.append([])
        bkey = node_bucket_key_for_point(p)
        node_bucket.setdefault(bkey, []).append(idx)
        return idx

    def register_directed_edge(rec: DirectedEdgeRecord) -> None:
        key = f"{rec.from_idx}:{rec.to_idx}"
        edge_map[key] = rec

    def add_edge_with_direction(
        p_from: Point,
        p_to: Point,
        dir_s: str,
        cost: float,
        raw_dist: float,
        pts_forward: List[Point],
        link_id: str,
        path_type: str,
    ) -> None:
        i = get_or_add(p_from)
        j = get_or_add(p_to)
        if i == j or cost < 1e-6:
            return
        fwd = dedupe_path_points(pts_forward if pts_forward else [p_from, p_to])
        rev = list(reversed(fwd))
        meta_ij = DirectedEdgeRecord(i, j, cost, raw_dist, fwd, link_id, path_type, dir_s)
        register_directed_edge(meta_ij)
        if dir_s == "both":
            adj[i].append((j, cost))
            adj[j].append((i, cost))
            register_directed_edge(
                DirectedEdgeRecord(j, i, cost, raw_dist, rev, link_id, path_type, dir_s)
            )
        elif dir_s == "counter_clockwise":
            adj[j].append((i, cost))
            adj[i].append((j, reverse_cost))
            register_directed_edge(
                DirectedEdgeRecord(i, j, reverse_cost, raw_dist, fwd, link_id, path_type, dir_s)
            )
        else:
            adj[i].append((j, cost))
            adj[j].append((i, reverse_cost))
            register_directed_edge(
                DirectedEdgeRecord(j, i, reverse_cost, raw_dist, rev, link_id, path_type, dir_s)
            )

    path_list = _layout_path_objects(layout)
    prepared: List[Tuple[dict, List[Point]]] = []
    for obj in path_list:
        if (
            omit_other_runway_exits
            and selected_arr_ret_id is not None
            and obj.get("pathType") == "runway_exit"
            and str(obj.get("id", "")) != str(selected_arr_ret_id)
        ):
            continue
        op = get_ordered_points(obj, layout, cell_size)
        if op and len(op) >= 2:
            prepared.append((obj, op))
    apron_node_stand: List[dict] = []
    min_d2 = 1e-6

    for obj, pts in prepared:
        tw_id = str(obj.get("id", "") or "")
        junctions: List[Tuple[float, Point]] = []
        for seg in range(len(pts) - 1):
            a, b = pts[seg], pts[seg + 1]
            for other, other_ord in prepared:
                if other.get("id") == obj.get("id"):
                    continue
                if not other_ord or len(other_ord) < 2:
                    continue
                for oseg in range(len(other_ord) - 1):
                    c, d = other_ord[oseg], other_ord[oseg + 1]
                    isec = segment_segment_intersection(a, b, c, d)
                    if isec:
                        t, pr = project_on_segment(a, b, isec)
                        junctions.append((seg + t, pr))
                    else:
                        ov = collinear_segment_overlap_on_ab(a, b, c, d)
                        if ov:
                            ax, ay = a[0], a[1]
                            bx, by = b[0], b[1]
                            dx_, dy_ = bx - ax, by - ay
                            p0 = (ax + ov[0] * dx_, ay + ov[0] * dy_)
                            p1ov = (ax + ov[1] * dx_, ay + ov[1] * dy_)
                            pr0 = project_on_segment(a, b, p0)
                            junctions.append((seg + pr0[0], pr0[1]))
                            if _dist2(p0, p1ov) > SPLIT_TOL_D2:
                                pr1 = project_on_segment(a, b, p1ov)
                                junctions.append((seg + pr1[0], pr1[1]))
                        else:
                            for q in (c, d):
                                if _dist2(a, q) <= SPLIT_TOL_D2 or _dist2(b, q) <= SPLIT_TOL_D2:
                                    t, proj = project_on_segment(a, b, q)
                                    if 0 <= t <= 1:
                                        junctions.append((seg + t, proj))
                for q in other_ord:
                    if not point_on_segment_strict(a, b, q):
                        continue
                    t, proj = project_on_segment(a, b, q)
                    junctions.append((seg + t, proj))
            if obj.get("pathType") != "runway":
                for lk in layout.get("apronLinks") or []:
                    if not isinstance(lk, dict):
                        continue
                    if str(lk.get("taxiwayId", "")) != tw_id:
                        continue
                    if lk.get("tx") is None or lk.get("ty") is None:
                        continue
                    link_pt = (float(lk["tx"]), float(lk["ty"]))
                    t, p = project_on_segment(a, b, link_pt)
                    if 0 <= t <= 1 and _dist2(p, link_pt) <= SPLIT_TOL_D2:
                        junctions.append((seg + t, p))
                        pbb = find_stand_by_id(layout, str(lk.get("pbbId", "")))
                        if pbb:
                            stand_pt = get_stand_connection_px(pbb, cell_size)
                            mids: List[Point] = []
                            for mv in lk.get("midVertices") or []:
                                if isinstance(mv, dict):
                                    mids.append(_vertex_to_px(mv, cell_size))
                            chain: List[Point] = [stand_pt] + mids + [p]
                            apron_node_stand.append(
                                {"nodeP": p, "standPt": stand_pt, "standId": lk.get("pbbId"), "chain": chain}
                            )
        if obj.get("pathType") == "runway":
            ldm = get_effective_runway_lineup_dist_m(obj)
            rpath = get_runway_path_px(layout, cell_size, obj.get("id"))
            if rpath and len(rpath["pts"]) >= 2 and ldm > 1e-6:
                r_pts = rpath["pts"]
                total = sum(path_dist(r_pts[i], r_pts[i + 1]) for i in range(len(r_pts) - 1))
                d = min(ldm, total)
                if d > 1e-6:
                    acc = 0.0
                    for ri in range(len(r_pts) - 1):
                        p1, p2 = r_pts[ri], r_pts[ri + 1]
                        seg_len = path_dist(p1, p2)
                        if seg_len < 1e-9:
                            continue
                        if acc + seg_len >= d - 1e-6:
                            t = max(0.0, min(1.0, (d - acc) / seg_len))
                            px = p1[0] + t * (p2[0] - p1[0])
                            py = p1[1] + t * (p2[1] - p1[1])
                            junctions.append((ri + t, (px, py)))
                            break
                        acc += seg_len
        waypoints: List[Tuple[float, Point, bool]] = [
            (0.0, pts[0], False),
            (float(len(pts) - 1), pts[-1], False),
        ]
        for t_along, p in junctions:
            waypoints.append((t_along, p, True))
        waypoints.sort(key=lambda x: x[0])
        chain: List[Tuple[float, Point, bool]] = []
        for t_along, p, is_j in waypoints:
            if (
                chain
                and abs(t_along - chain[-1][0]) < 1e-9
                and _dist2(p, chain[-1][1]) < min_d2
            ):
                continue
            chain.append((t_along, p, is_j))
        if obj.get("pathType") == "runway":
            rw_id = str(obj.get("id", ""))
            st = runway_node_indices_by_id.setdefault(rw_id, set())
            for t_along, p, _is_j in chain:
                st.add(get_or_add(p))
        dir_s = get_taxiway_direction(obj, direction_modes)
        is_runway_exit = obj.get("pathType") == "runway_exit"
        is_taxiway = obj.get("pathType") == "taxiway"
        path_type = str(obj.get("pathType", "taxiway") or "taxiway")
        for i in range(len(chain) - 1):
            seg_pts = polyline_points_between_along(pts, chain[i][0], chain[i + 1][0])
            dlen = polyline_distance_between_along(pts, chain[i][0], chain[i + 1][0])
            cost = dlen
            if is_runway_exit and not is_runway_exit_direction_allowed(
                obj, runway_direction_for_exit, rw_exit_allowed_default
            ):
                cost = reverse_cost
            if selected_arr_ret_id is not None and is_taxiway:
                cost = dlen + taxiway_heuristic_cost
            if pure_ground_exclude_runway and obj.get("pathType") == "runway":
                cost = reverse_cost
            add_edge_with_direction(
                chain[i][1],
                chain[i + 1][1],
                dir_s,
                cost,
                dlen,
                seg_pts,
                tw_id,
                path_type,
            )

    stand_id_to_node_index: Dict[str, int] = {}
    for entry in apron_node_stand:
        node_p = entry["nodeP"]
        stand_pt = entry["standPt"]
        stand_id = entry.get("standId")
        i = get_or_add(node_p)
        j = get_or_add(stand_pt)
        if stand_id is not None:
            stand_id_to_node_index[str(stand_id)] = j
        chain_pts = entry.get("chain") or []
        d_pts = dedupe_path_points(chain_pts if len(chain_pts) >= 2 else [node_p, stand_pt])
        if len(d_pts) < 2 or i == j:
            continue
        total_dist = sum(path_dist(d_pts[k], d_pts[k + 1]) for k in range(len(d_pts) - 1))
        if total_dist <= 1e-6:
            continue
        lid = "apron_link"
        adj[i].append((j, total_dist))
        adj[j].append((i, total_dist))
        register_directed_edge(
            DirectedEdgeRecord(
                i, j, total_dist, total_dist, list(reversed(d_pts)), lid, "apron_link", "both"
            )
        )
        register_directed_edge(
            DirectedEdgeRecord(
                j, i, total_dist, total_dist, d_pts, lid, "apron_link", "both"
            )
        )

    g = PathGraph(
        nodes=nodes,
        adj=adj,
        edge_map=edge_map,
        runway_node_indices_by_id=runway_node_indices_by_id,
        stand_id_to_node_index=stand_id_to_node_index,
        merge_radius_px=merge_rm,
        reverse_cost=reverse_cost,
    )
    return g


def path_indices_to_edge_segments(g: PathGraph, path_indices: List[int]) -> List[DirectedEdgeRecord]:
    """Ordered DirectedEdgeRecords along path (for sim edge_list + motion)."""
    out: List[DirectedEdgeRecord] = []
    for i in range(len(path_indices) - 1):
        a, b = path_indices[i], path_indices[i + 1]
        key = f"{a}:{b}"
        rec = g.edge_map.get(key)
        if rec is None:
            rd = path_dist(g.nodes[a], g.nodes[b])
            rec = DirectedEdgeRecord(
                a,
                b,
                max(rd, 0.01),
                rd,
                [g.nodes[a], g.nodes[b]],
                "_gap",
                "taxiway",
                "both",
            )
        out.append(rec)
    return out


def _stand_end_node_index(g: PathGraph, layout: dict, apron_id: str, cell_size: float) -> Optional[int]:
    """Apron graph end: standIdToNodeIndex, or nearest node to stand connection px if no apron link."""
    j = g.stand_id_to_node_index.get(str(apron_id))
    if j is not None:
        return j
    st = find_stand_by_id(layout, str(apron_id))
    if not st:
        return None
    sp = get_stand_connection_px(st, cell_size)
    if not g.nodes:
        return None
    return g.nearest_path_node(sp)


def _norm_sim_edge_path_dir(raw: Any) -> str:
    """Missing pathDir → clockwise (``from``→``to`` cheap, reverse uses ``reverse_cost``)."""
    if raw is None:
        return "clockwise"
    s = str(raw).strip().lower()
    if not s:
        return "clockwise"
    if s in ("clockwise", "cw"):
        return "clockwise"
    if s in ("counter_clockwise", "ccw", "counter-clockwise"):
        return "counter_clockwise"
    if s == "both":
        return "both"
    return "clockwise"


def path_graph_from_layout_sim_export(
    layout: dict,
    rw_dir: str,
    *,
    pure_ground_exclude_runway: bool,
    reverse_cost: float,
    merge_radius_px: float,
    taxiway_heuristic_bonus: float,
    apply_taxiway_ret_heuristic: bool,
) -> Optional[PathGraph]:
    """
    Assemble PathGraph from Layout_Design serializeCurrentLayout().simPathGraph (designer.js
    buildSimPathGraphExport). Same junction-split edges and weights as the HTML designer when
    Information pathSearch matches the layout export; falls back to build_path_graph if missing.
    """
    raw = layout.get("simPathGraph")
    if not isinstance(raw, dict) or int(raw.get("version", 0) or 0) != 1:
        return None
    nd = normalize_rw_direction_value(rw_dir)
    branch = raw.get("counter_clockwise") if nd == "counter_clockwise" else raw.get("clockwise")
    if not isinstance(branch, dict):
        return None
    inner_key = "pureGroundExcludeRunway" if pure_ground_exclude_runway else "standard"
    inner = branch.get(inner_key)
    if not isinstance(inner, dict):
        return None
    nodes_raw = inner.get("nodes")
    edges_raw = inner.get("edges")
    if not isinstance(nodes_raw, list) or len(nodes_raw) < 1:
        return None
    if not isinstance(edges_raw, list):
        return None

    nodes: List[Point] = []
    for p in nodes_raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            nodes.append((float(p[0]), float(p[1])))
        else:
            nodes.append((0.0, 0.0))

    n = len(nodes)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    edge_map: Dict[str, DirectedEdgeRecord] = {}
    rc = max(float(reverse_cost), 1.0)
    th = max(0.0, float(taxiway_heuristic_bonus))

    def register_sim_arc(
        u: int,
        v: int,
        c: float,
        raw_d: float,
        pts_uv: List[Point],
        lid: str,
        ptype: str,
        pdir: str,
        *,
        reverse_penalty: bool,
    ) -> None:
        if c < 1e-6:
            return
        if not reverse_penalty and (c >= rc * 0.999 or c < 1e-6):
            return
        adj[u].append((v, c))
        edge_map[f"{u}:{v}"] = DirectedEdgeRecord(
            from_idx=u,
            to_idx=v,
            cost=c,
            raw_dist=raw_d,
            pts=dedupe_path_points(pts_uv),
            link_id=lid,
            path_type=ptype,
            direction=pdir,
        )

    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        try:
            a = int(e["from"])
            b = int(e["to"])
        except (KeyError, TypeError, ValueError):
            continue
        if a < 0 or b < 0 or a >= n or b >= n:
            continue
        base_cost = float(e.get("dist", 0))
        raw_d = float(e.get("rawDist", base_cost))
        if apply_taxiway_ret_heuristic and str(e.get("pathType") or "") == "taxiway":
            base_cost += th
        if base_cost >= rc * 0.999 or base_cost < 1e-6:
            continue
        pts_raw = e.get("pts")
        pts: List[Point] = []
        if isinstance(pts_raw, list):
            for p in pts_raw:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        if len(pts) < 2:
            pts = [nodes[a], nodes[b]]
        link_id = str(e.get("linkId") or "")
        path_type = str(e.get("pathType") or "taxiway")
        path_dir_raw = e.get("pathDir")
        pd = _norm_sim_edge_path_dir(path_dir_raw)
        fwd_pts = dedupe_path_points(pts)
        rev_pts = dedupe_path_points(list(reversed(pts))) if len(pts) >= 2 else fwd_pts

        if pd == "both":
            register_sim_arc(a, b, base_cost, raw_d, fwd_pts, link_id, path_type, pd, reverse_penalty=False)
            register_sim_arc(b, a, base_cost, raw_d, rev_pts, link_id, path_type, pd, reverse_penalty=False)
        elif pd == "counter_clockwise":
            register_sim_arc(b, a, base_cost, raw_d, rev_pts, link_id, path_type, pd, reverse_penalty=False)
            register_sim_arc(a, b, rc, raw_d, fwd_pts, link_id, path_type, pd, reverse_penalty=True)
        else:
            register_sim_arc(a, b, base_cost, raw_d, fwd_pts, link_id, path_type, pd, reverse_penalty=False)
            register_sim_arc(b, a, rc, raw_d, rev_pts, link_id, path_type, pd, reverse_penalty=True)

    stand_raw = inner.get("standIdToNodeIndex") or {}
    stand_id_to_node_index: Dict[str, int] = {}
    if isinstance(stand_raw, dict):
        for k, v in stand_raw.items():
            try:
                stand_id_to_node_index[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

    runway_raw = inner.get("runwayNodeIndicesById") or {}
    runway_node_indices_by_id: Dict[str, set] = {}
    if isinstance(runway_raw, dict):
        for rwid, arr in runway_raw.items():
            if not isinstance(arr, list):
                continue
            s: set = set()
            for x in arr:
                try:
                    xi = int(x)
                    if 0 <= xi < n:
                        s.add(xi)
                except (TypeError, ValueError):
                    continue
            runway_node_indices_by_id[str(rwid)] = s

    merge_r = float(merge_radius_px) if math.isfinite(float(merge_radius_px)) else 7.0
    return PathGraph(
        nodes=nodes,
        adj=adj,
        edge_map=edge_map,
        runway_node_indices_by_id=runway_node_indices_by_id,
        stand_id_to_node_index=stand_id_to_node_index,
        merge_radius_px=max(merge_r, 1e-6),
        reverse_cost=rc,
    )


def solve_arrival_path_indices(
    layout: dict,
    information: dict,
    runway_id: str,
    apron_id: str,
    schedule_arr_ret_id: Optional[str],
    sampled_arr_ret: Optional[str],
    arr_runway_dir_used: str,
) -> Tuple[Optional[PathGraph], Optional[List[int]]]:
    """
    designer.js arrival path solve (solveByRunwayDir + tryCw/tryCcw), simplified:
    uses flight's arr_runway_dir_used when cw/ccw; otherwise tries both and picks lower totalD.
    """
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    algo = (
        information.get("tiers", {})
        .get("algorithm", {})
        if isinstance(information, dict)
        else {}
    )
    path_cfg = algo.get("pathSearch", {}) if isinstance(algo, dict) else {}
    reverse_cost = float(path_cfg.get("reverseCost", 1_000_000) or 1_000_000)
    th = path_cfg.get("taxiwayHeuristicCost")
    if th is not None and math.isfinite(float(th)) and float(th) == 0.0:
        taxiway_heuristic_cost = 0.0
    elif th is not None and float(th) > 0:
        taxiway_heuristic_cost = float(th)
    else:
        taxiway_heuristic_cost = 200.0
    merge_r = float(path_cfg.get("junctionMergeRadiusPx", 7.0) or 7.0)
    flight_sched = information.get("tiers", {}).get("flight_schedule", {}) if isinstance(information, dict) else {}
    allow_rw_ground = bool(flight_sched.get("defaultAllowRunwayInGroundSegment", False))
    rw_exit_raw = flight_sched.get("rwExitAllowedDefaultRaw")
    rw_exit_default = normalize_allowed_runway_directions(rw_exit_raw)
    direction_modes = layout.get("directionModes") or []
    if not isinstance(direction_modes, list):
        direction_modes = []

    path_list = _layout_path_objects(layout)
    sched_trim = (str(schedule_arr_ret_id).strip() if schedule_arr_ret_id else "") or ""
    selected = sampled_arr_ret
    if sched_trim and any(
        t.get("id") == sched_trim and t.get("pathType") == "runway_exit" for t in path_list
    ):
        selected = sched_trim
    valid_ret = None
    if selected is not None and any(
        t.get("id") == selected and t.get("pathType") == "runway_exit" for t in path_list
    ):
        valid_ret = str(selected)

    r = get_runway_path_px(layout, cell_size, runway_id)
    if not r:
        return None, None

    def solve_by_runway_dir(rw_dir: str) -> Tuple[Optional[PathGraph], Optional[List[int]], float]:
        runway_px = r["endPx"] if rw_dir == "counter_clockwise" else r["startPx"]
        exclude_ground = not allow_rw_ground
        g_full_opts = {"pureGroundExcludeRunway": exclude_ground}
        ret_heuristic = valid_ret is not None
        g_full = path_graph_from_layout_sim_export(
            layout,
            rw_dir,
            pure_ground_exclude_runway=exclude_ground,
            reverse_cost=reverse_cost,
            merge_radius_px=merge_r,
            taxiway_heuristic_bonus=taxiway_heuristic_cost,
            apply_taxiway_ret_heuristic=ret_heuristic,
        )
        if g_full is None:
            g_full = build_path_graph(
                layout,
                cell_size,
                reverse_cost,
                taxiway_heuristic_cost,
                merge_r,
                rw_exit_default,
                direction_modes,
                None,
                rw_dir,
                g_full_opts,
            )
        end_full = _stand_end_node_index(g_full, layout, str(apron_id), cell_size)
        if end_full is None:
            return None, None, float("inf")

        if valid_ret is not None:
            ret_tw = next(
                (t for t in path_list if t.get("id") == valid_ret and t.get("pathType") == "runway_exit"),
                None,
            )
            r_pts = get_ordered_points(ret_tw, layout, cell_size) if ret_tw else None
            if r_pts and len(r_pts) >= 2:
                ret_end_px = r_pts[-1]
                g1 = build_path_graph(
                    layout,
                    cell_size,
                    reverse_cost,
                    taxiway_heuristic_cost,
                    merge_r,
                    rw_exit_default,
                    direction_modes,
                    valid_ret,
                    rw_dir,
                    {"omitOtherRunwayExits": True},
                )
                start_node = nearest_path_node_on_runway_polyline(g1, runway_id, runway_px)
                pivot_idx = g1.nearest_path_node(ret_end_px)
                pivot_px_g1 = g1.nodes[pivot_idx]
                p1 = path_dijkstra(g1, start_node, pivot_idx)
                p2: Optional[List[int]] = None
                pivot_idx_full: Optional[int] = None
                if p1:
                    cand = gather_ret_exit_pivot_indices_on_g_full(g_full, ret_end_px, pivot_px_g1, r_pts)
                    p2, pivot_idx_full = path_dijkstra_from_ret_exit_to_stand(g_full, end_full, cand)
                if p1 and len(p1) >= 2 and p2 and len(p2) >= 2:
                    merged = ret_split_path_indices_on_g_full(
                        g1, g_full, p1, p2, pivot_idx, pivot_idx_full
                    )
                    d = path_total_dist(g1, p1) + path_total_dist(g_full, p2)
                    if merged and len(merged) >= 2 and d < reverse_cost:
                        return g_full, merged, d

        g = path_graph_from_layout_sim_export(
            layout,
            rw_dir,
            pure_ground_exclude_runway=False,
            reverse_cost=reverse_cost,
            merge_radius_px=merge_r,
            taxiway_heuristic_bonus=taxiway_heuristic_cost,
            apply_taxiway_ret_heuristic=ret_heuristic,
        )
        if g is None:
            g = build_path_graph(
                layout,
                cell_size,
                reverse_cost,
                taxiway_heuristic_cost,
                merge_r,
                rw_exit_default,
                direction_modes,
                valid_ret,
                rw_dir,
                None,
            )
        end_node = _stand_end_node_index(g, layout, str(apron_id), cell_size)
        if end_node is None:
            return None, None, float("inf")
        start_node = nearest_path_node_on_runway_polyline(g, runway_id, runway_px)
        p = path_dijkstra(g, start_node, end_node)
        if not p or len(p) < 2:
            return None, None, float("inf")
        d = path_total_dist(g, p)
        if d >= reverse_cost:
            return None, None, float("inf")
        return g, p, d

    nd = normalize_rw_direction_value(arr_runway_dir_used)
    if nd == "clockwise":
        g0, p0, _ = solve_by_runway_dir("clockwise")
        return g0, p0
    if nd == "counter_clockwise":
        g0, p0, _ = solve_by_runway_dir("counter_clockwise")
        return g0, p0

    g_cw, p_cw, d_cw = solve_by_runway_dir("clockwise")
    g_ccw, p_ccw, d_ccw = solve_by_runway_dir("counter_clockwise")
    if valid_ret is not None:
        # Prefer split when only one side has split — designer.js uses usedRetSplit flags; we approximate by path length / same graph
        pass
    if g_ccw and p_ccw and (not g_cw or not p_cw or d_ccw < d_cw):
        return g_ccw, p_ccw
    return g_cw, p_cw


def plan_taxi_route(
    layout: dict,
    information: dict,
    origin: Point,
    dest: Point,
    runway_direction_for_exit: str,
) -> Tuple[List[DirectedEdgeRecord], Optional[PathGraph]]:
    """Point-to-point taxi on full designer graph (e.g. pushback → runway threshold)."""
    cell_size = float(layout.get("grid", {}).get("cellSize", 20.0))
    algo = information.get("tiers", {}).get("algorithm", {}) if isinstance(information, dict) else {}
    path_cfg = algo.get("pathSearch", {}) if isinstance(algo, dict) else {}
    reverse_cost = float(path_cfg.get("reverseCost", 1_000_000) or 1_000_000)
    th = path_cfg.get("taxiwayHeuristicCost")
    if th is not None and math.isfinite(float(th)) and float(th) == 0.0:
        taxiway_heuristic_cost = 0.0
    elif th is not None and float(th) > 0:
        taxiway_heuristic_cost = float(th)
    else:
        taxiway_heuristic_cost = 200.0
    merge_r = float(path_cfg.get("junctionMergeRadiusPx", 7.0) or 7.0)
    flight_sched = information.get("tiers", {}).get("flight_schedule", {}) if isinstance(information, dict) else {}
    rw_exit_default = normalize_allowed_runway_directions(flight_sched.get("rwExitAllowedDefaultRaw"))
    direction_modes = layout.get("directionModes") or []
    if not isinstance(direction_modes, list):
        direction_modes = []
    g = build_path_graph(
        layout,
        cell_size,
        reverse_cost,
        taxiway_heuristic_cost,
        merge_r,
        rw_exit_default,
        direction_modes,
        None,
        runway_direction_for_exit,
        None,
    )
    if not g.nodes:
        return [], g
    o_idx = g.nearest_path_node(origin)
    d_idx = g.nearest_path_node(dest)
    path = path_dijkstra(g, o_idx, d_idx)
    if not path:
        return [], g
    return path_indices_to_edge_segments(g, path), g


def motion_span_for_record(g: PathGraph, rec: DirectedEdgeRecord) -> Tuple[Point, Point]:
    if rec.pts and len(rec.pts) >= 2:
        return rec.pts[0], rec.pts[-1]
    return g.nodes[rec.from_idx], g.nodes[rec.to_idx]


def polyline_apron_junctions_xy_for_sim_result(g: PathGraph) -> List[dict]:
    """Degree>=2 nodes (valid junctions), designer.js validJunctionsForDraw analogue."""
    out: List[dict] = []
    for i, p in enumerate(g.nodes):
        if i < len(g.adj) and len(g.adj[i]) >= 2:
            out.append({"x": round(float(p[0]), 3), "y": round(float(p[1]), 3)})
    return out
