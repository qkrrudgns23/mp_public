"""
Airside Simulation Engine.

layout_design.py의 Run Simulation에서 호출된다.
입력 layout JSON을 받아 동일한 layout 구조에 timeline을 주입한 dict를 반환한다.
반환값은 JS의 applyLayoutObject(result)로 바로 주입된다.

- Runway: start → end 방향만 허용 (역방향 불가).
- Taxiway: start → end 방향만 허용.
- 최소 거리 경로 탐색(Dijkstra)으로 runway start ↔ apron 간 정방향만 사용.
- 경로가 없으면 noWay=True, timeline은 빈 배열 또는 초기점만 두고 "No way" 경고 유지.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[float, float]
_EPS = 1.0  # 픽셀 단위 동일점 판정


@dataclass
class LayoutContext:
    cell_size: float
    taxiways: Dict[str, Dict[str, Any]]
    runways: Dict[str, Dict[str, Any]]
    pbb_stands: Dict[str, Dict[str, Any]]
    remote_stands: Dict[str, Dict[str, Any]]
    apron_links: List[Dict[str, Any]]
    direction_modes: Dict[str, Dict[str, Any]]


def _cell_to_pixel(col: float, row: float, cell_size: float) -> Point:
    return float(col) * cell_size, float(row) * cell_size


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _dist(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_key(p: Point) -> Tuple[float, float]:
    """그래프 노드 병합용 키 (소수점 1자리)."""
    return (round(p[0], 1), round(p[1], 1))


# ---------------------------------------------------------------------------
# Runway / Taxiway: start → end 정방향만 사용
# ---------------------------------------------------------------------------


def _get_runway_path(ctx: LayoutContext, runway_id: Optional[str]) -> Optional[Tuple[Point, Point, List[Point]]]:
    """
    Runway polyline을 start → end 순서의 점 목록으로 반환.
    반환: (start_px, end_px, [start, ..., end]) 또는 None.
    """
    runway_path = None
    if runway_id and runway_id in ctx.taxiways:
        tw = ctx.taxiways[runway_id]
        if (tw.get("pathType") == "runway" or "runway" in (tw.get("name") or "").lower()) and tw.get("vertices"):
            runway_path = tw
    if not runway_path:
        runway_path = next(
            (
                tw
                for tw in ctx.taxiways.values()
                if tw.get("pathType") == "runway" and tw.get("vertices")
            ),
            None,
        )
    if not runway_path or not runway_path.get("vertices"):
        runway_path = next(
            (
                tw
                for tw in ctx.taxiways.values()
                if "runway" in (tw.get("name") or "").lower() and tw.get("vertices")
            ),
            None,
        )
    if not runway_path or not runway_path.get("vertices"):
        return None

    verts = runway_path["vertices"]
    pts = [_cell_to_pixel(v.get("col", 0), v.get("row", 0), ctx.cell_size) for v in verts]
    if len(pts) < 2:
        return None

    sp = runway_path.get("start_point")
    ep = runway_path.get("end_point")
    if sp is not None and ep is not None:
        start_px = _cell_to_pixel(float(sp.get("col", 0)), float(sp.get("row", 0)), ctx.cell_size)
        end_px = _cell_to_pixel(float(ep.get("col", 0)), float(ep.get("row", 0)), ctx.cell_size)
        # vertices 순서가 start→end인지 확인
        d_start_first = _dist2(pts[0], start_px)
        d_start_last = _dist2(pts[-1], start_px)
        if d_start_last < d_start_first:
            pts = list(reversed(pts))
    else:
        start_px = pts[0]
        end_px = pts[-1]
    return (start_px, end_px, pts)


def _get_taxiway_ordered_points(tw: Dict[str, Any], cell_size: float) -> Optional[List[Point]]:
    """
    Taxiway를 start → end 정방향 점 목록으로 반환.
    """
    verts = tw.get("vertices") or []
    if len(verts) < 2:
        return None
    pts = [_cell_to_pixel(v.get("col", 0), v.get("row", 0), cell_size) for v in verts]
    sp = tw.get("start_point")
    ep = tw.get("end_point")
    if sp is not None and ep is not None:
        start_px = _cell_to_pixel(float(sp.get("col", 0)), float(sp.get("row", 0)), cell_size)
        if _dist2(pts[-1], start_px) < _dist2(pts[0], start_px):
            pts = list(reversed(pts))
    return pts


def _get_taxiway_direction(tw: Dict[str, Any], dm_by_id: Dict[str, Dict[str, Any]]) -> str:
    dm_id = tw.get("directionModeId")
    if dm_id and dm_id in dm_by_id:
        d = (dm_by_id[dm_id].get("direction") or "both").lower()
        if d in {"clockwise", "counter_clockwise", "both"}:
            return d
    d = (tw.get("direction") or "both").lower()
    if d not in {"clockwise", "counter_clockwise"}:
        return "both"
    return d


def _build_taxiway_path_pixels(
    tw: Dict[str, Any],
    from_pt: Point,
    to_pt: Point,
    cell_size: float,
    dm_by_id: Dict[str, Dict[str, Any]],
) -> Optional[List[Point]]:
    """
    한 taxiway polyline 위에서 from_pt → to_pt 경로를 vertices 기준으로 생성.
    정방향(start→end)만 허용; 역방향이면 None.
    """
    verts = tw.get("vertices") or []
    if len(verts) < 2:
        return None

    pts: List[Point] = [
        _cell_to_pixel(v.get("col", 0), v.get("row", 0), cell_size) for v in verts
    ]
    ordered = _get_taxiway_ordered_points(tw, cell_size)
    if not ordered:
        return None

    src_idx = min(range(len(ordered)), key=lambda i: _dist2(ordered[i], from_pt))
    dst_idx = min(range(len(ordered)), key=lambda i: _dist2(ordered[i], to_pt))
    if src_idx > dst_idx:
        return None  # 역방향 불가
    seq: List[Point] = [from_pt] + ordered[src_idx : dst_idx + 1] + [to_pt]
    return seq


def _find_runway_touch_point(ctx: LayoutContext, runway_id: Optional[str]) -> Optional[Point]:
    """
    JS findRunwayTouchPoint 포트.
    - runways 배열에 있으면 그 객체의 중심(cx, cy)을 사용.
    - 아니면 taxiways 중 pathType='runway' 또는 이름에 'runway'가 포함된 polyline의 중간점을 사용.
    """
    if ctx.runways:
        rw = ctx.runways.get(runway_id) if runway_id else next(iter(ctx.runways.values()), None)
        if rw is not None:
            cx = float(rw.get("cx", 0.0))
            cy = float(rw.get("cy", 0.0))
            return cx, cy

    taxiways_to_use = list(ctx.taxiways.values())
    runway_path: Optional[Dict[str, Any]]
    if runway_id:
        runway_path = next(
            (tw for tw in taxiways_to_use if tw.get("id") == runway_id and tw.get("vertices")), None
        )
    else:
        runway_path = next(
            (
                tw
                for tw in taxiways_to_use
                if tw.get("pathType") == "runway" and tw.get("vertices")
            ),
            None,
        )
        if runway_path is None:
            runway_path = next(
                (
                    tw
                    for tw in taxiways_to_use
                    if "runway" in (tw.get("name") or "").lower() and tw.get("vertices")
                ),
                None,
            )

    if runway_path and runway_path.get("vertices"):
        verts = runway_path["vertices"]
        pts = [
            _cell_to_pixel(v.get("col", 0), v.get("row", 0), ctx.cell_size) for v in verts
        ]
        total_len = 0.0
        for i in range(1, len(pts)):
            total_len += hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if total_len > 0:
            target = total_len / 2.0
            acc = 0.0
            for i in range(1, len(pts)):
                x1, y1 = pts[i - 1]
                x2, y2 = pts[i]
                seg_len = hypot(x2 - x1, y2 - y1)
                if acc + seg_len >= target:
                    t = (target - acc) / (seg_len or 1.0)
                    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                acc += seg_len
        # fallback: 첫 점
        return pts[0]

    return None


# ---------------------------------------------------------------------------
# 방향 보존 그래프: runway start→end, taxiway start→end 만 엣지로 사용
# ---------------------------------------------------------------------------


@dataclass
class _PathGraph:
    """정방향만 허용하는 노드/엣지 그래프. 노드 = (x,y) 픽셀, 키는 _point_key로 통일."""
    nodes: List[Point] = field(default_factory=list)
    key_to_idx: Dict[Tuple[float, float], int] = field(default_factory=dict)
    edges: List[Tuple[int, int, float]] = field(default_factory=list)  # (from, to, dist)
    stand_id_to_node_index: Dict[str, int] = field(default_factory=dict)  # apron link로 연결된 주기장만

    def get_or_add(self, p: Point) -> int:
        k = _point_key(p)
        if k in self.key_to_idx:
            return self.key_to_idx[k]
        idx = len(self.nodes)
        self.nodes.append(p)
        self.key_to_idx[k] = idx
        return idx

    def add_edge_if_near(self, p_from: Point, p_to: Point, max_merge: float = _EPS) -> None:
        i = self.get_or_add(p_from)
        j = self.get_or_add(p_to)
        if i == j:
            return
        d = _dist(self.nodes[i], self.nodes[j])
        if d < 1e-6:
            return
        self.edges.append((i, j, d))


def _project_on_segment(a: Point, b: Point, q: Point) -> Tuple[float, Point]:
    """점 q를 선분 a-b에 투영한 t(0~1)와 투영점."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    qx, qy = q[0], q[1]
    dx = bx - ax
    dy = by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return (0.0, a)
    t = ((qx - ax) * dx + (qy - ay) * dy) / den
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return (t, (px, py))


def _point_on_segment(a: Point, b: Point, q: Point, tol: float = 2.0) -> bool:
    """q가 선분 a-b 위에 있는지 (거리 기준)."""
    _, proj = _project_on_segment(a, b, q)
    return _dist2(proj, q) <= tol * tol


def _build_path_graph(ctx: LayoutContext) -> _PathGraph:
    """
    Runway( start→end ), Taxiway( start→end ), Apron link (tx,ty)→stand, stand→(tx,ty) 로
    방향 보존 그래프 구성. Runway와 taxiway 접점에서 runway를 분할해 연결.
    """
    g = _PathGraph()

    # Runway: start → end. Runway–Taxiway 교차점은 그래프에 넣지 않음(다익스트라에서 바이패스).
    for tw in ctx.taxiways.values():
        if tw.get("pathType") != "runway" and "runway" not in (tw.get("name") or "").lower():
            continue
        r = _get_runway_path(ctx, tw.get("id"))
        if not r:
            continue
        _start, _end, pts = r
        # Taxiway가 runway 선분을 교차해도 junction으로 수집하지 않음 → runway는 원래 꼭짓점만 사용
        run_pts: List[Point] = list(pts)
        for i in range(len(run_pts) - 1):
            g.add_edge_if_near(run_pts[i], run_pts[i + 1])

    # Taxiway: start → end 만 (역방향 엣지 없음)
    for tw in ctx.taxiways.values():
        if tw.get("pathType") == "runway":
            continue
        ordered = _get_taxiway_ordered_points(tw, ctx.cell_size)
        if not ordered or len(ordered) < 2:
            continue
        for i in range(len(ordered) - 1):
            g.add_edge_if_near(ordered[i], ordered[i + 1])

    # Apron link 점 (tx,ty)를 해당 taxiway 세그먼트에 연결
    for lk in ctx.apron_links:
        tx_val = lk.get("tx")
        ty_val = lk.get("ty")
        if tx_val is None or ty_val is None:
            continue
        link_pt: Point = (float(tx_val), float(ty_val))
        tid = lk.get("taxiwayId")
        tw = ctx.taxiways.get(tid or "") if tid else None
        if tw:
            ordered = _get_taxiway_ordered_points(tw, ctx.cell_size)
            if ordered and len(ordered) >= 2:
                best_i = -1
                best_d2 = 4.0 * ctx.cell_size * ctx.cell_size
                for i in range(len(ordered) - 1):
                    _, proj = _project_on_segment(ordered[i], ordered[i + 1], link_pt)
                    d2 = _dist2(proj, link_pt)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_i = i
                if best_i >= 0:
                    g.add_edge_if_near(ordered[best_i], link_pt)
                    g.add_edge_if_near(link_pt, ordered[best_i + 1])

        # (tx,ty) ↔ stand (도착: runway→stand, 출발: stand→runway)
        pbb_id = lk.get("pbbId")
        stand = ctx.pbb_stands.get(pbb_id or "") or ctx.remote_stands.get(pbb_id or "")
        if stand:
            if "x2" in stand and "y2" in stand:
                stand_pt: Point = (float(stand["x2"]), float(stand["y2"]))
            else:
                stand_pt = _cell_to_pixel(
                    float(stand.get("col", 0)), float(stand.get("row", 0)), ctx.cell_size
                )
            g.add_edge_if_near(link_pt, stand_pt)
            g.add_edge_if_near(stand_pt, link_pt)
            if pbb_id:
                g.stand_id_to_node_index[pbb_id] = g.get_or_add(stand_pt)

    return g


def _dijkstra(
    g: _PathGraph,
    start_idx: int,
    end_idx: int,
) -> Optional[List[int]]:
    """최소 거리 경로의 노드 인덱스 리스트. 없으면 None."""
    n = len(g.nodes)
    dist = [1e99] * n
    dist[start_idx] = 0.0
    prev: List[Optional[int]] = [None] * n
    heap: List[Tuple[float, int]] = [(0.0, start_idx)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end_idx:
            path: List[int] = []
            cur = end_idx
            while cur is not None:
                path.append(cur)
                cur = prev[cur]  # type: ignore
            path.reverse()
            return path
        for (v, w, edge_len) in g.edges:
            if v != u:
                continue
            nd = dist[u] + edge_len
            if nd < dist[w]:
                dist[w] = nd
                prev[w] = u
                heapq.heappush(heap, (nd, w))
    return None


def _nearest_node_index(g: _PathGraph, p: Point) -> int:
    best = 0
    best_d2 = _dist2(g.nodes[0], p)
    for i in range(1, len(g.nodes)):
        d2 = _dist2(g.nodes[i], p)
        if d2 < best_d2:
            best_d2 = d2
            best = i
    return best


def _path_points_from_indices(g: _PathGraph, path_indices: List[int]) -> List[Point]:
    return [g.nodes[i] for i in path_indices]


def _points_to_timeline(
    pts: List[Point],
    base_t_sec: float,
    velocity: float,
    dwell_sec: float,
) -> List[Dict[str, float]]:
    if not pts:
        return []
    timeline: List[Dict[str, float]] = []
    t = base_t_sec
    timeline.append({"t": t, "x": float(pts[0][0]), "y": float(pts[0][1])})
    for i in range(1, len(pts)):
        seg_len = _dist(pts[i - 1], pts[i])
        if seg_len < 1e-6:
            continue
        t += seg_len / max(1e-6, velocity)
        timeline.append({"t": t, "x": float(pts[i][0]), "y": float(pts[i][1])})
    if dwell_sec > 0 and timeline:
        last = timeline[-1]
        timeline.append({"t": last["t"] + dwell_sec, "x": last["x"], "y": last["y"]})
    return timeline


def _build_arrival_timeline(
    ctx: LayoutContext,
    flight: Dict[str, Any],
    graph: Optional[_PathGraph],
) -> Tuple[List[Dict[str, float]], bool]:
    """
    Runway start → apron(stand) 최소거리 경로(정방향만)로 타임라인 생성.
    경로가 없으면 ([], True) 반환하여 noWay 처리.
    """
    token = flight.get("token") or {}
    runway_id = token.get("arrRunwayId") or token.get("runwayId")
    apron_id = token.get("apronId") or flight.get("standId")

    if not runway_id or not apron_id:
        return ([], True)

    rw = _get_runway_path(ctx, runway_id)
    if not rw:
        return ([], True)
    runway_start_px, _, _ = rw

    stand = ctx.pbb_stands.get(apron_id) or ctx.remote_stands.get(apron_id)
    if not stand:
        return ([], True)
    if "x2" in stand and "y2" in stand:
        stand_px: Point = (float(stand["x2"]), float(stand["y2"]))
    else:
        stand_px = _cell_to_pixel(
            float(stand.get("col", 0)), float(stand.get("row", 0)), ctx.cell_size
        )

    if graph is None:
        graph = _build_path_graph(ctx)

    # apron link로 연결된 주기장만 허용 (녹색 점이 있어야 경로 허용)
    end_idx = graph.stand_id_to_node_index.get(apron_id)
    if end_idx is None:
        return ([], True)

    start_idx = _nearest_node_index(graph, runway_start_px)
    path_indices = _dijkstra(graph, start_idx, end_idx)
    if not path_indices:
        return ([], True)

    pts = _path_points_from_indices(graph, path_indices)
    # S(d) 계열 사용 (sibtMin_d), 없으면 Original timeMin
    sibt_d = flight.get("sibtMin_d")
    sobt_d = flight.get("sobtMin_d")
    time_min = float(flight.get("timeMin", 0) or 0)
    dwell_min = float(flight.get("dwellMin", 0) or 0)
    base_min = float(sibt_d) if sibt_d is not None else time_min
    if sobt_d is not None and sibt_d is not None:
        dwell_sec = max(0.0, (float(sobt_d) - float(sibt_d)) * 60.0)
    else:
        dwell_sec = max(0.0, dwell_min * 60.0)
    base_t = base_min * 60.0
    v = max(1.0, float(flight.get("velocity", 15) or 15))
    timeline = _points_to_timeline(pts, base_t, v, dwell_sec)
    return (timeline, False)


def _build_departure_timeline(
    ctx: LayoutContext,
    flight: Dict[str, Any],
    graph: Optional[_PathGraph],
) -> Tuple[List[Dict[str, float]], bool]:
    """
    Apron(stand) → Runway start 최소거리 경로(정방향만)로 출발 타임라인.
    경로 없으면 ([], True).
    """
    token = flight.get("token") or {}
    runway_id = token.get("depRunwayId") or token.get("arrRunwayId") or token.get("runwayId")
    apron_id = token.get("apronId") or flight.get("standId")

    if not runway_id or not apron_id:
        return ([], True)

    rw = _get_runway_path(ctx, runway_id)
    if not rw:
        return ([], True)
    runway_start_px, _, _ = rw

    stand = ctx.pbb_stands.get(apron_id) or ctx.remote_stands.get(apron_id)
    if not stand:
        return ([], True)
    if "x2" in stand and "y2" in stand:
        stand_px: Point = (float(stand["x2"]), float(stand["y2"]))
    else:
        stand_px = _cell_to_pixel(
            float(stand.get("col", 0)), float(stand.get("row", 0)), ctx.cell_size
        )

    if graph is None:
        graph = _build_path_graph(ctx)

    # apron link로 연결된 주기장만 허용 (녹색 점이 있어야 경로 허용)
    start_idx = graph.stand_id_to_node_index.get(apron_id)
    if start_idx is None:
        return ([], True)

    end_idx = _nearest_node_index(graph, runway_start_px)
    path_indices = _dijkstra(graph, start_idx, end_idx)
    if not path_indices:
        return ([], True)

    pts = _path_points_from_indices(graph, path_indices)
    # S(d) 계열 사용 (sobtMin_d), 없으면 Original timeMin + dwellMin
    sobt_d = flight.get("sobtMin_d")
    time_min = float(flight.get("timeMin", 0) or 0)
    dwell_min = float(flight.get("dwellMin", 0) or 0)
    base_min = float(sobt_d) if sobt_d is not None else (time_min + dwell_min)
    base_t = base_min * 60.0
    v = max(1.0, float(flight.get("velocity", 15) or 15))
    timeline = _points_to_timeline(pts, base_t, v, 0.0)
    return (timeline, False)


def _build_context(layout: Dict[str, Any]) -> LayoutContext:
    grid = layout.get("grid") or {}
    cell_size = float(grid.get("cellSize", 20.0) or 20.0)
    taxiways_list = layout.get("taxiways") or []
    runways_list = layout.get("runways") or []
    pbb_list = layout.get("pbbStands") or []
    remote_list = layout.get("remoteStands") or []
    dm_list = layout.get("directionModes") or []
    return LayoutContext(
        cell_size=cell_size,
        taxiways={tw.get("id"): tw for tw in taxiways_list if tw.get("id")},
        runways={rw.get("id"): rw for rw in runways_list if rw.get("id")},
        pbb_stands={st.get("id"): st for st in pbb_list if st.get("id")},
        remote_stands={st.get("id"): st for st in remote_list if st.get("id")},
        apron_links=list(layout.get("apronLinks") or []),
        direction_modes={dm.get("id"): dm for dm in dm_list if dm.get("id")},
    )


def run_simulation(
    layout: Dict[str, Any],
    time_step_sec: int = 5,
    use_discrete_engine: bool = True,
    layout_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    입력 layout(JSON)을 받아 result를 반환.
    - Runway/Taxiway는 start→end 정방향만 사용하여 최소거리 경로 탐색.
    - 도착: runway start → apron(stand), 출발: apron → runway start.
    - 경로가 없으면 noWay=True, timeline은 시작점 한 점만 두어 "No way" 경고와 함께 표시.
    """
    print(f"[airside_sim] 사용 JSON: {layout_name or '(이름 미지정)'}")
    if not use_discrete_engine:
        return copy.deepcopy(layout or {})

    base_layout = copy.deepcopy(layout or {})
    ctx = _build_context(base_layout)
    graph = _build_path_graph(ctx)

    flights = base_layout.get("flights") or []
    out_flights: List[Dict[str, Any]] = []
    for f in flights:
        out = {k: v for k, v in f.items() if k not in ("timeline", "noWay")}
        timeline, no_way = _build_arrival_timeline(ctx, f, graph)
        if no_way:
            out["noWay"] = True
            token = f.get("token") or {}
            rw_id = token.get("arrRunwayId") or token.get("runwayId")
            rw = _get_runway_path(ctx, rw_id) if rw_id else None
            if rw:
                start_px = rw[0]
                sibt_d = f.get("sibtMin_d")
                time_min = float(f.get("timeMin", 0) or 0)
                base_min = float(sibt_d) if sibt_d is not None else time_min
                base_t = base_min * 60.0
                out["timeline"] = [{"t": base_t, "x": start_px[0], "y": start_px[1]}]
            else:
                out["timeline"] = []
        else:
            out["timeline"] = timeline
            out["noWay"] = False
        out_flights.append(out)

    base_layout["flights"] = out_flights
    # region agent log
    try:
        ts = int(__import__("time").time() * 1000)
        log_path = Path(__file__).resolve().parents[1] / ".cursor" / "debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        non_empty = sum(1 for fl in out_flights if fl.get("timeline"))
        payload = {
            "id": f"log_airside_{ts}",
            "timestamp": ts,
            "location": "utils/airside_sim.py:run_simulation",
            "message": "airside timelines",
            "data": {
                "layoutName": layout_name,
                "flightCount": len(out_flights),
                "withTimeline": non_empty,
            },
            "runId": "sim_debug",
            "hypothesisId": "A",
        }
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion

    return base_layout


# ─────────────── __main__: JSON 파일 입력 → run_simulation → result 파일 저장 ───────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airside simulation (JSON 입출력 테스트)")
    parser.add_argument(
        "--input",
        default="data/Layout_storage/default_layout.json",
        help="Input layout JSON",
    )
    parser.add_argument("--step", type=int, default=5, help="Time step (sec)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(1)
    json_name = path.stem
    print(f"[airside_sim] 입력 JSON 파일: {path} (이름: {json_name})")
    with open(path, encoding="utf-8") as f:
        layout = json.load(f)

    result = run_simulation(layout, time_step_sec=args.step, use_discrete_engine=True, layout_name=json_name)
    out_path = path.parent / "sim_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[완료] 결과 저장: {out_path}")
