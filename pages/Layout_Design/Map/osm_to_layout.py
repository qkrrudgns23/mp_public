"""
Convert a saved OpenAirportMap / Overpass bundle (``data/map_storage/{ICAO}_map.json``)
into a designer layout JSON matching ``data/Layout_storage/default_layout.json`` shape.

Coordinate system: local ENU-style metres with origin at the south-west corner of the
computed grid (all x,y >= 0). Grid cell size matches layout ``cellSize`` (metres per cell edge).
Layout stores projected local metres shifted to grid origin with Y mirrored to match
the designer's screen-space orientation.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, unary_union

# ---------------------------------------------------------------------------
# Human-readable rules (edit here). Python code below reads these dicts.
# ---------------------------------------------------------------------------
OSM_TO_LAYOUT_RULES: Dict[str, Any] = {
    "description": "OSM GeoJSON → Layout_storage/{ICAO}_OSM.json (default_layout-compatible keys).",
    "grid": {
        "cell_size_m": 10,
        "margin_m": 120,
        "min_cols_rows": 400,
        "max_cols_rows": 1000,
        "position_decimal_places": 2,
    },
    "projection": {
        "kind": "local_enu_metres",
        "earth_radius_m": 6378137.0,
        "origin_lon_lat": "bounds_center",
        "axes": "x=east_metres, y=north_metres in projection; layout uses shifted local metres with Y flip",
    },
    "stands": {
        "default_category": "C",
        "default_category_mode": "icao",
        "gate_contact_rule": (
            "Pick nearest OSM parking_position (lead-in) LineString within "
            "gate_to_leadin_search_m of the gate Point. If that line is within "
            "leadin_to_terminal_max_m of any terminal polygon → pbbStands (contact), "
            "else remoteStands. If no parking line in range → measure gate Point to "
            "terminal polygon; within leadin_to_terminal_max_m → contact."
        ),
        "gate_to_leadin_search_m": 70.0,
        "leadin_to_terminal_max_m": 50.0,
        "pbb_wall_length_m": 22.0,
        "pbb_bridge_stub_m": 28.0,
        "pbb_default_pbb_count": 1,
    },
    "taxiways": {
        "default_width_m": 23.0,
        "runway_default_width_m": 45.0,
        "apron_taxiway_rule": (
            "aeroway=taxiway LineString becomes pathType apron_taxiway when the line is "
            "within apron_taxiway_near_terminal_m of a merged terminal polygon; otherwise "
            "pathType taxiway."
        ),
        "apron_taxiway_near_terminal_m": 50.0,
        "include_aeroway_linestrings": (
            "taxiway",
            "taxilane",
        ),
        "jet_bridge_near_stand_m": 55.0,
    },
    "navigationaid_islands": {
        "enabled": False,
        "rule": (
            "If enabled: tags.aeroway == navigationaid AND lower(tags.navigationaid) contains "
            "island_navigationaid_substr → layoutMarkers kind=island. Point geometry "
            "becomes a small square polygon (see island_square_half_side_m)."
        ),
        "island_navigationaid_substr": "txe",
        "island_square_half_side_m": 12.0,
        "marker_outer_width_m": 15.0,
        "marker_inner_width_m": 8.0,
    },
    "terminals": {
        "aeroway_polygon_as_terminal": ("terminal",),
    },
    "runways": {
        "aeroway_line_as_runway_centerline": ("runway",),
        "default_direction": "clockwise",
    },
    "empty_collections": {
        "runway_taxiways": "RET / rapid-exit tagging not inferred from OSM in v1 — empty list.",
        "holding_points": "Derived from aeroway=holding_position when present.",
        "apron_links": "Derived from gates + parking_position lead-ins + taxiways when present.",
        "flights": "Empty until scheduled in designer.",
        "networkJunctions": "Designer rebuilds path graph when needed.",
        "Edge": "Designer rebuilds derived edges when needed.",
    },
}

# OSM tag mapping reference (aeroway / related → layout role). Values are hints only.
OSM_TAG_TO_LAYOUT_ROLE: Dict[str, str] = {
    "aeroway=runway": "runwayPaths (centerline polyline)",
    "aeroway=taxiway": "taxiways or apron_taxiway (near terminal)",
    "aeroway=taxilane": "taxiways",
    "aeroway=parking_position": "apronLinks (stand end) + stand placement; not a taxiway path",
    "aeroway=jet_bridge": "ignored for pbbCount in v1 (contact stands use pbbCount=1); not a taxiway path",
    "aeroway=holding_position": "holdingPoints (hpKind runway_holding if near runway centerline)",
    "aeroway=terminal": "terminals polygon → building vertices",
    "aeroway=gate": "pbbStands or remoteStands (see OSM_TO_LAYOUT_RULES.stands)",
    "aeroway=apron": "ignored in v1 (could become layoutMarkers area later)",
    "aeroway=hangar": "ignored in v1",
    "aeroway=helipad": "ignored in v1",
    "aeroway=navigationaid + navigationaid≈txe": "layoutMarkers kind=island",
}


def _tags(props: Dict[str, Any]) -> Dict[str, str]:
    raw = props.get("tags")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _safe_name(s: str, fallback: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    return t if t else fallback


def _float_tag(tags: Dict[str, str], key: str, default: float) -> float:
    raw = tags.get(key)
    if raw is None:
        return default
    try:
        m = re.search(r"[-+]?\d*\.?\d+", str(raw))
        if not m:
            return default
        v = float(m.group(0))
        return v if math.isfinite(v) and v > 0 else default
    except (TypeError, ValueError):
        return default


def _position_decimal_places(rules: Dict[str, Any]) -> int:
    g = rules.get("grid") if isinstance(rules.get("grid"), dict) else {}
    raw = g.get("position_decimal_places", 2)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 2
    return max(0, min(n, 8))


def _quantize_m(v: float, places: int) -> float:
    if not math.isfinite(v):
        return 0.0
    return round(float(v), places)


def _round_vertex_list_xy(vertices: List[Dict[str, Any]], places: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for v in vertices:
        if not isinstance(v, dict):
            continue
        out.append({"x": _quantize_m(float(v.get("x", 0)), places), "y": _quantize_m(float(v.get("y", 0)), places)})
    return out


def _iter_coords(geom: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Flatten GeoJSON geometry to (lon, lat) samples."""
    t = geom.get("type")
    coords = geom.get("coordinates")
    out: List[Tuple[float, float]] = []
    if t == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        out.append((float(coords[0]), float(coords[1])))
    elif t == "LineString" and isinstance(coords, list):
        for c in coords:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                out.append((float(c[0]), float(c[1])))
    elif t == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
        if isinstance(ring, list):
            for c in ring:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    out.append((float(c[0]), float(c[1])))
    elif t == "MultiPolygon" and isinstance(coords, list):
        for poly in coords:
            if not isinstance(poly, list) or not poly:
                continue
            ring = poly[0]
            if isinstance(ring, list):
                for c in ring:
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        out.append((float(c[0]), float(c[1])))
    return out


def _lonlat_bounds(features: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    min_lon, min_lat = 1e9, 1e9
    max_lon, max_lat = -1e9, -1e9
    for feat in features:
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue
        for lon, lat in _iter_coords(geom):
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
    if min_lon > max_lon:
        raise ValueError("no valid coordinates in geojson features")
    return min_lon, min_lat, max_lon, max_lat


def _project_lonlat(
    lon: float,
    lat: float,
    lon0: float,
    lat0: float,
    r_earth: float,
) -> Tuple[float, float]:
    cos_lat = math.cos(math.radians(lat0))
    x = r_earth * math.radians(lon - lon0) * cos_lat
    y = r_earth * math.radians(lat - lat0)
    return x, y


def _layout_xy_from_raw(x_raw: float, y_raw: float, x_off: float, y_off: float, y_span_m: float) -> Tuple[float, float]:
    """Projected metres shifted to grid origin with layout-space Y mirroring."""
    x = x_raw - x_off
    y0 = y_raw - y_off
    return x, y_span_m - y0


def _geom_to_xy_shape(
    geom: Dict[str, Any],
    lon0: float,
    lat0: float,
    r_earth: float,
    x_off: float,
    y_off: float,
    y_span_m: float,
):
    def conv_xy(lon: float, lat: float) -> Tuple[float, float]:
        x_raw, y_raw = _project_lonlat(lon, lat, lon0, lat0, r_earth)
        return _layout_xy_from_raw(x_raw, y_raw, x_off, y_off, y_span_m)

    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return Point(conv_xy(float(coords[0]), float(coords[1])))
    if t == "LineString" and isinstance(coords, list):
        pts = [conv_xy(float(c[0]), float(c[1])) for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        return LineString(pts) if len(pts) >= 2 else None
    if t == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
        if isinstance(ring, list):
            pts = [conv_xy(float(c[0]), float(c[1])) for c in ring if isinstance(c, (list, tuple)) and len(c) >= 2]
            return Polygon(pts) if len(pts) >= 3 else None
    if t == "MultiPolygon" and isinstance(coords, list):
        polys = []
        for poly in coords:
            if not isinstance(poly, list) or not poly:
                continue
            ring = poly[0]
            if isinstance(ring, list):
                pts = [conv_xy(float(c[0]), float(c[1])) for c in ring if isinstance(c, (list, tuple)) and len(c) >= 2]
                if len(pts) >= 3:
                    polys.append(Polygon(pts))
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    return None


def _line_vertices_xy(
    geom: Dict[str, Any], lon0: float, lat0: float, r: float, x_off: float, y_off: float, y_span_m: float
) -> List[Dict[str, float]]:
    g = _geom_to_xy_shape(geom, lon0, lat0, r, x_off, y_off, y_span_m)
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [{"x": float(x), "y": float(y)} for x, y in g.coords]
    return []


def _polygon_vertices_xy(
    geom: Dict[str, Any], lon0: float, lat0: float, r: float, x_off: float, y_off: float, y_span_m: float
) -> List[Dict[str, float]]:
    g = _geom_to_xy_shape(geom, lon0, lat0, r, x_off, y_off, y_span_m)
    if g is None or g.is_empty:
        return []
    poly = g
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.geom_type != "Polygon":
        return []
    xys = list(poly.exterior.coords[:-1])
    return [{"x": float(x), "y": float(y)} for x, y in xys]


def _union_exterior(geom: Any):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom.exterior
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda p: p.area)
        return largest.exterior
    try:
        return geom.convex_hull.exterior
    except Exception:
        return None


def _nearest_line_dist_point(lines: Sequence[Tuple[LineString, Any]], px: float, py: float) -> Tuple[Optional[LineString], float]:
    best: Optional[LineString] = None
    best_d = 1e18
    pt = Point(px, py)
    for ln, _meta in lines:
        if ln is None or ln.is_empty:
            continue
        d = ln.distance(pt)
        if d < best_d:
            best_d = d
            best = ln
    return best, best_d


def _d2(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


def _parking_stand_and_taxi_endpoints(
    ln: LineString,
    taxi_geoms: Sequence[LineString],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = list(ln.coords)
    if len(coords) < 2:
        return (float(coords[0][0]), float(coords[0][1])), (float(coords[0][0]), float(coords[0][1]))
    ax, ay = float(coords[0][0]), float(coords[0][1])
    bx, by = float(coords[-1][0]), float(coords[-1][1])
    if not taxi_geoms:
        return (bx, by), (ax, ay)

    def _taxi_d(px: float, py: float) -> float:
        p = Point(px, py)
        return min(float(g.distance(p)) for g in taxi_geoms if g is not None and not g.is_empty)

    da, db = _taxi_d(ax, ay), _taxi_d(bx, by)
    if da <= db:
        return (bx, by), (ax, ay)
    return (ax, ay), (bx, by)


def _midvertices_along_parking(
    ln: LineString,
    stand_pt: Tuple[float, float],
    taxi_pt: Tuple[float, float],
    pos_pl: int,
) -> List[Dict[str, float]]:
    coords = list(ln.coords)
    if len(coords) <= 2:
        return []

    def nearest_vertex_index(px: float, py: float) -> int:
        best_i = 0
        best_d = 1e30
        for i, c in enumerate(coords):
            d = _d2(px, py, float(c[0]), float(c[1]))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    i_s = nearest_vertex_index(stand_pt[0], stand_pt[1])
    i_t = nearest_vertex_index(taxi_pt[0], taxi_pt[1])
    lo, hi = (i_s, i_t) if i_s <= i_t else (i_t, i_s)
    inner = coords[lo + 1 : hi]
    out: List[Dict[str, float]] = []
    for c in inner:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append({"x": _quantize_m(float(c[0]), pos_pl), "y": _quantize_m(float(c[1]), pos_pl)})
    return out


def _nearest_point_on_lines(
    px: float,
    py: float,
    lines: Sequence[Tuple[LineString, str]],
) -> Tuple[float, float, str, float]:
    p = Point(px, py)
    best_d = 1e30
    best_x, best_y = px, py
    best_id = ""
    for ln, lid in lines:
        if ln is None or ln.is_empty or len(ln.coords) < 2:
            continue
        q = ln.interpolate(ln.project(p))
        d = float(p.distance(q))
        if d < best_d:
            best_d = d
            best_x, best_y = float(q.x), float(q.y)
            best_id = lid
    return best_x, best_y, best_id, best_d


def _runway_width_m(tags: Dict[str, str], default_m: float) -> float:
    w = _float_tag(tags, "width", 0.0)
    return w if w > 0 else default_m


def _runway_displaced_m(tags: Dict[str, str]) -> Tuple[float, float]:
    s = max(_float_tag(tags, "displaced_threshold:start", 0.0), _float_tag(tags, "start_displaced_threshold", 0.0))
    e = max(_float_tag(tags, "displaced_threshold:end", 0.0), _float_tag(tags, "end_displaced_threshold", 0.0))
    g = _float_tag(tags, "displaced_threshold", 0.0)
    if s <= 0 and e <= 0 and g > 0:
        s = g
    return s, e


def _linemerge_to_linestrings(geoms: Sequence[LineString]) -> List[LineString]:
    if not geoms:
        return []
    merged = linemerge(list(geoms))
    if merged.is_empty:
        return []
    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return [g for g in merged.geoms if isinstance(g, LineString) and len(g.coords) >= 2]
    geoms_out: List[LineString] = []
    for g in getattr(merged, "geoms", []):
        if isinstance(g, LineString) and len(g.coords) >= 2:
            geoms_out.append(g)
    return geoms_out


def _piece_tags_at_runway_endpoint(
    pt_xy: Point,
    pieces: Sequence[Tuple[LineString, Dict[str, str]]],
    tol_m: float,
) -> Dict[str, str]:
    best: Dict[str, str] = {}
    best_d = 1e30
    for ln, tags in pieces:
        if ln is None or ln.is_empty:
            continue
        for vx, vy in (ln.coords[0], ln.coords[-1]):
            d = float(pt_xy.distance(Point(float(vx), float(vy))))
            if d < best_d:
                best_d = d
                best = tags
    return best if best_d <= tol_m else {}


def _point_near_any_line(pt: Tuple[float, float], lines: Sequence[LineString], tol_m: float) -> bool:
    p = Point(pt[0], pt[1])
    for ln in lines:
        if ln is None or ln.is_empty:
            continue
        if float(ln.distance(p)) <= tol_m:
            return True
    return False


def _parking_endpoint_class(
    pt: Tuple[float, float],
    own_idx: int,
    parking_lines: Sequence[Tuple[LineString, Dict[str, Any]]],
    taxi_lines: Sequence[LineString],
    tol_m: float,
) -> str:
    if _point_near_any_line(pt, taxi_lines, tol_m):
        return "taxiway"
    p = Point(pt[0], pt[1])
    for i, (ln, _tags) in enumerate(parking_lines):
        if i == own_idx or ln is None or ln.is_empty:
            continue
        if float(ln.distance(p)) <= tol_m:
            return "parking_intersection"
    return "apron_origin"


def _parking_edge_into_apron_origin_unit_deg(pln: LineString, origin_at_start: bool) -> Tuple[float, float, float]:
    """PBB / remote apron: unit (nx, ny) and angleDeg, 180° opposite to the parking edge into apron_origin."""
    cc = list(pln.coords)
    if len(cc) < 2:
        return -1.0, 0.0, 180.0
    if origin_at_start:
        vx = float(cc[0][0]) - float(cc[1][0])
        vy = float(cc[0][1]) - float(cc[1][1])
    else:
        vx = float(cc[-1][0]) - float(cc[-2][0])
        vy = float(cc[-1][1]) - float(cc[-2][1])
    lnlen = math.hypot(vx, vy)
    if lnlen < 1e-9:
        return -1.0, 0.0, 180.0
    ex, ey = vx / lnlen, vy / lnlen
    nx, ny = -ex, -ey
    return nx, ny, math.degrees(math.atan2(ny, nx))


def _match_parking_apron_edge_unit_deg_for_xy(
    parking_lines: Sequence[Tuple[LineString, Dict[str, Any]]],
    sx: float,
    sy: float,
    taxi_mate_geoms: Sequence[LineString],
    parking_tol_m: float,
    layout_xy_tol_m: float,
) -> Optional[Tuple[float, float, float]]:
    """If (sx,sy) matches a parking apron_origin, return (nx, ny, angleDeg) for that lead-in edge; else None."""
    for pi2, (pln2, _pt2) in enumerate(parking_lines):
        if pln2 is None or pln2.is_empty or len(pln2.coords) < 2:
            continue
        p0 = (float(pln2.coords[0][0]), float(pln2.coords[0][1]))
        p1 = (float(pln2.coords[-1][0]), float(pln2.coords[-1][1]))
        c0 = _parking_endpoint_class(p0, pi2, parking_lines, taxi_mate_geoms, parking_tol_m)
        c1 = _parking_endpoint_class(p1, pi2, parking_lines, taxi_mate_geoms, parking_tol_m)
        origin = p0 if c0 == "apron_origin" else (p1 if c1 == "apron_origin" else None)
        if origin is None:
            continue
        if math.hypot(origin[0] - sx, origin[1] - sy) > layout_xy_tol_m:
            continue
        origin_at_start = c0 == "apron_origin"
        return _parking_edge_into_apron_origin_unit_deg(pln2, origin_at_start)
    return None


def _parking_neighbors_at_point(
    pt: Tuple[float, float],
    own_idx: int,
    parking_lines: Sequence[Tuple[LineString, Dict[str, Any]]],
    tol_m: float,
) -> List[int]:
    p = Point(pt[0], pt[1])
    out: List[int] = []
    for i, (ln, _tags) in enumerate(parking_lines):
        if i == own_idx or ln is None or ln.is_empty:
            continue
        if float(ln.distance(p)) <= tol_m:
            out.append(i)
    return out


def _append_apron_link_candidate(
    out_list: List[Dict[str, Any]],
    pts: List[Tuple[float, float]],
    name: str,
    pbb_id: Optional[str],
    pos_pl: int,
) -> None:
    if len(pts) < 2:
        return
    dedup: List[Tuple[float, float]] = []
    for p in pts:
        if not dedup or abs(dedup[-1][0] - p[0]) > 1e-9 or abs(dedup[-1][1] - p[1]) > 1e-9:
            dedup.append(p)
    if len(dedup) < 2:
        return
    verts = _round_vertex_list_xy([{"x": p[0], "y": p[1]} for p in dedup], pos_pl)
    if len(verts) < 2:
        return
    out_list.append({"name": str(name), "pbbId": str(pbb_id) if pbb_id else "", "points": verts})


def _line_path_between_points(
    ln: LineString,
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> List[Tuple[float, float]]:
    if ln is None or ln.is_empty or len(ln.coords) < 2:
        return [a, b]
    coords = [(float(x), float(y)) for x, y in ln.coords]
    p_a = Point(a[0], a[1])
    p_b = Point(b[0], b[1])
    da = float(ln.project(p_a))
    db = float(ln.project(p_b))
    pa = ln.interpolate(da)
    pb = ln.interpolate(db)
    if da <= db:
        lo, hi = da, db
        rev = False
    else:
        lo, hi = db, da
        rev = True
    out: List[Tuple[float, float]] = []
    out.append((float(pa.x), float(pa.y)) if not rev else (float(pb.x), float(pb.y)))
    acc = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-9:
            continue
        seg_start = acc
        seg_end = acc + seg_len
        if seg_end <= lo + 1e-9:
            acc = seg_end
            continue
        if seg_start >= hi - 1e-9:
            break
        if seg_start >= lo - 1e-9 and seg_start <= hi + 1e-9:
            out.append((x0, y0))
        if seg_end >= lo - 1e-9 and seg_end <= hi + 1e-9:
            out.append((x1, y1))
        acc = seg_end
    out.append((float(pb.x), float(pb.y)) if not rev else (float(pa.x), float(pa.y)))
    dedup: List[Tuple[float, float]] = []
    for p in (reversed(out) if rev else out):
        if not dedup or abs(dedup[-1][0] - p[0]) > 1e-9 or abs(dedup[-1][1] - p[1]) > 1e-9:
            dedup.append(p)
    return dedup


def _nearest_gate_for_point(
    pt: Tuple[float, float],
    gates: Sequence[Tuple[float, float, Dict[str, str], str]],
    max_m: float,
) -> Optional[Tuple[float, float, Dict[str, str], str]]:
    best = None
    best_d = 1e30
    x, y = pt
    for gx, gy, tags, fid in gates:
        d = math.hypot(gx - x, gy - y)
        if d < best_d:
            best_d = d
            best = (gx, gy, tags, fid)
    if best is None or best_d > max_m:
        return None
    return best


def _nearest_terminal_label_for_point(
    pt: Tuple[float, float],
    terminals: Sequence[Tuple[str, Polygon]],
) -> str:
    if not terminals:
        return "Terminal"
    p = Point(float(pt[0]), float(pt[1]))
    best_name = "Terminal"
    best_d = 1e30
    for name, poly in terminals:
        if poly is None or poly.is_empty:
            continue
        d = float(poly.distance(p))
        if d < best_d:
            best_d = d
            best_name = str(name or "Terminal")
    return best_name


def _format_apron_taxiway_name(terminal_name: str, gate_name: str, branch_idx: Optional[int]) -> str:
    def _clean(s: str, fallback: str) -> str:
        t = str(s or "").strip()
        if not t:
            t = fallback
        t = re.sub(r"\s+", "_", t)
        t = re.sub(r"[^A-Za-z0-9_-]", "-", t)
        t = re.sub(r"-{2,}", "-", t).strip("-_")
        return t or fallback

    t_name = _clean(terminal_name, "Terminal")
    g_name = _clean(gate_name, "Gate")
    base = f"{t_name}_{g_name}_ATX"
    if branch_idx is None:
        return base
    return f"{base}_{branch_idx}"


def build_layout_from_map_storage_document(doc: Dict[str, Any], icao: str) -> Dict[str, Any]:
    """
    Build a layout dict from the same structure ``airport_overpass_fetch.build_storage_document`` writes:
    ``{ "icao", "geojson": FeatureCollection, ... }``.
    """
    if not isinstance(doc, dict):
        raise TypeError("document must be a dict")
    gj = doc.get("geojson")
    if not isinstance(gj, dict):
        raise ValueError("document.geojson missing")
    feats = gj.get("features")
    if not isinstance(feats, list):
        raise ValueError("document.geojson.features must be a list")

    rules = OSM_TO_LAYOUT_RULES
    cell = float(rules["grid"]["cell_size_m"])
    margin = float(rules["grid"]["margin_m"])
    min_cr = int(rules["grid"]["min_cols_rows"])
    max_cr = int(rules["grid"]["max_cols_rows"])
    pos_pl = _position_decimal_places(rules)
    r_earth = float(rules["projection"]["earth_radius_m"])

    min_lon, min_lat, max_lon, max_lat = _lonlat_bounds(feats)
    lon0 = (min_lon + max_lon) / 2.0
    lat0 = (min_lat + max_lat) / 2.0

    xs: List[float] = []
    ys: List[float] = []
    for feat in feats:
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue
        for lon, lat in _iter_coords(geom):
            x, y = _project_lonlat(lon, lat, lon0, lat0, r_earth)
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("no coordinates after projection")
    raw_min_x, raw_max_x = min(xs), max(xs)
    raw_min_y, raw_max_y = min(ys), max(ys)
    span_x = raw_max_x - raw_min_x + 2 * margin
    span_y = raw_max_y - raw_min_y + 2 * margin
    cols = int(math.ceil(span_x / cell))
    rows = int(math.ceil(span_y / cell))
    cols = max(min_cr, min(max_cr, cols))
    rows = max(min_cr, min(max_cr, rows))

    x_off = raw_min_x - margin
    y_off = raw_min_y - margin
    y_span_m = float(rows) * cell

    terminal_polys: List[Polygon] = []
    terminal_named_polys: List[Tuple[str, Polygon]] = []
    parking_lines: List[Tuple[LineString, Dict[str, Any]]] = []
    taxi_lines: List[Tuple[LineString, Dict[str, str], str]] = []  # line, tags, aeroway
    runway_lines: List[Tuple[LineString, Dict[str, str]]] = []
    holding_layout_xy: List[Tuple[float, float, Dict[str, str]]] = []
    gates: List[Tuple[float, float, Dict[str, str], str]] = []  # x,y,tags,feat_key
    txe_points: List[Tuple[float, float]] = []

    for feat in feats:
        props = feat.get("properties")
        if not isinstance(props, dict):
            continue
        tags = _tags(props)
        aw = tags.get("aeroway", "")
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue
        fid = str(props.get("id", _new_id("feat")))
        gxy = _geom_to_xy_shape(geom, lon0, lat0, r_earth, x_off, y_off, y_span_m)

        if aw in rules["terminals"]["aeroway_polygon_as_terminal"]:
            verts = _polygon_vertices_xy(geom, lon0, lat0, r_earth, x_off, y_off, y_span_m)
            if len(verts) >= 3:
                poly = Polygon([(v["x"], v["y"]) for v in verts])
                terminal_polys.append(poly)
                term_name = _safe_name(tags.get("name") or tags.get("ref"), f"Terminal-{len(terminal_named_polys)+1}")
                terminal_named_polys.append((term_name, poly))
        if aw == "parking_position" and isinstance(gxy, LineString) and len(gxy.coords) >= 2:
            parking_lines.append((gxy, tags))
        if aw in ("holding_position", "holding"):
            if isinstance(gxy, Point):
                holding_layout_xy.append((float(gxy.x), float(gxy.y), tags))
            elif isinstance(gxy, LineString) and len(gxy.coords) >= 2:
                mx = int(len(gxy.coords) / 2)
                cx, cy = float(gxy.coords[mx][0]), float(gxy.coords[mx][1])
                holding_layout_xy.append((cx, cy, tags))
        if aw in rules["taxiways"]["include_aeroway_linestrings"]:
            if isinstance(gxy, LineString) and len(gxy.coords) >= 2:
                taxi_lines.append((gxy, tags, aw))
        if aw in rules["runways"]["aeroway_line_as_runway_centerline"]:
            if isinstance(gxy, LineString) and len(gxy.coords) >= 2:
                runway_lines.append((gxy, tags))

        if aw == "gate" and geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                x_raw, y_raw = _project_lonlat(float(coords[0]), float(coords[1]), lon0, lat0, r_earth)
                gx, gy = _layout_xy_from_raw(x_raw, y_raw, x_off, y_off, y_span_m)
                gates.append((gx, gy, tags, fid))

        nav_cfg = rules["navigationaid_islands"]
        if nav_cfg.get("enabled") is True and aw == "navigationaid":
            nav = str(tags.get("navigationaid", "")).lower()
            sub = str(nav_cfg["island_navigationaid_substr"]).lower()
            if sub in nav and geom.get("type") == "Point":
                coords = geom.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    px_raw, py_raw = _project_lonlat(float(coords[0]), float(coords[1]), lon0, lat0, r_earth)
                    px, py = _layout_xy_from_raw(px_raw, py_raw, x_off, y_off, y_span_m)
                    txe_points.append((px, py))

    terminal_union = unary_union(terminal_polys) if terminal_polys else None
    leadin_max = float(rules["stands"]["leadin_to_terminal_max_m"])
    gate_search = float(rules["stands"]["gate_to_leadin_search_m"])
    wall_len = float(rules["stands"]["pbb_wall_length_m"])
    stub = float(rules["stands"]["pbb_bridge_stub_m"])
    taxi_mate_geoms = [ln for ln, _, _ in taxi_lines]

    pbb_stands: List[Dict[str, Any]] = []
    remote_stands: List[Dict[str, Any]] = []
    stand_records: List[Dict[str, Any]] = []
    apron_link_candidates: List[Dict[str, Any]] = []
    cat = str(rules["stands"]["default_category"])
    mode = str(rules["stands"]["default_category_mode"])
    parking_tol_m = 1.5
    contact_thresh_m = 80.0
    for pi, (pln, ptags) in enumerate(parking_lines):
        if pln is None or pln.is_empty or len(pln.coords) < 2:
            continue
        p0 = (float(pln.coords[0][0]), float(pln.coords[0][1]))
        p1 = (float(pln.coords[-1][0]), float(pln.coords[-1][1]))
        c0 = _parking_endpoint_class(p0, pi, parking_lines, taxi_mate_geoms, parking_tol_m)
        c1 = _parking_endpoint_class(p1, pi, parking_lines, taxi_mate_geoms, parking_tol_m)
        origin = p0 if c0 == "apron_origin" else (p1 if c1 == "apron_origin" else None)
        inter = p0 if c0 == "parking_intersection" else (p1 if c1 == "parking_intersection" else None)
        taxi_ep = p0 if c0 == "taxiway" else (p1 if c1 == "taxiway" else None)
        if origin is None:
            # Only create apron artifacts from explicit class-3 endpoint.
            continue
        gate_match = _nearest_gate_for_point(origin, gates, gate_search)
        # Rule lock: point-3 (apron_origin) is the stand attach center ("middle circle").
        sx, sy = origin
        own_name = str(ptags.get("ref") or ptags.get("name") or "").strip()
        if own_name:
            stand_name = _safe_name(own_name, f"PARK-{pi+1}")
        elif gate_match is not None:
            _gx_m, _gy_m, gtags_m, fid_m = gate_match
            stand_name = _safe_name(gtags_m.get("ref") or gtags_m.get("name"), f"GATE-{fid_m}")
        else:
            stand_name = f"PARK-{pi+1}"
        contact = False
        if terminal_union is not None and not terminal_union.is_empty:
            pt_st = Point(sx, sy)
            d_term = float(pt_st.distance(terminal_union))
            contact = d_term < contact_thresh_m
        source_pbb_id: Optional[str] = None
        if contact:
            origin_at_start = c0 == "apron_origin"
            nx, ny, ang = _parking_edge_into_apron_origin_unit_deg(pln, origin_at_start)
            pbx1, pby1 = sx - nx * wall_len * 0.3, sy - ny * wall_len * 0.3
            pbx2, pby2 = pbx1 + nx * wall_len, pby1 + ny * wall_len
            pbb_obj = {
                "id": _new_id("pbb"),
                "name": stand_name,
                "x1": _quantize_m(pbx1, pos_pl),
                "y1": _quantize_m(pby1, pos_pl),
                "x2": _quantize_m(pbx2, pos_pl),
                "y2": _quantize_m(pby2, pos_pl),
                "category": cat,
                "categoryMode": mode,
                "allowedAircraftTypes": [],
                "pbbCount": 1,
                "angleDeg": ang,
                # Keep stand attach point at class-3 apron origin / matched gate anchor.
                "apronSiteX": _quantize_m(sx, pos_pl),
                "apronSiteY": _quantize_m(sy, pos_pl),
                "boardingWidthM": 5,
                "boardingHeightM": 15,
                "pbbArmLenM": max(10.0, stub),
                "edgeCol": _quantize_m(sx / cell, pos_pl),
                "edgeRow": _quantize_m(sy / cell, pos_pl),
            }
            pbb_stands.append(pbb_obj)
            stand_records.append({"kind": "contact", "pbb": pbb_obj, "nearest_line": pln, "d_pl": 0.0})
            source_pbb_id = str(pbb_obj["id"])
        else:
            origin_at_start_r = c0 == "apron_origin"
            _rx, _ry, rang = _parking_edge_into_apron_origin_unit_deg(pln, origin_at_start_r)
            remote_obj = {
                "id": _new_id("remote"),
                "name": stand_name,
                "x": _quantize_m(sx, pos_pl),
                "y": _quantize_m(sy, pos_pl),
                "category": cat,
                "angleDeg": _quantize_m(rang, pos_pl),
                "categoryMode": mode,
                "allowedAircraftTypes": [],
                "allowedTerminals": [],
            }
            remote_stands.append(remote_obj)
            stand_records.append({"kind": "remote"})
            source_pbb_id = str(remote_obj["id"])
        atx_terminal_name = _nearest_terminal_label_for_point((sx, sy), terminal_named_polys)
        if inter is not None:
            neigh = _parking_neighbors_at_point(inter, pi, parking_lines, parking_tol_m)
            taxi_targets: List[Tuple[Tuple[float, float], LineString]] = []
            for ni in neigh:
                nln, _nt = parking_lines[ni]
                if nln is None or nln.is_empty or len(nln.coords) < 2:
                    continue
                np0 = (float(nln.coords[0][0]), float(nln.coords[0][1]))
                np1 = (float(nln.coords[-1][0]), float(nln.coords[-1][1]))
                cnp0 = _parking_endpoint_class(np0, ni, parking_lines, taxi_mate_geoms, parking_tol_m)
                cnp1 = _parking_endpoint_class(np1, ni, parking_lines, taxi_mate_geoms, parking_tol_m)
                if cnp0 == "taxiway":
                    taxi_targets.append((np0, nln))
                if cnp1 == "taxiway":
                    taxi_targets.append((np1, nln))
                # Fallback for branch: if explicit taxiway endpoint is absent, use the endpoint opposite from intersection.
                if cnp0 != "taxiway" and cnp1 != "taxiway":
                    d0 = math.hypot(np0[0] - inter[0], np0[1] - inter[1])
                    d1 = math.hypot(np1[0] - inter[0], np1[1] - inter[1])
                    far = np0 if d0 >= d1 else np1
                    taxi_targets.append((far, nln))
            if taxi_ep is not None:
                taxi_targets.append((taxi_ep, pln))
            uniq_targets: List[Tuple[Tuple[float, float], LineString]] = []
            seen_t = set()
            for tp, tln in taxi_targets:
                k = (_quantize_m(tp[0], 2), _quantize_m(tp[1], 2), id(tln))
                if k in seen_t:
                    continue
                seen_t.add(k)
                uniq_targets.append((tp, tln))
            # Rule lock: if 1-1/1-2 exists create two branches from 3->2, otherwise fallback to 3->1 direct.
            for ti, (tp, tln) in enumerate(uniq_targets[:2]):
                path_a = _line_path_between_points(pln, origin, inter)
                path_b = _line_path_between_points(tln, inter, tp)
                full_path = path_a + (path_b[1:] if len(path_b) >= 2 else [])
                _append_apron_link_candidate(
                    apron_link_candidates,
                    full_path,
                    _format_apron_taxiway_name(atx_terminal_name, stand_name, ti + 1),
                    source_pbb_id,
                    pos_pl,
                )
            if not uniq_targets and taxi_ep is not None:
                full_path = _line_path_between_points(pln, origin, taxi_ep)
                _append_apron_link_candidate(
                    apron_link_candidates,
                    full_path,
                    _format_apron_taxiway_name(atx_terminal_name, stand_name, None),
                    source_pbb_id,
                    pos_pl,
                )
        else:
            if taxi_ep is None:
                continue
            full_path = _line_path_between_points(pln, origin, taxi_ep)
            _append_apron_link_candidate(
                apron_link_candidates,
                full_path,
                _format_apron_taxiway_name(atx_terminal_name, stand_name, None),
                source_pbb_id,
                pos_pl,
            )

    # Fallback to legacy gate-based stand synthesis when parking_position lines are absent.
    if not parking_lines:
        for gx, gy, tags, fid in gates:
            nearest_pl, d_pl = _nearest_line_dist_point(parking_lines, gx, gy)
            contact = False
            if nearest_pl is not None and d_pl <= gate_search:
                if terminal_union is None or terminal_union.is_empty:
                    contact = False
                else:
                    contact = nearest_pl.distance(terminal_union) <= leadin_max
            else:
                if terminal_union is not None and not terminal_union.is_empty:
                    contact = Point(gx, gy).distance(terminal_union) <= leadin_max
            name = _safe_name(tags.get("ref") or tags.get("name"), f"GATE-{fid}")
            if contact:
                sx, sy = gx, gy
                parking_ln_gate: Optional[LineString] = None
                if nearest_pl is not None and isinstance(nearest_pl, LineString):
                    parking_ln_gate = nearest_pl
                    st, _txy = _parking_stand_and_taxi_endpoints(nearest_pl, taxi_mate_geoms)
                    sx, sy = st
                ang = 0.0
                nx, ny = 1.0, 0.0
                if parking_ln_gate is not None and not parking_ln_gate.is_empty and len(parking_ln_gate.coords) >= 2:
                    ccg = list(parking_ln_gate.coords)
                    d_s0 = math.hypot(sx - float(ccg[0][0]), sy - float(ccg[0][1]))
                    d_s1 = math.hypot(sx - float(ccg[-1][0]), sy - float(ccg[-1][1]))
                    origin_at_start = d_s0 <= d_s1
                    nx, ny, ang = _parking_edge_into_apron_origin_unit_deg(parking_ln_gate, origin_at_start)
                pbx1, pby1 = sx - nx * wall_len * 0.3, sy - ny * wall_len * 0.3
                pbx2, pby2 = pbx1 + nx * wall_len, pby1 + ny * wall_len
                pbb_obj: Dict[str, Any] = {
                    "id": _new_id("pbb"),
                    "name": name,
                    "x1": _quantize_m(pbx1, pos_pl),
                    "y1": _quantize_m(pby1, pos_pl),
                    "x2": _quantize_m(pbx2, pos_pl),
                    "y2": _quantize_m(pby2, pos_pl),
                    "category": cat,
                    "categoryMode": mode,
                    "allowedAircraftTypes": [],
                    "pbbCount": 1,
                    "angleDeg": ang,
                    "apronSiteX": _quantize_m(sx, pos_pl),
                    "apronSiteY": _quantize_m(sy, pos_pl),
                    "boardingWidthM": 5,
                    "boardingHeightM": 15,
                    "pbbArmLenM": max(10.0, stub),
                    "edgeCol": _quantize_m(sx / cell, pos_pl),
                    "edgeRow": _quantize_m(sy / cell, pos_pl),
                }
                pbb_stands.append(pbb_obj)
                stand_records.append({"kind": "contact", "pbb": pbb_obj, "nearest_line": nearest_pl, "d_pl": d_pl})
            else:
                remote_stands.append(
                    {
                        "id": _new_id("remote"),
                        "name": name,
                        "x": _quantize_m(gx, pos_pl),
                        "y": _quantize_m(gy, pos_pl),
                        "category": cat,
                        "angleDeg": 0,
                        "categoryMode": mode,
                        "allowedAircraftTypes": [],
                        "allowedTerminals": [],
                    }
                )
                stand_records.append({"kind": "remote"})

    rw_default_w = float(rules["taxiways"]["runway_default_width_m"])
    runway_paths: List[Dict[str, Any]] = []
    rwy_tol = 5.0
    merged_runways = _linemerge_to_linestrings([ln for ln, _ in runway_lines])
    for merged in merged_runways:
        verts = _round_vertex_list_xy([{"x": float(x), "y": float(y)} for x, y in merged.coords], pos_pl)
        if len(verts) < 2:
            continue
        st_tags = _piece_tags_at_runway_endpoint(Point(merged.coords[0]), runway_lines, rwy_tol)
        en_tags = _piece_tags_at_runway_endpoint(Point(merged.coords[-1]), runway_lines, rwy_tol)
        if not st_tags and runway_lines:
            st_tags = runway_lines[0][1]
        if not en_tags and runway_lines:
            en_tags = runway_lines[-1][1]
        st_thr, _st_e = _runway_displaced_m(st_tags)
        _en_s, en_thr = _runway_displaced_m(en_tags)
        w = max(_runway_width_m(st_tags, rw_default_w), _runway_width_m(en_tags, rw_default_w))
        for _, t in runway_lines:
            w = max(w, _runway_width_m(t, rw_default_w))
        rw_id = _new_id("rwy")
        rname = _safe_name(st_tags.get("ref") or st_tags.get("name") or en_tags.get("ref") or en_tags.get("name"), rw_id)
        surf = str(st_tags.get("surface", en_tags.get("surface", "asphalt")) or "asphalt")
        runway_paths.append(
            {
                "id": rw_id,
                "name": rname,
                "vertices": verts,
                "width": w,
                "direction": str(rules["runways"]["default_direction"]),
                "minArrVelocity": 15,
                "lineupDistM": 0,
                "avgMoveVelocity": 10,
                "startDisplacedThresholdM": _quantize_m(st_thr, pos_pl),
                "startBlastPadM": 0,
                "endDisplacedThresholdM": _quantize_m(en_thr, pos_pl),
                "endBlastPadM": 0,
                "pavement": surf,
            }
        )

    taxiways_out: List[Dict[str, Any]] = []
    for ln, tags, aw in taxi_lines:
        verts = _round_vertex_list_xy([{"x": float(x), "y": float(y)} for x, y in ln.coords], pos_pl)
        if len(verts) < 2:
            continue
        w = _float_tag(tags, "width", float(rules["taxiways"]["default_width_m"]))
        tw_id = _new_id("tw")
        taxiways_out.append(
            {
                "id": tw_id,
                "name": _safe_name(tags.get("ref") or tags.get("name"), f"{aw}-{tw_id}"),
                "vertices": verts,
                "width": w,
                "direction": "both",
                "avgMoveVelocity": 10,
                "pavement": str(tags.get("surface", "asphalt") or "asphalt"),
            }
        )

    taxi_spines: List[Tuple[LineString, str]] = []
    for tw in taxiways_out:
        verts = tw.get("vertices")
        if not isinstance(verts, list) or len(verts) < 2:
            continue
        pt = str(tw.get("pathType") or "taxiway")
        if pt not in ("taxiway", "apron_taxiway", "general_queue_taxiway"):
            continue
        tw_ln = LineString([(float(v["x"]), float(v["y"])) for v in verts if isinstance(v, dict)])
        if tw_ln.is_empty or len(tw_ln.coords) < 2:
            continue
        taxi_spines.append((tw_ln, str(tw["id"])))

    apron_links: List[Dict[str, Any]] = []
    lk_i = 0
    linked_pbb_ids: set[str] = set()
    for cand in apron_link_candidates:
        pbb_id = str(cand.get("pbbId") or "").strip()
        pts = cand.get("points") if isinstance(cand.get("points"), list) else []
        if not pbb_id or len(pts) < 2:
            continue
        taxi_pt = pts[-1]
        tx, ty, tw_id, d_snap = _nearest_point_on_lines(float(taxi_pt["x"]), float(taxi_pt["y"]), taxi_spines)
        if not tw_id or d_snap > 120.0:
            continue
        mids = []
        for v in pts[1:-1]:
            if not isinstance(v, dict):
                continue
            mids.append({"x": _quantize_m(float(v.get("x", 0.0)), pos_pl), "y": _quantize_m(float(v.get("y", 0.0)), pos_pl)})
        lk_i += 1
        apron_links.append(
            {
                "id": _new_id("alk"),
                "name": _safe_name(str(cand.get("name") or ""), f"Apron Taxiway {lk_i}"),
                "pbbId": pbb_id,
                "taxiwayId": tw_id,
                "tx": _quantize_m(tx, pos_pl),
                "ty": _quantize_m(ty, pos_pl),
                "midVertices": mids,
            }
        )
        linked_pbb_ids.add(pbb_id)
    for rec in stand_records:
        if rec.get("kind") != "contact":
            continue
        pbb = rec.get("pbb")
        pbb_id = str((pbb or {}).get("id") or "")
        if pbb_id in linked_pbb_ids:
            continue
        nearest_line = rec.get("nearest_line")
        d_pl = float(rec.get("d_pl", 999.0))
        if not isinstance(pbb, dict) or nearest_line is None or d_pl > gate_search:
            continue
        if not isinstance(nearest_line, LineString) or nearest_line.is_empty:
            continue
        stand_pt, taxi_pt = _parking_stand_and_taxi_endpoints(nearest_line, taxi_mate_geoms)
        tx, ty, tw_id, d_snap = _nearest_point_on_lines(taxi_pt[0], taxi_pt[1], taxi_spines)
        if not tw_id or d_snap > 120.0:
            continue
        mids = _midvertices_along_parking(nearest_line, stand_pt, taxi_pt, pos_pl)
        lk_i += 1
        apron_links.append(
            {
                "id": _new_id("alk"),
                "name": f"Apron Taxiway {lk_i}",
                "pbbId": pbb_id,
                "taxiwayId": tw_id,
                "tx": _quantize_m(tx, pos_pl),
                "ty": _quantize_m(ty, pos_pl),
                "midVertices": mids,
            }
        )

    runway_geom_for_hp: List[LineString] = _linemerge_to_linestrings([ln for ln, _ in runway_lines])
    holding_points_out: List[Dict[str, Any]] = []
    hp_idx = 0
    hp_near_rw_m = 120.0
    for hx, hy, htags in holding_layout_xy:
        hp_idx += 1
        hp_kind = "intermediate"
        hp_pt = Point(hx, hy)
        for rln in runway_geom_for_hp:
            if rln.distance(hp_pt) <= hp_near_rw_m:
                hp_kind = "runway_holding"
                break
        holding_points_out.append(
            {
                "id": _new_id("hp"),
                "name": _safe_name(htags.get("ref") or htags.get("name"), f"Position{hp_idx}"),
                "x": _quantize_m(hx, pos_pl),
                "y": _quantize_m(hy, pos_pl),
                "hpKind": hp_kind,
            }
        )

    layout_markers: List[Dict[str, Any]] = []
    if rules["navigationaid_islands"].get("enabled") is True:
        half = float(rules["navigationaid_islands"]["island_square_half_side_m"])
        ow = float(rules["navigationaid_islands"]["marker_outer_width_m"])
        iw = float(rules["navigationaid_islands"]["marker_inner_width_m"])
        for px, py in txe_points:
            pts = _round_vertex_list_xy(
                [
                    {"x": px - half, "y": py - half},
                    {"x": px + half, "y": py - half},
                    {"x": px + half, "y": py + half},
                    {"x": px - half, "y": py + half},
                ],
                pos_pl,
            )
            layout_markers.append(
                {
                    "kind": "island",
                    "id": _new_id("txe"),
                    "points": pts,
                    "outerWidthM": ow,
                    "innerWidthM": iw,
                    "pavement": "asphalt",
                }
            )

    terminals_out: List[Dict[str, Any]] = []
    t_idx = 0
    for feat in feats:
        props = feat.get("properties")
        if not isinstance(props, dict):
            continue
        tags = _tags(props)
        if tags.get("aeroway") not in rules["terminals"]["aeroway_polygon_as_terminal"]:
            continue
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            continue
        verts = _round_vertex_list_xy(_polygon_vertices_xy(geom, lon0, lat0, r_earth, x_off, y_off, y_span_m), pos_pl)
        if len(verts) < 3:
            continue
        t_idx += 1
        tid = _new_id("term")
        terminals_out.append(
            {
                "id": tid,
                "name": _safe_name(tags.get("name") or tags.get("ref"), f"Terminal-{t_idx}"),
                "vertices": verts,
                "closed": True,
                "floors": 1,
                "floorToFloor": 5,
                "floorHeight": 5,
                "departureCapacity": 100,
                "arrivalCapacity": 100,
                "buildingType": "terminal",
            }
        )

    # Enforce stand type rule on final terminal contours:
    # inside 80 m -> contact stand, 80 m+ -> remote stand.
    if terminals_out and remote_stands:
        term_polys_out: List[Polygon] = []
        for t in terminals_out:
            tv = t.get("vertices")
            if not isinstance(tv, list) or len(tv) < 3:
                continue
            pts = []
            for v in tv:
                if not isinstance(v, dict):
                    continue
                pts.append((float(v.get("x", 0.0)), float(v.get("y", 0.0))))
            if len(pts) >= 3:
                term_polys_out.append(Polygon(pts))
        term_union_out = unary_union(term_polys_out) if term_polys_out else None
        term_ring_out = _union_exterior(term_union_out) if term_union_out is not None else None
        kept_remote: List[Dict[str, Any]] = []
        promoted_contact: List[Dict[str, Any]] = []
        remote_id_to_promoted_pbb_id: Dict[str, str] = {}
        for rs in remote_stands:
            sx = float(rs.get("x", 0.0))
            sy = float(rs.get("y", 0.0))
            if term_union_out is None or term_union_out.is_empty:
                kept_remote.append(rs)
                continue
            d_term = float(Point(sx, sy).distance(term_union_out))
            if d_term >= contact_thresh_m or term_ring_out is None:
                kept_remote.append(rs)
                continue
            edge_orient = _match_parking_apron_edge_unit_deg_for_xy(
                parking_lines, sx, sy, taxi_mate_geoms, parking_tol_m, 2.0
            )
            if edge_orient is not None:
                nx, ny, ang = edge_orient
                x1 = sx - nx * wall_len * 0.3
                y1 = sy - ny * wall_len * 0.3
                x2 = x1 + nx * wall_len
                y2 = y1 + ny * wall_len
            else:
                nr = term_ring_out.interpolate(term_ring_out.project(Point(sx, sy)))
                nx, ny = float(nr.x), float(nr.y)
                dx = sx - nx
                dy = sy - ny
                dlen = math.hypot(dx, dy)
                if dlen < 1e-6:
                    dx, dy = 1.0, 0.0
                    dlen = 1.0
                dx /= dlen
                dy /= dlen
                ang = (math.degrees(math.atan2(dy, dx)) + 180.0) % 360.0
                x1 = nx - dx * wall_len * 0.5
                y1 = ny - dy * wall_len * 0.5
                x2 = nx + dx * wall_len * 0.5
                y2 = ny + dy * wall_len * 0.5
            old_remote_id = str(rs.get("id", "") or "")
            new_pbb_id = _new_id("pbb")
            if old_remote_id:
                remote_id_to_promoted_pbb_id[old_remote_id] = new_pbb_id
            promoted_contact.append(
                {
                    "id": new_pbb_id,
                    "name": str(rs.get("name", "")),
                    "x1": _quantize_m(x1, pos_pl),
                    "y1": _quantize_m(y1, pos_pl),
                    "x2": _quantize_m(x2, pos_pl),
                    "y2": _quantize_m(y2, pos_pl),
                    "category": cat,
                    "categoryMode": mode,
                    "allowedAircraftTypes": [],
                    "pbbCount": 1,
                    "angleDeg": _quantize_m(ang, 2),
                    "apronSiteX": _quantize_m(sx, pos_pl),
                    "apronSiteY": _quantize_m(sy, pos_pl),
                    "boardingWidthM": 5,
                    "boardingHeightM": 15,
                    "pbbArmLenM": max(10.0, stub),
                    "edgeCol": _quantize_m(sx / cell, pos_pl),
                    "edgeRow": _quantize_m(sy / cell, pos_pl),
                }
            )
        remote_stands = kept_remote
        if promoted_contact:
            pbb_stands.extend(promoted_contact)
        if remote_id_to_promoted_pbb_id:
            for al in apron_links:
                pid = str(al.get("pbbId") or "")
                repl = remote_id_to_promoted_pbb_id.get(pid)
                if repl:
                    al["pbbId"] = repl

    grid_layers = {
        "grid": False,
        "image": False,
        "pathLines": True,
        "pathFill": True,
        "standLines": True,
        "standFill": True,
        "islandAreaLines": True,
        "islandAreaFill": True,
        "buildingLines": True,
        "buildingFill": True,
        "textRuler": True,
        "dummyFlight": True,
        "junction": True,
    }

    layout: Dict[str, Any] = {
        "grid": {
            "cols": cols,
            "rows": rows,
            "cellSize": cell,
            "showGrid": False,
            "showImage": False,
            "showRoadWidth": True,
            "showLayoutMarkers": True,
            "layers": grid_layers,
            "layoutImageOverlay": None,
        },
        "networkJunctions": [],
        "Edge": [],
        "terminals": terminals_out,
        "pbbStands": pbb_stands,
        "remoteStands": remote_stands,
        "tempStands": [],
        "holdingPoints": holding_points_out,
        "runwayPaths": runway_paths,
        "runwayTaxiways": [],
        "taxiways": taxiways_out,
        "apronLinks": apron_links,
        "directionModes": [
            {"id": "dm_cw", "name": "CW", "direction": "clockwise"},
            {"id": "dm_ccw", "name": "CCW", "direction": "counter_clockwise"},
            {"id": "dm_both", "name": "Both", "direction": "both"},
        ],
        "layoutMarkers": layout_markers,
        "flights": [],
        "_osmImport": {
            "icao": icao,
            "converter": "osm_to_layout.py",
            "rulesVersion": 2,
            "lonLatOrigin": {"lon": lon0, "lat": lat0},
            "gridOriginShiftM": {"x": _quantize_m(x_off, pos_pl), "y": _quantize_m(y_off, pos_pl)},
            "gridYSpanM": _quantize_m(y_span_m, pos_pl),
            "counts": {
                "runwayPaths": len(runway_paths),
                "taxiways": len(taxiways_out),
                "terminals": len(terminals_out),
                "pbbStands": len(pbb_stands),
                "remoteStands": len(remote_stands),
                "txeIslands": len(layout_markers),
                "holdingPoints": len(holding_points_out),
                "apronLinks": len(apron_links),
            },
        },
    }
    return layout


def write_layout_osm_for_icao(doc: Dict[str, Any], icao: str, out_path: Any) -> None:
    """Write ``{ICAO}_OSM.json`` for a storage document."""
    layout = build_layout_from_map_storage_document(doc, icao)
    out_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
