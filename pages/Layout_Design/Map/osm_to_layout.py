"""
Convert a saved OpenAirportMap / Overpass bundle (``data/map_storage/{ICAO}_map.json``)
into a designer layout JSON matching ``data/Layout_storage/default_layout.json`` shape.

Coordinate system: local ENU-style metres with origin at the south-west corner of the
computed grid (all x,y >= 0). Grid cell size matches layout ``cellSize`` (metres per cell edge).
Layout Y is mirrored about the horizontal midline of the grid (``y' = rows*cellSize - y``)
so that the designer matches the expected airside orientation.
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
        "axes": "x=east_metres, y=north_metres in projection; layout Y then mirrored in grid height (rows*cellSize)",
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
    "aeroway=jet_bridge": "pbbCount on contact stands only; not a taxiway path",
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
    """Projected metres shifted to grid origin, then Y mirrored in full grid height."""
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


def _count_jet_bridges_near(
    stand_pt: Tuple[float, float],
    gate_pt: Tuple[float, float],
    parking_ln: Optional[LineString],
    jet_lines: Sequence[LineString],
    near_m: float,
) -> int:
    if not jet_lines:
        return 0
    ps = Point(stand_pt[0], stand_pt[1])
    pg = Point(gate_pt[0], gate_pt[1])
    buf = ps.buffer(near_m).union(pg.buffer(near_m))
    if parking_ln is not None and not parking_ln.is_empty:
        buf = unary_union([buf, parking_ln.buffer(8.0)])
    n = 0
    for jln in jet_lines:
        if jln is None or jln.is_empty:
            continue
        if jln.intersects(buf):
            n += 1
    return n


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
    parking_lines: List[Tuple[LineString, Dict[str, Any]]] = []
    taxi_lines: List[Tuple[LineString, Dict[str, str], str]] = []  # line, tags, aeroway
    runway_lines: List[Tuple[LineString, Dict[str, str]]] = []
    jet_bridge_lines: List[LineString] = []
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
                terminal_polys.append(Polygon([(v["x"], v["y"]) for v in verts]))
        if aw == "parking_position" and isinstance(gxy, LineString) and len(gxy.coords) >= 2:
            parking_lines.append((gxy, tags))
        if aw == "jet_bridge" and isinstance(gxy, LineString) and len(gxy.coords) >= 2:
            jet_bridge_lines.append(gxy)
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
    apron_near = float(rules["taxiways"]["apron_taxiway_near_terminal_m"])
    taxi_mate_geoms = [ln for ln, _, _ in taxi_lines]

    pbb_stands: List[Dict[str, Any]] = []
    remote_stands: List[Dict[str, Any]] = []
    stand_records: List[Dict[str, Any]] = []

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
        cat = str(rules["stands"]["default_category"])
        mode = str(rules["stands"]["default_category_mode"])

        if contact:
            sx, sy = gx, gy
            parking_ln_gate: Optional[LineString] = None
            if nearest_pl is not None and isinstance(nearest_pl, LineString):
                parking_ln_gate = nearest_pl
                st, _txy = _parking_stand_and_taxi_endpoints(nearest_pl, taxi_mate_geoms)
                sx, sy = st
            jb_near_m = float(rules["taxiways"].get("jet_bridge_near_stand_m", 55.0))
            jb_n = _count_jet_bridges_near((sx, sy), (gx, gy), parking_ln_gate, jet_bridge_lines, jb_near_m)
            pbb_count = max(1, jb_n)

            ang = 0.0
            if parking_ln_gate is not None:
                _, taxi_ep = _parking_stand_and_taxi_endpoints(parking_ln_gate, taxi_mate_geoms)
                ang_lead = math.degrees(math.atan2(sy - taxi_ep[1], sx - taxi_ep[0]))
                ang = (ang_lead + 180.0) % 360.0
            elif terminal_union is not None and not terminal_union.is_empty:
                try:
                    nr = _union_exterior(terminal_union)
                    if nr is not None:
                        nearest_pt = nr.interpolate(nr.project(Point(sx, sy)))
                        ang_t = math.degrees(math.atan2(sy - nearest_pt.y, sx - nearest_pt.x))
                        ang = (ang_t + 180.0) % 360.0
                except Exception:
                    ang = 0.0
            rad = math.radians(ang)
            dx, dy = math.cos(rad) * stub, math.sin(rad) * stub
            pbx1, pby1 = sx - math.cos(rad) * wall_len * 0.3, sy - math.sin(rad) * wall_len * 0.3
            pbx2, pby2 = pbx1 + math.cos(rad) * wall_len, pby1 + math.sin(rad) * wall_len
            pbb_id = _new_id("pbb")
            pbb_obj: Dict[str, Any] = {
                "id": pbb_id,
                "name": name,
                "x1": _quantize_m(pbx1, pos_pl),
                "y1": _quantize_m(pby1, pos_pl),
                "x2": _quantize_m(pbx2, pos_pl),
                "y2": _quantize_m(pby2, pos_pl),
                "category": cat,
                "categoryMode": mode,
                "allowedAircraftTypes": [],
                "pbbCount": int(pbb_count),
                "angleDeg": ang,
                "apronSiteX": _quantize_m(sx + dx, pos_pl),
                "apronSiteY": _quantize_m(sy + dy, pos_pl),
                "boardingWidthM": 5,
                "boardingHeightM": 15,
                "pbbArmLenM": max(10.0, stub),
                "edgeCol": _quantize_m(sx / cell, pos_pl),
                "edgeRow": _quantize_m(sy / cell, pos_pl),
            }
            pbb_stands.append(pbb_obj)
            stand_records.append(
                {
                    "kind": "contact",
                    "pbb": pbb_obj,
                    "gx": gx,
                    "gy": gy,
                    "nearest_line": nearest_pl,
                    "d_pl": d_pl,
                }
            )
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
        path_type = "taxiway"
        if aw == "taxiway":
            if terminal_union is not None and not terminal_union.is_empty:
                buf = ln.buffer(apron_near)
                if buf.intersects(terminal_union):
                    path_type = "apron_taxiway"
        tw_id = _new_id("tw")
        taxiways_out.append(
            {
                "id": tw_id,
                "name": _safe_name(tags.get("ref") or tags.get("name"), f"{aw}-{tw_id}"),
                "vertices": verts,
                "width": w,
                "direction": "both",
                "avgMoveVelocity": 10,
                "pathType": path_type,
                "pavement": str(tags.get("surface", "asphalt") or "asphalt"),
            }
        )

    taxi_spines: List[Tuple[LineString, str]] = []
    for tw in taxiways_out:
        verts = tw.get("vertices")
        if not isinstance(verts, list) or len(verts) < 2:
            continue
        pt = str(tw.get("pathType") or "taxiway")
        if pt not in ("taxiway", "apron_taxiway"):
            continue
        tw_ln = LineString([(float(v["x"]), float(v["y"])) for v in verts if isinstance(v, dict)])
        if tw_ln.is_empty or len(tw_ln.coords) < 2:
            continue
        taxi_spines.append((tw_ln, str(tw["id"])))

    apron_links: List[Dict[str, Any]] = []
    lk_i = 0
    for rec in stand_records:
        if rec.get("kind") != "contact":
            continue
        pbb = rec.get("pbb")
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
                "pbbId": str(pbb["id"]),
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
