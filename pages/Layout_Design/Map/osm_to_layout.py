"""
Convert a saved OpenAirportMap / Overpass bundle (``data/map_storage/{ICAO}_map.json``)
into a designer layout JSON matching ``data/Layout_storage/default_layout.json`` shape.

Coordinate system: local ENU-style metres with origin at the south-west corner of the
computed grid (all x,y >= 0). Grid cell size matches layout ``cellSize`` (metres per cell edge).
"""
from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

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
    },
    "projection": {
        "kind": "local_enu_metres",
        "earth_radius_m": 6378137.0,
        "origin_lon_lat": "bounds_center",
        "axes": "x=east_metres, y=north_metres; layout origin SW corner after shift",
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
            "parking_position",
            "jet_bridge",
        ),
        "parking_position_path_type": "apron_taxiway",
        "jet_bridge_path_type": "apron_taxiway",
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
        "holding_points": "Not derived from OSM in v1.",
        "apron_links": "Not derived from OSM in v1.",
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
    "aeroway=parking_position": "taxiways pathType apron_taxiway (lead-in)",
    "aeroway=jet_bridge": "taxiways pathType apron_taxiway",
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


def _geom_to_xy_shape(
    geom: Dict[str, Any],
    lon0: float,
    lat0: float,
    r_earth: float,
    x_off: float,
    y_off: float,
):
    def conv_xy(lon: float, lat: float) -> Tuple[float, float]:
        x, y = _project_lonlat(lon, lat, lon0, lat0, r_earth)
        return x - x_off, y - y_off

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


def _line_vertices_xy(geom: Dict[str, Any], lon0: float, lat0: float, r: float, x_off: float, y_off: float) -> List[Dict[str, float]]:
    g = _geom_to_xy_shape(geom, lon0, lat0, r, x_off, y_off)
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [{"x": float(x), "y": float(y)} for x, y in g.coords]
    return []


def _polygon_vertices_xy(geom: Dict[str, Any], lon0: float, lat0: float, r: float, x_off: float, y_off: float) -> List[Dict[str, float]]:
    g = _geom_to_xy_shape(geom, lon0, lat0, r, x_off, y_off)
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

    terminal_polys: List[Polygon] = []
    parking_lines: List[Tuple[LineString, Dict[str, Any]]] = []
    taxi_lines: List[Tuple[LineString, Dict[str, str], str]] = []  # line, tags, aeroway
    runway_lines: List[Tuple[LineString, Dict[str, str]]] = []
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
        gxy = _geom_to_xy_shape(geom, lon0, lat0, r_earth, x_off, y_off)

        if aw in rules["terminals"]["aeroway_polygon_as_terminal"]:
            verts = _polygon_vertices_xy(geom, lon0, lat0, r_earth, x_off, y_off)
            if len(verts) >= 3:
                terminal_polys.append(Polygon([(v["x"], v["y"]) for v in verts]))
        if aw == "parking_position" and isinstance(gxy, LineString) and len(gxy.coords) >= 2:
            parking_lines.append((gxy, tags))
        if aw in rules["taxiways"]["include_aeroway_linestrings"] and aw != "parking_position":
            if isinstance(gxy, LineString) and len(gxy.coords) >= 2:
                taxi_lines.append((gxy, tags, aw))
        if aw in rules["runways"]["aeroway_line_as_runway_centerline"]:
            if isinstance(gxy, LineString) and len(gxy.coords) >= 2:
                runway_lines.append((gxy, tags))

        if aw == "gate" and geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                gx, gy = _project_lonlat(float(coords[0]), float(coords[1]), lon0, lat0, r_earth)
                gates.append((gx - x_off, gy - y_off, tags, fid))

        nav_cfg = rules["navigationaid_islands"]
        if nav_cfg.get("enabled") is True and aw == "navigationaid":
            nav = str(tags.get("navigationaid", "")).lower()
            sub = str(nav_cfg["island_navigationaid_substr"]).lower()
            if sub in nav and geom.get("type") == "Point":
                coords = geom.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    px, py = _project_lonlat(float(coords[0]), float(coords[1]), lon0, lat0, r_earth)
                    txe_points.append((px - x_off, py - y_off))

    terminal_union = unary_union(terminal_polys) if terminal_polys else None
    leadin_max = float(rules["stands"]["leadin_to_terminal_max_m"])
    gate_search = float(rules["stands"]["gate_to_leadin_search_m"])
    wall_len = float(rules["stands"]["pbb_wall_length_m"])
    stub = float(rules["stands"]["pbb_bridge_stub_m"])
    apron_near = float(rules["taxiways"]["apron_taxiway_near_terminal_m"])

    pbb_stands: List[Dict[str, Any]] = []
    remote_stands: List[Dict[str, Any]] = []

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
            ang = 0.0
            if terminal_union is not None and not terminal_union.is_empty:
                try:
                    nr = _union_exterior(terminal_union)
                    if nr is not None:
                        nearest_pt = nr.interpolate(nr.project(Point(gx, gy)))
                        ang = math.degrees(math.atan2(gy - nearest_pt.y, gx - nearest_pt.x))
                except Exception:
                    ang = 0.0
            rad = math.radians(ang)
            dx, dy = math.cos(rad) * stub, math.sin(rad) * stub
            pbx1, pby1 = gx - math.cos(rad) * wall_len * 0.3, gy - math.sin(rad) * wall_len * 0.3
            pbx2, pby2 = pbx1 + math.cos(rad) * wall_len, pby1 + math.sin(rad) * wall_len
            pbb_stands.append(
                {
                    "id": _new_id("pbb"),
                    "name": name,
                    "x1": pbx1,
                    "y1": pby1,
                    "x2": pbx2,
                    "y2": pby2,
                    "category": cat,
                    "categoryMode": mode,
                    "allowedAircraftTypes": [],
                    "pbbCount": int(rules["stands"]["pbb_default_pbb_count"]),
                    "angleDeg": ang,
                    "apronSiteX": gx + dx,
                    "apronSiteY": gy + dy,
                    "boardingWidthM": 5,
                    "boardingHeightM": 15,
                    "pbbArmLenM": max(10.0, stub),
                    "edgeCol": gx / cell,
                    "edgeRow": gy / cell,
                }
            )
        else:
            remote_stands.append(
                {
                    "id": _new_id("remote"),
                    "name": name,
                    "x": gx,
                    "y": gy,
                    "category": cat,
                    "angleDeg": 0,
                    "categoryMode": mode,
                    "allowedAircraftTypes": [],
                    "allowedTerminals": [],
                }
            )

    runway_paths: List[Dict[str, Any]] = []
    for ln, tags in runway_lines:
        verts = [{"x": float(x), "y": float(y)} for x, y in ln.coords]
        if len(verts) < 2:
            continue
        w = _float_tag(tags, "width", float(rules["taxiways"]["runway_default_width_m"]))
        rw_id = _new_id("rwy")
        runway_paths.append(
            {
                "id": rw_id,
                "name": _safe_name(tags.get("ref") or tags.get("name"), rw_id),
                "vertices": verts,
                "width": w,
                "direction": str(rules["runways"]["default_direction"]),
                "minArrVelocity": 15,
                "lineupDistM": 0,
                "avgMoveVelocity": 10,
                "startDisplacedThresholdM": 0,
                "startBlastPadM": 0,
                "endDisplacedThresholdM": 0,
                "endBlastPadM": 0,
                "pavement": str(tags.get("surface", "asphalt") or "asphalt"),
            }
        )

    taxiways_out: List[Dict[str, Any]] = []
    for ln, tags, aw in taxi_lines:
        verts = [{"x": float(x), "y": float(y)} for x, y in ln.coords]
        if len(verts) < 2:
            continue
        w = _float_tag(tags, "width", float(rules["taxiways"]["default_width_m"]))
        path_type = "taxiway"
        if aw == "jet_bridge":
            path_type = str(rules["taxiways"]["jet_bridge_path_type"])
        elif aw == "taxiway":
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

    for ln, tags in parking_lines:
        verts = [{"x": float(x), "y": float(y)} for x, y in ln.coords]
        if len(verts) < 2:
            continue
        w = _float_tag(tags, "width", float(rules["taxiways"]["default_width_m"]))
        tw_id = _new_id("pp")
        taxiways_out.append(
            {
                "id": tw_id,
                "name": _safe_name(tags.get("ref") or tags.get("name"), f"parking-{tw_id}"),
                "vertices": verts,
                "width": w,
                "direction": "both",
                "avgMoveVelocity": 8,
                "pathType": str(rules["taxiways"]["parking_position_path_type"]),
                "pavement": "concrete",
            }
        )

    layout_markers: List[Dict[str, Any]] = []
    if rules["navigationaid_islands"].get("enabled") is True:
        half = float(rules["navigationaid_islands"]["island_square_half_side_m"])
        ow = float(rules["navigationaid_islands"]["marker_outer_width_m"])
        iw = float(rules["navigationaid_islands"]["marker_inner_width_m"])
        for px, py in txe_points:
            pts = [
                {"x": px - half, "y": py - half},
                {"x": px + half, "y": py - half},
                {"x": px + half, "y": py + half},
                {"x": px - half, "y": py + half},
            ]
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
        verts = _polygon_vertices_xy(geom, lon0, lat0, r_earth, x_off, y_off)
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
        "holdingPoints": [],
        "runwayPaths": runway_paths,
        "runwayTaxiways": [],
        "taxiways": taxiways_out,
        "apronLinks": [],
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
            "rulesVersion": 1,
            "lonLatOrigin": {"lon": lon0, "lat": lat0},
            "gridOriginShiftM": {"x": x_off, "y": y_off},
            "counts": {
                "runwayPaths": len(runway_paths),
                "taxiways": len(taxiways_out),
                "terminals": len(terminals_out),
                "pbbStands": len(pbb_stands),
                "remoteStands": len(remote_stands),
                "txeIslands": len(layout_markers),
            },
        },
    }
    return layout


def write_layout_osm_for_icao(doc: Dict[str, Any], icao: str, out_path: Any) -> None:
    """Write ``{ICAO}_OSM.json`` for a storage document."""
    layout = build_layout_from_map_storage_document(doc, icao)
    out_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
