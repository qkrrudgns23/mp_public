"""
Derive polylines and point features from a persisted layout dict (same coordinates as the designer / browser).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _is_real_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_xy(obj: Dict[str, Any], xk: str, yk: str) -> Optional[Tuple[float, float]]:
    x, y = obj.get(xk), obj.get(yk)
    if _is_real_number(x) and _is_real_number(y):
        return float(x), float(y)
    return None


def _vertices_to_coords(vertices: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(vertices, list):
        return out
    for v in vertices:
        if not isinstance(v, dict):
            continue
        p = _num_xy(v, "x", "y")
        if p is not None:
            out.append([p[0], p[1]])
    return out


def export_layout_geometry(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Return polylines, closed rings, and point samples from layout JSON."""
    if not isinstance(layout, dict):
        raise TypeError("layout must be a dict")

    grid = layout.get("grid") if isinstance(layout.get("grid"), dict) else {}
    cell_size = grid.get("cellSize")

    polylines: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []

    def push_poly(
        layer: str,
        obj_id: Any,
        name: Any,
        coordinates: List[List[float]],
        *,
        closed: bool = False,
        path_type: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(coordinates) < 2:
            return
        row: Dict[str, Any] = {
            "layer": layer,
            "id": obj_id,
            "name": name,
            "closed": closed,
            "coordinates": coordinates,
        }
        if path_type is not None:
            row["pathType"] = path_type
        if extra:
            row.update(extra)
        polylines.append(row)

    for bucket, layer in (
        ("taxiways", "taxiway"),
        ("runwayPaths", "runway_path"),
        ("runwayTaxiways", "runway_taxiway"),
    ):
        for tw in layout.get(bucket) or []:
            if not isinstance(tw, dict):
                continue
            coords = _vertices_to_coords(tw.get("vertices"))
            push_poly(
                layer,
                tw.get("id"),
                tw.get("name"),
                coords,
                closed=False,
                path_type=tw.get("pathType"),
                extra={"source": bucket},
            )

    for term in layout.get("terminals") or []:
        if not isinstance(term, dict):
            continue
        coords = _vertices_to_coords(term.get("vertices"))
        if len(coords) >= 3:
            push_poly(
                "terminal",
                term.get("id"),
                term.get("name"),
                coords,
                closed=True,
                extra={"buildingType": term.get("buildingType")},
            )

    for lk in layout.get("apronLinks") or []:
        if not isinstance(lk, dict):
            continue
        chain: List[List[float]] = []
        p0 = _num_xy(lk, "tx", "ty")
        if p0 is not None:
            chain.append([p0[0], p0[1]])
        chain.extend(_vertices_to_coords(lk.get("midVertices")))
        if len(chain) >= 2:
            push_poly("apron_link", lk.get("id"), lk.get("name"), chain, extra={"pbbId": lk.get("pbbId"), "taxiwayId": lk.get("taxiwayId")})

    for j in layout.get("networkJunctions") or []:
        if not isinstance(j, dict):
            continue
        p = _num_xy(j, "x", "y")
        if p is not None:
            points.append({"layer": "network_junction", "x": p[0], "y": p[1]})

    for hp in layout.get("holdingPoints") or []:
        if not isinstance(hp, dict):
            continue
        p = _num_xy(hp, "x", "y")
        if p is not None:
            points.append(
                {
                    "layer": "holding_point",
                    "id": hp.get("id"),
                    "name": hp.get("name"),
                    "x": p[0],
                    "y": p[1],
                    "hpKind": hp.get("hpKind"),
                }
            )

    for pb in layout.get("pbbStands") or []:
        if not isinstance(pb, dict):
            continue
        c1 = _num_xy(pb, "x1", "y1")
        c2 = _num_xy(pb, "x2", "y2")
        if c1 and c2:
            push_poly("pbb_stand_edge", pb.get("id"), pb.get("name"), [list(c1), list(c2)], extra={"category": pb.get("category")})
            mx, my = (c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5
            points.append({"layer": "pbb_stand_center", "id": pb.get("id"), "name": pb.get("name"), "x": mx, "y": my})

    for key, layer in (("remoteStands", "remote_stand"), ("tempStands", "temp_stand")):
        for st in layout.get(key) or []:
            if not isinstance(st, dict):
                continue
            p = _num_xy(st, "x", "y")
            if p is not None:
                points.append({"layer": layer, "id": st.get("id"), "name": st.get("name"), "x": p[0], "y": p[1]})

    for m in layout.get("layoutMarkers") or []:
        if not isinstance(m, dict):
            continue
        kind = str(m.get("kind") or "")
        pts = _vertices_to_coords(m.get("points"))
        if len(pts) >= 2:
            closed = kind in ("island", "area") and len(pts) >= 3
            push_poly(f"layout_marker_{kind}", m.get("id"), m.get("name"), pts, closed=closed, extra={"markerKind": kind})
        elif len(pts) == 1:
            points.append({"layer": f"layout_marker_{kind}", "id": m.get("id"), "name": m.get("name"), "x": pts[0][0], "y": pts[0][1]})

    return {
        "coordinateSpace": "layout_world_xy",
        "description": "x/y match vertices in Layout_storage JSON (designer world coordinates, same units as cellSize grid).",
        "gridCellSize": cell_size,
        "polylines": polylines,
        "points": points,
        "counts": {"polylines": len(polylines), "points": len(points)},
    }
