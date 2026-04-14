"""
Airport map bundle for data/map_storage/{ICAO}_map.json

Primary source: OpenAirportMap public API (same OSM JSON the site passes to osmtogeojson in the browser).
  GET https://openairportmap.org/api/airport/{ICAO}

Fallback: Overpass API if that endpoint fails.

GeoJSON: ``osm2geojson.json2geojson`` (pip: osm2geojson) — equivalent role to the site’s lib/osmtogeojson.js.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

OPENAIRPORTMAP_API = "https://openairportmap.org/api/airport/{icao}"

OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DMK-airside-sim/1.0; +https://github.com/)",
    "Accept": "application/json,*/*;q=0.9",
}


def sanitize_icao(code: str) -> Optional[str]:
    s = (code or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,4}", s):
        return None
    return s


def _oam_api_url(icao: str) -> str:
    safe = icao.replace(";", "_")
    return OPENAIRPORTMAP_API.format(icao=urllib.parse.quote(safe, safe=""))


def fetch_osm_from_openairportmap(icao: str) -> Dict[str, Any]:
    url = _oam_api_url(icao)
    req = urllib.request.Request(url, headers=_FETCH_HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    t = raw.lstrip()
    if t.startswith("<"):
        raise RuntimeError("OpenAirportMap API returned HTML (not JSON).")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("OpenAirportMap response is not a JSON object")
    els = data.get("elements")
    if not isinstance(els, list):
        raise ValueError("OpenAirportMap JSON missing elements[]")
    return data


def _overpass_query_airport_relation(icao: str) -> str:
    return f"""[out:json][timeout:120];
(
  relation["icao"="{icao}"]["aeroway"="aerodrome"];
);
(._;>;);
out geom;
"""


def _overpass_query_any_icao_tag(icao: str) -> str:
    return f"""[out:json][timeout:90];
(
  nwr["icao"="{icao}"];
);
out geom;
"""


def _post_overpass(endpoint: str, query: str, timeout_sec: float = 140.0) -> Dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={**_FETCH_HEADERS, "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    t = raw.lstrip()
    if t.startswith("<"):
        raise RuntimeError("Overpass returned HTML (server busy or error page).")
    return json.loads(raw)


def fetch_overpass_for_icao(icao: str) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None
    queries = [_overpass_query_airport_relation(icao), _overpass_query_any_icao_tag(icao)]
    for q in queries:
        for ep in OVERPASS_ENDPOINTS:
            try:
                return _post_overpass(ep, q)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as e:
                last_err = e
                continue
    raise RuntimeError(f"Overpass fetch failed for {icao}: {last_err!r}")


def fetch_osm_for_icao(icao: str) -> tuple[Dict[str, Any], str, Optional[str]]:
    """Return (osm_dict, data_source_label, api_url_used_or_none)."""
    try:
        o = fetch_osm_from_openairportmap(icao)
        return o, "openairportmap.org/api/airport", _oam_api_url(icao)
    except Exception:
        o = fetch_overpass_for_icao(icao)
        return o, "overpass", None


def _osm_to_geojson(osm: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import osm2geojson  # type: ignore[import-untyped]

        return osm2geojson.json2geojson(osm)
    except ImportError as e:
        raise RuntimeError(
            "Package osm2geojson is required for full GeoJSON export (same as OpenAirportMap’s osmtogeojson). "
            "Install: pip install osm2geojson"
        ) from e


def build_storage_document(icao: str) -> Dict[str, Any]:
    icao_u = sanitize_icao(icao)
    if not icao_u:
        raise ValueError("invalid ICAO code")
    parsed, source, api_url = fetch_osm_for_icao(icao_u)
    elements = parsed.get("elements")
    n_el = len(elements) if isinstance(elements, list) else 0
    gj = _osm_to_geojson(parsed)
    feats = gj.get("features") if isinstance(gj, dict) else None
    n_feat = len(feats) if isinstance(feats, list) else 0
    out: Dict[str, Any] = {
        "icao": icao_u,
        "openAirportMapUrl": f"https://openairportmap.org/{icao_u}",
        "openStreetMapAttribution": "Data © OpenStreetMap contributors, ODbL 1.0.",
        "fetchedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataSource": source,
        "openAirportMapApiUrl": api_url,
        "elementCount": n_el,
        "geojsonFeatureCount": n_feat,
        "remark": parsed.get("remark"),
        "osm": parsed,
        "geojson": gj,
    }
    osm3s = parsed.get("osm3s")
    if isinstance(osm3s, dict):
        out["osmTimestamp"] = osm3s.get("timestamp_osm_base")
    return out
