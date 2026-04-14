"""
Save Layout/load: data/Layout_storage/ Save by name to.
- Save: name If there is Layout_storage/{name}.json, If there is no current_layout.json (Run Simulationdragon).
- POST /api/save-layout: body { "layout": {...}, "name": "optional" } or the entire layout object.
- GET /api/load-layout?name=xxx: Layout_storage/{name}.json return.
- GET /api/export-layout-geometry?name=xxx: polylines + points derived from that layout (designer world x/y).
- POST /api/fetch-airport-map: body { "icao": "RPLL" } → OSM (OpenAirportMap source) → data/map_storage/{ICAO}_map.json
  and ``pages/Layout_Design/Map/osm_to_layout.py`` → data/Layout_storage/{ICAO}_OSM.json (same keys as default layout).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Optional

# Standalone receiver: Layout JSON must be data/Layout_storage/ Save only to
_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_STORAGE_DIR = (_ROOT / "data" / "Layout_storage").resolve()
MAP_STORAGE_DIR = (_ROOT / "data" / "map_storage").resolve()
RESULT_STORAGE_DIR = (_ROOT / "data" / "Result_storage").resolve()
LAYOUT_FILE = LAYOUT_STORAGE_DIR / "current_layout.json"
DEFAULT_LAYOUT_PATH = LAYOUT_STORAGE_DIR / "default_layout.json"

_GRID3D_ASSETS_ROOT = (_ROOT / "pages" / "Layout_Design" / "3D" / "assets").resolve()
_GRID3D_VIEWER_HTML = (_ROOT / "pages" / "Layout_Design" / "3D" / "grid3d-viewer.html").resolve()

LAYOUT_RECEIVER_PORT = 8765
_PORT = LAYOUT_RECEIVER_PORT
_RESERVED_NAMES = frozenset({"current_layout", "default_layout"})


@lru_cache(maxsize=1)
def _airport_overpass_fetch_module():
    path = (_ROOT / "pages" / "Layout_Design" / "Map" / "airport_overpass_fetch.py").resolve()
    spec = importlib.util.spec_from_file_location("airport_overpass_fetch", path)
    if spec is None or spec.loader is None:
        raise ImportError("airport_overpass_fetch spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _osm_to_layout_module():
    path = (_ROOT / "pages" / "Layout_Design" / "Map" / "osm_to_layout.py").resolve()
    spec = importlib.util.spec_from_file_location("osm_to_layout", path)
    if spec is None or spec.loader is None:
        raise ImportError("osm_to_layout spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_airport_map_for_icao(icao_raw: str) -> Dict[str, Any]:
    """Fetch OSM airport airside features and write data/map_storage/{ICAO}_map.json."""
    mod = _airport_overpass_fetch_module()
    icao = mod.sanitize_icao((icao_raw or "").strip())
    if not icao:
        raise ValueError("invalid or missing icao (use 3–4 letter/digit ICAO)")
    doc = mod.build_storage_document(icao)
    MAP_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{icao}_map.json"
    out_path = (MAP_STORAGE_DIR / out_name).resolve()
    if out_path.parent != MAP_STORAGE_DIR:
        raise ValueError("invalid output path")
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    gj = doc.get("geojson") if isinstance(doc.get("geojson"), dict) else {}
    feats = gj.get("features") if isinstance(gj.get("features"), list) else []
    layout_name = f"{icao}_OSM"
    layout_file = f"{layout_name}.json"
    layout_path = f"data/Layout_storage/{layout_file}"
    layout_error: Optional[str] = None
    try:
        ol = _osm_to_layout_module()
        layout_obj = ol.build_layout_from_map_storage_document(doc, icao)
        LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        layout_disk = (LAYOUT_STORAGE_DIR / layout_file).resolve()
        if layout_disk.parent != LAYOUT_STORAGE_DIR:
            raise ValueError("invalid layout output path")
        layout_disk.write_text(json.dumps(layout_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        layout_error = str(e)
    out: Dict[str, Any] = {
        "ok": True,
        "file": out_name,
        "path": f"data/map_storage/{out_name}",
        "featureCount": len(feats),
        "layoutName": layout_name,
        "layoutFile": layout_file,
        "layoutPath": layout_path,
    }
    if layout_error is not None:
        out["layoutError"] = layout_error
    return out


_sim_progress: Dict[str, Any] = {
    "running": False,
    "current": 0,
    "total": 0,
    "percent": 0,
    "error": None,
    "resultFile": None,
    "runningClockLabel": "",
}
_sim_lock = threading.Lock()
# Remove only dangerous characters so they can be used as file names (gap·Korean, etc. allowed)
def _sanitize_layout_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:200] if s else ""


def _remove_legacy_layout_storage_sim_files() -> None:
    """Older builds wrote sim_input/sim_output under Layout_storage; remove if present."""
    for legacy in (LAYOUT_STORAGE_DIR / "sim_input.json", LAYOUT_STORAGE_DIR / "sim_output.json"):
        try:
            if legacy.is_file():
                legacy.unlink()
        except OSError:
            pass


def _safe_layout_path(name: str) -> Optional[Path]:
    safe = _sanitize_layout_name(name)
    if not safe or safe.lower() in _RESERVED_NAMES:
        return None
    return LAYOUT_STORAGE_DIR / f"{safe}.json"


def _layout_path_for_read(name: str) -> Optional[Path]:
    """Load(read)dragon path. default_layout/current_layout Allow all saved names, including."""
    if not name or not (name or "").strip():
        return None
    safe = _sanitize_layout_name((name or "").strip())
    if not safe:
        return None
    path = LAYOUT_STORAGE_DIR / f"{safe}.json"
    if path.is_file():
        return path
    # streamlit etc. other cwdContrast when running on: Based on project root data/Layout_storage
    try:
        cwd_path = Path.cwd() / "data" / "Layout_storage" / f"{safe}.json"
        if cwd_path.is_file():
            return cwd_path
        parent_path = (Path.cwd().parent / "data" / "Layout_storage" / f"{safe}.json")
        if parent_path.is_file():
            return parent_path
    except Exception:
        pass
    return path


def build_layout_geometry_export(name: str) -> Dict[str, Any]:
    """Load Layout_storage/{name}.json and return polylines + points (designer world x/y)."""
    from utils.layout_geometry_export import export_layout_geometry

    n = (name or "").strip()
    if not n:
        raise ValueError("missing name")
    path = _layout_path_for_read(n)
    if not path or not path.is_file():
        raise FileNotFoundError(n)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("layout file must be a JSON object")
    geo = export_layout_geometry(raw)
    return {"ok": True, "layoutName": n, **geo}


def save_layout_to_file(layout: Dict[str, Any], name: Optional[str] = None) -> None:
    """Layout data/Layout_storage Save only to. name If there is no current_layout.json (Run Simulationdragon).
    name this default_layout/current_layout Overwrite even if it is(Save current state) allowance."""
    LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if name:
        safe = _sanitize_layout_name(name)
        if not safe and (name or "").strip():
            raise ValueError(f"Invalid layout name: {name!r}")
        # reservation name(default_layout, current_layout)Also allow overwriting (Paths are in lowercase letters)
        if safe and safe.lower() in _RESERVED_NAMES:
            path = (LAYOUT_STORAGE_DIR / f"{safe.lower()}.json").resolve()
        else:
            path = _safe_layout_path(name)
            if path is None:
                raise ValueError(f"Invalid layout name: {name!r}")
            path = path.resolve()
    else:
        # Run Simulation: certainly Layout_storage/current_layout.json
        path = (LAYOUT_STORAGE_DIR / "current_layout.json").resolve()
    path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")


def list_layout_names():
    """Layout_storage my .json name of the file(Excluding extension) returns a list."""
    try:
        LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if not LAYOUT_STORAGE_DIR.is_dir():
            return []
        names = []
        for p in LAYOUT_STORAGE_DIR.iterdir():
            if p.suffix.lower() == ".json" and p.is_file():
                names.append(p.stem)
        return sorted(names)
    except Exception:
        return []


def _try_resolve_sim_result_path(layout_name_safe: str) -> Optional[Path]:
    """Named result file under Result_storage (and cwd fallbacks for that name)."""
    RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ordered: list[Path] = []
    if layout_name_safe:
        ordered.append(RESULT_STORAGE_DIR / f"{layout_name_safe}_sim_result.json")
        try:
            ordered.append(Path.cwd() / "data" / "Result_storage" / f"{layout_name_safe}_sim_result.json")
            ordered.append(Path.cwd().parent / "data" / "Result_storage" / f"{layout_name_safe}_sim_result.json")
        except Exception:
            pass
    for raw in ordered:
        try:
            p = raw.resolve()
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def _grid3d_asset_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".gltf":
        return "model/gltf+json; charset=utf-8"
    if suf == ".glb":
        return "model/gltf-binary"
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".hdr":
        return "image/vnd.radiance"
    if suf == ".bin":
        return "application/octet-stream"
    return "application/octet-stream"


def _safe_grid3d_asset_file(rel: str) -> Optional[Path]:
    if not rel or not rel.strip():
        return None
    norm = rel.replace("\\", "/").strip("/")
    if not norm or ".." in norm.split("/"):
        return None
    candidate = (_GRID3D_ASSETS_ROOT / norm).resolve()
    try:
        candidate.relative_to(_GRID3D_ASSETS_ROOT)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def delete_layout(name: str) -> None:
    """Layout_storageof that name in json Delete file. default_layout/current_layout cannot be deleted."""
    if not name or (name or "").strip().lower() in _RESERVED_NAMES:
        raise ValueError("default_layout, current_layout cannot be deleted.")
    path = _safe_layout_path(name)
    if path is None:
        raise ValueError(f"Invalid layout name: {name!r}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Layout not found: {name}")
    if LAYOUT_STORAGE_DIR not in path.parents and path.parent.resolve() != LAYOUT_STORAGE_DIR:
        raise ValueError("Invalid path")
    path.unlink()


class LayoutReceiverHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        req_path = (urlparse(self.path).path or self.path or "/").split("?", 1)[0].rstrip("/") or "/"
        if req_path == "/api/fetch-airport-map":
            self.send_response(405)
            self.send_header("Content-Type", "application/json")
            self.send_header("Allow", "POST, OPTIONS")
            self._send_cors()
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "ok": False,
                        "error": "method_not_allowed",
                        "hint": "Use POST with JSON body: {\"icao\":\"RPLL\"}",
                    }
                ).encode("utf-8")
            )
            return
        if req_path == "/api/layout-receiver-health":
            try:
                mtime = Path(__file__).stat().st_mtime
            except OSError:
                mtime = None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "ok": True,
                        "module": "utils.layout_receiver",
                        "fetchAirportMapPost": True,
                        "receiverSourceMtime": mtime,
                        "hint": "POST /api/fetch-airport-map with {\"icao\":\"RPLL\"}. If you still get 404 on POST, another process may be bound to this port with old code — stop it and restart run_app.py.",
                    }
                ).encode("utf-8")
            )
            return
        if self.path.startswith("/api/list-layouts"):
            try:
                names = list_layout_names()
                body = json.dumps({"ok": True, "names": names}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if self.path.startswith("/api/load-sim-result"):
            qs = parse_qs(urlparse(self.path).query)
            names = qs.get("name", [])
            name = (names[0] or "").strip() if names else ""
            safe = _sanitize_layout_name(name)
            path = _try_resolve_sim_result_path(safe)
            if path is None or not path.is_file():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(
                    json.dumps({
                        "ok": False,
                        "error": "not_found",
                        "hint": "Run Pro Sim first, or check data/Result_storage/{name}_sim_result.json",
                    }).encode("utf-8")
                )
                return
            try:
                raw_text = path.read_text(encoding="utf-8")
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and "flight_edge_paths" in parsed:
                        parsed = dict(parsed)
                        parsed.pop("flight_edge_paths", None)
                        body = json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
                    else:
                        body = raw_text
                except json.JSONDecodeError:
                    body = raw_text
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if self.path.startswith("/api/load-layout"):
            qs = parse_qs(urlparse(self.path).query)
            names = qs.get("name", [])
            name = (names[0] or "").strip() if names else ""
            path = _layout_path_for_read(name) if name else None
            try:
                LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            if path is not None:
                try:
                    path = path.resolve()
                except Exception:
                    path = None
            if not path or not path.is_file():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "not_found"}).encode("utf-8"))
                return
            try:
                body = path.read_text(encoding="utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if self.path.startswith("/api/export-layout-geometry"):
            qs = parse_qs(urlparse(self.path).query)
            names = qs.get("name", [])
            name = (names[0] or "").strip() if names else ""
            try:
                payload = build_layout_geometry_export(name)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            except FileNotFoundError:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "not_found"}).encode("utf-8"))
            except ValueError as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if self.path.startswith("/api/sim-progress"):
            with _sim_lock:
                body = json.dumps(_sim_progress).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
            return
        _req_path = (urlparse(self.path).path or "").rstrip("/") or "/"
        if _req_path == "/api/grid3d-viewer-app":
            if not _GRID3D_VIEWER_HTML.is_file():
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(b"grid3d-viewer.html not found on server")
                return
            try:
                html_body = _GRID3D_VIEWER_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(html_body)
            except OSError as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
        if self.path.startswith("/api/grid3d-asset/"):
            parsed = urlparse(self.path)
            rel = unquote((parsed.path or "").strip())
            prefix = "/api/grid3d-asset/"
            if rel.startswith(prefix):
                rel = rel[len(prefix) :].lstrip("/")
            asset_path = _safe_grid3d_asset_file(rel)
            if asset_path is None:
                self.send_response(404)
                self._send_cors()
                self.end_headers()
                return
            try:
                data = asset_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", _grid3d_asset_mime(asset_path))
                self._send_cors()
                self.end_headers()
                self.wfile.write(data)
            except OSError as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
        self.send_response(404)
        self._send_cors()
        self.end_headers()

    def do_POST(self):
        path = (urlparse(self.path).path or self.path).rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        if path == "/api/delete-layout" or path.startswith("/api/delete-layout"):
            try:
                obj = json.loads(body)
                name = (obj.get("name") or "").strip() if isinstance(obj, dict) else ""
                delete_layout(name)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if path == "/api/fetch-airport-map" or path.startswith("/api/fetch-airport-map"):
            try:
                obj = json.loads(body) if body else {}
                icao_raw = ""
                if isinstance(obj, dict):
                    icao_raw = (obj.get("icao") or obj.get("ICAO") or "").strip()
                result = save_airport_map_for_icao(icao_raw)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if path == "/api/run-simulation" or path.startswith("/api/run-simulation"):
            try:
                obj = json.loads(body)
                layout = obj.get("layout", obj) if isinstance(obj, dict) else obj
                layout_name_raw = ""
                if isinstance(obj, dict):
                    layout_name_raw = (
                        obj.get("layoutName") or obj.get("name") or ""
                    ).strip()
                result_stem = _sanitize_layout_name(layout_name_raw) or "default_layout"
                RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                sim_input_path = (RESULT_STORAGE_DIR / f"{result_stem}_sim_input.json").resolve()
                rs_resolved_in = RESULT_STORAGE_DIR.resolve()
                if not (sim_input_path.parent == rs_resolved_in or rs_resolved_in in sim_input_path.parents):
                    raise ValueError("invalid sim input path")
                sim_input_path.write_text(
                    json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _remove_legacy_layout_storage_sim_files()
                with _sim_lock:
                    if _sim_progress["running"]:
                        self.send_response(409)
                        self.send_header("Content-Type", "application/json")
                        self._send_cors()
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "ok": False, "error": "simulation already running"
                        }).encode("utf-8"))
                        return
                    _sim_progress.update(
                        running=True,
                        current=0,
                        total=0,
                        percent=0,
                        error=None,
                        resultFile=None,
                        runningClockLabel="",
                    )
                t = threading.Thread(
                    target=_run_simulation_thread,
                    args=(layout, result_stem),
                    daemon=True,
                )
                t.start()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "message": "simulation started"}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if path != "/api/save-layout" and not path.startswith("/api/save-layout"):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": "not_found", "path": path}).encode("utf-8"))
            return
        try:
            obj = json.loads(body)
            # save name: bodyto "layout" If there is a key and its value is an object → layout + name use
            if isinstance(obj, dict) and "layout" in obj and isinstance(obj.get("layout"), dict):
                layout = obj["layout"]
                name = obj.get("name")
                if isinstance(name, str):
                    name = name.strip() or None
            else:
                layout = obj
                name = None
            save_layout_to_file(layout, name=name)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _format_sim_clock_5min(base_date: str, sim_sec_abs: float) -> str:
    s = (base_date or "").strip() or "2026-03-31"
    try:
        d0 = datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        d0 = datetime(2026, 1, 1)
    snapped = (max(0.0, float(sim_sec_abs)) // 300.0) * 300.0
    dt = d0 + timedelta(seconds=snapped)
    return dt.strftime("%Y-%m-%d %H:%M")


def _wall_elapsed_mm_ss(wall_t0: float) -> str:
    elapsed = max(0, int(time.time() - wall_t0))
    m, sec = divmod(elapsed, 60)
    return f"{m:02d}:{sec:02d}"


def _run_simulation_thread(layout: Dict[str, Any], result_stem: str) -> None:
    try:
        from utils.airside_sim import _deep_get, _load_information_json, run_simulation

        info = _load_information_json()
        base_date = str(
            _deep_get(
                info,
                "tiers",
                "algorithm",
                "simulation",
                "baseDate",
                default="2026-03-31",
            )
        )
        wall_t0 = time.time()

        def _progress_cb(
            current_time: float, total_time: float, sim_time_abs: Optional[float]
        ) -> None:
            pct = int(100 * current_time / total_time) if total_time > 0 else 0
            pct = max(0, min(100, pct))
            wall_part = _wall_elapsed_mm_ss(wall_t0)
            if sim_time_abs is None:
                sim_part = "준비 중"
            else:
                sim_part = _format_sim_clock_5min(base_date, float(sim_time_abs))
            running_clock = f"{sim_part} ({pct}% / {wall_part})"
            with _sim_lock:
                _sim_progress.update(
                    current=current_time,
                    total=total_time,
                    percent=pct,
                    runningClockLabel=running_clock,
                )

        result = run_simulation(layout, progress_cb=_progress_cb)
        output = dict(result) if isinstance(result, dict) else {}
        output.pop("flight_edge_paths", None)
        RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = _sanitize_layout_name(result_stem) or "default_layout"
        named_path = (RESULT_STORAGE_DIR / f"{safe_stem}_sim_result.json").resolve()
        rs_resolved = RESULT_STORAGE_DIR.resolve()
        if not (named_path.parent == rs_resolved or rs_resolved in named_path.parents):
            raise ValueError("invalid result path")
        payload = json.dumps(output, ensure_ascii=False, indent=2, default=str)
        named_path.write_text(payload, encoding="utf-8")
        _remove_legacy_layout_storage_sim_files()
        with _sim_lock:
            _sim_progress.update(
                running=False,
                percent=100,
                resultFile=f"{safe_stem}_sim_result.json",
                error=None,
                runningClockLabel="",
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _sim_lock:
            _sim_progress.update(running=False, error=str(e), runningClockLabel="")


_server: Optional[HTTPServer] = None
_thread: Optional[threading.Thread] = None
_receiver_boot_mtime: Optional[float] = None
_receiver_restart_lock = threading.Lock()


def start_layout_receiver(port: int = LAYOUT_RECEIVER_PORT) -> str:
    """Launch the layout receiving server in the background and connect to it URLreturns.

    When ``layout_receiver.py`` changes on disk (e.g. Streamlit reruns after a git pull), the
    embedded HTTP server is shut down and the module is reloaded so new routes (such as
    ``POST /api/fetch-airport-map``) take effect without a full Python process restart.
    """
    global _server, _thread, _receiver_boot_mtime
    with _receiver_restart_lock:
        try:
            cur_mtime = Path(__file__).stat().st_mtime
        except OSError:
            cur_mtime = None
        if _server is not None:
            unchanged = (
                _receiver_boot_mtime is not None
                and cur_mtime is not None
                and cur_mtime <= _receiver_boot_mtime + 0.01
            )
            if unchanged:
                return f"http://127.0.0.1:{_PORT}"
            try:
                _server.shutdown()
            except Exception:
                pass
            try:
                _server.server_close()
            except Exception:
                pass
            _server = None
            if _thread is not None and _thread.is_alive():
                _thread.join(timeout=5.0)
            _thread = None
            _receiver_boot_mtime = None
            reloaded = importlib.reload(sys.modules[__name__])
            return reloaded.start_layout_receiver(port)
        _server = HTTPServer(("127.0.0.1", port), LayoutReceiverHandler)
        _thread = threading.Thread(target=_server.serve_forever, daemon=True)
        _thread.start()
        _receiver_boot_mtime = cur_mtime
        return f"http://127.0.0.1:{port}"
