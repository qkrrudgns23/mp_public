"""
Save Layout/load: data/Layout_storage/ Save by name to.
- Save: name If there is Layout_storage/{name}.json, If there is no current_layout.json (Run Simulationdragon).
- POST /api/save-layout: body { "layout": {...}, "name": "optional" } or the entire layout object.
- GET /api/load-layout?name=xxx: Layout_storage/{name}.json return.
- GET /api/export-layout-geometry?name=xxx: polylines + points derived from that layout (designer world x/y).
- GET /api/airport-map-exists?icao=RPLL → ``airport_map_file_status`` → { ok, exists, file, icao }
- POST /api/fetch-airport-map: body { "icao": "RPLL" } → download OSM bundle, write ``{ICAO}_map.json``, build ``{ICAO}_OSM.json``.
- POST /api/process-stored-airport-map: body { "icao": "RPLL" } → read saved ``{ICAO}_map.json`` only, rebuild ``{ICAO}_OSM.json``.
- POST /api/ai-chat: body ``{ "messages": [ {"role":"user","content":"..."} ], "model": "kimi-k2.5" }`` → Moonshot Kimi (OpenAI-compatible). Set ``MOONSHOT_API_KEY`` (or ``KIMI_API_KEY``) in the process environment, or in project-root ``.env`` (loaded at import; not sent from the browser).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import time
from functools import lru_cache
from typing import Any, Dict, Optional


def _dbg_layout_sim_8ab4c9(payload: Dict[str, Any]) -> None:
    """Session 8ab4c9: NDJSON probe for Layout ProSim vs harness timings."""
    # #region agent log
    try:
        log_path = (_ROOT / "debug-8ab4c9.log").resolve()
        row = dict(payload)
        row.setdefault("sessionId", "8ab4c9")
        row.setdefault("timestamp", int(time.time() * 1000))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as _fp:
            _fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


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


def _load_dotenv_from_project_root() -> None:
    """Load project-root ``.env`` into ``os.environ`` if the file exists. Does not override existing vars."""
    p = _ROOT / ".env"
    if not p.is_file():
        return
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].lstrip()
        if "=" not in s:
            continue
        k, _, rest = s.partition("=")
        k = k.strip()
        if not k:
            continue
        v = rest.strip()
        if not v:
            continue
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k not in os.environ:
            os.environ[k] = v


_load_dotenv_from_project_root()


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


def airport_map_file_status(icao_raw: str) -> Dict[str, Any]:
    """Whether ``data/map_storage/{ICAO}_map.json`` exists (after ICAO sanitize)."""
    mod = _airport_overpass_fetch_module()
    icao = mod.sanitize_icao((icao_raw or "").strip())
    if not icao:
        return {"ok": False, "error": "invalid_icao", "exists": False, "icao": "", "file": ""}
    out_name = f"{icao}_map.json"
    p = (MAP_STORAGE_DIR / out_name).resolve()
    if p.parent != MAP_STORAGE_DIR:
        return {"ok": False, "error": "invalid_path", "exists": False, "icao": icao, "file": out_name}
    exists = p.is_file()
    return {"ok": True, "icao": icao, "exists": bool(exists), "file": out_name}


def process_stored_airport_map_for_icao(icao_raw: str) -> Dict[str, Any]:
    """Read existing ``{ICAO}_map.json`` and regenerate ``{ICAO}_OSM.json`` (no network fetch)."""
    mod = _airport_overpass_fetch_module()
    icao = mod.sanitize_icao((icao_raw or "").strip())
    if not icao:
        raise ValueError("invalid or missing icao (use 3–4 letter/digit ICAO)")
    out_name = f"{icao}_map.json"
    out_path = (MAP_STORAGE_DIR / out_name).resolve()
    if out_path.parent != MAP_STORAGE_DIR:
        raise ValueError("invalid output path")
    if not out_path.is_file():
        raise FileNotFoundError(f"No saved map file: data/map_storage/{out_name}")
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("saved map file is not a JSON object")
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
        "fromSavedMapOnly": True,
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


def _proxy_kimi_chat_completions(
    messages: list, model: str
) -> tuple[bool, str, Optional[str]]:
    """
    Call Moonshot OpenAI-compatible chat completions. Returns (ok, reply_or_empty, error_detail).
    """
    api_key = (
        (os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY") or "").strip()
    )
    if not api_key:
        return False, "", "missing_api_key"
    base = (os.environ.get("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1").rstrip("/")
    url = f"{base}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model or "kimi-k2.5",
        "messages": messages,
        "temperature": 0.6,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:2000]
        except Exception:
            pass
        return False, "", f"upstream_http_{e.code}:{detail}"
    except urllib.error.URLError as e:
        return False, "", f"upstream_url:{e.reason!r}"
    except Exception as e:
        return False, "", str(e)
    try:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return False, "", "no_choices"
        msg0 = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = (msg0 or {}).get("content") if isinstance(msg0, dict) else None
        if content is None:
            return False, "", "no_content"
        return True, str(content).strip(), None
    except Exception as e:
        return False, "", str(e)


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

    def _request_origin(self) -> str:
        host = (self.headers.get("Host") or f"127.0.0.1:{_PORT}").strip()
        return f"http://{host}"

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        req_path = (urlparse(self.path).path or self.path or "/").split("?", 1)[0].rstrip("/") or "/"
        if req_path in ("/", "/layout-design"):
            try:
                from utils.layout_designer_standalone import build_designer_html

                qs = parse_qs(urlparse(self.path).query)
                load_layout = (qs.get("load_layout", [""])[0] or "").strip() or None
                origin = self._request_origin()
                body = build_designer_html(
                    layout_api_url=origin,
                    grid3d_asset_api_url=origin,
                    load_layout=load_layout,
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8", errors="replace"))
            return
        if req_path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "ok": True,
                        "service": "standalone-layout-server",
                        "layoutApiUrl": self._request_origin(),
                    }
                ).encode("utf-8")
            )
            return
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
        if req_path == "/api/airport-map-exists":
            try:
                qs = parse_qs(urlparse(self.path).query)
                icao_q = (qs.get("icao", [""])[0] or "").strip()
                payload = airport_map_file_status(icao_q)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e), "exists": False}).encode("utf-8"))
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
                        "aiChatPost": True,
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
        if path == "/api/process-stored-airport-map" or path.startswith("/api/process-stored-airport-map"):
            try:
                obj = json.loads(body) if body else {}
                icao_raw = ""
                if isinstance(obj, dict):
                    icao_raw = (obj.get("icao") or obj.get("ICAO") or "").strip()
                result = process_stored_airport_map_for_icao(icao_raw)
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
                _post_t0 = time.time()
                _t_parse0 = time.time()
                obj = json.loads(body)
                _parse_ms = (time.time() - _t_parse0) * 1000
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
                _t_dump0 = time.time()
                sim_input_json = json.dumps(
                    layout,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                _dump_ms = (time.time() - _t_dump0) * 1000
                _t_write0 = time.time()
                sim_input_path.write_text(sim_input_json, encoding="utf-8")
                _write_ms = (time.time() - _t_write0) * 1000
                try:
                    _si_bytes = sim_input_path.stat().st_size
                except OSError:
                    _si_bytes = -1
                _post_handler_ms = (time.time() - _post_t0) * 1000
                # #region agent log
                _dbg_layout_sim_8ab4c9(
                    {
                        "runId": "prosim-http",
                        "hypothesisId": "H_SERIALIZE",
                        "location": "layout_receiver.py:POST/api/run-simulation",
                        "message": "request_parse_dump_write_ms",
                        "data": {
                            "resultStem": result_stem,
                            "request_parse_ms": round(_parse_ms, 3),
                            "sim_input_json_dumps_ms": round(_dump_ms, 3),
                            "sim_input_disk_write_ms": round(_write_ms, 3),
                            "post_handler_before_thread_ms": round(_post_handler_ms, 3),
                            "request_body_bytes": len(body.encode("utf-8"))
                            if isinstance(body, str)
                            else (len(body) if isinstance(body, (bytes, bytearray)) else 0),
                            "sim_input_bytes": _si_bytes,
                        },
                    }
                )
                # #endregion
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
                    args=(sim_input_path, result_stem),
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
        if path == "/api/ai-chat" or path.startswith("/api/ai-chat"):
            try:
                obj = json.loads(body) if body else {}
                if not isinstance(obj, dict):
                    raise ValueError("invalid_json_object")
                messages = obj.get("messages")
                if not isinstance(messages, list) or not messages:
                    single = (obj.get("message") or "").strip()
                    if not single:
                        raise ValueError("missing_messages")
                    messages = [{"role": "user", "content": single}]
                clean: list = []
                for m in messages[:40]:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role") or "").strip().lower()
                    content = m.get("content")
                    if role not in ("user", "assistant", "system"):
                        continue
                    if content is None:
                        continue
                    text = str(content).strip()
                    if not text:
                        continue
                    if len(text) > 12000:
                        text = text[:12000]
                    clean.append({"role": role, "content": text})
                if not clean:
                    raise ValueError("no_valid_messages")
                model_raw = (obj.get("model") or os.environ.get("MOONSHOT_MODEL") or "kimi-k2.5")
                model = str(model_raw).strip() or "kimi-k2.5"
                ok, reply, err = _proxy_kimi_chat_completions(clean, model)
                if not ok:
                    code = 503 if err == "missing_api_key" else 502
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self._send_cors()
                    self.end_headers()
                    hint = (
                        "Set MOONSHOT_API_KEY (or KIMI_API_KEY) in the environment and restart the layout receiver."
                        if err == "missing_api_key"
                        else ""
                    )
                    self.wfile.write(
                        json.dumps(
                            {"ok": False, "error": err or "chat_failed", "hint": hint},
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(
                    json.dumps({"ok": True, "reply": reply}, ensure_ascii=False).encode("utf-8")
                )
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


def _run_simulation_elapsed_sec_label(t0_mono: float) -> str:
    """Wall seconds for ``run_simulation`` only (``time.monotonic()`` span)."""
    el = max(0.0, float(time.monotonic()) - float(t0_mono))
    return f"{int(el):02d}sec"


def _prosim_worker_popen_extra() -> Dict[str, Any]:
    """Windows: keep Streamlit-spawned worker out of inherited low scheduling class."""
    if os.name != "nt":
        return {}
    flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    flags |= int(getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0x00008000))
    return {"creationflags": flags}


def _run_simulation_thread(sim_input_path: Path, result_stem: str) -> None:
    try:
        thread_wall_t0 = time.time()
        publish_throttle: Dict[str, Optional[float]] = {"last_mono": None}
        progress_publish_ctr = {"pub": 0}
        # #region agent log
        progress_pub_max_mono_elapsed: Dict[str, float] = {"sec": 0.0}
        # #endregion

        RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = _sanitize_layout_name(result_stem) or "default_layout"
        named_path = (RESULT_STORAGE_DIR / f"{safe_stem}_sim_result.json").resolve()
        rs_resolved = RESULT_STORAGE_DIR.resolve()
        if not (named_path.parent == rs_resolved or rs_resolved in named_path.parents):
            raise ValueError("invalid result path")
        sim_input_path = Path(sim_input_path).resolve()
        if not (
            sim_input_path.parent == rs_resolved or rs_resolved in sim_input_path.parents
        ):
            raise ValueError("invalid sim input path")
        progress_path = (RESULT_STORAGE_DIR / f".{safe_stem}_prosim_progress.json").resolve()
        if not (
            progress_path.parent == rs_resolved or rs_resolved in progress_path.parents
        ):
            raise ValueError("invalid progress path")
        try:
            progress_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        _engine_wall_mono0 = time.monotonic()
        _t_rs = time.time()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "harness.run",
                "--input",
                str(sim_input_path),
                "--output",
                str(named_path),
                "--no-validate",
                "--compact-output",
                "--metrics-json",
                "--stem",
                safe_stem,
                "--progress",
                str(progress_path),
            ],
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONHASHSEED": "0"},
            **_prosim_worker_popen_extra(),
        )

        while proc.poll() is None:
            now_m = time.monotonic()
            last_m = publish_throttle["last_mono"]
            if last_m is None or (now_m - float(last_m)) >= 0.25:
                publish_throttle["last_mono"] = now_m
                progress_publish_ctr["pub"] += 1
                _pub_elapsed = float(now_m - float(_engine_wall_mono0))
                if _pub_elapsed > progress_pub_max_mono_elapsed["sec"]:
                    progress_pub_max_mono_elapsed["sec"] = _pub_elapsed
                pct = 0
                current = 0.0
                total = 0.0
                try:
                    progress_obj = json.loads(progress_path.read_text(encoding="utf-8"))
                    if isinstance(progress_obj, dict):
                        pct = int(progress_obj.get("percent") or 0)
                        current = float(progress_obj.get("current") or 0.0)
                        total = float(progress_obj.get("total") or 0.0)
                except Exception:
                    pass
                with _sim_lock:
                    _sim_progress.update(
                        current=current,
                        total=total,
                        percent=max(0, min(100, pct)),
                        runningClockLabel=_run_simulation_elapsed_sec_label(
                            _engine_wall_mono0
                        ),
                    )
            time.sleep(0.05)

        stdout, stderr = proc.communicate()
        _t_re = time.time()
        # #region agent log
        _run_wall_mono_sec = float(time.monotonic() - float(_engine_wall_mono0))
        # #endregion
        if proc.returncode != 0:
            raise RuntimeError((stderr or stdout or "ProSim subprocess failed").strip())

        worker_metrics: Dict[str, Any] = {}
        for line in reversed((stdout or "").splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                worker_metrics = obj
                break

        with _sim_lock:
            _sim_progress.update(
                runningClockLabel="",
                percent=0,
            )
        _persist_mono0 = time.monotonic()
        _t_ds = _t_dd = _t_we = time.time()
        _persist_wall_ms = (time.monotonic() - _persist_mono0) * 1000
        _payload_bytes = int(worker_metrics.get("payloadUtf8Bytes") or 0)
        try:
            progress_path.unlink()
        except OSError:
            pass
        _dbg_layout_sim_8ab4c9(
            {
                "runId": "prosim-thread",
                "hypothesisId": "H1_H3_H4",
                "location": "layout_receiver.py:_run_simulation_thread",
                "message": "sim_core_vs_serialize_wall_sec",
                "data": {
                    "resultStem": safe_stem,
                    "persist_monotonic_wall_ms": round(_persist_wall_ms, 4),
                    "run_simulation_wall_sec": round(
                        float(
                            worker_metrics.get(
                                "runSimulationWallSec", _t_re - _t_rs
                            )
                        ),
                        4,
                    ),
                    "run_simulation_wall_mono_sec": round(_run_wall_mono_sec, 4),
                    "max_progress_publish_mono_elapsed_sec": round(
                        float(progress_pub_max_mono_elapsed["sec"]), 4
                    ),
                    "subprocess_total_wall_sec": round(_t_re - _t_rs, 4),
                    "worker_input_load_wall_sec": round(
                        float(worker_metrics.get("inputLoadWallSec") or 0.0), 4
                    ),
                    "worker_run_simulation_cpu_sec": round(
                        float(worker_metrics.get("runSimulationCpuSec") or 0.0), 4
                    ),
                    "worker_python_executable": str(
                        worker_metrics.get("pythonExecutable") or ""
                    ),
                    "worker_python_hash_seed": str(
                        worker_metrics.get("pythonHashSeed") or ""
                    ),
                    "worker_process_priority_class": str(
                        worker_metrics.get("processPriorityClass") or ""
                    ),
                    "worker_omp_num_threads": str(
                        worker_metrics.get("ompNumThreads") or ""
                    ),
                    "worker_pythonpath": str(worker_metrics.get("pythonPath") or ""),
                    "worker_input_sha256": str(worker_metrics.get("inputSha256") or ""),
                    "json_dumps_wall_sec": round(
                        float(worker_metrics.get("jsonDumpsWallSec") or 0.0), 4
                    ),
                    "result_write_wall_sec": round(
                        float(worker_metrics.get("resultWriteWallSec") or 0.0), 4
                    ),
                    "thread_total_wall_so_far_sec": round(_t_we - thread_wall_t0, 4),
                    "payload_utf8_bytes": _payload_bytes,
                    "progress_cb_calls": int(worker_metrics.get("progressCbCalls") or 0),
                    "worker_progress_writes": int(
                        worker_metrics.get("progressWrites") or 0
                    ),
                    "progress_cb_lock_updates": int(progress_publish_ctr["pub"]),
                },
            }
        )
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


def serve_layout_receiver_forever(
    host: str = "127.0.0.1", port: int = LAYOUT_RECEIVER_PORT
) -> None:
    """Run the Layout Design standalone HTTP server in the foreground."""
    global _PORT
    _PORT = int(port)
    server = HTTPServer((host, int(port)), LayoutReceiverHandler)
    url_host = host if host not in ("0.0.0.0", "") else "127.0.0.1"
    print(f"Standalone Layout Design: http://{url_host}:{port}/", flush=True)
    print(f"Health: http://{url_host}:{port}/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
