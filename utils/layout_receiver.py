"""
Save Layout/load: data/Layout_storage/ Save by name to.
- Save: name If there is Layout_storage/{name}.json, If there is no current_layout.json (Run Simulationdragon).
- POST /api/save-layout: body { "layout": {...}, "name": "optional" } or the entire layout object.
- GET /api/load-layout?name=xxx: Layout_storage/{name}.json return.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, Optional

# Standalone receiver: Layout JSON must be data/Layout_storage/ Save only to
_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_STORAGE_DIR = (_ROOT / "data" / "Layout_storage").resolve()
LAYOUT_FILE = LAYOUT_STORAGE_DIR / "current_layout.json"
DEFAULT_LAYOUT_PATH = LAYOUT_STORAGE_DIR / "default_layout.json"

_PORT = 8765
_RESERVED_NAMES = frozenset({"current_layout", "default_layout"})
# Remove only dangerous characters so they can be used as file names (gap·Korean, etc. allowed)
def _sanitize_layout_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    return s[:200] if s else ""


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
        if path != "/api/save-layout" and not path.startswith("/api/save-layout"):
            self.send_response(404)
            self._send_cors()
            self.end_headers()
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


_server: Optional[HTTPServer] = None
_thread: Optional[threading.Thread] = None


def start_layout_receiver(port: int = _PORT) -> str:
    """Launch the layout receiving server in the background and connect to it URLreturns."""
    global _server, _thread
    if _server is not None:
        return f"http://127.0.0.1:{_PORT}"
    _server = HTTPServer(("127.0.0.1", port), LayoutReceiverHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    return f"http://127.0.0.1:{port}"
