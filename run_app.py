"""
Streamlit과 레이아웃 저장/로드 API를 한 포트(8501)에서 제공.
data/Layout_storage/ 에 이름별 저장, default_layout / current_layout 사용.

사용법: python run_app.py
- 로컬: http://127.0.0.1:8501
- AWS EC2: 보안 그룹에서 8501 인바운드 허용 후 http://<EC2퍼블릭IP>:8501

프록시는 0.0.0.0 에 바인드되어 EC2 등 외부 접속 가능. Streamlit은 127.0.0.1:8502 로만 동작.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import Request, urlopen

from urllib.parse import parse_qs, urlparse

from utils.airside_sim import run_simulation as airside_run_simulation
from utils.layout_receiver import LAYOUT_STORAGE_DIR, save_layout_to_file, list_layout_names, delete_layout, _safe_layout_path, _layout_path_for_read

ROOT = Path(__file__).resolve().parent
STREAMLIT_PORT = 8502
PROXY_PORT = int(os.environ.get("PORT", "8501"))
PROXY_HOST = os.environ.get("HOST", "0.0.0.0")  # EC2 등에서는 0.0.0.0 으로 외부 접속 허용


class ProxyHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/load-layout"):
            qs = parse_qs(urlparse(self.path).query)
            names = qs.get("name", [])
            name = (names[0] or "").strip() if names else ""
            path = _layout_path_for_read(name) if name else None
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
        self._proxy()

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/api/run-simulation":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                raw = json.loads(body.decode("utf-8"))
                if isinstance(raw, dict) and "layout" in raw:
                    layout = raw["layout"]
                    layout_name = (raw.get("layoutName") or "").strip() or "current_layout"
                else:
                    layout = raw
                    layout_name = "current_layout"
                if not isinstance(layout, dict):
                    raise ValueError("layout must be a JSON object")
                # Run Simulation 시 Layout_storage/current_layout.json 저장 (timeline 제외)
                layout_to_save = dict(layout)
                flights = layout_to_save.get("flights") or []
                layout_to_save["flights"] = [{k: v for k, v in f.items() if k != "timeline"} for f in flights]
                save_layout_to_file(layout_to_save, name=None)
                result = airside_run_simulation(layout, time_step_sec=5, use_discrete_engine=True, layout_name=layout_name)
                out = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self._send_cors()
                self.end_headers()
                self.wfile.write(out)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return
        if path == "/api/delete-layout":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                obj = json.loads(body.decode("utf-8"))
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
        if path == "/api/save-layout":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            try:
                obj = json.loads(body.decode("utf-8"))
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
            return
        self._proxy()

    def _proxy(self):
        url = f"http://127.0.0.1:{STREAMLIT_PORT}{self.path}"
        if self.command == "GET":
            req = Request(url, headers=dict(self.headers))
        else:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            req = Request(url, data=body, method=self.command, headers=dict(self.headers))
        try:
            with urlopen(req, timeout=30) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(r.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Proxy error: {e}".encode())

    def log_message(self, format, *args):
        pass


def main():
    os.environ["LAYOUT_SAME_PORT"] = "1"
    os.environ["LAYOUT_API_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}"
    entry = ROOT / "Home.py"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(entry),
            "--server.port", str(STREAMLIT_PORT),
            "--server.address", "127.0.0.1",
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    server = HTTPServer((PROXY_HOST, PROXY_PORT), ProxyHandler)
    print(f"Layout API + Streamlit proxy: http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"  Layout storage: {LAYOUT_STORAGE_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        proc.wait()


if __name__ == "__main__":
    main()
