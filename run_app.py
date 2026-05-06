"""Run the layout API and web UI on one HTTP port (no Streamlit).

Default: http://127.0.0.1:8501
- ``/`` Layout designer (same as ``/layout-design``)
- ``/home`` BluPrint globe + optional sign-in
- ``/api/*`` layout, map, simulation, and related endpoints

EC2: bind externally with ``HOST=0.0.0.0 PORT=8501 python run_app.py`` (open the port in your security group).

How to use: ``python run_app.py``
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.layout_receiver import LAYOUT_STORAGE_DIR, serve_layout_receiver_forever

ROOT = Path(__file__).resolve().parent
PROXY_PORT = int(os.environ.get("PORT", "8501"))
PROXY_HOST = os.environ.get("HOST", "0.0.0.0")


def main() -> None:
    os.chdir(ROOT)
    os.environ["LAYOUT_SAME_PORT"] = "1"
    base = f"http://127.0.0.1:{PROXY_PORT}"
    os.environ["LAYOUT_API_BASE_URL"] = base
    os.environ["GRID3D_ASSET_API_URL"] = base
    print(f"Web UI + layout API: http://{PROXY_HOST}:{PROXY_PORT}/", flush=True)
    print(f"  Layout designer: http://127.0.0.1:{PROXY_PORT}/", flush=True)
    print(f"  Home globe: http://127.0.0.1:{PROXY_PORT}/home", flush=True)
    print(f"  Layout storage: {LAYOUT_STORAGE_DIR}", flush=True)
    serve_layout_receiver_forever(host=PROXY_HOST, port=PROXY_PORT)


if __name__ == "__main__":
    main()
