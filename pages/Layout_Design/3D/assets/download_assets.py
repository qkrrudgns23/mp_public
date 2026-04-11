"""
One-off / CI helper: fetch Grid 3D viewer assets into this folder.
Requires network. Run from repo root or any cwd:
  python pages/Layout_Design/3D/assets/download_assets.py

`vehicle_cybertruck.glb` is produced from `cybertruck (final).blend` (not downloaded here):
  blender "cybertruck (final).blend" --background --python export_cybertruck_glb.py
Optional Draco on that file:
  npx --yes @gltf-transform/cli@3.10.1 draco vehicle_cybertruck.glb vehicle_cybertruck.glb
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CTX = ssl.create_default_context()

_UA = "Mozilla/5.0 (compatible; Grid3D-assets/1; +https://polyhaven.com)"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
        dest.write_bytes(r.read())


def download_polyhaven_covered_car() -> None:
    api = "https://api.polyhaven.com/files/covered_car"
    req = urllib.request.Request(api, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    g = data["gltf"]["1k"]["gltf"]
    base = ROOT / "polyhaven_covered_car"
    fetch(g["url"], base / "covered_car_1k.gltf")
    for rel, info in (g.get("include") or {}).items():
        fetch(info["url"], base / rel)


def download_polyhaven_hdri() -> None:
    api = "https://api.polyhaven.com/files/kloppenheim_06"
    req = urllib.request.Request(api, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    url = data["hdri"]["1k"]["hdr"]["url"]
    fetch(url, ROOT / "polyhaven_kloppenheim_06_1k.hdr")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    download_polyhaven_covered_car()
    download_polyhaven_hdri()
    print("Done.")


if __name__ == "__main__":
    main()
