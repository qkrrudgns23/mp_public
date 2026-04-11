"""
One-off / CI helper: fetch Grid 3D viewer assets into this folder.
Requires network. Run from repo root or any cwd:
  python pages/Layout_Design/3D/assets/download_assets.py

Primary car mesh is authored in `cybertruck (final).blend`; export to GLB with:
  blender "cybertruck (final).blend" --background --python export_cybertruck_glb.py
The committed vehicle_car.glb / vehicle_aircraft.glb are often Draco-compressed for the
3D viewer. After re-fetching aircraft or ToyCar URLs, re-encode before use:
  npx --yes @gltf-transform/cli@3.10.1 draco vehicle_car.glb vehicle_car.glb
  npx --yes @gltf-transform/cli@3.10.1 draco vehicle_aircraft.glb vehicle_aircraft.glb
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CTX = ssl.create_default_context()

PRIMARY_ION_DRIVE = (
    "https://raw.githubusercontent.com/mrdoob/three.js/r128/examples/models/gltf/PrimaryIonDrive.glb"
)
TOY_CAR_GLB = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/ToyCar/glTF-Binary/ToyCar.glb"
)


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


def download_aircraft_glb() -> None:
    fetch(PRIMARY_ION_DRIVE, ROOT / "vehicle_aircraft.glb")


def download_vehicle_car_glb() -> None:
    fetch(TOY_CAR_GLB, ROOT / "vehicle_car.glb")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    download_polyhaven_covered_car()
    download_polyhaven_hdri()
    download_aircraft_glb()
    download_vehicle_car_glb()
    print("Done.")


if __name__ == "__main__":
    main()
