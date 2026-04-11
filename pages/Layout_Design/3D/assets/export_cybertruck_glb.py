"""
Export `cybertruck (final).blend` to `vehicle_car.glb` for the Grid 3D viewer.

Requires Blender (3.6+). From this directory (Git Bash / PowerShell), run:

  blender "cybertruck (final).blend" --background --python export_cybertruck_glb.py

Or with a full path to blender.exe on Windows:

  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" ^
    "cybertruck (final).blend" --background --python export_cybertruck_glb.py

After export, optionally Draco-compress for smaller download:

  npx --yes @gltf-transform/cli@3.10.1 draco vehicle_car.glb vehicle_car.glb
"""

from __future__ import annotations

from pathlib import Path

import bpy


def main() -> None:
    out = Path(__file__).resolve().parent / "vehicle_car.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(out),
        export_format="GLB",
        use_visible=True,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
