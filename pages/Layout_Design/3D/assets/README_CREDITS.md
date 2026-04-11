# Grid 3D — local assets (CC0 / permissive)

| File / folder | Source | License |
|---------------|--------|---------|
| `vehicle_cybertruck.glb` | Authored in `cybertruck (final).blend` (export via `export_cybertruck_glb.py`) | Your project |
| `cybertruck (final).blend` | Local Blender source for the driveable car | Your project |
| `polyhaven_covered_car/` | [Poly Haven — covered_car](https://polyhaven.com/a/covered_car) (1k glTF bundle; optional, not used by default viewer) | CC0 |
| `polyhaven_kloppenheim_06_1k.hdr` | [Poly Haven — kloppenheim_06](https://polyhaven.com/a/kloppenheim_06) (1k HDR) | CC0 |

**Grid 3D viewer:** 오른쪽 자동차 아이콘 → `vehicle_cybertruck.glb` (Blender에서 blend를 export).

Poly Haven `covered_car` glTF uses extensions that often render invisible or black in three.js **r128**; the viewer does not load it by default.

Re-download Poly Haven / HDR only: `python pages/Layout_Design/3D/assets/download_assets.py`
