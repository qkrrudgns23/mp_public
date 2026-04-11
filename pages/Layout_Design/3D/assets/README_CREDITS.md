# Grid 3D — local assets (CC0 / permissive)

| File / folder | Source | License |
|---------------|--------|---------|
| `vehicle_car.glb` | [Khronos glTF-Sample-Models — ToyCar](https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/ToyCar) (single GLB) | [Khronos sample model terms](https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/README.md#license) |
| `polyhaven_covered_car/` | [Poly Haven — covered_car](https://polyhaven.com/a/covered_car) (1k glTF bundle; optional, not used by default viewer) | CC0 |
| `polyhaven_kloppenheim_06_1k.hdr` | [Poly Haven — kloppenheim_06](https://polyhaven.com/a/kloppenheim_06) (1k HDR) | CC0 |
| `vehicle_aircraft.glb` | [three.js r128 examples — PrimaryIonDrive.glb](https://github.com/mrdoob/three.js/tree/r128/examples/models/gltf) | MIT (three.js project) |

**Grid 3D viewer:** **자동차** 버튼 → `vehicle_car.glb` · **항공기** 버튼 → `vehicle_aircraft.glb` (우주선 형태 스탠드인).

Poly Haven `covered_car` glTF uses extensions that often render invisible or black in three.js **r128**; the viewer loads `vehicle_car.glb` instead for reliable display.

Re-download: `python pages/Layout_Design/3D/assets/download_assets.py`
