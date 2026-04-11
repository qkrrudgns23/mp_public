# Grid 3D — local assets (CC0 / permissive)

| File / folder | Source | License |
|---------------|--------|---------|
| `cybertruck (final).blend` | Optional local reference mesh (not loaded by the viewer). | Your project |
| `polyhaven_covered_car/` | [Poly Haven — covered_car](https://polyhaven.com/a/covered_car) (1k glTF bundle; optional, not used by default viewer) | CC0 |
| `polyhaven_kloppenheim_06_1k.hdr` | [Poly Haven — kloppenheim_06](https://polyhaven.com/a/kloppenheim_06) (1k HDR) | CC0 |

**Grid 3D viewer:** 추적 대상은 **코드로 만든 구(Sphere)** 입니다. GLB 차량 자산은 사용하지 않습니다.

Optional Blender export helpers (not used by default viewer): `export_cybertruck_glb.py`, `export_cybertruck.cmd`.

Poly Haven `covered_car` glTF uses extensions that often render invisible or black in three.js **r128**; the viewer does not load it by default.

Re-download Poly Haven / HDR only: `python pages/Layout_Design/3D/assets/download_assets.py`
