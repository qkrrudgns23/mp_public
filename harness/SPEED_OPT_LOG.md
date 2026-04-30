# SPEED_OPT execution log

- τ: +2% (베이스라인 min wall_sec 대비; 노이즈 허용)
- N: 2 (SPEED_OPT_PLAN.md §12)

## 도구 (Step 0)

- `harness/golden_compare.py` — JSON deep equality, 첫 mismatch 경로 출력
- `data/Result_storage/_perf_*.json` — `.gitignore` 처리됨

---

### BASELINE 20260430Z (최적화 전, 코드 `221c99b` 시점 측정값)

`harness.run --no-validate`, 페어당 N=2, **min wall_sec**.

| pair            | run1   | run2   | min   |
| --------------- | ------ | ------ | ----- |
| default_layout  | 7.88s | 7.85s | 7.85s |
| large_flight    | 33.88s | 67.92s | 33.88s |
| MNL_OSM         | 17.09s | 13.95s | 13.95s |

- 3페어 골든: 모두 PASS (`golden_compare` + 베이스라인 코드)
- 결정성: `run_simulation` 동일 입력 2회 in-process, dict·sha256 일치

---

### ITER 20260430Z `touchdown_dep_rows` ( touchdown dep-window 사전 집계 )

**현상**: `cProfile`(large_flight)에서 `_compute_arr_touchdown_motion_abs_sec` 누적 ~56s, 매 호출마다 전체 `agents` 순회로 dep window 구축.

**대안 (요약)**:

1. **선택됨**: 런웨이별 `(dep_entry, dep_end, flight_id)` 를 tick당 1회 빌드 → `_compute…` 에 optional dict 전달, agent 본인 행만 제외해 리스트 구성. 동작 동일 · 중복 스캔 제거.
2. 선행기 스캔까지 runway별 인덱싱: 구조 더 큼, 동등성 리스크↑ → 보류.
3. 캐시만 강화: `apply_movement_controls`는 주석대로 캐시 비사용 유지 → 해당 경로 비용 그대로 → 보류.

**변경**:

- `utils/airside_sim.py`: `_touchdown_dep_window_rows_by_runway`, `_compute_arr_touchdown_motion_abs_sec(..., dep_window_rows_by_runway=...)`, `_refresh_touchdown_motion_cache`·`apply_movement_controls`에서 사전 집계 공유.

**결과** (`harness.run --no-validate`, N=2, 이후 추가 샘플 포함):

| pair            | min wall_sec (post) | golden | 비고 |
| --------------- | ------------------- | ------ | ---- |
| default_layout  | 7.88s (추가 샘플 7.88~17s) | PASS | OS 부하 분산 큼; baseline min과 동급 |
| large_flight    | **31.50s** (32.06 / 31.50) | PASS | baseline 33.88 대비 ~−7% |
| MNL_OSM         | **10.84s** (10.84 / 11.94) | PASS | baseline 13.95 대비 ~−22% |

- 결정성(변경 후): 3페어 `run_simulation` ×2 in-process, dict·sha256 일치.

**decision**: **ADOPT** (3페어 골든 PASS, 결정성 유지, large/MNL에서 측정 min 기준 개선).

**next**: `_stand_pushback_clearance_cooldown_active` / `_ensure_agent_apron_lists` 쪽 프로파일 후보(2차) — 변경 시 동일 루프로 검증.
