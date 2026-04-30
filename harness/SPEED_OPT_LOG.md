# SPEED_OPT execution log

- τ: +2% (베이스라인 min wall_sec 대비; **wall 절대값은 OS 부하에 따라 크게 흔들림** → 채택·지속 기준은 **골든 PASS + 결정성**)
- N: 2 (SPEED_OPT_PLAN.md §12)

## 도구 (Step 0)

- `harness/golden_compare.py` — JSON deep equality
- `data/Result_storage/_perf_*.json`, `_prof_*.prof` — `.gitignore`

---

## 누적 요약 표 (기능 동등성)

| 단계 | 변경 요약 | default | large_flight | MNL_OSM | 결정 |
| ---- | --------- | ------- | ------------ | ------- | ---- |
| OPT1 (커밋 `perf: opt1`) | touchdown `dep_window` 행 사전 집계 | PASS | PASS | PASS | ADOPT |
| LOOP2 | `_lookahead_depth_billed_count`: prefix path-type 한 번만 조회 후 재사용 | PASS | PASS | PASS | ADOPT |
| LOOP3 | `_agents_by_arr_runway` + touchdown 선행기 스캔 범위 축소 | PASS | PASS | PASS | ADOPT |
| LOOP4 | `_ensure_agent_apron_lists` 이미 충분한 길이면 즉시 return | PASS | PASS | PASS | ADOPT |
| LOOP5 | `_lookahead_depth_billed_count_pts` + `get_lookahead_edges`에서 `edge_pts` 1회 생성 | PASS | PASS | PASS | ADOPT |
| LOOP6 | touchdown fast-path: 그룹 스캔 시 `arr_runway` strip 재검사 생략 | PASS | PASS | PASS | ADOPT |

**실패·원복**: 이번 5루프(LOOP2~6)에서 골든 실패 없음 → 원복 없음.

---

### BASELINE (프로젝트 초기, 코드 `221c99b`)

`harness.run --no-validate`, N=2, min wall_sec.

| pair | min |
| ---- | --- |
| default_layout | 7.85s |
| large_flight | 33.88s |
| MNL_OSM | 13.95s |

---

### OPT1 — `touchdown_dep_rows` (커밋 `f940bae`)

| pair | min (당시 샘플) | golden |
| ---- | --------------- | ------ |
| default_layout | ~7.88s | PASS |
| large_flight | 31.50s | PASS |
| MNL_OSM | 10.84s | PASS |

---

### 프로파일 근거 (LOOP2 계획 시점, `large_flight`, post-OPT1)

`cProfile` 상위: `run_simulation` ~340s; `_stand_pushback_clearance_cooldown_active` ~111s; `_compute_arr_touchdown_motion_abs_sec` ~75s; `_ensure_agent_apron_lists` ~75s; `_lookahead_depth_billed_count` ~32s; `_layout_edge_path_type` ~9.3s; `str.strip` ~6600만 회.

**미채택 후보 (리스크/범위)**  

- 틱 단위로 전체 `agents`에 대해 `ensure` 1회만: 호출 경로 다수 · apron 길이 변화 시 불일치 위험.  
- `stand_pushback` 알고리즘 변경: 골든 깨지기 쉬움.

---

### LOOP2 — `lookahead` path-type 단일 슬라이스

- **아이디어**: 같은 prefix에 대해 `taxiway` 분기에서 `j`·`j-1` path-type을 두 번 `_layout_edge_path_type` 하지 않도록 `pts[]` 한 패스.
- **검증**: 3페어 골든 PASS.

---

### LOOP3 — `_agents_by_arr_runway`

- **아이디어**: touchdown 선행기 스캔을 전체 `agents`가 아니라 `arr_runway_id` 버킷만 순회.
- **검증**: 3페어 골든 PASS.

---

### LOOP4 — `ensure` fast path

- **아이디어**: `dwell_sec_list`·4개 apron 리스트 길이가 이미 `n` 이상이면 조기 return (동작 동일).
- **검증**: 3페어 골든 PASS.

---

### LOOP5 — `get_lookahead_edges` + `_lookahead_depth_billed_count_pts`

- **아이디어**: `k` loop마다 `_layout_edge_path_type` 반복 호출 대신 `k_max`까지 path-type을 한 번만 채우고, `_prefix_has_apron_taxiway_edges`는 `edge_pts`로 대체.
- **검증**: 3페어 골든 PASS (`large_flight` 입력 경로 오타 1회 후 재실행으로 확인).

---

### LOOP6 — touchdown 그룹 스캔 strip 생략

- **아이디어**: `agents_by_arr_runway`가 있을 때 버킷 항목은 이미 동일 r/w 키로 모였으므로 per-flight `arr_runway_id` strip 재검사 생략 (`None` legacy 경로는 기존과 동일).
- **검증**: 3페어 골든 PASS.
- **결정성**: `default_layout` 동일 입력 2회 `run_simulation` dict 동등 확인.

---

### 다음 탐색 후보 (미적용)

프로파일 잔여: `_stand_pushback_clearance_cooldown_active` (~110s cum), `_ensure_agent_apron_lists` 호출 빈도. 틱 단위 일괄 `ensure` + `stand_pushback` 내부 `ensure` 제거는 **불변식 증명** 후에만 시도 권장.

---

## 10x 방향 전환: Phase A + C (Phase B 제외)

사용자 판단: event-driven/Phase B는 매 tick 유동성 모델을 깨뜨릴 수 있으므로 제외.  
방향: **tick loop는 유지**하고, tick 내부 반복 scan을 인덱스/정규화 캐시로 줄인 뒤, pure kernel만 컴파일 후보로 분리.

### Phase A1 — stand cooldown index

**현상**: post-OPT1 profile에서 `_stand_pushback_clearance_cooldown_active`가 cum ~110s, 호출마다 전체 `agents`와 apron segment를 스캔.

**대안**:

1. 최소: cooldown 함수 내부의 `ensure`만 추가 축소 — 리스크 낮지만 효과 제한.
2. **선택**: tick/pass 단위 `stand_id -> [(agent_id, clear_until)]` 인덱스 생성 후 query는 stand bucket만 확인 — tick 구조 유지, 병목 직접 제거.
3. persistent dirty-index — 가장 빠를 수 있으나 상태 불변식 리스크 큼.

**변경**:

- `StandCooldownIndex` 타입 추가.
- `_build_stand_pushback_clearance_index(agents)` 추가.
- `_stand_pushback_clearance_cooldown_active(..., stand_cooldown_index=None)` fast path 추가.
- `_single_full_reservation_pass`, history append loop, `_destination_stand_history_snap`, `_stand_pipeline_allows_apron_inblocks_stamp`, `can_reserve_path`에 index 전달.

**검증**:

| pair | result |
| ---- | ------ |
| default_layout | PASS |
| large_flight | PASS |
| MNL_OSM | PASS |

**profile 확인** (`large_flight`, cProfile):

- `_stand_pushback_clearance_cooldown_active`가 상위 hotspot에서 사라짐.
- 대신 `_build_stand_pushback_clearance_index` cum ~14.8s로 대체됨.
- 함수 호출 수: 기존 profile ~3.8억 → 새 profile ~2.17억.

### Phase A2 — normalized path type cache

**현상**: A1 후 `get_lookahead_edges`/`_layout_edge_path_type`/`_lookahead_depth_billed_count`가 상위 hotspot. 동일 agent path의 path_type string strip이 반복됨.

**변경**:

- `Flight.segment_path_types_norm` 추가.
- path 변경/초기화 시 `_refresh_agent_segment_path_types_norm` 호출.
- `_finish_edge_segment`에서 norm list도 head pop.
- `_lookahead_and_reservation_depth_for_agent`, `get_lookahead_edges`가 norm list를 우선 사용.

**검증**:

| pair | result |
| ---- | ------ |
| default_layout | PASS |
| large_flight | PASS |
| MNL_OSM | PASS |

**다음 후보**:

- A3: `_compute_arr_touchdown_motion_abs_sec`의 arrival predecessor도 runway별 누적 상태로 줄이기.
- C1: `_lookahead_depth_billed_count_pts`는 `Sequence[str] -> int` pure-ish kernel이라 Cython/Rust 후보. 다만 먼저 Python 배열/정수 코드로 안정화 권장.

### Phase A3 — touchdown arrival prefix index

**현상**: A1/A2 후 `_compute_arr_touchdown_motion_abs_sec`가 여전히 top self-time. 각 flight마다 같은 runway의 선행 arrival를 반복 스캔.

**대안**:

1. 기존 `agents_by_arr_runway` bucket만 유지 — 이미 적용되어 추가 효과 제한.
2. **선택**: runway별 ELDT 정렬 prefix index를 만들고, `agent_id -> (any_pred, pred_missing_exit, max_exit)`로 선행 arrival 집계를 조회.
3. touchdown 전체 vector화 — dep runway window 반복 조정까지 얽혀 리스크 큼.

**변경**:

- `TouchdownArrivalPrefixIndex` 타입 추가.
- `_touchdown_arrival_prefix_index(agents_by_arr_runway)` 추가.
- `_compute_arr_touchdown_motion_abs_sec(..., arrival_prefix_by_id=...)` fast path 추가.
- `_refresh_touchdown_motion_cache`, `apply_movement_controls`에서 prefix index를 생성/전달.

**검증**:

| pair | result |
| ---- | ------ |
| default_layout | PASS |
| large_flight | PASS |
| MNL_OSM | PASS |

**profile 확인** (`large_flight`, cProfile):

- `_compute_arr_touchdown_motion_abs_sec` self: A1/A2 profile 약 51s → A3 profile 약 30s.
- 전체 profile wall은 측정 부하 영향으로 명확한 개선이라고 단정하지 않음.

### Rejected A4 — one-pass lookahead depth reach

**시도**: `get_lookahead_edges`에서 prefix depth를 매 k 재계산하지 않고 single pass로 첫 depth 도달 index를 찾도록 변경.

**결과**: `default_layout`에서 `deadlock_resolve_event_count`가 `0 -> 2`로 바뀌어 골든 실패.

**조치**: 즉시 원복. 현재 코드에는 A4 변경 없음.
