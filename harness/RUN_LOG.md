## Run Notes (최근 실행 결과 요약)

### Format
- **RUN_ID**: UTC timestamp + short tag
- **command**: 실행 커맨드
- **inputs/outputs**: 사용 파일 경로
- **result**: PASS/FAIL
- **evidence**: 로그/검증 실패 요약
- **next**: 다음 루프 최소 수정 계획

---

### (init) Harness created
- **RUN_ID**: N/A
- **command**: N/A
- **result**: N/A
- **evidence**: `harness/` 문서/스크립트 생성
- **next**: smoke check + 1회 실행 + validator 기록

---

### RUN 20260407T0706Z default_layout
- **RUN_ID**: 20260407T0706Z
- **command**: `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
- **inputs/outputs**:
  - input: `data/Result_storage/default_layout_sim_input.json`
  - output: `data/Result_storage/_validation_sim_result.json`
- **result**: PASS
- **evidence**: `PASS run+validate` (wall ~4s)
- **next**: `python -m harness.loop`로 1회 더 실행해 재현성 확인(동일 계약 유지)

---

### RUN 20260407T0710Z loop-pass
- **RUN_ID**: 20260407T0710Z
- **command**: `python -m harness.loop --max-runs 2 ...`
- **result**: PASS (1st run success → loop 종료)
- **evidence**:
  - (fix) `harness/loop.py` 문자열 이스케이프 오타로 `SyntaxError` 발생 → 수정
  - 이후 `smoke` + `run+validate` 통과
- **next**: 하네스 기반으로 `airside_sim.py` 개선 작업은 “작은 변경→run+validate→기록” 루프를 사용

---

### RUN 20260407T0730Z temp-apron-reroute-guard
- **RUN_ID**: 20260407T0730Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
- **result**: PASS
- **builder change**:
  - reroute 경로 생성에서 `temp occupied incident edges`를 패널티 대상이 아닌 **forbidden edge**로 처리
  - `wait_reason=temp_stand_busy:*`일 때 incident edges를 reroute penalized set에 반영
- **reviewer check**:
  - Information.json 변경 없음
  - I/O 계약 유지(`run+validate` PASS)
  - 영향 범위는 `utils/airside_sim.py` 경로탐색/재탐색 로직으로 제한
- **evidence**: 결과 파일 생성 및 validator 통과
- **note**: 현재 기본 데이터셋에서 R7/R8의 01:14 근접 지표는 기존과 동일(재현 케이스 추가 필요)

---

### RUN 20260407T0805Z unresolved-loop-continue
- **RUN_ID**: 20260407T0805Z
- **result**: FAIL (미해결)
- **evidence**:
  - 기존 수정으로 R7/R8 근접 지표가 개선되지 않았고, R8 경로 이상(사용자 제보) 대응 미흡
  - 원인 확인 없이 종료하면 안 되는 케이스였음
- **action**:
  - 루프 지속, 원인 재탐색
  - `LESSONS_LEARNED.md`에 “미해결 상태 종료 금지” 규칙 추가

---

### RUN 20260407T0820Z temp-incident-map-fix
- **RUN_ID**: 20260407T0820Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
- **result**: PASS (핵심 지표 개선)
- **builder change**:
  - temp stand incident edge 맵이 비어있던 원인을 수정
  - `_build_temp_stand_incident_edges()`에서 temp stand ID가 그래프 stand map에 없을 때, temp stand 좌표 기준 최근접 노드로 fallback 매핑
  - tie-break 비결정성(`hash`)을 안정 시드로 교체해 재현성 확보
- **reviewer evidence**:
  - `temp incident map`: `{id_ftz1gy7ir:3, id_c6efn4jsg:4}`로 정상 생성 확인
  - 동일 입력 2회 실행 결과 JSON 완전 동일 확인
  - R7/R8 근접 지표: `old min_dist≈5.34px` → `new min_dist≈62.76px` (01:14 근처 창)
- **next**: 사용자 시나리오에서 01:11 런웨이 워크 체감이 사라졌는지 현장 확인 후, 필요 시 해당 시각 구간만 추가 계측

---

### RUN 20260407T0835Z verification-gate-enforced
- **RUN_ID**: 20260407T0835Z
- **result**: PASS (검증 게이트 통과)
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
  - 재현 지표 스크립트(근접도 + phase regression 검사)
- **evidence**:
  - `old_min_dist = 5.3357px @ t=4438`
  - `new_min_dist = 62.7623px @ t=4386` (근접 통과 완화)
  - `r8_landing_after_arr = False` (Arr_taxi 이후 Landing 회귀 없음)
- **rule update**:
  - 수정 후 확인 의무/검증 게이트를 `RULES.md`, `LESSONS_LEARNED.md`에 반영

---

### RUN 20260407T0850Z runway-single-occupancy-guard
- **RUN_ID**: 20260407T0850Z
- **result**: PASS (가드 적용 + 재실행 통과)
- **builder change**:
  - `can_reserve_path()`에서 활주로는 `forced_open` 우회 금지
  - 활주로 자원(`runway_resources`)에 타 기체 점유가 있으면 즉시 `runway_occupied:*`로 예약 실패
  - `apply_movement_controls()`에서 WAIT/YIELD 체크 누락 분기(`elif`)를 독립 `if`로 수정
  - 이동 단계에서도 현재 엣지가 활주로이고 타 기체가 점유 중이면 강제 정지
- **evidence**:
  - `python -m harness.smoke` PASS
  - `python -m harness.run ...` PASS
  - 01:21 이후 R2/R6_2 최소거리 `~156.28px` (즉시 겹침/중첩은 관측되지 않음)
- **note**:
  - `HEAVY_DECISION_INTERVAL_SEC`는 보조 요인일 수 있으나, 이번 패치는 간격 조정보다 안전 불변식(활주로 단일 점유) 강제를 우선 적용

---

### RUN 20260409T0415Z path-graph-cache-perf
- **RUN_ID**: 20260409T0415Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_bench_default.json`
  - `python -m harness.run --input data/Result_storage/large_flight_sim_input.json --output data/Result_storage/_bench_large.json`
  - golden equality: parsed JSON `==` against `default_layout_sim_result.json`, `large_flight_sim_result.json`
- **result**: PASS
- **builder change**:
  - Per-`run_simulation` path graph build cache (`_PATH_GRAPH_BUILD_CACHE`, keyed by `id(layout)` + path-search params + runway ops direction).
  - Reroute/temp penalties applied in Dijkstra via optional penalized directed arcs instead of mutating `PathGraph` costs.
  - `path_dijkstra`: no-penalty fast path keeps original `adj` weight loop (avoids per-edge `edge_map` overhead).
  - `build_resource_model` / pair-index bootstraps use `_cached_path_graph_for_direction`.
- **evidence**:
  - `PASS run+validate` on both inputs; `default match: True`, `large match: True`.
  - Same-machine A/B vs stashed baseline (this runner): default ~10.3s → ~5.9–7.2s wall; large ~91.9s → ~80–84s wall (variance observed between runs).
- **next**: If more speed is needed, profile the time-step loop and hot geometry helpers; keep golden JSON equality as the regression gate.

---

### RUN 20260409T0455Z touchdown-motion-cache
- **RUN_ID**: 20260409T0455Z
- **evidence (cProfile, large input, pre-change)**:
  - `_arr_touchdown_motion_abs_sec` ~138s tottime, ~2.97M calls (dominant bottleneck).
  - `update_decisions_every_10s` ~7.3s cumtime — heavy-decision path not the main cost vs touchdown.
- **builder change**:
  - `_compute_arr_touchdown_motion_abs_sec` = former body; per-tick `touchdown_motion_by_id` on `SimulationControlState`.
  - `_refresh_touchdown_motion_cache` after `current_time_abs += dt_sec` and after `apply_movement_controls`.
  - Callers before movement use cache via `_arr_touchdown_motion_abs_sec(..., control_state=...)`.
  - **Second loop of `apply_movement_controls`**: always `_compute_...` (sequential moves / reroute can change runway exit state mid-tick).
- **command**: `harness.smoke` + `harness.run` both golden inputs + JSON `==` vs `*_sim_result.json`.
- **result**: PASS (`default True`, `large True`).
- **wall (this runner, no cProfile)**: large ~70.7s vs ~80s prior run on same trajectory of work.
- **next**: Optional: trim `str.strip` hot paths (~16s in profile); numpy only where float/tie semantics provably match golden.

---

### RUN 20260421T0500Z stand-dwell-heading-fix
- **RUN_ID**: 20260421T0500Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/_mnl_stand_heading_fix_result.json`
  - `python -m harness.validate --input data/Result_storage/MNL_OSM_sim_input.json --result data/Result_storage/_mnl_stand_heading_fix_result.json`
- **result**: PASS
- **builder change**:
  - `utils/airside_sim.py`의 `_compress_agent_history_for_dwell_export()`가 parked dwell 구간 양 끝점의 `motionForward`를 주기 직전 마지막 이동 샘플 기준으로 정규화
  - dwell endpoint rewrite 시 기존 tail 필드를 유지하도록 row 재구성
- **reviewer check**:
  - `Information.json` 변경 없음
  - I/O 계약 유지 (`PASS run+validate`)
  - 영향 범위는 dwell export 압축 로직으로 제한
- **evidence**:
  - apron 6 / `pbb-72f017a5a944` / flight `id_mmcuy5gfc`
  - parked dwell heading: `t=26000 -> 140.224`, `t=28769 -> 140.224`, `motionForward=True`
  - pushback start 이후: `t=28771 -> 140.223`, `motionForward=False` (다음 moving sample부터 reverse 적용)
- **next**: 동일한 dwell 압축 경로를 타는 다른 stand 케이스가 있으면 추가 샘플 검증

---

### RUN 20260421T0519Z mnl-r1-r2-parked-nose-fix
- **RUN_ID**: 20260421T0519Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/MNL_OSM_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/MNL_OSM_sim_input.json --result data/Result_storage/MNL_OSM_sim_result.json`
- **result**: PASS
- **builder change**:
  - `_compress_agent_history_for_dwell_export()`의 parked dwell `motionForward`를 단순 고정값/직전 row 플래그가 아니라
    `정지 직전 마지막 raw segment 방향`과 `정지 후 첫 moving display 방향`을 비교해서 선택하도록 수정
  - stationary interval에서 프런트가 `prev` 벡터를 우선 사용하는 점을 기준으로 `R1`/`R2` 케이스를 동시에 만족하도록 보정
- **reviewer check**:
  - `Information.json` 변경 없음
  - I/O 계약 유지 (`PASS run+validate`)
  - 영향 범위는 dwell export 압축 로직으로 제한
- **evidence**:
  - `R1`: stand nose `140.220`, parked heading `[140.227, 140.227, 140.225]`
  - `R2`: stand nose `160.977`, parked heading `[160.980, 160.980, 160.980]`
- **next**: 다른 공항 입력에서도 동일한 stationary fallback 규칙이 문제를 일으키는 stand가 있는지 샘플 점검

---

### RUN 20260421T0530Z stand-nose-root-cause-fix
- **RUN_ID**: 20260421T0530Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/MNL_OSM_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/MNL_OSM_sim_input.json --result data/Result_storage/MNL_OSM_sim_result.json`
- **result**: PASS
- **builder change**:
  - parked dwell `motionForward`를 history 추정만으로 정하지 않고, stand geometry에서 계산한 `nose heading`을 기준으로 선택
  - `_stand_nose_heading_deg()` 추가: `angleDeg`(tail/open 기준)를 nose 방향으로 변환하고, 필요 시 PBB anchor→apronSite 벡터로 fallback
- **root cause**:
  - 기존 보정은 stationary interval의 방향을 주변 sample만으로 추정해서 stand geometry와 분리돼 있었음
  - 게다가 `angleDeg`를 nose 방향으로 바꾸는 변환에서 180도 보정이 빠질 수 있는 구조여서 `R1`/`R2`가 엇갈렸음
- **reviewer check**:
  - `Information.json` 변경 없음
  - I/O 계약 유지 (`PASS run+validate`)
  - 영향 범위는 parked dwell 방향 선택 로직과 stand heading helper로 제한
- **evidence**:
  - `R1`: stand nose `140.220`, parked `[140.228, 140.228, 140.227]`
  - `R2`: stand nose `160.977`, parked `[160.965, 160.965, 160.980]`
- **next**: 필요 시 RKSI/RPLL 등 다른 입력에서 동일한 stand-nose 기준이 유지되는지 샘플 회귀 검증

---

### RUN 20260421T0539Z dynamic-stand-dwell-heading-fix
- **RUN_ID**: 20260421T0539Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/MNL_OSM_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/MNL_OSM_sim_input.json --result data/Result_storage/MNL_OSM_sim_result.json`
  - rerun same command once more + parked-heading verifier
- **result**: PASS
- **builder change**:
  - 입력 `standId/apronId`가 비어 있는 flight는 `_history_destination_stand_id()`로 dwell band의 `destinationApron.standId`를 복원
  - parked dwell nose 기준 계산이 정적 입력 stand가 아니라 실제 시뮬레이션 중 배정된 stand를 사용할 수 있도록 보정
- **root cause**:
  - `R3`~`R6`은 동적 stand 배정 케이스라 `ag.apron_stand_id`가 비어 있었고, 기존 압축 로직은 stand nose를 계산하지 못해 기본 `motionForward`를 그대로 남겼음
  - 그 결과 parked row가 일부 stand에서 정확히 180도 반대로 저장됐음
- **reviewer check**:
  - `Information.json` 변경 없음
  - I/O 계약 유지 (`PASS run+validate`)
  - 영향 범위는 parked dwell stand-id 복원과 nose heading lookup으로 제한
- **evidence**:
  - `R1`: stand nose `140.220`, parked `[140.223, 140.223, 140.226]`
  - `R2`: stand nose `160.977`, parked `[160.981, 160.981, 160.980]`
  - `R3`: stand nose `-135.050`, parked `[-135.046, -135.046, -135.047]`
  - `R4`: stand nose `177.460`, parked `[177.470, 177.470, 177.470]`
  - `R5`: stand nose `-120.014`, parked `[-120.013, -120.013, -120.014]`
  - `R6`: stand nose `-120.004`, parked `[-120.003, -120.003, -120.003]`
  - second rerun verifier: `bad []`
- **next**: 동적 stand 배정이 많은 다른 입력에서도 동일 fallback이 잘 동작하는지 샘플 회귀 검증

