## Run Notes (최근 실행 결과 요약)

> **Note (2026-05)**: 벽시간만 반복 측정하는 `opt_repeat_experiment` 및 벤치/스냅 스크립트는 하네스에서 제거됨. 성능 확인은 `golden_opt_cycle` 한 사이클(또는 프로파일)로만 기록하는 것을 권장.

### Phase summary (numpy-vectorize-airside-sim plan)

- **baseline (Phase 0)**: golden triple wall sum **25.31 s** (default 5.50 / large 9.47 / MNL 10.34).
- **Phase 1 (non-numpy 4 patches)**: P1.1 → P1.4 모두 채택. wall sum **25.31 → 21.04 s** (-4.27 s, -16.9 %). cProfile 측정으로 검증된 구조적 개선:
  - `_ensure_agent_apron_lists`: 11.15 M 호출 → 사실상 0 (sticky flag 첫 호출 후 즉시 return). 더 이상 top-12에 없음.
  - `_compute_arr_touchdown_motion_abs_sec`: 513 K 호출 → 257 K (P1.3 dirty flag로 정확히 절반).
  - `_build_stand_pushback_clearance_index`: 513 K 호출 → 29 (P1.4 캐시로 17,000× 감소).
- **Phase 2 (numpy 시도)**: P2.1+P2.2 결합 시도 → wall +2.8 s regression → REVERT. P2.3는 같은 리스크 프로파일이라 선제 cancel.
- **최종 결과**: wall sum **21.15 s** (베이스 대비 -16.5 %). golden triple loose gate(rtol=1e-9, atol=1e-6) 통과 (Phase 1 패치는 정확 비트-동등도 유지).
- **Lessons**: 본 워크로드(N≈50 agents, 2–3 runways)에서 NumPy 벡터라이즈는 per-op Python 오버헤드가 작은 배열 산술을 능가하여 손해. Phase 1 비-numpy 캐시/dirty-flag 전략이 모든 실측에서 우위. 향후 N이 한 자릿수 더 커지면(>500 agents) P2.2 재고 가치 있음.

### RUN 20260501 P1.1 _apron_lists_valid sticky flag (perf, golden-locked)

- **command**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag p1_1_apron_valid_flag`
- **change**: `Flight._apron_lists_valid: bool = False` 추가, `_ensure_agent_apron_lists`가 True면 즉시 return하고 lists 길이 정규화 후 True로 셋. apron_segments / dwell_sec_list 길이는 생성 후 변하지 않으므로 sticky 불변식 유지.
- **alternatives**: (A) sticky flag — **채택**. (B) lists를 numpy로 — 무관(no-op 체크가 비용). (C) hot caller에서 호출 인라인 제거 — 회귀 위험.
- **result**: golden triple PASS. wall sum **25.31 s → 21.27 s (-4.04 s, -16 %)**; default 5.50→4.81, large 9.47→8.29, MNL 10.34→8.17.

### RUN 20260501 P1.2 stand_phys snapshot for history snap (perf, golden-locked)

- **command**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag p1_2_stand_phys_snapshot`
- **change**: `_build_stand_phys_set_snapshot(control_state)` 신설 — tick당 한 번 `{stand_id → set(occupied_by)}`. `_destination_stand_history_snap`에 `phys_set_by_stand` kwarg 추가, 있으면 per-agent set-comp 생략 후 `len(s) - (1 if aid in s else 0)`로 즉답. `run_simulation` 히스토리 루프에서 1회 빌드하여 전 agent에 공유.
- **alternatives**: (A) snapshot dict — **채택**. (B) NxK NumPy bool 매트릭스 — agents/stands가 적어 무가치. (C) 히스토리 스냅 자체를 시뮬 종료 후로 미루기 — 데이터 흐름 큰 변경.
- **result**: golden triple PASS. wall sum **21.27 → 22.72 s** (+1.45 s, OS 지터 범위 ±1.5 s 내). cProfile로 확인: `_destination_stand_history_snap` tottime 0.69 s (이미 작아짐 — 절감 0.1 s 추정으로 노이즈에 묻힘). 채택은 **유지**(코드 명확성 + 미세 절감).

### RUN 20260501 P1.3 conditional 2nd touchdown-cache refresh (perf, golden-locked)

- **command**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag p1_3_td_cache_dirty_flag`
- **change**: `Flight._td_cache_input_dirty: bool = True`. `_finish_edge_segment`이 `runway_entry_abs_sec` / `path_completed_abs_sec`를 새로 쓸 때 True. `_refresh_touchdown_motion_cache`가 호출 끝에 모든 agent의 플래그 클리어. `run_simulation` 의 2nd refresh(line 9235)는 `any(ag._td_cache_input_dirty for ag in agents)`일 때만 실행.
- **alternatives**: (A) per-Flight 플래그 + 조건부 — **채택**. (B) 무조건 1번으로 줄이고 `_try_record_exit_runway_abs_sec` 재배치 — 계약 변경 위험. (C) `_finish_edge_segment` 시그니처에 control_state 주입 후 직접 invalidate — 콜사이트 다수, 침습.
- **result**: golden triple PASS. wall sum **22.72 → 21.72 s** (-1.0 s, -4.4 %). cProfile: `_compute_arr_touchdown_motion_abs_sec` 호출 수 **513,044 → 256,914 (정확히 절반)**, tottime **2.23 → 1.14 s**. 2nd refresh가 대략 절반 tick에서 스킵됨.

### RUN 20260501 P1.4 stand cooldown index cache + dirty flag (perf, golden-locked)

- **command**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag p1_4_pushback_cooldown_cache`
- **change**:
  - `Flight._pushback_clearance_dirty: bool = True` 추가, 진짜 mutator 두 곳에서 셋: `_agent_stamp_current_offblocks`(offblocks list 첫 기입), `_agent_set_current_apron_index`(apron_stand_id / 단일 offblocks 슬롯 시프트).
  - `SimulationControlState.stand_cooldown_index_cache: Optional[StandCooldownIndex]` 신설.
  - 새 헬퍼 `_get_stand_pushback_clearance_index(control_state, agents)`: 캐시 있고 dirty agent 없으면 캐시 즉답, 아니면 full rebuild + 모든 agent 플래그 클리어.
  - `_single_full_reservation_pass`(line 8632) / `run_simulation`(line 9251)의 두 build call 모두 헬퍼로 교체. 히스토리 루프의 in-place `_stand_pushback_clearance_merge_agent`는 그대로 유지(다음 tick rebuild 때 dirty로 정합성 보장).
- **alternatives**: (A) 캐시 + per-agent dirty flag — **채택**. (B) NxK NumPy cooldown 매트릭스 — N·K가 모두 수십 단위라 무가치. (C) build 빈도만 줄이기 — heavy/light pass와 history loop 두 위치 모두 별도 소비처라 단순 빈도 축소 어려움.
- **result**: golden triple PASS. wall sum **21.72 → 21.04 s** (-0.68 s, -3.1 %). cProfile 측정:
  - `_build_stand_pushback_clearance_index` 호출 **513,058 → 29회** (>17,000× 감소).
  - `_stand_pushback_clearance_merge_agent` **513,058 → 420회**.
  - `_get_stand_pushback_clearance_index` 36,646회(= 2 × 18,323 tick) 캐시 히트.
  - 클러스터 전체 cProfile time **2.63 → 0.052 s**(P1.3 이후 2.63s 기준 50× 절감).

### RUN 20260501 P2.1 + P2.2 NumPy vectorize touchdown cache (REVERTED, evidence-based)

- **command**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag p2_1_p2_2_vectorize_td_cache`
- **change attempted**: `numpy` 의존성 추가, `_AgentScalarArrays`(병렬 컬럼) + `_build_agent_scalar_arrays` + `_sync_agent_scalar_arrays_dirty` + `_refresh_touchdown_motion_cache_vectorized` (per-arr-runway `np.argsort` + `np.searchsorted` + `np.maximum.accumulate` + `np.cumsum`로 predecessor 통계 계산, dep-window는 per-agent legacy 루프 유지). `_try_record_exit_runway_abs_sec`도 `_td_cache_input_dirty=True`로 마킹.
- **result**: golden triple PASS. 그러나 **wall sum 21.04 → 23.84 s (+2.80 s, +13 %)**. cProfile: `_refresh_touchdown_motion_cache_vectorized` tottime **2.437 s** vs legacy 1.13 s — 정확히 두 배. per-runway 그룹이 5–30 agent 수준이라 `np.where` / `np.argsort` / `np.searchsorted`의 Python 호출 오버헤드가 작은 배열 산술을 압도.
- **decision**: **REVERT**. plan stop condition("If two consecutive numpy patches cost more than they save, abandon further numpy work")의 첫 패치에서 명백한 regression. 사용자 사전 지시("넘파이 어레이의 전환이 너무 잦아서 오히려 성능을 떨어뜨릴 수 있기 때문에 ... 효과가 있어보일만한 곳에 넣자가 전략") 합치. P2.3은 더 작은 단위(per-call 2.8 us)라 동일 리스크가 자명하므로 **선제 cancel**.
- **revert verify**: tag `p2_revert_verify` PASS, wall sum **21.15 s** (P1.4 21.04 s ±지터 범위 내).

### Loose golden gate (numpy-vectorize-airside-sim plan, Phase 0)

- **gate command (Phase 1/2 patches)**: `python -m harness.golden_opt_cycle --float-rtol 1e-9 --float-atol 1e-6 --tag <step>`
- **rationale**: NumPy 벡터라이즈 / 사전계산 캐시 패치는 합산/곱 순서가 바뀌면서 IEEE 비결합법칙으로 인한 ULP-급 미세 떨림이 발생할 수 있음. 사용자 승인하에 leaf number tolerance를 `rtol=1e-9 / atol=1e-6` 로 설정 (시간 1ms·위치 1um 수준).
- **baseline wall (golden triple, no profiler)**: default_layout 5.50 s · large_flight 9.47 s · MNL_OSM 10.34 s · **sum 25.31 s** (`--tag baseline_phase0`).
- **profile baseline (large_flight only, cProfile 5–10× overhead)**: `_prof_lf_opt10.pstats` 기준 cumulative **377.77 s**. 비율 분석은 유효, 절대 절감은 wall sum 기준으로 환산 필요.
- **top hot clusters (baseline)**: `_refresh_touchdown_motion_cache`+`_compute_arr_touchdown_motion_abs_sec` 106 s (28 %), `_single_full_reservation_pass` 105 s, `_ensure_agent_apron_lists` 22 s (11.15 M calls), `_stand_pushback_clearance_merge_agent` 16 s (4.17 M), `_destination_stand_history_snap` 17 s, `_lookahead_depth_billed_count` 13 s.
- **abort/revert rule**: 패치가 loose gate(`rtol=1e-9 atol=1e-6`)를 통과 못 하면 즉시 원복하고 사고 단계로 복귀.



### RUN 20260501 td_compute_aid_aa_bucket — **미채택(원복)**

- **패치**: `_compute_arr_touchdown_motion_abs_sec`에 `aid` 로컬, `deps` 합의 `float(20.0)` 제거, `agents_by_arr_runway`를 `aa`·`bucket`으로 조회.
- **golden**: `golden_opt_cycle --tag td_compute_aid_aa_bucket_v1` PASS.
- **과거 반복 측정(도구 제거 전)**: 8 reps median **33.51 s** — 동일 세션 패치 전 4 reps baseline median **33.33 s** 대비 악화 → **시간 줄이기** 미충족으로 **원복**.
- **note**: 이후 `golden_opt_cycle --tag revert_verify_td_compute` 로 원복 트리 PASS 확인.

### RUN 20260501 playback_t_key_apk_locals (perf, golden-locked)

- **command**: `python -m harness.smoke` → `python -m harness.golden_opt_cycle --skip-smoke --tag playback_apk_touchdown_locals_v1`
- **result**: golden triple PASS; 당시 합계 벽시간 샘플 **median ~33.15 s** (레거거 반복 하네스 기록).
- **대안**: (A) `_playback_schedule_point_t_key` + touchdown `else`에서 `apk`/`dep_rows_sel` 로컬 — **채택** (프로파일 구간 미세 절약, 계약 동일). (B) apron `_ensure_*` 호출 줄이기 — 변경면 큼. (C) `sorted`만 손보기만 — 단독 적용 시 한계 있어 A에 포함.

### RUN 20260501 runway_key_strip_elide (perf, golden-locked)

- **command**: `python -m harness.smoke` → `python -m harness.golden_opt_cycle --skip-smoke --tag runway_key_strip_elide_v1`
- **result**: golden triple PASS (deep-equal); 당시 합계 벽시간 샘플 **median ~33.14 s** (레거거 반복 하네스 기록).
- **대안 요약**: (A) `_agent_runway_id_key`: sim `Flight`의 활주로 id가 생성 시점에 이미 strip 정규화되므로 루프 내 `str(...).strip()` 제거 — **채택**. (B) 터치다운 모션 캐시 증분 갱신 — 이득 큼, 계약 리스크 큼 → 보류. (C) 재생 포인트 `sorted` 람다 치환 — 프로파일 대비 영향 미미 → 보류.
- **change**: `utils/airside_sim.py` — `_agent_runway_id_key`; touchdown/점유 등 agent 경로에서 arr/dep runway dict·비교 키 계산 단순화.

### RUN 20260424T0400Z temp-stand-passthrough + arr-temp-splice-snap
- **RUN_ID**: 20260424T0400Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/_validation_mnl_result.json`
- **result**: PASS run+validate (default_layout, MNL_OSM)
- **problem A (R7 미도달)**: default_layout에서 R7이 C012가 바쁠 때 temp stand T001로 우회해야 하지만 T001 도달 전 T002의 graph node(70) 통과 구간에서 `temp_stand_busy:T002`로 고착. 원인은 `_agent_occupies_temp_stand_slot`가 `temp_stand_id`만 세팅돼도 `sr_t.occupied_by`를 채워 — 아직 물리적으로 도착하지 않은 R8의 T002 claim이 동일 택시웨이의 incident edge(layout-edge-098/099)를 통과 트래픽까지 차단. T001(node 68)은 T002(node 70)보다 택시웨이상 하류이므로 R7이 T001로 가려면 node 70을 지나야 함 → 영구 고착.
- **problem B (RTX 뚜둑거림)**: Landing → `Arr_taxi_occupied`(PHASE_ARR_TAXI_TEMP) 전환 틱에서 속도가 `v=16.03 → 1.73` 로 급락하고 위치가 ~3 px 역행. 원인은 `_try_splice_temp_stand_arrival_detour`가 temp_prep을 만들 때 `snap_exact_start_xy=False` 이어서 Dijkstra 시작 노드(폴리라인 상 이전 그래프 노드)가 `start_xy`(Landing 마지막 끝점 = Arr_taxi 첫 시작점)와 불일치. 결과적으로 temp_prep 첫 micro-segment가 Landing 마지막 micro-segment와 **같은 구간을 중복 표현**하며, edge 전환 시 `_snap_agent_to_first_segment`가 기체를 그 세그먼트의 앞쪽으로 끌어당겨 역행·속도 리셋이 발생.
- **change**: `utils/airside_sim.py`
  - `_agent_occupies_temp_stand_slot`: `awaiting_apron_from_temp=True` 일 때만 `tid` 반환하도록 제한. 픽 중복 방지는 `_temp_stand_has_other_claimant_or_occupant`의 `ag2.temp_stand_id` 스캔으로 이미 유지. `can_reserve_path`의 `temp_stand_busy` 차단은 실제 주기 상태일 때만 발동 → 통과 트래픽 허용.
  - `_try_splice_temp_stand_arrival_detour`: `_build_prep_xy_to_xy_phase(..., snap_exact_start_xy=True, snap_exact_end_xy=True, ...)` 로 호출해 temp_prep 첫 micro-segment의 `p0`를 `start_xy`로 정확히 스냅 → Landing–ArrTaxi_Temp 이음새의 중복 구간·역행·속도 리셋 제거.
- **evidence** (default_layout, R7=id_gedp95dvr, R8=id_kpc2h0amv):
  - **A 확인**: 수정 후 R7 T001 최단거리 **0.00 px** (이전: 121 px에서 stuck). R7 phase sequence `Landing → Arr_taxi_occupied(temp) → (parked) → Arr_taxi` 로 정상 진행. R8 도 T002 도달 → EIBT **None → 03/31 01:29:56** 획득.
  - **B 확인**: R7 `Landing→Arr_taxi_occupied` 전환 속도 `1.73 → 14.98` m/s, 위치 역행 제거 (t=4070→4072: `(2855.6, 2993.9) → (2877.2, 2973.1)` 정방향). R8 동일 구간도 `12.23 → 14.98`.
  - **회귀 없음**: R1/R2/R3 EIBT 불변 (00:21:18, 00:18:32, 00:24:48), MNL_OSM PASS run+validate.
  - **결정성**: 동일 입력 2회 실행 R7 positions SHA-256 일치 (`2008bb7346c694ab`).
- **note**: default_layout sim 종료는 여전히 DEADLOCK_CAP에서 중단 (무관한 Dep_taxi 혼잡으로 R1/R2/R3 가 다른 이유로 ghost 됨) — 본 이슈와 별개.

---

### RUN 20260423T0903Z dep-runway-hold-buffer
- **RUN_ID**: 20260423T0903Z
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/MNL_OSM_sim_result.json`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/default_layout_sim_result.json`
- **result**: PASS (all runs), PASS run+validate
- **problem**: 활주로 용량이 찼을 때 RTX/RET(runway_taxiway / runway_exit)에서 정차하는 위치가 접근로의 폴리라인 꺾임점 개수(=graph 노드 밀도)에 좌우됨. 꺾임점 多 → `runway_holding` 노드가 활주로에 가까이 스냅되어, 출발기가 활주로에 너무 근접한 지점에서 `Holding_lineup` → `runway_occupied` WAIT에 빠짐.
- **change**: `utils/airside_sim.py`
  - 신규 상수 `DEP_RUNWAY_HOLD_BUFFER_M = 100.0` (along-path 거리, m)
  - 신규 helper `_dep_runway_entry_remaining_m(agent, ppm)` — 현재 micro-segment 진행 + 앞쪽 미끝 세그먼트들을 합쳐, 경로상 다음 `runway` 세그먼트까지 남은 m 거리 계산 (폴리라인 정점 밀도 무관).
  - `can_reserve_path` 내 `idx == 0` 블록에 **거리 기반 departure-runway hold 게이트** 추가: `ph0 ∈ {Dep_taxi, Holding_lineup, Lineup_departure}` 이고 `pt0 ∈ {runway_taxiway, runway_exit}` 일 때, `rem_m ≤ DEP_RUNWAY_HOLD_BUFFER_M` 이면 기존 dep_rwy 검사(`runway_rot_busy` / `runway_dep_busy`)를 조기 수행 → WAIT.
- **evidence** (MNL_OSM, id_26ltg5sfz 이륙 중 `rwy-1ea07258fb27`를 점유한 시간대):
  - `id_2njdlcz0j` 최초 hold 물리거리: **44.39 m → 63.46 m**
  - `id_fvwl7ufoo` : **35.67 m → 61.44 m**
  - `id_56g5h2ra5` : **33.71 m → 73.75 m**
  - `id_hmt0zdqsd` : (이전 `edge_capacity`로 129.55 m에서 먼저 대기) → `runway_dep_busy`로 58.61 m에서 대기
  - `max simultaneous on runway: 1; ticks with >1: 0` (warp-fix와 동일 invariant 유지)
  - 접근로 WAIT reason 분포: `[runway_dep_busy(161), temp_stand_busy(79), edge_capacity(44)]` (기존 `runway_occupied`가 더 이르고 깔끔한 `runway_dep_busy`로 대체)
  - 결정론 검사: 동일 입력 2회 실행 SHA256 일치 (`620b499fb7c6bb11`)
  - 회귀: `default_layout_sim_input.json` `PASS run+validate`
- **note**: `DEP_RUNWAY_HOLD_BUFFER_M`는 along-path 기준 100 m. MNL_OSM의 RET(runway_exit)가 활주로 코너에 접근하는 기하에서는 경로 대비 수직 물리 거리가 약 0.5× 비율(예: path 87 m ↔ physical 43 m, path 47 m ↔ physical 24 m)이라 path-100 m ≈ physical-50 m 수준의 holding offset을 확보.
- **next**: RPLL 등 다른 레이아웃에서도 동일 효과 회귀 확인 요청 시 추가 검증.

---



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

---

### RUN 20260423T — MNL RET·RTX → physical runway resource
- **command**: `python -m harness.smoke` → `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/_MNL_OSM_sim_result_fix2.json`
- **result**: PASS run+validate
- **root cause / fix**:
  - `runway_exit` / `runway_taxiway` graph `linkId`가 `tw-*`일 때 `RunwayResource` 키를 tw로 잡아 활주로(`rwy-*`)과 분리 → 동일 시각 다중 라인업(예: 001+003) 동시 점유.
  - `_physical_runway_id_for_graph_link` + `build_resource_model`에서 tw를 `_arr_ret_runway_junction_xy`로 `runwayPaths` id에 합침. `RUNWAY_LINK_PATH_TYPES`는 phase·재진입·우선순위 분기용으로 유지.
- **evidence (샘플 스크립트)**: old `MNL_OSM_sim_result` 동시 Hold on 키 엣지(001/002/003/105) max **3**; fix2 결과(full t) max **1**.

---

### RUN 20260423T0900 — MNL Holding_lineup reroute warp guard
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/MNL_OSM_sim_input.json --output data/Result_storage/MNL_OSM_sim_result.json`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_default_fix_warp.json`
- **result**: PASS run+validate (MNL + default_layout)
- **builder change**:
  - `_try_reroute_agent_off_path_block()`에서 `PHASE_HOLDING_LINEUP` / `PHASE_LINEUP_DEPARTURE`를 `PHASE_LANDING`과 동일하게 reroute 차단 대상에 포함.
- **root cause**:
  - R7(`id_56g5h2ra5`)가 `t=29100`에 RTX `layout-edge-003` 위 (392.74, 3667.23)에서 활주로 점유 해제를 기다리다가 `total_wait_sec`가 임계치를 넘어 `t=29134`에 non-aggressive reroute 발동.
  - `build_reroute_path_from_xy → _flight_route_impl`이 `RouteEndpoint(token_pixel_xy=start_xy)`를 `g.nearest_path_node()`로 스냅하면서 RTX 폴리라인 중간에 있는 (392.74, 3667.23)의 최근접 그래프 노드로 **활주로** 위의 node 0(389.56, 3707.93, 40.82px)이 선택됨(RTX의 두 끝 노드 node 1 / node 71은 각각 51.53 / 64.91px로 더 멀었음).
  - 그 결과 새 경로 = `node 0 → node 1` = `layout-edge-001`(runway)이 되고, 에이전트 위치가 segment 직선 위로 스냅되어 (409.53, 3696.46)로 **워프**. 이 탓에 R2가 이륙 중인 활주로에 R7이 동시에 점유해 occupancy가 2~5까지 치솟음.
  - Lineup 계열 단계는 경로가 이미 `RTX → lineup → runway takeoff`로 고정이며, 다른 대체 경로가 없고 단지 활주로가 비기를 기다리는 상태이므로 reroute가 필요 없음. Landing과 같은 선상에서 차단하는 것이 최소 변경.
- **reviewer check**:
  - `Information.json` 변경 없음
  - I/O 계약 유지(`PASS run+validate` 양쪽 입력)
  - `default_layout_sim_result.json` 골든과 결과 완전 일치(회귀 없음)
  - MNL 2회 독립 실행 결과 JSON 완전 동일(결정성 유지)
- **evidence**:
  - R7 trajectory (fix): `t=29100` WAIT@edge-003(392.74, 3667.23) → `t=29144` PROCEED@edge-003(411.91, 3672.94) → `t=29148` edge-002(449.72, 3673.38)로 자연스럽게 진입. 워프 소실.
  - 동시 runway 점유: 전체 시뮬레이션에서 old `max=5`, overlap 샘플 **289** → fix `max=1`, overlap 샘플 **0**.
  - MNL `deadlock_resolve_event_count = 0`.
- **next**: RTX·lineup 구간에 걸린 다른 기체 시나리오에서도 동일 가드가 과도하게 이동을 묶지 않는지 샘플 회귀 검증.


---

### RUN 20260428T0548Z remove-sd-schedule-layer
- **command**:
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/default_layout_sim_input.json --result data/Result_storage/_validation_sim_result.json`
- **result**: PASS run+validate
- **builder change**:
  - Removed active S(d) schedule fields from UI, simulation input/result contract, and stored JSON.
  - Plain `sldtMin/sibtMin/sobtMin/stotMin` are now the single S schedule source.
  - Removed automatic OVLP/stand-delay push; same-stand SIBT-SOBT overlap is blocked at assignment.
- **alternatives compared**:
  - Smoke only: fastest but would not prove schema run compatibility.
  - Default run+validate: selected because it checks the changed input/result contract with low runtime.
  - Broader MNL run: stronger coverage but unnecessary before the default schema smoke passes.
- **reviewer check**:
  - `Information.json` unchanged.
  - S(d) keys removed from active source/data search.
  - Default harness output generated at `_validation_sim_result.json`.
- **next**: If needed, run a larger airport input after UI manual checks.

---

### RUN 20260428T0628Z s-anchored-eldt-fix
- **command**:
  - `python -m py_compile utils/airside_sim.py`
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/default_layout_sim_input.json --result data/Result_storage/_validation_sim_result.json`
- **result**: PASS run+validate
- **root cause**:
  - S-only 전환 후에도 `_s_eldt_sec()`가 stale `eldtMin` 입력을 우선 사용해 `R6_5`가 S schedule보다 이른 16분 anchor로 시뮬레이션됨.
- **builder change**:
  - `_s_eldt_sec()` now anchors simulation from `sldtMin` only.
- **evidence**:
  - Before: `R6_5` SIBT `00:50:00`, EIBT `00:22:35`.
  - After validation run: ELDT `00:45:00`, EIBT `00:51:11`, EOBT `02:01:13`.
- **alternatives compared**:
  - Ignore `eldtMin` in `airside_sim.py`: selected, fixes simulation contract at source.
  - Remove stale E fields from input JSON: useful cleanup but not sufficient if future inputs include them.
  - Clamp output E fields after simulation: hides the symptom while positions/timing remain inconsistent.

---

### RUN 20260428T0634Z remove-e-min-input-fields
- **command**:
  - `node --check pages/Layout_Design/designer.js`
  - `python -m py_compile utils/airside_sim.py`
  - `python -m harness.smoke`
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
  - `python -m harness.validate --input data/Result_storage/default_layout_sim_input.json --result data/Result_storage/_validation_sim_result.json`
- **result**: PASS run+validate
- **builder change**:
  - Removed `eldtMin/eibtMin/eobtMin/etotMin` from sim/layout serialization allowlists.
  - Scrubbed stale E-minute fields from stored JSON payloads under `data/`.
  - Kept runtime E-minute cache fields for result display/playback.
- **evidence**:
  - `rg` found no E-minute fields in `data/*.json`.
  - `simFlightKeys` no longer contains `eldtMin`.
