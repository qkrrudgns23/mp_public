# `utils/airside_sim.py` 후속 최적화 후보 20선 (Golden 유지 전제)

> 작성 기준: 워크스페이스 현행 `utils/airside_sim.py` (`run_simulation` 메인 시간 루프 + 핫 헬퍼 기준).
> 모두 "동작 의미는 그대로, 같은 틱·같은 입력에서 산출 동등"을 목표로 한 **마이크로 최적화**(또는 캐시) 후보다.
> 채택 절차는 항상 `python -m harness.golden_opt_cycle --tag <id>` PASS 게이트 후 결정.
> 한 번에 1건만 적용 → 골든 → 측정.

## 점수 표 의미
- **Risk**: 1 = 거의 무위험(이름만 바인딩), 5 = 의미/순서/캐시 일관성 검토 필요.
- **Hot**: 1 = 드물게, 5 = 매 틱 × 모든 에이전트.

---

## 핵심 핫루프(메인 while) 안

### 1. `_arr_touchdown_motion_abs_sec` 매-에이전트 호출 결과 캐시 재활용
- **위치**: `apply_movement_controls`(8362, 8396), `refresh_resource_occupancy`(6026), `_update_deadlock_stagnation_probe`(7043), 메인 루프 히스토리 기록(`8795`).
- **현상**: 같은 틱 안에서 **각 ag마다 `td`를 여러 번** 다시 구함. `apply_movement_controls`는 의도적으로 re-compute(8395 주석)하지만, 그 외는 `control_state.touchdown_motion_by_id` 캐시(`_refresh_touchdown_motion_cache` 직후) 결과로 충분.
- **개선**: 같은 틱 시작 직후 캐시된 dict를 **로컬 변수에 한 번만 묶어** `td = cache.get(str(ag.id))` 형태로 호출 회피.
- **Risk**: 1 / **Hot**: 5

### 2. 메인 루프 ag-단위 변수 호이스트 (`ag.id`, `str(ag.id)`, `ag.edge_phases[0]`)
- **위치**: 8788~8899(히스토리 기록 루프).
- **현상**: 같은 ag에 대해 `ag.id`/`str(ag.id)`/`ag.edge_ids[0]`/`ag.edge_phases[0]` 등이 반복 접근.
- **개선**: 루프 첫 줄에서 `aid = ag.id`, `aid_s = str(aid)`, `eids = ag.edge_ids`, `phs = ag.edge_phases` 묶기.
- **Risk**: 1 / **Hot**: 5

### 3. 히스토리 튜플 `_pt0` / `_ph0` / `_eid0` 계산 통합
- **위치**: 8855~8866.
- **현상**: 동일 `len(ag.segment_path_types) == len(ag.edge_ids)` 검사가 두 번(메인·refresh) 반복. 메인 루프 내에서도 `_pt_eobt`, `_pt0`이 같은 분기를 사실상 두 번 계산.
- **개선**: 한 번 분기로 `pt0_s`를 계산하고 `_pt_eobt = pt0_s` 등으로 재사용.
- **Risk**: 2 / **Hot**: 5

### 4. `_destination_stand_history_snap` 재진입 줄이기
- **위치**: 호출(8851), 정의(6301).
- **현상**: 매 틱 × 모든 ag 호출되며 내부에서 `_stand_pushback_clearance_cooldown_active` 두 번(스냅 + 다른 호출 경로). `stand_cooldown_index`는 이미 매 틱 시작에 빌드돼 있어 **재계산 사실상 불필요**.
- **개선**: snap 내부에서 `stand_cooldown_index`가 None일 때 `agents` 풀스캔 폴백을 들어가지 않도록 호출자에서 항상 인덱스를 전달하는지 보강(이미 메인 루프는 전달 중). `cd` 재계산을 보장된 인덱스 경로로만.
- **Risk**: 2 / **Hot**: 5

### 5. `_build_stand_pushback_clearance_index` 다시 빌드 조건 정리
- **위치**: 8785, 8844.
- **현상**: 8843에서 한 ag에 대해 offblocks 스탬프하면 **그 즉시 다시 전체 빌드**. 보통 한 틱에 다수 ag가 동시에 스탬프되지 않지만, 빈번한 빌드 위험 존재.
- **개선**: 인덱스에 **부분 갱신**(스탬프된 ag의 sid 항목만 갱신) 헬퍼를 두고 한 틱 안 1회만 풀빌드.
- **Risk**: 3 / **Hot**: 4

### 6. `_temp_stand_pipeline_sort_key`의 `str(ag.id)` 한 번만
- **위치**: 5378, 사용 8678 부근.
- **현상**: `flights_by_id.get(str(ag.id))`와 `flight_input_order.get(str(ag.id), …)`, 마지막 튜플 키에 `str(ag.id)` 가 **3번** 호출.
- **개선**: 함수 시작에 `aid_s = str(ag.id)` 하나로 묶기.
- **Risk**: 1 / **Hot**: 4(틱마다 sorted key 호출)

### 7. `_resolve_all_head_on`의 `str(ag.edge_ids[0])` 캐시
- **위치**: 7239.
- **현상**: 한 틱에서 ag별로 `eid = str(ag.edge_ids[0])` 호출. 다른 곳에서도 동일값이 필요(`refresh_resource_occupancy` 6041). ag.id 단위로 같은 틱에서 자주 쓰인다.
- **개선**: `Flight`에 **틱 휘발성 `_eid0_cached`**(매 틱 시작에 invalidation)를 두고 함수들이 그것을 읽음.
- **Risk**: 3 / **Hot**: 5

### 8. `refresh_resource_occupancy` 내부 `str()` 폭 줄이기
- **위치**: 6041, 6049~6055.
- **현상**: 같은 ag에서 `str(ag.edge_ids[0])`, `str(er.runway_id)`, `str(ag.dep_runway_id or "")`, `str(ag.edge_phases[0])`, `str(ag.segment_path_types[0] or "")`가 각각 발생.
- **개선**: ag 단위 호이스트 후 `eid0 = str(ag.edge_ids[0])`/`ph0 = ...` 한 번만 계산.
- **Risk**: 1 / **Hot**: 5

### 9. `refresh_resource_occupancy` clear 4중 루프 단순화
- **위치**: 6007~6014.
- **현상**: 4개 자원 dict를 매 틱 풀스캔하며 `.occupied_by.clear()`.
- **개선**: 자원 dict 자체를 **틱 시작 시 한 번 만든 리스트 캐시**로 매 틱 같은 컨테이너 순회(이미 하지만 `for ... in d.values()`가 dict.values 객체 두 번 만듦). 단순히 `clear`만 묶어 **list comprehension 대신 직접 메서드** 사용.
- **Risk**: 1 / **Hot**: 5

### 10. `_resolve_all_head_on`의 사전 분류 + 조기 종료
- **위치**: 7232~7246.
- **현상**: 단일 ag만 있는 edge group도 `setdefault → 후속 검사`로 진입.
- **개선**: 1개짜리 그룹은 `by_eid` 자체에 안 넣거나, 첫 ag는 list가 아닌 placeholder로 두고, **두 번째 ag가 들어올 때만 list로 승격**(메모리·반복 절감).
- **Risk**: 2 / **Hot**: 4

---

## 의사결정/예약(heavy/light) 경로

### 11. `_single_full_reservation_pass` 의 `_decision_sort_key` 호출 중 `str(ag.id)` 캐시
- **위치**: 8136~8155.
- **현상**: sort key 안에서 `str(ag.id)` 두 번(stable_tie + 마지막 튜플), `agent_states_get(ag.id)`도 한 번 호출.
- **개선**: `ag.id`를 한 번만 변수에 잡고, `rank_cache`처럼 `id_str_cache`/`tw_cache`를 미리 만들어 sort key는 dict lookup만.
- **Risk**: 2 / **Hot**: 3(heavy/light 트리거 시)

### 12. `can_reserve_path`의 `str(agent.dep_runway_id or "").strip()` 중복 제거
- **위치**: 6830, 6878.
- **현상**: 함수 도입부에서 `dep_rwy`를 잡고도 6878에서 **재계산**.
- **개선**: 두 번째 줄을 제거(이미 같은 값).
- **Risk**: 1 / **Hot**: 4

### 13. `can_reserve_path`의 `aid_key`를 항상 사용
- **위치**: 6831 vs 6866 / 6878 / 6896 등.
- **현상**: `aid_key = str(aid)`을 만들어 두고도 일부에서 `aid`(=`agent.id`) 자체로 비교 → 자료형이 달라지면 set 비교가 비싸짐.
- **개선**: 자원 비교용으로는 **`aid_key`로 일관**(특히 `{x for x in sr.occupied_by if x != aid}`은 자료형 다양 가능).
- **Risk**: 2 / **Hot**: 3

### 14. `can_reserve_path`의 `_resource_use_count`/`_lookahead_depth_billed_count` 결과 캐시
- **위치**: 6953, 6979.
- **현상**: 하나의 `idx`에 대해 `billed_here is None`이면 두 분기에서 똑같이 계산. 또 인접 idx에서도 `agent.edge_ids` 동일 prefix 기반 누적값.
- **개선**: 한 lookahead pass에서 **누적 카운터를 incremental 갱신**. 우선은 동일 idx 안에서 두 곳을 한 번 호출로 통합.
- **Risk**: 3 / **Hot**: 3

### 15. `_temp_apron_hold_reservation_only_current_edge` 등 짧은 분기 인라인화 검토
- **위치**: 8175~8179 외 reservation_pass 분기.
- **현상**: 매 ag마다 보조 함수 다섯 단계 호출.
- **개선**: 호출 빈도가 큰 1~2개를 inline 또는 결과 캐시.
- **Risk**: 3 / **Hot**: 3

---

## 자료구조/캐시

### 16. `control_state.edge_path_type_norm` 의 `ptn.get(...)` 키 변환 비용
- **위치**: 6395, 6852.
- **현상**: 매번 `str(eids[i])` 변환. `agent.segment_path_types_norm`가 있으면 바로 쓰지만 없을 때마다 변환.
- **개선**: `eids`를 build할 때부터 **str로 정규화**하여 변환 자체를 제거.
- **Risk**: 2 / **Hot**: 3

### 17. `_destination_stand_history_snap`의 `phys_others` 계산
- **위치**: 6328.
- **현상**: 매 호출마다 `set comprehension`으로 다시 만든 뒤 `len`. capacity 작을 때도 풀스캔.
- **개선**: occupied_by 자체를 작은 list로 유지 가정하고 `sum(1 for x in sr.occupied_by if x != aid)`로 set 생성 비용 제거(중복 ID가 들어올 가능성을 어디선가 보장한다는 전제).
- **Risk**: 2 / **Hot**: 5

### 18. `_agent_deadlock_ghost_at_time` 호출 비용
- **위치**: 762, 호출 전역(메인 루프, decision pass 등).
- **현상**: `float(sim_time_abs) + 1e-9 < float(u)` 매번 두 번 `float()`.
- **개선**: 호출자 쪽에서 **이미 float인 t를 넘기는 호출 경로**에서는 두 번째 `float()` 제거(필요 시 별도 빠른 경로 함수 추가).
- **Risk**: 1 / **Hot**: 5

### 19. 메인 루프 `agents` 정렬 결과 재활용
- **위치**: 8678(`agents_temp_pipe = sorted(agents, key=...)`).
- **현상**: 매 틱 `sorted()`. 키 자체는 **eldt/sldt/입력순** 기반으로 거의 변하지 않는다.
- **개선**: 입력 순서가 바뀌지 않는 동안에는 정렬 결과를 캐시하고, **이벤트(스탬프/리젝)** 발생 시에만 무효화.
- **Risk**: 4 / **Hot**: 5

### 20. `Flight.history.append` 튜플 사이즈/타입 정리
- **위치**: 8867~8885.
- **현상**: 매 틱 × ag, 14요소 튜플을 `bool()`/`float()`/`str()` 변환과 함께 append.
- **개선**: 변환은 진짜 필요한 항목만(예: `_pt0/_ph0/_eid0`은 이미 str). 또 `(_st_dbg.clearance if _st_dbg is not None else None)` 두 번 나오는 패턴을 변수에 묶기.
- **Risk**: 1 / **Hot**: 5

---

## 채택 절차(공통)

1. 위 목록에서 한 건 선택 → 변경 위치만 최소 패치.
2. `python -m harness.golden_opt_cycle --tag <slug>` PASS 확인.
3. 측정은 같은 코드로 **3회 재측정** 후 wall median으로만 비교(noise 회피).
4. PASS 누적이면 `git add utils/airside_sim.py && git commit -m "perf: <slug>"`.
5. FAIL이면 즉시 원복(`git checkout -- utils/airside_sim.py`), 다음 후보로.

## 적용 로그 (`opt_candidates_batch`, 골든 PASS)

동일 채널에서 **항목별 3연속 중앙값 A/B**(20회 전부)은 지터·시간 때문에 생략하고, 문서 후보 중 **무위험·중저위험 묶음**을 한 번에 반영한 뒤 `golden_opt_cycle --tag opt_candidates_batch`로 검증했다.

### 반영한 후보 매핑 (대략)

| MD # | 변경 요약 |
|------|-----------|
| 1 | `touchdown_motion_by_id` 로컬 스냅 + `dict.get`로 `_arr_touchdown_motion_abs_sec` 우회 (`refresh_resource_occupancy`, `apply_movement_controls` 선행 루프, `_update_deadlock_stagnation_probe`, 히스토리 루프, `_single_full_reservation_pass`) |
| 2 / 3 / 20 | 히스토리 루프: `pt0_s`/`ph0_s`/에지 리스트 재사용, `clearance`/`wait_reason` 로컬 묶음 |
| 6 | `_temp_stand_pipeline_sort_key`: `aid_s = str(ag.id)` 단일화 |
| 8 | `refresh_resource_occupancy`: `rwid0` 재사용(활주로 OCC) |
| 11 | 예약 패스 `_decision_sort_key`: `tw`/`eldt_i`/`tie`/`str(id)` 선계산 dict |
| 12 | `can_reserve_path`: 루프 내 중복 `dep_rwy = str(...)` 제거 (**문법 복구 시 `if (` 헤더 복원**) |
| 13 | apron stand `phys_others`를 `aid_key`와 `str(x)`로 일관 |
| 16 | `get_lookahead_edges`: `ptn` 분기에서 `sid_eids` 프리패스 후 `get` |
| 18 | `_agent_deadlock_ghost_at_time`: `sim_time_abs`에 불필요한 `float()` 한 번 제거 |
| — | (**배치 이후 추가**) `#9`: `refresh_resource_occupancy` OCC clear를 네 `values()` 튜플 이중 루프로 통합(동작 동등) |

### 명시적으로 건너뜀·보류 (#4 근처 제외 명단)

**#4**(스냅만 전달 확인), **#5**(인덱스 부분 갱신), **#7**(Flight 휘발 필드), **#10**(그룹 조기 종료 — 구조 대비 이득 작음), **#14**(루프 내 `billed_here`는 이미 단일 lazy), **#15**(인라인화 — 리스크 대비 이득 불명), **#17**(set→sum은 중복 점유 의미 변경 위험), **#19**(정렬 캐시, risk 4).  
*(**#9**는 위 표에 통합 OCC clear로 반영됨.)*

---

## 의도적으로 제외한 패턴

- **재시작 캐시 / 자료구조 형 변경**: 효과는 크지만 골든 동일성 위험·검토 비용 큼. 별도 트랙으로 다룬다.
- **알고리즘 변경(예: head-on 검사 거리/선분 기반 단축, lookahead 깊이 동적 축소)**: 사양 영향 가능.
- **로깅/디버그 데이터 누락**: 외부 도구가 의존할 수 있어 보존.
