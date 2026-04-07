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

