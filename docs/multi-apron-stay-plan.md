# Multi-Apron-Stay Implementation Plan (V3 — Final)

## 0. 한 줄 요약

한 편(flight)이 N개 주기장을 시간순으로 옮겨다닐 수 있게 한다(`AP1..APN`).
N=1 이 기본·기존 동작. N≥2 는 사용자가 Apron Gantt 에서 split + 다른 stand 로
드래그했을 때만 발생.

## 1. Frozen Decisions

| #   | Decision                                                                                                                                                                                  |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | `flight.arrApronId`(=AP1), `flight.depApronId`(=AP_last) **두 필드로 분리**. 중간 stand 는 `apronStaySegments[i].standId` 만으로 보유. 기존 단일 `f.standId` 의존 코드는 `depApronId` 와 alias. |
| Q2  | AP 번호 매김 = **시간순 정렬 + 인접 standId 변경 지점마다 +1**. 인접 같은 stand 는 1개의 AP 로 합산. 같은 stand 가 비인접 위치에 다시 등장하면 새 번호.                                                |
| Q3  | **각 조각 dwell ≥ 20분**. 미만이면 split UI 비활성/거절(툴팁 안내).                                                                                                                       |
| Q4  | 자른 점은 **`SIBT_{i+1} = SOBT_i` 강제**(no gap).                                                                                                                                         |
| Q5  | Flight Schedule 컬럼 = **동적 K = max(N over all flights)**. 빈 셀 `—`.                                                                                                                  |
| Q6  | 별도 수동 K 없음. Q5 와 동일.                                                                                                                                                             |
| Q7  | AP 라벨 = **stand name** (없으면 id). 표·Gantt 일치.                                                                                                                                     |
| Q8  | `extract_point_to_paths` / phase 시퀀스를 **반복형**으로. 점유 시 `ARR_TAXI_TEMP_OCCUPIED` 분기 기존 그대로.                                                                              |
| Q9  | **반복 i=1 부터 N**. 첫 도착 ARR_TAXI 도 1회로 카운트. (N=1 ⇒ 기존 동작)                                                                                                                |
| Q10 | `ARR_TAXI_OCCUPIED` 의미 그대로 (점유 대기). 별도 dwell phase 신설 X.                                                                                                                    |
| Q11 | post-pushback cooldown 등 stand 자원 룰 **반복 적용**. 자기 자신 예외 없음.                                                                                                              |
| Q12 | deadlock / temp-stand reroute 룰 그대로 반복.                                                                                                                                             |
| Q13 | KPI 7개 정의: ELDT, EXIT_RUNWAY, **EIBT1**, **EOBT_LAST**, **E_PUSH_FINISHED_LAST**, E_LINEUP, ETOT 만 사용.                                                                              |
| Q14 | sim_result schedule row 에 `EIBT_LIST` / `EOBT_LIST` / `E_PUSH_FINISHED_LIST` / `STANDS` 부속 필드만 추가. KPI 계산 미반영.                                                              |
| Q15 | Split 트리거 = **호버 칩** + Alt+클릭 보조.                                                                                                                                               |
| Q16 | 빨간 점 = **드래그(시각 변경)** + **클릭(Merge 메뉴)**.                                                                                                                                   |
| Q17 | Pro Sim stale 상태에서도 split **가능** (변경 즉시 stale 마킹).                                                                                                                          |
| Q18 | 기존 sim_input.json 의 단일 segment = **로드 시 1구간으로 자동 보강** (메모리만, 디스크 안 건드림).                                                                                       |
| Q19 | 회귀 시나리오: ① N=1 (기존 시나리오 모두) ② N=2 같은 stand ③ N=2 다른 stand ④ N=3 다른 stand.                                                                                            |
| Q20 | split 만 하고 옮기지 않은 상태 = **시뮬은 N=1**, **시간 조정 잠금**.                                                                                                                     |
| F1  | `arrApronId = AP1`, `depApronId = AP_last`. N=1 이면 두 값 동일.                                                                                                                          |
| F2  | AP 번호: **시간순 + standId 변경 지점마다 +1**. 인접 같은 stand → 1개 AP. AP1→AP2→AP1 → 라벨 `AP1, AP2, AP3`.                                                                             |
| F3  | 동일 stand 인접 segment 의 빨간 점은 **드래그/Merge 잠금**. 다른 stand 행으로 드래그하는 동작만 가능 (활성화 트리거).                                                                     |
| F4  | K 변경 시 **Flight Schedule 표 전체 리렌더**. (가상 스크롤 안전성 우선)                                                                                                                  |
| F5  | **직렬화 단계에서 같은 stand 인접 segment 자동 병합** (sim_input.json 은 N=1 이 되도록).                                                                                                  |

## 2. Data Model

### 2.1 Flight 객체 (Designer state.flights[i])

```json
{
  "id": "flight_xxx",
  "arrApronId": "stand_ap1",
  "depApronId": "stand_ap2",
  "apronStaySegments": [
    { "standId": "stand_ap1", "sibtMin": 60, "sobtMin": 100 },
    { "standId": "stand_ap2", "sibtMin": 130, "sobtMin": 170 }
  ],
  "sibtMin": 60,
  "sobtMin": 170,
  "dwellMin": 80,
  "sldtMin": 55,
  "stotMin": 175,
  "standId": "stand_ap2"
}
```

- N=1 invariant: `arrApronId == depApronId == segments[0].standId`, `segments.length == 1`.
- Aggregate field invariants:
  - `sibtMin = segments[0].sibtMin`
  - `sobtMin = segments[N-1].sobtMin`
  - `dwellMin = Σ (sobtMin_i − sibtMin_i)`
  - `sldtMin = sibtMin − SCHED_SIBT_MINUS_SLDT_MIN` (5)
  - `stotMin = sobtMin + SCHED_STOT_MINUS_SOBT_MIN` (5)
- 호환 alias: `f.standId == depApronId`.
- AP 번호 매김 알고리즘:

```text
ap_label_index = 0
prev_stand = None
for seg in sorted(segments by sibtMin):
    if seg.standId != prev_stand:
        ap_label_index += 1
    seg._apLabel = "AP" + str(ap_label_index)
    prev_stand = seg.standId
N_logical = ap_label_index   # 같은 stand 인접 합산 후의 N
```

### 2.2 직렬화 (`pages/Layout_Design/flight-timeline.js`)

- `simFlightKeys` 에 `apronStaySegments`, `arrApronId`, `depApronId` 추가.
- **F5 정규화**: 출력 직전 같은 standId 인접 segment 를 1개로 병합.
  → 같은 stand 만 split 한 상태는 sim_input.json 에서 N=1 로 출력됨.

### 2.3 sim_result schedule row 추가 필드

```json
{
  "STANDS": ["stand_ap1", "stand_ap2"],
  "SIBT_LIST": [3600, 7800],
  "SOBT_LIST": [6000, 10200],
  "EIBT_LIST": [3650, 7850],
  "EOBT_LIST": [6050, 10250],
  "E_PUSH_FINISHED_LIST": [6080, 10280],

  "SIBT": 3600,
  "SOBT": 10200,
  "EIBT": 3650,
  "EOBT": 10250,
  "E_PUSH_FINISHED": 10280
}
```

- 단수 alias 규칙:
  - `SIBT = SIBT_LIST[0]`
  - `SOBT = SOBT_LIST[-1]`
  - `EIBT = EIBT_LIST[0]`
  - `EOBT = EOBT_LIST[-1]`
  - `E_PUSH_FINISHED = E_PUSH_FINISHED_LIST[-1]`
- N=1 일 때 LIST 길이 1, 단수 alias 와 동일 값.

### 2.4 듀레이션 매핑 (Q13)

| 컬럼            | 정의                                                  |
| --------------- | ----------------------------------------------------- |
| `ARR_ROT_SEC`   | `EXIT_RUNWAY − ELDT`                                  |
| `VTT_ARR_SEC`   | `EIBT_LIST[0] − EXIT_RUNWAY`                          |
| `PUSHBACK_SEC`  | `E_PUSH_FINISHED_LIST[-1] − EOBT_LIST[-1]`            |
| `DTT_ARR_SEC`   | `[EXIT_RUNWAY .. EIBT_LIST[0]]` 0속도 합              |
| `DTT_DEP_SEC`   | `[E_PUSH_FINISHED_LIST[-1] .. E_LINEUP]` 0속도 합     |
| `VTT_DEP_SEC`   | `E_LINEUP − E_PUSH_FINISHED_LIST[-1]`                 |
| `DEP_ROT_SEC`   | `ETOT − E_LINEUP`                                     |

## 3. Apron Gantt UI

### 3.1 막대 그리기

- 한 flight = N 개 막대. 시각상 갭 0 (빨간 점만).
- 막대 폭 = `(SOBT_i − SIBT_i)`.
- 각 막대는 그 segment 의 standId 행에 배치.
- **노란 선택 효과(`alloc-flight-selected`)** 는 **flight 단위** = AP1..AP_last 모두 동시 하이라이트.
  - 구현: `[data-flight-id="X"]` 전체에 클래스 부여.

### 3.2 Split

- `.alloc-flight:hover` → 우상단 `Split` 칩.
- Alt+클릭 단축키.
- 클릭한 X(분) 에서 segment 분할.
- **각 조각 ≥ 20분** 미만이면 거절 (칩 비활성/툴팁 “Each part must be ≥ 20 min”).
- 새 조각 standId = 부모 standId(같은 stand 로 시작).

### 3.3 빨간 점(JUNCTION)

- 위치 = `SOBT_i = SIBT_{i+1}`.
- 스타일: `width:7px; height:7px; background:#ef4444; border:1px solid #7f1d1d;`.
- **드래그**: 좌우 이동, 양쪽 dwell ≥ 20분 클램프.
- **클릭**: Merge 컨텍스트 메뉴 (인접 두 segment 병합).
- **F3 잠금**: `segments[i].standId == segments[i+1].standId` 일 때:
  - 드래그/Merge 비활성, 툴팁 “Move to a different stand to enable timing edits”.
  - 단, 빨간 점 자체는 시각적으로는 표시.

### 3.4 Stand drop / 행 배치

- 막대마다 `data-flight-id` + `data-segment-idx` 부여.
- 드롭 결과: **드래그한 segment 의 standId 만** 갱신.
- 드롭 후 즉시 `arrApronId / depApronId` 재계산 + AP 라벨 재계산.
- 충돌 검사: `flightWouldOverlapStandAssignment(flight, standId, segmentIdx)` segment 단위.

### 3.5 시간 조정 잠금/해제 (Q20 / F3)

- segment 의 `standId` 가 좌·우 인접 segment 와 다르면 시간 편집 가능 (SIBT/SOBT 핸들 + JUNCTION 드래그).
- 같으면 잠금. 다른 stand 로 옮긴 즉시 잠금 해제.

## 4. Flight Schedule Table

### 4.1 컬럼 (동적 K)

- `K = max(N_logical for all flights)`.
- 컬럼 순서:

```text
... 기본 | SLDT | SIBT1..K | SOBT1..K | ELDT | EIBT1..K | EOBT1..K | ETOT | AP1..K
```

- 행마다 `n < K` 위치는 `—` (`data-empty="1"`).
- 헤더 라벨: `SIBT1`, `SIBT2`, …; AP 컬럼은 stand **name** (없으면 id).

### 4.2 인덱스 상수

- 기존 `FLIGHT_SCHED_TD_*` 상수는 **K 의존 함수**로 변경:

```text
function flightSchedColIndex(field, k) { ... }
```

- `FLIGHT_SCHED_TABLE_COL_COUNT` = base + 4*K (SIBT/SOBT/EIBT/EOBT) + K (AP) + 4 (SLDT/ELDT/ETOT/...).

### 4.3 리렌더 (F4)

- K 변경 감지 (split / 새 split 막대 stand 변경 / flight 추가) → **표 전체 리렌더**.
- 가상 스크롤 컨테이너는 K 의존 너비/그리드 다시 계산.

## 5. airside_sim.py FSM (반복형)

### 5.1 phase 시퀀스 빌더 (sim 내부)

```text
[LANDING]
for i in 1..N:
    [ARR_TAXI → AP_i]                      # 점유 시 ARR_TAXI_TEMP_OCCUPIED 기존 분기
    [APRON dwell at AP_i]                  # phase = ARR_TAXI_OCCUPIED 의미 (점유 대기 + 정주)
    [PUSHBACK → AP_i 떠남]
[DEP_TAXI]
[HOLDING_LINEUP]
[LINEUP_DEPARTURE]
```

- N=1 ⇒ 결과 바이트 동일성 유지 (회귀 안전망).
- AP_i → AP_{i+1} 사이 전이는 다음 루프 `[ARR_TAXI → AP_{i+1}]` 가 자연 처리.

### 5.2 `extract_point_to_paths` 변경

- 정적 6-leg 가정 제거.
- N에 따라 leg 동적 생성:

```text
(td → A),
(A → AP_1),
(AP_1 → AP_2), ..., (AP_{N-1} → AP_N),
(AP_N → PB+1), (PB+1 → HOLD), (HOLD → LINEUP), (LINEUP → END)
```

- `_EXTRACT_LEG_PHASES` 상수 → **빌더 함수**.

### 5.3 Flight 상태 확장 (`utils/airside_sim.py:Flight`)

- `apron_segments: List[Dict]` 추가.
- `current_apron_segment_idx: int`.
- `actual_apron_inblocks_abs_sec_list: List[Optional[float]]`  (길이 N).
- `actual_apron_offblocks_abs_sec_list: List[Optional[float]]`.
- `pushback_finished_abs_sec_list: List[Optional[float]]`.
- 단일 alias 유지:
  - `actual_apron_inblocks_abs_sec ≡ list[0]`
  - `actual_apron_offblocks_abs_sec ≡ list[-1]`
  - `pushback_finished_abs_sec ≡ list[-1]`

### 5.4 dwell 게이팅

- 각 PUSHBACK 시작 = `inblocks_i + dwell_i` (`dwell_i = SOBT_i − SIBT_i`).
- `dep_taxi_start_*` 게이팅은 **마지막 segment** 기준만 적용.

### 5.5 stand 자원 / cooldown

- 기존 룰 그대로 반복 적용.
- post-pushback cooldown 도 동일 (자기 자신 예외 없음).

### 5.6 deadlock / temp-stand reroute

- 기존 로직 그대로 반복.
- 재계산 시 `current_apron_segment_idx` 보존 점검 추가.

## 6. 회귀 / 검증 (Q19)

시나리오:

1. N=1 — 모든 기존 시나리오. 결과 바이트 동일.
2. N=2 같은 stand — 직렬화 후 1구간으로 정규화(F5) → (1) 과 동일 결과.
3. N=2 다른 stand — 신규.
4. N=3 다른 stand — 신규.

검증 항목:

- sim_input.json `apronStaySegments` 필드 존재/형태.
- sim_result.json `STANDS / *_LIST / E_PUSH_FINISHED_LIST` 형태.
- KPI 7개 컬럼 값 (Q13 정의대로).
- Pro Sim positions 가 다구간일 때 phase 라벨 순서 일치.
- Apron Gantt: 막대 N개, 빨간 점 N-1개, 흰 핸들 좌우 1개씩, 노란 선택이 모든 막대에 적용됨.

## 7. PR 단위 (위험 최소화)

| PR  | 범위                                                                                                                | 회귀 안전                       |
| --- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 1   | Data model 도입(`apronStaySegments`, `arrApronId/depApronId`) + invariant 헬퍼 + 직렬화(F5 정규화)                 | sim_input.json N=1 시 바이트 동일 |
| 2   | Apron Gantt: multi-bar + 빨간 점 + Split UI(20분 제약) + 선택 동기화                                                | N=1 시 동일                     |
| 3   | JUNCTION 드래그/Merge + F3 잠금 + segment 단위 stand drop / 충돌 검사                                              | N=1 시 동일                     |
| 4   | Flight Schedule 표 동적 K + AP 라벨 + F4 전체 리렌더                                                               | N=1 시 동일                     |
| 5   | airside_sim.py: extract_point_to_paths 빌더화 + N=1 동등성 보장 + N>1 ValueError 가드 제거 준비                    | N=1 동일                        |
| 6   | airside_sim.py: Flight 상태 확장(LIST 필드) + 반복 FSM + dwell 게이팅                                              | N=1 동일                        |
| 7   | sim_result alias / LIST 필드 + KPI overlay 재맵 (Q13)                                                              | N=1 동일                        |
| 8   | 회귀 시나리오 4종 추가 + 문서 정리                                                                                  | —                               |

## 8. 알려진 위험·완화

- **가상 스크롤**: K 변경 시 행 높이/그리드 안정성 — 전체 리렌더(F4) 로 회피.
- **deadlock 보호**: N>1 비행이 reroute 될 때 segment list invariants — 5.6 점검 로직 추가.
- **선택 동기화**: 동일 flight 의 N개 DOM에 동시 선택 — `[data-flight-id]` 단위 일괄 조작.
- **빨간 점 hit-area**: 점이 작아 드래그 어려울 수 있음 — 투명 padding 영역 추가 권장.
