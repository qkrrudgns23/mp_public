## Lessons Learned (반복 방지용 교훈)

### Principles (누적/병합)
- **근거 우선**: 오류 원인은 스택트레이스/로그/입출력 파일/검증 결과로만 판단한다.
- **무한 재시도 금지**: 같은 커맨드를 원인 미확인 상태로 반복하지 않는다.
- **먼저 validator**: 결과가 “생성은 됐는데 내용이 깨짐”을 빠르게 잡기 위해 validator를 먼저 강화한다.
- **먼저 smoke**: 구문 오류/임포트 실패는 `python -m harness.smoke`로 선제 차단한다.
- **수정 후 확인 의무**: 코드 수정만으로 완료 처리하지 않는다. 반드시 재현 지표를 직접 재검증하고, PASS/FAIL 근거를 남긴다.
- **터미널 검증 의무(예외 없음)**: 어떤 수정이든 반영 후 반드시 내가 터미널에서 실행(`smoke` + `run`/필요시 `validate`)하고 결과를 확인한다.

### Known Pitfalls (현재까지)
- **RET/RTX `linkId`는 tw\***: `simPathGraph`의 `runway_exit` / `runway_taxiway`는 `linkId`가 `tw-*`인 경우가 많다. `RunwayResource` 키를 `rec.link_id`로만 쓰면 `rwy-*`와 따로 떨어져 동시 라인업(다른 엣지)이 capacity=1을 우회할 수 있다. `runwayPaths`와의 접점( `_arr_ret_runway_junction_xy` )으로 물리 활주로 id에 합친다.
- **하네스 스크립트 구문 오류**: 문자열 이스케이프/따옴표 실수로 `SyntaxError`가 나면, 실행 전 단계에서 smoke가 잡도록 한다.
- **미해결 상태 종료 금지**: 핵심 재현 지표가 unchanged/악화면 루프를 종료하지 않는다. 최소 1개 핵심 지표 개선을 확인할 때까지 재실행한다.
- **temp incident 맵 공백 리스크**: temp stand ID가 그래프 stand map에 없을 수 있으므로, 좌표 기반 최근접 노드 fallback 없이는 temp 점유 회피 로직이 무효화된다.
- **비결정성 리스크**: 의사결정 tie-break에 Python `hash()`를 쓰면 실행마다 결과가 흔들린다. 안정 해시로 고정한다.
- **검증 누락 리스크**: 사용자 보고 이슈(R7/R8, 특정 시각 회귀)는 수정 직후 동일 시나리오로 재측정하지 않으면 "해결"로 간주하지 않는다.
- **분기 누락 리스크**: `elif` 체인으로 WAIT/YIELD 체크가 건너뛰어질 수 있다. 이동 제어에서 안전 조건은 독립 `if`로 누락 없이 평가한다.
- **활주로 강제 규칙**: deadlock용 forced-open이 있더라도 활주로 점유 안전 규칙(타 기체 점유 시 진입 금지)은 예외 없이 유지한다.
- **말뿐인 완료 금지**: 터미널 실행 로그/검증 근거 없이 “해결”이라고 보고하지 않는다.
- **dwell 방향 보존**: parked dwell 구간을 압축할 때 endpoint의 `motionForward`를 이웃 sample에서 가져오면 pushback/reverse 상태가 parked 구간 전체로 번질 수 있다. dwell 구간은 주기 직전 마지막 이동 sample의 방향을 유지하고, reverse는 실제 taxi-out moving sample부터만 적용한다.
- **stationary fallback 기준 일치**: 프런트 `getFlightPoseAtTime()`는 정지 구간에서 `prev` 비정지 벡터를 우선 사용한다. 따라서 parked dwell의 `motionForward`는 “직전 row의 bool”이 아니라, `정지 직전 마지막 raw segment`와 `정지 후 첫 moving display 방향`이 일치하도록 선택해야 `R1`/`R2` 같이 서로 다른 stand geometry에서도 nose 방향이 안정적으로 유지된다.
- **stand heading 의미 고정**: layout의 `angleDeg`는 stand nose가 아니라 tail/apron-open 방향이다. stand에 주기한 기체 방향을 판정할 때는 반드시 `nose = angleDeg + 180°` 기준을 써야 하며, helper에서 단순 normalize만 하면 `R1`/`R2`처럼 서로 반대 증상이 생긴다.
- **동적 stand 배정 fallback**: 입력 flight의 `standId/apronId`가 비어 있어도 parked dwell 방향을 판정해야 한다. 이 경우 agent 초기 필드(`ag.apron_stand_id`)만 보면 `R3`~`R6`처럼 nose 기준을 잃으므로, dwell band의 history `destinationApron.standId`를 우선 복원해서 실제 배정 stand geometry를 사용해야 한다.
- **경로그래프 캐시 + 페널티**: `_flight_route_impl`에서 그래프를 매번 재구성하면 reroute/다중 레그에서 비용이 폭증한다. 레이아웃 객체 단위로 캐시하고, 레이아웃 엣지 페널티는 그래프 변형 대신 Dijkstra 가중치에만 반영한다. 페널티 없는 경우에는 `adj`에 저장된 가중치 루프를 유지해 미세 성능 회귀를 막는다.
- **시뮬레이션 시작 시 캐시 초기화**: 프로세스 내 다른 입력/레이아웃과 섞이지 않도록 `run_simulation` 진입 시 path graph 캐시를 비운다.
- **터치다운 시각 캐시는 이동 단계와 분리**: `_compute_arr_touchdown_motion_abs_sec`는 전 기체 상태에 의존한다. 틱 초반(점유 갱신·예약·정체 프로브)에는 캐시를 쓰되, `apply_movement_controls`의 **이동 루프**에서는 기체별 순차 이동/중간 reroute로 `exit_runway_abs_sec` 등이 바뀔 수 있으므로 매 기체마다 재계산한다.
- **Lineup 단계 reroute 금지**: RTX·lineup·takeoff 구간은 경로가 이미 고정이고 대체 경로가 없다. 이 단계에서 reroute를 돌리면 `build_reroute_path_from_xy → _flight_route_impl`의 `g.nearest_path_node()`가 RTX 폴리라인 중간 위치를 활주로 위 노드로 스냅할 수 있어(RTX 그래프 엣지는 두 끝 노드만 갖고 있고 그 사이 위치는 runway 쪽 노드가 더 가까워짐), 새 경로가 `runway` 엣지로 잡히고 segment 직선 스냅으로 인해 기체가 활주로 위로 **워프**한다. Landing과 동일하게 `PHASE_HOLDING_LINEUP` / `PHASE_LINEUP_DEPARTURE`에서는 reroute를 차단한다. 이는 활주로 동시 점유(capacity≥2) 회귀의 직접 원인이었다.
- **Runway holding은 거리 기반이어야 한다**: 출발기의 runway hold 지점을 "phase 경계(Holding_lineup 첫 micro-segment)"에 의존하게 두면, RTX/RET 폴리라인 꺾임점(graph 노드) 밀도에 따라 정지 거리가 크게 변한다(꺾임점 多 → `runway_holding` 스냅이 활주로에 가까워져 기체가 활주로 코앞에 정차). `DEP_RUNWAY_HOLD_BUFFER_M` + `_dep_runway_entry_remaining_m`로 **경로상 남은 m 거리가 버퍼 이하이면** `runway_dep_busy`를 조기 발동해 WAIT 시키는 방식으로, 정점 밀도와 무관한 hold offset을 강제한다. pt0 집합은 `runway_taxiway`만으로는 부족하고 `runway_exit`도 포함해야 한다(MNL_OSM처럼 RET가 lineup 접근로로 쓰이는 레이아웃 대응).
- **경로 거리 ≠ 물리 수직 거리**: lineup 코너에서 RET는 활주로와 예각으로 꺾인다. 경로상 rem_m=50 m라도 runway 폴리라인과의 수직 거리는 그 절반 수준일 수 있다(실측 ratio ≈ 0.5). 사용자가 체감하는 "활주로에서 먼 위치" 기준을 맞추려면 along-path 버퍼를 수직 거리 목표의 약 2× 이상으로 설정한다(현재 100 m → 물리 ≈ 50 m).

