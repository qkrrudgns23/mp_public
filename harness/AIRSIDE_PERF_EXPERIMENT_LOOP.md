# `airside_sim.py` 성능 개선 × 반복 실험 가이드

**목표**: 벽 시간(wall clock) 단축, 세 시나리오 **골든 JSON deep-equal** 유지, **시뮬레이션 의미·결과 동등성** 불변.

**상세 미시 후보**(줄번호·위험도·근거): `harness/OPT_CANDIDATES_v2.md`  
**실행 하네스**(스모크 포함 단일 PASS 사이클): `python -m harness.golden_opt_cycle`  
**반복 통계**(중앙값·분산): `python -m harness.opt_repeat_experiment`

---

## 1. 불변 규칙 (전체 로직 훼손 금지)

| 규칙 | 확인 방법 |
|------|-----------|
| 동일 입력 → 동일 결과 | `golden_opt_cycle`: `default_layout` / `large_flight` / `MNL_OSM` 각각 `golden_compare` PASS |
| 사양 변경·휴리스틱 완화 금지 | 동일 틱에 대한 예약·충돌·히스토리 필드 의미 변경 없음(리팩만) |
| 승인 없는 폴백 금지 | 조용한 기본값·예외 무시 패턴 추가하지 않음 |
| 최소 패치 | 한 PR/한 커밋당 후보 1건 권장(원인 분리) |

FAIL 시 **즉시** `utils/airside_sim.py` 원복 후 다음 후보로 넘긴다.

---

## 2. 코드베이스에 이미 반영된 항목

| 라벨 | 출처(MD #) | 요약 |
|------|------------|------|
| `opt_candidates_batch` | 1, 2, 3, 6, 8, 11, 12, 13, 16, 18, 20 근처 | 터치다운 dict 스냅, 히스토리 변수 호이스트·재사용, 정렬키/예약키 캐시, `can_reserve_path` 등 |
| 통합 OCC clear 루프 | **#9** (추가) | `refresh_resource_occupancy`: 네 자원맵 `.values()`를 한 튜플로 순회해 `occupied_by.clear()` — 동작 동일 |

> `OPT_CANDIDATES_v2.md` 적용 로그 표의 **“#9 보류”** 줄은 통합 클린 루프 적용 전 스냅샷이다. 현재 트리는 위 표와 동기화된다.

---

## 3. 다음 실험 큐 (위험도 낮은 순·권장 순서)

`OPT_CANDIDATES_v2.md` 원문 번호와 연결한다. **한 행만** 패치 → 골든 → 반복 측정.

| 순서 | MD # | 내용 요약 | Risk | 비고 |
|------|------|-----------|------|------|
| A1 | **#10** | `_resolve_all_head_on`: `len(grp)<2`는 이미 스킵; 추가로 **단일 ag인 edge에 대해 리스트 빌드를 줄이는** 자료구조 변경은 검토 필요(동등성 테스트 필수). | 2 | 이득·복잡도 트레이드오프; 미적용 상태 |
| A2 | **#4** | `_destination_stand_history_snap`: 호출측에서 `stand_cooldown_index`만 사용하는지 재확인·불필요 재계산 제거 가능성 | 2 | 변경이 0줄일 수 있음 |
| B1 | **#5** | `_build_stand_pushback_clearance_index` 부분 갱신으로 풀 재빌드 감소 | 3 | 틱 경계 검증 필요 |
| B2 | **#7** | `Flight` 틱별 `_eid0_cached` 무효화 | 3 | 무효화 누락 시 골든 붕괴 위험 |
| 보류 | **#15** ~ **#17** | 인라인·집합 합 변경 등 의미 변경 가능성 | 3–5 | 실험 루프 밖 검토 권장 |
| 보류 | **#19** | `sorted(agents)` 캐시·무효화 | 4 | 이벤트 경계 명세화 필요 |

---

## 4. 반복 실험 절차 (이 문서의 “표준 루프”)

### 4.1 패치 없이 현재 헤드 기준선

처음 실행은 스모크까지 포함하고, 이후는 시뮬+골든만 반복하는 편이 실측 분산이 줄어든다.

```bash
# 기준선: 스모크 1회 포함 + N회 사이클
python -m harness.opt_repeat_experiment --reps 8

# 이미 한 세션 안에서 연속 재측정만 할 때 (스모크 스킵)
python -m harness.opt_repeat_experiment --reps 10 --skip-smoke
```

출력 예: 각 `rep … sum_wall=…`, 마지막에 `median`, `mean`, `stdev`. **의사결정은 median 우선**(OS 노이즈 완화).

### 4.2 단일 미시 패치 채택 루프

1. **큐**에서 순서 선택 → `airside_sim.py` 최소 diff.
2. **골든 단발**:  
   `python -m harness.golden_opt_cycle --tag perf_<후보라벨>`  
   실패 시 원복 후 종료.
3. **반복 시간**: 동일 명령으로 `--reps` 맞춰 재실행 후 §5 표에 기록.
4. median이 기준선 대비 안정적으로 감소**하고** 골든 통과 유지 시 채택, 아니면 원복.

### 4.3 기대치

합계 `sum_wall`(3 시나리오 합)은 **수백 ms~1s급 JIT** 가능. 패치 채택은 **통계 + 골든** 둘 다 만족할 때만.

---

## 5. 실험 결과 로그 (수동 업데이트)

| 실행일 (UTC 권장) | git 짧은 SHA | 명령 | reps | median sum_wall(s) | stdev | golden |
|-------------------|--------------|------|------|---------------------|-------|--------|
| 2026-04-30 | *(working tree)* | `opt_repeat_experiment --skip-smoke` | 5 | **33.88** | 0.33 | PASS |

히스토리: `rep`별 `sum_wall` 33.46, 33.50, 33.88, 33.90, 34.25 (s).

---

## 6. 참고: 한 방에 검증하지 않을 것

- 골든 없이 시간만 재는 배치 최적화(회귀 감지 불가).
- 여러 미시 패치 동시 적용 후 실패 시 원인 분리 불능.
