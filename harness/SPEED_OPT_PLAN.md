# `airside_sim.py` 속도 최적화 계획 (Golden-Locked)

> 본 문서는 **계획서**다. 사용자의 명시적 승인 전에는 어떤 코드도 수정하지 않는다.
> 진행은 `Default-Rule.mdc`의 **Think 70% / Act 30%** 와 `harness-runner` SKILL을 따른다.

---

## 1. 목표 (Goal)

`utils/airside_sim.py::run_simulation` 의 **wall-time 단축**.
조건: **출력 결과·기능이 현재와 100% 동일해야 한다.** 한 페어라도 골든이 깨지면 그 변경은 즉시 롤백.

부수 목표:
- 핫스팟 식별과 구조 개선 (가능한 범위 내).
- 결정성(determinism) 유지 — 동일 입력 2회 실행 시 결과 SHA-256 동일.

비목표(Non-goals):
- 알고리즘적 동작/스케줄 의미 변경 금지.
- 출력 스키마/키/값 변경 금지.
- `data/Info_storage/Information.json` 변경 금지(read-only).
- `dt`(기본 1.0) 등 시뮬레이션 파라미터 변경 금지.
- 기존 result.json(골든) 파일 수정 금지.

---

## 2. 골든 페어 (Inputs / Goldens)

| pair_id          | sim_input                                                  | golden_result                                              |
| ---------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| `default_layout` | `data/Result_storage/default_layout_sim_input.json`        | `data/Result_storage/default_layout_sim_result.json`       |
| `large_flight`   | `data/Result_storage/large_flight_sim_input.json`          | `data/Result_storage/large_flight_sim_result.json`         |
| `MNL_OSM`        | `data/Result_storage/MNL_OSM_sim_input.json`               | `data/Result_storage/MNL_OSM_sim_result.json`              |

페어별 산출물(검증용, gitignore 권장):
- `data/Result_storage/_perf_default_layout_result.json`
- `data/Result_storage/_perf_large_flight_result.json`
- `data/Result_storage/_perf_mnl_osm_result.json`

---

## 3. 도구 (Toolchain)

- **실행**: `python -m harness.run --input <X_input.json> --output <_perf_X_result.json> --no-validate`
  - 기존 `harness.run`이 이미 `dt_wall`을 출력 → 그대로 활용.
  - `--no-validate`: 기존 구조 검증은 골든 비교와 무관하므로 생략(시간 절약).
- **골든 비교 (신규 필요)**: `harness/golden_compare.py` (이번 계획 승인 후 `Step 0`에서 추가 예정)
  - 입력: 두 JSON 경로 (예상, 실측).
  - 동작: `json.load` 후 **deep value equality** (`==`) 비교.
  - 다른 경우: 첫 번째 차이 경로(JSON path), 두 값 일부, 차이 카운트를 출력. exit 1.
  - 같으면: `PASS golden <pair_id>`. exit 0.
  - 부동소수 허용오차는 사용하지 않는다(완전 동일 요구).
- **스모크**: `python -m harness.smoke` (변경 후 import/compile 안전 보증).
- **결정성 검사**: 같은 입력으로 2회 실행 → 두 산출 JSON SHA-256 동일.
- **시간 측정**: 페어당 **N회**(기본 N=3) 실행, **min(wall_sec)** 채택.
- **백업/롤백**: 각 시도 전 `git` 스냅샷. 실패 시 `git checkout -- utils/airside_sim.py`로 즉시 복구.

> 본 계획서는 코드 변경 권한 없이 작성됨. `golden_compare.py` 추가도 사용자 승인 후 별도 단계에서 수행.

---

## 4. 사고 절차 (매 iteration 필수, Think 70%)

각 iteration 시작 전 아래를 `harness/SPEED_OPT_LOG.md`에 기록한다.

1. **현상**: 직전 베이스라인/실패의 핵심 1~2줄 + 근거 파일/줄 위치.
2. **가설**: 가능한 병목 원인 ≥ 2개. 가급적 프로파일 근거 첨부.
3. **대안 ≥ 3개**: 각 대안에 대해
   - 변경 위치/범위
   - 기대 효과 (수치 추정 포함)
   - 리스크 (계약 파손, 결과 변동, 다른 경로 영향)
   - 복잡도 (S/M/L)
   - 1개는 반드시 **"가장 적은 변경으로 가설을 검증하는 방안"**.
4. **선택 + 근거**: 왜 이 대안이 이번 루프 목표/리스크 관점에서 최선인지 1~3줄.
5. **중단 판단**: 3개 모두 확신이 없으면 코드 변경 금지 → 추가 관찰 단계로 회귀(프로파일/로그 재수집).

---

## 5. 표준 루프 (Per-iteration)

### Step 0 (1회만, 사전 셋업)
- `harness/golden_compare.py` 추가 (3절 스펙).
- `data/Result_storage/_perf_*.json`을 `.gitignore`에 추가(권장).
- `harness/SPEED_OPT_LOG.md` 생성(빈 헤더만).

### Step A — 베이스라인 측정
1. `python -m harness.smoke`
2. 페어별로 N회(=3) `harness.run --no-validate` 실행, 각 회 wall_sec 기록.
3. 페어별 **min wall_sec** 을 baseline으로 채택.
4. 페어별 결정성 점검: 산출 JSON SHA-256 두 회 일치 확인.
5. 페어별 골든 비교: `python -m harness.golden_compare <golden> <_perf_*>` → 모두 PASS여야 시작 허용.
   - 만약 현재 코드가 골든과 이미 다르면 작업 시작 전에 사용자 보고 후 중단.

### Step B — 핫스팟 탐색 (Profile-driven)
- `cProfile` + `pstats`로 페어별 상위 함수 식별 (예: `python -X dev -m cProfile -o prof.out -m harness.run ...`).
- 결과는 표 형태로 `SPEED_OPT_LOG.md`에 첨부 (top N cumulative time).
- 추측 금지. 프로파일이 가리키는 실측 핫스팟에서만 후보를 도출.

### Step C — 후보 도출 (4절 절차 적용)
- 후보 카테고리(예시일 뿐, 실제는 프로파일 결과 기반):
  - 반복 검색의 dict/set 자료구조화
  - 매 틱 재계산되는 파생값 캐싱(불변 입력에 한정)
  - 함수 내부 로컬 바인딩(이름 검색 비용 절감)
  - 동등 결과를 보장하는 분기 단순화
  - 불필요한 중간 list/tuple 할당 제거
  - **금지**: 알고리즘 변경, 부동소수 연산 순서 임의 변경, 정렬 키 임의 변경.
- 각 후보에 대해 4절 사고 단계를 통과해야 진행.

### Step D — 적용 (Act 30%, 최소 변경)
- 단일 변경 단위(가능하면 한 함수/한 블록)로 적용.
- 변경 전 `git diff`가 깔끔한 상태인지 확인 후 시도.

### Step E — 검증 (반드시 모든 페어)
- `python -m harness.smoke` PASS.
- 페어별 N회 실행, **min wall_sec** 갱신.
- 페어별 골든 비교: **모두 PASS** 필수.
- 결정성: 동일 입력 2회 실행 SHA-256 동일.

### Step F — 채택/롤백 판정
- **PASS 조건 (모두 충족)**:
  - 3개 페어 골든 모두 deep-equal.
  - 3개 페어 모두 `min wall_sec` 가 baseline 대비 **회귀 임계치 τ 이내**.
  - 결정성 SHA-256 동일.
- **위 중 하나라도 위배 시**: `git checkout -- utils/airside_sim.py`로 즉시 롤백. 변경 채택 금지.
- **기록**: 결과(어느 페어가 어떻게 변했는지, 채택/롤백)를 `SPEED_OPT_LOG.md`에 1블록 추가.

### Step G — Baseline 업데이트
- 채택된 경우에만 baseline을 새 측정치로 갱신.
- 다음 iteration으로 이동.

---

## 6. 채택/롤백 임계치 τ (사용자 확인 필요)

회귀 판정 임계치 τ를 사전에 합의한다. 측정 노이즈를 고려한 후보:
- (a) **엄격**: τ = 0% (어떠한 회귀도 거부). 노이즈로 인한 false reject 가능성 큼.
- (b) **권장**: τ = +2% (3회 min 기준 노이즈 흡수). 누적 회귀를 베이스라인 갱신으로 반영해 견제.
- (c) **느슨**: τ = +5% (적극 탐색용, 단 누적 추적 강화 필요).

**제안: (b) τ = +2%**. 사용자 확인 시 적용. (다른 값 원하시면 알려주세요.)

추가로 **누적 한도**: 채택 누적 합산이 baseline 대비 항상 개선이어야 한다. 즉, 어떤 시점이든 "프로젝트 시작 baseline 대비 모든 페어가 비퇴보".

---

## 7. 리스크 & 안전장치

- **결과 차이의 잠복**: 부동소수 연산 순서 미세 변경이 다른 분기를 유발할 수 있음 → 의심 시 곧장 롤백, 이유를 LOG에 기록.
- **결정성 깨짐**: dict 순회 순서, set 순회, 해시 시드 등에 의존하지 않도록 주의.
- **JSON I/O 외란**: `harness.run`은 dict 그대로 비교 가능하나, 파일 비교 시 직렬화 옵션 변경 금지(`indent=2`, `ensure_ascii=False`).
- **외부 의존**: 새 라이브러리 추가 금지(표준 라이브러리만). 필요 시 사용자 선승인.
- **장시간 실행**: 페어 1회 실행 시간이 매우 길 수 있음 → 측정은 우선 N=3, 필요시 N 조정 합의.

---

## 8. 보고/기록 형식

`harness/SPEED_OPT_LOG.md`에 iteration 단위로 추가:

```
### ITER YYYYMMDDTHHMMZ <짧은 태그>
- baseline (min wall_sec, N=3)
  - default_layout: <s>
  - large_flight: <s>
  - MNL_OSM: <s>
- candidates (≥3): <요약 + 선택 + 근거>
- change: <파일/함수/줄, 한 문장>
- result (min wall_sec, N=3)
  - default_layout: <s> (Δ=<%>) golden=PASS|FAIL
  - large_flight:  <s> (Δ=<%>) golden=PASS|FAIL
  - MNL_OSM:       <s> (Δ=<%>) golden=PASS|FAIL
- determinism (sha256 x2): match|mismatch
- decision: ADOPT | ROLLBACK (이유)
- next: <다음 후보 또는 종료 사유>
```

---

## 9. 종료/일시 중지 조건 (Stop)

- 같은 실패 유형 2회 연속 → 중단(원인 근거 추적 필요).
- 3 페어 모두 추가 개선 여지 없음(연속 N=3 ROLLBACK 또는 Δ < 0.5% 누적) → 종료 보고.
- 사용자가 명시적으로 중지 요청.
- 입력 데이터/계약 해석 충돌 의심 → 중단 후 보고.

---

## 10. 워크플로 대안 비교 (Think 70%, MD 작성 시점 결정)

| 대안 | 설명                                                         | 장점                              | 단점                                             |
| ---- | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------ |
| A    | **현 계획**: `harness.run` + 신규 `golden_compare.py`, min-of-N | 기존 하네스 재사용, 도구 추가 1개 | `harness.run`이 매번 JSON 파일을 씀(소량 오버헤드, 모든 측정에 동일 적용) |
| B    | `run_simulation`을 직접 import해 in-process로 N회 측정       | 파일 I/O 제거로 측정 정밀         | 새 러너 스크립트 필요, 기존 하네스와 시간 비교 기준 갈림    |
| C    | `cProfile` 단독 비교(시간 비교 생략)                         | 핫스팟만 빠르게 특정              | 사용자의 "시간 단축이 목표" 요건을 직접 검증 못함         |

**선택: A**. 이유: (1) 사용자가 요구한 검증 기준이 "터미널에서 3쌍 돌린 wall-time" 이라 기존 `harness.run` 출력이 정확히 이를 만족, (2) 새로 추가할 도구가 `golden_compare.py` 하나로 최소, (3) 파일 I/O는 모든 측정에 균일 적용되어 비교 공정성 유지. B는 추후 정밀 측정이 필요하면 보조로 도입 가능.

---

## 11. 골든 비교 방식 대안 비교

| 대안 | 설명                                          | 장점                | 단점                                  |
| ---- | --------------------------------------------- | ------------------- | ------------------------------------- |
| α    | **deep value equality** (`json.load` 후 `==`) | 직렬화 옵션 무관    | 첫 차이 위치 출력은 별도 구현 필요    |
| β    | 정규화 직렬화(sort_keys=True) 후 byte-equal   | 간결                | `harness.run`의 직렬화 옵션과 불일치 시 false fail 위험 |
| γ    | 필드별 SHA-256 매니페스트 비교                | 빠름                | 구현 복잡, 차이 디버깅 어려움         |

**선택: α**. 이유: (1) 부동소수 포함 dict의 정확한 동등 검증에 가장 견고, (2) 차이 발생 시 경로 출력으로 디버깅 용이, (3) 외부 의존 없음.

---

## 12. 사용자 승인이 필요한 항목 (실행 전 합의 사항)

다음을 확정한 뒤에 Step 0(설정) → Step A(베이스라인)로 진입한다.

1. **임계치 τ** (6절): 제안 `+2%` 채택 여부.
2. **N(반복 횟수)**: 제안 `2`.
3. **`harness/golden_compare.py` 신규 추가** 허용 여부.
4. **`data/Result_storage/_perf_*.json`** 산출물 위치 사용 허용 여부 (기존 파일 덮어쓰지 않음).
5. **로그 위치**: `harness/SPEED_OPT_LOG.md` 신규 생성 허용 여부.

승인 후에만 Step 0 실행. 승인 없이 코드/파일 수정하지 않는다.
