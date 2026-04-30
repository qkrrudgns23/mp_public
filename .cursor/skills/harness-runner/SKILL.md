---
name: harness-runner
description: Execute and iterate the airside simulation harness loop (smoke -> run -> validate -> classify -> log) with strict evidence-based retries. Use when the user asks to run, verify, debug, or repeatedly improve airside_sim.py through terminal execution.
---

# Harness Runner

`airside_sim.py` 개발/검증을 위한 실행 하네스를 표준 루프로 수행한다.

## When to activate

- 사용자가 실행/검증/재실행을 요청할 때
- 사용자가 실패 원인 분석, 반복 루프, 결과 검토를 요청할 때
- `airside_sim.py` 변경 후 안전 확인이 필요할 때

## Non-negotiable rules

- `data/Info_storage/Information.json`은 **read-only** 로 취급한다.
- 추측하지 말고 코드/로그/입출력 파일을 근거로 판단한다.
- 원인 미확인 상태에서 같은 실행을 반복하지 않는다.
- 실행 전 `python -m harness.smoke`를 반드시 수행한다.
- 큰 변경 전에는 작은 검증 루프를 먼저 통과시킨다.
- **Think 70% / Act 30%**: 매 루프마다 "생각 70%, 실행 30%"의 비중을 유지한다. 실행 전 사고 단계에서 반드시 **최소 3개 이상의 대안**을 만들고, 각 대안의 장단점·리스크를 비교한 뒤 하나를 근거와 함께 선택한다. 충분한 근거가 쌓이기 전에는 코드를 건드리지 않는다.

## Think 70% / Act 30% (실행 전 필수 사고 절차)

하네스 루프의 각 반복(iteration)에서 "Builder pass" 또는 "재실행" 에 들어가기 **전에** 아래를 먼저 기록한다.

1. **현상 정의**: 직전 실패/관찰의 핵심을 1~2줄로 요약. 근거 파일(로그/코드 위치)을 명시.
2. **가설 / 원인 후보**: 최소 2개의 가능한 원인 가설을 쓴다.
3. **대안(≥ 3개)**: 이번에 시도할 수정안을 **최소 3가지** 제시하고 각각에 대해:
   - 변경 위치와 범위
   - 기대 효과
   - 리스크/부작용 (계약 파손, 성능, 다른 경로 영향)
   - 복잡도(최소/중/큰)
   - 한 대안은 반드시 **"가장 적은 변경으로 가설을 검증하는 방안"** 을 포함한다.
4. **선택 + 근거**: 위 중 하나를 고르고, **왜 그 대안이 이번 루프의 목표와 리스크 관점에서 최선인지** 1~3줄로 적는다.
5. **중단 판단**: 3개 대안 모두 확신이 없다면 실행하지 말고, 근거 수집 단계로 되돌아간다(코드/로그/입출력 추가 관찰).

이 절차를 건너뛴 실행은 무효로 간주하고, 그 실행 결과는 증거로 사용하지 않는다.

## Minimal context first

실행 전 필요한 최소 파일만 확인:

- `harness/RULES.md`
- `harness/TASK_PLAN.md`
- `harness/RUN_LOG.md`
- `harness/LESSONS_LEARNED.md`
- 필요 시 `utils/airside_sim.py`의 관련 함수만 부분 확인

불필요한 대형 파일 전체 읽기 금지.

## Golden-locked performance loop (``airside_sim.py``)

목적: **동작/산출 골든 JSON**을 깨지 않으면서 wall time·지역 연산만 줄이는 패치를 **한 건씩** 쌓는다.

**한 사이클 (코드 수정 직후 1회):**

1. 최소 패치 적용 (**Think 70%** 절차는 변경 없음 — 대안 비교 후 한 가지만).
2. ``python -m harness.golden_opt_cycle [--tag LABEL]``
   - 내부: ``harness.smoke`` → (default_layout / large_flight / MNL_OSM 각각) ``harness.run`` → 해당 골든과 ``golden_compare`` (deep ``==``, 부동 허용오차 없음).
3. **exit 1** 또는 골든 FAIL → 해당 패치 **즉시 원복** 후 원인 근거를 잡고 다시 진행한다.
4. **exit 0** → 패치 채택. 필요 시 같은 출력 로그의 wall 시간으로 전후 비교만 기록.

**연속 최적화(마라톤, 최대 ``N``회 시도)** — 레지스트리에 등록된 패치를 순서대로 적용하고 **매 패치 후** 위와 동일한 골든 사이클을 돌린다:

- ``python -m harness.golden_opt_marathon --iterations 50``
  - 패치 정의 목록은 ``harness/golden_opt_marathon_steps.py`` 의 ``_PATCHES`` 순서이다.
  - 인덱스가 레지스트리를 넘기면 해당 iteration은 SKIP(확장할 때까지)한다. 더 이상 스킵 없이 멈추려면 ``--abort-on-registry-exhaust`` 참고.
  - 한 iteration에서 골든이 깨지면 **그 패치만** 파일에서 되돌리고 프로그램이 exit 1 한다.

동일 코드로 wall time 만 반복 측정하는 것 (**``bench_triple_loop``** 등, 코드 변경 없음)은 OS 지터 분석에는 쓰이지만 **계약 검증으로는 패치 채택 근거가 되지 않는다.** 성능 튜닝의 PASS/FAIL은 항상 **골든 3페어**와 ``golden_opt_cycle`` exit 코드로만 판별한다.

## Standard loop


1. 목표 정의(이번 실행의 성공 기준 1~2개)
2. **사고 단계 (Think 70%)**
   - 현상 정의 + 원인 가설 2개 이상
   - **대안 ≥ 3개 제시 + 장단점/리스크 비교 + 하나 선택 + 근거 명시**
   - 대안이 부실하거나 근거가 약하면 여기서 멈추고 코드/로그를 더 읽는다.
3. 최소 변경 계획(변경 파일/영향 범위 명시)
4. Builder pass (구현, Act 30%)
5. 실행
   - `python -m harness.smoke`
   - `python -m harness.run --input <sim_input.json> --output <sim_result.json>`
   - 필요 시 `python -m harness.validate --input <sim_input.json> --result <sim_result.json>`
6. 실패 분류
   - `smoke` / `runtime` / `validate` / `path-config` / `contract-mismatch`
7. Reviewer pass
   - 과도한 변경 여부, 하드코딩 증가, I/O 계약 파손, 디버그 코드 잔존 점검
8. 불필요한 수정 제거
9. 재실행(최대 반복 횟수 내) — 재실행 전에도 2번의 사고 단계를 다시 수행한다.

## Stop conditions

- 같은 실패 유형 2회 연속 발생 시 중단
- 실패 원인 근거를 제시하지 못하면 중단
- 수동 검토 필요 조건:
  - 입력 데이터 결함 의심
  - 요구사항/계약 해석 충돌
  - 대규모 리팩터링이 필요한 경우

## Commands

- Smoke only:
  - `python -m harness.smoke`
- One-shot run + validate:
  - `python -m harness.run --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json`
- Managed loop:
  - `python -m harness.loop --input data/Result_storage/default_layout_sim_input.json --output data/Result_storage/_validation_sim_result.json --max-runs 2`
- Golden triple gate (변경 후 1회, smoke + run×3 + compare×3):
  - ``python -m harness.golden_opt_cycle [--tag LABEL]``
- Golden marathon (레지스트리 패치 순서대로 각각 동일 게이트, 기본 최대 반복 예: 50회):
  - ``python -m harness.golden_opt_marathon --iterations 50``

## Logging requirements

매 루프 후 아래를 갱신:

- `harness/RUN_LOG.md`
  - 실행 명령, PASS/FAIL, 핵심 근거, 다음 최소 수정
  - **이번 루프에서 비교한 대안 목록(≥ 3개)과 선택 근거**를 함께 기록 (한 줄씩 간단히)
- `harness/LESSONS_LEARNED.md`
  - 재발 방지 규칙(중복은 병합/일반화)

## Output style to user

- 실행 결과는 짧고 근거 중심으로 보고
- 실패 시 “무엇이 잘못되었는지 + 다음 수정 계획”을 반드시 포함
- 완료 판정은 Reviewer pass 이후에만 수행

