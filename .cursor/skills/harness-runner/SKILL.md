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

## Minimal context first

실행 전 필요한 최소 파일만 확인:

- `harness/RULES.md`
- `harness/TASK_PLAN.md`
- `harness/RUN_LOG.md`
- `harness/LESSONS_LEARNED.md`
- 필요 시 `utils/airside_sim.py`의 관련 함수만 부분 확인

불필요한 대형 파일 전체 읽기 금지.

## Standard loop

1. 목표 정의(이번 실행의 성공 기준 1~2개)
2. 최소 변경 계획(변경 파일/영향 범위 명시)
3. Builder pass (구현)
4. 실행
   - `python -m harness.smoke`
   - `python -m harness.run --input <sim_input.json> --output <sim_result.json>`
   - 필요 시 `python -m harness.validate --input <sim_input.json> --result <sim_result.json>`
5. 실패 분류
   - `smoke` / `runtime` / `validate` / `path-config` / `contract-mismatch`
6. Reviewer pass
   - 과도한 변경 여부, 하드코딩 증가, I/O 계약 파손, 디버그 코드 잔존 점검
7. 불필요한 수정 제거
8. 재실행(최대 반복 횟수 내)

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

## Logging requirements

매 루프 후 아래를 갱신:

- `harness/RUN_LOG.md`
  - 실행 명령, PASS/FAIL, 핵심 근거, 다음 최소 수정
- `harness/LESSONS_LEARNED.md`
  - 재발 방지 규칙(중복은 병합/일반화)

## Output style to user

- 실행 결과는 짧고 근거 중심으로 보고
- 실패 시 “무엇이 잘못되었는지 + 다음 수정 계획”을 반드시 포함
- 완료 판정은 Reviewer pass 이후에만 수행

