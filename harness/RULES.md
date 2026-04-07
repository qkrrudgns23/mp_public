## Stable Rules (변경 적음)

- **Source of truth**: 판단 근거는 코드/입출력/로그/결과 파일이다. 추측 금지.
- **Information.json**: `data/Info_storage/Information.json`은 **read-only** 로 취급한다. 의미/스키마를 임의 변경하지 않는다.
- **I/O contract**:
  - **sim_input**: 레이아웃+항공편 입력 JSON. 최소 요구 키: `grid`, `flights`.
  - **sim_result**: 시뮬레이션 결과 JSON. 최소 요구 키: `baseDate`, `positions`, `schedule`, `flights_detail`, `deadlock_resolve_event_count`.
- **Pre-run checks**:
  - **syntax/smoke check 필수**: 실행 전 `py_compile`로 최소 구문 체크를 통과해야 한다.
  - **작은 검증 루프 우선**: 큰 변경 전에는 최소 입력으로 1회 실행→검증→기록 루프를 먼저 돌린다.
- **Retry discipline**:
  - **원인 없는 재시도 금지**: 실패 시 원인 분류/증거(로그/스택트레이스/검증 실패 항목) 없이 같은 명령 반복 금지.
  - **최대 반복/중단 조건**: 자동 반복은 `max_runs`를 둔다. 같은 실패 유형이 2회 반복되면 전략을 바꾼다(validator 강화/더 작은 변경/입출력 스키마 확인 우선).
- **Verification gate**:
  - 수정 후에는 반드시 동일 시나리오 재실행 + 핵심 지표 재측정(예: 특정 시각 근접도/경로 회귀) 완료 전에는 완료 처리 금지.
  - 모든 코드 수정 후에는 예외 없이 **내가 터미널에서** `python -m harness.smoke`와 `python -m harness.run`(필요시 `validate`)를 직접 실행해 결과를 확인한다.
- **Cleanup**:
  - **임시 디버그 코드**는 최종 정리 대상이다(하네스 로그/validator로 대체).

## Worker Isolation (Builder/Reviewer 강제 분리)

- **Worker A = Builder**: 최소 변경 구현 + 실행 + 결과 생성까지 담당
- **Worker B = Reviewer**: 코드/실행 결과/산출물 기준으로 검토 후 승인

### Reviewer checklist
- 요구사항을 벗어난 과도한 변경/리팩터링이 있는가?
- 하드코딩이 늘어났는가?
- 입출력 계약(`sim_input`/`sim_result`)이 깨졌는가?
- `Information.json` read-only 원칙을 침범했는가?
- 예외 처리/로그/검증이 부족한가?
- 디버그 코드가 남았는가?
- 동일 실패를 반복 유발할 구조가 남아 있는가?
- 함수 책임이 과도하게 커졌는가?

## Standard Commands (표준 실행)

- **Smoke check**: `python -m harness.smoke`
- **Run once**: `python -m harness.run --input <sim_input.json> --output <sim_result.json>`
- **Validate only**: `python -m harness.validate --result <sim_result.json> --input <sim_input.json>`

