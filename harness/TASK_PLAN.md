## Task Plan (현재 목표/체크리스트)

### Goal
`utils/airside_sim.py` 개발을 안전하게 반복할 수 있도록, **실행→검증→기록**이 표준화된 하네스를 구축한다.

### Scope / 영향 범위
- **추가/변경 파일**: `harness/` 폴더(문서 + 실행/검증 스크립트)
- **기존 코드 변경**: 원칙적으로 없음(필요 시 최소 수정만)
- **금지**: `data/Info_storage/Information.json` 변경 금지

### Loop Checklist (각 루프 공통)
- [ ] 이번 변경에 필요한 최소 파일 목록 선언
- [ ] 변경 영향 범위(어떤 함수/어떤 출력이 바뀌는지) 기록
- [ ] smoke check 통과
- [ ] run once 실행
- [ ] 결과 파일 생성/갱신 확인
- [ ] validator 통과 확인(또는 실패 항목/근거 기록)
- [ ] 실패 유형 분류(입력/경로/런타임/결과스키마/논리오류/성능 등)
- [ ] Lessons Learned 갱신(중복 병합)

### Current Defaults (초기 루프)
- **default input**: `data/Result_storage/default_layout_sim_input.json`
- **default output**: `data/Result_storage/_validation_sim_result.json` (하네스 전용)

