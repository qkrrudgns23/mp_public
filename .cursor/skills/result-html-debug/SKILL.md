---
name: result-html-debug
description: >-
  Regenerates self-contained HTML debug reports from simulation JSON (e.g. pose
  vs track tables) and surfaces a local file path or opens the browser, because
  the chat cannot render HTML. Use when the user wants to re-run the same
  offline HTML debugging flow, see pose/track/R3/Layout alignment again, says
  "다시 띄워" or "HTML로 디버그", or references harness/pose_debug_from_result
  or Test_sim_result window inspection.
---

# Result HTML debug (오프라인 뷰)

채팅은 **HTML을 화면에 띄울 수 없다.** 대신 에이전트가 **스크립트를 실행 → 리포가 파일로 쓰여짐 → 사용자가 브라우저로 연다** 흐름이 “에이전트가 대신 띄워주는” 동등물이다. 사람이 터미널·경로·인자를 직접 맞출 필요를 줄이는 것이 이 스킬의 목적이다.

## When to activate (발동 조건)

- 사용자가 **똑같은 식의 디버그**를 **다시** 하고 싶다고 할 때 (예: "다시 띄워", "그 HTML 다시", "R3 pose 디버그 다시")
- **시뮬 결과 JSON**으로 **표/로그 HTML**을 보고 싶을 때 (`Test_sim_result`, pose vs track, `retro_nose_vs_track` 등)
- `harness/pose_debug_from_result.py` 또는 **자기 포함 HTML** 경로를 언급할 때
- "채팅에 띄워줘"가 아니라 **로컬 파일로 보면 됨**이 전제인 디버그일 때

## Canonical command (repo root)

Layout pose vs ground track (designer `getFlightPoseAtTime` 포트) 테이블:

```bash
python harness/pose_debug_from_result.py \
  --result data/Result_storage/Test_sim_result.json \
  --input data/Result_storage/Test_sim_input.json \
  --flight-id <flight_id> \
  --t0 <sec_from_midnight> --t1 <sec_from_midnight> \
  --out harness/r3_pose_debug_<label>.html
```

- `--t0` / `--t1`: `positions[].t`와 동일한 **자정 기준 초** (표에는 `HH:MM:SS`로도 보임).
- `--out`: 덮어쓰기 가능한 출력 경로; **레이블**은 윈도우를 알아볼 수 있게 (`01_14_30` 등).

필요 인자를 사용자에게 묻지 말고, **result/input JSON**과 **기준 구간**은 가능하면 기존 대화·파일·최근 커맨드에서 복원한다. 없을 때만 질문한다.

## Agent workflow

1. **실행**: 위 형태로 `python`을 **직접** 돌려서 HTML을 쓴다 (사용자에게 "직접 실행하세요"로 넘기지 않는다).
2. **성공 확인**: 스크립트의 `Wrote ...` 출력 또는 파일 존재로 확인.
3. **열람 안내**
   - **절대 경로** `file:///.../harness/....html` 을 응답에 넣는다 (한글 경로는 인코딩이 깨질 수 있으니 **탐색기에서 `harness` 폴더 열고 파일 더블클릭**을 함께 안내).
   - 선택: Windows에서 브라우저로 열기가 필요하면 `explorer path\to\file.html` 또는 사용자 환경에 맞는 **한 줄** 열기 명령을 쓴다(실패해도 파일 경로 안내는 유지).
4. **스크립트/템플릿 수정**이 요청되면, 변경 후 **같은 커맨드로 재생성**해 증거를 맞춘다.

## Related

- 시뮬 **실행·검증 루프** 자체: `.cursor/skills/harness-runner/SKILL.md`
- HTML **UI 폴리시** (다른 화면): `.cursor/skills/ui-design/SKILL.md` — 이 스킬은 **리포 HTML**에 필수는 아님

## 확장 (같은 패턴)

나중에 다른 `harness/*_debug*.py`가 생기면, **"JSON → 단일 HTML + 임베드 데이터"** 패턴은 이 스킬과 동일하게 취급: 에이전트는 생성 명령을 실행하고 로컬 파일로 열람을 연결한다.
