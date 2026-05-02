# Zone profile: 3 inputs × 3 repeats (mean per scenario)

Runs: `AIRSIDE_ZONE_PROFILE=1`, tags `r{1,2,3}_{default_layout|large_flight|MNL_OSM}`

Summaries parsed: **9**

## default_layout — mean wall over 3 runs (zones overlap; % is naive share of summed zone means)

| Zone | Σt mean (s) | % naive sum | Invokes mean (n) | µs / invoke |
|------|------------:|-----------:|-----------------:|-----------:|
| `setup_prep_flight_paths` | 0.1314 | 0.8% | 1 | 131,377.9 |
| `setup_adjust_eldt_spacing` | 0.0002 | 0.0% | 1 | 150.2 |
| `setup_spawn_agents` | 0.0020 | 0.0% | 1 | 2,029.5 |
| `setup_build_resource_model` | 0.0015 | 0.0% | 1 | 1,485.2 |
| `tick_touchdown_cache_pre` | 0.4792 | 2.9% | 7,776 | 61.6 |
| `tick_refresh_resource_occupancy` | 0.8822 | 5.3% | 7,776 | 113.4 |
| `tick_arr_temp_detour_flags` | 0.0375 | 0.2% | 7,776 | 4.8 |
| `tick_sort_temp_pipeline` | 0.1514 | 0.9% | 7,776 | 19.5 |
| `tick_temp_stand_ops` | 0.1313 | 0.8% | 7,776 | 16.9 |
| `tick_refresh_agent_edge_fsm` | 0.0772 | 0.5% | 7,776 | 9.9 |
| `tick_deadlock_stagnation_probe` | 0.0930 | 0.6% | 7,776 | 12.0 |
| `tick_heavy_decision_update` | 1.5153 | 9.0% | 519 | 2,919.6 |
| `tick_light_reservation_pass` | 5.4684 | 32.6% | 7,257 | 753.5 |
| `mv_halt_precheck_loop` | 0.1587 | 0.9% | 7,776 | 20.4 |
| `mv_same_direction_following_caps` | 0.0741 | 0.4% | 7,776 | 9.5 |
| `mv_touchdown_recompute_and_move_loop` | 1.6699 | 10.0% | 7,776 | 214.7 |
| `tick_touchdown_cache_post` | 0.4795 | 2.9% | 7,776 | 61.7 |
| `tick_stand_pushback_clearance_index` | 0.2613 | 1.6% | 7,776 | 33.6 |
| `tick_history_and_apron_snap_loop` | 1.1029 | 6.6% | 7,776 | 141.8 |
| `post_truncation_filter_history` | 0.0000 | 0.0% | 1 | 1.2 |
| `post_backfill_apron_offblocks` | 0.0000 | 0.0% | 1 | 7.5 |
| `post_trim_agent_micro_segments` | 0.0000 | 0.0% | 1 | 23.0 |
| `post_compress_dwell_histories_block` | 0.0672 | 0.4% | 1 | 67,196.6 |
| `post_build_positions` | 0.1096 | 0.7% | 1 | 109,595.5 |
| `post_build_schedule_detail` | 0.0093 | 0.1% | 1 | 9,252.1 |
| `post_overlay_schedule_timing` | 0.0356 | 0.2% | 1 | 35,575.0 |
| `post_compact_positions_v2` | 0.0615 | 0.4% | 1 | 61,469.9 |
| `fn_can_reserve_path` | 1.9670 | 11.7% | 54,125 | 36.3 |
| `fn_reserve_path` | 0.9654 | 5.8% | 52,749 | 18.3 |
| `fn_move_agent` | 0.8217 | 4.9% | 82,393 | 10.0 |

## large_flight — mean wall over 3 runs (zones overlap; % is naive share of summed zone means)

| Zone | Σt mean (s) | % naive sum | Invokes mean (n) | µs / invoke |
|------|------------:|-----------:|-----------------:|-----------:|
| `setup_prep_flight_paths` | 0.6413 | 0.5% | 1 | 641,250.2 |
| `setup_adjust_eldt_spacing` | 0.0004 | 0.0% | 1 | 429.9 |
| `setup_spawn_agents` | 0.0098 | 0.0% | 1 | 9,799.7 |
| `setup_build_resource_model` | 0.0025 | 0.0% | 1 | 2,487.2 |
| `tick_touchdown_cache_pre` | 11.5863 | 9.2% | 41,700 | 277.8 |
| `tick_refresh_resource_occupancy` | 3.7945 | 3.0% | 41,700 | 91.0 |
| `tick_arr_temp_detour_flags` | 0.4449 | 0.4% | 41,700 | 10.7 |
| `tick_sort_temp_pipeline` | 2.0012 | 1.6% | 41,700 | 48.0 |
| `tick_temp_stand_ops` | 2.1882 | 1.7% | 41,700 | 52.5 |
| `tick_refresh_agent_edge_fsm` | 1.0788 | 0.9% | 41,700 | 25.9 |
| `tick_deadlock_stagnation_probe` | 0.9819 | 0.8% | 41,700 | 23.5 |
| `tick_heavy_decision_update` | 2.3620 | 1.9% | 2,780 | 849.7 |
| `tick_light_reservation_pass` | 30.0089 | 23.8% | 38,920 | 771.0 |
| `mv_halt_precheck_loop` | 1.9534 | 1.6% | 41,700 | 46.8 |
| `mv_same_direction_following_caps` | 0.3830 | 0.3% | 41,700 | 9.2 |
| `mv_touchdown_recompute_and_move_loop` | 17.1409 | 13.6% | 41,700 | 411.1 |
| `tick_touchdown_cache_post` | 11.3844 | 9.0% | 41,700 | 273.0 |
| `tick_stand_pushback_clearance_index` | 3.4405 | 2.7% | 41,700 | 82.5 |
| `tick_history_and_apron_snap_loop` | 18.4049 | 14.6% | 41,700 | 441.4 |
| `post_truncation_filter_history` | 0.0000 | 0.0% | 1 | 0.5 |
| `post_backfill_apron_offblocks` | 0.0000 | 0.0% | 1 | 10.0 |
| `post_trim_agent_micro_segments` | 0.0000 | 0.0% | 1 | 36.1 |
| `post_compress_dwell_histories_block` | 0.7593 | 0.6% | 1 | 759,346.1 |
| `post_build_positions` | 3.0355 | 2.4% | 1 | 3,035,498.7 |
| `post_build_schedule_detail` | 0.0184 | 0.0% | 1 | 18,365.5 |
| `post_overlay_schedule_timing` | 0.0599 | 0.0% | 1 | 59,906.6 |
| `post_compact_positions_v2` | 0.1246 | 0.1% | 1 | 124,645.3 |
| `fn_can_reserve_path` | 7.5149 | 6.0% | 220,683 | 34.1 |
| `fn_reserve_path` | 3.3482 | 2.7% | 220,417 | 15.2 |
| `fn_move_agent` | 3.2790 | 2.6% | 967,145 | 3.4 |

## MNL_OSM — mean wall over 3 runs (zones overlap; % is naive share of summed zone means)

| Zone | Σt mean (s) | % naive sum | Invokes mean (n) | µs / invoke |
|------|------------:|-----------:|-----------------:|-----------:|
| `setup_prep_flight_paths` | 1.2389 | 8.9% | 1 | 1,238,930.7 |
| `setup_adjust_eldt_spacing` | 0.0002 | 0.0% | 1 | 160.7 |
| `setup_spawn_agents` | 0.0021 | 0.0% | 1 | 2,129.8 |
| `setup_build_resource_model` | 0.0051 | 0.0% | 1 | 5,131.4 |
| `tick_touchdown_cache_pre` | 0.1458 | 1.0% | 5,239 | 27.8 |
| `tick_refresh_resource_occupancy` | 0.5707 | 4.1% | 5,239 | 108.9 |
| `tick_arr_temp_detour_flags` | 0.0140 | 0.1% | 5,239 | 2.7 |
| `tick_sort_temp_pipeline` | 0.0522 | 0.4% | 5,239 | 10.0 |
| `tick_temp_stand_ops` | 0.0637 | 0.5% | 5,239 | 12.2 |
| `tick_refresh_agent_edge_fsm` | 0.0318 | 0.2% | 5,239 | 6.1 |
| `tick_deadlock_stagnation_probe` | 0.0345 | 0.2% | 5,239 | 6.6 |
| `tick_heavy_decision_update` | 4.4758 | 32.0% | 350 | 12,788.0 |
| `tick_light_reservation_pass` | 3.6161 | 25.9% | 4,889 | 739.7 |
| `mv_halt_precheck_loop` | 0.0684 | 0.5% | 5,239 | 13.1 |
| `mv_same_direction_following_caps` | 0.0337 | 0.2% | 5,239 | 6.4 |
| `mv_touchdown_recompute_and_move_loop` | 0.6762 | 4.8% | 5,239 | 129.1 |
| `tick_touchdown_cache_post` | 0.1439 | 1.0% | 5,239 | 27.5 |
| `tick_stand_pushback_clearance_index` | 0.0861 | 0.6% | 5,239 | 16.4 |
| `tick_history_and_apron_snap_loop` | 0.4194 | 3.0% | 5,239 | 80.1 |
| `post_truncation_filter_history` | 0.0000 | 0.0% | 1 | 0.6 |
| `post_backfill_apron_offblocks` | 0.0000 | 0.0% | 1 | 4.5 |
| `post_trim_agent_micro_segments` | 0.0000 | 0.0% | 1 | 14.2 |
| `post_compress_dwell_histories_block` | 0.0253 | 0.2% | 1 | 25,265.8 |
| `post_build_positions` | 0.0327 | 0.2% | 1 | 32,712.6 |
| `post_build_schedule_detail` | 0.0045 | 0.0% | 1 | 4,478.0 |
| `post_overlay_schedule_timing` | 0.0175 | 0.1% | 1 | 17,511.0 |
| `post_compact_positions_v2` | 0.0325 | 0.2% | 1 | 32,483.2 |
| `fn_can_reserve_path` | 1.1490 | 8.2% | 53,120 | 21.6 |
| `fn_reserve_path` | 0.5989 | 4.3% | 51,819 | 11.6 |
| `fn_move_agent` | 0.4264 | 3.1% | 57,432 | 7.4 |

---

## 알기 쉬운 요약 (한국어)

### 이 파일이 말하는 것

- 같은 입력 JSON으로 시뮬을 **각 시나리오당 3번** 돌린 뒤, **코드 블록(Zone)** 마다 걸린 **누적 시간(초)의 평균**을 모은 표입니다.
- **`% naive sum`** 은 「모든 존 시간 합」을 100으로 둘 때, 그 존만 **몇 점유**인지에 가까운 눈대중 비율입니다. **중첩 존**(한 구간 안에 또 존을 켠 경우 등) 때문에 **합쳐서 100%가 진짜 CPU 비중이 되지는 않습니다.** “어느 라벨이 숫자로 커 보이나” 보는 용도로 쓰면 됩니다.
- **`Invokes mean`** = 그 존이 **들어간 횟수** 평균, **`µs / invoke`** = 한 번 들어올 때 **평균 몇 마이크로초**.

### 존 이름 앞머리(구역)

| 접두사 | 의미 |
|--------|------|
| `setup_*` | 시뮬 시작 전 준비(레이아웃/에이전트 등) |
| `tick_*` | 매 시간 스텝마다 반복되는 틱 루프 안 |
| `mv_*` | 이동 제어(`apply_movement_controls` 근처) |
| `post_*` | 시뮬 종료 후 후처리 |
| `fn_*` | 특정 함수 전체 시간(예: 예약 검사·이동 호출 빈번) |

### 시나리오별 — 누적 시간 Σt 평균 상위 5존

| 순위 | default_layout (초, %) | large_flight (초, %) | MNL_OSM (초, %) |
|-----:|-------------------------|-----------------------|------------------|
| 1 | light_reservation 5.47 (32.6%) | light_reservation 30.01 (23.8%) | heavy_decision 4.48 (32.0%) |
| 2 | can_reserve_path 1.97 (11.7%) | history_apron_snap 18.40 (14.6%) | light_reservation 3.62 (25.9%) |
| 3 | mv_touchdown_loop 1.67 (10.0%) | mv_touchdown_loop 17.14 (13.6%) | setup_prep_paths 1.24 (8.9%) |
| 4 | heavy_decision 1.52 (9.0%) | touchdown_cache_pre 11.59 (9.2%) | can_reserve_path 1.15 (8.2%) |
| 5 | reserve_path 0.97 (5.8%) | touchdown_cache_post 11.38 (9.0%) | mv_touchdown_loop 0.68 (4.8%) |

(표에서 Zone 이름은 길이를 줄여 썼고, 원문은 위 본문 표와 동일합니다.)

### 한 줄 해석

- **세 시나리오 공통**: `tick_light_reservation_pass` 가 항상 크다 → **라이트 예약 패스**가 병목 후보 1순위.
- **default_layout**: `fn_can_reserve_path` 비중이 커서 **경로 예약 가능 여부 판단**도 같이 보면 좋음.
- **large_flight**: 틱 수·데이터가 커서 **터치다운 캐시(pre/post)**, **히스토리/에이프런 스냅**, **mv_touchdown** 이 함께 부풀음.
- **MNL_OSM**: `tick_heavy_decision_update` 가 1순위(32%) → 그 입력에선 **헤비 의사결정 갱신**이 돋보임. `setup_prep_flight_paths` 도 상대적으로 큼(OSM 레이아웃 전처리 부담 가능).
