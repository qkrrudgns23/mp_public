import json
import os
import random
import shutil
import string
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from utils.airside_sim import run_simulation
from utils.layout_receiver import (
    DEFAULT_LAYOUT_PATH,
    LAYOUT_FILE,
    LAYOUT_STORAGE_DIR,
    list_layout_names,
    _safe_layout_path,
    start_layout_receiver,
)

# run_app.py 사용 시(LAYOUT_SAME_PORT=1): 8501 프록시에서 API 처리. 그 외: layout_receiver(8765) 기동.
if os.environ.get("LAYOUT_SAME_PORT") == "1":
    LAYOUT_API_URL = os.environ.get("LAYOUT_API_BASE_URL", "http://127.0.0.1:8501")
else:
    LAYOUT_API_URL = start_layout_receiver()

st.set_page_config(
    page_title="Terminal & Airside Designer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Default grid; all settings (Grid / Terminal / Contact Stand / Remote) are in the right panel inside the iframe
GRID_COLS = 200
GRID_ROWS = 200
CELL_SIZE = 20.0

# data/Layout_storage 만 사용. data/ 직하위에는 current_layout/default_layout 생성·사용 안 함
_data_dir = Path(__file__).resolve().parents[1] / "data"
_fallback_default = _data_dir / "default_layout.json"
LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Info_storage: Information.json (runway separation, aircraft types, etc.)
INFO_STORAGE_DIR = _data_dir / "Info_storage"
INFO_FILE = INFO_STORAGE_DIR / "Information.json"
INFORMATION: dict = {}
try:
    INFO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if INFO_FILE.is_file():
        INFORMATION = json.loads(INFO_FILE.read_text(encoding="utf-8"))
except Exception:
    pass
if not DEFAULT_LAYOUT_PATH.is_file() and _fallback_default.is_file():
    shutil.copy2(_fallback_default, DEFAULT_LAYOUT_PATH)

# Load initial layout: Layout_storage/default_layout.json 우선, 없으면 data/default_layout.json
DEFAULT_LAYOUT: dict = {}
try:
    for _layout_path in (DEFAULT_LAYOUT_PATH, _fallback_default):
        if _layout_path.is_file():
            DEFAULT_LAYOUT = json.loads(_layout_path.read_text(encoding="utf-8"))
            break
except Exception:
    pass


def _random_reg_number() -> str:
    """Generate a pseudo aircraft registration like HL-AB123."""
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = f"{random.randint(0, 9999):04d}"
    return f"HL-{letters}{digits}"


def _ensure_random_regs(layout: dict) -> None:
    flights = layout.get("flights") or []
    for f in flights:
        reg = (f.get("reg") or "").strip()
        if not reg:
            f["reg"] = _random_reg_number()


_ensure_random_regs(DEFAULT_LAYOUT)


# --- Python 디스크리트 시뮬레이션 엔진 (5초 단위) ---
SIM_TIME_STEP_SEC = 5

# 쿼리 파라미터: load_layout=이름 → Layout_storage에서 해당 JSON 로드하여 표시 (API/포트 없이)
# 기본은 DEFAULT_LAYOUT; load_layout 이 있으면 해당 파일로 덮어씀
layout_for_html = DEFAULT_LAYOUT


def _get_query_one(key: str):
    """Streamlit 쿼리 파라미터에서 단일 값 추출 (query_params / experimental_get_query_params 모두 처리)."""
    try:
        _qp = getattr(st, "query_params", None)
        if _qp is not None:
            val = _qp.get(key) if hasattr(_qp, "get") else getattr(_qp, key, None)
        else:
            _qp = st.experimental_get_query_params()
            val = _qp.get(key) if isinstance(_qp, dict) else None
        if val is None:
            return None
        if isinstance(val, str):
            return val.strip() or None
        if isinstance(val, (list, tuple)) and len(val):
            return (val[0] or "").strip() or None
        return None
    except Exception:
        return None


try:
    load_name = _get_query_one("load_layout")
    if load_name:
        _load_path = _safe_layout_path(load_name)
        if _load_path and _load_path.is_file():
            layout_for_html = json.loads(_load_path.read_text(encoding="utf-8"))
            _ensure_random_regs(layout_for_html)
            layout_display_name = load_name
    # Run Simulation: ?run_simulation=1 이면 current_layout 읽어서 시뮬레이션 후 표시
    q = _get_query_one("run_simulation")
    if q and LAYOUT_FILE.is_file():
        layout_from_designer = json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
        layout_for_html = run_simulation(
            layout_from_designer,
            time_step_sec=SIM_TIME_STEP_SEC,
            use_discrete_engine=True,
        )
        layout_display_name = "Simulation"
except Exception:
    pass

# Layout_storage 파일 목록은 페이지 렌더 시 주입 (API 호출 없음)
layout_names = list_layout_names()

# 그리드 상단에 표시할 레이아웃 이름
layout_display_name = "default_layout"

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; min-height: 100%; height: 100%; background: #303030; color: #e5e7eb; font-family: system-ui, sans-serif; overflow: hidden; }}
    #app {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    #canvas-container {{ position: absolute; inset: 0; cursor: crosshair; }}
    #grid-canvas {{ width: 100%; height: 100%; display: block; }}
    #toolbar {{ position: absolute; bottom: 56px; right: 50vw; left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 8px; z-index: 30; pointer-events: auto; }}
    #sim-controls-container {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.5); background: rgba(55,65,81,0.95); box-shadow: 0 1px 3px rgba(0,0,0,0.3); }}
    #sim-controls-container .tool-btn {{ margin: 0; border: none; border-radius: 4px; font-size: 11px; padding: 6px 12px; background: rgba(75,85,99,0.9); color: #e5e7eb; cursor: pointer; }}
    #sim-controls-container .tool-btn:hover {{ background: rgba(100,116,139,0.9); }}
    #sim-controls-container select {{ width: auto; min-width: 70px; margin: 0; padding: 4px 8px; font-size: 11px; background: rgba(75,85,99,0.9); color: #e5e7eb; border: 1px solid rgba(100,100,100,0.5); border-radius: 4px; }}
    #sim-controls-container label {{ margin: 0 4px 0 0; font-size: 11px; color: #9ca3af; }}
    #sim-controls-container #flightSimSlider {{ width: 120px; margin: 0; vertical-align: middle; accent-color: #3b82f6; }}
    #sim-controls-container #flightSimTimeLabel {{ font-size: 11px; color: #e5e7eb; min-width: 72px; display: inline-block; }}
    #view-toggle {{ display: inline-flex; border-radius: 8px; overflow: hidden; border: 1px solid rgba(148,163,184,0.5); background: rgba(40,40,40,0.95); box-shadow: 0 1px 3px rgba(0,0,0,0.3); }}
    #view-toggle .tool-btn {{ margin: 0; border: none; border-radius: 0; border-right: 1px solid rgba(148,163,184,0.3); font-size: 11px; padding: 6px 12px; }}
    #view-toggle .tool-btn:last-child {{ border-right: none; }}
    #view-toggle .tool-btn.active {{ background: #1e3a5f; color: #93c5fd; }}
    #view-toggle .tool-btn:not(.active) {{ background: transparent; color: #9ca3af; }}
    #view-toggle .tool-btn:hover {{ background: rgba(60,60,60,0.9); }}
    #info-bar {{ display: none; position: absolute; bottom: 8px; left: 10px; right: 10px; padding: 8px 12px; min-height: 2.2em; border-radius: 8px; background: rgba(0,0,0,0.45); color: #9ca3af; font-size: 12px; align-items: center; pointer-events: none; z-index: 5; }}
    #right-panel {{ position: absolute; top: 0; right: 0; bottom: 0; width: 50vw; background: rgba(30,30,30,0.95); border-left: 1px solid rgba(100,100,100,0.5); padding: 12px; font-size: 12px; overflow-y: auto; z-index: 20; transition: width 0.2s, min-width 0.2s; }}
    #right-panel.collapsed {{ width: 44px; min-width: 44px; padding: 8px; overflow: hidden; }}
    #right-panel.collapsed .panel-content {{ display: none; }}
    #panel-toggle {{
      position: absolute;
      left: -26px;
      top: 50%;
      transform: translateY(-50%);
      width: 32px;
      height: 84px;
      border-radius: 9999px 0 0 9999px;
      border: 1px solid rgba(148,163,184,0.8);
      border-right: none;
      background: linear-gradient(180deg, #1f2937, #020617);
      color: #f9fafb;
      cursor: pointer;
      font-size: 16px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 12px rgba(0,0,0,0.65);
      backdrop-filter: blur(4px);
      z-index: 30;
    }}
    #panel-toggle:hover {{
      background: linear-gradient(180deg, #111827, #000000);
      color: #f97316;
      border-color: rgba(249,250,251,0.95);
    }}
    .section-title {{ font-size: 10px; font-weight: 600; text-transform: uppercase; color: #9ca3af; margin: 10px 0 4px 0; }}
    .section-title:first-child {{ margin-top: 0; }}
    label {{ font-size: 11px; color: #9ca3af; display: block; margin-top: 6px; }}
    input, select {{ width: 100%; background: #1a1a1a; color: #e5e7eb; border: 1px solid #444; border-radius: 4px; padding: 6px 8px; font-size: 11px; margin-top: 2px; }}
    button.small {{ padding: 4px 8px; font-size: 11px; margin-top: 4px; cursor: pointer; border-radius: 4px; border: 1px solid #555; background: #2a2a2a; color: #e5e7eb; }}
    button.small:hover {{ background: #333; }}
    .obj-list {{ max-height: 160px; overflow-y: auto; margin-top: 6px; }}
    /* Layout 탭의 object 리스트는 패널 높이까지 확장하고,
       전체 패널 높이를 초과하는 경우에만 우측 패널 스크롤을 사용 */
    #object-list.obj-list {{ max-height: none; }}
    /* Flight schedule는 기본 12편 정도 보이도록 높이 확장 (10% 증가) */
    #flightList.obj-list {{ max-height: 418px; }}
    /* Flight Configuration은 세로 스크롤 없이 20개 이상 표시, 가로는 스크롤 가능 */
    #flightConfigList.obj-list {{ max-height: none; overflow-y: visible; overflow-x: auto; }}
    .obj-item {{ padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; background: rgba(50,50,50,0.9); border: 1px solid #444; font-size: 11px; cursor: pointer; }}
    .obj-item:hover {{ background: #3a3a3a; }}
    .obj-item.selected {{ border-color: #3b82f6; background: rgba(59,130,246,0.15); }}
    .obj-item-header {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; }}
    .obj-item-title {{ font-weight: 500; flex: 1; min-width: 0; }}
    .obj-item-tag {{ font-size: 10px; color: #9ca3af; flex-shrink: 0; }}
    .obj-item-time {{ color: #94a3b8; font-weight: normal; margin-left: 4px; }}
    .obj-item-details {{ display: none; margin-top: 4px; color: #d1d5db; font-size: 10px; line-height: 1.4; }}
    .obj-item.expanded .obj-item-details {{ display: block; }}
    .obj-item-delete {{ flex-shrink: 0; padding: 2px 6px; font-size: 10px; border-radius: 3px; border: 1px solid #555; background: #2a2a2a; color: #f87171; cursor: pointer; }}
    .obj-item-delete:hover {{ background: #3a2a2a; }}
    .flight-row {{ display:flex; gap:8px; align-items:stretch; margin-bottom:4px; }}
    .flight-row .obj-item {{ flex: 1.3; }}
    .flight-assign-panel {{ flex: 1; display:flex; align-items:flex-start; justify-content:space-between; gap:12px; font-size:11px; }}
    .flight-assign-col {{ display:flex; flex-direction:column; align-items:flex-start; gap:4px; min-width:0; }}
    .flight-assign-col-arr {{ width:110px; }}
    .flight-assign-col-term {{ width:140px; }}
    .flight-assign-col-dep {{ width:110px; }}
    .flight-assign-label {{ font-size:10px; color:#9ca3af; }}
    .flight-assign-select {{ min-width:90px; padding:4px 8px; font-size:11px; background:#020617; color:#e5e7eb; border-radius:4px; border:1px solid #4b5563; }}
    .flight-schedule-table {{ width:100%; border-collapse:collapse; font-size:11px; margin-top:4px; }}
    .flight-schedule-table thead {{ position:sticky; top:0; z-index:1; background:rgba(30,30,30,0.98); }}
    .flight-schedule-table th {{ text-align:left; padding:6px 6px 6px 0; font-weight:600; color:#9ca3af; font-size:10px; text-transform:uppercase; letter-spacing:0.03em; border-bottom:1px solid #444; white-space:nowrap; }}
    .flight-schedule-table td {{ padding:4px 6px 4px 0; border-bottom:1px solid #333; vertical-align:middle; }}
    .flight-schedule-table tbody tr.flight-data-row {{ cursor:pointer; }}
    .flight-schedule-table tbody tr.flight-data-row:hover {{ background:rgba(60,60,60,0.6); }}
    .flight-schedule-table tbody tr.flight-data-row.selected {{ background:rgba(59,130,246,0.15); }}
    .flight-schedule-table .flight-td-reg {{ font-weight:500; min-width:60px; }}
    .flight-schedule-table .flight-td-time {{ color:#94a3b8; white-space:nowrap; }}
    .flight-schedule-table .flight-td-select {{ padding:2px 0; }}
    .flight-schedule-table .flight-td-select select {{ min-width:0; width:100%; max-width:100px; padding:2px 4px; font-size:10px; }}
    .flight-schedule-table .flight-td-del {{ width:28px; text-align:center; padding:2px 0; }}
    .flight-config-input {{ width:72px; font-size:11px; text-align:right; }}
    .flight-config-table th.sticky-col,
    .flight-config-table td.sticky-col {{ position:sticky; left:0; z-index:2; background:rgba(17,24,39,0.98); }}
    .flight-schedule-table .flight-td-sibt {{ background:rgba(239,68,68,0.06); }}
    .flight-schedule-table thead .flight-td-sibt {{ background:rgba(239,68,68,0.08); }}
    /* Allocation Gantt (Apron × Time, 10% 높이 증가) */
    #allocationGantt {{ margin-top: 10px; padding: 10px 12px 10px 0; background: rgba(15,23,42,0.98); border-radius: 10px; border: 1px solid rgba(55,65,81,0.95); max-height: 572px; overflow-y: auto; overflow-x: auto; }}
    .alloc-terminal-header {{ margin-top: 6px; margin-bottom: 2px; font-size: 11px; font-weight: 600; color: #e5e7eb; text-transform: uppercase; letter-spacing: 0.03em; position:sticky; left:0; z-index:10; padding-left:12px; background:rgba(15,23,42,0.98); }}
    .alloc-row {{ display:flex; align-items:stretch; margin:4px 0; font-size:11px; }}
    .alloc-row-label {{ width:122px; padding:0 8px 0 12px; color:#e5e7eb; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; position:sticky; left:0; z-index:10; background:rgba(15,23,42,0.98); }}
    .alloc-row-track {{ flex:1; position:relative; height:40px; border-radius:6px; background:transparent; border:none; overflow:visible; z-index:1; }}
    .alloc-time-grid-line {{ position:absolute; top:0; bottom:0; width:1px; background:#4b5563; opacity:0.4; transform:translateX(-0.5px); pointer-events:none; }}
    .alloc-apron-bg-slot {{ position:absolute; top:6px; bottom:6px; display:flex; align-items:center; justify-content:center; font-size:8px; font-weight:600; color:rgba(148,163,184,0.35); opacity:0.35; pointer-events:none; white-space:nowrap; }}
    .alloc-flight {{ position:absolute; top:4px; bottom:4px; border-radius:4px; background:#38bdf8; color:#0f172a; padding:2px 4px; font-size:10px; display:flex; flex-direction:column; justify-content:space-between; cursor:default; box-shadow:0 1px 3px rgba(0,0,0,0.5); overflow:visible; }}
    .alloc-flight.conflict {{ background:#f97373; color:#111827; }}
    .alloc-flight.alloc-flight-selected {{ outline:2px solid #fbbf24; outline-offset:2px; box-shadow:0 0 0 1px #0f172a, 0 2px 8px rgba(251,191,36,0.5); z-index:2; }}
    .alloc-apron-tag {{ font-size:9px; font-weight:500; padding:1px 4px; border-radius:9999px; background:rgba(15,23,42,0.92); color:#f9fafb; border:1px solid #4b5563; align-self:flex-start; margin-bottom:1px; white-space:nowrap; }}
    .alloc-flight-reg {{ font-weight:600; font-size:10px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:left; }}
    .alloc-flight-meta {{ font-size:9px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:right; opacity:0.95; }}
    /* SLDT/SIBT/SOBT/STOT 보조 바차트 */
    .alloc-s-bar {{ position:absolute; height:3px; background:#22c55e; border-radius:9999px; opacity:0.8; pointer-events:none; }}
    /* EIBT/EOBT 보조 바차트: 메인 바(SIBT~SOBT)와 두께·위치가 완전히 겹치는 쨍한 핫핑크 하이라이트 */
    .alloc-e-bar {{
      position:absolute;
      top:4px;
      bottom:4px;
      background:#fb37c5;
      border-radius:4px;
      opacity:0.45;
      pointer-events:none;
    }}
    /* ELDT/EIBT, EOBT/ETOT 보조 바차트: 같은 핫핑크 톤의 얇은 하이라이트 */
    .alloc-e2-bar {{ position:absolute; height:2px; background:#fb37c5; border-radius:9999px; opacity:0.9; pointer-events:none; }}
    /* Time axis overlay (Apron/Runway 공통, 간트 하단에 sticky) */
    .alloc-time-axis-overlay {{ position:sticky; bottom:0; z-index:4; background:rgba(15,23,42,0.98); padding-left:122px; }}
    .alloc-time-axis-inner {{ position:relative; height:24px; min-width:100%; font-size:9px; color:#9ca3af; }}
    .rwysep-timeline-root {{ display:flex; align-items:stretch; width:100%; }}
    .rwysep-timeline-label-col {{ display:flex; flex-direction:column; flex-shrink:0; width:110px; }}
    .rwysep-timeline-scroll-col {{ flex:1; overflow-x:auto; }}
    .rwysep-timeline-inner {{ min-width:100%; }}
    .rwysep-reg-tag {{ position:absolute; left:0; top:50%; transform:translate(-50%,-50%); font-size:9px; padding:2px 6px; border-radius:9999px; background:#111827; color:#e5e7eb; border:1px solid #4b5563; white-space:nowrap; pointer-events:none; }}
    .alloc-time-tick {{ position:absolute; bottom:0; transform:translateX(-50%); text-align:center; }}
    .alloc-time-tick-line {{ display:none; }}
    .alloc-time-tick-label {{ white-space:nowrap; }}
    /* Runway Separation Timeline 전용 (Reg × Time) */
    .rwysep-line-s {{ position:absolute; height:7%; top:26%; background:#38bdf8; border-radius:9999px; opacity:0.55; pointer-events:none; }}
    .rwysep-line-e {{ position:absolute; height:7%; top:60%; background:#fb923c; border-radius:9999px; opacity:0.55; pointer-events:none; }}
    .rwysep-tri {{ position:absolute; width:0; height:0; border-left:4px solid transparent; border-right:4px solid transparent; transform:translateX(-50%); pointer-events:none; }}
    .rwysep-head-row {{ display:flex; align-items:center; font-size:9px; color:#9ca3af; margin-bottom:2px; }}
    /* Runway Separation Timeline 헤더 라벨 폭을 Apron Gantt 라벨(alloc-row-label)과 동일하게 맞춰
       상단 S/E 헤더 트랙과 각 Reg 행, 아래 시간축의 세로 눈금이 정확히 정렬되도록 한다. */
    .rwysep-head-label {{ width:122px; padding:0 8px 0 12px; text-align:left; position:sticky; left:0; z-index:10; background:rgba(15,23,42,0.98); }}
    .rwysep-head-track {{ flex:1; position:relative; height:14px; z-index:1; }}
    #directionModesList {{ width: 100%; }}
    .direction-mode-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 6px; width: 100%; min-width: 0; }}
    .direction-mode-row .direction-mode-name {{ flex: 1; min-width: 90px; width: 0; max-width: 100%; }}
    .direction-mode-row select.direction-mode-dir {{ flex-shrink: 0; width: auto; min-width: 95px; }}
    .direction-mode-row .direction-mode-delete {{ flex-shrink: 0; padding: 2px 8px; font-size: 12px; }}
    #object-info {{ margin-top: 10px; padding: 8px; border-radius: 6px; background: rgba(0,0,0,0.3); border: 1px solid #444; font-size: 11px; display: none; }}
    #view3d-container {{ position: absolute; inset: 0; z-index: 10; display: none; pointer-events: auto; }}
    #view3d-container.active {{ display: block; }}
    #layout-name-bar {{ position: absolute; top: 8px; left: 50%; transform: translateX(-50%); z-index: 8; pointer-events: none; font-size: 11px; color: #9ca3af; background: rgba(0,0,0,0.4); padding: 4px 10px; border-radius: 6px; }}
    .right-panel-tabs {{ display: flex; gap: 2px; margin-bottom: 10px; border-bottom: 1px solid #444; }}
    .right-panel-tab {{ flex: 1; padding: 6px 8px; font-size: 11px; cursor: pointer; background: #252525; color: #9ca3af; border: none; border-radius: 4px 4px 0 0; }}
    .right-panel-tab:hover {{ background: #333; color: #e5e7eb; }}
    .right-panel-tab.active {{ background: #1a1a1a; color: #3b82f6; border-bottom: 1px solid #1a1a1a; margin-bottom: -1px; }}
    .tab-content {{ display: none; width: 100%; }}
    .token-nodes {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-bottom: 8px; }}
    .token-nodes .token-node {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #9ca3af; cursor: pointer; }}
    .token-nodes .token-node input {{ margin: 0; }}
    .token-object-pane {{ margin-bottom: 8px; }}
    .token-object-pane label {{ font-size: 11px; color: #9ca3af; display: block; margin-top: 4px; }}
    .token-object-pane select {{ width: 100%; max-width: 200px; font-size: 11px; padding: 4px; background: #1a1a1a; color: #e5e7eb; border: 1px solid #444; border-radius: 4px; }}
    .token-auto-label {{ font-size: 11px; color: #6b7280; }}
    .tab-content.active {{ display: block; }}
    .layout-save-load-tabs {{ display: flex; gap: 2px; margin-bottom: 8px; }}
    .layout-save-load-tab {{ flex: 1; padding: 4px 6px; font-size: 10px; cursor: pointer; background: #252525; color: #9ca3af; border: none; border-radius: 4px; }}
    .layout-save-load-tab:hover {{ background: #333; color: #e5e7eb; }}
    .layout-save-load-tab.active {{ background: #1e3a5f; color: #93c5fd; }}
    /* Flight 탭 내부 Schedule / Configuration 서브탭도 Save/Load 탭과 동일한 파란 음영 사용 */
    .flight-subtab.active {{ background: #1e3a5f; color: #93c5fd; }}
    .layout-save-load-pane {{ display: none; }}
    .layout-save-load-pane.active {{ display: block; }}
    #layoutLoadList {{ max-height: 140px; overflow-y: auto; margin-top: 6px; }}
    #layoutLoadList .layout-load-item {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; padding: 6px 8px; margin-bottom: 4px; border-radius: 4px; background: rgba(50,50,50,0.9); border: 1px solid #444; font-size: 11px; cursor: pointer; }}
    #layoutLoadList .layout-load-item:hover {{ background: #3a3a3a; border-color: #3b82f6; }}
    #layoutLoadList .layout-load-name {{ flex: 1; min-width: 0; }}
    #layoutLoadList .layout-load-delete {{ flex-shrink: 0; padding: 2px 6px; font-size: 12px; cursor: pointer; border: none; background: transparent; color: #9ca3af; border-radius: 3px; }}
    #layoutLoadList .layout-load-delete:hover {{ background: rgba(239,68,68,0.3); color: #f87171; }}

    /* Runway Separation tab */
    #rwySepPanel {{ font-size: 11px; color: #e5e7eb; }}
    .rwysep-rwy-bar {{ display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }}
    .rwysep-rwy-tabs {{ display:flex; flex-wrap:wrap; gap:4px; }}
    .rwysep-rwy-btn {{ background:#1a1a1a; border:1px solid #444; color:#e5e7eb; padding:4px 10px; font-size:11px; border-radius:9999px; cursor:pointer; }}
    .rwysep-rwy-btn.active {{ background:#111827; border-color:#3b82f6; color:#bfdbfe; }}
    .rwysep-rwy-del {{ background:transparent; border:none; color:#6b7280; cursor:pointer; font-size:11px; padding:0 4px; }}
    .rwysep-block {{ margin-top:8px; padding:8px 10px; border-radius:8px; border:1px solid #444; background:rgba(0,0,0,0.3); }}
    .rwysep-label {{ font-size:10px; color:#9ca3af; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em; }}
    .rwysep-row {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:6px; }}
    .rwysep-row select {{ width:auto; min-width:90px; max-width:140px; font-size:11px; padding:4px 8px; background:#1a1a1a; color:#e5e7eb; border-radius:4px; border:1px solid #444; }}
    .rwysep-matrix-wrap {{ margin-top:8px; overflow-x:auto; }}
    .rwysep-table {{ border-collapse:collapse; min-width:360px; }}
    .rwysep-table th, .rwysep-table td {{ border:1px solid #1f2937; padding:4px 6px; text-align:center; }}
    .rwysep-table th {{ background:#1a1a1a; color:#9ca3af; font-size:10px; }}
    .rwysep-table input {{ width:56px; background:#1a1a1a; border:1px solid #444; color:#e5e7eb; font-size:10px; padding:2px 4px; text-align:center; border-radius:3px; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="app">
    <div id="layout-name-bar"></div>
    <div id="toolbar">
      <div id="sim-controls-container" style="display:none;">
        <button type="button" class="tool-btn" id="btnPlayFlights" title="재생">▶ Play</button>
        <button type="button" class="tool-btn" id="btnPauseFlights" title="일시정지">⏸ Pause</button>
        <select id="flightSpeed">
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
          <option value="20" selected>20x</option>
          <option value="50">50x</option>
          <option value="100">100x</option>
          <option value="200">200x</option>
        </select>
        <label for="flightSimSlider">Current</label>
        <input type="range" id="flightSimSlider" min="0" max="100" value="0" step="1" />
        <span id="flightSimTimeLabel">00:00:00</span>
      </div>
      <div id="view-toggle">
        <button class="tool-btn" id="btnView2D" title="2D 뷰">2D</button>
        <button class="tool-btn" id="btnView3D" title="3D orbit 뷰">3D</button>
        <button class="tool-btn" id="btnResetView" title="전체 격자 보기">Fit</button>
        <button class="tool-btn" id="btnGlobalUpdate" title="모든 뷰/계산 새로고침" style="margin-left:8px;">Update</button>
      </div>
    </div>
    <div id="canvas-container">
      <canvas id="grid-canvas"></canvas>
    </div>
    <div id="info-bar">
      <span id="hint">Draw terminal: click grid points, then near first point to close. Drag to pan, scroll to zoom.</span>
      <span id="coord"></span>
    </div>
    <div id="flight-tooltip" style="position:absolute;pointer-events:none;padding:2px 6px;font-size:11px;border-radius:4px;background:rgba(15,23,42,0.9);color:#f9fafb;border:1px solid rgba(148,163,184,0.7);display:none;z-index:40;"></div>
    <div id="right-panel">
      <button id="panel-toggle" title="Toggle panel">◀</button>
      <div class="panel-content">
        <div id="api-warning-banner" style="display:none;background:#7f1d1d;color:#fecaca;padding:8px 10px;font-size:11px;border-radius:6px;margin-bottom:10px;line-height:1.4;">
          <strong>API 연결 안 됨</strong><br/>
          Save/Load/Run Simulation을 사용하려면 <strong>python run_app.py</strong>로 실행한 뒤<br/>
          <strong>http://127.0.0.1:8501</strong> 로 접속하세요. (streamlit run 사용 시 동작하지 않습니다)
        </div>
        <div class="right-panel-tabs">
          <button type="button" class="right-panel-tab active" data-tab="settings">Layout</button>
          <button type="button" class="right-panel-tab" data-tab="flight">Flight</button>
          <button type="button" class="right-panel-tab" data-tab="rwysep">Runway</button>
          <button type="button" class="right-panel-tab" data-tab="allocation">Apron</button>
          <button type="button" class="right-panel-tab" data-tab="simulation">Simulation</button>
          <button type="button" class="right-panel-tab" data-tab="saveload">Save/Load</button>
        </div>

        <div id="tab-settings" class="tab-content active">
        <div class="section-title">Layout</div>
        <label>Mode</label>
        <select id="settingMode">
          <option value="grid">Grid</option>
          <option value="terminal">Terminal</option>
          <option value="pbb">Contact Stand</option>
          <option value="remote">Remote stand</option>
          <option value="taxiway">Taxiway &amp; Runway</option>
          <option value="apronTaxiway">Apron Taxiway</option>
        </select>

        <div id="settings-grid" class="settings-pane">
          <label>Cell size (m)</label>
          <input type="number" id="gridCellSize" min="10" max="1000" value="20" step="10" />
          <label>Columns</label>
          <input type="number" id="gridCols" min="5" max="500" value="200" />
          <label>Rows</label>
          <input type="number" id="gridRows" min="5" max="500" value="200" />
        </div>
        <div id="settings-terminal" class="settings-pane" style="display:none;">
          <label>Building name</label>
          <input type="text" id="terminalName" placeholder="e.g. T1" />
          <label>Floors</label>
          <input type="number" id="terminalFloors" min="1" max="20" value="1" step="1" />
          <label>Floor-to-floor height (m)</label>
          <input type="number" id="terminalFloorToFloor" min="1" max="10" value="4" step="0.5" />
          <label>Departure capacity</label>
          <input type="number" id="terminalDepartureCapacity" min="0" value="0" step="1" />
          <label>Arrival capacity</label>
          <input type="number" id="terminalArrivalCapacity" min="0" value="0" step="1" />
          <button class="small" id="btnTerminalDraw">Draw</button>
        </div>
        <div id="settings-pbb" class="settings-pane" style="display:none;">
          <label>Name</label>
          <input type="text" id="standName" placeholder="e.g. Gate 1" />
          <label>Category (ICAO)</label>
          <select id="standCategory">
            <option value="A">A</option><option value="B">B</option><option value="C" selected>C</option>
            <option value="D">D</option><option value="E">E</option><option value="F">F</option>
          </select>
          <label>Contact Stand length (cells)</label>
          <select id="pbbLength"><option value="1">1</option><option value="2" selected>2</option><option value="3">3</option></select>
        </div>
        <div id="settings-remote" class="settings-pane" style="display:none;">
          <label>Category (ICAO)</label>
          <select id="remoteCategory">
            <option value="A">A</option><option value="B">B</option><option value="C" selected>C</option>
            <option value="D">D</option><option value="E">E</option><option value="F">F</option>
          </select>
        </div>
        <div id="settings-taxiway" class="settings-pane" style="display:none;">
          <label>Type</label>
          <select id="pathType">
            <option value="runway">Runway</option>
            <option value="runway_exit">Runway Exit Taxiway</option>
            <option value="taxiway" selected>Taxiway</option>
          </select>
          <label>Name</label>
          <input type="text" id="taxiwayName" placeholder="e.g. Taxiway A" />
          <label>Width (m)</label>
          <input type="number" id="taxiwayWidth" min="10" max="100" value="15" step="1" />
          <div id="runwayDepPointExtras" style="display:none;margin-top:6px;">
            <label>Runway line-up point position (m)</label>
            <input type="number" id="runwayDepPointPos" min="0" max="500" value="0" step="1" />
            <p style="font-size:10px;color:#9ca3af;margin-top:2px;">0 = Start Point (default), 최대 500&nbsp;m까지 입력 가능합니다.</p>
          </div>
          <div id="runwayExitExtras" style="display:none;margin-top:6px;">
            <label>Max Exit Velocity</label>
            <input type="number" id="taxiwayMaxExitVel" min="1" max="150" value="30" step="1" />
          </div>
          <label style="margin-top:10px;">Taxiway Direction Mode</label>
          <select id="taxiwayDirectionMode">
            <option value="clockwise">CW</option>
            <option value="counter_clockwise">CCW</option>
            <option value="both" selected>Both</option>
          </select>
          <div id="runwayDirectionWrap" style="display:none;margin-top:10px;">
            <label>Direction</label>
            <select id="runwayDirectionInTaxiwayPane">
              <option value="clockwise">Clockwise</option>
              <option value="counter_clockwise">Counter CW</option>
              <option value="both" selected>Both</option>
            </select>
          </div>
          <button class="small" id="btnTaxiwayDraw">Draw</button>
        </div>
        <div id="settings-apronTaxiway" class="settings-pane" style="display:none;">
          <p style="font-size:11px;color:#9ca3af;margin-top:4px;">
            Click a Contact Stand (PBB end) and a Taxiway point to create an Apron–Taxiway link.
          </p>
        </div>
        <div id="settings-runway" class="settings-pane" style="display:none;">
          <label>Name</label>
          <input type="text" id="runwayName" placeholder="e.g. RWY 09/27" />
          <label>Length (m)</label>
          <input type="number" id="runwayLength" min="100" max="6000" value="3000" step="10" />
          <label>Width (m)</label>
          <input type="number" id="runwayWidth" min="20" max="80" value="45" step="1" />
          <label style="margin-top:10px;">Direction</label>
          <select id="runwayDirection">
            <option value="clockwise">Clockwise</option>
            <option value="counter_clockwise">Counter CW</option>
            <option value="both" selected>Both</option>
          </select>
          <button class="small" id="btnRunwayDraw" style="margin-top:6px;">Draw</button>
          <p style="font-size:11px;color:#9ca3af;margin-top:4px;">
            In Runway mode: click start point, then end point (bearing). Runway is drawn from start in that direction with the given length.
          </p>
        </div>

        <div class="section-title">Objects</div>
        <div id="object-list" class="obj-list"></div>
        <div id="object-info">Select an object on the grid or from the list.</div>
        </div>

        <div id="tab-flight" class="tab-content">
          <div class="section-title">Flight</div>
          <div class="layout-save-load-tabs" style="margin-top:4px;">
            <button type="button" class="layout-save-load-tab flight-subtab active" data-flight-subtab="schedule">Flight Schedule</button>
            <button type="button" class="layout-save-load-tab flight-subtab" data-flight-subtab="config">Flight Configuration</button>
          </div>
          <!-- Arr / Dep 선택은 내부 호환성을 위해 남겨두되, UI에서는 숨김 -->
          <label style="display:none;">Arr / Dep</label>
          <select id="flightArrDep" style="display:none;">
            <option value="Arr" selected>Arr (Arrival)</option>
            <option value="Dep">Dep (Departure)</option>
          </select>
          <div id="flightPaneSchedule">
            <label>SIBT (Scheduled In Block Time)</label>
            <input type="text" id="flightTime" placeholder="예: 0, 12:30, 09:23:45 (분 / HH:MM:SS)" />
            <label>Aircraft type</label>
            <select id="flightAircraftType">
              <!-- Populated from INFORMATION.tiers.aircraft.types -->
            </select>
            <label>Velocity (m/s)</label>
            <input type="number" id="flightVelocity" min="1" max="200" value="15" step="1" />
            <label>Reg. number</label>
            <input type="text" id="flightReg" placeholder="예: HL1234" />
            <label>Dwell time (분, 스탠드 점유 시간)</label>
            <input type="number" id="flightDwell" min="0" max="600" value="60" step="5" />
            <label>Min Dwell (분, 최소 드웰·Turnaround 보장)</label>
            <input type="number" id="flightMinDwell" min="0" max="600" value="0" step="5" title="도착 지연 시 EOBT = EIBT + Min Dwell 로 조정 (0이면 미적용)" />
            <button class="small" id="btnAddFlight">+ Add Flight</button>
            <div id="flightError" style="color:#f97316;font-size:11px;margin-top:4px;"></div>
            <div class="section-title" style="margin-top:10px;">Flight schedule</div>
            <div id="flightList" class="obj-list"></div>
          </div>

          <div id="flightPaneConfig" style="display:none;margin-top:8px;">
            <div class="section-title">Flight Configuration</div>
            <div id="flightConfigList" class="obj-list"></div>
          </div>
        </div>

        <div id="tab-rwysep" class="tab-content">
          <div class="section-title">Runway</div>
          <div id="rwySepPanel" style="margin-top:6px;"></div>
        </div>

        <div id="tab-allocation" class="tab-content">
          <div class="section-title">Apron</div>
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
            <div style="font-size:11px;color:#9ca3af;">Stand allocation (Apron × Time)</div>
            <div style="display:flex;align-items:center;gap:8px;">
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowSldtBars" />
                SLDT-SIBT / SOBT-STOT
              </label>
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowEibtBars" />
                EIBT-EOBT
              </label>
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowEldtBars" />
                ELDT-EIBT / EOBT-ETOT
              </label>
            </div>
          </div>
          <div id="allocationGantt"></div>
        </div>

        <div id="tab-simulation" class="tab-content">
          <div class="section-title">Simulation</div>
          <button type="button" class="small" id="btnRunSimulation" style="background:#1e40af;color:#fff;">Run Simulation</button>
        </div>

        <div id="tab-saveload" class="tab-content">
          <div class="section-title">Layout Save / Load</div>
          <div class="layout-save-load-tabs">
            <button type="button" class="layout-save-load-tab active" data-sltab="save">Save</button>
            <button type="button" class="layout-save-load-tab" data-sltab="load">Load</button>
          </div>
          <div id="layout-save-pane" class="layout-save-load-pane active">
            <label>Layout name</label>
            <input type="text" id="layoutName" placeholder="예: base_layout" />
            <button type="button" class="small" id="btnSaveLayout" style="margin-top:6px;">Save</button>
            <div id="layoutMessage" style="font-size:11px;color:#9ca3af;margin-top:4px;"></div>
          </div>
          <div id="layout-load-pane" class="layout-save-load-pane">
            <div class="section-title" style="margin-top:0;">Layout storage</div>
            <div id="layoutLoadList" class="obj-list"></div>
          </div>
        </div>
      </div>
    </div>
    <div id="view3d-container"></div>
  </div>

  <script>
  (function() {{
    const LAYOUT_API_URL = {json.dumps(LAYOUT_API_URL)};
    const LAYOUT_NAMES = {json.dumps(layout_names)};
    const INITIAL_LAYOUT = {json.dumps(layout_for_html)};
    const INITIAL_LAYOUT_DISPLAY_NAME = {json.dumps(layout_display_name)};
    const INFORMATION = {json.dumps(INFORMATION)};
    let GRID_COLS = {GRID_COLS};
    let GRID_ROWS = {GRID_ROWS};
    let CELL_SIZE = {CELL_SIZE};

    const canvas = document.getElementById('grid-canvas');
    const container = document.getElementById('canvas-container');
    const hintEl = document.getElementById('hint');
    const coordEl = document.getElementById('coord');
    const objectInfoEl = document.getElementById('object-info');
    const objectListEl = document.getElementById('object-list');
    const flightTooltip = document.getElementById('flight-tooltip');
    const settingModeSelect = document.getElementById('settingMode');
    const panel = document.getElementById('right-panel');
    const panelToggle = document.getElementById('panel-toggle');
    const resetViewBtn = document.getElementById('btnResetView');

    function id() {{ return 'id_' + Math.random().toString(36).slice(2, 11); }}
    function escapeHtml(str) {{
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    const state = {{
      terminals: [],
      pbbStands: [],
      remoteStands: [],
      taxiways: [],
      apronLinks: [],
      runways: [],
      directionModes: [],
      // 현재 선택/로드된 레이아웃 이름 (Simulation 요청 시 사용)
      currentLayoutName: String(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout'),
      // Flight / simulation state
      flights: [],
      simTimeSec: 0,
      simStartSec: 0,
      simDurationSec: 0,
      simPlaying: false,
      simSpeed: 20,
      currentTerminalId: null,
      selectedObject: null,
      terminalDrawingId: null,
      taxiwayDrawingId: null,
      dragVertex: null,
      dragTaxiwayVertex: null,
      scale: 1,
      panX: 0,
      panY: 0,
      isPanning: false,
      dragStart: null,
      previewRemote: null,
      previewPbb: null,
      apronLinkTemp: null,
      runwayTemp: null,
      runwayDrawing: false,
      hoverCell: null,
    }};
    // #region agent log (No Way 디버깅)
    function _logNoWay(payload) {{
      try {{
        fetch('http://127.0.0.1:7242/ingest/d3690a39-df65-41bd-83b6-eb0ef6ed1c98', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ ...payload, timestamp: Date.now() }}) }}).catch(function() {{}});
      }} catch (e) {{}}
    }}
    // #endregion
    function applyLayoutObject(obj) {{
      if (!obj || typeof obj !== 'object') return;
      if (obj.grid) {{
        if (typeof obj.grid.cols === 'number') GRID_COLS = obj.grid.cols;
        if (typeof obj.grid.rows === 'number') GRID_ROWS = obj.grid.rows;
        if (typeof obj.grid.cellSize === 'number') CELL_SIZE = obj.grid.cellSize;
      }}
      if (Array.isArray(obj.terminals)) state.terminals = obj.terminals.slice();
      if (Array.isArray(obj.pbbStands)) state.pbbStands = obj.pbbStands.slice();
      if (Array.isArray(obj.remoteStands)) state.remoteStands = obj.remoteStands.slice();
      if (Array.isArray(obj.taxiways)) state.taxiways = obj.taxiways.slice();
      if (Array.isArray(obj.apronLinks)) state.apronLinks = obj.apronLinks.slice();
      if (Array.isArray(obj.directionModes) && obj.directionModes.length) {{
        state.directionModes = obj.directionModes.slice();
      }}
      if (Array.isArray(obj.flights)) {{
        state.flights = obj.flights.slice();
        state.flights.forEach(f => {{
          const t = f.token || {{}};
          // aircraftType/code: legacy JSON에는 code만 있을 수 있음. aircraftType이 있으면 code 유도, 없으면 code로 aircraftType 매칭
          if (f.aircraftType && typeof getCodeForAircraft === 'function') {{
            f.code = getCodeForAircraft(f.aircraftType);
          }} else if (f.code && typeof AIRCRAFT_TYPES !== 'undefined') {{
            const match = AIRCRAFT_TYPES.find(a => a.icao === f.code);
            f.aircraftType = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
          }}
          // JSON에서 저장된 최소 token 형식: arrRunwayId, apronId, terminalId, depRunwayId
          f.arrRunwayId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
          f.depRunwayId = f.depRunwayId || t.depRunwayId || null;
          f.terminalId = f.terminalId || t.terminalId || null;
          // 항공편-주기장ID 매칭: JSON token.apronId만 소스로 사용. 이 값이 유지되면 Allocation도 유지됨
          const apronId = t.apronId != null ? t.apronId : (f.standId != null ? f.standId : null);
          f.standId = apronId;
          f.token = {{
            nodes: Array.isArray(t.nodes) ? t.nodes.slice() : ['runway','taxiway','apron','terminal'],
            runwayId: f.arrRunwayId || null,
            apronId: apronId,
            terminalId: f.terminalId || null,
            depRunwayId: f.depRunwayId || null,
          }};
        }});
      }} else {{
        state.flights = [];
      }}
      // 주기장 배치는 JSON(apronId)에서만 복원. 타임라인/경로 기반 자동 재배정 없음
      // 시뮬레이션 자동 재생하지 않음
      state.simPlaying = false;
      // flights가 바뀌면 경로·타임라인 계산 후 시뮬레이션 길이·리스트 갱신 (재생 가능하도록)
      if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
      else {{
        if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
        if (typeof renderFlightList === 'function') renderFlightList();
        draw();
      }}
    }}
    function applyInitialLayoutFromJson() {{
      if (!INITIAL_LAYOUT || typeof INITIAL_LAYOUT !== 'object') return;
      applyLayoutObject(INITIAL_LAYOUT);
    }}
    function updateLayoutNameBar(name) {{
      const n = (name && String(name).trim()) || '';
      state.currentLayoutName = n || state.currentLayoutName || 'default_layout';
      const bar = document.getElementById('layout-name-bar');
      if (bar) bar.textContent = n || state.currentLayoutName;
    }}
    function ensureDefaultDirectionModes() {{
      if (state.directionModes.length === 0) {{
        state.directionModes = [
          {{ id: id(), name: 'Mode A', direction: 'clockwise' }},
          {{ id: id(), name: 'Mode B', direction: 'counter_clockwise' }},
          {{ id: id(), name: 'Mode C', direction: 'both' }}
        ];
      }}
    }}
    const undoStack = [];
    const maxUndoLevels = 50;
    function pushUndo() {{
      const snap = {{
        terminals: JSON.parse(JSON.stringify(state.terminals || [])),
        pbbStands: JSON.parse(JSON.stringify(state.pbbStands || [])),
        remoteStands: JSON.parse(JSON.stringify(state.remoteStands || [])),
        taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
        apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
        directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
        flights: JSON.parse(JSON.stringify(state.flights || []))
      }};
      undoStack.push(snap);
      if (undoStack.length > maxUndoLevels) undoStack.shift();
    }}
    function undo() {{
      if (!undoStack.length) return;
      const snap = undoStack.pop();
      state.terminals = snap.terminals;
      state.pbbStands = snap.pbbStands;
      state.remoteStands = snap.remoteStands;
      state.taxiways = snap.taxiways;
      state.apronLinks = snap.apronLinks;
      state.directionModes = snap.directionModes;
      state.flights = snap.flights;
      state.selectedObject = null;
      state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
      state.terminalDrawingId = null;
      state.taxiwayDrawingId = null;
      syncPanelFromState();
      updateObjectInfo();
      renderObjectList();
      if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      if (typeof scene3d !== 'undefined' && scene3d) update3DScene();
    }}
    function getTaxiwayDirection(tw) {{
      if (!tw) return 'both';
      // 새 UI: taxiwayDirectionMode에서 직접 tw.direction에 저장 (clockwise / counter_clockwise / both)
      if (tw.direction != null) {{
        const d = tw.direction;
        if (d === 'topToBottom') return 'clockwise';
        if (d === 'bottomToTop') return 'counter_clockwise';
        return d || 'both';
      }}
      // 구버전 JSON 호환: directionModeId + state.directionModes 사용
      if (tw.directionModeId) {{
        const m = state.directionModes.find(d => d.id === tw.directionModeId);
        if (m && m.direction) return m.direction;
      }}
      return 'both';
    }}

    // ---- Runway Separation config (from Information.json) ----
    const _info = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers) ? INFORMATION : {{}};
    const _rwy = _info.runway || {{}};
    const _stds = _rwy.standards || {{}};
    const RSEP_STD_CATS = {{
      'ICAO': (_stds.ICAO && _stds.ICAO.categories) ? _stds.ICAO.categories : ['J','H','M','L'],
      'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categories) ? _stds['RECAT-EU'].categories : ['A','B','C','D','E','F'],
    }};
    const RSEP_SEQ_TYPES = {{ 'ARR→ARR': 'matrix', 'DEP→DEP': 'matrix', 'ARR→DEP': 'lead-1d', 'DEP→ARR': 'trail-1d' }};
    const RSEP_MODE_SEQS = {{ ARR: ['ARR→ARR'], DEP: ['DEP→DEP'], MIX: ['ARR→ARR','DEP→DEP','ARR→DEP','DEP→ARR'] }};
    const RSEP_DEFAULTS = {{}};
    ['ICAO','RECAT-EU'].forEach(k => {{
      const s = _stds[k];
      if (!s) return;
      RSEP_DEFAULTS[k] = {{ ...(s.separationDefaults || {{}}), ROT: s.ROT || {{}} }};
    }});
    if (!RSEP_DEFAULTS['ICAO'] || !Object.keys(RSEP_DEFAULTS['ICAO']).length) {{
      RSEP_DEFAULTS['ICAO'] = {{ 'ARR→ARR': {{ J:{{J:90,H:120,M:180,L:240}}, H:{{J:90,H:90,M:120,L:180}}, M:{{J:90,H:90,M:90,L:180}}, L:{{J:90,H:90,M:90,L:90}} }}, 'DEP→DEP': {{ J:{{J:90,H:120,M:180,L:180}}, H:{{J:90,H:90,M:120,L:120}}, M:{{J:90,H:90,M:90,L:90}}, L:{{J:90,H:90,M:90,L:90}} }}, 'ARR→DEP': {{J:90,H:80,M:65,L:50}}, 'DEP→ARR': {{J:60,H:60,M:70,L:90}}, ROT: {{J:70,H:65,M:55,L:40}} }};
    }}
    if (!RSEP_DEFAULTS['RECAT-EU'] || !Object.keys(RSEP_DEFAULTS['RECAT-EU']).length) {{
      RSEP_DEFAULTS['RECAT-EU'] = {{ 'ARR→ARR': {{ A:{{A:80,B:100,C:120,D:140,E:160,F:180}}, B:{{A:80,B:80,C:100,D:120,E:120,F:140}}, C:{{A:80,B:80,C:80,D:100,E:100,F:120}}, D:{{A:80,B:80,C:80,D:80,E:80,F:100}}, E:{{A:80,B:80,C:80,D:80,E:80,F:100}}, F:{{A:80,B:80,C:80,D:80,E:80,F:80}} }}, 'DEP→DEP': {{ A:{{A:80,B:100,C:120,D:120,E:120,F:140}}, B:{{A:80,B:80,C:100,D:100,E:100,F:120}}, C:{{A:80,B:80,C:80,D:80,E:80,F:100}}, D:{{A:80,B:80,C:80,D:80,E:80,F:80}}, E:{{A:80,B:80,C:80,D:80,E:80,F:80}}, F:{{A:80,B:80,C:80,D:80,E:80,F:80}} }}, 'ARR→DEP': {{A:80,B:70,C:60,D:55,E:50,F:45}}, 'DEP→ARR': {{A:55,B:55,C:60,D:65,E:70,F:80}}, ROT: {{A:65,B:60,C:55,D:50,E:45,F:40}} }};
    }}
    const RSEP_STANDARDS = {{ 'ICAO': {{ ROT: RSEP_DEFAULTS['ICAO'] && RSEP_DEFAULTS['ICAO'].ROT ? RSEP_DEFAULTS['ICAO'].ROT : {{}} }}, 'RECAT-EU': {{ ROT: RSEP_DEFAULTS['RECAT-EU'] && RSEP_DEFAULTS['RECAT-EU'].ROT ? RSEP_DEFAULTS['RECAT-EU'].ROT : {{}} }} }};
    const RSEP_CAT_LABELS = {{
      'ICAO': (_stds.ICAO && _stds.ICAO.categoryLabels) ? _stds.ICAO.categoryLabels : {{ J:'Super', H:'Heavy', M:'Medium', L:'Light' }},
      'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categoryLabels) ? _stds['RECAT-EU'].categoryLabels : {{ A:'Super-Heavy', B:'Upper-Heavy', C:'Lower-Heavy', D:'Medium', E:'Light', F:'Very-Light' }},
    }};
    const RSEP_SEQ_META = _rwy.seqMeta || {{
      'ARR→ARR': {{ driver: 'Wake of leading arrival aircraft', refPoint: 'Touchdown / final approach point of the leading arrival', input: 'Lead (arrival) × Trail (arrival) matrix input' }},
      'DEP→DEP': {{ driver: 'Wake of leading departure aircraft', refPoint: 'Take-off / runway entry point of the leading departure', input: 'Lead (departure) × Trail (departure) matrix input' }},
      'ARR→DEP': {{ driver: 'Runway occupancy time (ROT) of leading arrival', refPoint: 'Runway vacation / ROT end of the leading arrival', input: 'Lead (arrival category) 1‑D input' }},
      'DEP→ARR': {{ driver: 'Wake / ROT of leading departure', refPoint: 'Runway vacation / ROT end of the leading departure', input: 'Trail (arrival category) 1‑D input' }},
    }};
    function rsepGetCatLabel(stdKey, cat) {{
      const t = RSEP_CAT_LABELS[stdKey];
      if (!t) return '';
      return t[cat] || '';
    }}
    function rsepGetSeqMeta(seq) {{
      return RSEP_SEQ_META[seq] || null;
    }}
    function rsepMakeMatrix(cats, src) {{
      const m = {{}};
      cats.forEach(l => {{
        m[l] = {{}};
        cats.forEach(t => {{
          m[l][t] = src && src[l] && src[l][t] != null ? String(src[l][t]) : '';
        }});
      }});
      return m;
    }}
    function rsepMake1D(cats, src) {{
      const d = {{}};
      cats.forEach(c => {{
        d[c] = src && src[c] != null ? String(src[c]) : '';
      }});
      return d;
    }}
    function rsepMakeSeqData(stdKey) {{
      const cats = RSEP_STD_CATS[stdKey] || [];
      const def = RSEP_DEFAULTS[stdKey] || {{}};
      return {{
        'ARR→ARR': rsepMakeMatrix(cats, def['ARR→ARR']),
        'DEP→DEP': rsepMakeMatrix(cats, def['DEP→DEP']),
        'ARR→DEP': rsepMake1D(cats, def['ARR→DEP']),
        'DEP→ARR': rsepMake1D(cats, def['DEP→ARR']),
      }};
    }}

    function rsepColorForValue(val) {{
      const n = Number(val);
      if (!isFinite(n) || val === '' || val == null) {{
        return {{ bg: '#1a1a1a', color: '#e5e7eb', border: '#444444' }};
      }}
      if (n < 90) {{
        return {{ bg: '#0d2018', color: '#68d391', border: '#68d39155' }};
      }}
      if (n < 120) {{
        return {{ bg: '#0d1a28', color: '#63b3ed', border: '#63b3ed55' }};
      }}
      if (n < 150) {{
        return {{ bg: '#1e1e08', color: '#f6e05e', border: '#f6e05e55' }};
      }}
      return {{ bg: '#280d0d', color: '#fc8181', border: '#fc818155' }};
    }}
    function rsepMakeConfig(stdKey) {{
      const std = RSEP_STANDARDS[stdKey] || RSEP_STANDARDS['ICAO'];
      const cats = RSEP_STD_CATS[stdKey];
      const rot = std.ROT || {{}};
      const rotCopy = {{}};
      cats.forEach(c => {{ rotCopy[c] = rot[c] != null ? String(rot[c]) : ''; }});
      return {{
        standard: stdKey,
        mode: 'MIX',
        activeSeq: 'ARR→ARR',
        seqData: rsepMakeSeqData(stdKey),
        rot: rotCopy,
      }};
    }}
    function rsepGetConfigForRunway(rw) {{
      if (!rw) return null;
      if (!rw.rwySepConfig) {{
        rw.rwySepConfig = rsepMakeConfig('ICAO');
      }}
      // 표준이 바뀐 JSON이 들어올 수도 있으므로 cats 수에 안 맞으면 리셋
      const cfg = rw.rwySepConfig;
      if (!RSEP_STD_CATS[cfg.standard]) {{
        rw.rwySepConfig = rsepMakeConfig('ICAO');
        return rw.rwySepConfig;
      }}
      return cfg;
    }}
    let dpr = window.devicePixelRatio || 1;
    let ctx = canvas.getContext('2d');

    function screenToWorld(sx, sy) {{
      return [(sx - state.panX) / state.scale, (sy - state.panY) / state.scale];
    }}
    function cellToPixel(col, row) {{ return [col * CELL_SIZE, row * CELL_SIZE]; }}
    function pixelToCell(x, y) {{
      let col = Math.round(x / CELL_SIZE);
      let row = Math.round(y / CELL_SIZE);
      col = Math.max(0, Math.min(GRID_COLS, col));
      row = Math.max(0, Math.min(GRID_ROWS, row));
      return [col, row];
    }}
    const ICAO_STAND_SIZE_M = {{ A: 20, B: 30, C: 40, D: 50, E: 60, F: 80 }};
    function getStandSizeMeters(cat) {{ return ICAO_STAND_SIZE_M[cat] || 40; }}
    function getStandBoundsRect(cx, cy, sizeM) {{
      const h = sizeM / 2;
      return {{ left: cx - h, right: cx + h, top: cy - h, bottom: cy + h }};
    }}
    function rectsOverlap(a, b) {{
      // Treat only positive-area intersection as overlap.
      // If 두 사각형이 선이나 점으로만 닿는 경우(변이 겹치거나 모서리만 맞닿는 경우)는 겹침으로 보지 않는다.
      return !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
    }}
    function getPBBStandAngle(pbb) {{ return Math.atan2(pbb.y2 - pbb.y1, pbb.x2 - pbb.x1); }}
    function getPBBStandCorners(pbb) {{
      const cx = pbb.x2, cy = pbb.y2;
      const size = getStandSizeMeters(pbb.category || 'C');
      const angle = getPBBStandAngle(pbb);
      const h = size / 2;
      const cos = Math.cos(angle), sin = Math.sin(angle);
      return [
        [cx + (-h)*cos - (-h)*sin, cy + (-h)*sin + (-h)*cos],
        [cx + ( h)*cos - (-h)*sin, cy + ( h)*sin + (-h)*cos],
        [cx + ( h)*cos - ( h)*sin, cy + ( h)*sin + ( h)*cos],
        [cx + (-h)*cos - ( h)*sin, cy + (-h)*sin + ( h)*cos]
      ];
    }}
    function pointInPolygonXY(p, verts) {{
      let inside = false;
      const n = verts.length;
      for (let i = 0, j = n - 1; i < n; j = i++) {{
        const vi = verts[i], vj = verts[j];
        if (((vi[1] > p[1]) !== (vj[1] > p[1])) && (p[0] < (vj[0]-vi[0])*(p[1]-vi[1])/(vj[1]-vi[1])+vi[0])) inside = !inside;
      }}
      return inside;
    }}
    function segIntersect(a1, a2, b1, b2) {{
      const [ax1,ay1]=a1,[ax2,ay2]=a2,[bx1,by1]=b1,[bx2,by2]=b2;
      const dax = ax2-ax1, day = ay2-ay1, dbx = bx2-bx1, dby = by2-by1;
      const den = dax*dby - day*dbx;
      if (Math.abs(den) < 1e-10) return false;
      const t = ((bx1-ax1)*dby - (by1-ay1)*dbx) / den;
      const s = ((bx1-ax1)*day - (by1-ay1)*dax) / den;
      return t >= 0 && t <= 1 && s >= 0 && s <= 1;
    }}
    function rotatedRectsOverlap(cornersA, cornersB) {{
      for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersA[i], cornersB)) return true;
      for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersB[i], cornersA)) return true;
      for (let i = 0; i < 4; i++) {{
        const a1 = cornersA[i], a2 = cornersA[(i+1)%4];
        for (let j = 0; j < 4; j++) {{
          if (segIntersect(a1, a2, cornersB[j], cornersB[(j+1)%4])) return true;
        }}
      }}
      return false;
    }}
    function pbbStandOverlapsTerminal(pbb) {{
      const corners = getPBBStandCorners(pbb);
      for (let t = 0; t < state.terminals.length; t++) {{
        const term = state.terminals[t];
        if (!term.closed || term.vertices.length < 3) continue;
        const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
        for (let k = 0; k < 4; k++) {{
          if (pointInPolygonXY(corners[k], termPix)) return true;
        }}
        for (let k = 0; k < termPix.length; k++) {{
          if (pointInPolygonXY(termPix[k], corners)) return true;
        }}
      }}
      return false;
    }}
    function pbbStandOverlapsExisting(pbb, excludeId) {{
      if (pbbStandOverlapsTerminal(pbb)) return true;
      const corners = getPBBStandCorners(pbb);
      for (let i = 0; i < state.pbbStands.length; i++) {{
        const other = state.pbbStands[i];
        if (excludeId && other.id === excludeId) continue;
        if (rotatedRectsOverlap(corners, getPBBStandCorners(other))) return true;
      }}
      for (let i = 0; i < state.remoteStands.length; i++) {{
        const st = state.remoteStands[i];
        const [cx, cy] = cellToPixel(st.col, st.row);
        const half = getStandSizeMeters(st.category || 'C') / 2;
        const r = {{ left: cx - half, right: cx + half, top: cy - half, bottom: cy + half }};
        if (pointInPolygonXY([cx, cy], corners)) return true;
        for (let k = 0; k < 4; k++) {{
          const p = corners[k];
          if (p[0] >= r.left && p[0] <= r.right && p[1] >= r.top && p[1] <= r.bottom) return true;
        }}
      }}
      return false;
    }}
    function taxiwayOverlapsAnyTerminal(tw) {{
      if (!tw || !tw.vertices || tw.vertices.length < 2) return false;
      const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
      // 각 vertex 가 어떤 터미널 안에 들어가는지 체크
      for (let t = 0; t < state.terminals.length; t++) {{
        const term = state.terminals[t];
        if (!term.closed || term.vertices.length < 3) continue;
        const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
        for (let i = 0; i < vertsPix.length; i++) {{
          if (pointInPolygonXY(vertsPix[i], termPix)) return true;
        }}
        // Segments vs terminal polygon edges 교차 여부 체크
        for (let i = 0; i < vertsPix.length - 1; i++) {{
          const a1 = vertsPix[i], a2 = vertsPix[i+1];
          for (let j = 0; j < termPix.length; j++) {{
            const b1 = termPix[j], b2 = termPix[(j+1) % termPix.length];
            if (segIntersect(a1, a2, b1, b2)) return true;
          }}
        }}
      }}
      return false;
    }}
    function terminalOverlapsAnyTaxiway(term) {{
      if (!term || !term.vertices || term.vertices.length < 3) return false;
      const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
      if (!state.taxiways || !state.taxiways.length) return false;
      for (let i = 0; i < state.taxiways.length; i++) {{
        const tw = state.taxiways[i];
        if (!tw.vertices || tw.vertices.length < 2) continue;
        const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
        // Taxiway vertex 가 터미널 안에 있는지
        for (let k = 0; k < vertsPix.length; k++) {{
          if (pointInPolygonXY(vertsPix[k], termPix)) return true;
        }}
        // Taxiway 세그먼트와 터미널 변 교차 여부
        for (let a = 0; a < vertsPix.length - 1; a++) {{
          const a1 = vertsPix[a], a2 = vertsPix[a+1];
          for (let b = 0; b < termPix.length; b++) {{
            const b1 = termPix[b], b2 = termPix[(b+1) % termPix.length];
            if (segIntersect(a1, a2, b1, b2)) return true;
          }}
        }}
      }}
      return false;
    }}
    function makeUniqueNamedCopy(list, prop) {{
      const nameCount = {{}};
      return (list || []).map(obj => {{
        const copy = Object.assign({{}}, obj);
        const baseRaw = (copy[prop] || '').trim();
        if (!baseRaw) return copy;
        nameCount[baseRaw] = (nameCount[baseRaw] || 0) + 1;
        const n = nameCount[baseRaw];
        copy[prop] = n > 1 ? (baseRaw + ' (' + n + ')') : baseRaw;
        return copy;
      }});
    }}

    function serializeTaxiwayWithEndpoints(tw) {{
      const copy = Object.assign({{}}, tw);
      const dir = getTaxiwayDirection(tw);
      if (dir === 'both') {{
        copy.start_point = null;
        copy.end_point = null;
      }} else {{
        if (tw.vertices && tw.vertices.length >= 2) {{
          const first = tw.vertices[0];
          const last = tw.vertices[tw.vertices.length - 1];
          if (dir === 'clockwise') {{
            copy.start_point = {{ col: first.col, row: first.row }};
            copy.end_point = {{ col: last.col, row: last.row }};
          }} else {{
            copy.start_point = {{ col: last.col, row: last.row }};
            copy.end_point = {{ col: first.col, row: first.row }};
          }}
        }} else {{
          copy.start_point = null;
          copy.end_point = null;
        }}
      }}
      // Runway line-up point를 좌표 형태로 JSON에 저장 (lineup_point)
      if (tw.pathType === 'runway') {{
        if (tw.dep_point && typeof tw.dep_point.col === 'number' && typeof tw.dep_point.row === 'number') {{
          copy.lineup_point = {{ col: tw.dep_point.col, row: tw.dep_point.row }};
        }} else if (typeof tw.depPointPos === 'number' && tw.vertices && tw.vertices.length >= 2) {{
          const verts = tw.vertices;
          // col/row 차이를 1m 단위로 보고 depPointPos(m)를 길이에 대한 비율로 변환
          const totalLen = verts.reduce((acc, v, i) =>
            acc + (i > 0 ? Math.hypot(v.col - verts[i-1].col, v.row - verts[i-1].row) : 0), 0);
          if (totalLen > 0) {{
            const desired = Math.max(0, Math.min(500, tw.depPointPos)); // 0~500m
            const ratio = Math.max(0, Math.min(1, desired / totalLen));
            const targetDist = totalLen * ratio;
            let acc = 0;
            let cx = verts[0].col, cy = verts[0].row;
            for (let i = 1; i < verts.length; i++) {{
              const seg = Math.hypot(verts[i].col - verts[i-1].col, verts[i].row - verts[i-1].row);
              if (acc + seg >= targetDist) {{
                const t = seg > 0 ? (targetDist - acc) / seg : 0;
                cx = verts[i-1].col + t * (verts[i].col - verts[i-1].col);
                cy = verts[i-1].row + t * (verts[i].row - verts[i-1].row);
                break;
              }}
              acc += seg;
            }}
            copy.lineup_point = {{ col: Math.round(cx), row: Math.round(cy) }};
          }}
        }}
      }}
      if (tw.rwySepConfig) copy.rwySepConfig = tw.rwySepConfig;
      return copy;
    }}
    function serializeRunwayWithEndpoints(rw) {{
      const copy = Object.assign({{}}, rw);
      const dir = getTaxiwayDirection(rw);
      if (dir === 'both') {{
        copy.start_point = null;
        copy.end_point = null;
      }} else {{
        const [sx, sy] = cellToPixel(rw.startCol, rw.startRow);
        const ex = 2 * rw.cx - sx;
        const ey = 2 * rw.cy - sy;
        const [endCol, endRow] = pixelToCell(ex, ey);
        if (dir === 'clockwise') {{
          copy.start_point = {{ col: rw.startCol, row: rw.startRow }};
          copy.end_point = {{ col: endCol, row: endRow }};
        }} else {{
          copy.start_point = {{ col: endCol, row: endRow }};
          copy.end_point = {{ col: rw.startCol, row: rw.startRow }};
        }}
      }}
      if (rw.rwySepConfig) copy.rwySepConfig = rw.rwySepConfig;
      return copy;
    }}
    function serializeCurrentLayout() {{
      return {{
        grid: {{
          cols: GRID_COLS,
          rows: GRID_ROWS,
          cellSize: CELL_SIZE
        }},
        // 이름이 중복될 경우 Objects 패널에서 보이는 형태(예: "Stand 1 (2)")로 저장
        terminals: makeUniqueNamedCopy(state.terminals, 'name'),
        pbbStands: makeUniqueNamedCopy(state.pbbStands, 'name'),
        remoteStands: state.remoteStands.slice(),
        taxiways: state.taxiways.map(tw => serializeTaxiwayWithEndpoints(tw)),
        apronLinks: state.apronLinks.slice(),
        directionModes: state.directionModes.slice(),
        // 항공편-주기장ID 매칭(apronId)을 JSON에 포함해 두면, 불러올 때 Allocation이 그대로 복원됨
        flights: state.flights.map(function(f) {{
          const copy = {{ }};
          for (const k in f) {{
            if (
              k === 'timeline' ||
              k === 'arrDep' ||
              k === 'token' ||
              k === 'arrRunwayId' ||
              k === 'depRunwayId' ||
              k === 'terminalId' ||
              k === 'standId'
            ) continue;
            copy[k] = f[k];
          }}
          const t = f.token || {{}};
          copy.token = {{
            arrRunwayId: f.arrRunwayId || t.arrRunwayId || t.runwayId || null,
            apronId: (f.standId != null ? f.standId : (t.apronId != null ? t.apronId : null)),
            terminalId: f.terminalId || t.terminalId || null,
            depRunwayId: f.depRunwayId || t.depRunwayId || null,
          }};
          if (!copy.token.apronId) copy.token.apronId = null;
          return copy;
        }})
      }};
    }}
    function getExistingStandBounds() {{
      const list = [];
      state.remoteStands.forEach(st => {{
        const [cx, cy] = cellToPixel(st.col, st.row);
        list.push(getStandBoundsRect(cx, cy, getStandSizeMeters(st.category || 'C')));
      }});
      state.pbbStands.forEach(pbb => {{
        const corners = getPBBStandCorners(pbb);
        let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
        for (let k = 1; k < 4; k++) {{
          left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
          top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
        }}
        list.push({{ left, right, top, bottom }});
      }});
      return list;
    }}
    function standOverlapsExisting(bounds) {{
      const existing = getExistingStandBounds();
      for (let i = 0; i < existing.length; i++) if (rectsOverlap(bounds, existing[i])) return true;
      return false;
    }}
    function dist2(a, b) {{ const dx = a[0]-b[0], dy = a[1]-b[1]; return dx*dx+dy*dy; }}
    function closestPointOnSegment(p1, p2, p) {{
      const [x1,y1]=p1,[x2,y2]=p2,[px,py]=p;
      const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
      if (len2===0) return null;
      let t = ((px-x1)*dx+(py-y1)*dy)/len2;
      t = Math.max(0,Math.min(1,t));
      return [x1+t*dx,y1+t*dy];
    }}

    function pointInPolygon(p, verts) {{
      let inside = false;
      const n = verts.length;
      for (let i = 0, j = n - 1; i < n; j = i++) {{
        const vi = cellToPixel(verts[i].col, verts[i].row);
        const vj = cellToPixel(verts[j].col, verts[j].row);
        if (((vi[1] > p[1]) !== (vj[1] > p[1])) && (p[0] < (vj[0]-vi[0])*(p[1]-vi[1])/(vj[1]-vi[1])+vi[0])) inside = !inside;
      }}
      return inside;
    }}

    function hitTest(wx, wy) {{
      const click = [wx, wy];
      for (let i = state.remoteStands.length - 1; i >= 0; i--) {{
        const st = state.remoteStands[i];
        const [cx, cy] = cellToPixel(st.col, st.row);
        const half = getStandSizeMeters(st.category || 'C') / 2;
        if (Math.abs(wx - cx) <= half && Math.abs(wy - cy) <= half)
          return {{ type: 'remote', id: st.id, obj: st }};
      }}
      for (let i = state.pbbStands.length - 1; i >= 0; i--) {{
        const pbb = state.pbbStands[i];
        const near = closestPointOnSegment([pbb.x1,pbb.y1],[pbb.x2,pbb.y2], click);
        if (near && dist2(near, click) < (CELL_SIZE*0.6)**2)
          return {{ type: 'pbb', id: pbb.id, obj: pbb }};
      }}
      for (let i = state.terminals.length - 1; i >= 0; i--) {{
        const t = state.terminals[i];
        if (t.closed && t.vertices.length >= 3 && pointInPolygon(click, t.vertices))
          return {{ type: 'terminal', id: t.id, obj: t }};
      }}
      if (!state.taxiwayDrawingId) {{
        for (let i = state.taxiways.length - 1; i >= 0; i--) {{
          const tw = state.taxiways[i];
          if (tw.vertices.length < 2) continue;
          const halfW = (tw.width != null ? tw.width : 23) / 2;
          const hitD2 = (CELL_SIZE * 0.8 + halfW) ** 2;
          for (let j = 0; j < tw.vertices.length - 1; j++) {{
            const [x1, y1] = cellToPixel(tw.vertices[j].col, tw.vertices[j].row);
            const [x2, y2] = cellToPixel(tw.vertices[j + 1].col, tw.vertices[j + 1].row);
            const near = closestPointOnSegment([x1, y1], [x2, y2], click);
            if (near && dist2(near, click) < hitD2) return {{ type: 'taxiway', id: tw.id, obj: tw }};
          }}
        }}
      }}
      return null;
    }}

    function hitTestTerminalVertex(wx, wy) {{
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * 0.6) ** 2;
      let best = null;
      let bestD2 = maxD2;
      state.terminals.forEach(t => {{
        t.vertices.forEach((v, idx) => {{
          const [vx, vy] = cellToPixel(v.col, v.row);
          const d2 = dist2([vx, vy], click);
          if (d2 < bestD2) {{
            bestD2 = d2;
            best = {{ terminalId: t.id, index: idx }};
          }}
        }});
      }});
      return best;
    }}

    function hitTestTaxiwayVertex(wx, wy) {{
      if (!state.selectedObject || state.selectedObject.type !== 'taxiway') return null;
      const tw = state.selectedObject.obj;
      if (!tw || !tw.vertices || tw.vertices.length === 0) return null;
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * 0.6) ** 2;
      let best = null;
      let bestD2 = maxD2;
      tw.vertices.forEach((v, idx) => {{
        const [vx, vy] = cellToPixel(v.col, v.row);
        const d2 = dist2([vx, vy], click);
        if (d2 < bestD2) {{
          bestD2 = d2;
          best = {{ taxiwayId: tw.id, index: idx }};
        }}
      }});
      return best;
    }}

    function getCurrentTerminal() {{
      if (state.currentTerminalId) {{
        const t = state.terminals.find(x => x.id === state.currentTerminalId);
        if (t) return t;
      }}
      return state.terminals[0] || null;
    }}

    function polygonAreaM2(vertices) {{
      if (!vertices || vertices.length < 3) return 0;
      let area = 0;
      const n = vertices.length;
      for (let i = 0; i < n; i++) {{
        const j = (i + 1) % n;
        area += vertices[i].col * vertices[j].row;
        area -= vertices[j].col * vertices[i].row;
      }}
      return Math.abs(area) * 0.5 * CELL_SIZE * CELL_SIZE;
    }}

    function syncPanelFromState() {{
      document.getElementById('gridCellSize').value = CELL_SIZE;
      document.getElementById('gridCols').value = GRID_COLS;
      document.getElementById('gridRows').value = GRID_ROWS;
      if (state.terminals.length && (!state.currentTerminalId || !state.terminals.some(t => t.id === state.currentTerminalId)))
        state.currentTerminalId = state.terminals[0].id;
      const term = getCurrentTerminal();
      if (term) {{
        document.getElementById('terminalName').value = term.name || '';
        const floors = term.floors != null ? Math.max(1, parseInt(term.floors, 10) || 1) : 1;
        const f2fRaw = term.floorToFloor != null ? Number(term.floorToFloor) : (term.floorHeight != null ? Number(term.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const totalH = term.floorHeight != null ? Number(term.floorHeight) || (floors * f2f) : (floors * f2f);
        term.floors = floors;
        term.floorToFloor = f2f;
        term.floorHeight = totalH;
        const floorsInput = document.getElementById('terminalFloors');
        const f2fInput = document.getElementById('terminalFloorToFloor');
        const totalInput = document.getElementById('terminalFloorHeight');
        if (floorsInput) floorsInput.value = floors;
        if (f2fInput) f2fInput.value = f2f;
        if (totalInput) totalInput.value = totalH;
        document.getElementById('terminalDepartureCapacity').value = term.departureCapacity != null ? term.departureCapacity : 0;
        document.getElementById('terminalArrivalCapacity').value = term.arrivalCapacity != null ? term.arrivalCapacity : 0;
      }}
      const drawBtn = document.getElementById('btnTerminalDraw');
      if (drawBtn) {{
        if (state.terminalDrawingId) {{
          drawBtn.textContent = 'Drawing...';
        }} else {{
          drawBtn.textContent = 'Draw';
        }}
      }}
      if (state.selectedObject && state.selectedObject.type === 'pbb') {{
        const pbb = state.selectedObject.obj;
        const nameInput = document.getElementById('standName');
        const catSel = document.getElementById('standCategory');
        if (nameInput) nameInput.value = pbb.name || '';
        if (catSel && pbb.category) catSel.value = pbb.category;
      }}
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        const nameInput = document.getElementById('taxiwayName');
        const widthInput = document.getElementById('taxiwayWidth');
        const typeSel = document.getElementById('pathType');
        const exitWrap = document.getElementById('runwayExitExtras');
        const maxExitInput = document.getElementById('taxiwayMaxExitVel');
        const depExtras = document.getElementById('runwayDepPointExtras');
        const depPosInput = document.getElementById('runwayDepPointPos');
        if (nameInput) nameInput.value = tw.name || '';
        const widthDefault = tw.pathType === 'runway' ? 60 : 15;
        if (widthInput) widthInput.value = tw.width != null ? tw.width : widthDefault;
        if (typeSel) typeSel.value = tw.pathType || 'taxiway';
        if (exitWrap) exitWrap.style.display = (tw.pathType === 'runway_exit') ? 'block' : 'none';
        if (maxExitInput) maxExitInput.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : 30;
        const modeSel = document.getElementById('taxiwayDirectionMode');
        if (modeSel) modeSel.value = getTaxiwayDirection(tw);
        // Dep Point 입력칸은 Runway 타입일 때만 보이도록
        if (depExtras) depExtras.style.display = (tw.pathType === 'runway') ? 'block' : 'none';
        if (depPosInput) {{
          if (tw.pathType === 'runway' && typeof tw.depPointPos === 'number') depPosInput.value = tw.depPointPos;
          else depPosInput.value = 0;
        }}
      }} else if (state.selectedObject && state.selectedObject.type === 'runway') {{
        const rw = state.selectedObject.obj;
        const dirSel = document.getElementById('runwayDirection');
        if (dirSel) dirSel.value = rw.direction || 'both';
        const rwWrap = document.getElementById('runwayDirectionWrap');
        const rwDirInPane = document.getElementById('runwayDirectionInTaxiwayPane');
        if (rwWrap) rwWrap.style.display = 'block';
        if (rwDirInPane) rwDirInPane.value = rw.direction || 'both';
      }} else {{
        const rwWrap = document.getElementById('runwayDirectionWrap');
        if (rwWrap) rwWrap.style.display = 'none';
        const exitWrap = document.getElementById('runwayExitExtras');
        if (exitWrap) exitWrap.style.display = 'none';
        const depExtras = document.getElementById('runwayDepPointExtras');
        if (depExtras) depExtras.style.display = 'none';
      }}
      const taxiwayDrawBtn = document.getElementById('btnTaxiwayDraw');
      if (taxiwayDrawBtn) {{
        taxiwayDrawBtn.textContent = state.taxiwayDrawingId ? 'Finish' : 'Draw';
      }}
      const runwayDrawBtn = document.getElementById('btnRunwayDraw');
      if (runwayDrawBtn) {{
        runwayDrawBtn.textContent = state.runwayDrawing ? 'Cancel' : 'Draw';
      }}
      renderObjectList();
    }}

    function syncStateFromPanel() {{
      var el = function(id) {{ return document.getElementById(id); }};
      if (el('gridCellSize')) CELL_SIZE = Math.max(10, Number(el('gridCellSize').value) || 10);
      if (el('gridCols')) GRID_COLS = Math.max(5, Math.min(500, parseInt(el('gridCols').value, 10) || 200));
      if (el('gridRows')) GRID_ROWS = Math.max(5, Math.min(500, parseInt(el('gridRows').value, 10) || 200));
      var t = getCurrentTerminal();
      if (t) {{
        if (el('terminalName')) t.name = (el('terminalName').value || '').trim() || t.name;
        if (el('terminalFloors')) t.floors = Math.max(1, parseInt(el('terminalFloors').value, 10) || 1);
        if (el('terminalFloorToFloor')) t.floorToFloor = Math.max(0.5, Number(el('terminalFloorToFloor').value) || 4);
        t.floorHeight = (t.floors || 1) * (t.floorToFloor || 4);
        if (el('terminalDepartureCapacity')) t.departureCapacity = Math.max(0, parseInt(el('terminalDepartureCapacity').value, 10) || 0);
        if (el('terminalArrivalCapacity')) t.arrivalCapacity = Math.max(0, parseInt(el('terminalArrivalCapacity').value, 10) || 0);
      }}
      if (state.selectedObject && state.selectedObject.type === 'pbb') {{
        var pbb = state.selectedObject.obj;
        if (el('standName')) pbb.name = (el('standName').value || '').trim();
        if (el('standCategory')) pbb.category = el('standCategory').value || 'C';
      }}
      if (state.selectedObject && (state.selectedObject.type === 'taxiway' || state.selectedObject.type === 'runway')) {{
        var tw = state.selectedObject.obj;
        if (el('taxiwayName')) tw.name = (el('taxiwayName').value || '').trim();
        if (el('taxiwayWidth')) tw.width = Math.max(10, Math.min(100, Number(el('taxiwayWidth').value) || 15));
        if (el('pathType')) tw.pathType = el('pathType').value || 'taxiway';
        if (el('taxiwayMaxExitVel')) {{
          const mv = Number(el('taxiwayMaxExitVel').value);
          if (tw.pathType === 'runway_exit') tw.maxExitVelocity = isFinite(mv) && mv > 0 ? mv : null;
          else delete tw.maxExitVelocity;
        }}
        if (el('taxiwayDirectionMode')) tw.direction = el('taxiwayDirectionMode').value || 'both';
        // Runway Dep Point 비율 입력값을 상태에 반영 (Runway 타입일 때만)
        if (tw.pathType === 'runway' && el('runwayDepPointPos')) {{
          const depRaw = Number(el('runwayDepPointPos').value);
          if (isFinite(depRaw)) tw.depPointPos = Math.max(0, Math.min(500, depRaw));
        }}
      }}
      const sel = state.selectedObject;
      if (sel && sel.type === 'runway') {{
        const dirVal = (el('runwayDirectionInTaxiwayPane') && el('runwayDirectionInTaxiwayPane').value) || (el('runwayDirection') && el('runwayDirection').value) || 'both';
        sel.obj.direction = dirVal;
      }}
    }}

    function syncSettingsPaneToMode() {{
      const mode = settingModeSelect ? settingModeSelect.value : 'grid';
      document.querySelectorAll('.settings-pane').forEach(el => {{ el.style.display = 'none'; }});
      const pane = document.getElementById('settings-' + mode);
      if (pane) pane.style.display = 'block';
      if (typeof renderObjectList === 'function') renderObjectList();
    }}

    settingModeSelect.addEventListener('change', function() {{
      syncSettingsPaneToMode();
    }});
    syncSettingsPaneToMode();

    let activeTab = 'settings';
    function switchToTab(tabId) {{
      activeTab = tabId;
      document.querySelectorAll('.right-panel-tab').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      const tabBtn = document.querySelector('.right-panel-tab[data-tab="' + tabId + '"]');
      const tabEl = document.getElementById('tab-' + tabId);
      if (tabBtn) tabBtn.classList.add('active');
      if (tabEl) tabEl.classList.add('active');
      // 패널 폭은 CSS에서 일관되게 관리 (Flight / Allocation / 다른 탭 동일 폭, 접기 기능 정상 동작)
      if (tabId === 'flight') {{
        if (state.selectedObject && state.selectedObject.type === 'flight' && typeof syncFlightPanelFromSelection === 'function') syncFlightPanelFromSelection();
      }}
      if (tabId === 'allocation') {{
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }}
      if (tabId === 'rwysep') {{
        if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
      }}
    }}
    document.querySelectorAll('.right-panel-tab').forEach(btn => {{
      btn.addEventListener('click', function() {{ switchToTab(this.getAttribute('data-tab')); }});
    }});

    // Apron 탭: SLDT-SIBT / SOBT-STOT, EIBT-EOBT 보조 바 표시 토글
    const chkShowSldtBars = document.getElementById('chkShowSldtBars');
    if (chkShowSldtBars) {{
      chkShowSldtBars.addEventListener('change', function() {{
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }});
    }}
    const chkShowEibtBars = document.getElementById('chkShowEibtBars');
    if (chkShowEibtBars) {{
      chkShowEibtBars.addEventListener('change', function() {{
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }});
    }}
    const chkShowEldtBars = document.getElementById('chkShowEldtBars');
    if (chkShowEldtBars) {{
      chkShowEldtBars.addEventListener('change', function() {{
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }});
    }}

    // Runway line-up point 값 변경 시 바로 상태/그리드에 반영
    const depPointPosInput = document.getElementById('runwayDepPointPos');
    if (depPointPosInput) {{
      depPointPosInput.addEventListener('input', function() {{
        let v = Number(this.value);
        if (!isFinite(v)) v = 0;
        v = Math.max(0, Math.min(500, v));
        this.value = v;
        if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
          const tw = state.selectedObject.obj;
          if (tw.pathType === 'runway') {{
            tw.depPointPos = v;
            // 좌표 기반 line-up point가 이미 있으면 depPointPos가 무시되므로, 입력 시 좌표 값을 리셋
            if (tw.lineup_point) delete tw.lineup_point;
            if (tw.dep_point) delete tw.dep_point;
            // dep_point/lineup_point는 저장 시 다시 계산하므로 여기서는 depPointPos만 갱신
            draw();
            if (typeof update3DScene === 'function') update3DScene();
          }}
        }}
      }});
    }}

    document.getElementById('gridCellSize').addEventListener('change', function() {{ CELL_SIZE = Math.max(10, Number(this.value) || 10); draw(); }});
    document.getElementById('gridCols').addEventListener('change', function() {{ GRID_COLS = Math.max(5, Math.min(500, parseInt(this.value,10)||400)); draw(); }});
    document.getElementById('gridRows').addEventListener('change', function() {{ GRID_ROWS = Math.max(5, Math.min(500, parseInt(this.value,10)||400)); draw(); }});

    const pathTypeSelect = document.getElementById('pathType');
    if (pathTypeSelect) {{
      pathTypeSelect.addEventListener('change', function() {{
        const nameInput = document.getElementById('taxiwayName');
        const widthInput = document.getElementById('taxiwayWidth');
        const exitWrap = document.getElementById('runwayExitExtras');
        const maxExitInput = document.getElementById('taxiwayMaxExitVel');
          const depExtras = document.getElementById('runwayDepPointExtras');
        if (!nameInput || !widthInput) return;
        const currentName = nameInput.value.trim();
        const currentWidth = Number(widthInput.value);
        const newType = this.value || 'taxiway';

        // 인풋 필드 기본값 조정
        if (newType === 'runway') {{
          if (!currentName || currentName.startsWith('Taxiway ') || currentName.startsWith('Runway Exit')) {{
            nameInput.value = 'Runway';
          }}
          if (!currentWidth || currentWidth === 15) widthInput.value = 60;
        }} else if (newType === 'runway_exit') {{
          if (!currentName || currentName === 'Runway') {{
            nameInput.value = 'Runway Exit TW';
          }}
          if (!currentWidth) widthInput.value = 15;
        }} else {{
          if (!currentName || currentName === 'Runway' || currentName.startsWith('Runway Exit')) {{
            nameInput.value = '';
          }}
          if (!currentWidth || currentWidth === 60) widthInput.value = 15;
        }}
        if (exitWrap) exitWrap.style.display = (newType === 'runway_exit') ? 'block' : 'none';
          if (depExtras) depExtras.style.display = (newType === 'runway') ? 'block' : 'none';

        // 선택된 Taxiway 객체에도 타입/폭/Max Exit Velocity 반영
        if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
          const tw = state.selectedObject.obj;
          tw.pathType = newType;
          const baseWidth = newType === 'runway' ? 60 : 15;
          const parsed = Number(widthInput.value);
          tw.width = Math.max(10, Math.min(100, parsed || baseWidth));
            if (newType === 'runway') {{
              const depPosInput = document.getElementById('runwayDepPointPos');
              const depVal = depPosInput ? Number(depPosInput.value) : 0;
              const clamped = isFinite(depVal) ? Math.max(0, Math.min(100, depVal)) : 0;
              tw.depPointPos = clamped; // 0~100 %, 0 = start point
            }} else {{
              delete tw.depPointPos;
            }}
          if (newType === 'runway_exit' && maxExitInput) {{
            const mv = Number(maxExitInput.value);
            tw.maxExitVelocity = isFinite(mv) && mv > 0 ? mv : null;
          }} else {{
            delete tw.maxExitVelocity;
          }}
          widthInput.value = tw.width;
          renderObjectList();
          updateObjectInfo();
          draw();
          if (typeof update3DScene === 'function') update3DScene();
        }}
      }});
    }}

    document.getElementById('terminalName').addEventListener('change', function() {{
      const t = getCurrentTerminal();
      if (t) {{
        t.name = this.value;
        draw();
        updateObjectInfo();
        if (typeof renderFlightList === 'function') renderFlightList();
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }}
    }});
    function recomputeTerminalFloorHeight() {{
      const t = getCurrentTerminal();
      if (!t) return;
      const floorsInput = document.getElementById('terminalFloors');
      const f2fInput = document.getElementById('terminalFloorToFloor');
      const totalInput = document.getElementById('terminalFloorHeight');
      let floors = floorsInput ? parseInt(floorsInput.value, 10) : t.floors;
      let f2f = f2fInput ? Number(f2fInput.value) : t.floorToFloor;
      floors = Math.max(1, floors || 1);
      f2f = Math.max(0.5, f2f || 4);
      const totalH = floors * f2f;
      t.floors = floors;
      t.floorToFloor = f2f;
      t.floorHeight = totalH;
      if (floorsInput) floorsInput.value = floors;
      if (f2fInput) f2fInput.value = f2f;
      if (totalInput) totalInput.value = totalH;
      draw();
      updateObjectInfo();
      if (typeof update3DScene === 'function') update3DScene();
    }}
    document.getElementById('terminalFloors').addEventListener('change', recomputeTerminalFloorHeight);
    document.getElementById('terminalFloorToFloor').addEventListener('change', recomputeTerminalFloorHeight);
    document.getElementById('terminalDepartureCapacity').addEventListener('change', function() {{
      const t = getCurrentTerminal();
      if (t) {{ t.departureCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }}
    }});
    document.getElementById('terminalArrivalCapacity').addEventListener('change', function() {{
      const t = getCurrentTerminal();
      if (t) {{ t.arrivalCapacity = Math.max(0, parseInt(this.value, 10) || 0); updateObjectInfo(); }}
    }});

    document.getElementById('standName').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'pbb') {{
        state.selectedObject.obj.name = this.value.trim();
        updateObjectInfo();
        renderObjectList();
        draw();
      }}
    }});
    document.getElementById('standCategory').addEventListener('change', function() {{
      const val = this.value || 'C';
      if (state.selectedObject && state.selectedObject.type === 'pbb') {{
        state.selectedObject.obj.category = val;
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }}
    }});

    document.getElementById('taxiwayName').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        state.selectedObject.obj.name = this.value.trim();
        updateObjectInfo();
        renderObjectList();
        draw();
      }}
    }});
    document.getElementById('taxiwayWidth').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        const baseWidth = tw.pathType === 'runway' ? 60 : 15;
        const val = Number(this.value);
        tw.width = Math.max(10, Math.min(100, val || baseWidth));
        this.value = tw.width;
        updateObjectInfo();
        draw();
        if (scene3d) update3DScene();
      }}
    }});
    document.getElementById('taxiwayDirectionMode').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        tw.direction = this.value || 'both';
        updateObjectInfo();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }}
    }});

    // ---- Flight helpers ----
    function parseTimeToMinutes(val) {{
      if (!val) return 0;
      const s = String(val).trim();
      if (!s) return 0;
      if (s.includes(':')) {{
        const parts = s.split(':');
        const h = parseInt(parts[0], 10) || 0;
        const m = parseInt(parts[1], 10) || 0;
        const sec = (parts.length >= 3) ? (parseInt(parts[2], 10) || 0) : 0;
        return Math.max(0, h * 60 + m + sec / 60);
      }}
      const num = parseFloat(s);
      return isNaN(num) ? 0 : Math.max(0, num);
    }}

    function recomputeSimDuration() {{
      let minT = 0, maxT = 0;
      state.flights.forEach(f => {{
        if (f.timeline && f.timeline.length) {{
          const first = f.timeline[0].t;
          const last = f.timeline[f.timeline.length - 1].t;
          if (minT === 0 || first < minT) minT = first;
          if (last > maxT) maxT = last;
        }}
      }});
      state.simStartSec = minT;
      state.simDurationSec = Math.max(maxT, minT);
      state.simTimeSec = Math.max(state.simStartSec, Math.min(state.simDurationSec, state.simTimeSec));
      const slider = document.getElementById('flightSimSlider');
      const label = document.getElementById('flightSimTimeLabel');
      if (slider) {{
        slider.min = state.simStartSec;
        slider.max = state.simDurationSec;
        slider.value = state.simTimeSec;
        if (state.simDurationSec <= state.simStartSec) slider.disabled = true;
        else slider.disabled = false;
      }}
      if (label) label.textContent = formatSecondsToHHMMSS(state.simTimeSec);
      const simContainer = document.getElementById('sim-controls-container');
      if (simContainer) simContainer.style.display = state.flights.length > 0 ? 'flex' : 'none';
    }}

    function formatMinutesToHHMM(minsRaw) {{
      const totalMin = Math.max(0, Math.floor(minsRaw || 0));
      const h = Math.floor(totalMin / 60);
      const m = totalMin % 60;
      const hh = (h < 10 ? '0' : '') + h;
      const mm = (m < 10 ? '0' : '') + m;
      return hh + ':' + mm;
    }}

    function formatMinutesToHHMMSS(minsRaw) {{
      const totalSec = Math.max(0, Math.round((minsRaw || 0) * 60));
      const h = Math.floor(totalSec / 3600);
      const m = Math.floor((totalSec % 3600) / 60);
      const s = totalSec % 60;
      const hh = (h < 10 ? '0' : '') + h;
      const mm = (m < 10 ? '0' : '') + m;
      const ss = (s < 10 ? '0' : '') + s;
      return hh + ':' + mm + ':' + ss;
    }}

    function formatSecondsToHHMMSS(secRaw) {{
      const totalSec = Math.max(0, Math.floor(secRaw || 0));
      const h = Math.floor(totalSec / 3600);
      const m = Math.floor((totalSec % 3600) / 60);
      const s = totalSec % 60;
      const hh = (h < 10 ? '0' : '') + h;
      const mm = (m < 10 ? '0' : '') + m;
      const ss = (s < 10 ? '0' : '') + s;
      return hh + ':' + mm + ':' + ss;
    }}

    function getStandBusyIntervals(standId, ignoreFlightId) {{
      const intervals = [];
      if (!standId) return intervals;
      (state.flights || []).forEach(f => {{
        if (!f || f.id === ignoreFlightId) return;
        if (f.arrDep !== 'Arr') return;
        if (f.standId !== standId) return;
        if (!f.timeline || !f.timeline.length) return;
        const end = f.timeline[f.timeline.length - 1].t;
        const dwellSec = Math.max(0, (f.dwellMin || 0) * 60);
        const start = Math.max(0, end - dwellSec);
        if (end > start) intervals.push({{ start, end }});
      }});
      intervals.sort((a, b) => a.start - b.start);
      return intervals;
    }}

    function findStandAvailableArrivalTime(standId, desiredArrival, dwellSec) {{
      let s = Math.max(0, desiredArrival);
      const intervals = getStandBusyIntervals(standId, null);
      for (let i = 0; i < intervals.length; i++) {{
        const iv = intervals[i];
        if (s + dwellSec <= iv.start) return s;
        if (s < iv.end) s = iv.end;
      }}
      return s;
    }}

    function assignStandForExistingFlights() {{
      if (!state.flights || !state.flights.length) return;
      if (!state.pbbStands || !state.pbbStands.length) return;
      const thr2 = (CELL_SIZE * 5) ** 2;
      state.flights.forEach(f => {{
        if (!f || f.standId || !f.timeline || !f.timeline.length) return;
        if (f.arrDep !== 'Arr') return;
        const last = f.timeline[f.timeline.length - 1];
        if (!last || typeof last.x !== 'number' || typeof last.y !== 'number') return;
        let bestId = null;
        let bestD2 = thr2;
        state.pbbStands.forEach(pbb => {{
          if (pbb.x2 == null || pbb.y2 == null) return;
          const dx = last.x - pbb.x2;
          const dy = last.y - pbb.y2;
          const d2 = dx*dx + dy*dy;
          if (d2 < bestD2) {{
            bestD2 = d2;
            bestId = pbb.id;
          }}
        }});
        if (bestId) f.standId = bestId;
      }});
    }}

    function findRunwayTouchPoint(runwayId) {{
      if (state.runways && state.runways.length) {{
        const rw = runwayId ? state.runways.find(r => r.id === runwayId) : state.runways[0];
        if (rw) return [rw.cx, rw.cy];
      }}
      const taxiwaysToUse = state.taxiways || [];
      const runwayPath = runwayId
        ? taxiwaysToUse.find(tw => tw.id === runwayId && tw.vertices && tw.vertices.length >= 2)
        : (taxiwaysToUse.find(tw => tw.pathType === 'runway' && tw.vertices && tw.vertices.length >= 2)
           || taxiwaysToUse.find(tw => (tw.name || '').toLowerCase().includes('runway') && tw.vertices && tw.vertices.length >= 2));
      if (runwayPath && runwayPath.vertices) {{
        const verts = runwayPath.vertices;
        const pts = verts.map(v => cellToPixel(v.col, v.row));
        let totalLen = 0;
        for (let i = 1; i < pts.length; i++) totalLen += Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
        if (totalLen > 0) {{
          const target = totalLen / 2;
          let acc = 0;
          for (let i = 1; i < pts.length; i++) {{
            const [x1,y1] = pts[i-1], [x2,y2] = pts[i];
            const segLen = Math.hypot(x2-x1, y2-y1);
            if (acc + segLen >= target) {{
              const t = (target - acc) / (segLen || 1);
              return [x1 + (x2-x1)*t, y1 + (y2-y1)*t];
            }}
            acc += segLen;
          }}
        }}
        return [pts[0][0], pts[0][1]];
      }}
      return null;
    }}

    function getTerminalForStand(stand) {{
      if (!stand || !state.terminals.length) return null;
      const [px, py] = (stand.x2 != null && stand.y2 != null)
        ? [stand.x2, stand.y2]
        : cellToPixel(stand.edgeCol != null ? stand.edgeCol : stand.col, stand.edgeRow != null ? stand.edgeRow : stand.row);
      let nearest = null;
      let nearestD2 = Infinity;
      for (let i = 0; i < state.terminals.length; i++) {{
        const t = state.terminals[i];
        if (!t.vertices || t.vertices.length < 1) continue;
        const termPix = t.vertices.map(v => cellToPixel(v.col, v.row));
        // 1) 먼저 폴리곤 내부인지 체크 (닫힌 터미널만)
        if (t.closed && termPix.length >= 3 && pointInPolygonXY([px, py], termPix)) return t;
        // 2) 아니면 가장 가까운 터미널을 기억해 둔다
        let cx = 0, cy = 0;
        termPix.forEach(p => {{ cx += p[0]; cy += p[1]; }});
        cx /= termPix.length;
        cy /= termPix.length;
        const dx = px - cx, dy = py - cy;
        const d2 = dx*dx + dy*dy;
        if (d2 < nearestD2) {{
          nearestD2 = d2;
          nearest = t;
        }}
      }}
      // 어떤 폴리곤에도 속하지 않으면, 가장 가까운 터미널을 반환
      return nearest;
    }}

    function getCandidateStandsForCode(code) {{
      const list = [];
      state.pbbStands.forEach(pbb => {{
        if (code && pbb.category && pbb.category !== code) return;
        if (state.apronLinks.some(lk => lk.pbbId === pbb.id)) list.push({{ id: pbb.id, name: (pbb.name || '').trim() || 'PBB', type: 'pbb', obj: pbb }});
      }});
      state.remoteStands.forEach(st => {{
        if (code && st.category && st.category !== code) return;
        list.push({{ id: st.id, name: (st.name || '').trim() || 'Remote', type: 'remote', obj: st }});
      }});
      return list;
    }}

    function getCandidatePbbStandsForCode(code) {{
      const list = [];
      state.pbbStands.forEach(pbb => {{
        // 코드가 지정된 경우 카테고리가 같을 때만 사용
        if (code && pbb.category && pbb.category !== code) return;
        const hasLink = state.apronLinks.some(lk => lk.pbbId === pbb.id);
        if (hasLink) list.push(pbb);
      }});
      return list;
    }}

    function pickRandom(arr) {{
      if (!arr.length) return null;
      const idx = Math.floor(Math.random() * arr.length);
      return arr[idx];
    }}

    function buildTaxiwayPathPixels(tw, fromPt, toPt, dirPreference) {{
      if (!tw || !tw.vertices || tw.vertices.length < 2) return null;
      const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
      let srcIdx = 0, dstIdx = pts.length - 1;
      let bestSrc = Infinity, bestDst = Infinity;
      for (let i = 0; i < pts.length; i++) {{
        const dSrc = (pts[i][0]-fromPt[0])**2 + (pts[i][1]-fromPt[1])**2;
        if (dSrc < bestSrc) {{ bestSrc = dSrc; srcIdx = i; }}
        const dDst = (pts[i][0]-toPt[0])**2 + (pts[i][1]-toPt[1])**2;
        if (dDst < bestDst) {{ bestDst = dDst; dstIdx = i; }}
      }}
      if (srcIdx === dstIdx) return [fromPt, pts[srcIdx], toPt];
      const dir = getTaxiwayDirection(tw); // 'clockwise' | 'counter_clockwise' | 'both'
      const needForward = srcIdx <= dstIdx;
      const allowForward = dir === 'both' || dir === 'clockwise';
      const allowBackward = dir === 'both' || dir === 'counter_clockwise';
      if (needForward && !allowForward) return null;
      if (!needForward && !allowBackward) return null;
      const forward = needForward;
      const seq = [];
      seq.push(fromPt);
      if (forward) {{
        for (let i = srcIdx; i <= dstIdx; i++) seq.push(pts[i]);
      }} else {{
        for (let i = srcIdx; i >= dstIdx; i--) seq.push(pts[i]);
      }}
      seq.push(toPt);
      return seq;
    }}

    function buildArrivalTimeline(flight) {{
      const runwayId = (flight.token && flight.token.runwayId) || null;
      const touch = findRunwayTouchPoint(runwayId);
      if (!touch) return null;
      const [rx, ry] = touch;

      // 1) JSON에서 불러온 기존 주기장 배정이 있으면 그대로 사용 (무조건 우선)
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
      let stand = null;
      if (flight.standId) {{
        stand = allStands.find(s => s.id === flight.standId) || null;
      }}

      // 2) 기존 배정이 없을 때만 자동 배정 로직 실행
      if (!stand) {{
        let candidates = getCandidatePbbStandsForCode(flight.code);
        if (!candidates.length) return null;
        if (flight.token && flight.token.terminalId) {{
          const inTerminal = candidates.filter(pbb => {{
            const t = getTerminalForStand(pbb);
            return t && t.id === flight.token.terminalId;
          }});
          if (inTerminal.length) candidates = inTerminal;
        }}
        stand = pickRandom(candidates);
        if (!stand) return null;
        flight.standId = stand.id;
      }}
      const links = state.apronLinks.filter(lk => lk.pbbId === stand.id);
      if (!links.length) return null;
      // 가장 가까운 링크 선택
      let best = links[0];
      let bestD2 = Infinity;
      links.forEach(lk => {{
        if (lk.tx == null || lk.ty == null) return;
        const dx = lk.tx - rx, dy = lk.ty - ry;
        const d2 = dx*dx + dy*dy;
        if (d2 < bestD2) {{ bestD2 = d2; best = lk; }}
      }});
      if (best.tx == null || best.ty == null) return null;
      const [ax, ay] = [stand.x2 != null ? stand.x2 : cellToPixel(stand.col, stand.row)[0],
                        stand.y2 != null ? stand.y2 : cellToPixel(stand.col, stand.row)[1]];
      const tw = state.taxiways.find(t => t.id === best.taxiwayId);
      let pts;
      if (tw) {{
        // Arrival: runway → taxiway → 링크 → stand, taxiway는 설정된 방향을 따르도록 시도
        pts = buildTaxiwayPathPixels(tw, [rx, ry], [best.tx, best.ty], 'forward');
        if (!pts) {{
          // taxiway 경로를 만들 수 없으면 기존 직선 경로로 fallback
          pts = [
            [rx, ry],
            [best.tx, best.ty]
          ];
        }}
        pts.push([ax, ay]);
      }} else {{
        pts = [
          [rx, ry],
          [best.tx, best.ty],
          [ax, ay]
        ];
      }}
      const baseT = flight.timeMin * 60;
      const v = Math.max(1, flight.velocity);
      const timeline = [];
      let tAcc = baseT;
      timeline.push({{ t: tAcc, x: pts[0][0], y: pts[0][1] }});
      for (let i = 1; i < pts.length; i++) {{
        const [x1,y1] = pts[i-1];
        const [x2,y2] = pts[i];
        const len = Math.hypot(x2-x1, y2-y1);
        const dt = len / v;
        tAcc += dt;
        timeline.push({{ t: tAcc, x: x2, y: y2 }});
      }}
      // Dwell time: 마지막 지점 고정
      const dwellSec = Math.max(0, flight.dwellMin) * 60;
      const arrivalTimeAtStand = tAcc;
      if (dwellSec > 0) {{
        tAcc += dwellSec;
        const last = timeline[timeline.length - 1];
        timeline.push({{ t: tAcc, x: last.x, y: last.y }});
      }}
      // Stand 점유 상태를 고려하여 도착 시간이 겹치면 전체 타임라인을 뒤로 밀어낸다.
      if (dwellSec > 0 && flight.standId) {{
        const desired = arrivalTimeAtStand;
        const available = findStandAvailableArrivalTime(flight.standId, desired, dwellSec);
        const delta = Math.max(0, available - desired);
        if (delta > 0) {{
          for (let i = 0; i < timeline.length; i++) {{
            timeline[i].t += delta;
          }}
          // 시간(Time)은 사용자가 바꾸지 않는 한 유지; timeMin 갱신하지 않음
        }}
      }}
      return timeline;
    }}

    function buildDepartureTimeline(flight) {{
      const runwayId = (flight.token && flight.token.runwayId) || null;
      const touch = findRunwayTouchPoint(runwayId);
      if (!touch) return null;
      const [rx, ry] = touch;

      // 1) 기존 standId가 있으면 그대로 사용 (JSON 배정 유지)
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
      let stand = null;
      if (flight.standId) {{
        stand = allStands.find(s => s.id === flight.standId) || null;
      }}

      // 2) 없는 경우에만 자동 후보에서 선택
      if (!stand) {{
        let candidates = getCandidatePbbStandsForCode(flight.code);
        if (!candidates.length) return null;
        if (flight.token && flight.token.terminalId) {{
          const inTerminal = candidates.filter(pbb => {{
            const t = getTerminalForStand(pbb);
            return t && t.id === flight.token.terminalId;
          }});
          if (inTerminal.length) candidates = inTerminal;
        }}
        stand = pickRandom(candidates);
        if (!stand) return null;
        flight.standId = stand.id;
      }}
      const links = state.apronLinks.filter(lk => lk.pbbId === stand.id);
      if (!links.length) return null;
      let best = links[0];
      let bestD2 = Infinity;
      links.forEach(lk => {{
        if (lk.tx == null || lk.ty == null) return;
        const [sx, sy] = [stand.x2 != null ? stand.x2 : cellToPixel(stand.col, stand.row)[0],
                          stand.y2 != null ? stand.y2 : cellToPixel(stand.col, stand.row)[1]];
        const dx = lk.tx - sx, dy = lk.ty - sy;
        const d2 = dx*dx + dy*dy;
        if (d2 < bestD2) {{ bestD2 = d2; best = lk; }}
      }});
      if (best.tx == null || best.ty == null) return null;
      const [sx, sy] = [stand.x2 != null ? stand.x2 : cellToPixel(stand.col, stand.row)[0],
                        stand.y2 != null ? stand.y2 : cellToPixel(stand.col, stand.row)[1]];
      const tw = state.taxiways.find(t => t.id === best.taxiwayId);
      let pts;
      if (tw) {{
        // Departure: stand → 링크 → taxiway → runway, taxiway 방향 반대로 이동 시도
        pts = buildTaxiwayPathPixels(tw, [sx, sy], [best.tx, best.ty], 'backward');
        if (!pts) {{
          pts = [
            [sx, sy],
            [best.tx, best.ty]
          ];
        }}
        pts.push([rx, ry]);
      }} else {{
        pts = [
          [sx, sy],
          [best.tx, best.ty],
          [rx, ry]
        ];
      }}
      const baseT = flight.timeMin * 60;
      const v = Math.max(1, flight.velocity);
      const timeline = [];
      let tAcc = baseT;
      timeline.push({{ t: tAcc, x: pts[0][0], y: pts[0][1] }});
      for (let i = 1; i < pts.length; i++) {{
        const [x1,y1] = pts[i-1];
        const [x2,y2] = pts[i];
        const len = Math.hypot(x2-x1, y2-y1);
        const dt = len / v;
        tAcc += dt;
        timeline.push({{ t: tAcc, x: x2, y: y2 }});
      }}
      return timeline;
    }}

    function getFlightPositionAtTime(flight, tSec) {{
      const tl = flight.timeline;
      if (!tl || !tl.length) return null;
      if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
      for (let i = 0; i < tl.length - 1; i++) {{
        const a = tl[i], b = tl[i+1];
        if (tSec >= a.t && tSec <= b.t) {{
          const span = b.t - a.t || 1;
          const u = (tSec - a.t) / span;
          return {{
            x: a.x + (b.x - a.x) * u,
            y: a.y + (b.y - a.y) * u
          }};
        }}
      }}
      return null;
    }}

    function getFlightPoseAtTime(flight, tSec) {{
      const tl = flight.timeline;
      if (!tl || !tl.length) return null;
      if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
      for (let i = 0; i < tl.length - 1; i++) {{
        const a = tl[i], b = tl[i+1];
        if (tSec >= a.t && tSec <= b.t) {{
          const span = b.t - a.t || 1;
          const u = (tSec - a.t) / span;
          const x = a.x + (b.x - a.x) * u;
          const y = a.y + (b.y - a.y) * u;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          return {{ x, y, dx, dy }};
        }}
      }}
      return null;
    }}

    function buildRunwayOptionsHtml(selectedId) {{
      const opts = [];
      const list = [];
      (state.runways || []).forEach(r => {{
        list.push({{ id: r.id, name: (r.name || '').trim() || 'Runway' }});
      }});
      (state.taxiways || []).filter(t =>
        t.pathType === 'runway' || (t.name || '').toLowerCase().includes('runway')
      ).forEach(t => {{
        list.push({{ id: t.id, name: (t.name || '').trim() || 'Runway' }});
      }});
      if (!list.length) {{
        opts.push('<option value=\"\">Runway</option>');
      }} else {{
        // Random/빈 값 옵션 없이, 실제 Runway 객체만 나열
        list.forEach(o => {{
          const sel = selectedId && o.id === selectedId ? ' selected' : '';
          opts.push('<option value=\"' + String(o.id || '').replace(/\"/g,'&quot;') + '\"' + sel + '>' +
            escapeHtml(o.name || o.id || 'Runway') + '</option>');
        }});
      }}
      return opts.join('');
    }}

    function buildTerminalOptionsHtml(selectedId) {{
      const opts = [];
      // 이름이 중복될 경우 makeUniqueNamedCopy를 사용해 'Pier Apron3 (2)' 형태로 표시
      const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(t => ({{
        id: t.id,
        name: (t.name || '').trim() || 'Terminal'
      }}));
      if (!terms.length) {{
        opts.push('<option value=\"\">Terminal</option>');
      }} else {{
        if (terms.length > 1) opts.push('<option value=\"\">Random</option>');
        terms.forEach(o => {{
          const sel = selectedId && o.id === selectedId ? ' selected' : '';
          opts.push('<option value=\"' + String(o.id || '').replace(/\"/g,'&quot;') + '\"' + sel + '>' +
            escapeHtml(o.name || o.id || 'Terminal') + '</option>');
        }});
      }}
      return opts.join('');
    }}

    function renderFlightList(skipAutoAllocate) {{
      const listEl = document.getElementById('flightList');
      const cfgEl = document.getElementById('flightConfigList');
      if (!listEl) return;
      if (!state.flights.length) {{
        listEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights yet.</div>';
        if (cfgEl) cfgEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights yet.</div>';
        const ganttEl = document.getElementById('allocationGantt');
        if (ganttEl) {{
          ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
        }}
        return;
      }}
      // 정렬은 표시용 복사본만 사용. state.flights 순서는 유지하여 Allocation 바차트/주기장 배치가 경로 갱신 시 바뀌지 않도록 함
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.timeMin != null ? a.timeMin : 0) - (b.timeMin != null ? b.timeMin : 0));
      flightsSorted.forEach(f => {{
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        if (f.noWayArr || f.noWayDep) f.timeline = null;
      }});
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
      const headerRow = '' +
        '<table class="flight-schedule-table">' +
        '<thead><tr>' +
          '<th>Reg</th>' +
          '<th>SLDT</th>' +
          '<th class="flight-td-sibt">SIBT</th>' +
          '<th>SOBT</th>' +
          '<th>STOT</th>' +
          '<th>ELDT</th>' +
          '<th>EIBT</th>' +
          '<th>EOBT</th>' +
          '<th>ETOT</th>' +
          '<th>VTT(Arr)</th>' +
          '<th>VTT(Dep)</th>' +
          '<th>Aircraft Type</th>' +
          '<th>Code(ICAO)</th>' +
          '<th>ICAO(J/H/M/L)</th>' +
          '<th>RECAT-EU(A-F)</th>' +
          '<th>v</th>' +
          '<th>Dwell(S)</th>' +
          '<th>Dwell(E)</th>' +
          '<th>Arr Rw</th>' +
          '<th>Arr RET</th>' +
          '<th>Terminal</th>' +
          '<th>Apron</th>' +
          '<th>Dep Rw</th>' +
          '<th class="flight-td-del"></th>' +
        '</tr></thead>' +
        '<tbody>';
      // Flight configuration 분포(μ, σ)를 aircraft type별로 정리
      const configByType = {{}};
      if (cfgEl) {{
        const seenTypeCfg = new Set();
        flightsSorted.forEach(f => {{
          const acInfo = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
          const typeKey = f.aircraftType || (acInfo && acInfo.id) || (acInfo && acInfo.name) || '';
          if (!typeKey || seenTypeCfg.has(typeKey)) return;
          seenTypeCfg.add(typeKey);
          const tdMu = (typeof acInfo?.touchdown_zone_avg_m === 'number') ? acInfo.touchdown_zone_avg_m : 900;
          const vMu = (typeof acInfo?.touchdown_speed_avg_ms === 'number') ? acInfo.touchdown_speed_avg_ms : 70;
          const aMu = (typeof acInfo?.deceleration_avg_ms2 === 'number') ? acInfo.deceleration_avg_ms2 : 2.5;
          // 초기 σ는 μ의 10% 수준으로 설정
          const tdSigma = Math.round(tdMu * 0.1);
          const vSigma = Math.round(vMu * 0.1);
          const aSigma = Math.round(aMu * 0.1 * 10) / 10;
          configByType[typeKey] = {{ tdMu, tdSigma, vMu, vSigma, aMu, aSigma }};
        }});
      }}
      const retStatsAll = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];

      const dataRows = flightsSorted.map(f => {{
        const tArrMin = f.timeMin != null ? f.timeMin : 0;
        const dwell = f.dwellMin != null ? f.dwellMin : 0;
        const tDepMin = tArrMin + dwell;
        const v = Math.max(1, f.velocity != null ? f.velocity : 15);
        let vttArrMin = 0;
        const arrPts = (typeof getPathForFlight === 'function') ? getPathForFlight(f) : null;
        if (arrPts && arrPts.length >= 2) {{
          let dist = 0;
          for (let i = 0; i < arrPts.length - 1; i++) dist += pathDist(arrPts[i], arrPts[i+1]);
          vttArrMin = dist / v / 60;
        }}
        const sldtMin = Math.max(0, tArrMin - vttArrMin);
        let vttDepMin = 0;
        const depPts = (typeof getPathForFlightDeparture === 'function') ? getPathForFlightDeparture(f) : null;
        if (depPts && depPts.length >= 2) {{
          let dist = 0;
          for (let i = 0; i < depPts.length - 1; i++) dist += pathDist(depPts[i], depPts[i+1]);
          vttDepMin = dist / v / 60;
        }}
        const stotMin = tDepMin + vttDepMin;
        // Flight 객체에 SLDT/STOT 저장하여 다른 탭(Apron Gantt 등에서 직접 참조 가능하게 함)
        f.sldtMin = sldtMin;
        f.stotMin = stotMin;
        // Flight 객체에 ELDT/ETOT 저장 (이미 값이 있는 경우 유지)
        const eldtMin = f.eldtMin != null ? f.eldtMin : sldtMin;
        const etotMin = f.etotMin != null ? f.etotMin : stotMin;
        f.eldtMin = eldtMin;
        f.etotMin = etotMin;

        const tArr = formatMinutesToHHMMSS(tArrMin);
        const tDep = formatMinutesToHHMMSS(tDepMin);
        const sldtStr = formatMinutesToHHMMSS(sldtMin);
        const stotStr = formatMinutesToHHMMSS(stotMin);
        const eibtMin = eldtMin + vttArrMin;
        const eobtMin = etotMin - vttDepMin;
        // Flight 객체에 EIBT/EOBT 저장 (Apron Gantt 등에서 직접 참조)
        f.eibtMin = eibtMin;
        f.eobtMin = eobtMin;

        const eldtStr = formatMinutesToHHMMSS(eldtMin);
        const etotStr = formatMinutesToHHMMSS(etotMin);
        const eibtStr = formatMinutesToHHMMSS(eibtMin);
        const eobtStr = formatMinutesToHHMMSS(eobtMin);
        const dwellS = dwell;
        const dwellE = Math.max(0, eobtMin - eibtMin);
        const vttArrStr = formatMinutesToHHMMSS(vttArrMin);
        const vttDepStr = formatMinutesToHHMMSS(vttDepMin);
        // Arr Runway는 Flight schedule에서 사용자가 선택한 값(또는 token의 runwayId)만 사용
        const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
        const arrOpt = buildRunwayOptionsHtml(arrRunwayId);
        const termOpt = buildTerminalOptionsHtml(f.terminalId || (f.token && f.token.terminalId));
        const depOpt = buildRunwayOptionsHtml(f.depRunwayId || (f.token && f.token.depRunwayId));
        const noWayBadge = (f.noWayArr || f.noWayDep) ? ' <span style="color:#dc2626;font-weight:600;font-size:10px;">⚠ No Way</span>' : '';
        const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
        const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
        const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
        const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
        const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';

        // --- RET 샘플링: Flight configuration 값(μ, σ)을 사용해 TD/VTD/a를 샘플링하고
        // 각 Runway Exit(RET)에 대해 감속 후 속도가 maxExitVelocity 이하가 되는 최초 RET 선택 ---
        let sampledRetName = '—';
        let sampledRetId = null;
        let retCandidateCount = 0;
        if (retStatsAll && retStatsAll.length && arrRunwayId != null) {{
          const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
          const cfg = typeKey ? configByType[typeKey] : null;
          if (cfg) {{
            // RET 선택은 Flight Configuration에 입력된 μ, σ를 모두 사용하여
            // 각 항공기에 대해 한 번씩 샘플링한 TD / VTD / a 값으로 결정하되,
            // 샘플링 값은 각 mean의 ±20% 범위 안으로 클리핑한다.
            const tdSample = sampleNormal(cfg.tdMu, cfg.tdSigma);
            const tdMin = cfg.tdMu * 0.8;
            const tdMax = cfg.tdMu * 1.2;
            const dTd = clamp(tdSample, Math.max(0, tdMin), Math.max(0, tdMax)); // m

            const vSample = sampleNormal(cfg.vMu, cfg.vSigma);
            const vMin = cfg.vMu * 0.8;
            const vMax = cfg.vMu * 1.2;
            const v0 = clamp(vSample, Math.max(0, vMin), Math.max(0, vMax)); // m/s

            const aSample = sampleNormal(cfg.aMu, cfg.aSigma);
            const aMin = Math.max(0.1, cfg.aMu * 0.8);
            const aMax = Math.min(6,   cfg.aMu * 1.2);
            const aDec = clamp(aSample, aMin, aMax); // m/s²
            const candidates = retStatsAll.filter(r => r.runway && r.runway.id === arrRunwayId);
            retCandidateCount = candidates.length;
            if (candidates.length) {{
              let chosen = null;
              candidates.forEach(r => {{
                if (chosen) return;
                const distFromTd = Math.max(0, r.distM - dTd);
                // v^2 = v0^2 - 2 a d
                const vAt = Math.sqrt(Math.max(0, v0*v0 - 2*aDec*distFromTd));
                if (vAt <= r.maxExitVelocity) {{
                  chosen = r;
                }}
              }});
              // 어떤 RET에서도 maxExitVelocity 조건을 만족하지 못하면 실패로 처리 (마지막 RET를 강제로 선택하지 않음)
              if (chosen) {{
                sampledRetName = chosen.name || 'RET';
                sampledRetId = chosen.exit && chosen.exit.id || null;
                f.sampledArrRet = sampledRetId;
              }} else {{
                sampledRetName = 'Failed';
                sampledRetId = null;
                f.sampledArrRet = null;
              }}
            }}
          }}
        }}

        // #region agent log (RET sampling per flight)
        try {{
          _logNoWay({{
            location: 'Layout_Design.js:renderFlightList',
            message: 'RET sampling decision',
            hypothesisId: 'H1_H2_H3_H4',
            runId: 'ret-debug-1',
            data: {{
              flightId: f.id,
              reg: f.reg || '',
              aircraftType: f.aircraftType || '',
              hasConfig: !!(f.aircraftType && configByType[f.aircraftType]),
              arrRunwayId: arrRunwayId || null,
              retCandidateCount,
              sampledRetName,
              sampledRetId
            }}
          }});
        }} catch (e) {{}}
        // #endregion
        return '' +
          '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
            '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + noWayBadge + '</td>' +
            '<td class="flight-td-time">' + sldtStr + '</td>' +
            '<td class="flight-td-time flight-td-sibt">' + tArr + '</td>' +
            '<td class="flight-td-time">' + tDep + '</td>' +
            '<td class="flight-td-time">' + stotStr + '</td>' +
            '<td class="flight-td-time">' + eldtStr + '</td>' +
            '<td class="flight-td-time">' + eibtStr + '</td>' +
            '<td class="flight-td-time">' + eobtStr + '</td>' +
            '<td class="flight-td-time">' + etotStr + '</td>' +
            '<td class="flight-td-time">' + vttArrStr + '</td>' +
            '<td class="flight-td-time">' + vttDepStr + '</td>' +
            '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
            '<td>' + escapeHtml(codeIcao) + '</td>' +
            '<td>' + escapeHtml(icaoJhl) + '</td>' +
            '<td>' + escapeHtml(recatEu) + '</td>' +
            '<td>' + (f.velocity != null ? f.velocity : '—') + '</td>' +
            '<td>' + (dwellS != null ? dwellS : 0) + '</td>' +
            '<td>' + (typeof dwellE === 'number' ? Math.round(dwellE * 10) / 10 : '—') + '</td>' +
            '<td class="flight-td-select"><select class="flight-assign-select" data-role="arr" data-id="' + f.id + '">' + arrOpt + '</select></td>' +
            '<td>' + escapeHtml(sampledRetName) + '</td>' +
            '<td class="flight-td-select"><select class="flight-assign-select" data-role="term" data-id="' + f.id + '">' + termOpt + '</select></td>' +
            '<td class="flight-td-reg">' + (function() {{ const st = (state.pbbStands || []).find(s => s.id === f.standId) || (state.remoteStands || []).find(s => s.id === f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); }})() + '</td>' +
            '<td class="flight-td-select"><select class="flight-assign-select" data-role="dep" data-id="' + f.id + '">' + depOpt + '</select></td>' +
            '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
          '</tr>';
      }});
      listEl.innerHTML = headerRow + dataRows.join('') + '</tbody></table>';
      // Flight Configuration 탭: Aircraft type 단위의 설정 테이블 (입력 UI만, 다른 로직과는 미연결)
      if (cfgEl) {{
        const seenType = new Set();
        const unique = [];
        flightsSorted.forEach(f => {{
          const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
          const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
          if (!typeKey || seenType.has(typeKey)) return;
          seenType.add(typeKey);
          unique.push({{
            key: typeKey,
            label: ac ? (ac.name || ac.id || typeKey) : typeKey
          }});
        }});
        if (!unique.length) {{
          cfgEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights yet.</div>';
        }} else {{
          // 기존 테이블이 있으면, 사용자가 수정한 값을 우선 읽어와서 유지
          const prevConfigByType = {{}};
          const prevInputs = cfgEl.querySelectorAll('.flight-config-input[data-ac][data-param]');
          prevInputs.forEach(inp => {{
            const acKey = inp.getAttribute('data-ac');
            const param = inp.getAttribute('data-param');
            if (!acKey || !param) return;
            const valNum = Number(inp.value);
            if (!isFinite(valNum)) return;
            if (!prevConfigByType[acKey]) prevConfigByType[acKey] = {{}};
            prevConfigByType[acKey][param] = valNum;
          }});
          const headerCols = unique.map(info =>
            '<th>' + escapeHtml(info.label) + '</th>'
          ).join('');
          const cfgHeader = '' +
            '<div style="font-size:10px;color:#9ca3af;margin-bottom:4px;">' +
              'Landing configuration per aircraft type (unit and statistic: mean μ / spread σ).' +
            '</div>' +
            '<table class="flight-schedule-table flight-config-table">' +
            '<thead><tr>' +
              '<th class="sticky-col">Parameter</th>' +
              '<th>Unit</th>' +
              '<th>Stat</th>' +
              headerCols +
            '</tr></thead><tbody>';
          const rows = [];
          // per-aircraft 기본값: Information.json의 static 값 사용하되, 기존 테이블에서 사용자가 수정한 값이 있으면 우선 사용
          const tdMeans = unique.map(info => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-mean'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const ac = getAircraftInfoByType(acKey) || {{}};
            return (typeof ac.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
          }});
          const vtdMeans = unique.map(info => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-mean'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const ac = getAircraftInfoByType(acKey) || {{}};
            return (typeof ac.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
          }});
          const aMeans = unique.map(info => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-mean'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const ac = getAircraftInfoByType(acKey) || {{}};
            return (typeof ac.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
          }});
          // σ는 μ의 10% 수준으로 설정 (사용자가 수정 가능), 기존 테이블에서 수정된 값이면 우선 사용
          const tdSigmas = unique.map((info, idx) => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-sigma'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const v = tdMeans[idx];
            return Math.round(v * 0.1);
          }});
          const vtdSigmas = unique.map((info, idx) => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-sigma'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const v = vtdMeans[idx];
            return Math.round(v * 0.1);
          }});
          const aSigmas = unique.map((info, idx) => {{
            const acKey = info.key;
            const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-sigma'];
            if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
            const v = aMeans[idx];
            return Math.round(v * 0.1 * 10) / 10;
          }});
          // Deceleration a 및 VTD의 mean 값으로부터, 26 m/s 에 도달하는 위치 (threshold로부터 거리, m)
          const vTarget = 26;
          const aMeanStopDists = aMeans.map((aMu, idx) => {{
            const vMu = vtdMeans[idx];
            const tdMu = tdMeans[idx];
            if (!(aMu > 0) || !(vMu > vTarget)) return Math.max(0, Math.round(tdMu || 0));
            const dFromTouchdown = (vMu*vMu - vTarget*vTarget) / (2 * aMu);
            const dTotal = (tdMu || 0) + (dFromTouchdown > 0 ? dFromTouchdown : 0);
            return dTotal > 0 ? Math.round(dTotal) : 0;
          }});

          // RET 샘플링용 configByType:
          // 최초에는 Information.json에서 시작하지만, 이후에는 사용자가 UI에서 수정한 tdMeans/vtdMeans/aMeans 및 시그마를 그대로 사용
          unique.forEach((info, idx) => {{
            const key = info.key;
            configByType[key] = {{
              tdMu: tdMeans[idx],
              tdSigma: tdSigmas[idx],
              vMu: vtdMeans[idx],
              vSigma: vtdSigmas[idx],
              aMu: aMeans[idx],
              aSigma: aSigmas[idx]
            }};
          }});

          rows.push(
            '<tr>' +
              '<td class="sticky-col">Touchdown zone distance from threshold</td>' +
              '<td>m</td>' +
              '<td>mean μ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-mean" type="number" min="0" max="10000" step="10" value="' + tdMeans[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          rows.push(
            '<tr>' +
              '<td class="sticky-col"></td>' +
              '<td>m</td>' +
              '<td>spread σ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-sigma" type="number" min="0" max="10000" step="10" value="' + tdSigmas[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          rows.push(
            '<tr>' +
              '<td class="sticky-col">Touchdown speed VTD</td>' +
              '<td>m/s</td>' +
              '<td>mean μ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-mean" type="number" min="0" max="150" step="1" value="' + vtdMeans[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          rows.push(
            '<tr>' +
              '<td class="sticky-col"></td>' +
              '<td>m/s</td>' +
              '<td>spread σ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-sigma" type="number" min="0" max="150" step="1" value="' + vtdSigmas[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          rows.push(
            '<tr>' +
              '<td class="sticky-col">Deceleration a</td>' +
              '<td>m/s²</td>' +
              '<td>mean μ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-mean" type="number" min="0" max="10" step="0.1" value="' + aMeans[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          rows.push(
            '<tr>' +
              '<td class="sticky-col"></td>' +
              '<td>m/s²</td>' +
              '<td>spread σ</td>' +
              unique.map((info, idx) =>
                '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-sigma" type="number" min="0" max="10" step="0.1" value="' + aSigmas[idx] + '" /></td>'
              ).join('') +
            '</tr>'
          );
          // Deceleration a: mean(μ) 값만 사용했을 때 26 m/s 에 도달하는 거리 (threshold 기준, 읽기 전용)
          rows.push(
            '<tr>' +
              '<td class="sticky-col" style="background:rgba(22,163,74,0.14);">Distance to 26 m/s (from threshold)</td>' +
              '<td style="background:rgba(22,163,74,0.14);">m</td>' +
              '<td style="background:rgba(22,163,74,0.14);">mean-based</td>' +
              unique.map((info, idx) =>
                '<td style="background:rgba(22,163,74,0.14);font-weight:600;color:#bbf7d0;">' + aMeanStopDists[idx] + '</td>'
              ).join('') +
            '</tr>'
          );
          // Runway Exit Taxiway (RET) 위치 표: threshold로부터의 거리 + Flight schedule에서 선택된 RET별 기종 카운트
          const retStats = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
          if (retStats && retStats.length) {{
            // 섹션 헤더 행
            rows.push(
              '<tr>' +
                '<td class="sticky-col" style="padding-top:10px;">Runway exits (distance from threshold)</td>' +
                '<td></td>' +
                '<td></td>' +
                unique.map(() => '<td></td>').join('') +
              '</tr>'
            );
            retStats.forEach((r, idx) => {{
              const rwLabel = r.runway && (r.runway.name || ('Runway ' + (idx + 1)));
              // 각 RET에 대해, Flight schedule에서 선택된 RET별 기종 카운트 계산
              const counts = unique.map(info => {{
                const typeKey = info.key;
                return (state.flights || []).filter(f =>
                  f.sampledArrRet === (r.exit && r.exit.id) &&
                  (f.aircraftType || '') === typeKey
                ).length;
              }});
              // 가장 많은 기종(1위), 2~3위 색상 구분
              const sortedIdx = counts
                .map((c, i) => [c, i])
                .filter(([c]) => c > 0)
                .sort((a, b) => b[0] - a[0]);
              const top1 = sortedIdx[0] ? sortedIdx[0][1] : -1;
              const top2 = sortedIdx[1] ? sortedIdx[1][1] : -1;
              const top3 = sortedIdx[2] ? sortedIdx[2][1] : -1;
              rows.push(
                '<tr>' +
                  '<td class="sticky-col">' +
                    '<span style="display:inline-flex;align-items:center;gap:4px;">' +
                      (rwLabel ? ('<span style="font-size:9px;color:#9ca3af;">' + escapeHtml(rwLabel) + '</span>') : '') +
                      '<span style="padding:2px 6px;border-radius:9999px;background:rgba(15,23,42,0.95);border:1px solid #4b5563;font-size:10px;color:#f9fafb;font-weight:600;">' +
                        escapeHtml(r.name) +
                      '</span>' +
                    '</span>' +
                  '</td>' +
                  '<td>m</td>' +
                  '<td>' + Math.round(r.distM) + '</td>' +
                  unique.map((info, colIdx) => {{
                    const cnt = counts[colIdx] || 0;
                    if (!cnt) return '<td></td>';
                    let bg = 'rgba(15,23,42,0.9)';
                    let color = '#e5e7eb';
                    if (colIdx === top1) {{
                      bg = 'rgba(22,163,74,0.35)'; // 1위: 녹색
                      color = '#bbf7d0';
                    }} else if (colIdx === top2 || colIdx === top3) {{
                      bg = 'rgba(251,146,60,0.3)'; // 2,3위: 주황색
                      color = '#fed7aa';
                    }}
                    return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
                  }}).join('') +
                '</tr>'
              );
            }});
            // RET로 탈출할 수 없었던 항공편(조건 미충족)을 별도 Failed 행으로 요약
            const failedCounts = unique.map(info => {{
              const typeKey = info.key;
              return (state.flights || []).filter(f =>
                (f.sampledArrRet === null || typeof f.sampledArrRet === 'undefined') &&
                (f.aircraftType || '') === typeKey
              ).length;
            }});
            const anyFailed = failedCounts.some(c => c > 0);
            if (anyFailed) {{
              const sortedFailed = failedCounts
                .map((c, i) => [c, i])
                .filter(([c]) => c > 0)
                .sort((a, b) => b[0] - a[0]);
              const fTop1 = sortedFailed[0] ? sortedFailed[0][1] : -1;
              const fTop2 = sortedFailed[1] ? sortedFailed[1][1] : -1;
              const fTop3 = sortedFailed[2] ? sortedFailed[2][1] : -1;
              rows.push(
                '<tr>' +
                  '<td class="sticky-col">' +
                    '<span style="padding:2px 6px;border-radius:9999px;background:rgba(127,29,29,0.9);border:1px solid #b91c1c;font-size:10px;color:#fee2e2;font-weight:600;">Failed</span>' +
                  '</td>' +
                  '<td></td>' +
                  '<td></td>' +
                  unique.map((info, colIdx) => {{
                    const cnt = failedCounts[colIdx] || 0;
                    if (!cnt) return '<td></td>';
                    let bg = 'rgba(30,30,30,0.9)';
                    let color = '#fecaca';
                    if (colIdx === fTop1) {{
                      bg = 'rgba(220,38,38,0.65)'; // 1위 실패: 진한 빨강
                      color = '#fee2e2';
                    }} else if (colIdx === fTop2 || colIdx === fTop3) {{
                      bg = 'rgba(239,68,68,0.45)'; // 2,3위 실패: 연한 빨강
                      color = '#fee2e2';
                    }}
                    return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
                  }}).join('') +
                '</tr>'
              );
            }}
          }}
          cfgEl.innerHTML = cfgHeader + rows.join('') + '</tbody></table>' +
            '<div style="font-size:10px;color:#6b7280;margin-top:4px;">' +
              'Note: sampling is clipped to stay within ±20% of each mean value.' +
            '</div>';
        }}
      }}
      listEl.querySelectorAll('.obj-item-delete').forEach(btn => {{
        btn.addEventListener('click', function(ev) {{
          const idVal = this.getAttribute('data-del');
          state.flights = state.flights.filter(f => f.id !== idVal);
          recomputeSimDuration();
          renderFlightList();
        }});
      }});
      // 클릭해서 해당 Flight 선택
      listEl.querySelectorAll('.obj-item').forEach(row => {{
        row.addEventListener('click', function(ev) {{
          if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
          const idVal = this.getAttribute('data-id');
          const f = state.flights.find(x => x.id === idVal);
          if (!f) return;
          state.selectedObject = {{ type: 'flight', id: idVal, obj: f }};
          // 선택 표시
          listEl.querySelectorAll('.obj-item').forEach(r => r.classList.remove('selected', 'expanded'));
          this.classList.add('selected', 'expanded');
          if (typeof updateObjectInfo === 'function') updateObjectInfo();
          if (typeof syncPanelFromState === 'function') syncPanelFromState();
          if (typeof draw === 'function') draw();
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }});
      }});
      // Arr Rw / Terminal / Dep Rw 선택 핸들러
      listEl.querySelectorAll('.flight-assign-select').forEach(sel => {{
        sel.addEventListener('change', function() {{
          const idVal = this.getAttribute('data-id');
          const role = this.getAttribute('data-role');
          const f = state.flights.find(x => x.id === idVal);
          if (!f) return;
          const val = this.value || null;
          if (!f.token) f.token = {{ nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null }};
          if (role === 'arr') {{
            f.arrRunwayId = val;
            f.token.runwayId = val;
          }} else if (role === 'term') {{
            f.terminalId = val;
            f.token.terminalId = val;
            // 선택된 터미널과 맞지 않는 Stand에 이미 배정되어 있으면 Unassigned로 이동
            if (f.standId) {{
              const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
              const st = allStands.find(s => s.id === f.standId);
              if (st) {{
                const term = getTerminalForStand(st);
                const standTermId = term ? term.id : null;
                if (!val || !standTermId || val !== standTermId) {{
                  f.standId = null;
                }}
              }}
            }}
          }} else if (role === 'dep') {{
            f.depRunwayId = val;
            f.token.depRunwayId = val;
          }}
          // Arr Rw / Terminal / Dep Rw 변경 후, RET 샘플링과 타임라인·Gantt를 모두 다시 계산
          if (typeof renderFlightList === 'function') renderFlightList();
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }});
      }});
      if (typeof renderFlightGantt === 'function') renderFlightGantt();
    }}

    // Allocation Gantt (세로: Apron/Stand, 가로: 시간)
    function renderFlightGantt() {{
      const ganttEl = document.getElementById('allocationGantt');
      if (!ganttEl) return;
      if (!state.flights.length) {{
        ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
        return;
      }}
      const flights = state.flights.slice();
      const stands = (state.pbbStands || []).concat(state.remoteStands || []);
      const standById = {{}};
      stands.forEach(s => {{ standById[s.id] = s; }});
      if (!flights.length) {{
        ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
        return;
      }}
      const intervals = flights.map(f => {{
        // 바차트: 스케줄 표의 SIBT/SOBT 기준
        const t0 = f.timeMin != null ? f.timeMin : 0;               // SIBT / IBT
        const dwell = f.dwellMin != null ? f.dwellMin : 0;
        const t1 = t0 + dwell;                                      // SOBT / OBT
        const sldt = f.sldtMin != null ? f.sldtMin : t0;            // SLDT from schedule
        const stot = f.stotMin != null ? f.stotMin : t1;            // STOT from schedule
        const eibt = f.eibtMin != null ? f.eibtMin : t0;            // EIBT from schedule
        const eobt = f.eobtMin != null ? f.eobtMin : t1;            // EOBT from schedule
        const eldt = f.eldtMin != null ? f.eldtMin : sldt;          // ELDT from schedule
        const etot = f.etotMin != null ? f.etotMin : stot;          // ETOT from schedule
        return {{ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot }};
      }});

      // 공통 시간축: Flight Schedule의 min(SLDT) - 20분, max(ETOT) + 20분
      let minS = Infinity;
      let maxE = -Infinity;
      intervals.forEach(it => {{
        if (it.sldt < minS) minS = it.sldt;
        const etot0 = (it.f && it.f.etotMin != null) ? it.f.etotMin : it.stot;
        if (etot0 > maxE) maxE = etot0;
      }});
      if (!isFinite(minS) || !isFinite(maxE)) {{
        ganttEl.innerHTML = '';
        return;
      }}
      // 기본 전체 범위 (zoom의 최소 축소 한계)
      const baseMinT = Math.max(0, minS - 20);
      const baseMaxT0 = maxE + 20;
      const baseMaxT = (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0;
      const baseSpan = baseMaxT - baseMinT;
      const zoom = (state.allocTimeZoom && state.allocTimeZoom > 1) ? state.allocTimeZoom : 1;
      const span = baseSpan;
      const minT = baseMinT;
      const maxT = baseMaxT;

      function formatMinToHM(m) {{
        const hh = Math.floor(m / 60);
        const mm = Math.floor(m % 60);
        return hh + ':' + (mm < 10 ? '0' : '') + mm;
      }}
      function flightCanUseStand(f, stand) {{
        if (!stand) return true;
        const order = {{ A:1,B:2,C:3,D:4,E:5,F:6 }};
        const fCode = (f.code || 'C').toUpperCase();
        const sCat = (stand.category || 'F').toUpperCase();
        const fc = order[fCode] || 99;
        const sc = order[sCat] || 0;
        if (fc > sc) return false;
        // 터미널 호환성: Flight에 선택된 터미널이 있으면, 해당 터미널에 속한 Stand에만 배정
        const ft = (f.terminalId || (f.token && f.token.terminalId)) || null;
        if (!ft) return true;
        const term = getTerminalForStand(stand);
        const standTermId = term ? term.id : null;
        if (!standTermId) return false;
        return ft === standTermId;
      }}
      // Allocation/바차트/주기장 배치는 사용자가 명시적으로 변경할 때만 갱신. 택시웨이 등 경로 갱신 시 자동 재배정하지 않음

      // 시간축 눈금 위치 (Apron 전체 공통) - 어떤 화면에서도 최대 6개까지만 표시
      const tickPositions = [];
      const axisStep = span <= 60 ? 10 : (span <= 240 ? 30 : 60); // minutes
      let tt = Math.floor(minT / axisStep) * axisStep;
      while (tt <= maxT) {{
        const leftPct = ((tt - baseMinT) / baseSpan) * 100 * zoom;
        const label = formatMinToHM(tt);
        tickPositions.push({{ leftPct, label }});
        tt += axisStep;
      }}
      if (tickPositions.length > 6) {{
        const stepTicks = Math.ceil(tickPositions.length / 6);
        const reduced = [];
        for (let i = 0; i < tickPositions.length; i += stepTicks) {{
          reduced.push(tickPositions[i]);
        }}
        const last = tickPositions[tickPositions.length - 1];
        if (reduced[reduced.length - 1] !== last) reduced.push(last);
        tickPositions.length = 0;
        Array.prototype.push.apply(tickPositions, reduced);
      }}

      function buildRowHtml(label, standId) {{
        const showAuxBarsEl = document.getElementById('chkShowSldtBars');
        const showAuxBars = !showAuxBarsEl || showAuxBarsEl.checked;
        const showEibtBarsEl = document.getElementById('chkShowEibtBars');
        const showEibtBars = !showEibtBarsEl || showEibtBarsEl.checked;
        const showEldtBarsEl = document.getElementById('chkShowEldtBars');
        const showEldtBars = !showEldtBarsEl || showEldtBarsEl.checked;
        const rowFlights = intervals.filter(it => {{
          const f = it.f;
          const sid = (f.standId || null);
          return (standId == null) ? !sid : sid === standId;
        }});
        // 동일 Apron/Stand 안에서 겹치는 구간이 있는 Flight를 conflict로 표시
        const conflictMap = {{}};
        for (let i = 0; i < rowFlights.length; i++) {{
          for (let j = i + 1; j < rowFlights.length; j++) {{
            const a = rowFlights[i];
            const b = rowFlights[j];
            if (a.t0 < b.t1 && b.t0 < a.t1) {{ // 구간 겹침
              conflictMap[a.f.id] = true;
              conflictMap[b.f.id] = true;
            }}
          }}
        }}
        const sBars = showAuxBars ? [] : null;
        const eBars = showEibtBars ? [] : null;
        const e2Bars = showEldtBars ? [] : null;
        const blocks = rowFlights.map(it => {{
          const f = it.f;
          const t0 = it.t0;
          const t1 = it.t1;
          const sldt = it.sldt;
          const stot = it.stot;
          const eibt = it.eibt;
          const eobt = it.eobt;
          const eldt = it.eldt;
          const etot = it.etot;
          const tStart = Math.max(t0, baseMinT);
          const tEnd = Math.min(t1, baseMaxT);
          if (tEnd <= tStart) return '';
          const leftPct = ((tStart - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct = Math.max(2, ((tEnd - tStart) / baseSpan) * 100 * zoom);
          const regSafe = escapeHtml(f.reg || '');
          const codeSafe = escapeHtml((f.code || '').toUpperCase());
          const dwellVal = f.dwellMin != null ? f.dwellMin : 0;
          const dwellLabel = dwellVal ? (dwellVal + 'm') : '';
          let meta = '';
          if (codeSafe && dwellLabel) meta = codeSafe + ' · ' + dwellLabel;
          else if (codeSafe) meta = codeSafe;
          else meta = dwellLabel;
          const conflictClass = conflictMap[f.id] ? ' conflict' : '';
          const selectedClass = (state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id) ? ' alloc-flight-selected' : '';
          const noWayLabel = (f.noWayArr || f.noWayDep) ? ' <span style="color:#fff;font-size:9px;">No way</span>' : '';
          const sibtLabel = formatMinToHM(t0);
          const sobtLabel = formatMinToHM(t1);
          const barTitle =
            'SIBT: ' + sibtLabel +
            '\\nSOBT: ' + sobtLabel +
            '\\nReg: ' + (f.reg || '');
          if (showEibtBars && eBars && isFinite(eibt) && isFinite(eobt) && eobt > eibt) {{
            const eStart = Math.max(eibt, baseMinT);
            const eEnd = Math.min(eobt, baseMaxT);
            if (eEnd > eStart) {{
              const eLeft = ((eStart - baseMinT) / baseSpan) * 100 * zoom;
              const eWidth = ((eEnd - eStart) / baseSpan) * 100 * zoom;
            eBars.push(
              '<div class="alloc-e-bar" style="left:' +
                eLeft +
                '%;width:' +
                Math.max(2, eWidth) +
                '%;"></div>'
            );
            }}
          }}
          if (showEldtBars && e2Bars) {{
            // ELDT~EIBT (pre-block, 아래쪽 얇은 핫핑크)
            if (isFinite(eldt) && isFinite(eibt) && eibt >= eldt) {{
              const s1 = Math.max(eldt, baseMinT);
              const s2 = Math.min(eibt, baseMaxT);
              if (s2 > s1) {{
                const preLeft = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
                const preWidth = ((s2 - s1) / baseSpan) * 100 * zoom;
              e2Bars.push(
                '<div class="alloc-e2-bar" style="left:' +
                  preLeft +
                  '%;width:' +
                  Math.max(0.5, preWidth) +
                  '%;bottom:4px;"></div>'
              );
              }}
            }}
            // EOBT~ETOT (post-block, 위쪽 얇은 핫핑크)
            if (isFinite(eobt) && isFinite(etot) && etot >= eobt) {{
              const s1 = Math.max(eobt, baseMinT);
              const s2 = Math.min(etot, baseMaxT);
              if (s2 > s1) {{
                const postLeft = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
                const postWidth = ((s2 - s1) / baseSpan) * 100 * zoom;
              e2Bars.push(
                '<div class="alloc-e2-bar" style="left:' +
                  postLeft +
                  '%;width:' +
                  Math.max(0.5, postWidth) +
                  '%;top:4px;"></div>'
              );
              }}
            }}
          }}
          if (showAuxBars && sBars) {{
            // SLDT~SIBT (pre-block) 보조 바
            if (isFinite(sldt) && sldt <= t0) {{
              const s1 = Math.max(sldt, baseMinT);
              const s2 = Math.min(t0, baseMaxT);
              if (s2 > s1) {{
                const preLeft = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
                const preWidth = ((s2 - s1) / baseSpan) * 100 * zoom;
              // SLDT~SIBT: 메인 바의 아래쪽에 붙여 표시
              sBars.push(
                '<div class="alloc-s-bar" style="left:' +
                  preLeft +
                  '%;width:' +
                  Math.max(0.5, preWidth) +
                  '%;bottom:4px;"></div>'
              );
              }}
            }}
            // SOBT~STOT (post-block) 보조 바: 메인 바 위쪽에 붙여 표시
            if (isFinite(stot) && stot >= t1) {{
              const s1 = Math.max(t1, baseMinT);
              const s2 = Math.min(stot, baseMaxT);
              if (s2 > s1) {{
                const postLeft = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
                const postWidth = ((s2 - s1) / baseSpan) * 100 * zoom;
              sBars.push(
                '<div class="alloc-s-bar" style="left:' +
                  postLeft +
                  '%;width:' +
                  Math.max(0.5, postWidth) +
                  '%;top:4px;"></div>'
              );
              }}
            }}
          }}
          return '' +
            '<div class="alloc-flight' + conflictClass + selectedClass + '" draggable="true" data-flight-id="' + f.id + '" ' +
              'style="left:' + leftPct + '%;width:' + widthPct + '%;min-width:4px;"' +
              ' title="' + barTitle + '">' +
              '<div class="alloc-flight-reg">' + regSafe + noWayLabel + '</div>' +
              '<div class="alloc-flight-meta">' + meta + '</div>' +
            '</div>';
        }}).join('');
        const sidAttr = standId ? String(standId) : '';
        const gridLines = tickPositions.map(tp =>
          '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>'
        ).join('');
        // 시간축과 시간축 "사이"의 중앙에 배경 텍스트를 배치
        const bgSlots = (tickPositions.length > 1)
          ? tickPositions.slice(0, -1).map((tp, idx) => {{
              const next = tickPositions[idx + 1];
              const midLeft = (tp.leftPct + next.leftPct) / 2;
              return (
                '<div class="alloc-apron-bg-slot" style="left:' + midLeft + '%;transform:translateX(-50%);">' +
                  escapeHtml(label) +
                '</div>'
              );
            }}).join('')
          : '';
        return '' +
          '<div class="alloc-row" data-stand-id="' + sidAttr + '">' +
            '<div class="alloc-row-label">' + escapeHtml(label) + '</div>' +
            '<div class="alloc-row-track" data-stand-id="' + sidAttr + '">' +
              gridLines +
              bgSlots +
              blocks +
              (showEibtBars && eBars ? eBars.join('') : '') +
              (showEldtBars && e2Bars ? e2Bars.join('') : '') +
              (showAuxBars && sBars ? sBars.join('') : '') +
            '</div>' +
          '</div>';
      }}
      const rows = [];
      // Unassigned 행
      rows.push(buildRowHtml('Unassigned', null));
      // 터미널별 Stand 그룹화
      const terminalCopies = makeUniqueNamedCopy(state.terminals || [], 'name');
      const termLabelById = {{}};
      terminalCopies.forEach(t => {{ termLabelById[t.id] = (t.name || '').trim() || 'Terminal'; }});
      const grouped = {{}};
      const order = [];
      const sortedStands = stands.slice().sort((a, b) => {{
        const ta = getTerminalForStand(a);
        const tb = getTerminalForStand(b);
        const la = ta ? (termLabelById[ta.id] || ta.name || '') : '';
        const lb = tb ? (termLabelById[tb.id] || tb.name || '') : '';
        if (la < lb) return -1;
        if (la > lb) return 1;
        const na = (a.name || '').toLowerCase();
        const nb = (b.name || '').toLowerCase();
        if (na < nb) return -1;
        if (na > nb) return 1;
        return 0;
      }});
      sortedStands.forEach(s => {{
        const term = getTerminalForStand(s);
        const key = term ? term.id : '__no_terminal__';
        if (!grouped[key]) {{
          grouped[key] = {{ term, stands: [] }};
          order.push(key);
        }}
        grouped[key].stands.push(s);
      }});
      order.forEach(key => {{
        const group = grouped[key];
        if (!group) return;
        const term = group.term;
        const headerLabel = term
          ? (termLabelById[term.id] || term.name || 'Terminal')
          : 'No Terminal';
        // 터미널 헤더: 왼쪽에 고정 (CSS에서 sticky)
        rows.push('<div class="alloc-terminal-header">' + escapeHtml(headerLabel) + '</div>');
        // 각 주기장 행: 기존처럼 Stand 이름/코드만 표시 (터미널 이름은 헤더에)
        group.stands.forEach(s => {{
          const label = (s.name || '') + ' (' + (s.category || '') + ')';
          rows.push(buildRowHtml(label, s.id));
        }});
      }});
      // Time axis overlay at bottom (세로 그리드라인과 같은 위치에 시간 라벨만 표시)
      const axisTicks = tickPositions.map(tp =>
        '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
          '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
        '</div>'
      );
      const axisHtml =
        '<div class="alloc-time-axis-overlay">' +
          '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
        '</div>';

      ganttEl.innerHTML = rows.join('') + axisHtml;

      // Shift + 마우스 휠로 시간축 줌 (Apron)
      if (!ganttEl._allocZoomBound) {{
        ganttEl._allocZoomBound = true;
        ganttEl.addEventListener('wheel', function(e) {{
          if (!e.shiftKey) return;
          e.preventDefault();
          // 위로 스크롤 = 확대, 아래로 스크롤 = 축소
          const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
          let z = state.allocTimeZoom || 1;
          z *= factor;
          if (z < 1) z = 1;           // 최소: 전체 범위
          if (z > 8) z = 8;           // 최대 확대 배율
          state.allocTimeZoom = z;
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }}, {{ passive: false }});
      }}
      // Drag & drop wiring
      ganttEl.querySelectorAll('.alloc-flight').forEach(el => {{
        el.addEventListener('dragstart', function(ev) {{
          ev.dataTransfer.setData('text/plain', this.getAttribute('data-flight-id') || '');
          ev.dataTransfer.effectAllowed = 'move';
        }});
        el.addEventListener('click', function(ev) {{
          ev.stopPropagation();
          const flightId = this.getAttribute('data-flight-id');
          if (!flightId) return;
          const f = state.flights.find(x => x.id === flightId);
          if (!f) return;
          state.selectedObject = {{ type: 'flight', id: flightId, obj: f }};
          if (typeof updateObjectInfo === 'function') updateObjectInfo();
          if (typeof syncPanelFromState === 'function') syncPanelFromState();
          if (typeof draw === 'function') draw();
          const listEl = document.getElementById('flightList');
          if (listEl) {{
            listEl.querySelectorAll('.obj-item').forEach(r => r.classList.remove('selected', 'expanded'));
            const row = listEl.querySelector('.obj-item[data-id="' + flightId + '"]');
            if (row) row.classList.add('selected', 'expanded');
          }}
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }});
      }});
      ganttEl.querySelectorAll('.alloc-row-track').forEach(track => {{
        track.addEventListener('dragover', function(ev) {{
          ev.preventDefault();
          ev.dataTransfer.dropEffect = 'move';
        }});
        track.addEventListener('drop', function(ev) {{
          ev.preventDefault();
          const flightId = ev.dataTransfer.getData('text/plain');
          if (!flightId) return;
          const f = state.flights.find(x => x.id === flightId);
          if (!f) return;
          const sidAttr = this.getAttribute('data-stand-id') || '';
          const standId = sidAttr || null;
          if (standId) {{
            const stand = standById[standId];
            if (!flightCanUseStand(f, stand)) {{
              alert('Code 또는 선택된 Terminal이 맞지 않아 이 Apron(Stand)에 배정할 수 없습니다.');
              return;
            }}
          }}
          f.standId = standId;
          if (f.token) f.token.apronId = standId;
          renderFlightGantt();
          renderFlightList();
          if (typeof draw === 'function') draw();
        }});
      }});
    }}

    function validateNetworkForFlights() {{
      const msgs = [];
      const hasRunwayObject = state.runways && state.runways.length > 0;
      const hasRunwayPath = state.taxiways && state.taxiways.some(tw => tw.pathType === 'runway' || (tw.name || '').toLowerCase().includes('runway'));
      if (!hasRunwayObject && !hasRunwayPath) msgs.push('Runway가 없습니다.');
      if (!state.taxiways || !state.taxiways.length) msgs.push('Taxiway가 없습니다.');
      const stands = state.pbbStands || [];
      const linked = state.apronLinks || [];
      // 적어도 하나의 PBB가 실제 존재하는 Taxiway와 Apron Taxiway 링크로 연결되어 있어야 한다.
      const hasApronLink = stands.some(pbb =>
        linked.some(lk =>
          lk.pbbId === pbb.id &&
          state.taxiways &&
          state.taxiways.some(tw => tw.id === lk.taxiwayId)
        )
      );
      if (!stands.length || !hasApronLink) msgs.push('Apron(PBB)과 Taxiway를 연결하는 링크가 최소 1개 이상 필요합니다.');
      return msgs;
    }}

    function updateFlightError(msgs) {{
      const el = document.getElementById('flightError');
      if (!el) return;
      el.textContent = Array.isArray(msgs) ? msgs.join(' / ') : (msgs || '');
    }}

    // ---- Layout Design 최소경로: Node/Edge 그래프, 역방향 비용 1,000,000 ----
    const REVERSE_COST = 1000000;
    function pathDist2(a, b) {{ return (a[0]-b[0])**2 + (a[1]-b[1])**2; }}
    function pathDist(a, b) {{ return Math.hypot(a[0]-b[0], a[1]-b[1]); }}

    function clamp(v, min, max) {{
      return Math.max(min, Math.min(max, v));
    }}
    function sampleNormal(mu, sigma) {{
      const u1 = Math.random() || 1e-9;
      const u2 = Math.random() || 1e-9;
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return mu + sigma * z;
    }}

    // 동일 격자(같은 셀)는 항상 같은 노드로: 셀 좌표 기반 키
    function pathPointKey(p) {{
      const cellCol = Math.round(p[0] / (typeof CELL_SIZE === 'number' ? CELL_SIZE : 20));
      const cellRow = Math.round(p[1] / (typeof CELL_SIZE === 'number' ? CELL_SIZE : 20));
      return cellCol + ',' + cellRow;
    }}

    // Runway separation: SLDT(Arr)·STOT(Dep) 단일 타임라인 시간순 정렬, 동일 시각은 위쪽(리스트 순)을 선행으로 보고
    // 선행 E계열 + 분리기준으로 후행 E계열 계산 (도미노). Arr→ELDT, Dep→ETOT
    // 반환값: 활주로 ID별로 events/minT/maxT를 담은 맵 (시각화용)
    function computeSeparationAdjustedTimes() {{
      const runwaysRaw = (state.taxiways || []).filter(t => t.pathType === 'runway');
      if (!runwaysRaw.length) return {{}};
      const flights = state.flights || [];
      // ELDT/ETOT 전역 초기화
      flights.forEach(f => {{ delete f.eldtMin; delete f.etotMin; }});

      // 동일 항공기에서 Arr 활주로가 Dep 활주로보다 먼저 처리되도록 활주로 순서를 정렬
      const runways = (function() {{
        const idToIndex = {{}};
        runwaysRaw.forEach((r, i) => {{ if (r && r.id != null) idToIndex[r.id] = i; }});
        const n = runwaysRaw.length;
        const indeg = new Array(n).fill(0);
        const adj = new Array(n).fill(0).map(() => []);
        flights.forEach(f => {{
          if (!f) return;
          let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
          let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
          if (!arrRwy || !depRwy || arrRwy === depRwy) return;
          const ai = idToIndex[arrRwy];
          const di = idToIndex[depRwy];
          if (ai == null || di == null) return;
          adj[ai].push(di);
          indeg[di] += 1;
        }});
        const q = [];
        for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i);
        const orderIdx = [];
        while (q.length) {{
          const i = q.shift();
          orderIdx.push(i);
          adj[i].forEach(j => {{
            indeg[j] -= 1;
            if (indeg[j] === 0) q.push(j);
          }});
        }}
        // 순환 등으로 모든 노드를 방문하지 못한 경우, 원래 순서를 사용
        if (orderIdx.length !== n) return runwaysRaw;
        return orderIdx.map(i => runwaysRaw[i]);
      }})();

      const byRunway = {{}};
      function getSec(val) {{ const n = Number(val); return isFinite(n) && n >= 0 ? n : 90; }}
      runways.forEach(rwy => {{
        const cfg = rsepGetConfigForRunway(rwy);
        if (!cfg) return;
        const stdKey = cfg.standard || 'ICAO';
        const arrArr = (cfg.seqData && cfg.seqData['ARR→ARR']) ? cfg.seqData['ARR→ARR'] : {{}};
        const depDep = (cfg.seqData && cfg.seqData['DEP→DEP']) ? cfg.seqData['DEP→DEP'] : {{}};
        const depArr = (cfg.seqData && cfg.seqData['DEP→ARR']) ? cfg.seqData['DEP→ARR'] : {{}};
        const rot = (cfg.rot) ? cfg.rot : {{}};
        const events = [];
        let eventIndex = 0;
        flights.forEach((f, flightIdx) => {{
          if (f.noWayArr || f.noWayDep) return;
          let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
          let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
          if (arrRwy == null && depRwy == null && runways.length === 1) {{ arrRwy = rwy.id; depRwy = rwy.id; }}
          else if (depRwy == null && arrRwy === rwy.id) depRwy = rwy.id;
          else if (arrRwy == null && depRwy === rwy.id) arrRwy = rwy.id;
          if (arrRwy !== rwy.id && depRwy !== rwy.id) return;
          const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
          const cat = stdKey === 'ICAO' ? (ac && ac.icaoJHL ? ac.icaoJHL : 'M') : (ac && ac.recatEu ? ac.recatEu : 'D');
          const tArrMin = f.timeMin != null ? f.timeMin : 0;
          // Dwell / Min Dwell 하한 20분 및 Dwell >= Min Dwell 보장
          let dwell = f.dwellMin != null ? f.dwellMin : 0;
          let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
          dwell = Math.max(20, dwell);
          minDwell = Math.max(20, minDwell);
          if (minDwell > dwell) minDwell = dwell;
          f.dwellMin = dwell;
          f.minDwellMin = minDwell;
          const tDepMin = tArrMin + dwell;
          const v = Math.max(1, f.velocity != null ? f.velocity : 15);
          let vttArrMin = 0, vttDepMin = 0;
          const arrPts = (typeof getPathForFlight === 'function') ? getPathForFlight(f) : null;
          if (arrPts && arrPts.length >= 2) {{
            let dist = 0;
            for (let i = 0; i < arrPts.length - 1; i++) dist += pathDist(arrPts[i], arrPts[i+1]);
            vttArrMin = dist / v / 60;
          }}
          const depPts = (typeof getPathForFlightDeparture === 'function') ? getPathForFlightDeparture(f) : null;
          if (depPts && depPts.length >= 2) {{
            let dist = 0;
            for (let i = 0; i < depPts.length - 1; i++) dist += pathDist(depPts[i], depPts[i+1]);
            vttDepMin = dist / v / 60;
          }}
          const sldtMin = Math.max(0, tArrMin - vttArrMin);
          const stotMin = tDepMin + vttDepMin;
          const sobtMin = tDepMin;
          if (arrRwy === rwy.id) events.push({{ time: sldtMin, type: 'arr', flight: f, cat: cat, vttArrMin, index: eventIndex++ }});
          if (depRwy === rwy.id) events.push({{ time: stotMin, type: 'dep', flight: f, cat: cat, vttDepMin, vttArrMin, sobtMin, index: eventIndex++ }});
        }});
        events.sort((a, b) => a.time - b.time || a.index - b.index);
        let lastArrETime = -1e9, lastArrCat = null;
        let lastDepETime = -1e9, lastDepCat = null;
        events.forEach(ev => {{
          if (ev.type === 'arr') {{
            let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec((arrArr[lastArrCat] && arrArr[lastArrCat][ev.cat]) != null ? arrArr[lastArrCat][ev.cat] : 90) / 60 : -1e9;
            let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec(depArr[ev.cat]) / 60 : -1e9;
            const eTime = Math.max(ev.time, minFromArr, minFromDep);
            ev.flight.eldtMin = eTime;
            lastArrETime = eTime;
            lastArrCat = ev.cat;
          }} else {{
            let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec(rot[lastArrCat]) / 60 : -1e9;
            let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec((depDep[lastDepCat] && depDep[lastDepCat][ev.cat]) != null ? depDep[lastDepCat][ev.cat] : 90) / 60 : -1e9;
            const etotSep = Math.max(ev.time, minFromArr, minFromDep);
            const eibtMin = (ev.flight.eldtMin != null ? ev.flight.eldtMin : 0) + (ev.vttArrMin || 0);
            const minDwellMin = ev.flight.minDwellMin != null ? Math.max(20, ev.flight.minDwellMin) : 20;
            const eobtMin = Math.max(etotSep - (ev.vttDepMin || 0), ev.sobtMin, eibtMin + minDwellMin);
            const etotMin = eobtMin + (ev.vttDepMin || 0);
            ev.flight.etotMin = etotMin;
            lastDepETime = etotMin;
            lastDepCat = ev.cat;
          }}
        }});
        // 시각화를 위한 시간 범위 계산
        if (events.length) {{
          let minT = Infinity;
          let maxT = -Infinity;
          events.forEach(ev => {{
            const s = ev.time;
            const e = ev.type === 'arr'
              ? (ev.flight && ev.flight.eldtMin != null ? ev.flight.eldtMin : s)
              : (ev.flight && ev.flight.etotMin != null ? ev.flight.etotMin : s);
            if (s < minT) minT = s;
            if (e < minT) minT = e;
            if (s > maxT) maxT = s;
            if (e > maxT) maxT = e;
          }});
          if (!isFinite(minT) || !isFinite(maxT)) {{
            minT = 0; maxT = 60;
          }} else if (maxT <= minT) {{
            maxT = minT + 60;
          }}
          byRunway[rwy.id] = {{ events, minT, maxT }};
        }} else {{
          byRunway[rwy.id] = {{ events: [], minT: 0, maxT: 0 }};
        }}
      }});
      return byRunway;
    }}

    function getRunwayPath(runwayId) {{
      const taxiways = state.taxiways || [];
      let rw = runwayId ? taxiways.find(t => t.id === runwayId && (t.pathType === 'runway' || (t.name||'').toLowerCase().includes('runway')) && t.vertices && t.vertices.length >= 2) : null;
      if (!rw) rw = taxiways.find(t => t.pathType === 'runway' && t.vertices && t.vertices.length >= 2);
      if (!rw) rw = taxiways.find(t => (t.name||'').toLowerCase().includes('runway') && t.vertices && t.vertices.length >= 2);
      if (!rw || !rw.vertices.length) return null;
      const pts = rw.vertices.map(v => cellToPixel(v.col, v.row));
      const sp = rw.start_point, ep = rw.end_point;
      if (sp && ep) {{
        const startPx = cellToPixel(sp.col, sp.row);
        const endPx = cellToPixel(ep.col, ep.row);
        if (pathDist2(pts[pts.length-1], startPx) < pathDist2(pts[0], startPx)) pts.reverse();
      }}
      return {{ startPx: pts[0], endPx: pts[pts.length-1], pts }};
    }}

    function syncStartEndFromVertices(obj) {{
      if (!obj || !obj.vertices || obj.vertices.length < 2) return;
      const first = obj.vertices[0], last = obj.vertices[obj.vertices.length - 1];
      obj.start_point = {{ col: first.col, row: first.row }};
      obj.end_point = {{ col: last.col, row: last.row }};
    }}
    function getTaxiwayOrderedPoints(tw) {{
      if (!tw.vertices || tw.vertices.length < 2) return null;
      const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
      if (tw.start_point && tw.end_point) {{
        const startPx = cellToPixel(tw.start_point.col, tw.start_point.row);
        if (pathDist2(pts[pts.length-1], startPx) < pathDist2(pts[0], startPx)) pts.reverse();
      }}
      return pts;
    }}
    function getOrderedPoints(obj) {{
      if (!obj || !obj.vertices || obj.vertices.length < 2) return null;
      const isRunway = obj.pathType === 'runway' || (obj.name||'').toLowerCase().includes('runway');
      if (isRunway) {{ const r = getRunwayPath(obj.id); return r && r.pts ? r.pts : null; }}
      return getTaxiwayOrderedPoints(obj);
    }}

    function projectOnSegment(a, b, q) {{
      const ax = a[0], ay = a[1], bx = b[0], by = b[1], qx = q[0], qy = q[1];
      const dx = bx - ax, dy = by - ay, den = dx*dx + dy*dy;
      if (den < 1e-12) return {{ t: 0, p: a }};
      let t = ((qx-ax)*dx + (qy-ay)*dy) / den;
      t = Math.max(0, Math.min(1, t));
      return {{ t, p: [ax+t*dx, ay+t*dy] }};
    }}
    // 세그먼트 (a,b)와 (c,d)의 교차점. 실제 교차 시에만 반환 (0<=t,s<=1).
    function segmentSegmentIntersection(a, b, c, d) {{
      const ax = a[0], ay = a[1], bx = b[0], by = b[1];
      const cx = c[0], cy = c[1], dx = d[0], dy = d[1];
      const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
      const cross = rx * sy - ry * sx;
      if (Math.abs(cross) < 1e-12) return null;
      const t = ((cx - ax) * sy - (cy - ay) * sx) / cross;
      const s = ((cx - ax) * ry - (cy - ay) * rx) / cross;
      if (t < 0 || t > 1 || s < 0 || s > 1) return null;
      return {{ p: [ax + t * rx, ay + t * ry] }};
    }}
    const SPLIT_TOL_D2 = 0.25;
    function pointOnSegmentStrict(a, b, q) {{
      const {{ p }} = projectOnSegment(a, b, q);
      return pathDist2(p, q) <= SPLIT_TOL_D2;
    }}

    // Runway별로 연결된 Runway Exit Taxiway(RET)의 threshold로부터 거리(m)를 계산
    function computeRunwayExitDistances() {{
      const taxiways = state.taxiways || [];
      const runways = taxiways.filter(t => t.pathType === 'runway' && Array.isArray(t.vertices) && t.vertices.length >= 2);
      const exits = taxiways.filter(t => t.pathType === 'runway_exit' && Array.isArray(t.vertices) && t.vertices.length >= 2);
      const results = [];
      if (!runways.length || !exits.length) return results;

      runways.forEach(rw => {{
        // Runway 중심선(격자 좌표) 정렬: start_point 기준 방향 정리
        let rVerts = rw.vertices.map(v => [v.col, v.row]);
        if (rw.start_point && rw.end_point && rVerts.length >= 2) {{
          const sp = [rw.start_point.col, rw.start_point.row];
          if (pathDist2(rVerts[rVerts.length - 1], sp) < pathDist2(rVerts[0], sp)) rVerts.reverse();
        }}
        if (rVerts.length < 2) return;
        // prefix 거리(셀 단위)
        const prefixDist = [0];
        for (let i = 1; i < rVerts.length; i++) {{
          prefixDist[i] = prefixDist[i - 1] + pathDist(rVerts[i - 1], rVerts[i]);
        }}

        exits.forEach(tw => {{
          let best = null;
          const exitName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Exit ' + String(results.length + 1));
          tw.vertices.forEach(v => {{
            const q = [v.col, v.row];
            for (let i = 0; i < rVerts.length - 1; i++) {{
              const a = rVerts[i], b = rVerts[i + 1];
              if (!pointOnSegmentStrict(a, b, q)) continue;
              const segLen = pathDist(a, b);
              if (!(segLen > 1e-6)) continue;
              const proj = projectOnSegment(a, b, q);
              // proj.p는 a-b 세그먼트 상의 점
              const t = Math.max(0, Math.min(1, segLen > 0 ? pathDist(a, proj.p) / segLen : 0));
              const distCells = prefixDist[i] + segLen * t;
              const distM = distCells * CELL_SIZE;
              const maxExitVel = (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0)
                ? tw.maxExitVelocity
                : 30;
              if (!best || distM < best.distM) {{
                best = {{ runway: rw, exit: tw, name: exitName, distM, maxExitVelocity: maxExitVel }};
              }}
            }}
          }});
          if (best) results.push(best);
        }});
      }});

      // threshold로부터 거리 순으로 정렬
      results.sort((a, b) => a.distM - b.distM);

      // #region agent log (RET stats summary)
      try {{
        _logNoWay({{
          location: 'Layout_Design.js:computeRunwayExitDistances',
          message: 'RET stats summary',
          hypothesisId: 'H3',
          runId: 'ret-debug-1',
          data: {{
            runwayCount: runways.length,
            exitCount: exits.length,
            resultCount: results.length,
            runways: runways.map(r => ({{ id: r.id, name: r.name || '' }})),
            sampleResults: results.slice(0, 5).map(r => ({{ runwayId: r.runway && r.runway.id, name: r.name, distM: r.distM, maxExitVelocity: r.maxExitVelocity }}))
          }}
        }});
      }} catch (e) {{}}
      // #endregion

      return results;
    }}

    function buildPathGraph() {{
      const nodes = [], keyToIdx = {{}}, edges = [], adj = [], junctionPts = [], junctionKeys = {{}};
      function addJunction(p) {{
        const k = pathPointKey(p);
        if (junctionKeys[k]) return;
        junctionKeys[k] = true;
        junctionPts.push(p);
      }}
      function getOrAdd(p) {{
        const k = pathPointKey(p);
        if (keyToIdx[k] != null) return keyToIdx[k];
        const idx = nodes.length;
        nodes.push(p);
        keyToIdx[k] = idx;
        adj[idx] = [];
        return idx;
      }}
      function addEdgeWithDirection(pFrom, pTo, dir, cost) {{
        const i = getOrAdd(pFrom), j = getOrAdd(pTo);
        if (i === j || cost < 1e-6) return;
        edges.push({{ from: i, to: j, dist: cost }});
        if (dir === 'both') {{
          adj[i].push([j, cost]);
          adj[j].push([i, cost]);
          edges.push({{ from: j, to: i, dist: cost }});
        }} else if (dir === 'counter_clockwise') {{
          adj[j].push([i, cost]);
          adj[i].push([j, REVERSE_COST]);
          edges.push({{ from: i, to: j, dist: REVERSE_COST }});
        }} else {{
          adj[i].push([j, cost]);
          adj[j].push([i, REVERSE_COST]);
          edges.push({{ from: j, to: i, dist: REVERSE_COST }});
        }}
      }}

      const pathList = state.taxiways || [];
      const apronNodeStand = [];
      const minD2 = 1e-6;
      pathList.forEach(obj => {{
        const pts = getOrderedPoints(obj);
        if (!pts || pts.length < 2) return;
        const junctions = [];
        for (let seg = 0; seg < pts.length - 1; seg++) {{
          const a = pts[seg], b = pts[seg+1];
          pathList.forEach(other => {{
            if (other.id === obj.id) return;
            const otherOrd = getOrderedPoints(other);
            if (!otherOrd || otherOrd.length < 2) return;
            for (let oseg = 0; oseg < otherOrd.length - 1; oseg++) {{
              const c = otherOrd[oseg], d = otherOrd[oseg+1];
              const isec = segmentSegmentIntersection(a, b, c, d);
              if (isec) {{
                const {{ t }} = projectOnSegment(a, b, isec.p);
                junctions.push({{ tAlong: seg + t, p: isec.p }});
              }} else {{
                // 공선이면 교차가 null. 끝점이 겹치는 경우 정션으로 추가 (활주로-택시웨이 끝끼리 연결)
                [c, d].forEach(function(q, idx) {{
                  if (pathDist2(a, q) <= SPLIT_TOL_D2 || pathDist2(b, q) <= SPLIT_TOL_D2) {{
                    const {{ t, p: proj }} = projectOnSegment(a, b, q);
                    if (t >= 0 && t <= 1) junctions.push({{ tAlong: seg + t, p: proj }});
                  }}
                }});
              }}
            }}
            otherOrd.forEach(q => {{
              if (!pointOnSegmentStrict(a, b, q)) return;
              const {{ t, p: proj }} = projectOnSegment(a, b, q);
              junctions.push({{ tAlong: seg + t, p: proj }});
            }});
          }});
          const isRunway = obj.pathType === 'runway' || (obj.name||'').toLowerCase().includes('runway');
          if (!isRunway) {{
            (state.apronLinks || []).forEach(lk => {{
              if (lk.taxiwayId !== obj.id || lk.tx == null || lk.ty == null) return;
              const linkPt = [Number(lk.tx), Number(lk.ty)];
              const {{ t, p }} = projectOnSegment(a, b, linkPt);
              if (t >= 0 && t <= 1 && pathDist2(p, linkPt) <= SPLIT_TOL_D2) {{
                junctions.push({{ tAlong: seg + t, p }});
                const pbb = (state.pbbStands || []).find(s => s.id === lk.pbbId) || (state.remoteStands || []).find(s => s.id === lk.pbbId);
                if (pbb) apronNodeStand.push({{ nodeP: p, standPt: (pbb.x2 != null && pbb.y2 != null) ? [Number(pbb.x2), Number(pbb.y2)] : cellToPixel(pbb.col || 0, pbb.row || 0), standId: lk.pbbId }});
              }}
            }});
          }}
        }}
        const waypoints = [];
        for (let i = 0; i < pts.length; i++) waypoints.push({{ tAlong: i, p: pts[i], isJunction: false }});
        junctions.forEach(({{ tAlong, p }}) => waypoints.push({{ tAlong, p, isJunction: true }}));
        waypoints.sort((x, y) => x.tAlong - y.tAlong);
        const chain = [];
        waypoints.forEach(({{ p, isJunction }}) => {{
          if (chain.length && pathDist2(p, chain[chain.length-1]) < minD2) {{
            if (isJunction) addJunction(p);
            return;
          }}
          chain.push(p);
          if (isJunction) addJunction(p);
        }});
        const dir = getTaxiwayDirection(obj);
        for (let i = 0; i < chain.length - 1; i++) {{
          const d = pathDist(chain[i], chain[i+1]);
          addEdgeWithDirection(chain[i], chain[i+1], dir, d);
        }}
      }});

      const standNodeIndices = [];
      const standIdToNodeIndex = {{}};
      apronNodeStand.forEach(({{ nodeP, standPt, standId }}) => {{
        const i = getOrAdd(nodeP), j = getOrAdd(standPt);
        standNodeIndices.push(j);
        if (standId != null) standIdToNodeIndex[standId] = j;
        if (i === j) return;
        const d = pathDist(nodes[i], nodes[j]);
        if (d < 1e-6) return;
        adj[i].push([j, d]);
        adj[j].push([i, d]);
        edges.push({{ from: i, to: j, dist: d }});
        edges.push({{ from: j, to: i, dist: d }});
      }});
      // BFS: cost < REVERSE_COST인 간선만 따라 도달 가능한 노드 집합
      function bfsReachable(startIndices) {{
        const out = new Set();
        const q = startIndices.slice();
        startIndices.forEach(function(idx) {{ out.add(idx); }});
        while (q.length) {{
          const u = q.shift();
          (adj[u] || []).forEach(function(tuple) {{
            const v = tuple[0], w = tuple[1];
            if (w >= REVERSE_COST) return;
            if (!out.has(v)) {{ out.add(v); q.push(v); }}
          }});
        }}
        return out;
      }}
      function nearestNode(p) {{
        let best = 0, bestD2 = pathDist2(nodes[0], p);
        for (let i = 1; i < nodes.length; i++) {{
          const d2 = pathDist2(nodes[i], p);
          if (d2 < bestD2) {{ bestD2 = d2; best = i; }}
        }}
        return best;
      }}
      const runwayNodeIndices = [];
      const runways = (state.taxiways || []).filter(function(t) {{ return t.pathType === 'runway' || (t.name||'').toLowerCase().indexOf('runway') >= 0; }});
      if (runways.length) {{
        const r = getRunwayPath(runways[0].id);
        if (r) {{
          runwayNodeIndices.push(nearestNode(r.startPx));
          runwayNodeIndices.push(nearestNode(r.endPx));
        }}
      }}
      const runwayReachable = runwayNodeIndices.length ? bfsReachable(runwayNodeIndices) : new Set();
      const standReachable = standNodeIndices.length ? bfsReachable(standNodeIndices) : new Set();
      const connected = new Set();
      runwayReachable.forEach(function(i) {{ if (standReachable.has(i)) connected.add(i); }});
      // 활주로-주기장이 서로 도달 가능한 연결된 구간 안의 분기점(degree>=2)만 녹색 정션으로 표시
      const junctionsForDraw = junctionPts.filter(function(p) {{
        const i = keyToIdx[pathPointKey(p)];
        return i != null && adj[i] && adj[i].length >= 2 && connected.has(i);
      }});
      return {{ nodes, edges, adj, getOrAdd, junctions: junctionsForDraw, standIdToNodeIndex }};
    }}

    class MinHeap {{
      constructor() {{ this.h = []; }}
      push(item) {{
        this.h.push(item);
        let i = this.h.length - 1;
        while (i > 0) {{
          const p = (i - 1) >> 1;
          if (this.h[p][0] <= this.h[i][0]) break;
          [this.h[p], this.h[i]] = [this.h[i], this.h[p]];
          i = p;
        }}
      }}
      pop() {{
        const top = this.h[0];
        const last = this.h.pop();
        if (this.h.length) {{
          this.h[0] = last;
          let i = 0;
          while (true) {{
            let s = i, l = 2*i+1, r = 2*i+2;
            if (l < this.h.length && this.h[l][0] < this.h[s][0]) s = l;
            if (r < this.h.length && this.h[r][0] < this.h[s][0]) s = r;
            if (s === i) break;
            [this.h[s], this.h[i]] = [this.h[i], this.h[s]];
            i = s;
          }}
        }}
        return top;
      }}
      get size() {{ return this.h.length; }}
    }}

    function pathDijkstra(g, startIdx, endIdx) {{
      const n = g.nodes.length;
      const dist = Array(n).fill(Infinity);
      const prev = Array(n).fill(null);
      dist[startIdx] = 0;
      const heap = new MinHeap();
      heap.push([0, startIdx]);
      while (heap.size) {{
        const [d, u] = heap.pop();
        if (d > dist[u]) continue;
        if (u === endIdx) break;
        for (const [v, w] of g.adj[u]) {{
          const nd = d + w;
          if (nd < dist[v]) {{
            dist[v] = nd;
            prev[v] = u;
            heap.push([nd, v]);
          }}
        }}
      }}
      if (dist[endIdx] === Infinity || dist[endIdx] >= REVERSE_COST) return null;
      const path = [];
      for (let cur = endIdx; cur !== null; cur = prev[cur]) path.push(cur);
      return path.reverse();
    }}

    function nearestPathNode(g, p) {{
      let best = 0, bestD2 = pathDist2(g.nodes[0], p);
      for (let i = 1; i < g.nodes.length; i++) {{
        const d2 = pathDist2(g.nodes[i], p);
        if (d2 < bestD2) {{ bestD2 = d2; best = i; }}
      }}
      return best;
    }}

    function pathTotalDist(g, pathIndices) {{
      let d = 0;
      for (let i = 0; i < pathIndices.length - 1; i++) {{
        const a = g.nodes[pathIndices[i]], b = g.nodes[pathIndices[i+1]];
        const e = g.edges.find(x => x.from === pathIndices[i] && x.to === pathIndices[i+1]);
        if (e) d += e.dist; else d += pathDist(a, b);
      }}
      return d;
    }}

    function getPathForFlight(f) {{
      const token = f.token || {{}};
      let runwayId = token.arrRunwayId || token.runwayId || f.arrRunwayId;
      const apronId = f.standId != null ? f.standId : (token.apronId || null);
      if (!apronId) return null;
      if (!runwayId && state.taxiways && state.taxiways.length) {{
        const runways = state.taxiways.filter(t => (t.pathType === 'runway' || (t.name||'').toLowerCase().includes('runway')) && t.vertices && t.vertices.length >= 2);
        if (runways.length) runwayId = runways[Math.floor(Math.random() * runways.length)].id;
      }}
      if (!runwayId) return null;
      const r = getRunwayPath(runwayId);
      if (!r) return null;
      const stand = (state.pbbStands || []).find(s => s.id === apronId) || (state.remoteStands || []).find(s => s.id === apronId);
      if (!stand) return null;
      const g = buildPathGraph();
      state.pathGraphJunctions = g.junctions || [];
      // 녹색 점(연결점)이 있는 경우에만 경로 허용: apron link로 연결된 주기장 노드만 사용
      const endIdx = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (endIdx == null) {{
        f.noWayArr = true;
        if (typeof _logNoWay === 'function') _logNoWay({{ location: 'getPathForFlight:noWayArr', message: 'No Way Arr (no apron link)', data: {{ flightId: f.id, reg: f.reg, apronId: apronId, runwayId: runwayId }} }});
        return null;
      }}
      const startIdx = nearestPathNode(g, r.startPx);
      const endRunwayIdx = nearestPathNode(g, r.endPx);
      const fromStart = pathDijkstra(g, startIdx, endIdx);
      const fromEnd = pathDijkstra(g, endRunwayIdx, endIdx);
      const distFromStart = fromStart ? pathTotalDist(g, fromStart) : Infinity;
      const distFromEnd = fromEnd ? pathTotalDist(g, fromEnd) : Infinity;
      let pathIndices = fromStart;
      if (fromEnd && (!fromStart || distFromEnd < distFromStart)) pathIndices = fromEnd;
      const totalD = pathIndices ? pathTotalDist(g, pathIndices) : Infinity;
      if (!pathIndices || pathIndices.length < 2 || totalD >= REVERSE_COST) {{
        f.noWayArr = true;
        if (typeof _logNoWay === 'function') _logNoWay({{ location: 'getPathForFlight:noWayArr', message: 'No Way Arr', data: {{ flightId: f.id, reg: f.reg, apronId: apronId, runwayId: runwayId, startIdx: startIdx, endIdx: endIdx, endRunwayIdx: endRunwayIdx, fromStartLen: fromStart ? fromStart.length : null, fromEndLen: fromEnd ? fromEnd.length : null, distFromStart: distFromStart, distFromEnd: distFromEnd, totalD: totalD, nodesCount: g.nodes.length, junctionsCount: (g.junctions && g.junctions.length) || 0, taxiwayNames: (state.taxiways || []).map(function(t) {{ return t.name || t.id; }}) }} }});
        return null;
      }}
      f.noWayArr = false;
      return pathIndices.map(i => g.nodes[i]);
    }}

    function getPathForFlightDeparture(f) {{
      const token = f.token || {{}};
      let runwayId = token.depRunwayId || token.runwayId || f.depRunwayId || f.arrRunwayId;
      const apronId = f.standId != null ? f.standId : (token.apronId || null);
      if (!apronId) return null;
      if (!runwayId && state.taxiways && state.taxiways.length) {{
        const runways = state.taxiways.filter(t => (t.pathType === 'runway' || (t.name||'').toLowerCase().includes('runway')) && t.vertices && t.vertices.length >= 2);
        if (runways.length) runwayId = runways[Math.floor(Math.random() * runways.length)].id;
      }}
      if (!runwayId) return null;
      const r = getRunwayPath(runwayId);
      if (!r) return null;
      const stand = (state.pbbStands || []).find(s => s.id === apronId) || (state.remoteStands || []).find(s => s.id === apronId);
      if (!stand) return null;
      const g = buildPathGraph();
      // 녹색 점(연결점)이 있는 경우에만 경로 허용: apron link로 연결된 주기장 노드만 사용
      const startIdx = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (startIdx == null) {{
        f.noWayDep = true;
        if (typeof _logNoWay === 'function') _logNoWay({{ location: 'getPathForFlightDeparture:noWayDep', message: 'No Way Dep (no apron link)', data: {{ flightId: f.id, reg: f.reg, apronId: apronId, runwayId: runwayId }} }});
        return null;
      }}
      const toStart = pathDijkstra(g, startIdx, nearestPathNode(g, r.startPx));
      const toEnd = pathDijkstra(g, startIdx, nearestPathNode(g, r.endPx));
      const distToStart = toStart ? pathTotalDist(g, toStart) : Infinity;
      const distToEnd = toEnd ? pathTotalDist(g, toEnd) : Infinity;
      let pathIndices = toStart;
      if (toEnd && (!toStart || distToEnd < distToStart)) pathIndices = toEnd;
      const totalD = pathIndices ? pathTotalDist(g, pathIndices) : Infinity;
      if (!pathIndices || pathIndices.length < 2 || totalD >= REVERSE_COST) {{
        f.noWayDep = true;
        if (typeof _logNoWay === 'function') _logNoWay({{ location: 'getPathForFlightDeparture:noWayDep', message: 'No Way Dep', data: {{ flightId: f.id, reg: f.reg, apronId: apronId, runwayId: runwayId, startIdx: startIdx, toStartLen: toStart ? toStart.length : null, toEndLen: toEnd ? toEnd.length : null, distToStart: distToStart, distToEnd: distToEnd, totalD: totalD, nodesCount: g.nodes.length, taxiwayNames: (state.taxiways || []).map(function(t) {{ return t.name || t.id; }}) }} }});
        return null;
      }}
      f.noWayDep = false;
      // 기본 그래프 경로는 apron → runway 접속점까지.
      // Departure 항공기는 항상 Runway End Point까지 이어지도록,
      // Runway path의 endPx를 추가로 연결해 준다.
      const pts = pathIndices.map(i => g.nodes[i]);
      if (r && r.endPx && Array.isArray(r.endPx) && r.endPx.length === 2) {{
        const last = pts[pts.length - 1];
        const dx = r.endPx[0] - last[0];
        const dy = r.endPx[1] - last[1];
        // 이미 거의 같은 위치면 중복 추가는 피한다.
        if (Math.hypot(dx, dy) > 1e-3) pts.push(r.endPx);
      }}
      return pts;
    }}

    function updateAllFlightPaths() {{
      if (!state.flights || !state.flights.length) {{ draw(); return; }}
      state.flights.forEach(f => {{
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        if (f.noWayArr || f.noWayDep) f.timeline = null;
        else {{
          const tl = typeof buildArrivalTimeline === 'function' ? buildArrivalTimeline(f) : null;
          if (tl && tl.length) f.timeline = tl;
        }}
      }});
      if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
      // 경로만 갱신: Allocation/바차트/주기장 배치는 건드리지 않고 리스트·캔버스만 갱신
      if (typeof renderFlightList === 'function') renderFlightList(true);
      draw();
    }}

    function drawPathJunctions() {{
      let g = null;
      if (state.taxiways && state.taxiways.length) {{
        g = buildPathGraph();
      }}
      if (!g || !g.junctions.length) return;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const r = Math.max(4, CELL_SIZE * 0.35);
      ctx.fillStyle = '#22c55e';
      ctx.strokeStyle = '#14532d';
      ctx.lineWidth = 1.5;
      g.junctions.forEach(p => {{
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }});
      ctx.fillStyle = '#0f172a';
      ctx.font = (Math.max(7, CELL_SIZE * 0.18)) + 'px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      (g.edges || []).forEach(e => {{
        if (e.dist >= REVERSE_COST || e.dist < 1e-6) return;
        const a = g.nodes[e.from], b = g.nodes[e.to];
        if (!a || !b) return;
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        ctx.fillText(Math.round(e.dist).toString(), mx, my);
      }});
      ctx.restore();
    }}

    function drawFlightPathHighlight() {{
      const sel = state.selectedObject;
      if (!sel || sel.type !== 'flight' || !sel.obj) return;
      const f = sel.obj;
      if (f.noWayArr) return;
      const pathPts = getPathForFlight(f);
      if (!pathPts || pathPts.length < 2) return;
      let totalDist = 0;
      for (let i = 0; i < pathPts.length - 1; i++) totalDist += pathDist(pathPts[i], pathPts[i+1]);
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 10;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(pathPts[0][0], pathPts[0][1]);
      for (let i = 1; i < pathPts.length; i++) ctx.lineTo(pathPts[i][0], pathPts[i][1]);
      ctx.stroke();
      let cx = 0, cy = 0;
      pathPts.forEach(p => {{ cx += p[0]; cy += p[1]; }});
      cx /= pathPts.length; cy /= pathPts.length;
      const badgeText = 'VTT: ' + Math.round(totalDist);
      ctx.font = 'bold ' + Math.max(10, CELL_SIZE * 0.4) + 'px system-ui';
      const tw = ctx.measureText(badgeText).width;
      const bh = CELL_SIZE * 0.5, pad = CELL_SIZE * 0.2, r = 4;
      const bw = tw + pad*2;
      const bx = cx - bw/2, by = cy - bh/2 - 4;
      ctx.fillStyle = 'rgba(239, 68, 68, 0.95)';
      ctx.strokeStyle = '#b91c1c';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.lineTo(bx + bw - r, by);
      ctx.arcTo(bx + bw, by, bx + bw, by + r, r);
      ctx.lineTo(bx + bw, by + bh - r);
      ctx.arcTo(bx + bw, by + bh, bx + bw - r, by + bh, r);
      ctx.lineTo(bx + r, by + bh);
      ctx.arcTo(bx, by + bh, bx, by + bh - r, r);
      ctx.lineTo(bx, by + r);
      ctx.arcTo(bx, by, bx + r, by, r);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeText, cx, cy - 4);
      ctx.restore();
    }}

    function drawDeparturePathHighlight() {{
      const sel = state.selectedObject;
      if (!sel || sel.type !== 'flight' || !sel.obj) return;
      const f = sel.obj;
      if (f.noWayDep) return;
      const pathPts = getPathForFlightDeparture(f);
      if (!pathPts || pathPts.length < 2) return;
      let totalDist = 0;
      for (let i = 0; i < pathPts.length - 1; i++) totalDist += pathDist(pathPts[i], pathPts[i+1]);
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.setLineDash([16, 12]);
      ctx.beginPath();
      ctx.moveTo(pathPts[0][0], pathPts[0][1]);
      for (let i = 1; i < pathPts.length; i++) ctx.lineTo(pathPts[i][0], pathPts[i][1]);
      ctx.stroke();
      let cx = 0, cy = 0;
      pathPts.forEach(p => {{ cx += p[0]; cy += p[1]; }});
      cx /= pathPts.length; cy /= pathPts.length;
      const badgeText = 'VTT: ' + Math.round(totalDist);
      ctx.font = 'bold ' + Math.max(10, CELL_SIZE * 0.4) + 'px system-ui';
      const tw = ctx.measureText(badgeText).width;
      const bh = CELL_SIZE * 0.5, pad = CELL_SIZE * 0.2, r = 4;
      const bw = tw + pad*2;
      const bx = cx - bw/2, by = cy - bh/2 - 4;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.lineTo(bx + bw - r, by);
      ctx.arcTo(bx + bw, by, bx + bw, by + r, r);
      ctx.lineTo(bx + bw, by + bh - r);
      ctx.arcTo(bx + bw, by + bh, bx + bw - r, by + bh, r);
      ctx.lineTo(bx + r, by + bh);
      ctx.arcTo(bx, by + bh, bx, by + bh - r, r);
      ctx.lineTo(bx, by + r);
      ctx.arcTo(bx, by, bx + r, by, r);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeText, cx, cy - 4);
      ctx.restore();
    }}

    function drawFlights2D() {{
      if (!state.flights.length) return;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const tSec = state.simTimeSec;
      state.flights.forEach(f => {{
        const hasNoWay = f.noWayArr || f.noWayDep;
        if (hasNoWay) {{
          // No way: 항공기 위치는 그리지 않고, 링크가 끊기는 지점(스탠드)에만 No way 딱지
          if (!f.standId) return;
          const stand = (state.pbbStands || []).find(s => s.id === f.standId) || (state.remoteStands || []).find(s => s.id === f.standId);
          if (!stand) return;
          const x = stand.x2 != null && stand.y2 != null ? stand.x2 : cellToPixel(stand.col || 0, stand.row || 0)[0];
          const y = stand.x2 != null && stand.y2 != null ? stand.y2 : cellToPixel(stand.col || 0, stand.row || 0)[1];
          const badgeH = CELL_SIZE * 0.85;
          const badgePad = CELL_SIZE * 0.3;
          let label = 'No way';
          if (f.noWayArr && !f.noWayDep) label = 'No way (Arr)';
          else if (!f.noWayArr && f.noWayDep) label = 'No way (Dep)';
          ctx.save();
          ctx.font = 'bold ' + Math.round(badgeH * 0.62) + 'px system-ui';
          const textW = ctx.measureText(label).width;
          const badgeW = textW + badgePad * 2;
          const bx = x - badgeW / 2;
          const by = y - badgeH - 8;
          const r = badgeH * 0.35;
          ctx.fillStyle = 'rgba(220, 38, 38, 0.92)';
          ctx.strokeStyle = 'rgba(185, 28, 28, 0.9)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(bx + r, by);
          ctx.lineTo(bx + badgeW - r, by);
          ctx.arcTo(bx + badgeW, by, bx + badgeW, by + r, r);
          ctx.lineTo(bx + badgeW, by + badgeH - r);
          ctx.arcTo(bx + badgeW, by + badgeH, bx + badgeW - r, by + badgeH, r);
          ctx.lineTo(bx + r, by + badgeH);
          ctx.arcTo(bx, by + badgeH, bx, by + badgeH - r, r);
          ctx.lineTo(bx, by + r);
          ctx.arcTo(bx, by, bx + r, by, r);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = '#ffffff';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, x, by + badgeH / 2);
          ctx.restore();
          return;
        }}
        const pose = getFlightPoseAtTime(f, tSec);
        if (!pose) return;
        const x = pose.x, y = pose.y, dx = pose.dx, dy = pose.dy;
        const len = Math.hypot(dx, dy) || 1;
        const nx = dx / len, ny = dy / len;
        let scale = 1.0;
        const code = (f.code || '').toUpperCase();
        if (code === 'A' || code === 'B') scale = 0.8;
        else if (code === 'C') scale = 1.0;
        else if (code === 'D') scale = 1.2;
        else if (code === 'E') scale = 1.4;
        else if (code === 'F') scale = 1.6;
        const size = CELL_SIZE * scale;
        ctx.save();
        ctx.translate(x, y);
        const ang = Math.atan2(ny, nx);
        ctx.rotate(ang);
        ctx.fillStyle = '#ff2f92';
        ctx.beginPath();
        ctx.moveTo(size * 0.6, 0);
        ctx.lineTo(-size * 0.5, size * 0.35);
        ctx.lineTo(-size * 0.3, 0);
        ctx.lineTo(-size * 0.5, -size * 0.35);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }});
      ctx.restore();
    }}

    function ensureSimLoop() {{
      if (ensureSimLoop._running) return;
      ensureSimLoop._running = true;
      ensureSimLoop._lastTs = null;
      function tick(ts) {{
        if (ensureSimLoop._lastTs == null) ensureSimLoop._lastTs = ts;
        const dt = (ts - ensureSimLoop._lastTs) / 1000; // 초 단위 경과 시간
        ensureSimLoop._lastTs = ts;
        if (state.simPlaying && state.simDurationSec > state.simStartSec) {{
          const speed = state.simSpeed != null ? state.simSpeed : 1;
          state.simTimeSec = Math.min(state.simTimeSec + Math.max(0, dt) * speed, state.simDurationSec);
          const slider = document.getElementById('flightSimSlider');
          const label = document.getElementById('flightSimTimeLabel');
          if (slider) slider.value = state.simTimeSec;
          if (label) label.textContent = formatSecondsToHHMMSS(state.simTimeSec);
          try {{ draw(); }} catch(e) {{}}
          if (typeof update3DScene === 'function') update3DScene();
        }}
        window.requestAnimationFrame(tick);
      }}
      window.requestAnimationFrame(tick);
    }}

    // ---- Aircraft helpers (from Information.json) ----
    const AIRCRAFT_TYPES = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers && INFORMATION.tiers.aircraft && Array.isArray(INFORMATION.tiers.aircraft.types)) ? INFORMATION.tiers.aircraft.types : [];
    const AIRCRAFT_BY_ID = {{}};
    AIRCRAFT_TYPES.forEach(a => {{ AIRCRAFT_BY_ID[a.id || a.name] = a; }});
    function getAircraftInfoByType(typeId) {{
      return AIRCRAFT_BY_ID[typeId] || null;
    }}
    function getCodeForAircraft(typeId) {{
      const a = getAircraftInfoByType(typeId);
      return (a && a.icao) ? a.icao : 'C';
    }}
    function populateAircraftSelect(sel) {{
      if (!sel) return;
      const opts = AIRCRAFT_TYPES.map(a => '<option value="' + escapeHtml(String(a.id || a.name || '')) + '">' + escapeHtml(a.name || a.id || '') + '</option>').join('');
      sel.innerHTML = opts || '<option value="A320">Airbus A320</option>';
      if (!opts && sel.options.length) sel.value = 'A320';
      else if (sel.options.length) sel.value = sel.options[0].value;
    }}

    // ---- Flight UI wiring ----
    (function initFlightUI() {{
      const arrDepEl = document.getElementById('flightArrDep');
      const dwellEl = document.getElementById('flightDwell');
      const minDwellEl = document.getElementById('flightMinDwell');
      const addBtn = document.getElementById('btnAddFlight');
      const playBtn = document.getElementById('btnPlayFlights');
      const pauseBtn = document.getElementById('btnPauseFlights');
      const resetBtn = document.getElementById('btnResetFlights');
      const simSlider = document.getElementById('flightSimSlider');
      const simTimeLabel = document.getElementById('flightSimTimeLabel');
      const speedSelect = document.getElementById('flightSpeed');
      const timeInputEl = document.getElementById('flightTime');
      const aircraftEl = document.getElementById('flightAircraftType');
      const velEl = document.getElementById('flightVelocity');
      const regEl = document.getElementById('flightReg');
      const layoutNameInput = document.getElementById('layoutName');
      const saveLayoutBtn = document.getElementById('btnSaveLayout');
      const layoutMsgEl = document.getElementById('layoutMessage');
      const layoutLoadListEl = document.getElementById('layoutLoadList');
      const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
      if (!arrDepEl) return;
      populateAircraftSelect(aircraftEl);

      // 현재 이미 생성된 Flight들의 SIBT(timeMin) 중 최대값 + 10분을 기본 SIBT로 사용
      function getDefaultSibtMinutes() {{
        let maxT = 0;
        (state.flights || []).forEach(f => {{
          if (f && typeof f.timeMin === 'number' && isFinite(f.timeMin)) {{
            if (f.timeMin > maxT) maxT = f.timeMin;
          }}
        }});
        return maxT + 10;
      }}
      if (dwellEl) {{
        const syncDwell = () => {{
          const isArr = arrDepEl.value === 'Arr';
          dwellEl.disabled = !isArr;
          if (!isArr) dwellEl.value = dwellEl.value || 0;
        }};
        arrDepEl.addEventListener('change', syncDwell);
        syncDwell();
      }}
      if (minDwellEl) {{
        const syncMinDwell = () => {{
          const isArr = arrDepEl.value === 'Arr';
          minDwellEl.disabled = !isArr;
          if (!isArr) minDwellEl.value = minDwellEl.value || 0;
        }};
        arrDepEl.addEventListener('change', syncMinDwell);
        syncMinDwell();
      }}
      const TOKEN_NODE_ORDER = ['runway','taxiway','apron','terminal'];
      function getRunwayOptions() {{
        const list = [];
        (state.runways || []).forEach(r => list.push({{ id: r.id, name: (r.name || '').trim() || 'Runway' }}));
        (state.taxiways || []).filter(t => t.pathType === 'runway' || (t.name || '').toLowerCase().includes('runway')).forEach(t => list.push({{ id: t.id, name: (t.name || '').trim() || 'Runway' }}));
        return list;
      }}
      function fillTokenSelects(flightCode) {{
        const runwaySel = document.getElementById('tokenRunwaySelect');
        const termSel = document.getElementById('tokenTerminalSelect');
        if (runwaySel) {{
          const opts = getRunwayOptions();
          runwaySel.innerHTML = '<option value="">Random</option>' + opts.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
        }}
        if (termSel) {{
          const terms = (state.terminals || []).map(t => ({{ id: t.id, name: (t.name || '').trim() || 'Terminal' }}));
          termSel.innerHTML = '<option value="">Random</option>' + terms.map(o => '<option value="' + (o.id || '').replace(/"/g, '&quot;') + '">' + (o.name || o.id || '').replace(/</g, '&lt;') + '</option>').join('');
        }}
      }}
      function updateTokenPanesVisibility(nodes) {{
        const arr = Array.isArray(nodes) ? nodes : TOKEN_NODE_ORDER;
        ['runway','taxiway','apron','terminal'].forEach((node, i) => {{
          const el = document.getElementById('tokenObject' + node.charAt(0).toUpperCase() + node.slice(1));
          if (el) el.style.display = arr.indexOf(node) >= 0 ? 'block' : 'none';
        }});
      }}
      // 화면 상단 Global Update 버튼: 현재 상태 기반으로 주요 뷰/계산을 모두 다시 실행
      if (globalUpdateBtn) {{
        globalUpdateBtn.addEventListener('click', function() {{
          try {{
            if (typeof syncPanelFromState === 'function') syncPanelFromState();
            if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
            else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
            if (typeof renderFlightList === 'function') renderFlightList();
            if (typeof renderFlightGantt === 'function') renderFlightGantt();
            if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
            if (typeof draw === 'function') draw();
          }} catch (e) {{
            console.error('Global update error', e);
          }}
        }});
      }}
      function applyTokenNodesFromCheckboxes() {{
        const nodes = [];
        TOKEN_NODE_ORDER.forEach((node, i) => {{
          const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
          if (cb && cb.checked) nodes.push(node);
          else return;
        }});
        return nodes;
      }}
      function setTokenCheckboxesFromNodes(nodes) {{
        const arr = Array.isArray(nodes) ? nodes : [];
        TOKEN_NODE_ORDER.forEach((node, i) => {{
          const cb = document.getElementById('token' + node.charAt(0).toUpperCase() + node.slice(1));
          if (cb) cb.checked = arr.indexOf(node) >= 0;
        }});
        updateTokenPanesVisibility(arr.length ? arr : TOKEN_NODE_ORDER);
      }}
      ['Runway','Taxiway','Apron','Terminal'].forEach((name, i) => {{
        const cb = document.getElementById('token' + name);
        if (!cb) return;
        cb.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          if (!f.token) f.token = {{ nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null }};
          if (this.checked) {{
            f.token.nodes = TOKEN_NODE_ORDER.slice(0, i + 1);
            setTokenCheckboxesFromNodes(f.token.nodes);
          }} else {{
            f.token.nodes = TOKEN_NODE_ORDER.slice(0, i);
            setTokenCheckboxesFromNodes(f.token.nodes);
          }}
          updateTokenPanesVisibility(f.token.nodes);
          rebuildSelectedFlightTimeline();
        }});
      }});
      const tokenRunwaySel = document.getElementById('tokenRunwaySelect');
      const tokenTerminalSel = document.getElementById('tokenTerminalSelect');
      if (tokenRunwaySel) tokenRunwaySel.addEventListener('change', function() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (!f.token) f.token = {{ nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null }};
        f.token.runwayId = this.value || null;
        rebuildSelectedFlightTimeline();
      }});
      if (tokenTerminalSel) tokenTerminalSel.addEventListener('change', function() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (!f.token) f.token = {{ nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null }};
        f.token.terminalId = this.value || null;
        rebuildSelectedFlightTimeline();
      }});
      // Flight 탭 내 Schedule / Configuration 서브탭 전환
      const flightSubtabButtons = document.querySelectorAll('.flight-subtab');
      const flightPaneSchedule = document.getElementById('flightPaneSchedule');
      const flightPaneConfig = document.getElementById('flightPaneConfig');
      if (flightSubtabButtons && flightPaneSchedule && flightPaneConfig) {{
        flightSubtabButtons.forEach(btn => {{
          btn.addEventListener('click', function() {{
            const target = this.getAttribute('data-flight-subtab') || 'schedule';
            flightSubtabButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            if (target === 'config') {{
              flightPaneSchedule.style.display = 'none';
              flightPaneConfig.style.display = 'block';
            }} else {{
              flightPaneSchedule.style.display = 'block';
              flightPaneConfig.style.display = 'none';
            }}
          }});
        }});
      }}
      if (addBtn) {{
        addBtn.addEventListener('click', function() {{
          const networkErrors = validateNetworkForFlights();
          if (networkErrors.length) {{
            updateFlightError(networkErrors);
            alert('Flight를 생성할 수 없습니다:\\n' + networkErrors.join('\\n'));
            return;
          }}
          let timeStr = (document.getElementById('flightTime').value || '').trim();
          if (!timeStr) {{
            const defMin = getDefaultSibtMinutes();
            timeStr = formatMinutesToHHMMSS(defMin);
            if (timeInputEl) timeInputEl.value = timeStr;
          }}
          const timeMin = parseTimeToMinutes(timeStr);
          const aircraftType = (document.getElementById('flightAircraftType').value || 'A320').trim();
          const code = getCodeForAircraft(aircraftType);
          const vel = parseFloat(document.getElementById('flightVelocity').value) || 15;
          const reg = (document.getElementById('flightReg').value || '').trim();
          let dwellMin = parseFloat(document.getElementById('flightDwell').value);
          let minDwellMin = parseFloat(document.getElementById('flightMinDwell').value);
          dwellMin = (typeof dwellMin === 'number' && !isNaN(dwellMin) && dwellMin >= 0) ? dwellMin : 0;
          minDwellMin = (typeof minDwellMin === 'number' && !isNaN(minDwellMin) && minDwellMin >= 0) ? minDwellMin : 0;
          dwellMin = Math.max(20, dwellMin);
          minDwellMin = Math.max(20, minDwellMin);
          if (minDwellMin > dwellMin) minDwellMin = dwellMin;
          // Arr/Dep은 하나의 Flight(Arr+Dep)를 표현하는 개념으로, 내부적으로는 Arr 기준으로만 관리
          const arrDep = 'Arr';
          const f = {{
            id: id(),
            arrDep,
            timeMin,
            aircraftType,
            code,
            velocity: Math.max(1, vel),
            reg,
            dwellMin,
            minDwellMin,
            timeline: null,
            token: {{ nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null }}
          }};
          if (typeof getPathForFlight === 'function') getPathForFlight(f);
          // Taxiway / Apron 경로가 없어도 Flight는 생성할 수 있도록, 경로 오류는 경고만 남기고 계속 진행
          let tl = null;
          if (typeof buildArrivalTimeline === 'function') {{
            tl = buildArrivalTimeline(f);
          }}
          if (!tl || !tl.length) {{
            // 경로가 없으면 timeline은 null로 두고, 우측 패널에만 경고 메시지 표시
            updateFlightError('참고: 해당 네트워크에서 유효한 Taxiway / Apron 경로를 찾지 못했습니다. (시뮬레이션 경로는 그려지지 않을 수 있습니다.)');
          }} else {{
            f.timeline = tl;
          }}
          state.flights.push(f);
          recomputeSimDuration();
          renderFlightList();
          // 다음 Flight 추가를 위한 기본 SIBT 인풋 갱신 (Max SIBT + 10분)
          if (timeInputEl) {{
            const nextDef = getDefaultSibtMinutes();
            timeInputEl.value = formatMinutesToHHMMSS(nextDef);
          }}
          updateFlightError('');
        }});
      }}
      // Flight 객체 선택 시 패널 인풋에 값 반영
      function syncFlightPanelFromSelection() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        // Arr/Dep은 UI에서 선택하지 않으며, 모든 Flight는 Arr+스탠드 점유(Dwell)를 포함하는 구조
        if (arrDepEl) arrDepEl.value = 'Arr';
        if (dwellEl) {{
          dwellEl.disabled = false;
          dwellEl.value = f.dwellMin || 0;
        }}
        if (minDwellEl) {{
          minDwellEl.disabled = false;
          minDwellEl.value = f.minDwellMin != null ? f.minDwellMin : 0;
        }}
        if (timeInputEl) timeInputEl.value = formatMinutesToHHMMSS(f.timeMin);
        if (aircraftEl) {{
          if (f.aircraftType && AIRCRAFT_BY_ID[f.aircraftType]) aircraftEl.value = f.aircraftType;
          else {{
            const match = AIRCRAFT_TYPES.find(a => a.icao === (f.code || 'C'));
            aircraftEl.value = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
          }}
        }}
        if (velEl) velEl.value = f.velocity != null ? f.velocity : 15;
        if (regEl) regEl.value = f.reg || '';
        if (!f.token) f.token = {{ nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null }};
        fillTokenSelects(f.code);
        setTokenCheckboxesFromNodes(f.token.nodes);
        if (tokenRunwaySel) tokenRunwaySel.value = f.token.runwayId || '';
        if (tokenTerminalSel) tokenTerminalSel.value = f.token.terminalId || '';
      }}
      // selection 변경 시마다 동기화될 수 있도록 hook
      const origSyncPanel = syncPanelFromState;
      syncPanelFromState = function() {{
        origSyncPanel();
        if (activeTab === 'flight') syncFlightPanelFromSelection();
      }};
      // Flight 설정 인풋 변경 시 선택된 Flight 객체에 반영 + 경로 재계산
      function rebuildSelectedFlightTimeline() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        const isArr = f.arrDep !== 'Dep';
        if (isArr && f.noWayArr) {{
          updateFlightError('경로 없음(No Way): 도착 경로를 찾을 수 없습니다.');
          f.timeline = null;
          draw();
          return;
        }}
        if (!isArr && f.noWayDep) {{
          updateFlightError('경로 없음(No Way): 출발 경로를 찾을 수 없습니다.');
          f.timeline = null;
          draw();
          return;
        }}
        const tl = isArr ? buildArrivalTimeline(f) : buildDepartureTimeline(f);
        if (!tl || !tl.length) {{
          updateFlightError('해당 네트워크에서 유효한 경로를 찾을 수 없습니다. (설정 변경 후)');
          return;
        }}
        f.timeline = tl;
        recomputeSimDuration();
        renderFlightList();
      }}
      if (arrDepEl) {{
        arrDepEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.arrDep = this.value === 'Dep' ? 'Dep' : 'Arr';
          if (dwellEl) {{
            dwellEl.disabled = f.arrDep !== 'Arr';
            if (f.arrDep !== 'Arr') {{
              f.dwellMin = 0;
              dwellEl.value = 0;
            }} else {{
              f.dwellMin = parseFloat(dwellEl.value) || 0;
            }}
          }}
          if (minDwellEl) {{
            minDwellEl.disabled = f.arrDep !== 'Arr';
            if (f.arrDep !== 'Arr') {{
              f.minDwellMin = 0;
              minDwellEl.value = 0;
            }} else {{
              f.minDwellMin = Math.max(0, parseFloat(minDwellEl.value) || 0);
              minDwellEl.value = f.minDwellMin;
            }}
          }}
          rebuildSelectedFlightTimeline();
        }});
      }}
      if (timeInputEl) {{
        timeInputEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          const mins = parseTimeToMinutes(this.value || '0');
          f.timeMin = mins;
          this.value = formatMinutesToHHMMSS(mins);
          rebuildSelectedFlightTimeline();
        }});
      }}
      if (aircraftEl) {{
        aircraftEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.aircraftType = this.value || 'A320';
          f.code = getCodeForAircraft(f.aircraftType);
          rebuildSelectedFlightTimeline();
        }});
      }}
      if (velEl) {{
        velEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          const v = parseFloat(this.value) || 15;
          f.velocity = Math.max(1, v);
          this.value = f.velocity;
          rebuildSelectedFlightTimeline();
        }});
      }}
      if (regEl) {{
        regEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.reg = this.value || '';
          renderFlightList();
          updateObjectInfo();
        }});
      }}
      if (dwellEl) {{
        dwellEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          let v = parseFloat(this.value);
          v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
          let dwell = Math.max(20, v);
          let minDwell = f.minDwellMin != null ? f.minDwellMin : dwell;
          minDwell = Math.max(20, minDwell);
          if (minDwell > dwell) minDwell = dwell;
          f.dwellMin = dwell;
          f.minDwellMin = minDwell;
          this.value = f.dwellMin;
          if (minDwellEl) minDwellEl.value = f.minDwellMin;
          rebuildSelectedFlightTimeline();
        }});
      }}
      if (minDwellEl) {{
        minDwellEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          let dwell = f.dwellMin != null ? f.dwellMin : 0;
          dwell = Math.max(20, dwell);
          let v = parseFloat(this.value);
          v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
          let minDwell = Math.max(20, v);
          if (minDwell > dwell) minDwell = dwell;
          f.dwellMin = dwell;
          f.minDwellMin = minDwell;
          if (dwellEl) dwellEl.value = f.dwellMin;
          this.value = f.minDwellMin;
          renderFlightList();
        }});
      }}
      if (playBtn) {{
        playBtn.addEventListener('click', function() {{
          const errs = validateNetworkForFlights();
          if (errs.length) {{
            state.simPlaying = false;
            updateFlightError(errs);
            alert('시뮬레이션을 재생할 수 없습니다:\\n' + errs.join('\\n'));
            return;
          }}
          if (!state.flights.length) {{
            updateFlightError('등록된 Flight가 없습니다.');
            alert('등록된 Flight가 없습니다.');
            return;
          }}
          // 재생 시작 시, 가장 이른 Flight 시간부터 시작
          let earliest = Infinity;
          state.flights.forEach(f => {{
            if (f.timeline && f.timeline.length) {{
              const t0 = f.timeline[0].t;
              if (t0 < earliest) earliest = t0;
            }}
          }});
          if (!isFinite(earliest)) earliest = state.simStartSec;
          state.simTimeSec = Math.max(state.simStartSec, Math.min(state.simDurationSec, earliest));
          if (simSlider) simSlider.value = state.simTimeSec;
          if (simTimeLabel) simTimeLabel.textContent = formatSecondsToHHMMSS(state.simTimeSec);
          state.simPlaying = true;
          ensureSimLoop();
        }});
      }}
      if (pauseBtn) {{
        pauseBtn.addEventListener('click', function() {{
          state.simPlaying = false;
        }});
      }}
      if (resetBtn) {{
        resetBtn.addEventListener('click', function() {{
          state.simPlaying = false;
          state.simTimeSec = state.simStartSec;
          if (simSlider) simSlider.value = state.simTimeSec;
          if (simTimeLabel) simTimeLabel.textContent = formatSecondsToHHMMSS(state.simTimeSec);
          try {{ draw(); }} catch(e) {{}}
          if (typeof update3DScene === 'function') update3DScene();
        }});
      }}
      if (simSlider) {{
        simSlider.addEventListener('input', function() {{
          const secs = parseFloat(this.value);
          if (!isNaN(secs)) {{
            state.simTimeSec = Math.max(state.simStartSec, Math.min(state.simDurationSec, secs));
            if (simTimeLabel) simTimeLabel.textContent = formatSecondsToHHMMSS(state.simTimeSec);
            try {{ draw(); }} catch(e) {{}}
            if (typeof update3DScene === 'function') update3DScene();
          }}
        }});
      }}
      if (speedSelect) {{
        speedSelect.addEventListener('change', function() {{
          const v = parseFloat(this.value);
          state.simSpeed = !isNaN(v) && v > 0 ? v : 1;
        }});
        const v0 = parseFloat(speedSelect.value);
        state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : 20;
      }}
      // Layout Save / Load: data/Layout_storage 에 이름별 저장·로드 (API)
      function setLayoutMessage(msg, isError) {{
        if (!layoutMsgEl) return;
        layoutMsgEl.textContent = msg || '';
        layoutMsgEl.style.color = isError ? '#f97316' : '#9ca3af';
      }}
      if (saveLayoutBtn) {{
        saveLayoutBtn.addEventListener('click', function() {{
          const name = (layoutNameInput && layoutNameInput.value || '').trim();
          if (!name) {{
            setLayoutMessage('저장명을 입력하세요.', true);
            return;
          }}
          if (name.toLowerCase() === 'default_layout' || name.toLowerCase() === 'current_layout') {{
            setLayoutMessage('해당 이름으로 저장할 수 없습니다.', true);
            return;
          }}
          try {{
            if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
            const data = serializeCurrentLayout();
            const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
            fetch(apiBase + '/api/save-layout', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ layout: data, name: name }})
            }}).then(function(r) {{
              if (r.ok) {{
                if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
                setLayoutMessage('Saved to Layout_storage as "' + name + '.json"', false);
              }} else setLayoutMessage('저장 실패 (status ' + r.status + ') — python run_app.py로 실행 후 http://127.0.0.1:8501 접속', true);
            }}).catch(function(e) {{
              setLayoutMessage('연결 실패: ' + (e && e.message) + ' — python run_app.py로 실행 후 http://127.0.0.1:8501 접속', true);
            }});
          }} catch (e) {{
            console.error(e);
            setLayoutMessage('레이아웃을 저장할 수 없습니다.', true);
          }}
        }});
      }}
      const runSimBtn = document.getElementById('btnRunSimulation');
      if (runSimBtn) {{
        runSimBtn.addEventListener('click', function() {{
          try {{
            if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
            const data = serializeCurrentLayout();
            const layoutName = (state.currentLayoutName && state.currentLayoutName.trim()) || (INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
            const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
            if (layoutMsgEl) {{ layoutMsgEl.textContent = '시뮬레이션 실행 중...'; layoutMsgEl.style.color = '#9ca3af'; }}
            fetch(apiBase + '/api/run-simulation', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ layout: data, layoutName: layoutName }})
            }}).then(function(r) {{
              if (!r.ok) throw new Error('시뮬레이션 실패');
              return r.json();
            }}).then(function(result) {{
              if (!result) return;
              applyLayoutObject(result);
              resizeCanvas();
              reset2DView();
              syncPanelFromState();
              if (typeof draw === 'function') draw();
              if (typeof update3DScene === 'function') update3DScene();
              // 레이아웃 제목은 기존 이름 유지 (Simulation으로 바꾸지 않음)
              if (layoutMsgEl) {{ layoutMsgEl.textContent = '시뮬레이션 완료.'; layoutMsgEl.style.color = '#9ca3af'; }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = '연결 실패: ' + ((e && e.message) || '시뮬레이션 실패') + ' — python run_app.py로 실행 후 http://127.0.0.1:8501 접속'; layoutMsgEl.style.color = '#f97316'; }}
            }});
          }} catch (e) {{
            if (layoutMsgEl) {{ layoutMsgEl.textContent = '오류: ' + (e && e.message); layoutMsgEl.style.color = '#f97316'; }}
          }}
        }});
      }}
      // Save / Load sub-tabs
      function switchLayoutTab(tabId) {{
        document.querySelectorAll('.layout-save-load-tab').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.layout-save-load-pane').forEach(p => p.classList.remove('active'));
        const btn = document.querySelector('.layout-save-load-tab[data-sltab="' + tabId + '"]');
        const pane = document.getElementById('layout-' + tabId + '-pane');
        if (btn) btn.classList.add('active');
        if (pane) pane.classList.add('active');
        if (tabId === 'load') fetchAndRefreshLayoutList();
      }}
      document.querySelectorAll('.layout-save-load-tab').forEach(btn => {{
        btn.addEventListener('click', function() {{ switchLayoutTab(this.getAttribute('data-sltab')); }});
      }});
      function getLayoutApiBase() {{
        if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
        try {{ if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin; }} catch(e) {{}}
        return '';
      }}
      function fetchAndRefreshLayoutList() {{
        if (!layoutLoadListEl) return;
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">목록 불러오는 중...</div>';
        const apiBase = getLayoutApiBase();
        fetch(apiBase + '/api/list-layouts').then(function(r) {{
          if (!r.ok) throw new Error('API 연결 실패 (status ' + r.status + ')');
          return r.json();
        }}).then(function(data) {{
          const names = (data && data.names) ? data.names : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
          refreshLayoutLoadList(names);
        }}).catch(function(e) {{
          layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#f97316;">연결 실패: ' + (e && e.message) + '</div><div style="font-size:10px;color:#9ca3af;margin-top:4px;">python run_app.py 로 실행 후 http://127.0.0.1:8501 접속</div>';
        }});
      }}
      function refreshLayoutLoadList(namesFromApi) {{
        if (!layoutLoadListEl) return;
        const names = namesFromApi != null ? (Array.isArray(namesFromApi) ? namesFromApi : []) : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
        if (!names.length) {{
          layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">저장된 레이아웃이 없습니다.</div>';
          return;
        }}
        const reserved = {{ 'default_layout': true, 'current_layout': true }};
        layoutLoadListEl.innerHTML = names.map(function(name) {{
          const n = (name || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
          const showDel = !reserved[(name || '').toLowerCase()];
          const delBtn = showDel ? '<button type="button" class="layout-load-delete" title="삭제" data-name="' + (name || '').replace(/"/g, '&quot;') + '">×</button>' : '';
          return '<div class="layout-load-item" data-name="' + (name || '').replace(/"/g, '&quot;') + '"><span class="layout-load-name">' + n + '</span>' + delBtn + '</div>';
        }}).join('');
        layoutLoadListEl.querySelectorAll('.layout-load-item').forEach(function(el) {{
          const name = el.getAttribute('data-name');
          el.addEventListener('click', function(ev) {{
            if (ev.target && ev.target.classList && ev.target.classList.contains('layout-load-delete')) return;
            if (!name) return;
            var apiBase = getLayoutApiBase();
            if (layoutMsgEl) {{ layoutMsgEl.textContent = '불러오는 중...'; layoutMsgEl.style.color = '#9ca3af'; }}
            fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(name)).then(function(r) {{
              if (!r.ok) throw new Error('not_found');
              return r.json();
            }}).then(function(obj) {{
              if (!obj || typeof obj !== 'object') {{ throw new Error('invalid_response'); }}
              try {{
                applyLayoutObject(obj);
                resizeCanvas();
                reset2DView();
                syncPanelFromState();
                if (typeof draw === 'function') draw();
                if (typeof update3DScene === 'function') update3DScene();
                if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
                if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Loaded \"' + name + '\"'; layoutMsgEl.style.color = '#9ca3af'; }}
              }} catch (err) {{
                console.error('applyLayoutObject error', err);
                throw err;
              }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = '불러오기 실패: ' + ((e && e.message) || name || '') + ' — python run_app.py로 실행 후 http://127.0.0.1:8501 접속'; layoutMsgEl.style.color = '#f97316'; }}
            }});
          }});
          el.querySelector('.layout-load-delete') && el.querySelector('.layout-load-delete').addEventListener('click', function(ev) {{
            ev.stopPropagation();
            const n = this.getAttribute('data-name');
            if (!n) return;
            const apiBase = getLayoutApiBase();
            fetch(apiBase + '/api/delete-layout', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ name: n }})
            }}).then(function(r) {{
              if (!r.ok) return r.json().then(function(d) {{ throw new Error(d.error || '삭제 실패'); }});
              return fetch(apiBase + '/api/list-layouts').then(function(r2) {{ return r2.json(); }});
            }}).then(function(data) {{
              if (data && data.names) refreshLayoutLoadList(data.names);
              if (layoutMsgEl) {{ layoutMsgEl.textContent = '삭제됨.'; layoutMsgEl.style.color = '#9ca3af'; }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = ((e && e.message) || '삭제 실패') + ' — python run_app.py로 실행 후 http://127.0.0.1:8501 접속'; layoutMsgEl.style.color = '#f97316'; }}
            }});
          }});
        }});
      }}
      // 페이지 로드 시 API 연결 확인 (405/404 시 안내 배너 표시)
      fetch((getLayoutApiBase() || '') + '/api/list-layouts').then(function(r) {{
        if (r.ok) return;
        var banner = document.getElementById('api-warning-banner');
        if (banner) banner.style.display = 'block';
      }}).catch(function() {{
        var banner = document.getElementById('api-warning-banner');
        if (banner) banner.style.display = 'block';
      }});
    }})();

    document.getElementById('btnTerminalDraw').addEventListener('click', function() {{
      // Draw를 시작/종료할 때는 기존 객체 선택 해제
      state.selectedObject = null;
      if (state.terminalDrawingId) {{
        const t = state.terminals.find(x => x.id === state.terminalDrawingId);
        if (t && !t.closed && t.vertices.length >= 3) {{
          t.closed = true;
          // 완료 시 Taxiway 중심선과의 겹침 검사
          if (terminalOverlapsAnyTaxiway(t)) {{
            alert('이 Apron/Terminal은 Taxiway 중심선과 겹칩니다. 다른 위치에 배치해주세요.');
            state.terminals = state.terminals.filter(term => term.id !== t.id);
          }}
        }}
        state.terminalDrawingId = null;
        syncPanelFromState();
        draw();
        return;
      }}
      const nameBase = document.getElementById('terminalName').value.trim() || 'Terminal ' + (state.terminals.length + 1);
      const floorsEl = document.getElementById('terminalFloors');
      const f2fEl = document.getElementById('terminalFloorToFloor');
      let floors = floorsEl ? parseInt(floorsEl.value, 10) : 1;
      let f2f = f2fEl ? Number(f2fEl.value) : 4;
      floors = Math.max(1, floors || 1);
      f2f = Math.max(0.5, f2f || 4);
      const totalH = floors * f2f;
      const term = {{ id: id(), name: nameBase, vertices: [], closed: false, floors, floorToFloor: f2f, floorHeight: totalH, departureCapacity: 0, arrivalCapacity: 0 }};
      pushUndo();
      state.terminals.push(term);
      state.currentTerminalId = term.id;
      state.terminalDrawingId = term.id;
      syncPanelFromState();
      draw();
      if (typeof renderFlightList === 'function') renderFlightList();
      if (typeof renderFlightGantt === 'function') renderFlightGantt();
    }});

    document.getElementById('btnTaxiwayDraw').addEventListener('click', function() {{
      // Draw를 시작/종료할 때는 기존 객체 선택 해제
      state.selectedObject = null;
      if (state.taxiwayDrawingId) {{
        const tw = state.taxiways.find(x => x.id === state.taxiwayDrawingId);
        if (tw && tw.vertices.length >= 2) {{
          // 완료 시 터미널과의 겹침 검사
          if (taxiwayOverlapsAnyTerminal(tw)) {{
            alert('이 Taxiway는 Terminal과 겹칩니다. 다른 경로로 그려주세요.');
            pushUndo();
            state.taxiways = state.taxiways.filter(t => t.id !== tw.id);
          }}
          state.taxiwayDrawingId = null;
          syncPanelFromState();
          if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
          if (scene3d) update3DScene();
        }}
        return;
      }}
      const pathTypeEl = document.getElementById('pathType');
      const pathType = pathTypeEl ? (pathTypeEl.value || 'taxiway') : 'taxiway';
      const rawName = document.getElementById('taxiwayName').value.trim();
      const nameBase = rawName || (pathType === 'runway'
        ? 'Runway'
        : (pathType === 'runway_exit' ? 'Runway Exit TW' : ('Taxiway ' + (state.taxiways.length + 1))));
      const inputWidth = Number(document.getElementById('taxiwayWidth').value);
      const baseWidth = pathType === 'runway' ? 60 : 15;
      const widthVal = Math.max(10, Math.min(100, inputWidth || baseWidth));
      const modeVal = document.getElementById('taxiwayDirectionMode').value || 'both';
      const maxExitInput = document.getElementById('taxiwayMaxExitVel');
      const maxExitVelocity = (pathType === 'runway_exit' && maxExitInput)
        ? (function() {{ const mv = Number(maxExitInput.value); return isFinite(mv) && mv > 0 ? mv : null; }})()
        : null;
      const depPosInput = document.getElementById('runwayDepPointPos');
      const depPosRaw = depPosInput ? Number(depPosInput.value) : 0;
      const depPointPos = pathType === 'runway'
        ? (isFinite(depPosRaw) ? Math.max(0, Math.min(500, depPosRaw)) : 0)
        : undefined;
      const taxiway = {{ id: id(), name: nameBase, vertices: [], width: widthVal, direction: modeVal, pathType, maxExitVelocity, depPointPos }};
      pushUndo();
      state.taxiways.push(taxiway);
      state.taxiwayDrawingId = taxiway.id;
      syncPanelFromState();
      if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
    }});
    document.getElementById('runwayDirection').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'runway') {{
        state.selectedObject.obj.direction = this.value || 'both';
        updateObjectInfo();
        draw();
        if (scene3d) update3DScene();
      }}
    }});
    const runwayDirInPaneEl = document.getElementById('runwayDirectionInTaxiwayPane');
    if (runwayDirInPaneEl) runwayDirInPaneEl.addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'runway') {{
        state.selectedObject.obj.direction = this.value || 'both';
        updateObjectInfo();
        draw();
        if (typeof scene3d !== 'undefined' && scene3d) update3DScene();
      }}
    }});
    document.getElementById('btnRunwayDraw').addEventListener('click', function() {{
      // Draw를 시작/종료할 때는 기존 객체 선택 해제
      state.selectedObject = null;
      if (state.runwayDrawing) {{
        // cancel current drawing
        state.runwayDrawing = false;
        state.runwayTemp = null;
        syncPanelFromState();
        draw();
        return;
      }}
      state.runwayDrawing = true;
      state.runwayTemp = null;
      syncPanelFromState();
      draw();
    }});

    panelToggle.addEventListener('click', function() {{
      panel.classList.toggle('collapsed');
      this.textContent = panel.classList.contains('collapsed') ? '▶' : '◀';
    }});

    function renderObjectList() {{
      if (!objectListEl) return;
      const mode = settingModeSelect.value;
      const seen = {{}};
      const nameCount = {{}};
      function uniqueTitle(baseName) {{
        nameCount[baseName] = (nameCount[baseName] || 0) + 1;
        return nameCount[baseName] > 1 ? baseName + ' (' + nameCount[baseName] + ')' : baseName;
      }}
      const items = [];
      if (mode === 'terminal') {{
        state.terminals.forEach((t, idx) => {{
          if (seen['terminal_' + t.id]) return;
          seen['terminal_' + t.id] = true;
          const areaM2 = t.vertices && t.vertices.length >= 3 ? polygonAreaM2(t.vertices) : 0;
          const floors = t.floors != null ? Math.max(1, parseInt(t.floors, 10) || 1) : 1;
          const f2fRaw = t.floorToFloor != null ? Number(t.floorToFloor) : (t.floorHeight != null ? Number(t.floorHeight) : 4);
          const f2f = Math.max(0.5, f2fRaw || 4);
          const floorH = t.floorHeight != null ? Number(t.floorHeight) || (floors * f2f) : (floors * f2f);
          const dep = t.departureCapacity != null ? t.departureCapacity : 0;
          const arr = t.arrivalCapacity != null ? t.arrivalCapacity : 0;
          const baseName = (t.name && t.name.trim()) ? t.name.trim() : ('Terminal ' + (idx + 1));
          items.push({{
            type: 'terminal',
            id: t.id,
            title: uniqueTitle('Terminal | ' + baseName),
            tag: 'Height ' + floorH.toFixed(1) + ' m',
            details:
              'Area: ' + areaM2.toFixed(1) + ' m²' +
              '<br>Height: ' + floorH.toFixed(1) + ' m' +
              '<br>Floors: ' + floors +
              '<br>Total floor area: ' + (areaM2 * floors).toFixed(1) + ' m²' +
              '<br>Departure: ' + dep +
              '<br>Arrival: ' + arr
          }});
        }});
      }} else if (mode === 'pbb') {{
        state.pbbStands.forEach((pbb, idx) => {{
          if (seen['pbb_' + pbb.id]) return;
          seen['pbb_' + pbb.id] = true;
          const baseName = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : ('Contact Stand ' + (idx + 1));
          items.push({{
            type: 'pbb',
            id: pbb.id,
            title: uniqueTitle('Contact Stand | ' + baseName),
            tag: 'Category ' + (pbb.category || 'C'),
            details: 'Edge cell: (' + pbb.edgeCol + ',' + pbb.edgeRow + ')'
          }});
        }});
      }} else if (mode === 'remote') {{
        state.remoteStands.forEach((st, idx) => {{
          if (seen['remote_' + st.id]) return;
          seen['remote_' + st.id] = true;
          items.push({{
            type: 'remote',
            id: st.id,
            title: uniqueTitle('Remote stand | ' + (idx + 1)),
            tag: 'Category ' + (st.category || 'C'),
            details: 'Cell: (' + st.col + ',' + st.row + ')'
          }});
        }});
      }} else if (mode === 'taxiway') {{
        state.taxiways.forEach((tw, idx) => {{
          if (seen['taxiway_' + tw.id]) return;
          seen['taxiway_' + tw.id] = true;
          const baseName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Taxiway ' + (idx + 1));
          const dirVal = getTaxiwayDirection(tw);
          const dirLabel = dirVal === 'clockwise' ? 'CW' : (dirVal === 'counter_clockwise' ? 'CCW' : 'Both');
          let lengthM = 0;
          if (tw.vertices && tw.vertices.length >= 2) {{
            for (let i = 1; i < tw.vertices.length; i++) {{
              const v0 = tw.vertices[i - 1];
              const v1 = tw.vertices[i];
              const dx = v1.col - v0.col;
              const dy = v1.row - v0.row;
              lengthM += CELL_SIZE * Math.hypot(dx, dy);
            }}
          }}
          const widthDefault = tw.pathType === 'runway' ? 60 : 15;
          const widthVal = tw.width != null ? tw.width : widthDefault;
          const serTw = serializeTaxiwayWithEndpoints(tw);
          const startStr = serTw.start_point != null ? '(' + serTw.start_point.col + ',' + serTw.start_point.row + ')' : '—';
          const endStr = serTw.end_point != null ? '(' + serTw.end_point.col + ',' + serTw.end_point.row + ')' : '—';
          items.push({{
            type: 'taxiway',
            id: tw.id,
            title: uniqueTitle('Taxiway | ' + baseName),
            tag: dirLabel,
            details:
              'Length: ' + lengthM.toFixed(0) + ' m' +
              '<br>Points: ' + tw.vertices.length +
              '<br>Width: ' + widthVal + ' m' +
              '<br>Start point: ' + startStr +
              '<br>End point: ' + endStr
          }});
        }});
      }} else if (mode === 'apronTaxiway') {{
        state.apronLinks.forEach((lk, idx) => {{
          if (seen['apron_' + lk.id]) return;
          seen['apron_' + lk.id] = true;
          const pbb = state.pbbStands.find(p => p.id === lk.pbbId);
          const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
          const title = 'Link ' + (idx + 1);
          const details = 'Stand: ' + (pbb && pbb.name ? pbb.name : lk.pbbId) +
            ', Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId);
          items.push({{
            type: 'apronLink',
            id: lk.id,
            title: uniqueTitle('Apron–Taxiway | ' + title),
            tag: 'Apron–Taxiway',
            details
          }});
        }});
      }} else if (mode === 'runway') {{
        state.runways.forEach((rw, idx) => {{
          if (seen['runway_' + rw.id]) return;
          seen['runway_' + rw.id] = true;
          const baseName = (rw.name && rw.name.trim()) ? rw.name.trim() : ('Runway ' + (idx + 1));
          const dirVal = getTaxiwayDirection(rw);
          const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter CW' : 'Both');
          const serRw = serializeRunwayWithEndpoints(rw);
          const startStr = serRw.start_point != null ? '(' + serRw.start_point.col + ',' + serRw.start_point.row + ')' : '—';
          const endStr = serRw.end_point != null ? '(' + serRw.end_point.col + ',' + serRw.end_point.row + ')' : '—';
          items.push({{
            type: 'runway',
            id: rw.id,
            title: uniqueTitle('Runway | ' + baseName),
            tag: dirLabel,
            details:
              'Length: ' + rw.length.toFixed(0) + ' m' +
              '<br>Width: ' + rw.width.toFixed(0) + ' m' +
              '<br>Start: (' + rw.startCol + ',' + rw.startRow + ')' +
              '<br>Start point: ' + startStr +
              '<br>End point: ' + endStr
          }});
        }});
      }}
      if (!items.length) {{
        objectListEl.innerHTML = '<div class="obj-item">No objects yet.</div>';
        return;
      }}
      objectListEl.innerHTML = items.map(it => (
        '<div class="obj-item" data-type="' + it.type + '" data-id="' + it.id + '">' +
          '<div class="obj-item-header">' +
            '<span class="obj-item-title">' + it.title + '</span>' +
            '<span class="obj-item-tag">' + it.tag + '</span>' +
            '<button type="button" class="obj-item-delete" title="Delete">×</button>' +
          '</div>' +
          '<div class="obj-item-details">' + it.details + '</div>' +
        '</div>'
      )).join('');
      const listItems = objectListEl.querySelectorAll('.obj-item');
      listItems.forEach(el => {{
        const type = el.getAttribute('data-type');
        const id = el.getAttribute('data-id');
        el.querySelector('.obj-item-delete').addEventListener('click', function(ev) {{
          ev.stopPropagation();
          pushUndo();
          if (type === 'terminal') state.terminals = state.terminals.filter(t => t.id !== id);
          else if (type === 'pbb') state.pbbStands = state.pbbStands.filter(p => p.id !== id);
          else if (type === 'remote') state.remoteStands = state.remoteStands.filter(r => r.id !== id);
          else if (type === 'taxiway') state.taxiways = state.taxiways.filter(tw => tw.id !== id);
          else if (type === 'apronLink') state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
          else if (type === 'runway') state.runways = state.runways.filter(rw => rw.id !== id);
          else if (type === 'flight') state.flights = state.flights.filter(f => f.id !== id);
          if (state.selectedObject && state.selectedObject.type === type && state.selectedObject.id === id)
            state.selectedObject = null;
          if (type === 'terminal' && state.currentTerminalId === id) {{
            state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
            if (state.terminalDrawingId === id) state.terminalDrawingId = null;
          }}
          if (type === 'taxiway' && state.taxiwayDrawingId === id) state.taxiwayDrawingId = null;
          syncPanelFromState();
          updateObjectInfo();
          if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        }});
        el.addEventListener('click', function(ev) {{
          if (ev.target.classList.contains('obj-item-delete')) return;
          const typ = this.getAttribute('data-type');
          const idr = this.getAttribute('data-id');
          let obj = null;
          if (typ === 'terminal') obj = state.terminals.find(t => t.id === idr);
          else if (typ === 'pbb') obj = state.pbbStands.find(p => p.id === idr);
          else if (typ === 'remote') obj = state.remoteStands.find(r => r.id === idr);
          else if (typ === 'taxiway') obj = state.taxiways.find(tw => tw.id === idr);
          else if (typ === 'apronLink') obj = state.apronLinks.find(lk => lk.id === idr);
          else if (typ === 'runway') obj = state.runways.find(rw => rw.id === idr);
          else if (typ === 'flight') obj = state.flights.find(f => f.id === idr);
          if (!obj) return;
          const wasExpanded = this.classList.contains('expanded');
          listItems.forEach(li => li.classList.remove('selected', 'expanded'));
          if (!wasExpanded) {{
            this.classList.add('selected', 'expanded');
            state.selectedObject = {{ type: typ, id: idr, obj }};
            if (typ === 'terminal') state.currentTerminalId = idr;
            syncPanelFromState();
            updateObjectInfo();
          }} else {{
            objectInfoEl.textContent = 'Select an object on the grid or from the list.';
          }}
          draw();
        }});
      }});
      if (state.selectedObject) {{
        const sel = objectListEl.querySelector('.obj-item[data-type="' + state.selectedObject.type + '"][data-id="' + state.selectedObject.id + '"]');
        if (sel) sel.classList.add('selected', 'expanded');
      }}
    }}

    function updateObjectInfo() {{
      if (state.selectedObject) {{
        const o = state.selectedObject.obj;
        if (state.selectedObject.type === 'terminal') {{
          const areaM2 = o.vertices && o.vertices.length >= 3 ? polygonAreaM2(o.vertices) : 0;
          const floors = o.floors != null ? Math.max(1, parseInt(o.floors, 10) || 1) : 1;
          const f2fRaw = o.floorToFloor != null ? Number(o.floorToFloor) : (o.floorHeight != null ? Number(o.floorHeight) : 4);
          const f2f = Math.max(0.5, f2fRaw || 4);
          const floorH = o.floorHeight != null ? Number(o.floorHeight) || (floors * f2f) : (floors * f2f);
          const totalArea = areaM2 * floors;
          const dep = o.departureCapacity != null ? o.departureCapacity : 0;
          const arr = o.arrivalCapacity != null ? o.arrivalCapacity : 0;
          objectInfoEl.innerHTML = '<strong>Terminal</strong><br>Name: ' + (o.name || o.id) + '<br>Vertices: ' + (o.vertices ? o.vertices.length : 0) +
            '<br>Footprint area: ' + areaM2.toFixed(1) + ' m²<br>Height: ' + floorH.toFixed(1) + ' m (Floors: ' + floors + ' × ' + f2f.toFixed(1) + ' m)' +
            '<br>Total floor area: ' + totalArea.toFixed(1) + ' m²' +
            '<br>Departure capacity: ' + dep + '<br>Arrival capacity: ' + arr;
        }} else if (state.selectedObject.type === 'pbb')
          objectInfoEl.innerHTML = '<strong>Contact Stand</strong><br>Name: ' + (o.name || '—') + '<br>Category: ' + o.category + '<br>Edge cell: (' + o.edgeCol + ',' + o.edgeRow + ')';
        else if (state.selectedObject.type === 'remote')
          objectInfoEl.innerHTML = '<strong>Remote stand</strong><br>Category: ' + o.category + '<br>Cell: (' + o.col + ',' + o.row + ')';
        else if (state.selectedObject.type === 'taxiway') {{
          const dirVal = getTaxiwayDirection(o);
          const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter Clockwise' : 'Both');
          const heading = o.pathType === 'runway' ? 'Runway' : (o.pathType === 'runway_exit' ? 'Runway Exit Taxiway' : 'Taxiway');
          const ser = serializeTaxiwayWithEndpoints(o);
          const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
          const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
          objectInfoEl.innerHTML = '<strong>' + heading + '</strong><br>Name: ' + (o.name || '—') + '<br>Direction: ' + dirLabel + '<br>Width: ' + (o.width != null ? o.width : 23) + ' m<br>Points: ' + (o.vertices ? o.vertices.length : 0) +
            '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
        }} else if (state.selectedObject.type === 'runway') {{
          const dirVal = getTaxiwayDirection(o);
          const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter CW' : 'Both');
          const ser = serializeRunwayWithEndpoints(o);
          const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
          const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
          objectInfoEl.innerHTML = '<strong>Runway</strong><br>Name: ' + (o.name || '—') +
            '<br>Direction: ' + dirLabel +
            '<br>Length: ' + (o.length != null ? o.length.toFixed(0) : '—') + ' m' +
            '<br>Width: ' + (o.width != null ? o.width.toFixed(0) : '—') + ' m' +
            '<br>Start cell: (' + o.startCol + ',' + o.startRow + ')' +
            '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
        }} else if (state.selectedObject.type === 'flight') {{
          const dir = o.arrDep === 'Dep' ? 'Departure' : 'Arrival';
          const sibt = formatMinutesToHHMMSS(o.timeMin != null ? o.timeMin : 0);
          const sobt = formatMinutesToHHMMSS((o.timeMin != null ? o.timeMin : 0) + (o.dwellMin != null ? o.dwellMin : 0));
          const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(o.aircraftType) : null;
          const acName = ac ? (ac.name || ac.id || '') : (o.aircraftType || '—');
          const codeIcao = (ac && ac.icao) ? ac.icao : (o.code || '—');
          const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
          const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';
          objectInfoEl.innerHTML =
            '<strong>Flight</strong><br>' +
            'Type: ' + dir +
            '<br>SIBT: ' + sibt + ' &nbsp; SOBT: ' + sobt +
            '<br>Aircraft: ' + (acName || '—') +
            '<br>Code(ICAO): ' + (codeIcao || '—') + ' &nbsp; ICAO(J/H/M/L): ' + (icaoJhl || '—') + ' &nbsp; RECAT-EU: ' + (recatEu || '—') +
            '<br>Velocity: ' + (o.velocity != null ? o.velocity.toFixed(1) : '—') + ' m/s' +
            '<br>Reg: ' + (o.reg || '—') +
            '<br>Dwell (Arr only): ' + (o.dwellMin || 0) + ' min';
        }}
      }} else
        objectInfoEl.textContent = 'Select an object on the grid or from the list.';
      renderObjectList();
    }}

    function reset2DView() {{
      let w = 0, h = 0;
      const rect = container.getBoundingClientRect();
      w = Number(rect.width) || 0;
      h = Number(rect.height) || 0;
      if (w <= 0 || h <= 0) {{
        if (canvas) {{
          w = canvas.clientWidth || canvas.width || 800;
          h = canvas.clientHeight || canvas.height || 600;
        }} else {{
          w = 800;
          h = 600;
        }}
      }}
      w = Math.max(1, w);
      h = Math.max(1, h);
      const maxX = GRID_COLS * CELL_SIZE;
      const maxY = GRID_ROWS * CELL_SIZE;
      const scaleX = w / maxX;
      const scaleY = h / maxY;
      const s = Math.min(scaleX, scaleY) * 0.9;
      state.scale = s;
      state.panX = (w - maxX * s) / 2;
      state.panY = (h - maxY * s) / 2;
      draw();
    }}

    function resizeCanvas() {{
      const rect = container.getBoundingClientRect();
      const w = Math.max(1, Number(rect.width) || 0);
      const h = Math.max(1, Number(rect.height) || 0);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      try {{ draw(); }} catch(e) {{}}
    }}

    function drawGrid() {{
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#303030';
      ctx.fillRect(0, 0, w, h);
      ctx.restore();
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
      const GRID_MAJOR = 10;
      for (let c = 0; c <= GRID_COLS; c++) {{
        const x = c * CELL_SIZE;
        ctx.strokeStyle = (c % GRID_MAJOR === 0) ? 'rgba(255,255,255,0.85)' : 'rgba(140,140,140,0.18)';
        ctx.lineWidth = (c % GRID_MAJOR === 0) ? 1.2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, maxY);
        ctx.stroke();
      }}
      for (let r = 0; r <= GRID_ROWS; r++) {{
        const y = r * CELL_SIZE;
        ctx.strokeStyle = (r % GRID_MAJOR === 0) ? 'rgba(255,255,255,0.85)' : 'rgba(140,140,140,0.18)';
        ctx.lineWidth = (r % GRID_MAJOR === 0) ? 1.2 : 1;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(maxX, y);
        ctx.stroke();
      }}
      ctx.fillStyle = '#aaa';
      ctx.font = '10px system-ui';
      ctx.fillText('0,0', 4, 2);
      // red dot at exact grid center
      const cx = (GRID_COLS * CELL_SIZE) / 2;
      const cy = (GRID_ROWS * CELL_SIZE) / 2;
      ctx.beginPath();
      ctx.fillStyle = '#ef4444';
      ctx.arc(cx, cy, CELL_SIZE * 0.15, 0, Math.PI * 2);
      ctx.fill();
      // hovered grid intersection: light red dot at crossing point
      if (state.hoverCell != null) {{
        const hc = state.hoverCell;
        const hx = hc.col * CELL_SIZE;
        const hy = hc.row * CELL_SIZE;
        ctx.beginPath();
        ctx.fillStyle = 'rgba(248, 113, 113, 0.45)';
        ctx.arc(hx, hy, CELL_SIZE * 0.2, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.restore();
    }}

    function drawTerminals() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      state.terminals.forEach(term => {{
        if (term.vertices.length === 0) return;
        const selected = state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.id === term.id;
        ctx.lineWidth = selected ? 3 : 2;
        ctx.strokeStyle = selected ? '#60a5fa' : '#38bdf8';
        ctx.fillStyle = selected ? 'rgba(56,189,248,0.2)' : 'rgba(56,189,248,0.12)';
        ctx.beginPath();
        for (let i = 0; i < term.vertices.length; i++) {{
          const [x,y] = cellToPixel(term.vertices[i].col, term.vertices[i].row);
          if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }}
        if (term.closed) {{ ctx.closePath(); ctx.fill(); }}
        ctx.stroke();
        // 선택된 터미널은 점선 컨투어로 한 번 더 강조
        if (selected) {{
          ctx.save();
          ctx.setLineDash([8, 6]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(248,250,252,0.85)';
          ctx.beginPath();
          for (let i = 0; i < term.vertices.length; i++) {{
            const [x,y] = cellToPixel(term.vertices[i].col, term.vertices[i].row);
            if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
          }}
          if (term.closed) ctx.closePath();
          ctx.stroke();
          ctx.restore();
        }}
        // 층고 표시: 겹쳐 그리지 않고 터미널 중심에 높이(m) 텍스트만 표시
        if (term.closed && term.vertices.length > 0) {{
          const floors = term.floors != null ? Math.max(1, parseInt(term.floors, 10) || 1) : 1;
          const f2f = term.floorToFloor != null ? Number(term.floorToFloor) : (term.floorHeight != null ? Number(term.floorHeight) : 4);
          const floorH = term.floorHeight != null ? Number(term.floorHeight) || (floors * f2f) : (floors * f2f);
          let cx = 0, cy = 0;
          term.vertices.forEach(v => {{
            const [px, py] = cellToPixel(v.col, v.row);
            cx += px; cy += py;
          }});
          cx /= term.vertices.length;
          cy /= term.vertices.length;
          ctx.fillStyle = 'rgba(56,189,248,0.95)';
          ctx.font = '11px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(floorH.toFixed(1) + ' m', cx, cy);
        }}
        term.vertices.forEach((v, i) => {{
          const [x,y] = cellToPixel(v.col, v.row);
          ctx.beginPath();
          ctx.fillStyle = i === 0 ? '#f97316' : '#e5e7eb';
          ctx.arc(x, y, 4, 0, Math.PI*2);
          ctx.fill();
        }});
      }});
      ctx.restore();
    }}

    function drawPBBs() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      state.pbbStands.forEach(pbb => {{
        const endSize = getStandSizeMeters(pbb.category || 'C');
        const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
        ctx.strokeStyle = sel ? '#fb923c' : '#f97316';
        ctx.lineWidth = sel ? 4 : 3;
        ctx.beginPath();
        ctx.moveTo(pbb.x1, pbb.y1);
        ctx.lineTo(pbb.x2, pbb.y2);
        ctx.stroke();
        const ex = pbb.x2, ey = pbb.y2;
        const angle = getPBBStandAngle(pbb);
        ctx.fillStyle = sel ? 'rgba(34,197,94,0.35)' : 'rgba(22,163,74,0.18)';
        ctx.strokeStyle = sel ? '#4ade80' : '#22c55e';
        ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.save();
        ctx.translate(ex, ey);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
        ctx.fill();
        ctx.stroke();
        if (sel) {{
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(248,250,252,0.9)';
          ctx.beginPath();
          ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
          ctx.stroke();
          ctx.restore();
        }}
        // Category label inside contact stand
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(pbb.category || 'C', 0, 0);
        ctx.restore();
      }});
      ctx.restore();
    }}

    function drawRemoteStands() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      state.remoteStands.forEach(st => {{
        const [cx,cy] = cellToPixel(st.col, st.row);
        const size = getStandSizeMeters(st.category || 'C');
        const sel = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
        ctx.fillStyle = sel ? 'rgba(34,197,94,0.35)' : 'rgba(22,163,74,0.18)';
        ctx.strokeStyle = sel ? '#4ade80' : '#22c55e';
        ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.rect(cx-size/2, cy-size/2, size, size);
        ctx.fill();
        ctx.stroke();
        if (sel) {{
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(248,250,252,0.9)';
          ctx.beginPath();
          ctx.rect(cx-size/2, cy-size/2, size, size);
          ctx.stroke();
          ctx.restore();
        }}
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(st.category, cx, cy);
      }});
      ctx.restore();
    }}

    function renderRunwaySeparation() {{
      const panel = document.getElementById('rwySepPanel');
      if (!panel) return;
      const runways = (state.taxiways || []).filter(tw => tw.pathType === 'runway');
      if (!runways.length) {{
        panel.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No runway-path taxiways. Mode Taxiway &amp; Runway에서 Type을 Runway로 설정한 객체를 먼저 생성하세요.</div>';
        return;
      }}
      if (!state.activeRwySepId || !runways.some(r => r.id === state.activeRwySepId)) {{
        state.activeRwySepId = runways[0].id;
      }}
      const active = runways.find(r => r.id === state.activeRwySepId) || runways[0];
      const cfg = rsepGetConfigForRunway(active);
      const stdKey = cfg.standard || 'ICAO';
      const cats = RSEP_STD_CATS[stdKey] || RSEP_STD_CATS['ICAO'];
      const mode = cfg.mode || 'MIX';
      const seq = cfg.activeSeq || (RSEP_MODE_SEQS[mode] && RSEP_MODE_SEQS[mode][0]) || 'ARR→ARR';
      const seqType = RSEP_SEQ_TYPES[seq] || 'matrix';
      const seqMeta = rsepGetSeqMeta(seq);

      let html = '';
      html += '<div class="rwysep-rwy-bar">';
      html += '<div class="rwysep-rwy-tabs">';
      runways.forEach(rw => {{
        const isActive = rw.id === active.id;
        const label = escapeHtml(rw.name || ('Runway ' + rw.id));
        html += '<button type="button" class="rwysep-rwy-btn' + (isActive ? ' active' : '') + '" data-rwy-id="' + rw.id + '">' + label + '</button>';
      }});
      html += '</div></div>';

      // 기본 탭을 Separation Timeline으로 먼저 표시 (Save/Load 탭 스타일 재사용)
      const activeSub = state.activeRwySepSubtab || 'timeline';
      html += '<div class="layout-save-load-tabs" style="margin-top:8px;">';
      ['timeline','input'].forEach(key => {{
        const isActive = activeSub === key;
        const label = key === 'timeline' ? 'Separation Timeline' : 'Separation Input';
        html += '<button type="button" class="layout-save-load-tab rwysep-subtab-btn' + (isActive ? ' active' : '') + '" data-subtab="' + key + '">' + label + '</button>';
      }});
      html += '</div>';

      // --- Subtab: Separation Input ---
      html += '<div id="rwysep-subtab-input" style="' + (activeSub === 'input' ? '' : 'display:none;') + '">';
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">Standard &amp; Mode</div>';
      html += '<div class="rwysep-row">';
      html += '<label style="font-size:11px;color:#9ca3af;">Standard&nbsp;</label>';
      html += '<select id="rwysep-standard">';
      html += '<option value="ICAO"' + (stdKey === 'ICAO' ? ' selected' : '') + '>ICAO (J/H/M/L)</option>';
      html += '<option value="RECAT-EU"' + (stdKey === 'RECAT-EU' ? ' selected' : '') + '>RECAT-EU (A~F)</option>';
      html += '</select>';
      html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Mode&nbsp;</label>';
      html += '<select id="rwysep-mode">';
      ['ARR','DEP','MIX'].forEach(m => {{
        const txt = m === 'ARR' ? 'Arrivals only' : (m === 'DEP' ? 'Departures only' : 'Mixed (Arr/Dep)');
        html += '<option value="' + m + '"' + (mode === m ? ' selected' : '') + '>' + txt + '</option>';
      }});
      html += '</select>';
      html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Seq&nbsp;</label>';
      html += '<select id="rwysep-seq">';
      (RSEP_MODE_SEQS[mode] || []).forEach(s => {{
        const lbl = s;
        html += '<option value="' + s + '"' + (seq === s ? ' selected' : '') + '>' + lbl + '</option>';
      }});
      html += '</select>';
      html += '</div></div>';

      if (seqMeta) {{
        html += '<div class="rwysep-block" style="margin-top:4px;">';
        html += '<div class="rwysep-label">Concept summary</div>';
        html += '<div style="font-size:10px;color:#d1d5db;line-height:1.5;background:#020617;border-radius:6px;border:1px solid #111827;padding:6px 8px;">';
        html += '<div><span style="color:#9ca3af;">Driving factor</span>&nbsp;&nbsp;: ' + escapeHtml(seqMeta.driver) + '</div>';
        html += '<div><span style="color:#9ca3af;">Reference point</span>&nbsp;: ' + escapeHtml(seqMeta.refPoint) + '</div>';
        html += '<div><span style="color:#9ca3af;">Input structure</span>: ' + escapeHtml(seqMeta.input) + '</div>';
        html += '</div>';
        html += '</div>';
      }}

      // ROT: Arr→Dep 조합에서만 표시
      if (seq === 'ARR→DEP') {{
        html += '<div class="rwysep-block">';
        html += '<div class="rwysep-label">ROT (Runway Occupancy Time, sec)</div>';

        // 색상 범례 + 채워진 ROT 개수
        const totalRot = cats.length;
        let filledRot = 0;
        cats.forEach(c => {{
          const val = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
          if (val !== '' && val != null) filledRot += 1;
        }});
        html += '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d2018;border-radius:2px;margin-right:4px;"></span><span style="color:#68d391;">&lt;90s</span></span>';
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d1a28;border-radius:2px;margin-right:4px;"></span><span style="color:#63b3ed;">90–119s</span></span>';
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:#1e1e08;border-radius:2px;margin-right:4px;"></span><span style="color:#f6e05e;">120–149s</span></span>';
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:#280d0d;border-radius:2px;margin-right:4px;"></span><span style="color:#fc8181;">≥150s</span></span>';
        const rotCountColor = filledRot === totalRot ? '#68d391' : '#9ca3af';
        html += '<span style="margin-left:4px;color:' + rotCountColor + ';">' + filledRot + '/' + totalRot + '</span>';
        html += '</div>';

        html += '<div class="rwysep-row" style="flex-wrap:wrap;">';
        cats.forEach(c => {{
          const rawVal = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
          const valStr = rawVal === null || rawVal === undefined ? '' : String(rawVal);
          const sub = rsepGetCatLabel(stdKey, c);
          const colInfo = rsepColorForValue(valStr);
          html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;">';
          html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">';
          html += 'Cat ' + c;
          if (sub) {{
            html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }}
          html += '</div>';
          html += '<input type="number" min="0" step="5" data-rwysep-rot="' + c + '" value="' + escapeHtml(valStr) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
          html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
          html += '</div>';
        }});
        html += '</div></div>';
      }}

      // Separation matrix / 1D
      if (seq === 'ARR→DEP') {{
        // For ARR→DEP, separation is effectively driven by ROT only
        html += '<div class="rwysep-block">';
        html += '<div class="rwysep-label">ROT‑based separation (sec)</div>';
        html += '<div style="font-size:10px;color:#9ca3af;line-height:1.5;">For ARR→DEP combinations, separation is determined by the ROT values above (runway occupancy time per wake category).</div>';
        html += '</div>';
      }} else {{
        html += '<div class="rwysep-block">';
        html += '<div class="rwysep-label">Separation (sec) — ' + escapeHtml(seq) + '</div>';
        if (seqType === 'matrix') {{
          const data = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMakeMatrix(cats, null);
          const total = cats.length * cats.length;
          let filled = 0;
          cats.forEach(lead => {{
            cats.forEach(trail => {{
              const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
              if (v !== '' && v != null) filled += 1;
            }});
          }});
          html += '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d2018;border-radius:2px;margin-right:4px;"></span><span style="color:#68d391;">&lt;90s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d1a28;border-radius:2px;margin-right:4px;"></span><span style="color:#63b3ed;">90–119s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#1e1e08;border-radius:2px;margin-right:4px;"></span><span style="color:#f6e05e;">120–149s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#280d0d;border-radius:2px;margin-right:4px;"></span><span style="color:#fc8181;">≥150s</span></span>';
          const countColor = filled === total ? '#68d391' : '#9ca3af';
          html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
          html += '</div>';

          html += '<div class="rwysep-matrix-wrap"><table class="rwysep-table"><thead><tr>';
          html += '<th>Lead↓ / Trail→</th>';
          cats.forEach(c => {{
            const sub = rsepGetCatLabel(stdKey, c);
            html += '<th><div style="line-height:1.2;">' + c;
            if (sub) {{
              html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(sub) + '</div>';
            }}
            html += '</div></th>';
          }});
          html += '</tr></thead><tbody>';
          cats.forEach(lead => {{
            const leadSub = rsepGetCatLabel(stdKey, lead);
            html += '<tr><td><div style="line-height:1.2;">' + lead;
            if (leadSub) {{
              html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(leadSub) + '</div>';
            }}
            html += '</div></td>';
            cats.forEach(trail => {{
              const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
              const colInfo = rsepColorForValue(v);
              html += '<td><input type="number" min="0" step="5" data-rwysep-matrix-lead="' + lead + '" data-rwysep-matrix-trail="' + trail + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" /></td>';
            }});
            html += '</tr>';
          }});
          html += '</tbody></table></div>';
        }} else {{
          const data1d = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMake1D(cats, null);
          const total = cats.length;
          let filled = 0;
          cats.forEach(cat => {{
            const v = data1d[cat] != null ? data1d[cat] : '';
            if (v !== '' && v != null) filled += 1;
          }});
          html += '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d2018;border-radius:2px;margin-right:4px;"></span><span style="color:#68d391;">&lt;90s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#0d1a28;border-radius:2px;margin-right:4px;"></span><span style="color:#63b3ed;">90–119s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#1e1e08;border-radius:2px;margin-right:4px;"></span><span style="color:#f6e05e;">120–149s</span></span>';
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:#280d0d;border-radius:2px;margin-right:4px;"></span><span style="color:#fc8181;">≥150s</span></span>';
          const countColor = filled === total ? '#68d391' : '#9ca3af';
          html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
          html += '</div>';

          html += '<div class="rwysep-row" style="flex-wrap:wrap;margin-top:4px;">';
          cats.forEach(cat => {{
            const v = data1d[cat] != null ? data1d[cat] : '';
            const colInfo = rsepColorForValue(v);
            const sub = rsepGetCatLabel(stdKey, cat);
            html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;border:1px solid #1f2937;border-radius:6px;padding:6px 8px;background:#020617;">';
            html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">Cat ' + cat;
            if (sub) {{
              html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
            }}
            html += '</div>';
            html += '<input type="number" min="0" step="5" data-rwysep-1d="' + cat + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
            html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
            html += '</div>';
          }});
          html += '</div>';
        }}
        html += '</div>';
      }}
      html += '</div>'; // end subtab input

      // --- Subtab: Separation Timeline ---
      html += '<div id="rwysep-subtab-timeline" style="' + (activeSub === 'timeline' ? '' : 'display:none;') + '">';
      html += '<div class="rwysep-block" style="margin-top:8px;">';
      html += '<div class="rwysep-label">Separation Timeline (Reg × Time)</div>';
      // 최대 약 12개 Reg 행만 한 화면에 보이고, 그 이상은 세로 스크롤
      html += '<div id="rwySepTimeWrap" style="width:100%;background:#020617;border-radius:8px;border:1px solid #1f2937;position:relative;overflow-x:auto;overflow-y:auto;margin-top:4px;max-height:calc(40px * 12 + 80px);"></div>';
      html += '<div style="font-size:9px;color:#9ca3af;margin-top:4px;">';
      html += 'Y: Reg Number · X: Time · Bars = S-series (SLDT–STOT) · Lines = E-series (ELDT–ETOT)';
      html += '</div></div>';
      html += '</div>'; // end subtab timeline

      panel.innerHTML = html;

      // draw timeline only when timeline subtab is visible (Apron Gantt 스타일, Reg × Time)
      function drawRwySeparationTimeline() {{
        if (state.activeRwySepSubtab && state.activeRwySepSubtab !== 'timeline') return;
        const wrap = panel.querySelector('#rwySepTimeWrap');
        if (!wrap) return;

        const allData = typeof computeSeparationAdjustedTimes === 'function' ? computeSeparationAdjustedTimes() : null;
        const data = allData && active && active.id != null ? allData[active.id] : null;
        if (!data || !data.events || !data.events.length) {{
          wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
          return;
        }}

        // Flight 별로 SLDT/STOT/ELDT/ETOT 모으기 (한 행 = 한 Reg)
        const byFlight = new Map();
        data.events.forEach(ev => {{
          const f = ev.flight;
          if (!f) return;
          let lane = byFlight.get(f);
          if (!lane) {{
            const reg = f.reg || f.id || '';
            lane = {{
              flight: f,
              reg,
              hasArr: false,
              hasDep: false,
              sldt: null,
              eldt: null,
              stot: null,
              etot: null
            }};
            byFlight.set(f, lane);
          }}
          if (ev.type === 'arr') {{
            lane.hasArr = true;
            lane.sldt = ev.time;
            lane.eldt = (f.eldtMin != null ? f.eldtMin : ev.time);
          }} else if (ev.type === 'dep') {{
            lane.hasDep = true;
            lane.stot = ev.time;
            lane.etot = (f.etotMin != null ? f.etotMin : ev.time);
          }}
        }});

        const lanes = Array.from(byFlight.values());
        if (!lanes.length) {{
          wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
          return;
        }}

        // 시간축: min(SLDT) - 10분, max(ETOT) + 10분 (Apron과 유사)
        let minT0 = Infinity;
        let maxT0 = -Infinity;
        lanes.forEach(ln => {{
          if (ln.sldt != null && ln.sldt < minT0) minT0 = ln.sldt;
          if (ln.etot != null && ln.etot > maxT0) maxT0 = ln.etot;
        }});
        if (!isFinite(minT0) || !isFinite(maxT0)) {{
          minT0 = data.minT;
          maxT0 = data.maxT;
        }}
        let baseMinT = Math.max(0, minT0 - 10);
        let baseMaxT = maxT0 + 10;
        if (baseMaxT <= baseMinT) baseMaxT = baseMinT + 60;
        const baseSpan = baseMaxT - baseMinT;
        const zoom = (state.rwySepTimeZoom && state.rwySepTimeZoom > 1) ? state.rwySepTimeZoom : 1;
        const span = baseSpan;
        const minT = baseMinT;
        const maxT = baseMaxT;

        function formatMinToHM(m) {{
          const hh = Math.floor(m / 60);
          const mm = Math.floor(m % 60);
          return hh + ':' + (mm < 10 ? '0' : '') + mm;
        }}

        lanes.sort((a, b) => {{
          const ta = (a.sldt ?? a.stot ?? a.eldt ?? a.etot ?? 0);
          const tb = (b.sldt ?? b.stot ?? b.eldt ?? b.etot ?? 0);
          return ta - tb;
        }});

        // 시간축 눈금 위치 (Runway Timeline 전체 공통) - 어떤 화면에서도 최대 6개까지만 표시
        const tickPositions = [];
        const axisStep = span <= 60 ? 10 : (span <= 240 ? 30 : 60);
        let tt = Math.floor(minT / axisStep) * axisStep;
        while (tt <= maxT) {{
          const leftPct = ((tt - baseMinT) / baseSpan) * 100 * zoom;
          const label = formatMinToHM(tt);
          tickPositions.push({{ leftPct, label }});
          tt += axisStep;
        }}
        if (tickPositions.length > 6) {{
          const stepTicks = Math.ceil(tickPositions.length / 6);
          const reduced = [];
          for (let i = 0; i < tickPositions.length; i += stepTicks) {{
            reduced.push(tickPositions[i]);
          }}
          const last = tickPositions[tickPositions.length - 1];
          if (reduced[reduced.length - 1] !== last) reduced.push(last);
          tickPositions.length = 0;
          Array.prototype.push.apply(tickPositions, reduced);
        }}

        // 상단 S/E 삼각형 타임라인용 데이터
        const sMarkers = [];
        const eMarkers = [];

        const rows = [];
        lanes.forEach(ln => {{
          const reg = ln.reg || '';
          const sStart = (ln.sldt != null ? ln.sldt : null);
          const sEnd = (ln.stot != null ? ln.stot : null);
          const eStart = (ln.eldt != null ? ln.eldt : null);
          const eEnd = (ln.etot != null ? ln.etot : null);

          let blocks = '';
          if (sStart != null && sEnd != null && span > 0) {{
            const s1 = Math.max(sStart, baseMinT);
            const s2 = Math.min(sEnd, baseMaxT);
            if (s2 <= s1) return;
            const leftPct = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
            const widthPct = Math.max(1, ((s2 - s1) / baseSpan) * 100 * zoom);
            const rightPct = leftPct + widthPct;
            // 상단 S계열 삼각형용 마커(시작/종료)
            sMarkers.push({{ t: sStart, leftPct, type: 'start' }});
            sMarkers.push({{ t: sEnd, leftPct: rightPct, type: 'end' }});
            // S-series: 얇은 파란 바 + 시작/종료 삼각형 (위쪽에 배치)
            blocks +=
              '<div class="rwysep-line-s" style="' +
                'left:' + leftPct + '%;' +
                'width:' + widthPct + '%;' +
              '"></div>' +
              // 시작점: 아래 방향 삼각형 (바를 향해)
              '<div class="rwysep-tri" style="' +
                'top:20%;' +
                'left:' + leftPct + '%;' +
                'border-top:6px solid #38bdf8;' +
              '"></div>' +
              // 종료점: 위 방향 삼각형
              '<div class="rwysep-tri" style="' +
                'top:20%;' +
                'left:' + rightPct + '%;' +
                'border-bottom:6px solid #38bdf8;' +
              '"></div>';
          }}
          if (eStart != null && eEnd != null && span > 0) {{
            const e1 = Math.max(eStart, baseMinT);
            const e2 = Math.min(eEnd, baseMaxT);
            if (e2 <= e1) return;
            const leftPct2 = ((e1 - baseMinT) / baseSpan) * 100 * zoom;
            const widthPct2 = Math.max(0.5, ((e2 - e1) / baseSpan) * 100 * zoom);
            const rightPct2 = leftPct2 + widthPct2;
            // 상단 E계열 삼각형용 마커(시작/종료)
            eMarkers.push({{ t: eStart, leftPct: leftPct2, type: 'start' }});
            eMarkers.push({{ t: eEnd, leftPct: rightPct2, type: 'end' }});
            // E-series: 얇은 주황 바 + 시작/종료 삼각형 (아래쪽에 배치)
            blocks +=
              '<div class="rwysep-line-e" style="' +
                'left:' + leftPct2 + '%;' +
                'width:' + widthPct2 + '%;' +
              '"></div>' +
              // 시작점: 아래 방향 삼각형
              '<div class="rwysep-tri" style="' +
                'top:54%;' +
                'left:' + leftPct2 + '%;' +
                'border-top:6px solid #fb923c;' +
              '"></div>' +
              // 종료점: 위 방향 삼각형
              '<div class="rwysep-tri" style="' +
                'top:54%;' +
                'left:' + rightPct2 + '%;' +
                'border-bottom:6px solid #fb923c;' +
              '"></div>';
          }}

          const gridLines = tickPositions.map(tp =>
            '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>'
          ).join('');

          rows.push(
            '<div class="alloc-row">' +
              '<div class="alloc-row-label">' + escapeHtml(reg) + '</div>' +
              // Runway Separation Timeline에서는 각 행 배경(트랙 배경색/테두리)을 제거
              '<div class="alloc-row-track" style="background:transparent;border:none;">' + gridLines + blocks + '</div>' +
            '</div>'
          );
        }});

        // 상단 S/E 삼각형 라인 HTML (시간순)
        sMarkers.sort((a, b) => a.t - b.t);
        eMarkers.sort((a, b) => a.t - b.t);

        const sHeadMarks = sMarkers.map(m =>
          '<div class="rwysep-tri" style="' +
            'top:60%;' +
            'left:' + m.leftPct + '%;' +
            (m.type === 'start'
              ? 'border-top:6px solid #38bdf8;'
              : 'border-bottom:6px solid #38bdf8;') +
          '"></div>'
        ).join('');

        const eHeadMarks = eMarkers.map(m =>
          '<div class="rwysep-tri" style="' +
            'top:60%;' +
            'left:' + m.leftPct + '%;' +
            (m.type === 'start'
              ? 'border-top:6px solid #fb923c;'
              : 'border-bottom:6px solid #fb923c;') +
          '"></div>'
        ).join('');

        const headHtml =
          '<div class="rwysep-head-row">' +
            '<div class="rwysep-head-label">S-series</div>' +
            '<div class="rwysep-head-track">' + sHeadMarks + '</div>' +
          '</div>' +
          '<div class="rwysep-head-row">' +
            '<div class="rwysep-head-label">E-series</div>' +
            '<div class="rwysep-head-track">' + eHeadMarks + '</div>' +
          '</div>';

        // Time axis overlay (Apron과 동일한 스타일, tickPositions 재사용)
        const axisTicks = tickPositions.map(tp =>
          '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
            '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
          '</div>'
        );
        // Runway Separation Timeline에서도 Apron과 동일하게 하단 시간축 오버레이를 사용
        const axisHtml =
          '<div class="alloc-time-axis-overlay">' +
            '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
          '</div>';

        // 많은 Reg가 있을 경우 세로 스크롤로 보이도록, 헤더는 그대로 두고 행들을 래핑
        const rowsHtml = '<div class="rwysep-rows">' + rows.join('') + '</div>';
        wrap.innerHTML = headHtml + rowsHtml + axisHtml;

        // Shift + 마우스 휠로 시간축 줌 (Runway Timeline)
        if (!wrap._rwySepZoomBound) {{
          wrap._rwySepZoomBound = true;
          wrap.addEventListener('wheel', function(e) {{
            if (!e.shiftKey) return;
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
            let z = state.rwySepTimeZoom || 1;
            z *= factor;
            if (z < 1) z = 1;
            if (z > 8) z = 8;
            state.rwySepTimeZoom = z;
            if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
          }}, {{ passive: false }});
        }}

        // 수평 스크롤 시에도 현재 시간축/배경이 다시 계산되도록 재랜더링
        if (!wrap._rwySepScrollBound) {{
          wrap._rwySepScrollBound = true;
          wrap.addEventListener('scroll', function() {{
            if (wrap._rwySepScrollRecalc) return;
            // 현재 스크롤 위치를 보존한 채로 타임라인 전체를 다시 그린다.
            const currentLeft = wrap.scrollLeft;
            wrap._rwySepScrollRecalc = true;
            drawRwySeparationTimeline();
            wrap.scrollLeft = currentLeft;
            wrap._rwySepScrollRecalc = false;
          }});
        }}
      }}

      drawRwySeparationTimeline();

      // Wiring: runway buttons
      panel.querySelectorAll('.rwysep-rwy-btn').forEach(btn => {{
        btn.addEventListener('click', function() {{
          const id = this.getAttribute('data-rwy-id');
          if (!id) return;
          state.activeRwySepId = id;
          renderRunwaySeparation();
        }});
      }});

      // Wiring: subtab buttons (Input / Timeline)
      panel.querySelectorAll('.rwysep-subtab-btn').forEach(btn => {{
        btn.addEventListener('click', function() {{
          const sub = this.getAttribute('data-subtab') || 'input';
          state.activeRwySepSubtab = sub;
          renderRunwaySeparation();
        }});
      }});

      const stdSel = panel.querySelector('#rwysep-standard');
      if (stdSel) {{
        stdSel.addEventListener('change', function() {{
          cfg.standard = this.value || 'ICAO';
          cfg.seqData = rsepMakeSeqData(cfg.standard);
          const catsNew = RSEP_STD_CATS[cfg.standard] || [];
          const rotNew = RSEP_STANDARDS[cfg.standard] && RSEP_STANDARDS[cfg.standard].ROT || {{}};
          cfg.rot = {{}};
          catsNew.forEach(c => {{ cfg.rot[c] = rotNew[c] != null ? String(rotNew[c]) : ''; }});
          renderRunwaySeparation();
        }});
      }}
      const modeSel = panel.querySelector('#rwysep-mode');
      if (modeSel) {{
        modeSel.addEventListener('change', function() {{
          cfg.mode = this.value || 'MIX';
          const seqs = RSEP_MODE_SEQS[cfg.mode] || ['ARR→ARR'];
          if (!seqs.includes(cfg.activeSeq)) cfg.activeSeq = seqs[0];
          renderRunwaySeparation();
        }});
      }}
      const seqSel = panel.querySelector('#rwysep-seq');
      if (seqSel) {{
        seqSel.addEventListener('change', function() {{
          cfg.activeSeq = this.value || 'ARR→ARR';
          renderRunwaySeparation();
        }});
      }}
      // ROT handlers
      panel.querySelectorAll('input[data-rwysep-rot]').forEach(inp => {{
        inp.addEventListener('change', function() {{
          const cat = this.getAttribute('data-rwysep-rot');
          if (!cat) return;
          cfg.rot[cat] = this.value;
          const colInfo = rsepColorForValue(this.value);
          this.style.background = colInfo.bg;
          this.style.borderColor = colInfo.border;
          this.style.color = colInfo.color;
        }});
      }});
      // Matrix handlers
      panel.querySelectorAll('input[data-rwysep-matrix-lead]').forEach(inp => {{
        inp.addEventListener('change', function() {{
          const lead = this.getAttribute('data-rwysep-matrix-lead');
          const trail = this.getAttribute('data-rwysep-matrix-trail');
          if (!lead || !trail) return;
          if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMakeMatrix(cats, null);
          if (!cfg.seqData[seq][lead]) cfg.seqData[seq][lead] = {{}};
          cfg.seqData[seq][lead][trail] = this.value;
          const colInfo = rsepColorForValue(this.value);
          this.style.background = colInfo.bg;
          this.style.borderColor = colInfo.border;
          this.style.color = colInfo.color;
        }});
      }});
      // 1D handlers
      panel.querySelectorAll('input[data-rwysep-1d]').forEach(inp => {{
        inp.addEventListener('change', function() {{
          const cat = this.getAttribute('data-rwysep-1d');
          if (!cat) return;
          if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMake1D(cats, null);
          cfg.seqData[seq][cat] = this.value;
          const colInfo = rsepColorForValue(this.value);
          this.style.background = colInfo.bg;
          this.style.borderColor = colInfo.border;
          this.style.color = colInfo.color;
        }});
      }});
    }}

    function drawTaxiways() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      state.taxiways.forEach(tw => {{
        const drawing = state.taxiwayDrawingId === tw.id;
        if (tw.vertices.length < 2 && !drawing) return;
        const isRunwayPath = tw.pathType === 'runway';
        const isRunwayExit = tw.pathType === 'runway_exit';
        const widthDefault = isRunwayPath ? 60 : 15;
        const width = tw.width != null ? tw.width : widthDefault;
        const sel = state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === tw.id;
        if (isRunwayPath || isRunwayExit) {{
          // Runway 및 Runway Exit Taxiway: 회색 계열
          ctx.strokeStyle = sel ? '#d1d5db' : '#6b7280';
          ctx.fillStyle = sel ? 'rgba(156,163,175,0.35)' : 'rgba(75,85,99,0.25)';
        }} else {{
          // 일반 Taxiway: 노란 계열
          ctx.strokeStyle = sel ? '#fbbf24' : (drawing ? '#facc15' : '#eab308');
          ctx.fillStyle = sel ? 'rgba(251,191,36,0.25)' : 'rgba(234,179,8,0.15)';
        }}
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {{
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        if (tw.vertices.length >= 2) ctx.stroke();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = (isRunwayPath || isRunwayExit)
          ? (sel ? '#e5e7eb' : '#9ca3af')
          : (sel ? '#f59e0b' : '#ca8a04');
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {{
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        if (tw.vertices.length >= 2) ctx.stroke();
        if (sel) {{
          ctx.save();
          ctx.setLineDash([8, 6]);
          ctx.lineWidth = 3;
          ctx.strokeStyle = 'rgba(248,250,252,0.9)';
          ctx.beginPath();
          for (let i = 0; i < tw.vertices.length; i++) {{
            const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }}
          ctx.stroke();
          ctx.restore();
        }}
        const dir = getTaxiwayDirection(tw);
        if (dir !== 'both' && tw.vertices.length >= 2) {{
          const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
          const totalLen = pts.reduce((acc, p, i) => acc + (i > 0 ? Math.hypot(p[0]-pts[i-1][0], p[1]-pts[i-1][1]) : 0), 0);
          const arrowSpacing = Math.max(22, Math.min(42, totalLen / 10));
          const numArrows = Math.max(2, Math.floor(totalLen / arrowSpacing));
          // 화살표 길이를 기존의 60% 수준으로 축소
          const arrLen = CELL_SIZE * 0.6;
          ctx.fillStyle = '#f59e0b';
          for (let k = 1; k <= numArrows; k++) {{
            const targetDist = totalLen * (k / (numArrows + 1));
            let acc = 0;
            let ax = pts[0][0], ay = pts[0][1];
            let angle = Math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]);
            for (let i = 1; i < pts.length; i++) {{
              const seg = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
              angle = Math.atan2(pts[i][1]-pts[i-1][1], pts[i][0]-pts[i-1][0]);
              if (acc + seg >= targetDist) {{
                const t = seg > 0 ? (targetDist - acc) / seg : 0;
                ax = pts[i-1][0] + t * (pts[i][0]-pts[i-1][0]);
                ay = pts[i-1][1] + t * (pts[i][1]-pts[i-1][1]);
                break;
              }}
              acc += seg;
            }}
            if (dir === 'counter_clockwise') angle += Math.PI;
            ctx.beginPath();
            ctx.moveTo(ax + arrLen * Math.cos(angle), ay + arrLen * Math.sin(angle));
            ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) + arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) - arrLen * 0.4 * Math.cos(angle));
            ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) - arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) + arrLen * 0.4 * Math.cos(angle));
            ctx.closePath();
            ctx.fill();
          }}
        }}
        // Runway path에 line-up point 마커 표시 (lineup_point 좌표가 우선, 없으면 depPointPos m 지점)
        if (isRunwayPath && tw.vertices.length >= 2 && (tw.lineup_point || tw.dep_point || typeof tw.depPointPos === 'number')) {{
          let px, py;
          const lup = tw.lineup_point || tw.dep_point;
          if (lup && typeof lup.col === 'number' && typeof lup.row === 'number') {{
            // 저장된 좌표 기반 (lineup_point 우선, dep_point 호환)
            const pt = cellToPixel(lup.col, lup.row);
            px = pt[0];
            py = pt[1];
          }} else {{
            const verts = tw.vertices;
            const pts = verts.map(v => cellToPixel(v.col, v.row));
            // col/row 차이를 1m 단위로 보고 depPointPos(m)를 길이에 대한 비율로 변환
            const totalLen = verts.reduce((acc, v, i) =>
              acc + (i > 0 ? Math.hypot(v.col - verts[i-1].col, v.row - verts[i-1].row) : 0), 0);
            const desired = Math.max(0, Math.min(500, tw.depPointPos || 0));
            const ratio = totalLen > 0 ? Math.max(0, Math.min(1, desired / totalLen)) : 0;
            const targetDist = totalLen * ratio;
            let acc = 0;
            px = pts[0][0]; py = pts[0][1];
            for (let i = 1; i < pts.length; i++) {{
              const seg = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
              if (acc + seg >= targetDist) {{
                const t = seg > 0 ? (targetDist - acc) / seg : 0;
                px = pts[i-1][0] + t * (pts[i][0]-pts[i-1][0]);
                py = pts[i-1][1] + t * (pts[i][1]-pts[i-1][1]);
                break;
              }}
              acc += seg;
            }}
          }}
          ctx.fillStyle = '#f97373';
          ctx.beginPath();
          ctx.arc(px, py, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#b91c1c';
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fillStyle = '#fef2f2';
          ctx.font = 'bold 9px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText('Line-up', px, py - 8);
        }}
        if ((drawing || sel) && tw.vertices.length >= 1) {{
          tw.vertices.forEach((v, i) => {{
            const [x, y] = cellToPixel(v.col, v.row);
            if (i === 0 && drawing) {{
              ctx.fillStyle = '#f97316';
              ctx.beginPath();
              ctx.arc(x, y, 7, 0, Math.PI*2);
              ctx.fill();
              ctx.strokeStyle = '#ea580c';
              ctx.lineWidth = 2;
              ctx.stroke();
              ctx.fillStyle = '#fff';
              ctx.font = 'bold 9px system-ui';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('Start', x, y - 11);
            }} else {{
              ctx.fillStyle = (i === 0 && sel) ? '#f97316' : '#e5e7eb';
              ctx.beginPath();
              ctx.arc(x, y, sel ? 5 : 4, 0, Math.PI*2);
              ctx.fill();
              if (sel) {{
                ctx.strokeStyle = '#ca8a04';
                ctx.lineWidth = 1.5;
                ctx.stroke();
              }}
            }}
          }});
        }}
      }});
      ctx.restore();
    }}

    function drawRunways() {{
      if (!state.runways.length && !state.runwayTemp) return;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      state.runways.forEach(rw => {{
        const cx = rw.cx, cy = rw.cy;
        const ux = rw.dx, uy = rw.dy;
        const len2 = rw.length / 2;
        const w2 = rw.width / 2;
        const px = -uy, py = ux; // left normal
        const p1x = cx - ux*len2 - px*w2, p1y = cy - uy*len2 - py*w2;
        const p2x = cx + ux*len2 - px*w2, p2y = cy + uy*len2 - py*w2;
        const p3x = cx + ux*len2 + px*w2, p3y = cy + uy*len2 + py*w2;
        const p4x = cx - ux*len2 + px*w2, p4y = cy - uy*len2 + py*w2;
        ctx.fillStyle = 'rgba(148,163,184,0.65)';
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p1x, p1y);
        ctx.lineTo(p2x, p2y);
        ctx.lineTo(p3x, p3y);
        ctx.lineTo(p4x, p4y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Direction arrows along runway centerline (like taxiway)
        const dirVal = getTaxiwayDirection(rw);
        if (dirVal !== 'both') {{
          const [startCol, startRow] = [rw.startCol, rw.startRow];
          const [sx, sy] = cellToPixel(startCol, startRow);
          const baseUx = ux;
          const baseUy = uy;
          const dirSign = dirVal === 'clockwise' ? 1 : -1;
          const vx = baseUx * dirSign;
          const vy = baseUy * dirSign;
          const totalLen = rw.length;
          const arrowSpacing = Math.max(80, Math.min(200, totalLen / 6));
          const arrLen = Math.max(18, Math.min(40, rw.width * 1.2));
          const numArrows = Math.floor(totalLen / arrowSpacing);
          ctx.fillStyle = '#f97316';
          ctx.strokeStyle = '#f97316';
          ctx.lineWidth = 1.2;
          for (let i = 1; i <= numArrows; i++) {{
            const d = (i * totalLen) / (numArrows + 1);
            const ax = sx + vx * d;
            const ay = sy + vy * d;
            const ang = Math.atan2(vy, vx);
            ctx.save();
            ctx.translate(ax, ay);
            ctx.rotate(ang);
            ctx.beginPath();
            ctx.moveTo(-arrLen * 0.5, -arrLen * 0.25);
            ctx.lineTo(arrLen * 0.5, 0);
            ctx.lineTo(-arrLen * 0.5, arrLen * 0.25);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
          }}
        }}
      }});
      // Runway drawing start point indicator (like taxiway)
      if (state.runwayTemp) {{
        const sx = state.runwayTemp.x;
        const sy = state.runwayTemp.y;
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0ea5e9';
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawApronTaxiwayLinks() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 6]);
      state.apronLinks.forEach(lk => {{
        const pbb = state.pbbStands.find(p => p.id === lk.pbbId);
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        if (!pbb || !tw || lk.tx == null || lk.ty == null) return;
        const [ax, ay] = [pbb.x2, pbb.y2];
        const [bx, by] = [lk.tx, lk.ty];
        ctx.strokeStyle = '#22d3ee';
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        if (state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lk.id) {{
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.lineWidth = 4;
          ctx.strokeStyle = 'rgba(248,250,252,0.95)';
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.restore();
          ctx.setLineDash([6,6]);
        }}
        // endpoint markers
        ctx.setLineDash([]);
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath(); ctx.arc(ax, ay, CELL_SIZE * 0.18, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(bx, by, CELL_SIZE * 0.18, 0, Math.PI*2); ctx.fill();
        ctx.setLineDash([6,6]);
      }});
      ctx.setLineDash([]);
      // temporary first endpoint marker
      if (state.apronLinkTemp) {{
        ctx.fillStyle = '#38bdf8';
        const t = state.apronLinkTemp;
        let px = null, py = null;
        if (t.kind === 'pbb') {{
          const pbb = state.pbbStands.find(p => p.id === t.pbbId);
          if (pbb) {{ px = pbb.x2; py = pbb.y2; }}
        }} else if (t.kind === 'taxiway') {{
          px = t.x; py = t.y;
        }}
        if (px != null && py != null) {{
          ctx.beginPath();
          ctx.arc(px, py, CELL_SIZE * 0.22, 0, Math.PI*2);
          ctx.fill();
        }}
      }}
      ctx.restore();
    }}

    function drawStandPreview() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const mode = settingModeSelect.value;
      if (mode === 'remote' && state.previewRemote) {{
        const [cx, cy] = cellToPixel(state.previewRemote.col, state.previewRemote.row);
        const category = document.getElementById('remoteCategory').value || 'C';
        const size = getStandSizeMeters(category);
        const overlap = state.previewRemote.overlap;
        ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
        ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.rect(cx - size/2, cy - size/2, size, size);
        ctx.fill();
        ctx.stroke();
      }}
      if (mode === 'pbb' && state.previewPbb) {{
        const ex = state.previewPbb.x2, ey = state.previewPbb.y2;
        const size = getStandSizeMeters(state.previewPbb.category || 'C');
        const overlap = state.previewPbb.overlap;
        const angle = getPBBStandAngle(state.previewPbb);
        ctx.fillStyle = overlap ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.25)';
        ctx.strokeStyle = overlap ? '#ef4444' : '#22c55e';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.save();
        ctx.translate(ex, ey);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.rect(-size/2, -size/2, size, size);
        ctx.fill();
        ctx.stroke();
        // Preview category label for contact stand
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(state.previewPbb.category || document.getElementById('standCategory').value || 'C', 0, 0);
        ctx.restore();
      }}
      ctx.restore();
    }}

    function draw() {{
      drawGrid();
      drawTerminals();
      drawTaxiways();
      drawRunways();
      drawPBBs();
      drawRemoteStands();
      drawApronTaxiwayLinks();
      drawStandPreview();
      drawPathJunctions();
      drawFlightPathHighlight();
      drawDeparturePathHighlight();
      drawFlights2D();
    }}

    document.addEventListener('keydown', function(ev) {{
      const el = document.activeElement;
      const inInput = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
      if (ev.ctrlKey && ev.key === 'z') {{
        if (!inInput) {{ ev.preventDefault(); undo(); }}
        return;
      }}
      if (ev.key !== 'Delete' && ev.key !== 'Backspace') return;
      if (inInput) return;
      if (!state.selectedObject) return;
      const type = state.selectedObject.type;
      const id = state.selectedObject.id;
      pushUndo();
      if (type === 'terminal') state.terminals = state.terminals.filter(t => t.id !== id);
      else if (type === 'pbb') state.pbbStands = state.pbbStands.filter(p => p.id !== id);
      else if (type === 'remote') state.remoteStands = state.remoteStands.filter(r => r.id !== id);
      else if (type === 'taxiway') state.taxiways = state.taxiways.filter(tw => tw.id !== id);
      else if (type === 'apronLink') state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
      else if (type === 'runway') state.runways = state.runways.filter(rw => rw.id !== id);
      else if (type === 'flight') state.flights = state.flights.filter(f => f.id !== id);
      else return;
      state.selectedObject = null;
      if (type === 'terminal' && state.currentTerminalId === id) {{
        state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
        if (state.terminalDrawingId === id) state.terminalDrawingId = null;
      }}
      if (type === 'taxiway' && state.taxiwayDrawingId === id) state.taxiwayDrawingId = null;
      syncPanelFromState();
      updateObjectInfo();
      if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      ev.preventDefault();
    }});

    const DRAG_THRESH = 5;
    container.addEventListener('mousedown', function(ev) {{
      if (ev.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      const mode = settingModeSelect.value;
      if (mode === 'terminal' && !state.terminalDrawingId) {{
        const vhit = hitTestTerminalVertex(wx, wy);
        if (vhit) {{
          pushUndo();
          state.dragVertex = vhit;
          const term = state.terminals.find(t => t.id === vhit.terminalId);
          if (term) {{
            state.selectedObject = {{ type: 'terminal', id: term.id, obj: term }};
            state.currentTerminalId = term.id;
            syncPanelFromState();
            updateObjectInfo();
            draw();
          }}
          return;
        }}
      }}
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const thit = hitTestTaxiwayVertex(wx, wy);
        if (thit && thit.taxiwayId === state.selectedObject.id) {{
          pushUndo();
          state.dragTaxiwayVertex = thit;
          draw();
          return;
        }}
      }}
      state.dragStart = {{ sx, sy, panX: state.panX, panY: state.panY }};
      state.isPanning = false;
    }});
    container.addEventListener('mousemove', function(ev) {{
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      const [col, row] = pixelToCell(wx, wy);
      if (coordEl) coordEl.textContent = 'cell: (' + col + ', ' + row + ')';
      const prev = state.hoverCell;
      state.hoverCell = {{ col, row }};
      const hoverChanged = !prev || prev.col !== col || prev.row !== row;
      if (state.dragVertex) {{
        const term = state.terminals.find(t => t.id === state.dragVertex.terminalId);
        if (term && term.vertices[state.dragVertex.index]) {{
          const v = term.vertices[state.dragVertex.index];
          v.col = col;
          v.row = row;
          draw();
        }}
        return;
      }}
      if (state.dragTaxiwayVertex) {{
        const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
        if (tw && tw.vertices[state.dragTaxiwayVertex.index]) {{
          const v = tw.vertices[state.dragTaxiwayVertex.index];
          v.col = col;
          v.row = row;
          draw();
          if (scene3d) update3DScene();
        }}
        return;
      }}
      if (state.dragStart) {{
        const dx = sx - state.dragStart.sx, dy = sy - state.dragStart.sy;
        if (!state.isPanning && (Math.abs(dx) > DRAG_THRESH || Math.abs(dy) > DRAG_THRESH))
          state.isPanning = true;
        if (state.isPanning) {{
          state.panX = state.dragStart.panX + dx;
          state.panY = state.dragStart.panY + dy;
          draw();
        }}
      }}
      const mode = settingModeSelect.value;
      if (!state.isPanning && !state.dragVertex && mode === 'remote') {{
        const category = document.getElementById('remoteCategory').value || 'C';
        const size = getStandSizeMeters(category);
        const [cx, cy] = cellToPixel(col, row);
        const bounds = getStandBoundsRect(cx, cy, size);
        const overlap = standOverlapsExisting(bounds);
        state.previewRemote = {{ col, row, overlap }};
        draw();
      }} else {{
        if (state.previewRemote) {{
          state.previewRemote = null;
          draw();
        }}
      }}
      // Flight hover tooltip (only in 2D view)
      if (flightTooltip && !state.isPanning && activeTab !== 'flight') {{
        let bestFlight = null;
        let bestD2 = (CELL_SIZE * 1.2) ** 2;
        const tSec = state.simTimeSec;
        state.flights.forEach(f => {{
          const pose = getFlightPoseAtTime(f, tSec);
          if (!pose || !f.reg) return;
          const dx = pose.x - wx;
          const dy = pose.y - wy;
          const d2 = dx*dx + dy*dy;
          if (d2 < bestD2) {{
            bestD2 = d2;
            bestFlight = f;
          }}
        }});
        if (bestFlight && bestFlight.reg) {{
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = bestFlight.reg;
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
        }} else {{
          flightTooltip.style.display = 'none';
        }}
      }}
      if (hoverChanged) try {{ draw(); }} catch(e) {{}}
    }});
    container.addEventListener('mouseleave', function() {{
      state.dragStart = null;
      state.hoverCell = null;
      try {{ draw(); }} catch(e) {{}}
    }});
    function hitTestPbbEnd(wx, wy) {{
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * 0.8) ** 2;
      let best = null;
      let bestD2 = maxD2;
      state.pbbStands.forEach(pbb => {{
        const ex = pbb.x2, ey = pbb.y2;
        const d2 = dist2([ex, ey], click);
        if (d2 < bestD2) {{
          bestD2 = d2;
          best = {{ id: pbb.id, x: ex, y: ey }};
        }}
      }});
      return best;
    }}

    function hitTestAnyTaxiwayVertex(wx, wy) {{
      // For Apron Taxiway links: allow connecting to any point along a taxiway polyline
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * 1.0) ** 2;
      let best = null;
      let bestD2 = maxD2;
      state.taxiways.forEach(tw => {{
        if (!tw.vertices || tw.vertices.length < 2) return;
        for (let i = 0; i < tw.vertices.length - 1; i++) {{
          const [x1, y1] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          const [x2, y2] = cellToPixel(tw.vertices[i+1].col, tw.vertices[i+1].row);
          const near = closestPointOnSegment([x1, y1], [x2, y2], click);
          if (!near) continue;
          const d2 = dist2(near, click);
          if (d2 < bestD2) {{
            bestD2 = d2;
            best = {{ taxiwayId: tw.id, x: near[0], y: near[1] }};
          }}
        }}
      }});
      return best;
    }}

    container.addEventListener('mouseup', function(ev) {{
      if (ev.button !== 0) return;
      if (state.dragVertex) {{
        state.dragVertex = null;
        return;
      }}
      if (state.dragTaxiwayVertex) {{
        const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
        if (tw && typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
        state.dragTaxiwayVertex = null;
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
        if (scene3d) update3DScene();
        draw();
        return;
      }}
      if (!state.dragStart) return;
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      if (!state.isPanning) {{
        const hit = hitTest(wx, wy);
        const mode = settingModeSelect.value;
        if (mode === 'apronTaxiway') {{
          const pbbHit = hitTestPbbEnd(wx, wy);
          const twHit = hitTestAnyTaxiwayVertex(wx, wy);
          const endpoint = pbbHit ? {{ kind: 'pbb', pbbId: pbbHit.id, x: pbbHit.x, y: pbbHit.y }} :
                            (twHit ? {{ kind: 'taxiway', taxiwayId: twHit.taxiwayId, x: twHit.x, y: twHit.y }} : null);
          if (endpoint) {{
            if (!state.apronLinkTemp) {{
              state.apronLinkTemp = endpoint;
            }} else {{
              const first = state.apronLinkTemp;
              if (first.kind !== endpoint.kind) {{
                let pbbId, taxiwayId, tx, ty;
                if (first.kind === 'pbb') {{
                  pbbId = first.pbbId;
                  taxiwayId = endpoint.taxiwayId;
                  tx = endpoint.x;
                  ty = endpoint.y;
                }} else {{
                  pbbId = endpoint.pbbId;
                  taxiwayId = first.taxiwayId;
                  tx = first.x;
                  ty = first.y;
                }}
                if (pbbId && taxiwayId) {{
                  pushUndo();
                  state.apronLinks.push({{ id: id(), pbbId, taxiwayId, tx, ty }});                  
                  syncPanelFromState();
                  if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
                }}
              }}
              state.apronLinkTemp = null;
            }}
            draw();
          }}
        }} else if (mode === 'runway') {{
          if (!state.runwayDrawing) return;
          const [col, row] = pixelToCell(wx, wy);
          const [sx, sy] = cellToPixel(col, row);
          if (!state.runwayTemp) {{
            state.runwayTemp = {{ x: sx, y: sy, col, row }};
          }} else {{
            const s = state.runwayTemp;
            const dx = sx - s.x;
            const dy = sy - s.y;
            const lenDir = Math.hypot(dx, dy);
            if (lenDir > 0.0001) {{
              const ux = dx / lenDir, uy = dy / lenDir;
              const lengthM = Math.max(50, Number(document.getElementById('runwayLength').value) || 3000);
              const widthM = Math.max(10, Number(document.getElementById('runwayWidth').value) || 45);
              ensureDefaultRunwayDirectionModes();
              const modeId = document.getElementById('runwayDirectionMode').value || (state.runwayDirectionModes[0] && state.runwayDirectionModes[0].id);
              const nameBase = document.getElementById('runwayName').value.trim() || ('Runway ' + (state.runways.length + 1));
              const cx = s.x + ux * (lengthM / 2);
              const cy = s.y + uy * (lengthM / 2);
              pushUndo();
              state.runways.push({{
                id: id(),
                name: nameBase,
                cx, cy,
                dx: ux, dy: uy,
                length: lengthM,
                width: widthM,
                startCol: s.col,
                startRow: s.row,
                directionModeId: modeId || null,
                direction: document.getElementById('runwayDirection').value || 'both'
              }});
            }}
            state.runwayTemp = null;
            state.runwayDrawing = false;
            if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
            syncPanelFromState();
            if (typeof update3DScene === 'function') update3DScene();
          }}
        }} else if (hit) {{
          state.selectedObject = hit;
          if (hit.type === 'terminal') state.currentTerminalId = hit.id;
          // 캔버스에서 클릭했을 때 해당 타입의 Mode 로 전환
          if (hit.type === 'terminal') settingModeSelect.value = 'terminal';
          else if (hit.type === 'pbb' || hit.type === 'remote') settingModeSelect.value = 'pbb';
          else if (hit.type === 'taxiway' || hit.type === 'runway' || hit.type === 'apronLink') settingModeSelect.value = 'taxiway';
          if (hit.type === 'flight' && typeof switchToTab === 'function') switchToTab('flight');
          if (hit.type === 'runway' && typeof switchToTab === 'function') switchToTab('rwysep');
          if (typeof syncSettingsPaneToMode === 'function') syncSettingsPaneToMode();
          syncPanelFromState();
          renderObjectList();
          updateObjectInfo();
          draw();
        }} else {{
          const [col, row] = pixelToCell(wx, wy);
          if (col < 0 || row < 0 || col > GRID_COLS || row > GRID_ROWS) {{ state.dragStart = null; return; }}
          if (mode === 'terminal') {{
            if (state.terminalDrawingId) {{
              let term = state.terminals.find(t => t.id === state.terminalDrawingId);
              if (!term) {{
                state.terminalDrawingId = null;
              }} else {{
                const pt = {{ col, row }};
                if (term.vertices.length === 0) {{
                  pushUndo();
                  term.vertices.push(pt);
                }} else {{
                  const [fx,fy] = cellToPixel(term.vertices[0].col, term.vertices[0].row);
                  const d2 = dist2([fx,fy], cellToPixel(col, row));
                  if (d2 < (CELL_SIZE*0.6)**2 && term.vertices.length >= 3) {{
                    term.closed = true;
                    state.terminalDrawingId = null;
                    syncPanelFromState();
                  }} else {{
                    const last = term.vertices[term.vertices.length-1];
                    if (last.col !== col || last.row !== row) {{ pushUndo(); term.vertices.push(pt); }}
                  }}
                }}
                draw();
              }}
            }}
          }} else if (mode === 'taxiway') {{
            if (state.taxiwayDrawingId) {{
              const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
              if (tw) {{
                const pt = {{ col, row }};
                const last = tw.vertices[tw.vertices.length - 1];
                if (!last || last.col !== col || last.row !== row) {{
                  // Runway 타입일 때는 딱 2개의 점(시작/끝)만 허용
                  if (tw.pathType === 'runway' && tw.vertices.length >= 2) return;
                  pushUndo();
                  tw.vertices.push(pt);
                  if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
                  // 두 점이 찍힌 순간 자동으로 그리기 종료
                  if (tw.pathType === 'runway' && tw.vertices.length >= 2) {{
                    state.taxiwayDrawingId = null;
                    syncPanelFromState();
                    if (scene3d) update3DScene();
                  }}
                  if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
                }}
              }}
            }}
          }} else if (mode === 'pbb') {{
            let bestEdge = null, bestD2 = Infinity;
            state.terminals.forEach(t => {{
              if (!t.closed || t.vertices.length < 2) return;
              let cx = 0, cy = 0;
              t.vertices.forEach(v => {{
                const [px, py] = cellToPixel(v.col, v.row);
                cx += px; cy += py;
              }});
              cx /= t.vertices.length || 1;
              cy /= t.vertices.length || 1;
              for (let i = 0; i < t.vertices.length; i++) {{
                const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
                const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
                const near = closestPointOnSegment(p1, p2, [wx, wy]);
                if (near) {{
                  const d2 = dist2(near, [wx, wy]);
                  if (d2 < bestD2) {{
                    bestD2 = d2;
                    bestEdge = {{ near, p1, p2, col: v1.col, row: v1.row, cx, cy }};
                  }}
                }}
              }}
            }});
            const maxD2 = (CELL_SIZE*1.0)**2;
            if (bestEdge && bestD2 < maxD2) {{
              const [ex,ey]=bestEdge.near, [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
              let nx = -(y2-y1), ny = x2-x1;
              const len = Math.hypot(nx,ny) || 1;
              nx /= len; ny /= len;
              const toClickX = wx - ex, toClickY = wy - ey;
              if (nx * toClickX + ny * toClickY < 0) {{ nx *= -1; ny *= -1; }}
              const category = document.getElementById('standCategory').value || 'C';
              const standSize = getStandSizeMeters(category);
              const minLen = standSize / 2 + 3;
              const lenCells = parseInt(document.getElementById('pbbLength').value || '2', 10);
              const lenPx = Math.max(lenCells * CELL_SIZE * 0.9, minLen);
              const newPbb = {{ x1: ex, y1: ey, x2: ex+nx*lenPx, y2: ey+ny*lenPx, category }};
              if (!pbbStandOverlapsExisting(newPbb)) {{
                pushUndo();
                state.pbbStands.push({{ id: id(), name: document.getElementById('standName').value.trim() || ('Contact Stand ' + (state.pbbStands.length + 1)), x1: ex, y1: ey, x2: ex+nx*lenPx, y2: ey+ny*lenPx, category: newPbb.category, edgeCol: bestEdge.col, edgeRow: bestEdge.row }});
                syncPanelFromState();
                draw();
              }}
            }}
          }} else if (mode === 'remote') {{
            const category = document.getElementById('remoteCategory').value || 'C';
            const [cx, cy] = cellToPixel(col, row);
            const size = getStandSizeMeters(category);
            const bounds = getStandBoundsRect(cx, cy, size);
            if (!standOverlapsExisting(bounds)) {{
              pushUndo();
              state.remoteStands.push({{ id: id(), col, row, category }});
              syncPanelFromState();
              draw();
            }}
          }}
        }}
      }}
      state.dragStart = null;
    }});
    let scene3d = null, camera3d = null, renderer3d = null, controls3d = null, grid3DMapper = null, raycaster3d = null, mouse3d = null, groundPlane3d = null, gridGroup3d = null;
    let mouse3dDown = null;
    const view3dContainer = document.getElementById('view3d-container');
    document.getElementById('btnView2D').classList.add('active');
    document.getElementById('btnView2D').addEventListener('click', function() {{
      document.getElementById('btnView2D').classList.add('active');
      document.getElementById('btnView3D').classList.remove('active');
      document.getElementById('canvas-container').style.display = 'block';
      view3dContainer.classList.remove('active');
      if (renderer3d) renderer3d.domElement.style.display = 'none';
      // display:block 직후에는 레이아웃이 아직 반영되지 않아 getBoundingClientRect가 0이 될 수 있음.
      // 한 프레임 뒤 resizeCanvas()로 캔버스 크기 갱신 및 draw() 호출
      requestAnimationFrame(function() {{
        if (typeof resizeCanvas === 'function') resizeCanvas();
      }});
    }});
    document.getElementById('btnView3D').addEventListener('click', function() {{
      document.getElementById('btnView3D').classList.add('active');
      document.getElementById('btnView2D').classList.remove('active');
      document.getElementById('canvas-container').style.display = 'none';
      view3dContainer.classList.add('active');
      init3D();
      animate3D();
    }});

    function reset3DView() {{
      if (!camera3d) return;
      const halfW = (GRID_COLS * CELL_SIZE) / 2;
      const halfH = (GRID_ROWS * CELL_SIZE) / 2;
      const maxDim = Math.max(halfW, halfH);
      camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
      camera3d.lookAt(0, 0, 0);
      if (controls3d) {{
        controls3d.target.set(0, 0, 0);
        controls3d.update();
      }}
    }}

    if (resetViewBtn) {{
      resetViewBtn.addEventListener('click', function() {{
        try {{
          resizeCanvas();
          if (view3dContainer.classList.contains('active')) reset3DView();
          else reset2DView();
          try {{ draw(); }} catch(e) {{}}
          if (typeof update3DScene === 'function') update3DScene();
        }} catch (e) {{ console.error('Fit button error:', e); }}
      }});
    }}

    class Grid3DMapper {{
      constructor(cols, rows, cellSize) {{
        this.cols = cols;
        this.rows = rows;
        this.cellSize = cellSize;
        this.ox = (cols * cellSize) / 2;
        this.oz = (rows * cellSize) / 2;
      }}
      pixelToWorldXZ(x, y) {{
        return {{ x: this.ox - x, z: this.oz - y }};
      }}
      cellToWorld(col, row, height) {{
        const [px, py] = cellToPixel(col, row);
        const p = this.pixelToWorldXZ(px, py);
        return new THREE.Vector3(p.x, height, p.z);
      }}
      worldFromPixel(x, y, height) {{
        const p = this.pixelToWorldXZ(x, y);
        return new THREE.Vector3(p.x, height, p.z);
      }}
      shapeFromCell(col, row) {{
        const [px, py] = cellToPixel(col, row);
        return {{ x: this.ox - px, y: py - this.oz }};
      }}
      worldToPixel(xWorld, zWorld) {{
        return {{ x: this.ox - xWorld, y: this.oz - zWorld }};
      }}
      worldToCell(xWorld, zWorld) {{
        const p = this.worldToPixel(xWorld, zWorld);
        let col = Math.round(p.x / this.cellSize);
        let row = Math.round(p.y / this.cellSize);
        col = Math.max(0, Math.min(this.cols, col));
        row = Math.max(0, Math.min(this.rows, row));
        return [col, row];
      }}
    }}

    function init3D() {{
      if (renderer3d) {{ renderer3d.domElement.style.display = 'block'; update3DScene(); return; }}
      const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
      scene3d = new THREE.Scene();
      scene3d.background = new THREE.Color(0x303030);
      // 3D 격자 + 축 전용 그룹 (update3DScene에서 지우지 않도록 별도 그룹으로 유지)
      gridGroup3d = new THREE.Group();
      scene3d.add(gridGroup3d);
      camera3d = new THREE.PerspectiveCamera(50, w/h, 1, 100000);
      const halfW = (GRID_COLS * CELL_SIZE) / 2, halfH = (GRID_ROWS * CELL_SIZE) / 2;
      const maxDim = Math.max(halfW, halfH);
      camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
      camera3d.lookAt(0, 0, 0);
      // 축 가이드: 그리드 평면을 X(빨강)–Y(초록)로, 수직축을 Z(파랑)로 표시
      const axisLen = CELL_SIZE * 8;
      const axisOrigin = new THREE.Vector3(-maxDim, 0, -maxDim);
      function addAxis(toVec, color) {{
        const pts = [axisOrigin, axisOrigin.clone().add(toVec)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({{ color }});
        const line = new THREE.Line(geo, mat);
        gridGroup3d.add(line);
      }}
      // x-axis: 그리드 X 방향
      addAxis(new THREE.Vector3(axisLen, 0, 0), 0xef4444);
      // y-axis: 그리드 Y 방향 (world Z 방향)
      addAxis(new THREE.Vector3(0, 0, axisLen), 0x22c55e);
      // z-axis: 수직 (world Y 방향)
      addAxis(new THREE.Vector3(0, axisLen, 0), 0x3b82f6);
      // 축 끝에 x,y,z 레이블 스프라이트 추가
      function createAxisLabel(text, color, endVec) {{
        const size = 128;
        const canvasLabel = document.createElement('canvas');
        canvasLabel.width = size;
        canvasLabel.height = size;
        const g = canvasLabel.getContext('2d');
        g.clearRect(0, 0, size, size);
        g.font = 'bold 72px system-ui';
        g.fillStyle = color;
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText(text, size / 2, size / 2);
        const tex = new THREE.CanvasTexture(canvasLabel);
        const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
        const sprite = new THREE.Sprite(mat);
        const s = CELL_SIZE * 3;
        sprite.scale.set(s, s, 1);
        sprite.position.copy(axisOrigin.clone().add(endVec));
        gridGroup3d.add(sprite);
      }}
      createAxisLabel('x', '#ef4444', new THREE.Vector3(axisLen * 1.1, 0, 0));
      createAxisLabel('y', '#22c55e', new THREE.Vector3(0, 0, axisLen * 1.1));
      createAxisLabel('z', '#3b82f6', new THREE.Vector3(0, axisLen * 1.1, 0));
      grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
      renderer3d = new THREE.WebGLRenderer({{ antialias: true }});
      renderer3d.setSize(w, h);
      renderer3d.setPixelRatio(window.devicePixelRatio || 1);
      view3dContainer.appendChild(renderer3d.domElement);
      controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
      controls3d.enableDamping = true;
      controls3d.dampingFactor = 0.05;
      raycaster3d = new THREE.Raycaster();
      mouse3d = new THREE.Vector2();
      groundPlane3d = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const dom3d = renderer3d.domElement;
      function getHitPoint(ev) {{
        const rect = dom3d.getBoundingClientRect();
        const ndcX = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        mouse3d.set(ndcX, ndcY);
        raycaster3d.setFromCamera(mouse3d, camera3d);
        const hit = new THREE.Vector3();
        return raycaster3d.ray.intersectPlane(groundPlane3d, hit) ? hit : null;
      }}
      dom3d.addEventListener('mousedown', function(ev) {{
        if (ev.button === 0) mouse3dDown = {{ x: ev.clientX, y: ev.clientY }};
      }});
      dom3d.addEventListener('mouseup', function(ev) {{
        if (ev.button !== 0 || !mouse3dDown) return;
        const dx = ev.clientX - mouse3dDown.x, dy = ev.clientY - mouse3dDown.y;
        if (dx*dx + dy*dy > 25) {{ mouse3dDown = null; return; }}
        mouse3dDown = null;
        const hit = getHitPoint(ev);
        if (!hit || !grid3DMapper) return;
        const mode = settingModeSelect.value;
        const p = grid3DMapper.worldToPixel(hit.x, hit.z);
        const wx = p.x, wy = p.y;
        const [col, row] = grid3DMapper.worldToCell(hit.x, hit.z);
        if (mode === 'pbb') {{
          let bestEdge = null, bestD2 = Infinity;
          state.terminals.forEach(t => {{
            if (!t.closed || t.vertices.length < 2) return;
            let cx = 0, cy = 0;
            t.vertices.forEach(v => {{ const [px, py] = cellToPixel(v.col, v.row); cx += px; cy += py; }});
            cx /= t.vertices.length || 1; cy /= t.vertices.length || 1;
            for (let i = 0; i < t.vertices.length; i++) {{
              const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
              const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
              const near = closestPointOnSegment(p1, p2, [wx, wy]);
              if (near) {{ const d2 = dist2(near, [wx, wy]); if (d2 < bestD2) {{ bestD2 = d2; bestEdge = {{ near, p1, p2, col: v1.col, row: v1.row, cx, cy }}; }} }}
            }}
          }});
          const maxD2 = (CELL_SIZE*1.0)**2;
          if (bestEdge && bestD2 < maxD2) {{
            const [ex,ey]=bestEdge.near, [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
            let nx = -(y2-y1), ny = x2-x1;
            const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
            const toClickX = wx - ex, toClickY = wy - ey;
            if (nx * toClickX + ny * toClickY < 0) {{ nx *= -1; ny *= -1; }}
            const category = document.getElementById('standCategory').value || 'C';
            const standSize = getStandSizeMeters(category);
            const minLen = standSize / 2 + 3;
            const lenCells = parseInt(document.getElementById('pbbLength').value || '2', 10);
            const lenPx = Math.max(lenCells * CELL_SIZE * 0.9, minLen);
            const newPbb = {{ x1: ex, y1: ey, x2: ex+nx*lenPx, y2: ey+ny*lenPx, category }};
            if (!pbbStandOverlapsExisting(newPbb)) {{
              state.pbbStands.push({{ id: id(), name: document.getElementById('standName').value.trim() || ('Contact Stand ' + (state.pbbStands.length + 1)), x1: ex, y1: ey, x2: ex+nx*lenPx, y2: ey+ny*lenPx, category: newPbb.category, edgeCol: bestEdge.col, edgeRow: bestEdge.row }});
              syncPanelFromState(); updateObjectInfo(); update3DScene();
            }}
          }}
        }} else if (mode === 'remote') {{
          if (col >= 0 && row >= 0 && col <= GRID_COLS && row <= GRID_ROWS) {{
            const category = document.getElementById('remoteCategory').value || 'C';
            const [cx, cy] = cellToPixel(col, row);
            const size = getStandSizeMeters(category);
            const bounds = getStandBoundsRect(cx, cy, size);
            if (!standOverlapsExisting(bounds)) {{
              state.remoteStands.push({{ id: id(), col, row, category }});
              syncPanelFromState(); updateObjectInfo(); update3DScene();
            }}
          }}
        }}
      }});
      const step = CELL_SIZE;
      const GRID_MAJOR = 10;
      const faintLines = [];
      const majorLines = [];
      let kx = 0;
      for (let x = -maxDim; x <= maxDim; x += step, kx++) {{
        const pts = [new THREE.Vector3(x, 0, -maxDim), new THREE.Vector3(x, 0, maxDim)];
        if (kx % GRID_MAJOR === 0) majorLines.push.apply(majorLines, pts);
        else faintLines.push.apply(faintLines, pts);
      }}
      let kz = 0;
      for (let z = -maxDim; z <= maxDim; z += step, kz++) {{
        const pts = [new THREE.Vector3(-maxDim, 0, z), new THREE.Vector3(maxDim, 0, z)];
        if (kz % GRID_MAJOR === 0) majorLines.push.apply(majorLines, pts);
        else faintLines.push.apply(faintLines, pts);
      }}
      if (faintLines.length) {{
        const faintGeo = new THREE.BufferGeometry().setFromPoints(faintLines);
        // 2D 보조격자와 비슷하지만 살짝 더 투명하게
        const faintMat = new THREE.LineBasicMaterial({{
          color: 0xd4d4d4,
          transparent: true,
          opacity: 0.16,
          depthTest: false
        }});
        gridGroup3d.add(new THREE.LineSegments(faintGeo, faintMat));
      }}
      if (majorLines.length) {{
        const majorGeo = new THREE.BufferGeometry().setFromPoints(majorLines);
        // 주격자도 약간 투명도를 주어 배경과 잘 어우러지게
        const majorMat = new THREE.LineBasicMaterial({{
          color: 0xffffff,
          transparent: true,
          opacity: 0.78,
          depthTest: false
        }});
        gridGroup3d.add(new THREE.LineSegments(majorGeo, majorMat));
      }}
      update3DScene();
    }}

    function update3DScene() {{
      if (!scene3d) return;
      while (scene3d.children.length > 1) scene3d.remove(scene3d.children[scene3d.children.length - 1]);
      if (!grid3DMapper) grid3DMapper = new Grid3DMapper(GRID_COLS, GRID_ROWS, CELL_SIZE);
      const ox = grid3DMapper.ox;
      const oz = grid3DMapper.oz;
      const maxDim = Math.max(ox, oz);
      state.terminals.forEach(term => {{
        if (!term.closed || term.vertices.length < 3) return;
        const shape = new THREE.Shape();
        for (let i = 0; i < term.vertices.length; i++) {{
          const pos = grid3DMapper.shapeFromCell(term.vertices[i].col, term.vertices[i].row);
          if (i === 0) shape.moveTo(pos.x, pos.y);
          else shape.lineTo(pos.x, pos.y);
        }}
        shape.closePath();
        const floors = term.floors != null ? Math.max(1, parseInt(term.floors, 10) || 1) : 1;
        const f2fRaw = term.floorToFloor != null ? Number(term.floorToFloor) : (term.floorHeight != null ? Number(term.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorHVal = term.floorHeight != null ? Number(term.floorHeight) || (floors * f2f) : (floors * f2f);
        const floorH = Math.max(0.5, floorHVal);
        const extrude = new THREE.ExtrudeGeometry(shape, {{ depth: floorH, bevelEnabled: false }});
        const mesh = new THREE.Mesh(extrude, new THREE.MeshPhongMaterial({{ color: 0x38bdf8, transparent: true, opacity: 0.55 }}));
        mesh.rotation.x = -Math.PI / 2;
        scene3d.add(mesh);
      }});
      state.pbbStands.forEach(pbb => {{
        const h = CELL_SIZE * 0.5;
        const start = grid3DMapper.worldFromPixel(pbb.x1, pbb.y1, h);
        const end = grid3DMapper.worldFromPixel(pbb.x2, pbb.y2, h);
        const dir = new THREE.Vector3().subVectors(end, start);
        const length = dir.length() || 1;
        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        dir.normalize();

        const corridorWidth = CELL_SIZE * 0.4;
        const corridorHeight = CELL_SIZE * 0.3;
        const corridorGeo = new THREE.BoxGeometry(length, corridorHeight, corridorWidth);
        const corridorMat = new THREE.MeshPhongMaterial({{ color: 0x7dd3fc }});
        const corridor = new THREE.Mesh(corridorGeo, corridorMat);
        corridor.position.copy(mid);
        corridor.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);

        const headSize = CELL_SIZE * 0.7;
        const headGeo = new THREE.BoxGeometry(headSize, corridorHeight * 1.1, headSize * 0.9);
        const headMat = new THREE.MeshPhongMaterial({{ color: 0x22c55e }});
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.copy(end);

        const baseSize = CELL_SIZE * 0.5;
        const baseGeo = new THREE.BoxGeometry(baseSize, corridorHeight * 1.1, baseSize);
        const baseMat = new THREE.MeshPhongMaterial({{ color: 0x1f2937 }});
        const base = new THREE.Mesh(baseGeo, baseMat);
        base.position.copy(start);

        // 3D green apron: PBB 3D 방향(dir)과 동일한 회전으로,
        // XZ 평면 위에 정사각형을 직접 구성한다. (2D와 최대한 동일한 회전 느낌)
        const standSize = getStandSizeMeters(pbb.category || 'C');
        const half = standSize / 2;
        const apronY = CELL_SIZE * 0.02;
        const center = grid3DMapper.worldFromPixel(pbb.x2, pbb.y2, apronY);

        const dirXZ = new THREE.Vector3(end.x - start.x, 0, end.z - start.z);
        let apronMesh = null;
        if (dirXZ.lengthSq() > 1e-6) {{
          dirXZ.normalize();
          const perp = new THREE.Vector3(-dirXZ.z, 0, dirXZ.x); // XZ 평면에서 90도 회전
          const v1 = center.clone().addScaledVector(dirXZ, -half).addScaledVector(perp, -half);
          const v2 = center.clone().addScaledVector(dirXZ,  half).addScaledVector(perp, -half);
          const v3 = center.clone().addScaledVector(dirXZ,  half).addScaledVector(perp,  half);
          const v4 = center.clone().addScaledVector(dirXZ, -half).addScaledVector(perp,  half);
          const apronGeo = new THREE.BufferGeometry();
          const vertices = new Float32Array([
            v1.x, v1.y, v1.z,
            v2.x, v2.y, v2.z,
            v3.x, v3.y, v3.z,
            v4.x, v4.y, v4.z
          ]);
          apronGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
          apronGeo.setIndex([0, 1, 2, 0, 2, 3]);
          apronGeo.computeVertexNormals();
          const apronMat = new THREE.MeshPhongMaterial({{
            color: 0x22c55e,
            transparent: true,
            opacity: 0.55,
            side: THREE.DoubleSide
          }});
          apronMesh = new THREE.Mesh(apronGeo, apronMat);
          apronMesh.receiveShadow = true;
        }}

        const group = new THREE.Group();
        group.add(corridor);
        group.add(head);
        group.add(base);
        if (apronMesh) group.add(apronMesh);
        scene3d.add(group);
      }});
      state.remoteStands.forEach(st => {{
        // Green remote apron area (same footprint as 2D)
        const size = getStandSizeMeters(st.category || 'C');
        const [px, py] = cellToPixel(st.col, st.row);
        const center = grid3DMapper.worldFromPixel(px, py, CELL_SIZE * 0.02);
        const apronGeo = new THREE.PlaneGeometry(size, size);
        const apronMat = new THREE.MeshPhongMaterial({{ color: 0x22c55e, transparent: true, opacity: 0.55 }});
        const apron = new THREE.Mesh(apronGeo, apronMat);
        apron.position.copy(center);
        apron.rotation.x = -Math.PI / 2; // flat on ground, axis-aligned like 2D
        scene3d.add(apron);

        const box = new THREE.Mesh(
          new THREE.BoxGeometry(CELL_SIZE * 0.7, CELL_SIZE * 0.3, CELL_SIZE * 0.7),
          new THREE.MeshPhongMaterial({{ color: 0x22c55e }})
        );
        box.position.copy(grid3DMapper.cellToWorld(st.col, st.row, CELL_SIZE * 0.15));
        scene3d.add(box);
      }});
      state.taxiways.forEach(tw => {{
        if (tw.vertices.length < 2) return;
        const w = tw.width != null ? tw.width : (tw.pathType === 'runway' ? 60 : 15);
        const isRunwayPath = tw.pathType === 'runway';
        const h = CELL_SIZE * 0.04;
        // 2D와 동일하게 vertex 사이를 직선 세그먼트로 연결하되,
        // 코너 지점에는 추가 패치를 넣어 시각적으로 라운드 느낌을 준다.
        const worldPts = tw.vertices.map(v => {{
          const [px, py] = cellToPixel(v.col, v.row);
          return grid3DMapper.worldFromPixel(px, py, h);
        }});
        // 기본 세그먼트
        for (let i = 0; i < worldPts.length - 1; i++) {{
          const start = worldPts[i];
          const end = worldPts[i + 1];
          const dirVec = new THREE.Vector3().subVectors(end, start);
          const length = dirVec.length() || 1;
          dirVec.normalize();
          const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
          const segGeo = new THREE.BoxGeometry(length, h * 0.5, w);
          const segMat = new THREE.MeshPhongMaterial({{ color: isRunwayPath ? 0x4b5563 : 0xeab308 }});
          const seg = new THREE.Mesh(segGeo, segMat);
          seg.position.copy(mid);
          seg.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dirVec);
          scene3d.add(seg);
        }}
        // 코너 라운드용 보조 박스 (joint patch)
        for (let i = 1; i < worldPts.length - 1; i++) {{
          const pPrev = worldPts[i - 1];
          const p = worldPts[i];
          const pNext = worldPts[i + 1];
          const v1 = new THREE.Vector3().subVectors(p, pPrev);
          const v2 = new THREE.Vector3().subVectors(pNext, p);
          if (v1.lengthSq() < 1e-4 || v2.lengthSq() < 1e-4) continue;
          v1.normalize();
          v2.normalize();
          const dot = v1.dot(v2);
          // 거의 일직선이면 스킵
          if (Math.abs(dot) > 0.999) continue;
          const bis = new THREE.Vector3().addVectors(v1, v2);
          if (bis.lengthSq() < 1e-4) continue;
          bis.normalize();
          const jointLen = w * 0.8;
          const jointGeo = new THREE.BoxGeometry(jointLen, h * 0.5, w * 1.02);
          const jointMat = new THREE.MeshPhongMaterial({{ color: isRunwayPath ? 0x4b5563 : 0xeab308 }});
          const joint = new THREE.Mesh(jointGeo, jointMat);
          joint.position.copy(p);
          joint.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), bis);
          scene3d.add(joint);
        }}
        // Direction arrows on top of taxiway, matching 2D polyline logic
        const dir = getTaxiwayDirection(tw);
        if (dir !== 'both' && tw.vertices.length >= 2) {{
          const ptsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
          const totalLen = ptsPix.reduce((acc, p, i) => acc + (i > 0 ? Math.hypot(p[0]-ptsPix[i-1][0], p[1]-ptsPix[i-1][1]) : 0), 0);
          const arrowSpacing = Math.max(22, Math.min(42, totalLen / 10));
          const numArrows = Math.max(2, Math.floor(totalLen / arrowSpacing));
          const arrowSize = Math.min(8, w * 0.4);
          for (let k = 1; k <= numArrows; k++) {{
            const targetDist = totalLen * (k / (numArrows + 1));
            let acc = 0;
            let ax = ptsPix[0][0], ay = ptsPix[0][1];
            let angle = Math.atan2(ptsPix[1][1]-ptsPix[0][1], ptsPix[1][0]-ptsPix[0][0]);
            let segStartPix = ptsPix[0];
            let segEndPix = ptsPix[1];
            for (let i = 1; i < ptsPix.length; i++) {{
              const segLen = Math.hypot(ptsPix[i][0]-ptsPix[i-1][0], ptsPix[i][1]-ptsPix[i-1][1]);
              angle = Math.atan2(ptsPix[i][1]-ptsPix[i-1][1], ptsPix[i][0]-ptsPix[i-1][0]);
              if (acc + segLen >= targetDist) {{
                const tSeg = segLen > 0 ? (targetDist - acc) / segLen : 0;
                ax = ptsPix[i-1][0] + tSeg * (ptsPix[i][0]-ptsPix[i-1][0]);
                ay = ptsPix[i-1][1] + tSeg * (ptsPix[i][1]-ptsPix[i-1][1]);
                segStartPix = ptsPix[i-1];
                segEndPix = ptsPix[i];
                break;
              }}
              acc += segLen;
            }}
            if (dir === 'counter_clockwise') angle += Math.PI;
            const pos = grid3DMapper.worldFromPixel(ax, ay, h + 0.8);
            const [sx, sy] = segStartPix;
            const [ex, ey] = segEndPix;
            const startW = grid3DMapper.worldFromPixel(sx, sy, h + 0.8);
            const endW = grid3DMapper.worldFromPixel(ex, ey, h + 0.8);
            const tangent = new THREE.Vector3().subVectors(endW, startW).normalize();
            if (dir === 'counter_clockwise') tangent.negate();
            const up = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(up, tangent);
            const coneGeo = new THREE.ConeGeometry(arrowSize * 0.6, arrowSize, 4);
            const coneMat = new THREE.MeshPhongMaterial({{ color: 0xf59e0b }});
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.copy(pos);
            cone.position.y = h + 0.8;
            cone.quaternion.copy(quat);
            scene3d.add(cone);
          }}
        }}
      }});
      // Runways
      state.runways.forEach(rw => {{
        const h = CELL_SIZE * 0.03;
        const cx = rw.cx, cy = rw.cy;
        const ux = rw.dx, uy = rw.dy;
        const len2 = rw.length / 2;
        const startPx = cx - ux * len2;
        const startPy = cy - uy * len2;
        const endPx = cx + ux * len2;
        const endPy = cy + uy * len2;
        const start = grid3DMapper.worldFromPixel(startPx, startPy, h);
        const end = grid3DMapper.worldFromPixel(endPx, endPy, h);
        const dirVec = new THREE.Vector3().subVectors(end, start);
        const length = dirVec.length() || 1;
        dirVec.normalize();
        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        const segGeo = new THREE.BoxGeometry(length, h * 0.6, rw.width);
        const segMat = new THREE.MeshPhongMaterial({{ color: 0x9ca3af }});
        const seg = new THREE.Mesh(segGeo, segMat);
        seg.position.copy(mid);
        seg.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dirVec);
        scene3d.add(seg);

        // Direction arrows in 3D along runway centerline (like taxiway)
        const dirVal = getTaxiwayDirection(rw);
        if (dirVal !== 'both') {{
          const dirSign = dirVal === 'clockwise' ? 1 : -1;
          const baseStart = dirSign === 1 ? start : end;
          const baseDir = dirSign === 1 ? dirVec.clone() : dirVec.clone().negate();
          const totalLen = rw.length;
          const arrowSpacing = Math.max(80, Math.min(200, totalLen / 6));
          const arrLen = Math.max(CELL_SIZE * 0.8, Math.min(CELL_SIZE * 2.0, rw.width * 0.8));
          const numArrows = Math.floor(totalLen / arrowSpacing);
          for (let i = 1; i <= numArrows; i++) {{
            const d = (i * totalLen) / (numArrows + 1);
            const pos = new THREE.Vector3().copy(baseStart).add(baseDir.clone().multiplyScalar(d));
            const coneGeo = new THREE.ConeGeometry(arrLen * 0.4, arrLen, 12);
            const coneMat = new THREE.MeshPhongMaterial({{ color: 0xf97316 }});
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.copy(pos);
            const up = new THREE.Vector3(0, 1, 0);
            const tangent = baseDir.clone().normalize();
            const quat = new THREE.Quaternion().setFromUnitVectors(up, tangent);
            cone.quaternion.copy(quat);
            scene3d.add(cone);
          }}
        }}
      }});
      // Apron–Taxiway links in 3D, matching 2D links
      const linkH = CELL_SIZE * 0.05;
      state.apronLinks.forEach(lk => {{
        const pbb = state.pbbStands.find(p => p.id === lk.pbbId);
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        if (!pbb || !tw || lk.tx == null || lk.ty == null) return;
        const start = grid3DMapper.worldFromPixel(pbb.x2, pbb.y2, linkH);
        const end = grid3DMapper.worldFromPixel(lk.tx, lk.ty, linkH);
        const dirVec = new THREE.Vector3().subVectors(end, start);
        const length = dirVec.length() || 1;
        dirVec.normalize();
        const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        const linkWidth = CELL_SIZE * 0.4;
        const linkGeo = new THREE.BoxGeometry(length, linkH * 0.5, linkWidth);
        const linkMat = new THREE.MeshPhongMaterial({{ color: 0x22d3ee, transparent: true, opacity: 0.9 }});
        const linkMesh = new THREE.Mesh(linkGeo, linkMat);
        linkMesh.position.copy(mid);
        linkMesh.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dirVec);
        scene3d.add(linkMesh);
      }});
      // Flights in 3D: simple airplane-shaped meshes following 2D timeline
      if (state.flights && state.flights.length) {{
        const tSec = state.simTimeSec;
        state.flights.forEach(f => {{
          const pose = getFlightPoseAtTime(f, tSec);
          if (!pose) return;
          const {{ x, y, dx, dy }} = pose;
          const pos3d = grid3DMapper.worldFromPixel(x, y, CELL_SIZE * 0.5);
          const len = Math.hypot(dx, dy) || 1;
          const dirVec = new THREE.Vector3(dx / len, 0, dy / len);
          let scale = 1.0;
          const code = (f.code || '').toUpperCase();
          if (code === 'A' || code === 'B') scale = 0.8;
          else if (code === 'C') scale = 1.0;
          else if (code === 'D') scale = 1.2;
          else if (code === 'E') scale = 1.4;
          else if (code === 'F') scale = 1.6;
          const bodyLen = CELL_SIZE * 1.2 * scale;
          const bodyWidth = CELL_SIZE * 0.4 * scale;
          const bodyHeight = CELL_SIZE * 0.2 * scale;
          const color = 0xff2f92; // 핫핑크
          const group = new THREE.Group();
          const bodyGeo = new THREE.BoxGeometry(bodyLen, bodyHeight, bodyWidth);
          const bodyMat = new THREE.MeshPhongMaterial({{ color }});
          const body = new THREE.Mesh(bodyGeo, bodyMat);
          group.add(body);
          const wingGeo = new THREE.BoxGeometry(bodyLen * 0.4, bodyHeight * 0.5, bodyWidth * 1.8);
          const wingMat = new THREE.MeshPhongMaterial({{ color }});
          const wings = new THREE.Mesh(wingGeo, wingMat);
          wings.position.y = -bodyHeight * 0.2;
          group.add(wings);
          group.position.copy(pos3d);
          const forward = new THREE.Vector3(1, 0, 0);
          const quat = new THREE.Quaternion().setFromUnitVectors(forward, dirVec);
          group.quaternion.copy(quat);
          scene3d.add(group);
        }});
      }}
      const light = new THREE.DirectionalLight(0xffffff, 0.8);
      light.position.set(maxDim, maxDim * 2, maxDim);
      scene3d.add(light);
      scene3d.add(new THREE.AmbientLight(0xffffff, 0.4));
    }}

    function animate3D() {{
      if (!renderer3d || !view3dContainer.classList.contains('active')) return;
      requestAnimationFrame(animate3D);
      if (controls3d) controls3d.update();
      if (renderer3d && scene3d && camera3d) renderer3d.render(scene3d, camera3d);
    }}

    container.addEventListener('wheel', function(ev) {{
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const wx = (mx - state.panX) / state.scale, wy = (my - state.panY) / state.scale;
      const factor = 1 - ev.deltaY * 0.002;
      state.scale *= factor;
      state.scale = Math.max(0.2, Math.min(5, state.scale));
      state.panX = mx - wx * state.scale;
      state.panY = my - wy * state.scale;
      try {{ draw(); }} catch(e) {{}}
    }}, {{ passive: false }});

    window.addEventListener('resize', function() {{
      resizeCanvas();
      if (renderer3d && view3dContainer.classList.contains('active')) {{
        const w = view3dContainer.clientWidth, h = view3dContainer.clientHeight;
        camera3d.aspect = w / h;
        camera3d.updateProjectionMatrix();
        renderer3d.setSize(w, h);
      }}
    }});
    // Apply initial layout from JSON (if provided from Python) so the designer starts with a default configuration.
    applyInitialLayoutFromJson();
    updateLayoutNameBar(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
    resizeCanvas();
    reset2DView();
    syncPanelFromState();
    if (typeof draw === 'function') draw();
    if (typeof update3DScene === 'function') update3DScene();
  }})();
  </script>
</body>
</html>
"""

# Home의 home_globe처럼 전체 화면에 HTML 표시 (position: fixed, 100vw x 100vh)
st.markdown("""
  <style>
    [data-testid="stHeader"], header[data-testid="stHeader"] { display: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; overflow: hidden !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; overflow: visible !important; min-height: 100vh !important; margin: 0 !important; }
    section.main [data-testid="stVerticalBlock"] { padding-top: 0 !important; }
    .block-container iframe, section.main iframe[title="streamlit_component"], section.main iframe[title="Streamlit component"] {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      min-height: 100vh !important;
      display: block !important;
      border: none !important;
      z-index: 0 !important;
    }
  </style>
""", unsafe_allow_html=True)
components.html(
    html,
    height=2000,
    scrolling=False,
)

