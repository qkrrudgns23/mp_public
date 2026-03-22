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

# run_app.py When used(LAYOUT_SAME_PORT=1): 8501 From proxy API treatment. Others: layout_receiver(8765) movement.
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

# data/Layout_storage use only. data/ Directly below current_layout/default_layout generation·Disabled
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


def _info_path(*keys, default=None):
    d = INFORMATION
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


_layout_info = _info_path("tiers", "layout") or {}
_grid_info = (_layout_info.get("grid") or {}) if isinstance(_layout_info, dict) else {}
if isinstance(_grid_info, dict):
    try:
        if _grid_info.get("cols") is not None:
            GRID_COLS = max(1, int(_grid_info["cols"]))
    except (TypeError, ValueError):
        pass
    try:
        if _grid_info.get("rows") is not None:
            GRID_ROWS = max(1, int(_grid_info["rows"]))
    except (TypeError, ValueError):
        pass
    try:
        if _grid_info.get("cellSize") is not None:
            CELL_SIZE = float(_grid_info["cellSize"])
    except (TypeError, ValueError):
        pass

_term_ui = (_layout_info.get("terminal") or {}) if isinstance(_layout_info, dict) else {}
_tw_ui = (_layout_info.get("taxiway") or {}) if isinstance(_layout_info, dict) else {}
_rw_path_ui = (_layout_info.get("runwayPath") or {}) if isinstance(_layout_info, dict) else {}
_rw_exit_ui = (_layout_info.get("runwayExit") or {}) if isinstance(_layout_info, dict) else {}

_flight_info = _info_path("tiers", "flight_schedule") or _info_path("tiers", "flight") or {}
_algo_info = _info_path("tiers", "algorithm") or {}
_sim_step = (_algo_info.get("simulation") or {}).get("timeStepSec") if isinstance(_algo_info, dict) else None

if not DEFAULT_LAYOUT_PATH.is_file() and _fallback_default.is_file():
    shutil.copy2(_fallback_default, DEFAULT_LAYOUT_PATH)

# Load initial layout: Layout_storage/default_layout.json First of all, if there is no data/default_layout.json
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


# --- Python Discrete simulation engine (Information.json algorithm.simulation.timeStepSec) ---
SIM_TIME_STEP_SEC = 5
try:
    if _sim_step is not None:
        SIM_TIME_STEP_SEC = max(1, int(_sim_step))
except (TypeError, ValueError):
    pass

# HTML / iframe initial value (Layout tab, Flight tab — tiers.flight_schedule)
_ui_g_min_cs = int(_grid_info.get("minCellSize", 10)) if isinstance(_grid_info, dict) else 10
_ui_g_max_cs = int(_grid_info.get("maxCellSize", 1000)) if isinstance(_grid_info, dict) else 1000
_ui_g_cs_step = int(_grid_info.get("cellSizeStep", 10)) if isinstance(_grid_info, dict) else 10
_ui_g_min_dim = int(_grid_info.get("minDim", 5)) if isinstance(_grid_info, dict) else 5
_ui_g_max_dim = int(_grid_info.get("maxDim", 500)) if isinstance(_grid_info, dict) else 500

_ui_term_floors = int(_term_ui.get("floorsDefault", 1)) if isinstance(_term_ui, dict) else 1
_ui_term_floors_min = int(_term_ui.get("floorsMin", 1)) if isinstance(_term_ui, dict) else 1
_ui_term_floors_max = int(_term_ui.get("floorsMax", 20)) if isinstance(_term_ui, dict) else 20
_ui_term_f2f = float(_term_ui.get("floorToFloor", 4)) if isinstance(_term_ui, dict) else 4.0
_ui_term_f2f_min = float(_term_ui.get("floorToFloorMin", 1)) if isinstance(_term_ui, dict) else 1.0
_ui_term_f2f_max = float(_term_ui.get("floorToFloorMax", 10)) if isinstance(_term_ui, dict) else 10.0
_ui_term_f2f_step = float(_term_ui.get("floorToFloorStep", 0.5)) if isinstance(_term_ui, dict) else 0.5
_ui_term_dep = int(_term_ui.get("departureCapacity", 0)) if isinstance(_term_ui, dict) else 0
_ui_term_arr = int(_term_ui.get("arrivalCapacity", 0)) if isinstance(_term_ui, dict) else 0

_ui_tw_w = int(_tw_ui.get("width", 15)) if isinstance(_tw_ui, dict) else 15
_ui_tw_avg = float(_tw_ui.get("avgMoveVelocity", 10)) if isinstance(_tw_ui, dict) else 10.0
_ui_rw_min_arr = int(_rw_path_ui.get("minArrVelocity", 15)) if isinstance(_rw_path_ui, dict) else 15
_ui_rw_lineup = int(_rw_path_ui.get("lineupDistM", 0)) if isinstance(_rw_path_ui, dict) else 0
_ui_ex_max = int(_rw_exit_ui.get("maxExitVelocity", 30)) if isinstance(_rw_exit_ui, dict) else 30
_ui_ex_min = int(_rw_exit_ui.get("minExitVelocity", 15)) if isinstance(_rw_exit_ui, dict) else 15

_ui_flight_dwell = int(_flight_info.get("defaultDwellMin", 60)) if isinstance(_flight_info, dict) else 60
_ui_flight_min_dwell = int(_flight_info.get("defaultMinDwellMin", 0)) if isinstance(_flight_info, dict) else 0
_ui_flight_dwell_max = int(_flight_info.get("dwellInputMax", 600)) if isinstance(_flight_info, dict) else 600
_ui_flight_dwell_step = int(_flight_info.get("dwellStep", 5)) if isinstance(_flight_info, dict) else 5
_ui_sim_speeds = (
    _flight_info.get("simSpeedOptions")
    if isinstance(_flight_info, dict) and isinstance(_flight_info.get("simSpeedOptions"), list)
    else [0.5, 1, 5, 10, 20, 50, 100, 200]
)
try:
    _ui_default_sim_speed = float(_flight_info.get("defaultSimSpeed", 20)) if isinstance(_flight_info, dict) else 20.0
except (TypeError, ValueError):
    _ui_default_sim_speed = 20.0
_flight_speed_options_html = "".join(
    f'<option value="{v}"{" selected" if float(v) == float(_ui_default_sim_speed) else ""}>{v}x</option>'
    for v in _ui_sim_speeds
)

# query parameters: load_layout=name → Layout_storagecorresponding to JSON Load and display (API/without port)
# The basics are DEFAULT_LAYOUT; load_layout If there is one, overwrite it with that file.
layout_for_html = DEFAULT_LAYOUT


def _get_query_one(key: str):
    """Streamlit Extract single value from query parameter (query_params / experimental_get_query_params handle it all)."""
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
    # Run Simulation: ?run_simulation=1 This side current_layout Read and display after simulation
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

# Layout_storage The file list is injected when the page is rendered. (API no call)
layout_names = list_layout_names()


def _grid_view_background(info: dict) -> str:
    """tiers.style.gridView.background → 2D/3D workspace fill color."""
    tiers = info.get("tiers") if isinstance(info, dict) else None
    st = (tiers.get("style") if isinstance(tiers, dict) else None) or {}
    gv = st.get("gridView") if isinstance(st.get("gridView"), dict) else {}
    bg = str(gv.get("background", "#252525")).strip()
    if not bg.startswith("#") or len(bg) not in (4, 7):
        bg = "#252525"
    return bg


def _right_panel_background_opacity(info: dict) -> float:
    """tiers.style.rightPanel.backgroundOpacity → 0–1 for color-mix over transparent."""
    tiers = info.get("tiers") if isinstance(info, dict) else None
    st = (tiers.get("style") if isinstance(tiers, dict) else None) or {}
    rp = st.get("rightPanel") if isinstance(st.get("rightPanel"), dict) else {}
    try:
        op = float(rp.get("backgroundOpacity", 0.95))
    except (TypeError, ValueError):
        op = 0.95
    return max(0.0, min(1.0, op))


def _style_root_css_from_information(info: dict) -> str:
    """tiers.style → :root CSS variables (Gantt, tables, canvas, 3D theme)."""
    tiers = info.get("tiers") if isinstance(info, dict) else None
    st = (tiers.get("style") if isinstance(tiers, dict) else None) or {}
    if not isinstance(st, dict):
        st = {}
    _gv_bg = _grid_view_background(info)
    _rp_bg_op = _right_panel_background_opacity(info)
    _rp_mix_pct = f"{round(_rp_bg_op * 100)}%"
    rp_style = st.get("rightPanel") if isinstance(st.get("rightPanel"), dict) else {}

    def _emit(name: str, value) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, float):
            pass
        elif isinstance(value, int):
            pass
        else:
            value = str(value).strip().replace("\n", " ").replace("\r", "")
            if not value:
                return None
        return f"  {name}: {value};"

    lines: list[str] = [":root {"]
    g = st.get("gantt") if isinstance(st.get("gantt"), dict) else {}
    pairs_gantt = [
        ("--style-gantt-s-bar", g.get("sBar")),
        ("--style-gantt-s-series", g.get("sSeries")),
        ("--style-gantt-e-bar", g.get("eBar")),
        ("--style-gantt-e-series", g.get("eSeries")),
        ("--style-gantt-conflict", g.get("conflict")),
        ("--style-gantt-conflict-stripe2", g.get("conflictStripe2")),
        ("--style-gantt-conflict-text", g.get("conflictText")),
        ("--style-gantt-conflict-border", g.get("conflictBorder")),
        ("--style-gantt-selected", g.get("selected")),
        ("--style-gantt-selected-ring", g.get("selectedShadowRing")),
        ("--style-gantt-selected-glow", g.get("selectedGlow")),
        ("--style-gantt-flight-bg", g.get("flightBarBg")),
        ("--style-gantt-flight-fg", g.get("flightBarFg")),
        ("--style-gantt-flight-shadow", g.get("flightBarShadow")),
        ("--style-gantt-flight-dim-opacity", g.get("flightBarDimOpacity")),
        ("--style-gantt-ovlp-bg", g.get("overlapBadgeBg")),
        ("--style-gantt-ovlp-fg", g.get("overlapBadgeFg")),
        ("--style-gantt-grid-opacity", g.get("timeGridLineOpacity")),
        ("--style-gantt-sbar-opacity", g.get("sBarOpacity")),
        ("--style-gantt-ebar-opacity", g.get("eBarOpacity")),
        ("--style-gantt-e2bar-opacity", g.get("e2BarOpacity")),
        ("--style-gantt-apron-slot-opacity", g.get("apronSlotLabelOpacity")),
    ]
    fs = st.get("flightScheduleTable") if isinstance(st.get("flightScheduleTable"), dict) else {}
    pairs_fs = [
        ("--style-fs-col-s", fs.get("colS")),
        ("--style-fs-col-sd", fs.get("colSd")),
        ("--style-fs-col-e", fs.get("colE")),
        ("--style-fs-col-rot", fs.get("colRot")),
    ]
    rwy = st.get("rwySepTimeline") if isinstance(st.get("rwySepTimeline"), dict) else {}
    pairs_rwy = [
        ("--style-rwysep-line-s", rwy.get("lineS")),
        ("--style-rwysep-line-e", rwy.get("lineE")),
        ("--style-rwysep-line-opacity", rwy.get("lineOpacity")),
    ]
    c2 = st.get("canvas2d") if isinstance(st.get("canvas2d"), dict) else {}
    pairs_c2 = [
        ("--style-c2d-path-dep-stroke", c2.get("pathDepartureStroke")),
        ("--style-c2d-vtt-badge-bg", c2.get("vttBadgeBg")),
        ("--style-c2d-vtt-badge-stroke", c2.get("vttBadgeStroke")),
        ("--style-c2d-vtt-badge-text", c2.get("vttBadgeText")),
        ("--style-c2d-noway-fill", c2.get("noWayFill")),
        ("--style-c2d-noway-stroke", c2.get("noWayStroke")),
        ("--style-c2d-noway-text", c2.get("noWayText")),
        ("--style-c2d-term-stroke-sel", c2.get("terminalStrokeSelected")),
        ("--style-c2d-term-stroke-def", c2.get("terminalStrokeDefault")),
        ("--style-c2d-term-fill-sel", c2.get("terminalFillSelected")),
        ("--style-c2d-term-fill-def", c2.get("terminalFillDefault")),
        ("--style-c2d-term-label-fill", c2.get("terminalLabelFill")),
        ("--style-c2d-term-dash", c2.get("terminalSelectedDash")),
        ("--style-c2d-obj-sel-stroke", c2.get("objectSelectedStroke")),
        ("--style-c2d-obj-sel-fill", c2.get("objectSelectedFill")),
        ("--style-c2d-obj-sel-dash", c2.get("objectSelectedDashStroke")),
        ("--style-c2d-obj-sel-glow", c2.get("objectSelectedGlow")),
        ("--style-c2d-obj-sel-glow-blur", c2.get("objectSelectedGlowBlur")),
    ]
    td = st.get("threeD") if isinstance(st.get("threeD"), dict) else {}
    pairs_3d = [
        ("--style-3d-remote-apron", td.get("remoteApron")),
        ("--style-3d-remote-apron-opacity", td.get("remoteApronOpacity")),
        ("--style-3d-remote-box", td.get("remoteStandBox")),
        ("--style-3d-taxiway", td.get("taxiway")),
        ("--style-3d-runway-path", td.get("runwayPath")),
        ("--style-3d-arrow-cone", td.get("arrowCone")),
        ("--style-3d-apron-link", td.get("apronLink")),
        ("--style-3d-apron-link-opacity", td.get("apronLinkOpacity")),
        ("--style-3d-dir-light", td.get("directionalLightIntensity")),
        ("--style-3d-amb-light", td.get("ambientLightIntensity")),
    ]
    for pairs in (pairs_gantt, pairs_fs, pairs_rwy, pairs_c2, pairs_3d):
        for name, val in pairs:
            row = _emit(name, val)
            if row:
                lines.append(row)
    def _rp_px(var_name: str, key: str, default: int) -> None:
        raw = rp_style.get(key, default)
        try:
            n = int(float(raw))
        except (TypeError, ValueError):
            n = default
        row = _emit(var_name, f"{max(0, n)}px")
        if row:
            lines.append(row)

    for name, val in (
        ("--style-grid-view-bg", _gv_bg),
        ("--style-right-panel-bg-mix-percent", _rp_mix_pct),
        ("--style-right-panel-toggle-bg", rp_style.get("panelToggleBg")),
        ("--style-right-panel-toggle-color", rp_style.get("panelToggleColor")),
        ("--style-right-panel-toggle-border", rp_style.get("panelToggleBorder")),
        ("--style-right-panel-width-half", rp_style.get("panelWidthHalf")),
        ("--style-right-panel-width-full", rp_style.get("panelWidthFull")),
        (
            "--layout-toolbar-right",
            rp_style.get("panelWidthFull") or "50vw",
        ),
    ):
        row = _emit(name, val)
        if row:
            lines.append(row)
    _rp_px("--style-right-panel-resize-collapsed", "panelResizeCollapsedPx", 44)
    _rp_px("--style-right-panel-resize-collapse-below", "panelResizeCollapseBelowPx", 96)
    _rp_px("--style-right-panel-resize-min-expanded", "panelResizeMinExpandedPx", 220)
    _rp_px("--style-right-panel-resize-viewport-margin", "panelResizeViewportMarginPx", 8)
    lines.append("}")
    return "\n".join(lines)


_style_root_css = _style_root_css_from_information(INFORMATION)
_GRID_VIEW_BG = _grid_view_background(INFORMATION)

# Layout name to display at the top of the grid
layout_display_name = "default_layout"

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
{_style_root_css}
    :root {{
      --ui-bg-base: #0d0d0f;
      --ui-bg-surface: #141416;
      --ui-bg-elevated: #1c1c1f;
      --ui-bg-overlay: #242428;
      --ui-bg-input: #1a1a1d;
      --ui-border-subtle: rgba(255, 255, 255, 0.06);
      --ui-border-default: rgba(255, 255, 255, 0.10);
      --ui-border-strong: rgba(255, 255, 255, 0.18);
      --ui-text-primary: #f0f0f2;
      --ui-text-secondary: #8b8b96;
      --ui-text-muted: #5c5c68;
      --ui-accent: #7c6af7;
      --ui-accent-hover: #9181f8;
      --ui-accent-muted: rgba(124, 106, 247, 0.16);
      --ui-accent-ring: rgba(124, 106, 247, 0.35);
      --ui-success-bg: rgba(61, 214, 140, 0.14);
      --ui-success-border: rgba(61, 214, 140, 0.45);
      --ui-success-text: #b8f0d0;
      --ui-error-bg: rgba(248, 113, 113, 0.12);
      --ui-error-border: rgba(248, 113, 113, 0.35);
      --ui-error-text: #fca5a5;
      --ui-font: 'Inter', 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif;
      --ui-transition: border-color 0.15s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.15s cubic-bezier(0.16, 1, 0.3, 1), color 0.15s cubic-bezier(0.16, 1, 0.3, 1), filter 0.15s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    button {{ border: none; }}
    #right-panel button {{ appearance: none; -webkit-appearance: none; }}
    html, body {{ width: 100%; min-height: 100%; height: 100%; background: var(--ui-bg-base); color: var(--ui-text-primary); font-family: var(--ui-font); font-size: 14px; line-height: 1.5; overflow: hidden; -webkit-font-smoothing: antialiased; }}
    #app {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    #canvas-container {{ position: absolute; inset: 0; cursor: crosshair; background: var(--style-grid-view-bg, #252525); }}
    #grid-canvas {{ width: 100%; height: 100%; display: block; }}
    #toolbar {{ position: absolute; bottom: 56px; right: var(--layout-toolbar-right, var(--style-right-panel-width-full, 50vw)); left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 8px; z-index: 30; pointer-events: auto; transition: right 0.2s cubic-bezier(0.16, 1, 0.3, 1); }}
    #sim-controls-container {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--ui-border-default); background: var(--ui-bg-elevated); transition: var(--ui-transition); }}
    #sim-controls-container .tool-btn {{ margin: 0; border: none; border-radius: 4px; font-size: 11px; font-weight: 500; padding: 6px 12px; background: var(--ui-bg-overlay); color: var(--ui-text-primary); cursor: pointer; transition: var(--ui-transition); }}
    #sim-controls-container .tool-btn:hover {{ background: var(--ui-bg-surface); }}
    #sim-controls-container select {{ width: auto; min-width: 70px; margin: 0; padding: 4px 8px; font-size: 11px; background: var(--ui-bg-input); color: var(--ui-text-primary); border: 1px solid var(--ui-border-default); border-radius: 4px; transition: var(--ui-transition); }}
    #sim-controls-container select:focus {{ outline: none; border-color: var(--ui-accent-ring); box-shadow: 0 0 0 2px var(--ui-accent-muted); }}
    #sim-controls-container label {{ margin: 0 4px 0 0; font-size: 11px; color: var(--ui-text-secondary); }}
    #sim-controls-container #flightSimSlider {{ width: 120px; margin: 0; vertical-align: middle; accent-color: var(--ui-accent); }}
    #sim-controls-container #flightSimTimeLabel {{ font-size: 11px; color: var(--ui-text-primary); min-width: 72px; display: inline-block; }}
    #view-toggle {{ display: inline-flex; border-radius: 8px; overflow: hidden; border: 1px solid var(--ui-border-default); background: var(--ui-bg-elevated); transition: var(--ui-transition); }}
    #view-toggle .tool-btn {{ margin: 0; border: none; border-radius: 0; font-size: 11px; font-weight: 500; padding: 6px 12px; transition: var(--ui-transition); }}
    #view-toggle .tool-btn.active {{ background: var(--ui-accent-muted); color: var(--ui-accent); }}
    #view-toggle .tool-btn:not(.active) {{ background: transparent; color: var(--ui-text-secondary); }}
    #view-toggle .tool-btn:hover {{ background: var(--ui-bg-overlay); color: var(--ui-text-primary); }}
    #info-bar {{ display: none; position: absolute; bottom: 8px; left: 10px; right: 10px; padding: 8px 12px; min-height: 2.2em; border-radius: 8px; background: var(--ui-bg-elevated); border: 1px solid var(--ui-border-subtle); color: var(--ui-text-secondary); font-size: 12px; align-items: center; pointer-events: none; z-index: 5; }}
    #right-panel {{ position: absolute; top: 0; right: 0; bottom: 0; width: var(--style-right-panel-width-full, 50vw); max-width: calc(100vw - var(--style-right-panel-resize-viewport-margin, 8px)); background: color-mix(in srgb, var(--ui-bg-surface) var(--style-right-panel-bg-mix-percent, 95%), transparent); border-left: 1px solid var(--ui-border-default); padding: 12px; font-size: 12px; overflow-y: auto; z-index: 20; transition: width 0.2s cubic-bezier(0.16, 1, 0.3, 1), min-width 0.2s cubic-bezier(0.16, 1, 0.3, 1); }}
    #right-panel.panel-resize-dragging {{ transition: none !important; }}
    #right-panel.collapsed {{ width: var(--style-right-panel-resize-collapsed, 44px) !important; min-width: var(--style-right-panel-resize-collapsed, 44px); max-width: none; padding: 8px; overflow: hidden; }}
    #right-panel.collapsed .panel-content {{ display: none; }}
    #panel-toggle {{
      position: absolute;
      left: -26px;
      top: 50%;
      transform: translateY(-50%);
      width: 32px;
      height: 84px;
      border-radius: 9999px 0 0 9999px;
      background: var(--style-right-panel-toggle-bg, #3d3d44);
      color: var(--style-right-panel-toggle-color, #ffffff);
      border: 1px solid var(--style-right-panel-toggle-border, rgba(255,255,255,0.28));
      cursor: ew-resize;
      user-select: none;
      -webkit-user-select: none;
      touch-action: none;
      font-size: 16px;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.45);
      backdrop-filter: blur(8px);
      z-index: 30;
      transition: var(--ui-transition);
    }}
    #panel-toggle:hover {{
      background: var(--ui-accent-muted);
      color: var(--ui-accent);
      border-color: var(--ui-accent-ring);
    }}
    #panel-toggle:focus-visible {{ outline: none; box-shadow: 0 0 0 2px var(--ui-accent-muted), 0 0 0 1px var(--ui-accent-ring); }}
    .section-title {{ font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ui-text-secondary); margin: 12px 0 4px 0; }}
    .section-title:first-child {{ margin-top: 0; }}
    label {{ font-size: 11px; color: var(--ui-text-secondary); display: block; margin-top: 8px; }}
    input, select {{ width: 100%; background: var(--ui-bg-input); color: var(--ui-text-primary); border: 1px solid var(--ui-border-default); border-radius: 4px; padding: 6px 8px; font-size: 11px; margin-top: 4px; transition: var(--ui-transition); }}
    input:focus, select:focus {{ outline: none; border-color: var(--ui-accent-ring); box-shadow: 0 0 0 2px var(--ui-accent-muted); }}
    button.small {{ padding: 6px 12px; font-size: 11px; font-weight: 500; margin-top: 4px; cursor: pointer; border-radius: 4px; background: var(--ui-bg-elevated); color: var(--ui-text-primary); transition: var(--ui-transition); }}
    button.small:hover {{ background: var(--ui-bg-overlay); }}
    button.small:focus-visible {{ outline: none; box-shadow: 0 0 0 2px var(--ui-accent-muted); }}
    .draw-toggle-btn.drawing {{ background: var(--ui-success-bg); color: var(--ui-success-text); }}
    .draw-toggle-btn.drawing:hover {{ filter: brightness(1.12); }}
    .obj-list {{ max-height: 160px; overflow-y: auto; margin-top: 6px; }}
    /* Layout of tab object The list extends to the panel height,
       Use right panel scrolling only if it exceeds the total panel height */
    #object-list.obj-list {{ max-height: none; }}
    /* Flight scheduleThe height is expanded to show about 12 basic episodes. (10% increase) */
    #flightList.obj-list {{ max-height: 418px; }}
    /* Flight ConfigurationMore than 20 items can be displayed without vertical scrolling, and horizontal scrolling is possible. */
    #flightConfigList.obj-list {{ max-height: none; overflow-y: visible; overflow-x: auto; }}
    .obj-item {{ padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; background: var(--ui-bg-elevated); border: 1px solid var(--ui-border-subtle); font-size: 11px; cursor: pointer; transition: var(--ui-transition); }}
    .obj-item:hover {{ background: var(--ui-bg-overlay); border-color: var(--ui-border-default); }}
    #right-panel select:not(:disabled):hover:not(:focus),
    #right-panel input:not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="file"]):not([type="hidden"]):not([type="button"]):not([type="submit"]):not([type="reset"]):not(:disabled):hover:not(:focus) {{ background: var(--ui-bg-overlay); border-color: var(--ui-border-default); }}
    .obj-item.selected {{ border-color: var(--ui-accent-ring); background: var(--ui-accent-muted); }}
    .obj-item-header {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; }}
    .obj-item-title {{ font-weight: 500; flex: 1; min-width: 0; }}
    .obj-item-tag {{ font-size: 10px; color: var(--ui-text-secondary); flex-shrink: 0; }}
    .obj-item-time {{ color: var(--ui-text-secondary); font-weight: normal; margin-left: 4px; }}
    .obj-item-details {{ display: none; margin-top: 4px; color: var(--ui-text-primary); font-size: 10px; line-height: 1.4; opacity: 0.92; }}
    .obj-item.expanded .obj-item-details {{ display: block; }}
    .obj-item-delete {{ flex-shrink: 0; padding: 2px 6px; font-size: 10px; border-radius: 3px; background: transparent; color: var(--ui-error-text); cursor: pointer; transition: var(--ui-transition); }}
    .obj-item-delete:hover {{ background: var(--ui-bg-overlay); color: var(--ui-error-text); }}
    .flight-row {{ display:flex; gap:8px; align-items:stretch; margin-bottom:4px; }}
    .flight-row .obj-item {{ flex: 1.3; }}
    .flight-assign-panel {{ flex: 1; display:flex; align-items:flex-start; justify-content:space-between; gap:12px; font-size:11px; }}
    .flight-assign-col {{ display:flex; flex-direction:column; align-items:flex-start; gap:4px; min-width:0; }}
    .flight-assign-col-arr {{ width:110px; }}
    .flight-assign-col-term {{ width:140px; }}
    .flight-assign-col-dep {{ width:110px; }}
    .flight-assign-label {{ font-size:10px; color:var(--ui-text-secondary); letter-spacing:0.04em; }}
    .flight-assign-select {{ min-width:90px; padding:4px 8px; font-size:11px; background:var(--ui-bg-input); color:var(--ui-text-primary); border-radius:4px; border:1px solid var(--ui-border-default); }}
    /* Flight schedule Table: The columns are wide, the columns are as wide as the font size, and all the text in the selected column is visible. */
    .flight-schedule-table {{ width:100%; border-collapse:collapse; font-size:11px; margin-top:4px; table-layout:auto; }}
    .flight-schedule-table thead {{ position:sticky; top:0; z-index:1; }}
    .flight-schedule-table th {{ text-align:left; padding:6px 8px 6px 0; font-weight:600; color:var(--ui-text-primary); font-size:10px; letter-spacing:0.06em; text-transform:uppercase; white-space:nowrap; border-bottom:1px solid var(--ui-border-default); background:var(--ui-bg-elevated); }}
    .flight-schedule-table td {{ padding:4px 8px 4px 0; border-bottom:1px solid var(--ui-border-subtle); vertical-align:middle; white-space:nowrap; background:transparent; }}
    .flight-schedule-table th.flight-col-s {{ color: var(--style-fs-col-s, #22c55e); }}
    .flight-schedule-table th.flight-col-sd {{ color: var(--style-fs-col-sd, #007aff); }}
    .flight-schedule-table th.flight-col-e {{ color: var(--style-fs-col-e, #ff69b4); }}
    .flight-schedule-table th.flight-col-e.flight-col-rot,
    .flight-schedule-table td.flight-col-e.flight-col-rot {{ color: var(--style-fs-col-rot, #ffffff); }}
    .flight-schedule-table .flight-td-reg {{ font-weight:500; min-width:72px; }}
    .flight-schedule-table .flight-td-time {{ min-width:1em; }}
    .flight-schedule-table .flight-td-select {{ padding:4px 6px 4px 0; min-width:0; }}
    .flight-schedule-table .flight-td-select select {{ min-width:100px; width:70%; max-width:100%; padding:4px 8px; font-size:11px; }}
    .flight-schedule-table .flight-td-del {{ width:36px; text-align:center; padding:2px 0; }}
    .flight-config-input {{ width:72px; font-size:11px; text-align:right; }}
    .flight-config-table th.sticky-col,
    .flight-config-table td.sticky-col {{ position:sticky; left:0; z-index:2; background:rgba(17,24,39,0.98); }}
    .flight-config-table thead tr th:nth-child(2),
    .flight-config-table thead tr th:nth-child(3) {{ background: var(--ui-bg-elevated); color: var(--ui-text-primary); }}
    .flight-config-table tbody tr td:nth-child(2),
    .flight-config-table tbody tr td:nth-child(3) {{ background: transparent; color: var(--ui-text-primary); }}
    /* Allocation Gantt (Apron × Time, 10% height increase)
       - Fixed height makes horizontal scrollbar always appear at bottom of viewport (Scroll only the right column)
       - vertical scroll: .alloc-gantt-scroll-col + .alloc-gantt-label-col (JSsync with)
       - horizontal scroll: .alloc-gantt-scroll-col Only in → Scrollbar is always at the bottom */
    #allocationGantt {{ margin-top: 10px; padding: 10px 12px 10px 0; background: var(--ui-bg-elevated); border-radius: 12px; border: 1px solid var(--ui-border-default); width: 100%; max-width: 100%; min-width: 0; height: 700px; max-height: 700px; overflow: hidden; display: flex; flex-direction: column; box-sizing: border-box; }}
    .alloc-gantt-root {{ display:flex; align-items:stretch; width:100%; flex:1; min-height:0; }}
    .alloc-gantt-label-col {{ flex-shrink:0; width:122px; padding-left:12px; padding-right:8px; box-sizing:border-box; display:flex; flex-direction:column; overflow-y:auto; overflow-x:hidden; min-height:0; scrollbar-width:none; -ms-overflow-style:none; background:var(--ui-bg-surface); border-right:1px solid var(--ui-border-default); }}
    .alloc-gantt-label-col::-webkit-scrollbar {{ display:none; }}
    .alloc-gantt-scroll-col {{ flex:1; overflow:auto; min-height:0; }}
    /* Zoom Horizontal reflecting magnification inner wrapper (Full timeline width = 100% × zoom) */
    .alloc-gantt-inner {{ position:relative; min-width:100%; height:100%; }}
    /* Right dummy track(24px)and vertical 1:1 alignment — margin doesn't exist */
    .alloc-terminal-header {{ margin:0; height:24px; min-height:24px; line-height:24px; font-size: 11px; font-weight: 600; color: var(--ui-text-primary); text-transform: uppercase; letter-spacing: 0.06em; flex-shrink:0; display:flex; align-items:center; gap:4px; box-sizing:border-box; }}
    .alloc-section-toggle-icon {{ display:inline-block; width:12px; text-align:center; font-size:10px; color:var(--ui-text-secondary); }}
    .alloc-row {{ display:flex; align-items:stretch; margin:0; font-size:11px; }}
    /* Only used in the left fixed label column. sticky Instead, it is separated into a separate column.. */
    /* bar height 70%(40→28px), spacing between bars 1/4(4px→1px) */
    .alloc-row-label {{ height:28px; line-height:28px; margin:0; color:var(--ui-text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex-shrink:0; }}
    .alloc-row-track {{ flex:1; position:relative; height:28px; margin:0; border-radius:6px; background:transparent; border:none; overflow:visible; z-index:1; }}
    /* Apron top runway LDT/TOT Overview row: Exclude drop targets */
    .alloc-row-track[data-runway-legend="1"] {{ pointer-events: none; }}
    .alloc-runway-legend-label {{ font-size:10px; color:var(--ui-text-secondary); letter-spacing:0.02em; }}
    .alloc-time-grid-line {{ position:absolute; top:0; bottom:0; width:1px; background:var(--ui-border-strong); opacity:var(--style-gantt-grid-opacity, 0.4); transform:translateX(-0.5px); pointer-events:none; }}
    .alloc-apron-bg-slot {{ position:absolute; top:4px; bottom:4px; display:flex; align-items:center; justify-content:center; font-size:4px; font-weight:600; color:rgba(148,163,184,0.35); opacity:var(--style-gantt-apron-slot-opacity, 0.35); pointer-events:none; white-space:nowrap; }}
    .alloc-flight {{ position:absolute; top:3px; bottom:3px; border-radius:4px; background:var(--style-gantt-flight-bg, #38bdf8); color:var(--style-gantt-flight-fg, #0f172a); padding:1px 2px; font-size:4px; display:flex; flex-direction:column; justify-content:space-between; cursor:default; box-shadow:0 1px 3px var(--style-gantt-flight-shadow, rgba(0,0,0,0.5)); overflow:visible; transition:opacity 0.15s ease-out; z-index:1; }}
    .alloc-flight.conflict {{
      background:repeating-linear-gradient(135deg, var(--style-gantt-conflict, #7f1d1d) 0, var(--style-gantt-conflict, #7f1d1d) 6px, var(--style-gantt-conflict-stripe2, #111827) 6px, var(--style-gantt-conflict-stripe2, #111827) 12px);
      color:var(--style-gantt-conflict-text, #fee2e2);
      border:1px solid var(--style-gantt-conflict-border, #f87171);
    }}
    /* S‑Bar When checking SIBT‑SOBT For adjusting basic bar chart transparency */
    .alloc-flight.alloc-flight-sbar-dim {{ opacity:var(--style-gantt-flight-dim-opacity, 0.4); }}
    .alloc-flight.alloc-flight-selected {{ outline:1px solid var(--style-gantt-selected, #fbbf24); outline-offset:1px; box-shadow:0 0 0 1px var(--style-gantt-selected-ring, #0f172a), 0 1px 4px var(--style-gantt-selected-glow, rgba(251,191,36,0.4)); z-index:2; }}
    .alloc-flight-ovlp-badge {{ position:absolute; top:0; right:1px; font-size:3px; font-weight:700; padding:0 1px; border-radius:1px; background:var(--style-gantt-ovlp-bg, #e879f9); color:var(--style-gantt-ovlp-fg, #0f172a); pointer-events:none; line-height:1.1; }}
    .alloc-apron-tag {{ font-size:3px; font-weight:500; padding:0 1px; border-radius:9999px; background:var(--ui-bg-overlay); color:var(--ui-text-primary); border:1px solid var(--ui-border-default); align-self:flex-start; margin-bottom:0; white-space:nowrap; }}
    .alloc-flight-reg {{ font-weight:600; font-size:9px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:left; }}
    .alloc-flight-meta {{ font-size:9px; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:right; opacity:0.95; }}
    /* SLDT/SIBT/SOBT/STOT Auxiliary bar chart (S-Pointdragon)
       - vertical position/height S series point(.alloc-time-dot-s)and fit */
    .alloc-s-bar {{
      position:absolute;
      top:calc(29% + 3px);
      height:2px;
      background:var(--style-gantt-s-bar, #007aff);
      border-radius:9999px;
      opacity:var(--style-gantt-sbar-opacity, 0.8);
      pointer-events:none;
      z-index:5;
    }}
    /* EOBT(orig) vertical baseline (Remove transparency, fully opaque) */
    .alloc-s-line-orig {{ position:absolute; top:2px; bottom:2px; width:0; border-left:1px dashed #0f172a; pointer-events:none; z-index:2; }}
    .alloc-s-line-orig-solid {{ border-left-style:solid; }}
    /* EIBT/EOBT Auxiliary bar chart (E-Bar) : Margins adjusted to 70% row height */
    .alloc-e-bar {{
      position:absolute;
      top:3px;
      bottom:3px;
      background:var(--style-gantt-e-bar, #fb37c5);
      border-radius:4px;
      opacity:var(--style-gantt-ebar-opacity, 0.45);
      pointer-events:none;
      z-index:6;
    }}
    /* ELDT/EIBT, EOBT/ETOT Auxiliary bar chart (E-Pointdragon highlights)
       - vertical position/height E series point(.alloc-time-dot-e)and fit */
    .alloc-e2-bar {{
      position:absolute;
      top:calc(71% + 1px);
      height:2px;
      background:var(--style-gantt-e-bar, #fb37c5);
      border-radius:9999px;
      opacity:var(--style-gantt-e2bar-opacity, 0.9);
      pointer-events:none;
      z-index:7;
    }}
    /* SLDT/STOT viewpoint marker point (on the rock)
       - S-Point: Place in the area above the row
       - E-Point: Place in bottom area of ​​row */
    .alloc-time-dot {{ position:absolute; width:6px; height:6px; border-radius:50%; pointer-events:none; z-index:8; transform:translateX(-50%); }}
    .alloc-time-dot-s {{ top:32%; background:var(--style-gantt-s-bar, #007aff) !important; }}
    .alloc-time-dot-sd {{ top:32%; background:var(--style-gantt-s-bar, #007aff) !important; }}
    .alloc-time-dot-e {{ top:68%; background:var(--style-gantt-e-bar, #fb37c5); }}
    /* S-Point / E-Point triangle (under: LDT, stomach: TOT) - 20% The degree is smaller, and only the colors are different for each series. */
    .alloc-s-tri,
    .alloc-e-tri {{
      position:absolute;
      width:0;
      height:0;
      border-left:3px solid transparent;
      border-right:3px solid transparent;
      transform:translateX(-50%);
      pointer-events:none;
      z-index:9;
    }}
    /* The triangle below is S/E Each award based on series/placed in lower area */
    .alloc-s-tri-down {{ border-top:5px solid var(--style-gantt-s-bar, #007aff); top:44%; }}
    .alloc-e-tri-down {{ border-top:5px solid var(--style-gantt-e-bar, #fb37c5); top:76%; }}
    /* The upper triangle is S/E Each award based on series/placed in lower area */
    .alloc-s-tri-up {{ border-bottom:5px solid var(--style-gantt-s-bar, #007aff); top:20%; }}
    /* STOT(orig)dragon black upward triangle (S-Point top area) */
    .alloc-s-tri-orig-up {{ border-bottom:5px solid #0d0d0f; top:20%; }}
    .alloc-e-tri-up {{ border-bottom:5px solid var(--style-gantt-e-bar, #fb37c5); top:60%; }}
    /* Time axis overlay (Apron/Runway Common, at the bottom of the Gantt sticky)
       - Right timeline column(.alloc-gantt-scroll-col) Only inside sticky action
       - Use a scroll context independent of the left label column
       - height 24px Fixed to maintain visual alignment with left spacer */
    .alloc-time-axis-overlay {{ position:sticky; bottom:0; z-index:4; background:var(--ui-bg-elevated); border-top:1px solid var(--ui-border-subtle); height:24px; min-height:24px; max-height:24px; overflow:hidden; box-sizing:border-box; padding:0; }}
    .alloc-label-axis-spacer {{ height:24px; min-height:24px; flex-shrink:0; box-sizing:border-box; }}
    /* Remote Margin above section: left·Right paired with a row of equal height spacers */
    .alloc-gantt-section-spacer {{ height:8px; min-height:8px; flex-shrink:0; box-sizing:border-box; }}
    /* Remote Header: Existing margin-bottom 2px Contain effect within height → Right dummy track(20px)Same as */
    .alloc-remote-header {{ margin:0; height:20px; min-height:20px; line-height:18px; font-size:11px; font-weight:600; color:var(--ui-text-secondary); display:flex; align-items:center; gap:4px; flex-shrink:0; box-sizing:border-box; padding-bottom:2px; }}
    .alloc-time-axis-inner {{ position:relative; height:24px; min-height:24px; max-height:24px; min-width:100%; font-size:9px; line-height:24px; color:var(--ui-text-secondary); overflow:hidden; box-sizing:border-box; padding:0; }}
    .rwysep-timeline-root {{ display:flex; align-items:stretch; width:100%; }}
    .rwysep-timeline-label-col {{ display:flex; flex-direction:column; flex-shrink:0; width:110px; }}
    .rwysep-timeline-scroll-col {{ flex:1; overflow-x:auto; }}
    .rwysep-timeline-inner {{ min-width:100%; }}
    .rwysep-reg-tag {{ position:absolute; left:0; top:50%; transform:translate(-50%,-50%); font-size:9px; padding:2px 6px; border-radius:9999px; background:var(--ui-bg-elevated); color:var(--ui-text-primary); border:1px solid var(--ui-border-default); white-space:nowrap; pointer-events:none; }}
    .alloc-time-tick {{ position:absolute; bottom:0; transform:translateX(-50%); text-align:center; }}
    .alloc-time-tick-line {{ display:none; }}
    .alloc-time-tick-label {{ white-space:nowrap; }}
    /* Runway Separation Timeline exclusive (Reg × Time) */
    .rwysep-line-s {{ position:absolute; height:7%; top:26%; background:var(--style-rwysep-line-s, #38bdf8); border-radius:9999px; opacity:var(--style-rwysep-line-opacity, 0.55); pointer-events:none; }}
    .rwysep-line-e {{ position:absolute; height:7%; top:60%; background:var(--style-rwysep-line-e, #fb923c); border-radius:9999px; opacity:var(--style-rwysep-line-opacity, 0.55); pointer-events:none; }}
    .rwysep-tri {{ position:absolute; width:0; height:0; border-left:4px solid transparent; border-right:4px solid transparent; transform:translateX(-50%); pointer-events:none; }}
    .rwysep-head-row {{ display:flex; align-items:center; font-size:9px; color:var(--ui-text-secondary); margin-bottom:2px; }}
    /* Runway Separation Timeline header label width Apron Gantt label(alloc-row-label)Match the same as
       top S/E header track and angle Reg Ensure that the vertical scales of the row and lower time axes are accurately aligned.. */
    .rwysep-head-label {{ width:122px; padding:0 8px 0 12px; text-align:left; position:sticky; left:0; z-index:10; background:var(--ui-bg-elevated); }}
    .rwysep-head-track {{ flex:1; position:relative; height:14px; z-index:1; }}
    #directionModesList {{ width: 100%; }}
    .direction-mode-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 6px; width: 100%; min-width: 0; }}
    .direction-mode-row .direction-mode-name {{ flex: 1; min-width: 90px; width: 0; max-width: 100%; }}
    .direction-mode-row select.direction-mode-dir {{ flex-shrink: 0; width: auto; min-width: 95px; }}
    .direction-mode-row .direction-mode-delete {{ flex-shrink: 0; padding: 2px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; background: transparent; color: var(--ui-text-secondary); transition: var(--ui-transition); }}
    .direction-mode-row .direction-mode-delete:hover {{ background: var(--ui-bg-overlay); color: var(--ui-text-primary); }}
    #object-info {{ margin-top: 10px; padding: 8px; border-radius: 8px; background: var(--ui-bg-elevated); border: 1px solid var(--ui-border-default); font-size: 11px; display: none; color: var(--ui-text-secondary); }}
    #view3d-container {{ position: absolute; inset: 0; z-index: 10; display: none; pointer-events: auto; }}
    #view3d-container.active {{ display: block; }}
    #layout-name-bar {{ position: absolute; top: 8px; left: 50%; transform: translateX(-50%); z-index: 8; pointer-events: none; font-size: 11px; font-weight: 500; color: var(--ui-text-secondary); background: var(--ui-bg-elevated); border: 1px solid var(--ui-border-subtle); padding: 6px 12px; border-radius: 8px; letter-spacing: 0.02em; }}
    .right-panel-tabs {{ display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }}
    .right-panel-tab {{ flex: 1; min-width: 0; padding: 8px 8px; font-size: 11px; font-weight: 500; cursor: pointer; background: var(--ui-bg-elevated); color: var(--ui-text-secondary); border-radius: 6px; transition: var(--ui-transition); }}
    .right-panel-tab:hover {{ background: var(--ui-bg-overlay); color: var(--ui-text-primary); }}
    .right-panel-tab.active {{ background: var(--ui-accent-muted); color: var(--ui-accent); }}
    .right-panel-tab.active:hover {{ filter: brightness(1.1); }}
    .tab-content {{ display: none; width: 100%; }}
    .token-nodes {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-bottom: 8px; }}
    .token-nodes .token-node {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--ui-text-secondary); cursor: pointer; }}
    .token-nodes .token-node input {{ margin: 0; }}
    .token-object-pane {{ margin-bottom: 8px; }}
    .token-object-pane label {{ font-size: 11px; color: var(--ui-text-secondary); display: block; margin-top: 4px; }}
    .token-object-pane select {{ width: 100%; max-width: 200px; font-size: 11px; padding: 4px 8px; background: var(--ui-bg-input); color: var(--ui-text-primary); border: 1px solid var(--ui-border-default); border-radius: 4px; }}
    .token-auto-label {{ font-size: 11px; color: var(--ui-text-muted); }}
    .tab-content.active {{ display: block; }}
    .layout-save-load-tabs {{ display: flex; gap: 4px; margin-bottom: 8px; flex-wrap: wrap; }}
    .layout-save-load-tab {{ flex: 1; min-width: 0; padding: 6px 8px; font-size: 10px; font-weight: 500; letter-spacing: 0.04em; cursor: pointer; background: var(--ui-bg-elevated); color: var(--ui-text-secondary); border-radius: 6px; transition: var(--ui-transition); }}
    .layout-save-load-tab:hover {{ background: var(--ui-bg-overlay); color: var(--ui-text-primary); }}
    .layout-save-load-tab.active {{ background: var(--ui-accent-muted); color: var(--ui-accent); }}
    .layout-save-load-tab.active:hover {{ filter: brightness(1.1); }}
    .flight-subtab.active {{ background: var(--ui-accent-muted); color: var(--ui-accent); }}
    .flight-subtab.active:hover {{ filter: brightness(1.1); }}
    .layout-save-load-pane {{ display: none; }}
    .layout-save-load-pane.active {{ display: block; }}
    #layoutLoadList {{ max-height: 140px; overflow-y: auto; margin-top: 6px; }}
    #layoutLoadList .layout-load-item {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; padding: 6px 8px; margin-bottom: 4px; border-radius: 6px; background: var(--ui-bg-elevated); border: 1px solid var(--ui-border-subtle); font-size: 11px; cursor: pointer; transition: var(--ui-transition); }}
    #layoutLoadList .layout-load-item:hover {{ background: var(--ui-bg-overlay); border-color: var(--ui-accent-ring); }}
    #layoutLoadList .layout-load-name {{ flex: 1; min-width: 0; }}
    #layoutLoadList .layout-load-delete {{ flex-shrink: 0; padding: 2px 6px; font-size: 12px; cursor: pointer; border: none; background: transparent; color: var(--ui-text-secondary); border-radius: 3px; transition: var(--ui-transition); }}
    #layoutLoadList .layout-load-delete:hover {{ background: var(--ui-bg-overlay); color: var(--ui-text-primary); }}

    /* Runway Separation tab */
    #rwySepPanel {{ font-size: 11px; color: var(--ui-text-primary); }}
    .rwysep-rwy-bar {{ display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }}
    .rwysep-rwy-tabs {{ display:flex; flex-wrap:wrap; gap:4px; }}
    .rwysep-rwy-btn {{ background:var(--ui-bg-input); color:var(--ui-text-primary); padding:4px 12px; font-size:11px; font-weight:500; border-radius:9999px; cursor:pointer; transition:var(--ui-transition); }}
    .rwysep-rwy-btn:hover:not(.active) {{ background: var(--ui-bg-overlay); }}
    .rwysep-rwy-btn.active {{ background:var(--ui-accent-muted); color:var(--ui-accent); }}
    .rwysep-rwy-btn.active:hover {{ filter: brightness(1.1); }}
    .rwysep-rwy-del {{ background:transparent; border:none; color:var(--ui-text-muted); cursor:pointer; font-size:11px; padding:0 4px; }}
    .rwysep-rwy-del:hover {{ background: var(--ui-bg-overlay); color:var(--ui-text-primary); border-radius: 4px; }}
    .rwysep-block {{ margin-top:8px; padding:8px 12px; border-radius:8px; border:1px solid var(--ui-border-default); background:var(--ui-bg-elevated); }}
    .rwysep-label {{ font-size:10px; color:var(--ui-text-secondary); margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em; }}
    .rwysep-row {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:6px; }}
    .rwysep-row select {{ width:auto; min-width:90px; max-width:140px; font-size:11px; padding:4px 8px; background:var(--ui-bg-input); color:var(--ui-text-primary); border-radius:4px; border:1px solid var(--ui-border-default); }}
    .rwysep-matrix-wrap {{ margin-top:8px; overflow-x:auto; }}
    .rwysep-table {{ border-collapse:collapse; min-width:360px; }}
    .rwysep-table th, .rwysep-table td {{ border:1px solid var(--ui-border-subtle); padding:4px 6px; text-align:center; }}
    .rwysep-table th {{ background:var(--ui-bg-elevated); color:var(--ui-text-secondary); font-size:10px; letter-spacing:0.04em; text-transform:uppercase; }}
    .rwysep-table input {{ width:56px; background:var(--ui-bg-input); border:1px solid var(--ui-border-default); color:var(--ui-text-primary); font-size:10px; padding:2px 4px; text-align:center; border-radius:4px; }}
    .kpi-shell {{ display:flex; flex-direction:column; gap:12px; margin-top:8px; }}
    .kpi-toolbar {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; padding:16px; border-radius:14px; border:1px solid var(--ui-border-default); background:linear-gradient(180deg, rgba(124, 106, 247, 0.16), rgba(28, 28, 31, 0.92)); }}
    .kpi-toolbar-copy {{ min-width:0; }}
    .kpi-toolbar-eyebrow {{ font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#c4b5fd; }}
    .kpi-toolbar-title {{ margin-top:6px; font-size:18px; font-weight:600; line-height:1.3; color:var(--ui-text-primary); }}
    .kpi-toolbar-subtitle {{ margin-top:6px; font-size:11px; line-height:1.55; color:var(--ui-text-secondary); max-width:520px; }}
    .kpi-status-chip {{ flex-shrink:0; display:inline-flex; align-items:center; gap:6px; padding:8px 10px; border-radius:9999px; background:rgba(13, 13, 15, 0.55); border:1px solid rgba(255,255,255,0.12); color:var(--ui-text-primary); font-size:10px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }}
    .kpi-status-chip::before {{ content:''; width:8px; height:8px; border-radius:9999px; background:var(--ui-accent); box-shadow:0 0 0 4px rgba(124, 106, 247, 0.16); }}
    #kpiDashboard {{ display:flex; flex-direction:column; gap:12px; }}
    .kpi-summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; }}
    .kpi-card {{ position:relative; overflow:hidden; padding:16px; border-radius:14px; border:1px solid rgba(124, 106, 247, 0.24); background:linear-gradient(180deg, rgba(124, 106, 247, 0.20), rgba(44, 33, 73, 0.58) 42%, rgba(20, 20, 22, 0.98)); box-shadow:0 8px 24px rgba(0,0,0,0.18); }}
    .kpi-card::after {{ content:''; position:absolute; inset:0 auto auto 0; width:100%; height:1px; background:linear-gradient(90deg, rgba(255,255,255,0.16), rgba(255,255,255,0)); pointer-events:none; }}
    .kpi-card.accent {{ background:linear-gradient(180deg, rgba(124, 106, 247, 0.22), rgba(44, 33, 73, 0.60) 42%, rgba(20, 20, 22, 0.98)); border-color:rgba(124, 106, 247, 0.26); }}
    .kpi-card.success {{ background:linear-gradient(180deg, rgba(124, 106, 247, 0.20), rgba(44, 33, 73, 0.56) 42%, rgba(20, 20, 22, 0.98)); border-color:rgba(124, 106, 247, 0.22); }}
    .kpi-card.warning {{ background:linear-gradient(180deg, rgba(124, 106, 247, 0.20), rgba(44, 33, 73, 0.56) 42%, rgba(20, 20, 22, 0.98)); border-color:rgba(124, 106, 247, 0.22); }}
    .kpi-card.danger {{ background:linear-gradient(180deg, rgba(124, 106, 247, 0.20), rgba(44, 33, 73, 0.56) 42%, rgba(20, 20, 22, 0.98)); border-color:rgba(124, 106, 247, 0.22); }}
    .kpi-card-label {{ font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:var(--ui-text-secondary); }}
    .kpi-card-value {{ margin-top:8px; font-size:24px; font-weight:700; line-height:1.1; color:var(--ui-text-primary); }}
    .kpi-card-meta {{ margin-top:8px; font-size:11px; line-height:1.55; color:var(--ui-text-secondary); }}
    .kpi-card-submeta {{ margin-top:6px; font-size:10px; color:var(--ui-text-muted); }}
    .kpi-panel-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
    .kpi-panel {{ padding:16px; border-radius:14px; border:1px solid var(--ui-border-default); background:var(--ui-bg-elevated); }}
    .kpi-panel-header {{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:12px; }}
    .kpi-panel-title {{ font-size:13px; font-weight:600; color:var(--ui-text-primary); }}
    .kpi-panel-badge {{ padding:4px 8px; border-radius:9999px; background:var(--ui-bg-overlay); border:1px solid var(--ui-border-subtle); color:var(--ui-text-secondary); font-size:10px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; }}
    .kpi-metric-list {{ display:flex; flex-direction:column; gap:10px; }}
    .kpi-metric-row {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; padding-top:10px; border-top:1px solid var(--ui-border-subtle); }}
    .kpi-metric-row:first-child {{ padding-top:0; border-top:none; }}
    .kpi-metric-label {{ font-size:11px; color:var(--ui-text-secondary); line-height:1.45; }}
    .kpi-metric-values {{ text-align:right; }}
    .kpi-metric-primary {{ font-size:15px; font-weight:700; color:#ffffff; }}
    .kpi-metric-secondary {{ margin-top:4px; font-size:10px; color:var(--ui-text-muted); }}
    .kpi-chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:12px; }}
    .kpi-chart-card {{ padding:16px; border-radius:14px; border:1px solid var(--ui-border-default); background:var(--ui-bg-elevated); }}
    .kpi-chart-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; margin-bottom:12px; }}
    .kpi-chart-title {{ font-size:13px; font-weight:600; color:var(--ui-text-primary); }}
    .kpi-chart-subtitle {{ margin-top:4px; font-size:11px; line-height:1.45; color:var(--ui-text-secondary); }}
    .kpi-chart-legend {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
    .kpi-legend-item {{ display:inline-flex; align-items:center; gap:6px; font-size:10px; color:var(--ui-text-secondary); }}
    .kpi-legend-swatch {{ width:10px; height:10px; border-radius:9999px; }}
    .kpi-chart-scroll {{ overflow-x:auto; overflow-y:hidden; padding-bottom:4px; }}
    .kpi-chart-frame {{ min-width:100%; }}
    .kpi-chart-frame text {{ font-family:var(--ui-font); }}
    .kpi-chart-axis {{ fill:var(--ui-text-secondary); font-size:10px; }}
    .kpi-chart-grid-line {{ stroke:rgba(255,255,255,0.08); stroke-width:1; }}
    .kpi-chart-dot {{ fill:#f0f0f2; stroke:#0d0d0f; stroke-width:2; }}
    .kpi-detail-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:12px; }}
    .kpi-table-card {{ padding:16px; border-radius:14px; border:1px solid var(--ui-border-default); background:var(--ui-bg-elevated); }}
    .kpi-table-wrap {{ margin-top:12px; max-height:340px; overflow:auto; border-radius:12px; border:1px solid var(--ui-border-subtle); }}
    .kpi-table {{ width:100%; min-width:100%; border-collapse:collapse; }}
    .kpi-table th {{ position:sticky; top:0; z-index:1; padding:10px 12px; background:var(--ui-bg-surface); color:var(--ui-text-secondary); font-size:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; text-align:left; border-bottom:1px solid var(--ui-border-default); }}
    .kpi-table td {{ padding:10px 12px; font-size:11px; color:var(--ui-text-primary); border-bottom:1px solid var(--ui-border-subtle); white-space:nowrap; }}
    .kpi-table tr:hover td {{ background:rgba(255,255,255,0.02); }}
    .kpi-table tr.kpi-row-highlight td {{ background:rgba(124, 106, 247, 0.08); }}
    .kpi-badge {{ display:inline-flex; align-items:center; padding:4px 8px; border-radius:9999px; font-size:10px; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; border:1px solid transparent; }}
    .kpi-badge.ok {{ background:rgba(61, 214, 140, 0.14); border-color:rgba(61, 214, 140, 0.3); color:#b8f0d0; }}
    .kpi-badge.fail {{ background:rgba(248, 113, 113, 0.14); border-color:rgba(248, 113, 113, 0.3); color:#fca5a5; }}
    .kpi-empty-state {{ padding:28px 16px; border-radius:14px; border:1px dashed var(--ui-border-default); background:rgba(255,255,255,0.02); text-align:center; color:var(--ui-text-secondary); font-size:11px; line-height:1.6; }}
    #api-warning-banner {{ display: none; background: var(--ui-error-bg); color: var(--ui-error-text); padding: 10px 12px; font-size: 11px; border-radius: 8px; margin-bottom: 12px; line-height: 1.45; border: 1px solid var(--ui-error-border); }}
    #api-warning-banner strong {{ color: var(--ui-text-primary); font-weight: 600; }}
    #flight-tooltip {{ position: absolute; pointer-events: none; padding: 4px 8px; font-size: 11px; border-radius: 6px; background: var(--ui-bg-elevated); color: var(--ui-text-primary); border: 1px solid var(--ui-border-default); display: none; z-index: 40; box-shadow: 0 1px 3px rgba(0,0,0,0.4); max-width: 280px; }}
    #btnRunSimulation {{ margin-left: 8px; }}
    #remoteTerminalAccess {{ margin-top: 4px; padding: 8px; border-radius: 8px; border: 1px solid var(--ui-border-default); background: var(--ui-bg-input); font-size: 11px; color: var(--ui-text-primary); max-height: 120px; overflow: auto; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="app">
    <div id="layout-name-bar"></div>
    <div id="toolbar">
      <div id="sim-controls-container" style="display:none;">
        <button type="button" class="tool-btn" id="btnPlayFlights" title="Play">▶ Play</button>
        <button type="button" class="tool-btn" id="btnPauseFlights" title="Pause">⏸ Pause</button>
        <select id="flightSpeed">
          {_flight_speed_options_html}
        </select>
        <label for="flightSimSlider">Current</label>
        <input type="range" id="flightSimSlider" min="0" max="100" value="0" step="1" />
        <span id="flightSimTimeLabel">00:00:00</span>
      </div>
      <div id="view-toggle">
        <button class="tool-btn" id="btnView2D" title="2D view">2D</button>
        <button class="tool-btn" id="btnView3D" title="3D orbit view">3D</button>
        <button class="tool-btn" id="btnResetView" title="Fit full grid">Fit</button>
        <button class="tool-btn" id="btnGlobalUpdate" title="Refresh all views and calculations" style="margin-left:8px;">Update</button>
        <button type="button" class="tool-btn" id="btnRunSimulation" title="Run simulation (requires API)">Run Sim</button>
      </div>
    </div>
    <div id="canvas-container">
      <canvas id="grid-canvas"></canvas>
    </div>
    <div id="info-bar">
      <span id="hint">Draw terminal: click grid points, then near first point to close. Drag to pan, scroll to zoom.</span>
      <span id="coord"></span>
    </div>
    <div id="flight-tooltip"></div>
    <div id="right-panel">
      <button type="button" id="panel-toggle" title="Drag to resize; click toggles open/closed">◀</button>
      <div class="panel-content">
        <div id="api-warning-banner">
          <strong>API Not connected</strong><br/>
          Save/Load/Run SimulationTo use <strong>python run_app.py</strong>After running it with<br/>
          <strong>http://127.0.0.1:8501</strong> Please access. (streamlit run Doesn't work when used)
        </div>
        <div class="right-panel-tabs">
          <button type="button" class="right-panel-tab active" data-tab="settings">Layout</button>
          <button type="button" class="right-panel-tab" data-tab="flight">Flight</button>
          <button type="button" class="right-panel-tab" data-tab="rwysep">Runway</button>
          <button type="button" class="right-panel-tab" data-tab="allocation">Apron</button>
          <button type="button" class="right-panel-tab" data-tab="simulation">KPI</button>
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
          <option value="runwayPath">Runway</option>
          <option value="runwayTaxiway">Runway Taxiway</option>
          <option value="taxiway">Taxiway</option>
          <option value="apronTaxiway">Apron Taxiway</option>
          <option value="edge">Edge</option>
        </select>

        <div id="settings-grid" class="settings-pane">
          <label>Cell size (m)</label>
          <input type="number" id="gridCellSize" min="{_ui_g_min_cs}" max="{_ui_g_max_cs}" value="{CELL_SIZE}" step="{_ui_g_cs_step}" />
          <label>Columns</label>
          <input type="number" id="gridCols" min="{_ui_g_min_dim}" max="{_ui_g_max_dim}" value="{GRID_COLS}" />
          <label>Rows</label>
          <input type="number" id="gridRows" min="{_ui_g_min_dim}" max="{_ui_g_max_dim}" value="{GRID_ROWS}" />
        </div>
        <div id="settings-terminal" class="settings-pane" style="display:none;">
          <label>Building name</label>
          <input type="text" id="terminalName" placeholder="e.g. T1" />
          <label>Floors</label>
          <input type="number" id="terminalFloors" min="{_ui_term_floors_min}" max="{_ui_term_floors_max}" value="{_ui_term_floors}" step="1" />
          <label>Floor-to-floor height (m)</label>
          <input type="number" id="terminalFloorToFloor" min="{_ui_term_f2f_min}" max="{_ui_term_f2f_max}" value="{_ui_term_f2f}" step="{_ui_term_f2f_step}" />
          <label>Departure capacity</label>
          <input type="number" id="terminalDepartureCapacity" min="0" value="{_ui_term_dep}" step="1" />
          <label>Arrival capacity</label>
          <input type="number" id="terminalArrivalCapacity" min="0" value="{_ui_term_arr}" step="1" />
          <button class="small draw-toggle-btn" id="btnTerminalDraw">Draw</button>
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
          <button class="small draw-toggle-btn" id="btnPbbDraw">Draw</button>
        </div>
        <div id="settings-remote" class="settings-pane" style="display:none;">
          <label>Name</label>
          <input type="text" id="remoteName" placeholder="e.g. R001" />
          <label>Category (ICAO)</label>
          <select id="remoteCategory">
            <option value="A">A</option><option value="B">B</option><option value="C" selected>C</option>
            <option value="D">D</option><option value="E">E</option><option value="F">F</option>
          </select>
          <label style="margin-top:6px;">Available terminals</label>
          <div id="remoteTerminalAccess">
            <!-- Terminal checkboxes are rendered dynamically -->
          </div>
          <button class="small draw-toggle-btn" id="btnRemoteDraw">Draw</button>
        </div>
        <div id="settings-edge" class="settings-pane" style="display:none;">
          <p style="font-size:11px;color:#9ca3af;line-height:1.45;margin:0;">Graph edges are derived from the current taxiway network (path graph). They are not saved in layout JSON. Click a segment on the map or pick an item in the list. Names 001–999 are assigned in stable graph order.</p>
        </div>
        <div id="settings-taxiway" class="settings-pane" style="display:none;">
          <label>Name</label>
          <input type="text" id="taxiwayName" placeholder="e.g. Taxiway A" />
          <label>Width (m)</label>
          <input type="number" id="taxiwayWidth" min="10" max="100" value="{_ui_tw_w}" step="1" />
          <div id="runwayMinArrVelocityWrap" style="display:none;margin-top:6px;">
            <label>Min Arr Velocity (m/s)</label>
            <input type="number" id="runwayMinArrVelocity" min="1" max="150" value="{_ui_rw_min_arr}" step="1" />
            <p style="font-size:10px;color:#9ca3af;margin-top:2px;">Arrival airspeed on a runway segment will not drop below this value..</p>
            <div id="runwayLineupDistWrap" style="margin-top:8px;">
              <label>Line up Point (m, start → end)</label>
              <input type="number" id="runwayLineupDistM" min="0" max="500000" value="{_ui_rw_lineup}" step="1" />
              <p style="font-size:10px;color:#9ca3af;margin-top:2px;">Startat End directionby this distance(m) spot. The departure route passes through this point and continues to the end of the runway..</p>
            </div>
          </div>
          <div id="taxiwayAvgVelocityWrap" style="display:none;">
            <label>Avg Move Velocity (m/s)</label>
            <input type="number" id="taxiwayAvgMoveVelocity" min="1" max="50" value="{_ui_tw_avg}" step="0.5" />
          </div>
          <div id="runwayExitExtras" style="display:none;margin-top:6px;">
            <label>Max Exit Velocity</label>
            <input type="number" id="taxiwayMaxExitVel" min="1" max="150" value="{_ui_ex_max}" step="1" />
            <label style="margin-top:4px;">Min Exit Velocity</label>
            <input type="number" id="taxiwayMinExitVel" min="1" max="150" value="{_ui_ex_min}" step="1" />
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
          <button class="small draw-toggle-btn" id="btnTaxiwayDraw">Draw</button>
        </div>
        <div id="settings-apronTaxiway" class="settings-pane" style="display:none;">
          <p style="font-size:11px;color:#9ca3af;margin-top:4px;">
            Click a Contact Stand (PBB end) and a Taxiway point to create an Apron–Taxiway link.
          </p>
          <button class="small draw-toggle-btn" id="btnApronLinkDraw">Draw</button>
        </div>

        <div class="section-title">Objects</div>
        <div id="object-list" class="obj-list"></div>
        <div id="object-info">Select an object on the grid or from the list.</div>
        </div>

        <div id="tab-flight" class="tab-content">
          <div class="section-title">Flight</div>
          <div class="layout-save-load-tabs" style="margin-top:4px;">
            <button type="button" class="layout-save-load-tab flight-subtab active" data-flight-subtab="schedule">Flight Schedule</button>
            <button type="button" class="layout-save-load-tab flight-subtab" data-flight-subtab="config">Arrival Configuration</button>
          </div>
          <!-- Arr / Dep The choice is reserved for internal compatibility., UIhidden in -->
          <label style="display:none;">Arr / Dep</label>
          <select id="flightArrDep" style="display:none;">
            <option value="Arr" selected>Arr (Arrival)</option>
            <option value="Dep">Dep (Departure)</option>
          </select>
          <div id="flightPaneSchedule">
            <label>SIBT (Scheduled In Block Time)</label>
            <input type="text" id="flightTime" placeholder="yes: 0, 12:30, 09:23:45 (minute / HH:MM:SS)" />
            <label>Aircraft type</label>
            <select id="flightAircraftType">
              <!-- Populated from INFORMATION.tiers.aircraft.types -->
            </select>
            <label>Reg. number</label>
            <input type="text" id="flightReg" placeholder="yes: HL1234" />
            <label>Airline Code</label>
            <input type="text" id="flightAirlineCode" placeholder="yes: KE" />
            <label>Flight Number</label>
            <input type="text" id="flightFlightNumber" placeholder="yes: KE0081" />
            <label>Dwell time (minutes, stand occupancy time)</label>
            <input type="number" id="flightDwell" min="0" max="{_ui_flight_dwell_max}" value="{_ui_flight_dwell}" step="{_ui_flight_dwell_step}" />
            <label>Min Dwell (minutes, minimum dwell·Turnaround guarantee)</label>
            <input type="number" id="flightMinDwell" min="0" max="{_ui_flight_dwell_max}" value="{_ui_flight_min_dwell}" step="{_ui_flight_dwell_step}" title="In case of delayed arrival EOBT = EIBT + Min Dwell Adjust to (0Not applicable to this side)" />
            <button class="small" id="btnAddFlight">+ Add Flight</button>
            <div id="flightError" style="color:#f97316;font-size:11px;margin-top:4px;"></div>
            <div class="section-title" style="margin-top:10px;">Flight schedule</div>
            <div id="flightList" class="obj-list"></div>
          </div>

          <div id="flightPaneConfig" style="display:none;margin-top:8px;">
            <div class="section-title">Arrival Configuration</div>
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
                <input type="checkbox" id="chkShowSPoints" />
                S‑Point
              </label>
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowSBars" checked />
                S‑Bar
              </label>
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowEBar" />
                E‑Bar
              </label>
              <label style="font-size:11px;color:#9ca3af;display:flex;align-items:center;gap:4px;margin:0;">
                <input type="checkbox" id="chkShowEPoints" />
                E‑Point
              </label>
            </div>
          </div>
          <div id="allocationGantt"></div>
        </div>

        <div id="tab-simulation" class="tab-content">
          <div class="section-title">KPI</div>
          <div class="kpi-shell">
            <div class="kpi-toolbar">
              <div class="kpi-toolbar-copy">
                <div class="kpi-toolbar-eyebrow">Operations Dashboard</div>
                <div class="kpi-toolbar-title">High-impact KPI snapshot for apron, runway, and delay performance</div>
                <div class="kpi-toolbar-subtitle">This dashboard is rendered only on the initial page load and when you click <strong>Update</strong>, so the KPI view stays stable while you edit schedules and layouts.</div>
              </div>
              <div class="kpi-status-chip" id="kpiSnapshotStatus">Initializing snapshot</div>
            </div>
            <div id="kpiDashboard"></div>
          </div>
        </div>

        <div id="tab-saveload" class="tab-content">
          <div class="section-title">Layout Save / Load</div>
          <div class="layout-save-load-tabs">
            <button type="button" class="layout-save-load-tab active" data-sltab="saveas">Save as</button>
            <button type="button" class="layout-save-load-tab" data-sltab="save">Save</button>
            <button type="button" class="layout-save-load-tab" data-sltab="load">Load</button>
          </div>
          <div id="layout-saveas-pane" class="layout-save-load-pane active">
            <label>Layout name</label>
            <input type="text" id="layoutName" placeholder="yes: base_layout" />
            <button type="button" class="small" id="btnSaveLayout" style="margin-top:6px;">Save as</button>
            <div id="layoutMessage" style="font-size:11px;color:#9ca3af;margin-top:4px;"></div>
          </div>
          <div id="layout-save-pane" class="layout-save-load-pane">
            <p style="font-size:11px;color:#9ca3af;">Overwrites the currently loaded layout file.</p>
            <button type="button" class="small" id="btnSaveCurrentLayout" style="margin-top:6px;">Save current state</button>
            <div id="layoutMessageSave" style="font-size:11px;color:#9ca3af;margin-top:4px;"></div>
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
    const GRID_VIEW_BG = {json.dumps(_GRID_VIEW_BG)};
    let GRID_COLS = {GRID_COLS};
    let GRID_ROWS = {GRID_ROWS};
    let CELL_SIZE = {CELL_SIZE};

    const _tiers = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers) ? INFORMATION.tiers : {{}};
    const _layoutTier = _tiers.layout || {{}};
    const _flightTier = _tiers.flight_schedule || _tiers.flight || {{}};
    const _algoTier = _tiers.algorithm || {{}};
    const _styleTier = _tiers.style || {{}};
    const _ganttStyle = _styleTier.gantt || {{}};
    const _canvas2dStyle = _styleTier.canvas2d || {{}};
    function c2dObjectSelectedStroke() {{ return _canvas2dStyle.objectSelectedStroke || '#e9d5ff'; }}
    function c2dObjectSelectedFill() {{ return _canvas2dStyle.objectSelectedFill || 'rgba(196, 181, 253, 0.42)'; }}
    function c2dObjectSelectedDashStroke() {{ return _canvas2dStyle.objectSelectedDashStroke || 'rgba(255, 252, 255, 0.95)'; }}
    function c2dObjectSelectedGlow() {{ return _canvas2dStyle.objectSelectedGlow || 'rgba(167, 139, 250, 0.9)'; }}
    function c2dObjectSelectedGlowBlur() {{
      const n = Number(_canvas2dStyle.objectSelectedGlowBlur);
      return (isFinite(n) && n >= 0) ? n : 22;
    }}
    const _threeDStyle = _styleTier.threeD || {{}};
    const GANTT_COLORS = {{
      S_BAR: _ganttStyle.sBar || '#007aff',
      S_SERIES: _ganttStyle.sSeries || '#38bdf8',
      E_BAR: _ganttStyle.eBar || '#fb37c5',
      E_SERIES: _ganttStyle.eSeries || '#fb923c',
      CONFLICT: _ganttStyle.conflict || '#7f1d1d',
      SELECTED: _ganttStyle.selected || '#fbbf24',
    }};
    const _apronAc = _layoutTier.apronAircraft || {{}};
    const _acScaleByCat = (_apronAc.scaleByIcaoCategory && typeof _apronAc.scaleByIcaoCategory === 'object') ? _apronAc.scaleByIcaoCategory : {{}};
    function apronAircraftScaleForIcao(code) {{
      const c = String(code || '').toUpperCase();
      const v = Number(_acScaleByCat[c]);
      if (isFinite(v) && v > 0) return v;
      const d = Number(_acScaleByCat.default);
      return (isFinite(d) && d > 0) ? d : 1.0;
    }}
    const _ac2d = _apronAc.twoD || {{}};
    const _acSil = (_ac2d.silhouette && typeof _ac2d.silhouette === 'object') ? _ac2d.silhouette : {{}};
    function apron2DGlyphFill() {{ return _ac2d.fillColor || '#ff2f92'; }}
    const _ac3d = _apronAc.threeD || {{}};
    function _numOr(v, def) {{
      const n = Number(v);
      return (isFinite(n) && n > 0) ? n : def;
    }}
    function hexToThreeColor(hex) {{
      const s = String(hex || '').replace('#', '').trim();
      if (!s || s.length < 6) return 0xff2f92;
      const n = parseInt(s.slice(0, 6), 16);
      return isFinite(n) ? n : 0xff2f92;
    }}
    function threeOpacity(v, def) {{
      const o = Number(v);
      return (isFinite(o) && o >= 0 && o <= 1) ? o : def;
    }}
    const _schedAlgo = _algoTier.scheduledTimes || {{}};
    const SCHED_DWELL_FLOOR_MIN = (function() {{
      const v = Number(_schedAlgo.dwellFloorMin);
      return (isFinite(v) && v >= 0) ? v : 20;
    }})();
    const RSEP_MISSING_MATRIX_SEC = (function() {{
      const v = Number(_schedAlgo.rsepMissingMatrixSeparationSec);
      return (isFinite(v) && v >= 0) ? v : 90;
    }})();
    const TIME_AXIS_CFG = _algoTier.timeAxis || {{}};
    function _taNum(k, def) {{
      const v = Number(TIME_AXIS_CFG[k]);
      return (isFinite(v) && v >= 0) ? v : def;
    }}
    const GANTT_PAD_MIN = _taNum('apronGanttPadMin', 20);
    const RWY_SEP_TIMELINE_PAD_MIN = _taNum('runwaySepTimelinePadMin', 10);
    const TICK_STEP_SPAN_LE60 = _taNum('tickStepWhenSpanLe60Min', 10);
    const TICK_STEP_SPAN_LE240 = _taNum('tickStepWhenSpanLe240Min', 30);
    const TICK_STEP_ELSE = _taNum('tickStepElseMin', 60);
    const MAX_TICKS_SHOWN = (function() {{
      const v = Math.floor(Number(TIME_AXIS_CFG.maxTicksShown));
      return (isFinite(v) && v >= 2) ? v : 6;
    }})();
    const PATH_SEARCH_CFG = _algoTier.pathSearch || {{}};
    const TAXIWAY_HEURISTIC_COST = (function() {{
      const v = Number(PATH_SEARCH_CFG.taxiwayHeuristicCost);
      return (isFinite(v) && v > 0) ? v : 200;
    }})();
    const _ix = _layoutTier.interaction || {{}};
    function _ixNum(k, def) {{
      const v = Number(_ix[k]);
      return (isFinite(v) && v >= 0) ? v : def;
    }}
    const DRAG_THRESH = _ixNum('dragThresholdPx', 5);
    const HIT_TERM_VTX_CF = _ixNum('hitTerminalVertexCellFactor', 0.6);
    const HIT_TW_VTX_CF = _ixNum('hitTaxiwayVertexCellFactor', 0.6);
    const HIT_TW_SEG_CF = _ixNum('hitTaxiwayAlongCellFactor', 0.8);
    const HIT_PBB_END_CF = _ixNum('hitPbbEndCellFactor', 0.8);
    const TRY_PBB_MAX_EDGE_CF = _ixNum('tryPlacePbbMaxEdgeCellFactor', 1.0);
    const FLIGHT_TOOLTIP_CF = _ixNum('flightTooltipCellFactor', 1.2);
    const TERM_CLOSE_POLY_CF = _ixNum('terminalClosePolygonCellFactor', 0.6);
    const PBB_PREVIEW_LEN_CF = _ixNum('pbbPreviewLengthCellFactor', 0.9);

    const canvas = document.getElementById('grid-canvas');
    const container = document.getElementById('canvas-container');
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
      directionModes: [],
      // current selection/Loaded layout name (Simulation Available upon request)
      currentLayoutName: String(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout'),
      // Flight / simulation state
      flights: [],
      simTimeSec: 0,
      simStartSec: 0,
      simDurationSec: 0,
      simPlaying: false,
      simSpeed: {json.dumps(float(_ui_default_sim_speed))},
      hasSimulationResult: false,
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
      pbbDrawing: false,
      remoteDrawing: false,
      apronLinkDrawing: false,
      apronLinkTemp: null,
      hoverCell: null,
      vttArrCacheRev: 0,
      derivedGraphEdges: [],
    }};
    let hookSyncFlightPanelFromSelection = null;
    const DEFAULT_AIRLINE_CODES = (function() {{
      const a = _flightTier.defaultAirlineCodes;
      return (Array.isArray(a) && a.length) ? a.map(String) : ['KE', '7C', 'DL'];
    }})();
    const PATH_LAYOUT_MODES = ['runwayPath', 'runwayTaxiway', 'taxiway'];
    function pathTypeFromLayoutMode(layoutMode) {{
      if (layoutMode === 'runwayPath') return 'runway';
      if (layoutMode === 'runwayTaxiway') return 'runway_exit';
      if (layoutMode === 'taxiway') return 'taxiway';
      return 'taxiway';
    }}
    function layoutModeFromPathType(pt) {{
      if (pt === 'runway') return 'runwayPath';
      if (pt === 'runway_exit') return 'runwayTaxiway';
      return 'taxiway';
    }}
    function isPathLayoutMode(m) {{
      return PATH_LAYOUT_MODES.indexOf(m) >= 0;
    }}
    function settingModeValueForHit(hit) {{
      if (!hit || !hit.type) return null;
      if (hit.type === 'terminal') return 'terminal';
      if (hit.type === 'pbb') return 'pbb';
      if (hit.type === 'remote') return 'remote';
      if (hit.type === 'taxiway') return layoutModeFromPathType((hit.obj && hit.obj.pathType) || 'taxiway');
      if (hit.type === 'apronLink') return 'apronTaxiway';
      return null;
    }}
    function cancelActiveLayoutDrawingState() {{
      state.pbbDrawing = false;
      state.remoteDrawing = false;
      state.apronLinkDrawing = false;
      state.previewPbb = null;
      state.previewRemote = null;
    }}
    function syncDrawToggleButton(elementId, isDrawing) {{
      const btn = document.getElementById(elementId);
      if (!btn) return;
      btn.textContent = isDrawing ? 'Drawing' : 'Draw';
      btn.classList.toggle('drawing', isDrawing);
    }}
    function toggleLayoutDrawMode(flagKey, previewKey, tempKey) {{
      state.selectedObject = null;
      if (state[flagKey]) {{
        state[flagKey] = false;
        if (previewKey) state[previewKey] = null;
        if (tempKey) state[tempKey] = null;
      }} else {{
        state[flagKey] = true;
        if (previewKey) state[previewKey] = null;
        if (tempKey) state[tempKey] = null;
      }}
      syncPanelFromState();
      draw();
    }}
    function handlePbbOrRemoteMouseUp2D(mode, wx, wy) {{
      if (mode === 'pbb' && state.pbbDrawing) {{
        if (tryPlacePbbAt(wx, wy)) {{ syncPanelFromState(); draw(); }}
        return true;
      }}
      if (mode === 'remote' && state.remoteDrawing) {{
        const prev = state.previewRemote;
        if (prev && !prev.overlap && tryPlaceRemoteAt(prev.col, prev.row)) {{ syncPanelFromState(); draw(); }}
        return true;
      }}
      return false;
    }}
    function tryCommitStandPlacement3D(mode, wx, wy, col, row) {{
      if (mode === 'pbb' && state.pbbDrawing) {{
        if (tryPlacePbbAt(wx, wy)) {{ syncPanelFromState(); updateObjectInfo(); update3DScene(); }}
        return;
      }}
      if (mode === 'remote' && state.remoteDrawing) {{
        if (tryPlaceRemoteAt(col, row)) {{ syncPanelFromState(); updateObjectInfo(); update3DScene(); }}
      }}
    }}
    function findLayoutObjectByListType(typ, idr) {{
      if (typ === 'terminal') return state.terminals.find(t => t.id === idr);
      if (typ === 'pbb') return state.pbbStands.find(p => p.id === idr);
      if (typ === 'remote') return state.remoteStands.find(r => r.id === idr);
      if (typ === 'taxiway') return state.taxiways.find(tw => tw.id === idr);
      if (typ === 'apronLink') return state.apronLinks.find(lk => lk.id === idr);
      if (typ === 'layoutEdge') return (state.derivedGraphEdges || []).find(function(e) {{ return e.id === idr; }});
      if (typ === 'flight') return state.flights.find(f => f.id === idr);
      return null;
    }}
    function removeLayoutObjectFromState(type, id) {{
      const removedTaxiway = (type === 'taxiway')
        ? (state.taxiways || []).find(function(tw) {{ return tw.id === id; }})
        : null;
      if (type === 'terminal') state.terminals = state.terminals.filter(t => t.id !== id);
      else if (type === 'pbb') state.pbbStands = state.pbbStands.filter(p => p.id !== id);
      else if (type === 'remote') state.remoteStands = state.remoteStands.filter(r => r.id !== id);
      else if (type === 'taxiway') state.taxiways = state.taxiways.filter(tw => tw.id !== id);
      else if (type === 'apronLink') state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
      else if (type === 'flight') state.flights = state.flights.filter(f => f.id !== id);
      else if (type === 'layoutEdge') {{}}
      if (removedTaxiway) {{
        if (removedTaxiway.pathType === 'runway_exit') {{
          (state.flights || []).forEach(function(f) {{
            if (!f || f.sampledArrRet !== id) return;
            f.sampledArrRet = null;
            f.arrRetFailed = false;
            f.arrRotSec = null;
            f.arrRetDistM = null;
            f.arrVRetInMs = null;
            f.arrVRetOutMs = null;
            f.__schedRetRotRev = null;
            f.__schedVttArrRev = null;
            f.__schedVttArrMin = null;
            f.noWayArr = false;
          }});
        }}
        if (typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
      }}
    }}
    function syncPathFieldVisibilityForPathType(pt) {{
      const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
      const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
      const exitWrap = document.getElementById('runwayExitExtras');
      const rwDirWrap = document.getElementById('runwayDirectionWrap');
      if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = (pt === 'taxiway') ? 'block' : 'none';
      if (runwayMinArrWrap) runwayMinArrWrap.style.display = (pt === 'runway') ? 'block' : 'none';
      if (exitWrap) exitWrap.style.display = (pt === 'runway_exit') ? 'block' : 'none';
      if (rwDirWrap) rwDirWrap.style.display = (pt === 'runway') ? 'block' : 'none';
    }}
    function mergeTaxiwaysFromLayoutObject(obj) {{
      if (!obj || typeof obj !== 'object') return [];
      const newSchema = Object.prototype.hasOwnProperty.call(obj, 'runwayPaths') ||
        Object.prototype.hasOwnProperty.call(obj, 'runwayTaxiways');
      if (newSchema) {{
        const out = [];
        (obj.runwayPaths || []).forEach(function(tw) {{
          const o = Object.assign({{}}, tw);
          o.pathType = 'runway';
          out.push(o);
        }});
        (obj.runwayTaxiways || []).forEach(function(tw) {{
          const o = Object.assign({{}}, tw);
          o.pathType = 'runway_exit';
          delete o.rwySepConfig;
          out.push(o);
        }});
        (obj.taxiways || []).forEach(function(tw) {{
          const o = Object.assign({{}}, tw);
          if (o.pathType !== 'runway' && o.pathType !== 'runway_exit') o.pathType = 'taxiway';
          if (o.pathType !== 'runway') delete o.rwySepConfig;
          out.push(o);
        }});
        return out;
      }}
      if (Array.isArray(obj.taxiways)) return obj.taxiways.slice();
      return [];
    }}
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
      state.taxiways = mergeTaxiwaysFromLayoutObject(obj);
      if (Array.isArray(obj.apronLinks)) state.apronLinks = obj.apronLinks.slice();
      if (Array.isArray(obj.directionModes) && obj.directionModes.length) {{
        state.directionModes = obj.directionModes.slice();
      }}
      if (Array.isArray(obj.flights)) {{
        state.flights = obj.flights.slice();
        state.flights.forEach(f => {{
          const t = f.token || {{}};
          // aircraftType/code: legacy JSONIn codeThere can only be. aircraftTypeIf there is code Judo, if there is no codeas aircraftType matching
          if (f.aircraftType && typeof getCodeForAircraft === 'function') {{
            f.code = getCodeForAircraft(f.aircraftType);
          }} else if (f.code && typeof AIRCRAFT_TYPES !== 'undefined') {{
            const match = AIRCRAFT_TYPES.find(a => a.icao === f.code);
            f.aircraftType = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
          }}
          // JSONMinimum saved at token form: arrRunwayId, apronId, terminalId, depRunwayId
          f.arrRunwayId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
          f.depRunwayId = f.depRunwayId || t.depRunwayId || null;
          f.terminalId = f.terminalId || t.terminalId || null;
          // Flight-parking areaID matching: JSON token.apronIdUsed only as a source. If this value holds Allocationalso maintained
          const apronId = t.apronId != null ? t.apronId : (f.standId != null ? f.standId : null);
          f.standId = apronId;
          f.token = {{
            nodes: Array.isArray(t.nodes) ? t.nodes.slice() : ['runway','taxiway','apron','terminal'],
            runwayId: f.arrRunwayId || null,
            apronId: apronId,
            terminalId: f.terminalId || null,
            depRunwayId: f.depRunwayId || null,
          }};
          if (!f.airlineCode) f.airlineCode = DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)];
          if (!f.flightNumber) f.flightNumber = f.airlineCode + String(Math.floor(1000 + Math.random() * 9000));
        }});
      }} else {{
        state.flights = [];
      }}
      // The layout of the parking lot is JSON(apronId)Restored only. timeline/No route-based automatic reassignment
      // Do not autoplay simulation
      state.simPlaying = false;
      // flightsWhen the path changes·Simulation length after timeline calculation·List update (to be playable)
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
      // bird UI: taxiwayDirectionModedirectly from tw.directionSave to (clockwise / counter_clockwise / both)
      if (tw.direction != null) {{
        const d = tw.direction;
        if (d === 'topToBottom') return 'clockwise';
        if (d === 'bottomToTop') return 'counter_clockwise';
        return d || 'both';
      }}
      // Old version JSON compatible: directionModeId + state.directionModes use
      if (tw.directionModeId) {{
        const m = state.directionModes.find(d => d.id === tw.directionModeId);
        if (m && m.direction) return m.direction;
      }}
      return 'both';
    }}

    // ---- Runway Separation config (from Information.json) ----
    const _rwy = _tiers.runway || {{}};
    const _sepUi = (_rwy.separationUi && typeof _rwy.separationUi === 'object') ? _rwy.separationUi : {{}};
    const RSEP_COLOR_THRESHOLDS = (function() {{
      const arr = _sepUi.inputColorThresholdsSec;
      if (Array.isArray(arr) && arr.length) {{
        return arr.map(x => Number(x)).filter(x => isFinite(x) && x > 0).sort((a, b) => a - b);
      }}
      return [90, 120, 150];
    }})();
    const RSEP_LEGEND_LAB = (_sepUi.legendLabels && typeof _sepUi.legendLabels === 'object') ? _sepUi.legendLabels : {{}};
    function rsepLegendFmt(tpl, a0, a1) {{
      let s = String(tpl || '');
      if (a1 != null && s.indexOf('{{1}}') >= 0) return s.replace('{{0}}', String(a0)).replace('{{1}}', String(a1));
      return s.replace('{{0}}', String(a0));
    }}
    const RSEP_COLOR_STYLES = [
      {{ bg: '#0d2018', color: '#68d391', border: '#68d39155' }},
      {{ bg: '#0d1a28', color: '#63b3ed', border: '#63b3ed55' }},
      {{ bg: '#1e1e08', color: '#f6e05e', border: '#f6e05e55' }},
      {{ bg: '#280d0d', color: '#fc8181', border: '#fc818155' }},
    ];
    const _stds = _rwy.standards || {{}};
    const RSEP_STD_CATS = {{
      'ICAO': (_stds.ICAO && _stds.ICAO.categories) ? _stds.ICAO.categories : ['J','H','M','L'],
      'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categories) ? _stds['RECAT-EU'].categories : ['A','B','C','D','E','F'],
    }};
    const RSEP_SEQ_TYPES = Object.assign({{ 'ARR→ARR': 'matrix', 'DEP→DEP': 'matrix', 'ARR→DEP': 'lead-1d', 'DEP→ARR': 'trail-1d' }}, _sepUi.seqTypes || {{}});
    const RSEP_MODE_SEQS = (function() {{
      const def = {{ ARR: ['ARR→ARR'], DEP: ['DEP→DEP'], MIX: ['ARR→ARR','DEP→DEP','ARR→DEP','DEP→ARR'] }};
      const ms = _sepUi.modeSequences || {{}};
      const out = {{}};
      ['ARR','DEP','MIX'].forEach(k => {{
        const a = ms[k];
        out[k] = (Array.isArray(a) && a.length) ? a.slice() : def[k].slice();
      }});
      return out;
    }})();
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
      const th = RSEP_COLOR_THRESHOLDS;
      for (let i = 0; i < th.length; i++) {{
        if (n < th[i]) return RSEP_COLOR_STYLES[i] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
      }}
      return RSEP_COLOR_STYLES[th.length] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
    }}
    function rsepLegendHtml(filled, total) {{
      const th = RSEP_COLOR_THRESHOLDS;
      const countColor = filled === total ? '#68d391' : '#9ca3af';
      let html = '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
      const lab = RSEP_LEGEND_LAB;
      if (th.length) {{
        const st0 = rsepColorForValue(Math.max(0, th[0] - 1));
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st0.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st0.color + ';">' + escapeHtml(rsepLegendFmt(lab.ltFirst || '<{{0}}s', th[0])) + '</span></span>';
        for (let i = 1; i < th.length; i++) {{
          const lo = th[i - 1], hi = th[i];
          const mid = lo + (hi - lo) / 2;
          const st = rsepColorForValue(mid);
          const text = rsepLegendFmt(lab.rangeMid || '{{0}}–{{1}}s', lo, hi - 1);
          html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st.color + ';">' + escapeHtml(text) + '</span></span>';
        }}
        const lastT = th[th.length - 1];
        const stL = rsepColorForValue(lastT + 1000);
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + stL.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + stL.color + ';">' + escapeHtml(rsepLegendFmt(lab.gteLast || '≥{{0}}s', lastT)) + '</span></span>';
      }}
      html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
      html += '</div>';
      return html;
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
      // standards have changed JSONBecause it may come in cats If the number does not match, reset
      const cfg = rw.rwySepConfig;
      if (!RSEP_STD_CATS[cfg.standard]) {{
        rw.rwySepConfig = rsepMakeConfig('ICAO');
        return rw.rwySepConfig;
      }}
      return cfg;
    }}
    let dpr = window.devicePixelRatio || 1;
    let ctx = (canvas && typeof canvas.getContext === 'function') ? canvas.getContext('2d') : null;

    function screenToWorld(sx, sy) {{
      return [(sx - state.panX) / state.scale, (sy - state.panY) / state.scale];
    }}
    function cellToPixel(col, row) {{ return [col * CELL_SIZE, row * CELL_SIZE]; }}
    function getTaxiwayAvgMoveVelocityForPath(path) {{
      if (path && typeof path.avgMoveVelocity === 'number' && isFinite(path.avgMoveVelocity) && path.avgMoveVelocity > 0)
        return Math.max(1, Math.min(50, path.avgMoveVelocity));
      const el = document.getElementById('taxiwayAvgMoveVelocity');
      const v = el ? Number(el.value) : 10;
      return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
    }}
    function pixelToCell(x, y) {{
      const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
      let col = Math.round(x / cs * 2) / 2;
      let row = Math.round(y / cs * 2) / 2;
      col = Math.max(0, Math.min(GRID_COLS, col));
      row = Math.max(0, Math.min(GRID_ROWS, row));
      return [col, row];
    }}
    const ICAO_STAND_SIZE_M = (function() {{
      const m = _layoutTier.standSizesMByIcaoCategory;
      if (m && typeof m === 'object') {{
        const o = {{}};
        Object.keys(m).forEach(k => {{ o[k] = Number(m[k]); }});
        return o;
      }}
      return {{ A: 20, B: 30, C: 40, D: 50, E: 60, F: 80 }};
    }})();
    function getStandSizeMeters(cat) {{ return ICAO_STAND_SIZE_M[cat] || 40; }}
    function getStandBoundsRect(cx, cy, sizeM) {{
      const h = sizeM / 2;
      return {{ left: cx - h, right: cx + h, top: cy - h, bottom: cy + h }};
    }}
    function rectsOverlap(a, b) {{
      // Treat only positive-area intersection as overlap.
      // If When two squares touch only by a line or point(When sides overlap or only edges touch)is not considered overlap..
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
    function tryPlacePbbAt(wx, wy) {{
      let bestEdge = null, bestD2 = Infinity;
      state.terminals.forEach(t => {{
        if (!t.closed || t.vertices.length < 2) return;
        let cx = 0, cy = 0;
        t.vertices.forEach(v => {{ const [px, py] = cellToPixel(v.col, v.row); cx += px; cy += py; }});
        cx /= t.vertices.length || 1; cy /= t.vertices.length || 1;
        for (let i = 0; i < t.vertices.length; i++) {{
          const v1 = t.vertices[i], v2 = t.vertices[(i + 1) % t.vertices.length];
          const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
          const near = closestPointOnSegment(p1, p2, [wx, wy]);
          if (near) {{
            const d2 = dist2(near, [wx, wy]);
            if (d2 < bestD2) {{ bestD2 = d2; bestEdge = {{ near, p1, p2, col: v1.col, row: v1.row, cx, cy }}; }}
          }}
        }}
      }});
      const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
      if (!bestEdge || bestD2 >= maxD2) return false;
      const [ex, ey] = bestEdge.near, [x1, y1] = bestEdge.p1, [x2, y2] = bestEdge.p2;
      let nx = -(y2 - y1), ny = x2 - x1;
      const len = Math.hypot(nx, ny) || 1; nx /= len; ny /= len;
      const toClickX = wx - ex, toClickY = wy - ey;
      if (nx * toClickX + ny * toClickY < 0) {{ nx *= -1; ny *= -1; }}
      const category = document.getElementById('standCategory').value || 'C';
      const standSize = getStandSizeMeters(category);
      const minLen = standSize / 2 + 3;
      const lenCells = parseInt(document.getElementById('pbbLength').value || '2', 10);
      const lenPx = Math.max(lenCells * CELL_SIZE * PBB_PREVIEW_LEN_CF, minLen);
      const newPbb = {{ x1: ex, y1: ey, x2: ex + nx * lenPx, y2: ey + ny * lenPx, category }};
      if (pbbStandOverlapsExisting(newPbb)) return false;
      pushUndo();
      state.pbbStands.push({{
        id: id(),
        name: document.getElementById('standName').value.trim() || ('Contact Stand ' + (state.pbbStands.length + 1)),
        x1: ex, y1: ey, x2: ex + nx * lenPx, y2: ey + ny * lenPx,
        category: newPbb.category, edgeCol: bestEdge.col, edgeRow: bestEdge.row
      }});
      return true;
    }}
    function tryPlaceRemoteAt(col, row) {{
      if (col < 0 || row < 0 || col > GRID_COLS || row > GRID_ROWS) return false;
      const category = document.getElementById('remoteCategory').value || 'C';
      const [cx, cy] = cellToPixel(col, row);
      const size = getStandSizeMeters(category);
      const bounds = getStandBoundsRect(cx, cy, size);
      if (standOverlapsExisting(bounds)) return false;
      pushUndo();
      const baseName = (document.getElementById('remoteName') && document.getElementById('remoteName').value.trim()) || 'R001';
      const usedNames = new Set((state.remoteStands || []).map(s => (s.name || '').trim()).filter(Boolean));
      let finalName = baseName;
      if (usedNames.has(finalName)) {{
        let idx = 1;
        while (usedNames.has(baseName + ' (' + idx + ')')) idx++;
        finalName = baseName + ' (' + idx + ')';
      }}
      state.remoteStands.push({{ id: id(), col, row, category, name: finalName }});
      return true;
    }}
    function taxiwayOverlapsAnyTerminal(tw) {{
      if (!tw || !tw.vertices || tw.vertices.length < 2) return false;
      const vertsPix = tw.vertices.map(v => cellToPixel(v.col, v.row));
      // each vertex Check which terminal is in
      for (let t = 0; t < state.terminals.length; t++) {{
        const term = state.terminals[t];
        if (!term.closed || term.vertices.length < 3) continue;
        const termPix = term.vertices.map(v => cellToPixel(v.col, v.row));
        for (let i = 0; i < vertsPix.length; i++) {{
          if (pointInPolygonXY(vertsPix[i], termPix)) return true;
        }}
        // Segments vs terminal polygon edges Check for intersection
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
        // Taxiway vertex is in the terminal
        for (let k = 0; k < vertsPix.length; k++) {{
          if (pointInPolygonXY(vertsPix[k], termPix)) return true;
        }}
        // Taxiway Whether the segment intersects the terminal edge
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
      // Avg move velocityis individual Taxiway Serialize the settings as is
      if (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) {{
        copy.avgMoveVelocity = tw.avgMoveVelocity;
      }}
      if (tw.pathType === 'runway' && typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0) {{
        copy.minArrVelocity = Math.max(1, Math.min(150, tw.minArrVelocity));
      }}
      if (tw.pathType === 'runway') {{
        if (typeof tw.lineupDistM === 'number' && isFinite(tw.lineupDistM) && tw.lineupDistM >= 0) copy.lineupDistM = tw.lineupDistM;
        else delete copy.lineupDistM;
        delete copy.lineup_point;
        delete copy.dep_point;
        delete copy.depPointPos;
      }}
      // Runway separationsilver physics runway(runway path)Meaning only; exit/common TWKeys attached to are not saved.
      if (tw.pathType === 'runway' && tw.rwySepConfig) copy.rwySepConfig = tw.rwySepConfig;
      else delete copy.rwySepConfig;
      return copy;
    }}
    function partitionTaxiwaysForPersist(list) {{
      const runwayPaths = [];
      const runwayTaxiways = [];
      const taxiways = [];
      (list || []).forEach(function(tw) {{
        const ser = serializeTaxiwayWithEndpoints(tw);
        const pt = tw.pathType || 'taxiway';
        delete ser.pathType;
        if (pt === 'runway') runwayPaths.push(ser);
        else if (pt === 'runway_exit') runwayTaxiways.push(ser);
        else taxiways.push(ser);
      }});
      return {{ runwayPaths: runwayPaths, runwayTaxiways: runwayTaxiways, taxiways: taxiways }};
    }}
    function serializeCurrentLayout() {{
      return {{
        grid: {{
          cols: GRID_COLS,
          rows: GRID_ROWS,
          cellSize: CELL_SIZE
        }},
        // In case of duplicate names Objects Shape visible on the panel(yes: "Stand 1 (2)")Save as
        terminals: makeUniqueNamedCopy(state.terminals, 'name'),
        pbbStands: makeUniqueNamedCopy(state.pbbStands, 'name'),
        remoteStands: state.remoteStands.slice(),
        ...(function() {{
          const p = partitionTaxiwaysForPersist(state.taxiways);
          return {{ runwayPaths: p.runwayPaths, runwayTaxiways: p.runwayTaxiways, taxiways: p.taxiways }};
        }})(),
        apronLinks: state.apronLinks.slice(),
        directionModes: state.directionModes.slice(),
        // Flight-parking areaID matching(apronId)second JSONIf you include it in , when loading AllocationRestored as is
        flights: state.flights.map(function(f) {{
          const copy = {{ }};
          // First the basics·Order in which you want time-related fields(S(orig) > S(d) > S(final) > E(final), Each group ldt > ibt > obt > tot net)Fill with
          // NOTE: E(orig) line(eldtMin_orig/eibtMin_orig/eobtMin_orig/etotMin_orig)silver
          //       JSONwithout saving to final Eline(eldtMin/eibtMin/eobtMin/etotMin)save only.
          const orderedKeys = [
            'id',
            'reg',
            'airlineCode',
            'flightNumber',
            'aircraftType',
            'code',
            'velocity',
            'timeMin',
            'dwellMin',
            'minDwellMin',
            'noWayArr',
            'noWayDep',
            // S (orig): SLDT, SIBT, SOBT, STOT
            'sldtMin_orig',
            'sibtMin_orig',
            'sobtMin_orig',
            'stotMin_orig',
            // S (d): SLDT(d), SIBT(d), SOBT(d), STOT(d)
            'sldtMin_d',
            'sibtMin_d',
            'sobtMin_d',
            'stotMin_d',
            // S (final): SLDT, SIBT, SOBT, STOT
            'sldtMin',
            'sibtMin',
            'sobtMin',
            'stotMin',
            // E (final): ELDT, EIBT, EOBT, ETOT
            'eldtMin',
            'eibtMin',
            'eobtMin',
            'etotMin',
            // Other indicators
            'vttADelayMin',
            'arrRotSec',
            'eOverlapPushed',
            'sampledArrRet',
            'sampledRetName',
            'arrRetFailed'
          ];
          orderedKeys.forEach(function(k) {{
            // sibtMinis explicitly final SIBTto leave a,
            // If there is no original field sibtMin_dMake a copy of.
            if (k === 'sibtMin') {{
              if (
                Object.prototype.hasOwnProperty.call(f, 'sibtMin') &&
                f.sibtMin != null
              ) {{
                copy.sibtMin = f.sibtMin;
              }} else if (
                Object.prototype.hasOwnProperty.call(f, 'sibtMin_d') &&
                f.sibtMin_d != null
              ) {{
                copy.sibtMin = f.sibtMin_d;
              }}
              return;
            }}
            if (
              Object.prototype.hasOwnProperty.call(f, k) &&
              k !== 'timeline' &&
              k !== 'arrDep' &&
              k !== 'token' &&
              k !== 'arrRunwayId' &&
              k !== 'depRunwayId' &&
              k !== 'terminalId' &&
              k !== 'standId'
            ) {{
              copy[k] = f[k];
            }}
          }});
          // The remaining fields are appended in the original order.
          for (const k in f) {{
            if (
              k === 'timeline' ||
              k === 'arrDep' ||
              k === 'token' ||
              k === 'arrRunwayId' ||
              k === 'depRunwayId' ||
              k === 'terminalId' ||
              k === 'standId' ||
              Object.prototype.hasOwnProperty.call(copy, k)
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
    function formatMinToHM(m) {{
      const hh = Math.floor(m / 60);
      const mm = Math.floor(m % 60);
      return hh + ':' + (mm < 10 ? '0' : '') + mm;
    }}
    function findNearestItem(candidates, getPoint, wx, wy, maxD2) {{
      const click = [wx, wy];
      let best = null;
      let bestD2 = maxD2;
      for (let i = 0; i < candidates.length; i++) {{
        const c = candidates[i];
        const pt = getPoint(c);
        if (!pt || pt.length < 2) continue;
        const d2 = dist2(pt, click);
        if (d2 < bestD2) {{
          bestD2 = d2;
          best = c;
        }}
      }}
      return best;
    }}
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
        const corners = getPBBStandCorners(pbb);
        if (pointInPolygonXY(click, corners))
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
          const hitD2 = (CELL_SIZE * HIT_TW_SEG_CF + halfW) ** 2;
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
      const maxD2 = (CELL_SIZE * HIT_TERM_VTX_CF) ** 2;
      const cands = [];
      state.terminals.forEach(t => {{
        t.vertices.forEach((v, idx) => {{
          cands.push({{ terminalId: t.id, index: idx, v }});
        }});
      }});
      const best = findNearestItem(cands, c => cellToPixel(c.v.col, c.v.row), wx, wy, maxD2);
      return best ? {{ terminalId: best.terminalId, index: best.index }} : null;
    }}

    function hitTestTaxiwayVertex(wx, wy) {{
      if (!state.selectedObject || state.selectedObject.type !== 'taxiway') return null;
      const tw = state.selectedObject.obj;
      if (!tw || !tw.vertices || tw.vertices.length === 0) return null;
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * HIT_TW_VTX_CF) ** 2;
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
      syncDrawToggleButton('btnTerminalDraw', !!state.terminalDrawingId);
      if (state.selectedObject && state.selectedObject.type === 'pbb') {{
        const pbb = state.selectedObject.obj;
        const nameInput = document.getElementById('standName');
        const catSel = document.getElementById('standCategory');
        if (nameInput) nameInput.value = pbb.name || '';
        if (catSel) catSel.value = pbb.category || 'C';
      }}
      if (state.selectedObject && state.selectedObject.type === 'remote') {{
        const st = state.selectedObject.obj;
        const nameInput = document.getElementById('remoteName');
        const catSel = document.getElementById('remoteCategory');
        if (nameInput) nameInput.value = st.name || '';
        if (catSel) catSel.value = st.category || 'C';
        const accWrap = document.getElementById('remoteTerminalAccess');
        if (accWrap) {{
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {{ return {{
            id: t.id,
            name: (t.name || '').trim() || 'Terminal'
          }}; }});
          const allowed = Array.isArray(st.allowedTerminals) ? st.allowedTerminals : [];
          if (!terms.length) {{
            accWrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No terminals.</div>';
          }} else {{
            accWrap.innerHTML = terms.map(function(t) {{
              const checked = allowed.includes(t.id) ? ' checked' : '';
              return '' +
                '<label class="remote-term-row" style="display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;">' +
                  '<span class="remote-term-checkbox" style="position:relative;display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:4px;border:1px solid #4b5563;background:#020617;">' +
                    '<input type="checkbox" class="remote-term-check" data-term-id="' + String(t.id || '').replace(/"/g,'&quot;') + '"' + checked + ' ' +
                           'style="position:absolute;inset:0;opacity:0;cursor:pointer;" />' +
                    (checked ? '<span style="width:8px;height:8px;border-radius:2px;background:#22c55e;"></span>' : '') +
                  '</span>' +
                  '<span class="remote-term-label" style="flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' +
                    escapeHtml(t.name || 'Terminal') +
                  '</span>' +
                '</label>';
            }}).join('');
          }}
        }}
      }}
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        const nameInput = document.getElementById('taxiwayName');
        const widthInput = document.getElementById('taxiwayWidth');
        const maxExitInput = document.getElementById('taxiwayMaxExitVel');
        const minExitInput = document.getElementById('taxiwayMinExitVel');
        if (nameInput) nameInput.value = tw.name || '';
        const widthDefault = tw.pathType === 'runway' ? 60 : 15;
        if (widthInput) widthInput.value = tw.width != null ? tw.width : widthDefault;
        const avgVelInput = document.getElementById('taxiwayAvgMoveVelocity');
        if (avgVelInput) avgVelInput.value = (tw.avgMoveVelocity != null ? tw.avgMoveVelocity : 10);
        syncPathFieldVisibilityForPathType(tw.pathType || 'taxiway');
        const runwayMinArrInput = document.getElementById('runwayMinArrVelocity');
        if (runwayMinArrInput) {{
          const mav = (typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
            ? Math.max(1, Math.min(150, tw.minArrVelocity))
            : 15;
          runwayMinArrInput.value = mav;
        }}
        const runwayLineupInput = document.getElementById('runwayLineupDistM');
        if (runwayLineupInput && tw.pathType === 'runway') {{
          const lv = getEffectiveRunwayLineupDistM(tw);
          runwayLineupInput.value = String(lv);
        }}
        if (maxExitInput) maxExitInput.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : 30;
        if (minExitInput) {{
          const minVal = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
            ? tw.minExitVelocity
            : 15;
          minExitInput.value = minVal;
        }}
        const modeSel = document.getElementById('taxiwayDirectionMode');
        const d = getTaxiwayDirection(tw);
        if (modeSel) modeSel.value = d;
        const rwDirInPane = document.getElementById('runwayDirectionInTaxiwayPane');
        if (rwDirInPane) rwDirInPane.value = d;
      }} else {{
        const rm = settingModeSelect ? settingModeSelect.value : '';
        if (isPathLayoutMode(rm)) syncPathFieldVisibilityForPathType(pathTypeFromLayoutMode(rm));
        else {{
          const rwWrap = document.getElementById('runwayDirectionWrap');
          if (rwWrap) rwWrap.style.display = 'none';
          const exitWrap = document.getElementById('runwayExitExtras');
          if (exitWrap) exitWrap.style.display = 'none';
          const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
          if (runwayMinArrWrap) runwayMinArrWrap.style.display = 'none';
          const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
          if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = 'none';
        }}
      }}
      syncDrawToggleButton('btnTaxiwayDraw', !!state.taxiwayDrawingId);
      syncDrawToggleButton('btnApronLinkDraw', !!state.apronLinkDrawing);
      syncDrawToggleButton('btnPbbDraw', !!state.pbbDrawing);
      syncDrawToggleButton('btnRemoteDraw', !!state.remoteDrawing);
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
      if (state.selectedObject && state.selectedObject.type === 'remote') {{
        var st = state.selectedObject.obj;
        if (el('remoteName')) st.name = (el('remoteName').value || '').trim();
        if (el('remoteCategory')) st.category = el('remoteCategory').value || 'C';
        const accWrap = document.getElementById('remoteTerminalAccess');
        if (accWrap) {{
          const checks = accWrap.querySelectorAll('.remote-term-check');
          const allowed = [];
          checks.forEach(function(ch) {{
            if (ch.checked) {{
              const id = ch.getAttribute('data-term-id');
              if (id) allowed.push(id);
            }}
          }});
          st.allowedTerminals = allowed;
        }}
      }}
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        var tw = state.selectedObject.obj;
        if (el('taxiwayName')) tw.name = (el('taxiwayName').value || '').trim();
        if (el('taxiwayWidth')) tw.width = Math.max(10, Math.min(100, Number(el('taxiwayWidth').value) || 15));
        if (el('taxiwayMaxExitVel')) {{
          const mv = Number(el('taxiwayMaxExitVel').value);
          if (tw.pathType === 'runway_exit') tw.maxExitVelocity = isFinite(mv) && mv > 0 ? mv : null;
          else delete tw.maxExitVelocity;
        }}
        if (el('taxiwayMinExitVel') && tw.pathType === 'runway_exit') {{
          const mv2 = Number(el('taxiwayMinExitVel').value);
          let v = isFinite(mv2) && mv2 > 0 ? mv2 : 15;
          if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
          tw.minExitVelocity = v;
        }} else if (tw.pathType !== 'runway_exit') {{
          delete tw.minExitVelocity;
        }}
        if (tw.pathType === 'runway' && el('runwayDirectionInTaxiwayPane')) {{
          tw.direction = el('runwayDirectionInTaxiwayPane').value || 'both';
        }} else if (el('taxiwayDirectionMode')) {{
          tw.direction = el('taxiwayDirectionMode').value || 'both';
        }}
        if (el('taxiwayAvgMoveVelocity')) {{
          var v = Number(el('taxiwayAvgMoveVelocity').value);
          tw.avgMoveVelocity = (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
        }}
        if (el('runwayMinArrVelocity')) {{
          const mav = Number(el('runwayMinArrVelocity').value);
          if (tw.pathType === 'runway') {{
            tw.minArrVelocity = (typeof mav === 'number' && isFinite(mav) && mav > 0) ? Math.max(1, Math.min(150, mav)) : 15;
          }} else {{
            delete tw.minArrVelocity;
          }}
        }}
        if (el('runwayLineupDistM') && tw.pathType === 'runway') {{
          const lx = Number(el('runwayLineupDistM').value);
          tw.lineupDistM = (typeof lx === 'number' && isFinite(lx) && lx >= 0) ? lx : 0;
        }} else if (tw.pathType !== 'runway') {{
          delete tw.lineupDistM;
        }}
      }}
    }}

    function syncSettingsPaneToMode() {{
      const mode = settingModeSelect ? settingModeSelect.value : 'grid';
      if (mode !== 'edge' && state.selectedObject && state.selectedObject.type === 'layoutEdge') state.selectedObject = null;
      document.querySelectorAll('.settings-pane').forEach(el => {{ el.style.display = 'none'; }});
      const paneKey = isPathLayoutMode(mode) ? 'taxiway' : mode;
      const pane = document.getElementById('settings-' + paneKey);
      if (pane) pane.style.display = 'block';
      if (isPathLayoutMode(mode)) syncPathFieldVisibilityForPathType(pathTypeFromLayoutMode(mode));
      if (typeof renderObjectList === 'function') renderObjectList();
    }}

    settingModeSelect.addEventListener('change', function() {{
      cancelActiveLayoutDrawingState();
      syncSettingsPaneToMode();
    }});
    syncSettingsPaneToMode();

    let activeTab = 'settings';
    function switchToTab(tabId) {{
      activeTab = tabId;
      cancelActiveLayoutDrawingState();
      document.querySelectorAll('.right-panel-tab').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      const tabBtn = document.querySelector('.right-panel-tab[data-tab="' + tabId + '"]');
      const tabEl = document.getElementById('tab-' + tabId);
      if (tabBtn) tabBtn.classList.add('active');
      if (tabEl) tabEl.classList.add('active');
      if (tabId === 'flight') {{
        if (state.selectedObject && state.selectedObject.type === 'flight' && typeof hookSyncFlightPanelFromSelection === 'function')
          hookSyncFlightPanelFromSelection();
      }}
      if (tabId === 'allocation' && typeof renderFlightGantt === 'function') renderFlightGantt();
      if (tabId === 'rwysep' && typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
    }}
    document.querySelectorAll('.right-panel-tab').forEach(btn => {{
      btn.addEventListener('click', function() {{ switchToTab(this.getAttribute('data-tab')); }});
    }});

    // Apron tab: S-Point / S-Bar / E-Bar / E-Point toggle
    ['chkShowSPoints', 'chkShowEBar', 'chkShowEPoints', 'chkShowSBars'].forEach(function(chkId) {{
      const el = document.getElementById(chkId);
      if (el) el.addEventListener('change', function() {{
        if (typeof renderFlightGantt === 'function') renderFlightGantt();
      }});
    }});

    document.getElementById('gridCellSize').addEventListener('change', function() {{ CELL_SIZE = Math.max(10, Number(this.value) || 10); draw(); }});
    document.getElementById('gridCols').addEventListener('change', function() {{ GRID_COLS = Math.max(5, Math.min(500, parseInt(this.value,10)||400)); draw(); }});
    document.getElementById('gridRows').addEventListener('change', function() {{ GRID_ROWS = Math.max(5, Math.min(500, parseInt(this.value,10)||400)); draw(); }});

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

    const remoteNameInput = document.getElementById('remoteName');
    if (remoteNameInput) {{
      remoteNameInput.addEventListener('change', function() {{
        if (state.selectedObject && state.selectedObject.type === 'remote') {{
          state.selectedObject.obj.name = this.value.trim();
          updateObjectInfo();
          renderObjectList();
          draw();
          if (typeof update3DScene === 'function') update3DScene();
        }}
      }});
    }}
    const remoteCategorySelect = document.getElementById('remoteCategory');
    if (remoteCategorySelect) {{
      remoteCategorySelect.addEventListener('change', function() {{
        const val = this.value || 'C';
        if (state.selectedObject && state.selectedObject.type === 'remote') {{
          state.selectedObject.obj.category = val;
          updateObjectInfo();
          renderObjectList();
          draw();
          if (typeof update3DScene === 'function') update3DScene();
        }}
      }});
    }}

    const remoteTerminalAccessEl = document.getElementById('remoteTerminalAccess');
    if (remoteTerminalAccessEl) {{
      remoteTerminalAccessEl.addEventListener('change', function(ev) {{
        const target = ev.target;
        if (!target || !target.classList.contains('remote-term-check')) return;
        if (!state.selectedObject || state.selectedObject.type !== 'remote') return;
        const st = state.selectedObject.obj;
        const checks = remoteTerminalAccessEl.querySelectorAll('.remote-term-check');
        const allowed = [];
        checks.forEach(function(ch) {{
          if (ch.checked) {{
            const id = ch.getAttribute('data-term-id');
            if (id) allowed.push(id);
          }}
        }});
        st.allowedTerminals = allowed;
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        updateObjectInfo();
        renderObjectList();
        draw();
      }});
    }}

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
    const avgVelInputEl = document.getElementById('taxiwayAvgMoveVelocity');
    if (avgVelInputEl) avgVelInputEl.addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        const val = Number(this.value);
        const v =
          (typeof val === 'number' && isFinite(val) && val > 0)
            ? Math.max(1, Math.min(50, val))
            : 10;
        tw.avgMoveVelocity = v;
        this.value = v;
        updateObjectInfo();
        renderObjectList();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }}
    }});
    document.getElementById('taxiwayMaxExitVel').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        const val = Number(this.value);
        if (tw.pathType === 'runway_exit') {{
          tw.maxExitVelocity = isFinite(val) && val > 0 ? val : null;
          // minExitVelocityIs maxExitVelocityAdjusted not to exceed
          if (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.maxExitVelocity != null && tw.minExitVelocity > tw.maxExitVelocity) {{
            tw.minExitVelocity = tw.maxExitVelocity;
          }}
        }} else {{
          delete tw.maxExitVelocity;
        }}
        if (isFinite(val) && val > 0) this.value = val; else this.value = tw.maxExitVelocity != null ? tw.maxExitVelocity : '';
        updateObjectInfo();
        renderObjectList();
        draw();
        if (scene3d) update3DScene();
      }}
    }});
    const minExitEl = document.getElementById('taxiwayMinExitVel');
    if (minExitEl) {{
      minExitEl.addEventListener('change', function() {{
        if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
          const tw = state.selectedObject.obj;
          const val = Number(this.value);
          if (tw.pathType === 'runway_exit') {{
            let v = isFinite(val) && val > 0 ? val : 15;
            if (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && v > tw.maxExitVelocity) v = tw.maxExitVelocity;
            tw.minExitVelocity = v;
            this.value = v;
          }} else {{
            delete tw.minExitVelocity;
          }}
          updateObjectInfo();
          renderObjectList();
          draw();
          if (scene3d) update3DScene();
        }}
      }});
    }}
    document.getElementById('taxiwayDirectionMode').addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
        const tw = state.selectedObject.obj;
        tw.direction = this.value || 'both';
        updateObjectInfo();
        draw();
        if (typeof update3DScene === 'function') update3DScene();
      }}
    }});
    const runwayMinArrVelEl = document.getElementById('runwayMinArrVelocity');
    if (runwayMinArrVelEl) {{
      runwayMinArrVelEl.addEventListener('change', function() {{
        if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
          const tw = state.selectedObject.obj;
          if (tw.pathType !== 'runway') return;
          const val = Number(this.value);
          const v = (typeof val === 'number' && isFinite(val) && val > 0) ? Math.max(1, Math.min(150, val)) : 15;
          tw.minArrVelocity = v;
          this.value = v;
          updateObjectInfo();
          renderObjectList();
          if (typeof renderFlightList === 'function') renderFlightList();
          draw();
        }}
      }});
    }}
    const runwayLineupEl = document.getElementById('runwayLineupDistM');
    if (runwayLineupEl) {{
      runwayLineupEl.addEventListener('change', function() {{
        if (state.selectedObject && state.selectedObject.type === 'taxiway') {{
          const tw = state.selectedObject.obj;
          if (tw.pathType !== 'runway') return;
          const val = Number(this.value);
          const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : 0;
          tw.lineupDistM = v;
          this.value = String(v);
          updateObjectInfo();
          if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
        }}
      }});
    }}

    // ---- Flight helpers ----
    function getMinArrVelocityMpsForRunwayId(runwayId) {{
      if (runwayId == null || runwayId === '') return 15;
      const list = state.taxiways || [];
      let tw = list.find(t => t.id === runwayId && t.pathType === 'runway');
      if (!tw) tw = list.find(t => t.id === runwayId && (t.name || '').toLowerCase().includes('runway'));
      if (!tw) return 15;
      const v = tw.minArrVelocity;
      if (typeof v === 'number' && isFinite(v) && v > 0) return Math.max(1, Math.min(150, v));
      return 15;
    }}
    /** v0deceleration from a(m/s²)as distM(m) When moving RET entrance speed·time taken. The speed is vFloor(m/s) Do not go below. */
    function runwayArrSpeedAndTimeToRet(v0, a, distM, vFloorIn) {{
      const vf0 = Math.max(1, Math.min(150, vFloorIn));
      const vf = Math.min(vf0, v0);
      if (!(a > 0) || distM <= 0) return {{ vAtRet: v0, tSec: 0 }};
      if (v0 <= vf) return {{ vAtRet: v0, tSec: distM / Math.max(v0, 1e-6) }};
      const dStop = (v0 * v0 - vf * vf) / (2 * a);
      if (distM < dStop) {{
        const vEnd = Math.sqrt(Math.max(0, v0 * v0 - 2 * a * distM));
        return {{ vAtRet: vEnd, tSec: (v0 - vEnd) / a }};
      }}
      const tDecel = (v0 - vf) / a;
      const tCruise = (distM - dStop) / vf;
      return {{ vAtRet: vf, tSec: tDecel + tCruise }};
    }}
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
      if (simContainer) simContainer.style.display = (state.hasSimulationResult && state.flights.length > 0) ? 'flex' : 'none';
    }}

    function formatMinutesToHHMM(minsRaw) {{
      const totalMin = Math.max(0, Math.floor(minsRaw || 0));
      const h = Math.floor(totalMin / 60);
      const m = totalMin % 60;
      const hh = (h < 10 ? '0' : '') + h;
      const mm = (m < 10 ? '0' : '') + m;
      return hh + ':' + mm;
    }}

    function formatTotalSecondsToHHMMSS(totalSec) {{
      const h = Math.floor(totalSec / 3600);
      const m = Math.floor((totalSec % 3600) / 60);
      const s = totalSec % 60;
      const hh = (h < 10 ? '0' : '') + h;
      const mm = (m < 10 ? '0' : '') + m;
      const ss = (s < 10 ? '0' : '') + s;
      return hh + ':' + mm + ':' + ss;
    }}
    function formatMinutesToHHMMSS(minsRaw) {{
      const totalSec = Math.max(0, Math.round((minsRaw || 0) * 60));
      return formatTotalSecondsToHHMMSS(totalSec);
    }}
    function formatSecondsToHHMMSS(secRaw) {{
      const totalSec = Math.max(0, Math.floor(secRaw || 0));
      return formatTotalSecondsToHHMMSS(totalSec);
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
        const dwellMin = (f.sobtMin_d != null && f.sibtMin_d != null) ? (f.sobtMin_d - f.sibtMin_d) : (f.dwellMin || 0);
        const dwellSec = Math.max(0, dwellMin * 60);
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
        // 1) First check if it is inside the polygon (Closed terminal only)
        if (t.closed && termPix.length >= 3 && pointInPolygonXY([px, py], termPix)) return t;
        // 2) Or remember the nearest terminal
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
      // If it does not belong to any polygon, return the nearest terminal
      return nearest;
    }}

    function flightCanUseStand(f, stand) {{
      if (!stand) return true;
      const order = {{ A:1,B:2,C:3,D:4,E:5,F:6 }};
      const fCode = (f.code || 'C').toUpperCase();
      const sCat = (stand.category || 'F').toUpperCase();
      const fc = order[fCode] || 99;
      const sc = order[sCat] || 0;
      if (fc > sc) return false;
      const ft = (f.terminalId || (f.token && f.token.terminalId)) || null;
      if (!ft) return true;
      const isRemote = (state.remoteStands || []).some(function(r) {{ return r.id === stand.id; }});
      if (isRemote) {{
        const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
        if (allowed.length) return allowed.includes(ft);
      }}
      const term = getTerminalForStand(stand);
      const standTermId = term ? term.id : null;
      if (!standTermId) return false;
      return ft === standTermId;
    }}

    function assignStandToFlight(f, standId) {{
      if (!f) return false;
      if (standId) {{
        const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
        const stand = allStands.find(function(s) {{ return s.id === standId; }});
        if (!flightCanUseStand(f, stand)) {{
          alert("Code or selected terminal: aircraft doesn't fit apron (stand); cannot be assigned.");
          return false;
        }}
      }}
      f.standId = standId;
      if (f.token) f.token.apronId = standId;
      delete f.sobtMin_orig;
      delete f.sldtMin_orig;
      delete f.sibtMin_orig;
      delete f.stotMin_orig;
      delete f.eldtMin_orig;
      delete f.eibtMin_orig;
      delete f.eobtMin_orig;
      delete f.etotMin_orig;
      if (typeof renderFlightGantt === 'function') renderFlightGantt();
      if (typeof renderFlightList === 'function') renderFlightList();
      if (typeof draw === 'function') draw();
      return true;
    }}

    function getCandidatePbbStandsForCode(code) {{
      const list = [];
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
      allStands.forEach(stand => {{
        // If a code is specified, only use it when the categories are the same
        if (code && stand.category && stand.category !== code) return;
        const hasLink = state.apronLinks.some(lk => lk.pbbId === stand.id);
        if (!hasLink) return;
        // Terminal constraints will not be checked here, and later flight.token.terminalIdand allowedTerminalsFilter by
        list.push(stand);
      }});
      return list;
    }}

    function pickRandom(arr) {{
      if (!arr.length) return null;
      const idx = Math.floor(Math.random() * arr.length);
      return arr[idx];
    }}

    function resolveStand(flight) {{
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
      if (flight.standId) {{
        return allStands.find(s => s.id === flight.standId) || null;
      }}
      let candidates = getCandidatePbbStandsForCode(flight.code);
      if (!candidates.length) return null;
      const termId = (flight.token && flight.token.terminalId) || null;
      if (termId) {{
        const filtered = candidates.filter(st => {{
          const allowed = Array.isArray(st.allowedTerminals) ? st.allowedTerminals : null;
          if (allowed && allowed.length) return allowed.includes(termId);
          const t = getTerminalForStand(st);
          return t && t.id === termId;
        }});
        if (filtered.length) candidates = filtered;
      }}
      const stand = pickRandom(candidates);
      if (stand) flight.standId = stand.id;
      return stand;
    }}

    function buildArrivalTimelineFromPts(flight, pts) {{
      if (!pts || pts.length < 2) return null;
      const sibtMin_d = flight.sibtMin_d != null ? flight.sibtMin_d : (flight.timeMin != null ? flight.timeMin : 0);
      const baseT = sibtMin_d * 60;
      const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
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
      const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (sibtMin_d + (flight.dwellMin != null ? flight.dwellMin : 0));
      const dwellSec = Math.max(0, (sobtMin_d - sibtMin_d) * 60);
      if (dwellSec > 0) {{
        tAcc = sobtMin_d * 60;
        const last = timeline[timeline.length - 1];
        timeline.push({{ t: tAcc, x: last.x, y: last.y }});
      }}
      return timeline;
    }}

    function buildDepartureTimelineFromPts(flight, pts) {{
      if (!pts || pts.length < 2) return null;
      const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
      const baseT = sobtMin_d * 60;
      const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
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

    function getRunwayOptions() {{
      const list = [];
      (state.taxiways || []).filter(t => t.pathType === 'runway' || (t.name || '').toLowerCase().includes('runway'))
        .forEach(t => list.push({{ id: t.id, name: (t.name || '').trim() || 'Runway' }}));
      return list;
    }}
    function buildRunwayOptionsHtml(selectedId) {{
      const opts = [];
      const list = getRunwayOptions();
      if (!list.length) {{
        opts.push('<option value=\"\">Runway</option>');
      }} else {{
        // Random/real, without empty value option Runway List only objects
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
      // In case of duplicate names makeUniqueNamedCopyuse 'Pier Apron3 (2)' displayed in the form
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

    // VTT(Arr) + RET/ROT(arrRotSec) share state.vttArrCacheRev; bump on Update / forced RET resample only.
    function bumpVttArrCacheRev() {{
      state.vttArrCacheRev = (state.vttArrCacheRev | 0) + 1;
    }}
    // Flight Schedule and S(d) same in calculation VTT(Arr) Helpers for using definitions
    // ※ Path/RET change: bump revision (Update or renderFlightList force RET) so VTT is recomputed once per flight.
    function getBaseVttArrMinutes(f) {{
      if (!f) return 0;
      const rev = state.vttArrCacheRev | 0;
      if (f.__schedVttArrRev === rev && f.__schedVttArrMin != null && isFinite(f.__schedVttArrMin)) {{
        return f.__schedVttArrMin;
      }}
      if (typeof sampleArrRetRotForFlightIfNeeded === 'function') {{
        const retStatsAll = getScheduleRetStatsAll();
        const rotCfgMap = {{}};
        sampleArrRetRotForFlightIfNeeded(f, retStatsAll, rotCfgMap, false);
      }}
      const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
      const arrPts = (typeof getPathForFlight === 'function') ? getPathForFlight(f) : null;
      let vttArrMin = 0;
      if (arrPts && arrPts.length >= 2) {{
        let startIdx = 0;
        if (f.sampledArrRet) {{
          const tw = (state.taxiways || []).find(t => t.id === f.sampledArrRet);
          if (tw && Array.isArray(tw.vertices) && tw.vertices.length) {{
            const last = tw.vertices[tw.vertices.length - 1];
            const retOutPt = cellToPixel(last.col, last.row);
            let bestD2 = Infinity;
            let bestIdx = 0;
            for (let i = 0; i < arrPts.length; i++) {{
              const dx = arrPts[i][0] - retOutPt[0];
              const dy = arrPts[i][1] - retOutPt[1];
              const d2 = dx*dx + dy*dy;
              if (d2 < bestD2) {{ bestD2 = d2; bestIdx = i; }}
            }}
            startIdx = Math.min(bestIdx, arrPts.length - 2);
          }}
        }}
        let dist = 0;
        for (let i = startIdx; i < arrPts.length - 1; i++) dist += pathDist(arrPts[i], arrPts[i+1]);
        vttArrMin = dist / v / 60;
      }}
      f.__schedVttArrMin = vttArrMin;
      f.__schedVttArrRev = rev;
      return vttArrMin;
    }}
    function getArrRotMinutes(f) {{
      const rotSec = f && f.arrRotSec;
      return (rotSec != null && isFinite(rotSec) && rotSec >= 0) ? rotSec / 60 : 0;
    }}
    function getBaseVttDepMinutes(f) {{
      const depPts = (typeof getPathForFlightDeparture === 'function') ? getPathForFlightDeparture(f) : null;
      if (!depPts || depPts.length < 2) return 0;
      const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
      let dist = 0;
      for (let i = 0; i < depPts.length - 1; i++) dist += pathDist(depPts[i], depPts[i+1]);
      return dist / v / 60;
    }}

    // By runway SLDT(d)The earliest arrival flight is ELDT = SLDT(d).
    // renderFlightList: SLDT=SIBT−VTT(Arr)−ROT(min); EIBT=ELDT+ROT+VTT+vttADelay (symmetric).
    function pinEarliestEldtToSldtPerRunway(flights) {{
      if (!Array.isArray(flights)) return;
      const byRwy = {{}};
      flights.forEach(f => {{
        if (!f || f.noWayArr) return;
        const rwy = f.arrRunwayId || (f.token && (f.token.arrRunwayId != null ? f.token.arrRunwayId : f.token.runwayId));
        if (rwy == null || rwy === '') return;
        const sldt = f.sldtMin_d;
        if (sldt == null || !isFinite(sldt)) return;
        if (!byRwy[rwy]) byRwy[rwy] = [];
        byRwy[rwy].push(f);
      }});
      Object.keys(byRwy).forEach(function(rwyId) {{
        const list = byRwy[rwyId];
        let minS = Infinity;
        let chosen = null;
        list.forEach(function(f) {{
          const s = f.sldtMin_d;
          if (s != null && isFinite(s) && s < minS) {{ minS = s; chosen = f; }}
        }});
        if (chosen) chosen.eldtMin = chosen.sldtMin_d;
      }});
    }}

    // One computeRunwayExitDistances() per user action when wrapped in begin/end (e.g. Global Update).
    var __schedRetStatsBatchActive = false;
    var __schedRetStatsCached = null;
    function beginScheduleRetStatsBatch() {{
      __schedRetStatsBatchActive = true;
      __schedRetStatsCached = null;
    }}
    function endScheduleRetStatsBatch() {{
      __schedRetStatsBatchActive = false;
      __schedRetStatsCached = null;
    }}
    function getScheduleRetStatsAll() {{
      if (__schedRetStatsBatchActive) {{
        if (__schedRetStatsCached === null) {{
          __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
        }}
        return __schedRetStatsCached;
      }}
      return typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    }}

    function warmFlightPathsForSchedule(flights) {{
      if (!Array.isArray(flights)) return;
      flights.forEach(f => {{
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        if (f.noWayArr || f.noWayDep) f.timeline = null;
      }});
    }}

    // Warm paths + RET/ROT sampling (single place; uses batched runway-exit stats when active).
    function warmPathsEnsureArrRetRot(flights, forceResampleRet) {{
      warmFlightPathsForSchedule(flights);
      return (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flights, !!forceResampleRet)
        : getScheduleRetStatsAll();
    }}

    function mutRotCfgEntryForType(configByType, f) {{
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
      if (!typeKey) return null;
      if (configByType[typeKey]) return configByType[typeKey];
      const tdMu = (typeof ac?.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
      const vMu = (typeof ac?.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
      const aMu = (typeof ac?.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
      const tdSigma = Math.round(tdMu * 0.1);
      const vSigma = Math.round(vMu * 0.1);
      const aSigma = Math.round(aMu * 0.1 * 10) / 10;
      configByType[typeKey] = {{ tdMu, tdSigma, vMu, vSigma, aMu, aSigma }};
      return configByType[typeKey];
    }}
    function isValidSampledArrRetForFlight(f, retStatsAll) {{
      if (!f || f.sampledArrRet == null) return false;
      if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
      const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
      return retStatsAll.some(function(r) {{
        if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
        if (arrRunwayId == null) return true;
        return !!(r.runway && r.runway.id === arrRunwayId);
      }});
    }}
    // RET + ROT(arrRotSec) for one flight; skipped if f.__schedRetRotRev matches vttArrCacheRev unless forceResample.
    function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {{
      if (!f || f.noWayArr) return;
      const rev = state.vttArrCacheRev | 0;
      if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
      if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
          f.sampledArrRet != null && f.arrRetFailed === false && f.arrRotSec != null && isFinite(f.arrRotSec) &&
          isValidSampledArrRetForFlight(f, retStatsAll)) {{
        f.__schedRetRotRev = rev;
        return;
      }}
      if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {{
        f.sampledArrRet = null;
        f.arrRetFailed = false;
        f.arrRotSec = null;
      }}
      const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
      const cfg = mutRotCfgEntryForType(configByType, f);
      if (!cfg || !retStatsAll || !retStatsAll.length || arrRunwayId == null) {{
        f.__schedRetRotRev = rev;
        return;
      }}
      const minArrVelRwy = getMinArrVelocityMpsForRunwayId(arrRunwayId);
      const tdSample = sampleNormal(cfg.tdMu, cfg.tdSigma);
      const tdMin = cfg.tdMu * 0.85;
      const tdMax = cfg.tdMu * 1.15;
      const dTd = clamp(tdSample, Math.max(0, tdMin), Math.max(0, tdMax));
      const vSample = sampleNormal(cfg.vMu, cfg.vSigma);
      const vMin = cfg.vMu * 0.85;
      const vMax = cfg.vMu * 1.15;
      const v0 = clamp(vSample, Math.max(0, vMin), Math.max(0, vMax));
      const aSample = sampleNormal(cfg.aMu, cfg.aSigma);
      const aMin = Math.max(0.1, cfg.aMu * 0.85);
      const aMax = Math.min(6,   cfg.aMu * 1.15);
      const aDec = clamp(aSample, aMin, aMax);
      const candidates = retStatsAll.filter(r => r.runway && r.runway.id === arrRunwayId);
      if (!candidates.length) {{
        f.__schedRetRotRev = rev;
        return;
      }}
      let chosen = null;
      candidates.forEach(r => {{
        if (chosen) return;
        const distFromTd = Math.max(0, r.distM - dTd);
        const vAt = runwayArrSpeedAndTimeToRet(v0, aDec, distFromTd, minArrVelRwy).vAtRet;
        if (vAt <= r.maxExitVelocity) {{ chosen = r; }}
      }});
      if (chosen) {{
        f.sampledArrRet = chosen.exit && chosen.exit.id || null;
        f.arrRetFailed = false;
        const MAX_DECEL_MS2 = 15;
        const distFromTdChosen = Math.max(0, chosen.distM - dTd);
        const aDecRot = Math.min(aDec, MAX_DECEL_MS2);
        const rtRunway = runwayArrSpeedAndTimeToRet(v0, aDecRot, distFromTdChosen, minArrVelRwy);
        const vAtChosen = rtRunway.vAtRet;
        const tToRetEntrance = rtRunway.tSec;
        const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
          ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
          : 15;
        let tExit = 0;
        if (vAtChosen > minExitVel) {{
          tExit = (vAtChosen - minExitVel) / aDecRot;
        }}
        f.arrRotSec = tToRetEntrance + tExit;
        f.arrRunwayIdUsed = arrRunwayId;
        f.arrTdDistM = dTd;
        f.arrRetDistM = chosen.distM;
        f.arrVTdMs = v0;
        f.arrVRetInMs = vAtChosen;
        f.arrVRetOutMs = minExitVel;
      }} else {{
        f.sampledArrRet = null;
        f.arrRetFailed = true;
        f.arrRotSec = null;
      }}
      f.__schedRetRotRev = rev;
    }}
    // RET selection + ROT(arrRotSec). Caller must have warmed paths. Uses getScheduleRetStatsAll(); same cache rev as VTT(Arr).
    function ensureArrRetRotSampled(flights, forceResampleRet) {{
      if (!Array.isArray(flights) || !flights.length) return [];
      const configByType = {{}};
      flights.forEach(f => {{ mutRotCfgEntryForType(configByType, f); }});
      const retStatsAll = getScheduleRetStatsAll();
      flights.forEach(function(f) {{
        sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, !!forceResampleRet);
      }});
      return retStatsAll;
    }}

    function renderFlightList(skipAutoAllocate, forceResampleRet) {{
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
      // Alignment uses display copies only. state.flights Maintain the order Allocation bar chart/Ensure that the parking lot layout does not change when route is updated.
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
      flightsSorted.forEach(f => {{
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        if (f.noWayArr || f.noWayDep) f.timeline = null;
      }});
      // Bump rev first so RET/ROT + VTT share the same generation (matches Global Update flow).
      if (forceResampleRet && typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
      // Paths (above) → RET/ROT → S(d)+E (compute* below). Order matches schedule dependency.
      const retStatsAll = (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flightsSorted, !!forceResampleRet)
        : (typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : []);
      const headerRow = '' +
        '<table class="flight-schedule-table">' +
        '<thead><tr>' +
          '<th>Reg</th>' +
          '<th>Airline Code</th>' +
          '<th>Flight Number</th>' +
          '<th class="flight-col-s flight-col-s-start">SLDT</th>' +
          '<th class="flight-td-sibt flight-col-s">SIBT</th>' +
          '<th class="flight-col-s">SOBT</th>' +
          '<th class="flight-col-s flight-col-s-last">STOT</th>' +
          '<th class="flight-col-sd flight-col-sd-start">SLDT(d)</th>' +
          '<th class="flight-col-sd">SIBT(d)</th>' +
          '<th class="flight-col-sd">SOBT(d)</th>' +
          '<th class="flight-col-sd flight-col-sd-last">STOT(d)</th>' +
          '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
          '<th class="flight-col-e">EIBT</th>' +
          '<th class="flight-col-e">EOBT</th>' +
          '<th class="flight-col-e">ETOT</th>' +
          '<th class="flight-col-e flight-col-rot">ROT</th>' +
          '<th>ARR_TAXI_TIME</th>' +
          '<th>ARR_TAXI_DELAY</th>' +
          '<th>DEP_TAXI_TIME</th>' +
          '<th>Aircraft Type</th>' +
          '<th>Code(ICAO)</th>' +
          '<th>ICAO(J/H/M/L)</th>' +
          '<th>RECAT-EU(A-F)</th>' +
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
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
      pinEarliestEldtToSldtPerRunway(flightsSorted);
      const dataRows = flightsSorted.map(function(f) {{
        const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
        const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
        let sampledRetName = '—';
        if (f.arrRetFailed) sampledRetName = 'Failed';
        else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {{
          const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
          sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
        }}
        const tArrMin = f.timeMin != null ? f.timeMin : 0;
        const dwell = f.dwellMin != null ? f.dwellMin : 0;
        const tDepMin = tArrMin + dwell;
        const vttArrMin = getBaseVttArrMinutes(f);
        const rotArrMin = getArrRotMinutes(f);
        const vttDepMin = getBaseVttDepMinutes(f);
        const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, tArrMin - vttArrMin - rotArrMin));
        const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
        const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
        const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (tDepMin + vttDepMin);
        const sldtStr = formatMinutesToHHMMSS(f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc);
        const stotStr = formatMinutesToHHMMSS(stotOrig);
        const sldtStr_d = formatMinutesToHHMMSS(f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
        const sibtStr_d = formatMinutesToHHMMSS(f.sibtMin_d != null ? f.sibtMin_d : tArrMin);
        const sobtStr_d = formatMinutesToHHMMSS(f.sobtMin_d != null ? f.sobtMin_d : tDepMin);
        const stotStr_d = formatMinutesToHHMMSS(f.stotMin_d != null ? f.stotMin_d : stotOrig);
        const eldtMin = f.eldtMin != null ? f.eldtMin : (f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
        const etotMin = f.etotMin != null ? f.etotMin : (f.stotMin_d != null ? f.stotMin_d : stotOrig);
        f.eldtMin = eldtMin;
        f.etotMin = etotMin;

        const tArr = formatMinutesToHHMMSS(tArrMin);
        const tDep = formatMinutesToHHMMSS(tDepMin);
        const vttADelayMin = f.vttADelayMin != null ? f.vttADelayMin : 0;
        const eibtMin = eldtMin + rotArrMin + vttArrMin + vttADelayMin;
        const eobtMin = etotMin - vttDepMin;
        // Flight Schedule Original standard S/E Save the series time, and then GanttIn *_orig Reference as standard
        if (f.sobtMin_orig == null) {{
          f.sldtMin_orig = sldtOrig;
          f.sibtMin_orig = tArrMin;
          f.sobtMin_orig = sobtOrig;
          f.stotMin_orig = stotOrig;
          f.eldtMin_orig = eldtMin;
          f.eibtMin_orig = eibtMin;
          f.eobtMin_orig = eobtMin;
          f.etotMin_orig = etotMin;
        }}
        // Flight to object EIBT/EOBT save (Apron Gantt Direct reference from etc.)
        f.eibtMin = eibtMin;
        f.eobtMin = eobtMin;

        const eldtStr = formatMinutesToHHMMSS(eldtMin);
        const etotStr = formatMinutesToHHMMSS(etotMin);
        const eibtStr = formatMinutesToHHMMSS(eibtMin);
        const eobtStr = formatMinutesToHHMMSS(eobtMin);
        const dwellS = dwell;
        const dwellE = Math.max(0, eobtMin - eibtMin);
        const vttArrStr = formatMinutesToHHMMSS(vttArrMin);
        const vttADelayStr = formatMinutesToHHMMSS(vttADelayMin);
        const vttDepStr = formatMinutesToHHMMSS(vttDepMin);
        // Arr Runway / options / No Way badge (arrRunwayId, acabove RET Already defined in the sampling step)
        const arrOpt = buildRunwayOptionsHtml(arrRunwayId);
        const termOpt = buildTerminalOptionsHtml(f.terminalId || (f.token && f.token.terminalId));
        const depOpt = buildRunwayOptionsHtml(f.depRunwayId || (f.token && f.token.depRunwayId));
        const noWayBadge = (f.noWayArr || f.noWayDep) ? ' <span style="color:#dc2626;font-weight:600;font-size:10px;">⚠ No Way</span>' : '';
        const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
        const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
        const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
        const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';

        const arrRetFailedBadge = (f.arrRetFailed || sampledRetName === 'Failed') ? ' <span style="color:#dc2626;font-weight:600;font-size:10px;">⚠ Failed</span>' : '';
        return '' +
          '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
            '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + noWayBadge + arrRetFailedBadge + '</td>' +
            '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
            '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
            '<td class="flight-td-time flight-col-s flight-col-s-start">' + sldtStr + '</td>' +
            '<td class="flight-td-time flight-td-sibt flight-col-s">' + tArr + '</td>' +
            '<td class="flight-td-time flight-col-s">' + tDep + '</td>' +
            '<td class="flight-td-time flight-col-s flight-col-s-last">' + stotStr + '</td>' +
            '<td class="flight-td-time flight-col-sd flight-col-sd-start">' + sldtStr_d + '</td>' +
            '<td class="flight-td-time flight-col-sd">' + sibtStr_d + '</td>' +
            '<td class="flight-td-time flight-col-sd">' + sobtStr_d + '</td>' +
            '<td class="flight-td-time flight-col-sd flight-col-sd-last">' + stotStr_d + '</td>' +
            '<td class="flight-td-time flight-col-e flight-col-e-start">' + eldtStr + '</td>' +
            '<td class="flight-td-time flight-col-e">' + eibtStr + '</td>' +
            '<td class="flight-td-time flight-col-e">' + eobtStr + '</td>' +
            '<td class="flight-td-time flight-col-e">' + etotStr + '</td>' +
            '<td class="flight-td-time flight-col-e flight-col-rot">' + (f.arrRotSec != null && isFinite(f.arrRotSec) ? (Math.round(f.arrRotSec) + ' s') : '—') + '</td>' +
            '<td class="flight-td-time">' + vttArrStr + '</td>' +
            '<td class="flight-td-time">' + vttADelayStr + '</td>' +
            '<td class="flight-td-time">' + vttDepStr + '</td>' +
            '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
            '<td>' + escapeHtml(codeIcao) + '</td>' +
            '<td>' + escapeHtml(icaoJhl) + '</td>' +
            '<td>' + escapeHtml(recatEu) + '</td>' +
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
      // Flight Configuration tab: Aircraft type Setting table of units (input UIHowever, it is not connected to other logic)
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
          // If there is an existing table, the values ​​modified by the user are first read and maintained.
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
          // per-aircraft default: Information.jsonof static Use the value, but if there is a value modified by the user in the existing table, use it first
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
          // σIs μset to 10% of (User editable), If the value has been modified from an existing table, use it first.
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
          // Deceleration a and VTDof mean from value, 26 m/s position to reach (thresholddistance from, m)
          const vTarget = 26;
          const aMeanStopDists = aMeans.map((aMu, idx) => {{
            const vMu = vtdMeans[idx];
            const tdMu = tdMeans[idx];
            if (!(aMu > 0) || !(vMu > vTarget)) return Math.max(0, Math.round(tdMu || 0));
            const dFromTouchdown = (vMu*vMu - vTarget*vTarget) / (2 * aMu);
            const dTotal = (tdMu || 0) + (dFromTouchdown > 0 ? dFromTouchdown : 0);
            return dTotal > 0 ? Math.round(dTotal) : 0;
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
          // Deceleration a: mean(μ) When only values ​​are used 26 m/s Distance to reach (threshold Baseline, read-only)
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
          // Runway Taxiway (RET) location table: thresholddistance from + Flight scheduleselected from RETstar model count
          const retStats = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
          if (retStats && retStats.length) {{
            // Section header row
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
              // each RETAbout, Flight scheduleselected from RETCalculate star model count
              const counts = unique.map(info => {{
                const typeKey = info.key;
                return (state.flights || []).filter(f =>
                  f.sampledArrRet === (r.exit && r.exit.id) &&
                  (f.aircraftType || '') === typeKey
                ).length;
              }});
              // most models(1stomach), 2~3color coded above
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
                      bg = 'rgba(22,163,74,0.35)'; // 1Above: green
                      color = '#bbf7d0';
                    }} else if (colIdx === top2 || colIdx === top3) {{
                      bg = 'rgba(251,146,60,0.3)'; // 2,3Above: Orange
                      color = '#fed7aa';
                    }}
                    return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
                  }}).join('') +
                '</tr>'
              );
            }});
            // RETFlights that could not be escaped by(Condition not met)Separately Failed Summarize by row
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
                      bg = 'rgba(220,38,38,0.65)'; // 1Stomach failure: dark red
                      color = '#fee2e2';
                    }} else if (colIdx === fTop2 || colIdx === fTop3) {{
                      bg = 'rgba(239,68,68,0.45)'; // 2,3Stomach failure: light red
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
              'Note: sampling is clipped to stay within ±15% of each mean value.' +
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
      // Click on the corresponding Flight select
      listEl.querySelectorAll('.obj-item').forEach(row => {{
        row.addEventListener('click', function(ev) {{
          if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
          const idVal = this.getAttribute('data-id');
          const f = state.flights.find(x => x.id === idVal);
          if (!f) return;
          state.selectedObject = {{ type: 'flight', id: idVal, obj: f }};
          // check mark
          listEl.querySelectorAll('.obj-item').forEach(r => r.classList.remove('selected', 'expanded'));
          this.classList.add('selected', 'expanded');
          if (typeof updateObjectInfo === 'function') updateObjectInfo();
          if (typeof syncPanelFromState === 'function') syncPanelFromState();
          if (typeof draw === 'function') draw();
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }});
      }});
      // Arr Rw / Terminal / Dep Rw selection handler
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
            // does not match the selected terminal StandIf it is already assigned to UnassignedGo to
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
          // Arr Rw / Terminal / Dep Rw After change, RET Sampling and Timeline·GanttRecalculate everything
          if (typeof renderFlightList === 'function') renderFlightList();
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
        }});
      }});
      if (typeof renderFlightGantt === 'function') renderFlightGantt();
    }}

    // GANTT_COLORS: from the top INFORMATION.tiers.style.gantt defined as

    // Allocation Gantt (length: Apron/Stand, horizontal: time)
    function renderFlightGantt() {{
      const ganttEl = document.getElementById('allocationGantt');
      if (!ganttEl) return;
      // current landscape/Remember the vertical scroll position DOM Restore after rebuild
      let prevScrollLeft = 0, prevScrollTop = 0;
      const prevScrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
      if (prevScrollCol) {{
        prevScrollLeft = prevScrollCol.scrollLeft || 0;
        prevScrollTop = prevScrollCol.scrollTop || 0;
      }}
      // existing DOMat Terminal / Remote Remember section folding state and restore it after re-render
      const prevCollapsedTerminals = new Set();
      let prevRemoteCollapsed = false;
      const prevLabelCol = ganttEl.querySelector('.alloc-gantt-label-col');
      if (prevLabelCol) {{
        const prevLabels = Array.from(prevLabelCol.children);
        prevLabels.forEach(function (el) {{
          if (el.classList && el.classList.contains('alloc-terminal-header')) {{
            if (el.getAttribute('data-collapsed') === '1') {{
              let txt = (el.textContent || '').trim();
              // Toggle icon in front(▶/▼) eliminate
              txt = txt.replace(/^[▶▼]\s*/, '');
              if (txt) prevCollapsedTerminals.add(txt);
            }}
          }}
          if (el.classList && el.classList.contains('alloc-remote-header')) {{
            if (el.getAttribute('data-collapsed') === '1') {{
              prevRemoteCollapsed = true;
            }}
          }}
        }});
      }}
      if (!state.flights.length) {{
        ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
        return;
      }}
      const flights = state.flights.slice();
      const stands = (state.pbbStands || []).concat(state.remoteStands || []);
      if (!flights.length) {{
        ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
        return;
      }}
      flights.forEach(f => {{
        if (typeof getPathForFlight === 'function') getPathForFlight(f);
        if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(f);
        if (f.noWayArr || f.noWayDep) f.timeline = null;
      }});
      if (typeof ensureArrRetRotSampled === 'function') ensureArrRetRotSampled(flights, false);
      // S(d)/E The series is Flight Schedule The values ​​shown in the table **directly** Read and use.
      // (Only if the table does not exist or is not rendered state.flights using value fallback)
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();

      let intervals = [];
      const schedTable = document.querySelector('.flight-schedule-table');
      if (schedTable) {{
        const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
        rows.forEach(row => {{
          const id = row.getAttribute('data-id');
          if (!id) return;
          const f = flights.find(ff => ff.id === id);
          if (!f) return;
          const tds = Array.from(row.querySelectorAll('td'));
          if (tds.length < 15) return;
          // Flight Schedule column index:
          // 0 Reg, 1 Airline, 2 FlightNo,
          // 3 SLDT, 4 SIBT, 5 SOBT, 6 STOT,
          // 7 SLDT(d), 8 SIBT(d), 9 SOBT(d), 10 STOT(d),
          // 11 ELDT, 12 EIBT, 13 EOBT, 14 ETOT, ...
          const getMin = (idx) => {{
            const txt = (tds[idx] && tds[idx].textContent || '').trim();
            if (!txt) return 0;
            try {{
              return parseTimeToMinutes(txt);
            }} catch (e) {{
              return 0;
            }}
          }};
          const sldt_d = getMin(7);
          const sibt_d = getMin(8);
          const sobt_d = getMin(9);
          const stot_d = getMin(10);
          const eldt   = getMin(11);
          const eibt   = getMin(12);
          const eobt   = getMin(13);
          const etot   = getMin(14);
          const t0 = sibt_d;
          const t1 = sobt_d || (t0 + (f.dwellMin != null ? f.dwellMin : 0));
          const sldt = sldt_d || t0;
          const stot = stot_d || t1;
          const sldtOrig = sldt;
          const sobtOrig = sobt_d || t1;
          const stotOrig = stot;
          intervals.push({{ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig }});
        }});
      }}
      if (!intervals.length) {{
        // If the table cannot be found or parsing fails, the existing state with based logic fallback
        intervals = flights.map(f => {{
          const t0 = f.sibtMin_d != null ? f.sibtMin_d : (f.timeMin != null ? f.timeMin : 0);
          const t1 = f.sobtMin_d != null ? f.sobtMin_d : (t0 + (f.dwellMin != null ? f.dwellMin : 0));
          const sldt = f.sldtMin_d != null ? f.sldtMin_d : t0;
          const stot = f.stotMin_d != null ? f.stotMin_d : t1;
          const eibt = f.eibtMin != null ? f.eibtMin : t0;
          const eobt = f.eobtMin != null ? f.eobtMin : t1;
          const eldt = f.eldtMin != null ? f.eldtMin : sldt;
          const etot = f.etotMin != null ? f.etotMin : stot;
          const sldtOrig = sldt;
          const sobtOrig = f.sobtMin_d != null ? f.sobtMin_d : t1;
          const stotOrig = stot;
          return {{ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig }};
        }});
      }}

      // common time base: Flight Scheduleof min(SLDT) - pad, max(ETOT) + pad (algorithm.timeAxis)
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
      // default full range (zoomThe minimum reduction limit of)
      const baseMinT = Math.max(0, minS - GANTT_PAD_MIN);
      const baseMaxT0 = maxE + GANTT_PAD_MIN;
      // The maximum range is baseMinT 24 hours standard(1440minute)Defensively limited to
      const baseMaxT = Math.min(
        (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0,
        baseMinT + 1440
      );
      const baseSpan = baseMaxT - baseMinT;
      const zoom = (state.allocTimeZoom && state.allocTimeZoom > 1) ? state.allocTimeZoom : 1;
      const span = baseSpan;
      const minT = baseMinT;
      const maxT = baseMaxT;

      // Allocation/bar chart/The apron layout is updated only when explicitly changed by the user. No automatic reassignment when updating routes such as taxiways

      // Time axis scale position (Apron common to all) - Only display up to 6 items on any screen
      const tickPositions = [];
      const axisStep = span <= 60 ? TICK_STEP_SPAN_LE60 : (span <= 240 ? TICK_STEP_SPAN_LE240 : TICK_STEP_ELSE); // minutes
      let tt = Math.floor(minT / axisStep) * axisStep;
      while (tt <= maxT) {{
        const leftPct = ((tt - baseMinT) / baseSpan) * 100 * zoom;
        const label = formatMinToHM(tt);
        tickPositions.push({{ leftPct, label }});
        tt += axisStep;
      }}
      if (tickPositions.length > MAX_TICKS_SHOWN) {{
        const stepTicks = Math.ceil(tickPositions.length / MAX_TICKS_SHOWN);
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
        const showSPointsEl = document.getElementById('chkShowSPoints');
        const showSPoints = !showSPointsEl || showSPointsEl.checked;
        const showSBarsEl = document.getElementById('chkShowSBars');
        // S‑Bar check: default SIBT‑SOBT Bars are opaque, uncheck: Apply transparency to the default bar.
        const dimSBars = !!(showSBarsEl && !showSBarsEl.checked);
        const showEBarEl = document.getElementById('chkShowEBar');
        const showEBar = !showEBarEl || showEBarEl.checked;
        const showEPointsEl = document.getElementById('chkShowEPoints');
        const showEPoints = !showEPointsEl || showEPointsEl.checked;
        // S‑Point: S Series auxiliary bar + dot + Full vertical control
        // E‑Bar : EIBT/EOBT thick auxiliary bar
        // E‑Point: ELDT/ETOT dot + triangle + ELDT/EIBT·EOBT/ETOT auxiliary bar
        const showAuxBars = showSPoints;
        const showEibtBars = showEBar;
        const showEldtBars = showEPoints;
        const showSDots = showSPoints;
        const showSdDots = showSPoints;
        const showEDots = showEPoints;
        const rowFlights = intervals.filter(it => {{
          const f = it.f;
          const sid = (f.standId || null);
          return (standId == null) ? !sid : sid === standId;
        }});
        // identification Apron/Stand There are overlapping sections within Flightcast conflictdisplayed as
        const conflictMap = {{}};
        for (let i = 0; i < rowFlights.length; i++) {{
          for (let j = i + 1; j < rowFlights.length; j++) {{
            const a = rowFlights[i];
            const b = rowFlights[j];
            if (a.t0 < b.t1 && b.t0 < a.t1) {{ // Section overlap
              conflictMap[a.f.id] = true;
              conflictMap[b.f.id] = true;
            }}
          }}
        }}
        const sBars = showAuxBars ? [] : null;
        const eBars = showEibtBars ? [] : null;
        const e2Bars = showEldtBars ? [] : null;
        const sDots = showSDots ? [] : null;
        const sdDots = showSdDots ? [] : null;
        const eDots = showEDots ? [] : null;
        function pushDot(arr, t, cls) {{
          if (!arr || !isFinite(t) || t < baseMinT || t > baseMaxT) return;
          const leftPct = ((t - baseMinT) / baseSpan) * 100 * zoom;
          arr.push('<div class="alloc-time-dot ' + cls + '" style="left:' + leftPct + '%;"></div>');
        }}
        const sLines = showSPoints ? [] : null;      // SOBT(orig) vertical line
        const sTrisDown = showSPoints ? [] : null;   // SLDTtriangle under dragon
        const sTrisUp = showSPoints ? [] : null;     // STOTtriangle above dragon
        const eTrisDown = showEPoints ? [] : null;   // ELDTtriangle under dragon
        const eTrisUp = showEPoints ? [] : null;     // ETOTtriangle above dragon
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
          const sobtOrig = (it.sobtOrig != null) ? it.sobtOrig : (it.stotOrig - (f.vttDepMin != null ? f.vttDepMin : 0));
          const tStart = Math.max(t0, baseMinT);
          const tEnd = Math.min(t1, baseMaxT);
          if (tEnd <= tStart) return '';
          const leftPct = ((tStart - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct = Math.max(2, ((tEnd - tStart) / baseSpan) * 100 * zoom);
          const regSafe = escapeHtml(f.reg || '');
          const codeSafe = escapeHtml((f.code || '').toUpperCase());
          const dwellVal = (t1 != null && t0 != null) ? Math.max(0, t1 - t0) : (f.dwellMin != null ? f.dwellMin : 0);
          const dwellLabel = dwellVal ? (Math.round(dwellVal * 10) / 10 + 'm') : '';
          let meta = '';
          if (codeSafe && dwellLabel) meta = codeSafe + ' · ' + dwellLabel;
          else if (codeSafe) meta = codeSafe;
          else meta = dwellLabel;
          const conflictClass = (conflictMap[f.id] || f.noWayArr || f.noWayDep) ? ' conflict' : '';
          const selectedClass = (state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id) ? ' alloc-flight-selected' : '';
          const sbarDimClass = dimSBars ? ' alloc-flight-sbar-dim' : '';
          const noWayLabel = (f.noWayArr || f.noWayDep)
            ? ' <span style="color:#fca5a5;font-size:9px;font-weight:700;">No way</span>'
            : '';
          const sibtLabel = formatMinToHM(t0);
          const sobtLabel = formatMinToHM(t1);
          const barTitle =
            'SIBT: ' + sibtLabel +
            '\\nSOBT: ' + sobtLabel +
            '\\nReg: ' + (f.reg || '') +
            '\\nAirline: ' + (f.airlineCode || '') + ' ' + (f.flightNumber || '');
          if (showEibtBars && eBars && isFinite(eibt) && isFinite(eobt) && eobt > eibt) {{
            const eStart = Math.max(eibt, baseMinT);
            const eEnd = Math.min(eobt, baseMaxT);
            if (eEnd > eStart) {{
              const eLeft = ((eStart - baseMinT) / baseSpan) * 100 * zoom;
              const eWidth = ((eEnd - eStart) / baseSpan) * 100 * zoom;
              eBars.push(
                '<div class="alloc-e-bar" style="left:' + eLeft + '%;width:' + Math.max(2, eWidth) + '%;"></div>'
              );
            }}
          }}
          const hasOverlap = (f.vttADelayMin != null && f.vttADelayMin > 0) || f.eOverlapPushed;
          const ovlpBadgeHtml = hasOverlap ? '<span class="alloc-flight-ovlp-badge">OVLP</span>' : '';
          if (showEldtBars && e2Bars) {{
            // ELDT~EIBT (pre-block, Center aligned thin hot pink bar)
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
                  '%;"></div>'
              );
              }}
            }}
            // EOBT~ETOT (post-block, Center aligned thin hot pink bar)
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
                  '%;"></div>'
              );
              }}
            }}
          }}
          if (showAuxBars && sBars) {{
            // SLDT~SIBT (pre-block) auxiliary bar
            if (isFinite(sldt) && sldt <= t0) {{
              const s1 = Math.max(sldt, baseMinT);
              const s2 = Math.min(t0, baseMaxT);
              if (s2 > s1) {{
                const preLeft = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
                const preWidth = ((s2 - s1) / baseSpan) * 100 * zoom;
              sBars.push(
                '<div class="alloc-s-bar" style="left:' +
                  preLeft +
                  '%;width:' +
                  Math.max(0.5, preWidth) +
                  '%;"></div>'
              );
              }}
            }}
            // SOBT~STOT (post-block) Auxiliary bar: Attached to the top of the main bar
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
                  '%;"></div>'
              );
              }}
            }}
          }}
          if (showSDots && sDots) {{
            // S-Point: auxiliary bar(sBars)same as S(d) series time(SLDT(d)/STOT(d))Show only circles
            pushDot(sDots, sldt, 'alloc-time-dot-s');
            pushDot(sDots, stot, 'alloc-time-dot-s');
          }}
          if (showSdDots && sdDots) {{
            // S(d) The series is also represented by the same blue dot.
            pushDot(sdDots, sldt, 'alloc-time-dot-sd');
            pushDot(sdDots, stot, 'alloc-time-dot-sd');
          }}
          if (showEDots && eDots) {{
            // E-Point: ELDT/ETOT dot + triangle (pink)
            pushDot(eDots, eldt, 'alloc-time-dot-e');
            pushDot(eDots, etot, 'alloc-time-dot-e');
            if (eTrisDown && isFinite(eldt) && eldt >= baseMinT && eldt <= baseMaxT) {{
              const leftPct = ((eldt - baseMinT) / baseSpan) * 100 * zoom;
              eTrisDown.push(
                '<div class="alloc-e-tri alloc-e-tri-down" style="left:' +
                  leftPct +
                  '%;"></div>'
              );
            }}
            if (eTrisUp && isFinite(etot) && etot >= baseMinT && etot <= baseMaxT) {{
              const leftPct2 = ((etot - baseMinT) / baseSpan) * 100 * zoom;
              eTrisUp.push(
                '<div class="alloc-e-tri alloc-e-tri-up" style="left:' +
                  leftPct2 +
                  '%;"></div>'
              );
            }}
          }}
        // S-Point: SLDT/STOTunder Edo/Add top triangle (E-PointSame design and color as GANTT_COLORS.S_BAR)
          if (showSPoints) {{
            if (sTrisDown && isFinite(sldt) && sldt >= baseMinT && sldt <= baseMaxT) {{
              const leftPctS1 = ((sldt - baseMinT) / baseSpan) * 100 * zoom;
              sTrisDown.push(
                '<div class="alloc-s-tri alloc-s-tri-down" style="left:' +
                  leftPctS1 +
                  '%;"></div>'
              );
            }}
            if (sTrisUp && isFinite(stot) && stot >= baseMinT && stot <= baseMaxT) {{
              const leftPctS2 = ((stot - baseMinT) / baseSpan) * 100 * zoom;
              sTrisUp.push(
                '<div class="alloc-s-tri alloc-s-tri-up" style="left:' +
                  leftPctS2 +
                  '%;"></div>'
              );
            }}
          }}
        // The black vertical dotted line is SOBT(orig)It is placed in,
        // "OVERLAP"Although it is an aircraft SOBT(orig) ≠ SOBT(d) Show only if
        if (sLines && ((f.vttADelayMin != null && f.vttADelayMin > 0) || f.eOverlapPushed) && isFinite(sobtOrig)) {{
          const sobtD = (f.sobtMin_d != null ? f.sobtMin_d : t1);
          if (!isNaN(sobtD) && Math.abs(sobtOrig - sobtD) > 1e-6) {{
            const sx = ((sobtOrig - baseMinT) / baseSpan) * 100 * zoom;
            sLines.push('<div class="alloc-s-line-orig" style="left:' + sx + '%;"></div>');
          }}
        }}
          return '' +
            '<div class="alloc-flight' + conflictClass + selectedClass + sbarDimClass + '" draggable="true" data-flight-id="' + f.id + '" ' +
              'style="left:' + leftPct + '%;width:' + widthPct + '%;min-width:4px;"' +
              ' title="' + barTitle + '">' +
              '<div class="alloc-flight-reg">' + regSafe + noWayLabel + '</div>' +
              '<div class="alloc-flight-meta">' + meta + '</div>' +
              ovlpBadgeHtml +
            '</div>';
        }}).join('');
        const sidAttr = standId ? String(standId) : '';
        const gridLines = tickPositions.map(tp =>
          '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>'
        ).join('');
        // time axis and time axis "between"Place background text in the center of
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
        const labelHtml =
          '<div class="alloc-row-label" data-stand-id="' + sidAttr + '">' +
            escapeHtml(label) +
          '</div>';
        const trackHtml =
          '<div class="alloc-row" data-stand-id="' + sidAttr + '">' +
            '<div class="alloc-row-track" data-stand-id="' + sidAttr + '">' +
              gridLines +
              bgSlots +
              blocks +
              (showEibtBars && eBars ? eBars.join('') : '') +
              (showEldtBars && e2Bars ? e2Bars.join('') : '') +
              (showAuxBars && sBars ? sBars.join('') : '') +
              (showSDots && sDots ? sDots.join('') : '') +
              (showSdDots && sdDots ? sdDots.join('') : '') +
              (showEDots && eDots ? eDots.join('') : '') +
              (sTrisDown ? sTrisDown.join('') : '') +
              (sTrisUp ? sTrisUp.join('') : '') +
              (eTrisDown ? eTrisDown.join('') : '') +
              (eTrisUp ? eTrisUp.join('') : '') +
              (sLines ? sLines.join('') : '') +
            '</div>' +
          '</div>';
        return {{ labelHtml, trackHtml }};
      }}
      // Unassigned Above: All flights SLDT/STOT·ELDT/ETOT Only dots (S/E Same class as point·Coordinate formula, existing row logic unchanged)
      function buildRunwayLegendPair() {{
        const gridLines = tickPositions.map(function(tp) {{
          return '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>';
        }}).join('');
        function pushLegendDot(arr, t, cls) {{
          if (!isFinite(t) || t < baseMinT || t > baseMaxT) return;
          const leftPct = ((t - baseMinT) / baseSpan) * 100 * zoom;
          arr.push('<div class="alloc-time-dot ' + cls + '" style="left:' + leftPct + '%;"></div>');
        }}
        const sDotsHtml = [];
        const eDotsHtml = [];
        intervals.forEach(function(it) {{
          pushLegendDot(sDotsHtml, it.sldt, 'alloc-time-dot-s');
          pushLegendDot(sDotsHtml, it.stot, 'alloc-time-dot-s');
          pushLegendDot(eDotsHtml, it.eldt, 'alloc-time-dot-e');
          pushLegendDot(eDotsHtml, it.etot, 'alloc-time-dot-e');
        }});
        const sLabelHtml = '<div class="alloc-row-label alloc-runway-legend-label" data-stand-id="" data-runway-legend="1">' + escapeHtml('S(LDT, TOT)') + '</div>';
        const sTrackHtml =
          '<div class="alloc-row" data-stand-id="" data-runway-legend="1">' +
            '<div class="alloc-row-track" data-stand-id="" data-runway-legend="1" style="background:transparent;border:none;">' +
              gridLines + sDotsHtml.join('') +
            '</div>' +
          '</div>';
        const eLabelHtml = '<div class="alloc-row-label alloc-runway-legend-label" data-stand-id="" data-runway-legend="1">' + escapeHtml('E(LDT, TOT)') + '</div>';
        const eTrackHtml =
          '<div class="alloc-row" data-stand-id="" data-runway-legend="1">' +
            '<div class="alloc-row-track" data-stand-id="" data-runway-legend="1" style="background:transparent;border:none;">' +
              gridLines + eDotsHtml.join('') +
            '</div>' +
          '</div>';
        return {{ sLabelHtml: sLabelHtml, sTrackHtml: sTrackHtml, eLabelHtml: eLabelHtml, eTrackHtml: eTrackHtml }};
      }}
      const labelRows = [];
      const trackRows = [];
      (function() {{
        const rw = buildRunwayLegendPair();
        labelRows.push(rw.sLabelHtml);
        trackRows.push(rw.sTrackHtml);
        labelRows.push(rw.eLabelHtml);
        trackRows.push(rw.eTrackHtml);
      }})();
      // Unassigned line
      (function() {{
        const row = buildRowHtml('Unassigned', null);
        labelRows.push(row.labelHtml);
        trackRows.push(row.trackHtml);
      }})();
      // By terminal Stand grouping
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
      const remoteIdSet = new Set((state.remoteStands || []).map(r => r.id));
      const allRemoteStands = [];
      order.forEach(key => {{
        const group = grouped[key];
        if (!group) return;
        const term = group.term;
        const headerLabel = term
          ? (termLabelById[term.id] || term.name || 'Terminal')
          : 'No Terminal';
        // Terminal header: Add one row each to the left label column and right timeline column.
        labelRows.push(
          '<div class="alloc-terminal-header" data-collapsed="0">' +
            '<span class="alloc-section-toggle-icon">▼</span>' +
            escapeHtml(headerLabel) +
          '</div>'
        );
        // For right header dummy Track terminal label height(24px)Set the row height to be the same..
        trackRows.push('<div class="alloc-row" data-stand-id="">' +
          '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:24px;"></div>' +
        '</div>');
        // Each apron row: Contact / RemoteDisplay separately (The terminal name is in the header)
        const contactStands = [];
        const remoteStandsInTerm = [];
        group.stands.forEach(s => {{
          if (remoteIdSet.has(s.id)) remoteStandsInTerm.push(s);
          else contactStands.push(s);
        }});
        // Contact stands first
        contactStands.forEach(s => {{
          const label = (s.name || '') + ' (' + (s.category || '') + ')';
          const row = buildRowHtml(label, s.id);
          labelRows.push(row.labelHtml);
          trackRows.push(row.trackHtml);
        }});
        // Remote standsgathers them into a global array, Terminal Show only once after
        if (remoteStandsInTerm.length) {{
          remoteStandsInTerm.forEach(s => allRemoteStands.push(s));
        }}
      }});
      // every Terminal behind, at the bottom Remote stand Add a dedicated section
      if (allRemoteStands.length) {{
        // left·Right same: Remote stomach 8px interval(nine margin-top)Separated by spacer rows to maintain row index 1:1
        labelRows.push('<div class="alloc-gantt-section-spacer" aria-hidden="true"></div>');
        trackRows.push(
          '<div class="alloc-row" data-stand-id="">' +
            '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:8px;min-height:8px;"></div>' +
          '</div>'
        );
        labelRows.push(
          '<div class="alloc-remote-header" data-collapsed="0">' +
            '<span class="alloc-section-toggle-icon">▼</span>' +
            'Remote stands' +
          '</div>'
        );
        trackRows.push(
          '<div class="alloc-row" data-stand-id="">' +
            '<div class="alloc-row-track" data-stand-id="" style="background:transparent;border:none;height:20px;min-height:20px;"></div>' +
          '</div>'
        );
        allRemoteStands.forEach(s => {{
          const label = (s.name || '') + ' (' + (s.category || '') + ')';
          const row = buildRowHtml(label, s.id);
          labelRows.push(row.labelHtml);
          trackRows.push(row.trackHtml);
        }});
      }}
      // Time axis overlay at bottom (Display only time labels at the same location as the vertical grid lines)
      const axisTicks = tickPositions.map(tp =>
        '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
          '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
        '</div>'
      );
      const axisHtml =
        '<div class="alloc-time-axis-overlay">' +
          '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
        '</div>';

      // The left label row also has the same height to match the bottom time axis. spaceradd
      labelRows.push('<div class="alloc-label-axis-spacer"></div>');

      const labelColHtml =
        '<div class="alloc-gantt-label-col">' +
          labelRows.join('') +
        '</div>';
      // zoom As much as the magnification inner Increase the width to the scrolled section .alloc-row-track hit areaAllow to expand (drop zone = full timeline)
      const innerMinWidthPct = Math.max(100, Math.round(zoom * 100));
      const trackColHtml =
        '<div class="alloc-gantt-scroll-col">' +
          '<div class="alloc-gantt-inner" style="min-width:' + innerMinWidthPct + '%;">' +
            trackRows.join('') +
            axisHtml +
          '</div>' +
        '</div>';
      const rootHtml =
        '<div class="alloc-gantt-root">' +
          labelColHtml +
          trackColHtml +
        '</div>';

      ganttEl.innerHTML = rootHtml;
      const newScrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
      const newLabelCol = ganttEl.querySelector('.alloc-gantt-label-col');
      // DOM Horizontal after reconstruction/Restore vertical scroll position
      if (newScrollCol) {{
        if (prevScrollLeft > 0) newScrollCol.scrollLeft = prevScrollLeft;
        if (prevScrollTop > 0) newScrollCol.scrollTop = prevScrollTop;
      }}
      // Left label column ↔ Synchronize vertical scrolling of right timeline column (Make the horizontal scrollbar always visible at the bottom of the viewport)
      if (newScrollCol && newLabelCol) {{
        newScrollCol.addEventListener('scroll', function() {{ newLabelCol.scrollTop = newScrollCol.scrollTop; }});
        newLabelCol.addEventListener('scroll', function() {{ newScrollCol.scrollTop = newLabelCol.scrollTop; }});
      }}
      // Terminal / Remote Collapse section/Expand
      if (newScrollCol && newLabelCol) {{
        const labelChildren = Array.from(newLabelCol.children);
        const innerEl = newScrollCol.querySelector('.alloc-gantt-inner');
        const trackChildren = innerEl ? Array.from(innerEl.children).filter(function(el) {{
          return el.classList.contains('alloc-row');
        }}) : [];
        labelChildren.forEach(function(el, idx) {{
          // Terminal header toggle
          if (el.classList.contains('alloc-terminal-header')) {{
            el.style.cursor = 'pointer';
            // which was collapsed in the previous render TerminalIf so, restore the initial state to the collapsed state.
            (function applyInitialCollapsed() {{
              let txt = (el.textContent || '').trim();
              txt = txt.replace(/^[▶▼]\s*/, '');
              if (txt && prevCollapsedTerminals.has(txt)) {{
                el.setAttribute('data-collapsed', '1');
                const icon0 = el.querySelector('.alloc-section-toggle-icon');
                if (icon0) icon0.textContent = '▶';
                for (let j = idx + 1; j < labelChildren.length; j++) {{
                  const lbl = labelChildren[j];
                  if (lbl.classList.contains('alloc-terminal-header') ||
                      lbl.classList.contains('alloc-remote-header') ||
                      lbl.classList.contains('alloc-label-axis-spacer') ||
                      lbl.classList.contains('alloc-gantt-section-spacer')) break;
                  lbl.style.display = 'none';
                  if (trackChildren[j]) trackChildren[j].style.display = 'none';
                }}
              }}
            }})();
            el.addEventListener('click', function() {{
              const prevCollapsed = el.getAttribute('data-collapsed') === '1';
              const nextCollapsed = !prevCollapsed;
              el.setAttribute('data-collapsed', nextCollapsed ? '1' : '0');
              const icon = el.querySelector('.alloc-section-toggle-icon');
              if (icon) icon.textContent = nextCollapsed ? '▶' : '▼';
              for (let j = idx + 1; j < labelChildren.length; j++) {{
                const lbl = labelChildren[j];
                if (lbl.classList.contains('alloc-terminal-header') ||
                    lbl.classList.contains('alloc-remote-header') ||
                    lbl.classList.contains('alloc-label-axis-spacer') ||
                    lbl.classList.contains('alloc-gantt-section-spacer')) break;
                lbl.style.display = nextCollapsed ? 'none' : '';
                if (trackChildren[j]) trackChildren[j].style.display = nextCollapsed ? 'none' : '';
              }}
            }});
          }}
          // Remote Toggle section header
          if (el.classList.contains('alloc-remote-header')) {{
            el.style.cursor = 'pointer';
            // From previous render Remote Restore initial state if section was collapsed
            if (prevRemoteCollapsed) {{
              el.setAttribute('data-collapsed', '1');
              const icon0 = el.querySelector('.alloc-section-toggle-icon');
              if (icon0) icon0.textContent = '▶';
              for (let j = idx + 1; j < labelChildren.length; j++) {{
                const lbl = labelChildren[j];
                if (lbl.classList.contains('alloc-terminal-header') ||
                    lbl.classList.contains('alloc-remote-header') ||
                    lbl.classList.contains('alloc-label-axis-spacer') ||
                    lbl.classList.contains('alloc-gantt-section-spacer')) break;
                lbl.style.display = 'none';
                if (trackChildren[j]) trackChildren[j].style.display = 'none';
              }}
            }}
            el.addEventListener('click', function() {{
              const prevCollapsed = el.getAttribute('data-collapsed') === '1';
              const nextCollapsed = !prevCollapsed;
              el.setAttribute('data-collapsed', nextCollapsed ? '1' : '0');
              const icon = el.querySelector('.alloc-section-toggle-icon');
              if (icon) icon.textContent = nextCollapsed ? '▶' : '▼';
              for (let j = idx + 1; j < labelChildren.length; j++) {{
                const lbl = labelChildren[j];
                if (lbl.classList.contains('alloc-terminal-header') ||
                    lbl.classList.contains('alloc-remote-header') ||
                    lbl.classList.contains('alloc-label-axis-spacer') ||
                    lbl.classList.contains('alloc-gantt-section-spacer')) break;
                lbl.style.display = nextCollapsed ? 'none' : '';
                if (trackChildren[j]) trackChildren[j].style.display = nextCollapsed ? 'none' : '';
              }}
            }});
          }}
        }});
      }}
      // Ctrl + Supports horizontal scrolling with wheel (Apron chart only)
      if (newScrollCol && !newScrollCol._allocWheelBound) {{
        newScrollCol._allocWheelBound = true;
        newScrollCol.addEventListener('wheel', function(ev) {{
          if (!ev.ctrlKey) return;
          ev.preventDefault();
          const delta = ev.deltaY || ev.deltaX || 0;
          newScrollCol.scrollLeft += delta;
        }}, {{ passive: false }});
      }}

      // The cursor tracks in the scrolled area./outside the bar(empty section)When in targetthis .alloc-gantt-scroll-colbecome this
      // preventDefaultPrevents the prohibition cursor from appearing because it is not called: only when off-track. scroll colat accept.
      // Dragging elementFromPointreturns a drag image and cannot find the track., clientYCalculate which row it is and store it.
      // The listener is ganttElPut it on renderFlightGantt() Make it work even after recall. _lastDropTracksilver ganttElSave to.
      (function() {{
        if (ganttEl._allocDropBound) return;
        ganttEl._allocDropBound = true;
        function getTrackAt(scrollCol, clientX, clientY) {{
          if (!scrollCol) return null;
          const inner = scrollCol.querySelector('.alloc-gantt-inner');
          if (!inner) return null;
          const rows = inner.querySelectorAll('.alloc-row');
          const tol = 2;
          for (let i = 0; i < rows.length; i++) {{
            const r = rows[i].getBoundingClientRect();
            if (clientY >= r.top - tol && clientY <= r.bottom + tol) {{
              const track = rows[i].querySelector('.alloc-row-track');
              if (track) return track;
            }}
          }}
          return null;
        }}
        ganttEl.addEventListener('dragover', function(ev) {{
          if (!ev.target || !ev.target.closest) return;
          if (!ev.target.closest('#allocationGantt')) return;
          const scrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
          if (!scrollCol) return;
          const rect = scrollCol.getBoundingClientRect();
          const x = Math.max(rect.left + 1, Math.min(rect.right - 1, ev.clientX));
          const y = ev.clientY;
          const el = document.elementFromPoint(ev.clientX, ev.clientY);
          let track = el && el.closest ? el.closest('.alloc-row-track') : null;
          if (!track && el && el.closest) {{
            const row = el.closest('.alloc-row');
            if (row) track = row.querySelector ? row.querySelector('.alloc-row-track') : null;
          }}
          if (!track) track = getTrackAt(scrollCol, x, y);
          ganttEl._lastDropTrack = track || null;
          if (!ev.target.closest('.alloc-row-track')) {{
            ev.preventDefault();
            ev.dataTransfer.dropEffect = 'move';
          }}
        }}, true);
        ganttEl.addEventListener('drop', function(ev) {{
          if (!ev.target || !ev.target.closest) return;
          if (!ev.target.closest('#allocationGantt')) return;
          ev.preventDefault();
          ev.stopPropagation();
          const scrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
          if (!scrollCol) return;
          let track = (ev.target && ev.target.closest('.alloc-row-track')) || null;
          if (!track) {{
            const el = document.elementFromPoint(ev.clientX, ev.clientY);
            track = el && el.closest ? el.closest('.alloc-row-track') : null;
          }}
          if (!track) track = ganttEl._lastDropTrack;
          if (!track) {{
            const rect = scrollCol.getBoundingClientRect();
            const x = Math.max(rect.left + 1, Math.min(rect.right - 1, ev.clientX));
            const y = ev.clientY;
            track = getTrackAt(scrollCol, x, y);
          }}
          if (!track) return;
          if (track.getAttribute('data-runway-legend') === '1') return;
          const flightId = ev.dataTransfer.getData('text/plain');
          if (!flightId) return;
          const f = state.flights.find(x => x.id === flightId);
          if (!f) return;
          const sidAttr = track.getAttribute('data-stand-id') || '';
          const standId = sidAttr || null;
          assignStandToFlight(f, standId);
        }}, true);
      }})();

      // Shift + Zoom on the time axis with the mouse wheel (Apron)
      if (!ganttEl._allocZoomBound) {{
        ganttEl._allocZoomBound = true;
        ganttEl.addEventListener('wheel', function(e) {{
          if (!e.shiftKey) return;
          e.preventDefault();
          // scroll up = Zoom in, scroll down = reduction
          const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
          let z = state.allocTimeZoom || 1;
          z *= factor;
          if (z < 1) z = 1;           // Minimum: full range
          if (z > 8) z = 8;           // maximum magnification
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
          if (this.getAttribute('data-runway-legend') === '1') return;
          ev.preventDefault();
          ev.dataTransfer.dropEffect = 'move';
        }});
        track.addEventListener('drop', function(ev) {{
          ev.preventDefault();
          if (this.getAttribute('data-runway-legend') === '1') return;
          const flightId = ev.dataTransfer.getData('text/plain');
          if (!flightId) return;
          const f = state.flights.find(x => x.id === flightId);
          if (!f) return;
          const sidAttr = this.getAttribute('data-stand-id') || '';
          const standId = sidAttr || null;
          assignStandToFlight(f, standId);
        }});
      }});
    }}

    function validateNetworkForFlights() {{
      const msgs = [];
      const hasRunwayPath = state.taxiways && state.taxiways.some(tw => tw.pathType === 'runway' || (tw.name || '').toLowerCase().includes('runway'));
      if (!hasRunwayPath) msgs.push('RunwayThere is no.');
      if (!state.taxiways || !state.taxiways.length) msgs.push('TaxiwayThere is no.');
      const stands = (state.pbbStands || []).concat(state.remoteStands || []);
      const linked = state.apronLinks || [];
      // at least one Stand(PBB/Remote)actually exists Taxiwayand Apron Taxiway Must be connected by link.
      const hasApronLink = stands.some(pbb =>
        linked.some(lk =>
          lk.pbbId === pbb.id &&
          state.taxiways &&
          state.taxiways.some(tw => tw.id === lk.taxiwayId)
        )
      );
      if (!stands.length || !hasApronLink) msgs.push('Apron(PBB)class TaxiwayAt least one link is required to connect.');
      // Remote standAvailable terminal constraints and Flight Scheduleof Terminal Check for conflicting settings
      const termsForLabel = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {{ return {{
        id: t.id,
        name: (t.name || '').trim() || 'Terminal'
      }}; }});
      function termNameById(id) {{
        const tt = termsForLabel.find(function(t) {{ return t.id === id; }});
        return tt ? tt.name : (id || 'Terminal');
      }}
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
      (state.flights || []).forEach(function(f) {{
        if (!f || !f.standId) return;
        const stand = allStands.find(function(s) {{ return s.id === f.standId; }});
        if (!stand) return;
        // Remote standOnly terminal access restrictions apply
        const isRemote = (state.remoteStands || []).some(function(r) {{ return r.id === stand.id; }});
        if (!isRemote) return;
        const termId = (f.token && f.token.terminalId) || null;
        // Flight Scheduleat TerminalOnly when this specific value (Random This side termId There is no)
        if (!termId) return;
        const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
        if (allowed.length && !allowed.includes(termId)) {{
          const flightLabel = f.id || f.flightNo || f.reg || '';
          const standLabel = stand.name || 'Remote';
          const termLabel = termNameById(termId);
          const allowedLabel = allowed.map(termNameById).join(', ');
          msgs.push('Flight ' + (flightLabel || '') + ' of Terminal setting(' + termLabel + ')this Remote stand ' + standLabel + ' Available terminal settings for(' + allowedLabel + ')does not match.');
        }}
      }});
      return msgs;
    }}

    function updateFlightError(msgs) {{
      const el = document.getElementById('flightError');
      if (!el) return;
      el.textContent = Array.isArray(msgs) ? msgs.join(' / ') : (msgs || '');
    }}

    // ---- Layout Design minimum path: Node/Edge Graph, reverse cost 1,000,000 ----
    const REVERSE_COST = 1000000;
    function pathDist2(a, b) {{ return dist2(a, b); }}
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

    // same grid(Same cell, half grid unit)Always with the same node: key based on cell coordinates
    function pathPointKey(p) {{
      const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
      const cellCol = Math.round(p[0] / cs * 2) / 2;
      const cellRow = Math.round(p[1] / cs * 2) / 2;
      return cellCol + ',' + cellRow;
    }}

    function kpiToNumber(value) {{
      const n = Number(value);
      return isFinite(n) ? n : null;
    }}

    function kpiRound(value, digits) {{
      const n = kpiToNumber(value);
      if (n == null) return null;
      const pow = Math.pow(10, digits || 0);
      return Math.round(n * pow) / pow;
    }}

    function kpiFormatCount(value) {{
      const n = kpiToNumber(value);
      return n == null ? '—' : String(Math.round(n));
    }}

    function kpiFormatMinutesCompact(value) {{
      const n = kpiToNumber(value);
      if (n == null) return '—';
      const totalSec = Math.max(0, Math.round(n * 60));
      const hours = Math.floor(totalSec / 3600);
      const mins = Math.floor((totalSec % 3600) / 60);
      const secs = totalSec % 60;
      if (hours > 0) return hours + 'h ' + mins + 'm';
      if (mins > 0) return mins + 'm' + (secs > 0 ? ' ' + secs + 's' : '');
      return secs + 's';
    }}

    function kpiFormatSecondsCompact(value) {{
      const n = kpiToNumber(value);
      if (n == null) return '—';
      const totalSec = Math.max(0, Math.round(n));
      const mins = Math.floor(totalSec / 60);
      const secs = totalSec % 60;
      if (mins > 0) return mins + 'm ' + secs + 's';
      return secs + 's';
    }}

    function kpiFormatMinutesValue(value) {{
      const n = kpiToNumber(value);
      if (n == null) return '—';
      return n.toFixed(1) + ' min';
    }}

    function kpiFormatSecondsValue(value) {{
      const n = kpiToNumber(value);
      if (n == null) return '—';
      return kpiRound(n, 0) + ' sec';
    }}

    function kpiFormatClockBucket(minute) {{
      const n = kpiToNumber(minute);
      if (n == null) return '—';
      const total = Math.floor(n);
      const hh = ((Math.floor(total / 60) % 24) + 24) % 24;
      return String(hh).padStart(2, '0') + ':00';
    }}

    function kpiFormatClock(minute) {{
      const n = kpiToNumber(minute);
      if (n == null) return '—';
      return formatMinutesToHHMMSS(n);
    }}

    function kpiFormatSnapshotTime() {{
      const now = new Date();
      const hh = String(now.getHours()).padStart(2, '0');
      const mm = String(now.getMinutes()).padStart(2, '0');
      const ss = String(now.getSeconds()).padStart(2, '0');
      return hh + ':' + mm + ':' + ss;
    }}

    function kpiSum(items, selector) {{
      return (items || []).reduce(function(acc, item) {{
        const value = selector(item);
        return acc + (kpiToNumber(value) || 0);
      }}, 0);
    }}

    function kpiAverage(items, selector) {{
      const vals = (items || []).map(selector).map(kpiToNumber).filter(v => v != null);
      if (!vals.length) return null;
      return kpiSum(vals, function(v) {{ return v; }}) / vals.length;
    }}

    function kpiStandLabelById(standId) {{
      const stands = (state.pbbStands || []).concat(state.remoteStands || []);
      const stand = stands.find(function(s) {{ return s && s.id === standId; }});
      return stand ? ((stand.name && stand.name.trim()) || stand.id || 'Stand') : 'Unassigned';
    }}

    function kpiBuildMetricRow(label, primary, secondary) {{
      return '' +
        '<div class="kpi-metric-row">' +
          '<div class="kpi-metric-label">' + escapeHtml(label) + '</div>' +
          '<div class="kpi-metric-values">' +
            '<div class="kpi-metric-primary">' + escapeHtml(primary) + '</div>' +
            '<div class="kpi-metric-secondary">' + escapeHtml(secondary) + '</div>' +
          '</div>' +
        '</div>';
    }}

    function kpiBuildSummaryCard(label, value, meta, tone, submeta) {{
      return '' +
        '<div class="kpi-card ' + escapeHtml(tone || '') + '">' +
          '<div class="kpi-card-label">' + escapeHtml(label) + '</div>' +
          '<div class="kpi-card-value">' + escapeHtml(value) + '</div>' +
          '<div class="kpi-card-meta">' + escapeHtml(meta) + '</div>' +
          '<div class="kpi-card-submeta">' + escapeHtml(submeta || '') + '</div>' +
        '</div>';
    }}

    function kpiBuildPanel(title, badge, rows) {{
      return '' +
        '<div class="kpi-panel">' +
          '<div class="kpi-panel-header">' +
            '<div class="kpi-panel-title">' + escapeHtml(title) + '</div>' +
            '<div class="kpi-panel-badge">' + escapeHtml(badge) + '</div>' +
          '</div>' +
          '<div class="kpi-metric-list">' + rows.join('') + '</div>' +
        '</div>';
    }}

    function kpiBuildOccupancyChart(buckets) {{
      if (!buckets || !buckets.length) return '<div class="kpi-empty-state">No gate occupancy data is available for the current snapshot.</div>';
      const width = Math.max(720, buckets.length * 54);
      const height = 240;
      const padL = 44, padR = 20, padT = 18, padB = 34;
      const innerW = width - padL - padR;
      const innerH = height - padT - padB;
      const maxY = Math.max(1, buckets.reduce(function(acc, bucket) {{ return Math.max(acc, bucket.occupancy || 0); }}, 0));
      const xFor = function(idx) {{
        return buckets.length === 1 ? (padL + innerW / 2) : (padL + (idx / (buckets.length - 1)) * innerW);
      }};
      const yFor = function(value) {{
        return padT + innerH - ((value || 0) / maxY) * innerH;
      }};
      let linePath = '';
      let areaPath = '';
      let dots = '';
      buckets.forEach(function(bucket, idx) {{
        const x = xFor(idx);
        const y = yFor(bucket.occupancy || 0);
        linePath += (idx === 0 ? 'M ' : ' L ') + x + ' ' + y;
        areaPath += (idx === 0 ? ('M ' + x + ' ' + (padT + innerH) + ' L ' + x + ' ' + y) : (' L ' + x + ' ' + y));
        dots += '<circle class="kpi-chart-dot" cx="' + x + '" cy="' + y + '" r="4"></circle>';
      }});
      areaPath += ' L ' + xFor(buckets.length - 1) + ' ' + (padT + innerH) + ' Z';
      const yLines = [];
      for (let i = 0; i < 4; i++) {{
        const ratio = i / 3;
        const y = padT + innerH - ratio * innerH;
        const label = kpiRound(maxY * ratio, 0);
        yLines.push('<line class="kpi-chart-grid-line" x1="' + padL + '" y1="' + y + '" x2="' + (padL + innerW) + '" y2="' + y + '"></line>');
        yLines.push('<text class="kpi-chart-axis" x="' + (padL - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + escapeHtml(String(label)) + '</text>');
      }}
      const step = Math.max(1, Math.ceil(buckets.length / 8));
      const xLabels = buckets.map(function(bucket, idx) {{
        if (idx % step !== 0 && idx !== buckets.length - 1) return '';
        return '<text class="kpi-chart-axis" x="' + xFor(idx) + '" y="' + (height - 8) + '" text-anchor="middle">' + escapeHtml(bucket.label) + '</text>';
      }}).join('');
      return '' +
        '<div class="kpi-chart-scroll">' +
          '<svg class="kpi-chart-frame" viewBox="0 0 ' + width + ' ' + height + '" width="' + width + '" height="' + height + '" aria-label="Hourly gate occupancy chart">' +
            '<defs>' +
              '<linearGradient id="kpiOccFill" x1="0" y1="0" x2="0" y2="1">' +
                '<stop offset="0%" stop-color="rgba(124,106,247,0.42)"></stop>' +
                '<stop offset="100%" stop-color="rgba(124,106,247,0.03)"></stop>' +
              '</linearGradient>' +
            '</defs>' +
            yLines.join('') +
            '<path d="' + areaPath + '" fill="url(#kpiOccFill)"></path>' +
            '<path d="' + linePath + '" fill="none" stroke="#a78bfa" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>' +
            dots +
            xLabels +
          '</svg>' +
        '</div>';
    }}

    function kpiBuildTrafficChart(buckets) {{
      if (!buckets || !buckets.length) return '<div class="kpi-empty-state">No arrival or departure events are available for the current snapshot.</div>';
      const width = Math.max(720, buckets.length * 56);
      const height = 240;
      const padL = 44, padR = 20, padT = 18, padB = 34;
      const innerW = width - padL - padR;
      const innerH = height - padT - padB;
      const maxY = Math.max(1, buckets.reduce(function(acc, bucket) {{
        return Math.max(acc, bucket.arrivals || 0, bucket.departures || 0, bucket.total || 0);
      }}, 0));
      const slotW = innerW / buckets.length;
      const barW = Math.max(8, Math.min(14, slotW * 0.22));
      const yFor = function(value) {{
        return padT + innerH - ((value || 0) / maxY) * innerH;
      }};
      const yLines = [];
      for (let i = 0; i < 4; i++) {{
        const ratio = i / 3;
        const y = padT + innerH - ratio * innerH;
        const label = kpiRound(maxY * ratio, 0);
        yLines.push('<line class="kpi-chart-grid-line" x1="' + padL + '" y1="' + y + '" x2="' + (padL + innerW) + '" y2="' + y + '"></line>');
        yLines.push('<text class="kpi-chart-axis" x="' + (padL - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + escapeHtml(String(label)) + '</text>');
      }}
      let bars = '';
      let linePath = '';
      let dots = '';
      buckets.forEach(function(bucket, idx) {{
        const cx = padL + slotW * idx + slotW / 2;
        const arrH = innerH - (yFor(bucket.arrivals || 0) - padT);
        const depH = innerH - (yFor(bucket.departures || 0) - padT);
        const arrY = yFor(bucket.arrivals || 0);
        const depY = yFor(bucket.departures || 0);
        const totalY = yFor(bucket.total || 0);
        bars += '<rect x="' + (cx - barW - 2) + '" y="' + arrY + '" width="' + barW + '" height="' + Math.max(2, arrH) + '" rx="4" fill="#38bdf8"></rect>';
        bars += '<rect x="' + (cx + 2) + '" y="' + depY + '" width="' + barW + '" height="' + Math.max(2, depH) + '" rx="4" fill="#fb923c"></rect>';
        linePath += (idx === 0 ? 'M ' : ' L ') + cx + ' ' + totalY;
        dots += '<circle class="kpi-chart-dot" cx="' + cx + '" cy="' + totalY + '" r="4" fill="#c4b5fd"></circle>';
      }});
      const step = Math.max(1, Math.ceil(buckets.length / 8));
      const xLabels = buckets.map(function(bucket, idx) {{
        if (idx % step !== 0 && idx !== buckets.length - 1) return '';
        const cx = padL + slotW * idx + slotW / 2;
        return '<text class="kpi-chart-axis" x="' + cx + '" y="' + (height - 8) + '" text-anchor="middle">' + escapeHtml(bucket.label) + '</text>';
      }}).join('');
      return '' +
        '<div class="kpi-chart-scroll">' +
          '<svg class="kpi-chart-frame" viewBox="0 0 ' + width + ' ' + height + '" width="' + width + '" height="' + height + '" aria-label="Hourly traffic chart">' +
            yLines.join('') +
            bars +
            '<path d="' + linePath + '" fill="none" stroke="#c4b5fd" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>' +
            dots +
            xLabels +
          '</svg>' +
        '</div>';
    }}

    function collectKpiSnapshot() {{
      const flights = Array.isArray(state.flights) ? state.flights.slice() : [];
      const rows = flights.map(function(f) {{
        const arrTaxiMin = kpiToNumber(typeof getBaseVttArrMinutes === 'function' ? getBaseVttArrMinutes(f) : null);
        const depTaxiMin = kpiToNumber(typeof getBaseVttDepMinutes === 'function' ? getBaseVttDepMinutes(f) : null);
        const rotSec = kpiToNumber(f && f.arrRotSec != null ? f.arrRotSec : (typeof getArrRotMinutes === 'function' ? getArrRotMinutes(f) * 60 : null));
        const sibt = kpiToNumber(f && f.sibtMin_orig != null ? f.sibtMin_orig : (f && f.timeMin != null ? f.timeMin : null));
        const sldt = kpiToNumber(f && f.sldtMin_orig != null ? f.sldtMin_orig : (sibt != null && arrTaxiMin != null && rotSec != null ? Math.max(0, sibt - arrTaxiMin - rotSec / 60) : null));
        const dwellMin = kpiToNumber(f && f.dwellMin != null ? f.dwellMin : null);
        const sobt = kpiToNumber(f && f.sobtMin_orig != null ? f.sobtMin_orig : (sibt != null && dwellMin != null ? sibt + dwellMin : null));
        const stot = kpiToNumber(f && f.stotMin_orig != null ? f.stotMin_orig : (sobt != null && depTaxiMin != null ? sobt + depTaxiMin : null));
        const eldt = kpiToNumber(f && f.eldtMin != null ? f.eldtMin : (f && f.sldtMin_d != null ? f.sldtMin_d : sldt));
        const eibt = kpiToNumber(f && f.eibtMin != null ? f.eibtMin : (eldt != null && arrTaxiMin != null && rotSec != null ? eldt + arrTaxiMin + rotSec / 60 + (kpiToNumber(f.vttADelayMin) || 0) : sibt));
        const eobt = kpiToNumber(f && f.eobtMin != null ? f.eobtMin : sobt);
        const etot = kpiToNumber(f && f.etotMin != null ? f.etotMin : (f && f.stotMin_d != null ? f.stotMin_d : stot));
        const failed = !!(f && (f.noWayArr || f.noWayDep || f.arrRetFailed));
        const paxArrDelay = (eibt != null && sibt != null) ? Math.max(0, eibt - sibt) : null;
        const paxDepDelay = (eobt != null && sobt != null) ? Math.max(0, eobt - sobt) : null;
        const acArrDelay = (eldt != null && sldt != null) ? Math.max(0, eldt - sldt) : null;
        const acDepDelay = (etot != null && stot != null) ? Math.max(0, etot - stot) : null;
        return {{
          flight: f,
          id: f && f.id ? f.id : '',
          reg: f && f.reg ? f.reg : '',
          flightNumber: f && f.flightNumber ? f.flightNumber : '',
          standId: f && f.standId ? f.standId : null,
          standName: kpiStandLabelById(f && f.standId ? f.standId : null),
          arrTaxiMin,
          depTaxiMin,
          rotSec,
          sibt,
          sobt,
          sldt,
          stot,
          eldt,
          eibt,
          eobt,
          etot,
          failed,
          paxArrDelay,
          paxDepDelay,
          acArrDelay,
          acDepDelay
        }};
      }});
      const timeValues = [];
      rows.forEach(function(row) {{
        [row.sldt, row.sibt, row.sobt, row.stot, row.eldt, row.eibt, row.eobt, row.etot].forEach(function(value) {{
          if (kpiToNumber(value) != null) timeValues.push(value);
        }});
      }});
      const minT = timeValues.length ? Math.min.apply(null, timeValues) : null;
      const maxT = timeValues.length ? Math.max.apply(null, timeValues) : null;
      const buckets = [];
      if (minT != null && maxT != null) {{
        const startHour = Math.floor(minT / 60) * 60;
        let endHour = Math.ceil((maxT + 1) / 60) * 60;
        if (endHour <= startHour) endHour = startHour + 60;
        for (let bucketStart = startHour; bucketStart < endHour; bucketStart += 60) {{
          const bucketEnd = bucketStart + 60;
          const activeStands = new Set();
          let arrivals = 0;
          let departures = 0;
          rows.forEach(function(row) {{
            const occStart = row.eibt != null ? row.eibt : row.sibt;
            const occEnd = row.eobt != null ? row.eobt : row.sobt;
            if (row.standId && occStart != null && occEnd != null && occEnd > bucketStart && occStart < bucketEnd) activeStands.add(row.standId);
            const arrMoment = row.eldt != null ? row.eldt : row.eibt;
            const depMoment = row.etot != null ? row.etot : row.eobt;
            if (arrMoment != null && arrMoment >= bucketStart && arrMoment < bucketEnd) arrivals += 1;
            if (depMoment != null && depMoment >= bucketStart && depMoment < bucketEnd) departures += 1;
          }});
          buckets.push({{
            label: kpiFormatClockBucket(bucketStart),
            occupancy: activeStands.size,
            arrivals: arrivals,
            departures: departures,
            total: arrivals + departures,
            bucketStart: bucketStart
          }});
        }}
      }}
      const failedFlights = rows.filter(function(row) {{ return row.failed; }});
      const operationalFlights = rows.filter(function(row) {{ return !row.failed; }});
      const peakBucket = buckets.reduce(function(best, bucket) {{
        if (!best) return bucket;
        return (bucket.occupancy || 0) > (best.occupancy || 0) ? bucket : best;
      }}, null);
      const busiestBucket = buckets.reduce(function(best, bucket) {{
        if (!best) return bucket;
        return (bucket.total || 0) > (best.total || 0) ? bucket : best;
      }}, null);
      const detailRows = rows.slice().sort(function(a, b) {{
        const delayA = (a.paxArrDelay || 0) + (a.paxDepDelay || 0) + (a.acArrDelay || 0) + (a.acDepDelay || 0);
        const delayB = (b.paxArrDelay || 0) + (b.paxDepDelay || 0) + (b.acArrDelay || 0) + (b.acDepDelay || 0);
        return delayB - delayA;
      }});
      return {{
        rows: rows,
        buckets: buckets,
        totalFlights: rows.length,
        failedFlights: failedFlights.length,
        operationalFlights: operationalFlights.length,
        peakBucket: peakBucket,
        busiestBucket: busiestBucket,
        rotTotalSec: kpiSum(rows, function(row) {{ return row.rotSec; }}),
        rotAvgSec: kpiAverage(rows, function(row) {{ return row.rotSec; }}),
        arrTaxiTotalMin: kpiSum(rows, function(row) {{ return row.arrTaxiMin; }}),
        arrTaxiAvgMin: kpiAverage(rows, function(row) {{ return row.arrTaxiMin; }}),
        depTaxiTotalMin: kpiSum(rows, function(row) {{ return row.depTaxiMin; }}),
        depTaxiAvgMin: kpiAverage(rows, function(row) {{ return row.depTaxiMin; }}),
        paxArrDelayTotalMin: kpiSum(rows, function(row) {{ return row.paxArrDelay; }}),
        paxArrDelayAvgMin: kpiAverage(rows, function(row) {{ return row.paxArrDelay; }}),
        paxDepDelayTotalMin: kpiSum(rows, function(row) {{ return row.paxDepDelay; }}),
        paxDepDelayAvgMin: kpiAverage(rows, function(row) {{ return row.paxDepDelay; }}),
        acArrDelayTotalMin: kpiSum(rows, function(row) {{ return row.acArrDelay; }}),
        acArrDelayAvgMin: kpiAverage(rows, function(row) {{ return row.acArrDelay; }}),
        acDepDelayTotalMin: kpiSum(rows, function(row) {{ return row.acDepDelay; }}),
        acDepDelayAvgMin: kpiAverage(rows, function(row) {{ return row.acDepDelay; }}),
        detailRows: detailRows
      }};
    }}

    function renderKpiDashboard(reasonLabel) {{
      const host = document.getElementById('kpiDashboard');
      const status = document.getElementById('kpiSnapshotStatus');
      if (!host) return;
      const snapshot = collectKpiSnapshot();
      if (!snapshot.totalFlights) {{
        host.innerHTML = '<div class="kpi-empty-state">No flights are available yet. Add or load a schedule, then click <strong>Update</strong> to refresh the KPI snapshot.</div>';
        if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · ' + kpiFormatSnapshotTime();
        return;
      }}
      const failureRate = snapshot.totalFlights > 0 ? ((snapshot.failedFlights / snapshot.totalFlights) * 100) : 0;
      const peakGateText = snapshot.peakBucket ? (kpiFormatCount(snapshot.peakBucket.occupancy) + ' gates') : '—';
      const peakGateMeta = snapshot.peakBucket ? (snapshot.peakBucket.label + ' peak occupancy') : 'No occupancy data';
      const busiestText = snapshot.busiestBucket ? (kpiFormatCount(snapshot.busiestBucket.total) + ' ops') : '—';
      const busiestMeta = snapshot.busiestBucket ? (snapshot.busiestBucket.label + ' arrivals + departures') : 'No movement data';
      const summaryCards = [
        kpiBuildSummaryCard('Total Flights', kpiFormatCount(snapshot.totalFlights), kpiFormatCount(snapshot.operationalFlights) + ' operational flights in snapshot', 'accent', 'Includes all loaded flights in the current layout'),
        kpiBuildSummaryCard('Failed Flights', kpiFormatCount(snapshot.failedFlights), kpiRound(failureRate, 1) + '% of total flights flagged as failed', snapshot.failedFlights > 0 ? 'danger' : 'success', 'Failed = no way or RET sampling failure'),
        kpiBuildSummaryCard('Average ROT', kpiFormatSecondsCompact(snapshot.rotAvgSec), 'Total ROT ' + kpiFormatSecondsCompact(snapshot.rotTotalSec), 'warning', 'Calculated from available ROT samples'),
        kpiBuildSummaryCard('Average Arr Taxi', kpiFormatMinutesCompact(snapshot.arrTaxiAvgMin), 'Total Arr Taxi ' + kpiFormatMinutesCompact(snapshot.arrTaxiTotalMin), '', 'Based on current arrival taxi path estimates'),
        kpiBuildSummaryCard('Average Dep Taxi', kpiFormatMinutesCompact(snapshot.depTaxiAvgMin), 'Total Dep Taxi ' + kpiFormatMinutesCompact(snapshot.depTaxiTotalMin), '', 'Based on current departure taxi path estimates'),
        kpiBuildSummaryCard('Peak Gate Occupancy', peakGateText, peakGateMeta, 'accent', 'Measured from EIBT–EOBT stand occupancy by hour')
      ].join('');
      const panelHtml = [
        kpiBuildPanel('Surface Movement', 'ROT + Taxi', [
          kpiBuildMetricRow('ROT time', 'Avg ' + kpiFormatSecondsValue(snapshot.rotAvgSec), 'Total ' + kpiFormatSecondsValue(snapshot.rotTotalSec)),
          kpiBuildMetricRow('Arrival taxi time', 'Avg ' + kpiFormatMinutesValue(snapshot.arrTaxiAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.arrTaxiTotalMin)),
          kpiBuildMetricRow('Departure taxi time', 'Avg ' + kpiFormatMinutesValue(snapshot.depTaxiAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.depTaxiTotalMin))
        ]),
        kpiBuildPanel('Passenger Delay', 'Gate View', [
          kpiBuildMetricRow('EIBT - SIBT', 'Avg ' + kpiFormatMinutesValue(snapshot.paxArrDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.paxArrDelayTotalMin)),
          kpiBuildMetricRow('EOBT - SOBT', 'Avg ' + kpiFormatMinutesValue(snapshot.paxDepDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.paxDepDelayTotalMin)),
          kpiBuildMetricRow('Busiest movement hour', busiestText, busiestMeta)
        ]),
        kpiBuildPanel('Aircraft Delay', 'Runway View', [
          kpiBuildMetricRow('ELDT - SLDT', 'Avg ' + kpiFormatMinutesValue(snapshot.acArrDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.acArrDelayTotalMin)),
          kpiBuildMetricRow('ETOT - STOT', 'Avg ' + kpiFormatMinutesValue(snapshot.acDepDelayAvgMin), 'Total ' + kpiFormatMinutesValue(snapshot.acDepDelayTotalMin)),
          kpiBuildMetricRow('Snapshot basis', kpiFormatCount(snapshot.totalFlights) + ' flights', 'Rendered only on initial load and Update')
        ])
      ].join('');
      const hourlyTableRows = (snapshot.buckets || []).map(function(bucket) {{
        const highlight = snapshot.peakBucket && bucket.bucketStart === snapshot.peakBucket.bucketStart ? ' class="kpi-row-highlight"' : '';
        return '' +
          '<tr' + highlight + '>' +
            '<td>' + escapeHtml(bucket.label) + '</td>' +
            '<td>' + escapeHtml(kpiFormatCount(bucket.occupancy)) + '</td>' +
            '<td>' + escapeHtml(kpiFormatCount(bucket.arrivals)) + '</td>' +
            '<td>' + escapeHtml(kpiFormatCount(bucket.departures)) + '</td>' +
            '<td>' + escapeHtml(kpiFormatCount(bucket.total)) + '</td>' +
          '</tr>';
      }}).join('');
      const topDelayRows = snapshot.detailRows.slice(0, 10).map(function(row) {{
        const statusClass = row.failed ? 'fail' : 'ok';
        const statusLabel = row.failed ? 'Failed' : 'Normal';
        return '' +
          '<tr>' +
            '<td>' + escapeHtml((row.reg || row.flightNumber || row.id || '—')) + '</td>' +
            '<td>' + escapeHtml(row.standName || 'Unassigned') + '</td>' +
            '<td>' + escapeHtml(kpiFormatMinutesValue(row.paxArrDelay)) + '</td>' +
            '<td>' + escapeHtml(kpiFormatMinutesValue(row.paxDepDelay)) + '</td>' +
            '<td>' + escapeHtml(kpiFormatMinutesValue((row.acArrDelay || 0) + (row.acDepDelay || 0))) + '</td>' +
            '<td><span class="kpi-badge ' + statusClass + '">' + escapeHtml(statusLabel) + '</span></td>' +
          '</tr>';
      }}).join('');
      host.innerHTML = '' +
        '<div class="kpi-summary-grid">' + summaryCards + '</div>' +
        '<div class="kpi-panel-grid">' + panelHtml + '</div>' +
        '<div class="kpi-chart-grid">' +
          '<div class="kpi-chart-card">' +
            '<div class="kpi-chart-head">' +
              '<div>' +
                '<div class="kpi-chart-title">Hourly Gate Occupancy</div>' +
                '<div class="kpi-chart-subtitle">Unique occupied stands by hour based on EIBT–EOBT occupancy windows.</div>' +
              '</div>' +
              '<div class="kpi-chart-legend">' +
                '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#a78bfa;"></span>Gate occupancy</span>' +
              '</div>' +
            '</div>' +
            kpiBuildOccupancyChart(snapshot.buckets) +
          '</div>' +
          '<div class="kpi-chart-card">' +
            '<div class="kpi-chart-head">' +
              '<div>' +
                '<div class="kpi-chart-title">Hourly Traffic Mix</div>' +
                '<div class="kpi-chart-subtitle">Arrival and departure counts by hour with total hourly movement overlay.</div>' +
              '</div>' +
              '<div class="kpi-chart-legend">' +
                '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#38bdf8;"></span>Arrivals</span>' +
                '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#fb923c;"></span>Departures</span>' +
                '<span class="kpi-legend-item"><span class="kpi-legend-swatch" style="background:#c4b5fd;"></span>Total</span>' +
              '</div>' +
            '</div>' +
            kpiBuildTrafficChart(snapshot.buckets) +
          '</div>' +
        '</div>' +
        '<div class="kpi-detail-grid">' +
          '<div class="kpi-table-card">' +
            '<div class="kpi-chart-title">Hourly Detail Table</div>' +
            '<div class="kpi-chart-subtitle">Quick lookup for gate occupancy and movement volume by hour.</div>' +
            '<div class="kpi-table-wrap">' +
              '<table class="kpi-table">' +
                '<thead><tr><th>Hour</th><th>Gate Occupancy</th><th>Arrivals</th><th>Departures</th><th>Total</th></tr></thead>' +
                '<tbody>' + hourlyTableRows + '</tbody>' +
              '</table>' +
            '</div>' +
          '</div>' +
          '<div class="kpi-table-card">' +
            '<div class="kpi-chart-title">Top Delay Flights</div>' +
            '<div class="kpi-chart-subtitle">Flights with the largest combined passenger and aircraft delay footprint.</div>' +
            '<div class="kpi-table-wrap">' +
              '<table class="kpi-table">' +
                '<thead><tr><th>Flight</th><th>Stand</th><th>Pax Arr Delay</th><th>Pax Dep Delay</th><th>Aircraft Delay</th><th>Status</th></tr></thead>' +
                '<tbody>' + topDelayRows + '</tbody>' +
              '</table>' +
            '</div>' +
          '</div>' +
        '</div>';
      if (status) status.textContent = (reasonLabel || 'Snapshot') + ' · ' + kpiFormatSnapshotTime();
    }}

    // S(d) Series: First S(d)=S(Original), Takes precedence when the same parking lot overlaps SOBT(d)-trailing SIBT(d) trailing as much as S(d) push. SLDT(d)=SLDT, SOBT(d)to Min Dwell reflect.
    // Original S The series is not referenced anywhere after being copied in this function. All calculations are S(d)use only.
    function computeScheduledDisplayTimes(flights) {{
      if (!flights || !flights.length) return;
      flights.forEach(f => {{
        if (f.noWayArr || f.noWayDep) return;
        f.vttADelayMin = 0;
        const tArrMin = f.timeMin != null ? f.timeMin : 0;
        let dwell = f.dwellMin != null ? f.dwellMin : 0;
        let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
        dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
        minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
        if (minDwell > dwell) minDwell = dwell;
        f.dwellMin = dwell;
        f.minDwellMin = minDwell;
        // VTT(Arr)Is Flight ScheduleReuse the same calculated value as the definition used in
        let vttArrMin = getBaseVttArrMinutes(f);
        const rotArrMin = getArrRotMinutes(f);
        const vttDepMin = getBaseVttDepMinutes(f);
        const sldtOrig = Math.max(0, tArrMin - vttArrMin - rotArrMin);
        const sobtOrig = tArrMin + dwell;
        const stotOrig = sobtOrig + vttDepMin;
        // SLDT/SIBT/SOBT/STOT(orig)is always updated with the internal calculated value.,
        // SLDT(d)Is SLDT(orig)Copy and use as is (JSON Ignore the initial value)
        f.sldtMin_orig = sldtOrig;
        f.sibtMin_orig = tArrMin;
        f.sobtMin_orig = sobtOrig;
        f.stotMin_orig = stotOrig;
        f.sldtMin_d = f.sldtMin_orig;
        f.sibtMin_d = tArrMin;
        f.sobtMin_d = sobtOrig;
        f.stotMin_d = stotOrig;
      }});
      const standToFlights = {{}};
      flights.forEach(f => {{
        if (f.noWayArr || f.noWayDep || !f.standId) return;
        const sid = f.standId;
        if (!standToFlights[sid]) standToFlights[sid] = [];
        standToFlights[sid].push(f);
      }});
      Object.keys(standToFlights).forEach(standId => {{
        const list = standToFlights[standId];
        list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
        let prevSOBT = -1e9;
        list.forEach(f => {{
          const vttDepMin = getBaseVttDepMinutes(f);
          const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
          const overlap = Math.max(0, prevSOBT - sibt0);
          f.vttADelayMin = overlap;
          // OVLP after reflection SIBT(d)
          f.sibtMin_d = sibt0 + overlap;
          // existing SOBT(d) candidate (yes: separation Value pushed out of logic)If you have it, keep it,
          // Min dwell Minimum required as standard SOBT(d)Use a larger value compared to
          const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
          const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
          const minSobtByDwell = f.sibtMin_d + minDwell;
          const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
          f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
          f.stotMin_d = f.sobtMin_d + vttDepMin;
          prevSOBT = f.sobtMin_d;
        }});
      }});
      // OVLP For all stand assigned flights, regardless of whether
      // Min dwell By enforcing the constraint once more, regular flights are corrected to follow the same rules.
      flights.forEach(f => {{
        if (!f || f.noWayArr || f.noWayDep || !f.standId) return;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const sibt = (f.sibtMin_d != null ? f.sibtMin_d
                     : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
        const minSobtByDwell = sibt + minDwell;
        const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
        if (sobtCurrent < minSobtByDwell) {{
          const delta = minSobtByDwell - sobtCurrent;
          f.sobtMin_d = minSobtByDwell;
          if (typeof f.stotMin_d === 'number') f.stotMin_d += delta;
        }}
      }});
      flights.forEach(f => {{
        if (f.noWayArr || f.noWayDep) return;
        f.sldtMin = f.sldtMin_d;
        f.stotMin = f.stotMin_d;
        f.sobtMin = f.sobtMin_d;
      }});
    }}

    function rsepGetSec(val) {{
      const n = Number(val);
      return isFinite(n) && n >= 0 ? n : RSEP_MISSING_MATRIX_SEC;
    }}

    function rsepApplySeparationToEvents(events, cfg) {{
      const arrArr = (cfg.seqData && cfg.seqData['ARR→ARR']) ? cfg.seqData['ARR→ARR'] : {{}};
      const depDep = (cfg.seqData && cfg.seqData['DEP→DEP']) ? cfg.seqData['DEP→DEP'] : {{}};
      const depArr = (cfg.seqData && cfg.seqData['DEP→ARR']) ? cfg.seqData['DEP→ARR'] : {{}};
      const rot = (cfg.rot) ? cfg.rot : {{}};
      const getSec = rsepGetSec;
      events.sort((a, b) => a.time - b.time || a.index - b.index);
      let lastArrETime = -1e9, lastArrCat = null;
      let lastDepETime = -1e9, lastDepCat = null;
      events.forEach(ev => {{
        if (ev.type === 'arr') {{
          let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec((arrArr[lastArrCat] && arrArr[lastArrCat][ev.cat]) != null ? arrArr[lastArrCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
          let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec(depArr[ev.cat]) / 60 : -1e9;
          const eTime = Math.max(ev.time, minFromArr, minFromDep);
          ev.flight.eldtMin = eTime;
          lastArrETime = eTime;
          lastArrCat = ev.cat;
        }} else {{
          let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec(rot[lastArrCat]) / 60 : -1e9;
          let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec((depDep[lastDepCat] && depDep[lastDepCat][ev.cat]) != null ? depDep[lastDepCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
          const etotSep = Math.max(ev.time, minFromArr, minFromDep);
          const vttADelay = ev.flight.vttADelayMin != null ? ev.flight.vttADelayMin : 0;
          const rotM = (ev.rotArrMin != null && isFinite(ev.rotArrMin)) ? ev.rotArrMin : getArrRotMinutes(ev.flight);
          const eibtMin = (ev.flight.eldtMin != null ? ev.flight.eldtMin : 0) + rotM + (ev.vttArrMin || 0) + vttADelay;
          const vttDep = ev.vttDepMin || 0;
          const etotMin = etotSep;
          const eobtMin = etotMin - vttDep;
          ev.flight.etotMin = etotMin;
          lastDepETime = etotMin;
          lastDepCat = ev.cat;
        }}
      }});
      let minT = Infinity, maxT = -Infinity;
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
      if (!isFinite(minT) || !isFinite(maxT)) {{ minT = 0; maxT = 60; }} else if (maxT <= minT) {{ maxT = minT + 60; }}
      return {{ minT, maxT }};
    }}

    function rsepCollectEventsForRunway(rwy, flights, runways) {{
      const cfg = rsepGetConfigForRunway(rwy);
      if (!cfg) return null;
      const stdKey = cfg.standard || 'ICAO';
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
        const sldtMin_d = f.sldtMin_d != null ? f.sldtMin_d : 0;
        const stotMin_d = f.stotMin_d != null ? f.stotMin_d : 0;
        const sobtMin_d = f.sobtMin_d != null ? f.sobtMin_d : 0;
        const vttArrMin = getBaseVttArrMinutes(f);
        const rotArrMin = getArrRotMinutes(f);
        const vttDepMin = getBaseVttDepMinutes(f);
        if (arrRwy === rwy.id) events.push({{ time: sldtMin_d, type: 'arr', flight: f, cat: cat, vttArrMin, rotArrMin, index: eventIndex++ }});
        if (depRwy === rwy.id) {{
          events.push({{ time: stotMin_d, type: 'dep', flight: f, cat: cat, vttDepMin, vttArrMin, rotArrMin, sobtMin: sobtMin_d, index: eventIndex++ }});
        }}
      }});
      return {{ cfg, events }};
    }}

    function runSeparationPass(runways, flights, byRunway, phase) {{
      if (phase === 'initial') {{
        runways.forEach(rwy => {{
          const pack = rsepCollectEventsForRunway(rwy, flights, runways);
          if (!pack) return;
          const {{ cfg, events }} = pack;
          if (!events.length) {{
            byRunway[rwy.id] = {{ events: [], minT: 0, maxT: 0 }};
            return;
          }}
          const {{ minT, maxT }} = rsepApplySeparationToEvents(events, cfg);
          byRunway[rwy.id] = {{ events, minT, maxT }};
        }});
      }} else {{
        runways.forEach(rwy => {{
          const cfg = rsepGetConfigForRunway(rwy);
          if (!cfg) return;
          const data = byRunway[rwy.id];
          if (!data || !data.events || !data.events.length) return;
          const events = data.events;
          events.forEach(ev => {{
            ev.time = ev.type === 'arr'
              ? (ev.flight.eldtMin != null ? ev.flight.eldtMin : ev.time)
              : (ev.flight.etotMin != null ? ev.flight.etotMin : ev.time);
          }});
          const {{ minT, maxT }} = rsepApplySeparationToEvents(events, cfg);
          byRunway[rwy.id] = {{ events, minT, maxT }};
        }});
      }}
    }}

    // Runway separation: SLDT(Arr)·STOT(Dep) Single timeline sorted chronologically, same time at the top(List order)See it as a good deed
    // preceding Eline + Trailing by separation criteria ESeries calculation (dominoes). Arr→ELDT, Dep→ETOT. All standards are S(d) Use series.
    // Returns: Runway IDnot really events/minT/maxTA map containing (For visualization)
    function computeSeparationAdjustedTimes() {{
      const flights = state.flights || [];
      // E line(ELDT/ETOT) When recalculating, the already calculated S(d) Use the series as is.
      // SOBT(d)·STOT(d) Coordination logic is the first S(d) calculation function(computeScheduledDisplayTimes)Perform only in.
      flights.forEach(f => {{ delete f.eldtMin; delete f.etotMin; }});
      const runwaysRaw = (state.taxiways || []).filter(t => t.pathType === 'runway');
      if (!runwaysRaw.length) return {{}};

      // on the same aircraft Arr the runway Dep Sort the runway order so that it is processed before the runway
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
        // If all nodes cannot be visited due to rotation, etc., the original order is used.
        if (orderIdx.length !== n) return runwaysRaw;
        return orderIdx.map(i => runwaysRaw[i]);
      }})();

      const byRunway = {{}};
      runSeparationPass(runways, flights, byRunway, 'initial');
      // Eline Apron Overlap: 1st RW result EBy parking lot EIBT After aligning the criteria, if they overlap, push them back, and then RWto final E confirmed
      flights.forEach(f => {{
        if (f.noWayArr || f.noWayDep) return;
        const vttArrMin = getBaseVttArrMinutes(f);
        const rotArrMin = getArrRotMinutes(f);
        const vttDepMin = getBaseVttDepMinutes(f);
        const vttADelay = f.vttADelayMin != null ? f.vttADelayMin : 0;
        f.eibtMin = (f.eldtMin != null ? f.eldtMin : 0) + rotArrMin + vttArrMin + vttADelay;
        f.eobtMin = (f.etotMin != null ? f.etotMin : 0) - vttDepMin;
      }});
      const standToFlightsE = {{}};
      flights.forEach(f => {{ if (f && !f.noWayArr && !f.noWayDep) f.eOverlapPushed = false; }});
      flights.forEach(f => {{
        if (f.noWayArr || f.noWayDep || !f.standId) return;
        const sid = f.standId;
        if (!standToFlightsE[sid]) standToFlightsE[sid] = [];
        standToFlightsE[sid].push(f);
      }});
      Object.keys(standToFlightsE).forEach(standId => {{
        const list = standToFlightsE[standId];
        list.sort((a, b) => (a.eibtMin != null ? a.eibtMin : 0) - (b.eibtMin != null ? b.eibtMin : 0));
        let prevEOBT = -1e9;
        list.forEach(f => {{
          const vttDepMin = getBaseVttDepMinutes(f);
          const vttArrMin = getBaseVttArrMinutes(f);
          const rotArrMin = getArrRotMinutes(f);
          const vttADelay = f.vttADelayMin != null ? f.vttADelayMin : 0;
          const eibtMin = f.eibtMin != null ? f.eibtMin : 0;
          const overlap = Math.max(0, prevEOBT - eibtMin);
          f.eOverlapPushed = overlap > 0;
          f.eibtMin = eibtMin + overlap;
          // S(d) Follows the dwell of the series.: dwellSd = SOBT(d) - SIBT(d)
          const dwellSd = (f.sobtMin_d != null && f.sibtMin_d != null)
            ? Math.max(0, f.sobtMin_d - f.sibtMin_d)
            : (f.dwellMin != null ? f.dwellMin : 0);
          const runwayEtot = f.etotMin != null ? f.etotMin : (f.eobtMin + vttDepMin);
          f.eobtMin = f.eibtMin + dwellSd;        // ✅ EOBT = EIBT + S(d) dwell
          f.eldtMin = f.eibtMin - rotArrMin - vttArrMin - vttADelay;
          // ELDTis physically SLDT(d)Hard clamp to prevent it from getting ahead of you
          const sldtBase = (f.sldtMin_d != null ? f.sldtMin_d
                           : (f.sldtMin_orig != null ? f.sldtMin_orig : 0));
          if (f.eldtMin < sldtBase) {{
            f.eldtMin = sldtBase;
            f.eibtMin = f.eldtMin + rotArrMin + vttArrMin + vttADelay;
            f.eobtMin = f.eibtMin + dwellSd;
          }}
          // IMPORTANT: keep ETOT fixed to the runway-based value, do not let apron overlap/dwell push ETOT further
          f.etotMin = runwayEtot;
          prevEOBT = f.eobtMin;
        }});
      }});
      // 2car RW: back EUse the series as the event time and perform the same as the original logic one more time.
      runSeparationPass(runways, flights, byRunway, 'refine');
      // SLDT(d)is the smallest(early) Flights are always ELDT = SLDT(d). Applies to only one arrival per runway.
      runways.forEach(rwy => {{
        const data = byRunway[rwy.id];
        if (!data || !data.events) return;
        const arrEvs = data.events.filter(e => e.type === 'arr');
        if (!arrEvs.length) return;
        let minSldt = Infinity, earliestArrFlight = null;
        arrEvs.forEach(ev => {{
          const sldt = (ev.flight.sldtMin_d != null ? ev.flight.sldtMin_d : (ev.flight.sldtMin_orig != null ? ev.flight.sldtMin_orig : Infinity));
          if (sldt < minSldt) {{ minSldt = sldt; earliestArrFlight = ev.flight; }}
        }});
        if (earliestArrFlight) {{
          const sldtBase = earliestArrFlight.sldtMin_d != null ? earliestArrFlight.sldtMin_d : (earliestArrFlight.sldtMin_orig != null ? earliestArrFlight.sldtMin_orig : 0);
          earliestArrFlight.eldtMin = sldtBase;
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

    function getRunwayPointAtDistance(runwayId, distM) {{
      const path = getRunwayPath(runwayId);
      if (!path || !path.pts || path.pts.length < 2) return null;
      const pts = path.pts;
      let acc = 0;
      for (let i = 0; i < pts.length - 1; i++) {{
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const segLen = pathDist(p1, p2);
        if (!(segLen > 1e-6)) continue;
        if (acc + segLen >= distM) {{
          const t = Math.max(0, Math.min(1, (distM - acc) / segLen));
          return [
            p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t
          ];
        }}
        acc += segLen;
      }}
      return pts[pts.length - 1];
    }}

    /** Runway departure lineup: JSON/When panel is not set 0(starting point). */
    function getEffectiveRunwayLineupDistM(tw) {{
      if (!tw || tw.pathType !== 'runway') return 0;
      const v = tw.lineupDistM;
      if (typeof v === 'number' && isFinite(v) && v >= 0) return v;
      return 0;
    }}

    function runwayPolylineLengthPx(pts) {{
      if (!pts || pts.length < 2) return 0;
      let s = 0;
      for (let i = 0; i < pts.length - 1; i++) s += pathDist(pts[i], pts[i + 1]);
      return s;
    }}

    /** On the runway centerline distPx Row of pixel coordinates from point to end (starting point dist branch included). */
    function polylineTailFromDistancePx(pts, distPx) {{
      if (!pts || pts.length < 2) return [];
      const total = runwayPolylineLengthPx(pts);
      const d = Math.max(0, Math.min(distPx, total));
      if (d <= 1e-9) return pts.map(p => [p[0], p[1]]);
      let acc = 0;
      for (let i = 0; i < pts.length - 1; i++) {{
        const p1 = pts[i], p2 = pts[i + 1];
        const segLen = pathDist(p1, p2);
        if (segLen < 1e-9) continue;
        if (acc + segLen >= d - 1e-6) {{
          const t = Math.max(0, Math.min(1, (d - acc) / segLen));
          const lp = [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
          const out = [lp];
          for (let j = i + 1; j < pts.length; j++) out.push([pts[j][0], pts[j][1]]);
          return out;
        }}
        acc += segLen;
      }}
      return [[pts[pts.length - 1][0], pts[pts.length - 1][1]]];
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
    // segment (a,b)and (c,d)intersection. Returns only when actual intersection occurs (0<=t,s<=1).
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

    // Runwaynot very connected Runway Taxiway(RET)of thresholddistance from(m)calculate
    function computeRunwayExitDistances() {{
      const taxiways = state.taxiways || [];
      const runways = taxiways.filter(t => t.pathType === 'runway' && Array.isArray(t.vertices) && t.vertices.length >= 2);
      const exits = taxiways.filter(t => t.pathType === 'runway_exit' && Array.isArray(t.vertices) && t.vertices.length >= 2);
      const results = [];
      if (!runways.length || !exits.length) return results;

      runways.forEach(rw => {{
        // Runway center line(grid coordinates) array: start_point Reference direction summary
        let rVerts = rw.vertices.map(v => [v.col, v.row]);
        if (rw.start_point && rw.end_point && rVerts.length >= 2) {{
          const sp = [rw.start_point.col, rw.start_point.row];
          if (pathDist2(rVerts[rVerts.length - 1], sp) < pathDist2(rVerts[0], sp)) rVerts.reverse();
        }}
        if (rVerts.length < 2) return;
        // prefix distance(cell unit)
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
              // proj.pIs a-b point on segment
              const t = Math.max(0, Math.min(1, segLen > 0 ? pathDist(a, proj.p) / segLen : 0));
              const distCells = prefixDist[i] + segLen * t;
              const distM = distCells * CELL_SIZE;
              const maxExitVelRaw = (typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0)
                ? tw.maxExitVelocity
                : 30;
              const minExitVelRaw = (typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
                ? tw.minExitVelocity
                : 15;
              const maxExitVel = maxExitVelRaw;
              const minExitVel = Math.min(minExitVelRaw, maxExitVel);
              if (!best || distM < best.distM) {{
                best = {{ runway: rw, exit: tw, name: exitName, distM, maxExitVelocity: maxExitVel, minExitVelocity: minExitVel }};
              }}
            }}
          }});
          if (best) results.push(best);
        }});
      }});

      // thresholdSort by distance from
      results.sort((a, b) => a.distM - b.distM);
      return results;
    }}

    function buildPathGraph(selectedArrRetId) {{
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
                // If it is a collinear line, the intersection is null. If endpoints overlap, add them as junctions (Runway-taxiway ends connected)
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
        if (obj.pathType === 'runway') {{
          const ldm = getEffectiveRunwayLineupDistM(obj);
          const rpath = getRunwayPath(obj.id);
          if (rpath && rpath.pts && rpath.pts.length >= 2 && ldm > 1e-6) {{
            let total = 0;
            for (let ri = 0; ri < rpath.pts.length - 1; ri++) total += pathDist(rpath.pts[ri], rpath.pts[ri + 1]);
            const d = Math.min(ldm, total);
            if (d > 1e-6) {{
              let acc = 0;
              for (let ri = 0; ri < rpath.pts.length - 1; ri++) {{
                const p1 = rpath.pts[ri], p2 = rpath.pts[ri + 1];
                const segLen = pathDist(p1, p2);
                if (segLen < 1e-9) continue;
                if (acc + segLen >= d - 1e-6) {{
                  const t = Math.max(0, Math.min(1, (d - acc) / segLen));
                  const px = p1[0] + t * (p2[0] - p1[0]), py = p1[1] + t * (p2[1] - p1[1]);
                  junctions.push({{ tAlong: ri + t, p: [px, py] }});
                  break;
                }}
                acc += segLen;
              }}
            }}
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
        const isRunwayExit = obj.pathType === 'runway_exit';
        const isTaxiway = obj.pathType === 'taxiway';
        for (let i = 0; i < chain.length - 1; i++) {{
          let d = pathDist(chain[i], chain[i+1]);
          let cost = d;
          if (selectedArrRetId != null) {{
            if (isRunwayExit && obj.id !== selectedArrRetId) cost = REVERSE_COST;
            else if (isTaxiway) cost = d + TAXIWAY_HEURISTIC_COST;
          }}
          addEdgeWithDirection(chain[i], chain[i+1], dir, cost);
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
      // BFS: cost < REVERSE_COSTA set of nodes that can be reached only along an edge.
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
      // Runway-park Junction within a connected section within reach of each other(degree>=2)Only marked with green junctions
      const junctionsForDraw = junctionPts.filter(function(p) {{
        const i = keyToIdx[pathPointKey(p)];
        return i != null && adj[i] && adj[i].length >= 2 && connected.has(i);
      }});
      return {{ nodes, edges, adj, getOrAdd, junctions: junctionsForDraw, standIdToNodeIndex }};
    }}

    function rebuildDerivedGraphEdges() {{
      state.derivedGraphEdges = [];
      if (!state.taxiways || !state.taxiways.length) return;
      let g;
      try {{
        g = buildPathGraph(null);
      }} catch (err) {{
        return;
      }}
      if (!g || !g.edges || !g.nodes) return;
      const seen = new Set();
      const raw = [];
      g.edges.forEach(function(e) {{
        if (e.dist >= REVERSE_COST || e.dist < 1e-6) return;
        const a = e.from, b = e.to;
        const lo = a < b ? a : b, hi = a < b ? b : a;
        const k = lo + ':' + hi;
        if (seen.has(k)) return;
        seen.add(k);
        const p0 = g.nodes[a], p1 = g.nodes[b];
        if (!p0 || !p1) return;
        raw.push({{ x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1], dist: e.dist, fromIdx: a, toIdx: b }});
      }});
      raw.sort(function(u, v) {{
        if (u.fromIdx !== v.fromIdx) return u.fromIdx - v.fromIdx;
        return u.toIdx - v.toIdx;
      }});
      const maxN = Math.min(raw.length, 999);
      for (let i = 0; i < maxN; i++) {{
        const label = String(i + 1).padStart(3, '0');
        const r = raw[i];
        state.derivedGraphEdges.push({{
          id: 'layout-edge-' + label,
          label: label,
          x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2,
          dist: r.dist,
          fromIdx: r.fromIdx,
          toIdx: r.toIdx
        }});
      }}
      if (state.selectedObject && state.selectedObject.type === 'layoutEdge') {{
        const sid = state.selectedObject.id;
        const fresh = (state.derivedGraphEdges || []).find(function(e) {{ return e.id === sid; }});
        if (fresh) state.selectedObject.obj = fresh;
        else state.selectedObject = null;
      }}
    }}

    function hitTestLayoutGraphEdge(wx, wy) {{
      if (!state.derivedGraphEdges || !state.derivedGraphEdges.length) return null;
      const click = [wx, wy];
      const tol = CELL_SIZE * 0.4;
      const tol2 = tol * tol;
      let best = null, bestD2 = tol2;
      state.derivedGraphEdges.forEach(function(ed) {{
        const near = closestPointOnSegment([ed.x1, ed.y1], [ed.x2, ed.y2], click);
        if (!near) return;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {{ bestD2 = d2; best = ed; }}
      }});
      return best;
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

    function graphPathArrival(f) {{
      if (f.arrRetFailed) {{ f.noWayArr = true; return null; }}
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
      const selectedArrRetId = f.sampledArrRet != null ? f.sampledArrRet : null;
      const validSelectedArrRetId = (selectedArrRetId != null && (state.taxiways || []).some(function(t) {{
        return t && t.id === selectedArrRetId && t.pathType === 'runway_exit';
      }})) ? selectedArrRetId : null;
      if (selectedArrRetId != null && validSelectedArrRetId == null) {{
        f.sampledArrRet = null;
        f.arrRetFailed = false;
        f.arrRotSec = null;
      }}
      const g = buildPathGraph(validSelectedArrRetId);
      state.pathGraphJunctions = g.junctions || [];
      const endIdx = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (endIdx == null) {{
        f.noWayArr = true;
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
        return null;
      }}
      f.noWayArr = false;
      return pathIndices.map(i => g.nodes[i]);
    }}

    function graphPathDeparture(f) {{
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
      const rwTw = (state.taxiways || []).find(t => t.id === runwayId && t.pathType === 'runway')
        || (state.taxiways || []).find(t => t.id === runwayId && (t.name||'').toLowerCase().includes('runway'));
      const stand = (state.pbbStands || []).find(s => s.id === apronId) || (state.remoteStands || []).find(s => s.id === apronId);
      if (!stand) return null;
      const g = buildPathGraph();
      const startIdx = (g.standIdToNodeIndex && g.standIdToNodeIndex[apronId] != null) ? g.standIdToNodeIndex[apronId] : null;
      if (startIdx == null) {{
        f.noWayDep = true;
        return null;
      }}
      const useLineup = rwTw && rwTw.pathType === 'runway';
      if (useLineup) {{
        const ldm = getEffectiveRunwayLineupDistM(rwTw);
        const lenPx = runwayPolylineLengthPx(r.pts);
        const dPx = Math.min(Math.max(0, ldm), lenPx);
        const lineupPx = getRunwayPointAtDistance(runwayId, dPx);
        if (!lineupPx) {{
          f.noWayDep = true;
          return null;
        }}
        const lineupIdx = nearestPathNode(g, lineupPx);
        const pathIndices = pathDijkstra(g, startIdx, lineupIdx);
        const totalD = pathIndices ? pathTotalDist(g, pathIndices) : Infinity;
        if (!pathIndices || pathIndices.length < 2 || totalD >= REVERSE_COST) {{
          f.noWayDep = true;
          return null;
        }}
        f.noWayDep = false;
        let pts = pathIndices.map(i => g.nodes[i]);
        const tail = polylineTailFromDistancePx(r.pts, dPx);
        if (tail.length) {{
          const last = pts[pts.length - 1];
          const firstTail = tail[0];
          if (pathDist2(last, firstTail) <= SPLIT_TOL_D2) pts = pts.concat(tail.slice(1));
          else pts = pts.concat(tail);
        }}
        if (r.endPx && Array.isArray(r.endPx) && r.endPx.length === 2) {{
          const last = pts[pts.length - 1];
          if (pathDist(last, r.endPx) > 1e-3) pts.push([r.endPx[0], r.endPx[1]]);
        }}
        return pts;
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
        return null;
      }}
      f.noWayDep = false;
      const pts = pathIndices.map(i => g.nodes[i]);
      if (r && r.endPx && Array.isArray(r.endPx) && r.endPx.length === 2) {{
        const last = pts[pts.length - 1];
        const dx = r.endPx[0] - last[0];
        const dy = r.endPx[1] - last[1];
        if (Math.hypot(dx, dy) > 1e-3) pts.push(r.endPx);
      }}
      return pts;
    }}

    function getPathForFlight(f) {{
      resolveStand(f);
      return graphPathArrival(f);
    }}

    function getPathForFlightDeparture(f) {{
      resolveStand(f);
      return graphPathDeparture(f);
    }}

    function computeFlightPath(flight, direction) {{
      resolveStand(flight);
      if (direction === 'arrival') {{
        const pts = graphPathArrival(flight);
        const timeline = (!flight.noWayArr && pts && pts.length >= 2)
          ? buildArrivalTimelineFromPts(flight, pts)
          : null;
        return {{ pts: pts || null, timeline }};
      }}
      const pts = graphPathDeparture(flight);
      const timeline = (!flight.noWayDep && pts && pts.length >= 2)
        ? buildDepartureTimelineFromPts(flight, pts)
        : null;
      return {{ pts: pts || null, timeline }};
    }}

    function updateAllFlightPaths() {{
      if (!state.flights || !state.flights.length) {{ draw(); return; }}
      state.flights.forEach(f => {{
        const arr = computeFlightPath(f, 'arrival');
        computeFlightPath(f, 'departure');
        if (f.noWayArr || f.noWayDep) f.timeline = null;
        else if (arr.timeline && arr.timeline.length) f.timeline = arr.timeline;
      }});
      if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
      // Update path only: Allocation/bar chart/List without touching the apron layout·Update only the canvas
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

    function drawSelectedLayoutEdge() {{
      const sel = state.selectedObject;
      if (!sel || sel.type !== 'layoutEdge' || !sel.obj) return;
      const e = sel.obj;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      function layoutEdgePath() {{
        ctx.beginPath();
        ctx.moveTo(e.x1, e.y1);
        ctx.lineTo(e.x2, e.y2);
      }}
      layoutEdgePath();
      ctx.save();
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(7, CELL_SIZE * 0.2);
      ctx.strokeStyle = c2dObjectSelectedStroke();
      ctx.shadowColor = c2dObjectSelectedGlow();
      ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      ctx.stroke();
      ctx.restore();
      layoutEdgePath();
      ctx.setLineDash([]);
      ctx.lineWidth = Math.max(4, CELL_SIZE * 0.12);
      ctx.strokeStyle = c2dObjectSelectedStroke();
      ctx.stroke();
      layoutEdgePath();
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 3;
      ctx.strokeStyle = c2dObjectSelectedDashStroke();
      ctx.stroke();
      ctx.restore();
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

      // --- Touch Down / RET entrance / RET outlet speed label ---
      ctx.font = 'bold ' + Math.max(9, CELL_SIZE * 0.35) + 'px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = '#fca5a5';
      function drawSpeedLabel(pt, label) {{
        if (!pt) return;
        const ox = 4, oy = -4;
        ctx.fillText(label, pt[0] + ox, pt[1] + oy);
      }}
      function drawTouchDownLabel(pt, distM, speedMs) {{
        if (!pt) return;
        const ox = 4, oy = -4;
        const x = pt[0] + ox, yBot = pt[1] + oy;
        const lh = Math.max(11, Math.round(CELL_SIZE * 0.36));
        let distPart = '---m';
        if (typeof distM === 'number' && isFinite(distM)) {{
          const r = Math.round(distM);
          distPart = (r >= 1000 ? String(r) : String(r).padStart(3, '0')) + 'm';
        }}
        let spdPart = '--.-m/s';
        if (typeof speedMs === 'number' && isFinite(speedMs)) {{
          spdPart = speedMs.toFixed(1) + 'm/s';
        }}
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText('(' + distPart + ',  ' + spdPart + ')', x, yBot);
        ctx.fillText('Touch Down', x, yBot - lh);
      }}
      let tdPt = null, retInPt = null, retOutPt = null;
      if (f.arrRunwayIdUsed && typeof getRunwayPointAtDistance === 'function') {{
        if (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM)) {{
          tdPt = getRunwayPointAtDistance(f.arrRunwayIdUsed, f.arrTdDistM);
        }}
        if (typeof f.arrRetDistM === 'number' && isFinite(f.arrRetDistM)) {{
          retInPt = getRunwayPointAtDistance(f.arrRunwayIdUsed, f.arrRetDistM);
        }}
      }}
      // RET OUT: selected RET(exit) Taxiwaylast point of
      if (!retOutPt && f.sampledArrRet) {{
        const tw = (state.taxiways || []).find(t => t.id === f.sampledArrRet);
        if (tw && Array.isArray(tw.vertices) && tw.vertices.length) {{
          const last = tw.vertices[tw.vertices.length - 1];
          retOutPt = cellToPixel(last.col, last.row);
        }}
      }}
      if (!tdPt && pathPts.length >= 1) tdPt = pathPts[0];
      if (!retInPt && pathPts.length >= 3) {{
        const idxIn = Math.max(1, Math.floor(pathPts.length * 0.4));
        retInPt = pathPts[Math.min(idxIn, pathPts.length - 1)];
      }}
      if (!retOutPt && pathPts.length >= 3) {{
        const idxOut = Math.max(2, Math.floor(pathPts.length * 0.7));
        retOutPt = pathPts[Math.min(idxOut, pathPts.length - 1)];
      }}
      if (tdPt && ((typeof f.arrVTdMs === 'number' && isFinite(f.arrVTdMs)) || (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM)))) {{
        drawTouchDownLabel(tdPt, f.arrTdDistM, f.arrVTdMs);
      }}
      if (typeof f.arrVRetInMs === 'number' && isFinite(f.arrVRetInMs)) {{
        drawSpeedLabel(retInPt, 'RET IN ' + f.arrVRetInMs.toFixed(1) + ' m/s');
      }}
      if (typeof f.arrVRetOutMs === 'number' && isFinite(f.arrVRetOutMs)) {{
        drawSpeedLabel(retOutPt, 'RET OUT ' + f.arrVRetOutMs.toFixed(1) + ' m/s');
      }}
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
      ctx.strokeStyle = _canvas2dStyle.pathDepartureStroke || '#000000';
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
      ctx.fillStyle = _canvas2dStyle.vttBadgeBg || 'rgba(15, 23, 42, 0.95)';
      ctx.strokeStyle = _canvas2dStyle.vttBadgeStroke || '#000000';
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
      ctx.fillStyle = _canvas2dStyle.vttBadgeText || '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeText, cx, cy - 4);
      ctx.restore();
    }}

    function drawFlights2D() {{
      if (!state.hasSimulationResult || !state.flights.length) return;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const tSec = state.simTimeSec;
      state.flights.forEach(f => {{
        const hasNoWay = f.noWayArr || f.noWayDep;
        if (hasNoWay) {{
          // No way: The aircraft position is not drawn, but the point where the link breaks.(stand)Only No way scab
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
          ctx.fillStyle = _canvas2dStyle.noWayFill || 'rgba(220, 38, 38, 0.92)';
          ctx.strokeStyle = _canvas2dStyle.noWayStroke || 'rgba(185, 28, 28, 0.9)';
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
          ctx.fillStyle = _canvas2dStyle.noWayText || '#ffffff';
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
        const code = (f.code || '').toUpperCase();
        const scale = apronAircraftScaleForIcao(code);
        const size = CELL_SIZE * scale;
        const silN = Number(_acSil.noseX), silWR = Number(_acSil.wingRearX), silUY = Number(_acSil.wingUpperY);
        const silTN = Number(_acSil.tailNeckX), silLY = Number(_acSil.wingLowerY);
        const nX = isFinite(silN) ? silN : 0.6;
        const wRx = isFinite(silWR) ? silWR : -0.5;
        const uY = isFinite(silUY) ? silUY : 0.35;
        const tX = isFinite(silTN) ? silTN : -0.3;
        const lY = isFinite(silLY) ? silLY : -0.35;
        const isFlightSel = state.selectedObject && state.selectedObject.type === 'flight' && state.selectedObject.id === f.id;
        if (isFlightSel) {{
          ctx.save();
          ctx.beginPath();
          ctx.arc(x, y, size * 0.62, 0, Math.PI * 2);
          ctx.strokeStyle = c2dObjectSelectedStroke();
          ctx.lineWidth = 2.5;
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.stroke();
          ctx.restore();
        }}
        ctx.save();
        ctx.translate(x, y);
        const ang = Math.atan2(ny, nx);
        ctx.rotate(ang);
        ctx.fillStyle = apron2DGlyphFill();
        ctx.beginPath();
        ctx.moveTo(size * nX, 0);
        ctx.lineTo(size * wRx, size * uY);
        ctx.lineTo(size * tX, 0);
        ctx.lineTo(size * wRx, size * lY);
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
        const dt = (ts - ensureSimLoop._lastTs) / 1000; // Elapsed time in seconds
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
      const regEl = document.getElementById('flightReg');
      const layoutNameInput = document.getElementById('layoutName');
      const saveLayoutBtn = document.getElementById('btnSaveLayout');
      const layoutMsgEl = document.getElementById('layoutMessage');
      const layoutLoadListEl = document.getElementById('layoutLoadList');
      const globalUpdateBtn = document.getElementById('btnGlobalUpdate');
      if (!arrDepEl) return;
      populateAircraftSelect(aircraftEl);

      function randomAirlineCode() {{ return DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)]; }}
      function randomFlightNumber(airlineCode) {{ return (airlineCode || randomAirlineCode()) + String(Math.floor(1000 + Math.random() * 9000)); }}
      // Currently already created Flightfield SIBT(d) The maximum value of + 10Minutes are basic SIBTused as
      function getDefaultSibtMinutes() {{
        let maxT = 0;
        (state.flights || []).forEach(f => {{
          if (!f) return;
          const sibt = f.sibtMin_d != null ? f.sibtMin_d : (typeof f.timeMin === 'number' ? f.timeMin : 0);
          if (isFinite(sibt) && sibt > maxT) maxT = sibt;
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
      // top of screen Global Update Button: Main view based on current state/Redo all calculations
      if (globalUpdateBtn) {{
        globalUpdateBtn.addEventListener('click', function() {{
          try {{
            if (typeof syncPanelFromState === 'function') syncPanelFromState();
            if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
            else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
            if (typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
            // Do all calculations first
            if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
            if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
            // Update the view using the calculation result
            if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
            if (typeof renderFlightGantt === 'function') renderFlightGantt();
            // at the end Flight schedule Render the table only once (RET/ELDT Full-length resampling)
            if (typeof renderFlightList === 'function') renderFlightList(false, true);
            if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Updated');
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
      // Flight My tab Schedule / Configuration Switch sub tabs
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
            alert('Flightcannot be created:\\n' + networkErrors.join('\\n'));
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
          const reg = (document.getElementById('flightReg').value || '').trim();
          let airlineCode = (document.getElementById('flightAirlineCode') && document.getElementById('flightAirlineCode').value || '').trim();
          let flightNumber = (document.getElementById('flightFlightNumber') && document.getElementById('flightFlightNumber').value || '').trim();
          if (!airlineCode) airlineCode = randomAirlineCode();
          if (!flightNumber) flightNumber = randomFlightNumber(airlineCode);
          let dwellMin = parseFloat(document.getElementById('flightDwell').value);
          let minDwellMin = parseFloat(document.getElementById('flightMinDwell').value);
          dwellMin = (typeof dwellMin === 'number' && !isNaN(dwellMin) && dwellMin >= 0) ? dwellMin : 0;
          minDwellMin = (typeof minDwellMin === 'number' && !isNaN(minDwellMin) && minDwellMin >= 0) ? minDwellMin : 0;
          dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, dwellMin);
          minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, minDwellMin);
          if (minDwellMin > dwellMin) minDwellMin = dwellMin;
          // Arr/Depis one Flight(Arr+Dep)It is a concept that expresses, and internally, Arr Manage only based on standards
          const arrDep = 'Arr';
          // basic Runway Select: currently defined Runway Use the first one in the list (If there is no null)
          const runwayOptions = getRunwayOptions();
          const defaultRunwayId = runwayOptions.length ? (runwayOptions[0].id || null) : null;
          const f = {{
            id: id(),
            arrDep,
            timeMin,
            aircraftType,
            code,
            reg,
            airlineCode,
            flightNumber,
            dwellMin,
            minDwellMin,
            arrRunwayId: defaultRunwayId,
            depRunwayId: defaultRunwayId,
            timeline: null,
            token: {{
              nodes: ['runway','taxiway','apron','terminal'],
              runwayId: defaultRunwayId,
              arrRunwayId: defaultRunwayId,
              depRunwayId: defaultRunwayId,
              apronId: null,
              terminalId: null
            }}
          }};
          const arrPath = computeFlightPath(f, 'arrival');
          computeFlightPath(f, 'departure');
          const tl = arrPath.timeline;
          if (!tl || !tl.length) {{
            // If there is no path timelinesilver nullLeave it at and display the warning message only on the right panel.
            updateFlightError('NOTE: Available on your network Taxiway / Apron path not found. (Simulation paths may not be drawn.)');
          }} else {{
            f.timeline = tl;
          }}
          state.flights.push(f);
          recomputeSimDuration();
          renderFlightList();
          // next Flight base for adding SIBT Input update (Max SIBT + 10minute)
          if (timeInputEl) {{
            const nextDef = getDefaultSibtMinutes();
            timeInputEl.value = formatMinutesToHHMMSS(nextDef);
          }}
          updateFlightError('');
        }});
      }}
      // Flight When selecting an object, the value is reflected in the panel input
      function syncFlightPanelFromSelection() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        // Arr/Depsilver UIdo not select from, all FlightIs Arr+stand occupancy(Dwell)Structure containing
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
        if (regEl) regEl.value = f.reg || '';
        const airlineCodeEl = document.getElementById('flightAirlineCode');
        const flightNumberEl = document.getElementById('flightFlightNumber');
        if (airlineCodeEl) airlineCodeEl.value = f.airlineCode || '';
        if (flightNumberEl) flightNumberEl.value = f.flightNumber || '';
        if (!f.token) f.token = {{ nodes: TOKEN_NODE_ORDER.slice(), runwayId: null, apronId: null, terminalId: null }};
        fillTokenSelects(f.code);
        setTokenCheckboxesFromNodes(f.token.nodes);
        if (tokenRunwaySel) tokenRunwaySel.value = f.token.runwayId || '';
        if (tokenTerminalSel) tokenTerminalSel.value = f.token.terminalId || '';
      }}
      hookSyncFlightPanelFromSelection = syncFlightPanelFromSelection;
      // selection So that every change is synchronized. hook
      const origSyncPanel = syncPanelFromState;
      syncPanelFromState = function() {{
        origSyncPanel();
        if (activeTab === 'flight') syncFlightPanelFromSelection();
      }};
      // Flight Selected when changing setting input Flight reflected in object + Recalculate route
      function rebuildSelectedFlightTimeline() {{
        if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
        const f = state.selectedObject.obj;
        const arr = computeFlightPath(f, 'arrival');
        const dep = computeFlightPath(f, 'departure');
        const isArr = f.arrDep !== 'Dep';
        if (isArr && f.noWayArr) {{
          updateFlightError('no path(No Way): Arrival route not found.');
          f.timeline = null;
          draw();
          return;
        }}
        if (!isArr && f.noWayDep) {{
          updateFlightError('no path(No Way): Departure route not found.');
          f.timeline = null;
          draw();
          return;
        }}
        const tl = isArr ? arr.timeline : dep.timeline;
        if (!tl || !tl.length) {{
          updateFlightError('No valid route found on that network. (After changing settings)');
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
      if (regEl) {{
        regEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.reg = this.value || '';
          renderFlightList();
          updateObjectInfo();
        }});
      }}
      const airlineCodeEl = document.getElementById('flightAirlineCode');
      const flightNumberEl = document.getElementById('flightFlightNumber');
      if (airlineCodeEl) {{
        airlineCodeEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.airlineCode = this.value || '';
          renderFlightList();
          updateObjectInfo();
        }});
      }}
      if (flightNumberEl) {{
        flightNumberEl.addEventListener('change', function() {{
          if (!state.selectedObject || state.selectedObject.type !== 'flight') return;
          const f = state.selectedObject.obj;
          f.flightNumber = this.value || '';
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
          let dwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
          let minDwell = f.minDwellMin != null ? f.minDwellMin : dwell;
          minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
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
          dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
          let v = parseFloat(this.value);
          v = (typeof v === 'number' && !isNaN(v) && v >= 0) ? v : 0;
          let minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, v);
          if (minDwell > dwell) minDwell = dwell;
          f.dwellMin = dwell;
          f.minDwellMin = minDwell;
          if (dwellEl) dwellEl.value = f.dwellMin;
          this.value = f.minDwellMin;
          // Flight Schedule graph + Apron Ganttrecalculate immediately/reflect
          const _prepSched = state.flights.slice();
          _prepSched.forEach(ff => {{
            if (typeof getPathForFlight === 'function') getPathForFlight(ff);
            if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(ff);
            if (ff.noWayArr || ff.noWayDep) ff.timeline = null;
          }});
          if (typeof ensureArrRetRotSampled === 'function') ensureArrRetRotSampled(_prepSched, false);
          if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
          if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
          if (typeof renderFlightGantt === 'function') renderFlightGantt();
          renderFlightList();
        }});
      }}
      if (playBtn) {{
        playBtn.addEventListener('click', function() {{
          const errs = validateNetworkForFlights();
          if (errs.length) {{
            state.simPlaying = false;
            updateFlightError(errs);
            alert('Simulation cannot be played:\\n' + errs.join('\\n'));
            return;
          }}
          if (!state.flights.length) {{
            updateFlightError('registered FlightThere is no.');
            alert('registered FlightThere is no.');
            return;
          }}
          // When playback starts, the earliest Flight start from time
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
        state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : {json.dumps(float(_ui_default_sim_speed))};
      }}
      // Flight Schedule the displayed value in the table state.flightsBy reflecting back on
      // Save/Run city JSONin table final values(especially Eline)sync to reflect this
      function syncTableToFlightState() {{
        const schedTable = document.querySelector('.flight-schedule-table');
        if (!schedTable || !Array.isArray(state.flights)) return;
        const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
        rows.forEach(function(row) {{
          const fid = row.getAttribute('data-id');
          if (!fid) return;
          const f = state.flights.find(function(ff) {{ return ff && ff.id === fid; }});
          if (!f) return;
          const tds = Array.from(row.querySelectorAll('td'));
          if (tds.length < 15) return;
          const getMin = function(idx) {{
            const txt = (tds[idx] && tds[idx].textContent || '').trim();
            if (!txt) return null;
            const parts = txt.split(':');
            if (parts.length >= 2) {{
              const h = parseInt(parts[0], 10) || 0;
              const m = parseInt(parts[1], 10) || 0;
              const s = parts.length >= 3 ? (parseInt(parts[2], 10) || 0) : 0;
              return h * 60 + m + s / 60;
            }}
            const n = parseFloat(txt);
            return isNaN(n) ? null : n;
          }};
          // Column order: 7=SLDT(d), 8=SIBT(d), 9=SOBT(d), 10=STOT(d)
          //            11=ELDT,  12=EIBT,   13=EOBT,   14=ETOT
          const map = {{
            sldtMin_d: 7, sibtMin_d: 8, sobtMin_d: 9,  stotMin_d: 10,
            eldtMin:  11, eibtMin:  12, eobtMin:  13, etotMin:   14
          }};
          Object.keys(map).forEach(function(key) {{
            const v = getMin(map[key]);
            if (v != null) f[key] = v;
          }});
        }});
      }}
      // Layout Save / Load: data/Layout_storage Save by name to·load (API)
      function setLayoutMessage(msg, isError) {{
        if (!layoutMsgEl) return;
        layoutMsgEl.textContent = msg || '';
        layoutMsgEl.style.color = isError ? '#f97316' : '#9ca3af';
      }}
      if (saveLayoutBtn) {{
        saveLayoutBtn.addEventListener('click', function() {{
          const name = (layoutNameInput && layoutNameInput.value || '').trim();
          if (!name) {{
            setLayoutMessage('Please enter a save name.', true);
            return;
          }}
          try {{
            if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
            if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
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
              }} else setLayoutMessage('save failed (status ' + r.status + ') — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
            }}).catch(function(e) {{
              setLayoutMessage('Connection failed: ' + (e && e.message) + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection', true);
            }});
          }} catch (e) {{
            console.error(e);
            setLayoutMessage('Unable to save layout.', true);
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
            if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Running simulation...'; layoutMsgEl.style.color = '#9ca3af'; }}
            fetch(apiBase + '/api/run-simulation', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ layout: data, layoutName: layoutName }})
            }}).then(function(r) {{
              if (!r.ok) throw new Error('Simulation failed');
              return r.json();
            }}).then(function(result) {{
              if (!result) return;
              state.hasSimulationResult = true;
              applyLayoutObject(result);
              resizeCanvas();
              reset2DView();
              syncPanelFromState();
              if (typeof draw === 'function') draw();
              if (typeof update3DScene === 'function') update3DScene();
              if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
              if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Simulation complete.'; layoutMsgEl.style.color = '#9ca3af'; }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Connection failed: ' + ((e && e.message) || 'Simulation failed') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }}
            }});
          }} catch (e) {{
            if (layoutMsgEl) {{ layoutMsgEl.textContent = 'error: ' + (e && e.message); layoutMsgEl.style.color = '#f97316'; }}
          }}
        }});
      }}
      // Save / Load sub-tabs (certainly #tab-saveload Inside only — Flight Prevent malfunctions during global queries by sharing subtabs and classes)
      function switchLayoutTab(tabId) {{
        const root = document.getElementById('tab-saveload');
        if (!root) return;
        root.querySelectorAll('.layout-save-load-tab').forEach(btn => btn.classList.remove('active'));
        root.querySelectorAll('.layout-save-load-pane').forEach(p => p.classList.remove('active'));
        const btn = root.querySelector('.layout-save-load-tab[data-sltab="' + tabId + '"]');
        const pane = document.getElementById('layout-' + tabId + '-pane');
        if (btn) btn.classList.add('active');
        if (pane) pane.classList.add('active');
        if (tabId === 'load') fetchAndRefreshLayoutList();
      }}
      const layoutMessageSaveEl = document.getElementById('layoutMessageSave');
      const btnSaveCurrent = document.getElementById('btnSaveCurrentLayout');
      if (btnSaveCurrent) btnSaveCurrent.addEventListener('click', function() {{
        const name = (state.currentLayoutName && state.currentLayoutName.trim()) || (INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
        try {{
          if (typeof syncStateFromPanel === 'function') syncStateFromPanel();
          if (typeof syncTableToFlightState === 'function') syncTableToFlightState();
          const data = serializeCurrentLayout();
          const apiBase = (typeof getLayoutApiBase === 'function') ? getLayoutApiBase() : (LAYOUT_API_URL || '');
          fetch(apiBase + '/api/save-layout', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ layout: data, name: name }})
          }}).then(function(r) {{
            if (r.ok) {{
              if (layoutMessageSaveEl) {{ layoutMessageSaveEl.textContent = 'saved: ' + name + '.json'; layoutMessageSaveEl.style.color = '#9ca3af'; }}
            }} else if (layoutMessageSaveEl) {{ layoutMessageSaveEl.textContent = 'save failed (status ' + r.status + ')'; layoutMessageSaveEl.style.color = '#f97316'; }}
          }}).catch(function(e) {{
            if (layoutMessageSaveEl) {{ layoutMessageSaveEl.textContent = 'Connection failed: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }}
          }});
        }} catch (e) {{ if (layoutMessageSaveEl) {{ layoutMessageSaveEl.textContent = 'error: ' + (e && e.message); layoutMessageSaveEl.style.color = '#f97316'; }} }}
      }});
      const saveLoadTabRoot = document.getElementById('tab-saveload');
      if (saveLoadTabRoot) {{
        saveLoadTabRoot.querySelectorAll('.layout-save-load-tab[data-sltab]').forEach(btn => {{
          btn.addEventListener('click', function() {{ switchLayoutTab(this.getAttribute('data-sltab')); }});
        }});
      }}
      function getLayoutApiBase() {{
        if (LAYOUT_API_URL && LAYOUT_API_URL !== 'null') return LAYOUT_API_URL;
        try {{ if (window.location && window.location.origin && window.location.origin !== 'null') return window.location.origin; }} catch(e) {{}}
        return '';
      }}
      function fetchAndRefreshLayoutList() {{
        if (!layoutLoadListEl) return;
        layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">Loading list...</div>';
        const apiBase = getLayoutApiBase();
        fetch(apiBase + '/api/list-layouts').then(function(r) {{
          if (!r.ok) throw new Error('API Connection failed (status ' + r.status + ')');
          return r.json();
        }}).then(function(data) {{
          const names = (data && data.names) ? data.names : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
          refreshLayoutLoadList(names);
        }}).catch(function(e) {{
          layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#f97316;">Connection failed: ' + (e && e.message) + '</div><div style="font-size:10px;color:#9ca3af;margin-top:4px;">python run_app.py After running with http://127.0.0.1:8501 connection</div>';
        }});
      }}
      function refreshLayoutLoadList(namesFromApi) {{
        if (!layoutLoadListEl) return;
        const names = namesFromApi != null ? (Array.isArray(namesFromApi) ? namesFromApi : []) : (Array.isArray(LAYOUT_NAMES) ? LAYOUT_NAMES : []);
        if (!names.length) {{
          layoutLoadListEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">There are no saved layouts.</div>';
          return;
        }}
        const reserved = {{ 'default_layout': true, 'current_layout': true }};
        layoutLoadListEl.innerHTML = names.map(function(name) {{
          const n = (name || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
          const showDel = !reserved[(name || '').toLowerCase()];
          const delBtn = showDel ? '<button type="button" class="layout-load-delete" title="Delete" data-name="' + (name || '').replace(/"/g, '&quot;') + '">×</button>' : '';
          return '<div class="layout-load-item" data-name="' + (name || '').replace(/"/g, '&quot;') + '"><span class="layout-load-name">' + n + '</span>' + delBtn + '</div>';
        }}).join('');
        layoutLoadListEl.querySelectorAll('.layout-load-item').forEach(function(el) {{
          const name = el.getAttribute('data-name');
          el.addEventListener('click', function(ev) {{
            if (ev.target && ev.target.classList && ev.target.classList.contains('layout-load-delete')) return;
            if (!name) return;
            var apiBase = getLayoutApiBase();
            if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Loading...'; layoutMsgEl.style.color = '#9ca3af'; }}
            fetch(apiBase + '/api/load-layout?name=' + encodeURIComponent(name)).then(function(r) {{
              if (!r.ok) throw new Error('not_found');
              return r.json();
            }}).then(function(obj) {{
              if (!obj || typeof obj !== 'object') {{ throw new Error('invalid_response'); }}
              try {{
                state.hasSimulationResult = false;
                applyLayoutObject(obj);
                resizeCanvas();
                reset2DView();
                syncPanelFromState();
                if (typeof draw === 'function') draw();
                if (typeof update3DScene === 'function') update3DScene();
                if (typeof updateLayoutNameBar === 'function') updateLayoutNameBar(name);
                if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
                if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Loaded \"' + name + '\"'; layoutMsgEl.style.color = '#9ca3af'; }}
              }} catch (err) {{
                console.error('applyLayoutObject error', err);
                throw err;
              }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = 'Failed to load: ' + ((e && e.message) || name || '') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }}
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
              if (!r.ok) return r.json().then(function(d) {{ throw new Error(d.error || 'Deletion failed'); }});
              return fetch(apiBase + '/api/list-layouts').then(function(r2) {{ return r2.json(); }});
            }}).then(function(data) {{
              if (data && data.names) refreshLayoutLoadList(data.names);
              if (layoutMsgEl) {{ layoutMsgEl.textContent = 'deleted.'; layoutMsgEl.style.color = '#9ca3af'; }}
            }}).catch(function(e) {{
              if (layoutMsgEl) {{ layoutMsgEl.textContent = ((e && e.message) || 'Deletion failed') + ' — python run_app.pyAfter running with http://127.0.0.1:8501 connection'; layoutMsgEl.style.color = '#f97316'; }}
            }});
          }});
        }});
      }}
      // On page load API Check connection (405/404 City information banner display)
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
      // Drawstart/Deselect existing objects when exiting
      state.selectedObject = null;
      if (state.terminalDrawingId) {{
        const t = state.terminals.find(x => x.id === state.terminalDrawingId);
        if (t && !t.closed && t.vertices.length >= 3) {{
          t.closed = true;
          // Upon completion Taxiway Overlap check with center line
          if (terminalOverlapsAnyTaxiway(t)) {{
            alert('this Apron/Terminalsilver Taxiway Overlaps the center line. Please place it in a different location.');
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
      // Drawstart/Deselect existing objects when exiting
      state.selectedObject = null;
      if (state.taxiwayDrawingId) {{
        const tw = state.taxiways.find(x => x.id === state.taxiwayDrawingId);
        if (tw && tw.vertices.length >= 2) {{
          // Check for overlap with terminal upon completion
          if (taxiwayOverlapsAnyTerminal(tw)) {{
            alert('this TaxiwayIs TerminalIt overlaps with . Please draw a different path.');
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
      const layoutMode = settingModeSelect ? settingModeSelect.value : 'taxiway';
      const pathType = pathTypeFromLayoutMode(isPathLayoutMode(layoutMode) ? layoutMode : 'taxiway');
      const rawName = document.getElementById('taxiwayName').value.trim();
      const nameBase = rawName || (pathType === 'runway'
        ? 'Runway'
        : (pathType === 'runway_exit' ? 'Runway Exit TW' : ('Taxiway ' + (state.taxiways.length + 1))));
      const inputWidth = Number(document.getElementById('taxiwayWidth').value);
      const baseWidth = pathType === 'runway' ? 60 : 15;
      const widthVal = Math.max(10, Math.min(100, inputWidth || baseWidth));
      const modeVal = document.getElementById('taxiwayDirectionMode').value || 'both';
      const maxExitInput = document.getElementById('taxiwayMaxExitVel');
      const minExitInput = document.getElementById('taxiwayMinExitVel');
      const maxExitVelocity = (pathType === 'runway_exit' && maxExitInput)
        ? (function() {{ const mv = Number(maxExitInput.value); return isFinite(mv) && mv > 0 ? mv : null; }})()
        : null;
      const minExitVelocity = (pathType === 'runway_exit' && minExitInput)
        ? (function() {{
            const mv = Number(minExitInput.value);
            if (!isFinite(mv) || mv <= 0) return 15;
            if (maxExitVelocity != null && mv > maxExitVelocity) return maxExitVelocity;
            return mv;
          }})()
        : undefined;
      const minArrVelInput = document.getElementById('runwayMinArrVelocity');
      const minArrVelocity = (pathType === 'runway' && minArrVelInput)
        ? (function() {{
            const mv = Number(minArrVelInput.value);
            return (isFinite(mv) && mv > 0) ? Math.max(1, Math.min(150, mv)) : 15;
          }})()
        : undefined;
      const lineupEl = document.getElementById('runwayLineupDistM');
      const lineupDistM = (pathType === 'runway' && lineupEl)
        ? (function() {{ const x = Number(lineupEl.value); return (isFinite(x) && x >= 0) ? x : 0; }})()
        : undefined;
      const taxiway = {{ id: id(), name: nameBase, vertices: [], width: widthVal, direction: modeVal, pathType, maxExitVelocity, minExitVelocity, minArrVelocity, lineupDistM, avgMoveVelocity: (function() {{
        const el = document.getElementById('taxiwayAvgMoveVelocity');
        const v = el ? Number(el.value) : 10;
        return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
      }})() }};
      if (pathType !== 'runway') delete taxiway.minArrVelocity;
      if (pathType !== 'runway') delete taxiway.lineupDistM;
      if (pathType !== 'runway_exit') {{ delete taxiway.maxExitVelocity; delete taxiway.minExitVelocity; }}
      pushUndo();
      state.taxiways.push(taxiway);
      state.taxiwayDrawingId = taxiway.id;
      syncPanelFromState();
      if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
    }});
    const runwayDirInPaneEl = document.getElementById('runwayDirectionInTaxiwayPane');
    if (runwayDirInPaneEl) runwayDirInPaneEl.addEventListener('change', function() {{
      if (state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.obj.pathType === 'runway') {{
        state.selectedObject.obj.direction = this.value || 'both';
        updateObjectInfo();
        draw();
        if (typeof scene3d !== 'undefined' && scene3d) update3DScene();
      }}
    }});
    const btnPbbDrawEl = document.getElementById('btnPbbDraw');
    if (btnPbbDrawEl) btnPbbDrawEl.addEventListener('click', function() {{
      toggleLayoutDrawMode('pbbDrawing', 'previewPbb', null);
    }});
    const btnRemoteDrawEl = document.getElementById('btnRemoteDraw');
    if (btnRemoteDrawEl) btnRemoteDrawEl.addEventListener('click', function() {{
      toggleLayoutDrawMode('remoteDrawing', 'previewRemote', null);
    }});
    const btnApronDrawEl = document.getElementById('btnApronLinkDraw');
    if (btnApronDrawEl) btnApronDrawEl.addEventListener('click', function() {{
      toggleLayoutDrawMode('apronLinkDrawing', null, 'apronLinkTemp');
    }});

    (function setupRightPanelDragResize() {{
      if (!panel || !panelToggle) return;
      const rootStyle = () => getComputedStyle(document.documentElement);
      function readPxVar(name, fallback) {{
        const v = parseFloat(rootStyle().getPropertyValue(name));
        return Number.isFinite(v) ? v : fallback;
      }}
      function readLenVar(name, fallback) {{
        const t = (rootStyle().getPropertyValue(name) || '').trim();
        return t || fallback;
      }}
      function parseCssLenToPx(s, vwBase) {{
        const str = String(s || '').trim().toLowerCase();
        const n = parseFloat(str);
        if (!Number.isFinite(n)) return vwBase * 0.5;
        if (str.endsWith('vw')) return (n / 100) * vwBase;
        if (str.endsWith('vh')) return (n / 100) * (typeof window !== 'undefined' ? window.innerHeight : 800);
        if (str.endsWith('%')) return (n / 100) * vwBase;
        if (str.endsWith('px')) return n;
        return n;
      }}
      function maxPanelPx() {{
        const m = readPxVar('--style-right-panel-resize-viewport-margin', 8);
        return Math.max(120, window.innerWidth - m);
      }}
      function collapsedPx() {{ return readPxVar('--style-right-panel-resize-collapsed', 44); }}
      function collapseBelowPx() {{ return readPxVar('--style-right-panel-resize-collapse-below', 96); }}
      function minExpandedPx() {{ return readPxVar('--style-right-panel-resize-min-expanded', 220); }}
      let lastExpandedWidthPx = Math.round(parseCssLenToPx(readLenVar('--style-right-panel-width-full', '50vw'), window.innerWidth));
      lastExpandedWidthPx = Math.min(maxPanelPx(), Math.max(minExpandedPx(), lastExpandedWidthPx));
      function syncToolbar(px) {{
        document.documentElement.style.setProperty('--layout-toolbar-right', Math.round(px) + 'px');
      }}
      function applyCollapsed() {{
        panel.classList.add('collapsed');
        panel.style.width = '';
        syncToolbar(collapsedPx());
        panelToggle.textContent = '▶';
      }}
      function applyExpandedWidthPx(px) {{
        const cap = maxPanelPx();
        let w = Math.min(cap, Math.round(px));
        w = Math.max(minExpandedPx(), w);
        panel.classList.remove('collapsed');
        panel.style.width = w + 'px';
        lastExpandedWidthPx = w;
        syncToolbar(w);
        panelToggle.textContent = '◀';
      }}
      function applyDragWidthPx(rawPx) {{
        const cap = maxPanelPx();
        const c0 = collapsedPx();
        const below = collapseBelowPx();
        let w = Math.min(cap, Math.max(c0, Math.round(rawPx)));
        if (w < below) {{
          panel.classList.add('collapsed');
          panel.style.width = '';
          syncToolbar(c0);
          panelToggle.textContent = '▶';
          return;
        }}
        panel.classList.remove('collapsed');
        panel.style.width = w + 'px';
        syncToolbar(w);
        panelToggle.textContent = '◀';
      }}
      function finishDragWidthPx(rawPx) {{
        const below = collapseBelowPx();
        const cap = maxPanelPx();
        let w = Math.min(cap, Math.max(collapsedPx(), Math.round(rawPx)));
        if (w < below) {{
          applyCollapsed();
          return;
        }}
        w = Math.min(cap, Math.max(minExpandedPx(), w));
        applyExpandedWidthPx(w);
      }}
      applyExpandedWidthPx(lastExpandedWidthPx);
      let dragStartClientX = 0;
      let dragStartWidth = 0;
      let lastMoveClientX = 0;
      let dragMoved = false;
      let resizePointerActive = false;
      let suppressToggleClick = false;
      const CLICK_MAX_MOVE = 6;
      function onResizeWindow() {{
        if (panel.classList.contains('collapsed')) {{
          syncToolbar(collapsedPx());
          return;
        }}
        const rw = panel.getBoundingClientRect().width;
        const cap = maxPanelPx();
        if (rw > cap) applyExpandedWidthPx(cap);
        else syncToolbar(rw);
      }}
      window.addEventListener('resize', onResizeWindow);
      panelToggle.addEventListener('click', function(ev) {{
        if (suppressToggleClick) {{
          ev.preventDefault();
          ev.stopImmediatePropagation();
          suppressToggleClick = false;
        }}
      }}, true);
      panelToggle.addEventListener('pointerdown', function(ev) {{
        if (ev.pointerType === 'mouse' && ev.button !== 0) return;
        ev.preventDefault();
        dragMoved = false;
        resizePointerActive = true;
        dragStartClientX = ev.clientX;
        lastMoveClientX = ev.clientX;
        const c0 = collapsedPx();
        dragStartWidth = panel.classList.contains('collapsed') ? c0 : panel.getBoundingClientRect().width;
        panel.classList.add('panel-resize-dragging');
        try {{ panelToggle.setPointerCapture(ev.pointerId); }} catch (e) {{}}
      }});
      panelToggle.addEventListener('pointermove', function(ev) {{
        if (!resizePointerActive) return;
        if (Math.abs(ev.clientX - dragStartClientX) > CLICK_MAX_MOVE) dragMoved = true;
        lastMoveClientX = ev.clientX;
        const w = dragStartWidth + (dragStartClientX - ev.clientX);
        applyDragWidthPx(w);
      }});
      function endPointerDrag(ev) {{
        if (!resizePointerActive) return;
        resizePointerActive = false;
        panel.classList.remove('panel-resize-dragging');
        try {{ if (ev && ev.pointerId != null) panelToggle.releasePointerCapture(ev.pointerId); }} catch (e) {{}}
        if (!dragMoved) {{
          if (panel.classList.contains('collapsed')) {{
            applyExpandedWidthPx(lastExpandedWidthPx);
          }} else {{
            lastExpandedWidthPx = Math.max(minExpandedPx(), Math.min(maxPanelPx(), panel.getBoundingClientRect().width));
            applyCollapsed();
          }}
          dragMoved = false;
          return;
        }}
        suppressToggleClick = true;
        const endX = ev && Number.isFinite(ev.clientX) ? ev.clientX : lastMoveClientX;
        const w = dragStartWidth + (dragStartClientX - endX);
        finishDragWidthPx(w);
        dragMoved = false;
      }}
      panelToggle.addEventListener('pointerup', endPointerDrag);
      panelToggle.addEventListener('pointercancel', endPointerDrag);
      panelToggle.addEventListener('lostpointercapture', function(ev) {{
        if (resizePointerActive) endPointerDrag(ev);
      }});
    }})();

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
          const baseName = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(idx + 1).padStart(3, '0'));
          let allowedLabel = 'All (by proximity)';
          if (Array.isArray(st.allowedTerminals) && st.allowedTerminals.length) {{
            const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {{ return {{
              id: t.id,
              name: (t.name || '').trim() || 'Terminal'
            }}; }});
            const names = st.allowedTerminals.map(function(id) {{
              const tt = terms.find(function(t) {{ return t.id === id; }});
              return tt ? tt.name : id;
            }});
            if (names.length) allowedLabel = names.join(', ');
          }}
          items.push({{
            type: 'remote',
            id: st.id,
            title: uniqueTitle('Remote stand | ' + baseName),
            tag: 'Category ' + (st.category || 'C'),
            details:
              'Category: ' + (st.category || '—') +
              '<br>Cell: (' + st.col + ',' + st.row + ')' +
              '<br>available terminals: ' + allowedLabel
          }});
        }});
      }} else if (isPathLayoutMode(mode)) {{
        const wantPt = pathTypeFromLayoutMode(mode);
        state.taxiways.forEach((tw, idx) => {{
          if (seen['taxiway_' + tw.id]) return;
          const pt = tw.pathType || 'taxiway';
          if (pt !== wantPt) return;
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
          const heading = tw.pathType === 'runway' ? 'Runway' : (tw.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
          const avgVel = (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) ? tw.avgMoveVelocity : 10;
          const maxExit = (tw.pathType === 'runway_exit' && typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0) ? tw.maxExitVelocity : null;
          const minExit = (tw.pathType === 'runway_exit' && typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
            ? (maxExit != null && tw.minExitVelocity > maxExit ? maxExit : tw.minExitVelocity)
            : null;
          const minArrDisplay = tw.pathType === 'runway'
            ? ((typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
              ? Math.max(1, Math.min(150, tw.minArrVelocity))
              : 15)
            : null;
          items.push({{
            type: 'taxiway',
            id: tw.id,
            title: uniqueTitle(heading + ' | ' + baseName),
            tag: dirLabel,
            details:
              'Length: ' + lengthM.toFixed(0) + ' m' +
              '<br>Points: ' + tw.vertices.length +
              '<br>Width: ' + widthVal + ' m' +
              (maxExit != null ? '<br>Max exit velocity: ' + maxExit + ' m/s' : '') +
              (minExit != null ? '<br>Min exit velocity: ' + minExit + ' m/s' : '') +
              (minArrDisplay != null ? '<br>Min arr velocity: ' + minArrDisplay + ' m/s' : '') +
              (tw.pathType === 'runway' ? '<br>Line up: ' + getEffectiveRunwayLineupDistM(tw) + ' m (start→end)' : '') +
              (tw.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
              '<br>Start point: ' + startStr +
              '<br>End point: ' + endStr
          }});
        }});
      }} else if (mode === 'apronTaxiway') {{
        state.apronLinks.forEach((lk, idx) => {{
          if (seen['apron_' + lk.id]) return;
          seen['apron_' + lk.id] = true;
          const stand = (state.pbbStands.find(p => p.id === lk.pbbId) ||
                         state.remoteStands.find(st => st.id === lk.pbbId));
          const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
          const title = 'Link ' + (idx + 1);
          const standLabel = stand && stand.name ? stand.name : lk.pbbId;
          const details = 'Stand: ' + standLabel +
            ', Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId);
          items.push({{
            type: 'apronLink',
            id: lk.id,
            title: uniqueTitle('Apron–Taxiway | ' + title),
            tag: 'Apron–Taxiway',
            details
          }});
        }});
      }} else if (mode === 'edge') {{
        rebuildDerivedGraphEdges();
        (state.derivedGraphEdges || []).forEach(function(ed) {{
          items.push({{
            type: 'layoutEdge',
            id: ed.id,
            title: 'Edge | ' + ed.label,
            tag: 'Graph',
            details:
              'Length (graph): ' + Math.round(ed.dist) +
              '<br>Pixel segment: (' + ed.x1.toFixed(0) + ', ' + ed.y1.toFixed(0) + ') → (' + ed.x2.toFixed(0) + ', ' + ed.y2.toFixed(0) + ')' +
              '<br>Node indices: ' + ed.fromIdx + ' → ' + ed.toIdx,
            noDelete: true
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
            '<button type="button" class="obj-item-delete" title="Delete"' + (it.noDelete ? ' style="display:none" tabindex="-1" aria-hidden="true"' : '') + '>×</button>' +
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
          removeLayoutObjectFromState(type, id);
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
          if (typ === 'layoutEdge') rebuildDerivedGraphEdges();
          const obj = findLayoutObjectByListType(typ, idr);
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
        else if (state.selectedObject.type === 'remote') {{
          let allowedLabel = 'All (by proximity)';
          if (Array.isArray(o.allowedTerminals) && o.allowedTerminals.length) {{
            const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {{ return {{
              id: t.id,
              name: (t.name || '').trim() || 'Terminal'
            }}; }});
            const names = o.allowedTerminals.map(function(id) {{
              const tt = terms.find(function(t) {{ return t.id === id; }});
              return tt ? tt.name : id;
            }});
            if (names.length) allowedLabel = names.join(', ');
          }}
          objectInfoEl.innerHTML =
            '<strong>Remote stand</strong>' +
            '<br>Name: ' + (o.name || '—') +
            '<br>Category: ' + (o.category || '—') +
            '<br>Cell: (' + o.col + ',' + o.row + ')' +
            '<br>available terminals: ' + allowedLabel;
        }}
        else if (state.selectedObject.type === 'taxiway') {{
          const dirVal = getTaxiwayDirection(o);
          const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter Clockwise' : 'Both');
          const heading = o.pathType === 'runway' ? 'Runway' : (o.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
          const ser = serializeTaxiwayWithEndpoints(o);
          const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
          const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
          const avgVel = (typeof o.avgMoveVelocity === 'number' && isFinite(o.avgMoveVelocity) && o.avgMoveVelocity > 0) ? o.avgMoveVelocity : 10;
          const minArr = (o.pathType === 'runway')
            ? ((typeof o.minArrVelocity === 'number' && isFinite(o.minArrVelocity) && o.minArrVelocity > 0) ? Math.max(1, Math.min(150, o.minArrVelocity)) : 15)
            : null;
          const lineupStr = (o.pathType === 'runway') ? (String(getEffectiveRunwayLineupDistM(o)) + ' m (from start toward end)') : '';
          const maxEx = (o.pathType === 'runway_exit' && typeof o.maxExitVelocity === 'number' && isFinite(o.maxExitVelocity) && o.maxExitVelocity > 0) ? o.maxExitVelocity : null;
          const minEx = (o.pathType === 'runway_exit' && typeof o.minExitVelocity === 'number' && isFinite(o.minExitVelocity) && o.minExitVelocity > 0) ? o.minExitVelocity : null;
          objectInfoEl.innerHTML = '<strong>' + heading + '</strong><br>Name: ' + (o.name || '—') +
            '<br>Direction: ' + dirLabel +
            '<br>Width: ' + (o.width != null ? o.width : 23) + ' m' +
            (o.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
            (minArr != null ? '<br>Min arr velocity: ' + minArr + ' m/s' : '') +
            (o.pathType === 'runway' ? '<br>Line up: ' + lineupStr : '') +
            (maxEx != null ? '<br>Max exit velocity: ' + maxEx + ' m/s' : '') +
            (minEx != null ? '<br>Min exit velocity: ' + minEx + ' m/s' : '') +
            '<br>Points: ' + (o.vertices ? o.vertices.length : 0) +
            '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
        }} else if (state.selectedObject.type === 'layoutEdge') {{
          const ed = state.selectedObject.obj;
          objectInfoEl.innerHTML =
            '<strong>Edge (derived)</strong><br>' +
            'Name: ' + (ed && ed.label ? ed.label : '—') +
            '<br>Graph length: ' + (ed && ed.dist != null ? Math.round(ed.dist) : '—') +
            '<br>Nodes: ' + (ed ? ed.fromIdx + ' → ' + ed.toIdx : '—') +
            '<br>Segment (px): (' + (ed ? ed.x1.toFixed(0) : '—') + ', ' + (ed ? ed.y1.toFixed(0) : '—') + ') → (' + (ed ? ed.x2.toFixed(0) : '—') + ', ' + (ed ? ed.y2.toFixed(0) : '—') + ')';
        }} else if (state.selectedObject.type === 'flight') {{
          const dir = o.arrDep === 'Dep' ? 'Departure' : 'Arrival';
          const sibt = formatMinutesToHHMMSS(o.sibtMin_d != null ? o.sibtMin_d : (o.timeMin != null ? o.timeMin : 0));
          const sobt = formatMinutesToHHMMSS(o.sobtMin_d != null ? o.sobtMin_d : ((o.timeMin != null ? o.timeMin : 0) + (o.dwellMin != null ? o.dwellMin : 0)));
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
            '<br>Reg: ' + (o.reg || '—') +
            '<br>Airline Code: ' + (o.airlineCode || '—') + ' &nbsp; Flight Number: ' + (o.flightNumber || '—') +
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
      if (!container || !canvas || !ctx) return;
      const rect = container.getBoundingClientRect();
      const w = Math.max(1, Number(rect.width) || 0);
      const h = Math.max(1, Number(rect.height) || 0);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      safeDraw();
    }}

    function drawGrid() {{
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = GRID_VIEW_BG;
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
        ctx.strokeStyle = (c % GRID_MAJOR === 0) ? 'rgba(255,255,255,0.35)' : 'rgba(140,140,140,0.2)';
        ctx.lineWidth = (c % GRID_MAJOR === 0) ? 1.2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, maxY);
        ctx.stroke();
      }}
      for (let r = 0; r <= GRID_ROWS; r++) {{
        const y = r * CELL_SIZE;
        ctx.strokeStyle = (r % GRID_MAJOR === 0) ? 'rgba(255,255,255,0.35)' : 'rgba(140,140,140,0.2)';
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
        ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : (_canvas2dStyle.terminalStrokeDefault || '#38bdf8');
        ctx.fillStyle = selected ? c2dObjectSelectedFill() : (_canvas2dStyle.terminalFillDefault || 'rgba(56,189,248,0.12)');
        ctx.beginPath();
        for (let i = 0; i < term.vertices.length; i++) {{
          const [x,y] = cellToPixel(term.vertices[i].col, term.vertices[i].row);
          if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }}
        if (term.closed) {{ ctx.closePath(); ctx.fill(); }}
        if (selected) {{
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
        }}
        ctx.stroke();
        if (selected) ctx.restore();
        // Selected terminals are highlighted once more with a dotted contour
        if (selected) {{
          ctx.save();
          ctx.setLineDash([8, 6]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = c2dObjectSelectedDashStroke();
          ctx.beginPath();
          for (let i = 0; i < term.vertices.length; i++) {{
            const [x,y] = cellToPixel(term.vertices[i].col, term.vertices[i].row);
            if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
          }}
          if (term.closed) ctx.closePath();
          ctx.stroke();
          ctx.restore();
        }}
        // Show terminal name centered in terminal (height eliminate)
        if (term.closed && term.vertices.length > 0) {{
          let cx = 0, cy = 0;
          term.vertices.forEach(v => {{
            const [px, py] = cellToPixel(v.col, v.row);
            cx += px; cy += py;
          }});
          cx /= term.vertices.length;
          cy /= term.vertices.length;
          const label = term.name || term.id || 'Terminal';
          ctx.fillStyle = _canvas2dStyle.terminalLabelFill || 'rgba(56,189,248,0.95)';
          ctx.font = '12px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, cx, cy);
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
        const x1 = Number(pbb.x1), y1 = Number(pbb.y1), x2 = Number(pbb.x2), y2 = Number(pbb.y2);
        if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;
        const endSize = getStandSizeMeters(pbb.category || 'C');
        const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#f97316';
        ctx.lineWidth = sel ? 4 : 3;
        if (sel) {{
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }}
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        if (sel) ctx.restore();
        const ex = x2, ey = y2;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.fillStyle = sel ? c2dObjectSelectedFill() : 'rgba(22,163,74,0.18)';
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#22c55e';
        ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.save();
        ctx.translate(ex, ey);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
        ctx.fill();
        if (sel) {{
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }}
        ctx.stroke();
        if (sel) ctx.restore();
        if (sel) {{
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = c2dObjectSelectedDashStroke();
          ctx.beginPath();
          ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
          ctx.stroke();
          ctx.restore();
        }}
        // Parking lot label: "Category / Name" form (If there is no name, there is a number)
        const nameRaw = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : String(state.pbbStands.indexOf(pbb) + 1);
        const label = (pbb.category || 'C') + ' / ' + nameRaw;
        const pad = 3;
        const tx = endSize / 2 - pad;
        const ty = -endSize / 2 + pad;
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '8px system-ui';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(String(label), tx, ty);
        ctx.restore();
      }});
      ctx.restore();
    }}

    function drawRemoteStands() {{
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.translate(state.panX, state.panY);
      ctx.scale(state.scale, state.scale);
      const mode = settingModeSelect ? settingModeSelect.value : 'grid';
      state.remoteStands.forEach(st => {{
        const [cx,cy] = cellToPixel(st.col, st.row);
        const size = getStandSizeMeters(st.category || 'C');
        const sel = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
        ctx.fillStyle = sel ? c2dObjectSelectedFill() : 'rgba(22,163,74,0.18)';
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#22c55e';
        ctx.lineWidth = sel ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.rect(cx-size/2, cy-size/2, size, size);
        ctx.fill();
        if (sel) {{
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }}
        ctx.stroke();
        if (sel) ctx.restore();
        if (sel) {{
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.lineWidth = 2;
          ctx.strokeStyle = c2dObjectSelectedDashStroke();
          ctx.beginPath();
          ctx.rect(cx-size/2, cy-size/2, size, size);
          ctx.stroke();
          ctx.restore();
        }}
        // Apron Taxiway Reference point for link: Remote stand Show small dot in center
        if (mode === 'apronTaxiway') {{
          ctx.save();
          ctx.fillStyle = sel ? '#f97316' : '#e5e7eb';
          ctx.beginPath();
          ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }}
        // Remote stand label: "Category / Name" form (Default if no name Rxxx) - Top left placement
        const nameRaw = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(state.remoteStands.indexOf(st) + 1).padStart(3, '0'));
        const label = (st.category || 'C') + ' / ' + nameRaw;
        ctx.fillStyle = '#bbf7d0';
        ctx.font = '8px system-ui';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        const labelOffset = 2;
        ctx.fillText(label, cx + size/2 - labelOffset, cy - size/2 + labelOffset);
      }});
      ctx.restore();
    }}

    function renderRunwaySeparation() {{
      const panel = document.getElementById('rwySepPanel');
      if (!panel) return;
      const runways = (state.taxiways || []).filter(tw => tw.pathType === 'runway');
      if (!runways.length) {{
        panel.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No runway paths. Layout Mode <strong>Runway</strong>Draw the runway polyline first with.</div>';
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

      // default tab name 'No Name', and no separate timeline graph is provided..
      const activeSub = 'noname';
      html += '<div class="layout-save-load-tabs" style="margin-top:8px;">';
      html += '<button type="button" class="layout-save-load-tab rwysep-subtab-btn active" data-subtab="noname">No Name</button>';
      html += '</div>';

      // --- Subtab: No Name (Maintain input form) ---
      html += '<div id="rwysep-subtab-input" style="">';
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

      // ROT: Arr→Dep Shown only in combination
      if (seq === 'ARR→DEP') {{
        html += '<div class="rwysep-block">';
        html += '<div class="rwysep-label">ROT (Runway Occupancy Time, sec)</div>';

        // color legend + filled ROT count
        const totalRot = cats.length;
        let filledRot = 0;
        cats.forEach(c => {{
          const val = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
          if (val !== '' && val != null) filledRot += 1;
        }});
        html += rsepLegendHtml(filledRot, totalRot);

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
          html += rsepLegendHtml(filled, total);

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
          html += rsepLegendHtml(filled, total);

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
      // Up to about 12 Reg Only one row appears on the screen, anything beyond that scrolls vertically
      html += '<div id="rwySepTimeWrap" style="width:100%;background:#020617;border-radius:8px;border:1px solid #1f2937;position:relative;overflow-x:auto;overflow-y:auto;margin-top:4px;max-height:calc(40px * 12 + 80px);"></div>';
      html += '<div style="font-size:9px;color:#9ca3af;margin-top:4px;">';
      html += 'Y: Reg Number · X: Time · Bars = S-series (SLDT–STOT) · Lines = E-series (ELDT–ETOT)';
      html += '</div></div>';
      html += '</div>'; // end subtab timeline

      panel.innerHTML = html;

      // draw timeline only when timeline subtab is visible (Apron Gantt style, Reg × Time)
      function drawRwySeparationTimeline() {{
        if (state.activeRwySepSubtab && state.activeRwySepSubtab !== 'timeline') return;
        const wrap = panel.querySelector('#rwySepTimeWrap');
        if (!wrap) return;

        const _prepRwy = state.flights.slice();
        _prepRwy.forEach(ff => {{
          if (typeof getPathForFlight === 'function') getPathForFlight(ff);
          if (typeof getPathForFlightDeparture === 'function') getPathForFlightDeparture(ff);
          if (ff.noWayArr || ff.noWayDep) ff.timeline = null;
        }});
        if (typeof ensureArrRetRotSampled === 'function') ensureArrRetRotSampled(_prepRwy, false);
        if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
        const allData = typeof computeSeparationAdjustedTimes === 'function' ? computeSeparationAdjustedTimes() : null;
        // Runway separation(Eline) After recalculation Flight Schedule The table is also up to date ELDT/ETOTImmediately re-render to reflect
        if (typeof renderFlightList === 'function') renderFlightList();
        const data = allData && active && active.id != null ? allData[active.id] : null;
        if (!data || !data.events || !data.events.length) {{
          wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
          return;
        }}

        // Flight not really SLDT/STOT/ELDT/ETOT collect (line = one Reg)
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

        // time axis: min(SLDT) - pad, max(ETOT) + pad (algorithm.timeAxis.runwaySepTimelinePadMin)
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
        let baseMinT = Math.max(0, minT0 - RWY_SEP_TIMELINE_PAD_MIN);
        let baseMaxT = maxT0 + RWY_SEP_TIMELINE_PAD_MIN;
        if (baseMaxT <= baseMinT) baseMaxT = baseMinT + 60;
        const baseSpan = baseMaxT - baseMinT;
        const zoom = (state.rwySepTimeZoom && state.rwySepTimeZoom > 1) ? state.rwySepTimeZoom : 1;
        const span = baseSpan;
        const minT = baseMinT;
        const maxT = baseMaxT;

        lanes.sort((a, b) => {{
          const ta = (a.sldt ?? a.stot ?? a.eldt ?? a.etot ?? 0);
          const tb = (b.sldt ?? b.stot ?? b.eldt ?? b.etot ?? 0);
          return ta - tb;
        }});

        // Time axis scale position (Runway Timeline common to all) - Only display up to 6 items on any screen
        const tickPositions = [];
        const axisStep = span <= 60 ? TICK_STEP_SPAN_LE60 : (span <= 240 ? TICK_STEP_SPAN_LE240 : TICK_STEP_ELSE);
        let tt = Math.floor(minT / axisStep) * axisStep;
        while (tt <= maxT) {{
          const leftPct = ((tt - baseMinT) / baseSpan) * 100 * zoom;
          const label = formatMinToHM(tt);
          tickPositions.push({{ leftPct, label }});
          tt += axisStep;
        }}
        if (tickPositions.length > MAX_TICKS_SHOWN) {{
          const stepTicks = Math.ceil(tickPositions.length / MAX_TICKS_SHOWN);
          const reduced = [];
          for (let i = 0; i < tickPositions.length; i += stepTicks) {{
            reduced.push(tickPositions[i]);
          }}
          const last = tickPositions[tickPositions.length - 1];
          if (reduced[reduced.length - 1] !== last) reduced.push(last);
          tickPositions.length = 0;
          Array.prototype.push.apply(tickPositions, reduced);
        }}

        // top S/E Data for triangle timeline
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
            // top SMarkers for Series Triangles(start/end)
            sMarkers.push({{ t: sStart, leftPct, type: 'start' }});
            sMarkers.push({{ t: sEnd, leftPct: rightPct, type: 'end' }});
            // S-series: thin blue bar + start/exit triangle (placed at the top)
            blocks +=
              '<div class="rwysep-line-s" style="' +
                'left:' + leftPct + '%;' +
                'width:' + widthPct + '%;' +
              '"></div>' +
              // Starting point: downward triangle (towards the bar)
              '<div class="rwysep-tri" style="' +
                'top:20%;' +
                'left:' + leftPct + '%;' +
                'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
              '"></div>' +
              // End point: upward triangle
              '<div class="rwysep-tri" style="' +
                'top:20%;' +
                'left:' + rightPct + '%;' +
                'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
              '"></div>';
          }}
          if (eStart != null && eEnd != null && span > 0) {{
            const e1 = Math.max(eStart, baseMinT);
            const e2 = Math.min(eEnd, baseMaxT);
            if (e2 <= e1) return;
            const leftPct2 = ((e1 - baseMinT) / baseSpan) * 100 * zoom;
            const widthPct2 = Math.max(0.5, ((e2 - e1) / baseSpan) * 100 * zoom);
            const rightPct2 = leftPct2 + widthPct2;
            // top EMarkers for Series Triangles(start/end)
            eMarkers.push({{ t: eStart, leftPct: leftPct2, type: 'start' }});
            eMarkers.push({{ t: eEnd, leftPct: rightPct2, type: 'end' }});
            // E-series: thin orange bar + start/exit triangle (placed at the bottom)
            blocks +=
              '<div class="rwysep-line-e" style="' +
                'left:' + leftPct2 + '%;' +
                'width:' + widthPct2 + '%;' +
              '"></div>' +
              // Starting point: downward triangle
              '<div class="rwysep-tri" style="' +
                'top:54%;' +
                'left:' + leftPct2 + '%;' +
                'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
              '"></div>' +
              // End point: upward triangle
              '<div class="rwysep-tri" style="' +
                'top:54%;' +
                'left:' + rightPct2 + '%;' +
                'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
              '"></div>';
          }}

          const gridLines = tickPositions.map(tp =>
            '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>'
          ).join('');

          rows.push(
            '<div class="alloc-row">' +
              '<div class="alloc-row-label">' + escapeHtml(reg) + '</div>' +
              // Runway Separation TimelineIn each row background(track background color/outline)remove
              '<div class="alloc-row-track" style="background:transparent;border:none;">' + gridLines + blocks + '</div>' +
            '</div>'
          );
        }});

        // top S/E triangle lines HTML (chronological order)
        sMarkers.sort((a, b) => a.t - b.t);
        eMarkers.sort((a, b) => a.t - b.t);

        const sHeadMarks = sMarkers.map(m =>
          '<div class="rwysep-tri" style="' +
            'top:60%;' +
            'left:' + m.leftPct + '%;' +
            (m.type === 'start'
              ? 'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';'
              : 'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';') +
          '"></div>'
        ).join('');

        const eHeadMarks = eMarkers.map(m =>
          '<div class="rwysep-tri" style="' +
            'top:60%;' +
            'left:' + m.leftPct + '%;' +
            (m.type === 'start'
              ? 'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';'
              : 'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';') +
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

        // Time axis overlay (Apronsame style as, tickPositions reuse)
        const axisTicks = tickPositions.map(tp =>
          '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
            '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
          '</div>'
        );
        // Runway Separation TimelineEven in ApronUse the bottom time base overlay the same as
        const axisHtml =
          '<div class="alloc-time-axis-overlay">' +
            '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
          '</div>';

        // many RegIf present, wrap the rows while leaving the header intact to display vertical scrolling.
        const rowsHtml = '<div class="rwysep-rows">' + rows.join('') + '</div>';
        wrap.innerHTML = headHtml + rowsHtml + axisHtml;

        // Shift + Zoom on the time axis with the mouse wheel (Runway Timeline)
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

        // 수평 스크롤 cityto도 현재 time axis/Re-render so the background is recalculated
        if (!wrap._rwySepScrollBound) {{
          wrap._rwySepScrollBound = true;
          wrap.addEventListener('scroll', function() {{
            if (wrap._rwySepScrollRecalc) return;
            // Redraw the entire timeline while preserving the current scroll position.
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
        if (sel) {{
          ctx.strokeStyle = c2dObjectSelectedStroke();
          ctx.fillStyle = c2dObjectSelectedFill();
        }} else if (isRunwayPath || isRunwayExit) {{
          // Runway and Runway Taxiway: One tone darker gray
          ctx.strokeStyle = '#374151';
          ctx.fillStyle = 'rgba(31,41,55,0.32)';
        }} else {{
          // common Taxiway: brighter yellow color (Arrow colors remain separate)
          ctx.strokeStyle = drawing ? '#facc15' : '#fbbf24';
          ctx.fillStyle = 'rgba(251,191,36,0.18)';
        }}
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {{
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        if (tw.vertices.length >= 2) {{
          if (sel) {{
            ctx.save();
            ctx.shadowColor = c2dObjectSelectedGlow();
            ctx.shadowBlur = c2dObjectSelectedGlowBlur();
            ctx.stroke();
            ctx.restore();
          }} else ctx.stroke();
        }}
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : ((isRunwayPath || isRunwayExit) ? '#52525b' : '#facc15');
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
          ctx.strokeStyle = c2dObjectSelectedDashStroke();
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
          // Arrow: 10% zoom out, color #f5930b
          const arrLen = CELL_SIZE * 0.54;
          ctx.fillStyle = '#f5930b';
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
        if (isRunwayPath && tw.vertices.length >= 2) {{
          const rp = getRunwayPath(tw.id);
          if (rp && rp.pts.length >= 2) {{
            const lenPx = runwayPolylineLengthPx(rp.pts);
            const d = Math.min(Math.max(0, getEffectiveRunwayLineupDistM(tw)), lenPx);
            const lp = getRunwayPointAtDistance(tw.id, d);
            if (lp) {{
              ctx.save();
              ctx.fillStyle = '#dc2626';
              ctx.strokeStyle = '#450a0a';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.arc(lp[0], lp[1], 5, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();
              ctx.fillStyle = '#fecaca';
              ctx.font = 'bold 11px system-ui, sans-serif';
              ctx.textAlign = 'left';
              ctx.textBaseline = 'bottom';
              ctx.fillText('Line up', lp[0] + 7, lp[1] - 4);
              ctx.restore();
            }}
          }}
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
                ctx.strokeStyle = c2dObjectSelectedStroke();
                ctx.lineWidth = 1.5;
                ctx.stroke();
              }}
            }}
          }});
        }}
      }});
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
        const stand = (state.pbbStands.find(p => p.id === lk.pbbId) ||
                       state.remoteStands.find(st => st.id === lk.pbbId));
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        if (!stand || !tw || lk.tx == null || lk.ty == null) return;
        const [ax, ay] = (stand.x2 != null && stand.y2 != null)
          ? [stand.x2, stand.y2]
          : cellToPixel(stand.col || 0, stand.row || 0);
        const [bx, by] = [lk.tx, lk.ty];
        // Link line uses taxiway yellow tone
        ctx.strokeStyle = '#facc15';
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        if (state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lk.id) {{
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.lineWidth = 4;
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.strokeStyle = c2dObjectSelectedDashStroke();
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.restore();
          ctx.setLineDash([6,6]);
        }}
        // endpoint markers
        ctx.setLineDash([]);
        // Endpoint markers use same taxiway yellow
        ctx.fillStyle = '#facc15';
        ctx.beginPath(); ctx.arc(ax, ay, CELL_SIZE * 0.18, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(bx, by, CELL_SIZE * 0.18, 0, Math.PI*2); ctx.fill();
        ctx.setLineDash([6,6]);
      }});
      ctx.setLineDash([]);
      // temporary first endpoint marker
      if (state.apronLinkTemp) {{
        ctx.fillStyle = '#facc15';
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

    function safeDraw() {{ try {{ draw(); }} catch(e) {{}} }}
    function draw() {{
      if (!ctx || !canvas) return;
      drawGrid();
      drawTerminals();
      drawTaxiways();
      drawPBBs();
      drawRemoteStands();
      drawApronTaxiwayLinks();
      drawStandPreview();
      drawPathJunctions();
      drawSelectedLayoutEdge();
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
      if (type !== 'terminal' && type !== 'pbb' && type !== 'remote' && type !== 'taxiway' && type !== 'apronLink' && type !== 'flight') return;
      pushUndo();
      removeLayoutObjectFromState(type, id);
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
      if ((mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing)) return;
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
      let drewThisMove = false;
      if (state.dragVertex) {{
        const term = state.terminals.find(t => t.id === state.dragVertex.terminalId);
        if (term && term.vertices[state.dragVertex.index]) {{
          const v = term.vertices[state.dragVertex.index];
          v.col = col;
          v.row = row;
          safeDraw(); drewThisMove = true;
        }}
        return;
      }}
      if (state.dragTaxiwayVertex) {{
        const tw = state.taxiways.find(t => t.id === state.dragTaxiwayVertex.taxiwayId);
        if (tw && tw.vertices[state.dragTaxiwayVertex.index]) {{
          const v = tw.vertices[state.dragTaxiwayVertex.index];
          v.col = col;
          v.row = row;
          safeDraw(); drewThisMove = true;
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
          safeDraw(); drewThisMove = true;
        }}
      }}
      const mode = settingModeSelect.value;
      if (!state.isPanning && !state.dragVertex && mode === 'remote' && state.remoteDrawing) {{
        const category = document.getElementById('remoteCategory').value || 'C';
        const size = getStandSizeMeters(category);
        const [cx, cy] = cellToPixel(col, row);
        const bounds = getStandBoundsRect(cx, cy, size);
        const overlap = standOverlapsExisting(bounds);
        state.previewRemote = {{ col, row, overlap }};
        safeDraw(); drewThisMove = true;
      }} else if (!state.isPanning && !state.dragVertex && mode === 'pbb' && state.pbbDrawing) {{
        let bestEdge = null, bestD2 = Infinity;
        state.terminals.forEach(t => {{
          if (!t.closed || t.vertices.length < 2) return;
          for (let i = 0; i < t.vertices.length; i++) {{
            const v1 = t.vertices[i], v2 = t.vertices[(i+1) % t.vertices.length];
            const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
            const near = closestPointOnSegment(p1, p2, [wx, wy]);
            if (near) {{
              const d2 = dist2(near, [wx, wy]);
              if (d2 < bestD2) {{ bestD2 = d2; bestEdge = {{ near, p1, p2 }}; }}
            }}
          }}
        }});
        const maxD2 = (CELL_SIZE*1.0)**2;
        if (bestEdge && bestD2 < maxD2) {{
          const nearPt = bestEdge.near;
          const ex = (nearPt && nearPt[0] != null) ? nearPt[0] : 0;
          const ey = (nearPt && nearPt[1] != null) ? nearPt[1] : 0;
          const [x1,y1]=bestEdge.p1, [x2,y2]=bestEdge.p2;
          let nx = -(y2-y1), ny = x2-x1;
          const len = Math.hypot(nx,ny) || 1; nx /= len; ny /= len;
          const toClickX = wx - ex, toClickY = wy - ey;
          if (nx * toClickX + ny * toClickY < 0) {{ nx *= -1; ny *= -1; }}
          const category = document.getElementById('standCategory').value || 'C';
          const standSize = getStandSizeMeters(category);
          const minLen = standSize / 2 + 3;
          const lenCells = parseInt(document.getElementById('pbbLength').value || '2', 10);
          const lenPx = Math.max(lenCells * CELL_SIZE * PBB_PREVIEW_LEN_CF, minLen);
          const px2 = ex + nx * lenPx, py2 = ey + ny * lenPx;
          const preview = {{ x1: ex, y1: ey, x2: px2, y2: py2, category }};
          const overlap = pbbStandOverlapsExisting(preview);
          state.previewPbb = {{ x1: ex, y1: ey, x2: px2, y2: py2, category: preview.category, overlap }};
          safeDraw(); drewThisMove = true;
        }} else {{
          if (state.previewPbb) {{ state.previewPbb = null; safeDraw(); drewThisMove = true; }}
        }}
      }} else {{
        let clearedPreview = false;
        if (state.previewRemote) {{ state.previewRemote = null; clearedPreview = true; }}
        if (state.previewPbb) {{ state.previewPbb = null; clearedPreview = true; }}
        if (clearedPreview) {{ safeDraw(); drewThisMove = true; }}
      }}
      // Object name tooltip on hover (Grid); flight tooltip when simulation result and near aircraft
      if (flightTooltip && !state.isPanning) {{
        const hit = hitTest(wx, wy);
        if (hit && hit.obj) {{
          const name = (hit.obj.name != null && String(hit.obj.name).trim()) ? String(hit.obj.name).trim() : (hit.type === 'terminal' ? 'Terminal' : hit.type === 'pbb' ? 'Contact Stand' : hit.type === 'remote' ? 'Remote Stand' : hit.type === 'taxiway' ? (hit.obj.name || 'Path') : hit.type);
          flightTooltip.style.display = 'block';
          flightTooltip.textContent = name;
          flightTooltip.style.left = (ev.clientX + 12) + 'px';
          flightTooltip.style.top = (ev.clientY + 12) + 'px';
        }} else if (state.hasSimulationResult) {{
          let bestFlight = null;
          let bestD2 = (CELL_SIZE * FLIGHT_TOOLTIP_CF) ** 2;
          const tSec = state.simTimeSec;
          state.flights.forEach(f => {{
            const pose = getFlightPoseAtTime(f, tSec);
            if (!pose || !f.reg) return;
            const dx = pose.x - wx;
            const dy = pose.y - wy;
            const d2 = dx*dx + dy*dy;
            if (d2 < bestD2) {{ bestD2 = d2; bestFlight = f; }}
          }});
          if (bestFlight && bestFlight.reg) {{
            flightTooltip.style.display = 'block';
            flightTooltip.textContent = bestFlight.reg;
            flightTooltip.style.left = (ev.clientX + 12) + 'px';
            flightTooltip.style.top = (ev.clientY + 12) + 'px';
          }} else {{
            flightTooltip.style.display = 'none';
          }}
        }} else {{
          flightTooltip.style.display = 'none';
        }}
      }}
      if (hoverChanged && !drewThisMove) {{ safeDraw(); drewThisMove = true; }}
    }});
    container.addEventListener('mouseleave', function() {{
      state.dragStart = null;
      state.isPanning = false;
      state.hoverCell = null;
      state.previewPbb = null;
      state.previewRemote = null;
      safeDraw();
    }});
    function hitTestPbbEnd(wx, wy) {{
      const maxD2 = (CELL_SIZE * HIT_PBB_END_CF) ** 2;
      const cands = [];
      state.pbbStands.forEach(pbb => {{
        cands.push({{ id: pbb.id, kind: 'pbb', x: pbb.x2, y: pbb.y2 }});
      }});
      state.remoteStands.forEach(st => {{
        const [cx, cy] = cellToPixel(st.col, st.row);
        cands.push({{ id: st.id, kind: 'remote', x: cx, y: cy }});
      }});
      const best = findNearestItem(cands, c => [c.x, c.y], wx, wy, maxD2);
      return best || null;
    }}

    function hitTestAnyTaxiwayVertex(wx, wy) {{
      // For Apron Taxiway links: allow connecting to any point along a taxiway polyline
      const click = [wx, wy];
      const maxD2 = (CELL_SIZE * TRY_PBB_MAX_EDGE_CF) ** 2;
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
      state.isPanning = false;
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
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      const mode = settingModeSelect.value;
      const inStandDrawingMode = (mode === 'pbb' && state.pbbDrawing) || (mode === 'remote' && state.remoteDrawing);
      if (!state.dragStart && !inStandDrawingMode) {{ state.dragStart = null; return; }}
      if (handlePbbOrRemoteMouseUp2D(mode, wx, wy)) {{
        state.dragStart = null;
        return;
      }}
      if (!state.dragStart) return;
      if (!state.isPanning) {{
        const mode = settingModeSelect.value;
        if (mode === 'edge') {{
          rebuildDerivedGraphEdges();
          const eh = hitTestLayoutGraphEdge(wx, wy);
          if (eh) {{
            state.selectedObject = {{ type: 'layoutEdge', id: eh.id, obj: eh }};
          }} else {{
            state.selectedObject = null;
          }}
          syncPanelFromState();
          updateObjectInfo();
          draw();
          state.dragStart = null;
          return;
        }}
        const hit = hitTest(wx, wy);
        if (mode === 'apronTaxiway' && state.apronLinkDrawing) {{
          const pbbHit = hitTestPbbEnd(wx, wy);
          const twHit = hitTestAnyTaxiwayVertex(wx, wy);
          const endpoint = pbbHit ? {{ kind: pbbHit.kind, standId: pbbHit.id, x: pbbHit.x, y: pbbHit.y }} :
                            (twHit ? {{ kind: 'taxiway', taxiwayId: twHit.taxiwayId, x: twHit.x, y: twHit.y }} : null);
          if (endpoint) {{
            if (!state.apronLinkTemp) {{
              state.apronLinkTemp = endpoint;
            }} else {{
              const first = state.apronLinkTemp;
              if (first.kind !== endpoint.kind) {{
                let standId, taxiwayId, tx, ty;
                if (first.kind === 'taxiway') {{
                  taxiwayId = first.taxiwayId;
                  standId = endpoint.standId;
                  tx = first.x;
                  ty = first.y;
                }} else {{
                  taxiwayId = endpoint.taxiwayId;
                  standId = first.standId;
                  tx = endpoint.x;
                  ty = endpoint.y;
                }}
                if (standId && taxiwayId) {{
                  pushUndo();
                  state.apronLinks.push({{ id: id(), pbbId: standId, taxiwayId, tx, ty }});                  
                  syncPanelFromState();
                  if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths();
                }}
              }}
              state.apronLinkTemp = null;
            }}
            draw();
          }}
        }} else if (hit) {{
          state.selectedObject = hit;
          if (hit.type === 'terminal') state.currentTerminalId = hit.id;
          // When clicking on the canvas, the corresponding type Mode switch to
          const sm = settingModeValueForHit(hit);
          if (sm) settingModeSelect.value = sm;
          if (hit.type === 'flight' && typeof switchToTab === 'function') switchToTab('flight');
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
                  if (d2 < (CELL_SIZE * TERM_CLOSE_POLY_CF) ** 2 && term.vertices.length >= 3) {{
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
          }} else if (isPathLayoutMode(mode)) {{
            if (state.taxiwayDrawingId) {{
              const tw = state.taxiways.find(t => t.id === state.taxiwayDrawingId);
              if (tw) {{
                const pt = {{ col, row }};
                const last = tw.vertices[tw.vertices.length - 1];
                if (!last || last.col !== col || last.row !== row) {{
                  // Runway When it comes to type, there are only two points(start/end)only allowed
                  if (tw.pathType === 'runway' && tw.vertices.length >= 2) return;
                  pushUndo();
                  tw.vertices.push(pt);
                  if (typeof syncStartEndFromVertices === 'function') syncStartEndFromVertices(tw);
                  // The drawing ends automatically the moment two points are struck.
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
            if (tryPlacePbbAt(wx, wy)) {{
              syncPanelFromState();
              draw();
            }}
          }} else if (mode === 'remote' && state.remoteDrawing) {{
            const prev = state.previewRemote;
            if (prev && !prev.overlap && tryPlaceRemoteAt(prev.col, prev.row)) {{
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
      // display:block Immediately after, the layout is not reflected yet. getBoundingClientRectcan be 0.
      // one frame later resizeCanvas()Update the canvas size with and draw() call
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
      scene3d.background = new THREE.Color(GRID_VIEW_BG);
      // 3D grid + Axis-only group (update3DSceneKeep them as separate groups to avoid erasing them from)
      gridGroup3d = new THREE.Group();
      scene3d.add(gridGroup3d);
      camera3d = new THREE.PerspectiveCamera(50, w/h, 1, 100000);
      const halfW = (GRID_COLS * CELL_SIZE) / 2, halfH = (GRID_ROWS * CELL_SIZE) / 2;
      const maxDim = Math.max(halfW, halfH);
      camera3d.position.set(maxDim * 1.2, maxDim * 1.2, maxDim * 1.2);
      camera3d.lookAt(0, 0, 0);
      // Axis Guide: Grid Plane X(red)–Y(abstract), the vertical axis Z(blue)displayed as
      const axisLen = CELL_SIZE * 8;
      const axisOrigin = new THREE.Vector3(-maxDim, 0, -maxDim);
      function addAxis(toVec, color) {{
        const pts = [axisOrigin, axisOrigin.clone().add(toVec)];
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({{ color }});
        const line = new THREE.Line(geo, mat);
        gridGroup3d.add(line);
      }}
      // x-axis: grid X direction
      addAxis(new THREE.Vector3(axisLen, 0, 0), 0xef4444);
      // y-axis: grid Y direction (world Z direction)
      addAxis(new THREE.Vector3(0, 0, axisLen), 0x22c55e);
      // z-axis: perpendicular (world Y direction)
      addAxis(new THREE.Vector3(0, axisLen, 0), 0x7c6af7);
      // at the end of the axis x,y,z Add label sprite
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
      createAxisLabel('z', '#7c6af7', new THREE.Vector3(0, axisLen * 1.1, 0));
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
        tryCommitStandPlacement3D(mode, wx, wy, col, row);
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
        // 2D Similar to auxiliary grid, but slightly more transparent
        const faintMat = new THREE.LineBasicMaterial({{
          color: 0xd4d4d4,
          transparent: true,
          opacity: 0.2,
          depthTest: false
        }});
        gridGroup3d.add(new THREE.LineSegments(faintGeo, faintMat));
      }}
      if (majorLines.length) {{
        const majorGeo = new THREE.BufferGeometry().setFromPoints(majorLines);
        // The main grid is also slightly transparent so that it blends well with the background.
        const majorMat = new THREE.LineBasicMaterial({{
          color: 0xffffff,
          transparent: true,
          opacity: 0.35,
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
        const mesh = new THREE.Mesh(extrude, new THREE.MeshPhongMaterial({{ color: hexToThreeColor(_canvas2dStyle.terminalStrokeDefault || '#38bdf8'), transparent: true, opacity: 0.55 }}));
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

        // 3D green apron: PBB 3D direction(dir)With the same rotation as,
        // XZ Construct a square directly on a plane. (2DRotation feeling as similar as possible to)
        const standSize = getStandSizeMeters(pbb.category || 'C');
        const half = standSize / 2;
        const apronY = CELL_SIZE * 0.02;
        const center = grid3DMapper.worldFromPixel(pbb.x2, pbb.y2, apronY);

        const dirXZ = new THREE.Vector3(end.x - start.x, 0, end.z - start.z);
        let apronMesh = null;
        if (dirXZ.lengthSq() > 1e-6) {{
          dirXZ.normalize();
          const perp = new THREE.Vector3(-dirXZ.z, 0, dirXZ.x); // XZ Rotate 90 degrees in a plane
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
        const apronMat = new THREE.MeshPhongMaterial({{
          color: hexToThreeColor(_threeDStyle.remoteApron || '#22c55e'),
          transparent: true,
          opacity: threeOpacity(_threeDStyle.remoteApronOpacity, 0.55),
        }});
        const apron = new THREE.Mesh(apronGeo, apronMat);
        apron.position.copy(center);
        apron.rotation.x = -Math.PI / 2; // flat on ground, axis-aligned like 2D
        scene3d.add(apron);

        const box = new THREE.Mesh(
          new THREE.BoxGeometry(CELL_SIZE * 0.7, CELL_SIZE * 0.3, CELL_SIZE * 0.7),
          new THREE.MeshPhongMaterial({{ color: hexToThreeColor(_threeDStyle.remoteStandBox || '#22c55e') }})
        );
        box.position.copy(grid3DMapper.cellToWorld(st.col, st.row, CELL_SIZE * 0.15));
        scene3d.add(box);
      }});
      state.taxiways.forEach(tw => {{
        if (tw.vertices.length < 2) return;
        const w = tw.width != null ? tw.width : (tw.pathType === 'runway' ? 60 : 15);
        const isRunwayPath = tw.pathType === 'runway';
        const isRunwayExit3d = tw.pathType === 'runway_exit';
        const rwGrayHex = _threeDStyle.runwayPath || '#374151';
        const h = CELL_SIZE * 0.04;
        // 2Dsame as vertex Connect with straight segments between,
        // Additional patches are added to the corner points to give a visual roundness..
        const worldPts = tw.vertices.map(v => {{
          const [px, py] = cellToPixel(v.col, v.row);
          return grid3DMapper.worldFromPixel(px, py, h);
        }});
        // default segment
        for (let i = 0; i < worldPts.length - 1; i++) {{
          const start = worldPts[i];
          const end = worldPts[i + 1];
          const dirVec = new THREE.Vector3().subVectors(end, start);
          const length = dirVec.length() || 1;
          dirVec.normalize();
          const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
          const segGeo = new THREE.BoxGeometry(length, h * 0.5, w);
          const segMat = new THREE.MeshPhongMaterial({{ color: hexToThreeColor((isRunwayPath || isRunwayExit3d) ? rwGrayHex : (_threeDStyle.taxiway || '#eab308')) }});
          const seg = new THREE.Mesh(segGeo, segMat);
          seg.position.copy(mid);
          seg.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dirVec);
          scene3d.add(seg);
        }}
        // Auxiliary box for corner rounding (joint patch)
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
          // If it's almost a straight line, skip it.
          if (Math.abs(dot) > 0.999) continue;
          const bis = new THREE.Vector3().addVectors(v1, v2);
          if (bis.lengthSq() < 1e-4) continue;
          bis.normalize();
          const jointLen = w * 0.8;
          const jointGeo = new THREE.BoxGeometry(jointLen, h * 0.5, w * 1.02);
          const jointMat = new THREE.MeshPhongMaterial({{ color: hexToThreeColor((isRunwayPath || isRunwayExit3d) ? rwGrayHex : (_threeDStyle.taxiway || '#eab308')) }});
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
            const coneMat = new THREE.MeshPhongMaterial({{ color: hexToThreeColor(_threeDStyle.arrowCone || '#f59e0b') }});
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.copy(pos);
            cone.position.y = h + 0.8;
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
        const linkMat = new THREE.MeshPhongMaterial({{
          color: hexToThreeColor(_threeDStyle.apronLink || '#22d3ee'),
          transparent: true,
          opacity: threeOpacity(_threeDStyle.apronLinkOpacity, 0.9),
        }});
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
          const pos3d = grid3DMapper.worldFromPixel(x, y, CELL_SIZE * _numOr(_ac3d.altitudeCellFactor, 0.5));
          const len = Math.hypot(dx, dy) || 1;
          const dirVec = new THREE.Vector3(dx / len, 0, dy / len);
          const code = (f.code || '').toUpperCase();
          const scale = apronAircraftScaleForIcao(code);
          const bl = _numOr(_ac3d.bodyLengthCellFactor, 1.2);
          const bw = _numOr(_ac3d.bodyWidthCellFactor, 0.4);
          const bh = _numOr(_ac3d.bodyHeightCellFactor, 0.2);
          const bodyLen = CELL_SIZE * bl * scale;
          const bodyWidth = CELL_SIZE * bw * scale;
          const bodyHeight = CELL_SIZE * bh * scale;
          const wLen = _numOr(_ac3d.wingLengthRatio, 0.4);
          const wHt = _numOr(_ac3d.wingHeightRatio, 0.5);
          const wWd = _numOr(_ac3d.wingWidthRatio, 1.8);
          const wYo = _numOr(_ac3d.wingYOffsetRatio, 0.2);
          const color = hexToThreeColor(_ac3d.meshColorHex || '#ff2f92');
          const group = new THREE.Group();
          const bodyGeo = new THREE.BoxGeometry(bodyLen, bodyHeight, bodyWidth);
          const bodyMat = new THREE.MeshPhongMaterial({{ color }});
          const body = new THREE.Mesh(bodyGeo, bodyMat);
          group.add(body);
          const wingGeo = new THREE.BoxGeometry(bodyLen * wLen, bodyHeight * wHt, bodyWidth * wWd);
          const wingMat = new THREE.MeshPhongMaterial({{ color }});
          const wings = new THREE.Mesh(wingGeo, wingMat);
          wings.position.y = -bodyHeight * wYo;
          group.add(wings);
          group.position.copy(pos3d);
          const forward = new THREE.Vector3(1, 0, 0);
          const quat = new THREE.Quaternion().setFromUnitVectors(forward, dirVec);
          group.quaternion.copy(quat);
          scene3d.add(group);
        }});
      }}
      const light = new THREE.DirectionalLight(0xffffff, threeOpacity(_threeDStyle.directionalLightIntensity, 0.8));
      light.position.set(maxDim, maxDim * 2, maxDim);
      scene3d.add(light);
      scene3d.add(new THREE.AmbientLight(0xffffff, threeOpacity(_threeDStyle.ambientLightIntensity, 0.4)));
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
    try {{ applyInitialLayoutFromJson(); }} catch(applyErr) {{ console.error('Layout apply failed:', applyErr); }}
    updateLayoutNameBar(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout');
    resizeCanvas();
    reset2DView();
    syncPanelFromState();
    if (typeof draw === 'function') draw();
    if (typeof update3DScene === 'function') update3DScene();
    if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Initial load');
  }})();
  </script>
</body>
</html>
"""

# Homeof home_globelike in full screen HTML mark (position: fixed, 100vw x 100vh)
st.markdown("""
  <style>
    .stApp, [data-testid="stAppViewContainer"], section.main { background-color: #0d0d0f !important; }
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

