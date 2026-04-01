import json
import logging
import os
import random
import shutil
import string
from pathlib import Path

_logger = logging.getLogger(__name__)

import streamlit as st
import streamlit.components.v1 as components

from utils.layout_flight_meta import ensure_flight_service_dates
from utils.layout_receiver import (
    DEFAULT_LAYOUT_PATH,
    LAYOUT_STORAGE_DIR,
    list_layout_names,
    _safe_layout_path,
    start_layout_receiver,
)

if os.environ.get("LAYOUT_SAME_PORT") == "1":
    LAYOUT_API_URL = os.environ.get("LAYOUT_API_BASE_URL", "http://127.0.0.1:8501")
else:
    LAYOUT_API_URL = start_layout_receiver()

st.set_page_config(
    page_title="Terminal & Airside Designer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


GRID_COLS = 200
GRID_ROWS = 200
CELL_SIZE = 20.0

_data_dir = Path(__file__).resolve().parents[1] / "data"
_fallback_default = _data_dir / "default_layout.json"
LAYOUT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

INFO_STORAGE_DIR = _data_dir / "Info_storage"
INFO_FILE = INFO_STORAGE_DIR / "Information.json"
INFORMATION: dict = {}
try:
    INFO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if INFO_FILE.is_file():
        INFORMATION = json.loads(INFO_FILE.read_text(encoding="utf-8"))
except Exception:
    _logger.warning("Failed to load Information.json", exc_info=True)


def _info_path(*keys, default=None):
    d = INFORMATION
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def _dict_or_empty(value) -> dict:
    return value if isinstance(value, dict) else {}


def _cfg_cast(section: dict, key: str, default, cast):
    if not isinstance(section, dict):
        return default
    raw = section.get(key)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _cfg_int(section: dict, key: str, default: int) -> int:
    return int(_cfg_cast(section, key, default, int))


def _cfg_float(section: dict, key: str, default: float) -> float:
    return float(_cfg_cast(section, key, default, float))


def _cfg_str(section: dict, key: str, default: str) -> str:
    value = default if not isinstance(section, dict) else section.get(key, default)
    return str(value)


def _cfg_bool(section: dict, key: str, default: bool) -> bool:
    if not isinstance(section, dict):
        return default
    raw = section.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return default


def _cfg_list(section: dict, key: str, default: list):
    if not isinstance(section, dict):
        return default
    value = section.get(key)
    return value if isinstance(value, list) else default


def _cfg_bundle(section: dict, specs: list[tuple[str, str, object, object]]) -> dict:
    values: dict[str, object] = {}
    for name, key, default, reader in specs:
        values[name] = reader(section, key, default)
    return values


_layout_info = _dict_or_empty(_info_path("tiers", "layout"))
_grid_info = _dict_or_empty(_layout_info.get("grid"))
_grid_img_info = _dict_or_empty(_grid_info.get("layoutImageOverlay"))

if _grid_info.get("cols") is not None:
    GRID_COLS = max(1, _cfg_int(_grid_info, "cols", GRID_COLS))
if _grid_info.get("rows") is not None:
    GRID_ROWS = max(1, _cfg_int(_grid_info, "rows", GRID_ROWS))
if _grid_info.get("cellSize") is not None:
    CELL_SIZE = _cfg_float(_grid_info, "cellSize", CELL_SIZE)

_term_ui = _dict_or_empty(_layout_info.get("terminal"))
_tw_ui = _dict_or_empty(_layout_info.get("taxiway"))
_rw_path_ui = _dict_or_empty(_layout_info.get("runwayPath"))
_rw_exit_ui = _dict_or_empty(_layout_info.get("runwayExit"))
_rw_exit_allowed_default_raw = _cfg_list(
    _rw_exit_ui,
    "allowedRwDirectionsDefault",
    ["clockwise", "counter_clockwise"],
)

_flight_info = _dict_or_empty(_info_path("tiers", "flight_schedule") or _info_path("tiers", "flight"))
_algo_tier = _dict_or_empty(_info_path("tiers", "algorithm"))
_algo_sim = _dict_or_empty(_algo_tier.get("simulation"))
_default_flight_service_date = _cfg_str(
    _flight_info, "defaultServiceDate", _cfg_str(_algo_sim, "baseDate", "2026-03-31")
)

if not DEFAULT_LAYOUT_PATH.is_file() and _fallback_default.is_file():
    shutil.copy2(_fallback_default, DEFAULT_LAYOUT_PATH)

DEFAULT_LAYOUT: dict = {}
try:
    for _layout_path in (DEFAULT_LAYOUT_PATH, _fallback_default):
        if _layout_path.is_file():
            DEFAULT_LAYOUT = json.loads(_layout_path.read_text(encoding="utf-8"))
            break
except Exception:
    _logger.warning("Failed to load default layout", exc_info=True)


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
ensure_flight_service_dates(DEFAULT_LAYOUT, _default_flight_service_date)

_grid_ui_defaults = _cfg_bundle(_grid_info, [
    ("min_cs", "minCellSize", 5, _cfg_int),
    ("max_cs", "maxCellSize", 1000, _cfg_int),
    ("cs_step", "cellSizeStep", 5, _cfg_int),
    ("min_dim", "minDim", 5, _cfg_int),
    ("max_dim", "maxDim", 1000, _cfg_int),
    ("major_interval", "majorInterval", 10, _cfg_int),
    ("major_line_w", "majorLineWidth", 1.2, _cfg_float),
    ("minor_line_w", "minorLineWidth", 1.0, _cfg_float),
    ("major_line_rgb", "majorLineBaseRgb", "255,255,255", _cfg_str),
    ("minor_line_rgb", "minorLineBaseRgb", "140,140,140", _cfg_str),
])
_grid_img_ui_defaults = _cfg_bundle(_grid_img_info, [
    ("opacity", "defaultOpacity", 0.45, _cfg_float),
    ("opacity_min", "minOpacity", 0.0, _cfg_float),
    ("opacity_max", "maxOpacity", 1.0, _cfg_float),
    ("opacity_step", "opacityStep", 0.05, _cfg_float),
    ("width_m", "defaultWidthM", 200.0, _cfg_float),
    ("height_m", "defaultHeightM", 200.0, _cfg_float),
    ("size_step", "sizeStepM", 1.0, _cfg_float),
    ("point_step", "pointStep", 0.5, _cfg_float),
    ("top_left_col", "defaultTopLeftCol", 0.0, _cfg_float),
    ("top_left_row", "defaultTopLeftRow", 0.0, _cfg_float),
])
_term_ui_defaults = _cfg_bundle(_term_ui, [
    ("floors", "floorsDefault", 1, _cfg_int),
    ("floors_min", "floorsMin", 1, _cfg_int),
    ("floors_max", "floorsMax", 20, _cfg_int),
    ("f2f", "floorToFloor", 4.0, _cfg_float),
    ("f2f_min", "floorToFloorMin", 1.0, _cfg_float),
    ("f2f_max", "floorToFloorMax", 10.0, _cfg_float),
    ("f2f_step", "floorToFloorStep", 0.5, _cfg_float),
    ("dep", "departureCapacity", 0, _cfg_int),
    ("arr", "arrivalCapacity", 0, _cfg_int),
])
_taxiway_ui_defaults = _cfg_bundle(_tw_ui, [
    ("width", "width", 1, _cfg_int),
    ("width_min", "minWidth", 1, _cfg_int),
    ("avg", "avgMoveVelocity", 10.0, _cfg_float),
])
_runway_path_ui_defaults = _cfg_bundle(_rw_path_ui, [
    ("min_arr", "minArrVelocity", 15, _cfg_int),
    ("lineup", "lineupDistM", 0, _cfg_int),
    ("disp_start", "startDisplacedThresholdM", 100, _cfg_int),
    ("blast_start", "startBlastPadM", 100, _cfg_int),
    ("disp_end", "endDisplacedThresholdM", 100, _cfg_int),
    ("blast_end", "endBlastPadM", 100, _cfg_int),
])
_runway_exit_ui_defaults = _cfg_bundle(_rw_exit_ui, [
    ("ex_width", "width", 1, _cfg_int),
    ("ex_max", "maxExitVelocity", 30, _cfg_int),
    ("ex_min", "minExitVelocity", 15, _cfg_int),
    ("ex_width_min", "minWidth", 1, _cfg_int),
])
_flight_ui_defaults = _cfg_bundle(_flight_info, [
    ("dwell", "defaultDwellMin", 60, _cfg_int),
    ("min_dwell", "defaultMinDwellMin", 0, _cfg_int),
    ("dwell_max", "dwellInputMax", 600, _cfg_int),
    ("dwell_step", "dwellStep", 5, _cfg_int),
    ("default_sim_speed", "defaultSimSpeed", 5.0, _cfg_float),
    ("sim_slider_snap_sec", "simTimeSliderSnapSec", 1, _cfg_int),
    ("allow_rw_ground", "defaultAllowRunwayInGroundSegment", False, _cfg_bool),
    ("dep_rot_min", "depRotMin", 2, _cfg_int),
])

_ui_g_min_cs = _grid_ui_defaults["min_cs"]
_ui_g_max_cs = _grid_ui_defaults["max_cs"]
_ui_g_cs_step = _grid_ui_defaults["cs_step"]
_ui_g_min_dim = _grid_ui_defaults["min_dim"]
_ui_g_max_dim = _grid_ui_defaults["max_dim"]
_ui_g_img_opacity = _grid_img_ui_defaults["opacity"]
_ui_g_img_opacity_min = _grid_img_ui_defaults["opacity_min"]
_ui_g_img_opacity_max = _grid_img_ui_defaults["opacity_max"]
_ui_g_img_opacity_step = _grid_img_ui_defaults["opacity_step"]
_ui_g_img_width_m = _grid_img_ui_defaults["width_m"]
_ui_g_img_height_m = _grid_img_ui_defaults["height_m"]
_ui_g_img_size_step = _grid_img_ui_defaults["size_step"]
_ui_g_img_point_step = _grid_img_ui_defaults["point_step"]
_ui_g_img_top_left_col = _grid_img_ui_defaults["top_left_col"]
_ui_g_img_top_left_row = _grid_img_ui_defaults["top_left_row"]

_ui_term_floors = _term_ui_defaults["floors"]
_ui_term_floors_min = _term_ui_defaults["floors_min"]
_ui_term_floors_max = _term_ui_defaults["floors_max"]
_ui_term_f2f = _term_ui_defaults["f2f"]
_ui_term_f2f_min = _term_ui_defaults["f2f_min"]
_ui_term_f2f_max = _term_ui_defaults["f2f_max"]
_ui_term_f2f_step = _term_ui_defaults["f2f_step"]
_ui_term_dep = _term_ui_defaults["dep"]
_ui_term_arr = _term_ui_defaults["arr"]

_ui_tw_w = _taxiway_ui_defaults["width"]
_ui_tw_w_min = max(1, int(_taxiway_ui_defaults["width_min"]))
_ui_rtx_w_min = max(1, int(_runway_exit_ui_defaults["ex_width_min"]))
_ui_shared_path_width_min = min(_ui_tw_w_min, _ui_rtx_w_min)
_ui_tw_avg = _taxiway_ui_defaults["avg"]
_ui_rw_min_arr = _runway_path_ui_defaults["min_arr"]
_ui_rw_lineup = _runway_path_ui_defaults["lineup"]
_ui_rw_disp_start = _runway_path_ui_defaults["disp_start"]
_ui_rw_blast_start = _runway_path_ui_defaults["blast_start"]
_ui_rw_disp_end = _runway_path_ui_defaults["disp_end"]
_ui_rw_blast_end = _runway_path_ui_defaults["blast_end"]
_ui_ex_max = _runway_exit_ui_defaults["ex_max"]
_ui_ex_min = _runway_exit_ui_defaults["ex_min"]

_ui_flight_dwell = _flight_ui_defaults["dwell"]
_ui_flight_min_dwell = _flight_ui_defaults["min_dwell"]
_ui_flight_dwell_max = _flight_ui_defaults["dwell_max"]
_ui_flight_dwell_step = _flight_ui_defaults["dwell_step"]
_ui_sim_speeds = _cfg_list(_flight_info, "simSpeedOptions", [0.5, 1, 5, 10, 20, 30])
_ui_default_sim_speed = _flight_ui_defaults["default_sim_speed"]
_ui_flight_sim_slider_snap_sec = max(1, int(_flight_ui_defaults["sim_slider_snap_sec"]))
_ui_flight_allow_rw_ground = _flight_ui_defaults["allow_rw_ground"]
_flight_speed_options_html = "".join(
    f'<option value="{v}"{" selected" if float(v) == float(_ui_default_sim_speed) else ""}>{v}x</option>'
    for v in _ui_sim_speeds
)

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
            ensure_flight_service_dates(layout_for_html, _default_flight_service_date)
            layout_display_name = load_name
except Exception:
    _logger.exception("Failed to load layout from query param")

layout_names = list_layout_names()


def _tiers_style_dict(info: dict) -> dict:
    tiers = _dict_or_empty(info).get("tiers")
    return _dict_or_empty(_dict_or_empty(tiers).get("style"))


def _tiers_style_section(info: dict, key: str) -> dict:
    return _dict_or_empty(_tiers_style_dict(info).get(key))


def _grid_view_background(info: dict) -> str:
    """tiers.style.gridView.background → 2D/3D workspace fill color."""
    gv = _tiers_style_section(info, "gridView")
    bg = str(gv.get("background", "#252525")).strip()
    if not bg.startswith("#") or len(bg) not in (4, 7):
        bg = "#252525"
    return bg


def _grid_view_line_opacity(info: dict, key: str, default: float) -> float:
    """tiers.style.gridView.(majorLineOpacity|minorLineOpacity) → 0–1."""
    gv = _tiers_style_section(info, "gridView")
    try:
        value = float(gv.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def _right_panel_background_opacity(info: dict) -> float:
    """tiers.style.rightPanel.backgroundOpacity → 0–1 for color-mix over transparent."""
    rp = _tiers_style_section(info, "rightPanel")
    try:
        op = float(rp.get("backgroundOpacity", 0.95))
    except (TypeError, ValueError):
        op = 0.95
    return max(0.0, min(1.0, op))


def _style_root_css_from_information(info: dict) -> str:
    """tiers.style → :root CSS variables (Gantt, tables, canvas, 3D theme)."""
    st = _tiers_style_dict(info)
    _gv_bg = _grid_view_background(info)
    _rp_bg_op = _right_panel_background_opacity(info)
    _rp_mix_pct = f"{round(_rp_bg_op * 100)}%"
    rp_style = _dict_or_empty(st.get("rightPanel"))

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

    ui = st.get("uiTheme") if isinstance(st.get("uiTheme"), dict) else {}
    _ui_map = [
        ("--ui-bg-base", "bgBase", "#0d0d0f"),
        ("--ui-bg-surface", "bgSurface", "#141416"),
        ("--ui-bg-elevated", "bgElevated", "#242428"),
        ("--ui-bg-overlay", "bgOverlay", "#242428"),
        ("--ui-bg-control", "bgControl", "#27272b"),
        ("--ui-bg-input", "bgInput", "#1a1a1d"),
        ("--ui-border-subtle", "borderSubtle", "rgba(255, 255, 255, 0.06)"),
        ("--ui-border-default", "borderDefault", "rgba(255, 255, 255, 0.10)"),
        ("--ui-border-strong", "borderStrong", "rgba(255, 255, 255, 0.18)"),
        ("--ui-text-primary", "textPrimary", "#f0f0f2"),
        ("--ui-text-secondary", "textSecondary", "#8b8b96"),
        ("--ui-text-muted", "textMuted", "#5c5c68"),
        ("--ui-accent", "accent", "#7c6af7"),
        ("--ui-accent-hover", "accentHover", "#9181f8"),
        ("--ui-accent-muted", "accentMuted", "rgba(124, 106, 247, 0.16)"),
        ("--ui-accent-ring", "accentRing", "rgba(124, 106, 247, 0.35)"),
        ("--ui-success-bg", "successBg", "rgba(61, 214, 140, 0.14)"),
        ("--ui-success-border", "successBorder", "rgba(61, 214, 140, 0.45)"),
        ("--ui-success-text", "successText", "#b8f0d0"),
        ("--ui-error-bg", "errorBg", "rgba(248, 113, 113, 0.12)"),
        ("--ui-error-border", "errorBorder", "rgba(248, 113, 113, 0.35)"),
        ("--ui-error-text", "errorText", "#fca5a5"),
        ("--ui-font", "font", "'Inter', 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif"),
        ("--ui-transition", "transition", "border-color 0.15s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.15s cubic-bezier(0.16, 1, 0.3, 1), color 0.15s cubic-bezier(0.16, 1, 0.3, 1), filter 0.15s cubic-bezier(0.16, 1, 0.3, 1)"),
        ("--ui-scrollbar-track", "scrollbarTrack", "rgba(0, 0, 0, 0.78)"),
        ("--ui-scrollbar-thumb", "scrollbarThumb", "rgba(255, 255, 255, 0.16)"),
        ("--ui-scrollbar-thumb-hover", "scrollbarThumbHover", "rgba(255, 255, 255, 0.24)"),
        ("--ui-scrollbar-border", "scrollbarBorder", "rgba(255, 255, 255, 0.06)"),
    ]
    for css_var, json_key, default in _ui_map:
        val = ui.get(json_key, default)
        row = _emit(css_var, val)
        if row:
            lines.append(row)

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
    for pairs in (pairs_gantt, pairs_fs, pairs_rwy, pairs_c2):
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
        ("--style-right-panel-bg", rp_style.get("backgroundColor")),
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
_GRID_MAJOR_LINE_OPACITY = _grid_view_line_opacity(INFORMATION, "majorLineOpacity", 0.35)
_GRID_MINOR_LINE_OPACITY = _grid_view_line_opacity(INFORMATION, "minorLineOpacity", 0.2)

_ui_g_major_interval = _grid_ui_defaults["major_interval"]
_ui_g_major_line_w = _grid_ui_defaults["major_line_w"]
_ui_g_minor_line_w = _grid_ui_defaults["minor_line_w"]
_ui_g_major_line_rgb = _grid_ui_defaults["major_line_rgb"]
_ui_g_minor_line_rgb = _grid_ui_defaults["minor_line_rgb"]
_ui_g_draw_viewport_margin_cells = max(0, _cfg_int(_grid_info, "drawViewportMarginCells", 2))
_ui_g_minor_grid_min_scale = max(0.0, _cfg_float(_grid_info, "minorGridMinScale", 0.0))

layout_display_name = "default_layout"

_IMAGE_DIR = _data_dir / "image"


def _read_svg_icon(filename: str) -> str:
    path = _IMAGE_DIR / filename
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        _logger.debug("Failed to read SVG icon: %s", filename, exc_info=True)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
        'viewBox="0 0 64 64" fill="none">'
        '<rect x="12" y="12" width="40" height="40" rx="10" stroke="white" stroke-width="4"/>'
        "</svg>"
    )


def _layout_mode_button_html(mode: str, label: str, icon_filename: str) -> str:
    icon_svg = _read_svg_icon(icon_filename)
    return (
        f'<button type="button" class="layout-mode-tab" data-mode="{mode}">'
        f'<span class="layout-mode-icon" aria-hidden="true">{icon_svg}</span>'
        f'<span class="layout-mode-label">{label}</span>'
        "</button>"
    )


def _top_panel_tab_html(tab_id: str, label: str, icon_filename: str, active: bool = False) -> str:
    icon_svg = _read_svg_icon(icon_filename)
    active_class = " active" if active else ""
    return (
        f'<button type="button" class="right-panel-tab{active_class}" data-tab="{tab_id}">'
        f'<span class="right-panel-tab-icon" aria-hidden="true">{icon_svg}</span>'
        f'<span class="right-panel-tab-label">{label}</span>'
        "</button>"
    )


def _build_layout_mode_tabs_html() -> tuple[str, str]:
    primary_html = "".join(
        [
            _layout_mode_button_html("grid", "Grid", "layout_mode_grid.svg"),
            _layout_mode_button_html("runwayPath", "Runway", "layout_mode_runway.svg"),
            _layout_mode_button_html("runwayTaxiway", "Runway Taxiway", "layout_mode_runway_taxiway.svg"),
            _layout_mode_button_html("taxiway", "Taxiway", "layout_mode_taxiway.svg"),
            _layout_mode_button_html("holdingPoint", "Holding Point", "layout_mode_holding_point.svg"),
            _layout_mode_button_html("edge", "Edge", "layout_mode_edge.svg"),
        ]
    )
    secondary_html = "".join(
        [
            _layout_mode_button_html("terminal", "Building", "layout_mode_terminal.svg"),
            _layout_mode_button_html("pbb", "Contact Stand", "layout_mode_pbb.svg"),
            _layout_mode_button_html("remote", "Remote Stand", "layout_mode_remote.svg"),
            _layout_mode_button_html("apronTaxiway", "Apron Taxiway", "layout_mode_apron_taxiway.svg"),
            _layout_mode_button_html("groundAccess", "Ground Access", "layout_mode_ground_access.svg"),
        ]
    )
    return primary_html, secondary_html


def _build_top_panel_tabs_html() -> str:
    return "".join(
        [
            _top_panel_tab_html("settings", "Layout", "top_tab_layout.svg", active=True),
            _top_panel_tab_html("flight", "Flight", "top_tab_flight.svg"),
            _top_panel_tab_html("rwysep", "Runway", "top_tab_runway.svg"),
            _top_panel_tab_html("allocation", "Apron", "top_tab_apron.svg"),
            _top_panel_tab_html("simulation", "KPI", "top_tab_kpi.svg"),
            _top_panel_tab_html("saveload", "Save", "top_tab_saveload.svg"),
        ]
    )


_layout_mode_tabs_primary_html, _layout_mode_tabs_secondary_html = _build_layout_mode_tabs_html()
_top_panel_tabs_html = _build_top_panel_tabs_html()
_PANEL_FORM_PREFIXES: tuple[str, ...] = (
    "#tab-settings .settings-pane",
    "#flightPaneSchedule",
    "#flightPaneConfig",
)


def _panel_form_join(lines: list[str]) -> str:
    return ",\n    ".join(lines)


_PANEL_FORM_SCOPE_SELECTORS = _panel_form_join(list(_PANEL_FORM_PREFIXES))
_PANEL_FORM_LABEL_SELECTORS = _panel_form_join([p + " label" for p in _PANEL_FORM_PREFIXES])
_PANEL_FORM_LABEL_FIRST_SELECTORS = _panel_form_join([p + " label:first-child" for p in _PANEL_FORM_PREFIXES])
_PANEL_FORM_CONTROL_SELECTORS = _panel_form_join([
        '#tab-settings .settings-pane input:not([type="checkbox"]):not([type="radio"]):not([type="range"])',
        '#tab-settings .settings-pane select',
        '#flightPaneSchedule input:not([type="checkbox"]):not([type="radio"]):not([type="range"])',
        '#flightPaneSchedule select',
        '#flightPaneConfig input:not([type="checkbox"]):not([type="radio"]):not([type="range"])',
        '#flightPaneConfig select',
    ])
_PANEL_FORM_SELECT_SELECTORS = _panel_form_join([p + " select" for p in _PANEL_FORM_PREFIXES])
_PANEL_FORM_FOCUS_SELECTORS = _panel_form_join([
        '#tab-settings .settings-pane input:focus',
        '#tab-settings .settings-pane select:focus',
        '#flightPaneSchedule input:focus',
        '#flightPaneSchedule select:focus',
        '#flightPaneConfig input:focus',
        '#flightPaneConfig select:focus',
    ])
_PANEL_FORM_SMALL_BUTTON_SELECTORS = _panel_form_join([p + " button.small" for p in _PANEL_FORM_PREFIXES])

def _build_designer_context() -> dict:
    return {
        "style_root_css": _style_root_css,
        "layout_mode_tabs_primary_html": _layout_mode_tabs_primary_html,
        "layout_mode_tabs_secondary_html": _layout_mode_tabs_secondary_html,
        "top_panel_tabs_html": _top_panel_tabs_html,
    }


_DESIGNER_ASSET_DIR = Path(__file__).resolve().parent / "Layout_Design"

_DESIGNER_JS_PARTS: tuple[str, ...] = (
    "config.js",
    "state.js",
    "layout-objects.js",
    "geometry.js",
    "path-engine.js",
    "panel-sync.js",
    "flight-bridge.js",
    "flight-timeline.js",
    "flight-ui.js",
    "gantt.js",
    "kpi.js",
    "runway-sep.js",
    "simulation.js",
    "designer-mid.js",
    "panel-dom.js",
    "canvas2d.js",
    "canvas3d.js",
)


def _load_designer_asset(name: str) -> str:
    path = _DESIGNER_ASSET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Designer asset missing: {path}")
    return path.read_text(encoding="utf-8")


def _designer_js_config_dict() -> dict[str, object]:
    return {
        "layoutApiUrl": LAYOUT_API_URL,
        "layoutNames": layout_names,
        "initialLayout": layout_for_html,
        "initialLayoutDisplayName": layout_display_name,
        "information": INFORMATION,
        "gridViewBg": _GRID_VIEW_BG,
        "gridMajorLineOpacity": _GRID_MAJOR_LINE_OPACITY,
        "gridMinorLineOpacity": _GRID_MINOR_LINE_OPACITY,
        "gridMajorInterval": _ui_g_major_interval,
        "gridMajorLineWidth": _ui_g_major_line_w,
        "gridMinorLineWidth": _ui_g_minor_line_w,
        "gridMajorLineRgb": _ui_g_major_line_rgb,
        "gridMinorLineRgb": _ui_g_minor_line_rgb,
        "gridDrawViewportMarginCells": _ui_g_draw_viewport_margin_cells,
        "gridMinorGridMinScale": _ui_g_minor_grid_min_scale,
        "gridCols": GRID_COLS,
        "gridRows": GRID_ROWS,
        "cellSize": CELL_SIZE,
        "flightSimSliderSnapSec": _ui_flight_sim_slider_snap_sec,
        "defaultAllowRunwayInGroundSegment": bool(_ui_flight_allow_rw_ground),
        "rwExitAllowedDefaultRaw": _rw_exit_allowed_default_raw,
        "gridLayoutImage": {
            "opacity": _ui_g_img_opacity,
            "opacityMin": _ui_g_img_opacity_min,
            "opacityMax": _ui_g_img_opacity_max,
            "widthM": _ui_g_img_width_m,
            "heightM": _ui_g_img_height_m,
            "topLeftCol": _ui_g_img_top_left_col,
            "topLeftRow": _ui_g_img_top_left_row,
        },
        "defaultSimSpeed": float(_ui_default_sim_speed),
        "defaultFlightServiceDate": _default_flight_service_date,
    }


def _apply_designer_placeholders(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, val in mapping.items():
        token = f"__{key}__"
        if token not in out:
            raise RuntimeError(f"Designer template missing placeholder: {token}")
        out = out.replace(token, val)
    return out


def _build_designer_html() -> str:
    """Assemble designer iframe document from Layout_Design/* assets."""
    context = _build_designer_context()
    style_root = context["style_root_css"]
    layout_mode_tabs_primary_html = context["layout_mode_tabs_primary_html"]
    layout_mode_tabs_secondary_html = context["layout_mode_tabs_secondary_html"]
    top_panel_tabs_html = context["top_panel_tabs_html"]

    css_tm = _load_designer_asset("designer.css")
    css = _apply_designer_placeholders(
        css_tm,
        {
            "PANEL_FORM_SCOPE_SELECTORS": _PANEL_FORM_SCOPE_SELECTORS,
            "PANEL_FORM_LABEL_SELECTORS": _PANEL_FORM_LABEL_SELECTORS,
            "PANEL_FORM_LABEL_FIRST_SELECTORS": _PANEL_FORM_LABEL_FIRST_SELECTORS,
            "PANEL_FORM_CONTROL_SELECTORS": _PANEL_FORM_CONTROL_SELECTORS,
            "PANEL_FORM_SELECT_SELECTORS": _PANEL_FORM_SELECT_SELECTORS,
            "PANEL_FORM_FOCUS_SELECTORS": _PANEL_FORM_FOCUS_SELECTORS,
            "PANEL_FORM_SMALL_BUTTON_SELECTORS": _PANEL_FORM_SMALL_BUTTON_SELECTORS,
        },
    )

    body_tm = _load_designer_asset("designer_body.html")
    body = _apply_designer_placeholders(
        body_tm,
        {
            "FLIGHT_SPEED_OPTIONS_HTML": _flight_speed_options_html,
            "TOP_PANEL_TABS_HTML": top_panel_tabs_html,
            "LAYOUT_MODE_TABS_PRIMARY_HTML": layout_mode_tabs_primary_html,
            "LAYOUT_MODE_TABS_SECONDARY_HTML": layout_mode_tabs_secondary_html,
            "UI_G_MIN_CS": str(_ui_g_min_cs),
            "UI_G_MAX_CS": str(_ui_g_max_cs),
            "CELL_SIZE": str(CELL_SIZE),
            "UI_G_CS_STEP": str(_ui_g_cs_step),
            "UI_G_MIN_DIM": str(_ui_g_min_dim),
            "UI_G_MAX_DIM": str(_ui_g_max_dim),
            "GRID_COLS": str(GRID_COLS),
            "GRID_ROWS": str(GRID_ROWS),
            "UI_G_IMG_OPACITY_MIN": str(_ui_g_img_opacity_min),
            "UI_G_IMG_OPACITY_MAX": str(_ui_g_img_opacity_max),
            "UI_G_IMG_OPACITY_STEP": str(_ui_g_img_opacity_step),
            "UI_G_IMG_OPACITY": str(_ui_g_img_opacity),
            "UI_G_IMG_SIZE_STEP": str(_ui_g_img_size_step),
            "UI_G_IMG_WIDTH_M": str(_ui_g_img_width_m),
            "UI_G_IMG_HEIGHT_M": str(_ui_g_img_height_m),
            "UI_G_IMG_POINT_STEP": str(_ui_g_img_point_step),
            "UI_G_IMG_TOP_LEFT_COL": str(_ui_g_img_top_left_col),
            "UI_G_IMG_TOP_LEFT_ROW": str(_ui_g_img_top_left_row),
            "UI_TERM_FLOORS_MIN": str(_ui_term_floors_min),
            "UI_TERM_FLOORS_MAX": str(_ui_term_floors_max),
            "UI_TERM_FLOORS": str(_ui_term_floors),
            "UI_TERM_F2F_MIN": str(_ui_term_f2f_min),
            "UI_TERM_F2F_MAX": str(_ui_term_f2f_max),
            "UI_TERM_F2F": str(_ui_term_f2f),
            "UI_TERM_F2F_STEP": str(_ui_term_f2f_step),
            "UI_TERM_DEP": str(_ui_term_dep),
            "UI_TERM_ARR": str(_ui_term_arr),
            "UI_SHARED_PATH_WIDTH_MIN": str(_ui_shared_path_width_min),
            "UI_TW_W": str(_ui_tw_w),
            "UI_RW_MIN_ARR": str(_ui_rw_min_arr),
            "UI_RW_LINEUP": str(_ui_rw_lineup),
            "UI_RW_DISP_START": str(_ui_rw_disp_start),
            "UI_RW_BLAST_START": str(_ui_rw_blast_start),
            "UI_RW_DISP_END": str(_ui_rw_disp_end),
            "UI_RW_BLAST_END": str(_ui_rw_blast_end),
            "UI_TW_AVG": str(_ui_tw_avg),
            "UI_EX_MAX": str(_ui_ex_max),
            "UI_EX_MIN": str(_ui_ex_min),
            "UI_FLIGHT_DWELL_MAX": str(_ui_flight_dwell_max),
            "UI_FLIGHT_DWELL": str(_ui_flight_dwell),
            "UI_FLIGHT_DWELL_STEP": str(_ui_flight_dwell_step),
            "UI_FLIGHT_MIN_DWELL": str(_ui_flight_min_dwell),
        },
    )

    js_combined = "\n\n".join(_load_designer_asset(n) for n in _DESIGNER_JS_PARTS)
    cfg_json = json.dumps(_designer_js_config_dict(), ensure_ascii=False)
    out = (
        "<!DOCTYPE html>\n<html>\n<head>\n  <meta charset=\"utf-8\" />\n  <style>\n"
        + style_root
        + "\n"
        + css
        + "\n  </style>\n"
        + body
        + "\n\n  <script>\n  window.__DESIGNER_CONFIG__ = "
        + cfg_json
        + ";\n  </script>\n  <script>\n"
        + js_combined
        + "\n  </script>\n</body>\n</html>\n"
    )
    for needle in (
        "__PANEL_FORM_",
        "__FLIGHT_SPEED_OPTIONS_HTML__",
        "__TOP_PANEL_TABS_HTML__",
        "__LAYOUT_MODE_TABS_",
        "__UI_G_",
        "__UI_TERM_",
        "__UI_TW_W__",
        "__UI_RW_",
        "__UI_EX_",
        "__UI_FLIGHT_",
        "__GRID_COLS__",
        "__GRID_ROWS__",
        "__CELL_SIZE__",
        "__UI_SHARED_",
    ):
        if needle in out:
            raise RuntimeError(f"Unreplaced designer template token: {needle}")
    if "{{" in out:
        raise RuntimeError("Stray '{{' in designer HTML output (f-string escape leak)")
    return out


html = _build_designer_html()

def _designer_background_base() -> str:
    ui_theme = _dict_or_empty(_info_path("tiers", "style", "uiTheme"))
    return _cfg_str(ui_theme, "bgBase", "#0d0d0f")


def _build_streamlit_shell_css(ui_bg_base: str) -> str:
    return """
  <style>
    .stApp, [data-testid="stAppViewContainer"], section.main { background-color: """ + ui_bg_base + """ !important; }
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
"""


def _mount_designer_component(component_html: str) -> None:
    st.markdown(_build_streamlit_shell_css(_designer_background_base()), unsafe_allow_html=True)
    components.html(
        component_html,
        height=2000,
        scrolling=False,
    )


_mount_designer_component(html)
