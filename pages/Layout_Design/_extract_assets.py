"""Regenerate designer.css / designer_body.html / designer.js from `3_⚙️ Layout_Design.py` f-string body.

Run from repo root::
    python pages/Layout_Design/_extract_assets.py
    python pages/Layout_Design/split_designer_js.py

The split step rewrites the 17 `*.js` parts from monolithic `designer.js`.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pages" / "3_⚙️ Layout_Design.py"
OUT = Path(__file__).resolve().parent
# Line slices are 0-based [start:end) into `_build_designer_html` f-string body (verified against main file).
_SLC_CSS = (673, 1285)
_SLC_BODY = (1287, 1737)
_SLC_JS = (1739, 15910)


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)

    css = "".join(lines[_SLC_CSS[0] : _SLC_CSS[1]])
    css = css.replace("{{", "{").replace("}}", "}")
    for old, new in (
        ("{_PANEL_FORM_SCOPE_SELECTORS}", "__PANEL_FORM_SCOPE_SELECTORS__"),
        ("{_PANEL_FORM_LABEL_SELECTORS}", "__PANEL_FORM_LABEL_SELECTORS__"),
        ("{_PANEL_FORM_LABEL_FIRST_SELECTORS}", "__PANEL_FORM_LABEL_FIRST_SELECTORS__"),
        ("{_PANEL_FORM_CONTROL_SELECTORS}", "__PANEL_FORM_CONTROL_SELECTORS__"),
        ("{_PANEL_FORM_SELECT_SELECTORS}", "__PANEL_FORM_SELECT_SELECTORS__"),
        ("{_PANEL_FORM_FOCUS_SELECTORS}", "__PANEL_FORM_FOCUS_SELECTORS__"),
        ("{_PANEL_FORM_SMALL_BUTTON_SELECTORS}", "__PANEL_FORM_SMALL_BUTTON_SELECTORS__"),
    ):
        css = css.replace(old, new)
    (OUT / "designer.css").write_text(css, encoding="utf-8")

    body = "".join(lines[_SLC_BODY[0] : _SLC_BODY[1]])
    for old, new in (
        ("{_flight_speed_options_html}", "__FLIGHT_SPEED_OPTIONS_HTML__"),
        ("{_top_panel_tabs_html}", "__TOP_PANEL_TABS_HTML__"),
        ("{_layout_mode_tabs_primary_html}", "__LAYOUT_MODE_TABS_PRIMARY_HTML__"),
        ("{_layout_mode_tabs_secondary_html}", "__LAYOUT_MODE_TABS_SECONDARY_HTML__"),
        ("{_ui_g_min_cs}", "__UI_G_MIN_CS__"),
        ("{_ui_g_max_cs}", "__UI_G_MAX_CS__"),
        ("{CELL_SIZE}", "__CELL_SIZE__"),
        ("{_ui_g_cs_step}", "__UI_G_CS_STEP__"),
        ("{_ui_g_min_dim}", "__UI_G_MIN_DIM__"),
        ("{_ui_g_max_dim}", "__UI_G_MAX_DIM__"),
        ("{GRID_COLS}", "__GRID_COLS__"),
        ("{GRID_ROWS}", "__GRID_ROWS__"),
        ("{_ui_g_img_opacity_min}", "__UI_G_IMG_OPACITY_MIN__"),
        ("{_ui_g_img_opacity_max}", "__UI_G_IMG_OPACITY_MAX__"),
        ("{_ui_g_img_opacity_step}", "__UI_G_IMG_OPACITY_STEP__"),
        ("{_ui_g_img_opacity}", "__UI_G_IMG_OPACITY__"),
        ("{_ui_g_img_size_step}", "__UI_G_IMG_SIZE_STEP__"),
        ("{_ui_g_img_width_m}", "__UI_G_IMG_WIDTH_M__"),
        ("{_ui_g_img_height_m}", "__UI_G_IMG_HEIGHT_M__"),
        ("{_ui_g_img_point_step}", "__UI_G_IMG_POINT_STEP__"),
        ("{_ui_g_img_top_left_col}", "__UI_G_IMG_TOP_LEFT_COL__"),
        ("{_ui_g_img_top_left_row}", "__UI_G_IMG_TOP_LEFT_ROW__"),
        ("{_ui_term_floors_min}", "__UI_TERM_FLOORS_MIN__"),
        ("{_ui_term_floors_max}", "__UI_TERM_FLOORS_MAX__"),
        ("{_ui_term_floors}", "__UI_TERM_FLOORS__"),
        ("{_ui_term_f2f_min}", "__UI_TERM_F2F_MIN__"),
        ("{_ui_term_f2f_max}", "__UI_TERM_F2F_MAX__"),
        ("{_ui_term_f2f}", "__UI_TERM_F2F__"),
        ("{_ui_term_f2f_step}", "__UI_TERM_F2F_STEP__"),
        ("{_ui_term_dep}", "__UI_TERM_DEP__"),
        ("{_ui_term_arr}", "__UI_TERM_ARR__"),
        ("{_ui_shared_path_width_min}", "__UI_SHARED_PATH_WIDTH_MIN__"),
        ("{_ui_tw_w}", "__UI_TW_W__"),
        ("{_ui_rw_min_arr}", "__UI_RW_MIN_ARR__"),
        ("{_ui_rw_lineup}", "__UI_RW_LINEUP__"),
        ("{_ui_rw_disp_start}", "__UI_RW_DISP_START__"),
        ("{_ui_rw_blast_start}", "__UI_RW_BLAST_START__"),
        ("{_ui_rw_disp_end}", "__UI_RW_DISP_END__"),
        ("{_ui_rw_blast_end}", "__UI_RW_BLAST_END__"),
        ("{_ui_tw_avg}", "__UI_TW_AVG__"),
        ("{_ui_ex_max}", "__UI_EX_MAX__"),
        ("{_ui_ex_min}", "__UI_EX_MIN__"),
        ("{_ui_flight_dwell_max}", "__UI_FLIGHT_DWELL_MAX__"),
        ("{_ui_flight_dwell}", "__UI_FLIGHT_DWELL__"),
        ("{_ui_flight_dwell_step}", "__UI_FLIGHT_DWELL_STEP__"),
        ("{_ui_flight_min_dwell}", "__UI_FLIGHT_MIN_DWELL__"),
    ):
        body = body.replace(old, new)
    (OUT / "designer_body.html").write_text(body, encoding="utf-8")

    js_lines = lines[_SLC_JS[0] : _SLC_JS[1]]
    js = "".join(s[2:] if s.startswith("  ") else s for s in js_lines)
    js = js.replace("{{", "{").replace("}}", "}")

    boot_old = """(function() {
  const LAYOUT_API_URL = {json.dumps(LAYOUT_API_URL)};
  const LAYOUT_NAMES = {json.dumps(layout_names)};
  const INITIAL_LAYOUT = {json.dumps(layout_for_html)};
  const INITIAL_LAYOUT_DISPLAY_NAME = {json.dumps(layout_display_name)};
  const INFORMATION = {json.dumps(INFORMATION)};
  const GRID_VIEW_BG = {json.dumps(_GRID_VIEW_BG)};
  const GRID_MAJOR_LINE_OPACITY = {json.dumps(_GRID_MAJOR_LINE_OPACITY)};
  const GRID_MINOR_LINE_OPACITY = {json.dumps(_GRID_MINOR_LINE_OPACITY)};
  const GRID_MAJOR_INTERVAL = {json.dumps(_ui_g_major_interval)};
  const GRID_MAJOR_LINE_WIDTH = {json.dumps(_ui_g_major_line_w)};
  const GRID_MINOR_LINE_WIDTH = {json.dumps(_ui_g_minor_line_w)};
  const GRID_MAJOR_LINE_RGB = {json.dumps(_ui_g_major_line_rgb)};
  const GRID_MINOR_LINE_RGB = {json.dumps(_ui_g_minor_line_rgb)};
  const GRID_DRAW_VIEWPORT_MARGIN_CELLS = {json.dumps(_ui_g_draw_viewport_margin_cells)};
  const GRID_MINOR_GRID_MIN_SCALE = {json.dumps(_ui_g_minor_grid_min_scale)};
  let GRID_COLS = {GRID_COLS};
  let GRID_ROWS = {GRID_ROWS};
  let CELL_SIZE = {CELL_SIZE};
"""
    boot_new = """(function() {
  var _dc = window.__DESIGNER_CONFIG__;
  if (!_dc || typeof _dc !== 'object') { throw new Error('__DESIGNER_CONFIG__ missing'); }
  const LAYOUT_API_URL = _dc.layoutApiUrl;
  const LAYOUT_NAMES = _dc.layoutNames;
  const INITIAL_LAYOUT = _dc.initialLayout;
  const INITIAL_LAYOUT_DISPLAY_NAME = _dc.initialLayoutDisplayName;
  const INFORMATION = _dc.information;
  const GRID_VIEW_BG = _dc.gridViewBg;
  const GRID_MAJOR_LINE_OPACITY = _dc.gridMajorLineOpacity;
  const GRID_MINOR_LINE_OPACITY = _dc.gridMinorLineOpacity;
  const GRID_MAJOR_INTERVAL = _dc.gridMajorInterval;
  const GRID_MAJOR_LINE_WIDTH = _dc.gridMajorLineWidth;
  const GRID_MINOR_LINE_WIDTH = _dc.gridMinorLineWidth;
  const GRID_MAJOR_LINE_RGB = _dc.gridMajorLineRgb;
  const GRID_MINOR_LINE_RGB = _dc.gridMinorLineRgb;
  const GRID_DRAW_VIEWPORT_MARGIN_CELLS = _dc.gridDrawViewportMarginCells;
  const GRID_MINOR_GRID_MIN_SCALE = _dc.gridMinorGridMinScale;
  let GRID_COLS = _dc.gridCols;
  let GRID_ROWS = _dc.gridRows;
  let CELL_SIZE = _dc.cellSize;
"""
    if boot_old not in js:
        raise SystemExit("boot_old not found in extracted JS")
    js = js.replace(boot_old, boot_new, 1)

    js = js.replace(
        "  const SIM_TIME_SLIDER_SNAP_SEC = Math.max(1, Number({json.dumps(_ui_flight_sim_slider_snap_sec)}) || 60);",
        "  const SIM_TIME_SLIDER_SNAP_SEC = Math.max(1, Number(_dc.flightSimSliderSnapSec) || 60);",
    )
    js = js.replace(
        "  const DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT = {json.dumps(bool(_ui_flight_allow_rw_ground))};",
        "  const DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT = _dc.defaultAllowRunwayInGroundSegment;",
    )
    js = js.replace(
        "  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections({json.dumps(_rw_exit_allowed_default_raw)});",
        "  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections(_dc.rwExitAllowedDefaultRaw);",
    )
    old_img = """  const GRID_LAYOUT_IMAGE_DEFAULTS = {
    opacity: {json.dumps(_ui_g_img_opacity)},
    opacityMin: {json.dumps(_ui_g_img_opacity_min)},
    opacityMax: {json.dumps(_ui_g_img_opacity_max)},
    widthM: {json.dumps(_ui_g_img_width_m)},
    heightM: {json.dumps(_ui_g_img_height_m)},
    topLeftCol: {json.dumps(_ui_g_img_top_left_col)},
    topLeftRow: {json.dumps(_ui_g_img_top_left_row)}
  };"""
    new_img = """  const GRID_LAYOUT_IMAGE_DEFAULTS = {
    opacity: _dc.gridLayoutImage.opacity,
    opacityMin: _dc.gridLayoutImage.opacityMin,
    opacityMax: _dc.gridLayoutImage.opacityMax,
    widthM: _dc.gridLayoutImage.widthM,
    heightM: _dc.gridLayoutImage.heightM,
    topLeftCol: _dc.gridLayoutImage.topLeftCol,
    topLeftRow: _dc.gridLayoutImage.topLeftRow
  };"""
    if old_img not in js:
        raise SystemExit("GRID_LAYOUT_IMAGE_DEFAULTS block not found")
    js = js.replace(old_img, new_img, 1)

    js = js.replace(
        "    simSpeed: {json.dumps(float(_ui_default_sim_speed))},",
        "    simSpeed: _dc.defaultSimSpeed,",
    )
    js = js.replace(
        "      state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : {json.dumps(float(_ui_default_sim_speed))};",
        "      state.simSpeed = !isNaN(v0) && v0 > 0 ? v0 : _dc.defaultSimSpeed;",
    )

    if "json.dumps" in js:
        raise SystemExit("Leftover python injection in JS: check manually")

    (OUT / "designer.js").write_text(js, encoding="utf-8")
    print("Wrote designer.css, designer_body.html, designer.js")


if __name__ == "__main__":
    main()
