"""
Simulation Viewer — displays airside simulation results with animated playback.

Reads sim_output.json (layout + positions + schedule + KPI) produced by
the Airside Simulation Engine and renders:
  1. A 2D canvas showing the layout (runways, taxiways, stands, terminals)
     with aircraft points animated over time.
  2. Playback controls (play/pause, speed, time slider).
  3. A-schedule table and KPI cards (below the canvas).
"""
import json
import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Simulation Viewer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _ROOT / "data"
_LAYOUT_STORAGE = _DATA_DIR / "Layout_storage"
_SIM_OUTPUT_FILE = _LAYOUT_STORAGE / "sim_output.json"
_DESIGNER_ASSET_DIR = Path(__file__).resolve().parent / "Layout_Design"
_INFO_FILE = _DATA_DIR / "Info_storage" / "Information.json"


def _load_json(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_asset(name: str) -> str:
    p = _DESIGNER_ASSET_DIR / name
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return ""


def _build_root_css(info: dict) -> str:
    ui = {}
    try:
        ui = info.get("tiers", {}).get("style", {}).get("uiTheme", {})
    except Exception:
        pass
    if not isinstance(ui, dict):
        ui = {}
    _map = [
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
        ("--ui-font", "font", "'Inter', 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif"),
        ("--ui-transition", "transition", "border-color 0.15s, background-color 0.15s, color 0.15s"),
        ("--ui-scrollbar-track", "scrollbarTrack", "rgba(0, 0, 0, 0.78)"),
        ("--ui-scrollbar-thumb", "scrollbarThumb", "rgba(255, 255, 255, 0.12)"),
        ("--ui-scrollbar-thumb-hover", "scrollbarThumbHover", "rgba(255, 255, 255, 0.24)"),
        ("--ui-scrollbar-border", "scrollbarBorder", "rgba(0, 0, 0, 0.6)"),
    ]
    lines = [":root {"]
    for var, key, default in _map:
        val = ui.get(key, default)
        lines.append(f"  {var}: {val};")
    gv = info.get("tiers", {}).get("style", {}).get("gridView", {})
    if isinstance(gv, dict):
        bg = gv.get("background", "#252525")
        lines.append(f"  --style-grid-view-bg: {bg};")
    lines.append("}")
    return "\n".join(lines)


def _build_viewer_html(sim_data: dict, info: dict) -> str:
    root_css = _build_root_css(info)

    playback_js = _load_asset("sim-playback.js")

    sim_json = json.dumps(sim_data, ensure_ascii=False)

    layout = sim_data.get("layout", {})
    config = {
        "gridCols": layout.get("grid", {}).get("cols", 200),
        "gridRows": layout.get("grid", {}).get("rows", 200),
        "cellSize": layout.get("grid", {}).get("cellSize", 20.0),
        "layout": layout,
    }
    config_json = json.dumps(config, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
{root_css}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width:100%; height:100%; background: var(--ui-bg-base); color: var(--ui-text-primary); font-family: var(--ui-font); font-size:14px; overflow:hidden; }}
#viewer-app {{ position:absolute; inset:0; display:flex; flex-direction:column; }}
#viewer-canvas-wrap {{ flex:1; position:relative; background: var(--style-grid-view-bg, #252525); }}
#viewer-canvas {{ width:100%; height:100%; display:block; }}
#viewer-controls {{ display:flex; align-items:center; gap:8px; padding:8px 12px; background:var(--ui-bg-elevated); border-top:1px solid var(--ui-border-default); }}
#viewer-controls button {{ border:none; border-radius:4px; font-size:11px; font-weight:500; padding:6px 12px; background:var(--ui-bg-overlay); color:var(--ui-text-primary); cursor:pointer; transition:var(--ui-transition); }}
#viewer-controls button:hover {{ background:var(--ui-bg-surface); }}
#viewer-controls select {{ padding:4px 8px; font-size:11px; background:var(--ui-bg-input); color:var(--ui-text-primary); border:1px solid var(--ui-border-default); border-radius:4px; }}
#viewer-controls label {{ font-size:11px; color:var(--ui-text-secondary); }}
#viewer-slider {{ flex:1; min-width:80px; accent-color:var(--ui-accent); }}
#viewer-time-label {{ font-size:11px; min-width:72px; display:inline-block; }}
#viewer-info {{ position:absolute; top:8px; left:8px; font-size:11px; color:var(--ui-text-secondary); pointer-events:none; }}
  </style>
</head>
<body>
  <div id="viewer-app">
    <div id="viewer-canvas-wrap">
      <canvas id="viewer-canvas"></canvas>
      <div id="viewer-info"></div>
    </div>
    <div id="viewer-controls">
      <button id="btnPlay">Play</button>
      <button id="btnPause">Pause</button>
      <select id="playSpeed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="5">5x</option>
        <option value="10">10x</option>
        <option value="30">30x</option>
      </select>
      <label>Time</label>
      <input type="range" id="viewer-slider" min="0" max="1000" value="0" step="1" />
      <span id="viewer-time-label">00:00:00</span>
    </div>
  </div>

  <script>
  window.__SIM_VIEWER_CONFIG__ = {config_json};
  window.__SIM_PLAYBACK_DATA__ = {sim_json};
  </script>
  <script>
{playback_js}
  </script>
</body>
</html>"""
    return html


sim_data = _load_json(_SIM_OUTPUT_FILE)
info = _load_json(_INFO_FILE)

if not sim_data or not sim_data.get("positions"):
    st.warning("No simulation results found. Run a simulation from Layout Design first.")
    st.stop()

html = _build_viewer_html(sim_data, info)

bg_base = "#0d0d0f"
try:
    bg_base = info.get("tiers", {}).get("style", {}).get("uiTheme", {}).get("bgBase", "#0d0d0f")
except Exception:
    pass

st.markdown(f"""
<style>
  .stApp, [data-testid="stAppViewContainer"], section.main {{ background-color: {bg_base} !important; }}
  [data-testid="stHeader"], header[data-testid="stHeader"] {{ display: none !important; height: 0 !important; }}
  [data-testid="stAppViewContainer"] {{ padding: 0 !important; }}
  .block-container {{ padding: 0 !important; max-width: 100% !important; overflow: visible !important; min-height: 100vh !important; margin: 0 !important; }}
  section.main [data-testid="stVerticalBlock"] {{ padding-top: 0 !important; }}
  .block-container iframe, section.main iframe[title="streamlit_component"], section.main iframe[title="Streamlit component"] {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    min-height: 100vh !important;
    display: block !important;
    border: none !important;
    z-index: 0 !important;
  }}
</style>
""", unsafe_allow_html=True)

components.html(html, height=2000, scrolling=False)
