(function() {
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
  function readCheckedDataItemIds(rootOrId, selectorClass) {
    const root = typeof rootOrId === 'string' ? document.getElementById(rootOrId) : rootOrId;
    const scope = root || document;
    return Array.from(scope.querySelectorAll(selectorClass)).filter(function(ch) { return ch.checked; }).map(function(ch) { return String(ch.getAttribute('data-item-id') || '').trim(); }).filter(Boolean);
  }
  function flightBlockedLikeNoWay(f) {
    if (!f) return false;
    if (f.noWayArr || f.noWayDep) return true;
    return f.arrDep !== 'Dep' && !!f.arrRetFailed;
  }
  function arrivalAirsideBlocked(f) {
    return !!(f && f.arrDep !== 'Dep' && (f.noWayArr || f.arrRetFailed));
  }
  window.flightBlockedLikeNoWay = flightBlockedLikeNoWay;

  const _tiers = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers) ? INFORMATION.tiers : {};
  const _layoutTier = _tiers.layout || {};
  const _pbbTier = _layoutTier.pbb || {};
  const _remoteTier = _layoutTier.remote || {};
  const _taxiwayTier = _layoutTier.taxiway || {};
  const _runwayPathTier = _layoutTier.runwayPath || {};
  const _runwayExitTier = _layoutTier.runwayExit || {};
  const _flightTier = _tiers.flight_schedule || _tiers.flight || {};
  const SCHED_DEP_ROT_MIN = Math.max(0, Number(_flightTier.depRotMin) || 2);
  const DEP_LINEUP_HOLD_SEC = Math.max(0, Number(_flightTier.depLineupHoldSec) != null && isFinite(Number(_flightTier.depLineupHoldSec)) ? Number(_flightTier.depLineupHoldSec) : 20);
  const DEP_TAKEOFF_ACCEL_SMALL_MS2 = Math.max(0.1, Number(_flightTier.depTakeoffAccelSmallMs2) || 2.5);
  const DEP_TAKEOFF_ACCEL_LARGE_MS2 = Math.max(0.1, Number(_flightTier.depTakeoffAccelLargeMs2) || 2.0);
  const DEP_MTOW_REF_SMALL_KG = Math.max(1, Number(_flightTier.depTakeoffAccelMtowRefSmallKg) || 50000);
  const DEP_MTOW_REF_LARGE_KG = Math.max(DEP_MTOW_REF_SMALL_KG + 1, Number(_flightTier.depTakeoffAccelMtowRefLargeKg) || 350000);
  const APRON_TAXIWAY_SPEED_MS = Math.max(0.1, Number(_flightTier.apronTaxiwaySpeedMs) || 1.5);
  const SIM_TIME_SLIDER_SNAP_SEC = Math.max(1, Number(_dc.flightSimSliderSnapSec) || 1);
  const DEFAULT_ALLOW_RUNWAY_IN_GROUND_SEGMENT = _dc.defaultAllowRunwayInGroundSegment;
  const _algoTier = _tiers.algorithm || {};
  const _algoSimTier = (_algoTier.simulation && typeof _algoTier.simulation === 'object') ? _algoTier.simulation : {};
  const APPROACH_OFFSET_WORLD_M = Math.max(0, Number(_algoSimTier.approachOffsetM) || 10000);
  const APPROACH_STRAIGHT_FINAL_M = Math.max(0, Number(_algoSimTier.approachStraightFinalM) || 3000);
  const AIRCRAFT_WINGSPAN_M = Math.max(1, Number(_algoSimTier.aircraftWingspanM) || 40);
  const AIRCRAFT_FUSELAGE_LENGTH_M = Math.max(1, Number(_algoSimTier.aircraftFuselageLengthM) || 50);
  const FLIGHT_TRAIL_LENGTH_M = Math.max(0, Number(_algoSimTier.trailLengthM) || 300);
  const PRE_TOUCHDOWN_HALO_ENABLED = (_algoSimTier.preTouchdownHaloEnabled !== false);
  const PLAYBACK_LEAD_BEFORE_FIRST_TD_SEC = Math.max(0, Number(_algoSimTier.playbackLeadBeforeFirstTouchdownSec) || 0);
  const MAX_LAZY_TIMELINE_BUILDS_PER_FRAME = Math.max(1, Math.min(64, Number(_algoSimTier.maxLazyTimelineBuildsPerFrame) || 6));
  const _styleTier = _tiers.style || {};
  const _ganttStyle = _styleTier.gantt || {};
  const GANTT_VISIBLE_WINDOW_MIN = Math.max(60, Number(_ganttStyle.visibleWindowMin) || 360);
  const GANTT_PAN_STEP_MIN = Math.max(15, Number(_ganttStyle.panStepMin) || 120);
  const _canvas2dStyle = _styleTier.canvas2d || {};
  const TAXIWAY_WIDTH_MIN = Math.max(1, Math.min(100, Number(_taxiwayTier.minWidth) || 1));
  const RUNWAY_EXIT_WIDTH_MIN = Math.max(1, Math.min(100, Number(_runwayExitTier.minWidth) || 1));
  const TAXIWAY_DEFAULT_WIDTH = Math.max(TAXIWAY_WIDTH_MIN, Math.min(100, Number(_taxiwayTier.width) || 1));
  const RUNWAY_PATH_DEFAULT_WIDTH = Math.max(5, Math.min(100, Number(_runwayPathTier.width) || 60));
  const RUNWAY_EXIT_DEFAULT_WIDTH = Math.max(RUNWAY_EXIT_WIDTH_MIN, Math.min(100, Number(_runwayExitTier.width) || 1));
  function minWidthMForTaxiwayPathType(pathType) {
    if (pathType === 'runway') return 5;
    if (pathType === 'runway_exit') return RUNWAY_EXIT_WIDTH_MIN;
    return TAXIWAY_WIDTH_MIN;
  }
  function clampTaxiwayWidthM(pathType, val, baseWidth) {
    const lo = minWidthMForTaxiwayPathType(pathType);
    const raw = Number(val);
    const use = (isFinite(raw) && raw > 0) ? raw : baseWidth;
    return Math.max(lo, Math.min(100, use));
  }
  function normalizeTaxiwayWidthInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    const pt = tw.pathType || 'taxiway';
    const fb = pt === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (pt === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    if (tw.width != null) tw.width = clampTaxiwayWidthM(pt, tw.width, fb);
  }
  const RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.startDisplacedThresholdM) || 100);
  const RUNWAY_START_BLAST_PAD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.startBlastPadM) || 100);
  const RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.endDisplacedThresholdM) || 100);
  const RUNWAY_END_BLAST_PAD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.endBlastPadM) || 100);
  function c2dObjectSelectedStroke() { return _canvas2dStyle.objectSelectedStroke || 'rgba(233, 213, 255, 0.62)'; }
  function c2dObjectSelectedFill() { return _canvas2dStyle.objectSelectedFill || 'rgba(196, 181, 253, 0.28)'; }
  function c2dObjectSelectedDashStroke() { return _canvas2dStyle.objectSelectedDashStroke || 'rgba(255, 252, 255, 0.55)'; }
  function c2dObjectSelectedGlow() { return _canvas2dStyle.objectSelectedGlow || 'rgba(167, 139, 250, 0.45)'; }
  function c2dRunwayStroke() { return _canvas2dStyle.runwayStroke || 'rgba(156, 163, 175, 0.78)'; }
  function c2dRunwayFill() { return _canvas2dStyle.runwayFill || 'rgba(75, 85, 99, 0.78)'; }
  function c2dRunwayOutline() { return _canvas2dStyle.runwayOutline || '#cbd5e1'; }
  function c2dRunwayMarkingColor() { return _canvas2dStyle.runwayMarkingColor || '#f8fafc'; }
  function c2dRunwayThresholdColor() { return _canvas2dStyle.runwayThresholdColor || c2dRunwayMarkingColor(); }
  function c2dRunwayCenterlineColor() { return _canvas2dStyle.runwayCenterlineColor || c2dRunwayMarkingColor(); }
  function c2dRunwayTouchdownColor() { return _canvas2dStyle.runwayTouchdownColor || c2dRunwayMarkingColor(); }
  function c2dRunwayAimingPointColor() { return _canvas2dStyle.runwayAimingPointColor || c2dRunwayMarkingColor(); }
  function c2dRunwayExtensionFill() { return _canvas2dStyle.runwayExtensionFill || 'rgba(55, 65, 81, 0.78)'; }
  function c2dRunwayBlastChevronColor() { return _canvas2dStyle.runwayBlastChevronColor || '#facc15'; }
  function c2dObjectSelectedGlowBlur() {
    const n = Number(_canvas2dStyle.objectSelectedGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 22;
  }
  function c2dSimPreTouchdownHaloStroke() { return _canvas2dStyle.simPreTouchdownHaloStroke || 'rgba(239, 68, 68, 0.92)'; }
  function c2dSimPreTouchdownHaloFill() { return _canvas2dStyle.simPreTouchdownHaloFill || 'rgba(239, 68, 68, 0.18)'; }
  function c2dSimPreTouchdownHaloBlur() {
    const n = Number(_canvas2dStyle.simPreTouchdownHaloBlur);
    return (isFinite(n) && n >= 0) ? n : 14;
  }
  function c2dSimFlightTrailStroke() { return _canvas2dStyle.simFlightTrailStroke || 'rgba(255, 47, 146, 0.97)'; }
  function c2dSimFlightTrailStrokeEnd() { return _canvas2dStyle.simFlightTrailStrokeEnd || 'rgba(255, 47, 146, 0)'; }
  function c2dSimFlightTrailLineWidth() {
    const n = Number(_canvas2dStyle.simFlightTrailLineWidth);
    return (isFinite(n) && n > 0) ? n : 3.5;
  }
  function c2dApproachPreviewWidthM() {
    const n = Number(_canvas2dStyle.approachPreviewWidthM);
    return (isFinite(n) && n > 0) ? n : 30;
  }
  function c2dApproachPreviewStroke() {
    return _canvas2dStyle.approachPreviewStroke || 'rgba(255, 255, 255, 0.01)';
  }
  function c2dHoldingPointDiameterM() {
    const n = Number(_canvas2dStyle.holdingPointDiameterM);
    return (isFinite(n) && n > 0) ? n : 15;
  }
  function normalizeHoldingPointKind(raw) {
    return raw === 'runway_holding' ? 'runway_holding' : 'intermediate';
  }
  function pathTypeToHpKind(pathType) {
    return pathType === 'runway_exit' ? 'runway_holding' : 'intermediate';
  }
  function holdingPointKindDisplayLabel(kind) {
    return normalizeHoldingPointKind(kind) === 'runway_holding' ? 'Runway Holding Position' : 'Intermediate Holding Position';
  }
  function c2dHoldingPointFillForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayFill || 'rgba(239, 68, 68, 0.5)';
    return _canvas2dStyle.holdingPointIntermediateFill || 'rgba(249, 115, 22, 0.5)';
  }
  function c2dHoldingPointStrokeForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dHoldingPointPreviewFillForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayPreviewFill || 'rgba(239, 68, 68, 0.28)';
    return _canvas2dStyle.holdingPointIntermediatePreviewFill || 'rgba(249, 115, 22, 0.28)';
  }
  function c2dHoldingPointPreviewStrokeForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dSimStandOccupiedFill() { return _canvas2dStyle.simStandOccupiedFill || 'rgba(239, 68, 68, 0.32)'; }
  function c2dSimStandOccupiedStroke() { return _canvas2dStyle.simStandOccupiedStroke || 'rgba(220, 38, 38, 0.95)'; }
  function c2dPathDrawStartMarkerRadiusPx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerRadiusPx);
    const base = (isFinite(n) && n > 0) ? n : 3.5;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  function c2dPathDrawStartMarkerStrokePx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerStrokePx);
    const base = (isFinite(n) && n > 0) ? n : 1;
    return Math.max(0.5, base * LAYOUT_VERTEX_DOT_SCALE);
  }
  function c2dPathDrawStartLabelFontPx() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelFontPx);
    const base = (isFinite(n) && n >= 6) ? n : 8;
    return Math.max(6, Math.round(base * LAYOUT_VERTEX_DOT_SCALE));
  }
  function c2dPathDrawStartLabelOffsetY() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelOffsetY);
    const base = isFinite(n) ? n : -6;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  const GANTT_COLORS = {
    S_BAR: _ganttStyle.sBar || '#007aff',
    S_SERIES: _ganttStyle.sSeries || '#38bdf8',
    E_BAR: _ganttStyle.eBar || '#fb37c5',
    E_SERIES: _ganttStyle.eSeries || '#fb923c',
    CONFLICT: _ganttStyle.conflict || '#7f1d1d',
    SELECTED: _ganttStyle.selected || '#fbbf24',
  };
  const _apronAc = _layoutTier.apronAircraft || {};
  const _acScaleByCat = (_apronAc.scaleByIcaoCategory && typeof _apronAc.scaleByIcaoCategory === 'object') ? _apronAc.scaleByIcaoCategory : {};
  function apronAircraftScaleForIcao(code) {
    const c = String(code || '').toUpperCase();
    const v = Number(_acScaleByCat[c]);
    if (isFinite(v) && v > 0) return v;
    const d = Number(_acScaleByCat.default);
    return (isFinite(d) && d > 0) ? d : 1.0;
  }
  const _ac2d = _apronAc.twoD || {};
  const _acSil = (_ac2d.silhouette && typeof _ac2d.silhouette === 'object') ? _ac2d.silhouette : {};
  function apron2DGlyphFill() { return _ac2d.fillColor || '#ff2f92'; }
  const _schedAlgo = _algoTier.scheduledTimes || {};
  const SCHED_DWELL_FLOOR_MIN = (function() {
    const v = Number(_schedAlgo.dwellFloorMin);
    return (isFinite(v) && v >= 0) ? v : 20;
  })();
  const RSEP_MISSING_MATRIX_SEC = (function() {
    const v = Number(_schedAlgo.rsepMissingMatrixSeparationSec);
    return (isFinite(v) && v >= 0) ? v : 90;
  })();
  const TIME_AXIS_CFG = _algoTier.timeAxis || {};
  const DOM_OPT_CFG = (_algoTier.domOptimization && typeof _algoTier.domOptimization === 'object') ? _algoTier.domOptimization : {};
  const DOM_OPT_FLIGHT_VIRT_ENABLE = DOM_OPT_CFG.flightListVirtualScroll !== false;
  const DOM_OPT_FLIGHT_VIRT_MIN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualMinRows));
    return (isFinite(v) && v >= 8) ? v : 48;
  })();
  const DOM_OPT_FLIGHT_VIRT_OVERSCAN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualOverscan));
    return (isFinite(v) && v >= 0) ? v : 8;
  })();
  const DOM_OPT_FLIGHT_VIRT_ROW_H = (function() {
    const v = Number(DOM_OPT_CFG.flightListVirtualRowHeightPx);
    return (isFinite(v) && v >= 18) ? v : 28;
  })();
  const FLIGHT_SCHED_PAGE_SIZE = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightSchedulePageSize));
    if (!isFinite(v) || v < 0) return 20;
    return v;
  })();
  const GANTT_LEGEND_MAX_INTERVALS = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.ganttLegendMaxIntervals));
    if (!isFinite(v) || v < 1) return 100;
    return v;
  })();
  const KPI_ROLLING_TABLE_VISIBLE_ROWS = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.kpiRollingTableVisibleRows));
    if (!isFinite(v) || v < 1) return 24;
    return v;
  })();
  function _taNum(k, def) {
    const v = Number(TIME_AXIS_CFG[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  const GANTT_PAD_MIN = _taNum('apronGanttPadMin', 20);
  const RWY_SEP_TIMELINE_PAD_MIN = _taNum('runwaySepTimelinePadMin', 10);
  const TICK_STEP_SPAN_LE60 = _taNum('tickStepWhenSpanLe60Min', 10);
  const TICK_STEP_SPAN_LE240 = _taNum('tickStepWhenSpanLe240Min', 30);
  const TICK_STEP_ELSE = _taNum('tickStepElseMin', 60);
  const MAX_TICKS_SHOWN = (function() {
    const v = Math.floor(Number(TIME_AXIS_CFG.maxTicksShown));
    return (isFinite(v) && v >= 2) ? v : 6;
  })();
  const PATH_SEARCH_CFG = _algoTier.pathSearch || {};
  const TAXIWAY_HEURISTIC_COST = (function() {
    const v = Number(PATH_SEARCH_CFG.taxiwayHeuristicCost);
    return (isFinite(v) && v > 0) ? v : 200;
  })();
  const _ix = _layoutTier.interaction || {};
  function _interactionConfigNum(k, def) {
    const v = Number(_ix[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  function _ixBool(k, def) {
    const v = _ix[k];
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') {
      const s = v.trim().toLowerCase();
      if (s === 'true' || s === '1' || s === 'yes' || s === 'on') return true;
      if (s === 'false' || s === '0' || s === 'no' || s === 'off') return false;
    }
    return !!def;
  }
  const LAYOUT_VERTEX_DOT_SCALE = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutVertexDotScale', 0.7)));
  const LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutSelectedVertexRadiusFactor', 0.7)));
  const GRID_VISIBLE_DEFAULT = _ixBool('showGridDefault', true);
  const IMAGE_VISIBLE_DEFAULT = _ixBool('showImageDefault', true);
  const ROAD_WIDTH_VISIBLE_DEFAULT = _ixBool('showRoadWidthDefault', true);
  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections(_dc.rwExitAllowedDefaultRaw);
  function layoutPathVertexRadiusPx(vertexSelected, pathSelected) {
    if (vertexSelected) return 6 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    if (pathSelected) return 5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    return 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  function layoutTerminalVertexRadiusPx(vertexSelected) {
    return vertexSelected ? 5.5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR : 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  const DRAG_THRESH = _interactionConfigNum('dragThresholdPx', 0);
  const FREE_DRAW_STEP_CELL = Math.max(0.001, _interactionConfigNum('freeDrawStepCell', 0.05));
  const GRID_SNAP_STEP_CELL = Math.max(0.001, _interactionConfigNum('gridSnapStepCell', 0.5));
  const INSERT_VERTEX_HIT_CF = _interactionConfigNum('insertVertexHitCellFactor', 0.9);
  const CANVAS_MIN_ZOOM = Math.max(0.01, _interactionConfigNum('canvasMinZoom', 0.05));
  const CANVAS_MAX_ZOOM = Math.max(CANVAS_MIN_ZOOM, _interactionConfigNum('canvasMaxZoom', 10));
  const HIT_TERM_VTX_CF = _interactionConfigNum('hitTerminalVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_VTX_CF = _interactionConfigNum('hitTaxiwayVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_SEG_CF = _interactionConfigNum('hitTaxiwayAlongCellFactor', 0.8);
  const HIT_PBB_END_CF = _interactionConfigNum('hitPbbEndCellFactor', 0.8);
  const TRY_PBB_MAX_EDGE_CF = _interactionConfigNum('tryPlacePbbMaxEdgeCellFactor', 1.0);
  const FLIGHT_TOOLTIP_CF = _interactionConfigNum('flightTooltipCellFactor', 1.2);
  const FLIGHT_TOOLTIP_SCAN_MIN_MS = _interactionConfigNum('flightTooltipScanMinIntervalMs', 50);
  const TERM_CLOSE_POLY_CF = _interactionConfigNum('terminalClosePolygonCellFactor', 0.6);
  const PBB_PREVIEW_LEN_CF = _interactionConfigNum('pbbPreviewLengthCellFactor', 0.9);

  const canvas = document.getElementById('grid-canvas');
  const container = document.getElementById('canvas-container');
  const coordEl = document.getElementById('coord');
  const objectInfoEl = document.getElementById('object-info');
  const objectListEl = document.getElementById('object-list');
  const flightTooltip = document.getElementById('flight-tooltip');
  const settingModeSelect = document.getElementById('settingMode');
  const layoutModeTabs = document.getElementById('layoutModeTabs');
  const panel = document.getElementById('right-panel');
  const panelToggle = document.getElementById('panel-toggle');
  const resetViewBtn = document.getElementById('btnResetView');
  const gridToggleBtn = document.getElementById('btnGridToggle');
  const imageToggleBtn = document.getElementById('btnImageToggle');
  const roadWidthToggleBtn = document.getElementById('btnRoadWidthToggle');
  const GRID_LAYOUT_IMAGE_DEFAULTS = {
    opacity: _dc.gridLayoutImage.opacity,
    opacityMin: _dc.gridLayoutImage.opacityMin,
    opacityMax: _dc.gridLayoutImage.opacityMax,
    widthM: _dc.gridLayoutImage.widthM,
    heightM: _dc.gridLayoutImage.heightM,
    topLeftCol: _dc.gridLayoutImage.topLeftCol,
    topLeftRow: _dc.gridLayoutImage.topLeftRow
  };
