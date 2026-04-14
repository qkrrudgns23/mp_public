(function() {
  var _dc = window.__DESIGNER_CONFIG__;
  if (!_dc || typeof _dc !== 'object') { throw new Error('__DESIGNER_CONFIG__ missing'); }
  const LAYOUT_API_URL = _dc.layoutApiUrl;
  const GRID3D_ASSET_API_URL = _dc.grid3dAssetApiUrl != null ? String(_dc.grid3dAssetApiUrl) : '';
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
  const DEFAULT_SIBT_DATE = (function() {
    const s = (_dc.defaultFlightServiceDate != null) ? String(_dc.defaultFlightServiceDate).trim() : '';
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
    return '2026-03-31';
  })();
  /** Minutes from midnight for default SIBT time panel / empty Add Flight (07:00:00). */
  const DEFAULT_SIBT_TIME_MIN = 420;
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
  const _pathSearchTier = (_algoTier.pathSearch && typeof _algoTier.pathSearch === 'object') ? _algoTier.pathSearch : {};
  const _junctionMergeRadiusRaw = Number(_pathSearchTier.junctionMergeRadiusPx);
  const PATH_JUNCTION_MERGE_RADIUS_PX = (isFinite(_junctionMergeRadiusRaw) && _junctionMergeRadiusRaw >= 0) ? _junctionMergeRadiusRaw : 7;
  const _styleTier = _tiers.style || {};
  const _ganttStyle = _styleTier.gantt || {};
  const GANTT_VISIBLE_WINDOW_MIN = Math.max(60, Number(_ganttStyle.visibleWindowMin) || 1440);
  const GANTT_PAN_STEP_MIN = Math.max(15, Number(_ganttStyle.panStepMin) || 360);
  const _canvas2dStyle = _styleTier.canvas2d || {};
  const TAXIWAY_WIDTH_MIN = Math.max(1, Math.min(100, Number(_taxiwayTier.minWidth) || 1));
  const RUNWAY_EXIT_WIDTH_MIN = Math.max(1, Math.min(100, Number(_runwayExitTier.minWidth) || 1));
  const TAXIWAY_DEFAULT_WIDTH = Math.max(TAXIWAY_WIDTH_MIN, Math.min(100, Number(_taxiwayTier.width) || 1));
  const QUEUE_TAXIWAY_JUNCTION_SPACING_M = Math.max(5, Number(_taxiwayTier.queueJunctionSpacingM) || 40);
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
  function c2dTaxiwayPavementStroke() {
    const s = _canvas2dStyle.taxiwayPavementStroke;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#6d6a61';
  }
  function c2dTaxiwayPavementFill() {
    const s = _canvas2dStyle.taxiwayPavementFill;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#7b796d';
  }
  function c2dRunwayOutline() { return _canvas2dStyle.runwayOutline || '#cbd5e1'; }
  function c2dRunwayMarkingColor() { return _canvas2dStyle.runwayMarkingColor || '#f8fafc'; }
  function c2dRunwayThresholdColor() { return _canvas2dStyle.runwayThresholdColor || c2dRunwayMarkingColor(); }
  function c2dRunwayCenterlineColor() { return _canvas2dStyle.runwayCenterlineColor || c2dRunwayMarkingColor(); }
  function c2dRunwayTouchdownColor() { return _canvas2dStyle.runwayTouchdownColor || c2dRunwayMarkingColor(); }
  function c2dRunwayAimingPointColor() { return _canvas2dStyle.runwayAimingPointColor || c2dRunwayMarkingColor(); }
  function c2dRunwayExtensionFill() { return _canvas2dStyle.runwayExtensionFill || c2dRunwayStroke(); }
  function c2dRunwayBlastChevronColor() { return _canvas2dStyle.runwayBlastChevronColor || '#facc15'; }
  /** Strip alpha from rgba for solid road surface when showRoadWidth is on. */
  function c2dCssColorToOpaque(css) {
    const s = String(css || '').trim();
    const ra = s.match(/^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)/i);
    if (ra) return 'rgb(' + ra[1] + ',' + ra[2] + ',' + ra[3] + ')';
    return s;
  }
  const C2D_COLOR_SHADE_STEP_MUL = 0.88;
  function c2dParseCssRgbTriplet(css) {
    const s = String(css || '').trim();
    let m = s.match(/^#([0-9a-f]{3})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
    }
    m = s.match(/^#([0-9a-f]{6})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    }
    m = s.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
    return null;
  }
  function c2dCssColorLightenSteps(css, steps) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const n = Number(steps);
    if (!t || !(n > 0)) return opaque;
    const f = Math.pow(1 / C2D_COLOR_SHADE_STEP_MUL, n);
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Multiply RGB channels (e.g. 0.99 ≈ 1% darker). Expects opaque-ish CSS; alpha stripped first. */
  function c2dCssColorRgbChannelScale(css, mul) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const f = Number(mul);
    if (!t || !isFinite(f)) return opaque;
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Same rgb as layout marker kind=area fill (`drawLayoutAreaMarkers2DFloor`, 3 lighten steps). */
  function c2dRoadWidthBandSurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 3);
  }
  /** Taxiway / apron taxiway width band: one step darker than marker area (2 lighten steps vs runway stroke). */
  function c2dRoadWidthBandTaxiwaySurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 2);
  }
  /** Runway path & runway taxiway (runway_exit) width band: dark asphalt gray. */
  function c2dRoadWidthBandRunwayAsphaltColor() {
    return '#363636';
  }
  function pathPavementDefaultForPathType(pathType) {
    const pt = pathType || 'taxiway';
    if (pt === 'runway' || pt === 'runway_exit') return 'asphalt';
    return 'cement';
  }
  function pathPavementResolvedForTaxiway(tw) {
    if (!tw || typeof tw !== 'object') return 'cement';
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return v;
    return pathPavementDefaultForPathType(tw.pathType);
  }
  function c2dRoadWidthBandForPavement(pavement) {
    return pavement === 'cement' ? c2dRoadWidthBandTaxiwaySurfaceColor() : c2dRoadWidthBandRunwayAsphaltColor();
  }
  function normalizePathPavementInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return;
    tw.pavement = pathPavementDefaultForPathType(tw.pathType);
  }
  function islandMarkerPavementResolved(m) {
    if (!m || typeof m !== 'object') return 'asphalt';
    const pv = m.pavement;
    if (pv === 'asphalt' || pv === 'cement') return pv;
    const op = m.outerPavement;
    if (op === 'taxiway') return 'cement';
    if (op === 'runway') return 'asphalt';
    return 'asphalt';
  }
  function islandMarkerPavementFillCss(m) {
    return c2dRoadWidthBandForPavement(islandMarkerPavementResolved(m));
  }
  /** Selected road-width band only: ~1% darker than theme stroke/fill. */
  const ROAD_WIDTH_SURFACE_RGB_MUL = 0.99;
  function c2dObjectSelectedGlowBlur() {
    const n = Number(_canvas2dStyle.objectSelectedGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 22;
  }
  function c2dFlightSelectedRingStroke() { return _canvas2dStyle.flightSelectedRingStroke || '#facc15'; }
  function c2dFlightSelectedRingGlow() { return _canvas2dStyle.flightSelectedRingGlow || 'rgba(250, 204, 21, 0.55)'; }
  function c2dFlightSelectedRingGlowBlur() {
    const n = Number(_canvas2dStyle.flightSelectedRingGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 18;
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
  function c2dHoldingPointMarkingYellow() {
    return _canvas2dStyle.holdingPointMarkingYellow || '#facc15';
  }
  function c2dHoldingPointMarkingLineWidthWorld() {
    const n = Number(_canvas2dStyle.holdingPointMarkingLineWidthWorld);
    return (isFinite(n) && n > 0) ? n : 0.28;
  }
  function holdingPointMarkingDoubleLineGapM(lineW) {
    const n = Number(_canvas2dStyle.holdingPointMarkingDoubleLineGapM);
    const lw = Number(lineW);
    const baseLw = (isFinite(lw) && lw > 0) ? lw : c2dHoldingPointMarkingLineWidthWorld();
    return (isFinite(n) && n > 0) ? n : Math.max(0.28, baseLw * 1.2);
  }
  function taxiwayWorldWidthMForHolding(tw) {
    if (!tw) return TAXIWAY_DEFAULT_WIDTH;
    const typ = tw.pathType || 'taxiway';
    const base = typ === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (typ === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    return clampTaxiwayWidthM(typ, tw.width, base);
  }
  function holdingPointBarHalfLengthMFromPathWidth(pathWidthM) {
    const w = Number(pathWidthM);
    if (isFinite(w) && w > 0) return w * 0.5;
    return Math.max(3, c2dHoldingPointDiameterM() * 0.5);
  }
  function holdingPointPerpFromTangent(ux, uy) {
    return { px: -uy, py: ux };
  }
  function distPointToSegmentSq(x, y, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay;
    const apx = x - ax, apy = y - ay;
    const abLenSq = abx * abx + aby * aby;
    if (abLenSq < 1e-12) return apx * apx + apy * apy;
    let t = (apx * abx + apy * aby) / abLenSq;
    t = Math.max(0, Math.min(1, t));
    const qx = ax + t * abx, qy = ay + t * aby;
    const dx = x - qx, dy = y - qy;
    return dx * dx + dy * dy;
  }
  function findHoldingPointPathGeometry(hp) {
    const pt = [hp.x, hp.y];
    const wantRunway = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding';
    const maxD2 = Math.pow(Math.max(CELL_SIZE * 6, 55), 2);
    let bestD2 = Infinity;
    let ux = 1, uy = 0;
    let bestTw = null;
    (state.taxiways || []).forEach(function(tw) {
      const typ = tw.pathType || 'taxiway';
      if (wantRunway) {
