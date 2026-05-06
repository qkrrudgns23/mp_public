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
  const AIRPORTS_CATALOG = Array.isArray(_dc.airportsCatalog) ? _dc.airportsCatalog : [];
  const GRID_DRAW_VIEWPORT_MARGIN_CELLS = _dc.gridDrawViewportMarginCells;
  const GRID_MINOR_GRID_MIN_SCALE = _dc.gridMinorGridMinScale;
  let GRID_COLS = _dc.gridCols;
  let GRID_ROWS = _dc.gridRows;
  let CELL_SIZE = _dc.cellSize;
  function normalizeAirportCatalogRows(rows) {
    const out = [];
    (rows || []).forEach(function(r) {
      if (!r || typeof r !== 'object') return;
      const icao = String(r.icao || '').trim().toUpperCase();
      if (!icao) return;
      out.push({
        icao: icao,
        iata: String(r.iata || '').trim().toUpperCase(),
        name: String(r.name || '').trim(),
        country: String(r.country || '').trim(),
        city: String(r.city || '').trim()
      });
    });
    return out;
  }
  const AIRPORT_SEARCH_ROWS = normalizeAirportCatalogRows(AIRPORTS_CATALOG);

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
  const AIRCRAFT_TYPES = (typeof INFORMATION === 'object' && INFORMATION && INFORMATION.tiers && INFORMATION.tiers.aircraft && Array.isArray(INFORMATION.tiers.aircraft.types)) ? INFORMATION.tiers.aircraft.types : [];
  const AIRCRAFT_BY_ID = {};
  AIRCRAFT_TYPES.forEach(function(a) {
    const id = String(a.id || a.name || '').trim();
    if (id) AIRCRAFT_BY_ID[id] = a;
  });
  function getAircraftInfoByType(typeId) {
    return AIRCRAFT_BY_ID[typeId] || null;
  }
  function getCodeForAircraft(typeId) {
    const a = getAircraftInfoByType(typeId);
    if (a && a.icao != null) return String(a.icao).trim().toUpperCase()[0] || 'C';
    return 'C';
  }
  const ICAO_LETTERS_ORDER = ['A', 'B', 'C', 'D', 'E', 'F'];
  function normalizeAllowedIcaoCategories(raw) {
    const hit = {};
    (Array.isArray(raw) ? raw : []).forEach(function(x) {
      const c = String(x || '').trim().toUpperCase()[0];
      if (ICAO_LETTERS_ORDER.indexOf(c) >= 0) hit[c] = true;
    });
    return ICAO_LETTERS_ORDER.filter(function(c) { return hit[c]; });
  }
  function representativeCategoryFromLetters(letters) {
    const order = { A: 1, B: 2, C: 3, D: 4, E: 5, F: 6 };
    let best = 'C', bi = 0;
    normalizeAllowedIcaoCategories(letters).forEach(function(c) {
      const o = order[c] || 0;
      if (o > bi) { bi = o; best = c; }
    });
    return best;
  }
  function representativeCategoryFromAllowedTypes(typeIds) {
    const order = { A: 1, B: 2, C: 3, D: 4, E: 5, F: 6 };
    let best = 'C', bi = 0;
    (Array.isArray(typeIds) ? typeIds : []).forEach(function(tid) {
      const c = getCodeForAircraft(tid);
      const o = order[c] || 0;
      if (o > bi) { bi = o; best = c; }
    });
    return best;
  }
  function aircraftTypeIdsForIcaoLetters(letters) {
    const set = {};
    normalizeAllowedIcaoCategories(letters).forEach(function(c) { set[c] = true; });
    if (!Object.keys(set).length) return [];
    const out = [];
    AIRCRAFT_TYPES.forEach(function(a) {
      const id = String(a.id || a.name || '').trim();
      if (!id) return;
      const ic = String(a.icao || 'C').trim().toUpperCase()[0];
      if (set[ic] && out.indexOf(id) < 0) out.push(id);
    });
    return out;
  }
  function getAircraftConstraintOptionsForIcaoLetters(letters) {
    const set = {};
    normalizeAllowedIcaoCategories(letters).forEach(function(c) { set[c] = true; });
    if (!Object.keys(set).length) return [];
    const out = [];
    AIRCRAFT_TYPES.forEach(function(a) {
      const id = String(a.id || a.name || '').trim();
      if (!id) return;
      const ic = String(a.icao || 'C').trim().toUpperCase()[0];
      if (!set[ic]) return;
      const label = String(a.name || a.id || id || '').trim();
      out.push({ id: id, label: label || id });
    });
    return out;
  }
  function readIcaoCategoriesFromHost(hostId) {
    const host = typeof hostId === 'string' ? document.getElementById(hostId) : hostId;
    if (!host) return [];
    const sel = [];
    host.querySelectorAll('input[type="checkbox"].icao-letter-check').forEach(function(cb) {
      if (cb.checked) sel.push(cb.value);
    });
    return normalizeAllowedIcaoCategories(sel);
  }
  function applyIcaoCategoriesToHost(hostId, letters) {
    const host = typeof hostId === 'string' ? document.getElementById(hostId) : hostId;
    if (!host) return;
    const set = {};
    normalizeAllowedIcaoCategories(letters).forEach(function(c) { set[c] = true; });
    host.querySelectorAll('input[type="checkbox"].icao-letter-check').forEach(function(cb) {
      cb.checked = !!set[cb.value];
    });
  }
  function normalizeStandCategoryMode(rawMode, fallbackMode) {
    const mode = String(rawMode || fallbackMode || 'icao').trim().toLowerCase();
    return mode === 'aircraft' ? 'aircraft' : 'icao';
  }
  function standAllowedTypesMatchIcaoExpansion(panelTypes, letters) {
    const exp = aircraftTypeIdsForIcaoLetters(normalizeAllowedIcaoCategories(letters));
    const a = normalizeAllowedAircraftTypes(panelTypes).slice().sort().join('\0');
    const b = exp.slice().sort().join('\0');
    return a === b;
  }
  function deriveCategoryModeFromUnifiedStandPanel(panelTypes, allowedIcaoCategories) {
    const letters = normalizeAllowedIcaoCategories(allowedIcaoCategories);
    const pt = normalizeAllowedAircraftTypes(panelTypes);
    if (!pt.length || standAllowedTypesMatchIcaoExpansion(pt, letters)) return 'icao';
    return 'aircraft';
  }
  function readUnifiedNewStandConstraintFromPanel(icaoHostId, aircraftAccessId, defaultLettersIfEmpty) {
    let allowedIcaoCategories = readIcaoCategoriesFromHost(icaoHostId);
    if (!allowedIcaoCategories.length) allowedIcaoCategories = defaultLettersIfEmpty.slice();
    const expanded = aircraftTypeIdsForIcaoLetters(allowedIcaoCategories);
    let panelTypes = readCheckedDataItemIds(aircraftAccessId, '.aircraft-type-check');
    const categoryMode = deriveCategoryModeFromUnifiedStandPanel(panelTypes, allowedIcaoCategories);
    const allowedAircraftTypes = (!panelTypes.length || categoryMode === 'icao') ? expanded : panelTypes;
    const category = (categoryMode === 'aircraft' && panelTypes.length)
      ? representativeCategoryFromAllowedTypes(panelTypes)
      : representativeCategoryFromLetters(allowedIcaoCategories);
    return { allowedIcaoCategories: allowedIcaoCategories, allowedAircraftTypes: allowedAircraftTypes, categoryMode: categoryMode, category: category };
  }
  function applyUnifiedStandConstraintFromPanelToObject(stand, icaoHostId, aircraftAccessId) {
    let letters = readIcaoCategoriesFromHost(icaoHostId);
    if (!letters.length) letters = ['C'];
    stand.allowedIcaoCategories = letters;
    const expanded = aircraftTypeIdsForIcaoLetters(letters);
    let panelTypes = readCheckedDataItemIds(aircraftAccessId, '.aircraft-type-check');
    stand.categoryMode = deriveCategoryModeFromUnifiedStandPanel(panelTypes, letters);
    if (!panelTypes.length || stand.categoryMode === 'icao') {
      stand.allowedAircraftTypes = expanded;
      stand.category = representativeCategoryFromLetters(letters);
    } else {
      stand.allowedAircraftTypes = panelTypes;
      stand.category = panelTypes.length ? representativeCategoryFromAllowedTypes(panelTypes) : representativeCategoryFromLetters(letters);
    }
  }
  function panelRepresentativeCategoryForNewStand(which) {
    const hostId = which === 'pbb' ? 'standIcaoCategories' : (which === 'remote' ? 'remoteIcaoCategories' : 'tempStandIcaoCategories');
    const accId = which === 'pbb' ? 'standAircraftAccess' : (which === 'remote' ? 'remoteAircraftAccess' : 'tempStandAircraftAccess');
    let letters = readIcaoCategoriesFromHost(hostId);
    if (!letters.length) letters = ['C'];
    const types = readCheckedDataItemIds(accId, '.aircraft-type-check');
    if (types.length && !standAllowedTypesMatchIcaoExpansion(types, letters)) return representativeCategoryFromAllowedTypes(types);
    return representativeCategoryFromLetters(letters);
  }
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
  function getSimAircraftWorldDimsM(flight) {
    const ac = flight && flight.aircraftType ? getAircraftInfoByType(String(flight.aircraftType).trim()) : null;
    const lenR = ac && Number(ac.length_m);
    const wingR = ac && Number(ac.wingspan_m);
    const lenM = (isFinite(lenR) && lenR > 0) ? lenR : AIRCRAFT_FUSELAGE_LENGTH_M;
    const wingM = (isFinite(wingR) && wingR > 0) ? wingR : AIRCRAFT_WINGSPAN_M;
    return { lenM, wingM };
  }
  function detailedSilhouetteAxisSpans(silhouette2D) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < silhouette2D.length; i++) {
      const p = silhouette2D[i];
      if (!Array.isArray(p) || p.length < 2) continue;
      const x = Number(p[0]), y = Number(p[1]);
      if (!isFinite(x) || !isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) return { spanX: 1, spanY: 1 };
    return { spanX: Math.max(1e-9, maxX - minX), spanY: Math.max(1e-9, maxY - minY) };
  }
  const FLIGHT_TRAIL_LENGTH_M = Math.max(0, Number(_algoSimTier.trailLengthM) || 300);
  const PRE_TOUCHDOWN_HALO_ENABLED = (_algoSimTier.preTouchdownHaloEnabled !== false);
  const MAX_LAZY_TIMELINE_BUILDS_PER_FRAME = Math.max(1, Math.min(64, Number(_algoSimTier.maxLazyTimelineBuildsPerFrame) || 6));
  const _pathSearchTier = (_algoTier.pathSearch && typeof _algoTier.pathSearch === 'object') ? _algoTier.pathSearch : {};
  const _junctionMergeRadiusRaw = Number(_pathSearchTier.junctionMergeRadiusPx);
  const PATH_JUNCTION_MERGE_RADIUS_PX = (isFinite(_junctionMergeRadiusRaw) && _junctionMergeRadiusRaw >= 0) ? _junctionMergeRadiusRaw : 7;
  const _styleTier = _tiers.style || {};
  const _flightVizStyle = (_styleTier.flightVisualization && typeof _styleTier.flightVisualization === 'object') ? _styleTier.flightVisualization : {};
  const FLIGHT_SIM_VIZ_DEFAULT_PALETTE = [
    '#ff1493', '#39ff14', '#00f5ff', '#ff6600', '#ffffff', '#ff2d2d', '#ffff00', '#c026fc', '#2563eb', '#9ca3af',
  ];
  function flightSimVizPaletteList() {
    const p = _flightVizStyle.palette;
    if (Array.isArray(p) && p.length) {
      const out = [];
      for (let i = 0; i < p.length; i++) {
        const c = String(p[i] || '').trim();
        if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(c)) out.push(c);
      }
      if (out.length) return out;
    }
    return FLIGHT_SIM_VIZ_DEFAULT_PALETTE.slice();
  }
  function flightSimVizOverflowGray() {
    const g = _flightVizStyle.overflowGray;
    return (typeof g === 'string' && g.trim()) ? g.trim() : '#9ca3af';
  }
  const _ganttStyle = _styleTier.gantt || {};
  const GANTT_VISIBLE_WINDOW_MIN = Math.max(60, Number(_ganttStyle.visibleWindowMin) || 1440);
  const GANTT_PAN_STEP_MIN = Math.max(15, Number(_ganttStyle.panStepMin) || 360);
  const _canvas2dStyle = _styleTier.canvas2d || {};
  const TAXIWAY_WIDTH_MIN = Math.max(1, Math.min(100, Number(_taxiwayTier.minWidth) || 1));
  const RUNWAY_EXIT_WIDTH_MIN = Math.max(1, Math.min(100, Number(_runwayExitTier.minWidth) || 1));
  const TAXIWAY_DEFAULT_WIDTH = Math.max(TAXIWAY_WIDTH_MIN, Math.min(100, Number(_taxiwayTier.width) || 1));
  const QUEUE_TAXIWAY_JUNCTION_SPACING_M = Math.max(5, Number(_taxiwayTier.queueJunctionSpacingM) || 30);
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
  /** Queue junction spacing along path (matches backend designer_path_graph queue splits). */
  function taxiwayUsesQueueJunctionSpacing(tw) {
    if (!tw) return false;
    const pt = String(tw.pathType || '');
    if (pt === 'general_queue_taxiway') return true;
    if (pt === 'runway_exit' || pt === 'runway_taxiway') return tw.queueFlow !== false;
    return false;
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
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#827f76';
  }
  function c2dTaxiwayPavementFill() {
    const s = _canvas2dStyle.taxiwayPavementFill;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#908e82';
  }
  function c2dRunwayOutline() { return _canvas2dStyle.runwayOutline || '#cbd5e1'; }
