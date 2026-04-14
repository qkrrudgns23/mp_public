  const TRY_PBB_MAX_EDGE_CF = _interactionConfigNum('tryPlacePbbMaxEdgeCellFactor', 1.0);
  const PBB_STAND_CENTER_OFFSET_FROM_TERMINAL_WALL_M = 50;
  const FLIGHT_TOOLTIP_CF = _interactionConfigNum('flightTooltipCellFactor', 1.2);
  const FLIGHT_TOOLTIP_SCAN_MIN_MS = _interactionConfigNum('flightTooltipScanMinIntervalMs', 50);
  const TERM_CLOSE_POLY_CF = _interactionConfigNum('terminalClosePolygonCellFactor', 0.6);
  const PBB_PREVIEW_LEN_CF = _interactionConfigNum('pbbPreviewLengthCellFactor', 0.9);

  const canvas = document.getElementById('grid-canvas');
  if (canvas) {
    canvas.draggable = false;
    canvas.setAttribute('tabindex', '-1');
    canvas.style.outline = 'none';
  }
  function focusCanvasForLayoutHotkeys() {
    if (!canvas) return;
    try {
      canvas.focus({ preventScroll: true });
    } catch (e) {
      try { canvas.focus(); } catch (e2) {}
    }
  }
  const container = document.getElementById('canvas-container');
  const coordEl = document.getElementById('coord');
  const cursorPixelReadoutEl = document.getElementById('cursor-pixel-readout');
  const objectInfoEl = document.getElementById('object-info');
  const objectListEl = document.getElementById('object-list');
  const flightTooltip = document.getElementById('flight-tooltip');
  let _layoutReadoutLastCellKey = '';
  let _layoutReadoutLastPixelStr = '';
  let _layoutTooltipRafId = 0;
  let _layoutTooltipPending = null;
  const settingModeSelect = document.getElementById('settingMode');
  const layoutModeTabs = document.getElementById('layoutModeTabs');
  const panel = document.getElementById('right-panel');
  const panelToggle = document.getElementById('panel-toggle');
  const MARKER_BLAZER_COLOR_OPTIONS = ['#a78bfa', '#22d3ee', '#4ade80', '#f59e0b', '#f43f5e'];
  const markerFlightBlazerOverlayBtn = document.createElement('button');
  const markerFlightHeadingOverlayBtn = document.createElement('button');
  const markerFlightBlazerPaletteWrap = document.createElement('div');
  markerFlightBlazerOverlayBtn.type = 'button';
  markerFlightBlazerOverlayBtn.textContent = 'Blazer: OFF';
  markerFlightBlazerOverlayBtn.setAttribute('aria-label', 'Toggle flight marker blazer');
  markerFlightBlazerOverlayBtn.style.position = 'absolute';
  markerFlightBlazerOverlayBtn.style.zIndex = '35';
  markerFlightBlazerOverlayBtn.style.display = 'none';
  markerFlightBlazerOverlayBtn.style.padding = '6px 10px';
  markerFlightBlazerOverlayBtn.style.border = '1px solid var(--ui-border-default)';
  markerFlightBlazerOverlayBtn.style.borderRadius = '6px';
  markerFlightBlazerOverlayBtn.style.background = 'var(--ui-bg-control)';
  markerFlightBlazerOverlayBtn.style.color = 'var(--ui-text-primary)';
  markerFlightBlazerOverlayBtn.style.cursor = 'pointer';
  markerFlightBlazerOverlayBtn.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightHeadingOverlayBtn.type = 'button';
  markerFlightHeadingOverlayBtn.textContent = 'Heading: FWD';
  markerFlightHeadingOverlayBtn.setAttribute('aria-label', 'Toggle flight marker heading');
  markerFlightHeadingOverlayBtn.style.position = 'absolute';
  markerFlightHeadingOverlayBtn.style.zIndex = '35';
  markerFlightHeadingOverlayBtn.style.display = 'none';
  markerFlightHeadingOverlayBtn.style.padding = '6px 10px';
  markerFlightHeadingOverlayBtn.style.border = '1px solid var(--ui-border-default)';
  markerFlightHeadingOverlayBtn.style.borderRadius = '6px';
  markerFlightHeadingOverlayBtn.style.background = 'var(--ui-bg-control)';
  markerFlightHeadingOverlayBtn.style.color = 'var(--ui-text-primary)';
  markerFlightHeadingOverlayBtn.style.cursor = 'pointer';
  markerFlightHeadingOverlayBtn.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightBlazerPaletteWrap.style.position = 'absolute';
  markerFlightBlazerPaletteWrap.style.zIndex = '35';
  markerFlightBlazerPaletteWrap.style.display = 'none';
  markerFlightBlazerPaletteWrap.style.gap = '6px';
  markerFlightBlazerPaletteWrap.style.alignItems = 'center';
  markerFlightBlazerPaletteWrap.style.padding = '4px 6px';
  markerFlightBlazerPaletteWrap.style.border = '1px solid var(--ui-border-default)';
  markerFlightBlazerPaletteWrap.style.borderRadius = '6px';
  markerFlightBlazerPaletteWrap.style.background = 'var(--ui-bg-control)';
  markerFlightBlazerPaletteWrap.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightBlazerPaletteWrap.style.pointerEvents = 'auto';
  markerFlightBlazerPaletteWrap.style.display = 'none';
  markerFlightBlazerPaletteWrap.style.flexDirection = 'row';
  function swallowBlazerOverlayPointer(ev) {
    if (!ev) return;
    ev.preventDefault();
    ev.stopPropagation();
  }
  MARKER_BLAZER_COLOR_OPTIONS.forEach(function(c) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('data-blazer-color', c);
    b.style.width = '14px';
    b.style.height = '14px';
    b.style.minWidth = '14px';
    b.style.borderRadius = '2px';
    b.style.border = '1px solid rgba(255,255,255,0.45)';
    b.style.background = c;
    b.style.cursor = 'pointer';
    b.style.padding = '0';
    b.style.margin = '0';
    markerFlightBlazerPaletteWrap.appendChild(b);
  });
  if (container) {
    container.appendChild(markerFlightBlazerOverlayBtn);
    container.appendChild(markerFlightHeadingOverlayBtn);
    container.appendChild(markerFlightBlazerPaletteWrap);
  }
  const resetViewBtn = document.getElementById('btnResetView');
  const layerPopoverBtn = document.getElementById('btnLayerPopover');
  const layerPopoverPanel = document.getElementById('layerPopoverPanel');
  const layerPopoverWrap = document.getElementById('layerPopoverWrap');
  markerFlightBlazerOverlayBtn.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightBlazerOverlayBtn.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightHeadingOverlayBtn.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightHeadingOverlayBtn.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightBlazerPaletteWrap.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightBlazerPaletteWrap.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightBlazerOverlayBtn.addEventListener('click', function() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj || sel.obj.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(sel.obj);
    sel.obj.blazerEnabled = !sel.obj.blazerEnabled;
    if (sel.obj.blazerEnabled) appendMarkerFlightBlazerTrail(sel.obj);
    scheduleDraw();
    updateObjectInfo();
  });
  markerFlightBlazerPaletteWrap.addEventListener('click', function(ev) {
    const target = ev.target;
    if (!target || !target.getAttribute) return;
    const next = String(target.getAttribute('data-blazer-color') || '').trim();
    if (MARKER_BLAZER_COLOR_OPTIONS.indexOf(next) < 0) return;
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj || sel.obj.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(sel.obj);
    sel.obj.blazerColor = next;
    scheduleDraw();
    updateObjectInfo();
  });
  markerFlightHeadingOverlayBtn.addEventListener('click', function() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj || sel.obj.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(sel.obj);
    sel.obj.headingReversed = !sel.obj.headingReversed;
    scheduleDraw();
    updateObjectInfo();
  });
  const GRID_LAYOUT_IMAGE_DEFAULTS = {
    opacity: _dc.gridLayoutImage.opacity,
    opacityMin: _dc.gridLayoutImage.opacityMin,
    opacityMax: _dc.gridLayoutImage.opacityMax,
    widthM: _dc.gridLayoutImage.widthM,
    heightM: _dc.gridLayoutImage.heightM,
    topLeftCol: _dc.gridLayoutImage.topLeftCol,
    topLeftRow: _dc.gridLayoutImage.topLeftRow
  };
  let layoutImageBitmap = null;
  let layoutImageBitmapSrc = '';
  const BUILDING_TYPE_CFG = (_layoutTier.building && typeof _layoutTier.building === 'object') ? _layoutTier.building : {};
  const BUILDING_TYPES = Array.isArray(BUILDING_TYPE_CFG.types) && BUILDING_TYPE_CFG.types.length ? BUILDING_TYPE_CFG.types.slice() : [
    { id: 'passenger_terminal', label: 'Passenger Terminal' },
    { id: 'concourse', label: '위성터미널(콘코스)' },
    { id: 'control_tower', label: 'Control Tower' },
    { id: 'cargo_terminal', label: 'Cargo Terminal' },
    { id: 'hanger', label: 'Hanger' },


    { id: 'utility', label: 'Utility' },
    { id: 'wall', label: 'Wall' },
  ];
  const BUILDING_TYPE_DEFAULT = String(BUILDING_TYPE_CFG.defaultType || (BUILDING_TYPES[0] && BUILDING_TYPES[0].id) || 'passenger_terminal');
  const BUILDING_TYPE_BY_ID = {};
  BUILDING_TYPES.forEach(function(bt) { BUILDING_TYPE_BY_ID[String(bt.id || '')] = bt; });
  function normalizeBuildingType(rawType) {
    const key = String(rawType || '').trim();
    if (key && BUILDING_TYPE_BY_ID[key]) return key;
    return BUILDING_TYPE_DEFAULT;
  }
  function getBuildingTypeMeta(rawType) {
    return BUILDING_TYPE_BY_ID[normalizeBuildingType(rawType)] || BUILDING_TYPE_BY_ID[BUILDING_TYPE_DEFAULT] || { id: BUILDING_TYPE_DEFAULT, label: 'Passenger Terminal' };
  }
  function getBuildingTypeLabel(rawType) {
    const meta = getBuildingTypeMeta(rawType);
    return String(meta.label || meta.id || 'Building');
  }
  function getBuildingTypeNamePrefix(rawType) {
    const key = normalizeBuildingType(rawType);
    if (key === 'passenger_terminal') return 'Terminal';
    if (key === 'concourse') return 'Concourse';
    if (key === 'control_tower') return 'Tower';
    if (key === 'cargo_terminal') return 'Cargo';
    if (key === 'hanger') return 'Hanger';
    if (key === 'utility') return 'Utility';
    if (key === 'wall') return 'Wall';
    return 'Building';
  }
  function getBuildingTypeOptionsHtml(selectedType) {
    const current = normalizeBuildingType(selectedType);
    return BUILDING_TYPES.map(function(bt) {
      const id = String(bt.id || '');
      const label = String(bt.label || bt.id || id || 'Building');
      return '<option value="' + escapeHtml(id) + '"' + (id === current ? ' selected' : '') + '>' + escapeHtml(label) + '</option>';
    }).join('');
  }
  function getBuildingTheme(building) {
    const key = normalizeBuildingType(building && building.buildingType);
    const themes = (_canvas2dStyle.buildingTypes && typeof _canvas2dStyle.buildingTypes === 'object') ? _canvas2dStyle.buildingTypes : {};
    const theme = (themes && typeof themes[key] === 'object') ? themes[key] : {};
    return {
      key: key,
      label: getBuildingTypeLabel(key),
      stroke: theme.stroke || _canvas2dStyle.terminalStrokeDefault || '#0284c7',
      fill: theme.fill || _canvas2dStyle.terminalFillDefault || 'rgba(10,34,50,0.38)',
      labelFill: theme.labelFill || _canvas2dStyle.terminalLabelFill || 'rgba(186,230,253,0.96)',
      fillEnabled: theme.fillEnabled !== false,
      hatch: String(theme.hatch || '').trim().toLowerCase(),
    };
  }
  function c2dPassengerTerminalStroke() {
    return getBuildingTheme({ buildingType: 'passenger_terminal' }).stroke;
  }
  function c2dRunwayTaxiwayCenterlineStroke() {
    const s = _canvas2dStyle.runwayTaxiwayCenterlineStroke;
    return (typeof s === 'string' && s.trim()) ? s.trim() : c2dPassengerTerminalStroke();
  }
  function c2dTaxiwayCenterlineStroke() {
    const s = _canvas2dStyle.taxiwayCenterlineStroke;
    return (typeof s === 'string' && s.trim()) ? s.trim() : c2dRunwayTaxiwayCenterlineStroke();
  }
  function getDefaultBuildingNameForType(buildingType, currentId) {
    const prefix = getBuildingTypeNamePrefix(buildingType);
    const buildings = (state.terminals || []).filter(function(t) { return t && t.id !== currentId; });
    const used = new Set(buildings.map(function(t) { return (t.name && String(t.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet(prefix + String(buildings.length + 1), used);
  }

  function id() { return 'id_' + Math.random().toString(36).slice(2, 11); }
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function escapeAttr(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/\\r\\n|\\r|\\n/g, ' ');
  }
  function renderChoiceChipList(container, items, selectedIds, inputClass, inputName) {
    if (!container) return;
    const selected = new Set(Array.isArray(selectedIds) ? selectedIds.map(String) : []);
    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
      container.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No options.</div>';
      return;
    }
    container.innerHTML = '<div class="choice-chip-grid">' + list.map(function(item) {
      const itemId = String(item.id || '');
      const checked = selected.has(itemId);
      return '' +
        '<label class="choice-chip' + (checked ? ' is-checked' : '') + '">' +
          '<input type="checkbox" class="' + escapeHtml(inputClass || '') + '" name="' + escapeHtml(inputName || '') + '" data-item-id="' + escapeHtml(itemId) + '"' + (checked ? ' checked' : '') + ' />' +
          '<span class="choice-chip-label">' + escapeHtml(String(item.label || itemId || '')) + '</span>' +
        '</label>';
    }).join('') + '</div>';
  }
  function syncChoiceChipStates(container) {
    if (!container) return;
    container.querySelectorAll('.choice-chip').forEach(function(labelEl) {
      const input = labelEl.querySelector('input[type="checkbox"]');
      labelEl.classList.toggle('is-checked', !!(input && input.checked));
    });
  }
  function getNamedBuildings() {
    return makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {
      return { id: t.id, label: (t.name || '').trim() || 'Building' };
    });
  }
  function renderRemoteTerminalAccessChoices(selectedIds) {
    const container = document.getElementById('remoteTerminalAccess');
    renderChoiceChipList(container, getNamedBuildings(), selectedIds, 'remote-term-check', 'remote-building');
  }
  function renderTempStandTerminalAccessChoices(selectedIds) {
    const container = document.getElementById('tempStandTerminalAccess');
    renderChoiceChipList(container, getNamedBuildings(), selectedIds, 'remote-term-check', 'remote-building');
  }
  function renderRunwayDirectionChoices(selectedIds) {
    const container = document.getElementById('runwayExitAllowedDirection');
    renderChoiceChipList(container, [
      { id: 'clockwise', label: 'CW' },
      { id: 'counter_clockwise', label: 'CCW' },
    ], selectedIds, 'runway-exit-dir-check', 'runway-exit-dir');
  }
  function renderAircraftConstraintChoices(containerId, selectedIds) {
    const container = document.getElementById(containerId);
    renderChoiceChipList(container, getAircraftConstraintOptions(), selectedIds, 'aircraft-type-check', containerId);
  }
  function syncStandConstraintVisibility(prefix, mode) {
    const normMode = normalizeStandCategoryMode(mode, 'icao');
    const icaoWrap = document.getElementById(prefix + 'IcaoWrap');
    const aircraftWrap = document.getElementById(prefix + 'AircraftWrap');
    if (icaoWrap) icaoWrap.style.display = normMode === 'icao' ? 'grid' : 'none';
    if (aircraftWrap) aircraftWrap.style.display = normMode === 'aircraft' ? 'grid' : 'none';
  }

  const state = {
    terminals: [],
    pbbStands: [],
    remoteStands: [],
    tempStands: [],
    holdingPoints: [],
    taxiways: [],
    apronLinks: [],
    layoutEdgeNames: {},
    directionModes: [],
    currentLayoutName: String(INITIAL_LAYOUT_DISPLAY_NAME || 'default_layout'),
    flights: [],
    simTimeSec: 0,
    simStartSec: 0,
    simDurationSec: 0,
    simPlaybackEndCapSec: null,
    simPlaying: false,
    simSliderScrubbing: false,
    simSpeed: _dc.defaultSimSpeed,
    hasSimulationResult: false,
    simPlaybackDockVisible: false,
    showGrid: GRID_VISIBLE_DEFAULT,
    showImage: IMAGE_VISIBLE_DEFAULT,
    showRoadWidth: ROAD_WIDTH_VISIBLE_DEFAULT,
    aiAssistantDockOpen: false,
    currentTerminalId: null,
    selectedObject: null,
    terminalDrawingId: null,
    taxiwayDrawingId: null,
    dragVertex: null,
    dragTaxiwayVertex: null,
    dragPbbBridgeVertex: null,
    dragStandConnection: null,
    dragRemoteStandPosition: null,
    dragStandRotation: null,
    dragApronLinkVertex: null,
    selectedVertex: null,
    scale: 1,
    panX: 0,
    panY: 0,
    isPanning: false,
    dragStart: null,
    layoutImageOverlay: null,
    previewRemote: null,
    previewTempStand: null,
    previewPbb: null,
    pbbDrawing: false,
    remoteDrawing: false,
    tempStandDrawing: false,
    holdingPointDrawing: false,
    previewHoldingPoint: null,
    apronLinkDrawing: false,
    apronLinkTemp: null,
    apronLinkMidpoints: [],
    apronLinkPointerWorld: null,
    layoutPathDrawPointer: null,
    hoverCell: null,
    vttArrCacheRev: 0,
    derivedGraphEdges: [],
    globalUpdateFresh: false,
    activeRwySepId: null,
    activeRwySepSubtab: 'noname',
    rwySepPanelDirty: true,
    rwySepSnapshotStaleGen: 0,
    pathPolylineCacheRev: 0,
    flightSchedulePage: 0,
    kpiRollingDetailExpanded: false,
    flightPathRevealFlightId: null,
    allocGanttWindowStartMin: null,
    layoutMarkers: [],
    layers: Object.assign({}, DEFAULT_LAYERS),
    showLayoutMarkers: false,
    markerDrawing: false,
    markerRulerDraft: null,
    markerRulerHoverWorld: null,
    markerIslandDraft: null,
    markerIslandHoverWorld: null,
    markerAreaDraft: null,
    markerAreaHoverWorld: null,
    markerFlightHoverSnap: null,
    markerTextDraft: null,
    dragLayoutMarkerHandle: null,
    pathArcModeOn: false,
    pathArcDrag: null,
  };
  const LAYER_STATE_KEYS = [
    'grid', 'image', 'pathLines', 'pathFill', 'standLines', 'standFill',
    'islandAreaLines', 'islandAreaFill', 'buildingLines', 'buildingFill', 'textRuler', 'dummyFlight', 'junction'
  ];
  const LAYER_SECTION_KEYS = {
    lines: ['pathLines', 'standLines', 'islandAreaLines', 'buildingLines'],
    fill: ['pathFill', 'standFill', 'islandAreaFill', 'buildingFill'],
    etc: ['textRuler', 'dummyFlight', 'junction']
  };
  function syncLegacyViewFlagsFromLayers() {
    state.showGrid = !!state.layers.grid;
    state.showImage = !!state.layers.image;
    state.showRoadWidth = !!(state.layers.pathFill || state.layers.islandAreaFill);
    state.showLayoutMarkers = !!(state.layers.textRuler || state.layers.dummyFlight);
  }
  function mergeLayersFromObject(raw) {
    if (!raw || typeof raw !== 'object') return;
    for (let i = 0; i < LAYER_STATE_KEYS.length; i++) {
      const k = LAYER_STATE_KEYS[i];
      if (typeof raw[k] === 'boolean') state.layers[k] = raw[k];
    }
  }
  function hydrateLayersFromGridObject(grid, root) {
    state.layers = Object.assign({}, DEFAULT_LAYERS);
    const g = grid && typeof grid === 'object' ? grid : {};
    const r = root && typeof root === 'object' ? root : {};
    if (g.layers && typeof g.layers === 'object') {
      mergeLayersFromObject(g.layers);
    } else {
      if (typeof g.showGrid === 'boolean') state.layers.grid = g.showGrid;
      else if (typeof r.showGrid === 'boolean') state.layers.grid = r.showGrid;
      if (typeof g.showImage === 'boolean') state.layers.image = g.showImage;
      else if (typeof r.showImage === 'boolean') state.layers.image = r.showImage;
      const sr = typeof g.showRoadWidth === 'boolean' ? g.showRoadWidth
        : (typeof r.showRoadWidth === 'boolean' ? r.showRoadWidth : DEFAULT_LAYERS.pathFill);
      state.layers.pathFill = !!sr;
      state.layers.islandAreaFill = !!sr;
      if (typeof g.showLayoutMarkers === 'boolean') {
        state.layers.textRuler = g.showLayoutMarkers;
        state.layers.dummyFlight = g.showLayoutMarkers;
      }
    }
    syncLegacyViewFlagsFromLayers();
  }
  syncLegacyViewFlagsFromLayers();
  let hookSyncFlightPanelFromSelection = null;
  function bumpRwySepSnapshotStaleGen() {
    state.rwySepSnapshotStaleGen = (state.rwySepSnapshotStaleGen | 0) + 1;
  }
  function bumpPathPolylineCacheRev() {
    state.pathPolylineCacheRev = (state.pathPolylineCacheRev | 0) + 1;
  }
  function cloneFlightsWithoutPathPolylineCache(flights) {
    return (flights || []).map(function(f) {
      const raw = JSON.parse(JSON.stringify(f));
      delete raw.cachedArrPathPts;
      delete raw.cachedDepPathPts;
      delete raw._pathPolylineCacheRev;
      delete raw._pathPolylineArrRetKey;
      return raw;
    });
  }
  function markGlobalUpdateStale() {
    state.globalUpdateFresh = false;
    state.simPlaying = false;
    state.simSliderScrubbing = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    bumpPathPolylineCacheRev();
    state.rwySepPanelDirty = true;
    bumpRwySepSnapshotStaleGen();
    if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines({ keepDesResultTimelines: true });
    const dot = document.getElementById('globalUpdateSyncDot');
    if (dot) {
      dot.classList.remove('fresh');
      dot.classList.add('stale');
      dot.setAttribute('title', '레이아웃/스케줄 변경됨 — Pro Sim으로 재동기화 (완료 시 결과 자동 반영)');
    }
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function markGlobalUpdateFresh() {
    state.globalUpdateFresh = true;
    const dot = document.getElementById('globalUpdateSyncDot');
    if (dot) {
      dot.classList.remove('stale');
      dot.classList.add('fresh');
      dot.setAttribute('title', 'All views match the last Pro Sim run');
    }
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function redrawLayoutAfterEdit() {
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof draw === 'function') draw();
    if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
  }
  function setGlobalUpdateProgressUi(visible, label, pct) {
    const ov = document.getElementById('globalUpdateOverlay');
    const fill = document.getElementById('globalUpdateProgressFill');
    const lab = document.getElementById('globalUpdateOverlayLabel');
    const btn = document.getElementById('btnGlobalUpdate');
    if (!ov) return;
    if (visible) {
      ov.classList.add('is-visible');
      ov.setAttribute('aria-hidden', 'false');
      if (lab && label != null) lab.textContent = label;
      if (fill && pct != null) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
      if (btn) btn.disabled = true;
    } else {
      ov.classList.remove('is-visible');
      ov.setAttribute('aria-hidden', 'true');
      if (fill) fill.style.width = '0%';
      if (btn) btn.disabled = false;
    }
  }
  function scheduleAfterPaint(fn) {
    requestAnimationFrame(function() {
      requestAnimationFrame(function() { setTimeout(fn, 0); });
    });
  }
  const DEFAULT_AIRLINE_CODES = (function() {
    const a = _flightTier.defaultAirlineCodes;
    return (Array.isArray(a) && a.length) ? a.map(String) : ['KE', '7C', 'DL'];
  })();
  const PATH_LAYOUT_MODES = ['runwayPath', 'runwayTaxiway', 'taxiway'];
  function pathTypeFromLayoutMode(layoutMode) {
    if (layoutMode === 'runwayPath') return 'runway';
    if (layoutMode === 'runwayTaxiway') return 'runway_exit';
    if (layoutMode === 'taxiway') return 'taxiway';
    return 'taxiway';
  }


  function layoutModeFromPathType(pt) {
    if (pt === 'runway') return 'runwayPath';
    if (pt === 'runway_exit') return 'runwayTaxiway';
    if (pt === 'apron_taxiway') return 'taxiway';
    if (pt === 'general_queue_taxiway') return 'taxiway';
    return 'taxiway';
  }
  function isPathLayoutMode(m) {
    return PATH_LAYOUT_MODES.indexOf(m) >= 0;
  }
  function standHasApronTaxiwayLink(standId) {
    if (standId == null || standId === '') return false;
    const links = state.apronLinks || [];
    const tws = state.taxiways || [];
    for (let i = 0; i < links.length; i++) {
      const lk = links[i];
      if (!lk || lk.pbbId !== standId) continue;
      const tid = lk.taxiwayId;
      for (let j = 0; j < tws.length; j++) {
        if (tws[j] && tws[j].id === tid) return true;
      }
    }
    return false;
  }
  function ganttESeriesMinutesFromTimelineMeta(f) {
    const m = f && f.timeline_meta;
    if (!m || typeof m !== 'object') {
      return { eldt: NaN, eibt: NaN, eobt: NaN, etot: NaN };
    }
    const toMin = function(sec) {
      const n = sec != null ? Number(sec) : NaN;
      return (isFinite(n) ? n / 60 : NaN);
    };
    return {
      eldt: toMin(m.eldtSec),
      eibt: toMin(m.eibtSec),
      eobt: toMin(m.eobtSec),
      etot: toMin(m.etotSec),
    };
  }
  function settingModeValueForHit(hit) {
    if (!hit || !hit.type) return null;
    if (hit.type === 'terminal') return 'terminal';
    if (hit.type === 'pbb') return 'pbb';
    if (hit.type === 'remote') return 'remote';
    if (hit.type === 'tempStand') return 'tempStand';
    if (hit.type === 'holdingPoint') return 'holdingPoint';
    if (hit.type === 'taxiway') return layoutModeFromPathType((hit.obj && hit.obj.pathType) || 'taxiway');
    if (hit.type === 'apronLink') return 'apronTaxiway';
    if (hit.type === 'layoutMarker') return 'marker';
    return null;
  }
  function cancelActiveLayoutDrawingState() {
    state.pbbDrawing = false;
    state.remoteDrawing = false;
    state.tempStandDrawing = false;
    state.holdingPointDrawing = false;
    state.previewHoldingPoint = null;
    state.apronLinkDrawing = false;
    state.apronLinkTemp = null;
    state.apronLinkMidpoints = [];
    state.apronLinkPointerWorld = null;
    state.layoutPathDrawPointer = null;
    state.previewPbb = null;
    state.previewRemote = null;
    state.previewTempStand = null;
    state.markerDrawing = false;
    state.markerRulerDraft = null;
    state.markerRulerHoverWorld = null;
    state.markerIslandDraft = null;
    state.markerIslandHoverWorld = null;
    state.markerAreaDraft = null;
    state.markerAreaHoverWorld = null;
    state.markerFlightHoverSnap = null;
    if (state.markerTextDraft && state.markerTextDraft.active) {
      state.markerTextDraft = null;
      hideMarkerTextDraftEditor();
    }
    state.dragLayoutMarkerHandle = null;
    syncDrawToggleButton('btnMarkerDraw', false);
  }
  function syncDrawToggleButton(elementId, isDrawing) {
    const btn = document.getElementById(elementId);
    if (!btn) return;
    btn.textContent = isDrawing ? 'Drawing' : 'Draw';
    btn.classList.toggle('drawing', isDrawing);
  }
  function syncLayerPopoverFromState() {
    const panel = document.getElementById('layerPopoverPanel');
    if (panel) {
      panel.querySelectorAll('input[data-layer-key]').forEach(function(inp) {
        const k = inp.getAttribute('data-layer-key');
        if (k && typeof state.layers[k] === 'boolean') inp.checked = !!state.layers[k];
      });
    }
    const allOn = LAYER_STATE_KEYS.every(function(k) { return !!state.layers[k]; });
    const btnAll = document.getElementById('btnLayerPopoverAll');
    if (btnAll) {
      btnAll.classList.toggle('active', allOn);
      btnAll.setAttribute('aria-pressed', allOn ? 'true' : 'false');
      btnAll.title = allOn ? 'Turn all layers off' : 'Turn all layers on';
    }
    if (panel) {
      panel.querySelectorAll('input[data-layer-section-parent]').forEach(function(parentInp) {
        const sec = parentInp.getAttribute('data-layer-section-parent');
        const keys = sec && LAYER_SECTION_KEYS[sec];
        if (!keys || !keys.length) return;
        const secAll = keys.every(function(k) { return !!state.layers[k]; });
        const secSome = keys.some(function(k) { return !!state.layers[k]; });
        parentInp.checked = secAll;
        parentInp.indeterminate = !secAll && secSome;
      });
    }
    const btn = document.getElementById('btnLayerPopover');
    if (btn) {
      btn.classList.toggle('active', allOn);
    }
    const on = !!state.showLayoutMarkers;
    const t1 = document.getElementById('btnLayoutMarkersToggle');
    const t2 = document.getElementById('btnGridMarkerOverlayToggle');
    [t1, t2].forEach(function(el) {
      if (!el) return;
      el.classList.toggle('active', on);
      el.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }
  function setLayerPopoverOpen(open) {
    const btn = document.getElementById('btnLayerPopover');
    const panel = document.getElementById('layerPopoverPanel');
    if (!btn || !panel) return;
    const o = !!open;
    btn.setAttribute('aria-expanded', o ? 'true' : 'false');
    if (o) panel.removeAttribute('hidden');
    else panel.setAttribute('hidden', '');
  }
  function clampLayoutImageOpacity(value) {
    const n = Number(value);
    if (!isFinite(n)) return GRID_LAYOUT_IMAGE_DEFAULTS.opacity;
    return Math.max(GRID_LAYOUT_IMAGE_DEFAULTS.opacityMin, Math.min(GRID_LAYOUT_IMAGE_DEFAULTS.opacityMax, n));
  }
  function clampLayoutImageSize(value, fallback) {
    const n = Number(value);
    if (!isFinite(n) || n <= 0) return fallback;
    return n;
  }
  function clampLayoutImagePoint(value, fallback) {
    const n = Number(value);
    return isFinite(n) ? n : fallback;
  }
  function getLayoutImageAspectRatio(overlay) {
    if (!overlay || typeof overlay !== 'object') return 1;
    const ow = Number(overlay.originalWidthPx);
    const oh = Number(overlay.originalHeightPx);
    if (isFinite(ow) && ow > 0 && isFinite(oh) && oh > 0) return oh / ow;
    const w = Number(overlay.widthM);
    const h = Number(overlay.heightM);
    if (isFinite(w) && w > 0 && isFinite(h) && h > 0) return h / w;
    return 1;
  }
  function applyLayoutImageWidthByAspect(widthM) {
    if (!state.layoutImageOverlay) return;
    const nextWidth = clampLayoutImageSize(widthM, state.layoutImageOverlay.widthM);
    const aspect = getLayoutImageAspectRatio(state.layoutImageOverlay);
    state.layoutImageOverlay.widthM = nextWidth;
    state.layoutImageOverlay.heightM = clampLayoutImageSize(nextWidth * aspect, state.layoutImageOverlay.heightM);
  }
  function applyLayoutImageHeightByAspect(heightM) {
    if (!state.layoutImageOverlay) return;
    const nextHeight = clampLayoutImageSize(heightM, state.layoutImageOverlay.heightM);
    const aspect = getLayoutImageAspectRatio(state.layoutImageOverlay);
    state.layoutImageOverlay.heightM = nextHeight;
    state.layoutImageOverlay.widthM = clampLayoutImageSize(nextHeight / Math.max(aspect, 1e-9), state.layoutImageOverlay.widthM);
  }
  function normalizeLayoutImageOverlay(raw) {
    if (!raw || typeof raw !== 'object' || !raw.dataUrl) return null;
    const widthM = clampLayoutImageSize(raw.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM);
    const heightM = clampLayoutImageSize(raw.heightM, GRID_LAYOUT_IMAGE_DEFAULTS.heightM);
    const originalWidthPx = clampLayoutImageSize(raw.originalWidthPx, widthM);
    const originalHeightPx = clampLayoutImageSize(raw.originalHeightPx, heightM);
    return {
      name: String(raw.name || 'Layout image'),
      type: String(raw.type || 'image/png'),
      dataUrl: String(raw.dataUrl || ''),
      opacity: clampLayoutImageOpacity(raw.opacity),
      widthM: widthM,
      heightM: heightM,
      originalWidthPx: originalWidthPx,
      originalHeightPx: originalHeightPx,
      topLeftCol: clampLayoutImagePoint(raw.topLeftCol, GRID_LAYOUT_IMAGE_DEFAULTS.topLeftCol),
      topLeftRow: clampLayoutImagePoint(raw.topLeftRow, GRID_LAYOUT_IMAGE_DEFAULTS.topLeftRow)
    };
  }
  function syncLayoutImageBitmap() {
    const overlay = state.layoutImageOverlay;
    if (!overlay || !overlay.dataUrl) {
      layoutImageBitmap = null;
      layoutImageBitmapSrc = '';
      return;
    }
    if (layoutImageBitmap && layoutImageBitmapSrc === overlay.dataUrl) return;
    layoutImageBitmap = null;
    layoutImageBitmapSrc = '';
    const img = new Image();
    const src = overlay.dataUrl;
    img.onload = function() {
      if (!state.layoutImageOverlay || state.layoutImageOverlay.dataUrl !== src) return;
      layoutImageBitmap = img;
      layoutImageBitmapSrc = src;
      invalidateGridUnderlay();
      safeDraw();
    };
    img.onerror = function() {
      if (!state.layoutImageOverlay || state.layoutImageOverlay.dataUrl !== src) return;
      layoutImageBitmap = null;
      layoutImageBitmapSrc = '';
      invalidateGridUnderlay();
      safeDraw();
    };
    img.src = src;
  }
  function toggleLayoutDrawMode(flagKey, previewKey, tempKey) {
    state.selectedObject = null;
    if (state[flagKey]) {
      state[flagKey] = false;
      if (previewKey) state[previewKey] = null;
      if (tempKey) state[tempKey] = null;
      if (flagKey === 'apronLinkDrawing') {
        state.apronLinkMidpoints = [];
        state.apronLinkPointerWorld = null;
      }
    } else {
      state[flagKey] = true;
      if (previewKey) state[previewKey] = null;
      if (tempKey) state[tempKey] = null;
      if (flagKey === 'apronLinkDrawing') {
        state.apronLinkMidpoints = [];
        state.apronLinkPointerWorld = null;
      }
    }
    syncPanelFromState();
    draw();
  }
  function handlePbbOrRemoteMouseUp2D(mode, wx, wy) {
    if (mode === 'pbb' && state.pbbDrawing) {
      if (tryPlacePbbAt(wx, wy)) { syncPanelFromState(); draw(); }
      return true;
    }
    if (mode === 'remote' && state.remoteDrawing) {
      const prev = state.previewRemote;
      if (prev && !prev.overlap && tryPlaceRemoteAt(prev.x, prev.y)) { syncPanelFromState(); draw(); }
      return true;
    }
    if (mode === 'tempStand' && state.tempStandDrawing) {
      const prev = state.previewTempStand;
      if (prev && !prev.overlap && tryPlaceTempStandAt(prev.x, prev.y)) { syncPanelFromState(); draw(); }
      return true;
    }
    if (mode === 'holdingPoint' && state.holdingPointDrawing) {
      const prev = state.previewHoldingPoint;
      if (prev && tryPlaceHoldingPointAt(prev.x, prev.y, prev.pathType || 'taxiway')) { syncPanelFromState(); draw(); }
      return true;
    }
    return false;
  }
  function tryCommitStandPlacement3D(mode, wx, wy, col, row) {
    if (mode === 'pbb' && state.pbbDrawing) {
      if (tryPlacePbbAt(wx, wy)) { syncPanelFromState(); updateObjectInfo(); update3DScene(); }
      return;
    }
    if (mode === 'remote' && state.remoteDrawing) {
      if (tryPlaceRemoteAt(wx, wy)) { syncPanelFromState(); updateObjectInfo(); update3DScene(); }
    }
    if (mode === 'tempStand' && state.tempStandDrawing) {
      if (tryPlaceTempStandAt(wx, wy)) { syncPanelFromState(); updateObjectInfo(); update3DScene(); }
    }
  }
  function findLayoutObjectByListType(typ, idr) {
    if (typ === 'terminal') return state.terminals.find(t => t.id === idr);
    if (typ === 'pbb') return state.pbbStands.find(p => p.id === idr);
    if (typ === 'remote') return state.remoteStands.find(r => r.id === idr);
    if (typ === 'tempStand') return (state.tempStands || []).find(function(s) { return s.id === idr; });
    if (typ === 'holdingPoint') return (state.holdingPoints || []).find(h => h.id === idr);
    if (typ === 'taxiway') return state.taxiways.find(tw => tw.id === idr);
    if (typ === 'apronLink') return state.apronLinks.find(lk => lk.id === idr);
    if (typ === 'layoutEdge') return (state.derivedGraphEdges || []).find(function(e) { return e.id === idr; });
    if (typ === 'flight') return state.flights.find(f => f.id === idr);
    if (typ === 'layoutMarker') return (state.layoutMarkers || []).find(function(m) { return m && m.id === idr; });
    return null;
  }
  function removeLayoutObjectFromState(type, id) {
    const removedTaxiway = (type === 'taxiway')
      ? (state.taxiways || []).find(function(tw) { return tw.id === id; })
      : null;
    if (type === 'terminal') state.terminals = state.terminals.filter(t => t.id !== id);
    else if (type === 'pbb') state.pbbStands = state.pbbStands.filter(p => p.id !== id);
    else if (type === 'remote') state.remoteStands = state.remoteStands.filter(r => r.id !== id);
    else if (type === 'tempStand') state.tempStands = (state.tempStands || []).filter(function(s) { return s.id !== id; });
    else if (type === 'holdingPoint') state.holdingPoints = (state.holdingPoints || []).filter(h => h.id !== id);
    else if (type === 'taxiway') state.taxiways = state.taxiways.filter(tw => tw.id !== id);
    else if (type === 'apronLink') state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
    else if (type === 'flight') {
      state.flights = state.flights.filter(f => f.id !== id);
      bumpRwySepSnapshotStaleGen();
      state.rwySepPanelDirty = true;
    }
    else if (type === 'layoutMarker') state.layoutMarkers = (state.layoutMarkers || []).filter(function(m) { return m && m.id !== id; });
    else if (type === 'layoutEdge') {}
    if (removedTaxiway) {
      const shouldResampleRet = (removedTaxiway.pathType === 'runway' || removedTaxiway.pathType === 'runway_exit');
      if (removedTaxiway.pathType === 'runway_exit') {
