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
    /** Map apron link id -> true: draw taxiway×apron polyline junction overlay until path graph sync (no full graph rebuild). */
    apronLinkJunctionOverlayDirtyIds: null,
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
    pathGraphCache: null,
    pathGraphCacheValid: false,
    pathGraphCacheSig: '',
    pathGraphCacheDirty: false,
    pathGraphInvalidatedAtMs: 0,
    pathGraphAllowHeavySimExport: false,
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
  const PATH_GRAPH_REBUILD_DEBOUNCE_MS = 400;
  const PATH_GRAPH_ASYNC_REBUILD_MIN_TW = 200;
  /** When true, never run buildPathGraph from draw/serialize except via applyPathGraphSyncNow() (Update / Pro Sim). */
  const PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION = true;
  let _pathGraphRebuildTimer = null;
  let _pathGraphRebuildSigQueued = '';
  function cancelPathGraphRebuildTimer() {
    if (_pathGraphRebuildTimer) {
      clearTimeout(_pathGraphRebuildTimer);
      _pathGraphRebuildTimer = null;
    }
    _pathGraphRebuildSigQueued = '';
  }
  function queuePathGraphRebuild(graphSig) {
    if (graphSig === _pathGraphRebuildSigQueued && _pathGraphRebuildTimer) return;
    if (_pathGraphRebuildTimer) clearTimeout(_pathGraphRebuildTimer);
    _pathGraphRebuildSigQueued = graphSig;
    _pathGraphRebuildTimer = setTimeout(function() {
      _pathGraphRebuildTimer = null;
      _pathGraphRebuildSigQueued = '';
      const sigNow = computeTaxiwaysGraphSig();
      if (sigNow !== graphSig) return;
      try {
        const gNew = buildPathGraph();
        state.pathGraphCache = gNew;
        state.pathGraphCacheValid = true;
        state.pathGraphCacheSig = sigNow;
        state.pathGraphCacheDirty = false;
      } catch (e) {
        state.pathGraphCache = null;
        state.pathGraphCacheValid = false;
        state.pathGraphCacheSig = '';
        state.pathGraphCacheDirty = true;
        console.error('queuePathGraphRebuild: buildPathGraph failed', e);
      }
      scheduleDraw();
    }, PATH_GRAPH_REBUILD_DEBOUNCE_MS);
  }
  function applyPathGraphSyncNow() {
    cancelPathGraphRebuildTimer();
    const graphSig = computeTaxiwaysGraphSig();
    try {
      const gNew = buildPathGraph();
      state.pathGraphCache = gNew;
      state.pathGraphCacheValid = true;
      state.pathGraphCacheSig = graphSig;
      state.pathGraphCacheDirty = false;
      state.apronLinkJunctionOverlayDirtyIds = null;
    } catch (e) {
      state.pathGraphCache = null;
      state.pathGraphCacheValid = false;
      state.pathGraphCacheSig = '';
      state.pathGraphCacheDirty = true;
      console.error('applyPathGraphSyncNow: buildPathGraph failed', e);
    }
  }
  function markApronLinkJunctionOverlayDirty(linkId) {
    if (linkId == null || linkId === '') return;
    if (!state.apronLinkJunctionOverlayDirtyIds) state.apronLinkJunctionOverlayDirtyIds = {};
    state.apronLinkJunctionOverlayDirtyIds[String(linkId)] = true;
  }
  function invalidatePathGraphCache(hardReset) {
    cancelPathGraphRebuildTimer();
    if (hardReset) {
      state.pathGraphCache = null;
      state.pathGraphCacheValid = false;
      state.pathGraphCacheSig = '';
      state.pathGraphCacheDirty = true;
      state.pathGraphInvalidatedAtMs = 0;
      state.apronLinkJunctionOverlayDirtyIds = null;
      return;
    }
    state.pathGraphCacheDirty = true;
    state.pathGraphInvalidatedAtMs = Date.now();
    if (!state.pathGraphCacheValid) {
      state.pathGraphCache = null;
      state.pathGraphCacheSig = '';
    }
  }
  function computeTaxiwaysGraphSig() {
    return (state.taxiways || []).map(function(tw) {
      if (!tw || !tw.vertices) return '';
      const verts = tw.vertices.map(function(v) { return String(Number(v.col)) + ',' + String(Number(v.row)); }).join(';');
      return String(tw.id || '') + '|' + String(tw.pathType || '') + '|' + String(tw.direction || '') + '|' + verts;
    }).join('||');
  }
  function stripPathGraphCacheJunctionsNearTaxiwayWorld(tw) {
    const g = state.pathGraphCache;
    if (!g || g.__junctionStale || !tw) return;
    const pts = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : null;
    if (!pts || pts.length < 2) return;
    const mergeR = (typeof PATH_JUNCTION_MERGE_RADIUS_PX === 'number' && isFinite(PATH_JUNCTION_MERGE_RADIUS_PX)) ? PATH_JUNCTION_MERGE_RADIUS_PX : 8;
    const tol = Math.max(mergeR * 2.2, 12);
    const tol2 = tol * tol;
    function distPointToSegSq(p, a, b) {
      const pr = projectOnSegment(a, b, p);
      const dpx = p[0] - pr.p[0], dpy = p[1] - pr.p[1];
      return dpx * dpx + dpy * dpy;
    }
    function pointNearDeletedTw(p) {
      if (!p || !Array.isArray(p) || p.length < 2) return false;
      if (!isFinite(p[0]) || !isFinite(p[1])) return false;
      for (let seg = 0; seg < pts.length - 1; seg++) {
        const a = pts[seg], b = pts[seg + 1];
        if (!a || !b || !isFinite(a[0]) || !isFinite(b[0])) continue;
        if (distPointToSegSq(p, a, b) <= tol2) return true;
      }
      return false;
    }
    function filt(arr) {
      if (!Array.isArray(arr)) return arr;
      return arr.filter(function(pt) { return !pointNearDeletedTw(pt); });
    }
    if (Array.isArray(g.validJunctions)) g.validJunctions = filt(g.validJunctions);
    if (Array.isArray(g.connectedJunctions)) g.connectedJunctions = filt(g.connectedJunctions);
    if (Array.isArray(g.junctions)) g.junctions = filt(g.junctions);
    if (Array.isArray(g.disconnectedValidJunctions)) g.disconnectedValidJunctions = filt(g.disconnectedValidJunctions);
  }
  function markPathGraphJunctionStaleShellAfterLayoutEdit() {
    cancelPathGraphRebuildTimer();
    state.pathGraphCache = {
      __junctionStale: true,
      validJunctions: [],
      connectedJunctions: [],
      disconnectedValidJunctions: [],
      junctions: [],
      nodes: [],
      edges: [],
      adj: [],
      edgeMap: {},
      runwayNodeIndicesById: {},
      standIdToNodeIndex: {}
    };
    state.pathGraphCacheValid = true;
    state.pathGraphCacheSig = computeTaxiwaysGraphSig();
    state.pathGraphCacheDirty = true;
    state.pathGraphInvalidatedAtMs = Date.now();
  }
  function graphSigParseRecords(sig) {
    const m = {};
    if (!sig || typeof sig !== 'string') return m;
    const chunks = sig.split('||');
    for (let i = 0; i < chunks.length; i++) {
      const rec = chunks[i];
      if (!rec) continue;
      const pipe = rec.indexOf('|');
      const id = pipe >= 0 ? rec.slice(0, pipe) : rec;
      if (id) m[id] = rec;
    }
    return m;
  }
  function graphSigTaxiwayDiff(oldSig, newSig) {
    const o = graphSigParseRecords(oldSig);
    const n = graphSigParseRecords(newSig);
    const removed = [];
    const changed = [];
    Object.keys(o).forEach(function(id) {
      if (!(id in n)) removed.push(id);
    });
    Object.keys(n).forEach(function(id) {
      if (!(id in o)) changed.push(id);
      else if (o[id] !== n[id]) changed.push(id);
    });
    return { removed: removed, changed: changed };
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
    // Full reset rebuilds junctions in draw(); manual-sync mode keeps last graph for display until Update / Pro Sim.
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
      invalidatePathGraphCache(false);
    } else {
      invalidatePathGraphCache(true);
    }
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
