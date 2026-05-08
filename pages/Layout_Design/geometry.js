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
  /** Flight Schedule default: 3 uppercase letters + 5 digits (e.g. ABC12345). */
  function randomRegNumber() {
    let letters = '';
    for (let i = 0; i < 3; i++) letters += String.fromCharCode(65 + Math.floor(Math.random() * 26));
    return letters + String(Math.floor(Math.random() * 100000)).padStart(5, '0');
  }
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
  function renderAircraftConstraintChoices(containerId, selectedIds, icaoLetters) {
    const container = document.getElementById(containerId);
    if (!container) return;
    let letters = normalizeAllowedIcaoCategories(icaoLetters);
    if (!letters.length) letters = ['C'];
    const items = getAircraftConstraintOptionsForIcaoLetters(letters);
    const allowedIds = {};
    items.forEach(function(it) { allowedIds[it.id] = true; });
    const selectedArr = Array.isArray(selectedIds) ? selectedIds.map(String) : [];
    const filteredSelected = selectedArr.filter(function(id) { return allowedIds[id]; });
    renderChoiceChipList(container, items, filteredSelected, 'aircraft-type-check', containerId);
  }
  function syncStandConstraintVisibility(prefix) {
    const icaoWrap = document.getElementById(prefix + 'IcaoWrap');
    const aircraftWrap = document.getElementById(prefix + 'AircraftWrap');
    if (icaoWrap) icaoWrap.style.display = 'grid';
    if (aircraftWrap) aircraftWrap.style.display = 'grid';
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
    /** Replay / Pro Sim SIBT window [simWindowStartSec, simWindowEndSec]; full slider axis is [simStartSec, simDurationSec]. */
    simWindowStartSec: 0,
    simWindowEndSec: 0,
    _simScheduleAxisKey: '',
    /** Set by applyLayoutObject when layout JSON has designerPersist sim window; consumed once in recomputeSimDuration. */
    _pendingPersistSimWindow: null,
    /** ``true``: 재생 타임 슬라이더 미세 드래그 모드에서 시각 변경을 대략 일반 스크럽 대비 SIM_TIME_SLIDER_FINE_DIVISOR 배 더 미세하게. 썹은 빨간색 표시. */
    simTimeSliderFineMode: false,
    simPlaybackEndCapSec: null,
    simPlaying: false,
    simSliderScrubbing: false,
    prosimBusy: false,
    grid3dPopupRef: null,
    simSpeed: _dc.defaultSimSpeed,
    hasSimulationResult: false,
    /** Last Pro Sim ``payload.positions`` — keeps x,y playback samples off flights when Play is blocked (lighter pan/zoom). */
    simPlaybackPositionsByFlightId: null,
    /** Copy of ``payload.schedule`` for timeline_meta / E-fields when rehydrating from ``simPlaybackPositionsByFlightId``. */
    simPlaybackScheduleSnapshot: null,
    /** True when timelines were evicted from flights but ``simPlaybackPositionsByFlightId`` still holds data. */
    simPlaybackTimelinesEvictedForMemory: false,
    simPlaybackDockVisible: false,
    /** Derived after Pro Sim: first deadlockGhost sample per flight; slider markers + left dock banner. */
    simDeadlockGhostPlayback: { events: [], bodyLines: '', resolveCount: 0 },
    /** After Resolve lookahead bump: show rerun hint banner until next Pro Sim result. */
    deadlockMitigateBannerRerunHint: false,
    /** flight_id keys with any deadlock ghost in last compact_v2 playback (survives timeline eviction). */
    deadlockFlightIdsFromLastSim: Object.create(null),
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
    /** Map Leadin Taxiway link id -> true: draw taxiway junction overlay until path graph sync (no full graph rebuild). */
    apronLinkJunctionOverlayDirtyIds: null,
    layoutPathDrawPointer: null,
    hoverCell: null,
    vttArrCacheRev: 0,
    derivedGraphEdges: [],
    duplicateApronByStandId: {},
    globalUpdateFresh: false,
    /** Path graph / views match last Designer 'Update' (applyPathGraphSyncNow), not Pro Sim. */
    designerPageUpdateFresh: false,
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
    /** Pro Sim 2D: all | airline | icao | intdom | building */
    flightColorMode: 'all',
    /** Layer popover: monotone overrides per section (Lines / Fill / ETC). */
    layerMono: Object.assign({}, DEFAULT_LAYER_MONO),
    /** Map Type: normal | heatmap (overlays when hasSimulationResult). */
    mapTypeMode: 'normal',
    heatmapTrafficPhases: { rotArr: true, vttArr: true, vttDep: true, rotDep: true },
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
  const LAYER_MONO_KEYS = ['lines', 'fill', 'etc'];
  function layerMonoLinesOn() { return !!(state.layerMono && state.layerMono.lines); }
  function layerMonoFillOn() { return !!(state.layerMono && state.layerMono.fill); }
  function layerMonoEtcOn() { return !!(state.layerMono && state.layerMono.etc); }
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
  function mergeLayerMonoFromObject(raw) {
    if (!raw || typeof raw !== 'object') return;
    for (let i = 0; i < LAYER_MONO_KEYS.length; i++) {
      const k = LAYER_MONO_KEYS[i];
      if (typeof raw[k] === 'boolean') state.layerMono[k] = raw[k];
    }
  }
  function hydrateLayersFromGridObject(grid, root) {
    state.layers = Object.assign({}, DEFAULT_LAYERS);
    state.layerMono = Object.assign({}, DEFAULT_LAYER_MONO);
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
