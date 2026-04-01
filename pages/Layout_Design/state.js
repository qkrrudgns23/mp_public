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
      stroke: theme.stroke || _canvas2dStyle.terminalStrokeDefault || '#38bdf8',
      fill: theme.fill || _canvas2dStyle.terminalFillDefault || 'rgba(56,189,248,0.12)',
      labelFill: theme.labelFill || _canvas2dStyle.terminalLabelFill || 'rgba(56,189,248,0.95)',
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
  function buildNoWayTooltip(f) {
    if (!f) return '경로를 찾을 수 없습니다.';
    const parts = [];
    if (f.noWayArr) {
      const d = f._noWayArrDetail != null ? String(f._noWayArrDetail).trim() : '';
      parts.push('도착: ' + (d || '사유를 판별하지 못했습니다.'));
    }
    if (f.noWayDep) {
      const d = f._noWayDepDetail != null ? String(f._noWayDepDetail).trim() : '';
      parts.push('출발: ' + (d || '사유를 판별하지 못했습니다.'));
    }
    if (f.arrDep !== 'Dep' && f.arrRetFailed && !f.noWayArr) {
      parts.push('도착: RET 실패(제약 또는 샘플링).');
    }
    if (!parts.length) return '경로를 찾을 수 없습니다.';
    return parts.join(' ');
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
    simPlaying: false,
    simSliderScrubbing: false,
    simSpeed: _dc.defaultSimSpeed,
    hasSimulationResult: false,
    simPlaybackDockVisible: false,
    showGrid: GRID_VISIBLE_DEFAULT,
    showImage: IMAGE_VISIBLE_DEFAULT,
    showRoadWidth: ROAD_WIDTH_VISIBLE_DEFAULT,
    currentTerminalId: null,
    selectedObject: null,
    terminalDrawingId: null,
    taxiwayDrawingId: null,
    dragVertex: null,
    dragTaxiwayVertex: null,
    dragPbbBridgeVertex: null,
    dragStandConnection: null,
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
    previewPbb: null,
    pbbDrawing: false,
    remoteDrawing: false,
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
  };
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
    state.simPlaybackDockVisible = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    bumpPathPolylineCacheRev();
    state.rwySepPanelDirty = true;
    bumpRwySepSnapshotStaleGen();
    if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines();
    const dot = document.getElementById('globalUpdateSyncDot');
    if (dot) {
      dot.classList.remove('fresh');
      dot.classList.add('stale');
      dot.setAttribute('title', 'Results may be outdated — click Light Sim to refresh');
    }
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function markGlobalUpdateFresh() {
    state.globalUpdateFresh = true;
    const dot = document.getElementById('globalUpdateSyncDot');
    if (dot) {
      dot.classList.remove('stale');
      dot.classList.add('fresh');
      dot.setAttribute('title', 'All views match the last Light Sim run');
    }
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function redrawLayoutAfterEdit() {
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof draw === 'function') draw();
    if (typeof scene3d !== 'undefined' && scene3d && typeof update3DScene === 'function') update3DScene();
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
