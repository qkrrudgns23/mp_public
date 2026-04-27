      ctx.shadowBlur = 0;
    }
    function strokeBarAtOffset(ofs) {
      const sx = cx - px * halfLen + g.ux * ofs;
      const sy = cy - py * halfLen + g.uy * ofs;
      const ex = cx + px * halfLen + g.ux * ofs;
      const ey = cy + py * halfLen + g.uy * ofs;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }
    if (k === 'intermediate') {
      ctx.setLineDash([dashLen, gapLen]);
      strokeBarAtOffset(0);
      ctx.setLineDash([]);
    } else {
      ctx.setLineDash([]);
      strokeBarAtOffset(-pairHalf);
      strokeBarAtOffset(pairHalf);
      const R = closestPointOnAnyRunwayCenterlineWorld(cx, cy);
      const rx = R ? R[0] : cx + g.ux * (CELL_SIZE * 40);
      const ry = R ? R[1] : cy + g.uy * (CELL_SIZE * 40);
      const midM = [cx - g.ux * pairHalf, cy - g.uy * pairHalf];
      const midP = [cx + g.ux * pairHalf, cy + g.uy * pairHalf];
      const ofsR = dist2(midM, [rx, ry]) <= dist2(midP, [rx, ry]) ? -pairHalf : pairHalf;
      const pathW = Number(g.pathWidthM);
      const toothLen = Math.max(0.75, (isFinite(pathW) && pathW > 0 ? pathW : 12) * 0.24) * 0.25;
      const toothSpacing = Math.max(0.55, pathSpanM * 0.065);
      const toothLw = Math.max(lw, lw * 1.12);
      ctx.save();
      ctx.lineWidth = toothLw;
      for (let s = -halfLen + toothSpacing * 0.5; s <= halfLen - toothSpacing * 0.25; s += toothSpacing) {
        const bx = cx + px * s + g.ux * ofsR;
        const by = cy + py * s + g.uy * ofsR;
        const mx = cx + px * s;
        const my = cy + py * s;
        const vx = rx - mx;
        const vy = ry - my;
        const signT = (g.ux * vx + g.uy * vy) >= 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + g.ux * signT * toothLen, by + g.uy * signT * toothLen);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.shadowBlur = 0;
  }
  function c2dSimStandOccupiedFill() { return _canvas2dStyle.simStandOccupiedFill || 'rgba(239, 68, 68, 0.32)'; }
  function c2dSimStandOccupiedStroke() { return _canvas2dStyle.simStandOccupiedStroke || 'rgba(220, 38, 38, 0.95)'; }
  function c2dStandSafetyStroke() { return _canvas2dStyle.standSafetyStroke || 'rgba(255, 45, 110, 0.95)'; }
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
  function getApronAircraftDetailedSilhouettePoints() {
    const raw = _ac2d.detailedSilhouettePoints;
    if (!Array.isArray(raw) || raw.length < 3) return [];
    const out = [];
    for (let i = 0; i < raw.length; i++) {
      const row = raw[i];
      if (!Array.isArray(row) || row.length < 2) continue;
      const x = Number(row[0]);
      const y = Number(row[1]);
      if (isFinite(x) && isFinite(y)) out.push([x, y]);
    }
    return out.length >= 3 ? out : [];
  }
  const _schedAlgo = _algoTier.scheduledTimes || {};
  const SCHED_DWELL_FLOOR_MIN = (function() {
    const v = Number(_schedAlgo.dwellFloorMin);
    return (isFinite(v) && v >= 0) ? v : 20;
  })();
  /** Dispatched schedule (d): SLDT(d) = SIBT(d) − this many minutes; STOT(d) = SOBT(d) + SCHED_SD_STOT_PLUS_SOBD_MIN. */
  const SCHED_SD_SIBT_MINUS_SLD_MIN = 5;
  const SCHED_SD_STOT_PLUS_SOBD_MIN = 5;
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
  const DEFAULT_LAYERS = {
    grid: GRID_VISIBLE_DEFAULT,
    image: IMAGE_VISIBLE_DEFAULT,
    pathLines: true,
    pathFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    standLines: true,
    standFill: true,
    islandAreaLines: true,
    islandAreaFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    buildingLines: true,
    buildingFill: true,
    textRuler: false,
    dummyFlight: false,
    junction: true
  };
  const DEFAULT_LAYER_MONO = { lines: false, fill: false, etc: false };
  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections(_dc.rwExitAllowedDefaultRaw);
  function layoutPathVertexRadiusPx(vertexSelected, pathSelected) {
    if (vertexSelected) return 6 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    if (pathSelected) return 5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    return 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  function layoutTerminalVertexRadiusPx(vertexSelected) {
    return vertexSelected ? 5.5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR : 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  const _dragThreshPx = _interactionConfigNum('dragThresholdPx', 4);
  const DRAG_THRESH = _dragThreshPx > 0 ? Math.max(1, _dragThreshPx) : 4;
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
  const MARKER_BLAZER_COLOR_OPTIONS = ['#ff1493', '#39ff14', '#00f5ff', '#ff6600', '#ffffff'];
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
      }
    }
    if (g.layerMono && typeof g.layerMono === 'object') mergeLayerMonoFromObject(g.layerMono);
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
