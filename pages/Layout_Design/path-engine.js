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
      panel.querySelectorAll('input[data-layer-mono]').forEach(function(inp) {
        const mk = inp.getAttribute('data-layer-mono');
        if (mk && state.layerMono && typeof state.layerMono[mk] === 'boolean') inp.checked = !!state.layerMono[mk];
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
    if (o) {
      panel.removeAttribute('hidden');
      const cp = document.getElementById('colorPopoverPanel');
      const cb = document.getElementById('btnColorPopover');
      if (cp && cb && !cp.hasAttribute('hidden')) {
        cp.setAttribute('hidden', '');
        cb.setAttribute('aria-expanded', 'false');
      }
    } else {
      panel.setAttribute('hidden', '');
    }
  }
  function syncMapTypePopoverFromState() {
    const btnHt = document.getElementById('btnHeatmapToggle');
    const heatOk = !!state.hasSimulationResult;
    const failedRegs = typeof getArrRetFailedRegsForProSimUi === 'function' ? getArrRetFailedRegsForProSimUi() : [];
    const hasRetFail = failedRegs.length > 0;
    const proSimUiFresh = !hasRetFail && !!state.globalUpdateFresh;
    const heatmapAllowed = heatOk && proSimUiFresh;
    if (!heatmapAllowed && state.mapTypeMode === 'heatmap') state.mapTypeMode = 'normal';
    if (!heatOk && state.mapTypeMode !== 'normal') state.mapTypeMode = 'normal';
    const mode = state.mapTypeMode || 'normal';
    const heatOn = heatmapAllowed && mode === 'heatmap';
    if (btnHt) {
      btnHt.disabled = !heatmapAllowed;
      btnHt.classList.toggle('active', heatOn);
      btnHt.setAttribute('aria-pressed', heatOn ? 'true' : 'false');
    }
    if (typeof syncHeatmapTrafficLegend === 'function') syncHeatmapTrafficLegend();
  }
  /** After layout load: restore heatmap on/off from `designerPersist` when timelines exist. */
  function applyDesignerPersistMapTypeAfterLoad(dp) {
    if (!dp || dp.v !== 1) return;
    const m = String(dp.mapTypeMode || '');
    const wantHeat = m === 'heatmap' || m === 'heatmap_traffic' || m === 'heatmap_queue';
    state.mapTypeMode = (state.hasSimulationResult && wantHeat) ? 'heatmap' : 'normal';
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
    state.layoutImageOverlay.widthM = clampLayoutImageSize(widthM, state.layoutImageOverlay.widthM);
  }
  function applyLayoutImageHeightByAspect(heightM) {
    if (!state.layoutImageOverlay) return;
    state.layoutImageOverlay.heightM = clampLayoutImageSize(heightM, state.layoutImageOverlay.heightM);
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
  function layoutDrawModePreventsBackgroundObjectPick() {
    return !!(state.pbbDrawing || state.remoteDrawing || state.tempStandDrawing || state.holdingPointDrawing ||
      state.apronLinkDrawing || state.terminalDrawingId || state.taxiwayDrawingId || state.markerDrawing);
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
    else if (type === 'taxiway') {
      state.taxiways = state.taxiways.filter(tw => tw.id !== id);
      if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION) {
        invalidatePathGraphCache(false);
      } else {
        markPathGraphJunctionStaleShellAfterLayoutEdit();
      }
    }
    else if (type === 'apronLink') {
      state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
      if (state.apronLinkJunctionOverlayDirtyIds) delete state.apronLinkJunctionOverlayDirtyIds[String(id)];
    }
    else if (type === 'flight') {
      state.flights = state.flights.filter(f => f.id !== id);
      bumpRwySepSnapshotStaleGen();
      state.rwySepPanelDirty = true;
    }
    else if (type === 'layoutMarker') state.layoutMarkers = (state.layoutMarkers || []).filter(function(m) { return m && m.id !== id; });
    else if (type === 'layoutEdge') {}
    if (removedTaxiway) {
      if (typeof bumpScheduleRetExitDistCache === 'function') bumpScheduleRetExitDistCache();
      if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && state.pathGraphCacheValid && state.pathGraphCache && !state.pathGraphCache.__junctionStale) {
        stripPathGraphCacheJunctionsNearTaxiwayWorld(removedTaxiway);
      }
      if (removedTaxiway.pathType === 'runway_exit') {
        (state.flights || []).forEach(function(f) {
          if (!f || f.sampledArrRet !== id) return;
          f.sampledArrRet = null;
          f.arrRetFailed = false;
          f.arrRotSec = null;
          f.arrRetDistM = null;
          f.arrVRetInMs = null;
          f.arrVRetOutMs = null;
          f.__schedRetRotRev = null;
          f.__schedVttArrRev = null;
          f.__schedVttArrMin = null;
          f.noWayArr = false;
          delete f._noWayArrDetail;
        });
      }
      if (typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    }
  }
  function syncPathFieldVisibilityForPathType(pt) {
    const taxiwayAvgWrap = document.getElementById('taxiwayAvgVelocityWrap');
    const runwayMinArrWrap = document.getElementById('runwayMinArrVelocityWrap');
    const runwayLineupWrap = document.getElementById('runwayLineupDistWrap');
    const runwayStartDispWrap = document.getElementById('runwayStartDisplacedThresholdWrap');
    const runwayStartBlastWrap = document.getElementById('runwayStartBlastPadWrap');
    const runwayEndDispWrap = document.getElementById('runwayEndDisplacedThresholdWrap');
    const runwayEndBlastWrap = document.getElementById('runwayEndBlastPadWrap');
    const maxExitWrap = document.getElementById('runwayMaxExitVelWrap');
    const minExitWrap = document.getElementById('runwayMinExitVelWrap');
    const rwDirWrap = document.getElementById('runwayExitAllowedDirectionWrap');
    const taxiwayTypeWrap = document.getElementById('taxiwayTypeWrap');
    if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = (pt === 'taxiway' || pt === 'apron_taxiway' || pt === 'general_queue_taxiway') ? 'grid' : 'none';
    if (taxiwayTypeWrap) taxiwayTypeWrap.style.display = (pt === 'taxiway' || pt === 'general_queue_taxiway') ? 'grid' : 'none';
    if (runwayMinArrWrap) runwayMinArrWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (runwayLineupWrap) runwayLineupWrap.style.display = (pt === 'runway') ? 'block' : 'none';
    if (runwayStartDispWrap) runwayStartDispWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (runwayStartBlastWrap) runwayStartBlastWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (runwayEndDispWrap) runwayEndDispWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (runwayEndBlastWrap) runwayEndBlastWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (maxExitWrap) maxExitWrap.style.display = (pt === 'runway_exit') ? 'grid' : 'none';
    if (minExitWrap) minExitWrap.style.display = (pt === 'runway_exit') ? 'grid' : 'none';
    if (rwDirWrap) rwDirWrap.style.display = (pt === 'runway_exit') ? 'grid' : 'none';
    refreshTaxiwayDirectionModeSelect(pt);
  }
  function refreshTaxiwayDirectionModeSelect(pathType) {
    const sel = document.getElementById('taxiwayDirectionMode');
    if (!sel) return;
    const cur = String(sel.value || '').trim();
    const htmlTwo = '<option value="clockwise">CW</option><option value="counter_clockwise">CCW</option>';
    const htmlThree = htmlTwo + '<option value="both">Both</option>';
    sel.innerHTML = (pathType === 'runway') ? htmlTwo : htmlThree;
    if (pathType === 'runway') {
      if (cur === 'clockwise' || cur === 'counter_clockwise') sel.value = cur;
      else sel.value = 'clockwise';
    } else {
      if (cur === 'clockwise' || cur === 'counter_clockwise' || cur === 'both') sel.value = cur;
      else sel.value = 'both';
    }
  }
  function _layoutCellSizeForPersistLoad() {
    return (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
  }
  function layoutVerticesPersistToCellsLoad(vertices) {
    const cs = _layoutCellSizeForPersistLoad();
    if (!Array.isArray(vertices)) return [];
    return vertices.map(function(v) {
      if (!v || typeof v !== 'object') return { col: 0, row: 0 };
      const x = Number(v.x), y = Number(v.y);
      if (isFinite(x) && isFinite(y)) return { col: x / cs, row: y / cs };
      return { col: Number(v.col) || 0, row: Number(v.row) || 0 };
    });
  }
  function layoutPointPersistToCellLoad(pt) {
    if (!pt || typeof pt !== 'object') return null;
    const cs = _layoutCellSizeForPersistLoad();
    const x = Number(pt.x), y = Number(pt.y);
    if (isFinite(x) && isFinite(y)) return { col: x / cs, row: y / cs };
    if (pt.col != null || pt.row != null) return { col: Number(pt.col) || 0, row: Number(pt.row) || 0 };
    return null;
  }
  function normalizeTaxiwayVerticesFromPersistLoad(tw) {
    const o = tw;
    if (!o || typeof o !== 'object') return;
    if (Array.isArray(o.vertices)) o.vertices = layoutVerticesPersistToCellsLoad(o.vertices);
    if (o.start_point) {
      const sp = layoutPointPersistToCellLoad(o.start_point);
      if (sp) o.start_point = sp;
    }
    if (o.end_point) {
      const ep = layoutPointPersistToCellLoad(o.end_point);
      if (ep) o.end_point = ep;
    }
  }
  function mergeTaxiwaysFromLayoutObject(obj) {
    if (!obj || typeof obj !== 'object') return [];
    const newSchema = Object.prototype.hasOwnProperty.call(obj, 'runwayPaths') ||
      Object.prototype.hasOwnProperty.call(obj, 'runwayTaxiways');
    if (newSchema) {
      const out = [];
      (obj.runwayPaths || []).forEach(function(tw) {
        const o = Object.assign({}, tw);
        o.pathType = 'runway';
        normalizeTaxiwayVerticesFromPersistLoad(o);
        out.push(o);
      });
      (obj.runwayTaxiways || []).forEach(function(tw) {
        const o = Object.assign({}, tw);
        o.pathType = 'runway_exit';
        delete o.rwySepConfig;
        normalizeTaxiwayVerticesFromPersistLoad(o);
        out.push(o);
      });
      (obj.taxiways || []).forEach(function(tw) {
        const o = Object.assign({}, tw);
        if (o.pathType !== 'runway' && o.pathType !== 'runway_exit' && o.pathType !== 'apron_taxiway' && o.pathType !== 'general_queue_taxiway') o.pathType = 'taxiway';
        if (o.pathType !== 'runway') delete o.rwySepConfig;
        normalizeTaxiwayVerticesFromPersistLoad(o);
        out.push(o);
      });
      out.forEach(normalizeTaxiwayWidthInPlace);
      out.forEach(normalizePathPavementInPlace);
      return out;
    }
    if (Array.isArray(obj.taxiways)) {
      const sliced = obj.taxiways.slice();
      sliced.forEach(function(tw) {
        normalizeTaxiwayVerticesFromPersistLoad(tw);
        normalizeTaxiwayWidthInPlace(tw);
        normalizePathPavementInPlace(tw);
      });
      return sliced;
    }
    return [];
  }
  function normalizeLayoutMarkerFromLoad(m) {
    if (!m || typeof m !== 'object') return null;
    const k = m.kind || m.type;
    if (k === 'text') {
      const x = Number(m.x), y = Number(m.y);
      if (!isFinite(x) || !isFinite(y)) return null;
      const text = m.text != null ? String(m.text).slice(0, 500) : '';
      return { kind: 'text', id: m.id || id(), x: x, y: y, text: text || 'Text' };
    }
    if (k === 'ruler') {
      const x1 = Number(m.x1), y1 = Number(m.y1), x2 = Number(m.x2), y2 = Number(m.y2);
      if (![x1, y1, x2, y2].every(isFinite)) return null;
      return { kind: 'ruler', id: m.id || id(), x1: x1, y1: y1, x2: x2, y2: y2 };
    }
    if (k === 'island') {
      const rawPts = Array.isArray(m.points) ? m.points : [];
      const points = rawPts.map(function(p) {
        return { x: Number(p && p.x), y: Number(p && p.y) };
      }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); });
      if (points.length < 3) return null;
      let w = Number(m.widthM);
      if (!isFinite(w) || w < 0) {
        const legacy = Number(m.outerWidthM);
        w = (isFinite(legacy) && legacy >= 0) ? Math.min(200, legacy) : LAYOUT_ISLAND_WIDTH_DEFAULT_M;
      } else {
        w = Math.min(200, w);
      }
      return { kind: 'island', id: m.id || id(), points: points, widthM: w };
    }
    if (k === 'area') {
      const rawPts = Array.isArray(m.points) ? m.points : [];
      const points = rawPts.map(function(p) {
        return { x: Number(p && p.x), y: Number(p && p.y) };
      }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); });
      if (points.length < 3) return null;
      return { kind: 'area', id: m.id || id(), points: points };
    }
    if (k === 'flight') {
      if (m.taxiwayId == null || m.taxiwayId === '') return null;
      const segIndex = Math.max(0, parseInt(m.segIndex, 10) || 0);
      let t = Number(m.t);
      if (!isFinite(t)) t = 0.5;
      const leftTrail = Array.isArray(m.blazerLeftTrail) ? m.blazerLeftTrail : [];
      const rightTrail = Array.isArray(m.blazerRightTrail) ? m.blazerRightTrail : [];
      return {
        kind: 'flight',
        id: m.id || id(),
        taxiwayId: m.taxiwayId,
        segIndex: segIndex,
        t: Math.max(0, Math.min(1, t)),
        aircraftType: String(m.aircraftType || '').trim() || ((AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320'),
        blazerEnabled: !!m.blazerEnabled,
        headingReversed: !!m.headingReversed,
        blazerColor: MARKER_BLAZER_COLOR_OPTIONS.indexOf(String(m.blazerColor || '').trim()) >= 0 ? String(m.blazerColor).trim() : MARKER_BLAZER_COLOR_OPTIONS[0],
        blazerLeftTrail: leftTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); }),
        blazerRightTrail: rightTrail.map(function(p) { return { x: Number(p && p.x), y: Number(p && p.y) }; }).filter(function(p) { return isFinite(p.x) && isFinite(p.y); })
      };
    }
    if (k === 'navaid') {
      const x = Number(m.x), y = Number(m.y);
      if (!isFinite(x) || !isFinite(y)) return null;
      const subRaw = String(m.subType || m.sub || 'papi').trim().toLowerCase();
      const sub = subRaw === 'ils' ? 'ils' : 'papi';
      return { kind: 'navaid', id: m.id || id(), subType: sub, x: x, y: y };
    }
    return null;
  }
  function isLayoutPolygonMarkerKind(kind) {
    return kind === 'island' || kind === 'area';
  }
  /** Area markers are drawn under other layout objects; keep them at the front of the array (low z in reverse hit-test). */
  function normalizeLayoutMarkerAreaZOrder(markers) {
    if (!Array.isArray(markers) || !markers.length) return markers || [];
    const areas = [];
    const rest = [];
    for (let i = 0; i < markers.length; i++) {
      const m = markers[i];
      if (!m) continue;
      if (m.kind === 'area') areas.push(m);
      else rest.push(m);
    }
    return areas.concat(rest);
  }
  function _flightApronFallbackStandId(f) {
    if (!f) return null;
    const t = f.token && typeof f.token === 'object' ? f.token : {};
    const raw = f.depApronId != null ? f.depApronId
      : (f.standId != null ? f.standId
        : (t.apronId != null ? t.apronId
          : (f.arrApronId != null ? f.arrApronId : null)));
    return raw != null && String(raw).trim() !== '' ? String(raw) : null;
  }
  function _flightApronBaseSibtMin(f) {
    const v = f && f.sibtMin != null ? Number(f.sibtMin) : (f && f.timeMin != null ? Number(f.timeMin) : 0);
    return isFinite(v) ? Math.max(0, v) : 0;
  }
  function _flightApronBaseSobtMin(f, sibtMin) {
    const raw = f && f.sobtMin != null ? Number(f.sobtMin) : NaN;
    if (isFinite(raw) && raw > sibtMin) return raw;
    const dwell = f && f.dwellMin != null && isFinite(Number(f.dwellMin)) ? Math.max(0, Number(f.dwellMin)) : 0;
    return Math.max(sibtMin, sibtMin + dwell);
  }
  function normalizeFlightApronStaySegments(f) {
    if (!f) return [];
    const fallbackStandId = _flightApronFallbackStandId(f);
    const rawSegs = Array.isArray(f.apronStaySegments) ? f.apronStaySegments : [];
    let segs = rawSegs.map(function(seg) {
      if (!seg || typeof seg !== 'object') return null;
      const sibt = Number(seg.sibtMin);
      const sobt = Number(seg.sobtMin);
      if (!isFinite(sibt) || !isFinite(sobt) || sobt <= sibt) return null;
      const sidRaw = seg.standId != null ? seg.standId : fallbackStandId;
      return {
        standId: sidRaw != null && String(sidRaw).trim() !== '' ? String(sidRaw) : null,
        sibtMin: Math.max(0, sibt),
        sobtMin: Math.max(0, sobt)
      };
    }).filter(Boolean);
    if (!segs.length) {
      const sibt = _flightApronBaseSibtMin(f);
      segs = [{
        standId: fallbackStandId,
        sibtMin: sibt,
        sobtMin: _flightApronBaseSobtMin(f, sibt)
      }];
    }
    segs.sort(function(a, b) {
      if (a.sibtMin !== b.sibtMin) return a.sibtMin - b.sibtMin;
      return a.sobtMin - b.sobtMin;
    });
    f.apronStaySegments = segs;
    return segs;
  }
  function mergeAdjacentSameStandApronSegments(segs) {
    const out = [];
    (segs || []).forEach(function(seg) {
      if (!seg) return;
      const sid = seg.standId != null && String(seg.standId).trim() !== '' ? String(seg.standId) : null;
      const sibt = Number(seg.sibtMin);
      const sobt = Number(seg.sobtMin);
      if (!isFinite(sibt) || !isFinite(sobt) || sobt <= sibt) return;
      const prev = out.length ? out[out.length - 1] : null;
      if (prev && String(prev.standId || '') === String(sid || '') && sibt <= prev.sobtMin + 1e-6) {
        prev.sobtMin = Math.max(prev.sobtMin, sobt);
      } else {
        out.push({ standId: sid, sibtMin: sibt, sobtMin: sobt });
      }
    });
    return out;
  }
  function syncFlightApronStayAggregate(f) {
    if (!f) return [];
    const segs = normalizeFlightApronStaySegments(f);
    if (!segs.length) return segs;
    const first = segs[0];
    const last = segs[segs.length - 1];
    f.arrApronId = first.standId || null;
    f.depApronId = last.standId || null;
    f.standId = f.depApronId || null;
    if (f.token && typeof f.token === 'object') f.token.apronId = f.depApronId || null;
    f.sibtMin = first.sibtMin;
    f.timeMin = first.sibtMin;
    f.sobtMin = last.sobtMin;
    let dwell = 0;
    for (let i = 0; i < segs.length; i++) dwell += Math.max(0, segs[i].sobtMin - segs[i].sibtMin);
    f.dwellMin = dwell;
    if (typeof SCHED_SIBT_MINUS_SLDT_MIN === 'number') f.sldtMin = Math.max(0, f.sibtMin - SCHED_SIBT_MINUS_SLDT_MIN);
    if (typeof SCHED_STOT_MINUS_SOBT_MIN === 'number') f.stotMin = f.sobtMin + SCHED_STOT_MINUS_SOBT_MIN;
    return segs;
  }
  function serializableApronStaySegmentsForFlight(f) {
    const segs = syncFlightApronStayAggregate(f).map(function(seg) {
      return { standId: seg.standId, sibtMin: seg.sibtMin, sobtMin: seg.sobtMin };
    });
    return mergeAdjacentSameStandApronSegments(segs);
  }
  const APRON_STAY_SPLIT_MIN_PART_MIN = 20;
  function splitFlightApronStaySegmentAtMinute(f, segIdx, cutMin) {
    if (!f || flightBlockedLikeNoWay(f)) return false;
    const segs = normalizeFlightApronStaySegments(f);
    const idx = Math.max(0, parseInt(segIdx, 10) || 0);
    if (idx >= segs.length) return false;
    const seg = segs[idx];
    const cut = Number(cutMin);
    if (!isFinite(cut)) return false;
    const snap = (typeof GANTT_SIBT_SOBT_HANDLE_SNAP_MIN === 'number' && GANTT_SIBT_SOBT_HANDLE_SNAP_MIN > 0)
      ? GANTT_SIBT_SOBT_HANDLE_SNAP_MIN
      : 1;
    const t = Math.max(0, Math.round(cut / snap) * snap);
    if (t - seg.sibtMin < APRON_STAY_SPLIT_MIN_PART_MIN || seg.sobtMin - t < APRON_STAY_SPLIT_MIN_PART_MIN) {
      return false;
    }
    const left = { standId: seg.standId || null, sibtMin: seg.sibtMin, sobtMin: t };
    const right = { standId: seg.standId || null, sibtMin: t, sobtMin: seg.sobtMin };
    segs.splice(idx, 1, left, right);
    f.apronStaySegments = segs;
    syncFlightApronStayAggregate(f);
    return true;
  }
  function _sameApronStayStand(a, b) {
    const sa = a && a.standId != null ? String(a.standId) : '';
    const sb = b && b.standId != null ? String(b.standId) : '';
    return sa === sb;
  }
  function applyApronStaySegmentHandleMinute(f, segIdx, role, minutes) {
    if (!f || flightBlockedLikeNoWay(f)) return false;
    const segs = normalizeFlightApronStaySegments(f);
    const idx = Math.max(0, parseInt(segIdx, 10) || 0);
    if (idx >= segs.length) return false;
    const raw = Number(minutes);
    if (!isFinite(raw)) return false;
    const snap = (typeof GANTT_SIBT_SOBT_HANDLE_SNAP_MIN === 'number' && GANTT_SIBT_SOBT_HANDLE_SNAP_MIN > 0)
      ? GANTT_SIBT_SOBT_HANDLE_SNAP_MIN
      : 1;
    const tRaw = Math.max(0, Math.round(raw / snap) * snap);
    if (role === 'sibt') {
      if (idx === 0) {
        segs[0].sibtMin = Math.min(tRaw, segs[0].sobtMin - APRON_STAY_SPLIT_MIN_PART_MIN);
      } else {
        const prev = segs[idx - 1];
        const cur = segs[idx];
        if (_sameApronStayStand(prev, cur)) return false;
        const lo = prev.sibtMin + APRON_STAY_SPLIT_MIN_PART_MIN;
        const hi = cur.sobtMin - APRON_STAY_SPLIT_MIN_PART_MIN;
        if (hi < lo) return false;
        const t = Math.max(lo, Math.min(hi, tRaw));
        prev.sobtMin = t;
        cur.sibtMin = t;
      }
    } else if (role === 'sobt') {
      if (idx >= segs.length - 1) {
        segs[idx].sobtMin = Math.max(tRaw, segs[idx].sibtMin + APRON_STAY_SPLIT_MIN_PART_MIN);
      } else {
        const cur = segs[idx];
        const next = segs[idx + 1];
        if (_sameApronStayStand(cur, next)) return false;
        const lo = cur.sibtMin + APRON_STAY_SPLIT_MIN_PART_MIN;
        const hi = next.sobtMin - APRON_STAY_SPLIT_MIN_PART_MIN;
        if (hi < lo) return false;
        const t = Math.max(lo, Math.min(hi, tRaw));
        cur.sobtMin = t;
        next.sibtMin = t;
      }
    } else {
      return false;
    }
    f.apronStaySegments = segs;
    syncFlightApronStayAggregate(f);
    return true;
  }
  function buildApronStayGanttIntervalsForFlight(f, eSer) {
    const segs = normalizeFlightApronStaySegments(f);
    if (!segs.length) return [];
    syncFlightApronStayAggregate(f);
    const first = segs[0];
    const last = segs[segs.length - 1];
    const sldt = f.sldtMin != null ? f.sldtMin : Math.max(0, first.sibtMin - SCHED_SIBT_MINUS_SLDT_MIN);
    const stot = f.stotMin != null ? f.stotMin : (last.sobtMin + SCHED_STOT_MINUS_SOBT_MIN);
    return segs.map(function(seg, idx) {
      return {
        f: f,
        t0: seg.sibtMin,
        t1: seg.sobtMin,
        sldt: sldt,
        stot: stot,
        eibt: eSer.eibt,
        eobt: eSer.eobt,
        eldt: eSer.eldt,
        etot: eSer.etot,
        sldtOrig: sldt,
        sobtOrig: last.sobtMin,
        stotOrig: stot,
        segmentIdx: idx,
        segmentCount: segs.length,
        segmentStandId: seg.standId || null
      };
    });
  }
  function applyLayoutObject(obj) {
    if (!obj || typeof obj !== 'object') return;
    state.simPlaybackEndCapSec = null;
    const dp = obj.designerPersist;
    const restoreProSimSyncUi = !!(dp && dp.v === 1 && dp.globalUpdateFresh === true);
    if (obj.grid) {
      if (typeof obj.grid.cols === 'number') GRID_COLS = obj.grid.cols;
      if (typeof obj.grid.rows === 'number') GRID_ROWS = obj.grid.rows;
      if (typeof obj.grid.cellSize === 'number') CELL_SIZE = obj.grid.cellSize;
    }
    hydrateLayersFromGridObject(obj.grid || null, obj);
    state.layoutImageOverlay = normalizeLayoutImageOverlay(
      (obj.grid && obj.grid.layoutImageOverlay) || obj.layoutImageOverlay || null
    );
    invalidateGridUnderlay();
    syncLayoutImageBitmap();
    syncLayerPopoverFromState();
    if (Array.isArray(obj.terminals)) state.terminals = obj.terminals.map(normalizeBuildingObject);
    if (Array.isArray(obj.pbbStands)) state.pbbStands = obj.pbbStands.map(normalizePbbStandObject);
    if (Array.isArray(obj.remoteStands)) state.remoteStands = obj.remoteStands.map(normalizeRemoteStandObject);
    if (Array.isArray(obj.tempStands)) state.tempStands = obj.tempStands.map(normalizeTempStandObject);
    else state.tempStands = [];
    state.taxiways = mergeTaxiwaysFromLayoutObject(obj);
    invalidatePathGraphCache(true);
    if (Array.isArray(obj.holdingPoints)) {
      state.holdingPoints = obj.holdingPoints.map(function(h) {
        const hx = Number(h && h.x);
        const hy = Number(h && h.y);
        let hpKind = null;
        if (h && h.hpKind != null) hpKind = normalizeHoldingPointKind(h.hpKind);
        if (!hpKind) {
          const snap = snapHoldingPointOnAllowedTaxiways(hx, hy);
          hpKind = (snap && snap.pathType) ? pathTypeToHpKind(snap.pathType) : 'intermediate';
        }
        return {
          id: (h && h.id) ? h.id : id(),
          name: h && h.name != null ? String(h.name) : '',
          x: hx,
          y: hy,
          hpKind: hpKind
        };
      }).filter(function(h) { return h && isFinite(h.x) && isFinite(h.y); });
    } else state.holdingPoints = [];
    if (Array.isArray(obj.apronLinks)) {
      const csAL = _layoutCellSizeForPersistLoad();
      state.apronLinks = obj.apronLinks.map(function(lk) {
        const copy = Object.assign({}, lk);
        if (Array.isArray(copy.midVertices)) {
          copy.midVertices = copy.midVertices.map(function(v) {
            if (!v || typeof v !== 'object') return { col: 0, row: 0 };
            const x = Number(v.x), y = Number(v.y);
            if (isFinite(x) && isFinite(y)) return { col: x / csAL, row: y / csAL };
            return { col: Number(v.col) || 0, row: Number(v.row) || 0 };
          });
        }
        return copy;
      });
    }
    if (Array.isArray(obj.directionModes) && obj.directionModes.length) {
      state.directionModes = obj.directionModes.slice();
    }
    if (Array.isArray(obj.layoutMarkers)) {
      state.layoutMarkers = normalizeLayoutMarkerAreaZOrder(obj.layoutMarkers.map(normalizeLayoutMarkerFromLoad).filter(Boolean));
    } else if (!Array.isArray(state.layoutMarkers)) {
      state.layoutMarkers = [];
    }
    if (Array.isArray(obj.flights)) {
      state.simPlaybackPositionsByFlightId = null;
      state.simPlaybackScheduleSnapshot = null;
      state.simPlaybackTimelinesEvictedForMemory = false;
      state.flights = obj.flights.slice();
      state.flights.forEach(f => {
        const rawTl = Array.isArray(f.timeline) ? f.timeline : null;
        const rawMeta = f.timeline_meta;
        const rawProSimEl = Array.isArray(f.proSimEdgeList) ? f.proSimEdgeList.slice() : null;
        const restorePlaybackFlight = !!(rawTl && rawTl.length >= 2);
        const t = f.token || {};
        const exitTwPersist = (t.ExitTaxiwayId != null && t.ExitTaxiwayId !== '') ? t.ExitTaxiwayId : null;
        const arrRetFailedPersist = f.arrRetFailed === true;
        if (f.aircraftType && typeof getCodeForAircraft === 'function') {
          f.code = getCodeForAircraft(f.aircraftType);
        } else if (f.code && typeof AIRCRAFT_TYPES !== 'undefined') {
          const match = AIRCRAFT_TYPES.find(a => a.icao === f.code);
          f.aircraftType = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
        }
        f.arrRunwayId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
        f.depRunwayId = f.depRunwayId || t.depRunwayId || null;
        f.terminalId = f.terminalId || t.terminalId || null;
        const apronId = f.depApronId != null ? f.depApronId : (t.apronId != null ? t.apronId : (f.standId != null ? f.standId : f.arrApronId || null));
        f.standId = apronId;
        f.token = {
          nodes: Array.isArray(t.nodes) ? t.nodes.slice() : ['runway','taxiway','apron','terminal'],
          runwayId: f.arrRunwayId || null,
          apronId: apronId,
          terminalId: f.terminalId || null,
          depRunwayId: f.depRunwayId || null,
        };
        if (exitTwPersist) f.token.ExitTaxiwayId = exitTwPersist;
        f.noWayArr = false;
        f.noWayDep = false;
        delete f._noWayArrDetail;
        delete f._noWayDepDetail;
        if (restoreProSimSyncUi && f.arrDep !== 'Dep') {
          if (exitTwPersist) {
            f.sampledArrRet = exitTwPersist;
            f.arrRetFailed = arrRetFailedPersist;
          } else {
            f.sampledArrRet = null;
            f.arrRetFailed = false;
          }
        } else {
          f.arrRetFailed = false;
          f.sampledArrRet = null;
        }
        if (!restorePlaybackFlight) {
          f.arrRotSec = null;
          delete f.proSimVttArrSec;
          delete f.proSimVttDepSec;
          delete f.proSimDttArrSec;
          delete f.proSimPushbackSec;
          delete f.proSimDttDepSec;
          delete f.proSimDepLineupSec;
          f.arrRunwayIdUsed = null;
          f.arrTdDistM = null;
          f.arrRetDistM = null;
          f.arrVTdMs = null;
          f.arrDecelMs2 = null;
          f.arrVRetInMs = null;
          f.arrVRetOutMs = null;
          f.timeline = null;
          delete f.timeline_meta;
          delete f.proSimEdgeList;
          delete f.eldtMin;
          delete f.eibtMin;
          delete f.eobtMin;
          delete f.etotMin;
          delete f.eldtMin;
          delete f.eibtMin;
          delete f.eobtMin;
          delete f.etotMin;
        } else {
          const tlNorm = rawTl.map(function(p) {
            const x = p.x != null && p.x !== '' ? Number(p.x) : Number(p.col);
            const y = p.y != null && p.y !== '' ? Number(p.y) : Number(p.row);
            const dg = p.deadlockGhost === true || p.deadlock_ghost === true;
            const o = { t: Number(p.t), x: x, y: y, deadlockGhost: dg };
            if (p.pathType != null && p.pathType !== '') o.pathType = String(p.pathType);
            if (p.phase != null && p.phase !== '') o.phase = String(p.phase);
            if (p.edgeId != null && String(p.edgeId).trim()) o.edgeId = String(p.edgeId).trim();
            return o;
          }).filter(function(k) {
            return isFinite(k.t) && isFinite(k.x) && isFinite(k.y);
          }).sort(function(a, b) { return a.t - b.t; });
          if (tlNorm.length >= 2) {
            f.timeline = tlNorm;
            if (rawMeta && typeof rawMeta === 'object') f.timeline_meta = Object.assign({}, rawMeta);
            else delete f.timeline_meta;
            if (rawProSimEl && rawProSimEl.length) {
              f.proSimEdgeList = rawProSimEl;
            } else if (Array.isArray(f.edge_list) && f.edge_list.length) {
              f.proSimEdgeList = f.edge_list.slice();
            }
          } else {
            f.arrRotSec = null;
            delete f.proSimVttArrSec;
            delete f.proSimVttDepSec;
            delete f.proSimDttArrSec;
            delete f.proSimPushbackSec;
            delete f.proSimDttDepSec;
            delete f.proSimDepLineupSec;
            f.arrRunwayIdUsed = null;
            f.arrTdDistM = null;
            f.arrRetDistM = null;
            f.arrVTdMs = null;
            f.arrDecelMs2 = null;
            f.arrVRetInMs = null;
            f.arrVRetOutMs = null;
            f.timeline = null;
            delete f.timeline_meta;
            delete f.proSimEdgeList;
            delete f.eldtMin;
            delete f.eibtMin;
            delete f.eobtMin;
            delete f.etotMin;
            delete f.eldtMin;
            delete f.eibtMin;
            delete f.eobtMin;
            delete f.etotMin;
          }
        }
        delete f.cachedArrPathPts;
        delete f.cachedDepPathPts;
        delete f._pathPolylineCacheRev;
        delete f._pathPolylineArrRetKey;
        f.__schedRetRotRev = null;
        f.__schedVttArrRev = null;
        f.__schedVttArrMin = null;
        if (!f.airlineCode) f.airlineCode = DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)];
        if (!f.flightNumber) f.flightNumber = f.airlineCode + String(Math.floor(1000 + Math.random() * 9000));
        if (!String(f.reg || '').trim()) f.reg = randomRegNumber();
        {
          const idRaw = String(f.intDom || '').trim();
          f.intDom = (idRaw.toLowerCase() === 'dom') ? 'Dom' : 'Int';
        }
        if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
      });
    } else {
      state.flights = [];
    }
    if (Object.prototype.hasOwnProperty.call(obj, '_airsideSimApply')) delete obj._airsideSimApply;
    state.simPlaying = false;
    state.layoutPathDrawPointer = null;
    if (typeof refreshHasSimulationResultFromPlaybackSources === 'function') {
      refreshHasSimulationResultFromPlaybackSources();
    } else {
      let playbackFlightCount = 0;
      (state.flights || []).forEach(function(f) {
        if (f && f.timeline && f.timeline.length >= 2) playbackFlightCount++;
      });
      state.hasSimulationResult = playbackFlightCount > 0;
