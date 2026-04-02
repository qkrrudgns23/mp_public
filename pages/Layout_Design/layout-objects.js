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
    return 'taxiway';
  }
  function isPathLayoutMode(m) {
    return PATH_LAYOUT_MODES.indexOf(m) >= 0;
  }
  function settingModeValueForHit(hit) {
    if (!hit || !hit.type) return null;
    if (hit.type === 'terminal') return 'terminal';
    if (hit.type === 'pbb') return 'pbb';
    if (hit.type === 'remote') return 'remote';
    if (hit.type === 'holdingPoint') return 'holdingPoint';
    if (hit.type === 'taxiway') return layoutModeFromPathType((hit.obj && hit.obj.pathType) || 'taxiway');
    if (hit.type === 'apronLink') return 'apronTaxiway';
    return null;
  }
  function cancelActiveLayoutDrawingState() {
    state.pbbDrawing = false;
    state.remoteDrawing = false;
    state.holdingPointDrawing = false;
    state.previewHoldingPoint = null;
    state.apronLinkDrawing = false;
    state.apronLinkTemp = null;
    state.apronLinkMidpoints = [];
    state.apronLinkPointerWorld = null;
    state.layoutPathDrawPointer = null;
    state.previewPbb = null;
    state.previewRemote = null;
  }
  function syncDrawToggleButton(elementId, isDrawing) {
    const btn = document.getElementById(elementId);
    if (!btn) return;
    btn.textContent = isDrawing ? 'Drawing' : 'Draw';
    btn.classList.toggle('drawing', isDrawing);
  }
  function syncGridToggleButton() {
    if (!gridToggleBtn) return;
    const on = !!state.showGrid;
    gridToggleBtn.classList.toggle('active', on);
    gridToggleBtn.title = on ? 'Grid visible (click to hide)' : 'Grid hidden (click to show)';
  }
  function syncImageToggleButton() {
    if (!imageToggleBtn) return;
    const on = !!state.showImage;
    imageToggleBtn.classList.toggle('active', on);
    imageToggleBtn.title = on ? 'Image visible (click to hide)' : 'Image hidden (click to show)';
  }
  function syncRoadWidthToggleButton() {
    if (!roadWidthToggleBtn) return;
    const on = !!state.showRoadWidth;
    roadWidthToggleBtn.classList.toggle('active', on);
    roadWidthToggleBtn.title = on
      ? 'Road width visible (click for schematic centerlines)'
      : 'Schematic mode (click to show road width)';
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
  }
  function findLayoutObjectByListType(typ, idr) {
    if (typ === 'terminal') return state.terminals.find(t => t.id === idr);
    if (typ === 'pbb') return state.pbbStands.find(p => p.id === idr);
    if (typ === 'remote') return state.remoteStands.find(r => r.id === idr);
    if (typ === 'holdingPoint') return (state.holdingPoints || []).find(h => h.id === idr);
    if (typ === 'taxiway') return state.taxiways.find(tw => tw.id === idr);
    if (typ === 'apronLink') return state.apronLinks.find(lk => lk.id === idr);
    if (typ === 'layoutEdge') return (state.derivedGraphEdges || []).find(function(e) { return e.id === idr; });
    if (typ === 'flight') return state.flights.find(f => f.id === idr);
    return null;
  }
  function removeLayoutObjectFromState(type, id) {
    const removedTaxiway = (type === 'taxiway')
      ? (state.taxiways || []).find(function(tw) { return tw.id === id; })
      : null;
    if (type === 'terminal') state.terminals = state.terminals.filter(t => t.id !== id);
    else if (type === 'pbb') state.pbbStands = state.pbbStands.filter(p => p.id !== id);
    else if (type === 'remote') state.remoteStands = state.remoteStands.filter(r => r.id !== id);
    else if (type === 'holdingPoint') state.holdingPoints = (state.holdingPoints || []).filter(h => h.id !== id);
    else if (type === 'taxiway') state.taxiways = state.taxiways.filter(tw => tw.id !== id);
    else if (type === 'apronLink') state.apronLinks = state.apronLinks.filter(lk => lk.id !== id);
    else if (type === 'flight') {
      state.flights = state.flights.filter(f => f.id !== id);
      bumpRwySepSnapshotStaleGen();
      state.rwySepPanelDirty = true;
    }
    else if (type === 'layoutEdge') {}
    if (removedTaxiway) {
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
    if (taxiwayAvgWrap) taxiwayAvgWrap.style.display = (pt === 'taxiway') ? 'grid' : 'none';
    if (runwayMinArrWrap) runwayMinArrWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
    if (runwayLineupWrap) runwayLineupWrap.style.display = (pt === 'runway') ? 'grid' : 'none';
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
        if (o.pathType !== 'runway' && o.pathType !== 'runway_exit') o.pathType = 'taxiway';
        if (o.pathType !== 'runway') delete o.rwySepConfig;
        normalizeTaxiwayVerticesFromPersistLoad(o);
        out.push(o);
      });
      out.forEach(normalizeTaxiwayWidthInPlace);
      return out;
    }
    if (Array.isArray(obj.taxiways)) {
      const sliced = obj.taxiways.slice();
      sliced.forEach(function(tw) {
        normalizeTaxiwayVerticesFromPersistLoad(tw);
        normalizeTaxiwayWidthInPlace(tw);
      });
      return sliced;
    }
    return [];
  }
  function applyLayoutObject(obj) {
    if (!obj || typeof obj !== 'object') return;
    if (obj.grid) {
      if (typeof obj.grid.cols === 'number') GRID_COLS = obj.grid.cols;
      if (typeof obj.grid.rows === 'number') GRID_ROWS = obj.grid.rows;
      if (typeof obj.grid.cellSize === 'number') CELL_SIZE = obj.grid.cellSize;
      if (typeof obj.grid.showGrid === 'boolean') state.showGrid = obj.grid.showGrid;
      if (typeof obj.grid.showImage === 'boolean') state.showImage = obj.grid.showImage;
    }
    if (typeof obj.showGrid === 'boolean') state.showGrid = obj.showGrid;
    if (typeof obj.showImage === 'boolean') state.showImage = obj.showImage;
    state.layoutImageOverlay = normalizeLayoutImageOverlay(
      (obj.grid && obj.grid.layoutImageOverlay) || obj.layoutImageOverlay || null
    );
    invalidateGridUnderlay();
    syncLayoutImageBitmap();
    syncGridToggleButton();
    syncImageToggleButton();
    if (Array.isArray(obj.terminals)) state.terminals = obj.terminals.map(normalizeBuildingObject);
    if (Array.isArray(obj.pbbStands)) state.pbbStands = obj.pbbStands.map(normalizePbbStandObject);
    if (Array.isArray(obj.remoteStands)) state.remoteStands = obj.remoteStands.map(normalizeRemoteStandObject);
    state.taxiways = mergeTaxiwaysFromLayoutObject(obj);
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
    if (Array.isArray(obj.flights)) {
      state.flights = obj.flights.slice();
      state.flights.forEach(f => {
        const t = f.token || {};
        if (f.aircraftType && typeof getCodeForAircraft === 'function') {
          f.code = getCodeForAircraft(f.aircraftType);
        } else if (f.code && typeof AIRCRAFT_TYPES !== 'undefined') {
          const match = AIRCRAFT_TYPES.find(a => a.icao === f.code);
          f.aircraftType = match ? match.id : (AIRCRAFT_TYPES[0] && AIRCRAFT_TYPES[0].id) || 'A320';
        }
        f.arrRunwayId = f.arrRunwayId || t.arrRunwayId || t.runwayId || null;
        f.depRunwayId = f.depRunwayId || t.depRunwayId || null;
        f.terminalId = f.terminalId || t.terminalId || null;
        const apronId = t.apronId != null ? t.apronId : (f.standId != null ? f.standId : null);
        f.standId = apronId;
        f.token = {
          nodes: Array.isArray(t.nodes) ? t.nodes.slice() : ['runway','taxiway','apron','terminal'],
          runwayId: f.arrRunwayId || null,
          apronId: apronId,
          terminalId: f.terminalId || null,
          depRunwayId: f.depRunwayId || null,
        };
        f.noWayArr = false;
        f.noWayDep = false;
        delete f._noWayArrDetail;
        delete f._noWayDepDetail;
        f.arrRetFailed = false;
        f.sampledArrRet = null;
        f.arrRotSec = null;
        f.arrRunwayIdUsed = null;
        f.arrTdDistM = null;
        f.arrRetDistM = null;
        f.arrVTdMs = null;
        f.arrDecelMs2 = null;
        f.arrVRetInMs = null;
        f.arrVRetOutMs = null;
        f.timeline = null;
        delete f.timeline_meta;
        delete f.cachedArrPathPts;
        delete f.cachedDepPathPts;
        delete f._pathPolylineCacheRev;
        delete f._pathPolylineArrRetKey;
        f.__schedRetRotRev = null;
        f.__schedVttArrRev = null;
        f.__schedVttArrMin = null;
        if (!f.airlineCode) f.airlineCode = DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)];
        if (!f.flightNumber) f.flightNumber = f.airlineCode + String(Math.floor(1000 + Math.random() * 9000));
        f.deferPathCompute = true;
        delete f.eldtMin;
        delete f.eibtMin;
        delete f.eobtMin;
        delete f.etotMin;
        delete f.eldtMin_orig;
        delete f.eibtMin_orig;
        delete f.eobtMin_orig;
        delete f.etotMin_orig;
      });
    } else {
      state.flights = [];
    }
    if (Object.prototype.hasOwnProperty.call(obj, '_airsideSimApply')) delete obj._airsideSimApply;
    state.simPlaying = false;
    state.layoutPathDrawPointer = null;
    state.hasSimulationResult = false;
    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
    else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else {
      draw();
    }
    if (Array.isArray(state.flights) && state.flights.length) {
      const flightTabBtn = document.querySelector('.right-panel-tab[data-tab="flight"]');
      if (flightTabBtn) flightTabBtn.click();
      if (typeof renderFlightList === 'function') {
        renderFlightList(false, true);
      }
    } else if (typeof renderFlightList === 'function') {
      renderFlightList();
    }
  }
  function applyAirsideSimulationResultPayload(payload) {
    if (!payload || typeof payload !== 'object') return;
    const flightsDetail = Array.isArray(payload.flights_detail) ? payload.flights_detail : null;
    if (flightsDetail) {
      const byId = {};
      flightsDetail.forEach(function(row) {
        if (!row || row.flight_id == null) return;
        const raw = row.edge_list;
        byId[String(row.flight_id)] = Array.isArray(raw) ? raw : [];
      });
      (state.flights || []).forEach(function(f) {
        if (!f || f.id == null) return;
        const raw = byId[String(f.id)];
        if (Array.isArray(raw) && raw.length) {
          f.edge_list = raw.slice();
          f.proSimEdgeList = f.edge_list.slice();
        } else {
          delete f.edge_list;
          delete f.proSimEdgeList;
        }
      });
    }
    const positions = payload.positions;
    const hasPositions = positions && typeof positions === 'object' && Object.keys(positions).length > 0;
    const scheduleList = Array.isArray(payload.schedule) ? payload.schedule : [];
    const layout = payload.layout;
    if (layout && typeof layout === 'object') {
      applyLayoutObject(layout);
    }
    const schedById = {};
    scheduleList.forEach(function(s) {
      if (s && s.flight_id != null) schedById[String(s.flight_id)] = s;
    });
    function secFromSimSched(srec, eKey, legacyAKey, sKey) {
      if (srec[eKey] != null && srec[eKey] !== '') return Number(srec[eKey]);
      if (legacyAKey && srec[legacyAKey] != null && srec[legacyAKey] !== '') return Number(srec[legacyAKey]);
      if (sKey && srec[sKey] != null && srec[sKey] !== '') return Number(srec[sKey]);
      return NaN;
    }
    let mergedTimelines = 0;
    if (hasPositions) {
      (state.flights || []).forEach(function(f) {
        if (!f || f.id == null) return;
        const rawPts = positions[f.id];
        if (rawPts == null) return;
        const pts = Array.isArray(rawPts) ? rawPts : [];
        if (pts.length < 2) return;
        const tl = pts.map(function(p) {
          const x = p.x != null && p.x !== '' ? Number(p.x) : Number(p.col);
          const y = p.y != null && p.y !== '' ? Number(p.y) : Number(p.row);
          return { t: Number(p.t), x: x, y: y };
        }).filter(function(k) {
          return isFinite(k.t) && isFinite(k.x) && isFinite(k.y);
        }).sort(function(a, b) { return a.t - b.t; });
        if (tl.length < 2) return;
        mergedTimelines++;
        f.timeline = tl;
        const srec = schedById[String(f.id)] || {};
        const eldtS = secFromSimSched(srec, 'ELDT', 'ALDT', 'SLDT');
        const eibtS = secFromSimSched(srec, 'EIBT', 'AIBT', 'SIBT');
        const eobtS = secFromSimSched(srec, 'EOBT', 'AOBT', 'SOBT');
        const etotS = secFromSimSched(srec, 'ETOT', 'ATOT', 'STOT');
        f.timeline_meta = {
          playbackSource: 'des_result',
          eldtSec: isFinite(eldtS) ? eldtS : undefined,
          eibtSec: isFinite(eibtS) ? eibtS : undefined,
          eobtSec: isFinite(eobtS) ? eobtS : undefined,
          etotSec: isFinite(etotS) ? etotS : undefined,
        };
        if (isFinite(eldtS)) f.eldtMin = eldtS / 60;
        if (isFinite(eibtS)) f.eibtMin = eibtS / 60;
        if (isFinite(eobtS)) f.eobtMin = eobtS / 60;
        if (isFinite(etotS)) f.etotMin = etotS / 60;
      });
      (state.flights || []).forEach(function(ff) {
        if (ff) ff.deferPathCompute = true;
      });
    }
    state.hasSimulationResult = mergedTimelines > 0;
    if (state.hasSimulationResult) {
      if (typeof markGlobalUpdateFresh === 'function') markGlobalUpdateFresh();
    } else if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof syncSimulationPlaybackAfterTimelines === 'function') syncSimulationPlaybackAfterTimelines();
    else if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (typeof resizeCanvas === 'function') resizeCanvas();
    if (typeof reset2DView === 'function') reset2DView();
    if (typeof syncPanelFromState === 'function') syncPanelFromState();
    if (typeof renderFlightList === 'function') renderFlightList(false, false);
    if (typeof renderKpiDashboard === 'function') renderKpiDashboard('Updated');
    if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
    if (typeof draw === 'function') draw();
    if (typeof scene3d !== 'undefined' && scene3d && typeof update3DScene === 'function') update3DScene();
    const applyBtn = document.getElementById('btnApplySimResult');
    const playDockBtn = document.getElementById('btnShowPlayDock');
    if (applyBtn) applyBtn.disabled = true;
    if (playDockBtn) playDockBtn.disabled = !state.hasSimulationResult;
  }
  function applyInitialLayoutFromJson() {
    if (!INITIAL_LAYOUT || typeof INITIAL_LAYOUT !== 'object') return;
    applyLayoutObject(INITIAL_LAYOUT);
  }
  function updateLayoutNameBar(name) {
    const n = (name && String(name).trim()) || '';
    state.currentLayoutName = n || state.currentLayoutName || 'default_layout';
    const bar = document.getElementById('layout-name-bar');
    if (bar) bar.textContent = n || state.currentLayoutName;
  }
  function uniqueNameAgainstSet(baseName, usedNames) {
    const base = (baseName && String(baseName).trim()) || 'Untitled';
    const used = usedNames instanceof Set ? usedNames : new Set();
    if (!used.has(base)) return base;
    let idx = 1;
    while (used.has(base + ' (' + idx + ')')) idx++;
    return base + ' (' + idx + ')';
  }
  function zeroPadNumber(num, width) {
    return String(Math.max(0, Number(num) || 0)).padStart(width, '0');
  }
  function getDefaultPathName(pathType, currentId) {
    const prefix = pathType === 'runway' ? 'RW' : (pathType === 'runway_exit' ? 'RTX' : 'TX');
    const sameType = (state.taxiways || []).filter(function(tw) { return tw && tw.id !== currentId && tw.pathType === pathType; });
    const used = new Set(sameType.map(function(tw) { return (tw.name && String(tw.name).trim()) || ''; }).filter(Boolean));
    let n = 1;
    let candidate = prefix + String(n);
    while (used.has(candidate)) {
      n++;
      candidate = prefix + String(n);
      if (n > 100000) break;
    }
    return candidate;
  }
  function getDefaultTerminalName(currentId) {
    return getDefaultBuildingNameForType(BUILDING_TYPE_DEFAULT, currentId);
  }
  function getDefaultPbbStandName(currentId) {
    const stands = (state.pbbStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('C' + zeroPadNumber(stands.length + 1, 3), used);
  }
  function getDefaultRemoteStandName(currentId) {
    const stands = (state.remoteStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('R' + zeroPadNumber(stands.length + 1, 3), used);
  }
  function getApronLinkDefaultName(linkOrId) {
    const linkId = (typeof linkOrId === 'object' && linkOrId) ? linkOrId.id : linkOrId;
    const idx = (state.apronLinks || []).findIndex(function(lk) { return lk && lk.id === linkId; });
    return 'Apron Taxiway ' + String(idx >= 0 ? idx + 1 : ((state.apronLinks || []).length + 1));
  }
  function getApronLinkDisplayName(link) {
    if (!link) return 'Apron Taxiway';
    return (link.name && String(link.name).trim()) || getApronLinkDefaultName(link);
  }
  function ensureUniqueApronLinkName(rawName, currentId) {
    const fallbackBase = getApronLinkDefaultName(currentId);
    const baseName = (rawName && String(rawName).trim()) || fallbackBase;
    const used = new Set((state.apronLinks || [])
      .filter(function(lk) { return lk && lk.id !== currentId; })
      .map(function(lk) { return (lk.name && String(lk.name).trim()) || getApronLinkDefaultName(lk); })
      .filter(Boolean));
    return uniqueNameAgainstSet(baseName, used);
  }
  function getLayoutEdgeDefaultName(edge) {
    if (!edge) return 'Edge';
    return 'Edge ' + (edge.label || '001');
  }
  function getLayoutEdgeDisplayName(edge) {
    if (!edge) return 'Edge';
    return (edge.name && String(edge.name).trim()) || getLayoutEdgeDefaultName(edge);
  }
  function ensureUniqueLayoutEdgeName(rawName, currentId, fallbackEdge) {
    const fallbackBase = getLayoutEdgeDefaultName(fallbackEdge || { label: '001' });
    const baseName = (rawName && String(rawName).trim()) || fallbackBase;
    const used = new Set(Object.keys(state.layoutEdgeNames || {})
      .filter(function(id) { return id !== currentId; })
      .map(function(id) { return state.layoutEdgeNames[id]; })
      .filter(Boolean));
    return uniqueNameAgainstSet(baseName, used);
  }
  function normalizeLayoutNameKey(name) {
    return String(name || '').trim().toLowerCase();
  }
  function findDuplicateLayoutName(objectKind, excludeId, proposedRaw) {
    const key = normalizeLayoutNameKey(proposedRaw);
    if (!key) return null;
    const ex = excludeId == null || excludeId === '' ? null : String(excludeId);
    function isOther(oid) {
      if (ex === null) return true;
      return String(oid) !== ex;
    }
    if (objectKind === 'terminal') {
      const arr = state.terminals || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'terminal', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'pbb') {
      const arr = state.pbbStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'pbb', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'remote') {
      const arr = state.remoteStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'remote', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'holdingPoint') {
      const arr = state.holdingPoints || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'holdingPoint', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'taxiway') {
      const arr = state.taxiways || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'taxiway', existing: disp || o.id };
      }
      return null;
    }
    if (objectKind === 'apronLink') {
      const arr = state.apronLinks || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = getApronLinkDisplayName(o);
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'apronLink', existing: disp };
      }
      return null;
    }
    if (objectKind === 'layoutEdge') {
      const map = state.layoutEdgeNames || {};
      const edgeIds = Object.keys(map);
      for (let ki = 0; ki < edgeIds.length; ki++) {
        const kid = edgeIds[ki];
        if (!isOther(kid)) continue;
        const disp = map[kid];
        if (disp != null && normalizeLayoutNameKey(disp) === key) return { kind: 'layoutEdge', existing: String(disp) };
      }
      return null;
    }
    return null;
  }
  function alertDuplicateLayoutName() {
    alert('설정 불가: 동일한 이름이 이미 사용 중입니다.');
  }
  function ensureDefaultDirectionModes() {
    if (state.directionModes.length === 0) {
      state.directionModes = [
        { id: id(), name: 'Mode A', direction: 'clockwise' },
        { id: id(), name: 'Mode B', direction: 'counter_clockwise' },
        { id: id(), name: 'Mode C', direction: 'both' }
      ];
    }
  }
  const undoStack = [];
  const maxUndoLevels = _interactionConfigNum('maxUndoLevels', 50);
  function pushUndo() {
    const snap = {
      terminals: JSON.parse(JSON.stringify(state.terminals || [])),
      pbbStands: JSON.parse(JSON.stringify(state.pbbStands || [])),
      remoteStands: JSON.parse(JSON.stringify(state.remoteStands || [])),
      holdingPoints: JSON.parse(JSON.stringify(state.holdingPoints || [])),
      taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
      apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
      layoutImageOverlay: JSON.parse(JSON.stringify(state.layoutImageOverlay || null)),
      layoutEdgeNames: JSON.parse(JSON.stringify(state.layoutEdgeNames || {})),
      directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
      flights: cloneFlightsWithoutPathPolylineCache(state.flights)
    };
    undoStack.push(snap);
    if (undoStack.length > maxUndoLevels) undoStack.shift();
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
  }
  function undo() {
    if (!undoStack.length) return;
    const snap = undoStack.pop();
    state.terminals = snap.terminals;
    state.pbbStands = snap.pbbStands;
    state.remoteStands = snap.remoteStands;
    state.holdingPoints = snap.holdingPoints || [];
    state.taxiways = snap.taxiways;
    state.apronLinks = snap.apronLinks;
    state.layoutImageOverlay = normalizeLayoutImageOverlay(snap.layoutImageOverlay);
    syncLayoutImageBitmap();
    state.layoutEdgeNames = snap.layoutEdgeNames || {};
    state.directionModes = snap.directionModes;
    state.flights = snap.flights;
    state.selectedObject = null;
    state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
    state.terminalDrawingId = null;
    state.taxiwayDrawingId = null;
    state.layoutPathDrawPointer = null;
    syncPanelFromState();
    updateObjectInfo();
    renderObjectList();
    if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
    else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
  }
  function getTaxiwayDirection(tw) {
    if (!tw) return 'both';
    if (tw.direction != null) {
      const d = tw.direction;
      if (d === 'topToBottom') return 'clockwise';
      if (d === 'bottomToTop') return 'counter_clockwise';
      return d || 'both';
    }
    if (tw.directionModeId) {
      const m = state.directionModes.find(d => d.id === tw.directionModeId);
      if (m && m.direction) return m.direction;
    }
    return 'both';
  }
  function normalizeRwDirectionValue(dir) {
    if (dir === 'clockwise' || dir === 'cw') return 'clockwise';
    if (dir === 'counter_clockwise' || dir === 'ccw') return 'counter_clockwise';
    return 'both';
  }
  function normalizeAllowedRunwayDirections(raw) {
    const out = [];
    const src = Array.isArray(raw) ? raw : [];
    src.forEach(function(v) {
      const d = normalizeRwDirectionValue(v);
      if (d === 'clockwise' && out.indexOf('clockwise') < 0) out.push('clockwise');
      if (d === 'counter_clockwise' && out.indexOf('counter_clockwise') < 0) out.push('counter_clockwise');
    });
    return out;
  }
  function getTaxiwayAllowedRunwayDirections(tw) {
    if (!tw || tw.pathType !== 'runway_exit') return (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT.slice() : ['clockwise', 'counter_clockwise'];
    const arr = normalizeAllowedRunwayDirections(tw.allowedRwDirections);
    if (!arr.length) return (RW_EXIT_ALLOWED_DEFAULT && RW_EXIT_ALLOWED_DEFAULT.length) ? RW_EXIT_ALLOWED_DEFAULT.slice() : ['clockwise', 'counter_clockwise'];
    return arr;
  }
  function isRunwayExitDirectionAllowed(tw, runwayDir) {
    const d = normalizeRwDirectionValue(runwayDir);
    if (d !== 'clockwise' && d !== 'counter_clockwise') return true;
    const allow = getTaxiwayAllowedRunwayDirections(tw);
    return allow.indexOf(d) >= 0;
  }
  function getRunwayExitAllowedDirectionsFromPanel() {
    const out = [];
    const container = document.getElementById('runwayExitAllowedDirection');
    if (!container) return out;
    container.querySelectorAll('.runway-exit-dir-check').forEach(function(ch) {
      if (!ch.checked) return;
      const value = String(ch.getAttribute('data-item-id') || '').trim();
      if (value === 'clockwise' || value === 'counter_clockwise') out.push(value);
    });
    return out;
  }

  const _rwy = _tiers.runway || {};
  const _sepUi = (_rwy.separationUi && typeof _rwy.separationUi === 'object') ? _rwy.separationUi : {};
  const RSEP_ARRDEP_BOOST_SEC = Math.max(0, Number(_sepUi.arrDepDefaultBoostSec) || 50);
  const RSEP_COLOR_THRESHOLDS = (function() {
