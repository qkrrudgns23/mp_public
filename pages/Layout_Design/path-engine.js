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
  function applyLayoutObject(obj) {
    if (!obj || typeof obj !== 'object') return;
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
        delete f.eldtMin;
        delete f.eibtMin;
        delete f.eobtMin;
        delete f.etotMin;
        delete f.eldtMin_orig;
        delete f.eibtMin_orig;
        delete f.eobtMin_orig;
        delete f.etotMin_orig;
        if (!f.airlineCode) f.airlineCode = DEFAULT_AIRLINE_CODES[Math.floor(Math.random() * DEFAULT_AIRLINE_CODES.length)];
        if (!f.flightNumber) f.flightNumber = f.airlineCode + String(Math.floor(1000 + Math.random() * 9000));
        if (!String(f.reg || '').trim()) f.reg = randomRegNumber();
        {
          const idRaw = String(f.intDom || '').trim();
          f.intDom = (idRaw.toLowerCase() === 'dom') ? 'Dom' : 'Int';
        }
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
    else draw();
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && state.layers && state.layers.junction) {
      try {
        if (typeof applyPathGraphSyncNow === 'function') applyPathGraphSyncNow();
        if (typeof draw === 'function') draw();
      } catch (ePg) {
        console.warn('applyLayoutObject: path graph sync', ePg);
      }
    }
    if (typeof renderFlightList === 'function') renderFlightList();
  }
  /** E-series minutes and ARR_ROT_SEC from ``airside_sim`` schedule row only (seconds → minutes). */
  function applyAirsideScheduleRowToFlight(f, srec) {
    if (!f) return;
    if (!srec || typeof srec !== 'object') {
      delete f.eldtMin;
      delete f.eibtMin;
      delete f.eobtMin;
      delete f.etotMin;
      delete f.eldtMin_orig;
      delete f.eibtMin_orig;
      delete f.eobtMin_orig;
      delete f.etotMin_orig;
      f.arrRotSec = null;
      return;
    }
    function secOpt(key) {
      if (srec[key] == null || srec[key] === '') return NaN;
      const n = Number(srec[key]);
      return isFinite(n) ? n : NaN;
    }
    const eldtS = secOpt('ELDT');
    const eibtS = secOpt('EIBT');
    const eobtS = secOpt('EOBT');
    const etotS = secOpt('ETOT');
    if (isFinite(eldtS)) f.eldtMin = eldtS / 60;
    else delete f.eldtMin;
    if (isFinite(eibtS)) f.eibtMin = eibtS / 60;
    else delete f.eibtMin;
    if (isFinite(eobtS)) f.eobtMin = eobtS / 60;
    else delete f.eobtMin;
    if (isFinite(etotS)) f.etotMin = etotS / 60;
    else delete f.etotMin;
    const rotS = secOpt('ARR_ROT_SEC');
    if (isFinite(rotS)) f.arrRotSec = rotS;
    else f.arrRotSec = null;
  }
  function applyAirsideSimulationResultPayload(payload) {
    if (!payload || typeof payload !== 'object') return;
    state.simPlaybackEndCapSec = null;
    if (payload.simulation_truncated_deadlock === true || payload.simulation_truncated_stot_horizon === true) {
      const rawCap = payload.simulation_playback_end_abs_sec;
      const c = Number(rawCap);
      if (isFinite(c)) state.simPlaybackEndCapSec = c;
    }
    const flightsDetail = Array.isArray(payload.flights_detail) ? payload.flights_detail : null;
    if (flightsDetail) {
      const byId = {};
      flightsDetail.forEach(function(row) {
        if (!row || row.flight_id == null) return;
        const fid = String(row.flight_id);
        const fin = row.edge_list_finished;
        const planned = row.edge_list;
        if (Array.isArray(fin) && fin.length) {
          byId[fid] = fin.slice();
        } else if (Array.isArray(planned) && planned.length) {
          byId[fid] = planned.slice();
        } else {
          byId[fid] = [];
        }
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
    let mergedTimelines = 0;
    (state.flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const srec = schedById[String(f.id)] || null;
      if (hasPositions) {
        const rawPts = positions[f.id];
        if (rawPts != null) {
          const pts = Array.isArray(rawPts) ? rawPts : [];
          if (pts.length >= 2) {
            const tl = pts.map(function(p) {
              const x = p.x != null && p.x !== '' ? Number(p.x) : Number(p.col);
              const y = p.y != null && p.y !== '' ? Number(p.y) : Number(p.row);
              const mf = p.motionForward !== false && p.motion_forward !== false;
              const dg = p.deadlockGhost === true || p.deadlock_ghost === true;
              return { t: Number(p.t), x: x, y: y, motionForward: mf, deadlockGhost: dg };
            }).filter(function(k) {
              return isFinite(k.t) && isFinite(k.x) && isFinite(k.y);
            }).sort(function(a, b) { return a.t - b.t; });
            if (tl.length >= 2) {
              mergedTimelines++;
              f.timeline = tl;
            }
          }
        }
      }
      if (srec && f.timeline && f.timeline.length >= 2) {
        const eldtS = srec.ELDT != null ? Number(srec.ELDT) : NaN;
        const eibtS = srec.EIBT != null ? Number(srec.EIBT) : NaN;
        const eobtS = srec.EOBT != null ? Number(srec.EOBT) : NaN;
        const etotS = srec.ETOT != null ? Number(srec.ETOT) : NaN;
        f.timeline_meta = {
          playbackSource: 'des_result',
          eldtSec: isFinite(eldtS) ? eldtS : undefined,
          eibtSec: isFinite(eibtS) ? eibtS : undefined,
          eobtSec: isFinite(eobtS) ? eobtS : undefined,
          etotSec: isFinite(etotS) ? etotS : undefined,
        };
      } else {
        delete f.timeline_meta;
      }
      applyAirsideScheduleRowToFlight(f, srec);
    });
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
    if (typeof update3DSceneWhenVisible === 'function') update3DSceneWhenVisible();
    const playDockBtn = document.getElementById('btnShowPlayDock');
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
    const prefix = pathType === 'runway' ? 'RW' : (pathType === 'runway_exit' ? 'RTX' : (pathType === 'apron_taxiway' ? 'ATX' : (pathType === 'general_queue_taxiway' ? 'QTX' : 'TX')));
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
  function getDefaultTempStandName(currentId) {
    const stands = (state.tempStands || []).filter(function(st) { return st && st.id !== currentId; });
    const used = new Set(stands.map(function(st) { return (st.name && String(st.name).trim()) || ''; }).filter(Boolean));
    return uniqueNameAgainstSet('T' + zeroPadNumber(stands.length + 1, 3), used);
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
