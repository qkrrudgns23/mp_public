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
      let ow = Number(m.outerWidthM);
      let iw = Number(m.innerWidthM);
      if (!isFinite(ow) || ow < 0) ow = LAYOUT_ISLAND_OUTER_WIDTH_DEFAULT_M;
      else ow = Math.min(500, ow);
      if (!isFinite(iw) || iw < 0) iw = LAYOUT_ISLAND_INNER_WIDTH_DEFAULT_M;
      else iw = Math.min(200, iw);
      return { kind: 'island', id: m.id || id(), points: points, outerWidthM: ow, innerWidthM: iw, pavement: islandMarkerPavementResolved(m) };
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
    if (objectKind === 'tempStand') {
      const arr = state.tempStands || [];
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i];
        if (!o || !isOther(o.id)) continue;
        const disp = (o.name && String(o.name).trim()) || '';
        if (normalizeLayoutNameKey(disp) === key) return { kind: 'tempStand', existing: disp || o.id };
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
      tempStands: JSON.parse(JSON.stringify(state.tempStands || [])),
      holdingPoints: JSON.parse(JSON.stringify(state.holdingPoints || [])),
      taxiways: JSON.parse(JSON.stringify(state.taxiways || [])),
      apronLinks: JSON.parse(JSON.stringify(state.apronLinks || [])),
      layoutImageOverlay: JSON.parse(JSON.stringify(state.layoutImageOverlay || null)),
      layoutEdgeNames: JSON.parse(JSON.stringify(state.layoutEdgeNames || {})),
      directionModes: JSON.parse(JSON.stringify(state.directionModes || [])),
      flights: cloneFlightsWithoutPathPolylineCache(state.flights),
      layoutMarkers: JSON.parse(JSON.stringify(state.layoutMarkers || []))
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
    state.tempStands = snap.tempStands || [];
    state.holdingPoints = snap.holdingPoints || [];
    state.taxiways = snap.taxiways;
    state.apronLinks = snap.apronLinks;
    state.apronLinkJunctionOverlayDirtyIds = null;
    state.layoutImageOverlay = normalizeLayoutImageOverlay(snap.layoutImageOverlay);
    syncLayoutImageBitmap();
    state.layoutEdgeNames = snap.layoutEdgeNames || {};
    state.directionModes = snap.directionModes;
    state.flights = snap.flights;
    state.layoutMarkers = normalizeLayoutMarkerAreaZOrder(Array.isArray(snap.layoutMarkers) ? snap.layoutMarkers : []);
    state.pathArcDrag = null;
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
    const arr = _sepUi.inputColorThresholdsSec;
    if (Array.isArray(arr) && arr.length) {
      return arr.map(x => Number(x)).filter(x => isFinite(x) && x > 0).sort((a, b) => a - b);
    }
    return [90, 120, 150];
  })();
  const RSEP_LEGEND_LAB = (_sepUi.legendLabels && typeof _sepUi.legendLabels === 'object') ? _sepUi.legendLabels : {};
  function rsepLegendFmt(tpl, a0, a1) {
    let s = String(tpl || '');
    if (a1 != null && s.indexOf('{1}') >= 0) return s.replace('{0}', String(a0)).replace('{1}', String(a1));
    return s.replace('{0}', String(a0));
  }
  const RSEP_COLOR_STYLES = [
    { bg: '#0d2018', color: '#68d391', border: '#68d39155' },
    { bg: '#0d1a28', color: '#63b3ed', border: '#63b3ed55' },
    { bg: '#1e1e08', color: '#f6e05e', border: '#f6e05e55' },
    { bg: '#280d0d', color: '#fc8181', border: '#fc818155' },
  ];
  const _stds = _rwy.standards || {};
  const RSEP_STD_CATS = {
    'ICAO': (_stds.ICAO && _stds.ICAO.categories) ? _stds.ICAO.categories : ['J','H','M','L'],
    'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categories) ? _stds['RECAT-EU'].categories : ['A','B','C','D','E','F'],
  };
  const RSEP_SEQ_TYPES = Object.assign({ 'ARR→ARR': 'matrix', 'DEP→DEP': 'matrix', 'ARR→DEP': 'lead-1d', 'DEP→ARR': 'trail-1d' }, _sepUi.seqTypes || {});
  const RSEP_MODE_SEQS = (function() {
    const def = { ARR: ['ARR→ARR'], DEP: ['DEP→DEP'], MIX: ['ARR→ARR','DEP→DEP','ARR→DEP','DEP→ARR'] };
    const ms = _sepUi.modeSequences || {};
    const out = {};
    ['ARR','DEP','MIX'].forEach(k => {
      const a = ms[k];
      out[k] = (Array.isArray(a) && a.length) ? a.slice() : def[k].slice();
    });
    return out;
  })();
  const RSEP_DEFAULTS = {};
  ['ICAO','RECAT-EU'].forEach(k => {
    const s = _stds[k];
    if (!s) return;
    RSEP_DEFAULTS[k] = { ...(s.separationDefaults || {}), ROT: s.ROT || {} };
  });
  if (!RSEP_DEFAULTS['ICAO'] || !Object.keys(RSEP_DEFAULTS['ICAO']).length) {
    RSEP_DEFAULTS['ICAO'] = { 'ARR→ARR': { J:{J:90,H:120,M:180,L:240}, H:{J:90,H:90,M:120,L:180}, M:{J:90,H:90,M:90,L:180}, L:{J:90,H:90,M:90,L:90} }, 'DEP→DEP': { J:{J:90,H:120,M:180,L:180}, H:{J:90,H:90,M:120,L:120}, M:{J:90,H:90,M:90,L:90}, L:{J:90,H:90,M:90,L:90} }, 'ARR→DEP': {J:90,H:80,M:65,L:50}, 'DEP→ARR': {J:60,H:60,M:70,L:90}, ROT: {J:70,H:65,M:55,L:40} };
  }
  if (!RSEP_DEFAULTS['RECAT-EU'] || !Object.keys(RSEP_DEFAULTS['RECAT-EU']).length) {
    RSEP_DEFAULTS['RECAT-EU'] = { 'ARR→ARR': { A:{A:80,B:100,C:120,D:140,E:160,F:180}, B:{A:80,B:80,C:100,D:120,E:120,F:140}, C:{A:80,B:80,C:80,D:100,E:100,F:120}, D:{A:80,B:80,C:80,D:80,E:80,F:100}, E:{A:80,B:80,C:80,D:80,E:80,F:100}, F:{A:80,B:80,C:80,D:80,E:80,F:80} }, 'DEP→DEP': { A:{A:80,B:100,C:120,D:120,E:120,F:140}, B:{A:80,B:80,C:100,D:100,E:100,F:120}, C:{A:80,B:80,C:80,D:80,E:80,F:100}, D:{A:80,B:80,C:80,D:80,E:80,F:80}, E:{A:80,B:80,C:80,D:80,E:80,F:80}, F:{A:80,B:80,C:80,D:80,E:80,F:80} }, 'ARR→DEP': {A:80,B:70,C:60,D:55,E:50,F:45}, 'DEP→ARR': {A:55,B:55,C:60,D:65,E:70,F:80}, ROT: {A:65,B:60,C:55,D:50,E:45,F:40} };
  }
  const RSEP_STANDARDS = { 'ICAO': { ROT: RSEP_DEFAULTS['ICAO'] && RSEP_DEFAULTS['ICAO'].ROT ? RSEP_DEFAULTS['ICAO'].ROT : {} }, 'RECAT-EU': { ROT: RSEP_DEFAULTS['RECAT-EU'] && RSEP_DEFAULTS['RECAT-EU'].ROT ? RSEP_DEFAULTS['RECAT-EU'].ROT : {} } };
  const RSEP_CAT_LABELS = {
    'ICAO': (_stds.ICAO && _stds.ICAO.categoryLabels) ? _stds.ICAO.categoryLabels : { J:'Super', H:'Heavy', M:'Medium', L:'Light' },
    'RECAT-EU': (_stds['RECAT-EU'] && _stds['RECAT-EU'].categoryLabels) ? _stds['RECAT-EU'].categoryLabels : { A:'Super-Heavy', B:'Upper-Heavy', C:'Lower-Heavy', D:'Medium', E:'Light', F:'Very-Light' },
  };
  const RSEP_SEQ_META = _rwy.seqMeta || {
    'ARR→ARR': { driver: 'Wake of leading arrival aircraft', refPoint: 'Touchdown / final approach point of the leading arrival', input: 'Lead (arrival) × Trail (arrival) matrix input' },
