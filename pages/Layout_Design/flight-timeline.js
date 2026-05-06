    const hi = Number(state.simDurationSec);
    evs.forEach(function(ev) {
      const t = Number(ev.t_abs);
      if (!isFinite(t)) return;
      if (t < lo - 2 || t > hi + 2) return;
      const pct = 100 * (t - lo) / span;
      const dot = document.createElement('span');
      dot.className = 'sim-slider-deadlock-dot';
      dot.style.left = Math.max(0, Math.min(100, pct)) + '%';
      dot.setAttribute('title', 'Deadlock @ ' + formatTotalSecondsToHHMMSS(t) + ' — click to jump');
      dot.setAttribute('role', 'button');
      dot.setAttribute('tabindex', '0');
      dot.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        seekSimToDeadlockMarkerEvent(ev);
      });
      dot.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          e.stopPropagation();
          seekSimToDeadlockMarkerEvent(ev);
        }
      });
      host.appendChild(dot);
    });
  }
  function applyAirsideSimulationResultPayload(payload) {
    if (!payload || typeof payload !== 'object') return;
    const truncCap = (payload.simulation_truncated_deadlock === true || payload.simulation_truncated_stot_horizon === true)
      ? (function() {
        const rawCap = payload.simulation_playback_end_abs_sec;
        const c = Number(rawCap);
        return isFinite(c) ? c : null;
      })()
      : null;
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
    state.simPlaybackEndCapSec = truncCap;
    const schedById = {};
    scheduleList.forEach(function(s) {
      if (s && s.flight_id != null) schedById[String(s.flight_id)] = s;
    });
    let mergedTimelines = 0;
    (state.flights || []).forEach(function(f) {
      if (!f || f.id == null) return;
      const srec = schedById[String(f.id)] || null;
      const track = hasPositions ? positions[String(f.id)] : null;
      const hasTrack = compactPlaybackTrackLength(track) >= 2;
      f.timeline = null;
      if (hasTrack) {
        mergedTimelines++;
        if (f.arrDep !== 'Dep') f.arrRetFailed = false;
      }
      if (srec && hasTrack) {
        const eldtS = srec.ELDT != null ? Number(srec.ELDT) : NaN;
        const eibtS = srec.EIBT != null ? Number(srec.EIBT) : NaN;
        const eobtS = srec.EOBT != null ? Number(srec.EOBT) : NaN;
        const etotS = srec.ETOT != null ? Number(srec.ETOT) : NaN;
        const prevMeta = f.timeline_meta || {};
        const builtDep = (typeof buildDepartureSurfaceTimelineSegments === 'function' && f.arrDep === 'Dep'
          && isFinite(eobtS) && isFinite(etotS))
          ? buildDepartureSurfaceTimelineSegments(f, eobtS, etotS)
          : null;
        const builtDepMeta = (builtDep && builtDep.meta) ? builtDep.meta : null;
        f.timeline_meta = Object.assign(
          {},
          prevMeta,
          builtDepMeta || {},
          {
            playbackSource: 'des_result',
            eldtSec: isFinite(eldtS) ? eldtS : undefined,
            eibtSec: isFinite(eibtS) ? eibtS : undefined,
            eobtSec: isFinite(eobtS) ? eobtS : undefined,
            etotSec: isFinite(etotS) ? etotS : undefined,
            eibtSecList: Array.isArray(srec.EIBT_LIST) ? srec.EIBT_LIST.slice() : undefined,
            eobtSecList: Array.isArray(srec.EOBT_LIST) ? srec.EOBT_LIST.slice() : undefined,
            ePushFinishedSecList: Array.isArray(srec.E_PUSH_FINISHED_LIST) ? srec.E_PUSH_FINISHED_LIST.slice() : undefined,
          }
        );
      } else {
        delete f.timeline_meta;
      }
      applyAirsideScheduleRowToFlight(f, srec);
    });
    state.hasSimulationResult = mergedTimelines > 0;
    state.simPlaybackPositionsByFlightId = hasPositions ? positions : null;
    state.simPlaybackScheduleSnapshot = scheduleList.length ? scheduleList.slice() : null;
    state.simPlaybackTimelinesEvictedForMemory = false;
    state.simDeadlockGhostPlayback = deriveDeadlockGhostPlaybackFromPayload(payload, state.flights);
    if (hasPositions && positions && typeof positions === 'object') {
      if (!state.deadlockFlightIdsFromLastSim) state.deadlockFlightIdsFromLastSim = Object.create(null);
      Object.keys(positions).forEach(function(pid) {
        const tr = positions[pid];
        if (allocFlightTrackHasDeadlock(tr)) state.deadlockFlightIdsFromLastSim[String(pid)] = true;
      });
    }
    if (state.hasSimulationResult) {
      if (typeof markGlobalUpdateFresh === 'function') markGlobalUpdateFresh();
      if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
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
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
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
    if (dir == null) return 'both';
    const s0 = String(dir).trim();
    if (!s0) return 'both';
    const s = s0.toLowerCase().replace(/[\s-]+/g, '_');
    if (s === 'clockwise' || s === 'cw') return 'clockwise';
    if (s === 'counter_clockwise' || s === 'ccw' || s === 'counterclockwise') return 'counter_clockwise';
    if (s === 'top_tobottom' || s === 'toptobottom' || s === 'ttb') return 'clockwise';
    if (s === 'bottom_totop' || s === 'bottomtotop' || s === 'btt') return 'counter_clockwise';
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
  /**
   * Arrival RET sampling (F2): runways in the property panel are always stored as CW or CCW;
   * legacy data may still have direction "both". The panel coerces "both" to the CW option for display,
   * so we use CW as the operational match for "Available RW direction" vs the runway.
   */
  function getRunwayOperationalDirForArrivalRetFilter2(rw) {
    if (!rw || rw.pathType !== 'runway') return 'clockwise';
    const raw = getTaxiwayDirection(rw);
    const d = normalizeRwDirectionValue(raw);
    if (d === 'clockwise' || d === 'counter_clockwise') return d;
    return 'clockwise';
  }
  /**
   * F2: same semantics as checkboxes, but an explicit `allowedRwDirections: []` (both unchecked) means
   * "no use" — do not re-expand to the default both-way list (pathfinding may still do that for legacy).
   */
  function isRunwayExitDirAllowedForArrivalFilter2(exitTw, runwayDir) {
    const d = normalizeRwDirectionValue(runwayDir);
    if (d !== 'clockwise' && d !== 'counter_clockwise') return true;
    if (!exitTw || exitTw.pathType !== 'runway_exit') return false;
    if (Object.prototype.hasOwnProperty.call(exitTw, 'allowedRwDirections')) {
      const arr = normalizeAllowedRunwayDirections(exitTw.allowedRwDirections);
      if (arr.length === 0) return false;
      return arr.indexOf(d) >= 0;
    }
    return isRunwayExitDirectionAllowed(exitTw, d);
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
    'DEP→DEP': { driver: 'Wake of leading departure aircraft', refPoint: 'Take-off / runway entry point of the leading departure', input: 'Lead (departure) × Trail (departure) matrix input' },
    'ARR→DEP': { driver: 'Leading aircraft ROT (runway occupancy time)', refPoint: 'Trailing aircraft: time from lineup to gear-off (lineup–gear-off)', input: 'Lead arrival category — 1D separation inputs' },
    'DEP→ARR': { driver: 'Wake / ROT of leading departure', refPoint: 'Runway vacation / ROT end of the leading departure', input: 'Trail (arrival category) 1‑D input' },
  };
  function rsepGetCatLabel(stdKey, cat) {
    const t = RSEP_CAT_LABELS[stdKey];
    if (!t) return '';
    return t[cat] || '';
  }
  function rsepGetSeqMeta(seq) {
    return RSEP_SEQ_META[seq] || null;
  }
  function _rsepStringValue(value) {
    return value != null ? String(value) : '';
  }
  function _rsepMakeCategoryValues(cats, src, asMatrix) {
    const out = {};
    cats.forEach(leadCat => {
      if (!asMatrix) {
        out[leadCat] = _rsepStringValue(src && src[leadCat]);
        return;
      }
      out[leadCat] = {};
      cats.forEach(trailCat => {
        out[leadCat][trailCat] = _rsepStringValue(src && src[leadCat] && src[leadCat][trailCat]);
      });
    });
    return out;
  }
  function rsepMakeMatrix(cats, src) {
    return _rsepMakeCategoryValues(cats, src, true);
  }
  function rsepMake1D(cats, src) {
    return _rsepMakeCategoryValues(cats, src, false);
  }
  function rsepMakeSeqData(stdKey) {
    const cats = RSEP_STD_CATS[stdKey] || [];
    const def = RSEP_DEFAULTS[stdKey] || {};
    const arrDep = rsepMake1D(cats, def['ARR→DEP']);
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      const s = arrDep[c];
      if (s === '' || s == null) return;
      const n = Number(s);
      if (isFinite(n)) arrDep[c] = String(Math.round(n + boost));
    });
    return {
      'ARR→ARR': rsepMakeMatrix(cats, def['ARR→ARR']),
      'DEP→DEP': rsepMakeMatrix(cats, def['DEP→DEP']),
      'ARR→DEP': arrDep,
      'DEP→ARR': rsepMake1D(cats, def['DEP→ARR']),
    };
  }

  function rsepColorForValue(val) {
    const n = Number(val);
    if (!isFinite(n) || val === '' || val == null) {
      return { bg: '#1a1a1a', color: '#e5e7eb', border: '#444444' };
    }
    const th = RSEP_COLOR_THRESHOLDS;
    for (let i = 0; i < th.length; i++) {
      if (n < th[i]) return RSEP_COLOR_STYLES[i] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
    }
    return RSEP_COLOR_STYLES[th.length] || RSEP_COLOR_STYLES[RSEP_COLOR_STYLES.length - 1];
  }
  function rsepLegendHtml(filled, total) {
    const th = RSEP_COLOR_THRESHOLDS;
    const countColor = filled === total ? '#68d391' : '#9ca3af';
    let html = '<div style="display:flex;align-items:center;gap:12px;margin-top:4px;margin-bottom:4px;font-size:10px;color:#9ca3af;">';
    const lab = RSEP_LEGEND_LAB;
    if (th.length) {
      const st0 = rsepColorForValue(Math.max(0, th[0] - 1));
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st0.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st0.color + ';">' + escapeHtml(rsepLegendFmt(lab.ltFirst || '<{0}s', th[0])) + '</span></span>';
      for (let i = 1; i < th.length; i++) {
        const lo = th[i - 1], hi = th[i];
        const mid = lo + (hi - lo) / 2;
        const st = rsepColorForValue(mid);
        const text = rsepLegendFmt(lab.rangeMid || '{0}–{1}s', lo, hi - 1);
        html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + st.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + st.color + ';">' + escapeHtml(text) + '</span></span>';
      }
      const lastT = th[th.length - 1];
      const stL = rsepColorForValue(lastT + 1000);
      html += '<span><span style="display:inline-block;width:10px;height:10px;background:' + stL.bg + ';border-radius:2px;margin-right:4px;"></span><span style="color:' + stL.color + ';">' + escapeHtml(rsepLegendFmt(lab.gteLast || '≥{0}s', lastT)) + '</span></span>';
    }
    html += '<span style="margin-left:4px;color:' + countColor + ';">' + filled + '/' + total + '</span>';
    html += '</div>';
    return html;
  }
  function rsepMakeConfig(stdKey) {
    const std = RSEP_STANDARDS[stdKey] || RSEP_STANDARDS['ICAO'];
    const cats = RSEP_STD_CATS[stdKey];
    const rot = std.ROT || {};
    const rotCopy = {};
    const boost = RSEP_ARRDEP_BOOST_SEC;
    cats.forEach(function(c) {
      if (rot[c] == null || rot[c] === '') rotCopy[c] = '';
      else {
        const n = Number(rot[c]);


        rotCopy[c] = isFinite(n) ? String(Math.round(n + boost)) : String(rot[c]);
      }
    });
    return {
      standard: stdKey,
      mode: 'MIX',
      activeSeq: 'ARR→ARR',
      seqData: rsepMakeSeqData(stdKey),
      rot: rotCopy,
    };
  }
  function rsepGetConfigForRunway(rw) {
    if (!rw) return null;
    if (!rw.rwySepConfig) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
    }
    const cfg = rw.rwySepConfig;
    if (!RSEP_STD_CATS[cfg.standard]) {
      rw.rwySepConfig = rsepMakeConfig('ICAO');
      return rw.rwySepConfig;
    }
    return cfg;
  }
  let dpr = window.devicePixelRatio || 1;
  let ctx = (canvas && typeof canvas.getContext === 'function') ? canvas.getContext('2d') : null;
  let layoutDrawCanvas = canvas;
  function layoutUseForegroundOverlay() {
    return !!(overlayCtx && overlayCanvas && layoutDrawCanvas === canvas);
  }

  function screenToWorld(sx, sy) {
    return [(sx - state.panX) / state.scale, (sy - state.panY) / state.scale];
  }
  function worldToScreenCanvas(wx, wy) {
    return [wx * state.scale + state.panX, wy * state.scale + state.panY];
  }
  function cellToPixel(col, row) { return [col * CELL_SIZE, row * CELL_SIZE]; }
  function getTaxiwayAvgMoveVelocityForPath(path) {
    if (path && typeof path.avgMoveVelocity === 'number' && isFinite(path.avgMoveVelocity) && path.avgMoveVelocity > 0)
      return Math.max(1, Math.min(50, path.avgMoveVelocity));
    const el = document.getElementById('taxiwayAvgMoveVelocity');
    const v = el ? Number(el.value) : 10;
    return (typeof v === 'number' && isFinite(v) && v > 0) ? Math.max(1, Math.min(50, v)) : 10;
  }
  function roundToStep(value, step) {
    const n = Number(value);
    const s = Number(step);
    if (!isFinite(n)) return 0;
    if (!isFinite(s) || s <= 0) return n;
    return Math.round(n / s) * s;
  }
  function clampToGridBounds(col, row) {
    const c = Math.max(0, Math.min(GRID_COLS, Number(col) || 0));
    const r = Math.max(0, Math.min(GRID_ROWS, Number(row) || 0));
    return [c, r];
  }
  function pixelToCell(x, y) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const snappedCol = roundToStep(x / cs, GRID_SNAP_STEP_CELL);
    const snappedRow = roundToStep(y / cs, GRID_SNAP_STEP_CELL);
    return clampToGridBounds(snappedCol, snappedRow);
  }
  function worldPointToCellPoint(wx, wy, snapToGrid) {
    const cs = (typeof CELL_SIZE === 'number' && CELL_SIZE > 0) ? CELL_SIZE : 20;
    const step = snapToGrid ? GRID_SNAP_STEP_CELL : FREE_DRAW_STEP_CELL;
    const col = roundToStep(wx / cs, step);
    const row = roundToStep(wy / cs, step);
    const clamped = clampToGridBounds(col, row);
    return { col: clamped[0], row: clamped[1] };
  }
  function worldPointToPixel(wx, wy, snapToGrid) {
    const pt = worldPointToCellPoint(wx, wy, snapToGrid);
    return cellToPixel(pt.col, pt.row);
  }
  const ICAO_STAND_SIZE_M = (function() {
    const m = _layoutTier.standSizesMByIcaoCategory;
    if (m && typeof m === 'object') {
      const o = {};
      Object.keys(m).forEach(k => { o[k] = Number(m[k]); });
      return o;
    }
    return { A: 20, B: 30, C: 40, D: 50, E: 60, F: 80 };
  })();
  function getStandSizeMeters(cat) { return ICAO_STAND_SIZE_M[cat] || 40; }
  const STAND_CONFIG_ROW_BY_CODE = (function() {
    const raw = _layoutTier.standConfig;
    const out = {};
    if (!raw || typeof raw !== 'object') return out;
    Object.keys(raw).forEach(function(k) {
      if (k === 'description') return;
      const row = raw[k];
      if (!row || typeof row !== 'object') return;
      const ws = Number(row.wingspan), g = Number(row.gap), rd = Number(row.road);
      const ln = Number(row.length), nc = Number(row.nose_clear), pb = Number(row.pushback);
      const dp = Number(row.depth);
      if (![ws, g, rd, ln, nc, pb, dp].every(isFinite)) return;
      const rowOut = { wingspan: ws, gap: g, road: rd, length: ln, nose_clear: nc, pushback: pb, depth: dp };
      const og = Number(row.outer_gear), nsc = Number(row.nose_side_clear), nw = Number(row.nose_width);
      if (isFinite(og)) rowOut.outer_gear = og;
      if (isFinite(nsc)) rowOut.nose_side_clear = nsc;
      if (isFinite(nw)) rowOut.nose_width = nw;
      out[String(k).toUpperCase().slice(0, 1)] = rowOut;
    });
    return out;
  })();
  function standConfigRowForIcaoCat(cat) {
    const s = String(cat == null ? 'C' : cat).toUpperCase();
    let c = 'C';
    for (let i = 0; i < s.length; i++) {
      const ch = s.charAt(i);
      if (ch >= 'A' && ch <= 'F') {
        c = ch;
        break;
      }
    }
    return STAND_CONFIG_ROW_BY_CODE[c] || null;
  }
  function getStandWidthMeters(cat) {
    const r = standConfigRowForIcaoCat(cat);
    if (r) return r.wingspan + r.gap * 2;
    return getStandSizeMeters(cat);
  }
  function getStandDepthMeters(cat) {
    const r = standConfigRowForIcaoCat(cat);
    if (r) return r.depth;
    return getStandSizeMeters(cat);
  }
  function getStandSpacingMeters(catA, catB) {
    const ra = standConfigRowForIcaoCat(catA);
    const rb = standConfigRowForIcaoCat(catB);
    if (ra && rb) return Math.max(ra.road, rb.road);
    return 0;
  }
  function standFootprintLocalToWorld(cx, cy, angleRad, lx, ly) {
    const cos = Math.cos(angleRad), sin = Math.sin(angleRad);
    return [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos];
  }
  function standStopbarCenterShiftLocalX(depM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || depM <= 0) return 0;
    const nc = Number(r.nose_clear);
    if (!isFinite(nc) || nc <= 0) return 0;
    const halfD = depM / 2;
    const xStopOld = -halfD + nc;
    if (!(xStopOld > -halfD && xStopOld < halfD)) return 0;
    return -xStopOld;
  }
  /** Nose (−X) width = nose_width; stop bar at nose_clear from nose edge; 45° flare to full stand width (±halfW), then rectangle to tail (+halfD). */
  function buildStandSafetyPolygonPath(ctx, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return false;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return false;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return false;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return false;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return false;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return false;
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    ctx.beginPath();
    ctx.moveTo(xNose + shiftX, -noseHalf);
    ctx.lineTo(xNose + shiftX, noseHalf);
    ctx.lineTo(xStop + shiftX, noseHalf);
    ctx.lineTo(Math.min(xBendEnd, halfD) + shiftX, halfW);
    if (xBendEnd < halfD - eps) {
      ctx.lineTo(halfD + shiftX, halfW);
      ctx.lineTo(halfD + shiftX, -halfW);
      ctx.lineTo(xBendEnd + shiftX, -halfW);
    } else {
      ctx.lineTo(halfD + shiftX, -halfW);
    }
    ctx.lineTo(xStop + shiftX, -noseHalf);
    ctx.closePath();
    return true;
  }
  /** Stand-local +X = tail/apron-open, −X = nose/terminal-ward. Red dashed: stop bar (nose_clear), pushback (pushback), lateral gap bounds (wingspan ± vs stand width). Clipped to safety footprint when nose geometry applies. */
  function drawStandApronMarkingsInLocalAxes(ctx, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return;
    const halfD = depM / 2, halfW = widM / 2;
    const eps = 0.12;
    ctx.save();
    if (buildStandSafetyPolygonPath(ctx, depM, widM, category)) {
      ctx.clip();
    }
    ctx.strokeStyle = layerMonoLinesOn() ? c2dLayerMonoLineStrokeCss() : c2dStandSafetyStroke();
    ctx.lineWidth = Math.max(0.35, 0.42 / Math.max(state.scale, 0.1));
    ctx.setLineDash([2, 2.5]);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    const nc = Number(r.nose_clear), pb = Number(r.pushback);
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    if (isFinite(nc) && isFinite(pb)) {
      const xStop = 0;
      const xPush = (halfD - pb) + shiftX;
      const xMin = (-halfD) + shiftX;
      const xMax = (halfD) + shiftX;
      if (xStop > -halfD + eps && xStop < halfD - eps) {
        ctx.beginPath();
        ctx.moveTo(xStop, -halfW);
        ctx.lineTo(xStop, halfW);
        ctx.stroke();
      }
      if (xPush < xMax - eps && xPush > xMin + eps) {
        ctx.beginPath();
        ctx.moveTo(xPush, -halfW);
        ctx.lineTo(xPush, halfW);
        ctx.stroke();
      }
    }
    const g = Number(r.gap), ws = Number(r.wingspan);
    if (isFinite(g) && g > eps && isFinite(ws) && ws > 0) {
      const yLim = halfW - g;
      if (yLim > eps && yLim < halfW - eps) {
        ctx.beginPath();
        ctx.moveTo(-halfD + shiftX, yLim);
        ctx.lineTo(halfD + shiftX, yLim);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-halfD + shiftX, -yLim);
        ctx.lineTo(halfD + shiftX, -yLim);
        ctx.stroke();
      }
    }
    ctx.restore();
  }
  function fillStandSafetyFootprintInLocalAxes(ctx, depM, widM, category) {
    if (buildStandSafetyPolygonPath(ctx, depM, widM, category)) {
      ctx.fill();
      return;
    }
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    ctx.beginPath();
    ctx.rect((-depM / 2) + shiftX, -widM / 2, depM, widM);
    ctx.fill();
  }
  function drawStandSafetyContourInLocalAxes(ctx, depM, widM, category, selected) {
    if (!buildStandSafetyPolygonPath(ctx, depM, widM, category)) return;
    ctx.save();
    ctx.strokeStyle = (!selected && layerMonoLinesOn()) ? c2dLayerMonoLineStrokeCss() : c2dStandSafetyStroke();
    const baseLw = Math.max(0.55, 0.65 / Math.max(state.scale, 0.1));
    ctx.lineWidth = selected ? baseLw * 1.35 : baseLw;
    ctx.setLineDash([]);
    ctx.lineJoin = 'miter';
    ctx.lineCap = 'butt';
    if (selected) {
      ctx.shadowColor = c2dObjectSelectedGlow();
      ctx.shadowBlur = c2dObjectSelectedGlowBlur();
    }
    ctx.stroke();
    ctx.restore();
  }
  /** Local +X from stand box center to end of 45° flare (full-width main body); 0 if nose geometry unused. */
  function standSafetyAircraftCenterLocalXM(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return 0;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return 0;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return 0;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return 0;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return 0;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return 0;
    return xBendEnd;
  }
  function getStandAircraftMarkerWorldPxForPbb(pbb) {
    const cxy = getStandConnectionPx(pbb);
    return [cxy[0], cxy[1]];
  }
  function getStandAircraftMarkerWorldPxForRemoteLike(st) {
    const cxy = getStandConnectionPx(st);
    return [cxy[0], cxy[1]];
  }
  const APRON_SITE_MARKER_WIDTH_M = 5;
  const APRON_SITE_MARKER_HEIGHT_M = 1;
  function _ensureLayoutToastStack() {
    let stack = document.getElementById('layout-toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'layout-toast-stack';
      stack.className = 'layout-toast-stack';
      document.body.appendChild(stack);
    }
    return stack;
  }
  function _formatToastTimestamp(d) {
    const pad = function(n) { return String(n).padStart(2, '0'); };
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }
  function showLayoutSavedToast(layoutName, kind, subText) {
    const stack = _ensureLayoutToastStack();
    const el = document.createElement('div');
    const variant = kind === 'error' ? 'is-error' : 'is-success';
    el.className = 'layout-toast ' + variant;
    const title = document.createElement('div');
    title.className = 'layout-toast-title';
    const ts = _formatToastTimestamp(new Date());
    const name = String(layoutName || '').trim() || 'layout';
    if (kind === 'error') {
      title.textContent = ts + ' · save failed · ' + name;
    } else {
      title.textContent = ts + ' · saved · ' + name;
    }
    el.appendChild(title);
    if (subText) {
      const sub = document.createElement('div');
      sub.className = 'layout-toast-sub';
      sub.textContent = String(subText);
      el.appendChild(sub);
    }
    stack.appendChild(el);
    requestAnimationFrame(function() { el.classList.add('is-visible'); });
    const lifeMs = kind === 'error' ? 4200 : 2600;
    setTimeout(function() {
      el.classList.remove('is-visible');
      setTimeout(function() { if (el.parentNode === stack) stack.removeChild(el); }, 220);
    }, lifeMs);
  }
  /**
   * Render apron site anchor as a 5m × 1m rectangle. Long side (5m) is aligned
   * with the stand's stopbar direction (local Y = perpendicular to the stand
   * depth axis), so it matches the dashed stopbar line drawn near the nose.
   * ``standAngleRad`` is the stand's depth-axis angle (same as ctx.rotate used
   * when drawing the stand footprint). When omitted, the 5m side falls back to
   * world horizontal.
   */
  function drawApronSiteMarker(ctx, cx, cy, fillStyle, strokeStyle, selected, standAngleRad) {
    const w = APRON_SITE_MARKER_WIDTH_M;
    const h = APRON_SITE_MARKER_HEIGHT_M;
    const angle = (typeof standAngleRad === 'number' && isFinite(standAngleRad))
      ? standAngleRad + Math.PI / 2
      : 0;
    ctx.save();
    ctx.translate(cx, cy);
    if (angle) ctx.rotate(angle);
    ctx.beginPath();
    ctx.rect(-w / 2, -h / 2, w, h);
    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (strokeStyle) {
      const baseLw = Math.max(0.18, 0.24 / Math.max(state.scale, 0.1));
      ctx.lineWidth = selected ? baseLw * 1.5 : baseLw;
      ctx.strokeStyle = strokeStyle;
      ctx.stroke();
    }
    ctx.restore();
  }
  /** Apron–taxiway UI attach point: PBB = aircraft marker; remote/temp = same local xBendEnd offset as Contact (getStandAircraftMarkerWorldPxFor*). */
  function getStandApronTaxiwayAttachWorldPx(stand) {
    if (!stand) return [0, 0];
    const isPbb = (state.pbbStands || []).some(function(s) { return s && s.id === stand.id; });
    if (isPbb) return getStandConnectionPx(stand);
    return getStandAircraftMarkerWorldPxForRemoteLike(stand);
  }
  function getStandBoundsRect(cx, cy, sizeM) {
    const h = sizeM / 2;
    return { left: cx - h, right: cx + h, top: cy - h, bottom: cy + h };
  }
  function normalizeAngleDeg(deg) {
    let a = Number(deg);
    if (!isFinite(a)) a = 0;
    while (a > 180) a -= 360;
    while (a <= -180) a += 360;
    return a;
  }
  function getRemoteStandCenterPx(st) {
    if (!st) return [0, 0];
    if (st.apronSiteX != null && st.apronSiteY != null) {
      return [Number(st.apronSiteX), Number(st.apronSiteY)];
    }
    if (typeof st.x === 'number' && isFinite(st.x) && typeof st.y === 'number' && isFinite(st.y)) {
      return [Number(st.x), Number(st.y)];
    }
    return cellToPixel(st.col || 0, st.row || 0);
  }
  /** Temp stand: taxiway centerline snap (sim_input junctionX/Y); defaults to stand x,y. */
  function getTempStandTaxiwayJunctionPx(st) {
    if (!st) return [0, 0];
    const jx = st.junctionX != null ? Number(st.junctionX) : NaN;
    const jy = st.junctionY != null ? Number(st.junctionY) : NaN;
    if (Number.isFinite(jx) && Number.isFinite(jy)) return [jx, jy];
    return getRemoteStandCenterPx(st);
  }
  function getRemoteStandAngleRad(st) {
    const deg = normalizeAngleDeg(st && st.angleDeg != null ? st.angleDeg : 0);
    return deg * Math.PI / 180;
  }
  function getRemoteStandCorners(stLike) {
    const [cx, cy] = getRemoteStandCenterPx(stLike);
    const cat = (stLike && stLike.category) || 'C';
    const dep = getStandDepthMeters(cat);
    const halfD = dep / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getRemoteStandAngleRad(stLike);
    const shiftX = standStopbarCenterShiftLocalX(dep, cat);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + ((-halfD + shiftX))*cos - (-halfW)*sin, cy + ((-halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - (-halfW)*sin, cy + (( halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - ( halfW)*sin, cy + (( halfD + shiftX))*sin + ( halfW)*cos],
      [cx + ((-halfD + shiftX))*cos - ( halfW)*sin, cy + ((-halfD + shiftX))*sin + ( halfW)*cos]
    ];
  }
  function rectsOverlap(a, b) {
    return !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
  }
  function getPbbAnchorPx(pbb) {
    const x1 = Number(pbb && pbb.x1);
    const y1 = Number(pbb && pbb.y1);
    if (Number.isFinite(x1) && Number.isFinite(y1)) return [x1, y1];
    const bridges = Array.isArray(pbb && pbb.pbbBridges) ? pbb.pbbBridges : [];
    const starts = bridges.map(function(bridge) {
      const pts = Array.isArray(bridge.points) ? bridge.points : [];
      return pts.length ? [Number(pts[0].x) || 0, Number(pts[0].y) || 0] : null;
    }).filter(Boolean);
    if (starts.length) {
      let sx = 0, sy = 0;
      starts.forEach(function(pt) { sx += pt[0]; sy += pt[1]; });
      return [sx / starts.length, sy / starts.length];
    }
    return [0, 0];
  }
  function getPBBStandAngle(pbb) {
    if (pbb && pbb.angleDeg != null) return normalizeAngleDeg(pbb.angleDeg) * Math.PI / 180;
    const anchor = getPbbAnchorPx(pbb);
    const center = getStandConnectionPx(pbb);
    return Math.atan2(center[1] - anchor[1], center[0] - anchor[0]);
  }
  function getPBBStandCorners(pbb) {
    const center = getStandConnectionPx(pbb);
    const cx = center[0], cy = center[1];
    const cat = pbb.category || 'C';
    const dep = getStandDepthMeters(cat);
    const halfD = dep / 2;
    const halfW = getStandWidthMeters(cat) / 2;
    const angle = getPBBStandAngle(pbb);
    const shiftX = standStopbarCenterShiftLocalX(dep, cat);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      [cx + ((-halfD + shiftX))*cos - (-halfW)*sin, cy + ((-halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - (-halfW)*sin, cy + (( halfD + shiftX))*sin + (-halfW)*cos],
      [cx + (( halfD + shiftX))*cos - ( halfW)*sin, cy + (( halfD + shiftX))*sin + ( halfW)*cos],
      [cx + ((-halfD + shiftX))*cos - ( halfW)*sin, cy + ((-halfD + shiftX))*sin + ( halfW)*cos]
    ];
  }
  function pointInPolygonXY(p, verts) {
    let inside = false;
    const n = verts.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const vi = verts[i], vj = verts[j];
      if (((vi[1] > p[1]) !== (vj[1] > p[1])) && (p[0] < (vj[0]-vi[0])*(p[1]-vi[1])/(vj[1]-vi[1])+vi[0])) inside = !inside;
    }
    return inside;
  }
  function segIntersect(a1, a2, b1, b2) {
    const [ax1,ay1]=a1,[ax2,ay2]=a2,[bx1,by1]=b1,[bx2,by2]=b2;
    const dax = ax2-ax1, day = ay2-ay1, dbx = bx2-bx1, dby = by2-by1;
    const den = dax*dby - day*dbx;
    if (Math.abs(den) < 1e-10) return false;
    const t = ((bx1-ax1)*dby - (by1-ay1)*dbx) / den;
    const s = ((bx1-ax1)*day - (by1-ay1)*dax) / den;
    return t >= 0 && t <= 1 && s >= 0 && s <= 1;
  }
  function rotatedRectsOverlap(cornersA, cornersB) {
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersA[i], cornersB)) return true;
    for (let i = 0; i < 4; i++) if (pointInPolygonXY(cornersB[i], cornersA)) return true;
    for (let i = 0; i < 4; i++) {
      const a1 = cornersA[i], a2 = cornersA[(i+1)%4];
      for (let j = 0; j < 4; j++) {
        if (segIntersect(a1, a2, cornersB[j], cornersB[(j+1)%4])) return true;
      }
    }
    return false;
  }
  function polygonsOverlapXY(polyA, polyB) {
    if (!Array.isArray(polyA) || !Array.isArray(polyB) || polyA.length < 3 || polyB.length < 3) return false;
    for (let i = 0; i < polyA.length; i++) if (pointInPolygonXY(polyA[i], polyB)) return true;
    for (let i = 0; i < polyB.length; i++) if (pointInPolygonXY(polyB[i], polyA)) return true;
    for (let i = 0; i < polyA.length; i++) {
      const a1 = polyA[i], a2 = polyA[(i + 1) % polyA.length];
      for (let j = 0; j < polyB.length; j++) {
        if (segIntersect(a1, a2, polyB[j], polyB[(j + 1) % polyB.length])) return true;
      }
    }
    return false;
  }
  function polygonAabbXY(poly) {
    if (!Array.isArray(poly) || !poly.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const x = Number(p && p[0]), y = Number(p && p[1]);
      if (!isFinite(x) || !isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)
      ? { left: minX, right: maxX, top: minY, bottom: maxY }
      : null;
  }
  function aabbRectOverlap(a, b) {
    return !!(a && b && !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom));
  }
  function distPointToSegment(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay;
    const wx = px - ax, wy = py - ay;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return Math.hypot(px - ax, py - ay);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(px - bx, py - by);
    const t = c1 / c2;
    const projx = ax + t * vx, projy = ay + t * vy;
    return Math.hypot(px - projx, py - projy);
  }
  function minDistanceConvexQuads(quadA, quadB) {
    if (rotatedRectsOverlap(quadA, quadB)) return 0;
    let minD = Infinity;
    for (let i = 0; i < 4; i++) {
      const p = quadA[i];
      for (let j = 0; j < 4; j++) {
        const q1 = quadB[j], q2 = quadB[(j + 1) % 4];
        minD = Math.min(minD, distPointToSegment(p[0], p[1], q1[0], q1[1], q2[0], q2[1]));
      }
    }
    for (let i = 0; i < 4; i++) {
      const p = quadB[i];
      for (let j = 0; j < 4; j++) {
        const q1 = quadA[j], q2 = quadA[(j + 1) % 4];
        minD = Math.min(minD, distPointToSegment(p[0], p[1], q1[0], q1[1], q2[0], q2[1]));
      }
    }
    return minD;
  }
  function standFootprintsTooClose(cornersA, catA, cornersB, catB) {
    const need = getStandSpacingMeters(catA, catB);
    if (need <= 0) return rotatedRectsOverlap(cornersA, cornersB);
    return minDistanceConvexQuads(cornersA, cornersB) < need;
  }
  function buildStandSafetyPolygonLocalPoints(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return null;
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(nw) || nw <= 0 || !isFinite(nc) || nc <= 0) return null;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const eps = 0.08;
    if (noseHalf >= halfW - eps) return null;
    const xNose = -halfD;
    const xStop = -halfD + nc;
    if (xStop <= xNose + eps || xStop >= halfD - eps) return null;
    const latRun = halfW - noseHalf;
    if (latRun <= eps) return null;
    const xBendEnd = xStop + latRun;
    if (xBendEnd > halfD + eps) return null;
    const pts = [];
    pts.push([xNose, -noseHalf]);
    pts.push([xNose, noseHalf]);
    pts.push([xStop, noseHalf]);
    pts.push([Math.min(xBendEnd, halfD), halfW]);
    if (xBendEnd < halfD - eps) {
      pts.push([halfD, halfW]);
      pts.push([halfD, -halfW]);
      pts.push([xBendEnd, -halfW]);
    } else {
      pts.push([halfD, -halfW]);
    }
    pts.push([xStop, -noseHalf]);
    return pts;
  }
  function standOuterContourWorldPolygonForSpec(cx, cy, angleRad, depM, widM, category) {
    const polyLocal = buildStandSafetyPolygonLocalPoints(depM, widM, category);
    if (polyLocal && polyLocal.length >= 3) {
      return polyLocal.map(function(p) { return standFootprintLocalToWorld(cx, cy, angleRad, p[0], p[1]); });
    }
    return [
      standFootprintLocalToWorld(cx, cy, angleRad, -depM / 2, -widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, depM / 2, -widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, depM / 2, widM / 2),
      standFootprintLocalToWorld(cx, cy, angleRad, -depM / 2, widM / 2),
    ];
  }
  function buildStandDuplicateSafetyPolygonLocalPoints(depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return null;
    const g = Number(r.gap), ws = Number(r.wingspan), pb = Number(r.pushback);
    const nw = Number(r.nose_width), nc = Number(r.nose_clear);
    if (!isFinite(g) || !isFinite(ws) || !isFinite(pb) || !isFinite(nw) || !isFinite(nc) || g <= 0 || ws <= 0 || nw <= 0 || nc <= 0) return null;
    const halfD = depM / 2, halfW = widM / 2;
    const noseHalf = nw / 2;
    const shiftX = standStopbarCenterShiftLocalX(depM, category);
    const xNose = -halfD + shiftX;
    const xStop = 0;
    const xBendEnd = xStop + (halfW - noseHalf);
    const xPush = (halfD - pb) + shiftX;
    const yLim = halfW - g;
    const xMin = (-halfD) + shiftX;
    const xMax = halfD + shiftX;
    const eps = 0.12;
    if (!(noseHalf < halfW - eps)) return null;
    if (!(xBendEnd <= xMax + eps)) return null;
    if (!(yLim > eps && yLim < halfW - eps)) return null;
    if (!(xStop > xMin + eps && xStop < xMax - eps)) return null;
    const xA = Math.max(xMin, Math.min(xMax, Math.min(xStop, xPush)));
    const xB = Math.max(xMin, Math.min(xMax, Math.max(xStop, xPush)));
    if (!(xB > xA + eps)) return null;
    const contour = [
      [xNose, -noseHalf],
      [xNose, noseHalf],
      [xStop, noseHalf],
      [Math.min(xBendEnd, xMax), halfW],
      [xMax, halfW],
      [xMax, -halfW],
      [Math.min(xBendEnd, xMax), -halfW],
      [xStop, -noseHalf],
    ];
    return clipPolygonToAxisAlignedBox(contour, xA, xB, -yLim, yLim);
  }
  function clipPolygonToAxisAlignedBox(poly, minX, maxX, minY, maxY) {
    if (!Array.isArray(poly) || poly.length < 3) return null;
    function clip(input, axis, keepGreater, value) {
      const out = [];
      for (let i = 0; i < input.length; i++) {
        const a = input[i];
        const b = input[(i + 1) % input.length];
        const av = axis === 'x' ? a[0] : a[1];
        const bv = axis === 'x' ? b[0] : b[1];
        const ain = keepGreater ? av >= value - 1e-9 : av <= value + 1e-9;
        const bin = keepGreater ? bv >= value - 1e-9 : bv <= value + 1e-9;
        if (ain && bin) {
          out.push(b);
        } else if (ain !== bin) {
          const denom = bv - av;
          if (Math.abs(denom) > 1e-12) {
            const t = (value - av) / denom;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
          }
          if (!ain && bin) out.push(b);
        }
      }
      return out;
    }
    let out = poly.slice();
    out = clip(out, 'x', true, minX);
    out = clip(out, 'x', false, maxX);
    out = clip(out, 'y', true, minY);
    out = clip(out, 'y', false, maxY);
    return out.length >= 3 ? out : null;
  }
  function standDuplicateSafetyWorldPolygonForSpec(cx, cy, angleRad, depM, widM, category) {
    const polyLocal = buildStandDuplicateSafetyPolygonLocalPoints(depM, widM, category);
    if (!polyLocal || polyLocal.length < 3) return null;
    return polyLocal.map(function(p) { return standFootprintLocalToWorld(cx, cy, angleRad, p[0], p[1]); });
  }
  function standSafetyOverlapSpec(stand) {
    if (!stand || stand.id == null) return null;
    const id = String(stand.id);
    const isPbb = (state.pbbStands || []).some(function(s) { return s && String(s.id) === id; });
    const center = isPbb ? getStandConnectionPx(stand) : getRemoteStandCenterPx(stand);
    const angle = isPbb ? getPBBStandAngle(stand) : getRemoteStandAngleRad(stand);
    const category = stand.category || 'C';
    const dep = getStandDepthMeters(category);
    const wid = getStandWidthMeters(category);
    if (!center || !isFinite(center[0]) || !isFinite(center[1]) || !isFinite(dep) || !isFinite(wid)) return null;
    const poly = standDuplicateSafetyWorldPolygonForSpec(center[0], center[1], angle, dep, wid, category);
    const aabb = polygonAabbXY(poly);
    return poly && aabb ? { id: id, stand: stand, poly: poly, aabb: aabb } : null;
  }
  function recomputeDuplicateApronByStandId() {
    const specs = (typeof allStandsForFlightAssignment === 'function' ? allStandsForFlightAssignment() : [])
      .map(standSafetyOverlapSpec)
      .filter(Boolean);
    const map = {};
    for (let i = 0; i < specs.length; i++) {
      const a = specs[i];
      if (!map[a.id]) map[a.id] = [];
      for (let j = i + 1; j < specs.length; j++) {
        const b = specs[j];
        if (!aabbRectOverlap(a.aabb, b.aabb)) continue;
        if (!polygonsOverlapXY(a.poly, b.poly)) continue;
        if (!map[a.id]) map[a.id] = [];
        if (!map[b.id]) map[b.id] = [];
        map[a.id].push(b.id);
        map[b.id].push(a.id);
      }
    }
    specs.forEach(function(spec) {
      const list = (map[spec.id] || []).slice().sort();
      spec.stand.duplicate_apron_list = list;
      map[spec.id] = list;
    });
    state.duplicateApronByStandId = map;
    return map;
  }
  function duplicateApronStandIdsForStand(standId) {
    const key = standId != null ? String(standId) : '';
    if (!key) return [];
    const map = state.duplicateApronByStandId || {};
    const arr = Array.isArray(map[key]) ? map[key] : [];
    return arr.map(function(id) { return String(id); }).filter(Boolean);
  }
  function standGapSegmentsWorldForSpec(cx, cy, angleRad, depM, widM, category) {
    const r = standConfigRowForIcaoCat(category);
    if (!r || !isFinite(depM) || !isFinite(widM) || depM <= 0 || widM <= 0) return [];
    const g = Number(r.gap), ws = Number(r.wingspan);
    const halfD = depM / 2, halfW = widM / 2;
    const eps = 0.12;
    if (!(isFinite(g) && g > eps && isFinite(ws) && ws > 0)) return [];
    const yLim = halfW - g;
    if (!(yLim > eps && yLim < halfW - eps)) return [];
    const a0 = standFootprintLocalToWorld(cx, cy, angleRad, -halfD, yLim);
    const a1 = standFootprintLocalToWorld(cx, cy, angleRad, halfD, yLim);
