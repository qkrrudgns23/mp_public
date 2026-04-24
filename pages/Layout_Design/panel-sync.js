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
