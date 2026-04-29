    }
    if (dp && dp.v === 1 && dp.simPlaybackEndCapSec != null && isFinite(Number(dp.simPlaybackEndCapSec))) {
      state.simPlaybackEndCapSec = Number(dp.simPlaybackEndCapSec);
    }
    applyDesignerPersistMapTypeAfterLoad(dp);
    syncMapTypePopoverFromState();
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
    if (restoreProSimSyncUi) {
      state.globalUpdateFresh = true;
      if (dp.designerPageUpdateFresh === true) {
        if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
      } else {
        state.designerPageUpdateFresh = false;
        const ddot = document.getElementById('designerPageUpdateSyncDot');
        if (ddot) {
          ddot.classList.remove('fresh');
          ddot.classList.add('stale');
          ddot.setAttribute('title', '레이아웃/객체 변경됨 — Update를 눌러 경로 그래프·뷰를 동기화하세요');
        }
      }
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
    }
    if (typeof renderFlightList === 'function') renderFlightList(false, false);
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  /** E-series minutes; ARR_ROT_SEC / VTT_* / DTT_* / DEP_ROT_SEC from ``airside_sim`` schedule. */
  function applyAirsideScheduleRowToFlight(f, srec) {
    if (!f) return;
    if (!srec || typeof srec !== 'object') {
      delete f.eldtMin;
      delete f.eibtMin;
      delete f.eobtMin;
      delete f.etotMin;
      delete f.eldtMin;
      delete f.eibtMin;
      delete f.eobtMin;
      delete f.etotMin;
      delete f.eibtMinList;
      delete f.eobtMinList;
      delete f.ePushFinishedMinList;
      f.arrRotSec = null;
      delete f.proSimVttArrSec;
      delete f.proSimVttDepSec;
      delete f.proSimDttArrSec;
      delete f.proSimPushbackSec;
      delete f.proSimDttDepSec;
      delete f.proSimDepLineupSec;
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
    function secListToMinList(key) {
      const raw = srec[key];
      if (!Array.isArray(raw)) return null;
      const out = raw.map(function(v) {
        const n = Number(v);
        return isFinite(n) ? n / 60 : null;
      });
      return out.length ? out : null;
    }
    const eibtList = secListToMinList('EIBT_LIST');
    const eobtList = secListToMinList('EOBT_LIST');
    const epushList = secListToMinList('E_PUSH_FINISHED_LIST');
    if (eibtList) f.eibtMinList = eibtList;
    else delete f.eibtMinList;
    if (eobtList) f.eobtMinList = eobtList;
    else delete f.eobtMinList;
    if (epushList) f.ePushFinishedMinList = epushList;
    else delete f.ePushFinishedMinList;
    const rotS = secOpt('ARR_ROT_SEC');
    if (isFinite(rotS)) f.arrRotSec = rotS;
    else f.arrRotSec = null;
    const vttArrS = secOpt('VTT_ARR_SEC');
    if (isFinite(vttArrS)) f.proSimVttArrSec = vttArrS;
    else delete f.proSimVttArrSec;
    const pushbackS = secOpt('PUSHBACK_SEC');
    if (isFinite(pushbackS)) f.proSimPushbackSec = pushbackS;
    else delete f.proSimPushbackSec;
    const vttDepS = secOpt('VTT_DEP_SEC');
    if (isFinite(vttDepS)) f.proSimVttDepSec = vttDepS;
    else delete f.proSimVttDepSec;
    const dttArrS = secOpt('DTT_ARR_SEC');
    if (isFinite(dttArrS)) f.proSimDttArrSec = dttArrS;
    else delete f.proSimDttArrSec;
    const dttDepS = secOpt('DTT_DEP_SEC');
    if (isFinite(dttDepS)) f.proSimDttDepSec = dttDepS;
    else delete f.proSimDttDepSec;
    const depRotS = secOpt('DEP_ROT_SEC');
    if (isFinite(depRotS)) f.proSimDepLineupSec = depRotS;
    else delete f.proSimDepLineupSec;
  }
  function flightEMinListForSchedule(f, listKey, metaSecListKey, scalarKey) {
    const raw = f && Array.isArray(f[listKey]) ? f[listKey] : null;
    if (raw && raw.length) return raw.map(function(v) {
      const n = Number(v);
      return isFinite(n) ? n : null;
    });
    const meta = f && f.timeline_meta && typeof f.timeline_meta === 'object' ? f.timeline_meta : null;
    const mraw = meta && Array.isArray(meta[metaSecListKey]) ? meta[metaSecListKey] : null;
    if (mraw && mraw.length) return mraw.map(function(v) {
      const n = Number(v);
      return isFinite(n) ? n / 60 : null;
    });
    const scalar = f && f[scalarKey] != null ? Number(f[scalarKey]) : NaN;
    return isFinite(scalar) ? [scalar] : [];
  }
  function buildFlightTimelineFromPlaybackPoints(rawPts) {
    const pts = Array.isArray(rawPts) ? rawPts : [];
    if (pts.length < 2) return null;
    const tl = pts.map(function(p) {
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
    return tl.length >= 2 ? tl : null;
  }
  function refreshHasSimulationResultFromPlaybackSources() {
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (f && f.timeline && f.timeline.length >= 2) {
        state.hasSimulationResult = true;
        return;
      }
    }
    const pos = state.simPlaybackPositionsByFlightId;
    if (pos && typeof pos === 'object') {
      for (const k in pos) {
        if (!Object.prototype.hasOwnProperty.call(pos, k)) continue;
        const arr = pos[k];
        if (Array.isArray(arr) && arr.length >= 2) {
          state.hasSimulationResult = true;
          return;
        }
      }
    }
    state.hasSimulationResult = false;
  }
  function evictFlightPlaybackTimelinesWhenPlayBlocked() {
    if (!state.simPlaybackPositionsByFlightId || typeof state.simPlaybackPositionsByFlightId !== 'object') return false;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f) continue;
      f.timeline = null;
      delete f.timeline_meta;
    }
    state.simPlaybackTimelinesEvictedForMemory = true;
    _lazyTimelineLastEvictSimSec = NaN;
    refreshHasSimulationResultFromPlaybackSources();
    return true;
  }
  function rehydrateFlightPlaybackTimelinesAfterPlayAllowed() {
    if (!state.simPlaybackTimelinesEvictedForMemory) return false;
    const positions = state.simPlaybackPositionsByFlightId;
    const scheduleList = Array.isArray(state.simPlaybackScheduleSnapshot) ? state.simPlaybackScheduleSnapshot : [];
    const schedById = {};
    for (let si = 0; si < scheduleList.length; si++) {
      const s = scheduleList[si];
      if (s && s.flight_id != null) schedById[String(s.flight_id)] = s;
    }
    if (!positions || typeof positions !== 'object') {
      state.simPlaybackTimelinesEvictedForMemory = false;
      refreshHasSimulationResultFromPlaybackSources();
      return true;
    }
    let mergedTimelines = 0;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || f.id == null) continue;
      const srec = schedById[String(f.id)] || null;
      const rawPts = positions[f.id];
      if (rawPts != null) {
        const tl = buildFlightTimelineFromPlaybackPoints(rawPts);
        if (tl) {
          mergedTimelines++;
          f.timeline = tl;
        }
      }
      if (srec && f.timeline && f.timeline.length >= 2) {
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
    }
    state.hasSimulationResult = mergedTimelines > 0;
    state.simPlaybackTimelinesEvictedForMemory = false;
    refreshHasSimulationResultFromPlaybackSources();
    return true;
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
    if (truncCap != null) {
      state.simPlaybackEndCapSec = truncCap;
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
          const tl = buildFlightTimelineFromPlaybackPoints(rawPts);
          if (tl) {
            mergedTimelines++;
            f.timeline = tl;
          }
        }
      }
      if (srec && f.timeline && f.timeline.length >= 2) {
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
