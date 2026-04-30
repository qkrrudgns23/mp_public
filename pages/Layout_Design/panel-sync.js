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
        f.arrTerminalId = f.arrTerminalId || t.arrTerminalId || f.terminalId || null;
        f.depTerminalId = f.depTerminalId || t.depTerminalId || f.terminalId || null;
        const apronId = f.depApronId != null ? f.depApronId : (t.apronId != null ? t.apronId : (f.standId != null ? f.standId : f.arrApronId || null));
        f.standId = apronId;
        f.token = {
          nodes: Array.isArray(t.nodes) ? t.nodes.slice() : ['runway','taxiway','apron','terminal'],
          runwayId: f.arrRunwayId || null,
          apronId: apronId,
          terminalId: f.terminalId || null,
          arrTerminalId: f.arrTerminalId || null,
          depTerminalId: f.depTerminalId || null,
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
          f.timeline = null;
          if (rawMeta && typeof rawMeta === 'object') {
            f.timeline_meta = Object.assign({}, rawMeta);
            hydrateEMinutesFromTimelineMetaIfMissing(f);
          } else {
            delete f.timeline_meta;
          }
          if (rawProSimEl && rawProSimEl.length) f.proSimEdgeList = rawProSimEl;
          else delete f.proSimEdgeList;
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
            hydrateEMinutesFromTimelineMetaIfMissing(f);
          } else {
            f.timeline = null;
            if (rawMeta && typeof rawMeta === 'object') {
              f.timeline_meta = Object.assign({}, rawMeta);
              hydrateEMinutesFromTimelineMetaIfMissing(f);
            } else {
              delete f.timeline_meta;
            }
            if (rawProSimEl && rawProSimEl.length) f.proSimEdgeList = rawProSimEl;
            else delete f.proSimEdgeList;
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
    if (typeof deriveDeadlockGhostPlaybackFromPayload === 'function') {
      state.simDeadlockGhostPlayback = deriveDeadlockGhostPlaybackFromPayload(payload, state.flights);
    } else {
      state.simDeadlockGhostPlayback = { events: [], bodyLines: '', resolveCount: 0 };
    }
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
