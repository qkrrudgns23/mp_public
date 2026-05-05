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
  function syncSingleApronStaySegmentFromAggregate(f) {
    if (!f || !Array.isArray(f.apronStaySegments) || f.apronStaySegments.length !== 1) return false;
    const sibt = f.sibtMin != null && isFinite(Number(f.sibtMin))
      ? Number(f.sibtMin)
      : (f.timeMin != null && isFinite(Number(f.timeMin)) ? Number(f.timeMin) : null);
    const sobt = f.sobtMin != null && isFinite(Number(f.sobtMin))
      ? Number(f.sobtMin)
      : (sibt != null ? sibt + Math.max(0, Number(f.dwellMin) || 0) : null);
    if (sibt == null || sobt == null || sobt <= sibt) return false;
    const cur = f.apronStaySegments[0] || {};
    const standId = cur.standId != null && String(cur.standId).trim() !== ''
      ? String(cur.standId)
      : (f.standId != null && String(f.standId).trim() !== '' ? String(f.standId) : null);
    f.apronStaySegments = [{ standId: standId, sibtMin: Math.max(0, sibt), sobtMin: Math.max(0, sobt) }];
    syncFlightApronStayAggregate(f);
    return true;
  }
  function collapseFlightApronStaySegmentsIfSingleStand(f) {
    if (!f || !Array.isArray(f.apronStaySegments) || f.apronStaySegments.length <= 1) return false;
    const segs = normalizeFlightApronStaySegments(f);
    if (segs.length <= 1) return false;
    const firstStandKey = String(segs[0].standId || '');
    for (let i = 1; i < segs.length; i++) {
      if (String(segs[i].standId || '') !== firstStandKey) return false;
    }
    const first = segs[0];
    const last = segs[segs.length - 1];
    f.apronStaySegments = [{
      standId: first.standId || null,
      sibtMin: first.sibtMin,
      sobtMin: last.sobtMin
    }];
    syncFlightApronStayAggregate(f);
    return true;
  }
  function collapseSingleStandApronStaySegmentsForFlights(flights) {
    const changed = [];
    (flights || []).forEach(function(f) {
      if (!f || typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f)) return;
      if (collapseFlightApronStaySegmentsIfSingleStand(f)) changed.push(f);
    });
    return changed;
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
    const eibtList = Array.isArray(eSer && eSer.eibtList) ? eSer.eibtList : [];
    const eobtList = Array.isArray(eSer && eSer.eobtList) ? eSer.eobtList : [];
    const hasSegmentEList = eibtList.length >= segs.length && eobtList.length >= segs.length;
    return segs.map(function(seg, idx) {
      const segEibt = hasSegmentEList ? eibtList[idx] : eSer.eibt;
      const segEobt = hasSegmentEList ? eobtList[idx] : eSer.eobt;
      return {
        f: f,
        t0: seg.sibtMin,
        t1: seg.sobtMin,
        sldt: sldt,
        stot: stot,
        eibt: segEibt,
        eobt: segEobt,
        eldt: eSer.eldt,
        etot: eSer.etot,
        eBarSegmented: hasSegmentEList,
        sldtOrig: sldt,
        sobtOrig: last.sobtMin,
        stotOrig: stot,
        segmentIdx: idx,
        segmentCount: segs.length,
        segmentStandId: seg.standId || null
      };
    });
  }
  /** If E minute fields were omitted from persist but Pro Sim left them on ``timeline_meta``, fill them for schedule UI. */
  function hydrateEMinutesFromTimelineMetaIfMissing(f) {
    if (!f || !f.timeline_meta || typeof f.timeline_meta !== 'object') return;
    const m = f.timeline_meta;
    function setMin(minKey, secKey) {
      if (f[minKey] != null && isFinite(Number(f[minKey]))) return;
      const n = m[secKey] != null ? Number(m[secKey]) : NaN;
      if (isFinite(n)) f[minKey] = n / 60;
    }
    setMin('eldtMin', 'eldtSec');
    setMin('eibtMin', 'eibtSec');
    setMin('eobtMin', 'eobtSec');
    setMin('etotMin', 'etotSec');
  }
  function applyLayoutObject(obj) {
    if (!obj || typeof obj !== 'object') return;
    state.simPlaybackEndCapSec = null;
    const dp = obj.designerPersist;
    const persistPlaybackSnapshot = !!(dp && dp.v === 1 && dp.hasSimulationPlayback === true);
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
        f.arrTerminalId = f.arrTerminalId || t.arrTerminalId || f.terminalId || null;
        f.depTerminalId = f.depTerminalId || t.depTerminalId || f.terminalId || null;
        if (f.lookaheadTaxi == null || f.lookaheadTaxi === '' || !isFinite(Number(f.lookaheadTaxi))) {
          f.lookaheadTaxi = 9;
        } else {
          f.lookaheadTaxi = Math.max(0, Math.min(200, Math.floor(Number(f.lookaheadTaxi))));
        }
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
        if (persistPlaybackSnapshot && f.arrDep !== 'Dep') {
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
    if (typeof recomputeDuplicateApronByStandId === 'function') recomputeDuplicateApronByStandId();
    if (obj.simPlaybackPositionsByFlightId && typeof obj.simPlaybackPositionsByFlightId === 'object') {
      state.simPlaybackPositionsByFlightId = obj.simPlaybackPositionsByFlightId;
      state.simPlaybackTimelinesEvictedForMemory = false;
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
    const savedDlp = obj.simDeadlockGhostPlayback;
    let savedRc = 0;
    if (savedDlp && isFinite(Number(savedDlp.resolveCount))) savedRc = Math.floor(Number(savedDlp.resolveCount));
    if (state.simPlaybackPositionsByFlightId && typeof state.simPlaybackPositionsByFlightId === 'object') {
      state.simDeadlockGhostPlayback = deriveDeadlockGhostPlaybackFromPayload(
        { positions: state.simPlaybackPositionsByFlightId, deadlock_resolve_event_count: savedRc },
        state.flights
      );
    } else if (savedDlp && typeof savedDlp === 'object' && Array.isArray(savedDlp.events) && savedDlp.events.length) {
      state.simDeadlockGhostPlayback = {
        events: savedDlp.events.map(function(ev) {
          const o = { t_abs: Number(ev.t_abs), labels: Array.isArray(ev.labels) ? ev.labels.slice() : [] };
          if (ev.focusWorldX != null && isFinite(Number(ev.focusWorldX))) o.focusWorldX = Number(ev.focusWorldX);
          if (ev.focusWorldY != null && isFinite(Number(ev.focusWorldY))) o.focusWorldY = Number(ev.focusWorldY);
          return o;
        }),
        bodyLines: typeof savedDlp.bodyLines === 'string' ? savedDlp.bodyLines : '',
        resolveCount: savedRc,
      };
    } else {
      state.simDeadlockGhostPlayback = { events: [], bodyLines: '', resolveCount: 0 };
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
    let didAutoPathGraphSync = false;
    if (PATH_GRAPH_SYNC_ONLY_ON_EXPLICIT_ACTION && state.layers && state.layers.junction) {
      try {
        if (typeof applyPathGraphSyncNow === 'function') applyPathGraphSyncNow();
        didAutoPathGraphSync = true;
        if (typeof draw === 'function') draw();
        if (typeof markDesignerPageUpdateFresh === 'function') markDesignerPageUpdateFresh();
      } catch (ePg) {
        console.warn('applyLayoutObject: path graph sync', ePg);
      }
    }
    if (dp && dp.v === 1) {
      state.globalUpdateFresh = !!dp.globalUpdateFresh;
      if (!didAutoPathGraphSync) {
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
  function isCompactPlaybackTrack(raw) {
    return !!(
      raw && typeof raw === 'object' && raw.format === 'compact_v2' &&
      Array.isArray(raw.t) && Array.isArray(raw.x) && Array.isArray(raw.y) && Array.isArray(raw.v) &&
      raw.t.length === raw.x.length && raw.t.length === raw.y.length && raw.t.length === raw.v.length
    );
  }
  function compactPlaybackTrackLength(raw) {
    return isCompactPlaybackTrack(raw) ? raw.t.length : 0;
  }
  function compactPlaybackTrackForFlight(f) {
    if (!f || f.id == null) return null;
    const map = state.simPlaybackPositionsByFlightId;
    if (!map || typeof map !== 'object') return null;
    const tr = map[String(f.id)];
    return isCompactPlaybackTrack(tr) ? tr : null;
  }
  function compactPlaybackDghostSet(track) {
    if (!track) return new Set();
    if (track.__dghostSet instanceof Set) return track.__dghostSet;
    const s = new Set();
    const arr = Array.isArray(track.dghost_t) ? track.dghost_t : [];
    for (let i = 0; i < arr.length; i++) {
      const t = Math.round(Number(arr[i]));
      if (isFinite(t)) s.add(t);
    }
    track.__dghostSet = s;
    return s;
  }
  function compactPlaybackDghostMergedRangesSec(track) {
    if (!track || !Array.isArray(track.dghost_t) || !track.dghost_t.length) return [];
    const nums = [];
    for (let i = 0; i < track.dghost_t.length; i++) {
      const v = Math.round(Number(track.dghost_t[i]));
      if (isFinite(v)) nums.push(v);
    }
    if (!nums.length) return [];
    nums.sort(function(a, b) { return a - b; });
    const out = [];
    let s0 = nums[0], e0 = nums[0];
    for (let j = 1; j < nums.length; j++) {
      const n = nums[j];
      if (n <= e0 + 1) e0 = n;
      else {
        out.push([s0, e0]);
        s0 = e0 = n;
      }
    }
    out.push([s0, e0]);
    return out;
  }
  /** True if compact playback track records any deadlock ghost sample (simulation seconds). */
  function allocFlightTrackHasDeadlock(trDead) {
    return !!(trDead && Array.isArray(trDead.dghost_t) && trDead.dghost_t.length > 0);
  }
  /** Flight Schedule row: deadlock from last Pro Sim compact playback, timeline ghost, or persisted id set. */
  function flightScheduleRowHasDeadlock(f) {
    if (!f || f.id == null) return false;
    const idStr = String(f.id);
    if (state.deadlockFlightIdsFromLastSim && state.deadlockFlightIdsFromLastSim[idStr]) return true;
    const tr = compactPlaybackTrackForFlight(f);
    if (allocFlightTrackHasDeadlock(tr)) return true;
    if (f.timeline && Array.isArray(f.timeline)) {
      for (let i = 0; i < f.timeline.length; i++) {
        if (f.timeline[i] && f.timeline[i].deadlockGhost === true) return true;
      }
    }
    return false;
  }
  /** Gantt apron bar: red overlay only where sim seconds fall in merged dghost ranges (time axis = sec/60). */
  function allocFlightDeadlockOverlayHtml(trDead, segT0Min, segT1Min, visT0Min, visT1Min) {
    if (!trDead || !Array.isArray(trDead.dghost_t) || !trDead.dghost_t.length) return '';
    const ranges = compactPlaybackDghostMergedRangesSec(trDead);
    if (!ranges.length) return '';
    const denom = visT1Min - visT0Min;
    if (!(denom > 1e-12)) return '';
    const parts = [];
    for (let r = 0; r < ranges.length; r++) {
      const ds = ranges[r][0], de = ranges[r][1];
      const m0 = ds / 60;
      const m1 = (de + 1) / 60;
      const a = Math.max(visT0Min, m0, segT0Min);
      const b = Math.min(visT1Min, m1, segT1Min);
      if (!(b > a + 1e-12)) continue;
      const leftRel = ((a - visT0Min) / denom) * 100;
      const wRel = Math.max(0.5, ((b - a) / denom) * 100);
      parts.push('<div class="alloc-flight-deadlock-seg" style="left:' + leftRel + '%;width:' + wRel + '%;"></div>');
    }
    return parts.length ? parts.join('') : '';
  }
  function compactPlaybackTugIntervals(track) {
    if (!track) return [];
    if (Array.isArray(track.__tugIntervals)) return track.__tugIntervals;
    const raw = Array.isArray(track.tug_intervals) ? track.tug_intervals : [];
    const out = [];
    for (let i = 0; i < raw.length; i++) {
      const it = raw[i];
      if (!it || typeof it !== 'object') continue;
      const start = Number(it.start != null ? it.start : it.t0);
      const end = Number(it.end != null ? it.end : it.t1);
      if (isFinite(start) && isFinite(end) && end > start) out.push({ start: start, end: end });
    }
    track.__tugIntervals = out;
    return out;
  }
  function compactPlaybackNeedsTugAt(track, tSec) {
