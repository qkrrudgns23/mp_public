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
        if (typeof window.ensureFlightLookaheadArrDepFlight === 'function') window.ensureFlightLookaheadArrDepFlight(f);
        else {
          (function() {
            function clampLA(v) {
              if (v == null || v === '' || !isFinite(Number(v))) return null;
              return Math.max(0, Math.min(200, Math.floor(Number(v))));
            }
            var leg = clampLA(f.lookaheadTaxi);
            var a = clampLA(f.lookaheadArr);
            var dep = clampLA(f.lookaheadDep);
            var base = leg !== null ? leg : 9;
            f.lookaheadArr = a !== null ? a : base;
            f.lookaheadDep = dep !== null ? dep : base;
          })();
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
    state._pendingPersistSimWindow = null;
    if (dp && dp.v === 1
        && dp.simWindowStartSec != null && dp.simWindowEndSec != null
        && isFinite(Number(dp.simWindowStartSec)) && isFinite(Number(dp.simWindowEndSec))) {
      state._pendingPersistSimWindow = {
        lo: Number(dp.simWindowStartSec),
        hi: Number(dp.simWindowEndSec),
      };
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
        } else if (typeof markDesignerPageUpdateStale === 'function') {
          markDesignerPageUpdateStale();
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
