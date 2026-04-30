  }
  
  function runwayArrSpeedAndTimeToRet(v0, a, distM, vFloorIn) {
    const vf0 = Math.max(1, Math.min(150, vFloorIn));
    const vf = Math.min(vf0, v0);
    if (!(a > 0) || distM <= 0) return { vAtRet: v0, tSec: 0 };
    if (v0 <= vf) return { vAtRet: v0, tSec: distM / Math.max(v0, 1e-6) };
    const dStop = (v0 * v0 - vf * vf) / (2 * a);
    if (distM < dStop) {
      const vEnd = Math.sqrt(Math.max(0, v0 * v0 - 2 * a * distM));
      return { vAtRet: vEnd, tSec: (v0 - vEnd) / a };
    }
    const tDecel = (v0 - vf) / a;
    const tCruise = (distM - dStop) / vf;
    return { vAtRet: vf, tSec: tDecel + tCruise };
  }
  function parseTimeToMinutes(val) {
    if (!val) return 0;
    const s = String(val).trim();
    if (!s) return 0;
    const isoDt = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?/);
    if (isoDt) {
      const h = parseInt(isoDt[4], 10) || 0;
      const m = parseInt(isoDt[5], 10) || 0;
      const sec = isoDt[6] ? (parseInt(isoDt[6], 10) || 0) : 0;
      return Math.max(0, h * 60 + m + sec / 60);
    }
    if (s.includes(':')) {
      const parts = s.split(':');
      const h = parseInt(parts[0], 10) || 0;
      const m = parseInt(parts[1], 10) || 0;
      const sec = (parts.length >= 3) ? (parseInt(parts[2], 10) || 0) : 0;
      return Math.max(0, h * 60 + m + sec / 60);
    }
    const num = parseFloat(s);
    return isNaN(num) ? 0 : Math.max(0, num);
  }

  function snapSimTimeSecForSlider(tSec) {
    const lo = state.simStartSec;
    const hi = state.simDurationSec;
    const step = SIM_TIME_SLIDER_SNAP_SEC;
    const t = Number(tSec);
    if (!isFinite(t)) return lo;
    if (!isFinite(lo) || !isFinite(hi) || hi < lo) return t;
    const clamped = Math.max(lo, Math.min(hi, t));
    if (!(step > 0)) return clamped;
    let snapped = lo + Math.round((clamped - lo) / step) * step;
    if (snapped < lo) snapped = lo;
    if (snapped > hi) snapped = hi;
    return snapped;
  }
  function updateFlightSimPlaybackLabelsDom() {
    const label = document.getElementById('flightSimTimeLabel');
    const t = state.simTimeSec;
    if (label) label.textContent = formatSecondsToHHMMSS(t);
  }
  
  function minFirstArrivalTouchdownSecAmongFlights() {
    let minS = Infinity;
    (state.flights || []).forEach(function(f) {
      if (!f || f.arrDep === 'Dep') return;
      if (arrivalAirsideBlocked(f)) return;
      const w = getFlightAirsideWindowSec(f);
      if (!w) return;
      const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], NaN));
      if (!isFinite(eldtMin)) return;
      const eldtS = eldtMin * 60;
      if (eldtS < minS) minS = eldtS;
    });
    return (isFinite(minS) && minS < Infinity) ? minS : null;
  }
  function recomputeSimDuration() {
    let minT = Infinity;
    let maxT = -Infinity;
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      const w = getFlightAirsideWindowSec(f);
      if (!w) return;
      if (w.t0 < minT) minT = w.t0;
      if (w.t1 > maxT) maxT = w.t1;
    });
    if (!isFinite(minT) || !isFinite(maxT)) {
      minT = 0;
      maxT = 0;
    }
    let simLo = minT;
    const firstTdS = minFirstArrivalTouchdownSecAmongFlights();
    if (firstTdS != null) {
      simLo = Math.max(0, firstTdS - 10);
    }
    let durSec = Math.max(maxT, minT);
    const capAbs = state.simPlaybackEndCapSec;
    if (capAbs != null && isFinite(Number(capAbs))) {
      durSec = Math.min(durSec, Number(capAbs));
    }
    state.simDurationSec = durSec;
    if (simLo > state.simDurationSec - 1e-6) {
      simLo = Math.max(0, state.simDurationSec - 1);
    }
    state.simStartSec = simLo;
    if ((state.flights || []).length > 0 && isFinite(minT) && isFinite(maxT) && state.simDurationSec <= state.simStartSec) {
      state.simDurationSec = state.simStartSec + 1;
    }
    state.simTimeSec = Math.max(state.simStartSec, Math.min(state.simDurationSec, state.simTimeSec));
    state.simTimeSec = snapSimTimeSecForSlider(state.simTimeSec);
    const slider = document.getElementById('flightSimSlider');
    if (slider) {
      slider.min = state.simStartSec;
      slider.max = state.simDurationSec;
      slider.step = String(SIM_TIME_SLIDER_SNAP_SEC);
      slider.value = state.simTimeSec;
      if (state.simDurationSec <= state.simStartSec) slider.disabled = true;
      else slider.disabled = false;
    }
    if (typeof renderFlightSimSliderDeadlockMarkers === 'function') renderFlightSimSliderDeadlockMarkers();
    updateFlightSimPlaybackLabelsDom();
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function applySimPlaybackBarDomVisibility() {
    const wrap = document.getElementById('sim-controls-wrap');
    const inner = document.getElementById('sim-controls-container');
    const hideBtn = document.getElementById('btnHideSimPlaybackBar');
    const hasSim = state.hasSimulationResult && state.globalUpdateFresh && state.flights.length > 0;
    if (!wrap) return;
    if (!hasSim || !state.simPlaybackDockVisible) {
      wrap.style.display = 'none';
      return;
    }
    wrap.style.display = 'flex';
    if (inner) inner.style.display = 'flex';
    if (hideBtn) hideBtn.setAttribute('aria-expanded', 'true');
  }
  function syncSimulationPlaybackAfterTimelines() {
    if (typeof recomputeSimDuration === 'function') recomputeSimDuration();
    if (!state.hasSimulationResult) return;
    const simSliderAfter = document.getElementById('flightSimSlider');
    state.simTimeSec = snapSimTimeSecForSlider(Math.max(state.simStartSec, Math.min(state.simDurationSec, state.simStartSec)));
    if (simSliderAfter) simSliderAfter.value = state.simTimeSec;
    updateFlightSimPlaybackLabelsDom();
  }

  function formatTotalSecondsToHHMMSS(totalSec) {
    const parts = _splitTotalSeconds(totalSec);
    return parts.hh + ':' + parts.mm + ':' + parts.ss;
  }
  function formatMinutesToHHMMSS(minsRaw) {
    return formatTotalSecondsToHHMMSS(_normalizeTimeToSeconds(minsRaw, 'minutes', 'round'));
  }
  function flightScheduleBaseDateIso(f) {
    if (!f) return DEFAULT_SIBT_DATE;
    const raw = f.sibtDate != null ? f.sibtDate : (f.serviceDate != null ? f.serviceDate : null);
    const d = (raw == null ? '' : String(raw)).trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(d)) return d;
    return DEFAULT_SIBT_DATE;
  }
  function formatFlightScheduleDateTime(f, minsRaw) {
    const base = flightScheduleBaseDateIso(f);
    const sec = _normalizeTimeToSeconds(minsRaw, 'minutes', 'round');
    const minTotal = sec / 60;
    const ps = base.split('-');
    const Y = parseInt(ps[0], 10);
    const Mo = parseInt(ps[1], 10) - 1;
    const D = parseInt(ps[2], 10);
    if (!isFinite(Y) || !isFinite(Mo) || !isFinite(D)) return formatMinutesToHHMMSS(minsRaw);
    const t0 = new Date(Y, Mo, D, 0, 0, 0);
    t0.setMinutes(t0.getMinutes() + minTotal);
    const pad = function(n) { return (n < 10 ? '0' : '') + n; };
    return t0.getFullYear() + '-' + pad(t0.getMonth() + 1) + '-' + pad(t0.getDate()) + ' ' + pad(t0.getHours()) + ':' + pad(t0.getMinutes()) + ':' + pad(t0.getSeconds());
  }
  function formatSignedMinutesToHHMMSS(minsRaw) {
    const n = Number(minsRaw);
    if (!isFinite(n)) return '—';
    const sign = n < 0 ? '-' : '';
    return sign + formatMinutesToHHMMSS(Math.abs(n));
  }
  function formatSecondsToHHMMSS(secRaw) {
    return formatTotalSecondsToHHMMSS(_normalizeTimeToSeconds(secRaw, 'seconds', 'floor'));
  }

  function getStandBusyIntervals(standId, ignoreFlightId) {
    const intervals = [];
    if (!standId) return intervals;
    (state.flights || []).forEach(f => {
      if (!f || f.id === ignoreFlightId) return;
      if (f.arrDep !== 'Arr') return;
      if (f.standId !== standId) return;
      const win = getFlightAirsideWindowSec(f);
      if (!win) return;
      const end = win.t1;
      const dwellMin = (f.sobtMin != null && f.sibtMin != null) ? (f.sobtMin - f.sibtMin) : (f.dwellMin || 0);
      const dwellSec = Math.max(0, dwellMin * 60);
      const start = Math.max(0, end - dwellSec);
      if (end > start) intervals.push({ start, end });
    });
    intervals.sort((a, b) => a.start - b.start);
    return intervals;
  }

  function isStandOccupiedAtSimSec(standId, tSec) {
    if (!standId || !state.hasSimulationResult) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const flights = state.flights || [];
    for (let i = 0; i < flights.length; i++) {
      const f = flights[i];
      if (!f || f.standId !== standId) continue;
      const m = f.timeline_meta;
      if (m && typeof m.eibtSec === 'number' && typeof m.eobtSec === 'number') {
        if (t + 1e-3 >= m.eibtSec && t <= m.eobtSec + 1e-3) return true;
        continue;
      }
      if (f.arrDep !== 'Dep' && (f.noWayArr || f.arrRetFailed)) {
        const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
        const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
        const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
        const eibtS = eibtMin * 60;
        const eobtS = eobtMin * 60;
        if (t + 1e-3 >= eibtS && t <= eobtS + 1e-3) return true;
      }
    }
    return false;
  }

  function findStandAvailableArrivalTime(standId, desiredArrival, dwellSec) {
    let s = Math.max(0, desiredArrival);
    const intervals = getStandBusyIntervals(standId, null);
    for (let i = 0; i < intervals.length; i++) {
      const iv = intervals[i];
      if (s + dwellSec <= iv.start) return s;
      if (s < iv.end) s = iv.end;
    }
    return s;
  }

  function getTerminalForStand(stand) {
    if (!stand || !state.terminals.length) return null;
    const [px, py] = getStandConnectionPx(stand);
    let nearest = null;
    let nearestD2 = Infinity;
    for (let i = 0; i < state.terminals.length; i++) {
      const t = state.terminals[i];
      if (!t.vertices || t.vertices.length < 1) continue;
      const termPix = t.vertices.map(v => cellToPixel(v.col, v.row));
      if (t.closed && termPix.length >= 3 && pointInPolygonXY([px, py], termPix)) return t;
      let cx = 0, cy = 0;
      termPix.forEach(p => { cx += p[0]; cy += p[1]; });
      cx /= termPix.length;
      cy /= termPix.length;
      const dx = px - cx, dy = py - cy;
      const d2 = dx*dx + dy*dy;
      if (d2 < nearestD2) {
        nearestD2 = d2;
        nearest = t;
      }
    }
    return nearest;
  }

  function allStandsForFlightAssignment() {
    return (state.pbbStands || []).concat(state.remoteStands || []).concat(state.tempStands || []);
  }

  function flightCanUseStand(f, stand) {
    if (!stand) return true;
    const mode = getStandCategoryMode(stand);
    const allowedTypes = getStandAllowedAircraftTypes(stand);
    if (allowedTypes.length) {
      const flightType = String(f.aircraftType || '').trim();
      if (!flightType || allowedTypes.indexOf(flightType) < 0) return false;
    } else if (mode === 'aircraft') {
      return false;
    } else {
      const order = { A:1,B:2,C:3,D:4,E:5,F:6 };
      const fCode = String(f.code || 'C').toUpperCase()[0];
      const sCat = String(stand.category || 'F').toUpperCase()[0];
      const fc = order[fCode] || 99;
      const sc = order[sCat] || 0;
      if (fc > sc) return false;
    }
    const ft = (f.terminalId || (f.token && f.token.terminalId)) || null;
    if (!ft) return true;
    const isRemoteLike = (state.remoteStands || []).some(function(r) { return r.id === stand.id; })
      || (state.tempStands || []).some(function(r) { return r.id === stand.id; });
    if (isRemoteLike) {
      const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
      if (allowed.length) return allowed.indexOf(ft) >= 0;
    }
    const term = getTerminalForStand(stand);
    const standTermId = term ? term.id : null;
    if (!standTermId) return false;
    return ft === standTermId;
  }

  function assignStandToFlight(f, standId, segmentIdx) {
    if (!f) return false;
    if (standId) {
      const allStands = allStandsForFlightAssignment();
      const stand = allStands.find(function(s) { return s.id === standId; });
      if (!flightCanUseStand(f, stand)) {
        alert("Stand constraints or selected building do not match this aircraft, so it cannot be assigned.");
        return false;
      }
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (flightWouldOverlapStandAssignment(f, standId)) {
        alert("This stand already has an overlapping flight in the selected SIBT-SOBT window.");
        return false;
      }
    }
    const prevStandForSched = f.standId || null;
    const segIdx = segmentIdx != null && isFinite(Number(segmentIdx)) ? Math.max(0, parseInt(segmentIdx, 10) || 0) : null;
    if (segIdx != null) {
      const segs = normalizeFlightApronStaySegments(f);
      if (segIdx < segs.length) {
        segs[segIdx].standId = standId || null;
        f.apronStaySegments = segs;
      }
    } else {
      f.standId = standId;
      if (f.token) f.token.apronId = standId;
      f.arrApronId = standId || null;
      f.depApronId = standId || null;
      f.apronStaySegments = [{
        standId: standId || null,
        sibtMin: (f.sibtMin != null && isFinite(f.sibtMin)) ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : 0),
        sobtMin: (f.sobtMin != null && isFinite(f.sobtMin)) ? Number(f.sobtMin) : ((f.timeMin != null ? Number(f.timeMin) : 0) + (f.dwellMin != null ? Number(f.dwellMin) : 0))
      }];
    }
    if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    const touchedSt = [];
    if (prevStandForSched) touchedSt.push(prevStandForSched);
    if (standId) touchedSt.push(standId);
    if (typeof renderFlightList === 'function') {
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touchedSt, skipGanttRefresh: true });
    }
    if (typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
    if (typeof draw === 'function') {
      // Stand-only change: skip path graph / pro-sim / junction overlays (saves a large 2D pass; geometry unchanged).
      draw({ skipPathGeometryOverlays: true });
    }
    return true;
  }

  function flightScheduleStandWindowMinutes(f) {
    if (!f) return null;
    if (typeof syncFlightApronStayAggregate === 'function') syncFlightApronStayAggregate(f);
    const sibt = (f.sibtMin != null && isFinite(f.sibtMin)) ? Number(f.sibtMin) : (f.timeMin != null ? Number(f.timeMin) : 0);
    const dwell = (f.dwellMin != null && isFinite(f.dwellMin)) ? Number(f.dwellMin) : 0;
    const sobt = (f.sobtMin != null && isFinite(f.sobtMin)) ? Number(f.sobtMin) : (sibt + dwell);
    if (!isFinite(sibt) || !isFinite(sobt) || sobt <= sibt) return null;
    return { sibt, sobt };
  }

  function flightWouldOverlapStandAssignment(f, standId) {
    if (!f || !standId) return false;
    const win = flightScheduleStandWindowMinutes(f);
    if (!win) return false;
    return (state.flights || []).some(function(other) {
      if (!other || other === f || flightBlockedLikeNoWay(other) || other.standId !== standId) return false;
      const ow = flightScheduleStandWindowMinutes(other);
      return !!ow && win.sibt < ow.sobt && ow.sibt < win.sobt;
    });
  }

  function getCandidatePbbStandsForCode(code, flight) {
    const list = [];
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
    allStands.forEach(stand => {
      if (flight && !flightCanUseStand(flight, stand)) return;
      if (!flight && code && getStandCategoryMode(stand) === 'icao') {
        const c = String(code || '').toUpperCase()[0];
        const letters = normalizeAllowedIcaoCategories(stand.allowedIcaoCategories);
        if (letters.length && letters.indexOf(c) < 0) return;
        if (!letters.length && stand.category && String(stand.category).toUpperCase()[0] !== c) return;
      }
      const hasLink = state.apronLinks.some(lk => lk.pbbId === stand.id);
      if (!hasLink) return;
      list.push(stand);
    });
    return list;
  }

  function pickRandom(arr) {
    if (!arr.length) return null;
    const idx = Math.floor(Math.random() * arr.length);
    return arr[idx];
  }

  function resolveStand(flight) {
    const allStands = allStandsForFlightAssignment();
    if (flight.standId) {
      return allStands.find(s => s.id === flight.standId) || null;
    }
    let candidates = getCandidatePbbStandsForCode(flight.code, flight);
    if (!candidates.length) return null;
    const termId = (flight.token && flight.token.terminalId) || null;
    if (termId) {
      const filtered = candidates.filter(st => {
        const allowed = Array.isArray(st.allowedTerminals) ? st.allowedTerminals : null;
        if (allowed && allowed.length) return allowed.includes(termId);
        const t = getTerminalForStand(st);
        return t && t.id === termId;
      });
      if (filtered.length) candidates = filtered;
    }
    const stand = pickRandom(candidates);
    if (stand) flight.standId = stand.id;
    return stand;
  }

  function buildArrivalTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sibtMin = flight.sibtMin != null ? flight.sibtMin : (flight.timeMin != null ? flight.timeMin : 0);
    const baseT = sibtMin * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    const sobtMin = flight.sobtMin != null ? flight.sobtMin : (sibtMin + (flight.dwellMin != null ? flight.dwellMin : 0));
    const dwellSec = Math.max(0, (sobtMin - sibtMin) * 60);
    if (dwellSec > 0) {
      tAcc = sobtMin * 60;
      const last = timeline[timeline.length - 1];
      timeline.push({ t: tAcc, x: last.x, y: last.y });
    }
    return timeline;
  }

  function buildDepartureTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sobtMin = flight.sobtMin != null ? flight.sobtMin : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
    const baseT = sobtMin * 60;
    const v = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    const timeline = [];
    let tAcc = baseT;
    timeline.push({ t: tAcc, x: pts[0][0], y: pts[0][1] });
    for (let i = 1; i < pts.length; i++) {
      const [x1,y1] = pts[i-1];
      const [x2,y2] = pts[i];
      const len = Math.hypot(x2-x1, y2-y1);
      const dt = len / v;
      tAcc += dt;
      timeline.push({ t: tAcc, x: x2, y: y2 });
    }
    return timeline;
  }

  /**
   * Walk distM on the timeline polyline from (fx,fy) on segment segIndex.
   * forward: toward +t; !forward: toward earlier samples (e.g. rear reference from front point).
   */
  function walkTimelinePolylineFromPoint(tl, segIndex, fx, fy, distM, forward) {
    const eps = 1e-6;
    if (!tl || tl.length < 2 || !(distM > eps) || !isFinite(fx) || !isFinite(fy) || !isFinite(distM)) {
      return null;
    }
    if (segIndex < 0 || segIndex > tl.length - 2) return null;
    let rem = distM;
    let x = fx, y = fy;
    let s = segIndex;
    while (rem > eps) {
      if (forward) {
        if (s > tl.length - 2) {
          if (tl.length < 2) return { x, y };
          const n = tl.length, pa = tl[n - 2], pb = tl[n - 1];
          const bx = pb.x - pa.x, by = pb.y - pa.y;
          const bl = Math.hypot(bx, by);
          if (bl < eps) return { x, y };
          const inv = 1 / bl;
          return { x: x + bx * inv * rem, y: y + by * inv * rem };
        }
        const b = tl[s + 1];
        const ddx = b.x - x, ddy = b.y - y;
        const dlen = Math.hypot(ddx, ddy);
        if (dlen < eps) { x = b.x; y = b.y; s++; continue; }
        const step = Math.min(rem, dlen), inv = 1 / dlen;
        x += ddx * inv * step; y += ddy * inv * step; rem -= step;
        if (rem < eps) return { x, y };
        if (dlen - step < eps) { x = b.x; y = b.y; s++; }
      } else {
        if (s < 0) {
          if (tl.length < 2) return { x, y };
          const p0 = tl[0], p1 = tl[1];
          const bx = p0.x - p1.x, by = p0.y - p1.y;
          const bl = Math.hypot(bx, by);
          if (bl < eps) return { x, y };
          const inv = 1 / bl;
          return { x: x + bx * inv * rem, y: y + by * inv * rem };
        }
        const tx = tl[s].x, ty = tl[s].y;
        const ddx = tx - x, ddy = ty - y;
        const dlen = Math.hypot(ddx, ddy);
        if (dlen < eps) { x = tx; y = ty; s--; continue; }
        const step = Math.min(rem, dlen), inv = 1 / dlen;
        x += ddx * inv * step; y += ddy * inv * step; rem -= step;
        if (rem < eps) return { x, y };
        if (dlen - step < eps) { x = tx; y = ty; s--; }
      }
    }
    return { x, y };
  }

  function getFlightPositionAtTime(flight, tSec) {
    const tl = flight.timeline;
    if (!tl || !tl.length) return null;
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        return {
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u
        };
      }
    }
    return null;
  }

  function getFlightPoseAtTime(flight, tSec) {
    const tl = flight.timeline;
    if (!tl || !tl.length) return null;
    if (tl.length === 1) {
      const a = tl[0];
      if (tSec + 1e-6 < a.t || tSec - 1e-6 > a.t) return null;
      const dg = a.deadlockGhost === true;
      return { x: a.x, y: a.y, dx: 1, dy: 0, deadlockGhost: dg };
    }
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    const motionChordEps = 0.08;
    const motionChordEps2 = motionChordEps * motionChordEps;
    function segmentUnitDir(segIdx) {
      if (segIdx < 0 || segIdx > tl.length - 2) return null;
      const p = tl[segIdx], q = tl[segIdx + 1];
      const ddx = q.x - p.x, ddy = q.y - p.y;
      const l2 = ddx * ddx + ddy * ddy;
      if (l2 < motionChordEps2) return null;
      const inv = 1 / Math.sqrt(l2);
      return { dx: ddx * inv, dy: ddy * inv };
    }
    function lastMotionUnitDirBefore(i) {
      for (let j = i - 1; j >= 0; j--) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function firstMotionUnitDirFrom(startSeg) {
      for (let j = startSeg; j <= tl.length - 2; j++) {
        const u = segmentUnitDir(j);
        if (u) return u;
      }
      return null;
    }
    function headingForInterval(i) {
      const a = tl[i], b = tl[i + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const l2 = dx * dx + dy * dy;
      if (l2 >= motionChordEps2) return { dx: dx, dy: dy };
      const prev = lastMotionUnitDirBefore(i);
      if (prev) return { dx: prev.dx, dy: prev.dy };
      const next = firstMotionUnitDirFrom(i + 1);
      if (next) return { dx: next.dx, dy: next.dy };
      return { dx: 1, dy: 0 };
    }
    function frBicyclePose(R, x, y, lenM, bmin, dg) {
      if (!R || lenM <= 1e-6) return null;
      const vdx = x - R.x, vdy = y - R.y, vl = Math.hypot(vdx, vdy);
      if (vl < bmin) return null;
      return { x, y, dx: vdx / vl, dy: vdy / vl, deadlockGhost: dg };
    }
    for (let i = 0; i < tl.length - 1; i++) {
      let a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        let useI = i;
        // At a time-key at the end of [a,b], the first matching segment is the *incoming* leg.
        // The bicycle (rear on polyline) + that chord can show nose 180° off next-second motion.
        // Prefer the *outgoing* segment [b, next] (same (x,y) at t=b.t, u=0) so F/R wheels stay
        // consistent with time-forward motion. Last segment has no outgoing — keep i.
        if (i + 1 < tl.length - 1) {
          const a2 = tl[i+1], b2 = tl[i+2];
          if (a2 && b2 && b2.t > a2.t && Math.abs(tSec - b.t) < 1e-5) {
            if (Math.abs(b.t - a2.t) < 1e-5) {
              useI = i + 1;
              a = a2;
              b = b2;
            }
          }
        }
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;
        const h = headingForInterval(useI);
        const dg = !!(a.deadlockGhost || b.deadlockGhost);
        const { lenM } = getSimAircraftWorldDimsM(flight);
        const wheelBaseM = 0.55 * lenM;
        const bicycleMin = Math.max(0.15 * motionChordEps, 0.005 * lenM, 0.04);
        let out = frBicyclePose(
          walkTimelinePolylineFromPoint(tl, useI, x, y, wheelBaseM, false), x, y, lenM, bicycleMin, dg);
        if (!out) {
          out = { x, y, dx: h.dx, dy: h.dy, deadlockGhost: dg };
        }
        return out;
      }
    }
    return null;
  }

  /**
   * After EOBT, while on apron_link (departure push/taxi) only: if the bicycle nose points with
   * the ground track step (nose . track &gt; 0), flip dx/dy 180&deg; so the silhouette shows
   * towed/reverse (retro) like R3, without changing (x,y) or the underlying bicycle trace.
   * Does not run before EObT, not off apron, not Arr_taxi, and does not change already-retro
   * pose. Other flights/pathTypes unchanged.
   */
  function applyEobtApronDepTaxiPushbackNoseIfNeeded(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const m = flight.timeline_meta;
    if (!m || typeof m.eobtSec !== 'number' || !isFinite(m.eobtSec)) return pose;
    if (tSec + 1e-3 < m.eobtSec) return pose;
    const tl = flight.timeline;
    if (!tl || !tl.length) return pose;
    const tKey = Math.round(Number(tSec));
    const byT = Object.create(null);
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
    }
    const cur = byT[tKey];
    if (!cur) return pose;
    const ph = String(cur.phase || '');
    if (ph !== 'Pushback') return pose;
    const prev = byT[tKey - 1];
    if (!prev) return pose;
    const ddx = cur.x - prev.x, ddy = cur.y - prev.y;
    const dlen = Math.hypot(ddx, ddy);
    if (dlen < 1e-9) return pose;
    const ux = ddx / dlen, uy = ddy / dlen;
    const pl = Math.hypot(pose.dx, pose.dy);
    if (pl < 1e-9) return pose;
    const px = pose.dx / pl, py = pose.dy / pl;
    const dotU = px * ux + py * uy;
    if (dotU <= 0.05) return pose;
    return { x: pose.x, y: pose.y, dx: -pose.dx, dy: -pose.dy, deadlockGhost: !!pose.deadlockGhost };
  }

  /**
   * Pushback tail-first motion: no bicycle model. Fuselage nose=0, tail=100; path
   * sample is station 70 (70% nose→tail). Draw anchor (≈10% aft of nose) at C + h * (0.70−0.1) * lenM
   * with h = unit nose from pose (after applyEobt). Forward taxi leaves pose unchanged.
   */
  function applyApronLinkDepReverseFuselageStation75PoseIfNeeded(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const m = flight.timeline_meta;
    if (!m || typeof m.eobtSec !== 'number' || !isFinite(m.eobtSec)) return pose;
    const t = Number(tSec);
    if (!isFinite(t) || t + 1e-3 < m.eobtSec) return pose;
    const tl = flight.timeline;
    if (!tl || !tl.length) return pose;
    const tKey = Math.round(t);
    const byT = Object.create(null);
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
    }
    const cur = byT[tKey];
    if (!cur) return pose;
    const ph = String(cur.phase || '');
    if (ph !== 'Pushback') return pose;
    let a = null;
    let b = null;
    for (let i = 0; i < tl.length - 1; i++) {
      const p = tl[i];
      const q = tl[i + 1];
      if (t + 1e-9 >= p.t && t - 1e-9 <= q.t) {
        a = p;
        b = q;
        break;
      }
    }
    if (!a || !b) return pose;
    const ddx = b.x - a.x;
    const ddy = b.y - a.y;
    const segLen = Math.hypot(ddx, ddy);
    if (segLen < 0.08) return pose;
    const vx = ddx / segLen;
    const vy = ddy / segLen;
    const pl = Math.hypot(pose.dx, pose.dy);
    if (pl < 1e-9) return pose;
    const hx = pose.dx / pl;
    const hy = pose.dy / pl;
    if (hx * vx + hy * vy > -0.05) return pose;
    const C = getFlightPositionAtTime(flight, t);
    if (!C) return pose;
    const { lenM } = getSimAircraftWorldDimsM(flight);
    const NOSE_TO_STATION75_FRAC = 0.70;
    const NOSE_TO_FRONT_WHEEL_FRAC = 0.1;
    const alongNoseM = (NOSE_TO_STATION75_FRAC - NOSE_TO_FRONT_WHEEL_FRAC) * lenM;
    return {
      x: C.x + hx * alongNoseM,
      y: C.y + hy * alongNoseM,
      dx: pose.dx,
      dy: pose.dy,
      deadlockGhost: !!pose.deadlockGhost,
    };
  }

  function getPushbackRearWheelOnPathPoseForDraw(flight, tSec, pose) {
    if (!pose || !flight) return pose;
    const tl = flight.timeline;
    if (!tl || tl.length < 2) return pose;
    const t = Number(tSec);
    if (!isFinite(t)) return pose;
    const tKey = Math.round(t);
    const byT = Object.create(null);
    let transitionStartT = null;
    for (let i = 0; i < tl.length; i++) {
      const w = tl[i];
      if (!w) continue;
      const tt = Math.round(Number(w.t));
      if (isFinite(tt)) byT[tt] = w;
      if (i > 0 && String(w.phase || '') === 'Dep_taxi' && String(tl[i - 1].phase || '') === 'Pushback') {
        transitionStartT = Number(w.t);
      }
    }
    const curPhase = String((byT[tKey] && byT[tKey].phase) || '');
    const prevPhase = String((byT[tKey - 1] && byT[tKey - 1].phase) || '');
    const PUSHBACK_TO_DEP_TAXI_BLEND_SEC = 1.0;
    const inBlend = transitionStartT != null
      && t + 1e-9 >= transitionStartT
      && t <= transitionStartT + PUSHBACK_TO_DEP_TAXI_BLEND_SEC + 1e-9;
    const inPushback = curPhase === 'Pushback' || (curPhase === 'Dep_taxi' && prevPhase === 'Pushback') || inBlend;
    if (!inPushback) return pose;
    let segIdx = -1;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (t + 1e-9 >= a.t && t - 1e-9 <= b.t) {
        segIdx = i;
        if (i + 1 < tl.length - 1 && Math.abs(t - b.t) < 1e-5) {
          const n = tl[i + 1];
          const nn = tl[i + 2];
          if (n && nn && String(n.phase || '') === 'Pushback' && String(nn.phase || '') === 'Pushback') segIdx = i + 1;
        }
        break;
      }
    }
    if (segIdx < 0) return pose;
    const { lenM } = getSimAircraftWorldDimsM(flight);
    const wheelBaseM = 0.55 * lenM;
    const rear = walkPushbackPolylineFromFront(tl, segIdx, pose.x, pose.y, wheelBaseM);
    if (!rear) return pose;
    const dx = pose.x - rear.x;
    const dy = pose.y - rear.y;
    const dl = Math.hypot(dx, dy);
    if (dl < Math.max(0.005 * lenM, 0.04)) return pose;
    const pushPose = { x: pose.x, y: pose.y, dx: dx / dl, dy: dy / dl, deadlockGhost: !!pose.deadlockGhost };
    if (!inBlend || transitionStartT == null || curPhase === 'Pushback') return pushPose;
    const alpha = Math.max(0, Math.min(1, (t - transitionStartT) / PUSHBACK_TO_DEP_TAXI_BLEND_SEC));
    return blendPoseHeading(pushPose, pose, alpha);
  }

  function blendPoseHeading(fromPose, toPose, alpha) {
    if (!fromPose || !toPose) return fromPose || toPose || null;
    const a = Math.max(0, Math.min(1, Number(alpha) || 0));
    const a0 = Math.atan2(fromPose.dy, fromPose.dx);
    const a1 = Math.atan2(toPose.dy, toPose.dx);
    let da = a1 - a0;
    while (da > Math.PI) da -= Math.PI * 2;
    while (da < -Math.PI) da += Math.PI * 2;
    const th = a0 + da * a;
    return {
      x: toPose.x,
      y: toPose.y,
      dx: Math.cos(th),
      dy: Math.sin(th),
      deadlockGhost: !!(fromPose.deadlockGhost || toPose.deadlockGhost),
    };
  }

  function walkPushbackPolylineFromFront(tl, segIndex, fx, fy, distM) {
    const eps = 1e-6;
    const motionEps = 0.08;
    if (!tl || tl.length < 2 || !(distM > eps)) return null;
    let rem = distM;
    let x = fx, y = fy;
    let s = segIndex;
    let lastUx = null;
    let lastUy = null;
    while (rem > eps && s <= tl.length - 2) {
      const a = tl[s];
      const b = tl[s + 1];
      if (String(a.phase || '') !== 'Pushback' || String(b.phase || '') !== 'Pushback') break;
      const ddx = b.x - x;
      const ddy = b.y - y;
      const dlen = Math.hypot(ddx, ddy);
      if (dlen < eps) {
        const sx = b.x - a.x;
        const sy = b.y - a.y;
        const sl = Math.hypot(sx, sy);
        if (sl > motionEps) {
          lastUx = sx / sl;
          lastUy = sy / sl;
        }
        x = b.x;
        y = b.y;
        s++;
        continue;
      }
      const ux = ddx / dlen;
      const uy = ddy / dlen;
      if (dlen > motionEps) {
        lastUx = ux;
        lastUy = uy;
      }
      const step = Math.min(rem, dlen);
      x += ux * step;
      y += uy * step;
      rem -= step;
      if (rem < eps) return { x, y };
      if (dlen - step < eps) {
        x = b.x;
        y = b.y;
        s++;
      }
    }
    if (lastUx == null || lastUy == null) {
      for (let j = Math.min(segIndex, tl.length - 2); j >= 0; j--) {
        const a = tl[j];
        const b = tl[j + 1];
        if (String(a.phase || '') !== 'Pushback' || String(b.phase || '') !== 'Pushback') continue;
        const sx = b.x - a.x;
        const sy = b.y - a.y;
        const sl = Math.hypot(sx, sy);
        if (sl > motionEps) {
          lastUx = sx / sl;
          lastUy = sy / sl;
          break;
        }
      }
    }
    if (lastUx == null || lastUy == null) return null;
    return { x: x + lastUx * rem, y: y + lastUy * rem };
  }

  function getFlightPoseAtTimeForDraw(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || !tl.length) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    return getPushbackRearWheelOnPathPoseForDraw(flight, t, getFlightPoseAtTime(flight, t));
  }

  function isFlightPreTouchdownForDraw(f, tSec) {
    if (!PRE_TOUCHDOWN_HALO_ENABLED) return false;
    if (!f || f.arrDep === 'Dep') return false;
    const m = f.timeline_meta;
    if (!m || typeof m.eldtSec !== 'number' || !isFinite(m.eldtSec)) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    return t < m.eldtSec - 1e-3;
  }

  function isFlightAirsideCycleCompleteAtSimTime(f, tSec) {
    const m = f && f.timeline_meta;
    const t = Number(tSec);
    if (!isFinite(t) || !m || m.error) return false;
    if (typeof m.etotSec !== 'number' || !isFinite(m.etotSec)) return false;
    return t >= m.etotSec - 1e-3;
  }

  
  function isFlightTimelineStationaryAtSimTime(f, tSec) {
    const tl = f && f.timeline;
    if (!tl || tl.length < 2) return false;
    const t = Number(tSec);
    if (!isFinite(t)) return false;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t < t0 - 1e-9 || t > t1 + 1e-9) return false;
    const stillEps = 0.08;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i + 1];
      if (!(t + 1e-9 >= a.t && t - 1e-9 <= b.t)) continue;
      const dt = b.t - a.t;
      if (dt < 1e-9) continue;
      const dist = Math.hypot(b.x - a.x, b.y - a.y);
      if (dist < stillEps) return true;
    }
    return false;
  }

  function isFlightTrailHiddenAtSimTime(f, tSec) {
    if (isFlightAirsideCycleCompleteAtSimTime(f, tSec)) return true;
    if (isFlightTimelineStationaryAtSimTime(f, tSec)) return true;
    return false;
  }

  function getFlightTrailPolylineBackward(f, tEnd, maxDistM) {
    const tl = f && f.timeline;
    if (!tl || tl.length < 2 || !(maxDistM > 0)) return [];
    const tMin = tl[0].t, tMax = tl[tl.length - 1].t;
    let t = Math.min(Math.max(tEnd, tMin), tMax);
    let seg = 0;
    for (let i = 0; i < tl.length - 1; i++) {
      if (t >= tl[i].t && t <= tl[i + 1].t) { seg = i; break; }
      if (t > tl[i + 1].t) seg = i;
    }
    const pts = [];
    function xyAt(T) {
      if (T <= tMin) return [tl[0].x, tl[0].y];
      if (T >= tMax) return [tl[tl.length - 1].x, tl[tl.length - 1].y];
      for (let i = 0; i < tl.length - 1; i++) {
        const a = tl[i], b = tl[i + 1];
        if (T >= a.t && T <= b.t) {
          const sp = b.t - a.t || 1;
          const uu = (T - a.t) / sp;
          return [a.x + (b.x - a.x) * uu, a.y + (b.y - a.y) * uu];
        }
      }
      return [tl[tl.length - 1].x, tl[tl.length - 1].y];
    }
    pts.push(xyAt(t));
    let rem = maxDistM;
    let curSeg = seg;
    let curT = t;
    let guard = 0;
    while (rem > 1e-6 && curSeg >= 0 && guard++ < 10000) {
      const A = tl[curSeg], B = tl[curSeg + 1];
      const ta = A.t, tb = B.t;
      const dt = tb - ta || 1e-12;
      const distAB = Math.hypot(B.x - A.x, B.y - A.y) || 1e-12;
      let u = Math.max(0, Math.min(1, (curT - ta) / dt));
      if (u < 1e-12) {
        if (curSeg <= 0) break;
        curSeg--;
        curT = tl[curSeg + 1].t;
        continue;
      }
      const distToA = u * distAB;
      if (distToA <= rem) {
        rem -= distToA;
        pts.push([A.x, A.y]);
        curSeg--;
        curT = ta;
      } else {
        const frac = rem / distAB;
        const uu = u - frac;
        const nx = A.x + uu * (B.x - A.x);
        const ny = A.y + uu * (B.y - A.y);
        pts.push([nx, ny]);
        rem = 0;
        break;
      }
    }
    return pts.slice().reverse();
  }

  function getRunwayOptions() {
    const list = [];
    (state.taxiways || []).filter(t => t.pathType === 'runway')
      .forEach(t => list.push({ id: t.id, name: (t.name || '').trim() || 'Runway' }));
    return list;
  }

  function buildRunwayOptionsHtml(selectedId) {
    const opts = [];
    const list = getRunwayOptions();
    if (!list.length) {
      opts.push('<option value=\"\">Runway</option>');
    } else {
      list.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Runway') + '</option>');
      });
    }
    return opts.join('');
  }
  function buildTerminalOptionsHtml(selectedId) {
    const opts = [];
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) {
      return { id: t.id, name: (t.name || '').trim() || 'Building' };
    });
    if (!terms.length) {
      opts.push('<option value=\"\">Building</option>');
    } else {
      if (terms.length > 1) opts.push('<option value=\"\">Random</option>');
      terms.forEach(function(o) {
        const sel = selectedId && o.id === selectedId ? ' selected' : '';
        opts.push('<option value=\"' + String(o.id || '').replace(/\"/g, '&quot;') + '\"' + sel + '>' +
          escapeHtml(o.name || o.id || 'Building') + '</option>');
      });
    }
    return opts.join('');
  }
  function resolveRunwayIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const list = getRunwayOptions();
    for (let i = 0; i < list.length; i++) {
      if (list[i].id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < list.length; i++) {
      if (String(list[i].name || '').trim().toLowerCase() === vl) return list[i].id;
    }
    return undefined;
  }
  function resolveTerminalIdFromInput(raw) {
    const v = (raw || '').trim();
    if (!v) return null;
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (t.id === v) return v;
    }
    const vl = v.toLowerCase();
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      if (String(t.name || '').trim().toLowerCase() === vl) return t.id;
    }
    return undefined;
  }
  function syncFlightAssignInputDisplay(el, f) {
    const role = el.getAttribute('data-role');
    if (role === 'arr') el.value = resolveArrivalRunwayIdForFlight(f) || '';
    else if (role === 'term' || role === 'arrterm') el.value = resolveFlightArrTerminalId(f) || '';
    else if (role === 'depterm') el.value = resolveFlightDepTerminalId(f) || '';
    else if (role === 'dep') el.value = f.depRunwayId || (f.token && f.token.depRunwayId) || '';
    else if (role === 'intdom') el.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
  }
  function getRunwayDisplayLabelById(rwId) {
    if (rwId == null || rwId === '') return '—';
    const list = getRunwayOptions();
    const o = list.find(function(x) { return x.id === rwId; });
    return o ? (o.name || o.id || 'Runway') : '—';
  }
  function getTerminalDisplayLabelById(termId) {
    if (termId == null || termId === '') return '—';
    const terms = makeUniqueNamedCopy(state.terminals || [], 'name');
    const t = terms.find(function(x) { return x.id === termId; });
    return t ? ((t.name || '').trim() || 'Building') : '—';
  }
  function resolveFlightBaseTerminalId(f) {
    if (!f) return null;
    return f.terminalId || (f.token && f.token.terminalId) || null;
  }
  function resolveFlightArrTerminalId(f) {
    if (!f) return null;
    return f.arrTerminalId || (f.token && f.token.arrTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function resolveFlightDepTerminalId(f) {
    if (!f) return null;
    return f.depTerminalId || (f.token && f.token.depTerminalId) || resolveFlightBaseTerminalId(f);
  }
  function ensureFlightSplitTerminalDefaults(f) {
    if (!f) return;
    const base = resolveFlightBaseTerminalId(f);
    if (!f.arrTerminalId && base) f.arrTerminalId = base;
    if (!f.depTerminalId && base) f.depTerminalId = base;
    if (f.token) {
      if (!f.token.arrTerminalId && f.arrTerminalId) f.token.arrTerminalId = f.arrTerminalId;
      if (!f.token.depTerminalId && f.depTerminalId) f.token.depTerminalId = f.depTerminalId;
    }
  }
  function flightColorGroupKeyForSim(f, mode) {
    if (mode === 'all') return '*';
    if (mode === 'airline') return 'a:' + (String(f.airlineCode || '').trim() || '—');
    if (mode === 'icao') {
      const c0 = (typeof getCodeForAircraft === 'function') ? String(getCodeForAircraft(f.aircraftType) || 'C').trim().toUpperCase()[0] : 'C';
      return 'i:' + (c0 || 'C');
    }
    if (mode === 'intdom') {
      return 'd:' + ((String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int');
    }
    if (mode === 'building') {
      const tid = f.terminalId || (f.token && f.token.terminalId) || '';
      return 'b:' + (tid ? getTerminalDisplayLabelById(tid) : '—');
    }
    return '*';
  }
  function buildFlightSim2DColorKeyIndexMap() {
    const mode = state.flightColorMode || 'all';
    if (mode === 'all') return new Map([['*', 0]]);
    const flights = state.flights || [];
    const keys = new Set();
    for (let i = 0; i < flights.length; i++) {
      if (!flights[i]) continue;
      keys.add(flightColorGroupKeyForSim(flights[i], mode));
    }
    const sorted = Array.from(keys).sort();
    const m = new Map();
    for (let j = 0; j < sorted.length; j++) m.set(sorted[j], j);
    return m;
  }
  function resolveFlightSim2DGlyphFillRgba(f, isDeadlockGhost, keyIdxMap, pal, overflow, mode) {
    if (isDeadlockGhost) return 'rgba(148, 163, 184, 0.45)';
    if (mode === 'all') return apron2DGlyphFill();
    const k = flightColorGroupKeyForSim(f, mode);
    const idx = keyIdxMap.get(k);
    if (idx == null || idx >= 10) return overflow;
    return pal[idx] || overflow;
  }
  function parseCssColorToRgbOptional(css) {
    const s = String(css || '').trim();
    const hex6 = s.match(/^#([0-9a-fA-F]{6})$/);
    if (hex6) {
      const h = hex6[1];
      return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
    }
    const rgba = s.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/);
    if (rgba) return { r: +rgba[1], g: +rgba[2], b: +rgba[3] };
    return null;
  }
  /** Trail stroke gradient: same hue as aircraft fill, fading to transparent along the tail. */
  function simFlightTrailGradientFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return { near: c2dSimFlightTrailStroke(), far: c2dSimFlightTrailStrokeEnd() };
    }
    return {
      near: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.96)',
      far: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0)',
    };
  }
  /** Pre-TD ring: same hue as fill, with soft fill + stroke + glow. */
  function simPreTouchdownHaloFromFillCss(fillCss) {
    const rgb = parseCssColorToRgbOptional(fillCss);
    if (!rgb) {
      return {
        fill: c2dSimPreTouchdownHaloFill(),
        stroke: c2dSimPreTouchdownHaloStroke(),
        shadow: c2dSimPreTouchdownHaloStroke(),
      };
    }
    return {
      fill: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.18)',
      stroke: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.92)',
      shadow: 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.55)',
    };
  }
  function syncFlightAssignStripFromFlight(f) {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    if (f) ensureFlightSplitTerminalDefaults(f);
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
    }
    if (intDomEl) {
      intDomEl.value = (f && String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    }
    if (arrTermEl) {
      const tid = f ? (resolveFlightArrTerminalId(f) || '') : '';
      arrTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      arrTermEl.value = tid;
    }
    if (depTermEl) {
      const tid = f ? (resolveFlightDepTerminalId(f) || '') : '';
      depTermEl.innerHTML = buildTerminalOptionsHtml(tid);
      depTermEl.value = tid;
    }
    if (depEl) {
      const did = f ? (f.depRunwayId || (f.token && f.token.depRunwayId) || '') : '';
      depEl.innerHTML = buildRunwayOptionsHtml(did);
      depEl.value = did;
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const arrTermEl = document.getElementById('flightAssignStripArrTerm');
    const depTermEl = document.getElementById('flightAssignStripDepTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const intDomEl = document.getElementById('flightAssignStripIntDom');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, arrTermEl, depTermEl, depEl, intDomEl].forEach(function(el) {
      if (el) el.disabled = dis;
    });
    if (!f) {
      syncFlightAssignStripFromFlight(null);
      return;
    }
    syncFlightAssignStripFromFlight(f);
  }
  function commitFlightAssign(role, flightId, rawValue, st, listEl) {
    const f = st.flights.find(function(x) { return x.id === flightId; });
    if (!f) return;
    const raw = rawValue;
    if (role === 'intdom') {
      const next = (String(raw || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      const prev = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
      if (next === prev) return;
      f.intDom = next;
      syncFlightAssignStripFromFlight(f);
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      if (typeof draw === 'function') draw();
      if (typeof renderFlightList === 'function')
        renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: f.standId ? [f.standId] : [] });
      return;
    }
    var val = null;
    if (role === 'arr' || role === 'dep') {
      const r = resolveRunwayIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else if (role === 'term' || role === 'arrterm' || role === 'depterm') {
      const r = resolveTerminalIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else return;
    var prevArr = f.arrRunwayId || null;
    var prevDep = f.depRunwayId || (f.token && f.token.depRunwayId) || null;
    var prevArrTerm = resolveFlightArrTerminalId(f) || null;
    var prevDepTerm = resolveFlightDepTerminalId(f) || null;
    if (role === 'arr' && val === prevArr) return;
    if (role === 'dep' && val === prevDep) return;
    if ((role === 'term' || role === 'arrterm') && val === prevArrTerm) return;
    if (role === 'depterm' && val === prevDepTerm) return;
    var prevStand = f.standId || null;
    if (!f.token) f.token = { nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null, arrTerminalId: null, depTerminalId: null };
    if (role === 'arr') {
      f.arrRunwayId = val;
      f.token.runwayId = val;
    } else if (role === 'term' || role === 'arrterm') {
      f.arrTerminalId = val;
      f.token.arrTerminalId = val;
      if (!f.depTerminalId) {
        f.depTerminalId = val;
        f.token.depTerminalId = val;
      }
      f.terminalId = val;
      f.token.terminalId = val;
    } else if (role === 'depterm') {
      f.depTerminalId = val;
      f.token.depTerminalId = val;
      if (!f.arrTerminalId) {
        f.arrTerminalId = val;
        f.token.arrTerminalId = val;
      }
      f.terminalId = f.arrTerminalId || val;
      f.token.terminalId = f.terminalId || null;
    } else if (role === 'dep') {
      f.depRunwayId = val;
      f.token.depRunwayId = val;
    }
    syncFlightAssignStripFromFlight(f);
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    var touched = [];
    if (prevStand) touched.push(prevStand);
    if (f.standId) touched.push(f.standId);
    if (typeof renderFlightList === 'function')
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [flightId], touchedStandIds: touched });
  }
  function commitFlightAssignField(el, st, listEl) {
    const idVal = el.getAttribute('data-id');
    const role = el.getAttribute('data-role');
    commitFlightAssign(role, idVal, el.value, st, listEl);
  }
  function commitFlightAssignFromStrip(el, st, listEl) {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.id) return;
    const role = el.getAttribute('data-role');
    if (!role) return;
    commitFlightAssign(role, sel.id, el.value, st, listEl);
  }

  /** Flight schedule dynamic AP columns: 10 fixed cells, AP cells, Dep Rw, then S/E groups. */
  const FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT = 10;
  const FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT = 7;
  function flightScheduleLogicalSegmentCount(f) {
    if (!f) return 1;
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    if (!segs.length) return 1;
    let n = 0;
    let prev = null;
    for (let i = 0; i < segs.length; i++) {
      const sid = segs[i].standId != null ? String(segs[i].standId) : '';
      if (i === 0 || sid !== prev) n++;
      prev = sid;
    }
    return Math.max(1, n);
  }
  function flightScheduleColumnK() {
    const flights = state.flights || [];
    let k = 1;
    for (let i = 0; i < flights.length; i++) k = Math.max(k, flightScheduleLogicalSegmentCount(flights[i]));
    return k;
  }
  function flightSchedColIndex(field, k) {
    const n = Math.max(1, Number(k) || flightScheduleColumnK());
    const apStart = FLIGHT_SCHED_FIXED_BEFORE_AP_COL_COUNT;
    const base = apStart + n + 1;
    if (field === 'ap') return apStart;
    if (field === 'depRunway') return apStart + n;
    if (field === 'sibt') return base;
    if (field === 'sobt') return base + 1;
    if (field === 'eldt') return base + n * 2;
    if (field === 'eibt') return base + n * 2 + 1;
    if (field === 'eobt') return base + n * 2 + 2;
    if (field === 'etot') return base + n * 4 + 1;
    if (field === 'metrics') return base + n * 4 + 2;
    return base;
  }
  function flightScheduleTableColCount(k) {
    return flightSchedColIndex('metrics', k) + FLIGHT_SCHED_TRAILING_METRIC_COL_COUNT + 1;
  }
  /** Backward-compatible aliases for N=1 call sites. Dynamic code should use `flightSchedColIndex`. */
  const FLIGHT_SCHED_TD_SIBT = 12;
  const FLIGHT_SCHED_TD_SOBT = 13;
  const FLIGHT_SCHED_TD_ELDT = 14;
  const FLIGHT_SCHED_TD_EIBT = 15;
  const FLIGHT_SCHED_TD_EOBT = 16;
  const FLIGHT_SCHED_TD_ETOT = 17;
  function ensureFlightAssignStripWired() {
    if (window.__flightAssignStripWired) return;
    const wrap = document.getElementById('flightAssignStrip');
    if (!wrap) return;
    window.__flightAssignStripWired = true;
    wrap.querySelectorAll('.flight-assign-strip-select').forEach(function(inp) {
      inp.addEventListener('change', function(ev) {
        const listEl = document.getElementById('flightList');
        const el = ev.target;
        commitFlightAssignFromStrip(el, state, listEl);
      });
    });
  }

  function _flightListPaintVirtualSlice(listEl) {
    const vs = listEl._flightVirtState;
    if (!vs) return;
    const tbody = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"] tbody');
    if (!tbody) return;
    const flightsSorted = vs.flightsSorted;
    const retStatsAll = vs.retStatsAll;
    const total = flightsSorted.length;
    const rowH = vs.rowH;
    const overscan = vs.overscan;
    const scrollTop = listEl.scrollTop || 0;
    const vh = listEl.clientHeight || 418;
    const start = Math.max(0, Math.floor(scrollTop / rowH) - overscan);
    const rowCount = Math.ceil(vh / rowH) + overscan * 2 + 2;
    const end = Math.min(total, start + rowCount);
    const topPad = start * rowH;
    const botPad = Math.max(0, (total - end) * rowH);
    const parts = [];
    const colCount = flightScheduleTableColCount(vs.apronK || flightScheduleColumnK());
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + topPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    for (let i = start; i < end; i++) {
      parts.push(_buildFlightListRowHtml(flightsSorted[i], retStatsAll, vs.apronK));
    }
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + botPad + 'px\"><td colspan=\"' + colCount + '\"></td></tr>');
    tbody.innerHTML = parts.join('');
    _flightListWireEvents(listEl, state);
  }
  function _flightListTeardownVirtual(listEl) {
    listEl._flightVirtState = null;
  }
  function _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK) {
    const prevScroll = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]') ? (listEl.scrollTop || 0) : 0;
    listEl._flightVirtState = {
      flightsSorted: flightsSorted,
      retStatsAll: retStatsAll,
      rowH: DOM_OPT_FLIGHT_VIRT_ROW_H,
      overscan: DOM_OPT_FLIGHT_VIRT_OVERSCAN,
      apronK: apronK,
      raf: null
    };
    listEl.innerHTML = headerRow + '</tbody></table>';
    const tbl = listEl.querySelector('.flight-schedule-table');
    if (tbl) tbl.setAttribute('data-virtual-table', '1');
    _flightListPaintVirtualSlice(listEl);
    if (prevScroll > 0) listEl.scrollTop = prevScroll;
    if (!listEl._flightVirtScrollBound) {
      listEl._flightVirtScrollBound = true;
      listEl.addEventListener('scroll', function() {
        const vs = listEl._flightVirtState;
        if (!vs || !listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]')) return;
        if (vs.raf) cancelAnimationFrame(vs.raf);
        vs.raf = requestAnimationFrame(function() {
          vs.raf = null;
          _flightListPaintVirtualSlice(listEl);
        });
      });
    }
  }

  function bumpVttArrCacheRev() {
    state.vttArrCacheRev = (state.vttArrCacheRev | 0) + 1;
    bumpRwySepSnapshotStaleGen();
  }
  function getBaseVttArrMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getArrRotMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  function getBaseVttDepMinutes(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getBaseVttDepMinutesToLineup(f) {
    if (!f) return 0;
    return 0;
  }
  
  function getDepBlockOutMin(f) {
    const taxi = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
    const rollBundleSec = (typeof computeDepRollAndLineupOnlySec === 'function')
      ? computeDepRollAndLineupOnlySec(f)
      : (DEP_LINEUP_HOLD_SEC + takeoffRollSecForRunwayTailLenM(0, DEP_TAKEOFF_ACCEL_SMALL_MS2));
    return taxi + rollBundleSec / 60;
  }
  
  function getNormalizedStandDwellBounds(f) {
    let dwell = f.dwellMin != null ? f.dwellMin : 0;
    let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
    dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
    minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
    if (minDwell > dwell) minDwell = dwell;
    return { dwell, minDwell };
  }

  /**
   * Apron Gantt SIBT handle: if dwell can shrink (dwell > minDwell), fix SOBT at drag anchor and resize dwell;
   * EIBT shifts by the same Δ as SIBT. If already at min dwell, translate the S block and nudge EOBT/ETOT by Δ.
   */
  function _ganttApplySibtHandleSnappedMinutes(f, mSnapped, dragCtx) {
    if (!f || !dragCtx || flightBlockedLikeNoWay(f)) return false;
    const mClamped = Math.max(0, Number(mSnapped));
    if (!isFinite(mClamped)) return false;
    const anchor = dragCtx.anchorSobt;
    const startS = dragCtx.startSibt;
    const minD = dragCtx.minDwell0;
    const d0 = dragCtx.dwell0;
    if (!(typeof anchor === 'number' && isFinite(anchor)) || !(typeof startS === 'number' && isFinite(startS))) return false;
    const atMinDwell = !(d0 > minD + 1e-9);
    if (atMinDwell) {
      if (typeof applyScheduledGateTimingFromSField === 'function') applyScheduledGateTimingFromSField(f, 'sibt', mClamped);
      const ds = mClamped - startS;
      if (dragCtx.startEobt != null && isFinite(dragCtx.startEobt)) f.eobtMin = dragCtx.startEobt + ds;
      if (dragCtx.startEtot != null && isFinite(dragCtx.startEtot)) f.etotMin = dragCtx.startEtot + ds;
      return true;
    }
    let newDwell = anchor - mClamped;
    let sibtU = mClamped;
    if (newDwell < minD) {
      newDwell = minD;
      sibtU = anchor - minD;
    }
    f.timeMin = sibtU;
    f.sibtMin = sibtU;
    f.sldtMin = scheduledSldtFromSibtMinutes(f, sibtU);
    f.sobtMin = anchor;
    f.dwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, newDwell);
    if (f.minDwellMin != null) {
      f.minDwellMin = Math.max(SCHED_DWELL_FLOOR_MIN, Math.min(f.dwellMin, f.minDwellMin));
    }
    f.stotMin = scheduledStotFromSobtMinutes(f, anchor);
    const deibt = sibtU - startS;
    if (dragCtx.startEibt != null && isFinite(dragCtx.startEibt)) f.eibtMin = dragCtx.startEibt + deibt;
    return true;
  }

  function applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotRunwayCandidateMin) {
    if (!f) return;
    const eibt = eibtMin != null && isFinite(eibtMin) ? eibtMin : 0;
    const block = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const { dwell, minDwell } = getNormalizedStandDwellBounds(f);
    const low = eibt + minDwell;
    const high = eibt + dwell;
    const sobtPref = (f.sobtMin != null)
      ? f.sobtMin
      : (f.sibtMin != null
        ? f.sibtMin + dwell
        : (f.timeMin != null ? f.timeMin + dwell : low));
    const eobt = Math.min(Math.max(sobtPref, low), high);
    const etotDraft = eobt + block;
    let etot = etotDraft;
    if (etotRunwayCandidateMin != null && isFinite(etotRunwayCandidateMin)) {
      etot = Math.max(etotRunwayCandidateMin, etotDraft);
    }
    f.eobtMin = eobt;
    f.etotMin = etot;
    f.depTaxiDelayMin = Math.max(0, etot - etotDraft);
  }

  function pinEarliestEldtToSldtPerRunway(flights) {
    void flights;
  }

  var __schedRetStatsBatchActive = false;
  var __schedRetStatsCached = null;
  var __schedRetExitDistSig = '';
  var __schedRetExitDistMemo = null;
  function scheduleRetExitDistLayoutSig() {
    const tws = state.taxiways || [];
    const parts = [];
    for (let i = 0; i < tws.length; i++) {
      const t = tws[i];
      if (!t || (t.pathType !== 'runway' && t.pathType !== 'runway_exit')) continue;
      let line = String(t.id) + '\x1e' + String(t.pathType) + '\x1e' + JSON.stringify(t.vertices || []);
      if (t.pathType === 'runway' && typeof getTaxiwayDirection === 'function') {
        line += '\x1e' + String(getTaxiwayDirection(t));
      }
      if (t.pathType === 'runway_exit') {
        line += '\x1e' + JSON.stringify(t.allowedRwDirections || []);
        if (typeof getTaxiwayDirection === 'function') {
          line += '\x1e' + String(getTaxiwayDirection(t));
        }
      }
      parts.push(line);
    }
    parts.sort();
    return parts.join('\x1f') + '\x1e' + 'arrivalRetPathEdgeF1V1';
  }
  function bumpScheduleRetExitDistCache() {
    __schedRetExitDistSig = '';
    __schedRetExitDistMemo = null;
  }
  function beginScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = true;
    __schedRetStatsCached = null;
  }
  function endScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = false;
    if (__schedRetStatsCached != null) {
      const sig = scheduleRetExitDistLayoutSig();
      __schedRetExitDistSig = sig;
      __schedRetExitDistMemo = __schedRetStatsCached;
    }
    __schedRetStatsCached = null;
  }
  function getScheduleRetStatsAll() {
    if (__schedRetStatsBatchActive) {
      if (__schedRetStatsCached === null) {
        __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
      }
      return __schedRetStatsCached;
    }
    const sig = scheduleRetExitDistLayoutSig();
    if (sig === __schedRetExitDistSig && __schedRetExitDistMemo && Array.isArray(__schedRetExitDistMemo)) {
      return __schedRetExitDistMemo;
    }
    const res = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
    __schedRetExitDistSig = sig;
    __schedRetExitDistMemo = res;
    return res;
  }

  function warmFlightPathsForSchedule(flights) {
    void flights;
  }

  function warmPathsEnsureArrRetRot(flights, forceResampleRet) {
    warmFlightPathsForSchedule(flights);
    return (typeof ensureArrRetRotSampled === 'function')
      ? ensureArrRetRotSampled(flights, !!forceResampleRet)
      : getScheduleRetStatsAll();
  }

  function mutRotCfgEntryForType(configByType, f) {
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
    if (!typeKey) return null;
    if (configByType[typeKey]) return configByType[typeKey];
    const tdMu = (typeof ac?.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
    const vMu = (typeof ac?.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
    const aMu = (typeof ac?.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
    const tdSigma = Math.round(tdMu * 0.1);
    const vSigma = Math.round(vMu * 0.1);
    const aSigma = Math.round(aMu * 0.1 * 10) / 10;
    configByType[typeKey] = { tdMu, tdSigma, vMu, vSigma, aMu, aSigma };
    return configByType[typeKey];
  }
  /** Same runway resolution as graphPathArrival (token.arrRunwayId before generic runwayId). */
  function resolveArrivalRunwayIdForFlight(f) {
    if (!f) return null;
    const t = f.token || {};
    return t.arrRunwayId || t.runwayId || f.arrRunwayId || null;
  }
  function isValidSampledArrRetForFlight(f, retStatsAll) {
    if (!f || f.sampledArrRet == null) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    return retStatsAll.some(function(r) {
      if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
      if (arrRunwayId == null) return true;
      return !!(r.runway && r.runway.id === arrRunwayId);
    });
  }
  /** Runway-exit (RET) sampling for Arrival Configuration / schedule RET column. ROT(arr) seconds come from Pro Sim schedule (``ARR_ROT_SEC``), not from this function. */
  function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {
    if (!f) return;
    const rev = state.vttArrCacheRev | 0;
    if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
    if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
        f.sampledArrRet != null && f.arrRetFailed === false &&
        isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrDecelMs2 = null;
    }
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const cfg = mutRotCfgEntryForType(configByType, f);
    if (!cfg || !retStatsAll || !retStatsAll.length || arrRunwayId == null) {
      f.__schedRetRotRev = rev;
      return;
    }
    const minArrVelRwy = getMinArrVelocityMpsForRunwayId(arrRunwayId);
    const tdSample = sampleNormal(cfg.tdMu, cfg.tdSigma);
    const tdMin = cfg.tdMu * 0.85;
    const tdMax = cfg.tdMu * 1.15;
    const dTd = clamp(tdSample, Math.max(0, tdMin), Math.max(0, tdMax));
    const vSample = sampleNormal(cfg.vMu, cfg.vSigma);
    const vMin = cfg.vMu * 0.85;
    const vMax = cfg.vMu * 1.15;
    const v0 = clamp(vSample, Math.max(0, vMin), Math.max(0, vMax));
    const aSample = sampleNormal(cfg.aMu, cfg.aSigma);
    const aMin = Math.max(0.1, cfg.aMu * 0.85);
    const aMax = Math.min(6,   cfg.aMu * 1.15);
    const aDec = clamp(aSample, aMin, aMax);
    const candidates = retStatsAll.filter(function(r) {
      return !!(r && r.runway && r.runway.id === arrRunwayId && r.exit);
    });
    if (!candidates.length) {
      f.arrDecelMs2 = null;
      f.__schedRetRotRev = rev;
      return;
    }
    let chosen = null;
    candidates.forEach(r => {
      if (chosen) return;
      const distFromTd = Math.max(0, r.distM - dTd);
      const vAt = runwayArrSpeedAndTimeToRet(v0, aDec, distFromTd, minArrVelRwy).vAtRet;
      if (vAt <= r.maxExitVelocity) { chosen = r; }
    });
    if (chosen) {
      f.sampledArrRet = chosen.exit && chosen.exit.id || null;
      f.arrRetFailed = false;
      const MAX_DECEL_MS2 = 15;
      const distFromTdChosen = Math.max(0, chosen.distM - dTd);
      const aDecRot = Math.min(aDec, MAX_DECEL_MS2);
      const rtRunway = runwayArrSpeedAndTimeToRet(v0, aDecRot, distFromTdChosen, minArrVelRwy);
      const vAtChosen = rtRunway.vAtRet;
      const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
        ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
        : 15;
      f.arrRunwayIdUsed = arrRunwayId;
      f.arrTdDistM = dTd;
      f.arrRetDistM = chosen.distM;
      f.arrVTdMs = v0;
      f.arrVRetInMs = vAtChosen;
      f.arrVRetOutMs = minExitVel;
      f.arrDecelMs2 = aDecRot;
    } else {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrDecelMs2 = null;
    }
    f.__schedRetRotRev = rev;
  }
  function ensureArrRetRotSampled(flights, forceResampleRet) {
    if (!Array.isArray(flights) || !flights.length) return [];
    const configByType = {};
    flights.forEach(f => { mutRotCfgEntryForType(configByType, f); });
    const retStatsAll = getScheduleRetStatsAll();
    flights.forEach(function(f) {
      sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, !!forceResampleRet);
    });
    return retStatsAll;
  }

  function _flightListEmptyHtml(message) {
    return '<div style="font-size:11px;color:#9ca3af;">' + message + '</div>';
  }

  function _renderEmptyFlightListState(listEl, cfgEl) {
    state.flightSchedulePage = 0;
    const pgr = document.getElementById('flightSchedulePager');
    if (pgr) pgr.style.display = 'none';
    _flightListTeardownVirtual(listEl);
    listEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    if (cfgEl) cfgEl.innerHTML = _flightListEmptyHtml('No flights yet.');
    const ganttEl = document.getElementById('allocationGantt');
    if (ganttEl) ganttEl.innerHTML = _flightListEmptyHtml('No flights for Gantt.');
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
  }
  function _updateFlightSchedulePagerUI(totalCount) {
    const pager = document.getElementById('flightSchedulePager');
    if (!pager) return;
    const size = FLIGHT_SCHED_PAGE_SIZE;
    if (!size || size <= 0) {
      pager.style.display = 'none';
      return;
    }
    pager.style.display = 'flex';
    const maxPage = Math.max(0, Math.ceil(totalCount / size) - 1);
    if (state.flightSchedulePage > maxPage) state.flightSchedulePage = maxPage;
    if (state.flightSchedulePage < 0) state.flightSchedulePage = 0;
    const start = state.flightSchedulePage * size;
    const end = Math.min(totalCount, start + size);
    const pageNum = maxPage + 1;
    const cur = state.flightSchedulePage + 1;
    const tEl = document.getElementById('flightSchedulePagerTotal');
    const rEl = document.getElementById('flightSchedulePagerRange');
    if (tEl) tEl.textContent = String(totalCount);
    if (rEl) rEl.textContent = totalCount ? (String(start + 1) + '–' + String(end) + ' · p ' + String(cur) + '/' + String(pageNum)) : '0–0 · p 0/0';
    const bPrev = document.getElementById('btnFlightSchedPrev');
    const bNext = document.getElementById('btnFlightSchedNext');
    if (bPrev) bPrev.disabled = state.flightSchedulePage <= 0;
    if (bNext) bNext.disabled = state.flightSchedulePage >= maxPage;
  }

  /** Same predicate as Arrival Configuration "Failed" row (flight-ui _renderFlightConfigTable failedCounts). */
  function isFlightArrRetFailedInConfigTable(f, retStatsAll) {
    if (!f) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    return f.sampledArrRet === null || typeof f.sampledArrRet === 'undefined';
  }
  function arrivalConfigColumnKeyForFlight(f) {
    if (!f) return '';
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    return f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
  }
  function isFlightCountedInArrivalConfigFailedRow(f, retStats) {
    return isFlightArrRetFailedInConfigTable(f, retStats) && !!arrivalConfigColumnKeyForFlight(f);
  }

  function flightScheduleMinutesForRow(f) {
    const sibt = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const sobt = f.sobtMin != null ? f.sobtMin : (sibt + dwell);
    return {
      sldt: f.sldtMin != null ? f.sldtMin : Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN),
      sibt: sibt,
      sobt: sobt,
      stot: f.stotMin != null ? f.stotMin : (sobt + SCHED_STOT_MINUS_SOBT_MIN),
    };
  }

  function flightScheduleProSimTimedCell(displayStr, dotKind) {
    const d = '—';
    const has = displayStr != null && String(displayStr).trim() !== '' && displayStr !== d;
    const text = has ? String(displayStr) : d;
    const muted = has ? '' : ' flight-sched-dot--muted';
    let dotClass = 'flight-sched-dot--green';
    if (dotKind === 'vttarr') dotClass = 'flight-sched-dot--vttarr';
    else if (dotKind === 'dttarr') dotClass = 'flight-sched-dot--dttarr';
    else if (dotKind === 'dttdep') dotClass = 'flight-sched-dot--dttdep';
    else if (dotKind === 'pushback') dotClass = 'flight-sched-dot--pushback';
    else if (dotKind === 'red') dotClass = 'flight-sched-dot--red';
    else if (dotKind === 'pink') dotClass = 'flight-sched-dot--pink';
    return '<span class="flight-sched-cell-inner">' +
      '<span class="flight-sched-dot ' + dotClass + muted + '" aria-hidden="true"></span>' +
      '<span class="flight-sched-cell-text">' + (has ? escapeHtml(text) : d) + '</span></span>';
  }

  function _buildFlightListHeaderHtml(apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const sHeads = [];
    const eHeads = [];
    const apHeads = [];
    for (let i = 1; i <= k; i++) {
      sHeads.push('<th class="flight-col-s' + (i === 1 ? ' flight-col-s-start flight-td-sibt' : '') + '">SIBT' + i + '</th>');
      sHeads.push('<th class="flight-col-s' + (i === k ? ' flight-col-s-last' : '') + '">SOBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EIBT' + i + '</th>');
      eHeads.push('<th class="flight-col-e">EOBT' + i + '</th>');
      apHeads.push('<th class="flight-th-mixed">AP' + i + '</th>');
    }
    return '' +
      '<table class="flight-schedule-table">' +
      '<thead><tr>' +
        '<th>Reg</th>' +
        '<th class="flight-th-mixed">Airline</th>' +
        '<th class="flight-th-mixed">Flight Num</th>' +
        '<th>ICAO Code</th>' +
        '<th class="flight-th-mixed">ICAO CAT</th>' +
        '<th>Int/Dom</th>' +
        '<th>Arr Rw</th>' +
        '<th>Arr RET</th>' +
        '<th>Arr Building</th>' +
        '<th>Dep Building</th>' +
        apHeads.join('') +
        '<th>Dep Rw</th>' +
        sHeads.join('') +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        eHeads.join('') +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">VTT(Arr)</th>' +
        '<th class="flight-th-mixed">DTT(Arr)</th>' +
        '<th class="flight-th-mixed">PUSHBACK</th>' +
        '<th class="flight-th-mixed">DTT(Dep)</th>' +
        '<th class="flight-th-mixed">VTT(Dep)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-td-del"></th>' +
      '</tr></thead>' +
      '<tbody>';
  }

  function flightScheduleSegmentsForDisplay(f, apronK) {
    const raw = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const merged = mergeAdjacentSameStandApronSegments(raw);
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const out = [];
    for (let i = 0; i < k; i++) out.push(merged[i] || null);
    return out;
  }
  function flightScheduleStandLabelById(standId) {
    if (standId == null || standId === '') return '—';
    const st = typeof findStandById === 'function' ? findStandById(standId) : null;
    if (!st) return String(standId);
    return (st.name && String(st.name).trim()) || String(st.id || standId);
  }
  function _buildFlightListRowHtml(f, retStatsAll, apronK) {
    const k = Math.max(1, Number(apronK) || flightScheduleColumnK());
    const arrRunwayId = resolveArrivalRunwayIdForFlight(f);
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStatsAll);
    let sampledRetName = '—';
    if (arrRetFailed) sampledRetName = 'Failed';
    else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {
      const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
      sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
    }
    const tArrMin = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = f.sobtMin != null ? f.sobtMin : (tArrMin + dwell);
    const schedDepRotMin = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2);
    const sldtCalc = (f.sldtMin != null ? f.sldtMin : Math.max(0, tArrMin));
    const stotCalc = (f.stotMin != null) ? f.stotMin : (tDepMin + schedDepRotMin);
    if (f.sibtMin == null || f.sobtMin == null || f.sldtMin == null || f.stotMin == null) {
      f.sldtMin = sldtCalc;
      f.sibtMin = tArrMin;
      f.sobtMin = tDepMin;
      f.stotMin = stotCalc;
    }
    const schedM = flightScheduleMinutesForRow(f);
    const sibtDisp = formatFlightScheduleDateTime(f, schedM.sibt);
    const sobtDisp = formatFlightScheduleDateTime(f, schedM.sobt);
    const segCells = flightScheduleSegmentsForDisplay(f, k);
    function fmtFlightESchedCell(minVal) {
      if (!(typeof minVal === 'number' && isFinite(minVal))) return '—';
      return formatFlightScheduleDateTime(f, minVal);
    }
    const eldtStr = fmtFlightESchedCell(f.eldtMin);
    const eibtStr = fmtFlightESchedCell(f.eibtMin);
    const eobtStr = fmtFlightESchedCell(f.eobtMin);
    const etotStr = fmtFlightESchedCell(f.etotMin);
    const dash = '—';
    const rotArrStr = (f.arrRotSec != null && isFinite(f.arrRotSec)) ? formatSecondsToHHMMSS(f.arrRotSec) : dash;
    const vttArrStr = (f.proSimVttArrSec != null && isFinite(f.proSimVttArrSec)) ? formatSecondsToHHMMSS(f.proSimVttArrSec) : dash;
    const dttArrStr = (f.proSimDttArrSec != null && isFinite(f.proSimDttArrSec)) ? formatSecondsToHHMMSS(f.proSimDttArrSec) : dash;
    const pushbackStr = (f.proSimPushbackSec != null && isFinite(f.proSimPushbackSec)) ? formatSecondsToHHMMSS(f.proSimPushbackSec) : dash;
    const dttDepStr = (f.proSimDttDepSec != null && isFinite(f.proSimDttDepSec)) ? formatSecondsToHHMMSS(f.proSimDttDepSec) : dash;
    const vttDepStr = (f.proSimVttDepSec != null && isFinite(f.proSimVttDepSec)) ? formatSecondsToHHMMSS(f.proSimVttDepSec) : dash;
    const rotDepStr = (f.proSimDepLineupSec != null && isFinite(f.proSimDepLineupSec)) ? formatSecondsToHHMMSS(f.proSimDepLineupSec) : dash;
    const rotArrCell = flightScheduleProSimTimedCell(rotArrStr, 'green');
    const vttArrCell = flightScheduleProSimTimedCell(vttArrStr, 'vttarr');
    const dttArrCell = flightScheduleProSimTimedCell(dttArrStr, 'dttarr');
    const pushbackCell = flightScheduleProSimTimedCell(pushbackStr, 'pushback');
    const dttDepCell = flightScheduleProSimTimedCell(dttDepStr, 'dttdep');
    const vttDepCell = flightScheduleProSimTimedCell(vttDepStr, 'red');
    const rotDepCell = flightScheduleProSimTimedCell(rotDepStr, 'pink');
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    ensureFlightSplitTerminalDefaults(f);
    const arrTermId = resolveFlightArrTerminalId(f);
    const depTermId = resolveFlightDepTerminalId(f);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const arrBuildingRead = escapeHtml(getTerminalDisplayLabelById(arrTermId));
    const depBuildingRead = escapeHtml(getTerminalDisplayLabelById(depTermId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    function segTimeCell(seg, key, cls) {
      if (!seg) return '<td class="flight-td-time ' + cls + '" data-empty="1">—</td>';
      const m = Number(seg[key]);
      const txt = isFinite(m) ? formatFlightScheduleDateTime(f, m) : '—';
      return '<td class="flight-td-time ' + cls + '" data-sched-min="' + (isFinite(m) ? m : '') + '">' + escapeHtml(txt) + '</td>';
    }
    function eSeriesCell(minVal, labelIdx) {
      const txt = fmtFlightESchedCell(minVal);
      return '<td class="flight-td-time flight-col-e" data-e-series-index="' + labelIdx + '">' + escapeHtml(txt) + '</td>';
    }
    const sCells = segCells.map(function(seg, idx) {
      return [
        segTimeCell(seg, 'sibtMin', 'flight-col-s' + (idx === 0 ? ' flight-col-s-start flight-td-sibt' : '')),
        segTimeCell(seg, 'sobtMin', 'flight-col-s' + (idx === k - 1 ? ' flight-col-s-last' : ''))
      ].join('');
    }).join('');
    const eCells = segCells.map(function(_seg, idx) {
      const eibtList = flightEMinListForSchedule(f, 'eibtMinList', 'eibtSecList', 'eibtMin');
      const eobtList = flightEMinListForSchedule(f, 'eobtMinList', 'eobtSecList', 'eobtMin');
      return [
        eSeriesCell(eibtList[idx] != null ? eibtList[idx] : null, idx + 1),
        eSeriesCell(eobtList[idx] != null ? eobtList[idx] : null, idx + 1)
      ].join('');
    }).join('');
    const apCells = segCells.map(function(seg) {
      const lab = seg ? flightScheduleStandLabelById(seg.standId) : '—';
      return '<td class="flight-td-readonly" data-empty="' + (seg ? '0' : '1') + '">' + escapeHtml(lab) + '</td>';
    }).join('');
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const intDomVal = (String(f.intDom || '').trim().toLowerCase() === 'dom') ? 'Dom' : 'Int';
    return '' +
      '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
        '<td class="flight-td-readonly" title="Edit in Int/Dom above when flight is selected">' + escapeHtml(intDomVal) + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '">' + (arrRetFailed ? 'Failed' : escapeHtml(sampledRetName)) + '</td>' +
        '<td class="flight-td-readonly">' + arrBuildingRead + '</td>' +
        '<td class="flight-td-readonly">' + depBuildingRead + '</td>' +
        apCells +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        sCells +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + escapeHtml(eldtStr) + '</td>' +
        eCells +
        '<td class="flight-td-time flight-col-e">' + escapeHtml(etotStr) + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotArrCell + '</td>' +
        '<td class="flight-td-time">' + vttArrCell + '</td>' +
        '<td class="flight-td-time">' + dttArrCell + '</td>' +
        '<td class="flight-td-time">' + pushbackCell + '</td>' +
        '<td class="flight-td-time">' + dttDepCell + '</td>' +
        '<td class="flight-td-time">' + vttDepCell + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + rotDepCell + '</td>' +
        '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
      '</tr>';
  }

  function _buildFlightListRowsHtml(flightsSorted, retStatsAll, apronK) {
    return flightsSorted.map(function(f) {
      return _buildFlightListRowHtml(f, retStatsAll, apronK);
    });
  }

  const FLIGHT_LIST_PATH_YIELD_CHUNK = 6;
  const FLIGHT_LIST_ASYNC_PATH_MIN = 8;
  function _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt) {
    const skipGanttRefresh = domOpt && domOpt.skipGanttRefresh;
    const apronK = flightScheduleColumnK();
    const headerRow = _buildFlightListHeaderHtml(apronK);
    const dirtyIds = [];
    dirtySet.forEach(function(id) { if (id != null && id !== '') dirtyIds.push(id); });
    const deferOnlyDirty = false;
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    } else {
      if (!deferOnlyDirty && typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
    }
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
    const usePagination = FLIGHT_SCHED_PAGE_SIZE > 0;
    let flightsForDom = flightsSorted;
    if (usePagination) {
      const size = FLIGHT_SCHED_PAGE_SIZE;
      const n = flightsSorted.length;
      const maxPage = Math.max(0, Math.ceil(n / size) - 1);
      if (state.flightSchedulePage > maxPage) state.flightSchedulePage = maxPage;
      if (state.flightSchedulePage < 0) state.flightSchedulePage = 0;


      const start = state.flightSchedulePage * size;
      flightsForDom = flightsSorted.slice(start, start + size);
    }
    _updateFlightSchedulePagerUI(flightsSorted.length);
    const useVirt = !usePagination && DOM_OPT_FLIGHT_VIRT_ENABLE && flightsSorted.length >= DOM_OPT_FLIGHT_VIRT_MIN;
    if (useVirt) {
      _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow, apronK);
    } else {
      _flightListTeardownVirtual(listEl);
      const dataRows = _buildFlightListRowsHtml(flightsForDom, retStatsAll, apronK);
      listEl.innerHTML = headerRow + dataRows.join('') + '</tbody></table>';
      const tbl0 = listEl.querySelector('.flight-schedule-table');
      if (tbl0) {
        if (usePagination) tbl0.setAttribute('data-virtual-table', '1');
        else tbl0.removeAttribute('data-virtual-table');
      }
      _flightListWireEvents(listEl, state);
    }
    _renderFlightConfigTable(cfgEl, flightsSorted);
    if (typeof ensureFlightAssignStripWired === 'function') ensureFlightAssignStripWired();
    if (typeof syncFlightAssignStrip === 'function') syncFlightAssignStrip();
    if (!skipGanttRefresh && typeof renderFlightGantt === 'function') renderFlightGantt({ skipPathPrep: true });
  }
  function _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl, scheduleOpts) {
    if (forceResampleRet && typeof bumpVttArrCacheRev === 'function') bumpVttArrCacheRev();
    let retStatsAll = [];
    if (schedFull) {
      retStatsAll = (typeof ensureArrRetRotSampled === 'function')
        ? ensureArrRetRotSampled(flightsSorted, !!forceResampleRet)
        : (typeof getScheduleRetStatsAll === 'function' ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []));
    } else {
      const dirtyFlights = flightsSorted.filter(function(f) { return dirtySet.has(f.id); });
      const dirtyForRet = dirtyFlights.filter(function(f) { return f; });
      if (dirtyForRet.length && typeof ensureArrRetRotSampled === 'function')
        retStatsAll = ensureArrRetRotSampled(dirtyForRet, false);
      else
        retStatsAll = (typeof getScheduleRetStatsAll === 'function') ? getScheduleRetStatsAll() : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
    }
    const domOpt = (scheduleOpts && scheduleOpts.skipGanttRefresh) ? { skipGanttRefresh: true } : null;
    _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt);
  }

  function renderFlightList(skipAutoAllocate, forceResampleRet, scheduleOpts, onDone) {
    const listEl = document.getElementById('flightList');
    const cfgEl = document.getElementById('flightConfigList');
    const cb = typeof onDone === 'function' ? onDone : null;
    if (!listEl) return;
    if (!state.flights.length) {
      _renderEmptyFlightListState(listEl, cfgEl);
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      if (cb) cb();
      return;
    }
    if (scheduleOpts && scheduleOpts.pageTurnOnly === true && FLIGHT_SCHED_PAGE_SIZE > 0) {
      const flightsSorted = state.flights.slice();
      flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
      beginScheduleRetStatsBatch();
      var retStatsAll2 = [];
      try {
        retStatsAll2 = (typeof getScheduleRetStatsAll === 'function')
          ? getScheduleRetStatsAll()
          : ((typeof computeRunwayExitDistances === 'function') ? computeRunwayExitDistances() : []);
        _renderFlightListDomAndSchedule(flightsSorted, false, new Set(), new Set(), listEl, cfgEl, retStatsAll2, { skipGanttRefresh: true });
      } finally {
        endScheduleRetStatsBatch();
      }
      if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      if (cb) cb();
      return;
    }
    let schedFull = true;
    let dirtySet = new Set();
    let standSet = new Set();
    if (!forceResampleRet && scheduleOpts && scheduleOpts.scheduleMode === 'incremental') {
      schedFull = false;
      const d = scheduleOpts.dirtyFlightIds;
      if (d instanceof Set) d.forEach(function(id) { if (id != null && id !== '') dirtySet.add(id); });
      else if (Array.isArray(d)) d.forEach(function(id) { if (id != null && id !== '') dirtySet.add(id); });
      const s = scheduleOpts.touchedStandIds;
      if (s instanceof Set) s.forEach(function(id) { if (id != null && id !== '') standSet.add(id); });
      else if (Array.isArray(s)) s.forEach(function(id) { if (id != null && id !== '') standSet.add(id); });
      if (dirtySet.size === 0 && standSet.size === 0) schedFull = true;
    }
    if (forceResampleRet) schedFull = true;
    const flightsSorted = state.flights.slice();
    flightsSorted.sort((a, b) => (a.sibtMin != null ? a.sibtMin : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin != null ? b.sibtMin : (b.timeMin != null ? b.timeMin : 0)));
    function runTail() {
      beginScheduleRetStatsBatch();
      try {
        _renderFlightListAfterPathEnsure(flightsSorted, schedFull, forceResampleRet, dirtySet, standSet, listEl, cfgEl, scheduleOpts);
        if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
      } finally {
        endScheduleRetStatsBatch();
      }
      if (cb) cb();
    }
    runTail();
  }

  function _renderFlightConfigTable(cfgEl, flightsSorted) {
    if (!cfgEl) return;
    const seenType = new Set();
    const unique = [];
    flightsSorted.forEach(f => {
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const typeKey = f.aircraftType || (ac && ac.id) || (ac && ac.name) || '';
      if (!typeKey || seenType.has(typeKey)) return;
      seenType.add(typeKey);
      unique.push({
        key: typeKey,
        label: ac ? (ac.name || ac.id || typeKey) : typeKey
      });
    });
    if (!unique.length) {
      cfgEl.innerHTML = _flightListEmptyHtml('No flights yet.');
      return;
    }
    const prevConfigByType = {};
    const prevInputs = cfgEl.querySelectorAll('.flight-config-input[data-ac][data-param]');
    prevInputs.forEach(inp => {
      const acKey = inp.getAttribute('data-ac');
      const param = inp.getAttribute('data-param');
      if (!acKey || !param) return;
      const valNum = Number(inp.value);
      if (!isFinite(valNum)) return;
      if (!prevConfigByType[acKey]) prevConfigByType[acKey] = {};
      prevConfigByType[acKey][param] = valNum;
    });
    const headerCols = unique.map(info => '<th>' + escapeHtml(info.label) + '</th>').join('');
    const cfgHeader = '' +
      '<div style="font-size:10px;color:#9ca3af;margin-bottom:4px;">' +
        'Landing configuration per aircraft type (unit and statistic: mean μ / spread σ).' +
      '</div>' +
      '<table class="flight-schedule-table flight-config-table">' +
      '<thead><tr>' +
        '<th class="sticky-col">Parameter</th>' +
        '<th>Unit</th>' +
        '<th>Stat</th>' +
        headerCols +
      '</tr></thead><tbody>';
    const rows = [];
    const tdMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.touchdown_zone_avg_m === 'number') ? ac.touchdown_zone_avg_m : 900;
    });
    const vtdMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.touchdown_speed_avg_ms === 'number') ? ac.touchdown_speed_avg_ms : 70;
    });
    const aMeans = unique.map(info => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-mean'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const ac = getAircraftInfoByType(acKey) || {};
      return (typeof ac.deceleration_avg_ms2 === 'number') ? ac.deceleration_avg_ms2 : 2.5;
    });
    const tdSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['td-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = tdMeans[idx];
      return Math.round(v * 0.1);
    });
    const vtdSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['vtd-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = vtdMeans[idx];
      return Math.round(v * 0.1);
    });
    const aSigmas = unique.map((info, idx) => {
      const acKey = info.key;
      const fromUser = prevConfigByType[acKey] && prevConfigByType[acKey]['a-sigma'];
      if (typeof fromUser === 'number' && isFinite(fromUser)) return fromUser;
      const v = aMeans[idx];
      return Math.round(v * 0.1 * 10) / 10;
    });
    const vTarget = 26;
    const aMeanStopDists = aMeans.map((aMu, idx) => {
      const vMu = vtdMeans[idx];
      const tdMu = tdMeans[idx];
      if (!(aMu > 0) || !(vMu > vTarget)) return Math.max(0, Math.round(tdMu || 0));
      const dFromTouchdown = (vMu*vMu - vTarget*vTarget) / (2 * aMu);
      const dTotal = (tdMu || 0) + (dFromTouchdown > 0 ? dFromTouchdown : 0);
      return dTotal > 0 ? Math.round(dTotal) : 0;
    });

    rows.push(
      '<tr>' +
        '<td class="sticky-col">Touchdown zone distance from threshold</td>' +
        '<td>m</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-mean" type="number" min="0" max="10000" step="10" value="' + tdMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="td-sigma" type="number" min="0" max="10000" step="10" value="' + tdSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col">Touchdown speed VTD</td>' +
        '<td>m/s</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-mean" type="number" min="0" max="150" step="1" value="' + vtdMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m/s</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="vtd-sigma" type="number" min="0" max="150" step="1" value="' + vtdSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col">Deceleration a</td>' +
        '<td>m/s²</td>' +
        '<td>mean μ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-mean" type="number" min="0" max="10" step="0.1" value="' + aMeans[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col"></td>' +
        '<td>m/s²</td>' +
        '<td>spread σ</td>' +
        unique.map((info, idx) =>
          '<td><input class="flight-config-input" data-ac="' + info.key + '" data-param="a-sigma" type="number" min="0" max="10" step="0.1" value="' + aSigmas[idx] + '" /></td>'
        ).join('') +
      '</tr>'
    );
    rows.push(
      '<tr>' +
        '<td class="sticky-col" style="background:rgba(124,106,247,0.14);">Distance to 26 m/s (from threshold)</td>' +
        '<td style="background:rgba(124,106,247,0.14);">m</td>' +
        '<td style="background:rgba(124,106,247,0.14);">mean-based</td>' +
        unique.map((info, idx) =>
          '<td style="background:rgba(124,106,247,0.14);font-weight:600;color:#ede9fe;">' + aMeanStopDists[idx] + '</td>'
        ).join('') +
      '</tr>'
    );
    const retStats = (typeof getScheduleRetStatsAll === 'function')
      ? getScheduleRetStatsAll()
      : (typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : []);
    if (retStats && retStats.length) {
      rows.push(
        '<tr>' +
          '<td class="sticky-col" style="padding-top:10px;">Runway exits (distance from threshold)</td>' +
          '<td></td>' +
          '<td></td>' +
          unique.map(() => '<td></td>').join('') +
        '</tr>'
      );
      const byRunway = new Map();
      retStats.forEach(r => {
        const rwId = r && r.runway && r.runway.id ? String(r.runway.id) : '';
        const key = rwId || '__unknown__';
        if (!byRunway.has(key)) byRunway.set(key, []);
        byRunway.get(key).push(r);
      });
      function runwayGroupSortKey(rwKey) {
        if (!rwKey || rwKey === '__unknown__') return 'zzzz__unknown__';
        const disp = (typeof getRunwayDisplayLabelById === 'function') ? getRunwayDisplayLabelById(rwKey) : rwKey;
        return String(disp || rwKey);
      }
      const runwayKeysSorted = Array.from(byRunway.keys()).sort((a, b) => runwayGroupSortKey(a).localeCompare(runwayGroupSortKey(b)));
      runwayKeysSorted.forEach((rwKey, rwIdx) => {
        const list = byRunway.get(rwKey) || [];
        const rwLabel = (rwKey && rwKey !== '__unknown__')
          ? escapeHtml(getRunwayDisplayLabelById(rwKey) || rwKey)
          : '—';
        list
          .slice()
          .sort((a, b) => (a && isFinite(a.distM) ? a.distM : 0) - (b && isFinite(b.distM) ? b.distM : 0))
          .forEach((r, idxInRw) => {
            void idxInRw;
            const counts = unique.map(info => {
              const typeKey = info.key;
              return (state.flights || []).filter(f =>
                f.sampledArrRet === (r.exit && r.exit.id) &&
                arrivalConfigColumnKeyForFlight(f) === typeKey
              ).length;
            });
            const sortedIdx = counts
              .map((c, i) => [c, i])
              .filter(([c]) => c > 0)
              .sort((a, b) => b[0] - a[0]);
            const top1 = sortedIdx[0] ? sortedIdx[0][1] : -1;
            const top2 = sortedIdx[1] ? sortedIdx[1][1] : -1;
            const top3 = sortedIdx[2] ? sortedIdx[2][1] : -1;
            rows.push(
              '<tr>' +
                '<td class="sticky-col">' +
                  '<span style="display:inline-flex;align-items:center;gap:4px;">' +
                    '<span style="font-size:9px;color:#9ca3af;font-weight:700;">' + rwLabel + '</span>' +
                    '<span style="padding:2px 6px;border-radius:9999px;background:rgba(124,106,247,0.16);border:1px solid rgba(124,106,247,0.35);font-size:10px;color:#ede9fe;font-weight:600;">' +
                      escapeHtml(r.name) +
                    '</span>' +
                  '</span>' +
                '</td>' +
                '<td>m</td>' +
                '<td>' + Math.round(r.distM) + '</td>' +
                unique.map((info, colIdx) => {
                  const cnt = counts[colIdx] || 0;
                  if (!cnt) return '<td></td>';
                  let bg = 'rgba(39,29,61,0.72)';
                  let color = '#ede9fe';
                  if (colIdx === top1) {
                    bg = 'rgba(124,106,247,0.36)';
                    color = '#f5f3ff';
                  } else if (colIdx === top2 || colIdx === top3) {
                    bg = 'rgba(124,106,247,0.22)';
                    color = '#ede9fe';
                  }
                  return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
                }).join('') +
              '</tr>'
            );
          });
        const isLastGroup = rwIdx === runwayKeysSorted.length - 1;
        if (!isLastGroup) {
          rows.push(
            '<tr>' +
              '<td class="sticky-col" style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>' +
              unique.map(() => '<td style="padding:6px 0;border-bottom:1px solid rgba(107,114,128,0.35);"></td>').join('') +
            '</tr>'
          );
        }
      });
      const failedCounts = unique.map(info => {
        const typeKey = info.key;
        return (state.flights || []).filter(f =>
          isFlightArrRetFailedInConfigTable(f, retStats) &&
          arrivalConfigColumnKeyForFlight(f) === typeKey
        ).length;
      });
      if (failedCounts.some(c => c > 0)) {
        const sortedFailed = failedCounts
          .map((c, i) => [c, i])
          .filter(([c]) => c > 0)
          .sort((a, b) => b[0] - a[0]);
        const fTop1 = sortedFailed[0] ? sortedFailed[0][1] : -1;
        const fTop2 = sortedFailed[1] ? sortedFailed[1][1] : -1;
        const fTop3 = sortedFailed[2] ? sortedFailed[2][1] : -1;
        rows.push(
          '<tr>' +
            '<td class="sticky-col">' +
              '<span style="padding:2px 6px;border-radius:9999px;background:rgba(127,29,29,0.9);border:1px solid #b91c1c;font-size:10px;color:#fee2e2;font-weight:600;">Failed</span>' +
            '</td>' +
            '<td></td>' +
            '<td></td>' +
            unique.map((info, colIdx) => {
              const cnt = failedCounts[colIdx] || 0;
              if (!cnt) return '<td></td>';
              let bg = 'rgba(30,30,30,0.9)';
              let color = '#fecaca';
              if (colIdx === fTop1) {
                bg = 'rgba(220,38,38,0.65)';
                color = '#fee2e2';
              } else if (colIdx === fTop2 || colIdx === fTop3) {
                bg = 'rgba(239,68,68,0.45)';
                color = '#fee2e2';
              }
              return '<td style="background:' + bg + ';color:' + color + ';font-weight:600;text-align:center;">' + cnt + '</td>';
            }).join('') +
          '</tr>'
        );
      }
    }
    function _fmtFlightPhysVal(v) {
      if (v == null || v === '') return '—';
      const n = Number(v);
      if (!isFinite(n)) return '—';
      const r = Math.round(n * 100) / 100;
      return (Math.abs(r - Math.round(r)) < 0.005) ? String(Math.round(r)) : String(r);
    }
    const perFlightBody = flightsSorted.map(function(f) {
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const typeLabel = ac ? (ac.name || ac.id || f.aircraftType || '—') : (f.aircraftType || '—');
      const arrRetFailed = isFlightCountedInArrivalConfigFailedRow(f, retStats);
      let retDisp = '—';
      if (arrRetFailed) retDisp = 'Failed';
      else if (f.sampledArrRet != null && retStats && retStats.length) {
        const retInfo = retStats.find(r => r.exit && r.exit.id === f.sampledArrRet);
        retDisp = retInfo ? (retInfo.name || 'RET') : 'RET';
      }
      const retCellInner = arrRetFailed ? 'Failed' : escapeHtml(retDisp);
      return '' +
        '<tr>' +
          '<td>' + escapeHtml(f.reg || '—') + '</td>' +
          '<td>' + escapeHtml(f.airlineCode || '—') + '</td>' +
          '<td>' + escapeHtml(f.flightNumber || '—') + '</td>' +
          '<td>' + escapeHtml(String(typeLabel)) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrVTdMs) + '</td>' +
          '<td style="text-align:right;font-variant-numeric:tabular-nums;">' + _fmtFlightPhysVal(f.arrDecelMs2) + '</td>' +
          '<td class="flight-td-arr-ret' + (arrRetFailed ? ' flight-td-arr-ret-failed' : '') + '" style="text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;">' + retCellInner + '</td>' +
        '</tr>';
    }).join('');
    const perFlightBlock = '' +
      '<div class="flight-config-sampled-caption">' +
        '<span class="flight-config-sampled-caption-ko">항공기별 적용값 · 샘플링된 접지속도(VTD)와 활주로 감속도</span>' +
        '<span class="flight-config-sampled-caption-en">Per flight: sampled VTD &amp; deceleration (used after page reload / path compute)</span>' +
      '</div>' +
      '<div class="flight-config-sampled-scroll">' +
        '<table class="flight-schedule-table flight-config-per-flight-table">' +
          '<thead><tr>' +
            '<th>Reg</th>' +
            '<th>Airline</th>' +
            '<th>Flight</th>' +
            '<th>Aircraft type</th>' +
            '<th style="text-align:right;">VTD (m/s)</th>' +
            '<th style="text-align:right;">Decel (m/s²)</th>' +
            '<th style="text-align:right;">Arr RET</th>' +
          '</tr></thead>' +
          '<tbody>' + perFlightBody + '</tbody>' +
        '</table>' +
      '</div>';
    cfgEl.innerHTML = cfgHeader + rows.join('') + '</tbody></table>' +
      '<div style="font-size:10px;color:#6b7280;margin-top:8px;">' +
        'Note: sampling is clipped to stay within ±15% of each mean value.' +
      '</div>' +
      perFlightBlock;
  }

  function syncAllocGanttSelectionHighlight() {
    const ganttRoot = document.getElementById('allocationGantt');
    if (!ganttRoot || !ganttRoot.querySelector('.alloc-gantt-root')) return;
    ganttRoot.querySelectorAll('.alloc-flight').forEach(function(el) {
      el.classList.remove('alloc-flight-selected');
    });
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'flight' || !sel.id) return;
    const wantId = String(sel.id);
    ganttRoot.querySelectorAll('.alloc-flight').forEach(function(el) {
      if (el.getAttribute('data-flight-id') === wantId) el.classList.add('alloc-flight-selected');
    });
  }

  function _flightListWireEvents(listEl, st) {
    listEl.querySelectorAll('.obj-item-delete').forEach(function(btn) {
      btn.addEventListener('click', function(ev) {
        var idVal = this.getAttribute('data-del');
        var fDel = st.flights.find(function(x) { return x.id === idVal; });
        var delStand = (fDel && fDel.standId) ? fDel.standId : null;
        st.flights = st.flights.filter(function(f) { return f.id !== idVal; });
        recomputeSimDuration();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        if (delStand)
          renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [], touchedStandIds: [delStand] });
        else
          renderFlightList();
      });
    });
    listEl.querySelectorAll('.obj-item').forEach(function(row) {
      row.addEventListener('click', function(ev) {
        if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
        var idVal = this.getAttribute('data-id');
        var f = st.flights.find(function(x) { return x.id === idVal; });
        if (!f) return;
        state.flightPathRevealFlightId = null;
        st.selectedObject = { type: 'flight', id: idVal, obj: f };
        listEl.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
        this.classList.add('selected', 'expanded');
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
      row.addEventListener('dblclick', function(ev) {
        if ((ev.target.classList && ev.target.classList.contains('obj-item-delete')) || ev.target.getAttribute('data-del')) return;
        ev.preventDefault();
        var idVal = this.getAttribute('data-id');
        var f = st.flights.find(function(x) { return x.id === idVal; });
        if (!f) return;
        st.selectedObject = { type: 'flight', id: idVal, obj: f };
        state.flightPathRevealFlightId = idVal;
        listEl.querySelectorAll('.obj-item').forEach(function(r) { r.classList.remove('selected', 'expanded'); });
        this.classList.add('selected', 'expanded');
        if (typeof updateObjectInfo === 'function') updateObjectInfo();
        if (typeof syncPanelFromState === 'function') syncPanelFromState();
        if (typeof draw === 'function') draw();
        if (typeof syncAllocGanttSelectionHighlight === 'function') syncAllocGanttSelectionHighlight();
      });
    });
  }


  function _ganttSaveViewState(ganttEl) {
    let scrollLeft = 0, scrollTop = 0;
    const scrollCol = ganttEl.querySelector('.alloc-gantt-scroll-col');
    if (scrollCol) {
      scrollLeft = scrollCol.scrollLeft || 0;
      scrollTop = scrollCol.scrollTop || 0;
    }
    const collapsedTerminals = new Set();
    let remoteCollapsed = false;
    const labelCol = ganttEl.querySelector('.alloc-gantt-label-col');
    if (labelCol) {
      Array.from(labelCol.children).forEach(function (el) {
        if (el.classList && el.classList.contains('alloc-terminal-header')) {
          if (el.getAttribute('data-collapsed') === '1') {
            let txt = (el.textContent || '').trim().replace(/^[▶▼]\s*/, '');
            if (txt) collapsedTerminals.add(txt);
          }
        }
        if (el.classList && el.classList.contains('alloc-remote-header')) {
          if (el.getAttribute('data-collapsed') === '1') remoteCollapsed = true;
        }
      });
    }
    return { scrollLeft: scrollLeft, scrollTop: scrollTop, collapsedTerminals: collapsedTerminals, remoteCollapsed: remoteCollapsed };
  }

  function renderFlightGantt(opt) {
    const skipPathPrep = opt && opt.skipPathPrep;
    const ganttEl = document.getElementById('allocationGantt');
    if (!ganttEl) return;
    const viewState = _ganttSaveViewState(ganttEl);
    const prevScrollLeft = viewState.scrollLeft;
    const prevScrollTop = viewState.scrollTop;
    const prevCollapsedTerminals = viewState.collapsedTerminals;
    const prevRemoteCollapsed = viewState.remoteCollapsed;
    if (!state.flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty = document.getElementById('allocGanttWindowLabel');
      if (labEmpty) labEmpty.textContent = '';
      return;
    }
    const flights = state.flights.slice();
    const stands = allStandsForFlightAssignment();
    if (!flights.length) {
      state.allocGanttWindowStartMin = null;
      ganttEl.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No flights for Gantt.</div>';
      const labEmpty2 = document.getElementById('allocGanttWindowLabel');
      if (labEmpty2) labEmpty2.textContent = '';
      return;
    }
    if (!skipPathPrep) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
    }

    let intervals = [];
    const schedTable = document.querySelector('.flight-schedule-table');
    const domScheduleOk = schedTable && schedTable.getAttribute('data-virtual-table') !== '1';
    if (domScheduleOk) {
      const rows = Array.from(schedTable.querySelectorAll('tbody tr.flight-data-row'));
      const flightById = new Map();
      for (let fi = 0; fi < flights.length; fi++) {
        const ff = flights[fi];
        if (ff && ff.id != null) flightById.set(String(ff.id), ff);
      }
      rows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (!id) return;
        const f = flightById.get(String(id));
        if (!f) return;
        const tds = Array.from(row.querySelectorAll('td'));
        const k = flightScheduleColumnK();
        const sibtIdx = flightSchedColIndex('sibt', k);
        const sobtIdx = flightSchedColIndex('sobt', k);
        const eldtIdx = flightSchedColIndex('eldt', k);
        const eibtIdx = flightSchedColIndex('eibt', k);
        const eobtIdx = flightSchedColIndex('eobt', k);
        const etotIdx = flightSchedColIndex('etot', k);
        if (tds.length <= etotIdx) return;
        const getMin = (idx) => {
          const td = tds[idx];
          if (!td) return 0;
          const dm = td.getAttribute('data-sched-min');
          if (dm != null && String(dm).trim() !== '') {
            const n = parseFloat(dm);
            return isFinite(n) ? n : 0;
          }
          const txt = (td.textContent || '').trim();
          if (!txt) return 0;
          try {
            return parseTimeToMinutes(txt);
          } catch (e) {
            return 0;
          }
        };
        const sibt = getMin(sibtIdx);
        const sobt = getMin(sobtIdx);
        const sldt = Math.max(0, sibt - SCHED_SIBT_MINUS_SLDT_MIN);
        const stot = sobt + SCHED_STOT_MINUS_SOBT_MIN;
        const eSer = ganttESeriesMinutesFromTimelineMeta(f);
        const eldt = eSer.eldt != null ? eSer.eldt : getMin(eldtIdx);
        const eibt = eSer.eibt != null ? eSer.eibt : getMin(eibtIdx);
        const eobt = eSer.eobt != null ? eSer.eobt : getMin(eobtIdx);
        const etot = eSer.etot != null ? eSer.etot : getMin(etotIdx);
        if (Array.isArray(f.apronStaySegments) && f.apronStaySegments.length > 1 && typeof buildApronStayGanttIntervalsForFlight === 'function') {
          buildApronStayGanttIntervalsForFlight(f, eSer).forEach(function(it) { intervals.push(it); });
        } else {
          const t0 = sibt;
          const t1 = sobt || (t0 + (f.dwellMin != null ? f.dwellMin : 0));
          const sldtOrig = sldt;
          const sobtOrig = sobt || t1;
          const stotOrig = stot;
          intervals.push({ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig, segmentIdx: 0, segmentCount: 1, segmentStandId: f.standId || null });
        }
      });
    }
    if (!intervals.length) {
      flights.forEach(f => {
        const t0 = f.sibtMin != null ? f.sibtMin : (f.timeMin != null ? f.timeMin : 0);
        const t1 = f.sobtMin != null ? f.sobtMin : (t0 + (f.dwellMin != null ? f.dwellMin : 0));
        const sldt = f.sldtMin != null ? f.sldtMin : Math.max(0, t0 - SCHED_SIBT_MINUS_SLDT_MIN);
        const stot = f.stotMin != null ? f.stotMin : (t1 + SCHED_STOT_MINUS_SOBT_MIN);
        const eSer2 = ganttESeriesMinutesFromTimelineMeta(f);
        if (Array.isArray(f.apronStaySegments) && f.apronStaySegments.length > 1 && typeof buildApronStayGanttIntervalsForFlight === 'function') {
          buildApronStayGanttIntervalsForFlight(f, eSer2).forEach(function(it) { intervals.push(it); });
          return;
        }
        const eibt = eSer2.eibt;
        const eobt = eSer2.eobt;
        const eldt = eSer2.eldt;
        const etot = eSer2.etot;
        const sldtOrig = sldt;
        const sobtOrig = f.sobtMin != null ? f.sobtMin : t1;
        const stotOrig = stot;
        intervals.push({ f, t0, t1, sldt, stot, eibt, eobt, eldt, etot, sldtOrig, sobtOrig, stotOrig, segmentIdx: 0, segmentCount: 1, segmentStandId: f.standId || null });
      });
    }

    let minS = Infinity;
    let maxE = -Infinity;
    intervals.forEach(it => {
      if (it.sldt < minS) minS = it.sldt;
      const etot0 = (it.etot != null && isFinite(it.etot)) ? it.etot : it.stot;
      if (etot0 > maxE) maxE = etot0;
    });
    if (minS <= 0 && intervals.length) {
      const posSldt = intervals.map(function(it) { return it.sldt; }).filter(function(v) { return isFinite(v) && v > 1e-6; });
      if (posSldt.length) minS = Math.min.apply(null, posSldt);
    }
    if (!isFinite(minS) || !isFinite(maxE)) {
      ganttEl.innerHTML = '';
      return;
    }
    const baseMinT = Math.max(0, minS - GANTT_PAD_MIN);
    const baseMaxT0 = maxE + GANTT_PAD_MIN;
    const baseMaxT = (baseMaxT0 <= baseMinT) ? (baseMinT + 60) : baseMaxT0;
    const baseSpan = baseMaxT - baseMinT;
    const dataSpan = Math.max(1e-9, baseSpan);
    const visibleSpan = Math.min(GANTT_VISIBLE_WINDOW_MIN, dataSpan);
    const maxWinStart = Math.max(baseMinT, baseMaxT - visibleSpan);
    let winStart = state.allocGanttWindowStartMin;
    if (winStart == null || !isFinite(winStart)) winStart = baseMinT;
    const vpPin = state._allocGanttHandleDragViewportPin;
    if (vpPin && vpPin.active) {
      let w = vpPin.winStart0;
      if (w == null || !isFinite(w)) w = winStart;
      if (w > maxWinStart) w = maxWinStart;
      if (w + visibleSpan < baseMinT - 1e-6) w = Math.min(maxWinStart, baseMinT);
      winStart = w;
    } else {
      winStart = Math.min(Math.max(winStart, baseMinT), maxWinStart);
    }
    state.allocGanttWindowStartMin = winStart;
    const winEnd = winStart + visibleSpan;
    state._allocGanttClamp = { baseMinT: baseMinT, baseMaxT: baseMaxT, visibleSpan: visibleSpan };
    const displaySpan = visibleSpan;
    const zoom = (state.allocTimeZoom && state.allocTimeZoom > 1) ? state.allocTimeZoom : 1;

    const tickPositions = buildTimeAxisTicks(winStart, winEnd, winStart, displaySpan, zoom);

    function allocLeftPct(t) {
      return ((t - winStart) / displaySpan) * 100 * zoom;
    }
    function allocTrackSpanHtml(cls, leftPct, widthPct, minWidthPct) {
      return '<div class="' + cls + '" style="left:' + leftPct + '%;width:' + Math.max(minWidthPct, widthPct) + '%;"></div>';
    }
    function allocTrackMarkerHtml(cls, leftPct) {
      return '<div class="' + cls + '" style="left:' + leftPct + '%;"></div>';
    }
    function pushAllocDot(arr, t, cls) {
      if (!arr || !isFinite(t) || t < winStart || t > winEnd) return;
      arr.push(allocTrackMarkerHtml('alloc-time-dot ' + cls, allocLeftPct(t)));
    }
    function pushAllocSpan(arr, startT, endT, cls, minWidthPct) {
      if (!arr || !isFinite(startT) || !isFinite(endT) || endT <= startT) return;
      const clippedStart = Math.max(startT, winStart);
      const clippedEnd = Math.min(endT, winEnd);
      if (clippedEnd <= clippedStart) return;
      arr.push(allocTrackSpanHtml(cls, allocLeftPct(clippedStart), ((clippedEnd - clippedStart) / displaySpan) * 100 * zoom, minWidthPct));
    }
    function pushAllocTriangle(arr, t, cls) {
      if (!arr || !isFinite(t) || t < winStart || t > winEnd) return;
      arr.push(allocTrackMarkerHtml(cls, allocLeftPct(t)));
    }

    /** O(flights) — avoid per-row intervals.filter (was O(stands * flights) per gantt pass). */
    const intervalsByStandKey = (function() {
      const o = { __unassigned: [] };
      for (let gi = 0; gi < intervals.length; gi++) {
        const it = intervals[gi];
        const raw = it.segmentStandId != null ? it.segmentStandId : (it.f && it.f.standId);
        if (raw == null || raw === '') o.__unassigned.push(it);
        else {
          const sid = String(raw);
          if (!o[sid]) o[sid] = [];
          o[sid].push(it);
        }
      }
      return o;
    })();

    function buildRowHtml(label, standId) {
      const showSPointsEl = document.getElementById('chkShowSPoints');
      const showSPoints = !showSPointsEl || showSPointsEl.checked;
      const showSBarsEl = document.getElementById('chkShowSBars');
      const dimSBars = !!(showSBarsEl && !showSBarsEl.checked);
      const showEBarEl = document.getElementById('chkShowEBar');
      const showEBar = !showEBarEl || showEBarEl.checked;
      const showEPointsEl = document.getElementById('chkShowEPoints');
      const showEPoints = !showEPointsEl || showEPointsEl.checked;
      const showAuxBars = showSPoints;
      const showEibtBars = showEBar;
      const showEldtBars = showEPoints;
      const showSDots = showSPoints;
      const showSdDots = showSPoints;
      const showEDots = showEPoints;
      const rowFlights = (standId == null)
        ? (intervalsByStandKey.__unassigned || [])
        : (intervalsByStandKey[String(standId)] || []);
      const conflictMap = {};
      for (let i = 0; i < rowFlights.length; i++) {
        for (let j = i + 1; j < rowFlights.length; j++) {
          const a = rowFlights[i];
          const b = rowFlights[j];
          if (a.f && b.f && a.f.id === b.f.id) continue;
          if (a.t0 < b.t1 && b.t0 < a.t1) { // Section overlap
            conflictMap[a.f.id] = true;
            conflictMap[b.f.id] = true;
          }
        }
      }
      const sBars = showAuxBars ? [] : null;
      const eBars = showEibtBars ? [] : null;
      const e2Bars = showEldtBars ? [] : null;
      const sDots = showSDots ? [] : null;
      const sdDots = showSdDots ? [] : null;
      const eDots = showEDots ? [] : null;
      const sLines = showSPoints ? [] : null;      // SOBT(orig) vertical line
      const sTrisDown = showSPoints ? [] : null;   // SLDTtriangle under dragon
      const sTrisUp = showSPoints ? [] : null;     // STOTtriangle above dragon
      const eTrisDown = showEPoints ? [] : null;   // ELDTtriangle under dragon
      const eTrisUp = showEPoints ? [] : null;     // ETOTtriangle above dragon
      const blocks = rowFlights.map(it => {
        const f = it.f;
        const t0 = it.t0;
        const t1 = it.t1;
        const sldt = it.sldt;
        const stot = it.stot;
        const eibt = it.eibt;
        const eobt = it.eobt;
