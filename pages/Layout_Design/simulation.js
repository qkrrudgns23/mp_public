      const v = this.value || '';
      if (tw.pathType === 'runway') {
        runwayReverseVerticesIfDirectionChanged(tw, v);
        tw.direction = (v === 'counter_clockwise') ? 'counter_clockwise' : 'clockwise';
      } else tw.direction = v || 'both';
      updateObjectInfo();
      if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      draw();
      update3DSceneWhenVisible();
      if (shouldResampleRet) triggerArrivalConfigResampleFromLayoutEdit();
    }
  });
  const taxiwayPathTypeKindEl = document.getElementById('taxiwayPathTypeKind');
  if (taxiwayPathTypeKindEl) {
    taxiwayPathTypeKindEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        const ptCur = tw.pathType || 'taxiway';
        if (ptCur === 'taxiway' || ptCur === 'general_queue_taxiway') {
          const kind = String(this.value || 'normal');
          tw.pathType = (kind === 'queue') ? 'general_queue_taxiway' : 'taxiway';
        }
        updateObjectInfo();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        draw();
        update3DSceneWhenVisible();
      }
    });
  }
  const runwayMinArrVelEl = document.getElementById('runwayMinArrVelocity');
  if (runwayMinArrVelEl) {
    runwayMinArrVelEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val > 0) ? Math.max(1, Math.min(150, val)) : 15;
        tw.minArrVelocity = v;
        this.value = v;
        updateObjectInfo();
        renderObjectList();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
        draw();
      }
    });
  }
  [
    ['runwayLineupDistM_CW', 'clockwise'],
    ['runwayLineupDistM_CCW', 'counter_clockwise']
  ].forEach(function(item) {
    const lineupEl = document.getElementById(item[0]);
    if (!lineupEl) return;
    lineupEl.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : 0;
        if (item[1] === 'clockwise') tw.lineupDistM_CW = v;
        else tw.lineupDistM_CCW = v;
        tw.lineupDistM = getEffectiveRunwayLineupDistM(tw);
        this.value = String(v);
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      }
    });
  });
  [
    ['runwayStartDisplacedThresholdM', 'startDisplacedThresholdM', function(tw) { return getEffectiveRunwayStartDisplacedThresholdM(tw); }],
    ['runwayStartBlastPadM', 'startBlastPadM', function(tw) { return getEffectiveRunwayStartBlastPadM(tw); }],
    ['runwayEndDisplacedThresholdM', 'endDisplacedThresholdM', function(tw) { return getEffectiveRunwayEndDisplacedThresholdM(tw); }],
    ['runwayEndBlastPadM', 'endBlastPadM', function(tw) { return getEffectiveRunwayEndBlastPadM(tw); }]
  ].forEach(function(item) {
    const el = document.getElementById(item[0]);
    if (!el) return;
    el.addEventListener('change', function() {
      if (state.selectedObject && state.selectedObject.type === 'taxiway') {
        const tw = state.selectedObject.obj;
        if (tw.pathType !== 'runway') return;
        const val = Number(this.value);
        const v = (typeof val === 'number' && isFinite(val) && val >= 0) ? val : item[2](tw);
        tw[item[1]] = v;
        this.value = String(v);
        updateObjectInfo();
        draw();
        if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
      }
    });
  });

  function getMinArrVelocityMpsForRunwayId(runwayId) {
    if (runwayId == null || runwayId === '') return 15;
    const list = state.taxiways || [];
    let tw = list.find(t => t.id === runwayId && t.pathType === 'runway');
    if (!tw) return 15;
    const v = tw.minArrVelocity;
    if (typeof v === 'number' && isFinite(v) && v > 0) return Math.max(1, Math.min(150, v));
    return 15;
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
    if (PLAYBACK_LEAD_BEFORE_FIRST_TD_SEC > 0) {
      const firstTdS = minFirstArrivalTouchdownSecAmongFlights();
      if (firstTdS != null) {
        simLo = Math.max(0, firstTdS - PLAYBACK_LEAD_BEFORE_FIRST_TD_SEC);
      }
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
    updateFlightSimPlaybackLabelsDom();
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
  }
  function applySimPlaybackBarDomVisibility() {
    const wrap = document.getElementById('sim-controls-wrap');
    const inner = document.getElementById('sim-controls-container');
    const hideBtn = document.getElementById('btnHideSimPlaybackBar');
    const hasSim = state.hasSimulationResult && state.flights.length > 0;
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
    state.hasSimulationResult = (state.flights || []).length > 0;
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
      const dwellMin = (f.sobtMin_d != null && f.sibtMin_d != null) ? (f.sobtMin_d - f.sibtMin_d) : (f.dwellMin || 0);
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

  function assignStandToFlight(f, standId) {
    if (!f) return false;
    if (standId) {
      const allStands = allStandsForFlightAssignment();
      const stand = allStands.find(function(s) { return s.id === standId; });
      if (!flightCanUseStand(f, stand)) {
        alert("Stand constraints or selected building do not match this aircraft, so it cannot be assigned.");
        return false;
      }
    }
    const prevStandForSched = f.standId || null;
    f.standId = standId;
    if (f.token) f.token.apronId = standId;
    delete f.sobtMin_orig;
    delete f.sldtMin_orig;
    delete f.sibtMin_orig;
    delete f.stotMin_orig;
    delete f.eldtMin_orig;
    delete f.eibtMin_orig;
    delete f.eobtMin_orig;
    delete f.etotMin_orig;
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    const touchedSt = [];
    if (prevStandForSched) touchedSt.push(prevStandForSched);
    if (standId) touchedSt.push(standId);
    if (typeof renderFlightList === 'function')
      renderFlightList(false, false, { scheduleMode: 'incremental', dirtyFlightIds: [f.id], touchedStandIds: touchedSt });
    if (typeof draw === 'function') draw();
    return true;
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
    const sibtMin_d = flight.sibtMin_d != null ? flight.sibtMin_d : (flight.timeMin != null ? flight.timeMin : 0);
    const baseT = sibtMin_d * 60;
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
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (sibtMin_d + (flight.dwellMin != null ? flight.dwellMin : 0));
    const dwellSec = Math.max(0, (sobtMin_d - sibtMin_d) * 60);
    if (dwellSec > 0) {
      tAcc = sobtMin_d * 60;
      const last = timeline[timeline.length - 1];
      timeline.push({ t: tAcc, x: last.x, y: last.y });
    }
    return timeline;
  }

  function buildDepartureTimelineFromPts(flight, pts) {
    if (!pts || pts.length < 2) return null;
    const sobtMin_d = flight.sobtMin_d != null ? flight.sobtMin_d : (flight.timeMin != null ? flight.timeMin + (flight.dwellMin != null ? flight.dwellMin : 0) : 0);
    const baseT = sobtMin_d * 60;
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
      if (a.motionForward === false) return { x: a.x, y: a.y, dx: -1, dy: 0, deadlockGhost: dg };
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
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;
        const h = headingForInterval(i);
        const mfB = b.motionForward !== false;
        let rdx = h.dx, rdy = h.dy;
        if (mfB === false) {
          const len = Math.hypot(rdx, rdy) || 1;
          const ang = Math.atan2(rdy, rdx) + Math.PI;
          rdx = Math.cos(ang) * len;
          rdy = Math.sin(ang) * len;
        }
        const dg = !!(a.deadlockGhost || b.deadlockGhost);
        return { x, y, dx: rdx, dy: rdy, deadlockGhost: dg };
      }
    }
    return null;
  }

  
  function getFlightPoseAtTimeForDraw(flight, tSec) {
    const tl = flight && flight.timeline;
    if (!tl || !tl.length) return null;
    let t = Number(tSec);
    if (!isFinite(t)) return null;
    const t0 = tl[0].t, t1 = tl[tl.length - 1].t;
    if (t + 1e-9 < t0) return null;
    if (t > t1) t = t1;
    return getFlightPoseAtTime(flight, t);
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
    else if (role === 'term') el.value = f.terminalId || (f.token && f.token.terminalId) || '';
    else if (role === 'dep') el.value = f.depRunwayId || (f.token && f.token.depRunwayId) || '';
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
  function syncFlightAssignStripFromFlight(f) {
    const arrEl = document.getElementById('flightAssignStripArr');
    const termEl = document.getElementById('flightAssignStripTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    if (arrEl) {
      const sid = f ? (resolveArrivalRunwayIdForFlight(f) || '') : '';
      arrEl.innerHTML = buildRunwayOptionsHtml(sid);
      arrEl.value = sid;
    }
    if (termEl) {
      const tid = f ? (f.terminalId || (f.token && f.token.terminalId) || '') : '';
      termEl.innerHTML = buildTerminalOptionsHtml(tid);
      termEl.value = tid;
    }
    if (depEl) {
      const did = f ? (f.depRunwayId || (f.token && f.token.depRunwayId) || '') : '';
      depEl.innerHTML = buildRunwayOptionsHtml(did);
      depEl.value = did;
    }
  }
  function syncFlightAssignStrip() {
    const arrEl = document.getElementById('flightAssignStripArr');
    const termEl = document.getElementById('flightAssignStripTerm');
    const depEl = document.getElementById('flightAssignStripDep');
    const sel = state.selectedObject;
    const hasFlight = sel && sel.type === 'flight' && sel.id;
    const f = hasFlight ? state.flights.find(function(x) { return x.id === sel.id; }) : null;
    const dis = !f;
    [arrEl, termEl, depEl].forEach(function(el) {
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
    var val = null;
    if (role === 'arr' || role === 'dep') {
      const r = resolveRunwayIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else if (role === 'term') {
      const r = resolveTerminalIdFromInput(raw);
      if ((raw || '').trim() && r === undefined) {
        syncFlightAssignStripFromFlight(f);
        return;
      }
      val = r === undefined ? null : r;
    } else return;
    var prevArr = f.arrRunwayId || null;
    var prevDep = f.depRunwayId || (f.token && f.token.depRunwayId) || null;
    var prevTerm = f.terminalId || (f.token && f.token.terminalId) || null;
    if (role === 'arr' && val === prevArr) return;
    if (role === 'dep' && val === prevDep) return;
    if (role === 'term' && val === prevTerm) return;
    var prevStand = f.standId || null;
    if (!f.token) f.token = { nodes: ['runway','taxiway','apron','terminal'], runwayId: null, apronId: null, terminalId: null };
    if (role === 'arr') {
      f.arrRunwayId = val;
      f.token.runwayId = val;
    } else if (role === 'term') {
      f.terminalId = val;
      f.token.terminalId = val;
      if (f.standId) {
        var allStands = (st.pbbStands || []).concat(st.remoteStands || []).concat(st.tempStands || []);
        var stand = allStands.find(function(s) { return s.id === f.standId; });
        if (stand) {
          var term = getTerminalForStand(stand);
          var standTermId = term ? term.id : null;
          if (!val || !standTermId || val !== standTermId) f.standId = null;
        }
      }
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

  const FLIGHT_SCHED_TABLE_COL_COUNT = 27;
  /** tbody td index alignment with `_buildFlightListRowHtml` (0-based). S-group SLDT/STOT columns removed. */
  const FLIGHT_SCHED_TD_SLD = 10;
  const FLIGHT_SCHED_TD_SIBTD = 11;
  const FLIGHT_SCHED_TD_SOBTD = 12;
  const FLIGHT_SCHED_TD_STOTD = 13;
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
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + topPad + 'px\"><td colspan=\"' + FLIGHT_SCHED_TABLE_COL_COUNT + '\"></td></tr>');
    for (let i = start; i < end; i++) {
      parts.push(_buildFlightListRowHtml(flightsSorted[i], retStatsAll));
    }
    parts.push('<tr class=\"flight-virt-spacer\" aria-hidden=\"true\" style=\"height:' + botPad + 'px\"><td colspan=\"' + FLIGHT_SCHED_TABLE_COL_COUNT + '\"></td></tr>');
    tbody.innerHTML = parts.join('');
    _flightListWireEvents(listEl, state);
  }
  function _flightListTeardownVirtual(listEl) {
    listEl._flightVirtState = null;
  }
  function _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow) {
    const prevScroll = listEl.querySelector('.flight-schedule-table[data-virtual-table=\"1\"]') ? (listEl.scrollTop || 0) : 0;
    listEl._flightVirtState = {
      flightsSorted: flightsSorted,
      retStatsAll: retStatsAll,
      rowH: DOM_OPT_FLIGHT_VIRT_ROW_H,
      overscan: DOM_OPT_FLIGHT_VIRT_OVERSCAN,
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
  
  function applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotRunwayCandidateMin) {
    if (!f) return;
    const eibt = eibtMin != null && isFinite(eibtMin) ? eibtMin : 0;
    const block = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const { dwell, minDwell } = getNormalizedStandDwellBounds(f);
    const low = eibt + minDwell;
    const high = eibt + dwell;
    const sobtPref = (f.sobtMin_d != null)
      ? f.sobtMin_d
      : (f.sibtMin_d != null
        ? f.sibtMin_d + dwell
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
  function beginScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = true;
    __schedRetStatsCached = null;
  }
  function endScheduleRetStatsBatch() {
    __schedRetStatsBatchActive = false;
    __schedRetStatsCached = null;
  }
  function getScheduleRetStatsAll() {
    if (__schedRetStatsBatchActive) {
      if (__schedRetStatsCached === null) {
        __schedRetStatsCached = typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
      }
      return __schedRetStatsCached;
    }
    return typeof computeRunwayExitDistances === 'function' ? computeRunwayExitDistances() : [];
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
