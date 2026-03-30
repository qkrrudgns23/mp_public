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
      if (f.noWayArr && f.noWayDep) return;
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
    state.simDurationSec = Math.max(maxT, minT);
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
    const hasSim = state.hasSimulationResult && state.flights.length > 0 && state.globalUpdateFresh;
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
      if (!m || typeof m.eibtSec !== 'number' || typeof m.eobtSec !== 'number') continue;
      if (t + 1e-3 >= m.eibtSec && t <= m.eobtSec + 1e-3) return true;
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

  function flightCanUseStand(f, stand) {
    if (!stand) return true;
    const mode = getStandCategoryMode(stand);
    if (mode === 'aircraft') {
      const allowedTypes = getStandAllowedAircraftTypes(stand);
      const flightType = String(f.aircraftType || '').trim();
      if (!allowedTypes.length || !flightType || allowedTypes.indexOf(flightType) < 0) return false;
    } else {
      const order = { A:1,B:2,C:3,D:4,E:5,F:6 };
      const fCode = (f.code || 'C').toUpperCase();
      const sCat = (stand.category || 'F').toUpperCase();
      const fc = order[fCode] || 99;
      const sc = order[sCat] || 0;
      if (fc > sc) return false;
    }
    const ft = (f.terminalId || (f.token && f.token.terminalId)) || null;
    if (!ft) return true;
    const isRemote = (state.remoteStands || []).some(function(r) { return r.id === stand.id; });
    if (isRemote) {
      const allowed = Array.isArray(stand.allowedTerminals) ? stand.allowedTerminals : [];
      if (allowed.length) return allowed.includes(ft);
    }
    const term = getTerminalForStand(stand);
    const standTermId = term ? term.id : null;
    if (!standTermId) return false;
    return ft === standTermId;
  }

  function assignStandToFlight(f, standId) {
    if (!f) return false;
    if (standId) {
      const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
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
      if (!flight && code && getStandCategoryMode(stand) === 'icao' && stand.category && stand.category !== code) return;
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
    const allStands = (state.pbbStands || []).concat(state.remoteStands || []);
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
      return { x: a.x, y: a.y, dx: 1, dy: 0 };
    }
    if (tSec < tl[0].t || tSec > tl[tl.length - 1].t) return null;
    for (let i = 0; i < tl.length - 1; i++) {
      const a = tl[i], b = tl[i+1];
      if (tSec >= a.t && tSec <= b.t) {
        const span = b.t - a.t || 1;
        const u = (tSec - a.t) / span;
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        return { x, y, dx, dy };
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
    if (role === 'arr') el.value = f.arrRunwayId || (f.token && f.token.runwayId) || '';
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
      const sid = f ? (f.arrRunwayId || (f.token && f.token.runwayId) || '') : '';
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
        var allStands = (st.pbbStands || []).concat(st.remoteStands || []);
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

  const FLIGHT_SCHED_TABLE_COL_COUNT = 29;
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
    const rev = state.vttArrCacheRev | 0;
    if (f.__schedVttArrRev === rev && f.__schedVttArrMin != null && isFinite(f.__schedVttArrMin)) {
      return f.__schedVttArrMin;
    }
    if (typeof sampleArrRetRotForFlightIfNeeded === 'function') {
      const retStatsAll = getScheduleRetStatsAll();
      const rotCfgMap = {};
      sampleArrRetRotForFlightIfNeeded(f, retStatsAll, rotCfgMap, false);
    }
    const arrPts = (typeof getPathForFlight === 'function') ? getPathForFlight(f) : null;
    let vttArrMin = 0;
    if (arrPts && arrPts.length >= 2) {
      let startIdx = 0;
      if (f.sampledArrRet) {
        const tw = (state.taxiways || []).find(t => t.id === f.sampledArrRet);
        if (tw && Array.isArray(tw.vertices) && tw.vertices.length) {
          const last = tw.vertices[tw.vertices.length - 1];
          const retOutPt = cellToPixel(last.col, last.row);
          let bestD2 = Infinity;
          let bestIdx = 0;
          for (let i = 0; i < arrPts.length; i++) {
            const dx = arrPts[i][0] - retOutPt[0];
            const dy = arrPts[i][1] - retOutPt[1];
            const d2 = dx*dx + dy*dy;
            if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
          }
          startIdx = Math.min(bestIdx, arrPts.length - 2);
        }
      }
      const carry = { lastTaxiwayMs: null };
      let sec = 0;
      for (let i = startIdx; i < arrPts.length - 1; i++) {
        const len = pathDist(arrPts[i], arrPts[i + 1]);
        if (len < 1e-9) continue;
        const v = taxiSegmentVelocityMsForPolylineSegment(arrPts[i], arrPts[i + 1], carry);
        sec += len / Math.max(0.1, v);
      }
      vttArrMin = sec / 60;
    }
    f.__schedVttArrMin = vttArrMin;
    f.__schedVttArrRev = rev;
    return vttArrMin;
  }
  function getArrRotMinutes(f) {
    const rotSec = f && f.arrRotSec;
    return (rotSec != null && isFinite(rotSec) && rotSec >= 0) ? rotSec / 60 : 0;
  }
  function getBaseVttDepMinutes(f) {
    const depPts = (typeof getPathForFlightDeparture === 'function') ? getPathForFlightDeparture(f) : null;
    if (!depPts || depPts.length < 2) return 0;
    const carry = { lastTaxiwayMs: null };
    let sec = 0;
    for (let i = 0; i < depPts.length - 1; i++) {
      const len = pathDist(depPts[i], depPts[i + 1]);
      if (len < 1e-9) continue;
      const v = taxiSegmentVelocityMsForPolylineSegment(depPts[i], depPts[i + 1], carry);
      sec += len / Math.max(0.1, v);
    }
    return sec / 60;
  }
  
  function getBaseVttDepMinutesToLineup(f) {
    const depPts = (typeof graphPathDeparture === 'function') ? graphPathDeparture(f, { onlyToLineup: true }) : null;
    if (!depPts || depPts.length < 2) return 0;
    const carry = { lastTaxiwayMs: null };
    let sec = 0;
    for (let i = 0; i < depPts.length - 1; i++) {
      const len = pathDist(depPts[i], depPts[i + 1]);
      if (len < 1e-9) continue;
      const v = taxiSegmentVelocityMsForPolylineSegment(depPts[i], depPts[i + 1], carry);
      sec += len / Math.max(0.1, v);
    }
    return sec / 60;
  }
  
  function getDepBlockOutMin(f) {
    const taxi = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
    return taxi + SCHED_DEP_ROT_MIN;
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
    if (!Array.isArray(flights)) return;
    const byRwy = {};
    flights.forEach(f => {
      if (!f || f.noWayArr) return;
      const rwy = f.arrRunwayId || (f.token && (f.token.arrRunwayId != null ? f.token.arrRunwayId : f.token.runwayId));
      if (rwy == null || rwy === '') return;
      const sldt = f.sldtMin_d;
      if (sldt == null || !isFinite(sldt)) return;
      if (!byRwy[rwy]) byRwy[rwy] = [];
      byRwy[rwy].push(f);
    });
    Object.keys(byRwy).forEach(function(rwyId) {
      const list = byRwy[rwyId];
      let minS = Infinity;
      let chosen = null;
      list.forEach(function(f) {
        const s = f.sldtMin_d;
        if (s != null && isFinite(s) && s < minS) { minS = s; chosen = f; }
      });
      if (chosen) chosen.eldtMin = chosen.sldtMin_d;
    });
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
    if (!Array.isArray(flights)) return;
    flights.forEach(function(f) { ensureFlightPaths(f); });
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
  function isValidSampledArrRetForFlight(f, retStatsAll) {
    if (!f || f.sampledArrRet == null) return false;
    if (!Array.isArray(retStatsAll) || !retStatsAll.length) return false;
    const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
    const arrDir = resolveArrivalRunwayDirForRetGate(f);
    return retStatsAll.some(function(r) {
      if (!r || !r.exit || r.exit.id !== f.sampledArrRet) return false;
      if (arrRunwayId == null) return true;
      if (!(r.runway && r.runway.id === arrRunwayId)) return false;
      if (arrDir === 'clockwise' || arrDir === 'counter_clockwise') {
        if (!isRunwayExitDirectionAllowed(r.exit, arrDir)) return false;
      }
      return true;
    });
  }
  function sampleArrRetRotForFlightIfNeeded(f, retStatsAll, configByType, forceResample) {
    if (!f) return;
    const rev = state.vttArrCacheRev | 0;
    if (!forceResample && f.__schedRetRotRev === rev && isValidSampledArrRetForFlight(f, retStatsAll)) return;
    if (!forceResample && (f.__schedRetRotRev === undefined || f.__schedRetRotRev === null) &&
        f.sampledArrRet != null && f.arrRetFailed === false && f.arrRotSec != null && isFinite(f.arrRotSec) &&
        isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.__schedRetRotRev = rev;
      return;
    }
    if (f.sampledArrRet != null && !isValidSampledArrRetForFlight(f, retStatsAll)) {
      f.sampledArrRet = null;
      f.arrRetFailed = false;
      f.arrRotSec = null;
    }
    const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
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
    const arrDir = resolveArrivalRunwayDirForRetGate(f);
    const candidates = retStatsAll.filter(function(r) {
      if (!(r && r.runway && r.runway.id === arrRunwayId && r.exit)) return false;
      if (arrDir === 'clockwise' || arrDir === 'counter_clockwise') {
        return isRunwayExitDirectionAllowed(r.exit, arrDir);
      }
      return true;
    });
    if (!candidates.length) {
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
      const tToRetEntrance = rtRunway.tSec;
      const minExitVel = (typeof chosen.minExitVelocity === 'number' && isFinite(chosen.minExitVelocity) && chosen.minExitVelocity > 0)
        ? Math.min(chosen.minExitVelocity, chosen.maxExitVelocity || chosen.minExitVelocity)
        : 15;
      let tExit = 0;
      if (vAtChosen > minExitVel) {
        tExit = (vAtChosen - minExitVel) / aDecRot;
      }
      f.arrRotSec = tToRetEntrance + tExit;
      f.arrRunwayIdUsed = arrRunwayId;
      f.arrTdDistM = dTd;
      f.arrRetDistM = chosen.distM;
      f.arrVTdMs = v0;
      f.arrVRetInMs = vAtChosen;
      f.arrVRetOutMs = minExitVel;
    } else {
      f.sampledArrRet = null;
      f.arrRetFailed = true;
      f.arrRotSec = null;
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

  function _buildFlightListHeaderHtml() {
    return '' +
      '<table class="flight-schedule-table">' +
      '<thead><tr>' +
        '<th>Reg</th>' +
        '<th class="flight-th-mixed">Airline</th>' +
        '<th class="flight-th-mixed">Flight Num</th>' +
        '<th class="flight-col-s flight-col-s-start">SLDT</th>' +
        '<th class="flight-td-sibt flight-col-s">SIBT</th>' +
        '<th class="flight-col-s">SOBT</th>' +
        '<th class="flight-col-s flight-col-s-last">STOT</th>' +
        '<th class="flight-col-sd flight-col-sd-start">SLDT(d)</th>' +
        '<th class="flight-col-sd">SIBT(d)</th>' +
        '<th class="flight-col-sd">SOBT(d)</th>' +
        '<th class="flight-col-sd flight-col-sd-last">STOT(d)</th>' +
        '<th class="flight-col-e flight-col-e-start">ELDT</th>' +
        '<th class="flight-col-e">EIBT</th>' +
        '<th class="flight-col-e">EOBT</th>' +
        '<th class="flight-col-e">ETOT</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(arr)</th>' +
        '<th class="flight-th-mixed">STT(arr)</th>' +
        '<th class="flight-th-mixed">DTT(arr)</th>' +
        '<th class="flight-col-e flight-col-rot flight-th-mixed">ROT(dep)</th>' +
        '<th class="flight-th-mixed">STT(dep)</th>' +
        '<th class="flight-th-mixed">DTT(dep)</th>' +
        '<th>Aircraft Type</th>' +
        '<th class="flight-th-mixed">Code(ICAO)</th>' +
        '<th>Arr Rw</th>' +
        '<th>Arr RET</th>' +
        '<th>Building</th>' +
        '<th>Apron</th>' +
        '<th>Dep Rw</th>' +
        '<th class="flight-td-del"></th>' +
      '</tr></thead>' +
      '<tbody>';
  }

  function _buildFlightListRowHtml(f, retStatsAll) {
    const arrRunwayId = f.arrRunwayId || (f.token && f.token.runwayId) || null;
    const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
    let sampledRetName = '—';
    if (f.arrRetFailed) sampledRetName = 'Failed';
    else if (f.sampledArrRet != null && retStatsAll && retStatsAll.length) {
      const retInfo = retStatsAll.find(r => r.exit && r.exit.id === f.sampledArrRet);
      sampledRetName = retInfo ? (retInfo.name || 'RET') : 'RET';
    }
    const tArrMin = f.timeMin != null ? f.timeMin : 0;
    const dwell = f.dwellMin != null ? f.dwellMin : 0;
    const tDepMin = tArrMin + dwell;
    const vttArrMin = getBaseVttArrMinutes(f);
    const rotArrMin = getArrRotMinutes(f);
    const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
    const vttDepMinLineup = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : Math.max(0, depBlockOutMin - SCHED_DEP_ROT_MIN);
    const vttDepMinSlot = (typeof getBaseVttDepMinutesToHoldingSlot === 'function') ? getBaseVttDepMinutesToHoldingSlot(f) : vttDepMinLineup;
    const sldtCalc = (f.sldtMin_d != null ? f.sldtMin_d : Math.max(0, tArrMin - vttArrMin - rotArrMin));
    const sldtOrig = f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc;
    const sobtOrig = (f.sobtMin_orig != null) ? f.sobtMin_orig : tDepMin;
    const stotOrig = (f.stotMin_orig != null) ? f.stotMin_orig : (tDepMin + depBlockOutMin);
    const sldtStr = formatMinutesToHHMMSS(f.sldtMin_orig != null ? f.sldtMin_orig : sldtCalc);
    const stotStr = formatMinutesToHHMMSS(stotOrig);
    const sldtStr_d = formatMinutesToHHMMSS(f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
    const sibtStr_d = formatMinutesToHHMMSS(f.sibtMin_d != null ? f.sibtMin_d : tArrMin);
    const sobtStr_d = formatMinutesToHHMMSS(f.sobtMin_d != null ? f.sobtMin_d : tDepMin);
    const stotStr_d = formatMinutesToHHMMSS(f.stotMin_d != null ? f.stotMin_d : stotOrig);
    const eldtMin = f.eldtMin != null ? f.eldtMin : (f.sldtMin_d != null ? f.sldtMin_d : sldtOrig);
    const etotCandMin = f.etotMin != null ? f.etotMin : (f.stotMin_d != null ? f.stotMin_d : stotOrig);
    f.eldtMin = eldtMin;
    const tArr = formatMinutesToHHMMSS(tArrMin);
    const tDep = formatMinutesToHHMMSS(tDepMin);
    const vttADelayMin = f.vttADelayMin != null ? f.vttADelayMin : 0;
    const eibtMin = eldtMin + rotArrMin + vttArrMin + vttADelayMin;
    f.eibtMin = eibtMin;
    applyForwardEobtEtotAndDepTaxiDelay(f, eibtMin, etotCandMin);
    const eobtMin = f.eobtMin != null ? f.eobtMin : (f.etotMin != null ? f.etotMin - depBlockOutMin : 0);
    const etotMin = f.etotMin != null ? f.etotMin : (eobtMin + depBlockOutMin);
    if (f.sobtMin_orig == null) {
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.eldtMin_orig = eldtMin;
      f.eibtMin_orig = eibtMin;
      f.eobtMin_orig = eobtMin;
      f.etotMin_orig = etotMin;
    }
    const eldtStr = formatMinutesToHHMMSS(eldtMin);
    const etotStr = formatMinutesToHHMMSS(etotMin);
    const eibtStr = formatMinutesToHHMMSS(eibtMin);
    const eobtStr = formatMinutesToHHMMSS(eobtMin);
    const vttArrStr = formatMinutesToHHMMSS(vttArrMin);
    const vttADelayStr = formatMinutesToHHMMSS(vttADelayMin);
    const vttDepStr = formatMinutesToHHMMSS(vttDepMinSlot);
    const depRotSecVal = (typeof computeDepRotSecondsForFlight === 'function') ? computeDepRotSecondsForFlight(f) : Math.max(0, Number(SCHED_DEP_ROT_MIN) || 2) * 60;
    const depRotStr = formatTotalSecondsToHHMMSS(depRotSecVal);
    const depTaxiDelayStr = formatSignedMinutesToHHMMSS(f.depTaxiDelayMin != null ? f.depTaxiDelayMin : 0);
    const depRunwayId = f.depRunwayId || (f.token && f.token.depRunwayId);
    const termId = f.terminalId || (f.token && f.token.terminalId);
    const arrRwRead = escapeHtml(getRunwayDisplayLabelById(arrRunwayId));
    const buildingRead = escapeHtml(getTerminalDisplayLabelById(termId));
    const depRwRead = escapeHtml(getRunwayDisplayLabelById(depRunwayId));
    const noWayBadge = (f.noWayArr || f.noWayDep)
      ? ' <span class="flight-no-way-badge" style="color:#dc2626;font-weight:600;font-size:10px;cursor:help;" title="' + escapeAttr(buildNoWayTooltip(f)) + '">⚠ No Way</span>'
      : '';
    const aircraftTypeLabel = ac ? (ac.name || ac.id || '') : (f.aircraftType || '—');
    const codeIcao = (ac && ac.icao) ? ac.icao : (f.code || '—');
    const arrRetFailedBadge = (f.arrRetFailed || sampledRetName === 'Failed') ? ' <span style="color:#dc2626;font-weight:600;font-size:10px;">⚠ Failed</span>' : '';
    return '' +
      '<tr class="flight-data-row obj-item" data-id="' + f.id + '">' +
        '<td class="flight-td-reg">' + escapeHtml(f.reg || '') + noWayBadge + arrRetFailedBadge + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.airlineCode || '') + '</td>' +
        '<td class="flight-td-reg">' + escapeHtml(f.flightNumber || '') + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-start">' + sldtStr + '</td>' +
        '<td class="flight-td-time flight-td-sibt flight-col-s">' + tArr + '</td>' +
        '<td class="flight-td-time flight-col-s">' + tDep + '</td>' +
        '<td class="flight-td-time flight-col-s flight-col-s-last">' + stotStr + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-start">' + sldtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd">' + sibtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd">' + sobtStr_d + '</td>' +
        '<td class="flight-td-time flight-col-sd flight-col-sd-last">' + stotStr_d + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-e-start">' + eldtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eibtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + eobtStr + '</td>' +
        '<td class="flight-td-time flight-col-e">' + etotStr + '</td>' +
        '<td class="flight-td-time flight-col-e flight-col-rot">' + (f.arrRotSec != null && isFinite(f.arrRotSec) ? (Math.round(f.arrRotSec) + ' s') : '—') + '</td>' +
        '<td class="flight-td-time">' + vttArrStr + '</td>' +
        '<td class="flight-td-time">' + vttADelayStr + '</td>' +
        '<td class="flight-td-time">' + depRotStr + '</td>' +
        '<td class="flight-td-time">' + vttDepStr + '</td>' +
        '<td class="flight-td-time">' + depTaxiDelayStr + '</td>' +
        '<td>' + escapeHtml(aircraftTypeLabel) + '</td>' +
        '<td>' + escapeHtml(codeIcao) + '</td>' +
        '<td class="flight-td-readonly">' + arrRwRead + '</td>' +
        '<td>' + escapeHtml(sampledRetName) + '</td>' +
        '<td class="flight-td-readonly">' + buildingRead + '</td>' +
        '<td class="flight-td-reg">' + (function() { var st = findStandById(f.standId); return escapeHtml(st ? ((st.name && st.name.trim()) || st.id || '—') : '—'); })() + '</td>' +
        '<td class="flight-td-readonly">' + depRwRead + '</td>' +
        '<td class="flight-td-del"><button type="button" class="obj-item-delete" data-del="' + f.id + '">×</button></td>' +
      '</tr>';
  }

  function _buildFlightListRowsHtml(flightsSorted, retStatsAll) {
    return flightsSorted.map(function(f) {
      return _buildFlightListRowHtml(f, retStatsAll);
    });
  }

  const FLIGHT_LIST_PATH_YIELD_CHUNK = 6;
  const FLIGHT_LIST_ASYNC_PATH_MIN = 8;
  function _renderFlightListDomAndSchedule(flightsSorted, schedFull, dirtySet, standSet, listEl, cfgEl, retStatsAll, domOpt) {
    const skipGanttRefresh = domOpt && domOpt.skipGanttRefresh;
    const headerRow = _buildFlightListHeaderHtml();
    if (schedFull) {
      if (typeof computeScheduledDisplayTimes === 'function') computeScheduledDisplayTimes(state.flights);
      if (typeof computeSeparationAdjustedTimes === 'function') computeSeparationAdjustedTimes();
      pinEarliestEldtToSldtPerRunway(flightsSorted);
    } else {
      if (typeof computeScheduledDisplayTimesIncremental === 'function')
        computeScheduledDisplayTimesIncremental(state.flights, dirtySet, standSet);
    }
    flightsSorted.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : (a.timeMin != null ? a.timeMin : 0)) - (b.sibtMin_d != null ? b.sibtMin_d : (b.timeMin != null ? b.timeMin : 0)));
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
      _flightListMountVirtual(listEl, flightsSorted, retStatsAll, headerRow);
    } else {
      _flightListTeardownVirtual(listEl);
      const dataRows = _buildFlightListRowsHtml(flightsForDom, retStatsAll);
      listEl.innerHTML = headerRow + dataRows.join('') + '</tbody></table>';
