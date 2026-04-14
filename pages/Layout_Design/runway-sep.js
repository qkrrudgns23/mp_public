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
