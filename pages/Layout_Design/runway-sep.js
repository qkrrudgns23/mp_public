    const touchedStands = (touchedStandIds instanceof Set) ? touchedStandIds : new Set(touchedStandIds || []);
    const standsToRecompute = new Set();
    touchedStands.forEach(function(sid) { if (sid != null && sid !== '') standsToRecompute.add(sid); });
    const needStep1 = new Set();
    dirty.forEach(function(id) { if (id != null && id !== '') needStep1.add(id); });
    allFlights.forEach(function(f) {
      if (!f || f.noWayArr || f.noWayDep) return;
      if (f.standId && standsToRecompute.has(f.standId)) needStep1.add(f.id);
    });
    allFlights.forEach(function(f) {
      if (!f || !needStep1.has(f.id)) return;
      if (f.noWayArr || f.noWayDep) return;
      f.vttADelayMin = 0;
      const tArrMin = f.timeMin != null ? f.timeMin : 0;
      let dwell = f.dwellMin != null ? f.dwellMin : 0;
      let minDwell = f.minDwellMin != null ? f.minDwellMin : 0;
      dwell = Math.max(SCHED_DWELL_FLOOR_MIN, dwell);
      minDwell = Math.max(SCHED_DWELL_FLOOR_MIN, minDwell);
      if (minDwell > dwell) minDwell = dwell;
      f.dwellMin = dwell;
      f.minDwellMin = minDwell;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      const sldtOrig = Math.max(0, tArrMin - vttArrMin - rotArrMin);
      const sobtOrig = tArrMin + dwell;
      const stotOrig = sobtOrig + depBlockOutMin;
      f.sldtMin_orig = sldtOrig;
      f.sibtMin_orig = tArrMin;
      f.sobtMin_orig = sobtOrig;
      f.stotMin_orig = stotOrig;
      f.sldtMin_d = f.sldtMin_orig;
      f.sibtMin_d = tArrMin;
      f.sobtMin_d = sobtOrig;
      f.stotMin_d = stotOrig;
    });
    standsToRecompute.forEach(function(standId) {
      const list = allFlights.filter(function(f) {
        return f && !f.noWayArr && !f.noWayDep && f.standId === standId;
      });
      list.sort((a, b) => (a.sibtMin_d != null ? a.sibtMin_d : 0) - (b.sibtMin_d != null ? b.sibtMin_d : 0));
      let prevSOBT = -1e9;
      list.forEach(function(f) {
        const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
        const sibt0 = (f.sibtMin_d != null ? f.sibtMin_d : 0);
        const overlap = Math.max(0, prevSOBT - sibt0);
        f.vttADelayMin = overlap;
        f.sibtMin_d = sibt0 + overlap;
        const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
        const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
        const minSobtByDwell = f.sibtMin_d + minDwell;
        const sobtCandidate = (f.sobtMin_d != null ? f.sobtMin_d : (f.sibtMin_d + dwell));
        f.sobtMin_d = Math.max(sobtCandidate, minSobtByDwell);
        f.stotMin_d = f.sobtMin_d + depBlockOutMin;
        prevSOBT = f.sobtMin_d;
      });
    });
    allFlights.forEach(function(f) {
      if (!f || f.noWayArr || f.noWayDep || !f.standId) return;
      if (!standsToRecompute.has(f.standId)) return;
      const dwell = f.dwellMin != null ? f.dwellMin : SCHED_DWELL_FLOOR_MIN;
      const minDwell = f.minDwellMin != null ? f.minDwellMin : SCHED_DWELL_FLOOR_MIN;
      const sibt = (f.sibtMin_d != null ? f.sibtMin_d : (f.sibtMin_orig != null ? f.sibtMin_orig : 0));
      const minSobtByDwell = sibt + minDwell;
      const sobtCurrent = (f.sobtMin_d != null ? f.sobtMin_d : (sibt + dwell));
      if (sobtCurrent < minSobtByDwell) {
        const delta = minSobtByDwell - sobtCurrent;
        f.sobtMin_d = minSobtByDwell;
        if (typeof f.stotMin_d === 'number') f.stotMin_d += delta;
      }
    });
    allFlights.forEach(function(f) {
      if (!f || f.noWayArr || f.noWayDep) return;
      const onTouched = f.standId && standsToRecompute.has(f.standId);
      if (!needStep1.has(f.id) && !onTouched) return;
      f.sldtMin = f.sldtMin_d;
      f.stotMin = f.stotMin_d;
      f.sobtMin = f.sobtMin_d;
    });
  }

  function rsepGetSec(val) {
    const n = Number(val);
    return isFinite(n) && n >= 0 ? n : RSEP_MISSING_MATRIX_SEC;
  }

  function rsepApplySeparationToEvents(events, cfg) {
    const arrArr = (cfg.seqData && cfg.seqData['ARR→ARR']) ? cfg.seqData['ARR→ARR'] : {};
    const depDep = (cfg.seqData && cfg.seqData['DEP→DEP']) ? cfg.seqData['DEP→DEP'] : {};
    const depArr = (cfg.seqData && cfg.seqData['DEP→ARR']) ? cfg.seqData['DEP→ARR'] : {};
    const rot = (cfg.rot) ? cfg.rot : {};
    const getSec = rsepGetSec;
    events.sort((a, b) => a.time - b.time || a.index - b.index);
    let lastArrETime = -1e9, lastArrCat = null;
    let lastDepETime = -1e9, lastDepCat = null;
    events.forEach(ev => {
      if (ev.type === 'arr') {
        let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec((arrArr[lastArrCat] && arrArr[lastArrCat][ev.cat]) != null ? arrArr[lastArrCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
        let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec(depArr[ev.cat]) / 60 : -1e9;
        const eTime = Math.max(ev.time, minFromArr, minFromDep);
        ev.flight.eldtMin = eTime;
        lastArrETime = eTime;
        lastArrCat = ev.cat;
      } else {
        let minFromArr = lastArrETime >= -1e8 && lastArrCat ? lastArrETime + getSec(rot[lastArrCat]) / 60 : -1e9;
        let minFromDep = lastDepETime >= -1e8 && lastDepCat ? lastDepETime + getSec((depDep[lastDepCat] && depDep[lastDepCat][ev.cat]) != null ? depDep[lastDepCat][ev.cat] : RSEP_MISSING_MATRIX_SEC) / 60 : -1e9;
        const etotSep = Math.max(ev.time, minFromArr, minFromDep);
        const vttADelay = ev.flight.vttADelayMin != null ? ev.flight.vttADelayMin : 0;
        const rotM = (ev.rotArrMin != null && isFinite(ev.rotArrMin)) ? ev.rotArrMin : getArrRotMinutes(ev.flight);
        const eibtMin = (ev.flight.eldtMin != null ? ev.flight.eldtMin : 0) + rotM + (ev.vttArrMin || 0) + vttADelay;
        const vttDep = ev.vttDepMin || 0;
        const etotMin = etotSep;
        const eobtMin = etotMin - vttDep;
        ev.flight.etotMin = etotMin;
        lastDepETime = etotMin;
        lastDepCat = ev.cat;
      }
    });
    let minT = Infinity, maxT = -Infinity;
    events.forEach(ev => {
      const s = ev.time;
      const e = ev.type === 'arr'
        ? (ev.flight && ev.flight.eldtMin != null ? ev.flight.eldtMin : s)
        : (ev.flight && ev.flight.etotMin != null ? ev.flight.etotMin : s);
      if (s < minT) minT = s;
      if (e < minT) minT = e;
      if (s > maxT) maxT = s;
      if (e > maxT) maxT = e;
    });
    if (!isFinite(minT) || !isFinite(maxT)) { minT = 0; maxT = 60; } else if (maxT <= minT) { maxT = minT + 60; }
    return { minT, maxT };
  }

  function rsepCollectEventsForRunway(rwy, flights, runways) {
    const cfg = rsepGetConfigForRunway(rwy);
    if (!cfg) return null;
    const stdKey = cfg.standard || 'ICAO';
    const events = [];
    let eventIndex = 0;
    flights.forEach((f, flightIdx) => {
      if (f.noWayArr || f.noWayDep) return;
      let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
      let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
      if (arrRwy == null && depRwy == null && runways.length === 1) { arrRwy = rwy.id; depRwy = rwy.id; }
      else if (depRwy == null && arrRwy === rwy.id) depRwy = rwy.id;
      else if (arrRwy == null && depRwy === rwy.id) arrRwy = rwy.id;
      if (arrRwy !== rwy.id && depRwy !== rwy.id) return;
      const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(f.aircraftType) : null;
      const cat = stdKey === 'ICAO' ? (ac && ac.icaoJHL ? ac.icaoJHL : 'M') : (ac && ac.recatEu ? ac.recatEu : 'D');
      const sldtMin_d = f.sldtMin_d != null ? f.sldtMin_d : 0;
      const stotMin_d = f.stotMin_d != null ? f.stotMin_d : 0;
      const sobtMin_d = f.sobtMin_d != null ? f.sobtMin_d : 0;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const vttDepMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
      if (arrRwy === rwy.id) events.push({ time: sldtMin_d, type: 'arr', flight: f, cat: cat, vttArrMin, rotArrMin, index: eventIndex++ });
      if (depRwy === rwy.id) {
        events.push({ time: stotMin_d, type: 'dep', flight: f, cat: cat, vttDepMin, vttArrMin, rotArrMin, sobtMin: sobtMin_d, index: eventIndex++ });
      }
    });
    return { cfg, events };
  }

  function runSeparationPass(runways, flights, byRunway, phase) {
    if (phase === 'initial') {
      runways.forEach(rwy => {
        const pack = rsepCollectEventsForRunway(rwy, flights, runways);
        if (!pack) return;
        const { cfg, events } = pack;
        if (!events.length) {
          byRunway[rwy.id] = { events: [], minT: 0, maxT: 0 };
          return;
        }
        const { minT, maxT } = rsepApplySeparationToEvents(events, cfg);
        byRunway[rwy.id] = { events, minT, maxT };
      });
    } else {
      runways.forEach(rwy => {
        const cfg = rsepGetConfigForRunway(rwy);
        if (!cfg) return;
        const data = byRunway[rwy.id];
        if (!data || !data.events || !data.events.length) return;
        const events = data.events;
        events.forEach(ev => {
          ev.time = ev.type === 'arr'
            ? (ev.flight.eldtMin != null ? ev.flight.eldtMin : ev.time)
            : (ev.flight.etotMin != null ? ev.flight.etotMin : ev.time);
        });
        const { minT, maxT } = rsepApplySeparationToEvents(events, cfg);
        byRunway[rwy.id] = { events, minT, maxT };
      });
    }
  }

  function buildRunwaySeparationTimelineByRunwaySnapshot(flights) {
    const snapGen = state.rwySepSnapshotStaleGen | 0;
    if (state.__rwySepSnapCacheGen === snapGen && state.__rwySepSnapCache) return state.__rwySepSnapCache;
    const list = flights || state.flights || [];
    const runwaysRaw = (state.taxiways || []).filter(t => t.pathType === 'runway');
    if (!runwaysRaw.length) {
      state.__rwySepSnapCache = {};
      state.__rwySepSnapCacheGen = snapGen;
      return state.__rwySepSnapCache;
    }
    const runways = (function() {
      const idToIndex = {};
      runwaysRaw.forEach((r, i) => { if (r && r.id != null) idToIndex[r.id] = i; });
      const n = runwaysRaw.length;
      const indeg = new Array(n).fill(0);
      const adj = new Array(n).fill(0).map(() => []);
      list.forEach(f => {
        if (!f) return;
        let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
        let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
        if (!arrRwy || !depRwy || arrRwy === depRwy) return;
        const ai = idToIndex[arrRwy];
        const di = idToIndex[depRwy];
        if (ai == null || di == null) return;
        adj[ai].push(di);
        indeg[di] += 1;
      });
      const q = [];
      for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i);
      const orderIdx = [];
      while (q.length) {
        const i = q.shift();
        orderIdx.push(i);
        adj[i].forEach(j => {
          indeg[j] -= 1;
          if (indeg[j] === 0) q.push(j);
        });
      }
      if (orderIdx.length !== n) return runwaysRaw;
      return orderIdx.map(i => runwaysRaw[i]);
    })();
    const byRunway = {};
    runways.forEach(rwy => {
      const pack = rsepCollectEventsForRunway(rwy, list, runways);
      if (!pack || !pack.events.length) {
        byRunway[rwy.id] = { events: [], minT: 0, maxT: 0 };
        return;
      }
      const events = pack.events.slice().sort((a, b) => a.time - b.time || a.index - b.index);
      let minT = Infinity, maxT = -Infinity;
      events.forEach(ev => {
        const s = ev.time;
        const f = ev.flight;
        const e = ev.type === 'arr'
          ? (f && f.eldtMin != null && isFinite(f.eldtMin) ? f.eldtMin : s)
          : (f && f.etotMin != null && isFinite(f.etotMin) ? f.etotMin : s);
        if (s < minT) minT = s;
        if (e < minT) minT = e;
        if (s > maxT) maxT = s;
        if (e > maxT) maxT = e;
      });
      if (!isFinite(minT) || !isFinite(maxT)) { minT = 0; maxT = 60; } else if (maxT <= minT) maxT = minT + 60;
      byRunway[rwy.id] = { events, minT, maxT };
    });
    state.__rwySepSnapCache = byRunway;
    state.__rwySepSnapCacheGen = snapGen;
    return byRunway;
  }

  function computeSeparationAdjustedTimes() {
    const flights = state.flights || [];
    flights.forEach(f => { delete f.eldtMin; delete f.etotMin; });
    const runwaysRaw = (state.taxiways || []).filter(t => t.pathType === 'runway');
    if (!runwaysRaw.length) return {};

    const runways = (function() {
      const idToIndex = {};
      runwaysRaw.forEach((r, i) => { if (r && r.id != null) idToIndex[r.id] = i; });
      const n = runwaysRaw.length;
      const indeg = new Array(n).fill(0);
      const adj = new Array(n).fill(0).map(() => []);
      flights.forEach(f => {
        if (!f) return;
        let arrRwy = f.arrRunwayId || (f.token && f.token.runwayId);
        let depRwy = f.depRunwayId || (f.token && f.token.depRunwayId);
        if (!arrRwy || !depRwy || arrRwy === depRwy) return;
        const ai = idToIndex[arrRwy];
        const di = idToIndex[depRwy];
        if (ai == null || di == null) return;
        adj[ai].push(di);
        indeg[di] += 1;
      });
      const q = [];
      for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i);
      const orderIdx = [];
      while (q.length) {
        const i = q.shift();
        orderIdx.push(i);
        adj[i].forEach(j => {
          indeg[j] -= 1;
          if (indeg[j] === 0) q.push(j);
        });
      }
      if (orderIdx.length !== n) return runwaysRaw;
      return orderIdx.map(i => runwaysRaw[i]);
    })();

    const byRunway = {};
    runSeparationPass(runways, flights, byRunway, 'initial');
    flights.forEach(f => {
      if (f.noWayArr || f.noWayDep) return;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const vttADelay = f.vttADelayMin != null ? f.vttADelayMin : 0;
      f.eibtMin = (f.eldtMin != null ? f.eldtMin : 0) + rotArrMin + vttArrMin + vttADelay;
      applyForwardEobtEtotAndDepTaxiDelay(f, f.eibtMin, null);
    });
    const standToFlightsE = {};
    flights.forEach(f => { if (f && !f.noWayArr && !f.noWayDep) f.eOverlapPushed = false; });
    flights.forEach(f => {
      if (f.noWayArr || f.noWayDep || !f.standId) return;
      const sid = f.standId;
      if (!standToFlightsE[sid]) standToFlightsE[sid] = [];
      standToFlightsE[sid].push(f);
    });
    Object.keys(standToFlightsE).forEach(standId => {
      const list = standToFlightsE[standId];
      list.sort((a, b) => (a.eibtMin != null ? a.eibtMin : 0) - (b.eibtMin != null ? b.eibtMin : 0));
      let prevEOBT = -1e9;
      list.forEach(f => {
        const depBlockOutMin = (typeof getDepBlockOutMin === 'function') ? getDepBlockOutMin(f) : 0;
        const vttArrMin = getBaseVttArrMinutes(f);
        const rotArrMin = getArrRotMinutes(f);
        const vttADelay = f.vttADelayMin != null ? f.vttADelayMin : 0;
        const eibtMin = f.eibtMin != null ? f.eibtMin : 0;
        const overlap = Math.max(0, prevEOBT - eibtMin);
        f.eOverlapPushed = overlap > 0;
        f.eibtMin = eibtMin + overlap;
        const runwayEtotCand = f.etotMin != null ? f.etotMin : ((f.eobtMin != null ? f.eobtMin : f.eibtMin) + depBlockOutMin);
        applyForwardEobtEtotAndDepTaxiDelay(f, f.eibtMin, runwayEtotCand);
        f.eldtMin = f.eibtMin - rotArrMin - vttArrMin - vttADelay;
        const sldtBase = (f.sldtMin_d != null ? f.sldtMin_d
                         : (f.sldtMin_orig != null ? f.sldtMin_orig : 0));
        if (f.eldtMin < sldtBase) {
          f.eldtMin = sldtBase;
          f.eibtMin = f.eldtMin + rotArrMin + vttArrMin + vttADelay;
          applyForwardEobtEtotAndDepTaxiDelay(f, f.eibtMin, f.etotMin);
        }
        prevEOBT = f.eobtMin;
      });
    });
    runSeparationPass(runways, flights, byRunway, 'refine');
    runways.forEach(rwy => {
      const data = byRunway[rwy.id];
      if (!data || !data.events) return;
      const arrEvs = data.events.filter(e => e.type === 'arr');
      if (!arrEvs.length) return;
      let minSldt = Infinity, earliestArrFlight = null;
      arrEvs.forEach(ev => {
        const sldt = (ev.flight.sldtMin_d != null ? ev.flight.sldtMin_d : (ev.flight.sldtMin_orig != null ? ev.flight.sldtMin_orig : Infinity));
        if (sldt < minSldt) { minSldt = sldt; earliestArrFlight = ev.flight; }
      });
      if (earliestArrFlight) {
        const sldtBase = earliestArrFlight.sldtMin_d != null ? earliestArrFlight.sldtMin_d : (earliestArrFlight.sldtMin_orig != null ? earliestArrFlight.sldtMin_orig : 0);
        earliestArrFlight.eldtMin = sldtBase;
      }
    });
    flights.forEach(f => {
      if (!f || f.noWayArr || f.noWayDep) return;
      const vttArrMin = getBaseVttArrMinutes(f);
      const rotArrMin = getArrRotMinutes(f);
      const vttADelay = f.vttADelayMin != null ? f.vttADelayMin : 0;
      f.eibtMin = (f.eldtMin != null ? f.eldtMin : 0) + rotArrMin + vttArrMin + vttADelay;
      applyForwardEobtEtotAndDepTaxiDelay(f, f.eibtMin, f.etotMin);
    });
    return byRunway;
  }

  function getRunwayPath(runwayId) {
    const taxiways = state.taxiways || [];
    let rw = runwayId ? taxiways.find(t => t.id === runwayId && t.pathType === 'runway' && t.vertices && t.vertices.length >= 2) : null;
    if (!rw) rw = taxiways.find(t => t.pathType === 'runway' && t.vertices && t.vertices.length >= 2);
    if (!rw || !rw.vertices.length) return null;
    const pts = rw.vertices.map(v => cellToPixel(v.col, v.row));
    const sp = rw.start_point, ep = rw.end_point;
    if (sp && ep) {
