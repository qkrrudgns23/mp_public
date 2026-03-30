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
      const startPx = cellToPixel(sp.col, sp.row);
      const endPx = cellToPixel(ep.col, ep.row);
      if (dist2(pts[pts.length-1], startPx) < dist2(pts[0], startPx)) pts.reverse();
    }
    return { startPx: pts[0], endPx: pts[pts.length-1], pts };
  }

  function getRunwayPointAtDistance(runwayId, distM) {
    const path = getRunwayPath(runwayId);
    if (!path || !path.pts || path.pts.length < 2) return null;
    const pts = path.pts;
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const segLen = pathDist(p1, p2);
      if (!(segLen > 1e-6)) continue;
      if (acc + segLen >= distM) {
        const t = Math.max(0, Math.min(1, (distM - acc) / segLen));
        return [
          p1[0] + (p2[0] - p1[0]) * t,
          p1[1] + (p2[1] - p1[1]) * t
        ];
      }
      acc += segLen;
    }
    return pts[pts.length - 1];
  }

  function flightEMinutesPrefer(f, keys, fallback) {
    for (let ki = 0; ki < keys.length; ki++) {
      const v = f[keys[ki]];
      if (typeof v === 'number' && isFinite(v)) return v;
    }
    return fallback;
  }
  function touchdownDistMForTimeline(f) {
    if (typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM) && f.arrTdDistM >= 0) return f.arrTdDistM;
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f.aircraftType) : null;
    const z = ac && typeof ac.touchdown_zone_avg_m === 'number' ? ac.touchdown_zone_avg_m : null;
    if (typeof z === 'number' && z > 0) return z;
    return 400;
  }
  function touchdownSpeedMsForTimeline(f) {
    let v = f.arrVTdMs;
    if (typeof v === 'number' && isFinite(v) && v > 0) return Math.max(1, v);
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f.aircraftType) : null;
    v = ac && typeof ac.touchdown_speed_avg_ms === 'number' ? ac.touchdown_speed_avg_ms : 70;
    return Math.max(1, v);
  }
  
  function getRunwayInboundUxyAtDistance(runwayId, rwDir, distAlong) {
    const r = getRunwayPath(runwayId);
    const anchor = getRunwayPointAtDistance(runwayId, distAlong);
    if (!r || !r.pts || r.pts.length < 2 || !anchor) return null;
    const pts = r.pts;
    let segIdx = Math.max(0, pts.length - 2);
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const segLen = pathDist(pts[i], pts[i + 1]);
      if (segLen < 1e-9) continue;
      if (acc + segLen >= distAlong - 1e-6) { segIdx = i; break; }
      acc += segLen;
    }
    const p1 = pts[segIdx], p2 = pts[segIdx + 1];
    const segLen = pathDist(p1, p2) || 1;
    let ux = (p2[0] - p1[0]) / segLen, uy = (p2[1] - p1[1]) / segLen;
    if (rwDir === 'counter_clockwise') { ux = -ux; uy = -uy; }
    return { td: anchor, ux, uy };
  }
  
  function buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDistAlong, totalM) {
    const ax = getRunwayInboundUxyAtDistance(runwayId, rwDir, anchorDistAlong);
    if (!ax) return null;
    const td = ax.td, ux = ax.ux, uy = ax.uy;
    const tm = Math.max(0, Number(totalM) || 0);
    const tdxy = [td[0], td[1]];
    if (tm < 1e-6) return { pts: [tdxy, tdxy], pathLen: 0 };
    const outer = [td[0] - ux * tm, td[1] - uy * tm];
    return { pts: [outer, tdxy], pathLen: pathDist(outer, tdxy) };
  }
  
  function arrivalApproachAnchorDistM(runwayId, tdDistAlong) {
    let anchorDist = runwayApproachThresholdDistAlongM(runwayId, tdDistAlong);
    if (!(typeof anchorDist === 'number' && isFinite(anchorDist) && anchorDist >= 0)) anchorDist = tdDistAlong;
    else if (anchorDist > tdDistAlong + 1e-3) anchorDist = tdDistAlong;
    return anchorDist;
  }
  function arrivalApproachDurationSecBeforeEldt(f) {
    const vTd = Math.max(1, touchdownSpeedMsForTimeline(f));
    const token = f.token || {};
    const runwayId = f.arrRunwayIdUsed || token.arrRunwayId || token.runwayId || f.arrRunwayId;
    if (runwayId == null || runwayId === '') return APPROACH_OFFSET_WORLD_M / vTd;
    const rwDir = String(f.arrRunwayDirUsed || 'clockwise');
    const tdDist = touchdownDistMForTimeline(f);
    const anchorDist = arrivalApproachAnchorDistM(runwayId, tdDist);
    const pack = buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDist, APPROACH_OFFSET_WORLD_M);
    const rsPt = getRunwayPointAtDistance(runwayId, anchorDist);
    const tdPt = getRunwayPointAtDistance(runwayId, tdDist);
    if (pack && pack.pathLen > 1e-9) {
      let totalLen = pack.pathLen;
      if (rsPt && tdPt) totalLen += pathDist(rsPt, tdPt);
      return totalLen / vTd;
    }
    if (!tdPt) return APPROACH_OFFSET_WORLD_M / vTd;
    const apprPt = approachPointBeforeThresholdJs(runwayId, rwDir, APPROACH_OFFSET_WORLD_M, anchorDist);
    let straightLen = pathDist(apprPt, rsPt || tdPt);
    if (rsPt && tdPt) straightLen += pathDist(rsPt, tdPt);
    return straightLen / vTd;
  }
  
  function getFlightAirsideWindowSec(f) {
    if (!f) return null;
    if (f.noWayArr && f.noWayDep) return null;
    if (f.arrDep === 'Dep') {
      const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], flightEMinutesPrefer(f, ['timeMin'], 0) + (typeof f.dwellMin === 'number' ? f.dwellMin : 0));
      const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
      const eobtS = eobtMin * 60;
      const etotS = etotMin * 60;
      const depRotS = Math.max(0, Number(SCHED_DEP_ROT_MIN) || 0) * 60;
      let depMoveStart = eobtS + depRotS;
      if (depMoveStart > etotS) depMoveStart = eobtS;
      return { t0: depMoveStart, t1: etotS };
    }
    const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
    const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
    const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
    const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
    const eldtS = eldtMin * 60;
    const etotS = etotMin * 60;
    const tAppr = arrivalApproachDurationSecBeforeEldt(f);
    if (!isFinite(tAppr) || tAppr < 0) return null;
    const t0 = eldtS - tAppr;
    if (!isFinite(t0) || !isFinite(etotS)) return null;
    return { t0: t0, t1: etotS };
  }
  
  function simAirsideLazyPadSec() {
    return Math.max(90, SIM_TIME_SLIDER_SNAP_SEC + 45);
  }
  function isFlightAirsideActiveAtSimSec(f, tSec) {
    const w = getFlightAirsideWindowSec(f);
    if (!w || !isFinite(Number(tSec))) return false;
    const t = Number(tSec);
    return t >= w.t0 - 1e-3 && t <= w.t1 + 1e-3;
  }
  function isFlightAirsideLazyTimelineBuildEligible(f, tSec) {
    const w = getFlightAirsideWindowSec(f);
    if (!w || !isFinite(Number(tSec))) return false;
    const t = Number(tSec);
    const pad = simAirsideLazyPadSec();
    return t >= w.t0 - pad - 1e-3 && t <= w.t1 + 1e-3;
  }
  function nearestIndexOnPolylineForTd(pts, q) {
    if (!pts || pts.length < 2) return 0;
    let bestI = 0, bestD2 = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const pr = projectOnSegment(pts[i], pts[i + 1], q);
      const d2 = dist2(pr.p, q);
      if (d2 < bestD2) { bestD2 = d2; bestI = i; }
    }
    return bestI;
  }
  function trimPolylineFromNearPoint(pts, nearPt) {
    if (!pts || pts.length < 2) return pts ? pts.slice() : [];
    const idx = nearestIndexOnPolylineForTd(pts, nearPt);
    const a = pts[idx], b = pts[idx + 1];
    const pr = projectOnSegment(a, b, nearPt);
    const t = Math.max(0, Math.min(1, pr.t));
    const start = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    const out = [start];
    for (let j = idx + 1; j < pts.length; j++) out.push([pts[j][0], pts[j][1]]);
    return out.length >= 2 ? out : pts.slice();
  }
  function approachPointBeforeThresholdJs(runwayId, rwDir, offsetWorld, touchdownDistAlong) {
    const r = getRunwayPath(runwayId);
    const td = getRunwayPointAtDistance(runwayId, touchdownDistAlong);
    if (!r || !r.pts || r.pts.length < 2) return td || [0, 0];
    const pts = r.pts;
    let segIdx = Math.max(0, pts.length - 2);
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const segLen = pathDist(pts[i], pts[i + 1]);
      if (segLen < 1e-9) continue;
      if (acc + segLen >= touchdownDistAlong - 1e-6) { segIdx = i; break; }
      acc += segLen;
    }
    const p1 = pts[segIdx], p2 = pts[segIdx + 1];
    const segLen = pathDist(p1, p2) || 1;
    let ux = (p2[0] - p1[0]) / segLen, uy = (p2[1] - p1[1]) / segLen;
    if (rwDir === 'counter_clockwise') { ux = -ux; uy = -uy; }
    return [td[0] - ux * offsetWorld, td[1] - uy * offsetWorld];
  }
  function mergeTimelineSegments(a, b) {
    if (!a || !a.length) return b ? b.slice() : [];
    if (!b || !b.length) return a.slice();
    const out = a.slice();
    const last = out[out.length - 1], first = b[0];
    if (Math.abs(last.t - first.t) < 1e-3 && Math.abs(last.x - first.x) < 0.1) out.pop();
    for (let i = 0; i < b.length; i++) out.push(b[i]);
    return out;
  }
  function polylineTotalLength(pts) {
    if (!pts || pts.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < pts.length - 1; i++) s += pathDist(pts[i], pts[i + 1]);
    return s;
  }
  function polylinePointAtDistance(pts, distAlong) {
    if (!pts || !pts.length) return [0, 0];
    const d = Math.max(0, Number(distAlong) || 0);
    if (d <= 1e-12) return [pts[0][0], pts[0][1]];
    let acc = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const seg = pathDist(a, b);
      if (seg < 1e-9) continue;
      if (acc + seg >= d - 1e-9) {
        const t = Math.max(0, Math.min(1, (d - acc) / seg));
        return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      }
      acc += seg;
    }
    const last = pts[pts.length - 1];
    return [last[0], last[1]];
  }
  function polylineSplitAtDistance(pts, cutDist) {
    if (!pts || pts.length < 2) return { first: pts ? pts.slice() : [], second: [] };
    const cut = Math.max(0, Number(cutDist) || 0);
    if (cut <= 1e-9) return { first: [[pts[0][0], pts[0][1]]], second: pts.slice() };
    let acc = 0;
    const first = [[pts[0][0], pts[0][1]]];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const seg = pathDist(a, b);
      if (seg < 1e-9) continue;
      if (acc + seg >= cut - 1e-9) {
        const t = Math.max(0, Math.min(1, (cut - acc) / seg));
        const px = a[0] + t * (b[0] - a[0]), py = a[1] + t * (b[1] - a[1]);
        if (dist2(first[first.length - 1], [px, py]) > 1e-8) first.push([px, py]);
        const second = [[px, py]];
        for (let j = i + 1; j < pts.length; j++) second.push([pts[j][0], pts[j][1]]);
        return { first: dedupePathPoints(first), second: dedupePathPoints(second) };
      }
      acc += seg;
      if (dist2(first[first.length - 1], b) > 1e-8) first.push([b[0], b[1]]);
    }
    return { first: dedupePathPoints(first), second: [[pts[pts.length - 1][0], pts[pts.length - 1][1]]] };
  }
  function runwayDistanceAtElapsedSec(tau, v0, a, vFloorIn, distM) {
    const vf0 = Math.max(1, Math.min(150, vFloorIn));
    const vf = Math.min(vf0, v0);
    if (!(tau > 0)) return 0;
    if (!(a > 0) || distM <= 0) return Math.min(distM, v0 * tau);
    if (v0 <= vf) return Math.min(distM, v0 * tau);
    const dStop = (v0 * v0 - vf * vf) / (2 * a);
    if (distM < dStop) {
      const vEnd = Math.sqrt(Math.max(0, v0 * v0 - 2 * a * distM));
      const tFull = (v0 - vEnd) / a;
      const t = Math.min(tau, tFull);
      const s = v0 * t - 0.5 * a * t * t;
      return Math.min(distM, s);
    }
    const tDecel = (v0 - vf) / a;
    if (tau <= tDecel) return Math.min(distM, v0 * tau - 0.5 * a * tau * tau);
    const sDecel = dStop;
    const s = sDecel + vf * (tau - tDecel);
    return Math.min(distM, s);
  }
  function runwayPhysicsTimelineScaled(pts, distM, tStart, tEnd, v0, a, vFloorIn) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9 || distM <= 1e-9) {
      const p = pts && pts.length ? polylinePointAtDistance(pts, 0) : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }, { t: tEnd, x: p[0], y: p[1] }];
    }
    const phy = runwayArrSpeedAndTimeToRet(v0, a, distM, vFloorIn);
    const phyT = Math.max(1e-6, phy.tSec);
    const n = Math.max(6, Math.min(24, Math.ceil(distM / 40)));
    const tl = [];
    for (let i = 0; i <= n; i++) {
      const u = i / n;
      const t = tStart + u * (tEnd - tStart);
      const tauPhy = u * phyT;
      const s = Math.min(distM, runwayDistanceAtElapsedSec(tauPhy, v0, a, vFloorIn, distM));
      const pt = polylinePointAtDistance(pts, s);
      tl.push({ t: t, x: pt[0], y: pt[1] });
    }
    tl[0].t = tStart;
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  function aircraftDecelMs2ForTimeline(f) {
    const ac = (typeof getAircraftInfoByType === 'function') ? getAircraftInfoByType(f && f.aircraftType) : null;
    const a = ac && typeof ac.deceleration_avg_ms2 === 'number' ? ac.deceleration_avg_ms2 : null;
    if (typeof a === 'number' && isFinite(a) && a > 0.05) return Math.min(5, Math.max(0.05, a));
    return 1.2;
  }
  function nearestTaxiInfraD2ForMidpoint(mid) {
    let bestApronD2 = Infinity;
    let bestTaxiD2 = Infinity;
    let bestTw = null;
    const apronList = state.apronLinks || [];
    for (let ai = 0; ai < apronList.length; ai++) {
      const poly = getApronLinkPolylineWorldPts(apronList[ai]);
      if (!poly || poly.length < 2) continue;
      for (let j = 0; j < poly.length - 1; j++) {
        const pr = projectOnSegment(poly[j], poly[j + 1], mid);
        const d2 = dist2(pr.p, mid);
        if (d2 < bestApronD2) bestApronD2 = d2;
      }
    }
    const list = state.taxiways || [];
    for (let ti = 0; ti < list.length; ti++) {
      const tw = list[ti];
      const ot = getOrderedPoints(tw);
      if (!ot || ot.length < 2) continue;
      for (let j = 0; j < ot.length - 1; j++) {
        const pr = projectOnSegment(ot[j], ot[j + 1], mid);
        const d2 = dist2(pr.p, mid);
        if (d2 < bestTaxiD2) { bestTaxiD2 = d2; bestTw = tw; }
      }
    }
    return { bestApronD2, bestTaxiD2, bestTw };
  }
  function taxiHitFromMidpoint(mid) {
    const { bestApronD2, bestTaxiD2, bestTw } = nearestTaxiInfraD2ForMidpoint(mid);
    const hasA = bestApronD2 < Infinity;
    const hasT = bestTaxiD2 < Infinity;
    if (hasA && (!hasT || bestApronD2 <= bestTaxiD2)) return { kind: 'apron' };
    if (hasT && bestTw) return { kind: 'tw', tw: bestTw };
    return { kind: 'tw', tw: null };
  }
  function taxiSegmentVelocityMsFromHit(hit, carry) {
    const fallback = getTaxiwayAvgMoveVelocityForPath(null);
    if (hit.kind === 'apron') return Math.max(0.1, APRON_TAXIWAY_SPEED_MS);
    const tw = hit.tw;
    if (!tw) return Math.max(1, fallback);
    const pt = tw.pathType || 'taxiway';
    if (pt === 'runway_exit') {
      const v = carry.lastTaxiwayMs;
      return Math.max(1, (typeof v === 'number' && v > 0) ? v : fallback);
    }
    if (pt === 'taxiway') {
      const v = getTaxiwayAvgMoveVelocityForPath(tw);
      carry.lastTaxiwayMs = v;
      return Math.max(1, v);
    }
    if (pt === 'runway') return Math.max(1, getTaxiwayAvgMoveVelocityForPath(tw));
    return Math.max(1, getTaxiwayAvgMoveVelocityForPath(tw));
  }
  function taxiSegmentVelocityMsForPolylineSegment(p1, p2, carry) {
    const mx = (p1[0] + p2[0]) * 0.5, my = (p1[1] + p2[1]) * 0.5;
    const hit = taxiHitFromMidpoint([mx, my]);
    return taxiSegmentVelocityMsFromHit(hit, carry);
  }
  function makeTaxiSegmentVelocityCallback() {
    const carry = { lastTaxiwayMs: null };
    return function(i, a, b) { return taxiSegmentVelocityMsForPolylineSegment(a, b, carry); };
  }
  function polylineRawDurationSegmentVelocities(pts, velForSeg) {
    if (!pts || pts.length < 2) return 0;
    let total = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const len = pathDist(pts[i], pts[i + 1]);
      if (len < 1e-9) continue;
      const v = Math.max(1, velForSeg(i, pts[i], pts[i + 1]));
      total += len / v;
    }
    return total;
  }
  function polylineTimelineBySegmentSpeeds(pts, tStart, tEnd, velForSeg) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9) {
      const p = pts && pts.length ? pts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }];
    }
    const lengths = [];
    for (let i = 0; i < pts.length - 1; i++) lengths.push(pathDist(pts[i], pts[i + 1]));
    const rawDts = [];
    for (let i = 0; i < lengths.length; i++) {
      const v = Math.max(1, velForSeg(i, pts[i], pts[i + 1]));
      rawDts.push((lengths[i] < 1e-9 ? 0 : lengths[i] / v));
    }
    const rawTotal = rawDts.reduce(function(s, x) { return s + x; }, 0);
    const window = tEnd - tStart;
    if (rawTotal < 1e-9) {
      return [
        { t: tStart, x: pts[0][0], y: pts[0][1] },
        { t: tEnd, x: pts[pts.length - 1][0], y: pts[pts.length - 1][1] },
      ];
    }
    const scale = window / rawTotal;
    const tl = [{ t: tStart, x: pts[0][0], y: pts[0][1] }];
    let acc = 0;
    for (let i = 0; i < lengths.length; i++) {
      acc += rawDts[i] * scale;
      tl.push({ t: Math.min(tStart + acc, tEnd), x: pts[i + 1][0], y: pts[i + 1][1] });
    }
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  function polylineTimelineLinearRetSpeed(pts, tStart, tEnd, vIn, vOut) {
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-9) {
      const p = pts && pts.length ? pts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }];
    }
    const lengths = [];
    let totalLen = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const len = pathDist(pts[i], pts[i + 1]);
      lengths.push(len);
      totalLen += len;
    }
    const rawDts = [];
    let accLen = 0;
    for (let i = 0; i < lengths.length; i++) {
      const midLen = accLen + lengths[i] * 0.5;
      const u = totalLen > 1e-9 ? midLen / totalLen : 0;
      const v = Math.max(1, vIn + (vOut - vIn) * u);
      rawDts.push(lengths[i] < 1e-9 ? 0 : lengths[i] / v);
      accLen += lengths[i];
    }
    const rawTotal = rawDts.reduce(function(s, x) { return s + x; }, 0);
    const window = tEnd - tStart;
    if (rawTotal < 1e-9) {
      return [
        { t: tStart, x: pts[0][0], y: pts[0][1] },
        { t: tEnd, x: pts[pts.length - 1][0], y: pts[pts.length - 1][1] },
      ];
    }
    const scale = window / rawTotal;
    const tl = [{ t: tStart, x: pts[0][0], y: pts[0][1] }];
    let acc = 0;
    for (let i = 0; i < lengths.length; i++) {
      acc += rawDts[i] * scale;
      tl.push({ t: Math.min(tStart + acc, tEnd), x: pts[i + 1][0], y: pts[i + 1][1] });
    }
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  function splitTaxiInPartsForTimeline(f, runwayId, taxiInPts) {
    const vTaxiBase = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    if (!taxiInPts || taxiInPts.length < 2) {
      return {
        vTaxiBase,
        runwayPts: [],
        retPts: [],
        taxiPts: [],
        phyRw: 0,
        phyRet: 0,
        phyTaxi: 0,
        useRwPhy: false,
        runwayLenM: 0,
        vTd: 0,
        aDec: 0,
        vRetIn: 0,
        vRetOut: 0,
        vRetResolved: vTaxiBase,
        carryAfterRunway: { lastTaxiwayMs: null },
      };
    }
    const vTd = touchdownSpeedMsForTimeline(f);
    let vRetIn = typeof f.arrVRetInMs === 'number' && isFinite(f.arrVRetInMs) && f.arrVRetInMs > 0 ? f.arrVRetInMs : getMinArrVelocityMpsForRunwayId(runwayId);
    let vRetOut = typeof f.arrVRetOutMs === 'number' && isFinite(f.arrVRetOutMs) && f.arrVRetOutMs > 0 ? f.arrVRetOutMs : vTaxiBase;
    const aDec = aircraftDecelMs2ForTimeline(f);
    let runwayLenM = 0;
    if (typeof f.arrRetDistM === 'number' && isFinite(f.arrRetDistM) && typeof f.arrTdDistM === 'number' && isFinite(f.arrTdDistM)) {
      runwayLenM = Math.abs(f.arrRetDistM - f.arrTdDistM);
    }
    const totalInLen = polylineTotalLength(taxiInPts);
    runwayLenM = Math.min(runwayLenM, Math.max(0, totalInLen));
    const splitRw = polylineSplitAtDistance(taxiInPts, runwayLenM);
    const runwayPts = splitRw.first;
    const afterRw = splitRw.second;
    let retLenM = 0;
    if (f.sampledArrRet) {
      const retTw = (state.taxiways || []).find(function(t) { return t.id === f.sampledArrRet; });
      const rPts = retTw ? getOrderedPoints(retTw) : null;
      if (rPts && rPts.length >= 2) {
        retLenM = polylineTotalLength(rPts);
        const remLen = polylineTotalLength(afterRw);
        retLenM = Math.min(retLenM, Math.max(0, remLen));
      }
    }
    const splitRet = polylineSplitAtDistance(afterRw, retLenM);
    const retPts = splitRet.first;
    const taxiPts = splitRet.second;
    const useRwPhy = runwayLenM > 1 && runwayPts.length >= 2;
    let phyRw = 0;
    if (useRwPhy) {
      phyRw = runwayArrSpeedAndTimeToRet(vTd, aDec, runwayLenM, vRetIn).tSec;
    } else if (runwayPts.length >= 2) {
      phyRw = polylineTotalLength(runwayPts) / vTaxiBase;
    }
    const carryRw = { lastTaxiwayMs: null };
    if (runwayPts.length >= 2) {
      for (let ri = 0; ri < runwayPts.length - 1; ri++) {
        taxiSegmentVelocityMsForPolylineSegment(runwayPts[ri], runwayPts[ri + 1], carryRw);
      }
    }
    const vFallback = getTaxiwayAvgMoveVelocityForPath(null);
    const vRetResolved = (typeof carryRw.lastTaxiwayMs === 'number' && carryRw.lastTaxiwayMs > 0)
      ? carryRw.lastTaxiwayMs
      : vFallback;
    const retPathLen = polylineTotalLength(retPts);
    const phyRet = (retPts.length >= 2 && retPathLen > 1e-3) ? retPathLen / Math.max(1, vRetResolved) : 0;
    const carryTaxi = { lastTaxiwayMs: carryRw.lastTaxiwayMs };
    const phyTaxi = taxiPts.length >= 2
      ? polylineRawDurationSegmentVelocities(taxiPts, function(i, a, b) {
          return taxiSegmentVelocityMsForPolylineSegment(a, b, carryTaxi);
        })
      : 0;
    return {
      vTaxiBase, runwayPts, retPts, taxiPts, phyRw, phyRet, phyTaxi, useRwPhy, runwayLenM, vTd, aDec, vRetIn, vRetOut,
      vRetResolved, carryAfterRunway: { lastTaxiwayMs: carryRw.lastTaxiwayMs },
    };
  }
  
  function buildRunwayAndRetTimelineInWindow(f, runwayId, taxiInPts, tStart, tEnd) {
    const parts = splitTaxiInPartsForTimeline(f, runwayId, taxiInPts);
    const vTaxiBase = parts.vTaxiBase;
    const runwayPts = parts.runwayPts;
    const retPts = parts.retPts;
    const phyRw = parts.phyRw;
    const phyRet = parts.phyRet;
    const useRwPhy = parts.useRwPhy;
    const runwayLenM = parts.runwayLenM;
    const vTd = parts.vTd;
    const aDec = parts.aDec;
    const vRetIn = parts.vRetIn;
    const vRetOut = parts.vRetOut;
    const vRetResolved = Math.max(1, parts.vRetResolved != null ? parts.vRetResolved : vTaxiBase);
    if (!taxiInPts || taxiInPts.length < 2 || tEnd <= tStart + 1e-6) {
      const p = taxiInPts && taxiInPts.length ? taxiInPts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }, { t: tEnd, x: p[0], y: p[1] }];
    }
    const window = Math.max(1e-6, tEnd - tStart);
    const rawSum = phyRw + phyRet;
    if (rawSum < 1e-9) {
      return polylineSpeedScaledToWindow(runwayPts.length >= 2 ? runwayPts : taxiInPts, tStart, tEnd, vTaxiBase);
    }
    const scale = window / rawSum;
    let tCur = tStart;
    let merged = null;
    if (runwayPts.length >= 2 && (useRwPhy ? runwayLenM > 1 : phyRw > 1e-9)) {
      const tSegEnd = tCur + phyRw * scale;
      const seg = useRwPhy
        ? runwayPhysicsTimelineScaled(runwayPts, runwayLenM, tCur, tSegEnd, vTd, aDec, vRetIn)
        : polylineSpeedScaledToWindow(runwayPts, tCur, tSegEnd, vTaxiBase);
      merged = seg;
      tCur = tSegEnd;
    }
    if (retPts.length >= 2 && phyRet > 1e-9) {
      const tSegEnd = tCur + phyRet * scale;
      const seg = polylineSpeedScaledToWindow(retPts, tCur, tSegEnd, vRetResolved);
      merged = merged ? mergeTimelineSegments(merged, seg) : seg;
      tCur = tSegEnd;
    }
    if (!merged) {
      return polylineSpeedScaledToWindow(taxiInPts, tStart, tEnd, vTaxiBase);
    }
    if (tCur < tEnd - 1e-3) {
      const last = merged[merged.length - 1];
      merged = mergeTimelineSegments(merged, [{ t: tCur, x: last.x, y: last.y }, { t: tEnd, x: last.x, y: last.y }]);
    }
    return merged;
  }
  function buildApronTaxiTimelineAfterRet(f, runwayId, taxiInPts, tStart, tEnd) {
    const parts = splitTaxiInPartsForTimeline(f, runwayId, taxiInPts);
    const taxiPts = parts.taxiPts;
    const phyTaxi = parts.phyTaxi;
    const vTaxiBase = parts.vTaxiBase;
    const cr = parts.carryAfterRunway || { lastTaxiwayMs: null };
    const carryApron = { lastTaxiwayMs: cr.lastTaxiwayMs };
    if (!taxiInPts || taxiInPts.length < 2 || tEnd <= tStart + 1e-6) {
      const p = taxiInPts && taxiInPts.length ? taxiInPts[taxiInPts.length - 1] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }, { t: tEnd, x: p[0], y: p[1] }];
    }
    if (taxiPts.length >= 2 && phyTaxi > 1e-9) {
      return polylineTimelineBySegmentSpeeds(taxiPts, tStart, tEnd, function(i, a, b) {
        return taxiSegmentVelocityMsForPolylineSegment(a, b, carryApron);
      });
    }
    const last = taxiInPts[taxiInPts.length - 1];
    return [{ t: tStart, x: last[0], y: last[1] }, { t: tEnd, x: last[0], y: last[1] }];
  }
  function buildTaxiInCompositeTimeline(f, runwayId, taxiInPts, tTaxiStart, eibtS) {
    if (!taxiInPts || taxiInPts.length < 2) {
      const p = taxiInPts && taxiInPts.length ? taxiInPts[0] : [0, 0];
      return [{ t: tTaxiStart, x: p[0], y: p[1] }, { t: eibtS, x: p[0], y: p[1] }];
    }
    const parts = splitTaxiInPartsForTimeline(f, runwayId, taxiInPts);
    const { vTaxiBase, runwayPts, retPts, taxiPts, phyRw, phyRet, phyTaxi, useRwPhy, runwayLenM, vTd, aDec, vRetIn, vRetOut, vRetResolved, carryAfterRunway } = parts;
    const vRetRes = Math.max(1, vRetResolved != null ? vRetResolved : vTaxiBase);
    const crComp = carryAfterRunway || { lastTaxiwayMs: null };
    const carryCompTaxi = { lastTaxiwayMs: crComp.lastTaxiwayMs };
    const window = Math.max(1e-6, eibtS - tTaxiStart);
    let rawSum = phyRw + phyRet + phyTaxi;
    if (rawSum < 1e-9) {
      return polylineSpeedScaledToWindow(taxiInPts, tTaxiStart, eibtS, vTaxiBase);
    }
    const scale = window / rawSum;
    let tCur = tTaxiStart;
    let merged = null;
    if (runwayPts.length >= 2 && (useRwPhy ? runwayLenM > 1 : phyRw > 1e-9)) {
      const tEnd = tCur + phyRw * scale;
      const seg = useRwPhy
        ? runwayPhysicsTimelineScaled(runwayPts, runwayLenM, tCur, tEnd, vTd, aDec, vRetIn)
        : polylineSpeedScaledToWindow(runwayPts, tCur, tEnd, vTaxiBase);
      merged = seg;
      tCur = tEnd;
    }
    if (retPts.length >= 2 && phyRet > 1e-9) {
      const tEnd = tCur + phyRet * scale;
      const seg = polylineSpeedScaledToWindow(retPts, tCur, tEnd, vRetRes);
      merged = merged ? mergeTimelineSegments(merged, seg) : seg;
      tCur = tEnd;
    }
    if (taxiPts.length >= 2 && phyTaxi > 1e-9) {
      const seg = polylineTimelineBySegmentSpeeds(taxiPts, tCur, eibtS, function(i, a, b) {
        return taxiSegmentVelocityMsForPolylineSegment(a, b, carryCompTaxi);
      });
      merged = merged ? mergeTimelineSegments(merged, seg) : seg;
      tCur = eibtS;
    }
    if (!merged) {
      return polylineSpeedScaledToWindow(taxiInPts, tTaxiStart, eibtS, vTaxiBase);
    }
    if (tCur < eibtS - 1e-3) {
      const last = merged[merged.length - 1];
      merged = mergeTimelineSegments(merged, [{ t: tCur, x: last.x, y: last.y }, { t: eibtS, x: last.x, y: last.y }]);
    }
    return merged;
  }
  function polylineSpeedScaledToWindow(pts, tStart, tEnd, velocityMs) {
    const v = Math.max(1, velocityMs);
    if (!pts || pts.length < 2 || tEnd <= tStart + 1e-6) {
      const p = pts && pts.length ? pts[0] : [0, 0];
      return [{ t: tStart, x: p[0], y: p[1] }];
    }
    const lengths = [];
    for (let i = 0; i < pts.length - 1; i++) lengths.push(pathDist(pts[i], pts[i + 1]));
    const rawDts = lengths.map(function(len) { return len / v; });
    const rawTotal = rawDts.reduce(function(s, x) { return s + x; }, 0);
    const window = tEnd - tStart;
    if (rawTotal < 1e-6) {
      return [
        { t: tStart, x: pts[0][0], y: pts[0][1] },
        { t: tEnd, x: pts[pts.length - 1][0], y: pts[pts.length - 1][1] },
      ];
    }
    const scale = window / rawTotal;
    const tl = [{ t: tStart, x: pts[0][0], y: pts[0][1] }];
    let acc = 0;
    for (let i = 0; i < lengths.length; i++) {
      acc += rawDts[i] * scale;
      const tt = tStart + acc;
      tl.push({ t: Math.min(tt, tEnd), x: pts[i + 1][0], y: pts[i + 1][1] });
    }
    tl[tl.length - 1].t = tEnd;
    return tl;
  }
  
  function splitDeparturePathLineupAndRunwayTail(f) {
    const depFull = getPathForFlightDeparture(f);
    const depToLineup = (typeof graphPathDeparture === 'function') ? graphPathDeparture(f, { onlyToLineup: true }) : null;
    if (!depFull || depFull.length < 2 || !depToLineup || depToLineup.length < 2) return null;
    const lastLu = depToLineup[depToLineup.length - 1];
    const tol = 0.25;
    let k = -1;
    for (let i = 0; i < depFull.length; i++) {
      if (dist2(depFull[i], lastLu) <= tol) k = i;
    }
    let runwayTail = (k >= 0) ? depFull.slice(k) : null;
    if (!runwayTail || runwayTail.length < 2) {
      const runwayId = f.depRunwayId || (f.token && f.token.depRunwayId) || (f.token && f.token.runwayId) || f.arrRunwayId;
      const rp = runwayId ? getRunwayPath(runwayId) : null;
      const rEnd = rp && rp.endPx ? rp.endPx : (rp && rp.pts && rp.pts.length >= 2 ? rp.pts[rp.pts.length - 1] : null);
      if (rEnd && Array.isArray(rEnd) && rEnd.length >= 2) {
        const lx = lastLu[0], ly = lastLu[1];
        if (!runwayTail || runwayTail.length < 1) runwayTail = [[lx, ly], [rEnd[0], rEnd[1]]];
        else if (runwayTail.length === 1 && dist2(runwayTail[0], rEnd) > 1e-6) runwayTail = [runwayTail[0], [rEnd[0], rEnd[1]]];
      }
    }
    if (!runwayTail || runwayTail.length < 2) runwayTail = null;
    return { toLineup: depToLineup, runwayTail: runwayTail };
  }
  function lineupDepQueueFingerprint(flights) {
    const parts = [];
    const list = flights || [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!f || f.noWayDep) continue;
      const eob = flightEMinutesPrefer(f, ['eobtMin'], flightEMinutesPrefer(f, ['timeMin'], 0) + (typeof f.dwellMin === 'number' ? f.dwellMin : 0));
      const vtt = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
      const rw = f.depRunwayId || (f.token && (f.token.depRunwayId != null ? f.token.depRunwayId : f.token.runwayId)) || f.arrRunwayId || '';
      const st = f.standId != null ? f.standId : '';
      parts.push(String(f.id != null ? f.id : i) + ':' + String(rw) + ':' + String(st) + ':' + eob + ':' + vtt);
    }
    parts.sort();
    return String(state.pathPolylineCacheRev | 0) + '|' + parts.join(';');
  }
  function assignLineupQueueRanksAll(flights) {
    const list = flights || [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (f) delete f._lineupQueueRank;
    }
    const entries = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!f || f.noWayDep) continue;
      const split = splitDeparturePathLineupAndRunwayTail(f);
      if (!split || !split.toLineup || split.toLineup.length < 2) continue;
      const last = split.toLineup[split.toLineup.length - 1];
      const rw = f.depRunwayId || (f.token && (f.token.depRunwayId != null ? f.token.depRunwayId : f.token.runwayId)) || f.arrRunwayId || '';
      const key = String(rw) + '|' + (Math.round(last[0] * 10) / 10) + '|' + (Math.round(last[1] * 10) / 10);
      const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], flightEMinutesPrefer(f, ['timeMin'], 0) + (typeof f.dwellMin === 'number' ? f.dwellMin : 0));
      const lineupEtaSec = eobtMin * 60 + Math.max(0, (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0) * 60;
      entries.push({ f: f, key: key, lineupEtaSec: lineupEtaSec });
    }
    const byKey = {};
    for (let j = 0; j < entries.length; j++) {
      const e = entries[j];
      if (!byKey[e.key]) byKey[e.key] = [];
      byKey[e.key].push(e);
    }
    Object.keys(byKey).forEach(function(k) {
      const arr = byKey[k];
      arr.sort(function(a, b) {
        if (a.lineupEtaSec !== b.lineupEtaSec) return a.lineupEtaSec - b.lineupEtaSec;
        const ia = a.f.id != null ? String(a.f.id) : '';
        const ib = b.f.id != null ? String(b.f.id) : '';
        return ia.localeCompare(ib);
      });
      for (let r = 0; r < arr.length; r++) arr[r].f._lineupQueueRank = r;
    });
  }
  function ensureLineupQueueRanksForSimulation() {
    const flights = state.flights || [];
    const fp = lineupDepQueueFingerprint(flights);
    if (state.__lineupQueueRankFp === fp) return;
    state.__lineupQueueRankFp = fp;
    assignLineupQueueRanksAll(flights);
  }
  
  function buildDepartureSurfaceTimelineSegments(f, eobtS, etotS) {
    const eps = 1e-3;
    const split = splitDeparturePathLineupAndRunwayTail(f);
    if (!split || !split.toLineup || split.toLineup.length < 2) return null;
    const depTaxiLineupMin = (typeof getBaseVttDepMinutesToLineup === 'function') ? getBaseVttDepMinutesToLineup(f) : 0;
    const depTaxiLineupSecReq = Math.max(0, depTaxiLineupMin) * 60;
    const depTaxiDelaySecReq = (typeof f.depTaxiDelayMin === 'number' && isFinite(f.depTaxiDelayMin))
      ? Math.max(0, f.depTaxiDelayMin) * 60 : 0;
    const t0 = eobtS;
    const t3 = etotS;
    const rank = (typeof f._lineupQueueRank === 'number' && isFinite(f._lineupQueueRank)) ? Math.max(0, Math.floor(f._lineupQueueRank)) : 0;
    const backM = rank * LINEUP_QUEUE_SPACING_M;
    const toLineupOrig = split.toLineup;
    const totalLen = polylineTotalLength(toLineupOrig);
    const maxBack = Math.max(0, totalLen - 1e-3);
    const backClamped = Math.min(backM, maxBack);
    const alongCut = Math.max(1e-6, totalLen - backClamped);
    const splitCut = polylineSplitAtDistance(toLineupOrig, alongCut);
    let taxiLineupPts = (splitCut.first && splitCut.first.length >= 2) ? splitCut.first : toLineupOrig;
    if (taxiLineupPts.length < 2) taxiLineupPts = toLineupOrig;
    const lastLu = taxiLineupPts[taxiLineupPts.length - 1];
    const lx = lastLu[0], ly = lastLu[1];
    let runwayTailAdj = split.runwayTail;
    if (runwayTailAdj && runwayTailAdj.length >= 2 && dist2(runwayTailAdj[0], [lx, ly]) > 1e-4) {
      runwayTailAdj = [[lx, ly]].concat(runwayTailAdj.slice());
    }
    const makeVelTaxi = makeTaxiSegmentVelocityCallback();
    const makeVelRoll = makeTaxiSegmentVelocityCallback();
    if (!(t3 > t0 + eps)) {
      const tl = [{ t: t0, x: lx, y: ly }, { t: t3, x: lx, y: ly }];
      return {
        timeline: tl,
        meta: {
          eobtSec: t0, etotSec: t3,
          depTaxiLineupSec: 0, depTaxiDelaySec: 0, depTaxiLineupSecReq: depTaxiLineupSecReq, depTaxiDelaySecReq: depTaxiDelaySecReq,
          lineupArrivalSec: t0, depRollStartSec: t0, depRotSec: Math.max(0, t3 - t0),
          lineupQueueRank: rank, lineupBackM: backClamped,
        },
      };
    }
    const maxSpan = t3 - t0 - eps;
    const taxiSecUsed = Math.min(depTaxiLineupSecReq, maxSpan);
    const t1 = t0 + taxiSecUsed;
    const afterTaxi = Math.max(0, t3 - t1 - eps);
    const delaySecUsed = Math.min(depTaxiDelaySecReq, afterTaxi);
    const t2 = t1 + delaySecUsed;
    const taxiTl = polylineTimelineBySegmentSpeeds(taxiLineupPts, t0, t1, makeVelTaxi);
    let holdTl = (t2 > t1 + eps) ? [{ t: t1, x: lx, y: ly }, { t: t2, x: lx, y: ly }] : [];
    let rollTl;
    if (runwayTailAdj && runwayTailAdj.length >= 2 && t3 > t2 + eps) {
      rollTl = polylineTimelineBySegmentSpeeds(runwayTailAdj, t2, t3, makeVelRoll);
    } else {
      rollTl = [{ t: t2, x: lx, y: ly }, { t: t3, x: lx, y: ly }];
    }
    let merged = mergeTimelineSegments(taxiTl, holdTl);
    merged = mergeTimelineSegments(merged, rollTl);
    return {
      timeline: merged,
      meta: {
        eobtSec: t0, etotSec: t3,
        depTaxiLineupSec: taxiSecUsed, depTaxiDelaySec: delaySecUsed,
        depTaxiLineupSecReq: depTaxiLineupSecReq, depTaxiDelaySecReq: depTaxiDelaySecReq,
        lineupArrivalSec: t1, depRollStartSec: t2, depRotSec: Math.max(0, t3 - t2),
        lineupQueueRank: rank, lineupBackM: backClamped,
      },
    };
  }
  function buildFullAirsideTimelineForFlight(f) {
    if (!f) return;
    if (typeof ensureLineupQueueRanksForSimulation === 'function') ensureLineupQueueRanksForSimulation();
    const vTaxiBase = Math.max(1, typeof getTaxiwayAvgMoveVelocityForPath === 'function' ? getTaxiwayAvgMoveVelocityForPath(null) : 10);
    if (f.arrDep === 'Dep') {
      if (f.noWayDep) {
        f.timeline = null;
        f.timeline_meta = { error: 'no_path', leg: 'dep' };
        return;
      }
      const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], flightEMinutesPrefer(f, ['timeMin'], 0) + (typeof f.dwellMin === 'number' ? f.dwellMin : 0));
      const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
      const eobtS = eobtMin * 60;
      const etotS = etotMin * 60;
      const built = buildDepartureSurfaceTimelineSegments(f, eobtS, etotS);
      if (!built || !built.timeline || built.timeline.length < 2) {
        f.timeline = null;
        f.timeline_meta = { error: 'no_path', leg: 'dep' };
        return;
      }
      f.timeline = built.timeline;
      f.timeline_meta = Object.assign({ leg: 'dep' }, built.meta || {});
      return;
    }
    const arrPts = getPathForFlight(f);
    const depPts = getPathForFlightDeparture(f);
    if (f.noWayArr || f.noWayDep) {
      f.timeline = null;
      f.timeline_meta = { error: 'no_path' };
      return;
    }
    if (!arrPts || arrPts.length < 2 || !depPts || depPts.length < 2) {
      f.timeline = null;
      f.timeline_meta = { error: 'no_path' };
      return;
    }
    const token = f.token || {};
    const runwayId = f.arrRunwayIdUsed || token.arrRunwayId || token.runwayId || f.arrRunwayId;
    if (runwayId == null || runwayId === '') {
      f.timeline = null;
      f.timeline_meta = { error: 'no_runway' };
      return;
    }
    const rwDir = String(f.arrRunwayDirUsed || 'clockwise');
    const vTd = Math.max(1, touchdownSpeedMsForTimeline(f));
    const tdDist = touchdownDistMForTimeline(f);
    const anchorDist = arrivalApproachAnchorDistM(runwayId, tdDist);
    const offset = APPROACH_OFFSET_WORLD_M;
    const eldtMin = flightEMinutesPrefer(f, ['eldtMin'], flightEMinutesPrefer(f, ['timeMin'], 0));
    const eibtMin = flightEMinutesPrefer(f, ['eibtMin'], eldtMin + 15);
    const eobtMin = flightEMinutesPrefer(f, ['eobtMin'], eibtMin + (typeof f.dwellMin === 'number' && isFinite(f.dwellMin) ? f.dwellMin : 45));
    const etotMin = flightEMinutesPrefer(f, ['etotMin'], eobtMin + 30);
    const eldtS = eldtMin * 60;
    const eibtS = eibtMin * 60;
    const eobtS = eobtMin * 60;
    const etotS = etotMin * 60;
    const tdPt = getRunwayPointAtDistance(runwayId, tdDist);
    if (!tdPt) {
      f.timeline = null;
      f.timeline_meta = { error: 'no_td' };
      return;
    }
    const pack = buildStraightApproachPolylineWorld(runwayId, rwDir, anchorDist, offset);
    let apprPts;
    if (pack && pack.pts && pack.pts.length >= 2) {
      apprPts = pack.pts.slice();
      const lastAp = apprPts[apprPts.length - 1];
      if (Math.hypot(lastAp[0] - tdPt[0], lastAp[1] - tdPt[1]) > 1e-3) apprPts.push([tdPt[0], tdPt[1]]);
    } else {
      const rsPt = getRunwayPointAtDistance(runwayId, anchorDist);
      const outer = approachPointBeforeThresholdJs(runwayId, rwDir, offset, anchorDist);
      const mid = rsPt ? [rsPt[0], rsPt[1]] : [tdPt[0], tdPt[1]];
      apprPts = [outer, mid];
      if (rsPt && Math.hypot(rsPt[0] - tdPt[0], rsPt[1] - tdPt[1]) > 1e-3) apprPts.push([tdPt[0], tdPt[1]]);
    }
    const tAppr = arrivalApproachDurationSecBeforeEldt(f);
    const t0 = eldtS - tAppr;
    const airTl = polylineTimelineBySegmentSpeeds(apprPts, t0, eldtS, function() { return vTd; });
    const rotS = (typeof f.arrRotSec === 'number' && isFinite(f.arrRotSec)) ? Math.max(0, f.arrRotSec) : 0;
    const vttDelayS = (typeof f.vttADelayMin === 'number' && isFinite(f.vttADelayMin) ? f.vttADelayMin : 0) * 60;
    const tAfterRot = eldtS + rotS;
    const runwayEndT = Math.min(tAfterRot, eibtS);
    let tTaxiStart = Math.min(tAfterRot + vttDelayS, eibtS);
    if (tTaxiStart < runwayEndT) tTaxiStart = runwayEndT;
    const taxiInPts = trimPolylineFromNearPoint(arrPts, tdPt);
    let taxiInTl;
    if (runwayEndT > eldtS + 1e-3) {
      taxiInTl = buildRunwayAndRetTimelineInWindow(f, runwayId, taxiInPts, eldtS, runwayEndT);
    } else {
      taxiInTl = [{ t: eldtS, x: tdPt[0], y: tdPt[1] }];
    }
    if (tTaxiStart > runwayEndT + 1e-3 && taxiInTl && taxiInTl.length) {
      const lastRw = taxiInTl[taxiInTl.length - 1];
      taxiInTl = mergeTimelineSegments(taxiInTl, [
